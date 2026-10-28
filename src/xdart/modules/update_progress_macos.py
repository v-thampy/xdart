"""System AppKit progress host: its GUI runtime lives outside the environment being updated."""
import json


FOUNDATION = r'''
ObjC.import('Foundation');
ObjC.bindFunction('open', ['int', ['char *', 'int']]);
ObjC.bindFunction('flock', ['int', ['int', 'int']]);
ObjC.bindFunction('close', ['int', ['int']]);
function text(path) {
    var s = $.NSString.stringWithContentsOfFileEncodingError($(path), $.NSUTF8StringEncoding, null);
    return s.isNil() ? '' : ObjC.unwrap(s);
}
function read(path) { try { return JSON.parse(text(path)); } catch (_) { return {}; } }
function save(path, value) {
    if (!$(JSON.stringify(value)).writeToFileAtomicallyEncodingError($(path), true, $.NSUTF8StringEncoding, null))
        throw Error('Could not record the progress window readiness.');
}
function fileSize(path) {
    var h = $.NSFileHandle.fileHandleForReadingAtPath($(path));
    if (h.isNil()) return 0;
    try { return Number(h.seekToEndOfFile); } finally { h.closeFile; }
}
function lock(path) {
    // O_RDWR|O_CLOEXEC without O_CREAT (the app made the file), then LOCK_EX|LOCK_NB: no child inherits it.
    var fd = $.open($(path).UTF8String, 0x01000002);
    if (fd < 0) return -1;
    if ($.flock(fd, 6) !== 0) { $.close(fd); return -1; }
    return fd;
}
function logTail(path, start) {
    try {
        var h = $.NSFileHandle.fileHandleForReadingAtPath($(path));
        if (h.isNil()) return null;
        try {
            var end = Number(h.seekToEndOfFile);
            // This update's output only: from where the log ended when the window opened, at most its last 128 KiB.
            h.seekToFileOffset(Math.min(end, Math.max(start || 0, end - 131072)));
            var data = h.readDataOfLength(131072);
            // A bounded tail may start in a UTF-8 continuation byte; try the next complete character.
            for (var skip = 0; skip <= 3 && skip <= Number(data.length); skip++) {
                var bytes = data.subdataWithRange($.NSMakeRange(skip, Number(data.length) - skip));
                var value = $.NSString.alloc.initWithDataEncoding(bytes, $.NSUTF8StringEncoding);
                if (!value.isNil()) return ObjC.unwrap(value);
            }
            return null;
        } finally { h.closeFile; }
    } catch (_) { return null; }
}
'''


SOURCE = r'''
ObjC.import('Cocoa');
var job = read(jobPath), app = $.NSApplication.sharedApplication;
// No O_CREAT: launcher prepared these files. CLOEXEC keeps host custody out of its engine/restart children.
var hostFd = lock(job.host_lock);
if (hostFd < 0) throw Error('Another progress window holds this update.');
var engine = null, started = false, terminal = false, success = false, finalRead = false;
var logStart = fileSize(job.log);  // the shared log holds earlier updates too
var frame = $.NSMakeRect(0, 0, 660, 450);
var window = $.NSWindow.alloc.initWithContentRectStyleMaskBackingDefer(frame, 15, $.NSBackingStoreBuffered, false);
window.title = $('Update xdart');
window.releasedWhenClosed = false;
var content = window.contentView;
function label(value, x, y, w, h, tip) {
    var v = $.NSTextField.labelWithString($(value));
    v.frame = $.NSMakeRect(x, y, w, h); v.toolTip = $(tip);
    v.autoresizingMask = 10;  // width sizable, kept at the top
    content.addSubview(v); return v;
}
label('Installed: ' + job.from + '    •    Available: ' + job.target, 20, 406, 620, 24,
      'The installed version and the version selected for this update.');
var status = label('Waiting for xdart to close…', 20, 373, 620, 24,
                   'The update starts only after xdart closes safely.');
var progress = $.NSProgressIndicator.alloc.initWithFrame($.NSMakeRect(20, 345, 620, 18));
progress.indeterminate = true; progress.style = $.NSProgressIndicatorStyleBar; progress.autoresizingMask = 10;
progress.toolTip = $('Activity indicator. Package installation does not provide an exact overall percentage.');
content.addSubview(progress); progress.startAnimation(null);
var scroll = $.NSScrollView.alloc.initWithFrame($.NSMakeRect(20, 73, 620, 253));
scroll.hasVerticalScroller = true; scroll.autoresizingMask = 18;
var details = $.NSTextView.alloc.initWithFrame($.NSMakeRect(0, 0, 600, 253));
details.editable = false; details.selectable = true;
details.verticallyResizable = true; details.horizontallyResizable = false;
details.minSize = $.NSMakeSize(0, 253); details.maxSize = $.NSMakeSize(10000000, 10000000);
details.autoresizingMask = 2;
details.textContainer.containerSize = $.NSMakeSize(600, 10000000);
details.textContainer.widthTracksTextView = true;
details.font = $.NSFont.userFixedPitchFontOfSize(11);
details.toolTip = $('Recent update output. Scroll, select and copy details if the update fails.');
scroll.documentView = details; content.addSubview(scroll);
function button(title, x, action, tip) {
    var v = $.NSButton.alloc.initWithFrame($.NSMakeRect(x, 20, 160, 32));
    v.title = $(title); v.bezelStyle = $.NSBezelStyleRounded; v.action = action;
    v.toolTip = $(tip); content.addSubview(v); return v;
}
var restart = button('Restart xdart', 480, 'restart:', 'Open the updated xdart after its update has completed.');
restart.enabled = false; restart.autoresizingMask = 1;  // kept at the right
window.contentMinSize = $.NSMakeSize(560, 360);
var closeButton = button('Close', 20, 'closeWindow:', 'Close after the update has stopped; this does not cancel installation.');
closeButton.enabled = false;
function completed() {
    var state = read(job.state_dir + '/xdart-update-state.json'), h = read(job.state_dir + '/' + job.helper_file);
    return state.install === job.install && state.attempt && state.attempt.id === job.attempt_id &&
        state.attempt.authorized === true && h.attempt_id === job.attempt_id && h.state === 'completed' &&
        h.exit === 0 && !h.timed_out && !h.refused;
}
function tick() {
    if (terminal && !started) return; // a failed readiness check must never launch on a later tick
    if (!started) {
        if (!window.visible) throw Error('The update window is not visible.');
        started = true;
        // First timer callback after makeKeyAndOrderFront: app may now require both host and engine custody.
        save(job.host_file, {attempt_id: job.attempt_id, pid: Number($.NSProcessInfo.processInfo.processIdentifier), ready: true});
        engine = $.NSTask.alloc.init;
        engine.launchPath = $(python);
        engine.arguments = $(['-m', 'xdart._updater', jobPath]);
        engine.standardInput = $.NSFileHandle.fileHandleWithNullDevice;
        engine.standardOutput = $.NSFileHandle.fileHandleWithNullDevice;
        engine.standardError = $.NSFileHandle.fileHandleWithNullDevice;
        engine.launch;
    }
    if (terminal) { if (finalRead) return; finalRead = true; }  // one read after the engine exits, then stop
    var output = logTail(job.log, logStart);
    if (output !== null && ObjC.unwrap(details.string) !== output) { details.string = $(output); details.scrollRangeToVisible($.NSMakeRange(output.length, 0)); }
    if (!terminal && engine && !engine.running) {
        terminal = true;
        success = Number(engine.terminationStatus) === 0 && completed();
        progress.stopAnimation(null); progress.indeterminate = false; progress.doubleValue = success ? 100 : 0;
        closeButton.enabled = true; restart.enabled = success;
        status.stringValue = $(success ? 'Update complete. Restart xdart when you are ready.' : 'Update did not complete. Read the details below.');
    } else if (!terminal) {
        var h = read(job.state_dir + '/' + job.helper_file);
        if (h.attempt_id === job.attempt_id && h.phase === 'updating') status.stringValue = $('Installing the update…');
    }
}
ObjC.registerSubclass({name: 'XdartUpdateActions', superclass: 'NSObject', methods: {
    'tick:': {types: ['void', ['id']], implementation: function () {
        try { tick(); } catch (e) {
            status.stringValue = $('Progress error: ' + String(e));
            if (!engine || !engine.running) { terminal = true; closeButton.enabled = true; }
        }
    }},
    'restart:': {types: ['void', ['id']], implementation: function () {
        var admission = lock(job.state_dir + '/xdart-update-admission.lock');
        if (admission < 0) { status.stringValue = $('An update is starting. Please wait.'); return; }
        var helper = -1;
        try {
            helper = lock(job.state_dir + '/' + job.lock_file);
            if (!terminal || !success || helper < 0 || !completed()) throw Error('The update state changed; open xdart yourself.');
            // Helper lock excludes a new update; admission must be free when the new app bootstraps.
            $.close(admission); admission = -1;
            var task = $.NSTask.alloc.init;
            task.launchPath = $(job.relaunch_cmd[0]); task.arguments = $(job.relaunch_cmd.slice(1));
            task.standardInput = $.NSFileHandle.fileHandleWithNullDevice;
            task.standardOutput = $.NSFileHandle.fileHandleWithNullDevice;
            task.standardError = $.NSFileHandle.fileHandleWithNullDevice;
            task.launch;
            restart.enabled = false;
            status.stringValue = $('xdart is starting. You can close this window.');
        } catch (e) { status.stringValue = $(String(e)); }
        finally { if (helper >= 0) $.close(helper); if (admission >= 0) $.close(admission); }
    }},
    'closeWindow:': {types: ['void', ['id']], implementation: function () { if (terminal) app.terminate(null); }},
    'applicationShouldTerminate:': {types: ['long', ['id']], implementation: function () { return terminal ? 1 : 0; }},
    'windowShouldClose:': {types: ['bool', ['id']], implementation: function () { if (terminal) app.terminate(null); return false; }}
}});
var actions = $.XdartUpdateActions.alloc.init;
restart.target = actions; closeButton.target = actions; window.delegate = actions; app.delegate = actions;
if (!app.setActivationPolicy($.NSApplicationActivationPolicyRegular)) throw Error('Cannot show the update window.');
window.center; window.orderFrontRegardless; window.makeKeyAndOrderFront(null);
// macOS 14+ activates cooperatively (activate); earlier systems only have activateIgnoringOtherApps.
if (app.respondsToSelector('activate')) app.activate; else app.activateIgnoringOtherApps(true);
var timer = $.NSTimer.timerWithTimeIntervalTargetSelectorUserInfoRepeats(.25, actions, 'tick:', null, true);
$.NSRunLoop.mainRunLoop.addTimerForMode(timer, $.NSRunLoopCommonModes);
try { app.run; } finally { timer.invalidate; $.close(hostFd); }
'''


def script(job_path, python):
    return 'var jobPath = ' + json.dumps(str(job_path)) + ', python = ' + json.dumps(str(python)) + ';\n' + FOUNDATION + SOURCE
