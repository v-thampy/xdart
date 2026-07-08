# -*- coding: utf-8 -*-
"""
@author: walroth, vthampy
"""
# Top level script for running gui based program

# Standard library imports
import sys
import gc
import os
import signal
import logging
import faulthandler
faulthandler.enable()  # Print Python traceback on bus error / segfault
if hasattr(signal, "SIGUSR1"):
    # Live-freeze diagnostic: `kill -USR1 <pid>` dumps every thread's Python
    # stack to stderr, even while the GUI thread is busy (the dump runs in the
    # C signal handler, no GIL needed).  No root required, unlike py-spy on macOS.
    faulthandler.register(signal.SIGUSR1, all_threads=True)

# Set PySide6 as the Qt binding for pyqtgraph before any Qt imports.
# Also export MPLBACKEND so child processes (e.g. pyFAI-calib2) inherit it.
os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'
os.environ['MPLBACKEND'] = 'QtAgg'

# Set matplotlib backend before any matplotlib import can occur.
# Use QtAgg (the Qt6 backend) to match pyqtgraph's choice.
import matplotlib
matplotlib.use('QtAgg')

# Default root logging level — INFO is what every wrangler log line
# currently uses, so basicConfig(INFO) is enough to surface them.
# The DEBUG line below opts specific loggers into more verbose output;
# the basicConfig must happen first so the handler's threshold is open.
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s',
)

# Suppress pyFAI INFO logs (e.g. "No sensor configuration provided").
logging.getLogger('pyFAI').setLevel(logging.WARNING)
logging.getLogger('pyFAI.gui.matplotlib').setLevel(logging.ERROR)
# Suppress silx's "pyOpenCL has been imported but can't be used here"
# warning — OpenCL is optional and the message has no user action.
logging.getLogger('silx.opencl').setLevel(logging.ERROR)

# pyqtgraph's log-axis tick painter computes 10**range while the histogram
# axis still holds the previous LINEAR image's extent for one paint after a
# Log toggle (e.g. Eiger counts ~4e9 -> 10**4e9).  Harmless — the inf is
# clamped on the next paint — but it logged a RuntimeWarning on every
# toggle.  Scoped to exactly that message and module.
import warnings
warnings.filterwarnings(
    'ignore', message='overflow encountered in power',
    category=RuntimeWarning, module=r'pyqtgraph\.graphicsItems\.AxisItem')

logger = logging.getLogger(__name__)

# Qt imports
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    QtCore: Any = None
    QtGui: Any = None
    QtWidgets: Any = None
else:
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

# This module imports
from xdart.gui.mainWindow import Ui_MainWindow
from xdart.gui import tabs


QMainWindow = QtWidgets.QMainWindow


# One modal error dialog in flight at a time.  A repeating error source (e.g. a
# timer/paint slot that raises every tick) would otherwise schedule an unbounded
# stack of QMessageBoxes; every exception is still logged, only the dialog is
# coalesced until the current one is dismissed.
_error_dialog_pending = False


def _xdart_excepthook(exc_type, exc, tb):
    """Log uncaught GUI-slot exceptions without terminating the process."""
    global _error_dialog_pending
    if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
        sys.__excepthook__(exc_type, exc, tb)
        return
    logger.error("Unhandled exception in xdart GUI", exc_info=(exc_type, exc, tb))
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    if _error_dialog_pending:
        # Already showing/queuing a dialog — the exception is logged above; don't
        # pile up another modal for a fast-repeating error source.
        return
    _error_dialog_pending = True

    def _show_error():
        global _error_dialog_pending
        try:
            QtWidgets.QMessageBox.critical(
                None,
                "xdart error",
                f"{exc_type.__name__}: {exc}\n\n"
                "The error was logged; the application will stay open.",
            )
        except Exception:
            logger.debug("Could not show GUI exception dialog", exc_info=True)
        finally:
            _error_dialog_pending = False

    try:
        QtCore.QTimer.singleShot(0, _show_error)
    except Exception:
        logger.debug("Could not schedule GUI exception dialog", exc_info=True)
        _error_dialog_pending = False


class _UpdateCheckThread(QtCore.QThread):
    """One-shot worker: fetch the latest PyPI version OFF the GUI thread so the
    ~3 s network round-trip never blocks the event loop (updater spec section 4)."""
    result_ready = QtCore.Signal(object)   # latest version str, or None

    def run(self):                                     # pragma: no cover - Qt thread
        from xdart.modules import updater
        self.result_ready.emit(updater.fetch_latest_pypi())


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('xdart')
        self.ui.actionOpen.triggered.connect(self.openFile)
        self.ui.actionExit.triggered.connect(self.exit)
        self.fname = None

        # Embed the main widget directly (no tab container).
        # The widget chooses its own scratch directory via get_fname_dir().
        self.main_widget = tabs.static_scan.staticWidget()
        self.setCentralWidget(self.main_widget)
        # D2 (greenfield Phase 3): in the live app, rehydrate evicted frames off
        # the GUI thread so scroll-back no longer freezes on a ~5 s .nxs open.
        # Done here (not in widget construction) so headless widget tests keep
        # their synchronous reads.
        try:
            self.main_widget.enable_async_hydration()
        except Exception:
            pass
        self._init_theme_menu()
        self._init_shortcut_actions()

        # Default size: 90% of the available screen, centered (was a fixed
        # 1600x920, whose width clamped the middle display panels below
        # their intended 57% share).  setGeometry rather than resize() --
        # a post-show resize was unreliable for width on macOS.
        self.show()
        try:
            avail = self.screen().availableGeometry()
            w = int(avail.width() * 0.95)
            h = int(avail.height() * 0.90)
            self.setGeometry(avail.x() + (avail.width() - w) // 2,
                             avail.y() + (avail.height() - h) // 2, w, h)
        except Exception:
            self.resize(1600, 920)

    def _init_theme_menu(self):
        """Add a Theme (Dark/Light) submenu to the in-window Config menu.

        The visible File/Config controls are the H5Viewer toolbar's tool-buttons
        (``h5viewer.paramMenu``), NOT the QMainWindow menu bar (which on macOS is
        the native top-of-screen bar).  Add Theme there so it sits next to
        Save/Load/Advanced where the user expects it.  Persisted in QSettings;
        switching re-applies the QSS live (the pyqtgraph plot-canvas background
        is snapshotted at widget creation, so a full plot recolor needs a
        relaunch -- a later stage owns per-mode plot backgrounds)."""
        try:
            config_menu = self.main_widget.h5viewer.paramMenu
        except Exception:
            logger.exception("Could not locate the Config menu for the theme toggle")
            return
        settings = QtCore.QSettings("xdart", "xdart")
        current = settings.value("theme", "dark")
        if current not in ("dark", "light"):
            current = "dark"
        panel_font_size = settings.value(
            "control_panel_font_size", "default")
        if panel_font_size not in ("small", "default", "large"):
            panel_font_size = "default"
        config_menu.addSeparator()
        theme_menu = config_menu.addMenu("Theme")
        group = QtGui.QActionGroup(self)
        group.setExclusive(True)
        for name, label in (("dark", "Dark"), ("light", "Light")):
            action = QtGui.QAction(label, self)
            action.setCheckable(True)
            action.setChecked(name == current)
            action.triggered.connect(
                lambda _checked=False, n=name: self._set_theme(n))
            group.addAction(action)
            theme_menu.addAction(action)
        font_menu = config_menu.addMenu("Control Panel Font Size")
        font_group = QtGui.QActionGroup(self)
        font_group.setExclusive(True)
        for size, label in (
            ("small", "Small"),
            ("default", "Default"),
            ("large", "Large"),
        ):
            action = QtGui.QAction(label, self)
            action.setCheckable(True)
            action.setChecked(size == panel_font_size)
            action.triggered.connect(
                lambda _checked=False, s=size:
                    self._set_control_panel_font_size(s))
            font_group.addAction(action)
            font_menu.addAction(action)
        self.debugMenu = config_menu.addMenu("Debug")
        self.actionDebugWindowState = QtGui.QAction("Window State", self)
        self.actionDebugWindowState.triggered.connect(self._log_window_state)
        self.debugMenu.addAction(self.actionDebugWindowState)

        # Help toolbar group (a top-level group next to Config): Check for
        # Updates... now; help-doc links can join it later.
        try:
            help_menu = self.main_widget.h5viewer.helpMenu
            update_action = QtGui.QAction("Check for Updates…", self)
            update_action.triggered.connect(self._check_for_updates)
            help_menu.addAction(update_action)
        except Exception:
            logger.exception("Could not set up the Help menu")

    # ── In-app updater (Help → Check for Updates…) — spec section 4 ───────────
    def _run_active(self):
        """True while a processing run holds the display — never update mid-run."""
        try:
            return bool(getattr(
                self.main_widget.displayframe, "_processing_active", False))
        except Exception:
            return False

    def _check_for_updates(self):
        from xdart.modules import updater
        if self._run_active():
            QtWidgets.QMessageBox.information(
                self, "Check for Updates",
                "A processing run is active — finish or stop it before updating.")
            return
        kind = updater.install_kind()
        if kind == "editable":
            QtWidgets.QMessageBox.information(
                self, "Check for Updates",
                "This is a development checkout. Update with git, not the in-app "
                "updater.")
            return
        self._update_kind = kind
        self._update_meta = updater.resolve_update_meta(kind)
        # Fetch the latest version off the GUI thread; never block the event loop.
        self._update_thread = _UpdateCheckThread(self)
        self._update_thread.result_ready.connect(self._on_update_check_result)
        self._update_thread.finished.connect(self._update_thread.deleteLater)
        self._update_thread.start()

    def _on_update_check_result(self, latest):
        from xdart.modules import updater
        current = updater.current_version()
        if latest is None:
            # None = network failure OR PyPI has no such release yet (404).
            self.statusBar().showMessage(
                "Could not check for updates (offline, or no release published "
                "on PyPI yet).", 8000)
            return
        if not updater.update_available(current, latest):
            QtWidgets.QMessageBox.information(
                self, "Check for Updates",
                f"xdart is up to date (version {current}).")
            return
        kind = getattr(self, "_update_kind", "managed")
        meta = getattr(self, "_update_meta", None) or {}
        # Update-on-exit is POSIX-only in v1.0: on Windows the env's .pyd/.dll are
        # locked under a live app and the PID probe differs (B2), so Windows -- and
        # any pip/conda-managed install on every platform -- gets a COPYABLE
        # command instead of the in-app update-on-exit.
        if kind == "managed" or sys.platform.startswith("win"):
            cmd = (" ".join(str(c) for c in (meta.get("update_cmd") or []))
                   or 'pip install -U "xrd-tools[gui]"')
            QtWidgets.QMessageBox.information(
                self, "Update available",
                f"xdart {latest} is available (you have {current}).\n\n"
                f"Update it with:\n\n    {cmd}\n\nthen restart xdart.")
            return
        resp = QtWidgets.QMessageBox.question(
            self, "Update available",
            f"xdart {latest} is available (you have {current}).\n\n"
            "Update and restart now?")
        if resp != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        # S3: a processing run may have started during the async PyPI fetch or
        # while this dialog was open -- never rewrite the env under a live run.
        if self._run_active():
            QtWidgets.QMessageBox.information(
                self, "Check for Updates",
                "A processing run is now active — try again after it finishes.")
            return
        self._launch_updater_and_close()

    def _launch_updater_and_close(self):
        import json
        meta = getattr(self, "_update_meta", None) or {}
        app_root = meta.get("app_root", "") or ""
        update_cmd = meta.get("update_cmd") or []
        relaunch_cmd = meta.get("relaunch_cmd") or []
        log_path = (os.path.join(app_root, "update.log") if app_root
                    else "update.log")
        args = ["-m", "xdart._updater", str(os.getpid()), app_root,
                json.dumps(update_cmd), json.dumps(relaunch_cmd), log_path]
        started = QtCore.QProcess.startDetached(sys.executable, args)
        if not started:
            QtWidgets.QMessageBox.warning(
                self, "Update",
                "Could not launch the updater. Update manually with:\n\n"
                f"    {' '.join(str(c) for c in update_cmd)}")
            return
        # B1: the check thread's run() may still be returning; wait for it so it is
        # not destroyed mid-flight, then exit via the HARDENED teardown
        # (Main.exit -> main_widget.close joins the reduction QThreads).  Bare
        # self.close() skips that teardown and crashes on the running QThreads.
        thread = getattr(self, "_update_thread", None)
        if thread is not None:
            try:
                thread.wait(3000)
            except Exception:
                pass
        self.exit()

    @staticmethod
    def _qsize_text(size):
        try:
            return f"{size.width()}x{size.height()}"
        except Exception:
            return repr(size)

    @staticmethod
    def _qt_value_text(value):
        name = getattr(value, "name", None)
        raw = getattr(value, "value", None)
        if name is not None and raw is not None:
            return f"{name}({raw})"
        return str(value)

    @classmethod
    def _widget_size_state(cls, widget):
        return (
            f"size={cls._qsize_text(widget.size())} "
            f"minHint={cls._qsize_text(widget.minimumSizeHint())} "
            f"min={cls._qsize_text(widget.minimumSize())} "
            f"max={cls._qsize_text(widget.maximumSize())}"
        )

    @classmethod
    def _top_level_widget_summary(cls, widget):
        name = widget.objectName() or "-"
        title = widget.windowTitle() or "-"
        return (
            f"{type(widget).__name__}(name={name!r}, title={title!r}, "
            f"size={cls._qsize_text(widget.size())})"
        )

    def _splitter_diagnostic_children(self):
        ui = getattr(getattr(self, "main_widget", None), "ui", None)
        labels = (
            ("left browser", "leftFrame"),
            ("middle display", "middleFrame"),
            ("right controls", "rightFrame"),
        )
        children = []
        for label, attr in labels:
            widget = getattr(ui, attr, None)
            if widget is not None:
                children.append((label, widget))
        if children:
            return children
        splitter = getattr(ui, "mainSplitter", None)
        if splitter is None or not hasattr(splitter, "count"):
            return []
        return [
            (f"mainSplitter[{idx}]", splitter.widget(idx))
            for idx in range(splitter.count())
            if splitter.widget(idx) is not None
        ]

    def _log_window_state(self):
        """Log resize/cursor state for diagnosing sporadic window lockups."""
        logger.warning(
            "Window State main: %s minHint=%s min=%s max=%s flags=%s "
            "isMaximized=%s isFullScreen=%s",
            f"size={self._qsize_text(self.size())}",
            self._qsize_text(self.minimumSizeHint()),
            self._qsize_text(self.minimumSize()),
            self._qsize_text(self.maximumSize()),
            self._qt_value_text(self.windowFlags()),
            self.isMaximized(),
            self.isFullScreen(),
        )
        for label, widget in self._splitter_diagnostic_children():
            logger.warning(
                "Window State splitter child %s: %s",
                label,
                self._widget_size_state(widget),
            )
        cursor = QtWidgets.QApplication.overrideCursor()
        cursor_text = "None"
        if cursor is not None:
            cursor_text = f"shape={self._qt_value_text(cursor.shape())}"
        grabber = QtWidgets.QWidget.mouseGrabber()
        grabber_text = "None"
        if grabber is not None:
            grabber_text = self._top_level_widget_summary(grabber)
        logger.warning(
            "Window State input: overrideCursor=%s mouseGrabber=%s",
            cursor_text,
            grabber_text,
        )
        app = QtWidgets.QApplication.instance()
        top_levels = app.topLevelWidgets() if app is not None else []
        visible_parentless = [
            widget for widget in top_levels
            if widget.parent() is None and widget.isVisible()
        ]
        summary = ", ".join(
            self._top_level_widget_summary(widget)
            for widget in visible_parentless
        ) or "none"
        logger.warning(
            "Window State top-level widgets: total=%d visible_parentless=%d %s",
            len(top_levels),
            len(visible_parentless),
            summary,
        )

    def _init_shortcut_actions(self):
        """Add discoverable menu actions for the main processing shortcuts."""

        def _menu_action(menu, object_name, text, shortcut, slot, before=None):
            action = QtGui.QAction(text, self)
            action.setObjectName(object_name)
            action.setShortcut(shortcut)
            action.setShortcutContext(QtCore.Qt.WindowShortcut)
            action.triggered.connect(slot)
            if before is None:
                menu.addAction(action)
            else:
                menu.insertAction(before, action)
            self.addAction(action)
            return action

        file_menu = self.ui.menuFile
        file_menu.insertSeparator(self.ui.actionExit)
        self.actionLoadSettings = _menu_action(
            file_menu,
            "actionLoadSettings",
            "Load Settings",
            QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Open),
            self._shortcut_load_settings,
            before=self.ui.actionExit,
        )
        self.actionSaveSettings = _menu_action(
            file_menu,
            "actionSaveSettings",
            "Save Settings",
            QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Save),
            self._shortcut_save_settings,
            before=self.ui.actionExit,
        )

        self.ui.menuRun = QtWidgets.QMenu(self.ui.menubar)
        self.ui.menuRun.setObjectName("menuRun")
        self.ui.menuRun.setTitle("Run")
        self.ui.menubar.addAction(self.ui.menuRun.menuAction())
        self.actionRunPause = _menu_action(
            self.ui.menuRun,
            "actionRunPause",
            "Run / Pause",
            QtGui.QKeySequence("Ctrl+R"),
            self._shortcut_run_pause,
        )
        self.actionStopRun = _menu_action(
            self.ui.menuRun,
            "actionStopRun",
            "Stop",
            QtGui.QKeySequence("Ctrl+Shift+C"),
            self._shortcut_stop,
        )
        self.actionToggleWriteMode = _menu_action(
            self.ui.menuRun,
            "actionToggleWriteMode",
            "Toggle Append / Replace",
            QtGui.QKeySequence("Ctrl+Shift+A"),
            self._shortcut_toggle_write_mode,
        )
        self.actionPinSliceCut = _menu_action(
            self.ui.menuRun,
            "actionPinSliceCut",
            "Pin Slice Cut",
            QtGui.QKeySequence("Ctrl+P"),
            self._shortcut_pin_slice_cut,
        )

    def _main_widget_shortcut(self, method_name):
        method = getattr(getattr(self, "main_widget", None), method_name, None)
        if method is None:
            logger.warning("Shortcut target missing on main_widget: %s", method_name)
            return
        try:
            method()
        except Exception:
            logger.exception("Error handling shortcut %s", method_name)

    def _shortcut_run_pause(self):
        self._main_widget_shortcut("shortcut_run_pause")

    def _shortcut_stop(self):
        self._main_widget_shortcut("shortcut_stop")

    def _shortcut_toggle_write_mode(self):
        self._main_widget_shortcut("shortcut_toggle_write_mode")

    def _shortcut_pin_slice_cut(self):
        self._main_widget_shortcut("shortcut_pin_slice_cut")

    def _shortcut_load_settings(self):
        self._main_widget_shortcut("shortcut_load_settings")

    def _shortcut_save_settings(self):
        self._main_widget_shortcut("shortcut_save_settings")

    def _set_theme(self, name):
        """Apply theme ``name`` live and persist the choice."""
        from xdart.gui.themes import apply_theme
        settings = QtCore.QSettings("xdart", "xdart")
        panel_font_size = settings.value(
            "control_panel_font_size", "default")
        app = QtWidgets.QApplication.instance()
        if app is not None:
            apply_theme(
                app, name, control_panel_font_size=panel_font_size)
        settings.setValue("theme", name)

    def _set_control_panel_font_size(self, size):
        """Apply the Controls-panel-only font-size preset and persist it."""
        if size not in ("small", "default", "large"):
            size = "default"
        settings = QtCore.QSettings("xdart", "xdart")
        settings.setValue("control_panel_font_size", size)
        theme = settings.value("theme", "dark")
        if theme not in ("dark", "light"):
            theme = "dark"
        from xdart.gui.themes import apply_theme
        app = QtWidgets.QApplication.instance()
        if app is not None:
            apply_theme(app, theme, control_panel_font_size=size)

    def exit(self):
        try:
            self.main_widget.close()
        finally:
            self.close()
            gc.collect()
            try:
                os.killpg(os.getpid(), signal.SIGTERM)
            except ProcessLookupError:
                pass
            sys.exit(0)

    def openFile(self):
        try:
            self.main_widget.open_file()
        except Exception:
            logger.exception("Error opening file")


def _apply_cli_session_args(argv):
    """Parse ``-f``/``-n`` and point the session system at the right file via
    env vars BEFORE any widget loads its session.  Returns the argv (minus the
    consumed flags) to hand to Qt.

    ``xdart -f``      → fresh session (load nothing, persist nothing).
    ``xdart -n NAME`` → named saved session (NAME under ~/.xdart; the ``.json``
                        extension is forced if the user omits it).
    """
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(
        prog='xdart', description='xdart — SSRL XRD reduction GUI')
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-f', '--fresh', action='store_true',
        help='start a fresh session (does not load or modify your saved session)')
    group.add_argument(
        '-n', '--session', metavar='NAME',
        help='start from a named saved session (NAME under ~/.xdart; '
             '.json is appended if omitted)')
    args, rest = parser.parse_known_args(argv[1:])
    if args.fresh:
        os.environ['XDART_SESSION_FRESH'] = '1'
    elif args.session:
        name = args.session
        if not name.lower().endswith('.json'):
            name += '.json'                  # force the .json extension
        p = Path(name)
        if not p.is_absolute() and p.parent == Path('.'):
            p = Path.home() / '.xdart' / name   # bare name -> ~/.xdart/
        os.environ['XDART_SESSION_FILE'] = str(p)
    return [argv[0], *rest]


def run():
    argv = _apply_cli_session_args(sys.argv)
    app = QtWidgets.QApplication(argv)
    # PERF-3 diagnostic (XDART_PERF only): time garbage collections.  CPython 3.12
    # GC is stop-the-world, so a gen-2 sweep over the large frame/publication graph
    # could freeze the GUI -- a candidate for the end-of-run ~3s recurring stalls
    # the heartbeat catches.  Logs each non-trivial collection's duration so ONE
    # run confirms or refutes GC (a STW pause is exactly what a stack sampler reads
    # ambiguously).  Registered only in the real app, behind the flag.
    if os.environ.get("XDART_PERF"):
        import time as _time
        _gc_t0 = [0.0]

        def _gc_perf_probe(phase, info):
            try:
                if phase == "start":
                    _gc_t0[0] = _time.perf_counter()
                elif phase == "stop":
                    dt_ms = (_time.perf_counter() - _gc_t0[0]) * 1000.0
                    gen = info.get("generation", -1)
                    if dt_ms >= 100.0 or gen >= 2:
                        logger.info(
                            "[PERF] gc: gen=%d elapsed=%.0fms collected=%d "
                            "uncollectable=%d", gen, dt_ms,
                            info.get("collected", 0),
                            info.get("uncollectable", 0))
            except Exception:
                pass

        gc.callbacks.append(_gc_perf_probe)
    # Install the keep-alive excepthook only when the GUI is actually launched —
    # never as an import-time side effect (importing this module must not hijack
    # the process-global sys.excepthook for tests / headless / embedding hosts).
    sys.excepthook = _xdart_excepthook
    # N8: apply the saved theme before any widget construction so
    # pyqtgraph plot backgrounds are set in time (pyqtgraph
    # snapshots the config at widget creation).
    try:
        from xdart.gui.themes import apply_theme
        settings = QtCore.QSettings("xdart", "xdart")
        theme = settings.value("theme", "dark")
        if theme not in ("dark", "light"):
            theme = "dark"
        panel_font_size = settings.value(
            "control_panel_font_size", "default")
        if panel_font_size not in ("small", "default", "large"):
            panel_font_size = "default"
        apply_theme(app, theme, control_panel_font_size=panel_font_size)
    except Exception:
        logger.exception("Failed to apply theme; using Qt default")
    mw = Main()
    mw.show()
    app.exec()


main = run   # back-compat alias


if __name__ == '__main__':
    sys.exit(run())
