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
from logging.handlers import RotatingFileHandler
from pathlib import Path
import faulthandler

# NOTE: no import-time side effects here.  faulthandler.enable(), the SIGUSR1
# stack-dump hook, the MPLBACKEND/QtAgg backend flip, and the hard
# PYQTGRAPH_QT_LIB pin all moved into run() — importing this module (tests,
# headless tools, embedding hosts) must not hijack process-global state.  The
# QtAgg flip at import broke headless CI collection outright ("Cannot load
# backend 'QtAgg' ... as 'headless' is currently running"), and a mid-session
# import flipped the whole remaining offscreen suite from Agg to QtAgg.
# pyqtgraph binding safety on import is already covered by xdart/__init__'s
# setdefault pins.

# Default root logging level — INFO is what every wrangler log line
# currently uses, so basicConfig(INFO) is enough to surface them.
# The DEBUG line below opts specific loggers into more verbose output;
# the basicConfig must happen first so the handler's threshold is open.
if sys.stderr is not None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s',
    )
else:
    # pythonw (the Windows no-console interpreter, used by the Start-menu
    # shortcut) sets sys.stderr to None.  basicConfig would then build a
    # StreamHandler whose stream is None, and every root record would be
    # eaten by that broken handler's error path.  Skip the console handler
    # entirely and just open the root at INFO so records reach the rotating
    # file handler installed in run().
    logging.getLogger().setLevel(logging.INFO)

# Suppress pyFAI INFO logs (e.g. "No sensor configuration provided").
logging.getLogger('pyFAI').setLevel(logging.WARNING)
logging.getLogger('pyFAI.gui.matplotlib').setLevel(logging.ERROR)
# Suppress silx's "pyOpenCL has been imported but can't be used here"
# warning — OpenCL is optional and the message has no user action.
logging.getLogger('silx.opencl').setLevel(logging.ERROR)


def _resolve_log_file():
    """Discoverable, platform-appropriate path for the rotating log file.

    Overridable with ``XDART_LOG_FILE`` (full path) or ``XDART_LOG_DIR`` (dir).
    Defaults follow each OS's convention so the file is easy to find and tail:
    macOS ``~/Library/Logs/xdart``, Windows ``%LOCALAPPDATA%\\xdart\\Logs``,
    Linux ``$XDG_STATE_HOME/xdart`` (else ``~/.local/state/xdart``).
    """
    override = os.environ.get('XDART_LOG_FILE')
    if override:
        return Path(override).expanduser()
    base = os.environ.get('XDART_LOG_DIR')
    if base:
        d = Path(base).expanduser()
    elif sys.platform == 'darwin':
        d = Path.home() / 'Library' / 'Logs' / 'xdart'
    elif os.name == 'nt':
        d = Path(os.environ.get('LOCALAPPDATA') or Path.home()) / 'xdart' / 'Logs'
    else:
        d = Path(os.environ.get('XDG_STATE_HOME')
                 or (Path.home() / '.local' / 'state')) / 'xdart'
    return d / 'xdart.log'


#: Absolute path of the active rotating log file, set by
#: :func:`_install_file_log_handler` (None if it could not be created).  The
#: Help ▸ Open Log Location action and the startup banner read this.
LOG_FILE_PATH = None


def _install_file_log_handler():
    """Route root logging to a rotating file so output survives a windowless
    launch (``pixi global`` / the menuinst shortcut have no attached terminal,
    so the stderr StreamHandler goes nowhere).  Best-effort: any failure here
    must never stop the GUI from starting."""
    global LOG_FILE_PATH
    if LOG_FILE_PATH is not None:                 # idempotent
        return LOG_FILE_PATH
    try:
        path = _resolve_log_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s %(name)s: %(message)s'))
        logging.getLogger().addHandler(handler)
        LOG_FILE_PATH = path
    except Exception:
        logging.getLogger(__name__).warning(
            'could not open a log file for writing', exc_info=True)
    return LOG_FILE_PATH


#: Open handle of the faulthandler side file used on windowless launches.
#: Module global ON PURPOSE: faulthandler keeps writing to this fd for the
#: rest of the process, so the file object must never be garbage-collected
#: (a collected handle closes the fd and the next crash dump goes nowhere).
_FAULTHANDLER_FILE = None


def _stderr_has_fileno():
    """True when ``sys.stderr`` exists and exposes a real OS fd.

    pythonw (the Windows no-console interpreter) sets it to None; captured /
    in-memory stderrs (pytest capsys, StringIO) raise on ``fileno()``.  Both
    are unusable for faulthandler, which writes straight to the fd."""
    stderr = sys.stderr
    if stderr is None:
        return False
    try:
        stderr.fileno()
    except Exception:
        return False
    return True


def _enable_faulthandler():
    """Enable faulthandler (crash tracebacks + the SIGUSR1 stack dump) without
    EVER raising — a diagnostic must never stop the GUI from starting.

    Normal console: plain ``faulthandler.enable()`` to stderr, as before.
    Windowless launch (the Windows Start-menu shortcut runs ``pythonw.exe``,
    whose ``sys.stderr`` is None — bare ``enable()`` raises RuntimeError there
    and killed the app before the QApplication existed): dump to a dedicated
    ``faulthandler.log`` next to the rotating log file, truncated per launch.
    Deliberately NOT the RotatingFileHandler's own stream — rotation closes
    that fd underneath faulthandler.  No usable stderr and no log dir: skip.
    """
    global _FAULTHANDLER_FILE
    try:
        side_file = None                       # None -> dump to real stderr
        if not _stderr_has_fileno():
            if LOG_FILE_PATH is None:
                logger.debug(
                    "faulthandler disabled: no usable stderr and no log file")
                return
            side_file = open(Path(LOG_FILE_PATH).parent / 'faulthandler.log',
                             'w', encoding='utf-8')
        if side_file is None:
            faulthandler.enable()
        else:
            faulthandler.enable(file=side_file)
        if hasattr(signal, "SIGUSR1"):
            # Live-freeze diagnostic (POSIX-only): `kill -USR1 <pid>` dumps
            # every thread's Python stack — to stderr, or to the side file on
            # a windowless launch — even while the GUI thread is busy (the
            # dump runs in the C signal handler, no GIL needed).  No root
            # required, unlike py-spy on macOS.
            if side_file is None:
                faulthandler.register(signal.SIGUSR1, all_threads=True)
            else:
                faulthandler.register(signal.SIGUSR1, file=side_file,
                                      all_threads=True)
        old = _FAULTHANDLER_FILE
        _FAULTHANDLER_FILE = side_file
        if old is not None and old is not side_file:
            try:
                old.close()                    # re-enable (tests): drop the stale handle
            except Exception:
                pass
    except Exception:
        logger.debug("could not enable faulthandler", exc_info=True)


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
from xdart.gui.pages.catalog import (
    BUILTIN_PAGES,
    DEFAULT_PAGE_KEY,
    SCATTERING_WORKSPACE_PAGE,
)
from xdart.gui.pages.descriptors import PageDescriptor
from xdart.gui.pages.handle import validate_page_handle
from xdart.gui.pages.registry import PageRegistry
from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import (
    ActionCompleted,
    ActionRefused,
    CAPABILITY_UNAVAILABLE,
    CLEANUP_PENDING,
    EXIT_ONLY_PAGE,
    PAGE_ACTIVE,
    UNKNOWN_PAGE,
    PageCapability,
    PageCleanup,
    PageKey,
    PageLifecycle,
)
from xdart.gui.themes.typography import (
    FONT_SCALE_MENU,
    FONT_SCALE_SETTINGS_KEY,
    application_settings as _app_settings,
    capture_application_baseline,
    normalize_font_scale,
    resolve_font_scale,
)
from xdart.gui.themes.accent import (
    ACCENT_COLOR_MENU,
    ACCENT_COLOR_SETTINGS_KEY,
    normalize_accent_color,
    resolve_accent_color,
)
from xdart.gui.themes.spacing import (
    SPACING_MENU,
    SPACING_SETTINGS_KEY,
    normalize_spacing,
    resolve_spacing,
)
from xdart.gui.themes.corners import (
    CONTROLS_CARD_CORNERS_SETTINGS_KEY,
    normalize_controls_card_corners,
    resolve_controls_card_corners,
)


QMainWindow = QtWidgets.QMainWindow


def _resolve_theme(settings):
    """The saved theme, defaulting anything unrecognised to ``dark``."""
    theme = settings.value("theme", "dark")
    return theme if theme in ("dark", "light") else "dark"


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


class _StatusBarPresenter:
    def __init__(self, window):
        self._window = window

    def show(self, text, timeout_ms=0):
        self._window.statusBar().showMessage(text, timeout_ms)


class Main(QMainWindow):
    def __init__(
        self,
        *,
        page_descriptors=None,
        host_services=None,
        selected_page_key=None,
    ):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('xdart')
        self.ui.actionOpen.triggered.connect(self.openFile)
        self.ui.actionExit.triggered.connect(self.exit)
        self._menus_initialized = False
        self._close_retry_scheduled = False
        self._process_exit_requested = False

        descriptors = tuple(
            BUILTIN_PAGES if page_descriptors is None else page_descriptors)
        self.page_registry = PageRegistry(descriptors).freeze()
        default = self.page_registry.get(DEFAULT_PAGE_KEY)
        if not isinstance(default, PageDescriptor):
            default = next(
                (item for item in self.page_registry
                 if isinstance(item, PageDescriptor)),
                None,
            )
        if default is None:
            raise ValueError("the page catalog contains no page descriptor")
        self._default_page_key = default.key
        self._host_services = host_services or empty_host_services(
            _StatusBarPresenter(self))
        persisted = (
            selected_page_key if selected_page_key is not None
            else _app_settings().value("page.selected")
        )
        descriptor = self.page_registry.select(persisted, self._default_page_key)
        self._mount_page(descriptor)
        self._init_shortcut_actions()
        self._init_theme_menu()
        self._menus_initialized = True
        self._attach_application_menus()
        self._sync_capability_actions()

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

    def _mount_page(self, descriptor):
        services = self._host_services.for_page(descriptor.key)
        handle = descriptor.build(services, self)
        validate_page_handle(descriptor, handle)
        self.page_descriptor = descriptor
        self.page_handle = handle
        self.selected_page_key = descriptor.key
        # Compatibility name for consumers that need the mounted root widget;
        # host behavior routes only through the typed handle below.
        self.main_widget = handle.widget
        self.setCentralWidget(handle.widget)
        self.ui.statusbar.setHidden(
            descriptor.key == SCATTERING_WORKSPACE_PAGE.key
        )
        if self._menus_initialized:
            self._attach_application_menus()
            self._sync_capability_actions()

    def _init_theme_menu(self):
        """Build the one application-owned Config/Help action set."""
        settings = _app_settings()
        current = _resolve_theme(settings)
        font_scale = resolve_font_scale(settings)
        accent_color = resolve_accent_color(settings)
        spacing = resolve_spacing(settings)
        controls_card_corners = resolve_controls_card_corners(settings)
        self.host_config_menu = QtWidgets.QMenu("Config", self.ui.menubar)
        self.host_config_menu.setObjectName("menuConfig")
        self.host_help_menu = QtWidgets.QMenu("Help", self.ui.menubar)
        self.host_help_menu.setObjectName("menuHelp")
        self.ui.menubar.addAction(self.host_config_menu.menuAction())
        self.ui.menubar.addAction(self.host_help_menu.menuAction())
        self._config_separator = QtGui.QAction(self)
        self._config_separator.setSeparator(True)
        self.themeMenu = QtWidgets.QMenu("Theme", self)
        group = QtGui.QActionGroup(self)
        group.setExclusive(True)
        for name, label in (("dark", "Dark"), ("light", "Light")):
            action = QtGui.QAction(label, self)
            action.setCheckable(True)
            action.setChecked(name == current)
            action.triggered.connect(
                lambda _checked=False, n=name: self._set_theme(n))
            group.addAction(action)
            self.themeMenu.addAction(action)
        # ONE application-wide preference, five exclusive tiers.  The menu is
        # generated from the theme layer's table so a tier cannot exist in the
        # contract and be missing here (or vice versa).
        self.fontSizeMenu = QtWidgets.QMenu("Font Size", self)
        font_group = QtGui.QActionGroup(self)
        font_group.setExclusive(True)
        self.fontSizeActions = {}
        for scale, label in FONT_SCALE_MENU:
            action = QtGui.QAction(label, self)
            action.setCheckable(True)
            action.setChecked(scale == font_scale)
            action.triggered.connect(
                lambda _checked=False, s=scale:
                    self._set_application_font_size(s))
            font_group.addAction(action)
            self.fontSizeMenu.addAction(action)
            self.fontSizeActions[scale] = action
        self.accentColorMenu = QtWidgets.QMenu("Accent Color", self)
        accent_group = QtGui.QActionGroup(self)
        accent_group.setExclusive(True)
        self.accentColorActions = {}
        for choice, label in ACCENT_COLOR_MENU:
            action = QtGui.QAction(label, self)
            action.setCheckable(True)
            action.setChecked(choice == accent_color)
            action.triggered.connect(
                lambda _checked=False, value=choice:
                    self._set_accent_color(value)
            )
            accent_group.addAction(action)
            self.accentColorMenu.addAction(action)
            self.accentColorActions[choice] = action
        self.spacingMenu = QtWidgets.QMenu("Spacing", self)
        spacing_group = QtGui.QActionGroup(self)
        spacing_group.setExclusive(True)
        self.spacingActions = {}
        for choice, label in SPACING_MENU:
            action = QtGui.QAction(label, self)
            action.setCheckable(True)
            action.setChecked(choice == spacing)
            action.triggered.connect(
                lambda _checked=False, value=choice:
                    self._set_spacing(value)
            )
            spacing_group.addAction(action)
            self.spacingMenu.addAction(action)
            self.spacingActions[choice] = action
        self.actionRoundedCorners = QtGui.QAction(
            "Rounded Corners",
            self,
        )
        self.actionRoundedCorners.setCheckable(True)
        self.actionRoundedCorners.setChecked(controls_card_corners)
        self.actionRoundedCorners.triggered.connect(
            self._set_rounded_corners
        )
        self.debugMenu = QtWidgets.QMenu("Debug", self)
        self.actionDebugWindowState = QtGui.QAction("Window State", self)
        self.actionDebugWindowState.triggered.connect(self._log_window_state)
        self.debugMenu.addAction(self.actionDebugWindowState)
        self.actionCheckForUpdates = QtGui.QAction("Check for Updates…", self)
        self.actionCheckForUpdates.triggered.connect(self._check_for_updates)
        self.actionOpenLogLocation = QtGui.QAction("Open Log Location", self)
        self.actionOpenLogLocation.triggered.connect(self._open_log_location)
        self.application_config_actions = (
            self._config_separator,
            self.themeMenu.menuAction(),
            self.fontSizeMenu.menuAction(),
            self.accentColorMenu.menuAction(),
            self.spacingMenu.menuAction(),
            self.actionRoundedCorners,
            self.debugMenu.menuAction(),
        )
        self.application_help_actions = (
            self.actionCheckForUpdates,
            self.actionOpenLogLocation,
        )
        self._attached_config_menu = None
        self._attached_help_menu = None

    def _attach_application_menus(self):
        if PageCapability.APP_MENU_HOSTS in self.page_descriptor.capabilities:
            points = self.page_handle.app_menus.mount_points()
            config_menu, help_menu = points.config_menu, points.help_menu
        else:
            config_menu, help_menu = self.host_config_menu, self.host_help_menu
        try:
            if self._attached_config_menu is not None:
                for action in self.application_config_actions:
                    self._attached_config_menu.removeAction(action)
            if self._attached_help_menu is not None:
                for action in self.application_help_actions:
                    self._attached_help_menu.removeAction(action)
        except RuntimeError:
            pass  # a CLEAN page may already have destroyed its menu host
        for action in self.application_config_actions:
            config_menu.addAction(action)
            action.setEnabled(True)
        for action in self.application_help_actions:
            help_menu.addAction(action)
            action.setEnabled(True)
        self._attached_config_menu = config_menu
        self._attached_help_menu = help_menu

    def _open_log_location(self):
        """Help ▸ Open Log Location — reveal the rotating log file in the OS file
        manager (macOS Finder / Windows Explorer / Linux folder), falling back to
        a dialog that shows the path so it is always copyable + tail-able."""
        import subprocess
        path = LOG_FILE_PATH
        if path is None or not os.path.exists(path):
            QtWidgets.QMessageBox.information(
                self, "Log Location",
                "No log file has been created yet." if path is None
                else "Log file (not written yet):\n%s" % path)
            return
        try:
            p = str(path)
            if sys.platform == "darwin":
                subprocess.run(["open", "-R", p], check=False)
            elif os.name == "nt":
                subprocess.run(["explorer", "/select,", p], check=False)
            else:
                subprocess.run(["xdg-open", str(path.parent)], check=False)
        except Exception:
            QtWidgets.QMessageBox.information(self, "Log Location", str(path))

    # ── In-app updater (Help → Check for Updates…) — spec section 4 ───────────
    def _run_active(self, *, require_known=False):
        """Query selected-page activity without inspecting its widget."""
        if PageCapability.RUN_ACTIVITY not in self.page_descriptor.capabilities:
            return bool(require_known)
        try:
            return bool(self.page_handle.activity.active())
        except Exception:
            logger.exception("Selected page activity query failed")
            return True

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
        if self._run_active(require_known=True):
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
    def _top_level_widget_summary(cls, widget):
        name = widget.objectName() or "-"
        title = widget.windowTitle() or "-"
        return (
            f"{type(widget).__name__}(name={name!r}, title={title!r}, "
            f"size={cls._qsize_text(widget.size())})"
        )

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
        if PageCapability.LAYOUT_DIAGNOSTICS in self.page_descriptor.capabilities:
            try:
                logger.warning(
                    "Window State page: %s",
                    self.page_handle.diagnostics.describe_layout(),
                )
            except Exception:
                logger.exception("Selected page layout diagnostics failed")
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

    @staticmethod
    def _set_capability_actions(actions, enabled, reason):
        for action in actions:
            action.setEnabled(enabled)
            action.setToolTip("" if enabled else reason)
            action.setStatusTip("" if enabled else reason)

    def _sync_capability_actions(self):
        capabilities = self.page_descriptor.capabilities
        rows = (
            (PageCapability.OPEN_FOLDER, (self.ui.actionOpen,),
             "Open Folder is unavailable on this page."),
            (PageCapability.SETTINGS_PERSISTENCE,
             (self.actionLoadSettings, self.actionSaveSettings),
             "Settings persistence is unavailable on this page."),
            (PageCapability.RUN_CONTROL,
             (self.actionRunPause, self.actionStopRun),
             "Run control is unavailable on this page."),
            (PageCapability.WRITE_MODE_TOGGLE,
             (self.actionToggleWriteMode,),
             "Write-mode toggle is unavailable on this page."),
            (PageCapability.SLICE_PIN, (self.actionPinSliceCut,),
             "Slice pinning is unavailable on this page."),
        )
        for capability, actions, reason in rows:
            self._set_capability_actions(
                actions, capability in capabilities, reason)

    def _present_action_outcome(self, outcome):
        if isinstance(outcome, ActionRefused):
            self._host_services.status.show(outcome.reason, 5000)
        return outcome

    def _capability_absent(self):
        return self._present_action_outcome(
            ActionRefused(CAPABILITY_UNAVAILABLE))

    def _shortcut_run_pause(self):
        if PageCapability.RUN_CONTROL not in self.page_descriptor.capabilities:
            return self._capability_absent()
        return self._present_action_outcome(
            self.page_handle.run_control.run_pause())

    def _shortcut_stop(self):
        if PageCapability.RUN_CONTROL not in self.page_descriptor.capabilities:
            return self._capability_absent()
        return self._present_action_outcome(self.page_handle.run_control.stop())

    def _shortcut_toggle_write_mode(self):
        if PageCapability.WRITE_MODE_TOGGLE not in self.page_descriptor.capabilities:
            return self._capability_absent()
        return self._present_action_outcome(self.page_handle.write_mode.toggle())

    def _shortcut_pin_slice_cut(self):
        if PageCapability.SLICE_PIN not in self.page_descriptor.capabilities:
            return self._capability_absent()
        return self._present_action_outcome(self.page_handle.slice_pin.pin())

    def _shortcut_load_settings(self):
        if PageCapability.SETTINGS_PERSISTENCE not in self.page_descriptor.capabilities:
            return self._capability_absent()
        return self._present_action_outcome(self.page_handle.settings_io.load())

    def _shortcut_save_settings(self):
        if PageCapability.SETTINGS_PERSISTENCE not in self.page_descriptor.capabilities:
            return self._capability_absent()
        return self._present_action_outcome(self.page_handle.settings_io.save())

    def _apply_appearance(
        self,
        *,
        theme=None,
        font_scale=None,
        accent_color=None,
        spacing=None,
        controls_card_corners=None,
    ):
        """Persist one appearance change and apply the whole look ONCE.

        Whichever value changed, the other three are read back from settings
        and all four go into one ``apply_theme`` call.  A theme or font change
        therefore cannot reset the chosen toggle colour or spacing tier.
        """
        settings = _app_settings()
        if theme is not None:
            theme = theme if theme in ("dark", "light") else "dark"
            settings.setValue("theme", theme)
        else:
            theme = _resolve_theme(settings)
        if font_scale is not None:
            # Only ever persist an exact known tier: a malformed write would
            # otherwise be read back forever as a broken preference.
            font_scale = normalize_font_scale(font_scale)
            settings.setValue(FONT_SCALE_SETTINGS_KEY, font_scale)
        else:
            font_scale = resolve_font_scale(settings)
        if accent_color is not None:
            accent_color = normalize_accent_color(accent_color)
            settings.setValue(ACCENT_COLOR_SETTINGS_KEY, accent_color)
        else:
            accent_color = resolve_accent_color(settings)
        if spacing is not None:
            spacing = normalize_spacing(spacing)
            settings.setValue(SPACING_SETTINGS_KEY, spacing)
        else:
            spacing = resolve_spacing(settings)
        if controls_card_corners is not None:
            controls_card_corners = normalize_controls_card_corners(
                controls_card_corners
            )
            settings.setValue(
                CONTROLS_CARD_CORNERS_SETTINGS_KEY,
                controls_card_corners,
            )
        else:
            controls_card_corners = resolve_controls_card_corners(settings)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            from xdart.gui.themes import apply_theme
            apply_theme(
                app,
                theme,
                font_scale=font_scale,
                accent_color=accent_color,
                spacing=spacing,
                controls_card_corners=controls_card_corners,
            )
        return (
            theme,
            font_scale,
            accent_color,
            spacing,
            controls_card_corners,
        )

    def _set_theme(self, name):
        """Apply theme ``name`` live and persist the choice."""
        self._apply_appearance(theme=name)

    def _set_rounded_corners(self, rounded):
        """Apply panel/container corner treatment live and persist it."""
        self._apply_appearance(controls_card_corners=rounded)

    def _set_application_font_size(self, scale):
        """Apply the application-wide font tier live and persist the choice."""
        self._apply_appearance(font_scale=scale)

    def _set_accent_color(self, choice):
        """Apply the selected-control accent live and persist the choice."""
        self._apply_appearance(accent_color=choice)

    def _set_spacing(self, choice):
        """Apply application spacing live and persist the choice."""
        self._apply_appearance(spacing=choice)

    def select_page(self, key):
        target = self.page_registry.get(PageKey(str(key)))
        if not isinstance(target, PageDescriptor):
            return ActionRefused(UNKNOWN_PAGE)
        if target.key == self.selected_page_key:
            return ActionCompleted(str(target.key))
        if self.page_descriptor.lifecycle is PageLifecycle.EXIT_ONLY:
            return ActionRefused(EXIT_ONLY_PAGE)
        if (PageCapability.RUN_ACTIVITY in self.page_descriptor.capabilities
                and self._run_active()):
            return ActionRefused(PAGE_ACTIVE)
        receipt = self._close_page()
        if receipt.status is PageCleanup.PENDING:
            return ActionRefused(CLEANUP_PENDING)
        self._mount_page(target)
        _app_settings().setValue("page.selected", str(target.key))
        return ActionCompleted(str(target.key))

    def _close_page(self):
        try:
            return self.page_handle.close()
        except Exception as exc:
            logger.exception("Selected page close failed")
            from xdart.gui.pages.values import CloseReceipt
            return CloseReceipt(PageCleanup.PENDING, f"close failed: {exc}")

    def closeEvent(self, event):
        receipt = self._close_page()
        if receipt.status is PageCleanup.PENDING:
            event.ignore()
            if not self._close_retry_scheduled:
                self._close_retry_scheduled = True
                QtCore.QTimer.singleShot(50, self._retry_close)
            return
        self._close_retry_scheduled = False
        event.accept()
        if self._process_exit_requested:
            QtCore.QTimer.singleShot(0, self._terminate_process)

    def _retry_close(self):
        self._close_retry_scheduled = False
        self.close()

    def _terminate_process(self):
        gc.collect()
        # os.killpg is POSIX-only; on Windows it raises AttributeError.
        if hasattr(os, "killpg"):
            try:
                os.killpg(os.getpid(), signal.SIGTERM)
            except ProcessLookupError:
                pass
        sys.exit(0)

    def exit(self):
        self._process_exit_requested = True
        self.close()

    def openFile(self):
        if PageCapability.OPEN_FOLDER not in self.page_descriptor.capabilities:
            return self._capability_absent()
        return self._present_action_outcome(
            self.page_handle.open_folder.request())


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


def _start_gui(app, window_factory=None):
    """Apply the saved appearance, THEN build and show the main window.

    The order is the whole point of this function existing, so it is worth one
    place that owns it: pyqtgraph snapshots its config (plot backgrounds) at
    widget creation, and a widget built before ``QApplication.setFont`` keeps
    the platform default until something else forces a re-polish.  Both mean
    the saved theme AND the saved font tier must land before the first widget.

    Appearance failure must never stop the GUI from starting -- a user with a
    corrupt preferences file gets Qt's default look, not a dead launcher.
    """
    try:
        # Snapshot the platform baseline while the font is still pristine:
        # every tier is derived from it, and after the first apply it is gone.
        capture_application_baseline(app)
        from xdart.gui.themes import apply_theme
        settings = _app_settings()
        apply_theme(
            app,
            _resolve_theme(settings),
            resolve_font_scale(settings),
            accent_color=resolve_accent_color(settings),
            spacing=resolve_spacing(settings),
            controls_card_corners=resolve_controls_card_corners(settings),
        )
    except Exception:
        logger.exception("Failed to apply the saved appearance; using Qt default")
    mw = (window_factory or Main)()
    mw.show()
    return mw


def run():
    # Process-global state claimed HERE, at real GUI launch — never on import.
    # File logging FIRST: a windowless launch (the pythonw Start-menu
    # shortcut, pixi global) has no terminal, so everything below — the
    # faulthandler side-file fallback and any startup traceback — must land
    # in the rotating file, which exists for exactly this case.
    log_path = _install_file_log_handler()
    if log_path is not None:
        logger.info("xdart logging to %s", log_path)
    # Python traceback on bus error / segfault (+ the SIGUSR1 stack dump).
    # Never raises: under pythonw sys.stderr is None and a bare
    # faulthandler.enable() would RuntimeError before the QApplication exists
    # (the v1.0.0/v1.0.1 Start-menu flash-and-close crash).
    _enable_faulthandler()
    # Hard-pin the Qt binding for pyqtgraph and export MPLBACKEND so child
    # processes (e.g. pyFAI-calib2) inherit the Qt6 backend.
    os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'
    os.environ['MPLBACKEND'] = 'QtAgg'
    # QtAgg to match pyqtgraph's binding; set before any pyplot figure exists.
    import matplotlib
    matplotlib.use('QtAgg')
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
    _start_gui(app)
    app.exec()


main = run   # back-compat alias


if __name__ == '__main__':
    sys.exit(run())
