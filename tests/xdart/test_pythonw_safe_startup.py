"""pythonw-safe startup: the Windows Start-menu shortcut runs
``pythonw.exe -m xdart.xdart_main`` (recipe/menu/xdart.json), and under
pythonw ``sys.stdout``/``sys.stderr`` are None.  v1.0.0/v1.0.1 died on the
FIRST statement of run() — a bare ``faulthandler.enable()`` raises
``RuntimeError: sys.stderr is None`` — before the QApplication existed
(symptom: the shortcut flashes and closes).  pythonw itself is Windows-only,
but ``sys.stderr = None`` reproduces the failure mode on any platform, so
these tests run everywhere.

NB: ``_gui_main`` is imported INSIDE each test, never at module level, same
as every other _gui_main test (test_gui_logging, test_updater,
test_cli_session) — importing it pulls the whole Qt stack, which must not
happen at headless collection time.

Run with ``-p no:faulthandler`` (like the rest of tests/xdart) so pytest's
own faulthandler plugin does not fight over the enable/disable state.
"""
import os
import subprocess
import sys

import faulthandler


def _teardown_faulthandler(_gui_main):
    """Restore process-global faulthandler state and drop the side-file handle."""
    import signal as _signal
    if hasattr(_signal, "SIGUSR1"):
        try:
            faulthandler.unregister(_signal.SIGUSR1)
        except Exception:
            pass
    faulthandler.disable()
    f = _gui_main._FAULTHANDLER_FILE
    if f is not None:
        try:
            f.close()
        except Exception:
            pass
        _gui_main._FAULTHANDLER_FILE = None


def test_side_file_mode_when_stderr_is_none(monkeypatch, tmp_path):
    """pythonw simulation with a usable log dir: no raise, faulthandler comes
    up in side-file mode (``faulthandler.log`` next to the rotating log)."""
    from xdart import _gui_main
    faulthandler.disable()
    monkeypatch.setattr(_gui_main, "LOG_FILE_PATH", tmp_path / "xdart.log")
    monkeypatch.setattr(sys, "stderr", None)
    try:
        _gui_main._enable_faulthandler()          # must not raise
        assert faulthandler.is_enabled()
        side = tmp_path / "faulthandler.log"
        assert side.exists()
        handle = _gui_main._FAULTHANDLER_FILE
        assert handle is not None and not handle.closed
        assert os.path.abspath(handle.name) == str(side)
    finally:
        _teardown_faulthandler(_gui_main)


def test_plain_enable_with_real_stderr(monkeypatch, tmp_path):
    """A stderr with a real fd takes the plain enable() path — no side file."""
    from xdart import _gui_main
    faulthandler.disable()
    monkeypatch.setattr(_gui_main, "LOG_FILE_PATH", tmp_path / "xdart.log")
    with open(os.devnull, "w") as devnull:       # real fileno, no console spam
        monkeypatch.setattr(sys, "stderr", devnull)
        try:
            _gui_main._enable_faulthandler()
            assert faulthandler.is_enabled()
            assert _gui_main._FAULTHANDLER_FILE is None
            assert not (tmp_path / "faulthandler.log").exists()
        finally:
            # Disable BEFORE devnull closes so no dump targets a dead fd.
            _teardown_faulthandler(_gui_main)


def test_off_when_no_stderr_and_no_log_dir(monkeypatch):
    """Worst case (no stderr, file logging never came up): faulthandler is
    simply off — never an exception."""
    from xdart import _gui_main
    faulthandler.disable()
    monkeypatch.setattr(_gui_main, "LOG_FILE_PATH", None)
    monkeypatch.setattr(sys, "stderr", None)
    try:
        _gui_main._enable_faulthandler()          # must not raise
        assert not faulthandler.is_enabled()
        assert _gui_main._FAULTHANDLER_FILE is None
    finally:
        _teardown_faulthandler(_gui_main)


def test_unwritable_side_file_target_never_raises(monkeypatch, tmp_path):
    """LOG_FILE_PATH set but its dir unusable (open() raises): swallowed,
    faulthandler stays off, the GUI keeps starting."""
    from xdart import _gui_main
    faulthandler.disable()
    blocker = tmp_path / "afile"
    blocker.write_text("x")                       # a FILE where a dir is needed
    monkeypatch.setattr(_gui_main, "LOG_FILE_PATH",
                        blocker / "nested" / "xdart.log")
    monkeypatch.setattr(sys, "stderr", None)
    try:
        _gui_main._enable_faulthandler()          # must not raise
        assert not faulthandler.is_enabled()
        assert _gui_main._FAULTHANDLER_FILE is None
    finally:
        _teardown_faulthandler(_gui_main)


def test_import_gui_main_with_none_streams_subprocess():
    """Import-time safety pin — the closest honest pythonw simulation: a fresh
    interpreter with BOTH std streams None must import xdart._gui_main (the
    module-level basicConfig, warnings filters, and the full Qt import chain)
    without dying."""
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("MPLBACKEND", "Agg")
    code = ("import sys; sys.stderr = None; sys.stdout = None; "
            "import xdart._gui_main")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, (
        "importing xdart._gui_main with None std streams exited "
        f"{proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
