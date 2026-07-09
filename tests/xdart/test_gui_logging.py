"""File-logging so a windowless launch (pixi global / menuinst shortcut) keeps
its output — the real _gui_main handler machinery, no fakes on the seam.

NB: `_gui_main` is imported INSIDE each test, never at module level. Importing it
runs ``matplotlib.use('QtAgg')`` at import time, which raises during a HEADLESS
test *collection* (no QApplication up yet, Agg already committed via conftest's
MPLBACKEND=Agg). Deferring to run time — when the Qt stack is already up — matches
every other _gui_main test (test_updater, test_static_controls, test_cli_session).
"""
import logging


def _clear_file_handlers(_gui_main):
    root = logging.getLogger()
    for h in [h for h in root.handlers
              if isinstance(h, _gui_main.RotatingFileHandler)]:
        root.removeHandler(h)
        h.close()


def test_resolve_log_file_honours_explicit_file(monkeypatch, tmp_path):
    from xdart import _gui_main
    target = tmp_path / "custom" / "my.log"
    monkeypatch.setenv("XDART_LOG_FILE", str(target))
    assert _gui_main._resolve_log_file() == target


def test_resolve_log_file_honours_dir_override(monkeypatch, tmp_path):
    from xdart import _gui_main
    monkeypatch.delenv("XDART_LOG_FILE", raising=False)
    monkeypatch.setenv("XDART_LOG_DIR", str(tmp_path))
    assert _gui_main._resolve_log_file() == tmp_path / "xdart.log"


def test_install_writes_and_is_idempotent(monkeypatch, tmp_path):
    from xdart import _gui_main
    monkeypatch.delenv("XDART_LOG_FILE", raising=False)
    monkeypatch.setenv("XDART_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(_gui_main, "LOG_FILE_PATH", None)
    _clear_file_handlers(_gui_main)
    root = logging.getLogger()
    prev_level = root.level
    root.setLevel(logging.INFO)          # production sets INFO via basicConfig
    try:
        p = _gui_main._install_file_log_handler()
        assert p == tmp_path / "xdart.log"
        logging.getLogger("xdart.selftest").info("marker-xyz-99")
        for h in root.handlers:
            h.flush()
        assert "marker-xyz-99" in p.read_text()

        # Second install must NOT add a duplicate handler.
        assert _gui_main._install_file_log_handler() == p
        n = sum(1 for h in root.handlers
                if isinstance(h, _gui_main.RotatingFileHandler))
        assert n == 1
    finally:
        root.setLevel(prev_level)
        _clear_file_handlers(_gui_main)
        monkeypatch.setattr(_gui_main, "LOG_FILE_PATH", None, raising=False)


def test_install_survives_an_unwritable_target(monkeypatch, tmp_path):
    from xdart import _gui_main
    # A bad path must never stop the GUI from launching: best-effort, returns None.
    blocker = tmp_path / "afile"
    blocker.write_text("x")
    monkeypatch.setenv("XDART_LOG_FILE", str(blocker / "nested" / "xdart.log"))
    monkeypatch.setattr(_gui_main, "LOG_FILE_PATH", None)
    _clear_file_handlers(_gui_main)
    try:
        assert _gui_main._install_file_log_handler() is None
    finally:
        _clear_file_handlers(_gui_main)
        monkeypatch.setattr(_gui_main, "LOG_FILE_PATH", None, raising=False)
