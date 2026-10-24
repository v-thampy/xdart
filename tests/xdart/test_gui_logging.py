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


def test_debug_logging_switch_writes_debug_records_mid_session(monkeypatch, tmp_path):
    """Config ▸ Debug ▸ Debug Logging (1.2.4): DEBUG reaches the log file at once and off restores INFO."""
    import os

    from xdart import _gui_main
    monkeypatch.delenv("XDART_LOG_FILE", raising=False)
    monkeypatch.setenv("XDART_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(_gui_main, "LOG_FILE_PATH", None)
    monkeypatch.setattr(_gui_main, "_FILE_LOG_HANDLER", None)
    for flag in _gui_main.DEBUG_LOGGING_FLAGS:
        monkeypatch.delenv(flag, raising=False)
    for name in ("xdart", "xrd_tools"):
        monkeypatch.setattr(logging.getLogger(name), "level", logging.getLogger(name).level)
    _clear_file_handlers(_gui_main)
    try:
        path = _gui_main._install_file_log_handler()
        probe = logging.getLogger("xrd_tools.io.metadata")
        probe.debug("before the switch")
        _gui_main.set_debug_logging(True)
        assert all(os.environ.get(flag) == "1" for flag in _gui_main.DEBUG_LOGGING_FLAGS)
        probe.debug("while debugging")
        _gui_main.set_debug_logging(False)
        assert not any(flag in os.environ for flag in _gui_main.DEBUG_LOGGING_FLAGS)
        probe.debug("after the switch")
        _gui_main._FILE_LOG_HANDLER.flush()
        text = path.read_text(encoding="utf-8")
        assert "while debugging" in text
        assert "before the switch" not in text and "after the switch" not in text
        assert "Debug logging on" in text and "Debug logging off" in text
    finally:
        _clear_file_handlers(_gui_main)


def test_performance_monitoring_switch_logs_diagnostics_and_times_collections_while_on(monkeypatch, tmp_path):
    """Config ▸ Debug ▸ Performance Monitoring (maintainer, 2026-10-24): XDART_PERF on at once, so a Run's
    diagnostics reach the regular log, with garbage-collection timing; off clears it and the quartile timing."""
    import gc
    import os

    from xdart import _gui_main
    from xdart.utils import log_events
    monkeypatch.delenv("XDART_LOG_FILE", raising=False)
    monkeypatch.setenv("XDART_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(_gui_main, "LOG_FILE_PATH", None)
    monkeypatch.setattr(_gui_main, "_FILE_LOG_HANDLER", None)
    for flag in _gui_main.PERFORMANCE_FLAGS:
        monkeypatch.delenv(flag, raising=False)
    _clear_file_handlers(_gui_main)
    root, previous = logging.getLogger(), logging.getLogger().level
    root.setLevel(logging.INFO)  # the app's root level (basicConfig INFO); setLevel clears the loggers' caches
    probe = logging.getLogger("xdart.gui.tabs.scattering.adapters.run_executor")
    try:
        path = _gui_main._install_file_log_handler()
        log_events.diagnostic(probe, "[PERF-SUMMARY] before the switch")
        _gui_main.set_performance_monitoring(True)
        assert os.environ.get("XDART_PERF") == "1" and log_events.performance_monitoring()
        assert gc.callbacks.count(_gui_main._time_garbage_collections) == 1
        _gui_main.set_performance_monitoring(True)  # once registered, however often it is turned on
        assert gc.callbacks.count(_gui_main._time_garbage_collections) == 1
        log_events.diagnostic(probe, "[PERF-SUMMARY] while monitoring")
        os.environ["XDART_PERF_QUARTILES"] = "1"  # Config ▸ Performance Diagnostics…'s quartile timing
        _gui_main.set_performance_monitoring(False)
        assert not any(flag in os.environ for flag in _gui_main.PERFORMANCE_FLAGS)
        assert _gui_main._time_garbage_collections not in gc.callbacks
        log_events.diagnostic(probe, "[PERF-SUMMARY] after the switch")
        _gui_main._FILE_LOG_HANDLER.flush()
        text = path.read_text(encoding="utf-8")
        assert "while monitoring" in text and "before the switch" not in text and "after the switch" not in text
        assert "Performance monitoring on" in text and "Performance monitoring off" in text
    finally:
        _gui_main.set_performance_monitoring(False)
        root.setLevel(previous)
        _clear_file_handlers(_gui_main)
