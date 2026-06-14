# -*- coding: utf-8 -*-
"""CLI session options (``xdart -f`` / ``xdart -n NAME``) + the session
module's fresh/ephemeral behaviour and forced ``.json`` extension."""
import json
import os

import pytest

from xdart.utils import session as S


def test_load_reads_default_when_not_fresh(tmp_path, monkeypatch):
    f = tmp_path / "session.json"
    f.write_text(json.dumps({"a": 1}))
    monkeypatch.setenv("XDART_SESSION_FILE", str(f))
    monkeypatch.delenv("XDART_SESSION_FRESH", raising=False)
    assert S.load_session() == {"a": 1}


def test_fresh_session_loads_empty_and_save_is_noop(tmp_path, monkeypatch):
    """``-f``: the saved session is neither read nor clobbered."""
    f = tmp_path / "session.json"
    f.write_text(json.dumps({"a": 1}))
    monkeypatch.setenv("XDART_SESSION_FILE", str(f))
    monkeypatch.setenv("XDART_SESSION_FRESH", "1")
    assert S.load_session() == {}            # ignores existing state
    S.save_session({"b": 2})                  # persists nothing
    assert json.loads(f.read_text()) == {"a": 1}   # file unchanged


def test_explicit_path_is_honoured_even_when_fresh(tmp_path, monkeypatch):
    """An explicit ``path=`` (or ``-n`` redirect via XDART_SESSION_FILE) is the
    named session and must be read/written even under the fresh flag."""
    f = tmp_path / "named.json"
    f.write_text(json.dumps({"x": 9}))
    monkeypatch.setenv("XDART_SESSION_FRESH", "1")
    assert S.load_session(path=str(f)) == {"x": 9}


def test_cli_fresh_flag(monkeypatch):
    from xdart._gui_main import _apply_cli_session_args
    monkeypatch.delenv("XDART_SESSION_FRESH", raising=False)
    monkeypatch.delenv("XDART_SESSION_FILE", raising=False)
    rest = _apply_cli_session_args(["xdart", "-f", "--", "extra"])
    assert os.environ.get("XDART_SESSION_FRESH") == "1"
    assert "XDART_SESSION_FILE" not in os.environ
    assert rest[0] == "xdart"               # argv[0] preserved for Qt


def test_cli_named_session_forces_json_under_xdart_dir(monkeypatch):
    from xdart._gui_main import _apply_cli_session_args
    monkeypatch.delenv("XDART_SESSION_FRESH", raising=False)
    monkeypatch.delenv("XDART_SESSION_FILE", raising=False)
    _apply_cli_session_args(["xdart", "-n", "mysetup"])      # no .json
    p = os.environ["XDART_SESSION_FILE"]
    assert p.endswith(os.path.join(".xdart", "mysetup.json"))
    assert "XDART_SESSION_FRESH" not in os.environ


def test_cli_named_session_keeps_explicit_json_and_path(tmp_path, monkeypatch):
    from xdart._gui_main import _apply_cli_session_args
    monkeypatch.delenv("XDART_SESSION_FILE", raising=False)
    explicit = str(tmp_path / "run42.json")
    _apply_cli_session_args(["xdart", "-n", explicit])
    assert os.environ["XDART_SESSION_FILE"] == explicit


def test_cli_fresh_and_named_are_mutually_exclusive():
    from xdart._gui_main import _apply_cli_session_args
    with pytest.raises(SystemExit):
        _apply_cli_session_args(["xdart", "-f", "-n", "x"])
