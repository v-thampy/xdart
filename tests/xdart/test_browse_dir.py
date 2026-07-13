# -*- coding: utf-8 -*-
"""BD-1: the shared, session-persisted "last browsed directory".

Rule (maintainer, 2026-07-13): every Browse dialog opens in the directory of
the LAST successful pick — any field, any wrangler, any mode — persisted
across sessions; fallbacks are the current field's folder, then the caller's
fallback (e.g. Project Folder).  The rule lives in ``xdart.utils.browse``;
dialog call sites seed from ``browse_start_dir`` and record with
``remember_browse_path``."""

import os
import types
from types import MethodType

import pytest


@pytest.fixture()
def session_file(tmp_path, monkeypatch):
    p = tmp_path / "session.json"
    monkeypatch.setenv("XDART_SESSION_FILE", str(p))
    monkeypatch.delenv("XDART_SESSION_FRESH", raising=False)
    return p


def test_remember_and_seed_round_trip(session_file, tmp_path):
    from xdart.utils.browse import browse_start_dir, remember_browse_path

    picked_dir = tmp_path / "beamtime" / "run42"
    picked_dir.mkdir(parents=True)
    picked_file = picked_dir / "image_0001.tif"
    picked_file.write_bytes(b"")

    # A FILE pick records its parent directory...
    remember_browse_path(str(picked_file))
    assert browse_start_dir() == str(picked_dir)
    # ...and persists (fresh read of the session file, not module state).
    from xdart.utils.session import load_session
    assert load_session()["last_browse_dir"] == str(picked_dir)

    # A DIRECTORY pick records itself.
    other = tmp_path / "elsewhere"
    other.mkdir()
    remember_browse_path(str(other))
    assert browse_start_dir() == str(other)


def test_seed_precedence_last_then_current_then_fallback(session_file, tmp_path):
    from xdart.utils.browse import browse_start_dir, remember_browse_path

    last = tmp_path / "last"
    cur = tmp_path / "cur"
    fb = tmp_path / "fb"
    for d in (last, cur, fb):
        d.mkdir()

    # Nothing recorded: current field's folder wins over the fallback.
    assert browse_start_dir(str(cur / "a.poni"), str(fb)) == str(cur)
    # No current either: fallback.
    assert browse_start_dir("", str(fb)) == str(fb)
    # Nothing at all: '' (Qt decides).
    assert browse_start_dir("", "") == ""

    # A recorded pick beats both (the maintainer rule).
    remember_browse_path(str(last))
    assert browse_start_dir(str(cur / "a.poni"), str(fb)) == str(last)


def test_vanished_directory_seeds_surviving_parent(session_file, tmp_path):
    from xdart.utils.browse import browse_start_dir, remember_browse_path

    gone = tmp_path / "beamtime" / "run42"
    gone.mkdir(parents=True)
    remember_browse_path(str(gone))
    gone.rmdir()

    # A vanished last dir falls back to its surviving parent (deliberate:
    # a renamed run folder still lands you in the right neighbourhood).
    assert browse_start_dir("", "") == str(tmp_path / "beamtime")

    # Recording a path with no existing directory anywhere near it is a no-op
    # (keeps the prior value).
    remember_browse_path(str(tmp_path / "nowhere" / "deep" / "y.tif"))
    assert browse_start_dir("", "") == str(tmp_path / "beamtime")


def test_suggest_save_path_joins_start_dir(session_file, tmp_path):
    from xdart.utils.browse import remember_browse_path, suggest_save_path

    # Unset: the bare filename (Qt decides the directory).
    assert suggest_save_path("table.csv") == "table.csv"
    d = tmp_path / "outputs"
    d.mkdir()
    remember_browse_path(str(d))
    assert suggest_save_path("table.csv") == os.path.join(str(d), "table.csv")


def test_image_wrangler_browse_dir_prefers_last_pick(session_file, tmp_path, qapp):
    """Production seam: imageWrangler._browse_dir consults the shared rule —
    last pick first, then the current field's folder, then Project Folder."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler
    from xdart.utils.browse import remember_browse_path

    project = tmp_path / "project"
    field = tmp_path / "fieldhome"
    last = tmp_path / "lastpick"
    for d in (project, field, last):
        d.mkdir()
    poni = field / "cal.poni"
    poni.write_text("")

    host = types.SimpleNamespace(project_folder=str(project))
    browse_dir = MethodType(imageWrangler._browse_dir, host)

    # No shared pick yet: current field's folder, then Project Folder.
    assert browse_dir(str(poni)) == str(field)
    assert browse_dir("") == str(project)

    remember_browse_path(str(last))
    assert browse_dir(str(poni)) == str(last)
    assert browse_dir("") == str(last)


@pytest.fixture(scope="module")
def qapp():
    from pyqtgraph.Qt import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
