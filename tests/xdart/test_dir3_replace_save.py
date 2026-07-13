# -*- coding: utf-8 -*-
"""DIR-3 (bl17-2, Windows/SMB): the replace-save's ``os.replace`` failed with
WinError 5 (destination held open by another handle) and the PermissionError
escaped ``run()``, killing the whole directory run.

Three fixes under test: a bounded retry around the rename (the only cover for
OUT-of-process holders — antivirus/indexer/SMB clients), h5pool key
normalization (an in-process case/slash-variant key silently defeated
``pause()``), and the clean-stop delivery so nothing escapes the QThread.
The Windows sharing violation itself cannot be reproduced on POSIX, so the OS
boundary (``os.replace``) is fault-injected; everything else is the real
code path."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest


def test_replace_retry_survives_transient_lock(tmp_path, monkeypatch):
    from xdart.modules.ewald import nexus_writer

    monkeypatch.setattr(nexus_writer, "_REPLACE_RETRY_DELAY_S", 0.0)
    src = tmp_path / "new.tmp"
    dst = tmp_path / "out.nxs"
    src.write_bytes(b"NEW")
    dst.write_bytes(b"OLD")

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(a, b):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(13, "Access is denied", str(b), 5)
        return real_replace(a, b)

    monkeypatch.setattr(nexus_writer.os, "replace", flaky_replace)
    nexus_writer._replace_with_retry(src, dst)

    assert calls["n"] == 3
    assert dst.read_bytes() == b"NEW"
    assert not src.exists()


def test_replace_retry_fails_loud_and_preserves_destination(tmp_path, monkeypatch):
    from xdart.modules.ewald import nexus_writer

    monkeypatch.setattr(nexus_writer, "_REPLACE_RETRY_DELAY_S", 0.0)
    src = tmp_path / "new.tmp"
    dst = tmp_path / "out.nxs"
    src.write_bytes(b"NEW")
    dst.write_bytes(b"OLD")

    def stuck_replace(a, b):
        raise PermissionError(13, "Access is denied", str(b), 5)

    monkeypatch.setattr(nexus_writer.os, "replace", stuck_replace)
    with pytest.raises(PermissionError) as excinfo:
        nexus_writer._replace_with_retry(src, dst)

    # Actionable message + untouched destination.
    assert "held open by another program" in str(excinfo.value)
    assert dst.read_bytes() == b"OLD"


def test_h5pool_key_variants_share_one_handle(tmp_path):
    """A case/slash/relative variant of one path must hit the SAME pool slot,
    so pause() always closes the cached read handle before a write."""
    import h5py
    from xdart.utils.h5pool import H5FilePool

    p = tmp_path / "scan.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("x", data=np.arange(3))

    pool = H5FilePool()
    messy = str(tmp_path / "." / "scan.h5")          # separator variant
    handle = pool.get(messy)
    assert handle is not None

    pool.pause(str(p))                               # clean spelling
    try:
        assert not handle.id.valid, (
            "pause() missed the handle cached under a path variant "
            "(the DIR-3 in-process hole)")
        assert pool.get(messy) is None               # paused under one key
    finally:
        pool.resume(str(p))
    assert pool.get(str(p)) is not None
    pool.close_all()


def test_initialize_scan_write_error_stops_cleanly():
    """The clean-stop delivery: command='stop' + a user-visible message,
    mirroring the AppendConfigMismatch pattern (nothing escapes run())."""
    import types
    from types import MethodType

    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        imageThread,
    )

    emitted = []
    host = types.SimpleNamespace(
        command="continue",
        showLabel=types.SimpleNamespace(emit=emitted.append),
    )
    handler = MethodType(imageThread._handle_initialize_scan_write_error, host)

    handler(PermissionError(13, "Access is denied", "X:\\out.nxs", 5))

    assert host.command == "stop"
    assert emitted and "locked by another program" in emitted[0]
