# -*- coding: utf-8 -*-
"""R2 Commit 2 — lazy Bluesky metadata provider.

Proves per-frame rows match the existing ``xrd_tools.io.bluesky_nexus`` helpers,
that descriptor/frame-count/first-read do NOT eagerly build the full table, that
repeated ``metadata_for`` reads reuse the provider and reopen no master, and
that a plain NeXus stack keeps sidecar behavior (empty provider).
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from xrd_tools.sources.cursor import ContainerCursor
from xrd_tools.sources.metadata_provider import (
    BlueskyMetadataProvider,
    EmptyMetadataProvider,
)

NF = 5
IMG = (8, 8)


def _bluesky(path, *, finalized=True):
    hy = np.linspace(11.0, 11.6, NF).astype(np.float64)  # scanned motor
    i0 = np.linspace(100.0, 110.0, NF).astype(np.float64)  # counter
    epoch = np.linspace(1000.0, 1004.0, NF).astype(np.float64)
    with h5py.File(path, "w") as f:
        f.attrs["creator"] = "NXWriter"
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        inst = entry.create_group("instrument")
        inst.attrs["NX_class"] = "NXinstrument"
        bl = inst.create_group("bluesky")
        bl.attrs["NX_class"] = "NXnote"
        md = bl.create_group("metadata")
        md.create_dataset("motors", data=b"!!python/tuple\n- hy\n")
        pos = inst.create_group("positioners")
        pos.attrs["NX_class"] = "NXnote"
        hg = pos.create_group("hy")
        hg.attrs["NX_class"] = "NXpositioner"
        hg.create_dataset("value", data=hy)
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data.attrs["signal"] = "eiger_image"
        data.create_dataset("hy", data=hy)
        data.create_dataset("i0", data=i0)
        data.create_dataset("EPOCH", data=epoch)
        imgs = np.arange(NF * IMG[0] * IMG[1], dtype=np.uint32).reshape((NF, *IMG))
        img = data.create_dataset("eiger_image", data=imgs, chunks=(2, *IMG))
        img.attrs["signal_type"] = "detector"
        if finalized:
            entry.create_dataset("end_time", data=np.bytes_("2026-07-15T00:00:00"))
    return path, hy, i0, epoch


def _plain(path):
    with h5py.File(path, "w") as f:
        e = f.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        det = e.create_group("instrument/detector")
        det.create_dataset("data", data=np.zeros((4, 8, 8), np.uint16), chunks=(2, 8, 8))
    return path


def _count_h5_opens(monkeypatch) -> list:
    opens: list = []
    orig = h5py.File.__init__

    def counting(self, name, *a, **k):
        opens.append(str(name))
        return orig(self, name, *a, **k)

    monkeypatch.setattr(h5py.File, "__init__", counting)
    return opens


# --------------------------------------------------------------------------- #
# per-frame rows match the Bluesky helpers
# --------------------------------------------------------------------------- #
def test_provider_matches_bluesky_helpers(tmp_path):
    from xrd_tools.io.bluesky_nexus import (
        bluesky_angles,
        bluesky_per_frame_table,
        resolve_nxentry,
    )

    p, hy, i0, epoch = _bluesky(tmp_path / "bl.nxs")
    with ContainerCursor(p) as cur:
        provider = cur.metadata_provider()
        assert isinstance(provider, BlueskyMetadataProvider)
        # scanned motors surface as whole-array columns
        np.testing.assert_array_equal(provider.motors()["hy"], hy)
        # per-frame row: counters + EPOCH, motor excluded
        md1 = provider.metadata_for(1)
        assert "hy" not in md1
        assert md1["i0"] == pytest.approx(i0[1])
        assert md1["EPOCH"] == pytest.approx(epoch[1])
        assert all(isinstance(v, float) for v in md1.values())

    # independently recompute from the raw helpers and compare
    with h5py.File(p, "r") as f:
        entry = resolve_nxentry(f, "entry")
        raw_motors = bluesky_angles(entry)
        raw_table = bluesky_per_frame_table(entry)
    np.testing.assert_array_equal(raw_motors["hy"], hy)
    assert raw_table["i0"][1] == pytest.approx(i0[1])


def test_out_of_range_frame_returns_empty(tmp_path):
    p, *_ = _bluesky(tmp_path / "range.nxs")
    with ContainerCursor(p) as cur:
        provider = cur.metadata_provider()
        assert provider.metadata_for(999) == {}
        assert provider.metadata_for(-1) == {}


# --------------------------------------------------------------------------- #
# laziness: descriptor / frame count / first read build no table
# --------------------------------------------------------------------------- #
def test_descriptor_and_first_read_do_not_build_table(tmp_path):
    p, *_ = _bluesky(tmp_path / "lazy.nxs")
    with ContainerCursor(p) as cur:
        _ = cur.descriptor
        _ = cur.frame_count
        _ = cur.read_frame(0)
        # provider not even constructed yet
        assert cur._provider is None  # noqa: SLF001
        provider = cur.metadata_provider()
        assert isinstance(provider, BlueskyMetadataProvider)
        assert provider._table is None  # noqa: SLF001 — still not materialized
        _ = provider.metadata_for(0)   # first metadata read materializes it
        assert provider._table is not None  # noqa: SLF001


def test_repeated_metadata_reads_open_no_new_master(tmp_path, monkeypatch):
    p, *_ = _bluesky(tmp_path / "reuse.nxs")
    opens = _count_h5_opens(monkeypatch)
    with ContainerCursor(p) as cur:
        provider = cur.metadata_provider()
        for i in range(NF):
            _ = provider.metadata_for(i)
        _ = provider.motors()
        _ = provider.scan_table()
    assert opens.count(str(p)) == 1  # one master open for the whole cursor life


def test_materialized_provider_survives_cursor_close(tmp_path):
    p, hy, i0, _ = _bluesky(tmp_path / "survive.nxs")
    with ContainerCursor(p) as cur:
        provider = cur.metadata_provider()
        provider.scan_table()  # materialize while the handle is open
    # cursor closed; provider now reads only memory
    assert cur.closed is True
    np.testing.assert_array_equal(provider.motors()["hy"], hy)
    assert provider.metadata_for(2)["i0"] == pytest.approx(i0[2])


def test_wavelength_available_without_table_build(tmp_path):
    p, *_ = _bluesky(tmp_path / "wl.nxs")
    with ContainerCursor(p) as cur:
        provider = cur.metadata_provider()
        # wavelength does not require the per-frame table
        _ = provider.wavelength()
        assert provider._table is None  # noqa: SLF001


# --------------------------------------------------------------------------- #
# plain NeXus stack keeps sidecar behavior (empty provider)
# --------------------------------------------------------------------------- #
def test_provider_read_after_close_before_materialize_fails_clearly(tmp_path):
    """If the owning cursor closes before the provider materializes, a later
    read must FAIL CLEARLY (not silently return empty) — handoff §4.2.5."""
    from xrd_tools.sources.metadata_provider import MetadataSourceClosedError

    p, *_ = _bluesky(tmp_path / "close_before_mat.nxs")
    cur = ContainerCursor(p).open()
    provider = cur.metadata_provider()  # obtained but NOT materialized
    cur.close()
    with pytest.raises(MetadataSourceClosedError):
        provider.metadata_for(0)


def test_plain_stack_gets_empty_provider(tmp_path):
    p = _plain(tmp_path / "plain.nxs")
    with ContainerCursor(p) as cur:
        provider = cur.metadata_provider()
        assert isinstance(provider, EmptyMetadataProvider)
        assert provider.metadata_for(0) == {}
        assert provider.motors() == {}


def test_nexus_stack_source_metadata_via_provider(tmp_path):
    """NexusStackSource.metadata_for / .motors now consume the provider and
    preserve the exact prior semantics."""
    from xrd_tools.sources.nexus import NexusStackSource

    p, hy, i0, epoch = _bluesky(tmp_path / "src.nxs")
    src = NexusStackSource(p)
    np.testing.assert_array_equal(src.motors["hy"], hy)
    md0 = src.metadata_for(0)
    assert md0["i0"] == pytest.approx(i0[0])
    assert "hy" not in md0

    plain = _plain(tmp_path / "plain_src.nxs")
    psrc = NexusStackSource(plain)
    assert psrc.motors == {}
    assert psrc.metadata_for(0) == {}
