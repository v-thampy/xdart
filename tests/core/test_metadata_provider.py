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

from xrd_tools.core.energy import WavelengthUnit
from xrd_tools.sources.cursor import ContainerCursor
from xrd_tools.sources.metadata_provider import (
    BlueskyMetadataProvider,
    EmptyMetadataProvider,
    MetadataProvider,
    metadata_provider_for_open_entry,
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


ACQUISITION_FRAMES = 3


def _acquisition(path, *, external=False):
    """A non-Bluesky acquisition NeXus container (no NXWriter marker): images in entry/instrument/detector/data or
    linked from a split data file, with the CAL-HELD-BATCH-20261009 v1 supported metadata beside items that contract
    excludes or ignores. Returns what it resolves: motors in degrees, the energy in eV."""
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        if external:
            member = path.with_name(path.stem + "_data_000001.h5")
            with h5py.File(member, "w") as data_file:
                data_file.create_dataset("entry/data/data", data=np.ones((ACQUISITION_FRAMES, 8, 8), np.uint16),
                                         chunks=(1, 8, 8))
            entry.create_group("data")["data_000001"] = h5py.ExternalLink(member.name, "/entry/data/data")
        else:
            entry.create_dataset("instrument/detector/data", data=np.ones((ACQUISITION_FRAMES, 8, 8), np.uint16))
        data = entry.require_group("data")
        data.attrs["NX_class"] = "NXdata"
        data.attrs["axes"] = "tth"
        data.create_dataset("tth", data=[5.0, 6.0, 7.0])  # an NXdata axis: a motor

        def positioner(where, name, value, units=None):
            group = entry.require_group(f"{where}/positioners/{name}")
            group.attrs["NX_class"] = "NXpositioner"
            dataset = group.create_dataset("value", data=value)
            if units is not None:
                dataset.attrs["units"] = units

        positioner("sample", "eta", [0.11, 0.25, 0.37], "deg")  # wins the name clash below
        positioner("instrument", "eta", [9.0, 9.0, 9.0], "deg")
        positioner("instrument", "phi", np.radians([10.0, 20.0, 30.0]), "rad")  # converted to degrees
        positioner("instrument", "chi", 12.5, "degrees")  # 0-d: a constant
        positioner("instrument", "omega", [1.0, 1.5, 2.0])  # no units: degrees, the legacy assumption
        positioner("instrument", "bad", [1.0, 2.0, 3.0, 4.0, 5.0], "deg")  # 5 values for 3 frames: excluded
        positioner("instrument", "x", [0.1, 0.2, 0.3], "mm")  # not an angle: excluded
        scan = entry.create_group("scan_data")
        scan.create_dataset("i0", data=[100.0, 101.0, 102.0])  # a counter, never a motor
        scan.create_dataset("foo", data=[7.0, 8.0, 9.0])  # not a supported item: ignored
        energy = entry.require_group("instrument/monochromator").create_dataset("energy", data=[16.0, 16.0, 16.001])
        energy.attrs["units"] = "keV"
    return {"eta": [0.11, 0.25, 0.37], "phi": [10.0, 20.0, 30.0], "omega": [1.0, 1.5, 2.0], "tth": [5.0, 6.0, 7.0],
            "chi": 12.5, "i0": [100.0, 101.0, 102.0], "energy": [16000.0, 16000.0, 16001.0]}


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
# X1 Slice 3a0 (R3-P1): explicit provider wavelength-unit declaration
# --------------------------------------------------------------------------- #
def test_cursor_provider_declares_angstrom_unit(tmp_path):
    """The descriptor's `_read_wavelength` explicitly returns angstroms, so the
    cursor composition point declares ANGSTROM — no magnitude inference."""
    p, *_ = _bluesky(tmp_path / "wl_unit.nxs")
    with ContainerCursor(p) as cur:
        provider = cur.metadata_provider()
        assert provider.wavelength_unit() is WavelengthUnit.ANGSTROM
        # value semantics unchanged: still the as-provided angstrom value
        assert provider.wavelength() == cur.wavelength

    plain = _plain(tmp_path / "plain_unit.nxs")
    with ContainerCursor(plain) as cur:
        # plain stacks get the empty provider, still angstrom-declared
        assert cur.metadata_provider().wavelength_unit() \
            is WavelengthUnit.ANGSTROM


def test_direct_provider_construction_is_unknown_unit_by_default():
    """Direct generic construction remains unknown-unit unless the caller
    declares one; a value with no declared unit contributes no evidence."""
    assert MetadataProvider().wavelength_unit() is None
    assert EmptyMetadataProvider(wavelength=1.54).wavelength_unit() is None
    assert BlueskyMetadataProvider(
        None, frame_count=3, wavelength=1.54).wavelength_unit() is None

    declared = EmptyMetadataProvider(
        wavelength=1.54, wavelength_unit=WavelengthUnit.ANGSTROM)
    assert declared.wavelength_unit() is WavelengthUnit.ANGSTROM
    assert declared.wavelength() == pytest.approx(1.54)


def test_provider_factory_threads_wavelength_unit(tmp_path):
    p, *_ = _bluesky(tmp_path / "factory_unit.nxs")
    with h5py.File(p, "r") as f:
        provider = metadata_provider_for_open_entry(
            f["entry"], frame_count=NF, wavelength=1.033,
            wavelength_unit=WavelengthUnit.ANGSTROM)
        assert isinstance(provider, BlueskyMetadataProvider)
        assert provider.wavelength_unit() is WavelengthUnit.ANGSTROM
    none_provider = metadata_provider_for_open_entry(
        None, frame_count=0, wavelength=None,
        wavelength_unit=WavelengthUnit.METRE)
    assert isinstance(none_provider, EmptyMetadataProvider)
    assert none_provider.wavelength_unit() is WavelengthUnit.METRE


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


@pytest.mark.parametrize("external", [False, True])
def test_a_non_bluesky_acquisition_entry_serves_its_supported_metadata(tmp_path, external):
    """CAL-HELD-BATCH-20261009 (maintainer: "I want these things to be addressed in this batch"): a NeXus acquisition
    container without the NXWriter marker, here misleadingly called scan_master, serves its supported metadata
    through the existing provider seam, frame-aligned: per-frame motors in degrees from sample positioners (winning
    a clash), instrument positioners and NXdata axes; a 0-d positioner as a constant; counters and the energy (eV)
    per frame. Misaligned, non-angle and unsupported items are not served."""
    path = tmp_path / "scan_master.h5"
    expected = _acquisition(path, external=external)
    with ContainerCursor(path) as cur:
        provider = cur.metadata_provider()
        motors = provider.motors()
        assert sorted(motors) == ["eta", "omega", "phi", "tth"]
        for name, values in motors.items():
            np.testing.assert_allclose(values, expected[name], rtol=0, atol=1e-9)
        assert dict(provider.metadata_for(1)) == pytest.approx({"chi": 12.5, "i0": 101.0, "energy": 16000.0},
                                                               rel=0, abs=1e-9)
        assert dict(provider.complete_metadata_for(2)) == pytest.approx(
            {"eta": 0.37, "omega": 2.0, "phi": 30.0, "tth": 7.0, "chi": 12.5, "i0": 102.0, "energy": 16001.0},
            rel=0, abs=1e-9)


def test_a_vendor_eiger_master_has_no_scan_metadata_whatever_its_name(tmp_path):
    """CAL-HELD-BATCH-20261009: a detector-only DECTRIS master records images and detector configuration, no scan
    metadata (its sample goniometer and transformations are not supported locations), renamed or not."""
    from tests.core.test_container_descriptor import _vendor_eiger_master
    for name in ("scan_master.h5", "renamed.h5"):
        directory = tmp_path / name.replace(".", "_")
        directory.mkdir()
        master = _vendor_eiger_master(directory / name, [directory / "scan_data_000001.h5"])
        with ContainerCursor(master) as cur:
            provider = cur.metadata_provider()
            assert (provider.motors(), dict(provider.metadata_for(0))) == ({}, {})


def test_a_generic_provider_read_after_close_before_materialize_fails_clearly(tmp_path):
    """As the Bluesky provider (handoff §4.2.5): a non-Bluesky acquisition provider obtained but not materialized
    before its cursor closes fails clearly, and one materialized first keeps serving from memory."""
    from xrd_tools.sources.metadata_provider import MetadataSourceClosedError
    path = tmp_path / "acquisition.nxs"
    _acquisition(path)
    cur = ContainerCursor(path).open()
    provider = cur.metadata_provider()
    cur.close()
    with pytest.raises(MetadataSourceClosedError):
        provider.metadata_for(0)
    with ContainerCursor(path) as cur:
        provider = cur.metadata_provider()
        provider.scan_table()  # materialize while the handle is open
    assert provider.complete_metadata_for(1)["eta"] == pytest.approx(0.25, rel=0, abs=1e-12)


@pytest.mark.parametrize("field,units,value,expected", [
    ("energy", "eV", 16000.0, 16000.0),
    ("energy", None, 16.0, 16000.0),  # below 1000: keV
    ("energy", None, 16000.0, 16000.0),  # 1000 or more: eV
    ("energy", "keV", 16000.0, 16000.0),  # a mislabel cannot mislead: the size decides
    ("energy", "eV", 16.0, 16000.0),
    ("wavelength", None, 0.7749, 1.239841984e-6 / 0.7749e-10),  # no units: the legacy angstrom
    ("wavelength", "m", 0.7749e-10, 1.239841984e-6 / 0.7749e-10),
    ("energy", "J", 1.0, None),  # unsupported units: not served
])
def test_a_generic_entrys_energy_is_served_in_ev_through_the_energy_owner(tmp_path, field, units, value, expected):
    """CAL-HELD-BATCH-20261009 (added after implementation): the monochromator energy, or its wavelength when no
    energy is recorded, is served as eV through core.energy. An X-ray energy's scale is decided by its size (maintainer:
    "It should be easy enough to figure out if the energy is in eV or keV for X-rays"); an undeclared wavelength keeps
    the legacy angstrom."""
    path = tmp_path / "mono.nxs"
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset("instrument/detector/data", data=np.ones((2, 4, 4), np.uint16))
        dataset = entry.create_dataset(f"instrument/monochromator/{field}", data=value)
        if units is not None:
            dataset.attrs["units"] = units
    with ContainerCursor(path) as cur:
        row = dict(cur.metadata_provider().metadata_for(1))
    assert row == ({} if expected is None else {"energy": pytest.approx(expected, rel=1e-12, abs=0)})
