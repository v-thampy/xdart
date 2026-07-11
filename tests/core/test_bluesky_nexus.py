"""Tests for the Bluesky / apstools ``NXWriter`` acquisition-file reader.

A committable synthetic fixture reproduces the apstools ``NXWriter`` structure
(creator=NXWriter, ``entry/instrument/bluesky/metadata``, ``positioners/hy``,
a flat ``entry/data`` with the image flagged ``@signal_type='detector'`` while
the NXdata ``@signal`` points at a scalar counter).  Real-file assertions run
only when ``$XDART_TEST_DATA`` points at the shipped
``nexus/Pt_10nm_00013.nxs``.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from xrd_tools.io import (
    ImageSourceKind,
    classify_image_source,
    read_nexus,
)
from xrd_tools.io.bluesky_nexus import (
    bluesky_motor_names,
    is_bluesky_nxwriter,
    resolve_nxentry,
)
from xrd_tools.io.image import count_frames, read_image
from xrd_tools.io.nexus import (
    find_nexus_image_dataset_in_open_file,
    open_nexus_image_stack,
)
from xrd_tools.io.read import get_metadata
from xrd_tools.sources.nexus import NexusStackSource


NFRAMES = 5
IMG_SHAPE = (4, 4)
WAVELENGTH = 1.2345
ENERGY_EV = 10000.0  # -> 10 keV


# ---------------------------------------------------------------------------
# synthetic apstools-NXWriter fixture
# ---------------------------------------------------------------------------

def _write_bluesky_nxwriter(path: Path, *, n: int = NFRAMES) -> Path:
    """Hand-build a minimal apstools ``NXWriter`` file with h5py."""
    hy = np.linspace(11.0, 11.6, n).astype(np.float64)
    counters = {
        "i0": np.linspace(30000, 31000, n),
        "i1": np.linspace(20000, 20500, n),
        "i2": np.linspace(150, 200, n),
        "pd": np.linspace(0, 4, n),
    }
    epoch = np.linspace(0.0, 4.0, n)
    gate = np.full(n, 0.5)  # the (scalar-ish 1-D) NXdata @signal counter
    images = np.arange(n * IMG_SHAPE[0] * IMG_SHAPE[1], dtype=np.uint32).reshape(
        (n, *IMG_SHAPE))

    with h5py.File(path, "w") as f:
        f.attrs["creator"] = "NXWriter"
        f.attrs["NeXus_release"] = "v2020.1"
        f.attrs["default"] = "entry"

        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["default"] = "data"
        entry.create_dataset("title", data=b"S0006-scan-synthetic")
        entry.create_dataset("plan_name", data=b"scan")
        entry.create_dataset("program_name", data=b"bluesky")
        entry.create_dataset("start_time", data=b"2026-07-10T00:00:00")
        entry.create_dataset("end_time", data=b"2026-07-10T00:01:00")
        entry.create_dataset("duration", data=np.float64(31.5))
        entry.create_dataset("entry_identifier", data=b"deadbeef")

        inst = entry.create_group("instrument")
        inst.attrs["NX_class"] = "NXinstrument"

        bluesky = inst.create_group("bluesky")
        bluesky.attrs["NX_class"] = "NXnote"
        md = bluesky.create_group("metadata")
        md.create_dataset("motors", data=b"!!python/tuple\n- hy\n")
        md.create_dataset("num_points", data=np.int64(n))
        md.create_dataset("detectors", data=b"- gate\n- i0\n- i1\n- i2\n- pd\n- eiger\n")
        eiger_cfg = md.create_group("configuration/eiger/data")
        eiger_cfg.create_dataset("eiger_cam_wavelength", data=np.float64(WAVELENGTH))
        eiger_cfg.create_dataset("eiger_cam_photon_energy", data=np.float64(ENERGY_EV))

        positioners = inst.create_group("positioners")
        positioners.attrs["NX_class"] = "NXnote"
        hy_grp = positioners.create_group("hy")
        hy_grp.attrs["NX_class"] = "NXpositioner"
        hy_grp.create_dataset("value", data=hy)

        detectors = inst.create_group("detectors")
        detectors.attrs["NX_class"] = "NXnote"
        det_eiger = detectors.create_group("eiger")
        det_eiger.attrs["NX_class"] = "NXdetector"

        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        # The Bluesky signpost trap: @signal names a scalar counter, NOT the image.
        data.attrs["signal"] = "gate"
        data.attrs["axes"] = ["hy"]
        data.create_dataset("hy", data=hy)
        for name, arr in counters.items():
            data.create_dataset(name, data=arr)
        data.create_dataset("EPOCH", data=epoch)
        data.create_dataset("gate", data=gate)
        img = data.create_dataset("eiger_image", data=images)
        img.attrs["signal_type"] = "detector"
        # a hard-linked twin under the detector group (as apstools writes)
        det_eiger["data"] = img

    return path


@pytest.fixture
def bluesky_file(tmp_path) -> Path:
    return _write_bluesky_nxwriter(tmp_path / "synthetic_bluesky_00001.nxs")


@pytest.fixture
def plain_nexus_file(tmp_path) -> Path:
    """A non-Bluesky NeXus file (no creator, no bluesky group)."""
    p = tmp_path / "plain.nxs"
    with h5py.File(p, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        inst = entry.create_group("instrument")
        mono = inst.create_group("monochromator")
        mono.create_dataset("energy", data=12.0)
        mono.create_dataset("wavelength", data=1.033)
        data = entry.create_group("data")
        data.create_dataset("data", data=np.zeros((3, 4, 4), dtype=np.uint32))
    return p


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------

def test_is_bluesky_nxwriter_true(bluesky_file):
    with h5py.File(bluesky_file, "r") as f:
        assert is_bluesky_nxwriter(f) is True
        assert is_bluesky_nxwriter(f["entry"]) is True
        assert resolve_nxentry(f).name == "/entry"
        assert bluesky_motor_names(f["entry"]) == ["hy"]


def test_is_bluesky_nxwriter_false_plain(plain_nexus_file):
    with h5py.File(plain_nexus_file, "r") as f:
        assert is_bluesky_nxwriter(f) is False


def test_is_bluesky_nxwriter_false_when_xdart_schema(tmp_path):
    """A processed xdart file (ssrl_schema present) is never Bluesky — even if
    it somehow carried a creator=NXWriter attr."""
    p = tmp_path / "processed.nxs"
    with h5py.File(p, "w") as f:
        f.attrs["creator"] = "NXWriter"
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["ssrl_schema"] = "ssrl_xrd_tools"
        entry.create_group("instrument").create_group("bluesky")
    with h5py.File(p, "r") as f:
        assert is_bluesky_nxwriter(f) is False


# ---------------------------------------------------------------------------
# motors / counters / wavelength via read_nexus
# ---------------------------------------------------------------------------

def test_read_nexus_motors_counters_wavelength(bluesky_file):
    meta = read_nexus(bluesky_file)
    assert list(meta.angles.keys()) == ["hy"]          # ONLY the scan motor
    assert set(meta.counters.keys()) == {"i0", "i1", "i2", "pd"}
    assert meta.angles["hy"].shape == (NFRAMES,)
    assert meta.wavelength == pytest.approx(WAVELENGTH)
    assert meta.energy == pytest.approx(ENERGY_EV / 1000.0)  # eV -> keV


# ---------------------------------------------------------------------------
# image resolution: 2-D detector frame, NOT the 1-D counter signal
# ---------------------------------------------------------------------------

def test_read_image_returns_detector_frame_not_counter(bluesky_file):
    frame0 = read_image(bluesky_file, frame=0)
    assert frame0.ndim == 2                     # the ndim>=2/signal_type pin
    assert frame0.shape == IMG_SHAPE
    assert count_frames(bluesky_file) == NFRAMES


def test_nexus_image_resolver_finds_detector(bluesky_file):
    with h5py.File(bluesky_file, "r") as f:
        path = find_nexus_image_dataset_in_open_file(f, "entry")
        assert path is not None
        # resolves to the detector-flagged pixel stack (ndim 3), not the counter
        assert f[path].ndim == 3
    with open_nexus_image_stack(bluesky_file) as stack:
        assert len(stack) == NFRAMES
        assert stack[0].shape == IMG_SHAPE


def test_classify_raw_master(bluesky_file):
    info = classify_image_source(bluesky_file)
    assert info.kind is ImageSourceKind.RAW_MASTER
    assert info.n_frames == NFRAMES
    assert info.has_raw is True


# ---------------------------------------------------------------------------
# Plot Metadata surfaces the per-frame columns
# ---------------------------------------------------------------------------

def test_get_metadata_columns(bluesky_file):
    meta = get_metadata(bluesky_file)
    assert meta["scan_data"]                              # non-empty
    assert "hy" in meta["scan_data"]
    assert "i0" in meta["scan_data"]
    assert "hy" in meta["positioners"]
    assert meta["wavelength_A"] == pytest.approx(WAVELENGTH)
    assert meta["n_frames"] == NFRAMES


def test_nexus_stack_source_motors_and_metadata(bluesky_file):
    src = NexusStackSource(bluesky_file)
    assert "hy" in src.motors
    assert src.motors["hy"].shape == (NFRAMES,)
    md0 = src.metadata_for(0)
    assert set(md0) == {"i0", "i1", "i2", "pd", "EPOCH"}
    assert all(isinstance(v, float) for v in md0.values())


def test_plain_nexus_stack_has_no_bluesky_columns(plain_nexus_file):
    """A non-Bluesky stack is untouched: no motors, empty metadata."""
    src = NexusStackSource(plain_nexus_file)
    assert src.motors == {}
    assert src.metadata_for(0) == {}


# ---------------------------------------------------------------------------
# FIXED (non-scanned) GI incidence motor: value in positioners + baseline,
# NOT a per-frame entry/data column (the usual GI acquisition — the incidence
# angle is held constant while something else, or nothing, is scanned).
# ---------------------------------------------------------------------------

HALPHA_FIXED = 0.2000  # value_start (the acquisition incidence angle)
HALPHA_END = 0.2003    # value_end differs slightly -> logged, value_start used


def _write_bluesky_fixed_incidence(path: Path, *, n: int = NFRAMES) -> Path:
    """A Bluesky NXWriter file where ``hy`` is scanned per-frame but the GI
    incidence motor ``halpha`` is FIXED — recorded ONLY in the baseline stream
    (``value_start``/``value_end``, with an EpicsMotor field-spray sibling to
    prove it is filtered) and ``positioners/halpha/value``, never in
    ``entry/data``."""
    hy = np.linspace(11.0, 11.6, n).astype(np.float64)
    with h5py.File(path, "w") as f:
        f.attrs["creator"] = "NXWriter"
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"

        inst = entry.create_group("instrument")
        inst.attrs["NX_class"] = "NXinstrument"
        bluesky = inst.create_group("bluesky")
        bluesky.attrs["NX_class"] = "NXnote"
        md = bluesky.create_group("metadata")
        md.create_dataset("motors", data=b"!!python/tuple\n- hy\n")
        eiger_cfg = md.create_group("configuration/eiger/data")
        eiger_cfg.create_dataset("eiger_cam_wavelength", data=np.float64(WAVELENGTH))

        # Baseline stream: value_start/value_end scalars per signal.
        baseline = bluesky.create_group("streams/baseline")
        h_grp = baseline.create_group("halpha")
        h_grp.create_dataset("value_start", data=np.float64(HALPHA_FIXED))
        h_grp.create_dataset("value_end", data=np.float64(HALPHA_END))
        # EpicsMotor field spray — must NOT be surfaced as a motor.
        for spray, val in (("halpha_hlm", 5.0), ("halpha_llm", -5.0),
                           ("halpha_dial", 0.9), ("halpha_user_setpoint", 0.2)):
            g = baseline.create_group(spray)
            g.create_dataset("value_start", data=np.float64(val))
            g.create_dataset("value_end", data=np.float64(val))

        # positioners: BOTH the scanned hy and the fixed halpha are groups here
        # (the authoritative motor-name source).
        positioners = inst.create_group("positioners")
        positioners.attrs["NX_class"] = "NXnote"
        hy_grp = positioners.create_group("hy")
        hy_grp.attrs["NX_class"] = "NXpositioner"
        hy_grp.create_dataset("value", data=hy)
        ha_grp = positioners.create_group("halpha")
        ha_grp.attrs["NX_class"] = "NXpositioner"
        ha_grp.create_dataset("value", data=np.float64(HALPHA_FIXED))

        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data.attrs["signal"] = "hy"
        data.create_dataset("hy", data=hy)          # scanned; halpha is NOT here
        images = np.arange(
            n * IMG_SHAPE[0] * IMG_SHAPE[1], dtype=np.uint32
        ).reshape((n, *IMG_SHAPE))
        img = data.create_dataset("eiger_image", data=images)
        img.attrs["signal_type"] = "detector"
    return path


@pytest.fixture
def fixed_incidence_file(tmp_path) -> Path:
    return _write_bluesky_fixed_incidence(tmp_path / "fixed_incidence_00001.nxs")


def test_fixed_motor_appears_in_motor_names(fixed_incidence_file):
    """Both the scanned and the FIXED motor are offered (positioners/*); the
    EpicsMotor field-spray fields are not."""
    from xrd_tools.io.bluesky_nexus import bluesky_motor_names

    with h5py.File(fixed_incidence_file, "r") as f:
        names = bluesky_motor_names(f["entry"])
    assert set(names) == {"hy", "halpha"}
    assert not any(s in names for s in ("halpha_hlm", "halpha_llm",
                                        "halpha_dial", "halpha_user_setpoint"))


def test_fixed_motor_value_from_baseline_value_start(fixed_incidence_file):
    from xrd_tools.io.bluesky_nexus import bluesky_fixed_motor_values

    with h5py.File(fixed_incidence_file, "r") as f:
        entry = f["entry"]
        # Exclude the scanned column: only the FIXED halpha remains.
        fixed = bluesky_fixed_motor_values(entry, exclude={"hy"})
    assert set(fixed) == {"halpha"}
    assert fixed["halpha"] == pytest.approx(HALPHA_FIXED)   # value_start, not _end
    # Field-spray fields are never surfaced.
    assert "halpha_hlm" not in fixed and "halpha_dial" not in fixed


def test_scanned_motor_is_not_in_fixed_values(fixed_incidence_file):
    """A per-frame scanned motor stays per-frame, never broadcast as a constant."""
    from xrd_tools.io.bluesky_nexus import bluesky_fixed_motor_values

    with h5py.File(fixed_incidence_file, "r") as f:
        fixed = bluesky_fixed_motor_values(
            f["entry"], exclude={"hy", "i0", "i1", "i2", "pd", "EPOCH"})
    assert "hy" not in fixed


def test_baseline_values_raw_includes_field_spray(fixed_incidence_file):
    """The RAW baseline reader returns value_start for EVERY signal (incl. the
    EpicsMotor field spray) — which is exactly why the motor-filtered
    :func:`bluesky_fixed_motor_values` exists."""
    from xrd_tools.io.bluesky_nexus import bluesky_baseline_values

    with h5py.File(fixed_incidence_file, "r") as f:
        raw = bluesky_baseline_values(f["entry"])
    assert raw["halpha"] == pytest.approx(HALPHA_FIXED)  # value_start, not value_end
    assert "halpha_hlm" in raw and "halpha_dial" in raw  # raw keeps the spray


def test_fixed_value_falls_back_to_positioner_without_baseline(tmp_path):
    """No baseline -> the positioner's constant ``value`` supplies the angle."""
    from xrd_tools.io.bluesky_nexus import bluesky_fixed_motor_values

    p = _write_bluesky_nxwriter(tmp_path / "nb_00001.nxs")
    # Add a fixed positioner with a constant value, no baseline.
    with h5py.File(p, "a") as f:
        ha = f["entry/instrument/positioners"].create_group("halpha")
        ha.attrs["NX_class"] = "NXpositioner"
        ha.create_dataset("value", data=np.float64(0.35))
    with h5py.File(p, "r") as f:
        fixed = bluesky_fixed_motor_values(f["entry"], exclude={"hy"})
    assert fixed["halpha"] == pytest.approx(0.35)


# ===========================================================================
# Real-file assertions (shipped Pt_10nm_00013.nxs; skip without test data)
# ===========================================================================

_DEFAULT_DATA = Path(__file__).resolve().parents[2] / "test_data"
_DATA = Path(os.environ.get("XDART_TEST_DATA", _DEFAULT_DATA))
_REAL = _DATA / "nexus" / "Pt_10nm_00013.nxs"

real_data = pytest.mark.skipif(
    not _REAL.exists(),
    reason=f"real Bluesky test file not found: {_REAL}",
)


@real_data
def test_real_read_nexus():
    meta = read_nexus(_REAL)
    assert "hy" in meta.angles                 # the real scan motor, not 'th'
    assert "th" not in meta.angles
    assert set(meta.counters.keys()) >= {"i0", "i1", "i2", "pd"}
    assert meta.wavelength == pytest.approx(1.033201653610002, rel=1e-6)


@real_data
def test_real_read_image_and_count():
    assert read_image(_REAL, frame=0).shape == (2167, 2070)
    assert count_frames(_REAL) == 31
    assert classify_image_source(_REAL).kind is ImageSourceKind.RAW_MASTER


@real_data
def test_real_get_metadata_columns():
    meta = get_metadata(_REAL)
    assert "hy" in meta["scan_data"]
    assert "i0" in meta["scan_data"]
    assert meta["n_frames"] == 31
    src = NexusStackSource(_REAL)
    assert "hy" in src.motors
    md0 = src.metadata_for(0)
    assert "i0" in md0 and isinstance(md0["i0"], float)
