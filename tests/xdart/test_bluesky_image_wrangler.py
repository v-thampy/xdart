"""The IMAGE wrangler reads a Bluesky/apstools ``NXWriter`` ``.nxs`` source's
EMBEDDED metadata.

A ``.nxs`` is always loaded through the IMAGE wrangler (there is no "Nexus"
source type — the Source dropdown is Image Series / Directory / Single Image).
The image half already reads frames via ``read_image``; these tests pin the
METADATA half that this branch wires in:

* GUI: selecting a Bluesky ``.nxs`` populates the GI Theta-Motor dropdown with
  the file's real motor (``hy``) + ``Manual`` (not the hardcoded ``th``), the
  Normalize dropdown with the counters (``i0``..``pd``), and emits the motor
  list to the integrator's GI-motor combo (``sigGIMotorOptions``).
* Thread: each processed frame's ``scan_info`` carries the per-frame motor +
  counter values, so the GI incidence angle resolves from the file's motor and
  the source wavelength is stamped onto the scan (no NaN in the output).
* Regression: a non-Bluesky source (plain NeXus, a TIFF path) is byte-identical
  to before — the Bluesky path is guarded behind ``is_bluesky_nxwriter``.

Real-file assertions run only when ``$XDART_TEST_DATA`` points at the shipped
``nexus/Pt_10nm_00013.nxs``.
"""
from __future__ import annotations

import os
import types
from pathlib import Path
from types import MethodType

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# The committable synthetic apstools-NXWriter fixture lives with the core reader
# tests; reuse it here so the GUI test drives the SAME structure the readers pin.
from tests.core.test_bluesky_nexus import (  # noqa: E402
    HALPHA_FIXED,
    IMG_SHAPE,
    NFRAMES,
    WAVELENGTH,
    _write_bluesky_fixed_incidence,
    _write_bluesky_nxwriter,
)

from xrd_tools.core.metadata import resolve_incident_angle  # noqa: E402
from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (  # noqa: E402
    imageThread,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bluesky_file(tmp_path) -> Path:
    return _write_bluesky_nxwriter(tmp_path / "Pt_synthetic_00001.nxs")


@pytest.fixture
def fixed_incidence_file(tmp_path) -> Path:
    """hy scanned per-frame; halpha the FIXED GI incidence motor (baseline +
    positioners, not in entry/data)."""
    return _write_bluesky_fixed_incidence(tmp_path / "fixed_incidence_00001.nxs")


@pytest.fixture
def plain_nexus_file(tmp_path) -> Path:
    """A non-Bluesky NeXus file (no creator, no bluesky group)."""
    import h5py

    p = tmp_path / "plain_00001.nxs"
    with h5py.File(p, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data.create_dataset("data", data=np.zeros((3, 4, 4), dtype=np.uint32))
    return p


# ---------------------------------------------------------------------------
# GUI: a light holder drives the REAL wrangler metadata methods against a real
# param tree (no heavy widget __init__), mirroring tests/xdart/test_n1_disclosure.
# ---------------------------------------------------------------------------

class _FakeSignal:
    def __init__(self):
        self.emitted = []

    def emit(self, *args):
        self.emitted.append(args)


def _wrangler_holder():
    import xdart.gui.gui_utils  # noqa: F401  # registers the 'str_browse' param type
    from pyqtgraph.parametertree import Parameter

    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import (
        imageWrangler,
        params,
    )

    root = Parameter.create(name="image_wrangler", type="group", children=params)
    h = types.SimpleNamespace(
        parameters=root,
        img_file="",
        img_dir="",
        img_ext="",
        inp_type="Image Series",
        single_img=False,
        include_subdir=False,
        meta_ext="auto",
        meta_dir="",
        file_filter="",
        scan_parameters=[],
        motors=[],
        counters=[],
        incidence_motor="th",
        poni=None,
        _bluesky_cols_cache=None,
        sigGIMotorOptions=_FakeSignal(),
    )
    for name in (
        "_read_bluesky_source_columns",
        "get_scan_parameters",
        "set_pars_from_meta",
        "set_gi_motor_options",
        "set_gi_th_motor",
        "set_bg_norm_options",
        "set_bg_matching_options",
        "exists_meta_file",
        "_sync_meta_ext_to_img_ext",
        "get_img_fname",
    ):
        setattr(h, name, MethodType(getattr(imageWrangler, name), h))
    return h, root


def _select_image_file(holder, root, path):
    """Drive the real ``File``-param -> ``get_img_fname`` selection flow."""
    root.child("Signal").child("File").setValue(str(path))
    holder.get_img_fname()


def test_gui_bluesky_populates_gi_motor_and_norm(bluesky_file):
    """Selecting a Bluesky .nxs surfaces the file's real motor + counters."""
    holder, root = _wrangler_holder()
    _select_image_file(holder, root, bluesky_file)

    th_motor = root.child("GI").child("th_motor")
    values = list(th_motor.opts["limits"])
    # The file's real scan motor, plus Manual — NOT the hardcoded 'th'.
    assert "hy" in values
    assert "Manual" in values
    assert "th" not in values
    # Exactly one motor -> default the selection to it (leaves Manual available).
    assert th_motor.value() == "hy"
    assert holder.incidence_motor == "hy"

    # Counters become Normalize options.
    norm_values = list(root.child("BG").child("norm_channel").opts["limits"])
    for counter in ("i0", "i1", "i2", "pd"):
        assert counter in norm_values

    # The integrator's GI-motor combo receives the motor list too.
    assert holder.sigGIMotorOptions.emitted == [(["hy"],)]


def test_gui_fixed_incidence_motor_in_options(fixed_incidence_file):
    """A FIXED GI incidence motor (halpha, baseline-only) is offered in the GI
    Theta-Motor dropdown and, being in the preference list, is the default."""
    holder, root = _wrangler_holder()
    _select_image_file(holder, root, fixed_incidence_file)

    th_motor = root.child("GI").child("th_motor")
    values = list(th_motor.opts["limits"])
    assert "halpha" in values  # the fixed incidence motor
    assert "hy" in values      # the scanned motor
    assert "Manual" in values
    # The EpicsMotor field-spray fields are never offered as motors.
    assert not any(v.startswith("halpha_") for v in values)
    # halpha is in the GI-motor preference order -> auto-selected as incidence.
    assert th_motor.value() == "halpha"
    assert holder.incidence_motor == "halpha"


def test_thread_fixed_incidence_constant_across_frames(fixed_incidence_file):
    """The fixed motor resolves to a CONSTANT per-frame incidence angle
    (value_start), while the scanned motor still varies per frame."""
    worker = _bare_thread(fixed_incidence_file)

    si0 = worker._frame_scan_info(str(fixed_incidence_file), 0)
    si_last = worker._frame_scan_info(str(fixed_incidence_file), NFRAMES - 1)

    # halpha is broadcast constant; hy is scanned per-frame.
    assert si0["halpha"] == pytest.approx(HALPHA_FIXED)
    assert si_last["halpha"] == pytest.approx(HALPHA_FIXED)
    assert si0["hy"] != si_last["hy"]

    # Incidence resolves from the FIXED motor, same angle every frame.
    assert resolve_incident_angle(si0, "halpha") == pytest.approx(HALPHA_FIXED)
    assert resolve_incident_angle(si_last, "halpha") == pytest.approx(HALPHA_FIXED)


def test_gui_plain_nexus_does_not_populate_from_embedded(plain_nexus_file):
    """Regression: a non-Bluesky .nxs stays on the sidecar/clear path — no
    embedded-motor harvest, GI Theta Motor collapses to Manual as before."""
    holder, root = _wrangler_holder()
    _select_image_file(holder, root, plain_nexus_file)

    assert holder._read_bluesky_source_columns(str(plain_nexus_file)) is None
    values = list(root.child("GI").child("th_motor").opts["limits"])
    assert values == ["Manual"]
    assert holder.motors == []
    assert holder.sigGIMotorOptions.emitted == [([],)]


def test_gui_helper_returns_none_for_non_nxs(tmp_path):
    """A non-HDF5 extension never opens a file — returns None immediately."""
    holder, _root = _wrangler_holder()
    tif = tmp_path / "frame_0001.tif"
    tif.write_bytes(b"II*\x00")  # not a real tiff; extension guard returns first
    assert holder._read_bluesky_source_columns(str(tif)) is None


# ---------------------------------------------------------------------------
# Thread: per-frame scan_info + incidence resolution + wavelength stamp
# ---------------------------------------------------------------------------

def _bare_thread(img_file):
    worker = imageThread.__new__(imageThread)
    worker.meta_ext = None
    worker.meta_dir = None
    worker._eiger_metadata_cache = {}
    worker._bluesky_source_cache = {}
    worker.img_file = str(img_file)
    return worker


def test_thread_per_frame_scan_info_from_bluesky(bluesky_file):
    worker = _bare_thread(bluesky_file)

    si0 = worker._frame_scan_info(str(bluesky_file), 0)
    si_last = worker._frame_scan_info(str(bluesky_file), NFRAMES - 1)

    # Per-frame motor + counter values are present (not an empty sidecar).
    for key in ("hy", "i0", "i1", "i2", "pd"):
        assert key in si0
    # The motor value advances frame-to-frame (per-frame, not a shared row).
    assert si0["hy"] != si_last["hy"]

    # The GI incidence angle resolves from the file's motor.
    assert resolve_incident_angle(si0, "hy") == pytest.approx(si0["hy"])


def test_thread_wavelength_stamped_on_scan(bluesky_file):
    from xdart.modules.wavelength import DEFAULT_WAVELENGTH_SENTINEL_M

    worker = _bare_thread(bluesky_file)
    scan = types.SimpleNamespace(
        mg_args={"wavelength": DEFAULT_WAVELENGTH_SENTINEL_M}
    )
    worker._stamp_bluesky_wavelength(scan)
    assert scan.mg_args["wavelength"] == pytest.approx(WAVELENGTH * 1e-10)


def test_thread_wavelength_does_not_clobber_real_value(bluesky_file):
    """A PONI-supplied wavelength wins for geometry — the file's is only a
    fallback, so a real mg_args value is left untouched."""
    worker = _bare_thread(bluesky_file)
    scan = types.SimpleNamespace(mg_args={"wavelength": 1.54e-10})
    worker._stamp_bluesky_wavelength(scan)
    assert scan.mg_args["wavelength"] == 1.54e-10


def test_thread_non_bluesky_scan_info_unchanged(plain_nexus_file):
    """Regression: a plain .nxs frame gets exactly the sidecar metadata (empty
    here, meta_ext off) — the Bluesky overlay is a no-op."""
    worker = _bare_thread(plain_nexus_file)
    assert worker._frame_scan_info(str(plain_nexus_file), 0) == {}
    assert worker._bluesky_frame_row(str(plain_nexus_file), 0) == {}
    # And it never stamps a wavelength.
    scan = types.SimpleNamespace(mg_args={"wavelength": 1e-10})
    worker._stamp_bluesky_wavelength(scan)
    assert scan.mg_args["wavelength"] == 1e-10


def test_thread_bluesky_shapes(bluesky_file):
    worker = _bare_thread(bluesky_file)
    info = worker._bluesky_source_for(str(bluesky_file))
    assert info is not None
    assert set(info["table"]) >= {"hy", "i0", "i1", "i2", "pd"}
    assert info["table"]["hy"].shape == (NFRAMES,)
    assert info["wavelength_A"] == pytest.approx(WAVELENGTH)
    assert IMG_SHAPE  # fixture sanity


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
def test_real_gui_motor_options():
    holder, root = _wrangler_holder()
    _select_image_file(holder, root, _REAL)
    values = list(root.child("GI").child("th_motor").opts["limits"])
    assert "hy" in values and "Manual" in values and "th" not in values
    assert holder.incidence_motor == "hy"
    norm_values = list(root.child("BG").child("norm_channel").opts["limits"])
    assert {"i0", "i1", "i2", "pd"} <= set(norm_values)


@real_data
def test_real_thread_per_frame_and_wavelength():
    from xdart.modules.wavelength import DEFAULT_WAVELENGTH_SENTINEL_M

    worker = _bare_thread(_REAL)
    si0 = worker._frame_scan_info(str(_REAL), 0)
    assert {"hy", "i0", "i1", "i2", "pd"} <= set(si0)
    assert resolve_incident_angle(si0, "hy") == pytest.approx(si0["hy"])
    assert si0["hy"] != worker._frame_scan_info(str(_REAL), 30)["hy"]

    scan = types.SimpleNamespace(
        mg_args={"wavelength": DEFAULT_WAVELENGTH_SENTINEL_M}
    )
    worker._stamp_bluesky_wavelength(scan)
    assert scan.mg_args["wavelength"] == pytest.approx(1.033201653610002e-10, rel=1e-6)
