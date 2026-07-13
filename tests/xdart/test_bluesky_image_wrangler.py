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
    DETX_FIXED,
    EIGER_TIME,
    GATE_TIME,
    HALPHA_FIXED,
    IMG_SHAPE,
    NFRAMES,
    SBSX_FIXED,
    WAVELENGTH,
    _write_bluesky_baseline_only_motors,
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
def baseline_only_file(tmp_path) -> Path:
    """halpha scanned per-frame; detx/sbsx fixed motors recorded ONLY in the
    baseline (not positioners); per-frame gate + eiger counting times."""
    return _write_bluesky_baseline_only_motors(tmp_path / "baseline_only_00001.nxs")


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
    # 'hy' is a HEIGHT motor, not a named preference nor a rotation-sounding axis,
    # so the default is Manual (the user enters the incidence angle) rather than
    # silently treating a translation stage as the incidence motor.
    assert th_motor.value() == "Manual"

    # Counters become Normalize options.
    norm_values = list(root.child("BG").child("norm_channel").opts["limits"])
    for counter in ("i0", "i1", "i2", "pd"):
        assert counter in norm_values

    # The integrator's GI-motor combo still receives the file's real motor list.
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


def test_gui_gi_dropdown_lists_all_motors(baseline_only_file):
    """The GI θ-motor dropdown offers EVERY real motor (scanned + baseline-fixed),
    not just the scanned one — beamlines use oddly-named incidence axes — and
    default-selects the named-preference motor (halpha)."""
    holder, root = _wrangler_holder()
    _select_image_file(holder, root, baseline_only_file)

    th_motor = root.child("GI").child("th_motor")
    values = list(th_motor.opts["limits"])
    # All three real motors of the file (halpha scanned; detx/sbsx baseline-fixed)
    # plus Manual — and NOT the legacy hardcoded 'th'.
    for motor in ("halpha", "detx", "sbsx"):
        assert motor in values
    assert "Manual" in values
    assert "th" not in values
    # No scaler / EpicsMotor field-spray leaks in as a "motor".
    assert "i0" not in values
    assert not any(v.startswith(("detx_", "sbsx_")) for v in values)
    # halpha is the named-preference incidence axis -> the default selection.
    assert th_motor.value() == "halpha"
    assert holder.incidence_motor == "halpha"
    # The integrator combo receives the same full motor list.
    assert holder.sigGIMotorOptions.emitted
    emitted_motors = holder.sigGIMotorOptions.emitted[-1][0]
    assert set(emitted_motors) == {"halpha", "detx", "sbsx"}


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


def test_thread_scan_info_carries_fixed_motors_and_counting_time(baseline_only_file):
    """The frame-metadata scan_info surfaces the baseline-only FIXED motors
    (detx/sbsx) as constants AND both counting times — this is exactly what the
    Frame metadata popup / Plot Metadata show for a Bluesky file."""
    worker = _bare_thread(baseline_only_file)
    si0 = worker._frame_scan_info(str(baseline_only_file), 0)
    si_last = worker._frame_scan_info(str(baseline_only_file), NFRAMES - 1)

    # Fixed motors broadcast constant across every frame.
    for si in (si0, si_last):
        assert si["detx"] == pytest.approx(DETX_FIXED)
        assert si["sbsx"] == pytest.approx(SBSX_FIXED)
        assert si["eiger_count_time"] == pytest.approx(EIGER_TIME)
        assert si["gate_actual_counting_time"] == pytest.approx(GATE_TIME)
    # The scanned motor stays per-frame (not broadcast as a constant).
    assert si0["halpha"] != si_last["halpha"]
    # The EpicsMotor field-spray is never surfaced as a column.
    assert not any(k.startswith(("detx_", "sbsx_")) for k in si0)


# ---------------------------------------------------------------------------
# Image DIRECTORY mode over a folder of Bluesky .nxs masters (bl17-2 closed
# loop): each .nxs is a self-contained master, discovered + processed like a
# directory of Eiger masters.  Pins that .nxs rides the master path — discovery,
# the append-skip cursor, and the seed's GI motor columns.
# ---------------------------------------------------------------------------

def test_directory_of_nxs_masters(tmp_path):
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        _paths_with_suffix, imageThread)

    d = tmp_path / "series"
    d.mkdir()
    stems = ["pos0_scan0001", "pos0_scan0002", "pos0_scan0003"]
    for stem in stems:
        _write_bluesky_nxwriter(d / f"{stem}.nxs")

    # Discovery: the same '.nxs' suffix glob the read path uses
    # (_eiger_refill_master_queue) finds every master.
    found = sorted(p.name for p in _paths_with_suffix(d, ".nxs"))
    assert found == [f"{s}.nxs" for s in stems]

    # Append-skip priming enumerates each .nxs as its OWN output scan (regression:
    # this branch used to exclude '.nxs', so append re-runs re-read every frame).
    t = imageThread.__new__(imageThread)
    t.write_mode = "Append"
    t.xye_only = False
    t.inp_type = "Image Directory"
    t.img_dir = str(d)
    t.img_ext = "nxs"
    t.img_file = str(d / "pos0_scan0001.nxs")
    t.file_filter = ""
    t.include_subdir = False
    assert sorted(t._append_run_start_scan_names()) == stems

    # The seed (first .nxs) yields the GI motor + counter columns that populate
    # the θ-motor / Normalize dropdowns in directory mode.
    holder, _root = _wrangler_holder()
    cols = holder._read_bluesky_source_columns(str(d / "pos0_scan0001.nxs"))
    assert cols is not None
    motors, counters = cols
    assert "hy" in motors
    assert {"i0", "i1", "i2", "pd"} <= set(counters)


# ---------------------------------------------------------------------------
# F5 (Codex review 2026-07-11, maintainer decision: finalized-.nxs-only): the
# live directory watch must DEFER an in-progress NXWriter container instead of
# consuming it — a partial .nxs used to be exhausted and permanently RETIRED
# into _eiger_done_masters, silently losing every frame written afterwards.
# ---------------------------------------------------------------------------

def _real_dir_watch_thread(watch_dir, out_dir):
    """A REAL imageThread (full __init__) watching a directory of .nxs masters."""
    import threading
    from queue import Queue

    from xrd_tools.core.containers import PONI
    from xdart.modules.live import LiveScan

    scan = LiveScan("scan", data_file=str(out_dir / "scan.nxs"), static=True)
    return imageThread(
        Queue(),                     # command_queue
        {},                          # scan_args
        threading.RLock(),           # file_lock
        "",                          # fname
        str(out_dir),                # h5_dir
        "scan",                      # scan_name
        False,                       # single_img
        PONI(dist=0.2, poni1=0.1, poni2=0.1, wavelength=1e-10),
        "Image Directory",           # inp_type
        "",                          # img_file
        str(watch_dir),              # img_dir
        False,                       # include_subdir
        "nxs",                       # img_ext
        False,                       # series_average
        None,                        # meta_ext
        "",                          # file_filter
        None,                        # mask_file
        "Full",                      # write_mode
        "None",                      # bg_type
        "",                          # bg_file
        "",                          # bg_dir
        None,                        # bg_matching_par
        "",                          # bg_match_fname
        "",                          # bg_file_filter
        1.0,                         # bg_scale
        None,                        # bg_norm_channel
        False,                       # gi
        None,                        # th_mtr
        1,                           # sample_orientation
        0.0,                         # tilt_angle
        "q_total",                   # gi_mode_1d
        "qip_qoop",                  # gi_mode_2d
        "start",                     # command
        scan,                        # scan
        live_mode=True,
        max_cores=1,
    )


def test_f5_unfinalized_nxs_deferred_then_consumed_in_full(tmp_path):
    """Mid-run NXWriter .nxs (no entry/end_time): the watch defers it — sentinel,
    NOT retired.  Once finalized, EVERY frame is consumed (none lost to the
    partial-read-then-retire path this fix kills)."""
    import h5py

    watch = tmp_path / "watch"
    watch.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    p = watch / "grow_00001.nxs"
    # Mid-run state: the NXWriter tree with only 2 frames flushed, no end_time.
    _write_bluesky_nxwriter(p, n=2)
    with h5py.File(p, "r+") as f:
        del f["entry/end_time"]

    t = _real_dir_watch_thread(watch, out)
    item = t._get_next_eiger_frame_sync()
    assert item[3] is None                       # deferred: end-of-stream sentinel
    assert str(p) not in t._eiger_done_masters   # NOT retired (the F5 data-loss)
    assert len(t._eiger_master_queue) == 0       # not queued either — re-polled

    # The run closes: NXWriter finalizes the SAME path with all 5 frames.
    _write_bluesky_nxwriter(p, n=5)

    frames = []
    for _ in range(10):
        item = t._get_next_eiger_frame_sync()
        if item[3] is None:
            break
        frames.append(item)
    assert len(frames) == 5                          # every frame, none lost
    assert {it[1] for it in frames} == {"grow_00001"}
    assert [it[2] for it in frames] == [1, 2, 3, 4, 5]
    assert str(p) in t._eiger_done_masters           # retired only after finalize+drain


def test_f5_refill_defers_only_unfinalized_bluesky(tmp_path):
    """The defer-guard is scoped: finalized Bluesky and PLAIN (non-Bluesky)
    .nxs queue immediately — a plain container has no end_time contract and
    must never be deferred forever — while an unreadable (mid-copy/torn) one
    waits."""
    import h5py

    watch = tmp_path / "watch"
    watch.mkdir()
    out = tmp_path / "out"
    out.mkdir()

    done = watch / "done_00001.nxs"
    _write_bluesky_nxwriter(done)
    inprog = watch / "inprog_00001.nxs"
    _write_bluesky_nxwriter(inprog)
    with h5py.File(inprog, "r+") as f:
        del f["entry/end_time"]
    plain = watch / "plain_00001.nxs"
    with h5py.File(plain, "w") as f:
        e = f.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        e.create_group("data").create_dataset(
            "data", data=np.zeros((3, 4, 4), dtype=np.uint32))
    torn = watch / "torn_00001.nxs"
    torn.write_bytes(b"\x89HDF\r\n partial garbage")

    t = _real_dir_watch_thread(watch, out)
    t._eiger_refill_master_queue()
    assert sorted(t._eiger_master_queue) == [str(done), str(plain)]
    assert not t._eiger_done_masters             # nothing retired by discovery

    # end_time lands -> the very next refill poll picks the deferred file up.
    with h5py.File(inprog, "r+") as f:
        f["entry"].create_dataset("end_time", data=b"2026-07-12T00:01:00")
    t._eiger_refill_master_queue()
    assert str(inprog) in t._eiger_master_queue


# ---------------------------------------------------------------------------
# F6 live-watch half (review wf_3614041c P2): the path-variant finder was
# strict-3D, so a one-exposure 2-D NXWriter .nxs opened fine through the
# HEADLESS seam but the live watch read ZERO frames (fabio can't read Bluesky
# layouts) and silently retired the file after a 30 s stall.
# ---------------------------------------------------------------------------

def _write_single_exposure_bluesky(path):
    """A one-exposure NXWriter count: the detector image is a lone 2-D dataset
    flagged @signal_type='detector' (finalized — end_time present)."""
    import h5py

    img = np.arange(16, dtype=np.uint32).reshape(4, 4)
    with h5py.File(path, "w") as f:
        f.attrs["creator"] = "NXWriter"
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["default"] = "data"
        entry.create_dataset("start_time", data=b"2026-07-12T00:00:00")
        entry.create_dataset("end_time", data=b"2026-07-12T00:00:01")
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data.attrs["signal"] = "gate"
        data.create_dataset("gate", data=np.array([0.5]))
        det = data.create_dataset("eiger_image", data=img)
        det.attrs["signal_type"] = "detector"
    return img


def test_f6_single_exposure_2d_nxs_consumed_by_live_watch(tmp_path):
    """The watch must read the one frame — the FULL 2-D image, never a row of
    it — and only then retire the master."""
    watch = tmp_path / "watch"
    watch.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    p = watch / "single_00001.nxs"
    img = _write_single_exposure_bluesky(p)

    t = _real_dir_watch_thread(watch, out)
    item = t._get_next_eiger_frame_sync()
    assert item[3] is not None, "single 2-D exposure must yield a frame"
    assert item[1] == "single_00001"
    assert item[2] == 1
    frame = np.asarray(item[3])
    assert frame.shape == img.shape           # the frame, not a (W,) row
    assert np.array_equal(frame, img)

    item = t._get_next_eiger_frame_sync()
    assert item[3] is None                    # exactly one frame, then done
    assert str(p) in t._eiger_done_masters


# ---------------------------------------------------------------------------
# bl17-2 (2026-07-12): a mixed beamline directory — diode/alignment scans with
# NO image dataset alongside real data scans.  An imageless container sorting
# FIRST used to return the end-of-stream sentinel; in batch that ended the run
# with 'Total Files Processed: 0' unless the user knew to add a name filter.
# ---------------------------------------------------------------------------

def test_imageless_containers_are_skipped_not_stream_ending(tmp_path):
    """Imageless containers must be retired-and-skipped; every image frame in
    the directory must still flow, in order, with no sentinel in between."""
    import h5py

    watch = tmp_path / "watch"
    watch.mkdir()
    out = tmp_path / "out"
    out.mkdir()

    # Three imageless alignment scans that SORT FIRST…
    for i in range(1, 4):
        with h5py.File(watch / f"align_{i:05d}.nxs", "w") as f:
            e = f.create_group("entry")
            e.attrs["NX_class"] = "NXentry"
            d = e.create_group("data")
            d.create_dataset("i0", data=np.linspace(0.0, 1.0, 11))
            d.create_dataset("EPOCH", data=np.linspace(0.0, 10.0, 11))
    # …then two real data scans.
    _write_bluesky_nxwriter(watch / "combi_data_00001.nxs", n=2)
    _write_bluesky_nxwriter(watch / "combi_data_00002.nxs", n=3)

    t = _real_dir_watch_thread(watch, out)
    frames = []
    for _ in range(12):
        item = t._get_next_eiger_frame_sync()
        if item[3] is None:
            break
        frames.append((item[1], item[2]))
    assert frames == [
        ("combi_data_00001", 1), ("combi_data_00001", 2),
        ("combi_data_00002", 1), ("combi_data_00002", 2),
        ("combi_data_00002", 3),
    ]
    for i in range(1, 4):
        assert str(watch / f"align_{i:05d}.nxs") in t._eiger_done_masters


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
    th_motor = root.child("GI").child("th_motor")
    values = list(th_motor.opts["limits"])
    assert "hy" in values and "Manual" in values and "th" not in values
    # 'hy' (height) is neither a named preference nor rotation-sounding -> Manual.
    assert th_motor.value() == "Manual"
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
