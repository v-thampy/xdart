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
import time
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

from tests.xdart._accepted_run import (  # noqa: E402
    accepted_run,
    container_source,
    directory_source,
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
        "_directory_metadata_preview_suffixes",
        "_directory_metadata_preview_file",
        "_adopt_directory_metadata_preview",
        "get_img_fname",
        # §13.6/§21.4 structured GI hydration helpers used by the emit path.
        "_gi_source_fingerprint",
        "_next_gi_hydration_generation",
        "_gi_hydration_registry",
        "_retire_gi_hydration_token",
        "_emit_gi_hydration",
        "_announce_gi_hydration",
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

    # The integrator's GI-motor combo still receives the file's real motor list,
    # now as a source-qualified GIMotorHydration (§13.6): a proved inspection
    # that found motors -> KNOWN_NONEMPTY.
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import (
        GIMotorHydration,
    )
    hydration = holder.sigGIMotorOptions.emitted[-1][0]
    assert isinstance(hydration, GIMotorHydration)
    assert hydration.state == GIMotorHydration.KNOWN_NONEMPTY
    assert tuple(hydration.motors) == ("hy",)


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
    # The integrator combo receives the same full motor list (§13.6 hydration).
    assert holder.sigGIMotorOptions.emitted
    emitted_motors = holder.sigGIMotorOptions.emitted[-1][0].motors
    assert set(emitted_motors) == {"halpha", "detx", "sbsx"}


def test_thread_fixed_incidence_constant_across_frames(fixed_incidence_file):
    """The fixed motor resolves to a CONSTANT per-frame incidence angle
    (value_start), while the scanned motor still varies per frame."""
    worker = _bare_thread(fixed_incidence_file)

    si0 = worker._frame_scan_info(worker.run_configuration, str(fixed_incidence_file), 0)
    si_last = worker._frame_scan_info(worker.run_configuration, str(fixed_incidence_file), NFRAMES - 1)

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
    # §13.7: a RESOLVED file that was inspected and has no readable motors is a
    # PROVED KNOWN_EMPTY hydration (motors ()), not an unqualified empty list.
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import (
        GIMotorHydration,
    )
    hydration = holder.sigGIMotorOptions.emitted[-1][0]
    assert isinstance(hydration, GIMotorHydration)
    assert hydration.state == GIMotorHydration.KNOWN_EMPTY
    assert tuple(hydration.motors) == ()


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
    """A worker plus the accepted configuration its routed reads consume."""
    worker = imageThread.__new__(imageThread)
    worker.meta_dir = None
    worker._eiger_metadata_cache = {}
    worker._bluesky_source_cache = {}
    worker.img_file = str(img_file)
    worker.run_configuration = worker._admitted_run_configuration = accepted_run(
        source_spec=container_source(img_file))
    return worker


def test_thread_per_frame_scan_info_from_bluesky(bluesky_file):
    worker = _bare_thread(bluesky_file)

    si0 = worker._frame_scan_info(worker.run_configuration, str(bluesky_file), 0)
    si_last = worker._frame_scan_info(worker.run_configuration, str(bluesky_file), NFRAMES - 1)

    # Per-frame motor + counter values are present (not an empty sidecar).
    for key in ("hy", "i0", "i1", "i2", "pd"):
        assert key in si0
    # The motor value advances frame-to-frame (per-frame, not a shared row).
    assert si0["hy"] != si_last["hy"]

    # The GI incidence angle resolves from the file's motor.
    assert resolve_incident_angle(si0, "hy") == pytest.approx(si0["hy"])


def test_thread_wavelength_stamped_on_scan(bluesky_file):
    from xrd_tools.core.energy import DEFAULT_WAVELENGTH_SENTINEL_M

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
    assert worker._frame_scan_info(worker.run_configuration, str(plain_nexus_file), 0) == {}
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
    si0 = worker._frame_scan_info(worker.run_configuration, str(baseline_only_file), 0)
    si_last = worker._frame_scan_info(worker.run_configuration, str(baseline_only_file), NFRAMES - 1)

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
    t.img_file = str(d / "pos0_scan0001.nxs")
    t.run_configuration = accepted_run(
        output_mode="Append",
        source_spec=directory_source(d, ext="nxs"))
    assert sorted(t._append_run_start_scan_names(t.run_configuration)) == stems

    # The seed (first .nxs) yields the GI motor + counter columns that populate
    # the θ-motor / Normalize dropdowns in directory mode.
    holder, _root = _wrangler_holder()
    cols = holder._read_bluesky_source_columns(str(d / "pos0_scan0001.nxs"))
    assert cols is not None
    motors, counters = cols
    assert "hy" in motors
    assert {"i0", "i1", "i2", "pd"} <= set(counters)


def test_directory_metadata_preview_keeps_gi_motor_options_direct_and_cached(
        tmp_path):
    """Lazy directory status must not remove pre-Run GI motor discovery.

    One direct matching container may provide the metadata schema.  Nested
    files are not searched merely because Subdirs is enabled, and repeating
    setup against the same source stamp does not reopen the representative.
    """
    from xrd_tools.sources import DirectorySourceSpec

    direct = _write_bluesky_baseline_only_motors(
        tmp_path / "direct_00001.nxs")
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    _write_bluesky_nxwriter(nested_dir / "nested_00001.nxs")

    holder, root = _wrangler_holder()
    holder.inp_type = "Image Directory"
    holder.source_spec = DirectorySourceSpec(
        root=tmp_path,
        recursive=True,
        suffixes=(".nxs",),
    )
    signal = root.child("Signal")
    signal.child("inp_type").setValue("Image Directory")
    signal.child("img_dir").setValue(str(tmp_path))
    signal.child("img_ext").setValue("nxs")
    signal.child("include_subdir").setValue(True)

    calls = []
    discovery_calls = []
    read_columns = holder._read_bluesky_source_columns
    discover = holder._directory_metadata_preview_file

    def counted_read(_self, path):
        calls.append(str(path))
        return read_columns(path)

    def counted_discover(_self):
        discovery_calls.append(True)
        return discover()

    holder._read_bluesky_source_columns = MethodType(counted_read, holder)
    holder._directory_metadata_preview_file = MethodType(
        counted_discover, holder)

    holder.get_img_fname()
    assert holder.img_file == ""
    assert calls == [str(direct)]
    assert discovery_calls == [True]
    choices = list(root.child("GI").child("th_motor").opts["limits"])
    assert {"Manual", "halpha", "detx", "sbsx"} <= set(choices)
    assert root.child("GI").child("th_motor").value() == "halpha"

    holder.get_img_fname()
    assert calls == [str(direct)]
    assert discovery_calls == [True]


def test_directory_hdf5_preview_accepts_master_h5_alias(tmp_path):
    """The hdf5 selector's accepted _master.h5 alias still supplies GI motors."""
    source = _write_bluesky_baseline_only_motors(
        tmp_path / "scan_master.h5")
    holder, root = _wrangler_holder()
    holder.inp_type = "Image Directory"
    signal = root.child("Signal")
    signal.child("inp_type").setValue("Image Directory")
    signal.child("img_dir").setValue(str(tmp_path))
    signal.child("img_ext").setValue("hdf5")

    holder.get_img_fname()

    assert holder.img_file == ""
    assert holder._directory_metadata_preview_path == str(source)
    choices = root.child("GI").child("th_motor").opts["limits"]
    assert "halpha" in choices


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
    # R4-G: retired policy arguments removed.  The watched directory, the
    # container extension and live mode now reach the worker only through the
    # accepted configuration admitted below.
    worker = imageThread(
        Queue(),                     # command_queue
        threading.RLock(),           # file_lock
        "",                          # fname
        "scan",                      # scan_name
        PONI(dist=0.2, poni1=0.1, poni2=0.1, wavelength=1e-10),
        "",                          # img_file
        "None",                      # bg_type
        "",                          # bg_file
        "",                          # bg_dir
        None,                        # bg_matching_par
        "",                          # bg_match_fname
        "",                          # bg_file_filter
        1.0,                         # bg_scale
        None,                        # bg_norm_channel
        "q_total",                   # gi_mode_1d
        "qip_qoop",                  # gi_mode_2d
        "start",                     # command
        scan,                        # scan
    )
    # O-1a-W1R-D2: worker execution consumes the ACCEPTED configuration passed
    # explicitly, so a harness driving it must admit one that names the same
    # directory of .nxs containers this thread watches.
    worker.run_configuration = worker._admitted_run_configuration = accepted_run(
        live_mode=True,
        save_path=str(out_dir),
        source_spec=directory_source(watch_dir, ext="nxs"),
    )
    return worker


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
    item = t._get_next_eiger_frame_sync(t.run_configuration)
    assert item[3] is None                       # deferred: end-of-stream sentinel
    assert str(p) not in t._eiger_done_masters   # NOT retired (the F5 data-loss)
    assert len(t._eiger_master_queue) == 0       # not queued either — re-polled

    # The run closes: NXWriter finalizes the SAME path with all 5 frames.
    _write_bluesky_nxwriter(p, n=5)

    frames = []
    for _ in range(10):
        item = t._get_next_eiger_frame_sync(t.run_configuration)
        if item[3] is None:
            break
        frames.append(item)
    assert len(frames) == 5                          # every frame, none lost
    assert {it[1] for it in frames} == {"grow_00001"}
    assert [it[2] for it in frames] == [1, 2, 3, 4, 5]
    assert str(p) in t._eiger_done_masters           # retired only after finalize+drain


def test_ready_nexus_scans_pop_in_natural_order(tmp_path):
    watch = tmp_path / "watch"
    watch.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    for stem in ("scan_10", "scan_2", "scan_1"):
        _write_bluesky_nxwriter(watch / f"{stem}.nxs")

    t = _real_dir_watch_thread(watch, out)
    t._eiger_refill_master_queue(t.run_configuration)
    popped = []
    while True:
        path = t._eiger_pop_next_master(t.run_configuration)
        if path is None:
            break
        popped.append(Path(path).name)

    assert popped == ["scan_1.nxs", "scan_2.nxs", "scan_10.nxs"]


def test_unfinished_earlier_scan_does_not_block_ready_later_scan(tmp_path):
    import h5py

    watch = tmp_path / "watch"
    watch.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    earlier = watch / "scan_1.nxs"
    later = watch / "scan_2.nxs"
    _write_bluesky_nxwriter(earlier)
    _write_bluesky_nxwriter(later)
    with h5py.File(earlier, "r+") as h5:
        del h5["entry/end_time"]

    t = _real_dir_watch_thread(watch, out)
    # Selection itself is name/stat-only.  The first JIT cursor open classifies
    # scan_1 as provisional, defers it, and continues to the ready sibling in
    # the same reader call.
    first = t._get_next_eiger_frame_sync(t.run_configuration)
    assert first[1:3] == ("scan_2", 1)
    assert str(earlier) not in t._eiger_done_masters

    later_frames = [first]
    for _ in range(NFRAMES + 2):
        item = t._get_next_eiger_frame_sync(t.run_configuration)
        if item[3] is None:
            break
        later_frames.append(item)
    assert [item[2] for item in later_frames] == list(range(1, NFRAMES + 1))
    assert {item[1] for item in later_frames} == {"scan_2"}
    assert str(later) in t._eiger_done_masters

    with h5py.File(earlier, "r+") as h5:
        h5["entry"].create_dataset(
            "end_time", data=b"2026-07-14T00:01:00")
    earlier_frames = []
    for _ in range(NFRAMES + 2):
        item = t._get_next_eiger_frame_sync(t.run_configuration)
        if item[3] is None:
            break
        earlier_frames.append(item)
    assert [item[2] for item in earlier_frames] == list(range(1, NFRAMES + 1))
    assert {item[1] for item in earlier_frames} == {"scan_1"}
    assert str(earlier) in t._eiger_done_masters


def test_live_zero_frame_shell_is_retried_without_blocking_ready_scan(tmp_path):
    """A just-created HDF5 shell is not a permanently imageless scan.

    Some acquisition writers expose the path before detector data or NXWriter
    markers land. The live watch must let later ready scans flow, then retry the
    same path after it is populated without requiring Stop/Run.
    """
    import h5py

    watch = tmp_path / "watch"
    watch.mkdir()
    out = tmp_path / "out"
    out.mkdir()

    shell = watch / "01_shell_00001.nxs"
    with h5py.File(shell, "w"):
        pass
    ready = watch / "02_ready_00001.nxs"
    _write_bluesky_nxwriter(ready, n=1)

    t = _real_dir_watch_thread(watch, out)
    t.CONTAINER_READY_RETRY = 0.01

    first = t._get_next_eiger_frame_sync(t.run_configuration)
    assert first[1:3] == ("02_ready_00001", 1)
    assert str(shell) not in t._eiger_done_masters

    _write_bluesky_nxwriter(shell, n=2)
    time.sleep(0.02)
    frames = []
    for _ in range(6):
        item = t._get_next_eiger_frame_sync(t.run_configuration)
        if item[3] is not None:
            frames.append((item[1], item[2]))
    assert frames == [("01_shell_00001", 1), ("01_shell_00001", 2)]


def test_finalized_bluesky_detectorless_is_terminal_without_retry(tmp_path):
    """A positive NXWriter marker plus end_time proves a completed imageless run."""
    import h5py

    watch = tmp_path / "watch"
    watch.mkdir()
    out = tmp_path / "out"
    out.mkdir()

    alignment = watch / "01_alignment_00001.nxs"
    with h5py.File(alignment, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_group("instrument/bluesky")
        entry.create_group("data").create_dataset(
            "i0", data=np.linspace(0.0, 1.0, 5))
        entry.create_dataset(
            "end_time", data=b"2026-07-14T00:01:00")
    ready = watch / "02_ready_00001.nxs"
    _write_bluesky_nxwriter(ready, n=1)

    t = _real_dir_watch_thread(watch, out)
    t.CONTAINER_READY_RETRY = 60.0

    first = t._get_next_eiger_frame_sync(t.run_configuration)
    assert first[1:3] == ("02_ready_00001", 1)
    assert str(alignment) in t._eiger_done_masters
    assert str(alignment) not in t._eiger_retry_after


def test_live_stale_mtime_shell_gets_a_retry_before_retirement(tmp_path):
    """A beamline clock offset must not make first sighting permanently fatal.

    Network filesystems can expose a new path with an mtime that is already old
    according to the client clock.  The path still deserves one readiness retry:
    its detector tree may become visible on the next poll even though wall-clock
    age exceeds the normal deadline.
    """
    import h5py

    watch = tmp_path / "watch"
    watch.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    shell = watch / "stale_clock_00001.nxs"
    with h5py.File(shell, "w"):
        pass
    old = time.time() - 120.0
    os.utime(shell, (old, old))

    t = _real_dir_watch_thread(watch, out)
    t.CONTAINER_READY_RETRY = 0.01

    first = t._get_next_eiger_frame_sync(t.run_configuration)
    assert first[3] is None
    assert str(shell) not in t._eiger_done_masters

    _write_bluesky_nxwriter(shell, n=2)
    time.sleep(0.02)
    frames = []
    for _ in range(6):
        item = t._get_next_eiger_frame_sync(t.run_configuration)
        if item[3] is not None:
            frames.append((item[1], item[2]))
    assert frames == [("stale_clock_00001", 1), ("stale_clock_00001", 2)]


def test_live_prefetch_finds_shell_populated_after_idle_sentinel(tmp_path):
    """The production prefetch lifecycle recovers without Pause or Stop/Run."""
    import h5py

    watch = tmp_path / "watch"
    watch.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    shell = watch / "late_00001.nxs"
    with h5py.File(shell, "w"):
        pass

    t = _real_dir_watch_thread(watch, out)
    t.CONTAINER_READY_RETRY = 0.01
    try:
        assert t._get_next_eiger_frame(t.run_configuration)[3] is None
        assert str(shell) not in t._eiger_done_masters

        _write_bluesky_nxwriter(shell, n=2)
        deadline = time.monotonic() + 3.0
        frames = []
        while time.monotonic() < deadline and len(frames) < 2:
            item = t._get_next_eiger_frame(t.run_configuration)
            if item[3] is not None:
                frames.append((item[1], item[2]))
            else:
                time.sleep(0.02)
        assert frames == [("late_00001", 1), ("late_00001", 2)]
    finally:
        t.command = "stop"
        t._prefetch_stop_prior()


def test_f5_jit_open_classifies_each_reached_container_once(
    tmp_path, monkeypatch,
):
    """Refill/pop stay name-only; one JIT cursor open owns classification.

    Finalized Bluesky and plain NeXus flow. An in-progress or torn container is
    deferred without blocking its ready siblings, and a changed/finalized path
    becomes eligible immediately rather than waiting out the old retry timer.
    """
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

    from xrd_tools.sources.cursor import ContainerCursor

    opens = []
    real_open = ContainerCursor.open

    def counted_open(cursor):
        opens.append(str(cursor._path))
        return real_open(cursor)

    monkeypatch.setattr(ContainerCursor, "open", counted_open)

    t._eiger_refill_master_queue(t.run_configuration)
    assert sorted(t._eiger_master_queue) == sorted(
        [str(done), str(inprog), str(plain), str(torn)])
    assert opens == []

    # Pop is also name/stat-only; classification belongs to open_master's
    # sustained cursor, not a duplicate finalized-at-close preflight.
    assert t._eiger_pop_next_master(t.run_configuration) == str(done)
    assert opens == []
    t._eiger_master_queue.appendleft(str(done))

    frames = []
    for _ in range(2 * NFRAMES + 8):
        item = t._get_next_eiger_frame_sync(t.run_configuration)
        if item[3] is None:
            break
        frames.append((item[1], item[2]))
    assert {name for name, _number in frames} == {
        "done_00001", "plain_00001",
    }
    assert opens.count(str(done)) == 1
    assert opens.count(str(inprog)) == 1
    assert opens.count(str(plain)) == 1
    assert opens.count(str(torn)) == 1
    assert str(inprog) not in t._eiger_done_masters
    assert str(torn) not in t._eiger_done_masters

    # end_time changes the cheap identity stamp. The next poll bypasses the
    # delay and performs exactly one new cursor open for the finalized version.
    with h5py.File(inprog, "r+") as f:
        f["entry"].create_dataset("end_time", data=b"2026-07-12T00:01:00")
    inprog_frames = []
    for _ in range(NFRAMES + 2):
        item = t._get_next_eiger_frame_sync(t.run_configuration)
        if item[3] is None:
            break
        inprog_frames.append((item[1], item[2]))
    assert inprog_frames == [
        ("inprog_00001", number) for number in range(1, NFRAMES + 1)
    ]
    assert opens.count(str(inprog)) == 2


def test_dir2b_bluesky_master_h5_reads_via_h5py_fallback(tmp_path):
    """DIR-2b (bl17-2 live): a Bluesky NXWriter file NAMED *_master.h5 is
    rejected by fabio's EigerImage (NotGoodReader, a RuntimeError) — that
    used to escape _eiger_open_master's (IOError, OSError) catch, crash the
    prefetch worker and END the whole directory run ('No frames discovered').
    It must fall back to the same h5py reader the .nxs branch uses and serve
    every frame."""
    watch = tmp_path / "watch"
    watch.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    p = watch / "num_0_align_00001_master.h5"
    _write_bluesky_nxwriter(p, n=3)

    t = _real_dir_watch_thread(watch, out)
    t.run_configuration = t._admitted_run_configuration = accepted_run(
        live_mode=True,
        save_path=str(out),
        source_spec=directory_source(watch, ext="h5"))

    frames = []
    for _ in range(6):
        item = t._get_next_eiger_frame_sync(t.run_configuration)
        if item[3] is None:
            break
        frames.append(item)
    assert len(frames) == 3, "fabio-rejected master.h5 must read via h5py"
    assert [it[2] for it in frames] == [1, 2, 3]

    # count_frames on the same file: the master arm now has the h5py
    # fallback too, so growth re-checks can't return 0 and retire a
    # growing container.
    from xrd_tools.io.image import count_frames
    assert count_frames(p) == 3


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
    item = t._get_next_eiger_frame_sync(t.run_configuration)
    assert item[3] is not None, "single 2-D exposure must yield a frame"
    assert item[1] == "single_00001"
    assert item[2] == 1
    frame = np.asarray(item[3])
    assert frame.shape == img.shape           # the frame, not a (W,) row
    assert np.array_equal(frame, img)

    item = t._get_next_eiger_frame_sync(t.run_configuration)
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
    # This test models completed, known-imageless history rather than a path
    # that has only just appeared and may still receive its detector dataset.
    t.CONTAINER_READY_DEADLINE = 0.0
    frames = []
    for _ in range(12):
        item = t._get_next_eiger_frame_sync(t.run_configuration)
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
    from xrd_tools.core.energy import DEFAULT_WAVELENGTH_SENTINEL_M

    worker = _bare_thread(_REAL)
    si0 = worker._frame_scan_info(worker.run_configuration, str(_REAL), 0)
    assert {"hy", "i0", "i1", "i2", "pd"} <= set(si0)
    assert resolve_incident_angle(si0, "hy") == pytest.approx(si0["hy"])
    assert si0["hy"] != worker._frame_scan_info(worker.run_configuration, str(_REAL), 30)["hy"]

    scan = types.SimpleNamespace(
        mg_args={"wavelength": DEFAULT_WAVELENGTH_SENTINEL_M}
    )
    worker._stamp_bluesky_wavelength(scan)
    assert scan.mg_args["wavelength"] == pytest.approx(1.033201653610002e-10, rel=1e-6)


# ---------------------------------------------------------------------------
# R2 — production-wired ContainerCursor read path (through the REAL thread)
# ---------------------------------------------------------------------------

def test_r2_cursor_backed_read_one_open_native_dtype_and_provider(tmp_path, monkeypatch):
    """One sustained ContainerCursor supplies frame count + ReadPlan + provider +
    native-dtype reads with no per-frame master reopen; it closes on drain."""
    import h5py

    watch = tmp_path / "watch"; watch.mkdir()
    out = tmp_path / "out"; out.mkdir()
    p = watch / "cur_00001.nxs"
    _write_bluesky_nxwriter(p, n=NFRAMES)

    opens = []
    orig = h5py.File.__init__

    def counting(self, name, *a, **k):
        opens.append(os.path.abspath(str(name)))
        return orig(self, name, *a, **k)

    monkeypatch.setattr(h5py.File, "__init__", counting)

    t = _real_dir_watch_thread(watch, out)
    frames = []
    for _ in range(NFRAMES + 3):
        item = t._get_next_eiger_frame_sync(t.run_configuration)
        if item[3] is None:
            break
        # the cursor + read plan are live while the master is being read
        assert t._eiger_cursor is not None
        assert t._eiger_read_plan is not None and t._eiger_read_plan.block_frames >= 1
        frames.append(item)

    assert [f[2] for f in frames] == list(range(1, NFRAMES + 1))
    # native detector dtype preserved end-to-end (no float64 upcast in source)
    assert all(f[3].dtype == np.uint32 for f in frames)
    # per-frame Bluesky metadata served from the cursor's materialized provider
    assert all("i0" in f[4] for f in frames)

    # the sustained cursor collapses the repeated traversal: the master is opened
    # at most twice (pop-time readiness probe + one consumption cursor), NEVER
    # once per frame.
    master_opens = opens.count(os.path.abspath(str(p)))
    assert master_opens <= 2, f"expected <=2 master opens, got {master_opens}"

    # fully drained -> the sync reader retired + closed the cursor
    assert t._eiger_cursor is None


def test_r2_cursor_closed_on_stop_and_clean_rerun(tmp_path):
    """Stop closes the sustained cursor; an immediate rerun reads fresh with no
    stale cursor/handle."""
    watch = tmp_path / "watch"; watch.mkdir()
    out = tmp_path / "out"; out.mkdir()
    _write_bluesky_nxwriter(watch / "s_00001.nxs", n=NFRAMES)

    t = _real_dir_watch_thread(watch, out)
    item = t._get_next_eiger_frame_sync(t.run_configuration)
    assert item[3] is not None
    cursor = t._eiger_cursor
    assert cursor is not None and cursor.closed is False

    # Stop closes the active master (prefetch owner thread's close path)
    t.command = "stop"
    t._eiger_close_master()
    assert t._eiger_cursor is None
    assert cursor.closed is True

    # immediate rerun: fresh read, no stale-handle error
    t.command = "start"
    t._eiger_master_path = None
    t._eiger_frame_idx = 0
    item2 = t._get_next_eiger_frame_sync(t.run_configuration)
    assert item2[3] is not None
    assert t._eiger_cursor is not None
    assert t._eiger_cursor is not cursor  # a NEW cursor, not the stopped one


def test_r2_bind_cursor_uses_actual_descriptor_read_plan_inputs(
        bluesky_file, monkeypatch):
    """The GUI binding cannot silently substitute plan shape/chunks/budget."""
    from xrd_tools.core import staging
    from xrd_tools.sources.cursor import ContainerCursor
    from xrd_tools.sources import read_plan as read_plan_module

    worker = _bare_thread(bluesky_file)
    worker._eiger_cursor = None
    worker._eiger_descriptor = None
    worker._eiger_read_plan = None
    worker._eiger_provider = None
    worker._eiger_fabio_handle = None
    budget = 2 * np.dtype(np.uint32).itemsize * np.prod(IMG_SHAPE)
    monkeypatch.setattr(staging, "source_block_budget_bytes", lambda: budget)
    real_plan_reads = read_plan_module.plan_reads
    seen = {}

    def spy_plan_reads(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return real_plan_reads(*args, **kwargs)

    monkeypatch.setattr(read_plan_module, "plan_reads", spy_plan_reads)
    cursor = ContainerCursor(bluesky_file).open()
    try:
        descriptor = cursor.descriptor
        worker._eiger_bind_cursor(cursor)
        assert seen["args"] == (
            descriptor.frame_count, descriptor.frame_shape, descriptor.dtype,
            descriptor.chunks, budget)
        assert seen["kwargs"] == {"two_d": descriptor.is_2d}
        assert worker._eiger_descriptor is descriptor
        assert worker._eiger_read_plan == real_plan_reads(
            descriptor.frame_count, descriptor.frame_shape, descriptor.dtype,
            descriptor.chunks, budget, two_d=descriptor.is_2d)
        assert worker._eiger_provider is cursor.metadata_provider()
    finally:
        worker._eiger_close_master()


def test_r2_cursor_refresh_failure_preserves_binding_then_recovers(tmp_path, monkeypatch):
    """A failed replacement retains the old coherent cursor for a later retry."""
    from xrd_tools.sources.cursor import ContainerCursor

    watch = tmp_path / "watch"; watch.mkdir()
    out = tmp_path / "out"; out.mkdir()
    path = watch / "refresh_00001.nxs"
    _write_bluesky_nxwriter(path, n=NFRAMES)

    worker = _real_dir_watch_thread(watch, out)
    assert worker._get_next_eiger_frame_sync(worker.run_configuration)[3] is not None
    old_cursor = worker._eiger_cursor
    old_descriptor = worker._eiger_descriptor
    old_plan = worker._eiger_read_plan
    old_provider = worker._eiger_provider
    assert old_cursor is not None

    real_open = ContainerCursor.open

    def transient_open(self):
        if self._path == path:  # noqa: SLF001 - exact reopened source path
            raise OSError("transient mid-write read failure")
        return real_open(self)

    monkeypatch.setattr(ContainerCursor, "open", transient_open)
    worker._eiger_refresh_master_handle()
    assert worker._eiger_cursor is old_cursor and old_cursor.closed is False
    assert worker._eiger_descriptor is old_descriptor
    assert worker._eiger_read_plan is old_plan
    assert worker._eiger_provider is old_provider

    monkeypatch.setattr(ContainerCursor, "open", real_open)
    worker._eiger_refresh_master_handle()
    assert worker._eiger_cursor is not None and worker._eiger_cursor is not old_cursor
    assert old_cursor.closed is True
    assert worker._eiger_descriptor is worker._eiger_cursor.descriptor
    assert worker._eiger_provider is not old_provider
    worker._eiger_close_master()


def test_r2_cursor_refresh_imageless_replacement_keeps_old_binding(
        tmp_path, monkeypatch):
    """A replacement with no dataset never publishes a mixed old/new binding."""
    import h5py
    import xrd_tools.sources.cursor as cursor_module

    from xrd_tools.sources.cursor import ContainerCursor

    watch = tmp_path / "watch"; watch.mkdir()
    out = tmp_path / "out"; out.mkdir()
    path = watch / "binding_00001.nxs"
    _write_bluesky_nxwriter(path, n=NFRAMES)
    imageless = tmp_path / "imageless.nxs"
    with h5py.File(imageless, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_group("instrument")

    worker = _real_dir_watch_thread(watch, out)
    assert worker._get_next_eiger_frame_sync(worker.run_configuration)[3] is not None
    old = (worker._eiger_cursor, worker._eiger_descriptor,
           worker._eiger_read_plan, worker._eiger_provider)

    monkeypatch.setattr(
        cursor_module, "ContainerCursor",
        lambda *_args, **_kwargs: ContainerCursor(imageless))
    assert worker._eiger_reopen_cursor() is False
    assert (worker._eiger_cursor, worker._eiger_descriptor,
            worker._eiger_read_plan, worker._eiger_provider) == old
    assert old[0] is not None and old[0].closed is False
    worker._eiger_close_master()


def test_r2_cursor_backed_rows_overlay_scanned_motors_and_preserve_order(tmp_path):
    """The real folder read carries motors, counters, constants, and EPOCH."""
    import h5py

    watch = tmp_path / "watch"; watch.mkdir()
    out = tmp_path / "out"; out.mkdir()
    path = watch / "rows_00001.nxs"
    _write_bluesky_baseline_only_motors(path, n=NFRAMES)
    with h5py.File(path, "r+") as h5:
        h5["entry"].create_dataset("end_time", data=np.bytes_("2026-07-17"))

    worker = _real_dir_watch_thread(watch, out)
    rows = []
    for _ in range(NFRAMES):
        item = worker._get_next_eiger_frame_sync(worker.run_configuration)
        assert item[3] is not None
        rows.append(item)

    assert [item[2] for item in rows] == list(range(1, NFRAMES + 1))
    for item in rows:
        row = item[4]
        assert {"halpha", "i0", "i1", "i2", "pd", "EPOCH", "detx",
                "sbsx", "eiger_count_time"} <= set(row)
        assert resolve_incident_angle(row, "halpha") == pytest.approx(row["halpha"])
    assert rows[0][4]["halpha"] != rows[-1][4]["halpha"]
    legacy = _bare_thread(path)._frame_scan_info(_bare_thread(path).run_configuration, str(path), 0)
    assert rows[0][4] == legacy
    assert worker._eiger_descriptor.wavelength == pytest.approx(WAVELENGTH)
    worker._eiger_close_master()


def test_r2_append_skip_avoids_real_cursor_reads_and_metadata(tmp_path, monkeypatch):
    """Append skips are decided before real sync/bulk cursor reads or metadata."""
    import queue
    import threading

    from xrd_tools.sources.cursor import ContainerCursor
    from xrd_tools.sources.metadata_provider import BlueskyMetadataProvider

    watch = tmp_path / "watch"; watch.mkdir()
    out = tmp_path / "out"; out.mkdir()
    path = watch / "append_00001.nxs"
    _write_bluesky_nxwriter(path, n=NFRAMES)

    read_indices = []
    block_ranges = []
    metadata_indices = []
    real_read = ContainerCursor.read_frame
    real_block = ContainerCursor.read_block
    real_metadata = BlueskyMetadataProvider.metadata_for

    def spy_read(self, index):
        read_indices.append(int(index))
        return real_read(self, index)

    def spy_metadata(self, index):
        metadata_indices.append(int(index))
        return real_metadata(self, index)

    def spy_block(self, start, stop):
        block_ranges.append((int(start), int(stop)))
        return real_block(self, start, stop)

    monkeypatch.setattr(ContainerCursor, "read_frame", spy_read)
    monkeypatch.setattr(ContainerCursor, "read_block", spy_block)
    monkeypatch.setattr(BlueskyMetadataProvider, "metadata_for", spy_metadata)

    worker = _real_dir_watch_thread(watch, out)
    worker.write_mode = "Append"
    worker._append_skip_frames_by_scan = {"append_00001": {1, 3}}
    worker._prefetch_queue = queue.Queue(maxsize=NFRAMES + 1)
    worker._prefetch_stop_evt = threading.Event()
    worker._prefetch_worker(worker.run_configuration)
    items = []
    while not worker._prefetch_queue.empty():
        item = worker._prefetch_queue.get_nowait()
        if item[3] is not None:
            items.append(item)

    assert [item[2] for item in items] == [2, 4, 5]
    assert read_indices == [1]
    assert block_ranges == [(3, 5)]
    assert metadata_indices == [1, 3, 4]
    assert worker._prefetch_error is None
    worker._eiger_close_master()
