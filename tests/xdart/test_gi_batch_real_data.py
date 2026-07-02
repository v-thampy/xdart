"""End-to-end batch-GI guards against real detector data.

These run only when ``$XDART_TEST_DATA`` (or the default sibling
``repos/test_data``) is present — they need the actual eiger/tiff frames,
the poni, and mask.  They encode the roadmap's Phase-2 verification:

* eiger masters carry no metadata, so a ``th`` incidence motor must raise
  :class:`IncidenceAngleUnresolved` rather than silently default to a
  degenerate 0° (the blank-cake regression).
* the GI 2D scout's frozen ``x_range``/``y_range`` (qip/qoop) must match a
  non-batch auto integration of the same frame and stay non-degenerate —
  i.e. the freeze faithfully captures the live grid, never collapses it.

Skipped automatically in CI / on machines without the data.
"""
import os
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

_DEFAULT_DATA = Path(__file__).resolve().parents[3] / "test_data"
DATA = Path(os.environ.get("XDART_TEST_DATA", _DEFAULT_DATA))
TIFF = DATA / "Tiff"
EIGER = DATA / "eiger"

pytestmark = pytest.mark.skipif(
    not TIFF.exists(), reason=f"detector test data not found at {DATA}",
)


_TIFF_PONI = "LaB6_detz190_dety72_th5_03261554_0001.poni"
# Two angle-dependence frames at DIFFERENT incidences — the case the
# stale-fiber-integrator pool got wrong (frame 2 integrated at frame 0's
# angle, collapsing its qoop).
_TIFF_FRAMES = [
    "Combi4_Angledependence_samz_4p9_03271002_0001.tif",  # th=0.15
    "Combi4_Angledependence_samz_4p9_03271002_0005.tif",  # th=0.35
]

# Mirror xdart.gui.tabs.static_scan.integrator.GI_MODES_1D / GI_MODES_2D
# (kept local so the parametrize lists don't pull the Qt integrator module
# at collection time).  Note the singular ``exit_angle`` (1D) vs plural
# ``exit_angles`` (2D).
GI_MODES_1D = ["q_total", "q_ip", "q_oop", "exit_angle", "chi_gi"]
GI_MODES_2D = ["qip_qoop", "q_chi", "exit_angles"]

# GI 1D modes whose *output* axis is out-of-plane (frozen on ``azimuth_range``
# rather than ``radial_range`` — the crux of the per-frame-drift fix).  chi_gi's
# output axis is χ_GI (the oop/azimuth grid), so it freezes on azimuth_range too.
_GI_1D_OOP_MODES = ("q_oop", "exit_angle", "chi_gi")


def _gi_1d_output_key(gi_mode_1d):
    return "azimuth_range" if gi_mode_1d in _GI_1D_OOP_MODES else "radial_range"


def _gi_2d_range_keys(gi_mode_2d):
    return ("x_range", "y_range") if gi_mode_2d == "qip_qoop" else (
        "radial_range", "azimuth_range")


def _load_tiff(name):
    import fabio
    from xrd_tools.io.metadata import read_image_metadata
    p = TIFF / name
    meta = read_image_metadata(str(p), meta_format="txt")
    img = fabio.open(str(p)).data.astype(np.float32)
    return img, float(meta["th"]), dict(meta)


def _tiff_poni():
    from xrd_tools.core.containers import PONI
    return PONI.from_poni_file(str(TIFF / _TIFF_PONI))


def _tiff_mask(img):
    import fabio
    mask_edf = fabio.open(str(TIFF / "mask.edf")).data
    return ((mask_edf != 0) | (img < 0)).astype(np.int8)


def _load_tiff_frame0():
    poni = _tiff_poni()
    img, th, _ = _load_tiff(_TIFF_FRAMES[0])
    return poni, th, img, _tiff_mask(img)


def _integrate_direct(poni, img, mask, incidence, bai_2d_args, sample_orientation=4):
    """Reference: integrate one frame at its OWN incidence, as the serial
    path does (fresh fiber integrator per frame)."""
    from xrd_tools.integrate.gid import create_fiber_integrator, integrate_gi_2d
    fi = create_fiber_integrator(poni, incident_angle=incidence,
                                 sample_orientation=sample_orientation,
                                 angle_unit="deg")
    a = dict(bai_2d_args)
    return integrate_gi_2d(
        img, fi, npt_rad=a.get("npt_rad", 500), npt_azim=a.get("npt_azim", 500),
        method="no", mask=mask,
        radial_range=a.get("x_range"), azimuth_range=a.get("y_range"),
    )


def _build_batch_thread(poni, mask, *, incidence_motor="th",
                        sample_orientation=4, gi=True):
    """Build a SimpleNamespace imageThread wired with the real integration
    helpers + an xye-flush spy.  Returns ``(w, captured)`` where ``captured``
    is the ``{img_number: LiveFrame}`` dict the spy fills at flush time.

    This is the shared rig behind both :func:`_run_batch_streaming` (drives the
    streaming ``_dispatch_batch_streaming``) and :func:`_frozen_gi_bai_args`
    (drives the scout freeze) so the freeze sees the same thread state as a real
    batch.
    """
    from types import SimpleNamespace, MethodType
    from threading import RLock
    from xdart.modules.reduction import StandardPlanCache
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import wranglerThread
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread

    w = SimpleNamespace(
        max_cores=2, gi=gi, incidence_motor=incidence_motor,
        sample_orientation=sample_orientation, tilt_angle=0,
        series_average=False, mask=mask, poni=poni, command="",
        batch_mode=True, xye_only=True,
        apply_threshold=False, threshold_min=0, threshold_max=0,
        _plan_cache=StandardPlanCache(), _xye_lock=RLock(), _xye_buffer=[],
        _cached_gi_incident_angle=None,
        showLabel=SimpleNamespace(emit=lambda *a: None),
        _middle_truncate=lambda t, **k: t,
    )
    # Bind the real wranglerThread helpers (these are the integration path).
    for meth in ("_resolve_frame_mask", "_prewarm_frame_mask",
                 "_apply_threshold_inline"):
        setattr(w, meth, MethodType(getattr(wranglerThread, meth), w))
    w._dispatch_batch_serial = MethodType(imageThread._dispatch_batch_serial, w)
    # D₂: the serial dispatch tail now routes through these (real _save_due gate
    # works — the rig sets LIVE_SAVE_INTERVAL + a real scan with frames).
    w.flush_serial_tail = MethodType(imageThread.flush_serial_tail, w)
    w._save_due = MethodType(imageThread._save_due, w)
    w._h5pool_bracket = MethodType(imageThread._h5pool_bracket, w)
    # Pause: the dispatch/submit loops now call _wait_if_paused() at the top; it
    # early-returns since this rig's command is never 'pause'.
    w._wait_if_paused = MethodType(imageThread._wait_if_paused, w)
    # Frame-shell builder extracted from the streaming dispatcher.
    w._build_batch_frames = MethodType(imageThread._build_batch_frames, w)
    # The real freeze+dispatch orchestrator (freezes, then routes to the
    # streaming dispatcher) — used to exercise the Int-1D-XYE path where the 2D
    # freeze self-skips on xye_only.
    w._dispatch_batch = MethodType(imageThread._dispatch_batch, w)
    # 4e: the rig drives the PRODUCTION streaming path (one write path — the
    # chunked dispatcher + its batch-execution selector are gone).  Wire the
    # persistent ReductionSession + QtNexusSink so _dispatch_batch routes here and
    # router-driven tests (xye-only uniform-grid, multichunk) get a real session.
    # gi_freeze_mode is left UNSET so the production scout freeze runs (the freeze
    # is itself under test) — _run_batch_streaming sets it None only when it passes
    # a pre-determined grid.
    w.file_lock = RLock()
    w.command_lock = RLock()
    w.sigUpdate = SimpleNamespace(emit=lambda *a: None)
    w.LIVE_SAVE_INTERVAL = 1000
    w._streaming_session = None
    w._streaming_sink = None
    w._streaming_scan_id = None
    w._scan_session_adapter = None
    w._reduction_session = None
    w._reduction_session_key = None
    w._gi_prepass_scan_id = None
    w._cancel_token = MethodType(wranglerThread._cancel_token, w)
    w._close_reduction_session = MethodType(
        wranglerThread._close_reduction_session, w)
    for _sm in ("_dispatch_batch_streaming", "_get_streaming_session"):
        setattr(w, _sm, MethodType(getattr(imageThread, _sm), w))
    # _dispatch_batch calls this one-time live-GI clip advisory (#75); bind it so
    # the double doesn't AttributeError.  No-op here (batch_mode=True → early
    # return), so the GI matrix assertions are unaffected.
    w._maybe_warn_live_gi_clip = MethodType(
        imageThread._maybe_warn_live_gi_clip, w)
    # The real scout-freeze methods (the code under test for the GI matrix).
    w._freeze_gi_1d_auto_range = MethodType(imageThread._freeze_gi_1d_auto_range, w)
    w._freeze_gi_2d_auto_ranges = MethodType(imageThread._freeze_gi_2d_auto_ranges, w)
    # #70: the freeze now scouts first+last via these helpers (union of extremes).
    w._scout_pending_frames = MethodType(imageThread._scout_pending_frames, w)
    w._build_scout = MethodType(imageThread._build_scout, w)
    # BLOCKER 1: the streaming dispatcher runs the whole-scan GI grid pre-pass;
    # bind it + its helpers.  With no source attrs on this rig, _enumerate_scan_files
    # returns [] so the pre-pass is a no-op (the chunk-local freeze is unchanged).
    for _m in ("_gi_freeze_whole_scan_prepass", "_gi_ranges_fully_pinned", "_gi_whole_scan_scout_entries",
               "_frame_source_for", "_enumerate_scan_files"):
        setattr(w, _m, MethodType(getattr(imageThread, _m), w))
    # Spy on the xye flush: snapshot the Phase-1 integrated frames rather
    # than writing xye files (and don't clear the buffer).
    captured = {}

    def _spy_flush(_scan, published_idxs=None):
        for num, fr in w._xye_buffer:
            captured[num] = fr
    w._flush_xye_buffer = _spy_flush
    return w, captured


def _make_scan(poni, mask, bai_1d_args, bai_2d_args, *, gi=True):
    from types import SimpleNamespace
    from xrd_tools.integrate.calibration import poni_to_integrator
    return SimpleNamespace(
        name="equivalence_scan",
        gi=gi,
        skip_2d=False,
        bai_1d_args=dict(bai_1d_args),
        bai_2d_args=dict(bai_2d_args),
        global_mask=mask,
        _cached_integrator=poni_to_integrator(poni),
        _cached_fiber_integrator=None,
        _cached_fiber_integrator_angle=None,
        _cached_data_mask=None,
        # QtNexusSink._due_to_save reads scan.frames._in_memory_cap for the
        # persist-before-evict bound; the streaming writer thread hits it on
        # EVERY frame, so the rig must carry it or the writer dies mid-run (a
        # crash a single-frame test silently swallows via abort()'s force-flush).
        frames=SimpleNamespace(_in_memory_cap=64),
    )


def _default_bai_1d(gi, gi_mode_1d):
    return {"gi_mode_1d": gi_mode_1d} if gi else {
        "numpoints": 256, "unit": "q_A^-1",
        "radial_range": (0.1, 6.0), "method": "csr",
    }


def _default_bai_2d(gi, gi_mode_2d):
    return (
        {"gi_mode_2d": gi_mode_2d, "x_range": None, "y_range": None,
         "npt_rad": 500, "npt_azim": 500}
        if gi else
        {"npt_rad": 96, "npt_azim": 72, "unit": "q_A^-1",
         "radial_range": (0.1, 6.0), "azimuth_range": (-90.0, 90.0),
         "method": "csr"}
    )


def _frozen_gi_bai_args(poni, img, meta, mask, *, gi_mode_1d, gi_mode_2d,
                        npt_rad=64, npt_azim=48, numpoints=128,
                        incidence_motor="th", sample_orientation=4):
    """Run the REAL mode-aware scout freeze (1D + 2D) from one scout frame and
    return the frozen ``(bai_1d, bai_2d)`` dicts — the production freeze that
    locks every frame onto one common grid.

    Faithful to ``_dispatch_batch``: the 2D freeze is gated on ``xye_only`` in
    production (the cake stack isn't written in Int-1D-XYE mode), so we run the
    freeze with ``xye_only=False`` to exercise BOTH halves here.
    """
    w, _ = _build_batch_thread(
        poni, mask, incidence_motor=incidence_motor,
        sample_orientation=sample_orientation, gi=True,
    )
    w.xye_only = False  # so _freeze_gi_2d_auto_ranges runs (it skips on xye_only)
    scan = _make_scan(
        poni, mask,
        {"gi_mode_1d": gi_mode_1d, "numpoints": numpoints},
        {"gi_mode_2d": gi_mode_2d, "npt_rad": npt_rad, "npt_azim": npt_azim},
        gi=True,
    )
    pending = [(_TIFF_FRAMES[0], 1, img, dict(meta), 0.0, 0.0)]
    w._freeze_gi_1d_auto_range(scan, pending)
    w._freeze_gi_2d_auto_ranges(scan, pending)
    return dict(scan.bai_1d_args), dict(scan.bai_2d_args)


def _run_batch_streaming(poni, pending_data, mask, *, incidence_motor="th",
                         sample_orientation=4, gi=True,
                         gi_mode_1d="q_total", gi_mode_2d="qip_qoop",
                         bai_1d_args=None, bai_2d_args=None,
                         data_file=None, xye_only=True):
    """Drive the REAL streaming dispatch — ``ReductionSession`` + ``QtNexusSink``
    via ``_dispatch_batch_streaming`` — on real frames.

    This is the PRODUCTION batch path (``batch_execution='streaming'`` is the
    module default), so the equivalence spine gates on the path that actually
    ships, not on the soon-to-be-retired chunked dispatcher.  Returns the same
    ``{img_number: LiveFrame}`` shape as :func:`_run_batch_parallel`, captured
    by the shared xye-flush spy (the sink hydrates int_1d + int_2d onto the
    registered LiveFrame and buffers ``(idx, live)`` before the final flush).

    The bai args are passed in PRE-FROZEN (by ``_frozen_gi_bai_args``) and
    ``gi_freeze_mode=None`` so the session uses that exact grid: the comparison
    isolates the integrate+write path, not the freeze — the identical contract
    the chunked leg uses (``freeze=False`` -> ``gi_freeze_mode=None``).
    """
    if bai_1d_args is None:
        bai_1d_args = _default_bai_1d(gi, gi_mode_1d)
    if bai_2d_args is None:
        bai_2d_args = _default_bai_2d(gi, gi_mode_2d)

    # _build_batch_thread now wires the full streaming session (one write path);
    # this runner only pins gi_freeze_mode=None so the session uses the args it is
    # GIVEN as-is — frozen args -> one uniform grid, unfrozen (None-range) args ->
    # per-frame auto-range drift — isolating the integrate+write path from the
    # freeze (the identical contract the retired chunked freeze=False leg used).
    w, captured = _build_batch_thread(
        poni, mask, incidence_motor=incidence_motor,
        sample_orientation=sample_orientation, gi=gi,
    )
    w.xye_only = bool(xye_only)
    w.gi_freeze_mode = None
    scan = _make_scan(poni, mask, bai_1d_args, bai_2d_args, gi=gi)
    if data_file is not None:
        scan.data_file = str(data_file)

    pending = [(name, i + 1, img, info, 0.0, 0.0)
               for i, (name, img, info) in enumerate(pending_data)]
    w._dispatch_batch_streaming(scan, pending)
    w._close_reduction_session()    # finish() -> QtNexusSink final flush -> spy
    return captured


def _run_live_single(poni, name, img, meta, mask, *, incidence_motor="th",
                     sample_orientation=4, gi=True,
                     gi_mode_1d="q_total", gi_mode_2d="qip_qoop",
                     bai_1d_args=None, bai_2d_args=None):
    """Drive the real sequential/live single-frame path and capture its frame."""

    from types import SimpleNamespace, MethodType
    from threading import Condition, RLock
    from xrd_tools.integrate.calibration import poni_to_integrator
    from xdart.modules.reduction import StandardPlanCache
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import wranglerThread
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread

    if bai_1d_args is None:
        bai_1d_args = _default_bai_1d(gi, gi_mode_1d)
    if bai_2d_args is None:
        bai_2d_args = _default_bai_2d(gi, gi_mode_2d)
    scan = SimpleNamespace(
        name="equivalence_scan",
        gi=gi,
        skip_2d=False,
        bai_1d_args=dict(bai_1d_args),
        bai_2d_args=dict(bai_2d_args),
        global_mask=mask,
        _cached_integrator=poni_to_integrator(poni),
        _cached_fiber_integrator=None,
        _cached_fiber_integrator_angle=None,
        _cached_data_mask=None,
    )
    w = SimpleNamespace(
        gi=gi, incidence_motor=incidence_motor,
        sample_orientation=sample_orientation, tilt_angle=0,
        series_average=False, mask=mask, poni=poni, command="",
        batch_mode=False, xye_only=True,
        apply_threshold=False, threshold_min=0, threshold_max=0,
        _plan_cache=StandardPlanCache(), _xye_lock=RLock(), _xye_buffer=[],
        _published_frames={}, _cached_gi_incident_angle=None,
        data_1d={}, data_2d={}, file_lock=Condition(),
        sub_label="",
        sigUpdate=SimpleNamespace(emit=lambda *a: None),
        showLabel=SimpleNamespace(emit=lambda *a: None),
        _middle_truncate=lambda t, **k: t,
    )
    for meth in ("_resolve_frame_mask", "_prewarm_frame_mask",
                 "_apply_threshold_inline"):
        setattr(w, meth, MethodType(getattr(wranglerThread, meth), w))
    w._process_one = MethodType(imageThread._process_one, w)
    imageThread._process_one(w, scan, name, 1, img, dict(meta), 0.0, 0.0)
    return w._published_frames[1]


def _canonicalize_thumbnail(frame, mask):
    """Make frame.thumbnail match the quantized/dequantized NeXus value."""

    from xdart.modules.ewald.nexus_writer import _quantize_thumbnail

    frame.make_thumbnail(global_mask=mask)
    thumb = getattr(frame, "thumbnail", None)
    if thumb is None:
        return None, None
    quant, (vmin, vmax, dtype) = _quantize_thumbnail(np.asarray(thumb, dtype=float))
    scale = 65535.0 if dtype == "uint16" else 255.0
    deq = vmin + (quant.astype(float) / scale) * (vmax - vmin)
    frame.thumbnail = deq
    return quant, (vmin, vmax, dtype)


def _write_publication_reload(path, frame, thumb_record=None):
    from xrd_tools.core import numeric_metadata
    from xrd_tools.io.nexus import write_integrated_stack
    from xdart.modules.frame_publication import publication_from_nexus_frame

    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        write_integrated_stack(
            entry,
            frame_indices=[int(frame.idx)],
            results_1d=[frame.int_1d],
            results_2d=[frame.int_2d],
        )
        numeric = numeric_metadata(frame.scan_info)
        if numeric:
            scan_data = entry.create_group("scan_data")
            scan_data.create_dataset("frame_index", data=np.array([int(frame.idx)], dtype=np.int64))
            for key, value in numeric.items():
                scan_data.create_dataset(str(key), data=np.array([value], dtype=np.float64))
        try:
            incident_angle = float(frame._get_incident_angle()) if getattr(frame, "gi", False) else None
        except Exception:
            incident_angle = None
        if incident_angle is not None:
            geom = entry.create_group("per_frame_geometry")
            geom.create_dataset("frame_index", data=np.array([int(frame.idx)], dtype=np.int64))
            geom.create_dataset("incident_angle", data=np.array([incident_angle], dtype=np.float64))
        if thumb_record is not None:
            quant, (vmin, vmax, dtype) = thumb_record
            fg = entry.create_group(f"frames/frame_{int(frame.idx):04d}")
            ds = fg.create_dataset("thumbnail", data=quant)
            ds.attrs["vmin"] = float(vmin)
            ds.attrs["vmax"] = float(vmax)
            ds.attrs["dtype"] = dtype
    return publication_from_nexus_frame(str(path), int(frame.idx))


def _assert_live_batch_reload_equivalent(tmp_path, *, gi,
                                         gi_mode_1d="q_total",
                                         gi_mode_2d="qip_qoop",
                                         unit_1d="q_A^-1"):
    """The equivalence spine: for one frame integrated under the given mode,
    the live (serial), batch (parallel), and NeXus-reload paths must produce
    byte-equivalent FrameViews.

    For GI the bai ranges are frozen by the REAL mode-aware scout freeze
    (:func:`_frozen_gi_bai_args`) so live and batch share one grid — exactly
    what ``_dispatch_batch`` does before either dispatch path runs.
    """
    from xrd_tools.core import assert_frameview_equivalent
    from xdart.modules.frame_publication import publication_from_live_frame

    poni = _tiff_poni()
    img, th, meta = _load_tiff(_TIFF_FRAMES[0])
    mask = _tiff_mask(img)

    if gi:
        bai_1d, bai_2d = _frozen_gi_bai_args(
            poni, img, meta, mask,
            gi_mode_1d=gi_mode_1d, gi_mode_2d=gi_mode_2d,
        )
    else:
        # Mode A (unit_1d='chi_deg'): the OUTPUT axis is chi; radial_range is the
        # q BAND to pool over.  Otherwise a standard radial 1D.
        bai_1d = {
            "numpoints": 256,
            "unit": unit_1d,
            "radial_range": (0.1, 6.0),
            "method": "csr",
        }
        bai_2d = {
            "npt_rad": 96,
            "npt_azim": 72,
            "unit": "q_A^-1",
            "radial_range": (0.1, 6.0),
            "azimuth_range": (-90.0, 90.0),
            "method": "csr",
        }

    live = _run_live_single(
        poni, _TIFF_FRAMES[0], img, meta, mask,
        gi=gi, bai_1d_args=bai_1d, bai_2d_args=bai_2d,
    )
    # The batch leg is the PRODUCTION streaming path (ReductionSession +
    # QtNexusSink) — the shipping default.  The chunked dispatcher it used to run
    # through was retired in 4e; streaming is now the sole batch reference.
    batch_stream = _run_batch_streaming(
        poni, [(_TIFF_FRAMES[0], img, meta)], mask,
        gi=gi, bai_1d_args=bai_1d, bai_2d_args=bai_2d,
    )[1]

    thumb_live = _canonicalize_thumbnail(live, mask)
    _canonicalize_thumbnail(batch_stream, mask)
    pub_live = publication_from_live_frame(live)
    pub_batch_stream = publication_from_live_frame(batch_stream)
    fname = (f"equiv_gi_{gi_mode_1d}_{gi_mode_2d}.nxs" if gi
             else "equiv_standard.nxs")
    pub_reload = _write_publication_reload(
        tmp_path / fname,
        live,
        thumb_live,
    )

    assert_frameview_equivalent(pub_live.view, pub_batch_stream.view)
    assert_frameview_equivalent(pub_live.view, pub_reload.view)


def _assert_good_gi_publication_passes(frame, *, min_dummy_headroom=0.05):
    from xdart.modules.frame_publication import publication_from_live_frame

    publication = publication_from_live_frame(frame)
    assert publication.diagnostics.ok
    assert publication.diagnostics.dummy_fraction_2d is not None
    assert publication.diagnostics.dummy_fraction_2d < (0.95 - min_dummy_headroom)


def test_batch_parallel_tiff_cakes_nondegenerate_and_match_serial():
    poni = _tiff_poni()
    frames = [_load_tiff(n) for n in _TIFF_FRAMES]   # [(img, th, meta), ...]
    mask = _tiff_mask(frames[0][0])
    bai = {"gi_mode_2d": "qip_qoop", "x_range": None, "y_range": None,
           "npt_rad": 500, "npt_azim": 500}

    pending_data = [(name, frames[i][0], frames[i][2])
                    for i, name in enumerate(_TIFF_FRAMES)]
    # Streaming with default (unfrozen, None-range) GI args + gi_freeze_mode=None
    # => each frame auto-ranges at its OWN incidence, exactly the per-frame path
    # the retired chunked freeze=False leg drove.  The assertions below are
    # therefore unchanged.
    out = _run_batch_streaming(poni, pending_data, mask)
    assert set(out) == {1, 2}

    for num, (img, th, _meta) in zip((1, 2), frames):
        batch_i2 = out[num].int_2d
        assert batch_i2 is not None
        az = np.asarray(batch_i2.azimuthal, float)
        # Non-degenerate qoop (the collapse bug gave span ~0.3; real ~5).
        assert (az.max() - az.min()) > 1.0, f"frame {num} qoop collapsed: {az.min()},{az.max()}"
        # Each batch frame integrated at ITS OWN incidence == direct ref.
        ref = _integrate_direct(poni, img, mask, th, bai)
        assert np.allclose(batch_i2.azimuthal, ref.azimuthal, atol=1e-4), \
            f"frame {num} (th={th}) qoop != serial-at-own-incidence"
        assert np.allclose(batch_i2.radial, ref.radial, atol=1e-4)
        _assert_good_gi_publication_passes(out[num])
        # The angle-dependence guard: frame 2 (th=0.35) must NOT match a
        # frame-0-incidence integration (the old stale-pool bug).
        if num == 2:
            wrong = _integrate_direct(poni, img, mask, frames[0][1], bai)
            assert not np.allclose(batch_i2.azimuthal, wrong.azimuthal, atol=1e-4), \
                "frame 2 integrated at frame-0 incidence — stale fiber pool regressed"


def test_batch_parallel_eiger_cake_nondegenerate_manual_incidence():
    # Eiger has no metadata -> incidence supplied manually (the GUI's
    # Manual Theta path: incidence_motor is the numeric angle string).
    # The batch must still produce a non-degenerate cake (not all-dummy,
    # not collapsed).
    import h5py
    from xrd_tools.core.containers import PONI

    poni_path = next(EIGER.glob("LaB6_detxn26*eta3.poni"), None)
    if poni_path is None:
        pytest.skip("eiger poni not found")
    poni = PONI.from_poni_file(str(poni_path))
    data_h5 = next(EIGER.glob("*_data_000001.h5"))
    with h5py.File(str(data_h5), "r") as f:
        img = np.asarray(f["entry/data/data"][0], dtype=np.float32)
    mask = (img < 0).astype(np.int8)

    out = _run_batch_streaming(
        poni, [(data_h5.name, img, {})], mask,
        incidence_motor="3.0",   # Manual incidence (deg)
    )
    i2 = out[1].int_2d
    assert i2 is not None
    az = np.asarray(i2.azimuthal, float)
    # Manual incidence resolved (3.0°), so the qoop axis must be a real
    # (non-collapsed) range, not the degenerate ~0 of a 0° integration.
    assert (az.max() - az.min()) > 1.0
    # And the cake must carry real signal, not an all-dummy grid.
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        _result_intensity_all_dummy,
    )
    assert not _result_intensity_all_dummy(i2)
    _assert_good_gi_publication_passes(out[1])


def test_standard_publication_live_batch_reload_equivalence(tmp_path):
    _assert_live_batch_reload_equivalent(tmp_path, gi=False)


def test_mode_a_chi_publication_live_batch_reload_equivalence(tmp_path):
    """Azimuthal Mode A (non-GI I-vs-chi, unit='chi_deg') must round-trip
    live == batch == reload, exactly like its GI sibling chi_gi (which is
    covered by test_gi_submode_...).  Guards the chi_deg run-path dispatch +
    the writer's 1D azimuthal axis_kind against a transpose/axis regression."""
    _assert_live_batch_reload_equivalent(tmp_path, gi=False, unit_1d="chi_deg")


def test_gi_qip_qoop_publication_live_batch_reload_equivalence(tmp_path):
    _assert_live_batch_reload_equivalent(tmp_path, gi=True)


def test_eiger_incidence_unresolved_without_metadata():
    # eiger masters have empty metadata -> 'th' motor can't resolve ->
    # must raise, not silently integrate at 0°.
    from xrd_tools.io.metadata import read_image_metadata
    from xdart.modules.live import LiveFrame, IncidenceAngleUnresolved

    master = next(EIGER.glob("*_master.h5"))
    meta = read_image_metadata(str(master), meta_format="txt")
    assert "th" not in meta and "eta" not in meta   # incidence is in the filename
    frame = LiveFrame(0, None, scan_info=dict(meta), gi=True, th_mtr="th")
    with pytest.raises(IncidenceAngleUnresolved):
        frame._get_incident_angle()


def test_tiff_frozen_gi_2d_range_matches_nonbatch_and_is_nondegenerate():
    from xrd_tools.integrate.gid import (
        create_fiber_integrator, integrate_gi_2d,
    )
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        _freeze_gi_2d_ranges_from_result, _result_intensity_all_dummy,
    )

    poni, th, img, mask = _load_tiff_frame0()
    assert th == pytest.approx(0.15)

    # GUI default sample orientation for this panel.
    fi = create_fiber_integrator(
        poni, incident_angle=th, sample_orientation=4, angle_unit="deg",
    )

    # Non-batch: auto range (what the live/serial path integrates).
    auto = integrate_gi_2d(img, fi, npt_rad=500, npt_azim=500, method="no",
                           mask=mask, radial_range=None, azimuth_range=None)
    assert not _result_intensity_all_dummy(auto)

    # Scout: freeze padded ranges from the auto result.
    args = {"gi_mode_2d": "qip_qoop", "x_range": None, "y_range": None}
    assert _freeze_gi_2d_ranges_from_result(args, auto)
    xr, yr = args["x_range"], args["y_range"]
    assert xr is not None and yr is not None
    assert (yr[1] - yr[0]) > 1.0          # qoop frozen range NOT degenerate
    assert (xr[1] - xr[0]) > 1.0          # qip frozen range NOT degenerate

    # Batch: re-integrate with the explicit frozen ranges
    # (integrate_2d maps x_range->radial_range, y_range->azimuth_range).
    frozen = integrate_gi_2d(img, fi, npt_rad=500, npt_azim=500, method="no",
                             mask=mask, radial_range=xr, azimuth_range=yr)
    assert not _result_intensity_all_dummy(frozen)

    # Frozen grid must track the non-batch auto grid (padding only).
    assert frozen.azimuthal.min() == pytest.approx(auto.azimuthal.min(), abs=0.3)
    assert frozen.azimuthal.max() == pytest.approx(auto.azimuthal.max(), abs=0.3)
    assert frozen.radial.min() == pytest.approx(auto.radial.min(), abs=0.3)
    assert frozen.radial.max() == pytest.approx(auto.radial.max(), abs=0.3)


# ---------------------------------------------------------------------------
# GI sub-mode coverage matrix (the publish gate).
#
# The Int-2D-mode crashes were really the *1D half* of a 2D run: a GI scan
# selects BOTH a gi_mode_1d and a gi_mode_2d, and on an angle-dependence batch
# the per-frame output axis drifts → the stacked NeXus writer (correctly)
# rejects the batch.  The freeze locks one common grid per scan.  These tests
# sweep every (gi_mode_1d × gi_mode_2d) combo on a MULTI-frame, two-incidence
# batch and assert the frozen stacks write uniformly AND that without the
# freeze they would not — so the regression lock demonstrably bites.
# ---------------------------------------------------------------------------


def _stack_results(out):
    """Sorted (frame_indices, results_1d, results_2d) from a batch capture."""
    idxs = sorted(out)
    return (idxs,
            [out[i].int_1d for i in idxs],
            [out[i].int_2d for i in idxs])


@pytest.mark.parametrize("gi_mode_2d", GI_MODES_2D)
@pytest.mark.parametrize("gi_mode_1d", GI_MODES_1D)
def test_gi_submode_multiframe_stack_writes_uniform(tmp_path, gi_mode_1d,
                                                    gi_mode_2d):
    """With the real scout freeze, a two-incidence GI batch writes BOTH the 1D
    and 2D stacks with no uniform-axes ValueError — for every sub-mode combo.

    This is the direct regression lock for the per-frame output-axis drift
    (incl. the ``azimuth_range`` key fix for q_oop / exit_angle): the freeze
    must produce one common grid, which the ssrl writer validators accept.
    """
    from xrd_tools.io.nexus import write_integrated_stack

    poni = _tiff_poni()
    raw = [_load_tiff(n) for n in _TIFF_FRAMES]
    mask = _tiff_mask(raw[0][0])
    img0, meta0 = raw[0][0], raw[0][2]

    bai_1d, bai_2d = _frozen_gi_bai_args(
        poni, img0, meta0, mask, gi_mode_1d=gi_mode_1d, gi_mode_2d=gi_mode_2d)

    # The freeze must have locked a non-degenerate output range on the RIGHT
    # key (q_oop/exit_angle → azimuth_range; else → radial_range).
    out_key = _gi_1d_output_key(gi_mode_1d)
    r1 = bai_1d.get(out_key)
    assert r1 is not None and (r1[1] - r1[0]) > 0.1, \
        f"{gi_mode_1d}: 1D output range {out_key} not frozen/degenerate: {r1}"
    for k in _gi_2d_range_keys(gi_mode_2d):
        r2 = bai_2d.get(k)
        assert r2 is not None and (r2[1] - r2[0]) > 0.1, \
            f"{gi_mode_2d}: 2D range {k} not frozen/degenerate: {r2}"

    pending = [(_TIFF_FRAMES[i], raw[i][0], raw[i][2]) for i in range(2)]
    # PRE-FROZEN args -> streaming locks ONE common grid for both frames.
    out = _run_batch_streaming(
        poni, pending, mask, gi=True,
        bai_1d_args=bai_1d, bai_2d_args=bai_2d,
    )
    assert set(out) == {1, 2}

    idxs, res_1d, res_2d = _stack_results(out)
    path = tmp_path / f"stack_{gi_mode_1d}_{gi_mode_2d}.nxs"
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        # Must NOT raise _require_uniform_axes_{1d,2d} — the freeze locked one
        # grid for both frames.  This mirrors scan._save_to_nexus()'s write.
        write_integrated_stack(
            entry, frame_indices=idxs, results_1d=res_1d, results_2d=res_2d)
    with h5py.File(path, "r") as h5:
        assert "entry" in h5  # round-tripped


@pytest.mark.parametrize("gi_mode_2d", GI_MODES_2D)
@pytest.mark.parametrize("gi_mode_1d", GI_MODES_1D)
def test_gi_submode_unfrozen_multiframe_stack_rejected(tmp_path, gi_mode_1d,
                                                       gi_mode_2d):
    """Proof the freeze is load-bearing: WITHOUT it, per-frame auto-range drift
    across the two incidences makes the stacked NeXus writer reject the batch.

    Every combo bites — the 1D modes drift on the 1D axis (q_ip/q_oop/
    exit_angle) and q_total drifts on the 2D axis — so the matrix above is not
    vacuously green.
    """
    from xrd_tools.io.nexus import write_integrated_stack

    poni = _tiff_poni()
    raw = [_load_tiff(n) for n in _TIFF_FRAMES]
    mask = _tiff_mask(raw[0][0])

    # No freeze: ranges defaulted to None → each frame auto-ranges at its own
    # incidence (the pre-fix per-frame path).  _run_batch_streaming pins
    # gi_freeze_mode=None, so with None-range args the session does NOT freeze a
    # common grid and the per-frame axes drift exactly as the old unfrozen path —
    # which the stacked writer must still reject.
    bai_1d = {"gi_mode_1d": gi_mode_1d, "numpoints": 128}
    bai_2d = {"gi_mode_2d": gi_mode_2d, "npt_rad": 64, "npt_azim": 48}

    pending = [(_TIFF_FRAMES[i], raw[i][0], raw[i][2]) for i in range(2)]
    out = _run_batch_streaming(
        poni, pending, mask, gi=True,
        bai_1d_args=bai_1d, bai_2d_args=bai_2d,
    )
    assert set(out) == {1, 2}

    idxs, res_1d, res_2d = _stack_results(out)
    with h5py.File(tmp_path / "unfrozen.nxs", "w") as h5:
        entry = h5.create_group("entry")
        # Anchor to the actual _require_uniform_axes_{1d,2d} messages
        # ("results_1d[i] has a different radial axis/unit ...",
        #  "results_2d[i] has a different q/chi axis ...") so an unrelated
        # ValueError that merely contains "different" can't mask a miss.
        with pytest.raises(ValueError,
                           match=r"results_[12]d\[\d+\] has a different"):
            write_integrated_stack(
                entry, frame_indices=idxs,
                results_1d=res_1d, results_2d=res_2d)


@pytest.mark.parametrize("gi_mode_2d", GI_MODES_2D)
@pytest.mark.parametrize("gi_mode_1d", GI_MODES_1D)
def test_gi_submode_publication_live_batch_reload_equivalence(
        tmp_path, gi_mode_1d, gi_mode_2d):
    """The equivalence spine, parametrized over every GI sub-mode combo:
    live ≡ batch ≡ reload for each (gi_mode_1d × gi_mode_2d)."""
    _assert_live_batch_reload_equivalent(
        tmp_path, gi=True, gi_mode_1d=gi_mode_1d, gi_mode_2d=gi_mode_2d)


def test_production_writer_persists_real_gi_accumulated_modes(tmp_path):
    from xdart.modules.ewald import LiveScan
    from xdart.modules.frame_publication import publication_from_live_frame
    from xrd_tools.core import FrameRecord, assert_frameview_equivalent
    from xrd_tools.io import read_frame_records
    from xrd_tools.io.nexus_record import quantize_thumbnail

    def _stored_thumbnail(thumbnail):
        if thumbnail is None:
            return None
        arr, (vmin, vmax, dtype) = quantize_thumbnail(np.asarray(thumbnail), dtype="uint8")
        scale = 65535.0 if dtype == "uint16" else 255.0
        return vmin + (arr.astype(float) / scale) * (vmax - vmin)

    def _as_written_record(record):
        views = tuple(record.results_1d.values()) + tuple(record.results_2d.values())
        thumbnail = next((view.thumbnail for view in views if view.thumbnail is not None), None)
        if thumbnail is None:
            return record
        stored = _stored_thumbnail(thumbnail)
        # This synthetic writer fixture persists the integrated stacks and compact
        # thumbnails, but not the optional per-frame geometry sidecar.
        return FrameRecord(
            label=record.label,
            results_1d={
                mode: replace(view, thumbnail=stored, incident_angle=None)
                for mode, view in record.results_1d.items()
            },
            results_2d={
                mode: replace(view, thumbnail=stored, incident_angle=None)
                for mode, view in record.results_2d.items()
            },
            active_mode_1d=record.active_mode_1d,
            active_mode_2d=record.active_mode_2d,
        )

    poni = _tiff_poni()
    raw = [_load_tiff(n) for n in _TIFF_FRAMES]
    mask = _tiff_mask(raw[0][0])
    pending = [
        (name, img, meta)
        for name, (img, _th, meta) in zip(_TIFF_FRAMES, raw)
    ]

    primary_1d, primary_2d = _frozen_gi_bai_args(
        poni, raw[0][0], raw[0][2], mask,
        gi_mode_1d="q_total", gi_mode_2d="qip_qoop",
    )
    extra_1d, extra_2d = _frozen_gi_bai_args(
        poni, raw[0][0], raw[0][2], mask,
        gi_mode_1d="q_ip", gi_mode_2d="q_chi",
    )
    primary = _run_batch_streaming(
        poni, pending, mask, gi=True,
        bai_1d_args=primary_1d, bai_2d_args=primary_2d,
    )
    extra = _run_batch_streaming(
        poni, pending, mask, gi=True,
        bai_1d_args=extra_1d, bai_2d_args=extra_2d,
    )

    out = tmp_path / "production_multimode.nxs"
    scan = LiveScan("production_multimode", data_file=str(out))
    scan.gi = True
    scan.skip_2d = False
    scan.bai_1d_args.update(primary_1d)
    scan.bai_2d_args.update(primary_2d)
    expected = []
    for label in sorted(primary):
        frame = primary[label]
        frame.gi = True
        frame.gi_1d = {"qtotal": frame.int_1d, "qip": extra[label].int_1d}
        frame.gi_2d = {"gi2d": frame.int_2d, "polar": extra[label].int_2d}
        frame.make_thumbnail(global_mask=getattr(scan, "global_mask", None))
        expected.append(_as_written_record(publication_from_live_frame(
            frame, active_mode_1d="q_total", active_mode_2d="qip_qoop",
        ).record))
        scan.add_frame(frame=frame, calculate=False, batch_save=True)

    scan._save_to_nexus(mode="w")
    reloaded = read_frame_records(out)
    assert len(reloaded) == len(expected)
    for want, got in zip(expected, reloaded):
        assert set(got.modes_1d) == {"q_total", "q_ip"}
        assert set(got.modes_2d) == {"qip_qoop", "q_chi"}
        assert_frameview_equivalent(want.view_1d("q_total"), got.view_1d("q_total"))
        assert_frameview_equivalent(want.view_1d("q_ip"), got.view_1d("q_ip"))
        assert_frameview_equivalent(want.view_2d("qip_qoop"), got.view_2d("qip_qoop"))
        assert_frameview_equivalent(want.view_2d("q_chi"), got.view_2d("q_chi"))


@pytest.mark.parametrize("gi_mode_1d", GI_MODES_1D)
def test_gi_submode_xye_only_uniform_xgrid(gi_mode_1d):
    """Int 1D (XYE) batch: per-frame .xye outputs must share ONE x-grid across
    incidences.  This path writes no .nxs (xye_only skips Phase-2), so the 1D
    scout freeze is what keeps the per-frame q axis from drifting.

    In production the xdart whole-scan prepass freezes ``scan.bai_1d_args`` from
    the real source files before the streaming session opens; this rig has no
    source files (the prepass self-skips), so we run the SAME 1D freeze
    explicitly first — then drive the real streaming ``_dispatch_batch`` (the 2D
    freeze self-skips on xye_only) and confirm the frozen grid holds per-frame.
    """
    poni = _tiff_poni()
    raw = [_load_tiff(n) for n in _TIFF_FRAMES]
    mask = _tiff_mask(raw[0][0])

    w, captured = _build_batch_thread(poni, mask, gi=True)
    w.xye_only = True              # Int 1D (XYE): no .nxs, per-frame xye only
    scan = _make_scan(
        poni, mask,
        {"gi_mode_1d": gi_mode_1d, "numpoints": 128},
        {"gi_mode_2d": "qip_qoop", "npt_rad": 64, "npt_azim": 48},
        gi=True,
    )
    scan.skip_2d = True            # Int 1D mode doesn't compute/write the cake
    pending = [(_TIFF_FRAMES[i], i + 1, raw[i][0], raw[i][2], 0.0, 0.0)
               for i in range(2)]
    w._freeze_gi_1d_auto_range(scan, pending)   # the prepass's 1D freeze, explicit
    w._dispatch_batch(scan, pending)            # streaming; session honours the grid
    w._close_reduction_session()                # finish -> QtNexusSink final flush

    assert set(captured) == {1, 2}
    r0 = np.asarray(captured[1].int_1d.radial, float)
    r1 = np.asarray(captured[2].int_1d.radial, float)
    assert r0.shape == r1.shape, \
        f"{gi_mode_1d}: per-frame xye x-grid length differs: {r0.shape} vs {r1.shape}"
    # Same tolerance the stacked writer uses for the uniform-axes check.
    assert np.allclose(r0, r1, rtol=1e-5, atol=1e-8), \
        f"{gi_mode_1d}: per-frame xye x-grid drifted across incidences"


# NOTE (4e): the old ``test_streaming_batch_xye_matches_chunked`` byte-equality
# test was retired with the chunked dispatcher.  Streaming per-frame 1D/2D
# correctness is now the spine's job — ``_assert_live_batch_reload_equivalent``
# proves live(serial) ≡ streaming ≡ reload for every GI sub-mode — and the
# per-frame-own-incidence integration is locked by
# ``test_batch_parallel_tiff_cakes_nondegenerate_and_match_serial`` (now on
# streaming).  A streaming-vs-single-frame-serial check is NOT equivalent: a
# multi-frame streaming batch derives its common output grid differently from a
# lone serial frame, so the grids legitimately differ.


def test_gi_freeze_covers_last_frame_extent():
    """Part B — freeze-clipping correctness check.

    The scout freeze locks the output range from frame 0 (+2% pad).  Assert the
    frozen 1D output axis (q_ip→radial, q_oop/exit_angle→azimuth) and the 2D
    out-of-plane y-range (qoop) COVER the LAST frame's natural (auto) extent
    across the tested incidence span (th 0.15→0.35°).

    Finding: on this data the 2% pad fully absorbs the incidence-induced output
    shift in BOTH scan directions (verified ascending AND descending), so the
    frame-0 freeze suffices — no clipping, and no freeze change is made.

    CAVEAT (flagged, not changed): a much wider incidence span — or a strongly
    non-monotonic scan whose widest-range (lowest-incidence) frame is not the
    scout — could shift the output axis beyond the 2% pad and clip later
    frames.  The robust fix would freeze from the union of the lowest- and
    highest-incidence scout frames.  The tested span does not clip, so per the
    task this is documented rather than applied.
    """
    from xrd_tools.integrate.calibration import poni_to_integrator
    from xrd_tools.integrate.gid import create_fiber_integrator
    from xdart.modules.live import LiveFrame

    poni = _tiff_poni()
    img0, th0, meta0 = _load_tiff(_TIFF_FRAMES[0])
    imgL, thL, metaL = _load_tiff(_TIFF_FRAMES[1])
    mask = _tiff_mask(img0)
    assert th0 < thL  # ascending: frame 0 is the lowest incidence (widest range)

    def _last_frame():
        return LiveFrame(
            5, imgL, poni=poni, scan_info=dict(metaL), static=True, gi=True,
            th_mtr="th", sample_orientation=4,
            integrator=poni_to_integrator(poni), mask=mask)

    fi = create_fiber_integrator(poni, incident_angle=thL,
                                 sample_orientation=4, angle_unit="deg")

    # ── 1D output-axis coverage, per mode ────────────────────────────────────
    for gi_mode_1d in GI_MODES_1D:
        bai_1d, _ = _frozen_gi_bai_args(
            poni, img0, meta0, mask, gi_mode_1d=gi_mode_1d, gi_mode_2d="qip_qoop")
        frozen = bai_1d[_gi_1d_output_key(gi_mode_1d)]
        frame = _last_frame()
        frame.integrate_1d(global_mask=mask, fiber_integrator=fi,
                           gi_mode_1d=gi_mode_1d, numpoints=128)
        r = np.asarray(frame.int_1d.radial, float)
        lo, hi = float(np.nanmin(r)), float(np.nanmax(r))
        # Tolerance at the writer's axis-precision floor (rtol 1e-5), NOT a
        # loose clipping margin — any real data truncation must fail here.
        tol = (frozen[1] - frozen[0]) * 1e-5
        assert frozen[0] - tol <= lo, \
            f"{gi_mode_1d}: frozen low {frozen[0]:.4f} clips last-frame low {lo:.4f}"
        assert hi <= frozen[1] + tol, \
            f"{gi_mode_1d}: frozen high {frozen[1]:.4f} clips last-frame high {hi:.4f}"

    # ── 2D out-of-plane (qoop / y_range) coverage ────────────────────────────
    _, bai_2d = _frozen_gi_bai_args(
        poni, img0, meta0, mask, gi_mode_1d="q_total", gi_mode_2d="qip_qoop")
    yr = bai_2d["y_range"]
    frame = _last_frame()
    frame.integrate_2d(global_mask=mask, fiber_integrator=fi,
                       gi_mode_2d="qip_qoop", npt_rad=64, npt_azim=48)
    az = np.asarray(frame.int_2d.azimuthal, float)
    lo, hi = float(np.nanmin(az)), float(np.nanmax(az))
    tol = (yr[1] - yr[0]) * 1e-5
    assert yr[0] - tol <= lo and hi <= yr[1] + tol, \
        f"2D qoop frozen y_range {yr} clips last-frame qoop ({lo:.4f},{hi:.4f})"


# ---------------------------------------------------------------------------
# Union-scout freeze coverage (#70): the freeze must bracket the WHOLE scan, not
# just frame 0.  The hard, data-independent bite (a single scout clipping a
# drifted second scout) is the headless ssrl freeze_common_axis unit test; here
# we confirm on real multi-incidence data that the union scout covers every
# frame and that the incidence extremes genuinely differ (so a single-frame
# pending[0] scout would clip the other extreme — the spine can't catch this
# because live/batch/reload all share the same frozen range).
#
# NOTE: the available angle-dependence fixtures span only th 0.15-0.35°, where
# the 2% pad largely absorbs the per-frame shift — so the *absolute* clip on
# this data is sub-pad.  A genuinely wide-incidence fixture would make the
# absolute clip large; the union fix is correct regardless and is bitten hard by
# the ssrl unit test.
# ---------------------------------------------------------------------------

def _freeze_1d_output_range(poni, mask, pending, gi_mode_1d):
    """Drive the production GI 1D freeze over ``pending`` and return the frozen
    output-axis (lo, hi) it wrote into bai_1d_args (or None)."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        gi_1d_output_axis_key,
    )
    w, _ = _build_batch_thread(poni, mask, gi=True)
    w.xye_only = False
    scan = _make_scan(
        poni, mask, {"gi_mode_1d": gi_mode_1d, "numpoints": 128},
        {"gi_mode_2d": "qip_qoop", "npt_rad": 64, "npt_azim": 48}, gi=True)
    w._freeze_gi_1d_auto_range(scan, pending)
    return scan.bai_1d_args.get(gi_1d_output_axis_key(gi_mode_1d))


def test_gi_union_scout_covers_all_frames_not_just_frame0():
    poni = _tiff_poni()
    names = [f"Combi4_Angledependence_samz_4p9_03271002_{i:04d}.tif"
             for i in range(1, 6)]                  # th 0.15 .. 0.35, 5 frames
    raw = [_load_tiff(n) for n in names]
    mask = _tiff_mask(raw[0][0])
    pending = [(names[i], i + 1, raw[i][0], raw[i][2], 0.0, 0.0)
               for i in range(len(names))]
    gi_mode_1d = "q_oop"                            # out-of-plane → incidence-sensitive

    union = _freeze_1d_output_range(poni, mask, pending, gi_mode_1d)
    assert union is not None
    singles = [_freeze_1d_output_range(poni, mask, [pending[i]], gi_mode_1d)
               for i in range(len(pending))]
    assert all(s is not None for s in singles)

    ulo, uhi = union
    # The union scout brackets EVERY frame's frozen extent — the whole scan.
    for i, (slo, shi) in enumerate(singles):
        assert ulo <= slo + 1e-9 and uhi >= shi - 1e-9, \
            f"union {union} does not cover frame {i} extent {(slo, shi)}"

    # Load-bearing: the two incidence extremes do NOT mutually bracket, so a
    # single-frame scout from one extreme clips the other — the union is needed.
    first, last = singles[0], singles[-1]

    def _brackets(a, b):
        return a[0] <= b[0] + 1e-9 and a[1] >= b[1] - 1e-9

    assert not (_brackets(first, last) and _brackets(last, first)), (
        "the incidence extremes have identical extents — union not exercised "
        "(need multi-incidence data)")
    # The union strictly extends beyond at least one single-frame scout.
    assert ulo < first[0] - 1e-12 or ulo < last[0] - 1e-12 \
        or uhi > first[1] + 1e-12 or uhi > last[1] + 1e-12

    # Order-independence (the operative fix vs the old pending[0] scout): the
    # freeze scouts by metadata incidence, so REVERSING the pending order yields
    # the SAME frozen range.  A single pending[0] scout is order-DEPENDENT (it
    # would freeze from whichever frame is first, clipping the rest on a
    # descending / unsorted scan) — which is exactly the bug this fixes.
    union_rev = _freeze_1d_output_range(
        poni, mask, list(reversed(pending)), gi_mode_1d)
    assert union_rev == pytest.approx(union), \
        "freeze is order-dependent — must scout by incidence, not position"
    # Sanity that the order test is meaningful: pending[0] differs between the
    # two orders, so an order-dependent (single-scout) freeze WOULD diverge.
    assert singles[0] != pytest.approx(singles[-1])


def test_gi_streaming_prepass_scouts_whole_scan_extremes():
    """BLOCKER 1: the streaming-batch GI pre-pass sweeps the WHOLE scan's
    metadata (not chunk 1) to find the global lowest+highest incidence frames,
    loads their images, and freezes the UNION grid -- so a multi-chunk angle-
    dependence batch can't clip a later, higher-incidence frame to the chunk-1
    grid.  Explicit contrast (3): a chunk-1-only freeze does NOT cover the
    global-max frame, which is exactly the bug this fixes.
    """
    from types import SimpleNamespace, MethodType
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread

    scan_name = "Combi4_Angledependence_samz_4p9_03271002"
    w = SimpleNamespace(
        incidence_motor="th",
        img_file=str(TIFF / f"{scan_name}_0001.tif"),
        img_dir=str(TIFF), scan_name=scan_name, img_ext="tif",
        meta_ext="txt", meta_dir=str(TIFF), inp_type=None,
        get_background=lambda *a, **k: 0.0,
    )
    w._enumerate_scan_files = MethodType(imageThread._enumerate_scan_files, w)
    w._frame_source_for = MethodType(imageThread._frame_source_for, w)
    w._gi_whole_scan_scout_entries = MethodType(
        imageThread._gi_whole_scan_scout_entries, w)

    # (1) The whole-scan metadata sweep finds the two incidence EXTREMES
    #     (th 0.15 = frame 1, th 0.35 = frame 5) and loads their images.
    status, entries = w._gi_whole_scan_scout_entries(None)
    assert status == "freeze"
    assert len(entries) == 2
    nums = sorted(e[1] for e in entries)
    assert nums == [1, 5], f"expected the global extremes (1, 5); got {nums}"
    for e in entries:
        assert e[2] is not None and np.asarray(e[2]).ndim == 2   # image loaded

    # (2) Those two extremes, handed to the production freeze, bracket EVERY
    #     frame's q_oop extent (the union the chunk-1 freeze would miss).
    poni = _tiff_poni()
    raw = [_load_tiff(f"{scan_name}_{i:04d}.tif") for i in range(1, 6)]
    mask = _tiff_mask(raw[0][0])
    extreme_pending = [(e[0], e[1], e[2], e[3], 0.0, 0.0) for e in entries]
    ulo, uhi = _freeze_1d_output_range(poni, mask, extreme_pending, "q_oop")
    singles = [_freeze_1d_output_range(
                   poni, mask,
                   [(f"f{i}", i + 1, raw[i][0], raw[i][2], 0.0, 0.0)], "q_oop")
               for i in range(5)]
    for i, (slo, shi) in enumerate(singles):
        assert ulo <= slo + 1e-9 and uhi >= shi - 1e-9, \
            f"prepass union ({ulo},{uhi}) clips frame {i + 1} extent {(slo, shi)}"

    # (3) Contrast (the bug): a chunk-1-only freeze (frames 1-2, th 0.15-0.20)
    #     does NOT cover the global-max frame 5 (th 0.35).
    chunk1 = [(f"f{i}", i + 1, raw[i][0], raw[i][2], 0.0, 0.0) for i in (0, 1)]
    c1_lo, c1_hi = _freeze_1d_output_range(poni, mask, chunk1, "q_oop")
    f5_lo, f5_hi = singles[4]
    assert not (c1_lo <= f5_lo + 1e-9 and c1_hi >= f5_hi - 1e-9), \
        "chunk-1-only freeze unexpectedly covers frame 5 -- test can't catch the bug"


def test_frame_source_for_rejects_neighbour_files(tmp_path):
    r"""ADR-0006 STEP 2: the strict factory anchors on ``^{scan}_\d+\.{ext}$``
    (NOT ``TiffSeriesSource.from_directory``).  Neighbour scans, background
    frames, and non-numeric / wrong-extension siblings sharing the folder must
    be EXCLUDED so the headless manifest sweeps only THIS scan's frames -- a
    directory sweep would scoop them up and skew the incidence extremes."""
    from types import SimpleNamespace, MethodType
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread
    from xrd_tools.sources.image import TiffSeriesSource

    keep = ["scan_0001.tif", "scan_0002.tif", "scan_0010.tif"]  # natural-int order
    decoys = [
        "scan_bg.tif",          # no numeric suffix
        "scan_0001_dark.tif",   # suffix is not pure digits
        "otherscan_0001.tif",   # different scan stem
        "scan_0003.txt",        # wrong extension
        "scan.tif",             # bare stem, no _<n>
    ]
    for name in keep + decoys:
        (tmp_path / name).touch()

    w = SimpleNamespace(
        single_img=False, inp_type=None,
        img_file=str(tmp_path / "scan_0001.tif"),
        img_dir=str(tmp_path), scan_name="scan", img_ext="tif",
        meta_ext="txt", meta_dir=str(tmp_path),
    )
    for m in ("_frame_source_for", "_enumerate_scan_files"):
        setattr(w, m, MethodType(getattr(imageThread, m), w))

    source = w._frame_source_for(None)
    assert isinstance(source, TiffSeriesSource)
    got = [Path(p).name for p in source.files]
    assert got == keep, f"strict factory admitted neighbour files: {got}"


def test_gi_prepass_scout_indices_map_back_to_noncontiguous_files(tmp_path):
    """Codex gate: a :class:`TiffSeriesSource` labels frames by POSITION
    (1..N) in the strict file list, so ``prepare_gi_freeze`` returns POSITIONAL
    scout indices.  With NON-contiguous, non-1-based on-disk numbering
    (recon_0003 / 0007 / 0012) the scout entries must carry the REAL on-disk
    img_numbers (3, 12), not the positional source indices (1, 3).  The
    contiguous Combi4 fixture (index == number) can't catch this confusion on
    its own, so the equivalence spine alone is a paper gate for the factory."""
    import shutil
    from types import SimpleNamespace, MethodType
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread

    src_scan = "Combi4_Angledependence_samz_4p9_03271002"
    # th 0.15 (global LO), 0.25 (mid), 0.35 (global HI), renumbered apart.
    renames = [("0001", "0003"), ("0003", "0007"), ("0005", "0012")]
    for src_n, dst_n in renames:
        shutil.copy(TIFF / f"{src_scan}_{src_n}.tif", tmp_path / f"recon_{dst_n}.tif")
        shutil.copy(TIFF / f"{src_scan}_{src_n}.txt", tmp_path / f"recon_{dst_n}.txt")

    w = SimpleNamespace(
        incidence_motor="th", single_img=False, inp_type=None,
        img_file=str(tmp_path / "recon_0003.tif"),
        img_dir=str(tmp_path), scan_name="recon", img_ext="tif",
        meta_ext="txt", meta_dir=str(tmp_path),
        get_background=lambda *a, **k: 0.0,
    )
    for m in ("_frame_source_for", "_enumerate_scan_files",
              "_gi_whole_scan_scout_entries"):
        setattr(w, m, MethodType(getattr(imageThread, m), w))

    status, entries = w._gi_whole_scan_scout_entries(None)
    assert status == "freeze"
    assert len(entries) == 2
    nums = sorted(e[1] for e in entries)
    assert nums == [3, 12], (
        "scout entries must carry the REAL on-disk img_numbers (3, 12), not the "
        f"positional source indices (1, 3); got {nums}")
    by_num = {e[1]: e for e in entries}
    # each scout loaded its 2D image and kept the matching incidence metadata
    assert np.asarray(by_num[3][2]).ndim == 2
    assert np.asarray(by_num[12][2]).ndim == 2
    assert float(by_num[3][3]["th"]) == 0.15    # lo scout == recon_0003
    assert float(by_num[12][3]["th"]) == 0.35   # hi scout == recon_0012


def test_gi_prepass_warns_and_proceeds_on_unestablishable_range():
    """T0-4 policy: when a multi-file scan's whole-scan incidence range CANNOT
    be established from metadata (>=2 files but no readable incidence), the
    pre-pass WARNS (user-visible advisory) and PROCEEDS on the session's own
    first-chunk freeze -- cropped extreme tails are accepted at the beamline;
    values inside the grid are exact.  The scan id latches so the advisory
    fires once per scan."""
    from types import SimpleNamespace, MethodType
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread

    scan_name = "Combi4_Angledependence_samz_4p9_03271002"
    emitted = []
    w = SimpleNamespace(
        gi=True, batch_mode=True, batch_execution="streaming",
        incidence_motor="no_such_motor",      # absent from every frame's metadata
        img_file=str(TIFF / f"{scan_name}_0001.tif"),
        img_dir=str(TIFF), scan_name=scan_name, img_ext="tif",
        meta_ext="txt", meta_dir=str(TIFF), inp_type=None,
        get_background=lambda *a, **k: 0.0, command="",
        showLabel=SimpleNamespace(emit=lambda m: emitted.append(m)),
    )
    for m in ("_gi_freeze_whole_scan_prepass", "_gi_ranges_fully_pinned", "_gi_whole_scan_scout_entries",
              "_frame_source_for", "_enumerate_scan_files", "_abort_gi_prepass",
              "_warn_gi_first_chunk_freeze"):
        setattr(w, m, MethodType(getattr(imageThread, m), w))

    # The sweep finds the 5 files but resolves no incidence for any -> "abort".
    status, entries = w._gi_whole_scan_scout_entries(None)
    assert status == "abort" and entries == []

    # T0-4: the orchestrator WARNS and PROCEEDS on the first-chunk freeze.
    scan = SimpleNamespace()
    proceed = w._gi_freeze_whole_scan_prepass(scan)
    assert proceed is True
    assert w.command == ""                       # run not stopped
    assert emitted and "set from the first frames" in emitted[-1]
    assert w._gi_prepass_scan_id == id(scan)     # latched: one advisory per scan


def test_gi_prepass_warns_and_proceeds_on_image_directory_source():
    """T0-4 policy: Image-Directory GI batch with a varying incidence motor
    returns 'unverifiable' (can't sweep the whole scan) and the orchestrator
    WARNS and PROCEEDS on the first-chunk freeze (cropped extreme tails are
    accepted at the beamline; values inside the grid are exact) instead of
    aborting -- the fail-closed posture turned common workflows into hard
    stops for a tail loss that doesn't matter scientifically."""
    from types import SimpleNamespace, MethodType
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread

    emitted = []
    w = SimpleNamespace(
        gi=True, batch_mode=True, batch_execution="streaming",
        incidence_motor="th",          # non-fixed motor (not float-parseable)
        img_file="/data/scan_0001.tif", inp_type="Image Directory",
        img_dir="/data", scan_name="scan", img_ext="tif",
        meta_ext="txt", meta_dir="/data",
        command="",
        showLabel=SimpleNamespace(emit=lambda m: emitted.append(m)),
    )
    for m in ("_gi_freeze_whole_scan_prepass", "_gi_ranges_fully_pinned", "_gi_whole_scan_scout_entries",
              "_frame_source_for", "_enumerate_scan_files", "_abort_gi_prepass",
              "_warn_gi_first_chunk_freeze"):
        setattr(w, m, MethodType(getattr(imageThread, m), w))

    # Image-Directory with a non-fixed motor -> "unverifiable" (can't sweep).
    status, entries = w._gi_whole_scan_scout_entries(None)
    assert status == "unverifiable" and entries == []

    # T0-4: orchestrator WARNS and PROCEEDS on the first-chunk freeze.
    scan = SimpleNamespace()
    proceed = w._gi_freeze_whole_scan_prepass(scan)
    assert proceed is True
    assert w.command == ""
    assert emitted and "set from the first frames" in emitted[-1]
    assert "Image Directory" in emitted[-1]      # advisory names the source
    assert w._gi_prepass_scan_id == id(scan)     # latched: one advisory per scan
    # Codex P2: the advisory is stamped on the scan so the writer persists it
    # in /entry/reduction/config -- a durable disclosure, not just a GUI label.
    assert "set from the first frames" in scan.gi_freeze_diagnostic


def _pinned_prepass_holder(emitted):
    from types import SimpleNamespace, MethodType
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread

    w = SimpleNamespace(
        gi=True, batch_mode=True, batch_execution="streaming",
        incidence_motor="th", inp_type="Image Directory",
        img_file="/data/scan_0001.tif", img_dir="/data", scan_name="scan",
        img_ext="tif", meta_ext="txt", meta_dir="/data",
        xye_only=False, command="",
        showLabel=SimpleNamespace(emit=lambda m: emitted.append(m)),
    )
    for m in ("_gi_freeze_whole_scan_prepass", "_gi_ranges_fully_pinned",
              "_abort_gi_prepass", "_warn_gi_first_chunk_freeze"):
        setattr(w, m, MethodType(getattr(imageThread, m), w))

    def _no_scout(scan):
        raise AssertionError(
            "scout sweep must not run when all GI ranges are pinned")
    w._gi_whole_scan_scout_entries = _no_scout
    return w


def test_gi_prepass_skips_scout_when_ranges_fully_pinned():
    """T0-3: with explicit 1D + 2D GI output ranges the freeze functions
    self-skip, so there is no auto grid to chunk-clip — an 'unverifiable'
    source (Image-Directory, varying motor) must proceed, not abort, and the
    whole-scan scout sweep must not even run."""
    from types import SimpleNamespace
    from xrd_tools.integrate.gid import gi_1d_output_axis_key

    emitted = []
    w = _pinned_prepass_holder(emitted)
    key_1d = gi_1d_output_axis_key("q_total")
    scan = SimpleNamespace(
        bai_1d_args={"gi_mode_1d": "q_total", key_1d: (0.0, 5.0)},
        bai_2d_args={"gi_mode_2d": "qip_qoop",
                     "x_range": (-10.0, 10.0), "y_range": (0.0, 5.0)},
        skip_2d=False,
    )

    proceed = w._gi_freeze_whole_scan_prepass(scan)

    assert proceed is True
    assert w.command == ""                       # no abort
    assert not emitted
    assert w._gi_prepass_scan_id == id(scan)     # decided for this scan


def test_gi_prepass_warns_when_ranges_partially_pinned():
    """T0-3/T0-4 counterpart: ONE missing range key means an auto freeze would
    still run somewhere — the run proceeds (T0-4 warn-and-proceed) but the
    pinned-ranges short-circuit must NOT silently claim full coverage: the
    first-chunk-freeze advisory still fires."""
    from types import SimpleNamespace, MethodType
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread
    from xrd_tools.integrate.gid import gi_1d_output_axis_key

    emitted = []
    w = _pinned_prepass_holder(emitted)
    # Restore the real scout path (the holder's raises) + its dependencies:
    # Image-Directory + non-fixed motor -> "unverifiable".
    for m in ("_gi_whole_scan_scout_entries", "_frame_source_for", "_enumerate_scan_files"):
        setattr(w, m, MethodType(getattr(imageThread, m), w))
    key_1d = gi_1d_output_axis_key("q_total")
    scan = SimpleNamespace(
        bai_1d_args={"gi_mode_1d": "q_total", key_1d: (0.0, 5.0)},
        bai_2d_args={"gi_mode_2d": "qip_qoop",
                     "x_range": (-10.0, 10.0), "y_range": None},  # one unpinned
        skip_2d=False,
    )

    proceed = w._gi_freeze_whole_scan_prepass(scan)

    assert proceed is True
    assert w.command == ""
    assert emitted and "set from the first frames" in emitted[-1]
    assert w._gi_prepass_scan_id == id(scan)


def test_gi_prepass_fails_closed_on_degenerate_scout_freeze():
    """BLOCKER 1 follow-up: a GIFreezeError raised by the whole-scan FREEZE (a
    degenerate / all-dummy scout cake) must FAIL CLOSED via _abort_gi_prepass --
    aborting the run loud -- not escape the worker thread (run() has no except,
    so an unhandled GIFreezeError would tear down the QThread)."""
    from types import SimpleNamespace, MethodType
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread
    from xrd_tools.reduction import GIFreezeError

    emitted = []
    w = SimpleNamespace(
        gi=True, batch_mode=True, batch_execution="streaming", command="",
        showLabel=SimpleNamespace(emit=lambda m: emitted.append(m)),
    )
    for m in ("_gi_freeze_whole_scan_prepass", "_gi_ranges_fully_pinned", "_abort_gi_prepass", "_warn_gi_first_chunk_freeze"):
        setattr(w, m, MethodType(getattr(imageThread, m), w))
    # The scout resolves two extremes ("freeze"), but the production freeze hits
    # a blank scout cake and raises GIFreezeError.
    w._gi_whole_scan_scout_entries = lambda scan: (
        "freeze", [("f", 1, None, {}, 0.0, 0.0)])

    def _boom(scan, scouts):
        raise GIFreezeError("blank GI scout cake")
    w._freeze_gi_1d_auto_range = _boom
    w._freeze_gi_2d_auto_ranges = _boom

    proceed = w._gi_freeze_whole_scan_prepass(SimpleNamespace())
    assert proceed is False                # fail closed, not raised
    assert w.command == "stop"
    assert emitted and "GI batch aborted" in emitted[-1]
    assert getattr(w, "_gi_prepass_scan_id", None) is None   # not latched


def test_gi_streaming_multichunk_later_chunk_uses_whole_scan_grid():
    """BLOCKER 1 TEST GAP: drive the streaming batch dispatcher over TWO chunks
    where the global-MAX-incidence frame (5, th 0.35) lands in the SECOND chunk
    (frame 1, th 0.15, is in chunk 1).  The whole-scan pre-pass runs on chunk 1
    and must freeze the UNION grid into ``scan.bai_1d_args`` so the persistent
    session integrates every later chunk onto an axis covering frame 5's full
    extent -- NOT clipped to chunk-1's (frames 1-2) grid, which is the multi-chunk
    bug.  This exercises the dispatch -> pre-pass -> session WIRING end-to-end; the
    single-chunk unit test (which calls the scout helper directly) can't, because
    the helper always sweeps the whole filesystem regardless of chunking."""
    from types import SimpleNamespace, MethodType
    from threading import RLock
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import wranglerThread
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        imageThread, gi_1d_output_axis_key)

    scan_name = "Combi4_Angledependence_samz_4p9_03271002"
    poni = _tiff_poni()
    raw = [_load_tiff(f"{scan_name}_{i:04d}.tif") for i in range(1, 6)]  # 1..5
    mask = _tiff_mask(raw[0][0])
    gi_mode_1d = "q_oop"                 # out-of-plane output: drifts with incidence

    w, captured = _build_batch_thread(poni, mask, gi=True)
    w.xye_only = True
    # Source attrs so the whole-scan pre-pass can metadata-sweep the 5-file series.
    w.img_file = str(TIFF / f"{scan_name}_0001.tif")
    w.img_dir = str(TIFF)
    w.scan_name = scan_name
    w.img_ext = "tif"
    w.meta_ext = "txt"
    w.meta_dir = str(TIFF)
    w.inp_type = None
    w.incidence_motor = "th"
    w.get_background = lambda *a, **k: 0.0
    # Streaming wiring (mirrors test_streaming_batch_xye_matches_chunked).
    w.batch_execution = "streaming"
    w.file_lock = RLock()
    w.sigUpdate = SimpleNamespace(emit=lambda *a: None)
    w.LIVE_SAVE_INTERVAL = 1000
    w._streaming_session = None
    w._streaming_sink = None
    w._streaming_scan_id = None
    w._reduction_session = None
    w._reduction_session_key = None
    w._cancel_token = MethodType(wranglerThread._cancel_token, w)
    w._close_reduction_session = MethodType(
        wranglerThread._close_reduction_session, w)
    for meth in ("_dispatch_batch_streaming", "_get_streaming_session",
                 "_gi_freeze_whole_scan_prepass", "_gi_ranges_fully_pinned", "_gi_whole_scan_scout_entries",
                 "_frame_source_for", "_enumerate_scan_files", "_abort_gi_prepass",
                 "_warn_gi_first_chunk_freeze", "_wait_if_paused"):
        setattr(w, meth, MethodType(getattr(imageThread, meth), w))

    scan = _make_scan(
        poni, mask, {"gi_mode_1d": gi_mode_1d, "numpoints": 128},
        {"gi_mode_2d": "qip_qoop", "npt_rad": 64, "npt_azim": 48}, gi=True)
    scan.skip_2d = True

    pending = [(str(TIFF / f"{scan_name}_{i + 1:04d}.tif"), i + 1,
                raw[i][0], raw[i][2], 0.0, 0.0) for i in range(5)]
    chunk1, chunk2 = pending[:2], pending[2:]   # the global max (frame 5) is in chunk 2

    w._dispatch_batch(scan, chunk1)
    assert w.command != "stop", "pre-pass aborted unexpectedly on real GI data"
    w._dispatch_batch(scan, chunk2)
    w._close_reduction_session()                # finish -> QtNexusSink final flush

    # The pre-pass froze the WHOLE-scan union into scan.bai_1d_args (the session
    # reads these to build its plan) -- not the chunk-1 grid.
    okey = gi_1d_output_axis_key(gi_mode_1d)
    frozen = scan.bai_1d_args.get(okey)
    union = _freeze_1d_output_range(poni, mask, pending, gi_mode_1d)
    c1 = _freeze_1d_output_range(poni, mask, chunk1, gi_mode_1d)
    assert frozen is not None, "pre-pass did not freeze a whole-scan grid"
    assert frozen == pytest.approx(union), \
        "pre-pass froze something other than the whole-scan incidence union"
    assert tuple(frozen) != pytest.approx(tuple(c1)), \
        "fixture degenerate: whole-scan union == chunk-1 grid (can't catch the bug)"

    # End-to-end: every frame (incl. frame 5 from chunk 2) was integrated onto
    # the one shared frozen grid.
    assert set(captured) == {1, 2, 3, 4, 5}
    np.testing.assert_array_equal(
        np.asarray(captured[5].int_1d.radial, float),
        np.asarray(captured[1].int_1d.radial, float))
