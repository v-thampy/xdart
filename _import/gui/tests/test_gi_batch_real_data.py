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
from pathlib import Path

import h5py
import numpy as np
import pytest

_DEFAULT_DATA = Path(__file__).resolve().parents[2] / "test_data"
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
GI_MODES_1D = ["q_total", "q_ip", "q_oop", "exit_angle"]
GI_MODES_2D = ["qip_qoop", "q_chi", "exit_angles"]

# GI 1D modes whose *output* axis is out-of-plane (frozen on ``azimuth_range``
# rather than ``radial_range`` — the crux of the per-frame-drift fix).
_GI_1D_OOP_MODES = ("q_oop", "exit_angle")


def _gi_1d_output_key(gi_mode_1d):
    return "azimuth_range" if gi_mode_1d in _GI_1D_OOP_MODES else "radial_range"


def _gi_2d_range_keys(gi_mode_2d):
    return ("x_range", "y_range") if gi_mode_2d == "qip_qoop" else (
        "radial_range", "azimuth_range")


def _load_tiff(name):
    import fabio
    from ssrl_xrd_tools.io.metadata import read_image_metadata
    p = TIFF / name
    meta = read_image_metadata(str(p), meta_format="txt")
    img = fabio.open(str(p)).data.astype(np.float32)
    return img, float(meta["th"]), dict(meta)


def _tiff_poni():
    from ssrl_xrd_tools.core.containers import PONI
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
    from ssrl_xrd_tools.integrate.gid import create_fiber_integrator, integrate_gi_2d
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

    This is the shared rig behind both :func:`_run_batch_parallel` (drives
    ``_dispatch_batch_parallel``) and :func:`_frozen_gi_bai_args` (drives the
    scout freeze) so the freeze sees the same thread state as a real batch.
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
        _executor=None, _executor_workers=0,
        showLabel=SimpleNamespace(emit=lambda *a: None),
        _middle_truncate=lambda t: t,
    )
    # Bind the real wranglerThread helpers (these are the integration path).
    for meth in ("_resolve_frame_mask", "_prewarm_frame_mask",
                 "_apply_threshold_inline", "_parallel_integrate",
                 "_get_executor", "_shutdown_executor"):
        setattr(w, meth, MethodType(getattr(wranglerThread, meth), w))
    w._borrow_fiber_integrator = wranglerThread._borrow_fiber_integrator  # staticmethod
    w._dispatch_batch_serial = MethodType(imageThread._dispatch_batch_serial, w)
    w._dispatch_batch_parallel = MethodType(imageThread._dispatch_batch_parallel, w)
    # The real freeze+dispatch orchestrator (freezes, then routes to the
    # parallel/serial dispatcher) — used to exercise the Int-1D-XYE path where
    # the 2D freeze self-skips on xye_only.
    w._dispatch_batch = MethodType(imageThread._dispatch_batch, w)
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
    from ssrl_xrd_tools.integrate.calibration import poni_to_integrator
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


def _run_batch_parallel(poni, pending_data, mask, *, incidence_motor="th",
                        sample_orientation=4, gi=True,
                        gi_mode_1d="q_total", gi_mode_2d="qip_qoop",
                        bai_1d_args=None, bai_2d_args=None, freeze=False):
    """Drive the REAL imageThread._dispatch_batch_parallel on real frames.

    ``pending_data`` is a list of ``(name, img, scan_info)``.
    xye_only=True skips the Phase-2 HDF5 write but Phase-1 still integrates
    2D (skip_2d=False) and buffers each integrated frame into _xye_buffer.
    Returns {img_number: LiveFrame}.

    ``gi_mode_1d`` / ``gi_mode_2d`` pick the GI sub-modes when the bai args are
    defaulted.  With ``freeze=False`` (default) every frame auto-ranges at its
    OWN incidence — the per-frame path the fiber-pool used to break, and the
    one that proves the un-frozen stack is non-uniform.  With ``freeze=True``
    the real scout freeze runs first (frame 0), locking one common grid for the
    whole batch — exactly what ``_dispatch_batch`` does before dispatching.
    """
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread

    if bai_1d_args is None:
        bai_1d_args = _default_bai_1d(gi, gi_mode_1d)
    if bai_2d_args is None:
        bai_2d_args = _default_bai_2d(gi, gi_mode_2d)

    w, captured = _build_batch_thread(
        poni, mask, incidence_motor=incidence_motor,
        sample_orientation=sample_orientation, gi=gi,
    )
    scan = _make_scan(poni, mask, bai_1d_args, bai_2d_args, gi=gi)

    # bg_raw=0 (no background) — matches _integrate_direct, which uses the
    # raw image; the real get_background returns 0 when no bg is configured.
    pending = [(name, i + 1, img, info, 0.0, 0.0)
               for i, (name, img, info) in enumerate(pending_data)]
    if freeze:
        # Mirror _dispatch_batch: freeze before dispatch.  The 2D freeze skips
        # on xye_only, so drop it for the freeze call only, then restore.
        prev_xye_only = w.xye_only
        w.xye_only = False
        w._freeze_gi_1d_auto_range(scan, pending)
        w._freeze_gi_2d_auto_ranges(scan, pending)
        w.xye_only = prev_xye_only
    imageThread._dispatch_batch_parallel(w, scan, pending)
    return captured


def _run_live_single(poni, name, img, meta, mask, *, incidence_motor="th",
                     sample_orientation=4, gi=True,
                     gi_mode_1d="q_total", gi_mode_2d="qip_qoop",
                     bai_1d_args=None, bai_2d_args=None):
    """Drive the real sequential/live single-frame path and capture its frame."""

    from types import SimpleNamespace, MethodType
    from threading import Condition, RLock
    from ssrl_xrd_tools.integrate.calibration import poni_to_integrator
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
        _middle_truncate=lambda t: t,
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
    from ssrl_xrd_tools.core import numeric_metadata
    from ssrl_xrd_tools.io.nexus import write_integrated_stack
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
                                         gi_mode_2d="qip_qoop"):
    """The equivalence spine: for one frame integrated under the given mode,
    the live (serial), batch (parallel), and NeXus-reload paths must produce
    byte-equivalent FrameViews.

    For GI the bai ranges are frozen by the REAL mode-aware scout freeze
    (:func:`_frozen_gi_bai_args`) so live and batch share one grid — exactly
    what ``_dispatch_batch`` does before either dispatch path runs.
    """
    from ssrl_xrd_tools.core import assert_frameview_equivalent
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
        bai_1d = {
            "numpoints": 256,
            "unit": "q_A^-1",
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
    batch = _run_batch_parallel(
        poni, [(_TIFF_FRAMES[0], img, meta)], mask,
        gi=gi, bai_1d_args=bai_1d, bai_2d_args=bai_2d,
    )[1]

    thumb_live = _canonicalize_thumbnail(live, mask)
    _canonicalize_thumbnail(batch, mask)
    pub_live = publication_from_live_frame(live)
    pub_batch = publication_from_live_frame(batch)
    fname = (f"equiv_gi_{gi_mode_1d}_{gi_mode_2d}.nxs" if gi
             else "equiv_standard.nxs")
    pub_reload = _write_publication_reload(
        tmp_path / fname,
        live,
        thumb_live,
    )

    assert_frameview_equivalent(pub_live.view, pub_batch.view)
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
    out = _run_batch_parallel(poni, pending_data, mask)
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
    from ssrl_xrd_tools.core.containers import PONI

    poni_path = next(EIGER.glob("LaB6_detxn26*eta3.poni"), None)
    if poni_path is None:
        pytest.skip("eiger poni not found")
    poni = PONI.from_poni_file(str(poni_path))
    data_h5 = next(EIGER.glob("*_data_000001.h5"))
    with h5py.File(str(data_h5), "r") as f:
        img = np.asarray(f["entry/data/data"][0], dtype=np.float32)
    mask = (img < 0).astype(np.int8)

    out = _run_batch_parallel(
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


def test_gi_qip_qoop_publication_live_batch_reload_equivalence(tmp_path):
    _assert_live_batch_reload_equivalent(tmp_path, gi=True)


def test_eiger_incidence_unresolved_without_metadata():
    # eiger masters have empty metadata -> 'th' motor can't resolve ->
    # must raise, not silently integrate at 0°.
    from ssrl_xrd_tools.io.metadata import read_image_metadata
    from xdart.modules.live import LiveFrame, IncidenceAngleUnresolved

    master = next(EIGER.glob("*_master.h5"))
    meta = read_image_metadata(str(master), meta_format="txt")
    assert "th" not in meta and "eta" not in meta   # incidence is in the filename
    frame = LiveFrame(0, None, scan_info=dict(meta), gi=True, th_mtr="th")
    with pytest.raises(IncidenceAngleUnresolved):
        frame._get_incident_angle()


def test_tiff_frozen_gi_2d_range_matches_nonbatch_and_is_nondegenerate():
    from ssrl_xrd_tools.integrate.gid import (
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
    from ssrl_xrd_tools.io.nexus import write_integrated_stack

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
    out = _run_batch_parallel(
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
    from ssrl_xrd_tools.io.nexus import write_integrated_stack

    poni = _tiff_poni()
    raw = [_load_tiff(n) for n in _TIFF_FRAMES]
    mask = _tiff_mask(raw[0][0])

    # No freeze: ranges defaulted to None → each frame auto-ranges at its own
    # incidence (the pre-fix per-frame path).
    bai_1d = {"gi_mode_1d": gi_mode_1d, "numpoints": 128}
    bai_2d = {"gi_mode_2d": gi_mode_2d, "npt_rad": 64, "npt_azim": 48}

    pending = [(_TIFF_FRAMES[i], raw[i][0], raw[i][2]) for i in range(2)]
    out = _run_batch_parallel(
        poni, pending, mask, gi=True,
        bai_1d_args=bai_1d, bai_2d_args=bai_2d, freeze=False,
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


@pytest.mark.parametrize("gi_mode_1d", GI_MODES_1D)
def test_gi_submode_xye_only_uniform_xgrid(gi_mode_1d):
    """Int 1D (XYE) batch: per-frame .xye outputs must share ONE x-grid across
    incidences.  This path writes no .nxs (xye_only skips Phase-2), so the 1D
    scout freeze — which runs even on xye_only — is what keeps the per-frame q
    axis from drifting.  We drive the real ``_dispatch_batch`` orchestrator so
    the 2D freeze self-skips (xye_only) exactly as in production.
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
    w._dispatch_batch(scan, pending)   # freezes 1D (runs) + 2D (skips), dispatches

    assert set(captured) == {1, 2}
    r0 = np.asarray(captured[1].int_1d.radial, float)
    r1 = np.asarray(captured[2].int_1d.radial, float)
    assert r0.shape == r1.shape, \
        f"{gi_mode_1d}: per-frame xye x-grid length differs: {r0.shape} vs {r1.shape}"
    # Same tolerance the stacked writer uses for the uniform-axes check.
    assert np.allclose(r0, r1, rtol=1e-5, atol=1e-8), \
        f"{gi_mode_1d}: per-frame xye x-grid drifted across incidences"


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
    from ssrl_xrd_tools.integrate.calibration import poni_to_integrator
    from ssrl_xrd_tools.integrate.gid import create_fiber_integrator
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
