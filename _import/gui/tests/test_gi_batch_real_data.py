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


def _run_batch_parallel(poni, pending_data, mask, *, incidence_motor="th",
                        sample_orientation=4, gi=True,
                        bai_1d_args=None, bai_2d_args=None):
    """Drive the REAL imageThread._dispatch_batch_parallel on real frames.

    ``pending_data`` is a list of ``(name, img, scan_info)``.
    xye_only=True skips the Phase-2 HDF5 write but Phase-1 still integrates
    2D (skip_2d=False) and buffers each integrated frame into _xye_buffer.
    Returns {img_number: LiveFrame}.  No freeze is applied (we call the
    parallel dispatcher directly), so every frame auto-ranges at its OWN
    incidence — exactly the per-frame path the fiber-pool used to break.
    """
    from types import SimpleNamespace, MethodType
    from threading import RLock
    from ssrl_xrd_tools.integrate.calibration import poni_to_integrator
    from xdart.modules.reduction import StandardPlanCache
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import wranglerThread
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread

    if bai_1d_args is None:
        bai_1d_args = {"gi_mode_1d": "q_total"} if gi else {
            "numpoints": 256,
            "unit": "q_A^-1",
            "radial_range": (0.1, 6.0),
            "method": "csr",
        }
    if bai_2d_args is None:
        bai_2d_args = (
            {
                "gi_mode_2d": "qip_qoop", "x_range": None, "y_range": None,
                "npt_rad": 500, "npt_azim": 500,
            }
            if gi else
            {
                "npt_rad": 96,
                "npt_azim": 72,
                "unit": "q_A^-1",
                "radial_range": (0.1, 6.0),
                "azimuth_range": (-90.0, 90.0),
                "method": "csr",
            }
        )

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
    # Spy on the xye flush: snapshot the Phase-1 integrated frames rather
    # than writing xye files (and don't clear the buffer).
    captured = {}

    def _spy_flush(_scan, published_idxs=None):
        for num, fr in w._xye_buffer:
            captured[num] = fr
    w._flush_xye_buffer = _spy_flush

    # bg_raw=0 (no background) — matches _integrate_direct, which uses the
    # raw image; the real get_background returns 0 when no bg is configured.
    pending = [(name, i + 1, img, info, 0.0, 0.0)
               for i, (name, img, info) in enumerate(pending_data)]
    imageThread._dispatch_batch_parallel(w, scan, pending)
    return captured


def _run_live_single(poni, name, img, meta, mask, *, incidence_motor="th",
                     sample_orientation=4, gi=True,
                     bai_1d_args=None, bai_2d_args=None):
    """Drive the real sequential/live single-frame path and capture its frame."""

    from types import SimpleNamespace, MethodType
    from threading import Condition, RLock
    from ssrl_xrd_tools.integrate.calibration import poni_to_integrator
    from xdart.modules.reduction import StandardPlanCache
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import wranglerThread
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread

    if bai_1d_args is None:
        bai_1d_args = {"gi_mode_1d": "q_total"} if gi else {
            "numpoints": 256,
            "unit": "q_A^-1",
            "radial_range": (0.1, 6.0),
            "method": "csr",
        }
    if bai_2d_args is None:
        bai_2d_args = (
            {
                "gi_mode_2d": "qip_qoop", "x_range": None, "y_range": None,
                "npt_rad": 500, "npt_azim": 500,
            }
            if gi else
            {
                "npt_rad": 96,
                "npt_azim": 72,
                "unit": "q_A^-1",
                "radial_range": (0.1, 6.0),
                "azimuth_range": (-90.0, 90.0),
                "method": "csr",
            }
        )
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


def _assert_live_batch_reload_equivalent(tmp_path, *, gi):
    from ssrl_xrd_tools.core import assert_frameview_equivalent
    from xdart.modules.frame_publication import publication_from_live_frame

    poni = _tiff_poni()
    img, th, meta = _load_tiff(_TIFF_FRAMES[0])
    mask = _tiff_mask(img)

    if gi:
        scout_args = {"gi_mode_2d": "qip_qoop", "x_range": None, "y_range": None,
                      "npt_rad": 96, "npt_azim": 72}
        scout = _integrate_direct(poni, img, mask, th, scout_args)
        bai_1d = {"gi_mode_1d": "q_total", "numpoints": 256}
        bai_2d = {
            "gi_mode_2d": "qip_qoop",
            "x_range": (float(np.nanmin(scout.radial)), float(np.nanmax(scout.radial))),
            "y_range": (float(np.nanmin(scout.azimuthal)), float(np.nanmax(scout.azimuthal))),
            "npt_rad": 96,
            "npt_azim": 72,
        }
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
    pub_reload = _write_publication_reload(
        tmp_path / ("equiv_gi.nxs" if gi else "equiv_standard.nxs"),
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
