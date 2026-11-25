"""P1–P6: real SPEC acquisitions through the mounted per-frame Live path."""

from dataclasses import replace
import math
import os
from pathlib import Path
import re
import sys
from threading import Event

import numpy as np
import pytest

from tests.core.test_stitch_operation import _native_form, _ring
from tests.xdart.pages import test_rsm_page as rsm_gui
from tests.xdart.pages.live_per_scan_support import tick_live, wait_until
from tests.xdart.pages.live_preview_support import advance_preview, start_preview, unimplemented_preview
from tests.xdart.pages.test_live_per_scan_native import (
    _append_spec, _click_run, _rsm_form, _same_science, _shots, _stop_live, _write_spec,
    make_page, qapp,  # noqa: F401
)
from tests.xdart.scattering.test_p1b_comprehensive_live import _write_flat
from xrd_tools.analysis import rsm_operation, rsm_values
from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
from xrd_tools.integrate.stitch_gi import StitchGI
from xrd_tools.io import analysis_artifact, descriptor_path, image, stat_identity
from xrd_tools.rsm.gridding import StreamingGridder
from xrd_tools.sources.point_images import PointImagePolicy

pytestmark = unimplemented_preview


def _scan(number, values, *, planned=64):
    lines = rsm_gui._spec_scan(number, 10).splitlines()
    lines[0] = f"#S {number} ascan eta 0 60 {planned - 1} 1"
    return "\n".join(lines[:6]) + "\n" + "".join(f"{value} 1 10 101\n" for value in values)


def _rsm_acquisition(page, root, *, count="1", values=(0,), planned=64):
    path = rsm_gui._synthetic_source(root, full=True)
    images = path.parent / "images"
    for file in images.iterdir():
        file.unlink()
    rsm_gui._spec_file(path, [_scan(1, values, planned=planned)])
    _rsm_form(page, root, count, root / "live")
    return path, images


def _raw(path, value, *, hot=None, header=b""):
    frame = np.full((195, 487), value, dtype=np.uint16)
    if hot is not None:
        frame[0, :4] = hot if isinstance(hot, tuple) else (hot, 275, value, value)
    path.write_bytes(header + frame.tobytes())
    return frame


def _stitch_scan(path, scan, deltas, *, planned=64):
    lines = path.read_text(encoding="utf-8").split("#S ", 1)[1].splitlines()
    assert lines[4].split()[:3] == ["#L", "del", "nu"]
    energy = " 17000.018" if len(lines[4].split()) == 4 else ""
    return (
        f"#S {scan} ascan del 14 35 {planned - 1} 1\n"
        + "\n".join(lines[1:5]) + "\n"
        + "".join(f"{delta:g} -10{energy}\n" for delta in deltas)
    )


def _stitch_acquisition(page, root, *, selection="1", backend="multigeometry", mode="1d"):
    form = _native_form(root, backend, scans=selection)
    path, images = Path(form.spec_path), Path(form.image_dir)
    text = path.read_text(encoding="utf-8")
    _write_spec(path, text[:text.index("#S 1")] + _stitch_scan(path, 1, (14.,)))
    for raw in images.iterdir():
        raw.unlink()
    page.load_form(replace(form, mode=mode, npt_azim_2d=30, npt_rad_2d=32))
    return form, path, images


def _stitch_raw(form, member, label, value):
    path = Path(form.image_dir) / f"sample_scan{member}_{label:04d}.raw"
    if value == 0 or form.backend == "xu_hist":
        pixels = np.full(form.detector_shape, value, dtype=np.int32)
    else:
        pixels = (_ring(label) * value).astype(np.int32)
    pixels.tofile(path)
    return path


def _next(page, owner, qapp, predicate, *, attempts=6):
    for _ in range(attempts):
        advance_preview(page, owner, qapp)
        if predicate():
            return
    assert predicate(), page.run_bar.readinessLabel.full_text()


def _prefix(owner, *keys):
    return owner.snapshot is not None and tuple(owner.snapshot.points) == keys


def _truncated_frame(path, kind):
    if kind == "raw":
        _raw(path, 45, header=b"SIXHDR")
        full = path.read_bytes()
        return full, full[:-1]
    _write_flat(path, kind, 45)
    full = path.read_bytes()
    if kind == "tif":
        return full, full[:16]
    starter = full.index(b"\x0c\x1a\x04\xd5")
    size = int(re.search(rb"X-Binary-Size:\s*(\d+)", full[:starter]).group(1))
    return full, full[:starter + 4 + size - 1]


@pytest.mark.parametrize("kind", ("raw", "tif", "cbf"))
def test_p1_partial_files_complete_without_finite_stillness(qapp, make_page, tmp_path, monkeypatch, kind):
    page = make_page("rsm")
    spec, images = _rsm_acquisition(page, tmp_path)
    frame = images / f"{spec.stem}_scan1_0000.{kind}"
    full, partial = _truncated_frame(frame, kind)
    frame.write_bytes(partial)
    if kind == "raw":
        page.set_field("adv_header", "6")
    raw_reader = image._read_raw_binary
    raw_calls = []

    def record_raw(path, *args, **kwargs):
        if Path(path) == frame:
            raw_calls.append(Path(path))
        return raw_reader(path, *args, **kwargs)

    monkeypatch.setattr(image, "_read_raw_binary", record_raw)
    owner = start_preview(page, qapp)
    advance_preview(page, owner, qapp)
    assert owner.snapshot is None and owner.progress.waiting
    assert not raw_calls and not _shots(tmp_path / "live")
    frame.write_bytes(full)
    advance_preview(page, owner, qapp)
    wait_until(qapp, lambda: owner.progress.included == 1)
    assert not _shots(tmp_path / "live")
    _next(page, owner, qapp, lambda: _prefix(owner, ("1.1", 0)))
    count = owner.progress.decodes
    advance_preview(page, owner, qapp)
    assert owner.progress.decodes == count == 1
    assert len(raw_calls) == int(kind == "raw")


@pytest.mark.parametrize("key", ("1", "1.1", "1.2"))
def test_p1_occurrence_origin_partial_row_and_point_mapping(qapp, make_page, tmp_path, key):
    page = make_page("rsm")
    spec, images = _rsm_acquisition(page, tmp_path)
    selected_key = key if "." in key else "1.1"
    page.set_field("image_stem", "{spec}_scan{occurrence}_")
    contents = spec.read_text(encoding="utf-8")
    prefix = contents[:contents.index("#S 1 ")]
    first = _scan(1, (11,), planned=1) if selected_key == "1.2" else ""
    _write_spec(spec, prefix + first + _scan(1, ()) + "0 1 10")
    page.set_field("scan", key)
    files = [images / f"{spec.stem}_scan{selected_key}_{label:04d}.raw" for label in range(2)]
    _raw(files[1], 60)
    owner = start_preview(page, qapp)
    advance_preview(page, owner, qapp)
    assert owner.snapshot is None and owner.progress.included == 0
    _append_spec(spec, " 101\n")
    advance_preview(page, owner, qapp)
    assert owner.snapshot is None, "_0001 alone has no established origin"
    _raw(files[0], 40)
    _next(page, owner, qapp, lambda: _prefix(owner, (selected_key, 0)))
    assert owner.progress.decodes == 1, "an extra raw index must not become another logical point"


def test_p1_grouped_point_waits_for_both_bound_raw_slots(qapp, make_page, tmp_path):
    page = make_page("rsm")
    spec, images = _rsm_acquisition(page, tmp_path)
    page.point_images.restore({"choices": [{
        "uri": str(spec), "scan": "1.1", "policy": PointImagePolicy(2, "difference").to_dict(),
    }]})
    _raw(images / f"{spec.stem}_scan1_0000.raw", 40)
    owner = start_preview(page, qapp)
    advance_preview(page, owner, qapp)
    assert owner.snapshot is None and owner.progress.included == 0
    _raw(images / f"{spec.stem}_scan1_0001.raw", 65)
    _next(page, owner, qapp, lambda: _prefix(owner, ("1.1", 0)))
    assert owner.progress.decodes == 2
    included = owner.snapshot.intensity[np.isfinite(owner.snapshot.intensity)]
    assert included.size
    np.testing.assert_allclose(included, 25)


def test_p1_denied_preview_source_read_waits_and_retries(qapp, make_page, tmp_path, monkeypatch):
    page = make_page("rsm")
    spec, images = _rsm_acquisition(page, tmp_path)
    open_file = os.open
    blocked, denied = True, []

    def sharing_denied(path, *args, **kwargs):
        if blocked and Path(path) == spec:
            denied.append(path)
            raise PermissionError("the acquisition temporarily denies a compatible read")
        return open_file(path, *args, **kwargs)

    owner = start_preview(page, qapp)
    monkeypatch.setattr(os, "open", sharing_denied)
    _raw(images / f"{spec.stem}_scan1_0000.raw", 40)
    _append_spec(spec, "0.01 1 10 101\n")
    advance_preview(page, owner, qapp)
    assert denied and owner.progress.waiting and owner.snapshot is None
    blocked = False
    _next(page, owner, qapp, lambda: _prefix(owner, ("1.1", 0)))
    assert not _shots(tmp_path / "live") and owner.progress.decodes == 1


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows share flags and rename semantics")
def test_p1_native_windows_preview_handles_allow_a_writer_and_rename(qapp, make_page, tmp_path, monkeypatch):
    page = make_page("rsm")
    spec, images = _rsm_acquisition(page, tmp_path)
    frame = images / f"{spec.stem}_scan1_0000.raw"
    landed = []
    real_open = descriptor_path._win32_open_existing

    def write_while_open(path, access, share, flags):
        descriptor = real_open(path, access, share, flags)
        if Path(path) == frame and not landed:
            try:
                required = descriptor_path._FILE_SHARE_WRITE | descriptor_path._FILE_SHARE_DELETE
                assert share & required == required
                with frame.open("r+b") as writer:
                    writer.write(b"\x29\x00")
                moving = frame.with_suffix(".moving")
                frame.replace(moving)
                moving.replace(frame)
                landed.append(frame)
            except BaseException:
                os.close(descriptor)
                raise
        return descriptor

    owner = start_preview(page, qapp)
    monkeypatch.setattr(descriptor_path, "_win32_open_existing", write_while_open)
    _raw(frame, 40)
    _next(page, owner, qapp, lambda: _prefix(owner, ("1.1", 0)))
    assert landed == [frame] and not _shots(tmp_path / "live")


@pytest.mark.parametrize("change", ("append", "rewrite", "replace-raw", "replace-spec", "calibration", "windows"))
def test_p2_adoption_rechecks_the_covered_prefix_and_assets(qapp, make_page, tmp_path, monkeypatch, change):
    page = make_page("rsm")
    spec, images = _rsm_acquisition(page, tmp_path, values=(0., 0.05))
    raw = images / f"{spec.stem}_scan1_0000.raw"
    if change == "windows":
        monkeypatch.setattr(stat_identity, "IDENTITY_CARRIES_CTIME", False)
    owner = start_preview(page, qapp)
    # Hold delivery of the real terminal after compute, with actual Qt and watcher pumping.
    worker = owner._worker
    poll = worker.poll
    terminal_waiting, open_delivery, delivered = Event(), Event(), Event()

    def hold(identity):
        if worker._terminal is not None and (not open_delivery.is_set() or delivered.is_set()):
            terminal_waiting.set()
            return None
        terminal = poll(identity)
        if terminal is not None and open_delivery.is_set():
            delivered.set()
        return terminal

    monkeypatch.setattr(worker, "poll", hold)
    _raw(raw, 32)
    _raw(raw.with_name(raw.name.replace("0000", "0001")), 55)
    advance_preview(page, owner, qapp, drain=False)
    wait_until(qapp, terminal_waiting.is_set)
    if change == "append":
        _append_spec(spec, "0.1 1 10 101\n")
    elif change == "replace-raw":
        replacement = raw.with_suffix(".replacement")
        _raw(replacement, 77)
        replacement.replace(raw)
    elif change == "replace-spec":
        replacement = spec.with_name("replacement")
        replacement.write_bytes(spec.read_bytes())
        replacement.replace(spec)
    elif change == "calibration":
        geometry = tmp_path / "calibration/rsm_canonical.xu.json"
        geometry.write_bytes(geometry.read_bytes() + b"\n")
    else:
        previous = spec.stat()
        _write_spec(spec, spec.read_text(encoding="utf-8").replace("0.05 1", "0.07 1"))
        if change == "windows":
            os.utime(spec, ns=(previous.st_atime_ns, previous.st_mtime_ns))
            assert spec.stat().st_size == previous.st_size
    open_delivery.set()
    owner.poll_due()
    wait_until(qapp, delivered.is_set)
    if change == "append":
        assert _prefix(owner, ("1.1", 0), ("1.1", 1))
    else:
        assert owner.snapshot is None
        assert owner.progress.invalidated > 0
    monkeypatch.setattr(worker, "poll", poll)


@pytest.mark.parametrize("order", ("[2-3]", "[3,2]", "2,[2-3]"))
def test_p3_preview_group_is_an_ordered_prefix_of_its_native_slot(qapp, make_page, tmp_path, order):
    page = make_page("stitch")
    form = replace(_native_form(tmp_path, "multigeometry", scans=order), output_family=None)
    spec = Path(form.spec_path)
    contents = spec.read_text(encoding="utf-8")
    start, end = contents.index("#S 2 "), contents.index("#S 3 ")
    _write_spec(spec, contents[:start] + _stitch_scan(spec, 2, (14, 14.1, 14.2), planned=3))
    for index in range(3):
        _stitch_raw(form, 2, index, 1)
    page.load_form(form)
    owner = start_preview(page, qapp)
    if order == "[3,2]":
        advance_preview(page, owner, qapp)
        assert owner.snapshot is None
    else:
        _next(page, owner, qapp, lambda: owner.snapshot is not None)
        assert {key for key, _ in owner.snapshot.points} == {"2.1"}
        if order == "2,[2-3]":
            wait_until(qapp, lambda: len(_shots(tmp_path / "output")) == 1)
        else:
            assert not _shots(tmp_path / "output")
    # Leave scan 3 incomplete long enough to distinguish the order, then seal it.
    _append_spec(spec, "\n" + _stitch_scan(spec, 3, (14, 14.1, 14.2), planned=4))
    for index in range(3):
        _stitch_raw(form, 3, index, 2)
    _next(page, owner, qapp, lambda: owner.snapshot is not None)
    if order == "[3,2]":
        assert {key for key, _ in owner.snapshot.points} == {"3.1"}
    _append_spec(spec, "14.3 -10\n")
    _stitch_raw(form, 3, 3, 2)
    for _ in range(4):
        tick_live(page, qapp)
    outputs = tmp_path / "output"
    expected_count = 2 if order == "2,[2-3]" else 1
    wait_until(qapp, lambda: len(_shots(outputs)) == expected_count)
    assert all(analysis_artifact.read_analysis_artifact(path).provenance_json for path in outputs.glob("*.nexus"))
    assert len(owner.retired_groups) == expected_count
    ordered = ("3.1", "2.1") if order == "[3,2]" else ("2.1", "3.1")
    assert ordered in [tuple(group.members) for group in owner.retired_groups]


@pytest.mark.parametrize(("requested", "preview_bins"), [
    ((12, 16, 20), (12, 16, 20)),
    ((1000, 40, 40), (855, 34, 34)),
    ((1_000_000, 2, 2), (250_000, 2, 2)),
])
def test_p4_preview_voxel_cap_is_independent_of_the_saved_grid(
    qapp, make_page, tmp_path, requested, preview_bins,
):
    page = make_page("rsm")
    spec, images = _rsm_acquisition(page, tmp_path, planned=2)
    for field, count in zip(("grid_h", "grid_k", "grid_l"), requested, strict=True):
        page.set_field(field, count)
    _raw(images / f"{spec.stem}_scan1_0000.raw", 40)
    owner = start_preview(page, qapp)
    _next(page, owner, qapp, lambda: owner.snapshot is not None)
    assert owner.snapshot.intensity.shape == preview_bins
    assert math.prod(preview_bins) <= 1_000_000
    if requested == (1000, 40, 40):
        _append_spec(spec, "1 1 10 101\n")
        _raw(images / f"{spec.stem}_scan1_0001.raw", 41)
        for _ in range(3):
            tick_live(page, qapp)
        wait_until(qapp, lambda: bool(_shots(tmp_path / "live")), timeout_ms=60000)
        [path] = list((tmp_path / "live").glob("*.nexus"))
        assert analysis_artifact.read_analysis_artifact(path).intensity.shape == requested


def _rsm_reference(page, snapshot, mask):
    """Bind the real finite source separately and feed its selected labels onto the preview axes."""
    native = rsm_values.prepare_rsm_tool_v2(page.build_forms()[0]).request
    assert len(native.preflight.members) == 1
    member, = native.preflight.members
    receipt, = native.module.source.members
    labels = {label for key, label in snapshot.points if key == "1.1"}
    contributions = tuple(item for item in member.contributions if item.label in labels)
    assert len(contributions) == len(snapshot.points)
    plan = native.plan
    geometry = plan.effective_geometry
    roi = geometry.roi
    r0, r1, c0, c1 = roi or (0, mask.shape[0], 0, mask.shape[1])
    cropped = np.array(mask[r0:r1, c0:c1], copy=True)
    cropped.setflags(write=False)
    with xu_runtime_session(geometry.runtime_requirements) as session:
        mapper = rsm_operation.rsm_effective_pixel_q_map(geometry)
        bins = tuple(len(axis) for _, axis in snapshot.axes)
        gridder = StreamingGridder(mapper, bins, runtime_session=session, coordinate_frame=plan.coordinate_frame)
        gridder.set_bounds(*((axis[0], axis[-1]) for _, axis in snapshot.axes))
        state = rsm_operation._rsm_correction_state(
            plan, geometry.detector_header, geometry.diffractometer_projection,
            native.preflight.geometry_asset_receipt,
        )
        with rsm_operation.requalified_analysis_source(receipt.analysis) as source:
            for item in contributions:
                angles = rsm_operation._rsm_v2_contribution_angles(member, (item,))
                lease = rsm_operation._make_rsm_v2_chunk_lease(
                    source, member, plan, (item,), cropped_static_mask=cropped,
                    cancel_token=None, corrections=state,
                )
                weights = rsm_operation._rsm_chunk_weights(state, mapper, session, angles, member.energy_eV, roi)
                gridder.add_leased(lease, angles, member.energy_eV, UB=np.array(member.ub), roi=roi,
                                   weight=1.0 if weights is None else weights)
        return gridder.to_volume().intensity


def test_p4_mask_shrinks_without_freezing_forward_progress(qapp, make_page, tmp_path):
    page = make_page("rsm")
    spec, images = _rsm_acquisition(page, tmp_path, planned=6)
    page.set_field("static_hot", "100")
    _raw(images / f"{spec.stem}_scan1_0000.raw", 40, hot=250)
    owner = start_preview(page, qapp)
    _next(page, owner, qapp, lambda: _prefix(owner, ("1.1", 0)))
    assert owner.snapshot.mask_pending
    mask = np.zeros((195, 487), dtype=bool)
    np.testing.assert_allclose(owner.snapshot.intensity, _rsm_reference(page, owner.snapshot, mask), equal_nan=True)
    for label, angle, hot in ((1, 0.001, 250), (2, 0.002, 42), (3, 0.003, 42)):
        _append_spec(spec, f"{angle} 1 10 101\n")
        _raw(images / f"{spec.stem}_scan1_{label:04d}.raw", 40 + label, hot=hot)
        _next(page, owner, qapp, lambda: _prefix(owner, *(("1.1", i) for i in range(label + 1))))
        mask[0, :2] = (label < 3, True)
        assert not owner.snapshot.mask_pending
        assert owner.snapshot.mask_update_queued == (label == 2)
        np.testing.assert_array_equal(owner.snapshot.member_masks["1.1"], mask)
        np.testing.assert_allclose(owner.snapshot.intensity, _rsm_reference(page, owner.snapshot, mask),
                                   rtol=1e-7, equal_nan=True)
    past = owner.snapshot.bounds
    _append_spec(spec, "55 1 10 101\n")
    _raw(images / f"{spec.stem}_scan1_0004.raw", 45, hot=42)
    _next(page, owner, qapp, lambda: owner.snapshot is not None and owner.snapshot.bounds != past)
    np.testing.assert_allclose(owner.snapshot.intensity, _rsm_reference(page, owner.snapshot, mask),
                               rtol=1e-7, equal_nan=True)
    _append_spec(spec, "56 1 10 101\n")
    _raw(images / f"{spec.stem}_scan1_0005.raw", 46, hot=42)
    for _ in range(3):
        tick_live(page, qapp)
    wait_until(qapp, lambda: bool(_shots(tmp_path / "live")), timeout_ms=60000)
    _stop_live(qapp, page)
    finite = make_page("rsm")
    _rsm_form(finite, tmp_path, "1", tmp_path / "finite")
    finite.set_field("static_hot", "100")
    _click_run(qapp, finite)
    [live_path] = (tmp_path / "live").glob("*.nexus")
    [finite_path] = (tmp_path / "finite").glob("*.nexus")
    _same_science(analysis_artifact.read_analysis_artifact(live_path),
                  analysis_artifact.read_analysis_artifact(finite_path))


def _stitch_reference(form, root, snapshot, mode):
    """The finite native Stitch Run of the same settled scan on the preview's own grid (Claude's oracle check).

    The preview's axes are bin centres; the finite range spans half a bin past the first and last centre, so the
    finite bins coincide with the preview's. Axes are compared first so a convention mismatch is named plainly.
    """
    from xrd_tools.analysis.stitch_operation import run_stitch_operation
    from xrd_tools.analysis.stitch_values import prepare_stitch_tool

    axes = tuple(np.asarray(axis, dtype=float) for _, axis in snapshot.axes)

    def span(axis):
        step = (axis[-1] - axis[0]) / (len(axis) - 1)
        return float(axis[0] - step / 2), float(axis[-1] + step / 2)

    if mode == "1d":
        grid = dict(q_range=span(axes[0]), npt_1d=len(axes[0]))
    else:
        grid = dict(map_radial_range=span(axes[0]), npt_rad_2d=len(axes[0]),
                    chi_range=span(axes[1]), npt_azim_2d=len(axes[1]))
    folder = root / "reference"
    folder.mkdir(exist_ok=True)
    form = replace(form, output_path=str(folder / "reference.nexus"), **grid)
    request, = prepare_stitch_tool(form).requests
    result = run_stitch_operation(request)
    assert result.terminal.disposition.value == "committed", result.terminal
    payload = result.payload
    for (_, actual), expected in zip(payload.axes, axes, strict=True):
        # Saved results keep float32 axes and intensities (the published format): tolerate float32 rounding,
        # scaled to the axis magnitude so a centre at zero is not held to a relative bound.
        np.testing.assert_allclose(np.asarray(actual, dtype=float), expected, rtol=1e-6,
                                   atol=1e-6 * float(np.max(np.abs(expected))))
    intensity = np.asarray(payload.intensity)
    # The saved 2-D result is (radial, azimuthal); the preview snapshot is (chi, q), as P5 asserts above.
    return intensity.T if mode == "2d" else intensity


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
@pytest.mark.parametrize("mode", ("1d", "2d"))
def test_p5_stitch_growth_preserves_the_screen_tail_and_native_axes(
    qapp, make_page, tmp_path, backend, mode,
):
    page = make_page("stitch")
    form, spec, _ = _stitch_acquisition(page, tmp_path, backend=backend, mode=mode)
    page.load_form(replace(form, mode=mode, npt_azim_2d=30, npt_rad_2d=32,
                           chi_range=(-179., 179.) if mode == "2d" else None))
    reference_form = page.build_form()  # the frozen Live inputs, built before Live starts watching
    _stitch_raw(form, 1, 0, 1)
    owner = start_preview(page, qapp)
    _next(page, owner, qapp, lambda: owner.progress.held == 1)
    assert owner.snapshot is None
    advance_preview(page, owner, qapp, seconds=6)
    assert owner.snapshot is None and owner.progress.held == 1
    suffix = " -10" if backend == "multigeometry" else " -10 17000.018"
    for i, delta in enumerate((14.2, 14.4, 29., 30., 31.), 1):
        _append_spec(spec, f"{delta}{suffix}\n")
        _stitch_raw(form, 1, i, 0 if i == 1 else 1)
        advance_preview(page, owner, qapp)
        if i == 2:
            assert owner.progress.held == 2 and owner.progress.included == 1
            prior = owner.snapshot.bounds
        if i == 4:
            assert owner.progress.excluded == 1
            assert owner.progress.held == 2 and owner.snapshot.bounds != prior
    _append_spec(spec, "\n" + _stitch_scan(spec, 2, (14.,), planned=2))
    advance_preview(page, owner, qapp)
    wait_until(qapp, lambda: owner.progress.end_of_stream and not owner.working)
    assert owner.progress.excluded == 1 and owner.progress.held == 0
    snapshot = owner.snapshot
    axes = tuple(np.asarray(axis) for _, axis in snapshot.axes)
    expected = tuple(len(axis) for axis in reversed(axes)) if mode == "2d" else (len(axes[0]),)
    assert snapshot.intensity.shape == expected
    assert all(np.all(np.diff(axis) > 0) for axis in axes)
    assert np.any(np.isfinite(snapshot.intensity)) and np.any(snapshot.normalization > 0)
    if backend == "xu_hist":
        assert snapshot.sigma is None
    else:
        assert np.any(snapshot.coverage > 0)
    if mode == "2d":
        assert np.min(axes[1]) >= -179. and np.max(axes[1]) <= 179.
    # Claude's oracle check: the settled preview equals the native finite science on the preview's own grid.
    np.testing.assert_allclose(snapshot.intensity, _stitch_reference(reference_form, tmp_path, snapshot, mode),
                               rtol=1e-6, equal_nan=True)


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
def test_p5_gi_preview_uses_the_native_primary_map(qapp, make_page, tmp_path, backend):
    page = make_page("stitch")
    form, spec, _ = _stitch_acquisition(page, tmp_path, backend=backend, mode="2d")
    gi = StitchGI(("q_ip", "q_oop"), incident_angle_deg=.3)
    page.load_form(replace(form, gi=gi, mode="2d", npt_rad_2d=24, npt_azim_2d=16))
    page.set_field("map_axis", "Qip-Qoop")
    suffix = " -10" if backend == "multigeometry" else " -10 17000.018"
    for i, delta in enumerate((14., 14.1, 14.2, 14.3)):
        if i:
            _append_spec(spec, f"{delta}{suffix}\n")
        _stitch_raw(form, 1, i, 3)
    owner = start_preview(page, qapp)
    _next(page, owner, qapp, lambda: owner.snapshot is not None)
    snapshot = owner.snapshot
    assert tuple(name for name, _ in snapshot.axes) == ("q_ip", "q_oop")
    assert page._last_map_state.image_axis == "qip_qoop"
    np.testing.assert_allclose(page._last_map_state.heavy.cake, snapshot.intensity, equal_nan=True)


def test_p6_replay_uses_fixed_targets_and_bounded_separate_work(qapp, make_page, tmp_path, monkeypatch):
    page = make_page("rsm")
    spec, images = _rsm_acquisition(page, tmp_path)
    page.set_field("static_hot", "100")
    _raw(images / f"{spec.stem}_scan1_0000.raw", 31, hot=(250, 251, 252, 253))
    owner = start_preview(page, qapp)
    _next(page, owner, qapp, lambda: _prefix(owner, ("1.1", 0)))
    initial = np.diff(np.asarray(owner.snapshot.bounds), axis=1).ravel()

    def grow(index, angle, hot):
        _append_spec(spec, f"{angle} 1 10 101\n")
        _raw(images / f"{spec.stem}_scan1_{index:04d}.raw", 30 + index, hot=hot)

    grow(1, 0., (250, 251, 252, 253))
    _next(page, owner, qapp, lambda: _prefix(owner, ("1.1", 0), ("1.1", 1)))
    for index in range(2, 8):
        hot = (42, 251 if index < 4 else 43, 252, 253)
        grow(index, 0., hot)
        _next(page, owner, qapp, lambda: _prefix(owner, *(("1.1", i) for i in range(index + 1))))
    entered, release = Event(), Event()
    grid = StreamingGridder.add_leased

    def hold(gridder, *args, **kwargs):
        result = grid(gridder, *args, **kwargs)
        if not entered.is_set():
            entered.set()
            assert release.wait(20), "the preview slice was never released"
        return result

    monkeypatch.setattr(StreamingGridder, "add_leased", hold)
    try:
        grow(8, 15., (42, 43, 44, 253))
        advance_preview(page, owner, qapp, drain=False)
        wait_until(qapp, entered.is_set)
        target = owner.progress.replay_target
        assert target is not None
        for extra in (9, 10):
            grow(extra, 15., (42, 43, 44, 253))
            advance_preview(page, owner, qapp, drain=False)
            assert owner.progress.replay_target == target
        assert owner.progress.pending_watermark > target
        assert "rebuild" in page.run_bar.readinessLabel.full_text().casefold()
    finally:
        release.set()

    _next(page, owner, qapp, lambda: _prefix(owner, *(("1.1", i) for i in range(11))))
    for index, angle in enumerate((18., 30., 50.), 11):
        grow(index, angle, (42, 43, 44, 253))
        _next(page, owner, qapp, lambda: _prefix(owner, *(("1.1", i) for i in range(index + 1))))
    counts = owner.progress
    replays = counts.replays
    assert counts.pending_watermark is None
    assert any(replay.growth and replay.mask for replay in replays)
    assert counts.physical_replays == len(replays)
    assert counts.replayed_points == sum(replay.points for replay in replays)
    assert counts.decodes == counts.forward_decodes + counts.replayed_points + counts.scout_decodes
    assert counts.resident_bytes <= counts.admitted_bytes
    assert counts.parser_opens <= counts.captures <= counts.observation_sweeps
    growth = [replay for replay in replays if replay.growth]
    final = np.diff(np.asarray(owner.snapshot.bounds), axis=1).ravel()
    assert len(growth) <= np.floor(np.log2(final / initial) + 1e-10).sum()
    for rebuild in growth:
        assert any(after >= 2 * before for before, after in zip(rebuild.span_before, rebuild.span_after, strict=True))
    mask_only = [replay.points for replay in replays if replay.mask and not replay.growth]
    assert len(mask_only) >= 2
    assert all(after >= 2 * before for before, after in zip(mask_only, mask_only[1:]))
    applied = owner.snapshot.member_masks["1.1"]
    np.testing.assert_array_equal(applied[0, :4], (False, False, False, True))
    np.testing.assert_allclose(owner.snapshot.intensity, _rsm_reference(page, owner.snapshot, applied),
                               rtol=1e-7, equal_nan=True)
