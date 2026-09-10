"""Focused headless Average Scan science and notebook-boundary oracle."""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import tifffile

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.reduction import (
    AverageContributor,
    AverageScanRecipe,
    Integration1DPlan,
    Integration2DPlan,
    GIMode,
    ReductionPlan,
)
from xrd_tools.reduction import average as module

def _science_of(plan):
    """Science payload for a prepared plan (the helper takes explicit inputs)."""
    return module._science_payload(
        plan.recipe, plan.numeric_metadata_keys, plan.invariant_metadata_keys,
    )


from xrd_tools.session.experiment_state import (
    CalibrationState, FactStatus, MaskState, PoniValues,
)
from xrd_tools.session.intent_store import IntentFreezeAccepted, RunIntentStore
from xrd_tools.session.run_configuration import RunIntent, ThresholdIntent
from xrd_tools.reduction.background import FrameBackgroundPlan
from xrd_tools.sources.selection import image_series_spec


def _series(tmp_path: Path, arrays=None) -> SourceSpec:
    arrays = arrays or (
        np.array([[1, 2], [3, 4]], dtype=np.uint16),
        np.array([[3, 4], [5, 6]], dtype=np.uint16),
        np.array([[5, 6], [7, 8]], dtype=np.uint16),
    )
    for index, value in enumerate(arrays, 1):
        tifffile.imwrite(tmp_path / f"scan_{index:04d}.tif", value)
    return image_series_spec(tmp_path / "scan_0001.tif", metadata_format=None)


def _frozen_configuration(source: SourceSpec, reduction: ReductionPlan):
    one = reduction.integration_1d or Integration1DPlan()
    two = reduction.integration_2d
    intent = RunIntent(
        source_spec=source,
        processing_mode="Int 2D" if two is not None else "Int 1D",
        output_mode="Overwrite",
        bai_1d_args={
            "npt": one.npt,
            "unit": one.unit,
            "method": one.method,
            "radial_range": one.radial_range,
            "azimuth_range": one.azimuth_range,
            "monitor": one.monitor_key,
            "error_model": one.error_model,
            "polarization_factor": one.polarization_factor,
            "chi_npt_rad": one.npt_rad,
            "chi_offset": one.azimuth_offset,
            **dict(one.extra),
        },
        bai_2d_args={} if two is None else {
            "npt_rad": two.npt_rad,
            "npt_azim": two.npt_azim,
            "unit": two.unit,
            "method": two.method,
            "radial_range": two.radial_range,
            "azimuth_range": two.azimuth_range,
            "chi_offset": two.azimuth_offset,
            "monitor": two.monitor_key,
            "error_model": two.error_model,
            "polarization_factor": two.polarization_factor,
            **dict(two.extra),
        },
        threshold=ThresholdIntent(
            apply_threshold=(
                reduction.threshold_min is not None
                or reduction.threshold_max is not None
            ),
            threshold_min=reduction.threshold_min,
            threshold_max=reduction.threshold_max,
            mask_saturation=reduction.mask_saturation,
        ),
        run_options={"series_average": True},
    )
    frozen = RunIntentStore(intent).freeze(expected_revision=0)
    assert type(frozen) is IntentFreezeAccepted
    return frozen.configuration


def _txt(path: str | Path, *, counters=(), motors=()) -> None:
    row = lambda items: ", ".join(f"{name} = {value}" for name, value in items)
    Path(path).with_suffix(".txt").write_text(
        f"# Counters\n{row(counters)}\n# Motors\n{row(motors)}\n"
        "User: p36, time: Mon Jan 15 10:30:00 2024  # Temp\n"
    )


def _stub_integrators(monkeypatch, observed, *, all_dummy_2d=False):
    from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
    from xrd_tools.reduction import core

    def one(image, _ai, *, npt, mask=None, normalization_factor=None, **_kwargs):
        observed.append(("1d", np.array(image, copy=True), None if mask is None else np.array(mask, copy=True), normalization_factor))
        usable = np.asarray(image)[~np.asarray(mask)] if mask is not None else np.asarray(image).ravel()
        value = float(np.mean(usable)) / (1.0 if normalization_factor is None else float(normalization_factor))
        return IntegrationResult1D(np.arange(npt, dtype=float), np.full(npt, value), None, "q_A^-1")

    def two(image, _ai, *, npt_rad, npt_azim, mask=None, normalization_factor=None, **_kwargs):
        observed.append(("2d", np.array(image, copy=True), None if mask is None else np.array(mask, copy=True), normalization_factor))
        data = np.full((npt_rad, npt_azim), np.nan if all_dummy_2d else float(np.nanmean(image)))
        return IntegrationResult2D(np.arange(npt_rad, dtype=float), np.arange(npt_azim, dtype=float), data, None, "q_A^-1", "chi_deg")

    monkeypatch.setattr(core, "integrate_1d", one)
    monkeypatch.setattr(core, "integrate_2d", two)


def _run_average_scan(recipe, **kwargs):
    runner = module.AverageScanRunner(recipe)
    result = runner.start(**kwargs)
    assert type(result) is module.AverageScanResult
    assert runner.close() is result
    return result


@pytest.mark.parametrize("thresholds", (False, True))
def test_real_average_counts_follow_each_frame_saturation(tmp_path, thresholds):
    frames = [np.full((4, 4), 20, dtype=np.uint32) for _ in range(3)]
    for index, frame in enumerate(frames):
        frame[index, 0] = np.iinfo(np.uint32).max
        frame[index, 1] = 2000
    source = _series(tmp_path, frames)
    static = np.zeros((4, 4), dtype=bool)
    static[3, 3] = True
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, static)
    calibration = CalibrationState(
        PoniValues(0.2, 0.001, 0.001, 0.0, 0.0, 0.0, 1e-10), "Detector",
        {"pixel1": 1e-4, "pixel2": 1e-4, "max_shape": [4, 4], "orientation": 3},
        status=FactStatus.PRESENT,
        mask=MaskState(str(mask_path), hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                       np.dtype(bool).str, static.shape, FactStatus.PRESENT),
    )
    result = _run_average_scan(AverageScanRecipe(
        source, tmp_path / "scan.nexus", ReductionPlan(
            integration_1d=Integration1DPlan(npt=4, method="numpy"),
            mask_saturation=True, threshold_max=100 if thresholds else None,
        ), calibration=calibration,
    ))
    assert result.disposition == "COMMITTED", result
    expected = np.full((4, 4), 3, dtype=np.uint32)
    expected[:3, 0] = 2
    if thresholds:
        expected[:3, 1] = 2
    expected[3, 3] = 0
    np.testing.assert_array_equal(module.get_average_finite_counts(result.target).values,
                                  expected)


def _prepare_average_scan(recipe):
    runner = module.AverageScanRunner(recipe)
    runner._execute_graph = lambda _graph: runner.plan
    value = runner.start()
    if type(value) is module.AverageScanResult:
        raise ValueError(value.diagnostic or value.diagnostic_code)
    if type(value) is module.AverageScanPending:
        raise RuntimeError(value.diagnostic)
    assert type(value) is module.AverageScanPlan
    runner.close()
    return value


def _public_run(tmp_path, monkeypatch, *, source=None, target_name="average.nxs",
                reduction=None, **recipe_kwargs):
    observed = []
    _stub_integrators(monkeypatch, observed)
    recipe = AverageScanRecipe(
        source or _series(tmp_path), tmp_path / target_name,
        reduction or ReductionPlan(integration_1d=Integration1DPlan(npt=4)),
        **recipe_kwargs,
    )
    result = _run_average_scan(recipe)
    return recipe, result, observed


def test_streaming_average_matches_reference_counts_metadata_and_order(tmp_path, monkeypatch) -> None:
    frames = (
        np.array([[65535, 12], [3, 14]], dtype=np.uint16),
        np.array([[2, 16], [5, 51]], dtype=np.uint16),
        np.array([[4, 20], [7, 18]], dtype=np.uint16),
    )
    source = _series(tmp_path, frames)
    admitted = []
    for index, path in enumerate(source.options["files"], 1):
        _txt(
            path, counters=(("i0", float(index)), ("temp", 4.0)),
            motors=(("theta", 99.0),),
        )
        admitted.append((str(path), "theta", 0.1 * index))
    source = SourceSpec(source.uri, source.kind, options={
        **dict(source.options), "metadata_format": "txt",
        "admitted_motor_values": tuple(admitted),
    })
    read_order, background_order, detector_masks, metadata_calls = [], [], [], []
    from xrd_tools.sources import execution_graph
    from xrd_tools.sources.image import TiffSeriesSource
    real_metadata = execution_graph._AverageSourceReadWindow.complete_metadata_for
    real_tiff_metadata = TiffSeriesSource.metadata_for
    real_detector_mask = module.detector_value_mask
    monkeypatch.setattr(
        execution_graph._AverageSourceReadWindow,
        "complete_metadata_for",
        lambda window, index: read_order.append(index) or real_metadata(window, index),
    )
    monkeypatch.setattr(
        TiffSeriesSource, "metadata_for",
        lambda owner, index, **kwargs: metadata_calls.append((index, kwargs))
        or real_tiff_metadata(owner, index, **kwargs),
    )
    monkeypatch.setattr(
        module, "detector_value_mask",
        lambda ceiling, first, **kwargs: detector_masks.append(np.array(first, copy=True))
        or real_detector_mask(ceiling, first, **kwargs),
    )
    monkeypatch.setattr(
        module, "resolve_frame_background",
        lambda *_a, **_k: background_order.append(len(background_order))
        or pytest.fail("inactive Average invoked the background resolver"),
    )
    reduction = ReductionPlan(
        integration_1d=Integration1DPlan(npt=4, monitor_key="I0"),
        threshold_max=50.0, mask_saturation=True,
    )
    mask_path = tmp_path / "static-mask.npy"
    static_mask = np.array([[False, False], [True, False]])
    np.save(mask_path, static_mask)
    mask_digest = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    calibration = CalibrationState(mask=MaskState(
        str(mask_path), mask_digest, np.dtype(bool).str, static_mask.shape,
        FactStatus.PRESENT,
    ))
    recipe, result, observed = _public_run(
        tmp_path, monkeypatch, source=source, reduction=reduction,
        calibration=calibration,
        numeric_metadata_keys=("I0", "theta"), invariant_metadata_keys=("temp",),
    )
    assert result.disposition == "COMMITTED" and result.committed_labels == (1,)
    assert result.metadata_denominators == (("I0", 3), ("theta", 3))
    finite = module.get_average_finite_counts(result.target)
    np.testing.assert_array_equal(finite.values, [[2, 3], [0, 2]])
    assert finite.evidence.contributor_extent == 3
    assert len(observed) == 1 and observed[0][0] == "1d"
    mean, mask, normalization = observed[0][1:]
    np.testing.assert_allclose(mean, [[3.0, 16.0], [np.nan, 16.0]], equal_nan=True)
    np.testing.assert_array_equal(mask, finite.values == 0)
    assert normalization == pytest.approx(2.0)
    assert read_order == [0, 0, 1, 2]
    assert metadata_calls == [(1, {"max_input_bytes": 65536}),
                              (1, {"max_input_bytes": 65536}),
                              (2, {"max_input_bytes": 65536}),
                              (3, {"max_input_bytes": 65536})]
    assert background_order == []
    assert len(detector_masks) == len(frames)
    for observed_mask, frame in zip(detector_masks, frames):
        np.testing.assert_array_equal(observed_mask, frame)
    assert recipe.batch_mode is False and result.finite_counts == finite.evidence


@pytest.mark.parametrize(
    ("dtype", "rows", "limits", "mask_saturation"),
    (
        pytest.param(
            "<u2",
            (
                ((1, 65535, 7), (9, 11, 13)),
                ((5, 6, 8), (10, 12, 14)),
                ((4, 7, 9), (11, 13, 15)),
            ),
            (5.0, None), True, id="u2",
        ),
        pytest.param(
            "<u4",
            (
                ((0, 4294967294, 3), (4, 5, 6)),
                ((1, 8, 4), (5, 6, 7)),
                ((2, 9, 5), (6, 7, 8)),
            ),
            (0.0, 100.0), False, id="u4-redundant-lower-bound",
        ),
        pytest.param(
            "<u8",
            (
                ((2**53 + 1, 2**53 + 3, 2**53 + 5),
                 (2**53 + 7, 2**53 + 9, 2**53 + 11)),
                ((2**53 + 13, 2**53 + 15, 2**53 + 17),
                 (2**53 + 19, 2**53 + 21, 2**53 + 23)),
                ((2**53 + 25, 2**53 + 27, 2**53 + 29),
                 (2**53 + 31, 2**53 + 33, 2**53 + 35)),
            ),
            (None, None), False, id="u8-above-binary64-integer-precision",
        ),
        pytest.param(
            "<i8",
            (
                ((-10, -5, 0), (5, 10, 100)),
                ((-9, -4, 1), (6, 11, 90)),
                ((-8, -3, 2), (7, 12, 80)),
            ),
            (-5.0, 50.0), False, id="signed",
        ),
        pytest.param(
            "<f8",
            (
                ((np.nan, -np.inf, np.inf), (-4.0, 0.0, 4.0)),
                ((-5.0, -3.0, 3.0), (-2.0, 1.0, 5.0)),
                ((-6.0, -1.0, 2.0), (-3.0, np.nan, 6.0)),
            ),
            (-5.0, 5.0), False, id="float-nonfinite",
        ),
    ),
)
def test_average_where_out_accumulator_is_bit_exact(
    tmp_path, monkeypatch, dtype, rows, limits, mask_saturation,
) -> None:
    frames = tuple(np.asarray(row, dtype=dtype) for row in rows)
    source = _series(tmp_path, frames)
    static_mask = np.array(
        ((False, False, True), (False, True, False)), dtype=bool,
    )
    mask_path = tmp_path / "static-mask.npy"
    np.save(mask_path, static_mask)
    calibration = CalibrationState(mask=MaskState(
        str(mask_path), hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        np.dtype(bool).str, static_mask.shape, FactStatus.PRESENT,
    ))
    minimum, maximum = limits
    reduction = ReductionPlan(
        integration_1d=Integration1DPlan(npt=3),
        threshold_min=minimum, threshold_max=maximum,
        mask_saturation=mask_saturation,
    )

    _recipe, result, observed = _public_run(
        tmp_path, monkeypatch, source=source, reduction=reduction,
        calibration=calibration, numeric_metadata_keys=(),
    )

    assert result.disposition == "COMMITTED" and len(observed) == 1
    actual = observed[0][1]
    actual_counts = module.get_average_finite_counts(result.target).values
    expected = np.zeros(frames[0].shape, dtype=np.float64)
    expected_counts = np.zeros(frames[0].shape, dtype="<u4")
    for native in frames:
        detector_mask = module.detector_value_mask(None, native, enabled=mask_saturation)
        conditioned = native.astype(np.float64, copy=True)
        invalid = np.zeros(native.shape, dtype=bool)
        if minimum is not None:
            np.logical_or(invalid, native < minimum, out=invalid)
        if maximum is not None:
            np.logical_or(invalid, native > maximum, out=invalid)
        np.logical_or(invalid, static_mask, out=invalid)
        if detector_mask is not None:
            np.logical_or(invalid, detector_mask, out=invalid)
        np.logical_or(invalid, ~np.isfinite(conditioned), out=invalid)
        valid = ~invalid
        expected[valid] += conditioned[valid]
        expected_counts[valid] += np.uint32(1)
    zero = expected_counts == 0
    np.divide(expected, expected_counts, out=expected, where=~zero)
    expected[zero] = np.nan

    np.testing.assert_array_equal(actual_counts, expected_counts)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(
        actual[finite].view(np.uint64), expected[finite].view(np.uint64),
    )


def test_average_persists_anisotropic_detector_axes_without_transposition(
    tmp_path, monkeypatch,
) -> None:
    calibration = CalibrationState(
        PoniValues(0.2, 0.01, 0.02, 0.0, 0.0, 0.0, 1.0e-10),
        "Detector",
        {"pixel1": 1.0e-4, "pixel2": 2.0e-4,
         "max_shape": [2, 2], "orientation": 3},
        status=FactStatus.PRESENT,
    )
    _recipe_value, result, _observed = _public_run(
        tmp_path, monkeypatch, calibration=calibration,
    )
    assert result.disposition == "COMMITTED"
    with h5py.File(result.target, "r") as handle:
        detector = handle["entry/instrument/detector"]
        assert float(detector["x_pixel_size"][()]) == pytest.approx(2.0e-4)
        assert float(detector["y_pixel_size"][()]) == pytest.approx(1.0e-4)


def test_all_nonfinite_and_invariant_mismatch_refuse_without_output(tmp_path, monkeypatch) -> None:
    dead = tmp_path / "dead"; dead.mkdir()
    source = _series(dead, (np.full((2, 2), np.nan),))
    recipe, result, observed = _public_run(dead, monkeypatch, source=source)
    assert result.disposition == "REFUSED"
    assert result.diagnostic_code == "AVERAGE_ALL_PIXELS_INVALID"
    assert observed == [] and not Path(recipe.target).exists()

    drift = tmp_path / "drift"; drift.mkdir()
    source = _series(drift, (np.ones((2, 2)), np.ones((2, 2))))
    for index, path in enumerate(source.options["files"], 1):
        _txt(path, counters=(("T", float(index)),))
    source = SourceSpec(source.uri, source.kind, options={
        **dict(source.options), "metadata_format": "txt",
    })
    recipe, result, observed = _public_run(
        drift, monkeypatch, source=source, numeric_metadata_keys=(),
        invariant_metadata_keys=("T",),
    )
    assert result.disposition == "REFUSED"
    assert result.diagnostic_code == "AVERAGE_INVARIANT_METADATA_CHANGED"
    assert observed == [] and not Path(recipe.target).exists()


def test_append_live_xye_refuse_before_source_or_target_effects(tmp_path, monkeypatch) -> None:
    effects = []
    source = _series(tmp_path)
    existing = tmp_path / "prior.nxs"
    with h5py.File(existing, "w") as handle: handle.create_dataset("prior", data=[1])
    prior = existing.read_bytes()
    from xrd_tools.sources import execution_graph
    monkeypatch.setattr(module, "qualify_source_execution_graph", lambda *_a, **_k: effects.append("source"))
    monkeypatch.setattr(execution_graph, "read_detector_image_layout", lambda *_a, **_k: effects.append("header"))
    monkeypatch.setattr(module, "resolve_session_policy", lambda *_a, **_k: effects.append("allocation"))
    monkeypatch.setattr(
        module.NexusSink, "for_finite_document",
        lambda *_a, **_k: effects.append("h23"),
    )
    canonical_targets = set()
    for target in (tmp_path / "absent.nxs", existing):
        for updates, code in (
            ({"live_mode": True, "output_mode": "Append", "save_xye": True},
             "AVERAGE_LIVE_TERMINALITY_UNSUPPORTED"),
            ({"output_mode": "Append", "save_xye": True}, "AVERAGE_APPEND_UNSUPPORTED"),
            ({"save_xye": True}, "AVERAGE_NEXUS_REQUIRED"),
        ):
            recipe = AverageScanRecipe(source, target, ReductionPlan(), **updates)
            canonical_targets.add(Path(recipe.target))
            result = _run_average_scan(recipe)
            assert (result.disposition, result.diagnostic_code) == ("REFUSED", code)
    assert effects == [] and not any(path.exists() for path in canonical_targets)
    assert not (tmp_path / "absent.nxs").exists()
    assert existing.read_bytes() == prior

    monkeypatch.undo()
    batch_root = tmp_path / "batch"; batch_root.mkdir()
    observed = []; counts = {"qualify": 0, "open": 0, "sink": 0}
    _stub_integrators(monkeypatch, observed)
    real_qualify = module.qualify_source_execution_graph
    real_open = module.open_source_execution_graph
    real_sink_factory = module.NexusSink.for_finite_document
    monkeypatch.setattr(module, "qualify_source_execution_graph", lambda *a, **k:
        counts.__setitem__("qualify", counts["qualify"] + 1) or real_qualify(*a, **k))
    monkeypatch.setattr(module, "open_source_execution_graph", lambda *a, **k:
        counts.__setitem__("open", counts["open"] + 1) or real_open(*a, **k))
    def finite_sink(*a, **k):
        counts["sink"] += 1
        return real_sink_factory(*a, **k)
    monkeypatch.setattr(module.NexusSink, "for_finite_document", finite_sink)
    source = _series(batch_root, (np.ones((2, 2), dtype="u2"),))
    recipes = tuple(AverageScanRecipe(
        source, batch_root / f"average-{mode}.nxs", ReductionPlan(), batch_mode=mode,
    ) for mode in (False, True))
    plans = []
    accepted = []
    for recipe in recipes:
        runner = module.AverageScanRunner(recipe)
        result = runner.start()
        assert type(result) is module.AverageScanResult
        plans.append(runner.plan)
        accepted.append(result)
        assert runner.close() is result
    plans = tuple(plans)
    assert _science_of(plans[0]) == _science_of(plans[1])
    assert plans[0].science_identity == plans[1].science_identity
    operation_payloads = [module._operation_payload(plan) for plan in plans]
    assert [payload["batch_mode"] for payload in operation_payloads] == [False, True]
    normalized = []
    for payload in operation_payloads:
        payload = dict(payload)
        # The old single "target" key split into the user-requested ANCHOR and
        # the immutable successor actually written.  These two plans differ only
        # in target filename, so BOTH carry that difference and both must be
        # normalised here.  version_identity is deliberately NOT normalised: it
        # is science-derived, not path-derived, and the assertions below pin
        # that.
        payload["target_anchor"] = payload["output_artifact"] = "<target>"
        payload["batch_mode"] = False
        normalized.append(payload)
    assert normalized[0] == normalized[1]
    # Two plans differing ONLY in target filename share a version identity...
    assert (operation_payloads[0]["version_identity"]
            == operation_payloads[1]["version_identity"])
    # ...and each artifact is that anchor's family plus the STABLE `_average`
    # slot, never the anchor itself: an Average never writes the path the user
    # pointed at.  The version stays in provenance and out of the public name.
    for payload in operation_payloads:
        assert payload["output_artifact"] != payload["target_anchor"]
        artifact = Path(payload["output_artifact"])
        anchor = Path(payload["target_anchor"])
        assert artifact.name == f"{anchor.stem}_average.nexus"
        assert artifact.parent == anchor.parent
        # No version leak: the identity is 64 hex, and none of it may surface.
        assert payload["version_identity"][:32] not in artifact.name
    assert plans[0].operation_identity != plans[1].operation_identity
    assert all(result.disposition == "COMMITTED" and result.contributor_extent == 1
               for result in accepted)
    assert all(result.logical_labels == result.committed_labels == (1,)
               for result in accepted)
    assert len(observed) == 2 and counts == {"qualify": 2, "open": 4, "sink": 2}
    # result.target is the ARTIFACT WRITTEN, never the anchor the recipe pointed
    # at.  Equality was the old in-place contract; pin the DERIVATION instead,
    # order-independently.
    anchors = {Path(recipe.target) for recipe in recipes}
    artifacts = {Path(result.target) for result in accepted}
    assert artifacts.isdisjoint(anchors)
    assert len(artifacts) == len(anchors)
    by_stem = {anchor.stem: anchor for anchor in anchors}
    for result in accepted:
        path = Path(result.target)
        stem, marker, slot = path.stem.rpartition("_average")
        assert (marker, slot) == ("_average", "")
        assert stem in by_stem and path.parent == by_stem[stem].parent
        # The version identity stays in provenance and out of the public name.
        assert result.version_identity[:32] not in path.name
    assert all(Path(result.target).suffix == ".nexus" for result in accepted)
    from xrd_tools.core.provenance import read_provenance
    from xrd_tools.io import get_1d, get_metadata
    products = []
    for result in accepted:
        products.append((
            get_1d(result.target, frame=1), module.get_average_finite_counts(result.target),
            get_metadata(result.target)["scan_data"],
            tuple(module.iter_average_contributors(result.target)),
            read_provenance(result.target)["config"]["average_scan_v1"],
        ))
    for field in ("q", "intensity", "sigma", "frames"):
        np.testing.assert_array_equal(getattr(products[0][0], field),
                                      getattr(products[1][0], field))
    np.testing.assert_array_equal(products[0][1].values, products[1][1].values)
    assert products[0][1].evidence == products[1][1].evidence
    assert set(products[0][2]) == set(products[1][2])
    for name in products[0][2]:
        np.testing.assert_array_equal(products[0][2][name], products[1][2][name])
    assert products[0][3] == products[1][3]
    provenance = [dict(product[4]) for product in products]
    operation_identities = tuple(value.pop("operation_identity") for value in provenance)
    assert provenance[0] == provenance[1]
    assert operation_identities == tuple(result.operation_identity for result in accepted)
    assert operation_identities[0] != operation_identities[1]
    assert accepted[0].science_identity == accepted[1].science_identity
    assert accepted[0].metadata_denominators == accepted[1].metadata_denominators


def test_average_background_none_is_canonical_and_active_refuses_pre_effect(
    tmp_path, monkeypatch,
) -> None:
    source = _series(tmp_path, (np.ones((2, 2), dtype="u2"),))
    target = tmp_path / "average.nxs"
    reduction = ReductionPlan(integration_1d=Integration1DPlan(npt=3))
    omitted = AverageScanRecipe(source, target, reduction)
    explicit = AverageScanRecipe(
        source, target, reduction, background=FrameBackgroundPlan(),
    )
    assert omitted.target == str((tmp_path / "average.nexus").resolve())
    reconstructed = replace(omitted, target=tmp_path / "reconstructed.nxs")
    assert reconstructed.target == str((tmp_path / "reconstructed.nexus").resolve())
    assert replace(reconstructed) == reconstructed
    assert omitted == explicit
    assert omitted.background is explicit.background is None
    assert module._recipe_payload(omitted) == module._recipe_payload(explicit)
    assert module._recipe_payload(explicit)["background"] is None
    plans = tuple(_prepare_average_scan(recipe)
                  for recipe in (omitted, explicit))
    assert plans[0].allocation == plans[1].allocation
    assert plans[0].science_identity == plans[1].science_identity
    assert plans[0].operation_identity == plans[1].operation_identity
    assert _science_of(plans[0])["background"] is None

    active = AverageScanRecipe(
        source, target, reduction,
        background=FrameBackgroundPlan(
            mode="Single BG File",
            locator=str(source.options["files"][0]),
        ),
    )
    effects = []

    def forbidden(name):
        return lambda *_args, **_kwargs: (
            effects.append(name), pytest.fail(f"active background reached {name}")
        )[1]

    with monkeypatch.context() as patch:
        for name in (
            "_source_from_recipe", "qualify_source_execution_graph",
            "open_source_execution_graph", "_average_allocation",
            "capture_target_snapshot", "_background_fact",
            "resolve_frame_background",
        ):
            patch.setattr(module, name, forbidden(name))
        with pytest.raises(
            ValueError,
            match="^AVERAGE_BACKGROUND_AGGREGATE_PROVENANCE_UNSUPPORTED$",
        ):
            _prepare_average_scan(active)
        refused = _run_average_scan(active)
    assert (refused.disposition, refused.diagnostic_code) == (
        "REFUSED", "AVERAGE_BACKGROUND_AGGREGATE_PROVENANCE_UNSUPPORTED",
    )
    assert effects == [] and not target.exists()

    run_configuration = []
    integrations = []
    real_sink_factory = module.NexusSink.for_finite_document

    def sink(*args, **kwargs):
        run_configuration.append(dict(kwargs["run_configuration_provenance"]))
        return real_sink_factory(*args, **kwargs)

    _stub_integrators(monkeypatch, integrations)
    monkeypatch.setattr(module.NexusSink, "for_finite_document", sink)
    committed = _run_average_scan(explicit)
    assert committed.disposition == "COMMITTED" and len(integrations) == 1
    assert len(run_configuration) == 1
    assert "background" not in run_configuration[0]
    from xrd_tools.core.provenance import read_provenance
    persisted = read_provenance(committed.target)["config"]["average_scan_v1"]
    assert persisted["background"] is None


def test_post_admission_source_drift_rolls_back_prior_target(tmp_path, monkeypatch) -> None:
    source = _series(tmp_path)
    member = Path(source.options["files"][0])
    target = tmp_path / "average.nxs"
    with h5py.File(target, "w") as handle:
        handle.create_dataset("prior", data=np.arange(7, dtype="<i4"))
    before = target.read_bytes()
    observed = []
    _stub_integrators(monkeypatch, observed)
    from xrd_tools.reduction import core
    integrate = core.integrate_1d

    def mutate_after_reduction(*args, **kwargs):
        value = integrate(*args, **kwargs)
        member.write_bytes(member.read_bytes() + b"post-admission-drift")
        return value

    monkeypatch.setattr(core, "integrate_1d", mutate_after_reduction)
    result = _run_average_scan(AverageScanRecipe(
        source, target, ReductionPlan(integration_1d=Integration1DPlan(npt=4)),
    ))
    assert result.disposition == "REFUSED"
    assert result.diagnostic_code == "AVERAGE_SOURCE_DRIFT"
    assert len(observed) == 1
    assert target.read_bytes() == before
    with pytest.raises((KeyError, ValueError)):
        module.get_average_finite_counts(target)


def test_pixel_and_metadata_sum_overflow_refuse_without_output(tmp_path, monkeypatch) -> None:
    maximum = np.finfo("f8").max
    for name, frames, rows, code in (
        ("pixel", (np.full((2, 2), maximum), np.full((2, 2), maximum)), ({}, {}),
         "AVERAGE_PIXEL_SUM_OVERFLOW"),
        ("metadata", (np.ones((2, 2)), np.ones((2, 2))),
         ({"I0": maximum}, {"I0": maximum}), "AVERAGE_METADATA_SUM_OVERFLOW"),
    ):
        root = tmp_path / name; root.mkdir()
        source = _series(root, frames)
        if name == "metadata":
            for row, path in zip(rows, source.options["files"], strict=True):
                _txt(path, counters=(("I0", row["I0"]),))
            source = SourceSpec(source.uri, source.kind, options={
                **dict(source.options), "metadata_format": "txt",
            })
        observed = []
        _stub_integrators(monkeypatch, observed)
        recipe = AverageScanRecipe(
            source, root / "average.nxs", ReductionPlan(),
            numeric_metadata_keys=("I0",) if name == "metadata" else (),
        )
        result = _run_average_scan(recipe)
        assert (result.disposition, result.diagnostic_code) == ("REFUSED", code)
        assert observed == [] and not Path(recipe.target).exists()


def test_partial_owner_block_grant_refuses_before_pixel_read(tmp_path, monkeypatch) -> None:
    from xrd_tools.sources import execution_graph
    source = _series(tmp_path); recipe = AverageScanRecipe(
        source, tmp_path / "average.nxs", ReductionPlan(), resource_requests={"workers": 1},
    )
    reads, targets = [], []
    original_read = execution_graph._AverageSourceReadWindow.read_native
    monkeypatch.setattr(
        execution_graph._AverageSourceReadWindow, "read_native",
        lambda window, index: reads.append(index) or original_read(window, index),
    )
    valid = module._average_allocation(
        recipe, (2, 2), np.dtype("<u2"), direct_eiger_eligible=False,
    ); counts = dict(valid.counts)
    key = next(iter(valid.categories)); categories = dict(valid.categories)
    mutants = (
        replace(valid, requirements=replace(
            valid.requirements, height=valid.requirements.height + 1,
        )),
        replace(valid, categories={**categories, key: categories[key] + 1}),
        replace(valid, counts={**counts, "owner_block_bytes": counts["owner_block_bytes"] - valid.requirements.native_frame_bytes}),
    )
    from types import SimpleNamespace
    for bad in mutants:
        with monkeypatch.context() as patch:
            patch.setattr(module, "resolve_session_policy", lambda *_a, bad=bad, **_k: SimpleNamespace(allocation=bad))
            patch.setattr(module, "capture_target_snapshot", lambda *_a: targets.append(1) or pytest.fail("target reached"))
            with pytest.raises(ValueError, match="AVERAGE_OWNER_BLOCK_GRANT_INCOMPLETE"):
                _prepare_average_scan(recipe)
    assert reads == [] and targets == [] and not Path(recipe.target).exists()


def test_recipe_deep_snapshot_survives_all_caller_mutation(tmp_path) -> None:
    from dataclasses import replace
    source = _series(tmp_path)
    source_options = dict(source.options)
    source_options["files"] = list(source_options["files"])
    source_options["admitted_motor_values"] = [
        [str(path), "theta", float(index)]
        for index, path in enumerate(source_options["files"], 1)
    ]
    source = SourceSpec(source.uri, source.kind, options=source_options)
    extra = {"nested": [1, {"value": 2.0}], "enabled_modes_1d": ["q"]}
    reduction_extra = {"x_range": [0.1, 0.9]}
    reduction = ReductionPlan(
        integration_1d=Integration1DPlan(npt=5, extra=extra),
        extra=reduction_extra,
    )
    requests, env = {"workers": 1}, {"OMP_NUM_THREADS": "1"}
    numeric, invariant = ["I0"], ["temperature"]
    detector_config = {
        "pixel1": 1.0e-4, "pixel2": 2.0e-4,
        "max_shape": [2, 2], "orientation": 3,
        "nested": {"labels": ["fast", True, None]},
    }
    calibration = CalibrationState(
        PoniValues(0.2, 0.01, 0.02, 0.0, 0.0, 0.0, 1.0e-10),
        "Detector", detector_config, "p" * 64, "c" * 64,
        str(tmp_path / "calibration.poni"),
        MaskState(str(tmp_path / "mask.npy"), "m" * 64, "|b1", (2, 2), FactStatus.PRESENT),
        FactStatus.PRESENT,
    )
    background = FrameBackgroundPlan(mode="Single BG File", locator=str(source_options["files"][0]))
    recipe = AverageScanRecipe(
        source, tmp_path / "average.nxs", reduction,
        calibration=calibration, background=background,
        numeric_metadata_keys=tuple(numeric), invariant_metadata_keys=tuple(invariant),
        resource_requests=requests, resource_env=env,
    )
    before = module._recipe_payload(recipe)
    assert tuple(before["calibration"]["detector_config"]["nested"]["labels"]) == (
        "fast", True, None,
    )
    variant_calibration = replace(
        recipe.calibration,
        detector_config={
            **detector_config,
            "nested": {"labels": ["slow", True, None]},
        },
    )
    variant = replace(recipe, calibration=variant_calibration)
    assert module._recipe_payload(variant) != before
    prepared = _prepare_average_scan(replace(recipe, background=None))
    variant_prepared = _prepare_average_scan(
        replace(variant, background=None),
    )
    assert tuple(_science_of(prepared)["calibration"]
                 ["detector_config"]["nested"]["labels"]) == ("fast", True, None)
    assert prepared.science_identity != variant_prepared.science_identity
    source_options["files"].append(str(tmp_path / "late.tif"))
    source_options["admitted_motor_values"][0][2] = 99.0
    extra["nested"][1]["value"] = 99.0
    extra["enabled_modes_1d"].append("chi")
    reduction_extra["x_range"][0] = -1.0
    reduction.integration_1d.npt = 9
    requests["workers"] = 8; env["OMP_NUM_THREADS"] = "8"
    numeric.append("changed"); invariant.clear()
    detector_config["max_shape"][0] = 99
    detector_config["nested"]["labels"].append("late")
    assert module._recipe_payload(recipe) == before
    assert recipe.source is not source and module._thaw_reduction(recipe) is not reduction
    assert recipe.calibration == calibration and recipe.calibration is not calibration
    assert recipe.calibration.detector_config["max_shape"] == (2, 2)
    assert recipe.calibration.detector_config["nested"]["labels"] == ("fast", True, None)
    assert recipe.background == background and recipe.background is not background
    assert not any(isinstance(getattr(recipe, item.name), (dict, list, np.ndarray))
                   for item in fields(recipe))
    for malformed in (
        {"bad": np.array([1])}, {"bad": float("nan")}, {"bad": {1: "value"}},
    ):
        with pytest.raises((TypeError, ValueError)):
            AverageScanRecipe(_series(tmp_path), tmp_path / "bad.nxs",
                              ReductionPlan(extra=malformed))


def test_recursive_json_extras_preserve_gi_ranges_scout_indices_and_enabled_modes(tmp_path) -> None:
    reduction_extra = {"nested": {"values": [1, True, None, {"z": "last", "a": "first"}]}}
    reduction = ReductionPlan(
        integration_1d=Integration1DPlan(extra={"x_range": [1, 2], "enabled_modes_1d": ["q"]}),
        integration_2d=Integration2DPlan(extra={"y_range": [3, 4], "gi_freeze_scout_indices": [0, 2]}),
        gi=GIMode(incident_angle=0.2, tilt_angle=0.1, sample_orientation=2,
                  mode_1d="q_ip", mode_2d="q_chi", npt_oop=7),
        extra=reduction_extra,
    )
    recipe = AverageScanRecipe(_series(tmp_path), tmp_path / "average.nxs", reduction)
    thawed = module._thaw_reduction(recipe)
    assert thawed.integration_1d.extra == {"enabled_modes_1d": ("q",), "x_range": (1, 2)}
    assert thawed.integration_2d.extra == {"gi_freeze_scout_indices": (0, 2), "y_range": (3, 4)}
    assert thawed.extra == {"nested": {"values": (1, True, None, {"a": "first", "z": "last"})}}
    before = module._recipe_payload(recipe)
    reduction_extra["nested"]["values"][3]["a"] = "changed"
    assert module._recipe_payload(recipe) == before
    twin = AverageScanRecipe(
        _series(tmp_path), tmp_path / "average.nxs", ReductionPlan(
            integration_1d=Integration1DPlan(extra={"enabled_modes_1d": ("q",), "x_range": (1, 2)}),
            integration_2d=Integration2DPlan(extra={"gi_freeze_scout_indices": (0, 2), "y_range": (3, 4)}),
            gi=GIMode(incident_angle=0.2, tilt_angle=0.1, sample_orientation=2,
                      mode_1d="q_ip", mode_2d="q_chi", npt_oop=7),
            extra={"nested": {"values": (1, True, None, {"a": "first", "z": "last"})}},
        ),
    )
    assert module._recipe_payload(twin) == before
    cycle = []; cycle.append(cycle)
    malformed = (
        {"cycle": cycle}, {"nonfinite": np.inf}, {"array": np.ones(1)},
        {"deep": [[[[[[[[[1]]]]]]]]]}, {"wide": list(range(4097))},
        {"large": "x" * (65536 + 1)},
    )
    for extra in malformed:
        with pytest.raises((TypeError, ValueError)):
            AverageScanRecipe(_series(tmp_path), tmp_path / "bad.nxs",
                              ReductionPlan(extra=extra))


def test_configured_metadata_keys_use_exact_first_casefold_and_refuse_ambiguity() -> None:
    assert module._metadata_value({"I0": 2.0, "i0": 2.0}, "I0") == 2.0
    assert module._metadata_value({"i0": 2.0}, "I0") == 2.0
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        module._metadata_value({"K": 2.0, "K": 3.0}, "k")


def test_configured_metadata_domains_emit_one_exact_downstream_spelling() -> None:
    rows = (
        module._project_metadata({"i0": 3.0, "other": 5.0}, ("I0",), ()),
        module._project_metadata({"I0": 4.0, "i0": 4.0}, ("I0",), ()),
    )
    assert rows == ({"I0": 3.0}, {"I0": 4.0})
    assert all(set(row) == {"I0"} for row in rows)


def test_identical_configured_role_spellings_coalesce_distinct_spellings_refuse() -> None:
    assert module._configured_key_union(("I0",), ("I0",), ()) == ("I0",)
    with pytest.raises(ValueError, match="DOMAIN"):
        module._configured_key_union(("I0",), ("i0",), ())
    with pytest.raises(ValueError, match="DOMAIN"):
        module._configured_key_union(("theta",), (), ("Theta",))


def test_preparation_metadata_row_drift_refuses_before_key_identity_derivation(tmp_path, monkeypatch) -> None:
    source = _series(tmp_path)
    sidecar = tmp_path / "scan_0001.txt"
    _txt(sidecar, counters=(("i0", 1.0),))
    source = SourceSpec(source.uri, source.kind, options={**dict(source.options), "metadata_format": "txt"})
    from xrd_tools.sources import execution_graph
    original = execution_graph._AverageSourceReadWindow.complete_metadata_for
    def row(window, index):
        value = original(window, index)
        _txt(sidecar, counters=(("i0", 2.0),))
        return value

    with monkeypatch.context() as patch:
        patch.setattr(execution_graph._AverageSourceReadWindow,
                      "complete_metadata_for", row)
        patch.setattr(module, "source_graph_digest", lambda *_a: pytest.fail("identity derived after metadata drift"))
        recipe = AverageScanRecipe(
            source, tmp_path / "average.nxs", ReductionPlan(),
            numeric_metadata_keys=("i0",),
        )
        with pytest.raises(ValueError, match="AVERAGE_SOURCE_DRIFT"):
            _prepare_average_scan(recipe)
    assert not Path(recipe.target).exists()
    gi_root = tmp_path / "gi"; gi_root.mkdir(); gi_source = _series(gi_root)
    for path in gi_source.options["files"]: _txt(path, motors=(("theta", 0.2),))
    gi_source = SourceSpec(gi_source.uri, gi_source.kind, options={**dict(gi_source.options), "metadata_format": "txt"})
    calls = []; real_prepare_q = module.qualify_source_execution_graph
    real_rq = module.requalify_source_execution_graph
    real_state_sweep = module.validate_source_state_sweep
    def prepare_q(*args, **kwargs): calls.append(("prepare", kwargs.get("selected_motor"))); return real_prepare_q(*args, **kwargs)
    def rq(*args, **kwargs): calls.append(("requalify", kwargs.get("selected_motor"))); return real_rq(*args, **kwargs)
    def state_sweep(*args, **kwargs): calls.append(("state-sweep", None)); return real_state_sweep(*args, **kwargs)
    with monkeypatch.context() as patch:
        _stub_integrators(patch, [])
        patch.setattr(module, "qualify_source_execution_graph", prepare_q)
        patch.setattr(module, "requalify_source_execution_graph", rq)
        patch.setattr(module, "validate_source_state_sweep", state_sweep)
        result = _run_average_scan(AverageScanRecipe(
            gi_source, gi_root / "average.nxs", ReductionPlan(),
        ))
    assert result.disposition == "COMMITTED"
    assert calls == [("prepare", None), ("requalify", None),
                     ("state-sweep", None), ("state-sweep", None)]


def test_average_tiff_fence_is_aligned_and_nexus_segment_lookup_is_binary(
    tmp_path, monkeypatch,
) -> None:
    from xrd_tools.io.nexus import NexusImageStack
    from xrd_tools.sources import execution_graph

    source = _series(tmp_path)
    for path in source.options["files"]:
        _txt(path, counters=(("I0", 1.0),))
    source = SourceSpec(source.uri, source.kind, options={
        **dict(source.options), "metadata_format": "txt",
    })
    graph = execution_graph.qualify_source_execution_graph(
        source, reader_binding="average_closed_v1",
    )

    class IndexedOnly(tuple):
        iterations = 0

        def __iter__(self):
            self.iterations += 1
            raise AssertionError("TIFF metadata inventory was scanned")

    metadata = IndexedOnly(graph.stamp.metadata_sources)
    object.__setattr__(graph.stamp, "metadata_sources", metadata)
    matched = []
    real_match = execution_graph.SourceFileState.matches_disk

    def matches(state):
        matched.append(state.path)
        return real_match(state)

    monkeypatch.setattr(
        execution_graph.SourceFileState, "matches_disk", matches,
    )
    module._validate_contributor(graph, 2, None)
    assert metadata.iterations == 0
    assert matched == [
        graph.stamp.members[2].path,
        graph.stamp.metadata_sources[2].metadata_file.path,
    ]

    class CountedOffsets:
        def __init__(self, values):
            self.values = values
            self.reads = 0

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            self.reads += 1
            return self.values[index]

    stack = NexusImageStack.__new__(NexusImageStack)
    offsets = CountedOffsets(tuple(range(652)))
    stack._offsets = offsets
    stack._dsets = (None,) * 651
    for index in (0, 325, 650):
        offsets.reads = 0
        assert stack._locate(index) == (index, 0)
        assert offsets.reads <= 12
    offsets.reads = 0
    with pytest.raises(IndexError):
        stack._locate(651)
    assert offsets.reads <= 12


@pytest.mark.parametrize("metadata_format", ("txt", "auto"))
def test_average_rejects_selected_sidecar_foreign_snapshot_then_restore(
    tmp_path, monkeypatch, metadata_format,
) -> None:
    """The second qualification read must not adopt briefly foreign bytes."""

    from xrd_tools.io import metadata as metadata_io
    from xrd_tools.sources import execution_graph

    source = _series(tmp_path, (np.ones((2, 2), dtype=np.uint16),))
    image = Path(source.options["files"][0])
    _txt(image, counters=(("I0", 2.0),))
    sidecar = image.with_suffix(".txt")
    original = sidecar.read_bytes()
    foreign = original.replace(b"I0 = 2.0", b"I0 = 9.0")
    assert foreign != original
    source = SourceSpec(source.uri, source.kind, options={
        **dict(source.options), "metadata_format": metadata_format,
    })
    metadata_io._AUTO_SIDECAR_CACHE.clear()
    real_snapshot = metadata_io._bounded_metadata_snapshot
    snapshots = []

    def transient_snapshot(path, limit, *, _with_revision=False):
        if Path(path) != sidecar:
            return real_snapshot(
                path, limit, _with_revision=_with_revision,
            )
        snapshots.append(1)
        if len(snapshots) != 2:
            return real_snapshot(
                path, limit, _with_revision=_with_revision,
            )
        sidecar.write_bytes(foreign)
        try:
            return real_snapshot(
                path, limit, _with_revision=_with_revision,
            )
        finally:
            sidecar.write_bytes(original)

    adopted = []
    monkeypatch.setattr(
        metadata_io, "_bounded_metadata_snapshot", transient_snapshot,
    )
    monkeypatch.setattr(
        execution_graph, "_finite_motors",
        lambda values: adopted.append(dict(values)) or {},
    )
    # The reader's OSError no longer escapes the runner: a mid-read source
    # change is converted into a TYPED refusal carrying the diagnostic, which is
    # the refuse-before-effects contract this module uses everywhere else.
    # _prepare_average_scan re-raises that as ValueError, so drive the runner
    # directly and assert on the typed object instead.
    runner = module.AverageScanRunner(AverageScanRecipe(
        source, tmp_path / f"average-{metadata_format}.nxs",
        ReductionPlan(integration_1d=Integration1DPlan(npt=2)),
        numeric_metadata_keys=("I0",),
    ))
    runner._execute_graph = lambda _graph: runner.plan
    refusal = runner.start()
    assert type(refusal) is module.AverageScanResult
    assert (refusal.disposition, refusal.diagnostic_code) == (
        "REFUSED", "AVERAGE_PREPARATION_FAILED"
    )
    assert "metadata source changed during read" in refusal.diagnostic
    assert runner.close() is refusal
    # The facts this test exists to protect, unchanged: the transient foreign
    # bytes were never adopted and the sidecar is back as it was.
    assert snapshots == [1, 1]
    assert adopted == []
    assert sidecar.read_bytes() == original


def test_metadata_key_domains_refuse_duplicate_and_overlap_before_effects(tmp_path) -> None:
    with pytest.raises(ValueError, match="duplicate|overlap"):
        AverageScanRecipe(_series(tmp_path), tmp_path / "a.nxs", ReductionPlan(), numeric_metadata_keys=("I0", "I0"))
    with pytest.raises(ValueError, match="duplicate|overlap"):
        AverageScanRecipe(_series(tmp_path), tmp_path / "b.nxs", ReductionPlan(), numeric_metadata_keys=("I0",), invariant_metadata_keys=("I0",))
    with pytest.raises(ValueError, match="nonempty|metadata"):
        AverageScanRecipe(_series(tmp_path), tmp_path / "c.nxs", ReductionPlan(), numeric_metadata_keys=("",))
    assert not any(
        (tmp_path / name).exists()
        for stem in ("a", "b", "c")
        for name in (f"{stem}.nxs", f"{stem}.nexus")
    )


def test_average_static_mask_loads_authenticated_edf(tmp_path) -> None:
    fabio = pytest.importorskip("fabio")
    expected = np.array([[False, True], [True, False]], dtype=bool)
    mask_path = tmp_path / "mask.edf"
    fabio.edfimage.EdfImage(data=expected.astype("u1")).write(
        str(mask_path)
    )
    accepted = module.load_mask(mask_path)
    calibration = CalibrationState(mask=MaskState(
        str(mask_path),
        hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        accepted.dtype.str,
        accepted.shape,
        FactStatus.PRESENT,
    ))
    recipe = AverageScanRecipe(
        _series(tmp_path),
        tmp_path / "average.nxs",
        ReductionPlan(),
        calibration=calibration,
    )

    actual = module._load_static_mask(recipe, expected.shape)

    assert actual is not None
    assert actual.dtype == np.dtype(bool)
    assert not actual.flags.writeable
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("suffix", (".npy", ".edf"))
def test_average_static_mask_decode_is_bound_to_authenticated_bytes(
    tmp_path, monkeypatch, suffix,
) -> None:
    expected = np.array([[False, True], [True, False]], dtype=bool)
    replacement = np.logical_not(expected)
    mask_path = tmp_path / f"mask{suffix}"

    def write_mask(value) -> None:
        if suffix == ".npy":
            np.save(mask_path, value)
        else:
            fabio = pytest.importorskip("fabio")
            fabio.edfimage.EdfImage(data=value.astype("u1")).write(
                str(mask_path)
            )

    write_mask(expected)
    accepted_payload = mask_path.read_bytes()
    accepted = (
        np.load(mask_path, allow_pickle=False)
        if suffix == ".npy"
        else module.load_mask(mask_path)
    )
    calibration = CalibrationState(mask=MaskState(
        str(mask_path), hashlib.sha256(accepted_payload).hexdigest(),
        accepted.dtype.str, accepted.shape, FactStatus.PRESENT,
    ))
    recipe = AverageScanRecipe(
        _series(tmp_path), tmp_path / "average.nxs", ReductionPlan(),
        calibration=calibration,
    )
    decode = module._decode_static_mask_snapshot
    observed = []

    def mutate_then_decode(path):
        observed.append((path, path.read_bytes()))
        write_mask(replacement)
        return decode(path)

    monkeypatch.setattr(
        module, "_decode_static_mask_snapshot", mutate_then_decode,
    )

    actual = module._load_static_mask(recipe, expected.shape)

    assert len(observed) == 1
    assert observed[0][0] != mask_path
    assert observed[0][1] == accepted_payload
    assert not observed[0][0].exists()
    assert hashlib.sha256(mask_path.read_bytes()).hexdigest() != calibration.mask.sha256
    assert not actual.flags.writeable
    np.testing.assert_array_equal(actual, expected)


def test_average_static_mask_limit_scales_and_refuses_before_decode(
    tmp_path, monkeypatch,
) -> None:
    large_shape = (4096, 4096)
    assert module._static_mask_snapshot_limit(large_shape) == (
        8 * np.prod(large_shape) + module._STATIC_MASK_HEADER_ALLOWANCE_BYTES
    )
    assert module._static_mask_snapshot_limit(large_shape) > 64 << 20

    mask_path = tmp_path / "oversize.npy"
    mask_path.write_bytes(b"five!")
    calibration = CalibrationState(mask=MaskState(
        str(mask_path), hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        np.dtype(bool).str, (1, 1), FactStatus.PRESENT,
    ))
    recipe = AverageScanRecipe(
        _series(tmp_path), tmp_path / "average.nxs", ReductionPlan(),
        calibration=calibration,
    )
    monkeypatch.setattr(module, "_static_mask_snapshot_limit", lambda _shape: 4)
    monkeypatch.setattr(
        module, "_qualify_static_mask_snapshot",
        lambda *_a: pytest.fail("oversize mask reached header qualification"),
    )
    monkeypatch.setattr(
        module, "_decode_static_mask_snapshot",
        lambda *_a: pytest.fail("oversize mask reached pixel decoder"),
    )

    with pytest.raises(ValueError, match="AVERAGE_STATIC_MASK_FILE_TOO_LARGE"):
        module._load_static_mask(recipe, (1, 1))


def test_average_static_mask_copy_cancellation_cleans_stream_and_snapshot(
    tmp_path, monkeypatch,
) -> None:
    from threading import Event

    shape = (2048, 1024)
    mask_path = tmp_path / "cancel-mask.npy"
    np.save(mask_path, np.zeros(shape, dtype=np.uint8))
    calibration = CalibrationState(mask=MaskState(
        str(mask_path),
        hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        np.dtype(bool).str,
        shape,
        FactStatus.PRESENT,
    ))
    recipe = AverageScanRecipe(
        _series(tmp_path), tmp_path / "average.nxs", ReductionPlan(),
        calibration=calibration,
    )
    token = Event()
    opened = []
    temporary_roots = []
    real_open = Path.open
    real_temporary = module.tempfile.TemporaryDirectory

    class CancellingSource:
        def __init__(self, stream):
            self.stream = stream
            self.closed = False
        def read(self, size=-1):
            payload = self.stream.read(size)
            token.set()
            return payload
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            self.closed = True
            self.stream.close()

    def controlled_open(path, *args, **kwargs):
        stream = real_open(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        if Path(path) == mask_path and mode == "rb":
            proxy = CancellingSource(stream)
            opened.append(proxy)
            return proxy
        return stream

    def tracked_temporary(*args, **kwargs):
        owner = real_temporary(*args, **kwargs)
        temporary_roots.append(Path(owner.name))
        return owner

    monkeypatch.setattr(Path, "open", controlled_open)
    monkeypatch.setattr(
        module.tempfile, "TemporaryDirectory", tracked_temporary,
    )
    monkeypatch.setattr(
        module, "_decode_static_mask_snapshot",
        lambda *_args, **_kwargs: pytest.fail(
            "cancelled mask snapshot entered the decoder"
        ),
    )

    with pytest.raises(module._AverageCancelled):
        module._load_static_mask(recipe, shape, token)

    assert len(opened) == 1 and opened[0].closed
    assert temporary_roots and all(
        not root.exists() for root in temporary_roots
    )


def test_source_science_and_operation_identity_domains_vary_independently(tmp_path) -> None:
    roots = [tmp_path / name for name in ("base", "source")]
    for root in roots: root.mkdir()
    base_source = _series(roots[0], (np.ones((2, 2), dtype="u2"),))
    other_source = _series(roots[1], (np.full((2, 2), 2, dtype="u2"),))
    base_target = tmp_path / "basé.nxs"
    resources = {"resource_requests": {"workers": 1},
                 "resource_env": {"XDART_PREFETCH_QUEUE_SIZE": "1"}}
    base = _prepare_average_scan(AverageScanRecipe(
        base_source, base_target, ReductionPlan(integration_1d=Integration1DPlan(npt=4)), **resources,
    ))
    changed_source = _prepare_average_scan(AverageScanRecipe(
        other_source, base_target, ReductionPlan(integration_1d=Integration1DPlan(npt=4)), **resources,
    ))
    changed_science = _prepare_average_scan(AverageScanRecipe(
        base_source, base_target, ReductionPlan(integration_1d=Integration1DPlan(npt=5)), **resources,
    ))
    changed_operation = _prepare_average_scan(AverageScanRecipe(
        base_source, tmp_path / "other.nxs", ReductionPlan(integration_1d=Integration1DPlan(npt=4)), **resources,
    ))
    assert base.source_graph_digest != changed_source.source_graph_digest
    assert base.science_identity == changed_source.science_identity
    assert base.source_graph_digest == changed_science.source_graph_digest
    assert base.science_identity != changed_science.science_identity
    assert base.operation_identity != changed_source.operation_identity
    assert base.operation_identity != changed_science.operation_identity
    assert base.operation_identity != changed_operation.operation_identity
    assert (base.source_graph_digest, base.science_identity) == (
        changed_operation.source_graph_digest, changed_operation.science_identity,
    )
    payload = module._operation_payload(base)
    assert set(payload) == {
        "api_version", "source_graph_digest", "science_identity",
        "target_anchor", "output_artifact", "version_identity",
        "entry", "source_base", "output_mode", "live_mode", "save_xye",
        "batch_mode", "contributor_extent", "detector_shape", "native_dtype",
        "numeric_metadata_keys", "invariant_metadata_keys", "logical_labels",
        "direct_eiger_eligible", "expected_target_snapshot", "resource_inputs",
        "allocation",
    }
    assert payload["api_version"] == "average_scan_v1"
    # The anchor is suffix-normalised to .nexus by the shared output owner
    # (_average_target -> resolve_output_target).  That normalisation PRE-DATES
    # the successor route -- it is byte-identical before 6dcfe89b -- so a .nxs
    # request has always been anchored at .nexus; this oracle simply said
    # otherwise.
    assert (payload["target_anchor"], payload["entry"], payload["source_base"],
            payload["output_mode"]) == (
        str(base_target.with_suffix(".nexus").resolve()), "entry", "", "Overwrite",
    )
    # target_anchor is where the caller pointed; output_artifact is the stable
    # `_average` slot actually written, derived from the anchor's family alone.
    # They are never the same path.  The 64-hex version identity is still
    # computed and still travels in provenance -- it just never reaches a
    # public filename, which is the whole point of the slot policy.
    assert payload["output_artifact"] != payload["target_anchor"]
    assert len(payload["version_identity"]) == 64
    assert payload["output_artifact"].endswith("_average.nexus")
    assert payload["version_identity"][:32] not in payload["output_artifact"]
    # The version tracks SCIENCE, not the operation: changing the operation
    # alone leaves it untouched, while changing the science moves it.
    assert (module._operation_payload(changed_operation)["version_identity"]
            == payload["version_identity"])
    assert (module._operation_payload(changed_science)["version_identity"]
            != payload["version_identity"])
    assert (payload["live_mode"], payload["save_xye"], payload["batch_mode"]) == (
        False, False, False,
    )
    assert (payload["contributor_extent"], tuple(payload["detector_shape"]),
            payload["native_dtype"], tuple(payload["logical_labels"])) == (1, (2, 2), "<u2", (1,))
    assert payload["direct_eiger_eligible"] is base.direct_eiger_eligible is False
    assert tuple(payload["numeric_metadata_keys"]) == base.numeric_metadata_keys
    assert tuple(payload["invariant_metadata_keys"]) == base.invariant_metadata_keys
    assert set(payload["expected_target_snapshot"]) == {
        "exists", "size", "mtime_ns", "device", "inode", "digest",
    }
    assert payload["expected_target_snapshot"] == {
        item.name: getattr(base.expected_target_snapshot, item.name)
        for item in fields(base.expected_target_snapshot)
    }
    assert set(payload["resource_inputs"]) == {
        "envelope_bytes", "resource_requests", "resource_env",
    }
    assert payload["resource_inputs"]["envelope_bytes"] is None
    assert tuple(map(tuple, payload["resource_inputs"]["resource_requests"])) == (("workers", 1),)
    assert tuple(map(tuple, payload["resource_inputs"]["resource_env"])) == (
        ("XDART_PREFETCH_QUEUE_SIZE", "1"),
    )
    allocation = payload["allocation"]
    assert set(allocation) == {
        "requirements", "envelope_bytes", "counts", "categories",
        "minimum_bytes", "floor_bytes", "assigned_bytes", "origin",
        "oversize_excess_bytes",
    }
    assert allocation["requirements"] == {
        item.name: getattr(base.allocation.requirements, item.name)
        for item in fields(base.allocation.requirements)
    }
    assert allocation["counts"] == dict(base.allocation.counts)
    assert allocation["categories"] == dict(base.allocation.categories)
    for name in ("envelope_bytes", "minimum_bytes", "floor_bytes",
                 "assigned_bytes", "origin", "oversize_excess_bytes"):
        assert allocation[name] == getattr(base.allocation, name)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False).encode()
    assert base.operation_identity == hashlib.sha256(
        b"xdart.average-operation.v1\0" + canonical,
    ).hexdigest()


def test_average_lineage_iterators_cover_container_tiff_and_eiger(tmp_path, monkeypatch) -> None:
    from xrd_tools.sources import execution_graph
    roots = {name: tmp_path / name for name in ("tiff", "container", "eiger")}
    for root in roots.values(): root.mkdir()
    sources = {"tiff": _series(roots["tiff"], (
        np.ones((2, 2), dtype="u2"), np.full((2, 2), 2, dtype="u2"),
    ))}
    container = roots["container"] / "scan.nxs"
    with h5py.File(container, "w") as handle:
        entry = handle.create_group("entry"); entry.attrs["NX_class"] = "NXentry"
        detector = entry.create_group("instrument/detector")
        detector.create_dataset("data", data=np.arange(12, dtype="u2").reshape(3, 2, 2), chunks=(1, 2, 2))
    sources["container"] = SourceSpec(container, SourceKind.NEXUS_STACK, entry="entry")
    member_paths = []
    for ordinal, extent in enumerate((2, 3), 1):
        path = roots["eiger"] / f"data_{ordinal:06d}.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("entry/data/data", data=np.full((extent, 2, 2), ordinal, dtype="u2"), chunks=(1, 2, 2))
        member_paths.append(path)
    master = roots["eiger"] / "scan_master.h5"
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry"); entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data"); data.attrs["NX_class"] = "NXdata"
        for ordinal, path in enumerate(member_paths, 1):
            data[f"data_{ordinal:06d}"] = h5py.ExternalLink(path.name, "/entry/data/data")
    sources["eiger"] = image_series_spec(master, metadata_format=None)
    graphs = {family: execution_graph.qualify_source_execution_graph(
        source, reader_binding="average_closed_v1",
    ) for family, source in sources.items()}
    assert sources["eiger"].kind is SourceKind.NEXUS_STACK
    assert graphs["eiger"].execution_source.kind is SourceKind.EIGER_MASTER

    observed = []
    routed = []
    validated = []
    real_route = module._external_contributor_route
    real_validate = module._validate_contributor

    def route(graph):
        value = real_route(graph)
        if value:
            routed.append(tuple((item.first, item.stop) for item in value))
        return value

    def validate(graph, index, token, external_member=None):
        if graph.stamp.external_members:
            validated.append((index, None if external_member is None
                              else external_member.first))
        return real_validate(graph, index, token, external_member)

    monkeypatch.setattr(module, "_external_contributor_route", route)
    monkeypatch.setattr(module, "_validate_contributor", validate)
    _stub_integrators(monkeypatch, observed)
    rows = {}
    for family, source in sources.items():
        result = _run_average_scan(AverageScanRecipe(
            source, roots[family] / "average.nxs", ReductionPlan(),
        ))
        assert result.disposition == "COMMITTED"
        iterator = module.iter_average_contributors(result.target)
        assert iter(iterator) is iterator
        rows[family] = tuple(iterator)
    expected = []
    for index, member in enumerate(graphs["tiff"].stamp.members):
        expected.append(AverageContributor(index, index, member.path, 0, None,
            index, index + 1, member.size, member.mtime_ns))
    assert rows["tiff"] == tuple(expected)
    root_state = graphs["container"].stamp.file
    dataset = graphs["container"].dataset_paths[0]
    assert rows["container"] == tuple(AverageContributor(
        index, index, root_state.path, index, dataset, index, index + 1,
        root_state.size, root_state.mtime_ns,
    ) for index in range(3))
    expected = []
    for member in graphs["eiger"].stamp.external_members:
        expected.extend(AverageContributor(
            index, index, member.file.path, index - member.first, member.dataset,
            member.first, member.stop, member.file.size, member.file.mtime_ns,
        ) for index in range(member.first, member.stop))
    assert rows["eiger"] == tuple(expected)
    assert routed == [((0, 2), (2, 5))]
    assert validated == [
        (index, 0 if index < 2 else 2)
        for index in range(5) for _fence in range(2)
    ]


def test_651_frame_inactive_average_has_exact_serial_cadence(
    tmp_path, monkeypatch,
) -> None:
    from xrd_tools.sources import execution_graph

    members = []
    for ordinal, extent in enumerate((217, 217, 217), 1):
        path = tmp_path / f"data_{ordinal:06d}.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset(
                "entry/data/data",
                data=np.arange(extent, dtype="u2").reshape(extent, 1, 1),
                chunks=(1, 1, 1),
            )
        members.append(path)
    source_path = tmp_path / "scan_master.h5"
    with h5py.File(source_path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        for ordinal, path in enumerate(members, 1):
            data[f"data_{ordinal:06d}"] = h5py.ExternalLink(
                path.name, "/entry/data/data",
            )
    source = image_series_spec(source_path, metadata_format=None)
    recipe = AverageScanRecipe(
        source, tmp_path / "average-651.nxs",
        ReductionPlan(integration_1d=Integration1DPlan(npt=3)),
        background=FrameBackgroundPlan(),
    )
    counts = {
        "prepare_qualify": 0, "requalify": 0, "source_state_sweep": 0,
        "read_native": 0, "metadata": 0,
        "contributor_fence": 0, "background_fact": 0,
        "contributor_state": 0, "background_resolver": 0, "count_increment": 0,
        "count_seal": 0,
    }
    fenced_paths = {}
    in_fence = [False]
    integrations = []
    real_prepare = module.qualify_source_execution_graph
    real_requalify = module.requalify_source_execution_graph
    real_state_sweep = module.validate_source_state_sweep
    real_read = execution_graph._AverageSourceReadWindow.read_native
    real_metadata = execution_graph._AverageSourceReadWindow.complete_metadata_for
    real_fence = module._validate_contributor
    real_matches = execution_graph.SourceFileState.matches_disk
    real_count_increment = module._increment_finite_counts
    real_count_seal = module._seal_invariant_finite_counts

    def prepare_qualify(*args, **kwargs):
        counts["prepare_qualify"] += 1
        return real_prepare(*args, **kwargs)

    def requalify(*args, **kwargs):
        counts["requalify"] += 1
        return real_requalify(*args, **kwargs)

    def source_state_sweep(*args, **kwargs):
        counts["source_state_sweep"] += 1
        return real_state_sweep(*args, **kwargs)

    def read_native(window, index):
        counts["read_native"] += 1
        return real_read(window, index)

    def metadata(window, index):
        counts["metadata"] += 1
        return real_metadata(window, index)

    def fence(graph, index, token, external_member=None):
        counts["contributor_fence"] += 1
        in_fence[0] = True
        try:
            return real_fence(graph, index, token, external_member)
        finally:
            in_fence[0] = False

    def matches(state):
        if in_fence[0]:
            counts["contributor_state"] += 1
            fenced_paths[state.path] = fenced_paths.get(state.path, 0) + 1
        return real_matches(state)

    def count_increment(*args, **kwargs):
        counts["count_increment"] += 1
        return real_count_increment(*args, **kwargs)

    def count_seal(*args, **kwargs):
        counts["count_seal"] += 1
        return real_count_seal(*args, **kwargs)

    def forbidden(name):
        def call(*_args, **_kwargs):
            counts[name] += 1
            pytest.fail(f"inactive Average invoked {name}")
        return call

    _stub_integrators(monkeypatch, integrations)
    monkeypatch.setattr(
        module, "qualify_source_execution_graph", prepare_qualify,
    )
    monkeypatch.setattr(module, "requalify_source_execution_graph", requalify)
    monkeypatch.setattr(module, "validate_source_state_sweep", source_state_sweep)
    monkeypatch.setattr(
        execution_graph._AverageSourceReadWindow, "read_native", read_native,
    )
    monkeypatch.setattr(
        execution_graph._AverageSourceReadWindow,
        "complete_metadata_for", metadata,
    )
    monkeypatch.setattr(module, "_validate_contributor", fence)
    monkeypatch.setattr(execution_graph.SourceFileState, "matches_disk", matches)
    monkeypatch.setattr(module, "_increment_finite_counts", count_increment)
    monkeypatch.setattr(
        module, "_seal_invariant_finite_counts", count_seal,
    )
    monkeypatch.setattr(
        module, "_background_fact", forbidden("background_fact"),
    )
    monkeypatch.setattr(
        module, "resolve_frame_background",
        forbidden("background_resolver"),
    )
    result = _run_average_scan(recipe)
    assert result.disposition == "COMMITTED"
    assert len(integrations) == 1
    assert counts == {
        "prepare_qualify": 1, "requalify": 1, "source_state_sweep": 2,
        "read_native": 651, "metadata": 1,
        "contributor_fence": 1302, "background_fact": 0,
        "contributor_state": 2604, "background_resolver": 0,
        "count_increment": 0,
        "count_seal": 1,
    }
    assert fenced_paths == {
        str(source_path.resolve()): 1302,
        **{str(path.resolve()): 434 for path in members},
    }


def test_average_elides_only_integer_thresholds_outside_native_domain() -> None:
    assert module._effective_integer_thresholds(
        np.dtype("<u4"), 0.0, 10000.0,
    ) == (None, 10000.0)
    assert module._effective_integer_thresholds(
        np.dtype("<i2"), -32768.0, 32767.0,
    ) == (None, None)
    assert module._effective_integer_thresholds(
        np.dtype("<i2"), -10.0, 10.0,
    ) == (-10.0, 10.0)
    assert module._effective_integer_thresholds(
        np.dtype("<f4"), 0.0, 10000.0,
    ) == (0.0, 10000.0)


def test_average_contributor_fields_are_exact_per_logical_frame_and_family(tmp_path) -> None:
    value = AverageContributor(2, 2, "/raw/data.h5", 1, "/entry/data", 2, 5, 100, 200)
    assert tuple(item.name for item in fields(value)) == (
        "ordinal", "logical_index", "source_path", "source_frame_index",
        "dataset_path", "source_start", "source_stop", "size", "mtime_ns",
    )
    with pytest.raises((TypeError, ValueError)):
        AverageContributor(1, 2, "/x", 0, None, 0, 1, 0, 0)
    with pytest.raises((TypeError, ValueError)):
        AverageContributor(0, 0, "/x", 0, None, 1, 1, 0, 0)
    malformed = tmp_path / "malformed.nxs"
    with h5py.File(malformed, "w") as handle:
        entry = handle.create_group("entry")
        entry.create_dataset("frame_index", data=[1])
        frames = entry.create_group("frames"); frame = frames.create_group("frame_0001")
        values = np.ones((1, 1), dtype="<u4")
        evidence = module.AverageFiniteCountsEvidence(
            "average_scan_v1", 1, (1, 1), "<u4",
            module._average_count_digest(values, 1, rows=1), 1, 1, 0,
            (1, 1), "gzip", 1, True, False,
        )
        from xrd_tools.io.nexus_record import write_average_finite_counts
        write_average_finite_counts(entry, module.AverageFiniteCounts(evidence, values))
    # This fixture is too minimal to reach the lineage/capability check: the
    # current-processed-record gate refuses first, which is the correct order
    # (refuse at the outermost contract).  COVERAGE NOTE: the lineage-specific
    # branch is therefore no longer exercised here and needs a fixture that IS a
    # current record and fails only on lineage.  Recorded as debt rather than
    # papered over.
    with pytest.raises(
        ValueError, match="lineage|capability|source|current xdart",
    ):
        tuple(module.iter_average_contributors(malformed))


def test_settlement_retry_revalidates_without_recompute(tmp_path, monkeypatch) -> None:
    from xrd_tools.io import get_average_finite_counts
    from xrd_tools.io.record_writer import NexusRecordWriter
    from xrd_tools.sources import execution_graph

    def inputs(case):
        root = tmp_path / case; root.mkdir()
        dependency = None
        if case == "dependency":
            dependency = root / "pixels.h5"
            with h5py.File(dependency, "w") as handle:
                handle.create_dataset("pixels", data=np.arange(12, dtype="u2").reshape(3, 2, 2))
            master = root / "scan.nxs"
            with h5py.File(master, "w", libver="latest") as handle:
                entry = handle.create_group("entry"); entry.attrs["NX_class"] = "NXentry"
                data = entry.create_group("data"); data.attrs["NX_class"] = "NXdata"
                layout = h5py.VirtualLayout(shape=(3, 2, 2), dtype="u2")
                layout[:] = h5py.VirtualSource(str(dependency), "/pixels", shape=(3, 2, 2))
                data.create_virtual_dataset("image", layout)
            source = SourceSpec(master, SourceKind.NEXUS_STACK, entry="entry")
        else:
            source = _series(root)
            if case == "sidecar":
                for path in source.options["files"]: _txt(path, counters=(("I0", 1.0),))
                source = SourceSpec(source.uri, source.kind, options={
                    **dict(source.options), "metadata_format": "txt",
                })
        target = root / "average.nxs"
        with h5py.File(target, "w") as handle: handle.create_dataset("prior", data=np.arange(3))
        mutate = (Path(source.options["files"][1]) if case == "member" else
            Path(source.options["files"][1]).with_suffix(".txt") if case == "sidecar"
            else dependency if case == "dependency" else target if case == "target" else None)
        return source, target, mutate

    for case in ("unchanged", "member", "sidecar", "dependency", "target", "commit-target"):
        source, target, mutate = inputs(case); before_target = target.read_bytes()
        recipe = AverageScanRecipe(
            source, target, ReductionPlan(integration_1d=Integration1DPlan(npt=3)),
            numeric_metadata_keys=("I0",) if case == "sidecar" else None,
        )
        counts = {name: 0 for name in ("open", "pixel", "metadata", "background", "write")}
        events = []; sinks = []; integrations = []
        source_sweeps = []
        real_open = module.open_source_execution_graph
        real_pixel = execution_graph._AverageSourceReadWindow.read_native
        real_metadata = execution_graph._AverageSourceReadWindow.complete_metadata_for
        real_background = module.resolve_frame_background
        real_state_sweep = module.validate_source_state_sweep
        real_target = module.capture_target_snapshot
        real_sink_factory = module.NexusSink.for_finite_document
        real_verify = NexusRecordWriter._verify_dirty_evidence
        held = []
        def opened(*args, **kwargs): counts["open"] += 1; return real_open(*args, **kwargs)
        def pixel(window, index): counts["pixel"] += 1; return real_pixel(window, index)
        def metadata(window, index): counts["metadata"] += 1; return real_metadata(window, index)
        def background(*args, **kwargs): counts["background"] += 1; return real_background(*args, **kwargs)
        def source_sweep(*args, **kwargs):
            source_sweeps.append(1)
            events.append("source-sweep")
            if case == "commit-target" and len(source_sweeps) == 2:
                # Same stale anchor as the "target" case above: the successor
                # route never writes average.nxs, so mutating it no longer
                # blocks the commit.  Mutate the artifact under settlement.
                artifact = Path(sinks[0]._writer.target)
                artifact.write_bytes(
                    (artifact.read_bytes() if artifact.exists() else b"") + b"x"
                )
            return real_state_sweep(*args, **kwargs)
        def snapshot(*args, **kwargs):
            events.append("target-sweep")
            return real_target(*args, **kwargs)
        def sink(*args, **kwargs):
            value = real_sink_factory(*args, **kwargs)
            sinks.append(value)
            return value
        def verify(owner):
            value = real_verify(owner)
            if len(held) < (2 if case == "unchanged" else 1) and f"{owner.entry}/frames/frame_0001/finite_counts" in owner._h5:
                held.append(1); raise OSError("p36 H23 precommit hold")
            return value
        with monkeypatch.context() as patch:
            _stub_integrators(patch, integrations)
            patch.setattr(module, "open_source_execution_graph", opened)
            patch.setattr(execution_graph._AverageSourceReadWindow, "read_native", pixel)
            patch.setattr(execution_graph._AverageSourceReadWindow, "complete_metadata_for", metadata)
            patch.setattr(module, "resolve_frame_background", background)
            patch.setattr(module, "validate_source_state_sweep", source_sweep)
            patch.setattr(module, "capture_target_snapshot", snapshot)
            patch.setattr(module.NexusSink, "for_finite_document", sink)
            patch.setattr(NexusRecordWriter, "_verify_dirty_evidence", verify)
            runner = module.AverageScanRunner(recipe)
            pending = runner.start()
            plan = runner.plan
            assert type(pending) is module.AverageScanPending
            assert pending.phase is module.AveragePendingPhase.OUTPUT_SETTLEMENT
            assert pending.operation_identity == plan.operation_identity
            assert pending.revision == 1
            assert len(sinks) == 1 and held == [1]
            owned_sink = sinks[0]; writer = owned_sink._writer
            assert writer.phase.value == "partial"
            custody = (id(owned_sink), id(writer), id(owned_sink._finite_candidate))
            finalization = writer._finalization.average_finite_counts
            assert finalization is not None
            frozen_counts = dict(counts); frozen_integrations = len(integrations); events.clear()
            if case == "target":
                # Under the immutable-successor route the ANCHOR (average.nxs)
                # is never written, so mutating it tests nothing -- this case
                # had silently become a duplicate of "unchanged".  Mutate the
                # artifact actually under settlement.  It need not exist yet:
                # something APPEARING at the successor path mid-run is exactly
                # what expected_target_snapshot guards (average.py:1454).
                artifact = Path(plan.output_artifact)
                artifact.write_bytes(
                    (artifact.read_bytes() if artifact.exists() else b"") + b"x"
                )
                mutate = None
            if mutate is not None: mutate.write_bytes(mutate.read_bytes() + b"x")
            if case == "unchanged":
                again = runner.command(module.AverageCommand.RETRY, pending)
                assert type(again) is module.AverageScanPending
                assert again.phase is module.AveragePendingPhase.OUTPUT_SETTLEMENT
                assert again.revision == 2 and held == [1, 1]
                assert events == []
                events.clear()
                pending = again
            terminal = runner.command(module.AverageCommand.RETRY, pending)
            assert events.count("source-sweep") <= 1
            assert events.count("target-sweep") <= 2
            if case in {"unchanged", "commit-target"}:
                assert "source-sweep" in events
            assert counts == frozen_counts and len(integrations) == frozen_integrations
            assert writer._finalization.average_finite_counts is finalization
            assert custody == (id(owned_sink), id(writer), id(owned_sink._finite_candidate))
            if case == "unchanged":
                assert terminal.disposition == "COMMITTED" and terminal.committed_labels == (1,)
            else:
                assert terminal.disposition == "ABORTED" and terminal.diagnostic_code
                if case in {"member", "sidecar", "dependency"}: assert terminal.diagnostic_code == "AVERAGE_SOURCE_DRIFT"
                else: assert terminal.diagnostic_code in {
                    "AVERAGE_H23_FAILED", "AVERAGE_TARGET_DRIFT",
                }
                assert target.read_bytes() == before_target
                # The ANCHOR is never written by this route, so its bytes cannot
                # fail on a botched rollback -- that oracle was vacuous.  The
                # ARTIFACT is what the route creates and must remove on abort.
                artifact = Path(module._average_output_artifact(
                    module._average_target(str(target))
                ))
                if case in {"target", "commit-target"}:
                    # THE TEST put this file at the slot, to simulate something
                    # appearing there mid-run.  Average must leave it EXACTLY
                    # alone: it did not create it, so it may not remove it.
                    # Under the parent the transaction admitted the SLOT, so
                    # rollback restored "absent" by UNLINKING an operator's
                    # file.  Writing to a hidden candidate instead means the
                    # only thing Average can roll back is its own.
                    assert artifact.exists()
                    assert artifact.read_bytes().endswith(b"x")
                else:
                    assert not artifact.exists()
                with pytest.raises((KeyError, ValueError)): get_average_finite_counts(target)
                assert not tuple(target.parent.glob(".xdart-finite-*"))
                assert not tuple(artifact.parent.glob(f".{artifact.name}*"))
                # The shared owner uses a `.xdart-finite-*` private candidate,
                # which the globs above do NOT match -- check it explicitly, or an
                # abort could strand a private file and nothing would say so.
                assert not tuple(
                    artifact.parent.glob(".xdart-finite-*")
                )
            terminal_events, terminal_counts = tuple(events), dict(counts)
            assert runner.close() is terminal
            assert tuple(events) == terminal_events and counts == terminal_counts

    from threading import Event
    source, target, _ = inputs("cancel-before-gate")
    before_target = target.read_bytes(); token = Event(); sinks = []; gate_calls = []
    real_gate = module.AverageScanRunner._gate
    real_sink_factory = module.NexusSink.for_finite_document
    def cancel_at_gate(owner): token.set(); return real_gate(owner)
    def capture_sink(*args, **kwargs):
        value = real_sink_factory(*args, **kwargs)
        sinks.append(value)
        return value
    with monkeypatch.context() as patch:
        _stub_integrators(patch, [])
        patch.setattr(module.AverageScanRunner, "_gate", cancel_at_gate)
        patch.setattr(module.NexusSink, "for_finite_document", capture_sink)
        terminal = _run_average_scan(AverageScanRecipe(
            source, target, ReductionPlan(integration_1d=Integration1DPlan(npt=3)),
        ), cancel_token=token, publication_gate=lambda: gate_calls.append(1) or True)
    assert (terminal.disposition, terminal.diagnostic_code) == ("CANCELLED", "AVERAGE_CANCELLED")
    assert gate_calls == [] and target.read_bytes() == before_target
    assert len(sinks) == 1
    assert not tuple(target.parent.glob(".xdart-finite-*"))

    source, target, _ = inputs("cancel-during-final-sweep")
    before_target = target.read_bytes(); token = Event(); sinks = []; gate_calls = []; sweeps = []
    real_state_sweep = module.validate_source_state_sweep
    def cancel_final_sweep(*args, **kwargs):
        sweeps.append(1)
        if len(sweeps) == 2: token.set(); raise InterruptedError("cancelled sweep")
        return real_state_sweep(*args, **kwargs)
    with monkeypatch.context() as patch:
        _stub_integrators(patch, [])
        patch.setattr(module, "validate_source_state_sweep", cancel_final_sweep)
        patch.setattr(module.NexusSink, "for_finite_document", capture_sink)
        terminal = _run_average_scan(AverageScanRecipe(
            source, target, ReductionPlan(integration_1d=Integration1DPlan(npt=3)),
        ), cancel_token=token, publication_gate=lambda: gate_calls.append(1) or True)
    assert (terminal.disposition, terminal.diagnostic_code) == ("CANCELLED", "AVERAGE_CANCELLED")
    assert sweeps == [1, 1] and gate_calls == [] and target.read_bytes() == before_target
    assert len(sinks) == 1

    source, target, _ = inputs("cancel-during-initial-qualification")
    before_target = target.read_bytes(); token = Event(); qualifications = []
    def interrupt_initial(*args, **kwargs):
        qualifications.append(args[0]); token.set(); raise InterruptedError("cancelled qualification")
    with monkeypatch.context() as patch:
        patch.setattr(module, "qualify_source_execution_graph", interrupt_initial)
        terminal = _run_average_scan(AverageScanRecipe(
            source, target, ReductionPlan(integration_1d=Integration1DPlan(npt=3)),
        ), cancel_token=token)
    assert (terminal.disposition, terminal.diagnostic_code) == ("CANCELLED", "AVERAGE_CANCELLED")
    assert len(qualifications) == 1 and target.read_bytes() == before_target


def test_late_cancel_after_accepted_gate_keeps_exact_sealed_result(
    tmp_path, monkeypatch,
) -> None:
    from threading import Event

    source = _series(tmp_path)
    target = tmp_path / "average.nxs"
    with h5py.File(target, "w") as handle:
        handle.create_dataset("prior", data=np.arange(3))
    _stub_integrators(monkeypatch, [])
    token = Event()

    def accept_gate():
        token.set()
        return True

    runner = module.AverageScanRunner(AverageScanRecipe(
        source, target,
        ReductionPlan(integration_1d=Integration1DPlan(npt=3)),
    ))
    terminal = runner.start(cancel_token=token, publication_gate=accept_gate)
    assert terminal.disposition == "COMMITTED"
    assert runner._gate_called and token.is_set()
    assert Path(terminal.target).exists()
    assert runner.close() is terminal


def test_finite_document_close_failure_stays_pending_until_actual_close(
    tmp_path, monkeypatch,
) -> None:
    source = _series(tmp_path)
    target = tmp_path / "average.nxs"
    with h5py.File(target, "w") as handle:
        handle.create_dataset("prior", data=np.arange(3))
    integrations = []
    _stub_integrators(monkeypatch, integrations)
    real_open = module.open_nexus_writer
    documents = []
    calls = []

    def opened(*args, **kwargs):
        document = real_open(*args, **kwargs)
        documents.append(document)
        real_close = document.close

        def fail_once():
            calls.append(1)
            if len(calls) == 1:
                raise OSError("injected actual HDF5 close failure")
            return real_close()

        monkeypatch.setattr(document, "close", fail_once)
        return document

    monkeypatch.setattr(module, "open_nexus_writer", opened)
    runner = module.AverageScanRunner(AverageScanRecipe(
        source, target,
        ReductionPlan(integration_1d=Integration1DPlan(npt=3)),
    ))
    pending = runner.start()
    assert type(pending) is module.AverageScanPending
    assert pending.phase is module.AveragePendingPhase.OUTPUT_SETTLEMENT
    assert len(documents) == 1
    document = documents[0]
    candidate = runner._publication.candidate
    writer = runner._sink._writer
    assert runner._publication._document is document and document.id.valid
    assert candidate.exists() and not Path(runner.plan.output_artifact).exists()
    assert len(integrations) == 1 and calls == [1]

    terminal = runner.command(module.AverageCommand.RETRY, pending)
    assert terminal.disposition == "COMMITTED"
    assert calls == [1, 1] and len(integrations) == 1
    assert runner._publication._document is None
    assert runner._sink._writer is writer and not document.id.valid
    assert not candidate.exists()
    assert runner.close() is terminal


def test_average_fail_loud_monitor_policy_is_identical_in_science_provenance_and_execution(tmp_path, monkeypatch) -> None:
    from xdart.gui.tabs.scattering.adapters.external_operation import OperationSlot
    from xdart.gui.tabs.scattering.operation_values import (
        OperationContextStamp, OperationTerminalStatus,
    )

    def gui_run(source, target, reduction, *, keys=("I0",)):
        slot = OperationSlot()
        identity = slot.begin_average(
            _frozen_configuration(source, reduction), target,
            numeric_metadata_keys=keys,
            stamp=OperationContextStamp(0),
        )
        assert identity is not None and slot._worker is not None
        slot._worker.join(5); assert not slot._worker.is_alive()
        update = slot.poll(identity)
        assert update is not None and update.terminal is not None
        assert update.terminal.status is OperationTerminalStatus.RETURNED
        return update.terminal.payload

    strict = {
        "policy": "average_reduction_strict_v1",
        "missing_normalization": True,
        "gi_all_dummy": True,
        "thumbnail_fallback": True,
    }
    for dimension in ("1d", "2d"):
        for ordinal, value in enumerate((None, 0.0, -1.0, np.inf, -np.inf, np.nan)):
            root = tmp_path / f"{dimension}-{ordinal}"; root.mkdir()
            source = _series(root, (np.ones((2, 2), dtype="u2"),))
            target = root / "direct.nxs"; gui_target = root / "gui.nxs"
            for path in (target, gui_target):
                with h5py.File(path, "w") as handle:
                    handle.create_dataset("prior", data=[ordinal])
            before = target.read_bytes(); gui_before = gui_target.read_bytes()
            row = {} if value is None else {"I0": value}
            if row:
                _txt(source.options["files"][0], counters=(("I0", value),))
            source = SourceSpec(source.uri, source.kind, options={
                **dict(source.options), "metadata_format": "txt",
            })
            calls = []
            _stub_integrators(monkeypatch, calls)
            reduction = (ReductionPlan(integration_1d=Integration1DPlan(npt=3, monitor_key="I0"))
                             if dimension == "1d" else ReductionPlan(
                                 integration_1d=Integration1DPlan(monitor_key="I0"),
                             integration_2d=Integration2DPlan(npt_rad=2, npt_azim=3, monitor_key="I0"),
                         ))
            recipe = AverageScanRecipe(source, target, reduction, numeric_metadata_keys=("I0",))
            plan = _prepare_average_scan(recipe)
            assert _science_of(plan)["reduction_strict_policy"] == strict
            result = _run_average_scan(recipe)
            gui = gui_run(source, gui_target, reduction)
            assert (result.disposition, result.diagnostic_code) == (
                "REFUSED", "AVERAGE_REDUCTION_MISSING_NORMALIZATION",
            )
            assert (gui.disposition, gui.diagnostic_code) == (
                result.disposition, result.diagnostic_code,
            )
            assert gui.science_identity == result.science_identity == plan.science_identity
            assert gui.metadata_denominators == result.metadata_denominators
            assert calls == [] and target.read_bytes() == before
            assert gui_target.read_bytes() == gui_before
            for path in (target, gui_target):
                with pytest.raises((KeyError, ValueError)):
                    module.get_average_finite_counts(path)

    valid = tmp_path / "valid"; valid.mkdir()
    source = _series(valid, (np.ones((2, 2), dtype="u2"),))
    _txt(
        source.options["files"][0],
        counters=(("I0", 2.0), ("optional", float("nan"))),
    )
    source = SourceSpec(source.uri, source.kind, options={
        **dict(source.options), "metadata_format": "txt",
    })
    calls = []
    _stub_integrators(monkeypatch, calls)
    reduction = ReductionPlan(
        integration_1d=Integration1DPlan(npt=3, monitor_key="I0"),
    )
    result = _run_average_scan(AverageScanRecipe(
        source, valid / "direct.nxs", reduction,
        numeric_metadata_keys=("I0", "optional"),
    ))
    gui = gui_run(
        source, valid / "gui.nxs", reduction, keys=("I0", "optional"),
    )
    assert result.disposition == gui.disposition == "COMMITTED" and len(calls) == 2
    assert result.science_identity == gui.science_identity
    assert dict(result.metadata_denominators)["optional"] == 0
    assert result.metadata_denominators == gui.metadata_denominators
    from xrd_tools.io import get_metadata
    direct_scan = get_metadata(result.target)["scan_data"]
    gui_scan = get_metadata(gui.target)["scan_data"]
    assert set(direct_scan) >= {"I0", "optional"} and set(gui_scan) >= {"I0", "optional"}
    assert np.isnan(direct_scan["optional"][0]) and np.isnan(gui_scan["optional"][0])
    from xrd_tools.core.provenance import read_provenance
    persisted = read_provenance(result.target)["config"]["average_scan_v1"]
    gui_persisted = read_provenance(gui.target)["config"]["average_scan_v1"]
    assert persisted["reduction_strict_policy"] == strict
    assert gui_persisted["reduction_strict_policy"] == strict


def test_average_fail_loud_all_dummy_2d_refuses_before_commit(tmp_path, monkeypatch) -> None:
    from xdart.gui.tabs.scattering.adapters.external_operation import OperationSlot
    from xdart.gui.tabs.scattering.operation_values import (
        OperationContextStamp, OperationTerminalStatus,
    )
    source = _series(tmp_path, (np.ones((2, 2), dtype="u2"),))
    target = tmp_path / "direct.nxs"; gui_target = tmp_path / "gui.nxs"
    for path in (target, gui_target):
        with h5py.File(path, "w") as handle:
            handle.create_dataset("prior", data=[1, 2, 3])
    before = target.read_bytes(); gui_before = gui_target.read_bytes()
    observed = []
    _stub_integrators(monkeypatch, observed, all_dummy_2d=True)
    reduction = ReductionPlan(
        integration_1d=Integration1DPlan(),
        integration_2d=Integration2DPlan(npt_rad=2, npt_azim=3),
    )
    recipe = AverageScanRecipe(source, target, reduction)
    plan = _prepare_average_scan(recipe)
    strict = {"policy": "average_reduction_strict_v1", "missing_normalization": True,
              "gi_all_dummy": True, "thumbnail_fallback": True}
    assert _science_of(plan)["reduction_strict_policy"] == strict
    result = _run_average_scan(recipe)
    slot = OperationSlot()
    identity = slot.begin_average(
        _frozen_configuration(source, reduction), gui_target,
        stamp=OperationContextStamp(0),
    )
    assert identity is not None and slot._worker is not None
    slot._worker.join(5); assert not slot._worker.is_alive()
    update = slot.poll(identity)
    assert update is not None and update.terminal is not None
    assert update.terminal.status is OperationTerminalStatus.RETURNED
    gui = update.terminal.payload
    assert (result.disposition, result.diagnostic_code) == (
        "REFUSED", "AVERAGE_REDUCTION_GI_ALL_DUMMY",
    )
    assert (gui.disposition, gui.diagnostic_code, gui.science_identity) == (
        result.disposition, result.diagnostic_code, result.science_identity,
    )
    assert result.science_identity == plan.science_identity
    assert result.diagnostic and gui.diagnostic == result.diagnostic
    assert result.commit_identity is gui.commit_identity is None
    assert result.finite_counts is gui.finite_counts is None
    assert [row[0] for row in observed] == ["1d", "2d", "1d", "2d"]
    assert target.read_bytes() == before and gui_target.read_bytes() == gui_before
    for path in (target, gui_target):
        with pytest.raises((KeyError, ValueError)):
            module.get_average_finite_counts(path)


def test_average_repeat_in_the_same_directory_atomically_replaces(
    tmp_path, monkeypatch,
) -> None:
    """A second Average for the same family REPLACES the slot. The ruling.

    This row previously asserted a REFUSAL: `_average_output_artifact` is
    version-independent, so ANY second Average for one anchor -- different frame
    selection, different npt, anything -- resolved the one slot and was refused
    at plan time by `AVERAGE_OUTPUT_EXISTS`. The operator had to delete the file
    by hand.

    Ruled 2026-09-04: an occupied slot is replaced unconditionally, and an
    identical repeat replaces too. The refusal is gone; Average now writes a
    hidden same-directory candidate and one atomic replacement publishes it.

    This FIXTURE names its scan `average`, so the family is `average` and the
    slot reads `average_average.nexus`. That doubling is the fixture's own name,
    not a product default -- `page.py:2553` builds the anchor from the save
    FOLDER plus the SCAN NAME, so a real scan gives `<scan>_average.nexus`.
    """
    source = _series(tmp_path)
    target = tmp_path / "average.nxs"
    _stub_integrators(monkeypatch, [])

    first = _run_average_scan(AverageScanRecipe(
        source, target, ReductionPlan(integration_1d=Integration1DPlan(npt=4)),
    ))
    assert first.disposition == "COMMITTED"
    artifact = Path(first.target)
    assert artifact.name == "average_average.nexus"
    committed_bytes = artifact.read_bytes()
    first_state = artifact.stat()

    second = _run_average_scan(AverageScanRecipe(
        source, target, ReductionPlan(integration_1d=Integration1DPlan(npt=6)),
    ))

    assert second.disposition == "COMMITTED"
    assert second.target == first.target
    # The SECOND science occupies the slot, and it is genuinely a new file.
    assert artifact.read_bytes() != committed_bytes
    assert (artifact.stat().st_dev, artifact.stat().st_ino) != (
        first_state.st_dev, first_state.st_ino
    )
    # The commit terminal describes the PUBLISHED slot, not the candidate that
    # produced it -- a copied receipt would name a file that no longer exists.
    assert second.commit_identity.target == second.target
    assert second.commit_identity.inode == artifact.stat().st_ino
    # Nothing private is left beside it.
    assert not tuple(artifact.parent.glob(".xdart-finite-*"))


def test_a_failed_average_leaves_the_prior_slot_byte_identical(
    tmp_path, monkeypatch,
) -> None:
    """The ADR guarantee this whole design exists to deliver.

    ADR-0010: "The prior slot remains visible and untouched until that
    publication step. STOP during Average discards the partial candidate and
    leaves the prior slot unchanged."

    Nothing pinned that until now. It is the reason Average writes a hidden
    candidate at all rather than simply dropping `AVERAGE_OUTPUT_EXISTS` -- the
    streamed transaction moves the prior to a BACKUP first, so a failure
    mid-Average would have left the public slot missing rather than intact.
    """
    import xrd_tools.reduction.average as module

    source = _series(tmp_path)
    target = tmp_path / "average.nxs"
    _stub_integrators(monkeypatch, [])

    first = _run_average_scan(AverageScanRecipe(
        source, target, ReductionPlan(integration_1d=Integration1DPlan(npt=4)),
    ))
    assert first.disposition == "COMMITTED"
    artifact = Path(first.target)
    prior_bytes = artifact.read_bytes()
    prior_state = artifact.stat()

    real_write = module.NexusSink.write

    def failing_write(owner, *args, **kwargs):
        real_write(owner, *args, **kwargs)
        raise OSError("injected mid-Average writer failure")

    with monkeypatch.context() as fault:
        fault.setattr(module.NexusSink, "write", failing_write)
        second = _run_average_scan(AverageScanRecipe(
            source, target,
            ReductionPlan(integration_1d=Integration1DPlan(npt=6)),
        ))

    assert second.disposition != "COMMITTED"
    # UNTOUCHED: same bytes, same inode -- not restored from a backup, never
    # moved in the first place.
    assert artifact.read_bytes() == prior_bytes
    assert (artifact.stat().st_dev, artifact.stat().st_ino) == (
        prior_state.st_dev, prior_state.st_ino
    )
    # And the partial candidate is discarded, not stranded.
    assert not tuple(artifact.parent.glob(".xdart-finite-*"))


def test_average_factory_failure_closes_cold_candidate_and_preserves_slot(
    tmp_path, monkeypatch,
) -> None:
    source = _series(tmp_path)
    target = tmp_path / "average.nxs"
    _stub_integrators(monkeypatch, [])

    first = _run_average_scan(AverageScanRecipe(
        source, target, ReductionPlan(integration_1d=Integration1DPlan(npt=4)),
    ))
    assert first.disposition == "COMMITTED"
    artifact = Path(first.target)
    prior_bytes = artifact.read_bytes()
    prior_state = artifact.stat()
    calls = []

    def fail_factory(path, document, **kwargs):
        calls.append((document, Path(document.filename)))
        raise OSError("injected finite sink factory failure")

    with monkeypatch.context() as fault:
        fault.setattr(module.NexusSink, "for_finite_document", fail_factory)
        runner = module.AverageScanRunner(AverageScanRecipe(
            source, target,
            ReductionPlan(integration_1d=Integration1DPlan(npt=6)),
        ))
        failed = runner.start()
        if type(failed) is module.AverageScanPending:
            failed = runner.command(module.AverageCommand.CANCEL, failed)

    assert len(calls) == 1
    document, candidate = calls[0]
    assert not document.id.valid and not candidate.exists()
    assert failed.disposition != "COMMITTED"
    assert artifact.read_bytes() == prior_bytes
    assert (artifact.stat().st_dev, artifact.stat().st_ino) == (
        prior_state.st_dev, prior_state.st_ino,
    )
    assert not tuple(artifact.parent.glob(".xdart-finite-*"))

    recovered = _run_average_scan(AverageScanRecipe(
        source, target, ReductionPlan(integration_1d=Integration1DPlan(npt=6)),
    ))
    assert recovered.disposition == "COMMITTED"


def test_average_persists_its_root_family_so_the_next_operation_cannot_chain(
    tmp_path, monkeypatch,
) -> None:
    """A committed Average carries `@artifact_family_v1`, refuting P-CHAIN.

    Fable F1 (P2) on `c167b71a`. Average writes through `NexusSink`, not the
    finite publisher, so before this change its output carried NO root family.
    `_predecessor` then fell through to a stem-derived family and the next
    operation published `<family>_average_reintegrate1d.nexus` -- the chained
    name ADR-0010 forbids by name, reachable today, in the same range that
    carried the ADR.

    Three things are asserted, and the third is the one that matters: the
    attribute exists, it is the ROOT family rather than the artifact's stem, and
    the family a following operation would CONSUME is the root -- so its slot is
    `<root>_reintegrate1d.nexus` and not a chain.
    """
    from xrd_tools.io.output_path import (
        artifact_family_from_source,
        resolve_finite_output_target,
    )
    from xrd_tools.io.schema import ARTIFACT_FAMILY_ATTR
    from xrd_tools.reduction.reintegrate_successor import _persisted_root_family

    source = _series(tmp_path)
    target = tmp_path / "scan12.nxs"
    _stub_integrators(monkeypatch, [])
    result = _run_average_scan(AverageScanRecipe(
        source, target, ReductionPlan(integration_1d=Integration1DPlan(npt=4)),
    ))
    assert result.disposition == "COMMITTED"
    artifact = Path(result.target)
    assert artifact.name == "scan12_average.nexus"

    with h5py.File(artifact, "r") as document:
        stored = document["entry"].attrs[ARTIFACT_FAMILY_ATTR]
    if isinstance(stored, bytes):
        stored = stored.decode("utf-8")
    # The ROOT, not this artifact's stem.
    assert str(stored) == "scan12"
    assert str(stored) != artifact.stem

    consumed = _persisted_root_family(artifact, "entry")
    assert consumed == "scan12"
    family = artifact_family_from_source(artifact, consumed)
    following = resolve_finite_output_target(
        artifact.parent, family, operation_token="reintegrate-1d",
    )
    assert following.name == "scan12_reintegrate1d.nexus"
    # The chain that was reachable before this change.
    assert following.name != "scan12_average_reintegrate1d.nexus"


def test_a_slot_named_artifact_without_a_family_keeps_its_stem(tmp_path) -> None:
    """No stamped family means the STEM is the family -- no suffix interpretation.

    An earlier revision of mine refused a stem ending in a known slot, reasoning
    that its root could not be recovered. Codex ruled that out on 2026-09-04 and
    was right on both counts:

    * it was WRONG, refusing `series_rsm.nexus` -- a scan legitimately named
      `series_rsm` -- with no way for the operator to proceed; and
    * it was UNNECESSARY, because every artifact the slot-aware writers produce
      carries `@artifact_family_v1`. A slot-shaped stem WITHOUT one was never
      written under this policy, so its stem IS its root family and retaining it
      verbatim is correct rather than a chain.

    Not interpreting the suffix at all is also the stronger reading of the "do
    not parse a generated suffix to recover a root family" stop condition.
    """
    from xrd_tools.reduction.reintegrate_successor import _persisted_root_family

    for stem in ("scan12", "series_rsm", "sample_average", "run_int1d"):
        artifact = tmp_path / f"{stem}.nexus"
        with h5py.File(artifact, "w") as document:
            document.create_group("entry")
        # No stamped family -> None, so the caller falls back to the stem.
        assert _persisted_root_family(artifact, "entry") is None
        # An explicit family still wins when the caller knows better.
        assert _persisted_root_family(artifact, "entry", "explicit") == "explicit"


def test_a_failure_after_the_rename_reports_published_not_aborted(
    tmp_path, monkeypatch,
) -> None:
    """Codex F3 (P2) on `e06d6123`: Average claimed a clean abort after publishing.

    One `except` covered both a failed rename and a failed read AFTER a
    successful rename, even though their visible effects are opposite. The
    second reported ABORTED with no commit identity -- so the operator was told
    there was no public effect while a NEW FILE sat at the slot.

    Now the two are separated. A post-rename failure reports
    PUBLISHED_UNVERIFIED and leaves the published file exactly where it is:
    restoring or deleting it would destroy the result the science produced.
    """
    import xrd_tools.reduction.average as module

    source = _series(tmp_path)
    target = tmp_path / "average.nxs"
    _stub_integrators(monkeypatch, [])
    first = _run_average_scan(AverageScanRecipe(
        source, target, ReductionPlan(integration_1d=Integration1DPlan(npt=4)),
    ))
    assert first.disposition == "COMMITTED"
    artifact = Path(first.target)
    prior_bytes = artifact.read_bytes()

    calls = []

    def failing_seal(path, ordinal):
        calls.append(path)
        raise OSError("injected post-rename seal failure")

    with monkeypatch.context() as fault:
        fault.setattr(module, "_seal_published", failing_seal)
        second = _run_average_scan(AverageScanRecipe(
            source, target,
            ReductionPlan(integration_1d=Integration1DPlan(npt=6)),
        ))

    # The injection actually fired -- this row proves nothing otherwise.
    assert len(calls) == 1

    assert second.disposition == "PUBLISHED_UNVERIFIED"
    assert second.diagnostic_code == "AVERAGE_PUBLISHED_UNVERIFIED"
    assert second.commit_identity is None and second.committed_labels == ()
    # NOT a clean abort: the replacement really landed and is still there.
    assert second.target == first.target
    assert artifact.exists() and artifact.read_bytes() != prior_bytes
    # The candidate was consumed by the rename, so nothing is stranded.
    assert not tuple(artifact.parent.glob(".xdart-finite-*"))


def test_a_failure_after_candidate_settlement_strands_nothing(
    tmp_path, monkeypatch,
) -> None:
    """A finished cold writer's post-write source sweep cleans its candidate."""
    source = _series(tmp_path)
    target = tmp_path / "average.nxs"
    _stub_integrators(monkeypatch, [])

    calls = []; sinks = []
    real_sweep = module.validate_source_state_sweep
    real_factory = module.NexusSink.for_finite_document

    def fail_after_finished(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            assert sinks and sinks[0]._writer.phase.value == "finished"
            raise OSError("injected failure after writer finished")
        return real_sweep(*args, **kwargs)

    def capture_sink(*args, **kwargs):
        value = real_factory(*args, **kwargs)
        sinks.append(value)
        return value

    with monkeypatch.context() as fault:
        fault.setattr(module, "validate_source_state_sweep", fail_after_finished)
        fault.setattr(module.NexusSink, "for_finite_document", capture_sink)
        result = _run_average_scan(AverageScanRecipe(
            source, target, ReductionPlan(integration_1d=Integration1DPlan(npt=4)),
        ))

    assert len(calls) == 2, "the injected failure never fired"
    assert result.disposition != "COMMITTED"
    # No slot was published, and NO private candidate remains.
    slot = Path(module._average_output_artifact(module._average_target(str(target))))
    assert not slot.exists()
    assert not tuple(slot.parent.glob(".xdart-finite-*"))


def test_average_refuses_a_slot_that_is_its_own_raw_input(tmp_path, monkeypatch):
    """The P1 data-loss refusal, pinned. Fable F3 on `55338dd9`.

    `AVERAGE_OUTPUT_IS_SOURCE` had ZERO test references: the defect was proved
    by a reviewer probe and by mine, and the fix shipped with nothing in the
    suite to hold it. A refactor of the plan path would have removed it silently
    and the reviews that found it would not run again.

    Codex's construction, exactly: a RAW NeXus stack sitting at the pathname the
    Average slot resolves to. No aliases, no concurrency. Before the fix this
    returned COMMITTED and overwrote the raw data.
    """
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.reduction.average import AverageScanRunner

    raw = tmp_path / "scan_average.nexus"          # IS the derived slot
    with h5py.File(raw, "w") as handle:
        handle.create_group("entry/instrument/detector").create_dataset(
            "data", data=np.arange(12, dtype="u2").reshape(3, 2, 2),
        )
    before = raw.read_bytes()
    _stub_integrators(monkeypatch, [])

    result = AverageScanRunner(AverageScanRecipe(
        SourceSpec(raw, SourceKind.NEXUS_STACK, entry="entry"),
        tmp_path / "scan.nexus",
        ReductionPlan(integration_1d=Integration1DPlan(npt=4)),
    )).start()

    assert result.disposition == "REFUSED"
    assert result.diagnostic_code == "AVERAGE_OUTPUT_IS_SOURCE"
    # Every source byte survives, and nothing was published or left behind.
    assert raw.read_bytes() == before
    assert not tuple(tmp_path.glob(".xdart-finite-*"))


def test_average_refuses_a_slot_that_is_its_own_active_mask(tmp_path, monkeypatch):
    """Codex R1 (P1) on `cca7d537`: the mask is READ, and it was not excluded.

    The first version of the input-collision guard covered the source graph
    only. I had checked `CalibrationState.source_uri` -- provenance-only -- and
    missed `CalibrationState.mask.source_uri`, which `load_mask` actually reads.
    So an Average whose derived slot resolved onto the operator's mask file
    OVERWROTE it: the same data-loss class as the raw-input case, one input
    short, and in a supported file format.

    The lesson is not "add the mask". It is that an inventory of admitted inputs
    is only as good as its completeness, and mine was assembled from the source
    graph because that is where I happened to be looking.
    """
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.reduction.average import AverageScanRunner

    source = _series(tmp_path)
    # The mask sits exactly where this Average's slot resolves.
    mask_path = tmp_path / "scan_average.nexus"
    static_mask = np.zeros((4, 4), dtype=bool)
    static_mask[0, 0] = True
    # Written directly: `np.save` would append `.npy` and defeat the collision.
    buffer = io.BytesIO()
    np.save(buffer, static_mask)
    mask_path.write_bytes(buffer.getvalue())
    before = mask_path.read_bytes()
    digest = hashlib.sha256(before).hexdigest()
    calibration = CalibrationState(mask=MaskState(
        str(mask_path), digest, np.dtype(bool).str, static_mask.shape,
        FactStatus.PRESENT,
    ))
    _stub_integrators(monkeypatch, [])

    result = AverageScanRunner(AverageScanRecipe(
        source, tmp_path / "scan.nexus",
        ReductionPlan(integration_1d=Integration1DPlan(npt=4)),
        calibration=calibration,
    )).start()

    assert result.disposition == "REFUSED"
    assert result.diagnostic_code == "AVERAGE_OUTPUT_IS_SOURCE"
    # The operator's mask survives byte for byte.
    assert mask_path.read_bytes() == before
