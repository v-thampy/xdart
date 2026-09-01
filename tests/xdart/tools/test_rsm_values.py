from __future__ import annotations

from dataclasses import fields, replace
import json
from pathlib import Path
from threading import Event

import numpy as np
import pytest

import xdart.gui.tools.rsm_values as rsm_values

from xdart.gui.tools.rsm_values import (
    RSMFrameSelector,
    RSMScanPreset,
    RSMToolForm,
    RSMToolPreflight,
    RSMToolPreflightRefused,
    RSMToolPreflightSummary,
    prepare_rsm_tool,
    rsm_tool_preset,
)
from xrd_tools.analysis.module_transaction import MetadataColumnSelector
from xrd_tools.analysis.rsm_operation import (
    RSMNormalizationMode,
    RSMNormalizationPolicy,
    RSMOperationPlan,
    rsm_normalization_divisors,
)
from xrd_tools.sources.spec import SpecSource


def test_scan43_preset_is_pure_and_locks_raw_and_selector_defaults(tmp_path):
    preset = rsm_tool_preset(RSMScanPreset.STO_ALIGN_SCAN43)
    assert preset.spec_relative_path == "STO_align"
    assert preset.scan == "43.1"
    assert preset.frame_selector == RSMFrameSelector(0, 60, 1)
    assert preset.detector_shape == (195, 487)
    assert preset.raw_dtype == "int32"
    assert preset.raw_header_skip == 0
    assert preset.normalization.exposure_selector == MetadataColumnSelector(
        "Seconds", 0
    )
    assert preset.quick_bins == (40, 40, 40)
    assert preset.full_bins == (200, 200, 200)
    assert preset.plan().bins == preset.quick_bins
    assert preset.plan(preset.full_bins).bins == preset.full_bins
    with pytest.raises(ValueError, match="quick or full"):
        preset.plan((41, 41, 41))

    form = preset.form(
        tmp_path / "not-probed",
        tmp_path / "not-probed" / "result.nexus",
    )
    assert form.spec_path.endswith("/not-probed/STO_align")
    assert form.image_dir.endswith("/not-probed/images")


def test_form_fingerprint_binds_exact_occurrence_and_grid(rsm_tool_form):
    original = rsm_tool_form
    normalization = RSMNormalizationPolicy(
        RSMNormalizationMode.FOIL_TRANSMISSION_EXPOSURE,
        original.plan.normalization.foil_selector,
        MetadataColumnSelector("Seconds", 0),
        original.plan.normalization.absorption_lengths,
    )
    changed_plan = RSMOperationPlan(
        original.plan.geometry,
        original.plan.conditioning,
        normalization,
        bins=original.plan.bins,
        chunk_size=original.plan.chunk_size,
        max_frame_bytes=original.plan.max_frame_bytes,
        max_chunk_bytes=original.plan.max_chunk_bytes,
    )
    changed = replace(original, plan=changed_plan)
    assert changed.fingerprint != original.fingerprint

    with pytest.raises(ValueError, match="match the RSM detector header"):
        replace(original, detector_shape=(5, 5))


def test_frame_selector_is_exact_label_membership():
    assert RSMFrameSelector(2, 6, 2).select((0, 2, 4, 6, 8)) == (2, 4, 6)
    with pytest.raises(RSMToolPreflightRefused) as refused:
        RSMFrameSelector(2, 8, 2).select((0, 2, 4, 8))
    assert refused.value.code == "FRAME_SELECTION_OUTSIDE_SOURCE"


def test_preflight_is_image_free_and_occurrence_aware(
    rsm_tool_form,
    monkeypatch,
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Preview must not decode detector images")

    monkeypatch.setattr(SpecSource, "load_frame", forbidden)
    preflight = prepare_rsm_tool(rsm_tool_form)
    summary = preflight.summary
    assert summary.form_fingerprint == rsm_tool_form.fingerprint
    assert summary.selected_labels == (0, 2)
    assert len(summary.members) == 2
    assert MetadataColumnSelector("Seconds", 1) in summary.required_selectors
    assert MetadataColumnSelector("Seconds", 0) not in summary.required_selectors
    assert all(
        any(name == "Seconds" and occurrence == 1 for name, occurrence, _ in item.values)
        for item in summary.members
    )
    expected = rsm_normalization_divisors(
        rsm_tool_form.plan.normalization,
        2,
        foil_status=np.asarray([101.0, 1010.0]),
        exposure_seconds=np.asarray([10.0, 40.0]),
    )
    np.testing.assert_allclose(
        [item.normalization_divisor for item in summary.members],
        expected,
        rtol=0,
        atol=0,
    )
    assert summary.energy_eV == 13000.007
    assert summary.ub == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert summary.detector_shape == (4, 5)
    assert summary.cropped_shape == (3, 4)
    assert summary.holds == (
        "gi-and-refraction-corrections",
        "hostile-shared-project-output-namespace",
        "multi-scan-rsm",
        "volume-rendering",
    )
    options = json.loads(preflight.request.preflight.source_options_json)
    assert options["read_image_kwargs"]["threshold"] is None
    assert options["read_image_kwargs"]["rotation"] == 0
    assert ["Seconds", 1] in options["metadata_column_projection"]
    assert ["Seconds", 0] not in options["metadata_column_projection"]


def test_preflight_enforces_project_output_and_cancellation(rsm_tool_form):
    outside = replace(
        rsm_tool_form,
        output_path=Path(rsm_tool_form.project_root).parent / "outside.nexus",
    )
    with pytest.raises(RSMToolPreflightRefused) as refused:
        prepare_rsm_tool(outside)
    assert refused.value.code == "OUTPUT_OUTSIDE_PROJECT"

    cancel = Event()
    cancel.set()
    with pytest.raises(RSMToolPreflightRefused) as refused:
        prepare_rsm_tool(rsm_tool_form, cancel_token=cancel)
    assert refused.value.code == "CANCELLED"


def test_outside_project_spec_is_refused_before_metadata_io(
    rsm_tool_form,
    tmp_path,
    monkeypatch,
):
    outside = tmp_path.parent / f"{tmp_path.name}-outside-spec"
    outside.write_text("outside must never be opened", encoding="utf-8")
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("outside SPEC must be refused before metadata I/O")

    monkeypatch.setattr(rsm_values, "run_metadata_table", forbidden)
    form = replace(rsm_tool_form, spec_path=outside)

    with pytest.raises(RSMToolPreflightRefused) as refused:
        prepare_rsm_tool(form)

    assert refused.value.code == "SPEC_OUTSIDE_PROJECT"
    assert calls == []


def test_preflight_summary_cannot_disagree_with_exact_request(prepared_rsm_tool):
    summary = prepared_rsm_tool.summary
    kwargs = {
        item.name: getattr(summary, item.name)
        for item in fields(RSMToolPreflightSummary)
        if item.init
    }
    kwargs["q_bounds"] = (
        (summary.q_bounds[0][0], summary.q_bounds[0][1] + 1.0e-6),
        summary.q_bounds[1],
        summary.q_bounds[2],
    )
    forged = RSMToolPreflightSummary(
        **kwargs,
        _claim=rsm_values._PREFLIGHT_FACTORY,
    )

    with pytest.raises(TypeError, match="facts disagree"):
        RSMToolPreflight(
            prepared_rsm_tool.form,
            prepared_rsm_tool.request,
            forged,
            rsm_values._PREFLIGHT_FACTORY,
        )


def test_preflight_form_intent_cannot_disagree_with_exact_request(
    prepared_rsm_tool,
):
    preflight = prepared_rsm_tool
    other_form = replace(
        preflight.form,
        output_path=Path(preflight.form.output_path).with_name("other.nexus"),
    )
    kwargs = {
        item.name: getattr(preflight.summary, item.name)
        for item in fields(RSMToolPreflightSummary)
        if item.init
    }
    kwargs["form_fingerprint"] = other_form.fingerprint
    forged_summary = RSMToolPreflightSummary(
        **kwargs,
        _claim=rsm_values._PREFLIGHT_FACTORY,
    )

    with pytest.raises(TypeError, match="facts disagree"):
        RSMToolPreflight(
            other_form,
            preflight.request,
            forged_summary,
            rsm_values._PREFLIGHT_FACTORY,
        )


def test_identity_normalization_is_a_valid_tool_preflight(rsm_tool_form):
    original = rsm_tool_form.plan
    identity_plan = RSMOperationPlan(
        original.geometry,
        original.conditioning,
        RSMNormalizationPolicy.identity(),
        bins=original.bins,
        chunk_size=original.chunk_size,
        max_frame_bytes=original.max_frame_bytes,
        max_chunk_bytes=original.max_chunk_bytes,
    )
    preflight = prepare_rsm_tool(replace(rsm_tool_form, plan=identity_plan))

    assert preflight.summary.normalization_mode is RSMNormalizationMode.IDENTITY
    assert preflight.summary.absorption_lengths == (0.0, 0.0, 0.0, 0.0)
    assert all(
        member.normalization_divisor == 1.0
        for member in preflight.summary.members
    )


def test_summary_foreign_member_is_rejected_as_type_error(prepared_rsm_tool):
    summary = prepared_rsm_tool.summary
    kwargs = {
        item.name: getattr(summary, item.name)
        for item in fields(RSMToolPreflightSummary)
        if item.init
    }
    kwargs["members"] = (object(),)

    with pytest.raises(TypeError, match="summary is invalid"):
        RSMToolPreflightSummary(
            **kwargs,
            _claim=rsm_values._PREFLIGHT_FACTORY,
        )


def test_form_rejects_extension_and_unsafe_decoder(rsm_tool_form):
    with pytest.raises(ValueError, match="extensionless"):
        replace(rsm_tool_form, spec_path=rsm_tool_form.spec_path + ".spec")
    with pytest.raises(ValueError, match="scalar real numeric"):
        replace(rsm_tool_form, raw_dtype="complex64")
