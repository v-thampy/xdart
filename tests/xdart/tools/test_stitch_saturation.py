"""Decision 39 in Stitch: each scan's automatic ceiling (its recorded value, else the calibration
detector's documented counter limit, else its data type's maximum) masks its frames, is recorded
with the output, decides whether Append redoes an output, and is the page's hint, never its value."""
from dataclasses import replace
import itertools
from pathlib import Path

import numpy as np
import pytest
import tifffile

from xdart.gui.pages.services import empty_host_services
from xdart.gui.tabs.stitch.page import StitchPage
from xdart.gui.tools.stitch_values import prepare_stitch_tool
from xrd_tools.analysis.module_transaction import (
    ModuleDisposition, ModuleKind, ModuleOutputRequest, ModuleSourceGroupReceipt,
)
from xrd_tools.analysis.stitch_native import label_saturation
from xrd_tools.analysis.stitch_operation import (
    StitchOperationRefused, prepare_stitch_operation, run_stitch_operation,
)
from xrd_tools.core import invalid
from xrd_tools.integrate.stitch_conditioning import StitchConditioning, condition_stitch_frame
from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, read_analysis_artifact
from xrd_tools.io.output_path import resolve_finite_output_target
from tests.xdart.pages.test_stitch_page import _Status, _dispose, _wait, qapp  # noqa: F401
from tests.xdart.tools.test_stitch_native_values import _native_form


def test_the_conditioning_records_a_saturation_only_when_resolved():
    assert StitchConditioning("auto").to_dict() == {"mode": "auto", "threshold_min": None, "threshold_max": None}
    resolved = StitchConditioning("auto", saturation=(1048575, None))
    assert resolved.to_dict()["saturation"] == [1048575, None]
    assert StitchConditioning(**resolved.to_dict()) == resolved  # a record's JSON list reads back
    for bad in ((None,), (), (0,), (1.5,), (np.int64(5),)):
        with pytest.raises(ValueError):
            StitchConditioning("auto", saturation=bad)
    with pytest.raises(ValueError):
        StitchConditioning("manual", None, 5., saturation=(4,))
    raw = np.array([[10, 1048574, 1048575, 2**31 - 1]], dtype=np.int32)
    image = condition_stitch_frame(raw, StitchConditioning("auto"), 1048575)
    assert np.isfinite(image).tolist() == [[True, True, False, False]]
    assert np.isfinite(condition_stitch_frame(raw, StitchConditioning("auto"))).tolist() == [[True, True, True, False]]


def _saturated(form, value=2_000_000):
    """The first frame of scan 1 with a block of counts at or above the PILATUS counter limit."""
    image = next(Path(form.image_dir).glob("sample_scan1_0000*"))
    frame = np.fromfile(image, dtype=np.int32)
    frame[1000:1400] = value
    frame.tofile(image)


def _science(form, conditioning):
    prepared = prepare_stitch_tool(replace(form, conditioning=conditioning))
    request, = prepared.requests
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    payload = read_analysis_artifact(request.module.output.target)
    return request.provenance["plan"]["conditioning"], payload.intensity


@pytest.mark.parametrize("backend", ["multigeometry", "xu_hist"])
def test_auto_leaves_out_counts_at_the_pilatus_counter_limit(tmp_path, backend):
    form = _native_form(tmp_path, backend, scans="1")  # the form replaces its output on each run
    _saturated(form)
    recorded, auto = _science(form, StitchConditioning("auto"))
    assert recorded == {"mode": "auto", "threshold_min": None, "threshold_max": None, "saturation": [1048575]}
    # Auto is the same science as keeping 1048574 and below (a manual maximum keeps its own value)
    manual = _science(form, StitchConditioning("manual", None, 1048574.))[1]
    np.testing.assert_array_equal(auto, manual)
    off = _science(form, StitchConditioning("off"))[1]
    assert not np.array_equal(np.nan_to_num(off), np.nan_to_num(auto))


def test_append_redoes_an_output_made_under_the_data_type_rule(tmp_path, monkeypatch):
    form = _native_form(tmp_path, "multigeometry", scans="1")
    monkeypatch.setattr(invalid, "documented_saturation", lambda _detector: None)  # the rule before
    old, = prepare_stitch_tool(form).requests
    assert "saturation" not in old.provenance["plan"]["conditioning"]
    assert run_stitch_operation(old).terminal.disposition is ModuleDisposition.COMMITTED
    assert prepare_stitch_tool(form).unchanged == (0,)
    monkeypatch.undo()
    fresh = prepare_stitch_tool(form)
    assert fresh.requests[0].provenance["plan"]["conditioning"]["saturation"] == [1048575]
    assert fresh.unchanged == ()  # made under another ceiling: Append redoes it
    assert run_stitch_operation(fresh.requests[0]).terminal.disposition is ModuleDisposition.COMMITTED
    assert prepare_stitch_tool(form).unchanged == (0,)


def _with_cutoffs(monkeypatch, cutoffs):
    """TIFF members written with PILATUS headers recording these Count_cutoff values in turn (the
    last one for any later file)."""
    original, values = tifffile.imwrite, itertools.chain(cutoffs, itertools.repeat(cutoffs[-1]))

    def write(path, data, **options):
        if options.get("description") is None:
            options["description"] = f"# Count_cutoff {next(values)} counts\r\n"
        return original(path, data, **options)
    monkeypatch.setattr(tifffile, "imwrite", write)


def _group(tmp_path, monkeypatch, first, second):
    from tests.core.test_stitch_operation import _prepared
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    _with_cutoffs(monkeypatch, first)
    one = _prepared(tmp_path / "one")
    _with_cutoffs(monkeypatch, second)
    two = _prepared(tmp_path / "two")
    monkeypatch.undo()
    source = ModuleSourceGroupReceipt.from_members((one.module.source, two.module.source), kind=ModuleKind.STITCH)
    plan = replace(one.plan, native_records=True, radial_range=None, conditioning=StitchConditioning("auto"))
    target = resolve_finite_output_target(tmp_path, "group", operation_token="stitch-1d")
    output = ModuleOutputRequest(target, AnalysisArtifactKind.STITCH_1D, artifact_family="group")
    return prepare_stitch_operation(source, output, plan, project_root=tmp_path)


def test_each_scan_is_masked_with_the_value_its_files_record(tmp_path, monkeypatch):
    request = _group(tmp_path, monkeypatch, [400] * 3, [450] * 3)
    assert request.provenance["plan"]["conditioning"]["saturation"] == [400, 450]
    labels = request.module.source.selected_labels
    assert label_saturation(request) == dict(zip(labels, [400, 400, 450, 450], strict=True))
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal


def test_the_files_of_one_scan_must_record_the_same_value(tmp_path, monkeypatch):
    with pytest.raises(StitchOperationRefused) as refused:
        _group(tmp_path, monkeypatch, [400, 400, 410], [450] * 3)  # frames 1 and 3 are selected
    assert refused.value.code == "STITCH_SATURATION_DIFFERS"


def test_the_page_shows_the_ceiling_as_a_hint_and_a_typed_number_is_manual(qapp, tmp_path):  # noqa: F811
    form = _native_form(tmp_path, "multigeometry", scans="1")
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(form)
        for _ in range(2):  # a second check keeps the field empty and the form unchanged
            page.begin_preview()
            _wait(page, qapp)
            assert page._owner.prepared is not None, page.run_bar.readinessLabel.full_text()
            assert page._value("threshold") is None and page.build_form().threshold is None
            assert page.threshold_range._high.placeholderText() == "Auto · 1048575 (saturation)"
            assert page.build_form().conditioning == StitchConditioning("auto")
        editor = page.threshold_range._high
        editor.setText("1048575")
        editor.setModified(True)
        editor.editingFinished.emit()
        assert page.build_form().conditioning == StitchConditioning("manual", 0.0, 1048575.0)
        page._show_auto_ceiling({(400, "saturation"), (450, "saturation")})
        assert editor.placeholderText() == "Auto · per scan" and page._auto_threshold_max is None
        page._show_auto_ceiling({(65535, "data type")})
        assert editor.placeholderText() == "Auto · 65535 (data type maximum)"
        assert page._auto_threshold_max == 65534.0
    finally:
        _dispose(page, qapp)
