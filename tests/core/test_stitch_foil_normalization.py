"""Stitch's foil correction (AUD-9 stage 2b C2): decision 4's four choices on both backends (maintainer,
2026-10-01: "There are monitors both before and after the foil. So add the 4th choice").

- None, and Monitor (one after the foils, which already includes their attenuation), are what Stitch had.
- Foil transmission × exposure divides each frame by its foils' transmission and its counting time.
- Monitor × foil transmission divides by a monitor before the foils and the transmission it does not see.

Each frame's divisor is the shared ``frame_divisors`` (RSM's too; ``test_beam_normalization``). On each backend a
foil choice divides each frame exactly as a monitor column holding that divisor would, and frames scaled by
their divisors come out as the unscaled ones. A plan without a foil choice keeps its identity and provenance.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.analysis.module_transaction import MetadataColumnSelector, ModuleDisposition
from xrd_tools.analysis.stitch_operation import (
    StitchFoilNormalization,
    StitchOperationRefused,
    prepare_stitch_operation,
    run_stitch_operation,
)
from xrd_tools.core.scan import ScanFrame
from xrd_tools.sources import MemoryFrameSource
from tests.core.test_beam_normalization import foil_to_factor

LENGTHS = (1.06, 3.04, 4.65, 9.5)
CHOICES = ("none", "monitor", "foil-transmission-exposure", "monitor-foil-transmission")


def _transmission(codes) -> np.ndarray:
    """The notebook's ``foil_to_factor`` (bit for bit, ``test_beam_normalization``)."""
    return foil_to_factor(np.asarray(codes, dtype=np.float64), LENGTHS)


def _divisors(choice, monitor, foil, seconds) -> np.ndarray:
    monitor, seconds = np.asarray(monitor, dtype=np.float64), np.asarray(seconds, dtype=np.float64)
    return {"none": np.ones(len(monitor)), "monitor": monitor,
            "foil-transmission-exposure": _transmission(foil) * seconds,
            "monitor-foil-transmission": monitor * _transmission(foil)}[choice]


def _keys(choice) -> dict:
    """A choice as a backend's keywords."""
    return {"none": {}, "monitor": dict(monitor_key="I0"),
            "foil-transmission-exposure": dict(foil_key="foil status", exposure_key="Seconds",
                                               absorption_lengths=LENGTHS),
            "monitor-foil-transmission": dict(monitor_key="I0", foil_key="foil status",
                                              absorption_lengths=LENGTHS)}[choice]


MONITOR, FOIL, SECONDS = (3.0, 5.0, 11.0), (0.0, 1101.0, 20.0), (1.0, 2.0, 0.5)


def _metadata(index, choice, extra):
    return {"I0": MONITOR[index], "foil status": FOIL[index], "Seconds": SECONDS[index],
            "div": float(_divisors(choice, MONITOR, FOIL, SECONDS)[index]), **extra}


# --- pyFAI MultiGeometry (run_stitch) ---

def _pyfai_case(choice, *, frames, scale):
    from xrd_tools.analysis import StitchPlan, run_stitch
    from xrd_tools.core.containers import PONI
    shape = (195, 487)
    base_poni = PONI(dist=0.2, poni1=shape[0] * 172e-6 / 2.0, poni2=shape[1] * 172e-6 / 2.0,
                     rot1=0.0, rot2=0.0, rot3=0.0, wavelength=1.0e-10, detector="Pilatus100k")

    def ring(seed):
        rng = np.random.default_rng(seed)
        y, x = np.mgrid[:shape[0], :shape[1]]
        r = np.sqrt((y - shape[0] / 2.0) ** 2 + (x - shape[1] / 2.0) ** 2)
        return 500.0 * np.exp(-((r - 60.0) / 10.0) ** 2) + rng.poisson(3, size=shape)
    divisors = _divisors(choice, MONITOR, FOIL, SECONDS)
    source = MemoryFrameSource([
        ScanFrame(i, image=ring(i) * (divisors[i] if scale else 1.0),
                  metadata=_metadata(i, choice, {"rot1": float(5 * i)})) for i in range(frames)], name="foil")
    return lambda keys: run_stitch(StitchPlan(base_poni=base_poni, rot1_key="rot1", mode="1d", npt_1d=200,
                                              **keys), source)


@pytest.mark.parametrize("choice", CHOICES[2:])
def test_on_pyfai_a_foil_choice_divides_each_frame_as_a_monitor_holding_its_divisor(choice):
    run = _pyfai_case(choice, frames=3, scale=False)
    foil, column = run(_keys(choice)), run(dict(monitor_key="div"))
    np.testing.assert_array_equal(foil.payload.intensity, column.payload.intensity)
    np.testing.assert_array_equal(foil.payload.radial, column.payload.radial)


def test_a_stitch_plan_records_its_foil_choice_only_when_set_and_restores_it():
    from xrd_tools.analysis import StitchPlan
    plain = StitchPlan(base_poni=object(), monitor_key="I0")
    foil = StitchPlan(base_poni=object(), monitor_key="I0", foil_key="foil status", absorption_lengths=LENGTHS)
    assert "foil_key" not in plain.provenance() and plain.provenance()["monitor_key"] == "I0"
    record = foil.provenance()
    assert (record["foil_key"], record["exposure_key"], record["absorption_lengths"]) == (
        "foil status", None, list(LENGTHS))
    restored = StitchPlan.from_provenance(record, base_poni=object())
    assert (restored.foil_key, restored.exposure_key, restored.absorption_lengths) == ("foil status", None, LENGTHS)


@pytest.mark.parametrize("choice", CHOICES)
def test_on_pyfai_a_frame_scaled_by_its_divisor_comes_out_unscaled(choice):
    """One frame (MultiGeometry weights frames by their monitors, so several frames' merge is a weighted one;
    the per-frame divisor is pinned above by the monitor-column equivalence)."""
    scaled = _pyfai_case(choice, frames=1, scale=True)(_keys(choice))
    plain = _pyfai_case("none", frames=1, scale=False)({})
    # pyFAI's MultiGeometry result is float32: equal to its precision
    np.testing.assert_allclose(scaled.payload.intensity, plain.payload.intensity, rtol=4e-7, equal_nan=True)


# --- xrayutilities histogram (run_xu_hist_stitch_1d) ---

def _xu_case(tmp_path, choice, *, scale):
    from xrd_tools.integrate.xu_stitch import run_xu_hist_stitch_1d
    from tests.core.test_xu_stitch import _receipt
    receipt = _receipt(tmp_path)
    divisors = _divisors(choice, MONITOR, FOIL, SECONDS)
    rng = np.random.default_rng(5)
    frames = [ScanFrame(i, image=(rng.uniform(10, 50, (195, 1475)) * (divisors[i] if scale else 1.0)),
                        metadata=_metadata(i, choice, {"del": d, "nu": n}))
              for i, (d, n) in enumerate(((14.0, -10.0), (25.0, 20.0), (18.0, 5.0)))]
    source = MemoryFrameSource(frames)

    def run(keys):
        keys = dict(keys)
        monitor = keys.pop("monitor_key", None)
        return run_xu_hist_stitch_1d(receipt, source, frame_indices=(0, 1, 2), npt=48, monitor_key=monitor,
                                     max_frame_bytes=16 * 1024 * 1024, **keys)
    return run


@pytest.mark.parametrize("choice", CHOICES[2:])
def test_on_xrayutilities_a_foil_choice_divides_each_frame_as_a_monitor_holding_its_divisor(tmp_path, choice):
    run = _xu_case(tmp_path, choice, scale=False)
    foil, column = run(_keys(choice)), run(dict(monitor_key="div"))
    np.testing.assert_array_equal(foil.payload.intensity, column.payload.intensity)
    np.testing.assert_array_equal(foil.diagnostics.normalization, column.diagnostics.normalization)


@pytest.mark.parametrize("choice", CHOICES)
def test_on_xrayutilities_frames_scaled_by_their_divisors_come_out_unscaled(tmp_path, choice):
    """xrayutilities divides each frame by its divisor before binning, so three frames scaled by theirs merge
    as the unscaled ones."""
    scaled = _xu_case(tmp_path / "a", choice, scale=True)(_keys(choice))
    plain = _xu_case(tmp_path / "b", "none", scale=False)({})
    np.testing.assert_allclose(scaled.payload.intensity, plain.payload.intensity, rtol=1e-12, equal_nan=True)


def test_an_xrayutilities_frame_whose_foil_status_is_not_a_whole_code_is_refused(tmp_path):
    from xrd_tools.integrate.xu_stitch import XuStitchScienceRefused
    run = _xu_case(tmp_path, "foil-transmission-exposure", scale=False)
    with pytest.raises(XuStitchScienceRefused) as refused:
        run(dict(foil_key="div", exposure_key="Seconds", absorption_lengths=LENGTHS))  # frame 1's is 0.1…
    assert refused.value.code == "FOIL_STATUS_INVALID"


@pytest.mark.parametrize("keys, code", [
    (dict(foil_key="absent", exposure_key="Seconds"), "FOIL_STATUS_INVALID"),
    (dict(foil_key="foil status", exposure_key="absent"), "EXPOSURE_INVALID"),
])
def test_an_xrayutilities_frame_missing_a_foil_column_is_refused_for_that_column(tmp_path, keys, code):
    from xrd_tools.integrate.xu_stitch import XuStitchScienceRefused
    run = _xu_case(tmp_path, "foil-transmission-exposure", scale=False)
    with pytest.raises(XuStitchScienceRefused) as refused:
        run(dict(keys, absorption_lengths=LENGTHS))
    assert refused.value.code == code


# --- the operation: plan, identity, provenance, preparation ---

def _foil_form(tmp_path, backend, *, name="sample_stitch1d.nexus"):
    """The native Stitch fixture with a monitor before the foils (I0), the foil status, the counting time and
    each choice's divisor as columns."""
    from xrd_tools.analysis.stitch_values import prepare_stitch_tool
    from tests.core.test_stitch_operation import _native_form
    tmp_path.mkdir(parents=True, exist_ok=True)
    form = _native_form(tmp_path, backend, scans="1")
    spec = Path(form.spec_path)
    lines, out, row = spec.read_text().splitlines(), [], 0
    for line in lines:
        if line.startswith("#L "):
            out.append(line + "  I0  foil status  Seconds  divFE  divMF")
            row = 0
        elif line and (line[0].isdigit() or line[0] == "-"):
            fe = _divisors("foil-transmission-exposure", MONITOR, FOIL, SECONDS)[row]
            mf = _divisors("monitor-foil-transmission", MONITOR, FOIL, SECONDS)[row]
            out.append(f"{line} {MONITOR[row]!r} {FOIL[row]!r} {SECONDS[row]!r} {float(fe)!r} {float(mf)!r}")
            row += 1
        elif line.startswith("#N "):
            out.append(f"#N {int(line.split()[1]) + 5}")
        else:
            out.append(line)
    spec.write_text("\n".join(out) + "\n")
    return prepare_stitch_tool(replace(form, output_path=Path(form.output_path).with_name(name))).request


def _with(request, **changes):
    """The request's plan with ``changes``, its source bound again with the columns the new plan reads (the
    page's Source rows bind them in C6)."""
    from xrd_tools.analysis.module_transaction import ModuleKind, ModuleSourceReceipt
    from xrd_tools.analysis.scan_operations import MetadataTablePlan, run_metadata_table
    plan = replace(request.plan, **changes)
    source = request.module.source
    table = run_metadata_table(MetadataTablePlan(source.analysis.source_spec))
    added = (plan.monitor_selector,) + (() if plan.foil_normalization is None else (
        plan.foil_normalization.foil_selector, plan.foil_normalization.exposure_selector))
    selectors = tuple(sorted({(s.name, s.occurrence): s for s in source.resolved_selectors + tuple(
        selector for selector in added if selector is not None)}.values(), key=lambda s: (s.name, s.occurrence)))
    source = ModuleSourceReceipt.from_metadata_table(table, kind=ModuleKind.STITCH,
                                                     selected_labels=source.selected_labels,
                                                     resolved_selectors=selectors, fresh=True)
    return prepare_stitch_operation(source, request.module.output, plan, project_root=request.manifest.project_root)


def _plan(request) -> dict:
    return json.loads(request.provenance_json)["plan"]


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
def test_a_foil_choice_enters_identity_and_provenance_only_when_set(tmp_path, backend):
    base = _foil_form(tmp_path, backend)
    assert base.plan.foil_normalization is None and "foil_normalization" not in _plan(base)
    exposure = StitchFoilNormalization(MetadataColumnSelector("foil status"), LENGTHS,
                                       MetadataColumnSelector("Seconds"))
    before = StitchFoilNormalization(MetadataColumnSelector("foil status"), LENGTHS)
    foil = _with(base, foil_normalization=exposure)
    monitored = _with(base, monitor_selector=MetadataColumnSelector("I0"), foil_normalization=before)
    assert len({base.plan.fingerprint, foil.plan.fingerprint, monitored.plan.fingerprint}) == 3
    assert _plan(foil)["foil_normalization"] == {
        "mode": "foil-transmission-exposure", "foil_selector": {"name": "foil status", "occurrence": 0},
        "exposure_selector": {"name": "Seconds", "occurrence": 0}, "absorption_lengths": list(LENGTHS)}
    assert _plan(monitored)["foil_normalization"]["mode"] == "monitor-foil-transmission"
    assert _with(base).provenance_json == base.provenance_json  # unchanged without one
    with pytest.raises(ValueError, match="never|with no monitor"):
        replace(base.plan, monitor_selector=MetadataColumnSelector("I0"), foil_normalization=exposure)
    with pytest.raises(ValueError, match="with no monitor"):
        replace(base.plan, foil_normalization=before)
    with pytest.raises(ValueError, match="not all zero"):
        StitchFoilNormalization(MetadataColumnSelector("foil status"), (0.0, 0.0, 0.0, 0.0))


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
def test_a_bad_foil_column_is_refused_at_preparation_naming_it(tmp_path, backend):
    base = _foil_form(tmp_path, backend)
    wrong = StitchFoilNormalization(MetadataColumnSelector("divFE"), LENGTHS, MetadataColumnSelector("Seconds"))
    with pytest.raises(StitchOperationRefused) as refused:
        _with(base, foil_normalization=wrong)  # divFE holds non-integral values
    assert refused.value.code == "FOIL_STATUS_INVALID" and "'divFE'" in str(refused.value)


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
@pytest.mark.parametrize("choice", CHOICES[2:])
def test_a_foil_choice_runs_as_its_divisor_column_through_the_operation(tmp_path, backend, choice):
    """End to end on both backends (the xrayutilities record's intent validator admits the foil record): the
    committed I(q) equals the run's with a monitor column holding the same divisors."""
    from xrd_tools.io.analysis_artifact import read_analysis_artifact
    foil_request = _foil_form(tmp_path / "foil", backend)
    column_request = _foil_form(tmp_path / "column", backend)
    exposure = choice == "foil-transmission-exposure"
    foil = _with(foil_request, monitor_selector=None if exposure else MetadataColumnSelector("I0"),
                 foil_normalization=StitchFoilNormalization(
                     MetadataColumnSelector("foil status"), LENGTHS,
                     MetadataColumnSelector("Seconds") if exposure else None))
    column = _with(column_request, monitor_selector=MetadataColumnSelector("divFE" if exposure else "divMF"))
    results = []
    for request in (foil, column):
        result = run_stitch_operation(request)
        assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
        results.append(read_analysis_artifact(Path(request.module.output.target)))
    for name in ("intensity", "coverage", "normalization"):
        np.testing.assert_array_equal(getattr(results[0], name), getattr(results[1], name))
    assert json.loads(results[0].provenance_json)["plan"]["foil_normalization"]["mode"] == choice


def test_the_xrayutilities_intent_validator_admits_only_a_well_formed_foil_record(tmp_path):
    from xrd_tools.analysis.module_transaction import _validate_xu_stitch_v2_intent
    base = _foil_form(tmp_path, "xu_hist")
    request = _with(base, foil_normalization=StitchFoilNormalization(
        MetadataColumnSelector("foil status"), LENGTHS, MetadataColumnSelector("Seconds")))
    provenance = json.loads(request.provenance_json)
    malformed = json.loads(request.provenance_json)
    malformed["plan"]["foil_normalization"]["mode"] = "foil-alone"
    with pytest.raises(Exception):
        _validate_xu_stitch_v2_intent(request.module, malformed)
    assert provenance["plan"]["foil_normalization"]["mode"] == "foil-transmission-exposure"
