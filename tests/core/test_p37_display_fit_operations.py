"""P3-7A headless single-displayed-trace Peak and Phase contracts.

Only optional fitting/CIF backends are faked.  The public runners retain
ownership of trace custody, validation, q guarding, policy, bounded CIF reads,
detachment, resource charging, and terminal truth.
"""

from __future__ import annotations

import builtins
import dataclasses
import hashlib
import importlib
import inspect
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import xrd_tools.analysis as analysis
from xrd_tools.analysis.plans import AnalysisResult, PeakFitPlan


_RUNNERS = {
    "run_metadata_table",
    "run_scan_plot",
    "run_roi_preview",
    "run_roi_scan",
    "run_displayed_peak_fit",
    "run_displayed_phase_fit",
}


class _StringAlias(str):
    pass


def _public(name: str):
    value = getattr(analysis, name, None)
    assert value is not None, f"missing frozen P3-7A public value {name}"
    return value


def _fit_ops():
    return importlib.import_module("xrd_tools.analysis.display_fit_operations")


def _word(value: object) -> str:
    return str(getattr(value, "value", value)).upper()


def _assert_terminal(result, disposition: str, code: str | None = None):
    assert _word(result.disposition) == disposition
    if code is not None:
        assert result.code == code
    return result


def _assert_owned_readonly(array: np.ndarray) -> None:
    assert isinstance(array, np.ndarray)
    assert array.dtype.kind != "O"
    assert array.flags.c_contiguous and array.flags.owndata
    assert not array.flags.writeable


def _trace(
    *,
    axis=None,
    intensity=None,
    axis_unit: str = "1/angstrom",
    label: str = "frame-7",
):
    axis = np.asarray(np.linspace(1.0, 4.0, 31) if axis is None else axis)
    intensity = np.asarray(
        2.0 + 20.0 * np.exp(-((axis - 2.2) / 0.12) ** 2)
        if intensity is None else intensity
    )
    receipt = _public("DisplayedTraceReceipt")(
        schema_version="displayed-trace-v1",
        run_generation=4,
        run_fingerprint="run-fingerprint",
        source_scan="7.1",
        artifact=Path("/data/result.nxs"),
        frame_label=label,
        work_ordinal=9,
    )
    return _public("DisplayedTraceInput")(
        axis=axis,
        intensity=intensity,
        label=label,
        axis_unit=axis_unit,
        title="retained displayed trace",
        epoch=12,
        receipt=receipt,
    )


def _peak_plan(trace, **changes):
    values = dict(
        trace=trace,
        fit_bounds=None,
        selection_mode="count",
        manual_centers=(),
        n_peaks=1,
        model="pseudovoigt",
        background="linear",
        sigma_init=None,
        sigma_bounds=None,
        amplitude_init=None,
        amplitude_bounds=None,
        center_bounds_delta=None,
        fraction_init=0.5,
        max_nfev=None,
    )
    values.update(changes)
    return _public("DisplayedPeakFitPlan")(**values)


def _phase_plan(trace, cif_paths, **changes):
    paths = tuple(Path(path) for path in cif_paths)
    values = dict(
        trace=trace,
        cif_paths=paths,
        expected_hashes=tuple(None for _ in paths),
        phase_names=tuple(path.stem for path in paths),
        wavelength_angstrom=0.6199,
        prefit_background="none",
        phase_profile="pseudovoigt",
        lattice_pct=0.05,
        min_intensity=0.0,
        max_nfev=None,
    )
    values.update(changes)
    return _public("DisplayedPhaseFitPlan")(**values)


def _peak_payload(
    x, y, *, name: str = "p0_center", success: bool = True,
    message: str | None = None,
):
    params = {
        name: SimpleNamespace(value=2.2, stderr=0.01),
        "p0_sigma": SimpleNamespace(value=0.12, stderr=0.01),
    }
    fit_result = SimpleNamespace(
        success=success,
        message=("converged" if success else "not converged") if message is None else message,
        params=params,
        best_fit=np.asarray(y, dtype=float) + 0.25,
        residual=np.asarray(y, dtype=float) - (np.asarray(y, dtype=float) + 0.25),
        eval_components=lambda x=None: {
            "background": np.full_like(np.asarray(y, dtype=float), 2.0)
        },
    )
    return SimpleNamespace(
        fit_result=fit_result,
        success=success,
        peak_centers=[2.2],
        peak_centers_err=[0.01],
        peak_sigmas=[0.12],
        peak_amplitudes=[20.0],
        n_peaks=1,
        model_name="pseudovoigt",
        background_name="linear",
        best_fit=fit_result.best_fit,
        params=params,
    )


def _install_peak_backend(
    monkeypatch, calls, *, parameter_name="p0_center", success=True,
    message=None, payloads=None,
):
    ops = _fit_ops()

    def run(plan, x, y):
        calls.append((plan, np.array(x, copy=True), np.array(y, copy=True)))
        payload = _peak_payload(
            x, y, name=parameter_name, success=success, message=message,
        )
        if payloads is not None:
            payloads.append(payload)
        return AnalysisResult(
            kind="peak_fit",
            payload=payload,
        )

    monkeypatch.setattr(ops, "_load_peak_backend", lambda: (PeakFitPlan, run))
    return ops


class _Structure:
    calls: list[tuple[str, str]] = []
    structures: list[object] = []

    @classmethod
    def from_str(cls, text: str, *, fmt: str):
        cls.calls.append((text, fmt))
        structure = SimpleNamespace(source_text=text)
        cls.structures.append(structure)
        return structure


class _PhaseModel:
    peak_count = 2
    peak_counts = None
    instances: list["_PhaseModel"] = []

    def __init__(self, name):
        self.name = name
        self.structure = None
        self.peaks = []
        self.calculate_calls = []
        type(self).instances.append(self)

    def calculate_peaks(self, *, wavelength):
        assert self.structure is not None
        self.calculate_calls.append(wavelength)
        count = (
            type(self).peak_count
            if type(self).peak_counts is None
            else type(self).peak_counts[self.name]
        )
        self.peaks = [
            SimpleNamespace(q=1.0 + index * 0.001, intensity=100.0, hkl=(1, 0, 0))
            for index in range(count)
        ]


class _PhaseFitter:
    parameter_count = 3
    parameter_names_override = None
    fit_success = True
    message_override = None
    fraction_override = None
    lattice_override = None
    summary_forbidden = False
    instances: list["_PhaseFitter"] = []

    def __init__(self, q, intensity, **options):
        self.q = np.array(q, copy=True)
        self.intensity = np.array(intensity, copy=True)
        self.options = dict(options)
        self.phases = []
        self.add_calls = 0
        self.fit_calls = 0
        self.background = np.zeros_like(self.intensity)
        self.last_params = None
        self.last_result = None
        type(self).instances.append(self)

    def add_phase(self, phase, *, q_range=None, min_intensity=0.5):
        self.add_calls += 1
        self.phases.append((phase, {"q_range": q_range, "min_intensity": min_intensity}))

    def build_parameters(self, **_kwargs):
        names = type(self).parameter_names_override
        if names is None:
            names = tuple(f"p{index}" for index in range(type(self).parameter_count))
        return {
            name: SimpleNamespace(value=float(index), stderr=0.1)
            for index, name in enumerate(names)
        }

    def eval_model(self, params):
        assert params is not None
        return self.intensity + 0.1

    def eval_phase(self, index, params):
        assert params is not None
        return np.full_like(self.intensity, 1.0 / max(len(self.phases), 1))

    def fit(self, params=None, **kwargs):
        self.fit_calls += 1
        params = self.build_parameters() if params is None else params
        self.last_params = params
        fractions = (
            {phase.name: 1.0 / len(self.phases) for phase, _ in self.phases}
            if type(self).fraction_override is None
            else dict(type(self).fraction_override)
        )

        def phase_fractions():
            if type(self).summary_forbidden:
                pytest.fail("phase fractions accessed after byte impossibility")
            return dict(fractions)

        def lattice_params(index):
            if type(self).summary_forbidden:
                pytest.fail("lattice result accessed after byte impossibility")
            if type(self).lattice_override is not None:
                return dict(type(self).lattice_override)
            return {"a": 5.0 + index}

        self.last_result = SimpleNamespace(
            params=params,
            success=type(self).fit_success,
            lmfit_result=SimpleNamespace(
                success=type(self).fit_success,
                message=(
                    "converged" if type(self).fit_success else "not converged"
                ) if type(self).message_override is None else type(self).message_override,
                params=params,
            ),
            phase_fractions=phase_fractions,
            lattice_params=lattice_params,
        )
        return self.last_result


def _install_phase_backend(monkeypatch):
    _Structure.calls = []
    _Structure.structures = []
    _PhaseModel.instances = []
    _PhaseFitter.instances = []
    monkeypatch.setattr(
        _fit_ops(),
        "_load_phase_backend",
        lambda: (_Structure, _PhaseModel, _PhaseFitter),
    )


def _recursive_values(value, seen=None):
    seen = set() if seen is None else seen
    marker = id(value)
    if marker in seen:
        return
    seen.add(marker)
    yield value
    if dataclasses.is_dataclass(value):
        for item in dataclasses.fields(value):
            yield from _recursive_values(getattr(value, item.name), seen)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _recursive_values(key, seen)
            yield from _recursive_values(item, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _recursive_values(item, seen)


def _receipt_value(receipt):
    return (
        receipt.schema_version, receipt.run_generation,
        receipt.run_fingerprint, receipt.source_scan, receipt.artifact,
        receipt.frame_label, receipt.work_ordinal,
    )


def _peak_storage_value(result):
    scalar = (
        "completed", result.code, result.diagnostics, result.trace_fingerprint,
        _receipt_value(result.trace_receipt), result.plan_fingerprint,
        result.policy_fingerprint, result.fit_success, result.message,
        result.label, result.axis_unit, result.parameter_names,
        result.parameter_values, result.parameter_stderr,
    )
    arrays = (
        result.fit, result.background, result.residual, result.marker_positions,
    )
    return scalar, arrays, result.result_fingerprint


def _phase_storage_value(result):
    receipts = tuple(
        (
            receipt.lexical_path, receipt.resolved_path, receipt.phase_name,
            receipt.byte_count, receipt.sha256, receipt.expected_sha256,
            receipt.pre_state, receipt.post_state, receipt.receipt_fingerprint,
        )
        for receipt in result.cif_receipts
    )
    scalar = (
        "completed", result.code, result.diagnostics, result.trace_fingerprint,
        _receipt_value(result.trace_receipt), result.plan_fingerprint,
        result.policy_fingerprint, result.wavelength_angstrom, receipts,
        result.fit_success, result.message, result.label, result.axis_unit,
        result.parameter_names, result.parameter_values,
        result.parameter_stderr, result.phase_fractions,
        result.lattice_parameters,
    )
    arrays = (
        result.fit, result.background, result.residual,
        result.marker_positions, *result.phase_components,
    )
    return scalar, arrays, result.result_fingerprint


def test_displayed_peak_fit_consumes_exact_trace_bytes_label_unit_and_identity(
    monkeypatch,
):
    ops = _fit_ops()
    assert ops._MAX_TRACE_POINTS == 1_000_000
    assert ops._MAX_TRACE_BYTES == 16 * 1024 * 1024
    assert ops._MAX_FIT_BYTES == 64 * 1024 * 1024
    axis = np.array([3.0, 1.0, 2.0, 4.0, 5.0])
    intensity = np.array([30.0, 10.0, 20.0, 40.0, 50.0])
    trace = _trace(axis=axis, intensity=intensity, label="display-identity")
    trace_identity = (
        trace.axis, trace.intensity, trace.label, trace.axis_unit, trace.title,
        trace.epoch, _receipt_value(trace.receipt),
    )
    assert trace.trace_fingerprint == ops._digest(trace_identity)
    assert trace.storage_bytes == ops._canonical_charge(
        (trace_identity, trace.trace_fingerprint)
    )
    original_fingerprint = trace.trace_fingerprint
    calls = []
    payloads = []
    _install_peak_backend(
        monkeypatch, calls, parameter_name="x" * 256, payloads=payloads,
    )
    result = _public("run_displayed_peak_fit")(_peak_plan(trace))
    _assert_terminal(result, "COMPLETED", "OK")
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][1], axis)
    np.testing.assert_array_equal(calls[0][2], intensity)
    assert result.trace_fingerprint == original_fingerprint
    assert result.label == "display-identity"
    assert result.axis_unit == "1/angstrom"
    assert result.trace_receipt == trace.receipt
    assert result.storage_bytes == ops._canonical_charge(
        _peak_storage_value(result)
    )
    scalar, arrays, _fingerprint = _peak_storage_value(result)
    assert result.result_fingerprint == ops._digest((scalar, arrays))
    for array in (
        result.fit, result.background, result.residual, result.marker_positions,
    ):
        _assert_owned_readonly(array)

    detached = tuple(
        np.array(value, copy=True)
        for value in (
            result.fit, result.background, result.residual, result.marker_positions,
        )
    )
    payloads[0].best_fit[:] = -100.0
    payloads[0].fit_result.best_fit[:] = -101.0
    payloads[0].fit_result.residual[:] = -102.0
    payloads[0].peak_centers[:] = [-103.0]
    for parameter in payloads[0].params.values():
        parameter.value = -104.0
    for actual, expected in zip(
        (result.fit, result.background, result.residual, result.marker_positions),
        detached,
    ):
        np.testing.assert_array_equal(actual, expected)
    assert not any(
        isinstance(value, SimpleNamespace) for value in _recursive_values(result)
    )

    nonconverged_calls = []
    _install_peak_backend(monkeypatch, nonconverged_calls, success=False)
    nonconverged = _public("run_displayed_peak_fit")(_peak_plan(trace))
    _assert_terminal(nonconverged, "COMPLETED", "OK")
    assert nonconverged.fit_success is False
    assert nonconverged.message == "not converged"
    assert len(nonconverged_calls) == 1

    diagnostic_calls = []
    _install_peak_backend(
        monkeypatch, diagnostic_calls, message="x" * 1024,
    )
    diagnostic_boundary = _public("run_displayed_peak_fit")(_peak_plan(trace))
    _assert_terminal(diagnostic_boundary, "COMPLETED", "OK")
    assert diagnostic_boundary.message == "x" * 1024
    diagnostic_calls.clear()
    _install_peak_backend(
        monkeypatch, diagnostic_calls, message="x" * 1025,
    )
    diagnostic_over = _public("run_displayed_peak_fit")(_peak_plan(trace))
    _assert_terminal(
        diagnostic_over, "REFUSED", "FIT_RESULT_LIMIT_EXCEEDED",
    )
    assert diagnostic_over.fit is None

    monkeypatch.setattr(ops, "_MAX_TRACE_BYTES", trace.storage_bytes)
    calls.clear()
    _install_peak_backend(monkeypatch, calls, parameter_name="x" * 256)
    trace_boundary = _public("run_displayed_peak_fit")(_peak_plan(trace))
    _assert_terminal(trace_boundary, "COMPLETED", "OK")
    monkeypatch.setattr(ops, "_MAX_TRACE_BYTES", trace.storage_bytes - 1)
    calls.clear()
    _install_peak_backend(monkeypatch, calls, parameter_name="x" * 256)
    trace_over = _public("run_displayed_peak_fit")(_peak_plan(trace))
    _assert_terminal(trace_over, "REFUSED", "TRACE_BYTE_LIMIT_EXCEEDED")
    assert calls == []
    monkeypatch.setattr(ops, "_MAX_TRACE_BYTES", 16 * 1024 * 1024)

    calls.clear()
    _install_peak_backend(monkeypatch, calls, parameter_name="x" * 257)
    refused = _public("run_displayed_peak_fit")(_peak_plan(trace))
    _assert_terminal(refused, "REFUSED", "FIT_RESULT_LIMIT_EXCEEDED")
    assert refused.fit is None

    calls.clear()
    _install_peak_backend(monkeypatch, calls, parameter_name="x" * 256)
    monkeypatch.setattr(ops, "_MAX_FIT_BYTES", result.storage_bytes)
    exact_fit = _public("run_displayed_peak_fit")(_peak_plan(trace))
    _assert_terminal(exact_fit, "COMPLETED", "OK")
    calls.clear()
    _install_peak_backend(monkeypatch, calls, parameter_name="x" * 256)
    monkeypatch.setattr(ops, "_MAX_FIT_BYTES", result.storage_bytes - 1)
    over = _public("run_displayed_peak_fit")(_peak_plan(trace))
    _assert_terminal(over, "REFUSED", "FIT_RESULT_LIMIT_EXCEEDED")
    assert over.fit is None

    too_many = _trace(axis=np.arange(1_000_001), intensity=np.ones(1_000_001))
    no_call = []
    _install_peak_backend(monkeypatch, no_call)
    refused = _public("run_displayed_peak_fit")(_peak_plan(too_many))
    _assert_terminal(refused, "REFUSED", "TRACE_POINT_LIMIT_EXCEEDED")
    assert no_call == []

    boundary_trace = _trace(
        axis=np.linspace(1.0, 2.0, 1_000_000),
        intensity=np.ones(1_000_000),
    )
    boundary_calls = []
    _install_peak_backend(monkeypatch, boundary_calls)
    monkeypatch.setattr(ops, "_MAX_FIT_BYTES", 64 * 1024 * 1024)
    boundary_points = _public("run_displayed_peak_fit")(_peak_plan(boundary_trace))
    _assert_terminal(boundary_points, "COMPLETED", "OK")
    assert len(boundary_calls) == 1

    def bad_backend(_plan, x, y):
        payload = _peak_payload(x, y)
        payload.best_fit = np.asarray(y).reshape(1, -1)
        return AnalysisResult(kind="peak_fit", payload=payload)

    monkeypatch.setattr(ops, "_load_peak_backend", lambda: (PeakFitPlan, bad_backend))
    bad_shape = _public("run_displayed_peak_fit")(_peak_plan(trace))
    _assert_terminal(bad_shape, "REFUSED", "FIT_RESULT_LIMIT_EXCEEDED")


def test_displayed_phase_fit_requires_inverse_angstrom_before_kernel(
    monkeypatch, tmp_path,
):
    cif = tmp_path / "phase.cif"
    cif.write_text("data_phase\n", encoding="utf-8")
    events = []
    real_open = builtins.open

    def unopened(path, *args, **kwargs):
        if Path(path) == cif:
            pytest.fail("CIF opened before inverse-angstrom guard")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", unopened)
    monkeypatch.setattr(
        _fit_ops(),
        "_load_phase_backend",
        lambda: events.append("dependency") or pytest.fail("dependency reached"),
    )
    for unit in ("2theta_deg", "", "angstrom"):
        result = _public("run_displayed_phase_fit")(
            _phase_plan(
                _trace(axis_unit=unit), (cif,), wavelength_angstrom=0.0,
            )
        )
        _assert_terminal(result, "REFUSED", "PHASE_Q_UNIT_REQUIRED")
    assert events == []


def test_phase_cif_capture_reads_once_hashes_and_parses_identical_bytes(
    monkeypatch, tmp_path,
):
    ops = _fit_ops()
    raw = b"\xef\xbb\xbfdata_phase\n_cell_length_a 5.0\n"
    cif = tmp_path / "phase.cif"
    cif.write_bytes(raw)
    expected_hash = hashlib.sha256(raw).hexdigest()
    reads = []
    real_open = builtins.open

    class _ObservedFile:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

        def fileno(self):
            return self._handle.fileno()

        def read(self, size=-1):
            reads.append(size)
            return self._handle.read(size)

    def observed_open(path, *args, **kwargs):
        handle = real_open(path, *args, **kwargs)
        return _ObservedFile(handle) if Path(path) == cif else handle

    monkeypatch.setattr(builtins, "open", observed_open)
    _install_phase_backend(monkeypatch)
    result = _public("run_displayed_phase_fit")(
        _phase_plan(
            _trace(),
            (cif,),
            expected_hashes=(expected_hash,),
            phase_names=("ExactPhase",),
        )
    )
    _assert_terminal(result, "COMPLETED", "OK")
    assert reads == [4 * 1024 * 1024 + 1]
    assert _Structure.calls == [(raw.decode("utf-8-sig"), "cif")]
    assert result.cif_receipts[0].sha256 == expected_hash
    assert result.cif_receipts[0].byte_count == len(raw)
    assert result.cif_receipts[0].phase_name == "ExactPhase"
    assert result.wavelength_angstrom == 0.6199
    assert result.storage_bytes == ops._canonical_charge(
        _phase_storage_value(result)
    )
    assert len(_PhaseModel.instances) == 1
    assert _PhaseModel.instances[0].structure is _Structure.structures[0]
    assert _PhaseModel.instances[0].calculate_calls == [0.6199]
    assert _PhaseFitter.instances[0].fit_calls == 1

    _install_phase_backend(monkeypatch)
    other_wavelength = _public("run_displayed_phase_fit")(
        _phase_plan(
            _trace(), (cif,), expected_hashes=(expected_hash,),
            phase_names=("ExactPhase",), wavelength_angstrom=1.0,
        )
    )
    _assert_terminal(other_wavelength, "COMPLETED", "OK")
    assert other_wavelength.wavelength_angstrom == 1.0
    assert other_wavelength.plan_fingerprint != result.plan_fingerprint
    assert other_wavelength.result_fingerprint != result.result_fingerprint
    assert _PhaseModel.instances[0].calculate_calls == [1.0]

    _install_phase_backend(monkeypatch)
    hash_mismatch = _public("run_displayed_phase_fit")(
        _phase_plan(
            _trace(),
            (cif,),
            expected_hashes=("0" * 64,),
            phase_names=("ExactPhase",),
        )
    )
    _assert_terminal(hash_mismatch, "REFUSED", "CIF_HASH_MISMATCH")
    assert _PhaseFitter.instances == []

    real_fstat = ops.os.fstat
    fstat_calls = []

    def drifting_fstat(descriptor):
        state = real_fstat(descriptor)
        fstat_calls.append(descriptor)
        if len(fstat_calls) == 2:
            return SimpleNamespace(
                st_mode=state.st_mode, st_dev=state.st_dev, st_ino=state.st_ino,
                st_size=state.st_size, st_mtime_ns=state.st_mtime_ns + 1,
                st_ctime_ns=state.st_ctime_ns,
            )
        return state

    _install_phase_backend(monkeypatch)
    with monkeypatch.context() as context:
        context.setattr(ops.os, "fstat", drifting_fstat)
        revision_changed = _public("run_displayed_phase_fit")(
            _phase_plan(_trace(), (cif,), phase_names=("ExactPhase",))
        )
    _assert_terminal(revision_changed, "REFUSED", "CIF_REVISION_CHANGED")
    assert len(fstat_calls) == 2
    assert _PhaseFitter.instances == []

    exact_cif = tmp_path / "exact-limit.cif"
    exact_cif.write_bytes(b"a" * (4 * 1024 * 1024))
    _install_phase_backend(monkeypatch)
    exact_limit = _public("run_displayed_phase_fit")(
        _phase_plan(_trace(), (exact_cif,))
    )
    _assert_terminal(exact_limit, "COMPLETED", "OK")
    assert exact_limit.cif_receipts[0].byte_count == 4 * 1024 * 1024

    oversized_cif = tmp_path / "over-limit.cif"
    oversized_cif.write_bytes(b"a" * (4 * 1024 * 1024 + 1))
    _install_phase_backend(monkeypatch)
    oversized = _public("run_displayed_phase_fit")(
        _phase_plan(_trace(), (oversized_cif,))
    )
    _assert_terminal(oversized, "REFUSED", "CIF_BYTE_LIMIT_EXCEEDED")
    assert _PhaseFitter.instances == []

    cumulative = []
    for index in range(4):
        path = tmp_path / f"cumulative-{index}.cif"
        path.write_bytes(bytes((97 + index,)) * (4 * 1024 * 1024))
        cumulative.append(path)
    _install_phase_backend(monkeypatch)
    cumulative_boundary = _public("run_displayed_phase_fit")(
        _phase_plan(_trace(), tuple(cumulative))
    )
    _assert_terminal(cumulative_boundary, "COMPLETED", "OK")
    assert sum(receipt.byte_count for receipt in cumulative_boundary.cif_receipts) == 16 * 1024 * 1024
    cumulative_over = tmp_path / "cumulative-over.cif"
    cumulative_over.write_bytes(b"z")
    _install_phase_backend(monkeypatch)
    cumulative_refused = _public("run_displayed_phase_fit")(
        _phase_plan(_trace(), tuple((*cumulative, cumulative_over)))
    )
    _assert_terminal(cumulative_refused, "REFUSED", "CIF_BYTE_LIMIT_EXCEEDED")
    assert all(instance.fit_calls == 0 for instance in _PhaseFitter.instances)


def test_single_trace_optional_dependency_refusal_is_typed(monkeypatch, tmp_path):
    ops = _fit_ops()
    monkeypatch.setattr(
        ops, "_load_peak_backend", lambda: (_ for _ in ()).throw(ImportError("lmfit"))
    )
    peak = _public("run_displayed_peak_fit")(_peak_plan(_trace()))
    _assert_terminal(peak, "REFUSED", "PEAK_FIT_DEPENDENCY_UNAVAILABLE")
    assert not any(isinstance(value, BaseException) for value in dataclasses.astuple(peak))

    monkeypatch.setattr(
        ops, "_auto_peak_policy",
        lambda *_a: (_ for _ in ()).throw(ImportError("scipy")),
    )
    auto_peak = _public("run_displayed_peak_fit")(
        _peak_plan(_trace(), selection_mode="auto")
    )
    _assert_terminal(
        auto_peak, "REFUSED", "PEAK_FIT_DEPENDENCY_UNAVAILABLE",
    )

    def missing_runner(*_args, **_kwargs):
        raise ImportError("lmfit")

    monkeypatch.setattr(ops, "_load_peak_backend", lambda: (PeakFitPlan, missing_runner))
    runner_peak = _public("run_displayed_peak_fit")(_peak_plan(_trace()))
    _assert_terminal(
        runner_peak, "REFUSED", "PEAK_FIT_DEPENDENCY_UNAVAILABLE",
    )

    cif = tmp_path / "phase.cif"
    cif.write_text("data_phase\n", encoding="utf-8")
    monkeypatch.setattr(
        ops,
        "_load_phase_backend",
        lambda: (_ for _ in ()).throw(ImportError("pymatgen")),
    )
    phase = _public("run_displayed_phase_fit")(_phase_plan(_trace(), (cif,)))
    _assert_terminal(phase, "REFUSED", "PHASE_FIT_DEPENDENCY_UNAVAILABLE")
    assert phase.wavelength_angstrom == 0.6199
    assert not any(isinstance(value, BaseException) for value in dataclasses.astuple(phase))


def test_peak_phase_batch_and_live_are_absent_from_p37_public_reachability():
    ops = _fit_ops()
    assert _RUNNERS.issubset(set(analysis.__all__))
    for absent in (
        "prepare_displayed_phase_fit",
        "run_displayed_peak_batch",
        "run_displayed_phase_batch",
        "run_displayed_phase_sequence",
        "run_live_peak_fit",
        "run_live_phase_fit",
    ):
        assert absent not in analysis.__all__
        assert not hasattr(analysis, absent)
    source = inspect.getsource(ops)
    for forbidden in (
        "run_batch", "fit_sequence", "fitting.batch", "OperationSlot",
        "LiveFrameSource", "NexusRecordWriter", "OutputTransaction",
    ):
        assert forbidden not in source
    with pytest.raises(TypeError):
        _phase_plan(_trace(), (), texture="march_dollase")


def test_p37_peak_range_finite_manual_auto_suggestion_policy_is_headless_and_versioned(
    monkeypatch,
):
    ops = _fit_ops()
    assert ops._PEAK_POLICY_VERSION == "p37-peak-selection-v1"
    assert ops._MAX_DIAGNOSTIC_BYTES == 1024
    trace = _trace(
        axis=np.linspace(0.0, 10.0, 801),
        intensity=(
            1.0
            + 8.0 * np.exp(-((np.linspace(0.0, 10.0, 801) - 2.0) / 0.08) ** 2)
            + 12.0 * np.exp(-((np.linspace(0.0, 10.0, 801) - 7.0) / 0.11) ** 2)
        ),
    )
    pre_cancel = threading.Event()
    pre_cancel.set()
    with monkeypatch.context() as context:
        context.setattr(
            ops, "_auto_peak_policy",
            lambda *_a: pytest.fail("pre-cancelled auto policy ran"),
        )
        cancelled = _public("run_displayed_peak_fit")(
            _peak_plan(trace, selection_mode="auto"),
            cancel_token=pre_cancel,
        )
    _assert_terminal(cancelled, "CANCELLED", "CANCELLED")

    calls = []
    _install_peak_backend(monkeypatch, calls)
    progress_cancel = threading.Event()
    cancelled_from_progress = _public("run_displayed_peak_fit")(
        _peak_plan(trace), cancel_token=progress_cancel,
        progress_callback=lambda *_a: progress_cancel.set(),
    )
    _assert_terminal(cancelled_from_progress, "CANCELLED", "CANCELLED")
    assert calls == []
    manual = _public("run_displayed_peak_fit")(
        _peak_plan(
            trace,
            selection_mode="auto",
            manual_centers=(8.5, 1.5),
            n_peaks=2,
            max_nfev=100_000,
        )
    )
    _assert_terminal(manual, "COMPLETED", "OK")
    assert calls[0][0].positions == (1.5, 8.5)
    assert calls[0][0].n_peaks == 2
    assert calls[0][0].fit_kwargs == {"max_nfev": 100_000}

    calls.clear()
    auto = _public("run_displayed_peak_fit")(
        _peak_plan(
            trace, selection_mode="auto", manual_centers=(), n_peaks=2,
            sigma_init=0.1, amplitude_init=1.0,
        )
    )
    _assert_terminal(auto, "COMPLETED", "OK")
    assert calls[0][0].positions == pytest.approx((2.0, 7.0), abs=0.03)
    assert calls[0][0].sigma_init == 0.1
    assert calls[0][0].amplitude_init == 1.0
    assert ops._auto_peak_policy(np.arange(160), np.ones(160)) == ()

    calls.clear()
    tuple_auto = _public("run_displayed_peak_fit")(
        _peak_plan(
            trace, selection_mode="auto", n_peaks=2,
            sigma_init=(0.1, 0.2),
        )
    )
    _assert_terminal(tuple_auto, "REFUSED", "INVALID_PEAK_PLAN")
    assert calls == []

    with monkeypatch.context() as context:
        context.setattr(ops, "_auto_peak_policy", lambda _x, _y: (1.0, 2.0, 3.0))
        eight_points = _trace(
            axis=np.linspace(0.0, 4.0, 8), intensity=np.ones(8),
        )
        calls.clear()
        resolved_count = _public("run_displayed_peak_fit")(
            _peak_plan(eight_points, selection_mode="auto", n_peaks=1)
        )
        _assert_terminal(
            resolved_count, "REFUSED", "INSUFFICIENT_FIT_POINTS",
        )
        assert calls == []

    list_plan = _peak_plan(
        trace, manual_centers=[2.0], n_peaks=1,
        fit_bounds=[0.0, 10.0], sigma_init=[0.1],
        sigma_bounds=[0.01, 1.0], amplitude_init=[1.0],
        amplitude_bounds=[0.0, 10.0],
    )
    assert list_plan.manual_centers == (2.0,)
    assert list_plan.fit_bounds == (0.0, 10.0)
    assert list_plan.sigma_init == (0.1,)
    assert list_plan.sigma_bounds == (0.01, 1.0)
    assert list_plan.amplitude_init == (1.0,)
    assert list_plan.amplitude_bounds == (0.0, 10.0)
    calls.clear()
    list_result = _public("run_displayed_peak_fit")(list_plan)
    _assert_terminal(list_result, "COMPLETED", "OK")
    assert calls[0][0].positions == (2.0,)
    for name, invalid_sequence in (
        ("manual_centers", {2.0}),
        ("manual_centers", iter((2.0,))),
        ("fit_bounds", {0.0, 10.0}),
        ("sigma_init", iter((0.1,))),
    ):
        with pytest.raises(ops.InvalidCanonicalValue):
            _peak_plan(trace, n_peaks=1, **{name: invalid_sequence})

    calls.clear()
    _install_peak_backend(monkeypatch, calls)
    twelve_centers = tuple(float(value) for value in np.linspace(0.5, 9.5, 12))
    twelve = _public("run_displayed_peak_fit")(
        _peak_plan(
            trace,
            manual_centers=twelve_centers,
            n_peaks=12,
        )
    )
    _assert_terminal(twelve, "COMPLETED", "OK")
    assert calls[0][0].positions == tuple(sorted(twelve_centers))

    for model, background in (
        ("pseudovoigt", "linear"),
        ("gaussian", "constant"),
        ("lorentzian", "none"),
        ("voigt", "chebyshev3"),
    ):
        calls.clear()
        _install_peak_backend(monkeypatch, calls)
        accepted = _public("run_displayed_peak_fit")(
            _peak_plan(trace, model=model, background=background)
        )
        _assert_terminal(accepted, "COMPLETED", "OK")
        assert calls[0][0].model == model
        assert calls[0][0].background == background

    five = _trace(axis=np.arange(5.0), intensity=np.ones(5))
    calls.clear()
    _install_peak_backend(monkeypatch, calls)
    finite_boundary = _public("run_displayed_peak_fit")(_peak_plan(five))
    _assert_terminal(finite_boundary, "COMPLETED", "OK")
    four = _trace(axis=np.arange(4.0), intensity=np.ones(4))
    calls.clear()
    _install_peak_backend(monkeypatch, calls)
    finite_over = _public("run_displayed_peak_fit")(_peak_plan(four))
    _assert_terminal(finite_over, "REFUSED", "INSUFFICIENT_FIT_POINTS")
    assert calls == []

    def parameter_backend(count):
        def run(_plan, x, y):
            payload = _peak_payload(x, y)
            params = {
                f"p{index}": SimpleNamespace(value=float(index), stderr=0.1)
                for index in range(count)
            }
            payload.params = params
            payload.fit_result.params = params
            return AnalysisResult(kind="peak_fit", payload=payload)
        return run

    monkeypatch.setattr(
        ops,
        "_load_peak_backend",
        lambda: (PeakFitPlan, parameter_backend(512)),
    )
    parameter_boundary = _public("run_displayed_peak_fit")(_peak_plan(trace))
    _assert_terminal(parameter_boundary, "COMPLETED", "OK")
    assert len(parameter_boundary.parameter_names) == 512
    monkeypatch.setattr(
        ops,
        "_load_peak_backend",
        lambda: (PeakFitPlan, parameter_backend(513)),
    )
    parameter_over = _public("run_displayed_peak_fit")(_peak_plan(trace))
    _assert_terminal(parameter_over, "REFUSED", "FIT_PARAMETER_LIMIT_EXCEEDED")

    invalid_changes = (
        {"selection_mode": "AUTO"},
        {"selection_mode": _StringAlias("count")},
        {"model": _StringAlias("gaussian")},
        {"background": _StringAlias("none")},
        {"n_peaks": True}, {"n_peaks": 0},
        {"n_peaks": 13}, {"manual_centers": (float("nan"),)},
        {
            "manual_centers": tuple(
                float(value) for value in np.linspace(0.1, 9.9, 13)
            ),
            "n_peaks": 12,
        },
        {"model": "pearson7"}, {"background": "polynomial"},
        {"sigma_init": 0.0}, {"sigma_bounds": (1.0, 1.0)},
        {"amplitude_init": -1.0}, {"amplitude_bounds": (2.0, 1.0)},
        {"center_bounds_delta": 20.0}, {"fraction_init": 1.01},
        {"max_nfev": 100_001}, {"max_nfev": True},
        {"fit_bounds": (5.0, 4.0)},
    )
    for changes in invalid_changes:
        calls.clear()
        result = _public("run_displayed_peak_fit")(_peak_plan(trace, **changes))
        _assert_terminal(result, "REFUSED", "INVALID_PEAK_PLAN")
        assert calls == []


def test_p37_phase_q_guard_precedes_dependency_cif_and_fit_work(
    monkeypatch, tmp_path,
):
    ops = _fit_ops()
    cif = tmp_path / "never-opened.cif"
    cif.write_text("data_never\n", encoding="utf-8")
    events = []

    def guard(unit, *, operation):
        events.append(("q_guard", unit, operation))
        raise ValueError("not inverse angstrom")

    monkeypatch.setattr(ops, "require_inverse_angstrom", guard)
    monkeypatch.setattr(
        ops,
        "_load_phase_backend",
        lambda: events.append(("dependency",)) or pytest.fail("late work reached"),
    )
    result = _public("run_displayed_phase_fit")(
        _phase_plan(
            _trace(axis_unit="degrees"), (cif,), wavelength_angstrom=0.0,
        )
    )
    _assert_terminal(result, "REFUSED", "PHASE_Q_UNIT_REQUIRED")
    assert events == [("q_guard", "degrees", "phase fitting")]


def test_p37_phase_reads_hashes_and_parses_each_cif_exactly_once_from_one_buffer(
    monkeypatch, tmp_path,
):
    ops = _fit_ops()
    assert ops._MAX_PHASES == 8
    assert ops._MAX_CIF_BYTES == 4 * 1024 * 1024
    assert ops._MAX_TOTAL_CIF_BYTES == 16 * 1024 * 1024
    assert ops._MAX_REFLECTIONS_PER_PHASE == 2048
    assert ops._MAX_REFLECTIONS_TOTAL == 4096
    assert ops._MAX_FIT_PARAMETERS == 512
    cifs = []
    hashes = []
    for index in range(2):
        path = tmp_path / f"phase{index}.cif"
        payload = f"data_phase{index}\n".encode()
        path.write_bytes(payload)
        cifs.append(path)
        hashes.append(hashlib.sha256(payload).hexdigest())

    _PhaseModel.peak_count = 2048
    _PhaseFitter.parameter_count = 512
    _install_phase_backend(monkeypatch)
    result = _public("run_displayed_phase_fit")(
        _phase_plan(
            _trace(), cifs,
            expected_hashes=tuple(hashes),
            phase_names=("A", "B"),
            prefit_background="snip",
            phase_profile="voigt",
            lattice_pct=0.25,
            min_intensity=100.0,
            max_nfev=100_000,
        )
    )
    _assert_terminal(result, "COMPLETED", "OK")
    assert len(_Structure.calls) == 2
    assert [phase.calculate_calls for phase in _PhaseModel.instances] == [
        [0.6199], [0.6199],
    ]
    assert [receipt.phase_name for receipt in result.cif_receipts] == ["A", "B"]
    assert [receipt.sha256 for receipt in result.cif_receipts] == hashes
    assert _PhaseFitter.instances[0].fit_calls == 1
    assert len(result.parameter_names) == 512
    assert result.wavelength_angstrom == 0.6199
    assert result.marker_positions.size == 4096
    assert result.storage_bytes == ops._canonical_charge(
        _phase_storage_value(result)
    )
    scalar, arrays, _fingerprint = _phase_storage_value(result)
    assert result.result_fingerprint == ops._digest((scalar, arrays))
    for array in (
        result.fit, result.background, result.residual,
        result.marker_positions, *result.phase_components,
    ):
        _assert_owned_readonly(array)

    phase_arrays = tuple(
        np.array(value, copy=True)
        for value in (
            result.fit, result.background, result.residual,
            result.marker_positions, *result.phase_components,
        )
    )
    phase_scalars = (
        result.parameter_names, result.parameter_values, result.parameter_stderr,
        result.phase_fractions, result.lattice_parameters,
    )
    fitter = _PhaseFitter.instances[0]
    fitter.q[:] = -10.0
    fitter.intensity[:] = -11.0
    fitter.background[:] = -12.0
    for parameter in fitter.last_params.values():
        parameter.value = -13.0
    fitter.phases[0][0].peaks.clear()
    for actual, expected in zip(
        (
            result.fit, result.background, result.residual,
            result.marker_positions, *result.phase_components,
        ),
        phase_arrays,
    ):
        np.testing.assert_array_equal(actual, expected)
    assert (
        result.parameter_names, result.parameter_values, result.parameter_stderr,
        result.phase_fractions, result.lattice_parameters,
    ) == phase_scalars
    assert not any(
        isinstance(value, (_PhaseFitter, _PhaseModel, SimpleNamespace))
        for value in _recursive_values(result)
    )

    phase_cancel = threading.Event()
    _install_phase_backend(monkeypatch)
    cancelled = _public("run_displayed_phase_fit")(
        _phase_plan(_trace(), (cifs[0],), phase_names=("A",)),
        cancel_token=phase_cancel,
        progress_callback=lambda done, _total: phase_cancel.set()
        if done == 1 else None,
    )
    _assert_terminal(cancelled, "CANCELLED", "CANCELLED")
    assert _PhaseFitter.instances == []

    invalid_plans = (
        _phase_plan(_trace(), ()),
        _phase_plan(_trace(), tuple(cifs[0] for _ in range(9))),
        _phase_plan(_trace(), cifs, phase_names=("A", "A")),
        _phase_plan(_trace(), cifs, expected_hashes=("ABC", None)),
        _phase_plan(_trace(), cifs, wavelength_angstrom=0.0),
        _phase_plan(_trace(), cifs, wavelength_angstrom=-0.1),
        _phase_plan(_trace(), cifs, wavelength_angstrom=float("nan")),
        _phase_plan(_trace(), cifs, wavelength_angstrom=float("inf")),
        _phase_plan(_trace(), cifs, wavelength_angstrom=True),
        _phase_plan(_trace(), cifs, prefit_background="SNIP"),
        _phase_plan(
            _trace(), cifs, prefit_background=_StringAlias("none"),
        ),
        _phase_plan(_trace(), cifs, phase_profile="pearson7"),
        _phase_plan(
            _trace(), cifs, phase_profile=_StringAlias("gaussian"),
        ),
        _phase_plan(_trace(), cifs, lattice_pct=0.251),
        _phase_plan(_trace(), cifs, min_intensity=100.1),
        _phase_plan(_trace(), cifs, max_nfev=100_001),
    )
    for plan in invalid_plans:
        _install_phase_backend(monkeypatch)
        with monkeypatch.context() as context:
            late = lambda *_a, **_k: pytest.fail(
                "invalid wavelength/phase plan reached dependency or CIF work"
            )
            context.setattr(ops, "_load_phase_backend", late)
            context.setattr(ops, "_capture_cif", late)
            refused = _public("run_displayed_phase_fit")(plan)
        _assert_terminal(refused, "REFUSED", "INVALID_PHASE_PLAN")
        assert _PhaseFitter.instances == []

    with pytest.raises(TypeError):
        _public("DisplayedPhaseFitPlan")(
            trace=_trace(), cif_paths=(cifs[0],), expected_hashes=(None,),
            phase_names=("A",),
        )

    normalized_plan = _public("DisplayedPhaseFitPlan")(
        trace=_trace(), cif_paths=list(cifs), expected_hashes=list(hashes),
        phase_names=["A", "B"], wavelength_angstrom=0.6199,
    )
    assert normalized_plan.cif_paths == tuple(cifs)
    assert normalized_plan.expected_hashes == tuple(hashes)
    assert normalized_plan.phase_names == ("A", "B")
    for name, invalid in (
        ("cif_paths", set(cifs)),
        ("expected_hashes", iter(hashes)),
        ("phase_names", {"A", "B"}),
    ):
        values = dict(
            trace=_trace(), cif_paths=tuple(cifs),
            expected_hashes=tuple(hashes), phase_names=("A", "B"),
            wavelength_angstrom=0.6199,
        )
        values[name] = invalid
        with pytest.raises(ops.InvalidCanonicalValue):
            _public("DisplayedPhaseFitPlan")(**values)

    _PhaseModel.peak_count = 2
    _PhaseModel.peak_counts = None
    _PhaseFitter.parameter_count = 3
    _install_phase_backend(monkeypatch)
    identity_boundary = _public("run_displayed_phase_fit")(
        _phase_plan(
            _trace(), (cifs[0],), expected_hashes=(hashes[0],),
            phase_names=("n" * 256,),
        )
    )
    _assert_terminal(identity_boundary, "COMPLETED", "OK")
    _install_phase_backend(monkeypatch)
    name_over = _public("run_displayed_phase_fit")(
        _phase_plan(_trace(), (cifs[0],), phase_names=("n" * 257,))
    )
    _assert_terminal(name_over, "REFUSED", "INVALID_PHASE_PLAN")
    _install_phase_backend(monkeypatch)
    uppercase_hash = _public("run_displayed_phase_fit")(
        _phase_plan(
            _trace(), (cifs[0],), expected_hashes=(hashes[0].upper(),),
            phase_names=("A",),
        )
    )
    _assert_terminal(uppercase_hash, "REFUSED", "INVALID_PHASE_PLAN")

    eight_cifs = []
    for index in range(8):
        path = tmp_path / f"phase-count-{index}.cif"
        path.write_bytes(f"data_count_{index}\n".encode())
        eight_cifs.append(path)
    _install_phase_backend(monkeypatch)
    eight = _public("run_displayed_phase_fit")(
        _phase_plan(
            _trace(), tuple(eight_cifs),
            phase_names=tuple(f"P{index}" for index in range(8)),
        )
    )
    _assert_terminal(eight, "COMPLETED", "OK")
    assert len(eight.cif_receipts) == 8
    assert len(eight.phase_components) == 8

    _PhaseModel.peak_count = 2049
    _PhaseFitter.parameter_count = 3
    _install_phase_backend(monkeypatch)
    reflection_over = _public("run_displayed_phase_fit")(
        _phase_plan(_trace(), (cifs[0],), phase_names=("A",))
    )
    _assert_terminal(reflection_over, "REFUSED", "PHASE_REFLECTION_LIMIT_EXCEEDED")
    if _PhaseFitter.instances:
        assert _PhaseFitter.instances[0].add_calls == 0
        assert _PhaseFitter.instances[0].fit_calls == 0

    third = tmp_path / "phase2.cif"
    third.write_text("data_phase2\n", encoding="utf-8")
    _PhaseModel.peak_count = 2
    _PhaseModel.peak_counts = {"A": 2048, "B": 2048, "C": 1}
    _install_phase_backend(monkeypatch)
    cumulative_reflections = _public("run_displayed_phase_fit")(
        _phase_plan(
            _trace(), (*cifs, third), phase_names=("A", "B", "C"),
        )
    )
    _assert_terminal(
        cumulative_reflections, "REFUSED", "PHASE_REFLECTION_LIMIT_EXCEEDED",
    )
    assert _PhaseFitter.instances == []
    _PhaseModel.peak_counts = None

    _PhaseModel.peak_count = 2
    _PhaseFitter.parameter_count = 513
    _install_phase_backend(monkeypatch)
    parameter_over = _public("run_displayed_phase_fit")(
        _phase_plan(_trace(), (cifs[0],), phase_names=("A",))
    )
    _assert_terminal(parameter_over, "REFUSED", "FIT_PARAMETER_LIMIT_EXCEEDED")
    assert len(_PhaseFitter.instances) == 1
    assert _PhaseFitter.instances[0].add_calls == 1
    assert _PhaseFitter.instances[0].fit_calls == 0

    _PhaseFitter.parameter_count = 3
    _install_phase_backend(monkeypatch)
    baseline = _public("run_displayed_phase_fit")(
        _phase_plan(_trace(), (cifs[0],), phase_names=("A",))
    )
    _assert_terminal(baseline, "COMPLETED", "OK")
    monkeypatch.setattr(ops, "_MAX_FIT_BYTES", baseline.storage_bytes)
    _install_phase_backend(monkeypatch)
    exact_fit = _public("run_displayed_phase_fit")(
        _phase_plan(_trace(), (cifs[0],), phase_names=("A",))
    )
    _assert_terminal(exact_fit, "COMPLETED", "OK")
    monkeypatch.setattr(ops, "_MAX_FIT_BYTES", baseline.storage_bytes - 1)
    _install_phase_backend(monkeypatch)
    over = _public("run_displayed_phase_fit")(
        _phase_plan(_trace(), (cifs[0],), phase_names=("A",))
    )
    _assert_terminal(over, "REFUSED", "FIT_RESULT_LIMIT_EXCEEDED")
    assert over.fit is None and over.cif_receipts == ()

    monkeypatch.setattr(ops, "_MAX_FIT_BYTES", 1)
    _PhaseFitter.summary_forbidden = True
    _install_phase_backend(monkeypatch)
    impossible = _public("run_displayed_phase_fit")(
        _phase_plan(_trace(), (cifs[0],), phase_names=("A",))
    )
    _assert_terminal(impossible, "REFUSED", "FIT_RESULT_LIMIT_EXCEEDED")
    _PhaseFitter.summary_forbidden = False

    monkeypatch.setattr(ops, "_MAX_FIT_BYTES", 64 * 1024 * 1024)
    for attribute, invalid in (
        ("fraction_override", {"A": float("nan")}),
        ("lattice_override", {"bad": 1.0}),
        ("lattice_override", {"a": float("inf")}),
    ):
        setattr(_PhaseFitter, attribute, invalid)
        _install_phase_backend(monkeypatch)
        invalid_summary = _public("run_displayed_phase_fit")(
            _phase_plan(_trace(), (cifs[0],), phase_names=("A",))
        )
        _assert_terminal(
            invalid_summary, "REFUSED", "FIT_RESULT_LIMIT_EXCEEDED",
        )
        setattr(_PhaseFitter, attribute, None)

    _PhaseFitter.message_override = "x" * 1024
    _install_phase_backend(monkeypatch)
    diagnostic_boundary = _public("run_displayed_phase_fit")(
        _phase_plan(_trace(), (cifs[0],), phase_names=("A",))
    )
    _assert_terminal(diagnostic_boundary, "COMPLETED", "OK")
    assert diagnostic_boundary.message == "x" * 1024
    _PhaseFitter.message_override = "x" * 1025
    _install_phase_backend(monkeypatch)
    diagnostic_over = _public("run_displayed_phase_fit")(
        _phase_plan(_trace(), (cifs[0],), phase_names=("A",))
    )
    _assert_terminal(diagnostic_over, "REFUSED", "FIT_RESULT_LIMIT_EXCEEDED")
    _PhaseFitter.message_override = None

    exact_names = tuple(
        f"{index:03d}".ljust(128, "p") for index in range(512)
    )
    _PhaseFitter.parameter_names_override = exact_names
    _install_phase_backend(monkeypatch)
    exact_name_bytes = _public("run_displayed_phase_fit")(
        _phase_plan(_trace(), (cifs[0],), phase_names=("A",))
    )
    _assert_terminal(exact_name_bytes, "COMPLETED", "OK")
    assert sum(len(name.encode("utf-8")) for name in exact_name_bytes.parameter_names) == 64 * 1024
    _PhaseFitter.parameter_names_override = (*exact_names[:-1], exact_names[-1] + "q")
    _install_phase_backend(monkeypatch)
    name_bytes_over = _public("run_displayed_phase_fit")(
        _phase_plan(_trace(), (cifs[0],), phase_names=("A",))
    )
    _assert_terminal(name_bytes_over, "REFUSED", "FIT_RESULT_LIMIT_EXCEEDED")
    _PhaseFitter.parameter_names_override = None

    _PhaseFitter.fit_success = False
    _install_phase_backend(monkeypatch)
    nonconverged = _public("run_displayed_phase_fit")(
        _phase_plan(_trace(), (cifs[0],), phase_names=("A",))
    )
    _assert_terminal(nonconverged, "COMPLETED", "OK")
    assert nonconverged.fit_success is False
    assert nonconverged.message == "not converged"

    _PhaseModel.peak_count = 2
    _PhaseModel.peak_counts = None
    _PhaseFitter.parameter_count = 3
    _PhaseFitter.parameter_names_override = None
    _PhaseFitter.fit_success = True
    _PhaseFitter.message_override = None
    _PhaseFitter.fraction_override = None
    _PhaseFitter.lattice_override = None
    _PhaseFitter.summary_forbidden = False
