"""One Qt-free scalar results/CSV projection for GUI and notebook fits (F9).

Unknown uncertainties are None, including rows kept compact by the runner. Failed inputs keep a status row.
"""
from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FitTableRow:
    input: str
    component: str
    parameter: str
    value: float | None
    stderr: float | None
    unit: str
    status: str
    message: str = ""
    result_identity: str = ""
    path: str = ""
    fingerprint: str = ""
    selection: tuple = ()
    coordinate_units: tuple = ()
    model: str = ""
    settings: tuple = ()


@dataclass(frozen=True, slots=True)
class FitTable:
    rows: tuple[FitTableRow, ...]


def _number(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def outcomes_table(outcomes, *, axis_unit="", contexts=None, model="", settings=()) -> FitTable:
    """Project complete or compact outcomes with optional detached input keys/snapshots.

    ``contexts`` is a sequence aligned with outcomes or a mapping from their labels to keys/snapshots. With
    no outcomes it projects those current requests as ``not fit`` rows, so settings and identity can be exported
    before the first successful fit. Scientific values/errors/units are the public owners' scalar projection.
    """
    outcomes = list(outcomes)
    is_mapping = isinstance(contexts, Mapping)
    context_sequence = () if contexts is None or is_mapping else tuple(contexts)
    if not outcomes and contexts:
        context_items = contexts.items() if is_mapping else (
            (getattr(getattr(item, "key", item), "result_identity", str(i)), item)
            for i, item in enumerate(context_sequence))
        from xrd_tools.analysis.runner import AnalysisOutcome
        outcomes = [AnalysisOutcome(label, False, message="not fit") for label, _context in context_items]
    settings_pairs = tuple(settings.items()) if isinstance(settings, Mapping) else tuple(settings)
    phase_names = tuple(dict(settings_pairs).get("phase_names", ()))
    rows = []
    for ordinal, outcome in enumerate(outcomes):
        context = (contexts.get(outcome.label) if is_mapping else
                   context_sequence[ordinal] if ordinal < len(context_sequence) else None)
        key = getattr(context, "key", context)
        selected = tuple(getattr(key, "selection", ()))
        units = tuple((axis.name, axis.unit) for axis in getattr(context, "axes", ()))
        if not units and axis_unit:
            units = ((dict(selected).get("coordinate", "axis"), axis_unit),)
        metadata = dict(result_identity=getattr(key, "result_identity", getattr(key, "label", outcome.label)),
                        path=getattr(key, "original_path", "") or getattr(key, "path", ""),
                        fingerprint=getattr(key, "fingerprint", getattr(key, "result_fingerprint", "")),
                        selection=selected, coordinate_units=units, model=model,
                        settings=settings_pairs or tuple(getattr(key, "settings", ())))
        status = "not fit" if outcome.message == "not fit" else "ok" if outcome.ok else "failed"
        result = outcome.result
        payload = getattr(result, "payload", None)
        names = getattr(result, "parameter_names", ())
        values = getattr(result, "parameter_values", ())
        errors = getattr(result, "parameter_stderr", ())
        unit = getattr(result, "axis_unit", "") or axis_unit
        scalar = getattr(outcome, "parameter_values", {})
        if scalar:
            errors = outcome.parameter_errors
            parameter_units = outcome.parameter_units
            for name, value in scalar.items():
                if name.startswith("frac_"):
                    continue
                component, parameter = _split_parameter(name)
                if phase_names and component.startswith("p") and component[1:].isdigit():
                    index = int(component[1:])
                    component = phase_names[index] if index < len(phase_names) else component
                rows.append(FitTableRow(outcome.label, component, parameter, _number(value),
                                        _number(errors.get(name)), parameter_units.get(name, ""),
                                        status, outcome.message, **metadata))
        elif payload is not None and hasattr(payload, "_flat_items"):
            for name, estimate, parameter_unit in payload._flat_items():
                if estimate is not None:
                    component, parameter = _split_parameter(name)
                    rows.append(FitTableRow(outcome.label, component, parameter, _number(estimate.value),
                                            _number(estimate.stderr), parameter_unit, status,
                                            outcome.message, **metadata))
        elif names:
            for i, (name, value) in enumerate(zip(names, values)):
                component, _, parameter = name.partition("_")
                if not parameter:
                    component, parameter = "", name
                parameter_unit = unit if parameter in {"center", "sigma", "fwhm"} else ""
                rows.append(FitTableRow(outcome.label, component, parameter, _number(value),
                                        _number(errors[i]) if i < len(errors) else None,
                                        parameter_unit, status, outcome.message, **metadata))
        elif outcome.params:
            for name, value in outcome.params.items():
                if name.startswith(("center_err_", "frac_")) or name.endswith("_err"):
                    continue
                parameter, _, component = name.rpartition("_")
                parameter = parameter or name
                error = (outcome.params.get(f"center_err_{component}") if parameter == "center" else
                         outcome.params.get(f"{name}_err"))
                rows.append(FitTableRow(outcome.label, component, parameter, _number(value), _number(error),
                                        unit if parameter in {"center", "fwhm"} else "", status, outcome.message, **metadata))
        else:
            rows.append(FitTableRow(outcome.label, "", "", None, None, "", status, outcome.message, **metadata))
        # Phase fractions are a named derived projection, not a raw lmfit parameter. The public analyzer's
        # compact params retains it; its uncertainty is unavailable unless that owner explicitly supplies one.
        for name, value in outcome.params.items():
            if name.startswith("frac_"):
                rows.append(FitTableRow(outcome.label, name[5:], "fraction", _number(value),
                                        _number(getattr(outcome, "parameter_errors", {}).get(name)), "",
                                        status, outcome.message, **metadata))
    return FitTable(tuple(rows))


def table_csv(table: FitTable) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("input", "component", "parameter", "value", "stderr", "unit", "status", "message",
                     "result_identity", "path", "fingerprint", "selection", "coordinate_units", "model", "settings"))
    for row in table.rows:
        writer.writerow((row.input, row.component, row.parameter, row.value, row.stderr, row.unit, row.status, row.message,
                         row.result_identity, row.path, row.fingerprint, _json(row.selection),
                         _json(row.coordinate_units), row.model, _json(row.settings)))
    return stream.getvalue()


def _split_parameter(name):
    if "." in name:
        return tuple(name.split(".", 1))
    component, _, parameter = name.partition("_")
    return (component, parameter) if parameter else ("", name)


def _json(value):
    from pathlib import Path
    def primitive(item):
        if isinstance(item, Path):
            return str(item)
        raise TypeError("table settings are primitive values or explicit asset paths")
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=primitive, separators=(",", ":"))
