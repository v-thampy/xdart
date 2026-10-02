"""One Qt-free scalar results/CSV projection for GUI and notebook fits (F9).

Unknown uncertainties are None, including rows kept compact by the runner. Failed inputs keep a status row.
"""
from __future__ import annotations

import csv
import io
import math
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


@dataclass(frozen=True, slots=True)
class FitTable:
    rows: tuple[FitTableRow, ...]


def _number(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def outcomes_table(outcomes, *, axis_unit="") -> FitTable:
    rows = []
    for outcome in outcomes:
        status = "ok" if outcome.ok else "failed"
        result = outcome.result
        names = getattr(result, "parameter_names", ())
        values = getattr(result, "parameter_values", ())
        errors = getattr(result, "parameter_stderr", ())
        unit = getattr(result, "axis_unit", "") or axis_unit
        if names:
            for i, (name, value) in enumerate(zip(names, values)):
                component, _, parameter = name.partition("_")
                if not parameter:
                    component, parameter = "", name
                parameter_unit = unit if parameter in {"center", "sigma", "fwhm"} else ""
                rows.append(FitTableRow(outcome.label, component, parameter, _number(value),
                                        _number(errors[i]) if i < len(errors) else None,
                                        parameter_unit, status, outcome.message))
        elif outcome.params:
            for name, value in outcome.params.items():
                if name.startswith("center_err_"):
                    continue
                parameter, _, component = name.rpartition("_")
                parameter = parameter or name
                error = outcome.params.get(f"center_err_{component}") if parameter == "center" else None
                rows.append(FitTableRow(outcome.label, component, parameter, _number(value), _number(error),
                                        unit if parameter in {"center", "fwhm"} else "", status, outcome.message))
        else:
            rows.append(FitTableRow(outcome.label, "", "", None, None, "", status, outcome.message))
    return FitTable(tuple(rows))


def table_csv(table: FitTable) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("input", "component", "parameter", "value", "stderr", "unit", "status", "message"))
    for row in table.rows:
        writer.writerow((row.input, row.component, row.parameter, row.value, row.stderr, row.unit, row.status, row.message))
    return stream.getvalue()
