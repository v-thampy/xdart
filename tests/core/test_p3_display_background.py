"""Qt-free public contract for bounded display-background aggregation."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.reduction import (
    DisplayBackgroundPlan, DisplayBackgroundResult, run_display_background,
)


def _plan(domain, values, axes=(), units=(), *, differing=False):
    axes = axes or ((),) * len(values)
    return DisplayBackgroundPlan(
        domain,
        tuple(f"source-{index}" for index in range(len(values))),
        tuple(value.shape for value in values),
        tuple(tuple(axis.shape for axis in row) for row in axes),
        tuple(units for _value in values),
        differing,
    )


def _root(value):
    while isinstance(value, np.ndarray):
        value = value.base
    return value


def test_public_runner_is_qt_free_and_direct_raw_call_is_immutable() -> None:
    first = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
    second = np.array([[3.0, 5.0], [np.nan, 8.0]], dtype=np.float32)
    before = tuple(value.copy() for value in (first, second))
    result = run_display_background(
        _plan("raw", (first, second)), ((first,), (second,)))
    assert type(result) is DisplayBackgroundResult
    np.testing.assert_allclose(result.values, [[2.0, 5.0], [np.nan, 6.0]],
                               equal_nan=True)
    np.testing.assert_array_equal(result.finite_counts, [[2, 1], [0, 2]])
    assert result.values.dtype == np.float64
    assert result.finite_counts.dtype == np.uint64
    assert isinstance(_root(result.values), bytes)
    assert isinstance(_root(result.finite_counts), bytes)
    assert not result.values.flags.writeable and not result.finite_counts.flags.writeable
    assert all(np.array_equal(value, saved, equal_nan=True)
               for value, saved in zip((first, second), before, strict=True))
    source = (Path(__file__).parents[2] / "src/xrd_tools/reduction/background.py").read_text()
    tree = ast.parse(source)
    imports = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module or "" for node in ast.walk(tree)
         if isinstance(node, ast.ImportFrom)}
    assert not any(name.startswith(("PyQt", "PySide", "qtpy", "xdart"))
                   for name in imports)
    assert sum(isinstance(node, ast.FunctionDef)
               and node.name == "run_display_background" for node in tree.body) == 1


@pytest.mark.parametrize("domain", ("integrated_1d", "integrated_2d"))
def test_same_grid_domains_promote_finite_mean_and_axes(domain: str) -> None:
    if domain == "integrated_1d":
        axes = ((np.array([0.0, 1.0, 2.0]),),) * 2
        values = (np.array([1.0, np.nan, 5.0]),
                  np.array([3.0, 7.0, np.nan]))
        units = ("q\0A^-1",)
        expected = np.array([2.0, 7.0, 5.0])
    else:
        row, column = np.array([-1.0, 1.0]), np.array([0.0, 1.0, 2.0])
        axes = ((row, column), (row.copy(), column.copy()))
        values = (np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]]),
                  np.array([[3.0, 4.0, np.nan], [6.0, np.nan, 8.0]]))
        units = ("chi\0deg", "q\0A^-1")
        expected = np.array([[2.0, 4.0, 3.0], [5.0, 5.0, 8.0]])
    result = run_display_background(
        _plan(domain, values, axes, units),
        tuple((value, *axis) for value, axis in zip(values, axes, strict=True)),
    )
    np.testing.assert_allclose(result.values, expected)
    assert result.axis_units == units
    assert all(axis.dtype == np.float64 and isinstance(_root(axis), bytes)
               and not axis.flags.writeable for axis in result.axes)


def test_all_nan_cells_are_results_but_all_nonfinite_contributors_refuse() -> None:
    first = np.array([1.0, np.nan, np.nan])
    second = np.array([3.0, np.nan, 7.0])
    axis = np.arange(3.0)
    result = run_display_background(
        _plan("integrated_1d", (first, second), ((axis,), (axis,)), ("q",)),
        ((first, axis), (second, axis)),
    )
    np.testing.assert_allclose(result.values, [2.0, np.nan, 7.0], equal_nan=True)
    np.testing.assert_array_equal(result.finite_counts, [2, 0, 1])
    assert result.diagnostics == ("contributors=2", "finite=3")
    bad = np.full(3, np.nan)
    with pytest.raises(ValueError, match="contributor has no finite"):
        run_display_background(
            _plan("integrated_1d", (first, bad), ((axis,), (axis,)), ("q",)),
            ((first, axis), (bad, axis)),
        )


def test_differing_grid_1d_uses_explicit_float64_interpolation() -> None:
    reference = np.array([0.0, 1.0, 2.0])
    shifted = np.array([1.0, 2.0, 3.0])
    first, second = np.array([1.0, 3.0, 5.0]), np.array([3.0, 5.0, 7.0])
    result = run_display_background(
        _plan("integrated_1d", (first, second), ((reference,), (shifted,)),
              ("q",), differing=True),
        ((first, reference), (second, shifted)),
    )
    np.testing.assert_allclose(result.values, [1.0, 3.0, 5.0])
    np.testing.assert_array_equal(result.finite_counts, [1, 2, 2])
    rejected = second.astype(np.float32)
    with pytest.raises(ValueError, match="C float64"):
        run_display_background(
            _plan("integrated_1d", (first, rejected),
                  ((reference,), (shifted,)), ("q",), differing=True),
            ((first, reference), (rejected, shifted)),
        )


def test_shape_unit_domain_grid_and_nonnumeric_refusals() -> None:
    value, axis = np.ones(3), np.arange(3.0)
    with pytest.raises(ValueError, match="plan is invalid"):
        DisplayBackgroundPlan("science", ("a",), ((3,),), ((),), ((),))
    with pytest.raises(ValueError, match="axis units differ"):
        DisplayBackgroundPlan("integrated_1d", ("a", "b"), ((3,), (3,)),
                              (((3,),), ((3,),)), (("q",), ("2theta",)))
    with pytest.raises(ValueError, match="identity differs"):
        run_display_background(
            _plan("integrated_1d", (value,), ((axis,),), ("q",)),
            ((np.ones(2), axis),))
    with pytest.raises(ValueError, match="identity differs"):
        run_display_background(
            _plan("integrated_1d", (value,), ((axis,),), ("q",)),
            ((np.array(["a", "b", "c"]), axis),))
    rows, columns = np.arange(2.0), np.arange(3.0)
    plan = _plan("integrated_2d", (np.ones((2, 3)),) * 2,
                 ((rows, columns), (rows, columns)), ("y", "x"))
    with pytest.raises(ValueError, match="grids differ"):
        run_display_background(plan, ((np.ones((2, 3)), rows, columns),
                                      (np.ones((2, 3)), rows + 1.0, columns)))


def test_cancel_is_checked_without_returning_a_partial_result() -> None:
    calls = 0
    value = np.arange(8.0)
    def cancelled():
        nonlocal calls
        calls += 1
        return calls > 1
    with pytest.raises(InterruptedError, match="cancelled"):
        run_display_background(
            _plan("integrated_1d", (value,), ((value,),), ("q",)),
            ((value, value),), cancelled=cancelled)
