"""``run_batch``'s lazy, compact mode (FIT-TOOLS-2026-10, Codex's design review F5; oracle row 7): descriptors loaded
one at a time, in order, cancelled before a load and between fits, one heavy outcome retained, the retained memory
flat in the number of results. The default mode is unchanged (``test_analysis_runner.py``)."""
from __future__ import annotations

import gc
import tracemalloc
import weakref

import numpy as np
import pytest

from xrd_tools.analysis.runner import AnalysisInput, DisplayedPeakFitAnalyzer, run_batch

_ANALYZER = DisplayedPeakFitAnalyzer({"model": "gaussian", "background": "constant"})


def _pattern(index, points=601):
    q = np.linspace(1.0, 4.0, points)
    return q, 20.0 + 500.0 * np.exp(-0.5 * ((q - (2.0 + 0.01 * index)) / 0.03) ** 2)


class _Results:
    """Descriptors that load their pattern only when called, recording each load and how many inputs loaded before
    it are still alive."""

    def __init__(self, count, points=601):
        self.log, self._alive, self.points = [], [], points
        self.descriptors = [self._loader(index) for index in range(count)]

    def _loader(self, index):
        def load():
            self.log.append(("load", index, sum(ref() is not None for ref in self._alive)))
            inp = AnalysisInput(f"result {index}", *_pattern(index, self.points), x_unit="A^-1")
            self._alive.append(weakref.ref(inp))
            return inp
        return load


def test_a_compact_batch_loads_each_result_only_at_its_turn_and_keeps_rows_but_the_last():
    results = _Results(4)
    progress = []
    outcomes = run_batch(_ANALYZER, results.descriptors, on_progress=lambda done, total: progress.append((done, total)),
                         on_frame=lambda outcome: results.log.append(("fit", outcome.label, outcome.overlay is not None)),
                         compact=True)
    assert results.log == [entry for index in range(4) for entry in (
        ("load", index, 0), ("fit", f"result {index}", True))]  # one input in memory at a time, in order
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert [outcome.label for outcome in outcomes] == [f"result {index}" for index in range(4)]
    assert all(outcome.ok for outcome in outcomes)
    np.testing.assert_allclose([outcome.params["center_0"] for outcome in outcomes],
                               [2.0, 2.01, 2.02, 2.03], atol=1e-4)
    assert all(outcome.overlay is None and outcome.result is None for outcome in outcomes[:-1])
    assert outcomes[-1].overlay is not None and outcomes[-1].result is not None  # the one shown


def test_a_compact_batch_cancels_before_a_load_and_between_fits():
    nothing = _Results(3)
    assert run_batch(_ANALYZER, nothing.descriptors, should_cancel=lambda: True, compact=True) == []
    assert nothing.log == []  # cancelled before anything was loaded
    two = _Results(5)
    fitted = []
    outcomes = run_batch(_ANALYZER, two.descriptors, should_cancel=lambda: len(fitted) >= 2,
                         on_frame=fitted.append, compact=True)
    assert [outcome.label for outcome in outcomes] == ["result 0", "result 1"]
    assert [entry[1] for entry in two.log] == [0, 1]  # the third was never loaded


def test_a_compact_batch_names_a_failed_load_and_goes_on_and_refuses_a_scan_unit_analyzer():
    def broken():
        raise OSError("result unreadable")

    outcomes = run_batch(_ANALYZER, [broken, _Results(1).descriptors[0]], compact=True)
    assert (outcomes[0].label, outcomes[0].ok, outcomes[0].message) == ("0", False, "result unreadable")
    assert outcomes[1].ok
    from xrd_tools.analysis.runner import Sin2PsiAnalyzer
    with pytest.raises(ValueError, match="frame-unit"):
        run_batch(Sin2PsiAnalyzer(plan=None), [], compact=True)


def _retained(count, *, compact):
    """The bytes a batch's returned outcomes keep (tracemalloc), for ``count`` results of 10,000 points each; a
    first batch beforehand, so the backend's one-time imports are not counted."""
    run_batch(_ANALYZER, _Results(1, 10_000).descriptors, compact=True)
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        if compact:
            outcomes = run_batch(_ANALYZER, _Results(count, 10_000).descriptors, compact=True)
        else:
            outcomes = run_batch(_ANALYZER, [AnalysisInput(f"result {index}", *_pattern(index, 10_000),
                                                           x_unit="A^-1") for index in range(count)])
        gc.collect()
        retained = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    assert len(outcomes) == count and all(outcome.ok for outcome in outcomes)
    return retained


def test_what_a_compact_batch_retains_does_not_grow_with_the_number_of_results():
    few, many = _retained(4, compact=True), _retained(24, compact=True)
    eager = _retained(24, compact=False)
    assert many < few + 128 * 1024, (few, many)  # flat: 20 more results add only their rows
    assert eager > 5 * many, (eager, many)  # the default mode keeps every outcome: the measure discriminates
