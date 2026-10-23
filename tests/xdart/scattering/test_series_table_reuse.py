"""A Run of single-frame series builds pyFAI's tables once and reuses them for every series (real executor, real TIFF
series with .txt sidecars, real pyFAI csr), with outputs identical to processing each series on its own."""

from __future__ import annotations

import time

import h5py
import numpy as np
import pytest

import xrd_tools.reduction.core as core
from tests.xdart.scattering.test_live_qualify_refusal_run import _stop, _wait
from tests.xdart.scattering.test_live_revalidation_cost_run import TINY, _count_labels, _poni, _Sidecar
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal, _start
from xdart.gui.tabs.scattering.adapters.dynamic_output import DynamicOutputAdapter
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec

_STEMS = ("posa", "posb", "posc")
_DATASETS = ("entry/integrated_1d/axis_1", "entry/integrated_1d/intensity", "entry/integrated_2d/axis_1",
             "entry/integrated_2d/axis_2", "entry/integrated_2d/intensity")


@pytest.fixture
def builds(monkeypatch):
    """The worker integrators the Run builds; each starts without pyFAI's tables."""
    built = []
    real = core._ReductionIntegratorProvider._new_standard

    def counting(self):
        built.append(self.scan.name)
        return real(self)

    monkeypatch.setattr(core._ReductionIntegratorProvider, "_new_standard", counting)
    return built


def _intent(tmp_path, raw, *, cores=1, live=False, processing="Int 2D"):
    return RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=(".tif",), metadata_format="txt"),
        poni_file=str(_poni(tmp_path, TINY)), project_root=str(raw), save_path=str(tmp_path / "processed"),
        output_mode="Append", processing_mode=processing, live_mode=live, max_cores=cores,
        bai_1d_args={"npt": 12, "method": "csr"}, bai_2d_args={"npt_rad": 10, "npt_azim": 8, "method": "csr"})


def _series(raw, stems):
    raw.mkdir(parents=True)
    for number, stem in enumerate(stems, 1):
        _Sidecar(raw, TINY, stem).frame(number)  # each series one frame, each frame its own values


def _outputs(tmp_path, stem):
    with h5py.File(tmp_path / "processed" / f"{stem}_int2d.nexus", "r") as handle:
        return [handle[name][()] for name in _DATASETS]


def _directory_run(tmp_path, stems, *, cores, request_value):
    _series(tmp_path / "raw", stems)
    executor, identity, events = _run_to_terminal(_intent(tmp_path, tmp_path / "raw", cores=cores),
                                                  request_value=request_value)
    executor.close(identity)
    terminal = [event for event in events if event.kind in _TERMINAL]
    assert [event.kind.name for event in terminal] == ["FINISHED"], terminal


@pytest.mark.parametrize("cores", [1, 4])
def test_a_directory_run_of_single_frame_series_builds_the_tables_once(tmp_path, builds, cores):
    _directory_run(tmp_path / "together", _STEMS, cores=cores, request_value=9510 + cores)
    assert len(builds) == 1, builds  # on the parent: one per series
    together = {stem: _outputs(tmp_path / "together", stem) for stem in _STEMS}
    for number, stem in enumerate(_STEMS):
        alone = tmp_path / f"alone-{stem}"
        raw = alone / "raw"
        raw.mkdir(parents=True)
        _Sidecar(raw, TINY, stem).frame(number + 1)
        executor, identity, events = _run_to_terminal(_intent(alone, raw, cores=cores), request_value=9520 + number)
        executor.close(identity)
        assert all(np.array_equal(left, right, equal_nan=True)
                   for left, right in zip(together[stem], _outputs(alone, stem))), stem


def test_a_failed_run_end_still_releases_the_runs_integrators(tmp_path, monkeypatch):
    adapter = DynamicOutputAdapter(_intent(tmp_path, tmp_path).freeze())
    adapter._integrator_reuse.give_back("key", object())

    def failing(*, stopped):
        raise RuntimeError("an output failed to settle")

    monkeypatch.setattr(adapter, "_finish_all", failing)
    with pytest.raises(RuntimeError, match="failed to settle"):
        adapter.finish_all()
    assert len(adapter._integrator_reuse) == 0


def test_a_live_run_reuses_the_tables_from_one_series_to_the_next(tmp_path, builds):
    raw = tmp_path / "raw"
    raw.mkdir()
    executor = StandardRunExecutor(join_timeout=2.0)
    identity = _start(executor, _intent(tmp_path, raw, live=True, processing="Int 1D"), request_value=9530)
    events = []
    try:
        for number, stem in enumerate(_STEMS, 1):
            _Sidecar(raw, TINY, stem).frame(number)
            _wait(executor, events, _count_labels(stem, 1), 60.0)
            time.sleep(0.2)  # the next series arrives after this one has settled, as at the beamline
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert len(builds) == 1, builds  # on the parent: one per series
