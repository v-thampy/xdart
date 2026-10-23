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


def _outputs(tmp_path, stem, name="int2d"):
    with h5py.File(tmp_path / "processed" / f"{stem}_{name}.nexus", "r") as handle:
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


@pytest.fixture
def promotions(monkeypatch):
    """GI integrators the Run promotes (each starts with no tables)."""
    import xrd_tools.integrate.gid as gid
    made = []
    real = gid._fiber_from_integrator

    def counting(*args, **kwargs):
        made.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(gid, "_fiber_from_integrator", counting)
    return made


def _gi_intent(tmp_path, raw, *, live=False, processing="Int 2D"):
    from xrd_tools.session.run_configuration import GIIntent
    intent = _intent(tmp_path, raw, live=live, processing=processing)
    intent.gi = GIIntent(enabled=True, incidence_motor="Manual", th_val=0.2)
    return intent


def test_a_gi_directory_run_at_one_angle_reuses_its_scout_and_worker_integrators(tmp_path, promotions):
    together = tmp_path / "together"
    _series(together / "raw", _STEMS)
    executor, identity, events = _run_to_terminal(_gi_intent(together, together / "raw"), request_value=9540)
    executor.close(identity)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
    assert len(promotions) <= 3, promotions  # on the parent: a scout and a worker per series (6)
    for number, stem in enumerate(_STEMS):
        alone = tmp_path / f"alone-{stem}"
        raw = alone / "raw"
        raw.mkdir(parents=True)
        _Sidecar(raw, TINY, stem).frame(number + 1)
        executor, identity, events = _run_to_terminal(_gi_intent(alone, raw), request_value=9550 + number)
        executor.close(identity)
        assert all(np.array_equal(left, right, equal_nan=True)
                   for left, right in zip(_outputs(together, stem, "gi_int2d"), _outputs(alone, stem, "gi_int2d"))), stem


def test_a_gi_live_run_reuses_its_integrators_from_one_series_to_the_next(tmp_path, promotions):
    raw = tmp_path / "raw"
    raw.mkdir()
    executor = StandardRunExecutor(join_timeout=2.0)
    identity = _start(executor, _gi_intent(tmp_path, raw, live=True, processing="Int 1D"), request_value=9560)
    events = []
    try:
        for number, stem in enumerate(_STEMS, 1):
            _Sidecar(raw, TINY, stem).frame(number)
            _wait(executor, events, _count_labels(f"{stem}_gi", 1), 60.0)
            time.sleep(0.2)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert len(promotions) <= 3, promotions  # on the parent: one or two per series


def test_on_windows_a_runs_tables_are_kept_within_its_physical_ram(tmp_path, monkeypatch):
    """Beamline 2026-10-23 (Windows, 64 GB): os.sysconf has no physical RAM there, so a Run kept only its funded
    integrator reserve, Cores x 1 GiB. A Rayonix table set is about 2 GB, so every 2-frame scan's second worker rebuilt
    one (about 8 s on that PC, against 1.2 s for a 1-frame scan). Windows' own RAM figure now lets the plan's unspent
    envelope count, as everywhere else."""
    import sys

    from xdart.gui.tabs.scattering.adapters import dynamic_output
    from xrd_tools.core import staging
    from xrd_tools.session.policy import INTEGRATOR_RESERVE_BYTES
    assert (staging.windows_physical_ram_bytes() is None) is (sys.platform != "win32")
    fits = []
    real_fit = core.WarmIntegrators.fit

    def fit(store, limit, allowance=None):
        fits.append((limit, allowance))
        return real_fit(store, limit, allowance)

    monkeypatch.setattr(core.WarmIntegrators, "fit", fit)
    monkeypatch.setattr(dynamic_output, "total_physical_ram_bytes", lambda: None)
    monkeypatch.setattr(dynamic_output, "windows_physical_ram_bytes", lambda: 64 << 30, raising=False)
    _directory_run(tmp_path, ("posa",), cores=4, request_value=9531)
    planned = [(limit, allowance) for limit, allowance in fits if allowance is not None]
    assert planned and all(allowance > limit * INTEGRATOR_RESERVE_BYTES for limit, allowance in planned), planned


def test_an_executor_keeps_a_runs_tables_for_its_next_run(tmp_path, builds):
    """Beamline 2026-10-23: "still slow for the first frame". Each Run emptied its store at its end, so every Run's
    first frame rebuilt pyFAI's tables (about 8 s for a Rayonix on that PC). The page's executor keeps each kind's
    latest tables, so a second Run with the same settings builds none."""
    from tests.xdart.scattering.test_p1b_output_graph import _drain_until
    executor = StandardRunExecutor(join_timeout=2.0)
    for number, stem in enumerate(_STEMS[:2]):
        run = tmp_path / f"run{number}"
        _series(run / "raw", (stem,))
        identity = _start(executor, _intent(run, run / "raw", cores=1), request_value=9541 + number)
        events = _drain_until(executor, lambda values: any(event.kind in _TERMINAL for event in values))
        executor.close(identity)
        assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"], events[-1]
    assert len(builds) == 1, builds  # on the parent: one per Run
    assert len(executor._warm_tables) == 1


@pytest.mark.parametrize(("frames", "warm", "table_bytes", "pixels", "expected"), (
    (5, 1, 2171788924, 16_777_216, 1),   # the beamline's scans after the first: one built set, keep to it
    (5, 4, 2171788924, 16_777_216, 4),   # four sets already built: use them all
    (3, 0, 0, 16_777_216, 1),            # the first scan of a large detector, before any set is measured
    (40, 1, 2171788924, 16_777_216, 3),  # a long scan repays two more builds
    (5, 0, 100 << 20, 16_777_216, 4),    # cheap tables: every granted worker
    (5, 0, 0, 1_000_000, 4),             # a small detector, unmeasured: every granted worker
))
def test_a_series_starts_the_workers_its_tables_can_repay(frames, warm, table_bytes, pixels, expected):
    """Beamline 2026-10-23 (Windows, Rayonix, 3-5 frame scans): "i ran with 1,2,3 and 4 cores. 1, 2 and 3 cores all
    took about 52-55s. 4 cores took 101s". Each worker builds its own 2.17 GB table set (~8 s on that PC, ~0.85 s a
    frame), which a few frames never repay."""
    from xdart.gui.tabs.scattering.adapters.dynamic_output import _series_workers
    assert _series_workers(4, frames, warm, table_bytes, pixels) == expected


def test_a_large_detectors_short_series_run_on_the_workers_with_built_tables(tmp_path, monkeypatch, builds):
    """With Cores 4, three 5-frame scans of a large detector start one worker each and build one table set."""
    from xdart.gui.tabs.scattering.adapters import dynamic_output
    monkeypatch.setattr(dynamic_output, "_LARGE_DETECTOR_PIXELS", 1, raising=False)  # TINY counts as large
    monkeypatch.setattr(dynamic_output, "_LARGE_TABLE_BYTES", 1, raising=False)
    executors = []
    real_open = dynamic_output.open_headless_scan_session

    def opening(*args, **kwargs):
        executors.append(kwargs["executor"])
        return real_open(*args, **kwargs)

    monkeypatch.setattr(dynamic_output, "open_headless_scan_session", opening)
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    for stem in _STEMS:
        sidecar = _Sidecar(raw, TINY, stem)
        for number in range(1, 6):
            sidecar.frame(number)
    executor, identity, events = _run_to_terminal(_intent(tmp_path, raw, cores=4), request_value=9551)
    executor.close(identity)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
    assert executors == [1, 1, 1] and len(builds) == 1, (executors, builds)
