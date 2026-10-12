"""LIVE-METADATA-DISCOVERY-RACE-20261012 frozen M2 oracle (Claude, October 12): GS1-GS2.

A real live Run (StandardRunExecutor on a watched directory whose images carry .txt sidecars). Stop is pressed while
the Run's own thread is validating a planned source: the live loop's validation is held at its entry until Stop has
been pressed, then runs for real and meets the cancellation it was asked to honour.
"""
import threading

from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_live_metadata_race_run import _arrive, _labels, _terminal
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _drain_until, _start
from xdart.gui.tabs.scattering.adapters import run_executor
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec


def _run(tmp_path, request_value):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _arrive(raw, 1)
    poni = tmp_path / 'cal.poni'
    write_poni(poni)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=('.tif',), metadata_format='txt'), poni_file=str(poni),
        project_root=str(raw), save_path=str(tmp_path / 'processed'), output_mode='Overwrite',
        processing_mode='Int 1D', live_mode=True, max_cores=1, bai_1d_args={'npt': 8, 'method': 'numpy'}),
        request_value=request_value)
    return raw, executor, identity


def _held(monkeypatch, select):
    """The live loop's validations, the first one ``select`` admits held at entry until ``pressed`` is set."""
    inside, pressed, real = threading.Event(), threading.Event(), run_executor.validate_planned_source

    def validate(item, **kwargs):
        if not inside.is_set() and select(item):
            inside.set()
            pressed.wait(30.)
        return real(item, **kwargs)
    monkeypatch.setattr(run_executor, 'validate_planned_source', validate)
    return inside, pressed


def _stop_and_close(executor, identity, events, pressed):
    pressed.set()
    if not _terminal(events):
        executor.stop(identity)
        try:
            events += _drain_until(executor, _terminal, timeout=60.)
        except AssertionError:
            pass
    executor.close(identity)
    return [(event.kind, event.detail) for event in events if event.kind in _TERMINAL]


def test_gs1_stop_while_the_live_loop_rechecks_a_settled_group_ends_stopped(tmp_path, monkeypatch):
    armed = threading.Event()
    inside, pressed = _held(monkeypatch, lambda _item: armed.is_set())
    _raw, executor, identity = _run(tmp_path, 2621)
    events = []
    try:
        events += _drain_until(executor, lambda values: 1 in _labels(values), timeout=60.)
        armed.set()  # frame 1 is reduced and settles: the loop's next validation re-checks its group
        assert inside.wait(30.), 'the live loop re-validated the settled group'
        executor.stop(identity)
        pressed.set()
        events += _drain_until(executor, _terminal, timeout=60.)
    finally:
        terminal = _stop_and_close(executor, identity, events, pressed)
    assert [kind for kind, _detail in terminal] == [StandardEventKind.STOPPED], terminal


def test_gs2_stop_while_a_new_frame_is_validated_for_reduction_ends_stopped(tmp_path, monkeypatch):
    inside, pressed = _held(monkeypatch, lambda item: len(item.source_stamp.members) == 2)
    raw, executor, identity = _run(tmp_path, 2622)
    events = []
    try:
        events += _drain_until(executor, lambda values: 1 in _labels(values), timeout=60.)
        _arrive(raw, 2)
        assert inside.wait(30.), "the live loop validated frame 2's group before reducing it"
        executor.stop(identity)
        pressed.set()
        events += _drain_until(executor, _terminal, timeout=60.)
    finally:
        terminal = _stop_and_close(executor, identity, events, pressed)
    assert [kind for kind, _detail in terminal] == [StandardEventKind.STOPPED], terminal
