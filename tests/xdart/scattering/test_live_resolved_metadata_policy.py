"""A Live Run's resolved metadata choice controls all later source reads."""
from tests.xdart.scattering.test_live_auto_metadata_wait import _asked, _frame_ready, ask_soon
from tests.xdart.scattering.test_live_qualify_refusal_run import (
    _frame, _labels, _output, _run, _said, _settle, _stop, _wait,
)
from tests.xdart.scattering.test_live_metadata_series import write_poni
from tests.xdart.scattering.test_p1b_output_graph import _start, _run_to_terminal, _TERMINAL
from tests.xdart.scattering.test_live_already_processed import _intent
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec


def test_answering_no_metadata_ignores_sidecars_that_arrive_later(tmp_path, ask_soon):
    raw = tmp_path / 'raw'
    raw.mkdir()
    executor, identity = _run(tmp_path, raw, 'auto', 3520)
    events = []
    try:
        _frame(raw, 'scan', 1, 'txt', 1., sidecar=False)
        _asked(executor, identity, events)
        executor.answer_metadata_question(identity, False)
        _frame_ready(executor, events, 1)
        _frame(raw, 'scan', 2, 'txt', 2.)
        _frame_ready(executor, events, 2)
        _frame(raw, 'scan', 1, 'txt', 1., image=False)
        _frame(raw, 'scan', 3, 'txt', 3.)
        _frame_ready(executor, events, 3)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert _output(tmp_path)[:2] == ((1, 2, 3), None)


def test_auto_binds_the_first_sidecar_format_across_folders(tmp_path):
    raw = tmp_path / 'raw'
    (raw / 'A').mkdir(parents=True)
    (raw / 'B').mkdir()
    poni = tmp_path / 'cal.poni'
    write_poni(poni)
    _frame(raw / 'A', 'a', 1, 'txt', 1.)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=('.tif',), metadata_format='auto', recursive=True),
        poni_file=str(poni), project_root=str(raw), save_path=str(tmp_path / 'processed'), output_mode='Overwrite',
        processing_mode='Int 1D', live_mode=True, max_cores=1, bai_1d_args={'npt': 8, 'method': 'numpy'}),
        request_value=3521)
    events = []
    try:
        _frame_ready(executor, events, 1, 'a')
        _frame(raw / 'B', 'b', 1, 'pdi', 9.)
        _settle(executor, events, .8)
        assert _labels(events, 'b') == [], 'the TXT Run read a different folder as PDI'
        _frame(raw / 'B', 'b', 1, 'txt', 7., image=False)
        _frame_ready(executor, events, 1, 'b')
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert _output(tmp_path, 'b')[:2] == ((1,), (7.,))


def test_auto_still_extends_a_saved_output_qualified_under_auto(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'scan', 1, 'txt', 1.)
    intent = _intent(tmp_path, raw, live=False)
    intent.source_spec = DirectorySourceSpec(raw, suffixes=('.tif',), metadata_format='auto')
    seed, seed_id, seeded = _run_to_terminal(intent, request_value=3522)
    seed.close(seed_id)
    assert [event.kind.name for event in seeded if event.kind in _TERMINAL] == ['FINISHED']
    executor, identity = _run(tmp_path, raw, 'auto', 3523)
    events = []
    try:
        _settle(executor, events, .5)
        _frame(raw, 'scan', 2, 'txt', 2.)
        _frame_ready(executor, events, 2)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert _output(tmp_path)[:2] == ((1, 2), (1., 2.))


def test_a_published_frame_cannot_be_left_out_when_its_sidecar_disappears(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    executor, identity = _run(tmp_path, raw, 'txt', 3524)
    events = []
    try:
        for label in (1, 2):
            _frame(raw, 'scan', label, 'txt', float(label))
        _frame_ready(executor, events, 2)
        (raw / 'scan_0001.txt').unlink()
        _frame(raw, 'scan', 3, 'txt', 3.)
        _wait(executor, events, lambda values: _said(values, 'stopped growing'), timeout=15.)
        left_out = tuple(executor._exact_run(identity).left_out)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert left_out == (), 'a published member was removed and its series relabelled'
    assert _output(tmp_path)[0] == (1, 2)


def test_a_saved_prefix_cannot_be_left_out_when_its_sidecar_disappears(tmp_path, caplog):
    raw = tmp_path / 'raw'
    raw.mkdir()
    for label in (1, 2):
        _frame(raw, 'scan', label, 'txt', float(label))
    seed, seed_id, seeded = _run_to_terminal(_intent(tmp_path, raw, live=False), request_value=3525)
    seed.close(seed_id)
    assert [event.kind.name for event in seeded if event.kind in _TERMINAL] == ['FINISHED']
    caplog.set_level('DEBUG', logger='xdart.gui.tabs.scattering.output_preflight')
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, live=True), request_value=3526)
    events = []
    try:
        _wait(executor, events, lambda values: len(executor.frame_catalog(identity).entries) == 2)
        assert not executor.processed_live_revisions(identity), 'the prefix is adopted from its saved record'
        (raw / 'scan_0001.txt').unlink()
        _frame(raw, 'scan', 3, 'txt', 3.)
        _frame(raw, 'other', 1, 'txt', 4.)
        _frame_ready(executor, events, 1, 'other')
        _settle(executor, events, 1.)
        left_out = tuple(executor._exact_run(identity).left_out)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert left_out == (), 'a saved member was removed and its series relabelled'
    assert _output(tmp_path)[0] == (1, 2)
    assert _output(tmp_path, 'other')[0] == (1,)
