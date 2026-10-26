"""A finite Run's format, timeout, omission accounting and committed membership have one owner."""
import os
import threading
import time
import pytest

from tests.xdart.scattering.test_directory_metadata_type import _intent, _aged
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame, _output, _wait
from tests.xdart.scattering.test_p1b_output_graph import _run_to_terminal, _start, _TERMINAL
from xdart.gui.tabs.scattering.adapters import run_executor
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xrd_tools.reduction.background import FrameBackgroundPlan
from xrd_tools.sources.selection import DirectorySourceSpec


def test_auto_qualification_uses_the_first_scans_resolved_format(tmp_path):
    raw = tmp_path / 'raw'
    (raw / 'A').mkdir(parents=True)
    (raw / 'B').mkdir()
    _frame(raw / 'A', 'a', 1, 'pdi', 2.)
    _frame(raw / 'B', 'b', 1, 'pdi', 7.)
    _frame(raw / 'B', 'b', 1, 'txt', 99., image=False)
    intent = _intent(tmp_path, 'auto')
    intent.source_spec = DirectorySourceSpec(raw, suffixes=('.tif',), metadata_format='auto', recursive=True)
    executor, identity, events = _run_to_terminal(intent, request_value=7901)
    executor.close(identity)
    assert next(event for event in events if event.kind in _TERMINAL).kind is StandardEventKind.FINISHED
    assert _output(tmp_path, 'b')[:2] == ((1,), (7.,)), 'Auto parsed TXT after binding this Run to PDI'


def test_future_server_mtime_cannot_extend_the_last_frame_wait(tmp_path, monkeypatch):
    monkeypatch.setitem(run_executor._METADATA_WAIT_LIMIT_S, 'sidecar', .7)
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'scan', 1, 'txt', 1.)
    image = _frame(raw, 'scan', 2, 'txt', 2., sidecar=False)
    future = time.time() + 30.
    os.utime(image, (future, future))
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path), request_value=7902)
    stop = threading.Timer(5., lambda: executor.stop(identity))
    stop.start()
    events = []
    try:
        _wait(executor, events, lambda values: any(event.kind in _TERMINAL for event in values), timeout=10.)
    finally:
        stop.cancel()
        executor.close(identity)
    assert next(event for event in events if event.kind in _TERMINAL).kind is StandardEventKind.FINISHED, (
        'a future server timestamp made the Run wait beyond its limit until Stop')


@pytest.mark.parametrize('late_sidecar', (False, True))
def test_retrying_metadata_admission_counts_each_omission_once(tmp_path, monkeypatch, late_sidecar):
    import xrd_tools.io.metadata as metadata
    raw = tmp_path / 'raw'
    raw.mkdir()
    for label, sidecar in ((1, True), (2, False), (3, True)):
        _frame(raw, 'scan', label, 'txt', float(label), sidecar=sidecar)
    armed, changed = threading.Event(), []
    filter_entry = StandardRunExecutor._without_unsidecarred
    read_txt = metadata.read_txt_metadata

    def filtered(self, *args):
        result = filter_entry(self, *args)
        armed.set()
        return result

    def read(path, *args, **kwargs):
        if armed.is_set() and not changed:
            changed.append(str(path))
            path.write_text(path.read_text().replace('I0 = 1.0', 'I0 = 8.0'))
            if late_sidecar:
                _frame(raw, 'scan', 2, 'txt', 2., image=False)
        return read_txt(path, *args, **kwargs)

    monkeypatch.setattr(StandardRunExecutor, '_without_unsidecarred', filtered)
    monkeypatch.setattr(metadata, 'read_txt_metadata', read)
    executor, identity, events = _run_to_terminal(_intent(tmp_path), request_value=7903)
    run = executor._exact_run(identity)
    skipped, left_out = run.files_skipped, tuple(run.metadata_left_out)
    executor.close(identity)
    terminal = next(event for event in events if event.kind in _TERMINAL)
    assert changed, 'the real metadata read was reached after filtering'
    assert terminal.kind is StandardEventKind.FINISHED, terminal.detail
    assert skipped == 1 and len(left_out) == 1, (skipped, left_out)
    assert _output(tmp_path)[:2] == ((1, 2), (8., 3.))


@pytest.mark.parametrize('rewritten', (False, True))
def test_a_background_append_never_leaves_out_an_already_committed_image(tmp_path, rewritten):
    raw = tmp_path / 'raw'
    raw.mkdir()
    background = _frame(tmp_path, 'bg', 1, 'txt', 1., sidecar=False)
    _frame(raw, 'scan', 1, 'txt', 1.)
    intent = _intent(tmp_path)
    intent.output_mode = 'Append'
    intent.background = FrameBackgroundPlan(mode='Single BG File', locator=str(background))
    seed, seed_id, seeded = _run_to_terminal(intent, request_value=7904)
    seed.close(seed_id)
    assert next(event for event in seeded if event.kind in _TERMINAL).kind is StandardEventKind.FINISHED
    (raw / 'scan_0001.txt').unlink()
    if rewritten:
        _frame(raw, 'scan', 1, 'txt', 1., sidecar=False)
    _frame(raw, 'scan', 2, 'txt', 2.)
    executor, identity, events = _run_to_terminal(intent, request_value=7905)
    run = executor._exact_run(identity)
    left_out = tuple(run.metadata_left_out)
    executor.close(identity)
    if rewritten:
        assert _output(tmp_path)[:2] == ((1,), (1.,)), 'a changed raw image was reprocessed without its chosen TXT'
        assert left_out, 'a changed image is new work, not a committed image protected from omission'
    else:
        assert _output(tmp_path)[0] == (1, 2), (left_out, [event.detail for event in events if event.kind in _TERMINAL])
        assert not left_out, 'a saved frame was filtered out because the optimized prefix was unavailable'


def test_a_filtered_energy_extension_can_rebuild_from_its_owned_receipt(tmp_path):
    from tests.xdart.scattering.test_energy_scan_append_grid import _frame as energy_frame
    raw = tmp_path / 'raw'
    raw.mkdir()
    intent = _intent(tmp_path)
    intent.output_mode = 'Append'
    for label, energy in enumerate((16., 16.1, 16.2), 1):
        energy_frame(raw, 'scan', label, energy)
    seed, seed_id, seeded = _run_to_terminal(intent, request_value=7906)
    seed.close(seed_id)
    assert next(event for event in seeded if event.kind in _TERMINAL).kind is StandardEventKind.FINISHED
    _aged(_frame(raw, 'scan', 4, 'txt', 4., sidecar=False))
    energy_frame(raw, 'scan', 5, 16.3)
    executor, identity, events = _run_to_terminal(intent, request_value=7907)
    executor.close(identity)
    terminal = next(event for event in events if event.kind in _TERMINAL)
    assert terminal.kind is StandardEventKind.FINISHED and terminal.rebuilt, terminal
    assert _output(tmp_path)[:2] == ((1, 2, 3, 4), (1., 2., 3., 5.))
