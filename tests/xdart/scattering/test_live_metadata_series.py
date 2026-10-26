"""Live keeps every series and folder going while one frame waits for its .txt (maintainer, 2026-10-26: "We want to
monitor multiple folders and integrate data in each of them as they arrive. If one folder doesn't have new data move
on to the next one and so on"). A frame waits only for its own .txt, which no other series' image ends. Real Live Runs
over real TIFF + .txt files."""
from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_live_auto_metadata_wait import _frame_ready
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame, _labels, _output, _run, _settle, _stop
from tests.xdart.scattering.test_p1b_output_graph import _start
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec


def test_interleaved_series_each_keep_their_own_txt_and_wait_for_no_other(tmp_path):
    """His cycle, in one folder: samples a, b and c measured in turn, each .txt landing half a second after its image.
    Every frame keeps its own I0; another series' image never ends a frame's wait."""
    raw = tmp_path / 'raw'
    raw.mkdir()
    executor, identity = _run(tmp_path, raw, 'txt', 3502)
    events = []
    try:
        for cycle in (1, 2):
            for stem, i0 in (('a', 1.0), ('b', 2.0), ('c', 3.0)):
                _frame(raw, stem, cycle, 'txt', i0 * cycle, sidecar=False)
                _settle(executor, events, .5)
                _frame(raw, stem, cycle, 'txt', i0 * cycle, image=False)
        for stem in ('a', 'b', 'c'):
            _frame_ready(executor, events, 2, stem)
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    for stem, i0 in (('a', 1.0), ('b', 2.0), ('c', 3.0)):
        assert _output(tmp_path, stem)[:2] == ((1, 2), (i0, 2 * i0)), stem


def test_a_folder_whose_frame_waits_for_its_txt_holds_up_no_other_folder(tmp_path):
    """Folder A's frame waits for its .txt while folder B's frames are processed as they land; A's frame is processed
    with its own .txt once it comes."""
    root = tmp_path / 'raw'
    (root / 'A').mkdir(parents=True)
    (root / 'B').mkdir()
    poni = tmp_path / 'cal.poni'
    write_poni(poni)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, RunIntent(
        source_spec=DirectorySourceSpec(root, suffixes=('.tif',), metadata_format='txt', recursive=True),
        poni_file=str(poni), project_root=str(root), save_path=str(tmp_path / 'processed'), output_mode='Overwrite',
        processing_mode='Int 1D', live_mode=True, max_cores=1, bai_1d_args={'npt': 8, 'method': 'numpy'}),
        request_value=3511)
    events = []
    try:
        _frame(root / 'A', 'a', 1, 'txt', 1.0, sidecar=False)  # its .txt has not come
        for index in (1, 2):
            _frame(root / 'B', 'b', index, 'txt', 2.0 * index)
            _frame_ready(executor, events, index, 'b')  # B goes on meanwhile
        held = 1 in _labels(events, 'a')
        _frame(root / 'A', 'a', 1, 'txt', 1.0, image=False)
        _frame_ready(executor, events, 1, 'a')
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert not held, 'a frame was processed without its .txt'
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert _output(tmp_path, 'a')[:2] == ((1,), (1.0,)) and _output(tmp_path, 'b')[:2] == ((1, 2), (2.0, 4.0))
