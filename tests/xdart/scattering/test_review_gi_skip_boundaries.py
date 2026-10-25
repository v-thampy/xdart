"""Real eager-GI admission and producer-error continuation checks."""
import pytest
from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_live_never_waits import _gi_frame, _stack
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec


def _intent(tmp_path, raw, *, npt=8, containers=False):
    poni = tmp_path / "cal.poni"
    if not poni.exists():
        write_poni(poni)
    return RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=(".nxs",) if containers else (".tif",),
                                       metadata_format=None if containers else "txt"),
        poni_file=str(poni), project_root=str(raw), save_path=str(tmp_path / "processed"),
        output_mode="Append", processing_mode="Int 1D", live_mode=False, max_cores=1,
        bai_1d_args={"npt": npt, "method": "numpy"},
        gi=GIIntent(enabled=True, incidence_motor="Manual" if containers else "th"))


def _run(intent, number):
    executor, identity, events = _run_to_terminal(intent, request_value=number)
    executor.close(identity)
    return [event for event in events if event.kind in _TERMINAL]


def test_an_all_append_refused_gi_directory_finishes_with_its_scan_named(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for k in (1, 2):
        _gi_frame(raw, "scan", k, float(k), th=.1 * k)
    assert [event.kind.name for event in _run(_intent(tmp_path, raw), 9830)] == ["FINISHED"]
    output = tmp_path / "processed" / "scan_gi_int1d.nexus"
    saved = output.read_bytes()
    terminal = _run(_intent(tmp_path, raw, npt=9), 9831)  # real incompatible Append settings
    print("ALL-SKIPPED-GI", [(event.kind.name, event.detail, event.warnings) for event in terminal])
    assert output.read_bytes() == saved
    assert [event.kind.name for event in terminal] == ["FINISHED"], terminal
    assert any("scan_gi_int1d.nexus" in note for note in terminal[0].warnings)


def test_a_refused_gi_scans_motor_choices_do_not_fence_the_surviving_scan(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for k in (1, 2):
        _gi_frame(raw, "bad", k, float(k), th=.1 * k)
    assert [event.kind.name for event in _run(_intent(tmp_path, raw, npt=9), 9835)] == ["FINISHED"]
    output = tmp_path / "processed" / "bad_gi_int1d.nexus"
    saved = output.read_bytes()
    for k in (1, 2):
        _gi_frame(raw, "good", k, float(k), th=.1 * k)
        sidecar = raw / f"good_{k:04d}.txt"
        sidecar.write_text(sidecar.read_text().replace("energy = 16", "extra_motor = 5, energy = 16"))
    terminal = _run(_intent(tmp_path, raw), 9836)
    assert output.read_bytes() == saved
    assert [event.kind.name for event in terminal] == ["FINISHED"], terminal
    assert any("bad_gi_int1d.nexus" in note for note in terminal[0].warnings)
    assert (tmp_path / "processed" / "good_gi_int1d.nexus").exists()


@pytest.mark.parametrize("corrupt", [1, 2])
def test_an_eager_gi_container_decode_failure_skips_its_scan_and_runs_the_next(tmp_path, corrupt):
    raw = tmp_path / "raw"
    raw.mkdir()
    _stack(raw / "bad.nxs", 3, corrupt=corrupt)  # real middle-frame or last-scout decode failure
    _stack(raw / "good.nxs", 3)
    terminal = _run(_intent(tmp_path, raw, containers=True), 9832)
    print("GI-CONTAINER-READ", [(event.kind.name, event.detail, event.warnings) for event in terminal])
    assert [event.kind.name for event in terminal] == ["FINISHED"], terminal
    assert any("bad_gi_int1d.nexus" in note for note in terminal[0].warnings)
    assert (tmp_path / "processed" / "good_gi_int1d.nexus").exists()
