"""Focused standalone-Calibrate launch and discovery oracles."""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
import signal
import subprocess
from threading import Event

import h5py
import numpy as np
import pytest
import tifffile
from silx.io.url import DataUrl

from xdart.gui.tabs.scattering import experiment_authoring as authoring
from xdart.gui.tabs.scattering.adapters.external_operation import OperationSlot
from xdart.gui.tabs.scattering.experiment_authoring import (
    CalibrationResult,
    prepare_calibration_request,
    run_calibration,
)
from xdart.gui.tabs.scattering.operation_values import (
    OperationContextStamp,
    OperationIdentity,
    OperationTerminalStatus,
)


_PONI = """poni_version: 2.1
Detector: Pilatus300kw
Detector_config: {"orientation":3}
Distance: 0.1234
Poni1: 0.05
Poni2: 0.06
Rot1: 0.01
Rot2: 0.02
Rot3: 0.03
Wavelength: 1e-10
"""


def _binary(tmp_path: Path, monkeypatch) -> Path:
    binary = tmp_path / "bin" / "pyFAI-calib2"
    binary.parent.mkdir(exist_ok=True)
    binary.write_text("fixture", encoding="utf-8")
    binary.chmod(0o700)
    monkeypatch.setenv("PATH", str(binary.parent))
    return binary.resolve()


def _tiff(path: Path) -> Path:
    tifffile.imwrite(path, np.arange(8, dtype=np.uint16).reshape(2, 4))
    return path


def _install_process(monkeypatch, hook, *, code=0):
    calls = []

    class Process:
        pid = 4242

        def __init__(self, argv, **options):
            calls.append((tuple(argv), options))
            hook(tuple(argv), options)

        def wait(self, *, timeout):
            return code

        def terminate(self):
            calls.append("terminate")

        def kill(self):
            calls.append("kill")

    monkeypatch.setattr(authoring, "_popen", Process)
    return calls


def _direct(request, *, seal=lambda _identity: True, cancelled=None):
    progress = []
    terminal = run_calibration(
        request,
        OperationIdentity(1),
        Event() if cancelled is None else cancelled,
        lambda *value: progress.append(value),
        seal,
    )
    assert type(terminal.payload) is CalibrationResult
    return terminal, progress


def _write_candidate(options, name="made.poni", payload=_PONI):
    Path(options["cwd"], name).write_text(payload, encoding="utf-8")


def test_tiff_is_preflighted_in_worker_and_launched_as_positional_input(
    tmp_path, monkeypatch,
) -> None:
    binary = _binary(tmp_path, monkeypatch)
    source = _tiff(tmp_path / "source.tiff")
    heavy = []
    real_qualify = authoring._qualify_tiff
    monkeypatch.setattr(
        authoring, "_qualify_tiff",
        lambda *args: (heavy.append("worker"), real_qualify(*args))[1],
    )
    request = prepare_calibration_request(str(source))
    assert heavy == []
    calls = _install_process(
        monkeypatch, lambda _argv, options: _write_candidate(options),
    )
    terminal, progress = _direct(request)
    result = terminal.payload
    assert terminal.status is OperationTerminalStatus.RETURNED
    assert [item[0] for item in progress] == ["launch", "discover"]
    assert heavy == ["worker"]
    argv, options = calls[0]
    assert argv == (str(binary), str(source.resolve()))
    assert "--poni" not in argv and "-o" not in argv
    assert options["cwd"] == str(tmp_path)
    assert options["shell"] is False
    assert options["stdin"] is options["stdout"] is options["stderr"] is subprocess.DEVNULL
    assert options["close_fds"] is True
    assert ("creationflags" in options) is authoring._WINDOWS
    assert ("start_new_session" in options) is not authoring._WINDOWS
    assert tuple(item.path for item in result.candidates) == (
        str(tmp_path / "made.poni"),
    )


@pytest.mark.parametrize(
    "suffix", (".tif", ".tiff", ".edf", ".cbf", ".img", ".mar3450", ".raw"),
)
def test_every_advertised_non_hdf_source_is_a_canonical_positional_input(
    tmp_path, monkeypatch, suffix,
) -> None:
    binary = _binary(tmp_path, monkeypatch)
    source = tmp_path / f"source{suffix}"
    if suffix in {".tif", ".tiff"}:
        _tiff(source)
    else:
        source.write_bytes(b"detector image")
    request = prepare_calibration_request(str(source))
    calls = _install_process(monkeypatch, lambda *_args: None)
    terminal, _progress = _direct(request)
    assert terminal.status is OperationTerminalStatus.RETURNED
    assert calls[0][0] == (str(binary), str(source.resolve()))


@pytest.mark.parametrize("suffix", (".h5", ".hdf5", ".nxs", ".nexus"))
def test_plain_multiframe_hdf_and_nexus_launch_without_positional_input(
    tmp_path, monkeypatch, suffix,
) -> None:
    binary = _binary(tmp_path, monkeypatch)
    source = tmp_path / f"master{suffix}"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("entry/data", data=np.ones((3, 2, 2), dtype="u2"))
    request = prepare_calibration_request(str(source))
    calls = _install_process(
        monkeypatch, lambda _argv, options: _write_candidate(options),
    )
    terminal, _progress = _direct(request)
    assert terminal.status is OperationTerminalStatus.RETURNED
    assert calls[0][0] == (str(binary),)
    assert terminal.payload.argv == (str(binary),)


@pytest.mark.parametrize("phase", ("prelaunch", "postprocess"))
def test_source_parent_rename_and_symlink_swap_is_never_accepted(
    tmp_path, monkeypatch, phase,
) -> None:
    _binary(tmp_path, monkeypatch)
    directory = tmp_path / "source-dir"
    directory.mkdir()
    source = _tiff(directory / "source.tif")
    request = prepare_calibration_request(str(source))
    moved = tmp_path / "moved-dir"

    def swap():
        directory.rename(moved)
        directory.symlink_to(moved, target_is_directory=True)

    if phase == "prelaunch":
        swap()
        calls = _install_process(
            monkeypatch, lambda *_args: pytest.fail("child launched"),
        )
    else:
        calls = _install_process(monkeypatch, lambda *_args: swap())
    terminal, _progress = _direct(request)
    assert terminal.status is OperationTerminalStatus.FAILED
    assert "source context changed" in terminal.diagnostic
    assert len(calls) == (0 if phase == "prelaunch" else 1)


def test_exact_owned_hdf_frame_url_is_preserved_and_hostile_link_refuses(
    tmp_path, monkeypatch,
) -> None:
    binary = _binary(tmp_path, monkeypatch)
    source = tmp_path / "owned.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("entry/data", data=np.ones((3, 2, 2), dtype="u2"))
    url = DataUrl(
        file_path=str(source), data_path="/entry/data", data_slice=(1,),
        scheme="silx",
    ).path()
    request = prepare_calibration_request(url)
    calls = _install_process(
        monkeypatch, lambda _argv, options: _write_candidate(options),
    )
    terminal, _progress = _direct(request)
    assert terminal.status is OperationTerminalStatus.RETURNED
    assert calls[0][0] == (str(binary), url)

    remote = tmp_path / "remote.h5"
    with h5py.File(remote, "w") as handle:
        handle.create_dataset("pixels", data=np.ones((3, 2, 2), dtype="u2"))
    hostile = tmp_path / "hostile.h5"
    with h5py.File(hostile, "w") as handle:
        entry = handle.create_group("entry")
        entry["data"] = h5py.ExternalLink(remote.name, "/pixels")
    hostile_url = DataUrl(
        file_path=str(hostile), data_path="/entry/data", data_slice=(1,),
        scheme="silx",
    ).path()
    hostile_request = prepare_calibration_request(hostile_url)
    before = len(calls)
    refused, _progress = _direct(hostile_request)
    assert refused.status is OperationTerminalStatus.FAILED
    assert "locally owned" in refused.diagnostic
    assert len(calls) == before


def test_exact_hdf_url_rebinds_relative_and_symlink_aliases_to_canonical_file(
    tmp_path, monkeypatch,
) -> None:
    binary = _binary(tmp_path, monkeypatch)
    directory = tmp_path / "data"
    directory.mkdir()
    source = directory / "owned.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset(
            "entry/data", data=np.ones((3, 2, 2), dtype="u2"),
        )
    monkeypatch.chdir(tmp_path)
    relative = DataUrl(
        file_path="data/owned.h5", data_path="/entry/data",
        data_slice=(1,), scheme="silx",
    ).path()
    request = prepare_calibration_request(relative)
    canonical = DataUrl(
        file_path=str(source.resolve()), data_path="/entry/data",
        data_slice=(1,), scheme="silx",
    ).path()
    assert request.source_path == str(source.resolve())
    assert request.exact_hdf_url == canonical
    calls = _install_process(monkeypatch, lambda *_args: None)
    assert _direct(request)[0].status is OperationTerminalStatus.RETURNED
    assert calls[-1][0] == (str(binary), canonical)

    alias = directory / "alias.h5"
    alias.symlink_to(source)
    aliased = DataUrl(
        file_path=str(alias), data_path="/entry/data", data_slice=(2,),
        scheme="silx",
    ).path()
    request = prepare_calibration_request(aliased)
    expected = DataUrl(
        file_path=str(source.resolve()), data_path="/entry/data",
        data_slice=(2,), scheme="silx",
    ).path()
    hostile = directory / "hostile.h5"
    with h5py.File(hostile, "w") as handle:
        handle.create_dataset("entry/data", data=np.zeros((3, 2, 2)))
    alias.unlink()
    alias.symlink_to(hostile)
    assert request.exact_hdf_url == expected
    with pytest.raises(ValueError, match="calibration request"):
        replace(request, exact_hdf_url=aliased)
    assert _direct(request)[0].status is OperationTerminalStatus.RETURNED
    assert calls[-1][0] == (str(binary), expected)


def test_hdf_url_rank_and_slice_select_exactly_one_numeric_frame(
    tmp_path, monkeypatch,
) -> None:
    binary = _binary(tmp_path, monkeypatch)
    source = tmp_path / "ranked.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("two", data=np.ones((2, 4), dtype="u2"))
        handle.create_dataset("three", data=np.ones((3, 2, 4), dtype="u2"))
    calls = _install_process(monkeypatch, lambda *_args: None)
    unsliced = DataUrl(
        file_path=str(source), data_path="/two", scheme="silx",
    ).path()
    assert _direct(prepare_calibration_request(unsliced))[0].status \
        is OperationTerminalStatus.RETURNED
    assert calls[-1][0] == (str(binary), unsliced)

    rank_two_slice = DataUrl(
        file_path=str(source), data_path="/two", data_slice=(0,),
        scheme="silx",
    ).path()
    before = len(calls)
    refused = _direct(prepare_calibration_request(rank_two_slice))[0]
    assert refused.status is OperationTerminalStatus.FAILED
    assert "bounded numeric frame" in refused.diagnostic
    assert len(calls) == before

    rank_three_slice = DataUrl(
        file_path=str(source), data_path="/three", data_slice=(1,),
        scheme="silx",
    ).path()
    assert _direct(prepare_calibration_request(rank_three_slice))[0].status \
        is OperationTerminalStatus.RETURNED
    assert calls[-1][0] == (str(binary), rank_three_slice)

    before = len(calls)
    external_raw = tmp_path / "external-raw.h5"
    with h5py.File(external_raw, "w") as handle:
        handle.create_dataset(
            "entry/data", shape=(3, 2, 2), dtype="u2",
            external=[("pixels.raw", 0, h5py.h5f.UNLIMITED)],
        )
    external_url = DataUrl(
        file_path=str(external_raw), data_path="/entry/data",
        data_slice=(1,), scheme="silx",
    ).path()
    external_request = prepare_calibration_request(external_url)
    refused, _progress = _direct(external_request)
    assert refused.status is OperationTerminalStatus.FAILED
    assert "bounded numeric frame" in refused.diagnostic
    assert len(calls) == before


def test_direct_child_discovery_filters_and_ranks_new_and_changed_candidates(
    tmp_path, monkeypatch,
) -> None:
    _binary(tmp_path, monkeypatch)
    source = _tiff(tmp_path / "source.tif")
    same = tmp_path / "same.poni"; same.write_text(_PONI)
    changed = tmp_path / "changed.poni"; changed.write_text(_PONI)
    nested = tmp_path / "nested"; nested.mkdir()
    request = prepare_calibration_request(str(source))

    def authored(_argv, options):
        directory = Path(options["cwd"])
        changed.write_text(_PONI.replace("0.1234", "0.2234"))
        older = directory / "older.poni"; older.write_text(_PONI)
        newer = directory / "newer.poni"; newer.write_text(_PONI)
        tie = directory / "alpha.poni"; tie.write_text(_PONI)
        (directory / "invalid.poni").write_text("not a PONI")
        (directory / "oversized.poni").write_bytes(b"x" * ((1 << 20) + 1))
        (directory / "linked.poni").symlink_to(newer)
        (nested / "nested.poni").write_text(_PONI)
        base = 1_800_000_000_000_000_000
        os.utime(older, ns=(base, base))
        os.utime(changed, ns=(base + 1, base + 1))
        os.utime(newer, ns=(base + 2, base + 2))
        os.utime(tie, ns=(base + 2, base + 2))

    _install_process(monkeypatch, authored)
    terminal, _progress = _direct(request)
    assert terminal.status is OperationTerminalStatus.RETURNED
    assert tuple(Path(item.path).name for item in terminal.payload.candidates) == (
        "alpha.poni", "newer.poni", "changed.poni", "older.poni",
    )
    assert same.read_text() == _PONI
    assert (nested / "nested.poni").exists()


def test_unstable_and_unchanged_candidates_are_inert_and_zero_is_returned(
    tmp_path, monkeypatch,
) -> None:
    _binary(tmp_path, monkeypatch)
    source = _tiff(tmp_path / "source.tif")
    (tmp_path / "unchanged.poni").write_text(_PONI)
    request = prepare_calibration_request(str(source))
    real_qualify = authoring._qualify

    def unstable(path):
        proof = real_qualify(path)
        if path.name == "unstable.poni":
            path.write_text(_PONI.replace("0.1234", "0.3234"))
        return proof

    monkeypatch.setattr(authoring, "_qualify", unstable)
    _install_process(
        monkeypatch,
        lambda _argv, options: _write_candidate(options, "unstable.poni"),
    )
    terminal, _progress = _direct(request)
    assert terminal.status is OperationTerminalStatus.RETURNED
    assert terminal.payload.candidates == ()


def test_existing_poni_chooser_rejects_lexical_symlink_before_qualification(
    tmp_path, monkeypatch,
) -> None:
    target = tmp_path / "target.poni"
    target.write_text(_PONI, encoding="utf-8")
    linked = tmp_path / "linked.poni"
    linked.symlink_to(target)
    monkeypatch.setattr(
        authoring, "_qualify",
        lambda *_args: pytest.fail("lexical symlink reached PONI qualification"),
    )
    with pytest.raises(ValueError, match="non-symlink"):
        authoring.qualify_calibration_candidate(str(linked))


def test_inventory_bounds_and_source_drift_refuse_before_child(
    tmp_path, monkeypatch,
) -> None:
    _binary(tmp_path, monkeypatch)
    source = _tiff(tmp_path / "source.tif")
    request = prepare_calibration_request(str(source))
    (tmp_path / "one").write_text("x")
    (tmp_path / "two").write_text("x")
    calls = _install_process(monkeypatch, lambda *_args: pytest.fail("child launched"))
    monkeypatch.setattr(authoring, "_DIRECT_CHILD_LIMIT", 1)
    terminal, _progress = _direct(request)
    assert terminal.status is OperationTerminalStatus.FAILED
    assert "direct-child limit" in terminal.diagnostic and calls == []
    monkeypatch.setattr(authoring, "_DIRECT_CHILD_LIMIT", 4096)
    first = tmp_path / "first.poni"; first.write_text(_PONI)
    second = tmp_path / "second.poni"; second.write_text(_PONI)
    monkeypatch.setattr(
        authoring, "_PONI_AGGREGATE_BYTES_LIMIT",
        first.stat().st_size + second.stat().st_size - 1,
    )
    terminal, _progress = _direct(request)
    assert terminal.status is OperationTerminalStatus.FAILED
    assert "PONI byte limit" in terminal.diagnostic and calls == []
    monkeypatch.setattr(authoring, "_PONI_AGGREGATE_BYTES_LIMIT", 16 << 20)
    source.write_bytes(b"drift")
    terminal, _progress = _direct(request)
    assert terminal.status is OperationTerminalStatus.FAILED and calls == []


def test_prepare_and_begin_do_not_scan_or_decode_on_calling_thread(
    tmp_path, monkeypatch,
) -> None:
    _binary(tmp_path, monkeypatch)
    source = _tiff(tmp_path / "source.tif")
    monkeypatch.setattr(
        authoring, "_qualify_tiff",
        lambda *_args: pytest.fail("GUI prepare decoded TIFF"),
    )
    monkeypatch.setattr(
        authoring, "_poni_inventory",
        lambda *_args: pytest.fail("GUI prepare scanned directory"),
    )
    request = prepare_calibration_request(str(source))
    captured = []
    slot = OperationSlot()
    monkeypatch.setattr(
        slot, "_begin",
        lambda *args: captured.append(args) or OperationIdentity(7),
    )
    assert slot.begin_calibrate(request, OperationContextStamp(0)) == OperationIdentity(7)
    assert captured and captured[0][0] is request


def test_nonzero_cancel_and_platform_signals_have_zero_application_effect(
    tmp_path, monkeypatch,
) -> None:
    _binary(tmp_path, monkeypatch)
    source = _tiff(tmp_path / "source.tif")
    request = prepare_calibration_request(str(source))
    calls = _install_process(
        monkeypatch, lambda _argv, options: _write_candidate(options), code=7,
    )
    terminal, _progress = _direct(request)
    assert terminal.status is OperationTerminalStatus.FAILED
    assert terminal.payload.candidates == () and len(calls) == 1
    cancelled = Event(); cancelled.set()
    before = len(calls)
    terminal, _progress = _direct(request, cancelled=cancelled)
    assert terminal.status is OperationTerminalStatus.CANCELLED
    assert len(calls) == before

    direct, groups = [], []
    process = type("Process", (), {
        "pid": 44,
        "terminate": lambda _self: direct.append("terminate"),
        "kill": lambda _self: direct.append("kill"),
    })()
    monkeypatch.setattr(authoring, "_WINDOWS", False)
    monkeypatch.setattr(authoring, "_killpg", lambda pid, sig: groups.append((pid, sig)))
    assert authoring._signal_child(process, kill=False) == ""
    assert groups == [(44, signal.SIGTERM)] and direct == []
    monkeypatch.setattr(authoring, "_WINDOWS", True)
    assert authoring._signal_child(process, kill=True) == ""
    assert direct == ["kill"]
