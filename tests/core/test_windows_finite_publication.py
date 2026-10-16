"""WINDOWS-FINITE-PUBLICATION-20260911: frozen contract rows W01–W46 (W35/W39 retired under D6).

Native credit only from actual Windows hosts (local NTFS and the approved SMB share), real handles and
files. On a descriptor host the ``host`` arm keeps the POSIX dir_fd branch; ``named`` forces
``finite_artifact._DESCRIPTOR_CUSTODY`` off (PR #1's ``_DESCRIPTOR_WALK`` idiom) and ``crt-emulated``
replays the CRT text-mode boundary (``test_stream_seed_preserves_binary_bytes``): DIAGNOSTIC ONLY.
Rows: W01-W08 science; W09-W12 binary bytes; W13-W24 custody; W25-W28 parent state/adoption/capacity;
W29-W34 calibration pair; W36-W38, W40 retry/flush; W41 Average reload (D1); W42 links (D2); W43-W45
identity (D3); W46 path length (T5); W47 Windows-width identities (hosted NTFS, 2026-10-16).
Applicability (T4): every row applies on NTFS and SMB, except that
W19 (hard link) and W42's symlink cell report INCOMPLETE (skip) where the link cannot be created, and
W42's junction cell is N/A on SMB (junctions need local NTFS; W16-W18 and W42's symlink cell cover it).
"""
from __future__ import annotations

from contextlib import nullcontext
import copy
from dataclasses import replace
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import threading
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import xrd_tools.io.finite_artifact as finite_module
import xrd_tools.io.output_transaction as transaction_module
from xrd_tools.io.finite_artifact import (
    FINITE_CANDIDATE_CLEANUP_WARNING, FINITE_PARENT_DIRECTORY_FSYNC_WARNING, FiniteArtifactCapacityError,
    FiniteArtifactCollision, FiniteArtifactDisposition, FiniteArtifactError, FiniteArtifactIntegrityError,
    FiniteArtifactPublisher, FiniteCandidateValidation, FiniteCommittedInspection,
    FiniteSeededDocumentAdapter, _FinitePublicationSession, capture_finite_source,
)
from xrd_tools.io.output_transaction import StreamTerminal, capture_target_snapshot, \
    get_output_transaction_coordinator
from xrd_tools.reduction.average import _seal_published
from tests.core.test_finite_artifact_publisher import _request

_WINDOWS = os.name == "nt"
_DESCRIPTOR_HOST = os.open in os.supports_dir_fd
_ARMS = ("host", "named") if _DESCRIPTOR_HOST else ("host",)

@pytest.fixture(params=_ARMS)
def arm(request, monkeypatch):
    """``named`` is the no-dir_fd arm: native on Windows, forced and diagnostic-only elsewhere. Disposal
    through a handle exists only on Windows; a forced named arm names its exact orphan instead."""
    if request.param == "named":
        monkeypatch.setattr(finite_module, "_DESCRIPTOR_CUSTODY", False)
    named = request.param == "named" or not _DESCRIPTOR_HOST
    return SimpleNamespace(named=named, disposes=_WINDOWS or not named)

@pytest.fixture(params=("host",) if _WINDOWS else ("host", "crt-emulated"))
def crt(request, monkeypatch):
    """DIAGNOSTIC on POSIX: a descriptor opened without O_BINARY reads/writes through CRT text mode."""
    if request.param == "host":
        return
    flag, text = 0x8000, set()
    real_open, real_read, real_write, real_close = os.open, os.read, os.write, os.close

    def crt_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags & ~flag, *args, **kwargs)
        (text.discard if flags & flag else text.add)(descriptor)
        return descriptor
    def crt_read(descriptor, size):
        data = real_read(descriptor, size)
        return data.split(b"\x1a", 1)[0].replace(b"\r\n", b"\n") if descriptor in text else data
    def crt_write(descriptor, data):
        if descriptor not in text:
            return real_write(descriptor, data)
        translated = bytes(data).replace(b"\n", b"\r\n")
        assert real_write(descriptor, translated) == len(translated)
        return len(data)
    def crt_close(descriptor):
        return text.discard(descriptor) or real_close(descriptor)
    monkeypatch.setattr(os, "O_BINARY", flag, raising=False)
    for name, value in (("open", crt_open), ("read", crt_read), ("write", crt_write), ("close", crt_close)):
        monkeypatch.setattr(os, name, value)

def _leftovers(directory) -> list[Path]:
    return sorted(path for path in Path(directory).iterdir() if path.name.startswith(".xdart-finite-"))

def _start(target, values=(1, 2, 3)) -> _FinitePublicationSession:
    session = _FinitePublicationSession(target)
    document = session.start(lambda path: h5py.File(path, "w"))
    document.create_dataset("value", data=np.asarray(values, dtype=np.int64))
    assert session.close_document()
    return session

def _publish(session, expected):
    return session.publish(expected, ordinal=session.ordinal,
                           observe=lambda path, ordinal: _seal_published(str(path), ordinal))

def _values(path) -> list[int]:
    with h5py.File(path, "r") as handle:
        return handle["value"][()].tolist()

def _prior(target, values=(1,)):
    facts = _publish(_start(target, values), capture_target_snapshot(target))
    assert facts.published and facts.error is None
    return target.read_bytes(), target.stat()

def _assert_prior(target, prior) -> None:
    raw, state = prior
    assert target.read_bytes() == raw
    assert (target.stat().st_dev, target.stat().st_ino) == (state.st_dev, state.st_ino)

def _assert_discarded(session, arm, directory) -> None:
    left = _leftovers(directory)
    if arm.disposes:
        assert left == [] and session.hidden_orphan is None, (left, session.hidden_orphan)
    else:  # a forced named arm cannot dispose by handle on POSIX: it names its exact orphan
        assert len(left) == 1 and session.hidden_orphan is not None, (left, session.hidden_orphan)
        assert os.path.samefile(session.hidden_orphan, left[0])

def _raising(error):
    def fail(*_args, **_kwargs):
        raise error
    return fail

def _assert_slot_free(target) -> None:
    coordinator = get_output_transaction_coordinator()
    coordinator.release_target(coordinator.hold_target(str(target), label="probe"))

def _parent_descriptor(session) -> int:
    parent = session._parent_descriptor
    return parent if type(parent) is int else parent.descriptor

@pytest.mark.parametrize("two_d", (False, True), ids=("W01-average-1d", "W02-average-2d"))
def test_w01_w02_average_publishes_replaces_and_reopens(tmp_path, monkeypatch, arm, two_d):
    from tests.core.test_p36_average import _run_average_scan, _series, _stub_integrators
    from xrd_tools.io.frame_view import read_frame_view
    from xrd_tools.reduction import AverageScanRecipe, Integration1DPlan, Integration2DPlan, ReductionPlan
    from xrd_tools.reduction import average as average_module, core
    _stub_integrators(monkeypatch, [])
    returned = []
    stubbed = core.integrate_2d if two_d else core.integrate_1d
    monkeypatch.setattr(core, "integrate_2d" if two_d else "integrate_1d",
                        lambda *a, **k: returned.append(stubbed(*a, **k)) or returned[-1])
    source = _series(tmp_path)
    states = []
    for npt in (4, 6):
        plan = (ReductionPlan(integration_2d=Integration2DPlan(npt_rad=npt, npt_azim=3)) if two_d
                else ReductionPlan(integration_1d=Integration1DPlan(npt=npt)))
        result = _run_average_scan(AverageScanRecipe(source, tmp_path / "average.nxs", plan))
        assert result.disposition == "COMMITTED", (result.diagnostic_code, result.diagnostic)
        artifact = Path(result.target)
        view = read_frame_view(artifact, 1, include_thumbnail=False)
        if two_d:
            np.testing.assert_array_equal(view.intensity_2d, np.asarray(returned[-1].intensity).T)
        else:
            np.testing.assert_array_equal(view.intensity_1d, returned[-1].intensity)
            np.testing.assert_array_equal(view.axis_1d.values, returned[-1].radial)
        assert average_module.get_average_finite_counts(artifact).evidence.contributor_extent == 3
        assert result.commit_identity.inode == artifact.stat().st_ino
        states.append(artifact.stat())
    assert (states[0].st_dev, states[0].st_ino) != (states[1].st_dev, states[1].st_ino)
    assert _leftovers(tmp_path) == []

def _stitch_requests(tmp_path, cell):
    from xrd_tools.analysis.module_transaction import ModuleOutputRequest
    from xrd_tools.analysis.stitch_operation import prepare_stitch_operation
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, AnalysisArtifactOverwrite
    from xrd_tools.io.output_path import resolve_finite_output_target
    from tests.core.test_stitch_operation import _prepared, _xu_prepared
    backend, mode = cell
    original = _prepared(tmp_path) if backend == "multigeometry" else _xu_prepared(tmp_path)
    grid = {"npt_rad_2d": 48, "npt_azim_2d": 32, "azimuth_range": (-180.0, 180.0)} if mode == "2d" else {}
    extra = ({"q_min_A_inverse": None, "q_max_A_inverse": None, "radial_range": (1.0, 5.2)}
             if backend == "xu_hist" else {})
    kind = AnalysisArtifactKind.STITCH_2D if mode == "2d" else AnalysisArtifactKind.STITCH_1D
    family = f"{backend}-{mode}"
    target = resolve_finite_output_target(tmp_path, family, operation_token=f"stitch-{mode}")
    for npt, overwrite in ((1000, AnalysisArtifactOverwrite.CREATE_NEW), (800, AnalysisArtifactOverwrite.REPLACE)):
        plan = replace(original.plan, native_records=True, mode=mode, npt_1d=npt, **grid, **extra)
        output = ModuleOutputRequest(target, kind, overwrite, artifact_family=family)
        yield prepare_stitch_operation(original.module.source, output, plan, project_root=tmp_path)

@pytest.mark.parametrize("cell", (("multigeometry", "1d"), ("xu_hist", "1d"), ("multigeometry", "2d"),
                                  ("xu_hist", "2d")),
                         ids=("W03-stitch1d-mg", "W04-stitch1d-xu", "W05-stitch2d-mg", "W06-stitch2d-xu"))
def test_w03_w06_stitch_publishes_replaces_and_reopens(tmp_path, arm, cell):
    from xrd_tools.analysis.module_transaction import ModuleDisposition
    from xrd_tools.analysis.stitch_operation import run_stitch_operation
    from xrd_tools.io.analysis_artifact import read_analysis_artifact
    from xrd_tools.io.frame_view import read_contributing_frame_view
    states, fingerprints = [], []
    for request in _stitch_requests(tmp_path, cell):
        result = run_stitch_operation(request)
        assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
        target = Path(request.module.output.target)
        payload = read_analysis_artifact(target, expected_kind=request.module.output.kind)
        assert payload.result_fingerprint == result.terminal.commit.result_fingerprint
        assert len(payload.contributing_frames) == len(request.module.source.selected_labels)
        row = payload.contributing_frames[0]
        frame = read_contributing_frame_view(target, row["identity"]["frame_index"], full_raw=True,
                                             scan=row["key"].split("/")[0][5:], artifact_payload=payload)
        assert frame.raw is not None and frame.raw.size > 0
        states.append(target.stat())
        fingerprints.append(payload.result_fingerprint)
    assert fingerprints[0] != fingerprints[1]
    assert (states[0].st_dev, states[0].st_ino) != (states[1].st_dev, states[1].st_ino)
    assert _leftovers(target.parent) == []

@pytest.mark.parametrize("frame", ("hkl", "q"), ids=("W07-rsm-hkl", "W08-rsm-q"))
def test_w07_w08_rsm_v2_publishes_replaces_and_reopens(tmp_path, monkeypatch, arm, frame):
    from xrd_tools.analysis.module_transaction import ModuleDisposition
    from xrd_tools.analysis.rsm_operation import run_rsm_operation_v2
    from xrd_tools.analysis.rsm_values import prepare_rsm_tool_v2
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, AnalysisArtifactOverwrite, \
        read_analysis_artifact
    from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame
    from tests.core.test_rsm_operation_r2 import _form, _runnable, _write_member
    _runnable(monkeypatch, level=10)
    form = replace(_form(tmp_path, (_write_member(tmp_path, 0),)), overwrite=AnalysisArtifactOverwrite.REPLACE)
    if frame == "q":
        form = replace(form, coordinate_frame=RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU)
    axes = ("qx", "qy", "qz") if frame == "q" else ("h", "k", "l")
    states, fingerprints = [], []
    for level in (10, 1000):
        _runnable(monkeypatch, level=level)
        committed = run_rsm_operation_v2(prepare_rsm_tool_v2(form).request)
        assert committed.terminal.disposition is ModuleDisposition.COMMITTED, committed.terminal.code
        target = Path(committed.request.module.output.target)
        reread = read_analysis_artifact(target, expected_kind=AnalysisArtifactKind.RSM)
        assert reread.result_fingerprint == committed.payload.result_fingerprint
        assert tuple(name for name, _values in reread.axes) == axes
        np.testing.assert_array_equal(reread.intensity, committed.payload.intensity)
        states.append(target.stat())
        fingerprints.append(reread.result_fingerprint)
    assert fingerprints[0] != fingerprints[1]
    assert (states[0].st_dev, states[0].st_ino) != (states[1].st_dev, states[1].st_ino)
    assert _leftovers(target.parent) == []

def _hdf5(path: Path) -> bytes:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("value", data=np.arange(4096, dtype="<u2"))
    raw = path.read_bytes()
    assert raw.startswith(b"\x89HDF\r\n\x1a\n")
    return raw

def _corpus(path: Path) -> bytes:
    unit = b"\r\n\x1a\n\r\x1a" + bytes(range(256))
    raw = unit * (finite_module._COPY_BLOCK_BYTES // len(unit) + 3)  # CR/LF/0x1A across block edges
    path.write_bytes(raw)
    return raw

@pytest.mark.parametrize("kind", ("hdf5", "corpus"), ids=("W09-capture-hdf5", "W10-capture-corpus"))
def test_w09_w10_capture_hashes_every_binary_byte(tmp_path, crt, kind):
    path = tmp_path / "source.nexus"
    raw = (_hdf5 if kind == "hdf5" else _corpus)(path)
    snapshot = capture_finite_source(path).snapshot
    assert (snapshot.size, snapshot.digest) == (len(raw), hashlib.sha256(raw).hexdigest())

def _seeded_publish(request, admission, raw: bytes):
    def validate(binding):
        binding.seek(0)
        blocks = []
        while block := binding.read(1 << 20):
            blocks.append(block)
        assert b"".join(blocks) == raw
        return FiniteCandidateValidation(request.lineage)
    def inspect(path, inspected):
        snapshot, state = capture_target_snapshot(path), os.stat(path)
        return FiniteCommittedInspection(StreamTerminal(
            str(path), int(snapshot.size), str(snapshot.digest), 1, int(state.st_dev), int(state.st_ino),
            int(state.st_mtime_ns), int(state.st_ctime_ns)), inspected.lineage)
    adapter = FiniteSeededDocumentAdapter(
        nullcontext, lambda _document, _seed, _candidate: None, nullcontext, validate,
    )
    return FiniteArtifactPublisher(request).publish(
        adapter, inspect_committed=inspect, seed=admission, accept_validated_commit=lambda _v: None,
    )

@pytest.mark.parametrize("kind", ("hdf5", "corpus"), ids=("W11-seed-hdf5", "W12-seed-corpus"))
def test_w11_w12_seeded_copy_is_byte_identical(tmp_path, arm, crt, kind):
    source = tmp_path / "source.nexus"
    raw = (_hdf5 if kind == "hdf5" else _corpus)(source)
    admission = capture_finite_source(source)
    request = _request(tmp_path, source, admission=admission)
    result = _seeded_publish(request, admission, raw)
    assert result.disposition is FiniteArtifactDisposition.COMMITTED
    assert result.seed_receipt.candidate_digest == hashlib.sha256(raw).hexdigest()
    assert Path(request.output_artifact).read_bytes() == raw and source.read_bytes() == raw
    assert _leftovers(tmp_path) == []

def _fixed_names(monkeypatch, letter: str) -> Path:
    monkeypatch.setattr(finite_module, "secrets", SimpleNamespace(token_hex=lambda n: letter * (2 * n)))
    return Path(f".xdart-finite-{letter * 32}-{letter * 32}.nexus")

def test_w13_candidate_collision_preserves_the_foreign_occupant(tmp_path, arm, monkeypatch):
    foreign = tmp_path / _fixed_names(monkeypatch, "c")
    foreign.write_bytes(b"foreign")
    session = _FinitePublicationSession(tmp_path / "slot.nexus")
    with pytest.raises(FiniteArtifactCollision):
        session.reserve()
    assert foreign.read_bytes() == b"foreign" and not (tmp_path / "slot.nexus").exists()
    assert not session.slot_held
    _assert_slot_free(tmp_path / "slot.nexus")

def test_w14_existing_slot_is_replaced_atomically(tmp_path, arm):
    target = tmp_path / "slot.nexus"
    _raw, before = _prior(target, (1, 2, 3))
    session = _start(target, (4, 5, 6))
    facts = _publish(session, capture_target_snapshot(target))
    assert facts.published and facts.error is None and _values(target) == [4, 5, 6]
    assert facts.terminal.inode == target.stat().st_ino != before.st_ino
    assert _leftovers(tmp_path) == []  # the named arm reports its unconfirmed directory barrier:
    assert any(FINITE_PARENT_DIRECTORY_FSYNC_WARNING in item for item in session.diagnostics) is arm.named

def test_w15_target_substituted_after_admission_is_refused(tmp_path, arm):
    target = tmp_path / "slot.nexus"
    target.write_bytes(b"prior")
    expected = capture_target_snapshot(target)
    session = _start(target)
    (tmp_path / "foreign").write_bytes(b"foreign")
    os.replace(tmp_path / "foreign", target)
    with pytest.raises(FiniteArtifactIntegrityError, match="public target changed"):
        _publish(session, expected)
    assert target.read_bytes() == b"foreign"
    _assert_discarded(session, arm, tmp_path)

@pytest.mark.parametrize("level", ("parent", "ancestor"), ids=("W16-parent", "W17-ancestor"))
def test_w16_w17_exchange_after_admission_never_publishes_elsewhere(tmp_path, arm, level):
    """Either the held parent makes the exchange impossible, or publication refuses the moved parent."""
    exchanged_root = tmp_path / "root" if level == "ancestor" else tmp_path / "root" / "out"
    parent = tmp_path / "root" / "out"
    parent.mkdir(parents=True)
    target = parent / "slot.nexus"
    session = _start(target)
    expected = capture_target_snapshot(target)
    moved = tmp_path / "moved"
    try:
        exchanged_root.rename(moved)
    except PermissionError:
        print("W16_W17", level, "exchange refused while the parent is held", flush=True)
        facts = _publish(session, expected)
        assert facts.published and facts.error is None and _values(target) == [1, 2, 3]
        return
    parent.mkdir(parents=True)  # the caller's name now leads to a different, empty directory
    moved_parent = moved / "out" if level == "ancestor" else moved
    with pytest.raises(FiniteArtifactIntegrityError, match="parent changed"):
        _publish(session, expected)
    assert not target.exists() and not (moved_parent / "slot.nexus").exists()
    assert list(parent.iterdir()) == []
    _assert_discarded(session, arm, moved_parent)

def _substitute(path: Path, raw: bytes = b"foreign") -> None:
    path.unlink()
    path.write_bytes(raw)

@pytest.mark.parametrize("moment", ("before-write", "after-close"), ids=("W18-before-write", "W40-before-flush"))
def test_w18_w40_substituted_candidate_is_refused_and_never_deleted(tmp_path, arm, moment):
    target = tmp_path / "slot.nexus"
    session = _FinitePublicationSession(target)
    session.reserve()
    candidate = session.candidate
    if moment == "before-write":
        _substitute(candidate)
        try:
            with pytest.raises(FiniteArtifactIntegrityError, match="candidate changed"):
                session.open_document(lambda path: h5py.File(path, "w"))
        finally:
            session.abort()
        assert candidate.read_bytes() == b"foreign" and session.hidden_orphan is None
        return
    document = session.open_document(lambda path: h5py.File(path, "w"))
    document.create_dataset("value", data=[1])
    assert session.close_document()
    if moment == "after-close":
        _substitute(candidate)
    foreign = candidate.read_bytes()
    with pytest.raises(FiniteArtifactIntegrityError, match="candidate changed") as raised:
        _publish(session, capture_target_snapshot(target))
    assert not target.exists() and candidate.read_bytes() == foreign
    assert session.hidden_orphan is None
    assert any(FINITE_CANDIDATE_CLEANUP_WARNING in note for note in raised.value.__notes__)

def test_w19_hard_linked_source_alias_is_refused(tmp_path, monkeypatch, arm):
    from tests.core.test_p36_average import _stub_integrators
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.reduction import AverageScanRecipe, Integration1DPlan, ReductionPlan
    from xrd_tools.reduction.average import AverageScanRunner
    raw = tmp_path / "raw.nexus"
    with h5py.File(raw, "w") as handle:
        handle.create_group("entry/instrument/detector").create_dataset(
            "data", data=np.arange(12, dtype="u2").reshape(3, 2, 2))
    before = raw.read_bytes()
    try:
        os.link(raw, tmp_path / "scan_average.nexus")  # the Average slot aliases the raw source
    except OSError as error:
        pytest.skip(f"INCOMPLETE: this filesystem cannot create a hard link ({error})")
    _stub_integrators(monkeypatch, [])
    result = AverageScanRunner(AverageScanRecipe(
        SourceSpec(raw, SourceKind.NEXUS_STACK, entry="entry"), tmp_path / "scan.nexus",
        ReductionPlan(integration_1d=Integration1DPlan(npt=4)),
    )).start()
    assert result.disposition == "REFUSED" and result.diagnostic_code == "AVERAGE_OUTPUT_IS_SOURCE"
    assert raw.read_bytes() == before and _leftovers(tmp_path) == []

def test_w20_foreign_cleanup_occupant_is_preserved(tmp_path, arm):
    session = _start(tmp_path / "slot.nexus")
    _substitute(session.candidate)
    assert session.abort()
    assert session.candidate.read_bytes() == b"foreign" and session.hidden_orphan is None
    assert any(FINITE_CANDIDATE_CLEANUP_WARNING in item for item in session.diagnostics)
    assert not session.slot_held

@pytest.mark.parametrize("fault", ("before", "after"), ids=("W21-before-effect", "W22-after-own-effect"))
def test_w21_w22_replacement_fault_keeps_one_truth(tmp_path, monkeypatch, arm, fault):
    target = tmp_path / "slot.nexus"
    prior = _prior(target)
    real = finite_module.replace_into_place

    def faulty(*args, **kwargs):
        if fault == "after":
            real(*args, **kwargs)
        raise OSError(f"injected {fault} rename")
    monkeypatch.setattr(finite_module, "replace_into_place", faulty)
    session = _start(target, (2,))
    if fault == "before":
        with pytest.raises(OSError, match="injected before"):
            _publish(session, capture_target_snapshot(target))
        _assert_prior(target, prior)
        _assert_discarded(session, arm, tmp_path)
        return
    facts = _publish(session, capture_target_snapshot(target))
    assert facts.published and facts.error is None and _values(target) == [2]
    assert _leftovers(tmp_path) == []

@pytest.mark.parametrize("occupant", ("regular", "directory"), ids=("W23-regular", "W23-directory"))
def test_w23_ambiguous_foreign_occupant_keeps_its_verdict(tmp_path, monkeypatch, arm, occupant):
    target = tmp_path / "slot.nexus"
    real = finite_module.replace_into_place

    def foreign_after(*args, **kwargs):
        real(*args, **kwargs)
        target.unlink()
        target.mkdir() if occupant == "directory" else target.write_bytes(b"foreign")
        raise OSError("ambiguous rename")
    monkeypatch.setattr(finite_module, "replace_into_place", foreign_after)
    session = _start(target)
    if occupant == "regular":
        with pytest.raises(OSError, match="ambiguous rename"):
            _publish(session, capture_target_snapshot(target))
        assert target.read_bytes() == b"foreign"
    else:
        facts = _publish(session, capture_target_snapshot(target))
        assert facts.published and type(facts.error) is FiniteArtifactCollision and target.is_dir()
    assert _leftovers(tmp_path) == [] and session.hidden_orphan is None

def test_w24_cleanup_failure_names_the_exact_hidden_orphan(tmp_path, monkeypatch, arm):
    session = _start(tmp_path / "slot.nexus")
    candidate = session.candidate
    # Windows: a real handle without delete sharing; POSIX descriptor arm: an injected unlinkat failure.
    holder = open(candidate, "rb") if _WINDOWS else nullcontext()
    if not _WINDOWS and not arm.named:
        monkeypatch.setattr(finite_module, "_unlink_candidate", _raising(PermissionError("cleanup denied")))
    with holder:
        assert session.abort()
    assert candidate.exists() and os.path.samefile(session.hidden_orphan, candidate)
    assert any(FINITE_CANDIDATE_CLEANUP_WARNING in item for item in session.diagnostics)

@pytest.mark.parametrize("arm", ("named",), indirect=True)
def test_w25_named_parent_publishes_without_descriptor_relative_calls(tmp_path, monkeypatch, arm):
    relative, closes = [], []
    for owner, name in ((os, "open"), (os, "stat"), (os, "unlink"), (transaction_module, "_replace")):
        real = getattr(owner, name)

        def spy(*args, _real=real, _name=name, **kwargs):
            if any(kwargs.get(key) is not None for key in ("dir_fd", "src_dir_fd", "dst_dir_fd")):
                relative.append(_name)
            return _real(*args, **kwargs)
        monkeypatch.setattr(owner, name, spy)
    real_close = finite_module._close_descriptor_once
    monkeypatch.setattr(finite_module, "_close_descriptor_once",
                        lambda descriptor: closes.append(descriptor) or real_close(descriptor))
    target = tmp_path / "slot.nexus"
    session = _start(target)
    parent = _parent_descriptor(session)
    facts = _publish(session, capture_target_snapshot(target))
    assert facts.published and facts.error is None and _values(target) == [1, 2, 3]
    assert relative == [] and closes.count(parent) == 1 and session._parent_descriptor is None
    assert any(FINITE_PARENT_DIRECTORY_FSYNC_WARNING in item for item in session.diagnostics)

@pytest.mark.parametrize("case", ("abort", "failed-reserve", "duplicate-reserve"),
                         ids=("W26-abort", "W26-failed-reserve", "W26-duplicate-reserve"))
def test_w26_owner_retirement_never_leaks_or_steals_the_slot(tmp_path, monkeypatch, arm, case):
    target = tmp_path / "slot.nexus"
    session = _FinitePublicationSession(target)
    if case == "failed-reserve":
        foreign = tmp_path / _fixed_names(monkeypatch, "d")
        foreign.write_bytes(b"foreign")
        with pytest.raises(FiniteArtifactCollision):
            session.reserve()
        assert foreign.read_bytes() == b"foreign"
    else:
        session.reserve()
    if case == "abort":
        assert session.abort()
        _assert_discarded(session, arm, tmp_path)
    if case == "duplicate-reserve":
        with pytest.raises(RuntimeError, match="one-shot"):
            session.reserve()
        assert session.slot_held and session.candidate.exists()
        session.open_document(lambda path: h5py.File(path, "w")).create_dataset("value", data=[7])
        assert session.close_document()
        facts = _publish(session, capture_target_snapshot(target))
        assert facts.published and _values(target) == [7] and _leftovers(tmp_path) == []
    assert not session.slot_held and session._parent_descriptor is None
    _assert_slot_free(target)

def test_w27_seeded_adoption_publishes_releases_and_strands_nothing(tmp_path, arm):
    source = tmp_path / "source.nexus"
    raw = _hdf5(source)
    admission = capture_finite_source(source)
    request = _request(tmp_path, source, admission=admission)
    result = _seeded_publish(request, admission, raw)
    target = Path(request.output_artifact)
    assert result.disposition is FiniteArtifactDisposition.COMMITTED
    assert (result.terminal.device, result.terminal.inode) == (target.stat().st_dev, target.stat().st_ino)
    assert result.hidden_orphan is None and _leftovers(tmp_path) == []
    assert any(FINITE_PARENT_DIRECTORY_FSYNC_WARNING in item for item in result.diagnostics) is arm.named
    _assert_slot_free(target)

@pytest.mark.parametrize("fault", ("fstat", "_snapshot_at"))
@pytest.mark.parametrize("arm", ("named",), indirect=True)
def test_named_reservation_observation_failure_keeps_cleanup_authority(tmp_path, monkeypatch, arm, fault):
    owner = os if fault == "fstat" else finite_module
    original = getattr(owner, fault)
    def fail_once(*args, **kwargs):
        result = original(*args, **kwargs)
        if fault == "fstat" and not stat.S_ISREG(result.st_mode):
            return result
        monkeypatch.setattr(owner, fault, original)
        raise OSError("injected reservation observation failure")
    monkeypatch.setattr(owner, fault, fail_once)
    session = _FinitePublicationSession(tmp_path / "slot.nexus")
    with pytest.raises(OSError, match="reservation observation failure"):
        session.reserve()
    _assert_discarded(session, arm, tmp_path)
    assert not session.slot_held

@pytest.mark.parametrize("arm", ("named",), indirect=True)
def test_named_disposal_retains_secondary_close_failure(tmp_path, monkeypatch, arm):
    session = _start(tmp_path / "slot.nexus")
    parent, candidate, reserved = session._parent_descriptor, session.candidate, session._reservation
    original = finite_module._close_descriptor_once
    def close_then_report(descriptor):
        return original(descriptor) or OSError("injected disposal close failure")
    monkeypatch.setattr(finite_module, "_close_descriptor_once", close_then_report)
    try:
        with pytest.raises(FiniteArtifactIntegrityError) as raised:
            finite_module._unlink_candidate(parent, candidate, reserved)
        assert "close" in str(raised.value) or any("close" in note for note in getattr(raised.value, "__notes__", ()))
    finally:
        monkeypatch.setattr(finite_module, "_close_descriptor_once", original)
        session.abort()

def test_w28_capacity_refusal_preserves_source_and_prior(tmp_path, monkeypatch, arm):
    source = tmp_path / "source.nexus"
    raw = _hdf5(source)
    admission = capture_finite_source(source)
    request = _request(tmp_path, source, admission=admission)
    target = Path(request.output_artifact)
    target.write_bytes(b"prior")
    required = len(raw) + max(finite_module._CANDIDATE_SPACE_MARGIN_BYTES, (len(raw) + 9) // 10)
    probes = []  # the named arm measures its admitted parent's volume; POSIX keeps fstatvfs
    if arm.named:
        monkeypatch.setattr(shutil, "disk_usage", lambda path: probes.append(path) or SimpleNamespace(
            total=required * 4, used=0, free=required - 1))
    else:
        monkeypatch.setattr(finite_module.os, "fstatvfs", lambda descriptor: probes.append(descriptor) or
                            SimpleNamespace(f_frsize=1, f_bavail=required - 1))
    with pytest.raises(FiniteArtifactCapacityError) as raised:
        _seeded_publish(request, admission, raw)
    assert probes and (raised.value.required_bytes, raised.value.available_bytes) == (required, required - 1)
    assert target.read_bytes() == b"prior" and source.read_bytes() == raw and _leftovers(tmp_path) == []

_ANGLES = [(6.0, -10.0), (15.0, 3.0), (25.0, 6.0)]

@pytest.fixture(scope="module")
def calibration_fit(tmp_path_factory):
    from tests.core.test_stitch_autocalibration import ROW_ANGLES, five_parameter_seed, ring_scan
    from xrd_tools.integrate.stitch_autocalibration import refine_arm_from_scan
    scan = ring_scan(tmp_path_factory.mktemp("fit") / "scan", [(d, 0.0) for d in (6.0, 20.0, 34.0)])
    return refine_arm_from_scan(scan, five_parameter_seed(ROW_ANGLES), max_frames=3)

def _pair_files(tmp_path):
    target = tmp_path / "saved.json"
    return target, (target.with_suffix(".json.provenance.json"), target, tmp_path / "saved.xu.json",
                    tmp_path / "saved.xu.json.provenance.json")

def _changed(fit):
    asset = copy.deepcopy(fit.asset)
    asset["calibration"]["param"][asset["calibration"]["param_names"].index("dist")] += 0.001
    return replace(fit, asset=asset)

def _snapshot_pair(files):
    return {path: path.read_bytes() for path in files}

def test_w29_w30_w31_pair_create_overwrite_and_prior_preservation(tmp_path, calibration_fit):
    from xrd_tools.integrate.stitch_calibration import paired_xu_calibration, save_calibration_pair
    target, files = _pair_files(tmp_path)
    save_calibration_pair(calibration_fit, target, validation_angles=_ANGLES)  # W29 create
    assert paired_xu_calibration(target) == files[2]
    binding = json.loads(files[3].read_text(encoding="utf-8"))["pyfai_calibration"]["sha256"]
    assert binding == hashlib.sha256(target.read_bytes()).hexdigest()
    created = _snapshot_pair(files)
    save_calibration_pair(_changed(calibration_fit), target, validation_angles=_ANGLES, overwrite=True)  # W30
    assert paired_xu_calibration(target) == files[2] and _snapshot_pair(files) != created
    assert not tuple(tmp_path.glob(".xdart-calibration-*"))
    current = _snapshot_pair(files)
    invalid = copy.deepcopy(_changed(calibration_fit).asset)  # W31: invalid and cancelled saves
    invalid["calibration"]["wavelength"] = float("nan")
    with pytest.raises(ValueError):
        save_calibration_pair(replace(calibration_fit, asset=invalid), target, validation_angles=_ANGLES,
                              overwrite=True)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(InterruptedError):
        save_calibration_pair(calibration_fit, target, validation_angles=_ANGLES, cancelled=cancel,
                              overwrite=True)
    assert _snapshot_pair(files) == current and not tuple(tmp_path.glob(".xdart-calibration-*"))

@pytest.mark.parametrize("fault", ("before-move", "after-move", "restore-fails", "foreign-occupant",
                                  "after-move-foreign", "foreign-during-restore"),
                         ids=("W32-before-move", "W33-after-move", "W34-restore-fails", "W34-foreign",
                              "W34-after-move-foreign", "W34-foreign-during-restore"))
def test_w32_w34_pair_failure_restores_owned_entries_and_reports(tmp_path, monkeypatch, calibration_fit, fault):
    from xrd_tools.integrate.stitch_calibration import save_calibration_pair
    target, files = _pair_files(tmp_path)
    save_calibration_pair(calibration_fit, target, validation_angles=_ANGLES)
    prior = _snapshot_pair(files)
    real = transaction_module.replace_into_place
    calls = []
    foreign = tmp_path / "foreign"
    foreign.write_bytes(b"foreign")
    write = Path.write_bytes
    def substitute_during_backup(path, raw):
        result = write(path, raw)
        if fault == "foreign-during-restore" and path.name == files[2].name and len(calls) >= 3:
            os.replace(foreign, files[2])
        return result
    monkeypatch.setattr(Path, "write_bytes", substitute_during_backup)

    def injected(source, destination, *args, **kwargs):
        calls.append((Path(destination).name, kwargs.get("verb", "publish")))
        publish_index = sum(verb == "publish" for _name, verb in calls)
        if kwargs.get("verb") == "restore calibration":
            if fault == "restore-fails" and Path(destination) == files[2]:
                raise OSError("injected restore failure")
            return real(source, destination, *args, **kwargs)
        if fault == "before-move" and publish_index == 1:
            raise OSError("injected before move")
        if fault in ("after-move", "after-move-foreign") and publish_index == 2:
            real(source, destination, *args, **kwargs)
            if fault == "after-move-foreign":
                os.replace(foreign, destination)
            raise OSError("injected after move")
        if fault in ("restore-fails", "foreign-occupant", "foreign-during-restore") and publish_index == 3:
            if fault == "foreign-occupant":
                _substitute(files[2])  # another writer replaced an entry we had moved
            raise OSError("injected third move")
        return real(source, destination, *args, **kwargs)
    monkeypatch.setattr(transaction_module, "replace_into_place", injected)
    with pytest.raises(OSError, match="injected") as raised:
        save_calibration_pair(_changed(calibration_fit), target, validation_angles=_ANGLES, overwrite=True)
    unrestored = files[2] if fault in ("restore-fails", "foreign-occupant", "after-move-foreign", "foreign-during-restore") else None
    for path in files:
        if path != unrestored:
            assert path.read_bytes() == prior[path], path.name
    if unrestored is None:
        assert not tuple(tmp_path.glob(".xdart-calibration-*"))
        return
    if fault in ("foreign-occupant", "after-move-foreign", "foreign-during-restore"):
        assert unrestored.read_bytes() == b"foreign"  # never clobber an entry we no longer own
    notes = " ".join(getattr(raised.value, "__notes__", ()))
    assert "rollback incomplete" in notes and unrestored.name in notes
    kept = [path for path in tmp_path.rglob("*") if path.is_file() and path not in files
            and path.read_bytes() == prior[unrestored]]
    assert kept, "the prior bytes of an unrestored entry must stay recoverable"

def test_w36_held_open_prior_released_within_the_retry_window_publishes(tmp_path, arm):
    target = tmp_path / "slot.nexus"
    _prior(target)
    session = _start(target, (2,))
    expected = capture_target_snapshot(target)
    holder = open(target, "rb")  # on Windows a CRT handle without delete sharing blocks the rename
    timer = threading.Timer(0.6, holder.close)
    timer.start()
    try:
        facts = _publish(session, expected)
    finally:
        timer.cancel()
        holder.close()
    assert facts.published and facts.error is None and _values(target) == [2]
    assert _leftovers(tmp_path) == []

@pytest.mark.parametrize("error", ("held-open", "generic"), ids=("W37-exhaustion", "W38-generic-not-retried"))
def test_w37_w38_failed_replacement_preserves_the_prior_slot(tmp_path, monkeypatch, arm, error):
    target = tmp_path / "slot.nexus"
    prior = _prior(target)
    session = _start(target, (2,))
    expected = capture_target_snapshot(target)
    native_holder = error == "held-open" and _WINDOWS  # a real holder for the whole retry window
    calls = []
    if not native_holder:
        def refuse(*_args, **_kwargs):
            calls.append(1)
            raise PermissionError(errno.EACCES, "held") if error == "held-open" else OSError(errno.EIO, "io")
        monkeypatch.setattr(transaction_module, "_replace", refuse)
    with open(target, "rb") if native_holder else nullcontext():
        with pytest.raises(PermissionError if error == "held-open" else OSError) as raised:
            _publish(session, expected)
    if error == "held-open":
        assert "held open by another program" in str(raised.value)
        assert native_holder or len(calls) == transaction_module._REPLACE_RETRIES
    else:
        assert type(raised.value) is OSError and len(calls) == 1
    _assert_prior(target, prior)
    _assert_discarded(session, arm, tmp_path)

def test_w41_mixed_case_average_reload_keeps_spelling_and_canonical_identity(tmp_path, monkeypatch):
    from tests.core.test_p36_average import _run_average_scan, _series, _stub_integrators
    from tests.xdart.scattering.test_workspace_operations import _owner_with_slot
    from xdart.gui.tabs.scattering.operation_values import OperationIdentity, OperationTerminal, \
        OperationTerminalStatus, OperationUpdate
    from xdart.gui.tabs.scattering.workspace_operations import AverageOperationState
    from xrd_tools.reduction import AverageScanRecipe, Integration1DPlan, ReductionPlan
    from xrd_tools.reduction.average import _committed_average_mismatch
    folder = tmp_path / "Mixed Case" / "Sample_A"
    folder.mkdir(parents=True)
    _stub_integrators(monkeypatch, [])
    anchor = folder / "Scan_A.nxs"
    result = _run_average_scan(AverageScanRecipe(_series(folder), anchor,
                                                 ReductionPlan(integration_1d=Integration1DPlan(npt=4))))
    assert result.disposition == "COMMITTED"
    assert Path(result.target).name in os.listdir(folder)  # the public spelling survives publication
    assert result.commit_identity.target == os.path.normcase(os.path.abspath(result.target))
    assert _committed_average_mismatch(result, anchor, "entry") is None
    owner, slot = _owner_with_slot()
    identity = OperationIdentity(41)
    slot._identity = identity
    owner._average = AverageOperationState(identity, 1, str(anchor), "entry")
    transition = owner.consume_average_update(OperationUpdate(identity, terminal=OperationTerminal(
        identity, OperationTerminalStatus.RETURNED, payload=result)), current_intent_revision=1)
    directive = transition.average_reload
    assert directive is not None, transition.notice
    sealed = directive.terminal_commit_identity.target
    assert sealed == os.path.normcase(os.path.abspath(sealed))  # begin_browse's writer-seal condition
    assert sealed == os.path.normcase(os.path.abspath(directive.target))

def _make_link(kind: str, link: Path, real: Path) -> None:
    if kind == "junction":
        import _winapi
        return _winapi.CreateJunction(str(real), str(link))
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"INCOMPLETE: this host cannot create a directory symlink ({error})")

@pytest.mark.parametrize("kind", ("junction", "symlink") if _WINDOWS else ("symlink",))
@pytest.mark.parametrize("layout", ("parent", "ancestor"), ids=("W42-parent-link", "W42-ancestor-link"))
def test_w42_immediate_parent_link_refused_ancestor_link_admitted(tmp_path, arm, layout, kind):
    real = tmp_path / "real"
    (real / "sub").mkdir(parents=True)
    link = tmp_path / "alias"
    try:
        _make_link(kind, link, real if layout == "ancestor" else real / "sub")
    except OSError as error:  # only the junction cell reaches here
        pytest.skip(f"junction unavailable ({error}): N/A on an SMB share, INCOMPLETE on local NTFS")
    target = (link / "sub" if layout == "ancestor" else link) / "slot.nexus"
    if layout == "parent":
        session = _FinitePublicationSession(target)
        with pytest.raises(FiniteArtifactIntegrityError, match="non-symlink directory"):
            session.reserve()
        assert not session.slot_held and list((real / "sub").iterdir()) == []
        return
    facts = _publish(_start(target), capture_target_snapshot(target))
    assert facts.published and facts.error is None and _values(real / "sub" / "slot.nexus") == [1, 2, 3]
    assert _leftovers(real / "sub") == []

def test_w43_w44_identities_are_distinct_nonzero_and_survive_the_rename(tmp_path, arm):
    states = []
    for name in ("one.nexus", "two.nexus"):
        session = _start(tmp_path / name)
        reserved = session.candidate.stat()
        assert _publish(session, capture_target_snapshot(tmp_path / name)).published
        published = (tmp_path / name).stat()
        assert (published.st_dev, published.st_ino) == (reserved.st_dev, reserved.st_ino)  # W44
        states.append(published)
    inodes = {state.st_ino for state in states} | {tmp_path.stat().st_ino}
    assert 0 not in inodes and len(inodes) == 3  # W43

class _ZeroInode:  # a stat view whose filesystem reports no file identity
    def __init__(self, real):
        self._real, self.st_ino = real, 0
    def __getattr__(self, name):
        return getattr(self._real, name)

def test_w45_unavailable_identity_is_refused_before_any_candidate(tmp_path, monkeypatch, arm):
    """Synthetic and DIAGNOSTIC: proves the guard, never a claim about a native server."""
    for name in ("stat", "lstat", "fstat"):
        real = getattr(os, name)
        monkeypatch.setattr(os, name, lambda *a, _real=real, **k: _ZeroInode(_real(*a, **k)))
    session = _FinitePublicationSession(tmp_path / "slot.nexus")
    with pytest.raises(FiniteArtifactIntegrityError, match="identity"):
        session.reserve()
    monkeypatch.undo()
    assert not session.slot_held and _leftovers(tmp_path) == []

def test_w46_deep_parent_candidate_beyond_max_path(tmp_path, arm):
    parent = tmp_path
    while len(str(parent / "slot.nexus")) < 230:
        parent = parent / ("d" * min(40, 230 - len(str(parent / "slot.nexus"))))
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / "slot.nexus"
    if not len(str(target)) < 260 < len(str(parent / f".xdart-finite-{'0' * 32}-{'0' * 32}.nexus")):
        pytest.skip(f"INCOMPLETE: basetemp {tmp_path} is too long to place the W46 target under MAX_PATH")
    if _WINDOWS:  # record the host setting (T7); never change it
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
                print("LongPathsEnabled", winreg.QueryValueEx(key, "LongPathsEnabled")[0], flush=True)
        except OSError as error:
            print("LongPathsEnabled unreadable", error, flush=True)
    session = _FinitePublicationSession(target)
    try:
        session.reserve()
    except FiniteArtifactError as refused:
        print("W46 typed pre-effect refusal", type(refused).__name__, refused, flush=True)
        assert not session.slot_held and _leftovers(parent) == [] and not target.exists()
        return
    document = session.open_document(lambda path: h5py.File(path, "w"))
    document.create_dataset("value", data=[46])
    assert session.close_document()
    facts = _publish(session, capture_target_snapshot(target))
    assert facts.published and facts.error is None and _values(target) == [46]
    assert _leftovers(parent) == []

class _WindowsWidth(_ZeroInode):  # Windows Python 3.12+: a 64-bit volume serial and a file ID up to 128 bits
    def __init__(self, real):
        self._real, self.st_dev, self.st_ino = real, real.st_dev | 1 << 63, real.st_ino | 1 << 127

@pytest.mark.parametrize("flow", (test_w14_existing_slot_is_replaced_atomically,
                                  test_w27_seeded_adoption_publishes_releases_and_strands_nothing),
                         ids=("W47-replace", "W47-seeded"))
def test_w47_windows_width_identities_publish(tmp_path, monkeypatch, arm, flow):
    """The hosted runner's NTFS reported st_dev 15923074127169020758 (> 2**63 - 1); identities have no width bound."""
    for name in ("stat", "lstat", "fstat"):
        monkeypatch.setattr(os, name, lambda *a, _real=getattr(os, name), **k: _WindowsWidth(_real(*a, **k)))
    flow(tmp_path, arm)
