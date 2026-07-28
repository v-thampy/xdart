"""O-3N.R.3 — completing the prepared NeXus execution transaction (handoff §19).

Frozen acceptance oracle for the §19.6 authorized completion packet, written
BEFORE any production edit and red at parent ``214eed2a``.

§19 accepted the §17.8 envelope design and found one bounded residual family:
preparation and output phases still reported or advanced before every resource
and artifact belonging to the phase reached its terminal state.  Four defects:

* §19.1 — raw pixels and scan metadata were NOT one accepted resource: the body
  reopened the accepted pathname with ``read_nexus()`` after strict binding, so
  a pathname replacement mixed accepted raw frames with replacement metadata;
* §19.2 — envelope publication/adoption and strict-opener construction were not
  ``BaseException``-total, leaking a published envelope or a sole HDF5 handle;
* §19.3 — a failed Overwrite rollback left the durable prior at the staging
  suffix, and the next exact retry unlinked that backup to reuse the suffix —
  destroying the only byte-identical prior result;
* §19.4 — stale-XYE cleanup retained its pending set but did not GATE this
  run's publication, so a persistent failure produced exactly the mixed-run
  state §17.8 item 6 forbids, reported as a clean completed run.

Row map — the seven independent §19 review rows, promoted in production shape
(§19.5: observe exact open handles, filesystem bytes/artifacts, published XYE
and the public/direct execution outcome; no private latch names):

1   complete preparation + metadata = ONE accepted-source open        (§19.1)
2   nothing re-resolves the accepted pathname after preparation       (§19.1)
3   a failed envelope adoption publishes nothing and leaks nothing    (§19.2)
4   the strict opener is BaseException-total at stack construction    (§19.2)
5ab the execution-source opener: one open, detached metadata, total   (§19.1/2)
6   a pending stale tail WITHHOLDS this run's XYE publication         (§19.4)
7   a persistent stale tail ends the run PENDING, never "Done"        (§19.4)
7b  a transiently blocked tail publishes the withheld XYE at run end  (§19.4)
8   a failed-rollback retry never destroys the durable prior          (§19.3)
9   an unowned .xdart-replacing artifact is refused, not destroyed    (§19.3)
10  a zero-publication run never sweeps the prior XYE tail       (panel F1)
11  a commit-resolved leftover backup is retried at the next commit (panel F2)
12  a stuck leftover backup is surfaced visibly at run end       (panel F2)
13  withheld XYE entries die with their envelope                 (panel F3)

Promotion provenance: rows 1/3/4 are the preserved reviewer rows from
``test_codex_o3nr2_single_source_handle.py`` and rows 6/8 the preserved rows
from ``test_codex_o3nr2_transaction_exact_review.py``, essentially verbatim.
The two inode rows those files proved with an unconditional ``pytest.fail``
demonstrator (metadata read from a replacement inode) are promoted as rows 2
and 5a asserting the CORRECTED §19.6 contract — the observation (handle count,
bound raw bytes, the replacement's motor value) is theirs.  Rows 7/7b/9 are the
§19.5-required production shapes the committed R.2 matrix lacked: the
only/final-flush outcome and rollback-failure-then-retry.

Production-wired (CLAUDE.md rule 2): the real ``staticWidget``, the real NeXus
page, real h5py containers, the real writer, and — rows 7/7b — a REAL complete
``thread.run()`` with real pyFAI integration on Pilatus100k-shaped frames.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pyqtgraph")
h5py = pytest.importorskip("h5py")

from tests.xdart.test_o3n_execution_owner import (  # noqa: E402,F401
    _select_nexus,
    _start_recorder,
    _started,
    _write_nexus,
    _write_poni,
    qapp,
    widget,
)
from tests.xdart.test_o3nr2_prepared_execution import (  # noqa: E402
    _collect_source_handles,
    _envelope,
)
from xdart.gui.tabs.static_scan.wranglers import (  # noqa: E402
    nexus_wrangler_thread as nwt,
)
from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (  # noqa: E402,E501
    nexusThread,
)
from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import (  # noqa: E402
    wranglerThread,
)
from xrd_tools.io import nexus as nexus_io  # noqa: E402


class _StopProbe(BaseException):
    """Escapes ``except Exception`` handlers, as an interrupt would."""


def _write_replacement_source(path, *, entry="entry", raw=91.0, motor=73.0):
    """A different-content container to occupy the accepted pathname."""
    with h5py.File(path, "w") as handle:
        group = handle.create_group(entry)
        group.attrs["NX_class"] = "NXentry"
        detector = group.create_group("instrument/detector")
        detector.attrs["NX_class"] = "NXdetector"
        detector.create_dataset(
            "data", data=np.full((2, 8, 8), raw, dtype=np.float32))
        scan_data = group.create_group("scan_data")
        scan_data.create_dataset(
            "halpha", data=np.asarray([motor, motor + 1.0]))
    return path


def _arm_pilatus(wrangler, tmp_path):
    """Arm a run whose frames the STANDARD fixture PONI can really integrate."""
    src = _write_nexus(tmp_path / "acq.nxs", entry="entry", shape=(195, 487))
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    wrangler.parameters.child("Calibration", "poni_file").setValue(
        _write_poni(tmp_path / "cal.poni"))
    wrangler.parameters.child("NeXus File", "nexus_file").setValue(str(src))
    wrangler.parameters.child("NeXus File", "entry").setValue("entry")
    wrangler.parameters.child("Output", "h5_dir").setValue(str(out))
    wrangler.parameters.child("Project", "project_folder").setValue(
        str(tmp_path))
    return src, out


def _started_with_motors(wrangler, tmp_path, monkeypatch):
    """An admitted run whose ACCEPTED container carries a real motor column.

    O-3N.R.3-3 oracle repair (adversarial-panel finding R3-F8): with the
    standard motorless fixture, the promoted replacement-metadata assertion
    was vacuously true — ``angles`` was always empty.  The preserved reviewer
    rows observed the motor VALUE; this fixture makes that observation
    load-bearing again: the accepted ``halpha`` is [5.0, 6.0], the replacement
    writes 73.0.
    """
    src = tmp_path / "acq-motors.nxs"
    with h5py.File(src, "w") as handle:
        group = handle.create_group("entry")
        group.attrs["NX_class"] = "NXentry"
        detector = group.create_group("instrument/detector")
        detector.attrs["NX_class"] = "NXdetector"
        detector.create_dataset(
            "data", data=np.arange(2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8))
        scan_data = group.create_group("scan_data")
        scan_data.create_dataset("halpha", data=np.asarray([5.0, 6.0]))
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    wrangler.parameters.child("Calibration", "poni_file").setValue(
        _write_poni(tmp_path / "cal.poni"))
    wrangler.parameters.child("NeXus File", "nexus_file").setValue(str(src))
    wrangler.parameters.child("NeXus File", "entry").setValue("entry")
    wrangler.parameters.child("Output", "h5_dir").setValue(str(out))
    wrangler.parameters.child("Project", "project_folder").setValue(
        str(tmp_path))
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()
    return src, out, wrangler.thread


# --------------------------------------------------------------------------- #
# 1/2 — §19.1: raw frames and scan metadata are ONE accepted resource
# --------------------------------------------------------------------------- #

def test_complete_worker_preparation_and_metadata_use_one_source_open(
        widget, tmp_path, monkeypatch):
    """Qualification, raw binding AND scan metadata share one source open.

    The committed one-open row counted strict preflight only; the complete
    body then called ``read_nexus(path, entry)`` — a second open (§19.1).
    """
    wrangler = _select_nexus(widget)
    source, _out, thread = _started(wrangler, tmp_path, monkeypatch)
    frozen = thread.run_configuration
    opens = _collect_source_handles(monkeypatch, source)
    prepared = nexusThread._prepare_execution(thread, frozen)

    monkeypatch.setattr(
        nexusThread, "_initialize_scan",
        lambda _self, _scan_name: (_ for _ in ()).throw(_StopProbe()))
    try:
        with pytest.raises(_StopProbe):
            nexusThread._run_body(thread, prepared)
    finally:
        nexusThread._release_execution(thread)

    assert len(opens) == 1, (
        "the complete worker reopened its accepted source after strict "
        f"raw-stack binding: observed {len(opens)} h5py.File opens")
    assert all(not handle.id.valid for handle in opens)


def test_nothing_reopens_the_accepted_pathname_after_preparation(
        widget, tmp_path, monkeypatch):
    """A pathname replacement after preparation reaches NOTHING.

    The §19.1 measured facts, asserted with corrected polarity: the raw stack
    stays bound to the accepted inode, no later open re-resolves the accepted
    pathname, and the execution's scan metadata carries the ACCEPTED motor
    values — never the replacement file's.  The accepted container carries a
    real ``halpha`` column so the motor-value observation is load-bearing
    (panel finding R3-F8; the preserved reviewer rows measured this value).
    """
    wrangler = _select_nexus(widget)
    source, _out, thread = _started_with_motors(wrangler, tmp_path, monkeypatch)
    frozen = thread.run_configuration
    prepared = nexusThread._prepare_execution(thread, frozen)
    accepted_raw = float(np.asarray(prepared.stack[0]).ravel()[1])
    opens = _collect_source_handles(monkeypatch, source)

    replacement = _write_replacement_source(tmp_path / "replacement.nxs")
    os.replace(replacement, source)

    monkeypatch.setattr(
        nexusThread, "_initialize_scan",
        lambda _self, _scan_name: (_ for _ in ()).throw(_StopProbe()))
    try:
        with pytest.raises(_StopProbe):
            nexusThread._run_body(thread, prepared)

        assert opens == [], (
            "the execution re-resolved the accepted pathname after strict "
            "preparation; a replacement inode reached the run")
        assert float(np.asarray(prepared.stack[0]).ravel()[1]) == accepted_raw
        assert accepted_raw != 91.0
        meta = prepared.scan_metadata
        assert meta is not None
        halpha = np.asarray(meta.angles["halpha"]).ravel().tolist()
        assert halpha == [5.0, 6.0], (
            f"scan metadata does not carry the accepted motor values: {halpha}")
        assert all(73.0 not in np.asarray(v).ravel()
                   for v in meta.angles.values()), (
            "scan metadata came from the replacement file while raw pixels "
            "remained bound to the prepared source handle")
    finally:
        nexusThread._release_execution(thread)


# --------------------------------------------------------------------------- #
# 3/4/5 — §19.2: publication/adoption and strict opening are BaseException-total
# --------------------------------------------------------------------------- #

def test_direct_impl_releases_exact_stack_if_envelope_adoption_raises(
        widget, tmp_path, monkeypatch):
    """The supported direct entry owns faults in the LAST prepare phase.

    The parent assigned ``self._execution`` and only then adopted; an adoption
    fault left the envelope published and the exact HDF5 handle live (§19.2).
    """
    wrangler = _select_nexus(widget)
    source, _out, thread = _started(wrangler, tmp_path, monkeypatch)
    frozen = thread.run_configuration
    opens = _collect_source_handles(monkeypatch, source)

    monkeypatch.setattr(
        nexusThread, "_adopt_frozen_source_target",
        lambda _self, _prepared: (_ for _ in ()).throw(_StopProbe()))
    with pytest.raises(_StopProbe):
        nexusThread._run_impl(thread, frozen)

    assert thread._execution is None, (
        "the failed preparation remained published as the active execution")
    assert opens and all(not handle.id.valid for handle in opens), (
        "the strict stack leaked when envelope adoption raised")


def test_exact_opener_closes_its_handle_if_stack_construction_is_interrupted(
        tmp_path, monkeypatch):
    """The one strict open must be total before an envelope can own it."""
    source = tmp_path / "constructor-interrupt.nxs"
    with h5py.File(source, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        detector = entry.create_group("instrument/detector")
        detector.create_dataset(
            "data", data=np.zeros((1, 4, 4), dtype=np.float32))

    handles = _collect_source_handles(monkeypatch, source)
    monkeypatch.setattr(
        nexus_io, "NexusImageStack",
        lambda _h5f, _paths: (_ for _ in ()).throw(_StopProbe()))

    with pytest.raises(_StopProbe):
        nexus_io.open_nexus_image_stack_exact(source, "entry")

    assert handles and all(not handle.id.valid for handle in handles), (
        "the strict opener leaked its sole HDF5 handle when stack "
        "construction was interrupted")


def test_execution_source_binds_stack_and_detached_metadata_in_one_open(
        tmp_path, monkeypatch):
    """§19.6 item 1: ONE strict open yields the raw stack AND detached scan
    metadata — the io-level operation the prepared envelope consumes."""
    source = tmp_path / "with-motors.nxs"
    with h5py.File(source, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        detector = entry.create_group("instrument/detector")
        detector.create_dataset(
            "data", data=np.arange(32, dtype=np.float32).reshape(2, 4, 4))
        scan_data = entry.create_group("scan_data")
        scan_data.create_dataset("halpha", data=np.asarray([5.0, 6.0]))

    opens = _collect_source_handles(monkeypatch, source)
    open_execution_source = getattr(nexus_io, "open_nexus_execution_source")

    execution_source = open_execution_source(source, "entry")
    try:
        stack = execution_source.stack
        meta = execution_source.scan_metadata
        assert all(p.startswith("/entry/") for p in stack.paths)
        assert meta is not None
        assert np.asarray(meta.angles["halpha"]).tolist() == [5.0, 6.0]
    finally:
        execution_source.stack.close()

    assert len(opens) == 1, (
        f"stack binding and metadata used {len(opens)} opens; they must be "
        "one held resource")
    # Detached: the metadata outlives the closed handle.
    assert np.asarray(meta.angles["halpha"]).tolist() == [5.0, 6.0]


def test_execution_source_opener_closes_its_handle_when_interrupted(
        tmp_path, monkeypatch):
    source = tmp_path / "constructor-interrupt-pair.nxs"
    with h5py.File(source, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        detector = entry.create_group("instrument/detector")
        detector.create_dataset(
            "data", data=np.zeros((1, 4, 4), dtype=np.float32))

    handles = _collect_source_handles(monkeypatch, source)
    open_execution_source = getattr(nexus_io, "open_nexus_execution_source")
    monkeypatch.setattr(
        nexus_io, "NexusImageStack",
        lambda _h5f, _paths: (_ for _ in ()).throw(_StopProbe()))

    with pytest.raises(_StopProbe):
        open_execution_source(source, "entry")

    assert handles and all(not handle.id.valid for handle in handles), (
        "the execution-source opener leaked its sole HDF5 handle")


# --------------------------------------------------------------------------- #
# 6/7 — §19.4: pending stale-XYE cleanup gates this run's publication
# --------------------------------------------------------------------------- #

def test_pending_stale_tail_withholds_this_runs_xye_write(
        tmp_path, monkeypatch):
    """All-or-pending may not publish the new run beside an undeleted tail."""
    root = tmp_path / "scan"
    root.mkdir()
    stale = root / "iq_scan_0009.xye"
    stale.write_text("old run")
    current = root / "iq_scan_0000.xye"
    frozen = SimpleNamespace(output_mode="Overwrite")
    prepared = _envelope(frozen, tmp_path / "scan.nxs")
    worker = SimpleNamespace(run_configuration=frozen, _execution=prepared)
    scan = SimpleNamespace(data_file=str(tmp_path / "scan.nxs"), name="scan")

    real_unlink = Path.unlink

    def refuse_stale(path, *args, **kwargs):
        if path == stale:
            raise OSError("persistent stale-tail failure")
        return real_unlink(path, *args, **kwargs)

    writes: list[str] = []

    def write_current(self, scan, published_idxs=None):
        writes.append("write")
        current.write_text("new run")

    monkeypatch.setattr(Path, "unlink", refuse_stale)
    monkeypatch.setattr(wranglerThread, "_flush_xye_buffer", write_current)

    nexusThread._flush_xye_buffer(worker, scan, published_idxs={0})

    assert stale.exists()
    assert prepared.xye_tail_pending == [stale]
    assert writes == [], (
        "new XYE was written while stale cleanup remained pending")
    assert not current.exists(), (
        "the output directory now mixes two run identities")


def test_persistent_stale_tail_yields_a_pending_outcome_not_done(
        widget, tmp_path, monkeypatch):
    """A COMPLETE real run against a permanently undeletable stale tail.

    §19.4: a one-chunk run has no later flush for the promised retry, so the
    parent published the new XYE beside the stale one and reported a clean
    completed run.  The corrected terminal outcome withholds publication and
    is visibly pending — never 'Done'.
    """
    wrangler = _select_nexus(widget)
    widget.controls.set_write_mode("Overwrite")
    src, out = _arm_pilatus(wrangler, tmp_path)
    scan_dir = out / Path(src).stem
    scan_dir.mkdir(parents=True)
    stale = scan_dir / f"iq_{Path(src).stem}_0009.xye"
    stale.write_text("prior run tail")
    real_unlink = Path.unlink

    def refuse_stale(path, *args, **kwargs):
        if path == stale:
            raise OSError("persistent stale-tail failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_stale)
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()
    thread = wrangler.thread
    said: list[str] = []
    thread.showLabel.connect(said.append)

    thread.run()

    fresh = sorted(p.name for p in scan_dir.glob("*.xye")
                   if p.name != stale.name)
    assert stale.exists()
    assert fresh == [], (
        f"the run published {fresh} beside an undeletable stale tail — the "
        "mixed-run state §17.8 item 6 forbids")
    assert not any(m.startswith("Done") for m in said), (
        f"a run with pending stale-XYE cleanup reported clean completion: "
        f"{said}")
    assert any("stale" in m.lower() for m in said), (
        f"no visible pending/refusal outcome was surfaced: {said}")


def test_transiently_blocked_tail_publishes_the_withheld_xye_at_run_end(
        widget, tmp_path, monkeypatch):
    """Withheld is NOT dropped: once the tail clears, the run's own XYE files
    land, and only then is the run clean.  (Kills withhold-by-discarding.)"""
    wrangler = _select_nexus(widget)
    widget.controls.set_write_mode("Overwrite")
    src, out = _arm_pilatus(wrangler, tmp_path)
    scan_dir = out / Path(src).stem
    scan_dir.mkdir(parents=True)
    stale = scan_dir / f"iq_{Path(src).stem}_0009.xye"
    stale.write_text("prior run tail")
    real_unlink = Path.unlink
    attempts: list[str] = []

    def fail_once(path, *args, **kwargs):
        if path == stale:
            attempts.append(str(path))
            if len(attempts) == 1:
                raise OSError("injected transient")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_once)
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()
    thread = wrangler.thread
    said: list[str] = []
    thread.showLabel.connect(said.append)

    thread.run()

    fresh = sorted(p.name for p in scan_dir.glob("*.xye"))
    assert not stale.exists(), (
        "the transient failure was never retried before run end")
    assert fresh == [f"iq_{Path(src).stem}_0000.xye",
                     f"iq_{Path(src).stem}_0001.xye"], (
        f"the withheld XYE tail was lost instead of published: {fresh}")
    assert any(m.startswith("Done") for m in said), said
    assert len(attempts) >= 2


# --------------------------------------------------------------------------- #
# 8/9 — §19.3: the envelope owns the unresolved backup/rollback fact
# --------------------------------------------------------------------------- #

def test_failed_rollback_retry_never_destroys_the_prior_result(
        tmp_path, monkeypatch):
    """A retry must repair/retain the prior backup before staging again."""
    target = tmp_path / "scan.nxs"
    backup = tmp_path / f"scan.nxs{nwt._REPLACING_SUFFIX}"
    prior = b"prior durable bytes"
    target.write_bytes(prior)
    frozen = SimpleNamespace(output_mode="Overwrite")
    prepared = _envelope(frozen, target)
    worker = SimpleNamespace(file_lock=threading.RLock())
    scan = SimpleNamespace(data_file=str(target))
    pool = SimpleNamespace(pause=lambda _path: None, resume=lambda _path: None)
    monkeypatch.setattr(nwt, "_get_h5pool", lambda: pool)

    real_unlink = Path.unlink
    target_unlink_attempts = 0

    def fail_first_target_unlink(path, *args, **kwargs):
        nonlocal target_unlink_attempts
        if path == target:
            target_unlink_attempts += 1
            if target_unlink_attempts == 1:
                raise OSError("injected rollback unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_first_target_unlink)
    writes = 0

    def fail_writer():
        nonlocal writes
        writes += 1
        target.write_bytes(f"partial-{writes}".encode())
        raise OSError(f"writer failure {writes}")

    with pytest.raises(OSError, match="writer failure 1"):
        nexusThread._write_run_result(worker, prepared, scan, fail_writer)

    assert backup.read_bytes() == prior
    assert target.read_bytes() == b"partial-1"
    assert prepared.target_replaced is False

    with pytest.raises(OSError, match="writer failure 2"):
        nexusThread._write_run_result(worker, prepared, scan, fail_writer)

    surviving = [
        path.read_bytes()
        for path in (target, backup)
        if path.exists()
    ]
    assert prior in surviving, (
        "the exact retry destroyed the only byte-identical prior result")
    assert prepared.target_replaced is False


def test_an_unowned_replacing_artifact_is_refused_not_destroyed(
        tmp_path, monkeypatch):
    """§19.3: never delete an unresolved prior backup merely to reuse the
    suffix.  A ``.xdart-replacing`` file this envelope never staged may be the
    only durable prior of a crashed run — staging must refuse, loudly, and the
    writer must not run."""
    target = tmp_path / "scan.nxs"
    target.write_bytes(b"current prior")
    stray = tmp_path / f"scan.nxs{nwt._REPLACING_SUFFIX}"
    stray_bytes = b"unresolved prior of a crashed run"
    stray.write_bytes(stray_bytes)
    frozen = SimpleNamespace(output_mode="Overwrite")
    prepared = _envelope(frozen, target)
    worker = SimpleNamespace(file_lock=threading.RLock())
    scan = SimpleNamespace(data_file=str(target))
    pool = SimpleNamespace(pause=lambda _path: None, resume=lambda _path: None)
    monkeypatch.setattr(nwt, "_get_h5pool", lambda: pool)
    writes: list[str] = []

    with pytest.raises(OSError):
        nexusThread._write_run_result(
            worker, prepared, scan, lambda: writes.append("write"))

    assert writes == [], "the writer ran over an unresolved prior backup"
    assert stray.read_bytes() == stray_bytes, (
        "the unowned backup was destroyed to reuse the staging suffix")
    assert target.read_bytes() == b"current prior"
    assert prepared.target_replaced is False


# --------------------------------------------------------------------------- #
# 10-13 — O-3N.R.3-3 correction rows (adversarial-panel findings R3-F1/F2/F3)
# --------------------------------------------------------------------------- #

def test_a_zero_publication_run_never_sweeps_the_prior_xye_tail(
        widget, tmp_path, monkeypatch):
    """§16.4 applied to the XYE half (panel finding R3-F1, P1).

    A COMPLETE real Overwrite run that publishes NOTHING — the operator hits
    Stop before the first chunk — must leave the prior run's XYE output
    byte-for-byte.  The terminal stale sweep exists to gate THIS run's
    publication; decoupled from any publication it silently destroys a
    durable prior result and creates nothing.
    """
    wrangler = _select_nexus(widget)
    widget.controls.set_write_mode("Overwrite")
    src, out = _arm_pilatus(wrangler, tmp_path)
    scan_dir = out / Path(src).stem
    scan_dir.mkdir(parents=True)
    prior = scan_dir / f"iq_{Path(src).stem}_0009.xye"
    prior.write_text("prior run output")
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()
    thread = wrangler.thread
    said: list[str] = []
    thread.showLabel.connect(said.append)
    thread.command = 'stop'                       # Stop before the first chunk

    thread.run()

    assert prior.exists() and prior.read_text() == "prior run output", (
        "a run that published nothing destroyed the prior run's XYE output")
    assert any(m.startswith("Done") for m in said), said


def test_a_commit_leftover_backup_is_retried_at_the_next_commit(
        tmp_path, monkeypatch):
    """Panel finding R3-F2: a backup RESOLVED by a successful commit whose
    unlink transiently failed is the envelope's fact to finish — the next
    successful commit of the same run removes it, so one transient lock does
    not strand a suffix that permanently refuses every later Overwrite run."""
    target = tmp_path / "scan.nxs"
    backup = tmp_path / f"scan.nxs{nwt._REPLACING_SUFFIX}"
    target.write_bytes(b"prior durable bytes")
    frozen = SimpleNamespace(output_mode="Overwrite")
    prepared = _envelope(frozen, target)
    worker = SimpleNamespace(file_lock=threading.RLock())
    scan = SimpleNamespace(data_file=str(target))
    pool = SimpleNamespace(pause=lambda _path: None, resume=lambda _path: None)
    monkeypatch.setattr(nwt, "_get_h5pool", lambda: pool)

    real_unlink = Path.unlink
    backup_unlink_attempts = 0

    def fail_first_backup_unlink(path, *args, **kwargs):
        nonlocal backup_unlink_attempts
        if path == backup:
            backup_unlink_attempts += 1
            if backup_unlink_attempts == 1:
                raise OSError("transient lock on the replaced prior")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_first_backup_unlink)

    def write_ok():
        target.write_bytes(b"new run bytes")

    nexusThread._write_run_result(worker, prepared, scan, write_ok)
    assert prepared.target_replaced is True
    assert backup.exists(), "the injected transient unlink failure never fired"

    nexusThread._write_run_result(worker, prepared, scan, write_ok)

    assert not backup.exists(), (
        "the commit-resolved leftover backup was forgotten; the next "
        "Overwrite run would refuse forever over a disposable file")
    assert target.read_bytes() == b"new run bytes"


def test_a_stuck_commit_backup_is_surfaced_at_run_end(
        widget, tmp_path, monkeypatch):
    """Panel finding R3-F2, visibility half: when the leftover cannot be
    removed by run end, the operator is TOLD — on the same visible label —
    which file to remove, at the moment it happens, not at the next run's
    unexplained failure."""
    wrangler = _select_nexus(widget)
    widget.controls.set_write_mode("Overwrite")
    src, out = _arm_pilatus(wrangler, tmp_path)
    target = out / f"{Path(src).stem}.nxs"
    target.write_bytes(b"prior durable result")
    backup = out / f"{Path(src).stem}.nxs{nwt._REPLACING_SUFFIX}"
    real_unlink = Path.unlink

    def refuse_backup_unlink(path, *args, **kwargs):
        if path == backup:
            raise OSError("persistent lock on the replaced prior")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_backup_unlink)
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()
    thread = wrangler.thread
    said: list[str] = []
    thread.showLabel.connect(said.append)

    thread.run()

    assert backup.exists()
    assert any(backup.name in m for m in said), (
        f"the stuck superseded backup was never surfaced to the operator: "
        f"{said}")


def test_release_drops_withheld_xye_entries_with_their_envelope(
        widget, tmp_path, monkeypatch):
    """Panel finding R3-F3: withheld XYE entries can never publish once their
    envelope dies — left in the buffer, a later run on the same worker would
    drain them under a FOREIGN identity.  They go with the envelope."""
    wrangler = _select_nexus(widget)
    _src, _out, thread = _started(wrangler, tmp_path, monkeypatch)
    prepared = nexusThread._prepare_execution(thread, thread.run_configuration)
    prepared.xye_tail_pending = [tmp_path / "scan" / "iq_scan_0009.xye"]
    prepared.xye_withheld_idxs = {0, 1}
    with thread._xye_lock:
        thread._xye_buffer.append((0, object()))
        thread._xye_buffer.append((1, object()))

    nexusThread._release_execution(thread)

    with thread._xye_lock:
        leftover = list(thread._xye_buffer)
    assert leftover == [], (
        "withheld XYE entries outlived their envelope; a later run would "
        "publish them under its own identity")
