"""Qt-free output target transactions and run-owned XYE publication.

This module is an unmounted value/state kernel.  It deliberately owns no
NeXus schema, writer, path-selection policy, source preparation, or GUI
composition.  A later consumer may compose it with those owners after the
kernel's contract has been accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import errno
import hashlib
import os
from pathlib import Path
import secrets
import sys
import threading
import time
from contextlib import nullcontext
from typing import Callable, Mapping, Protocol, TypeVar, runtime_checkable

from xrd_tools.io.stat_identity import identity_ctime_ns
from xrd_tools.session import get_pool


_BACKUP_SUFFIX = ".xdart-replacing"
_HASH_CHUNK_BYTES = 1024 * 1024
_REPLACE_RETRIES = 10
_REPLACE_RETRY_DELAY_S = 0.25

# Narrow seams for deterministic fault injection.  Production still performs
# the standard-library operations directly through these aliases.
_replace = os.replace
_link = os.link


def _unlink(path) -> None:
    Path(path).unlink()


def _normalize_target(path) -> str:
    """Use the same absolute/normcase identity as H10's public HDF5 pool."""
    return os.path.normcase(os.path.abspath(str(path)))


def _lease_key(target: str) -> str:
    """The identity two spellings of ONE file cannot escape.

    Codex F1 on `da228738`.  The lease registry keyed on
    `normcase(abspath(...))`, which is a pure STRING transform -- and on POSIX
    `normcase` is the identity function.  So `<root>/processed/slot.nexus` and
    `<root>/PROCESSED/slot.nexus`, which are the SAME FILE on a case-insensitive
    volume (verified with `os.path.samefile`), produced two different keys and
    two simultaneous leases.  That is survivable while a no-clobber `os.link`
    still refuses the second writer; it is not survivable once replacement
    removes that link, which is exactly why this lands with it.

    The fix is an IDENTITY, not a spelling rule.  The parent directory is
    stat'ed, and its `(st_dev, st_ino)` replaces the whole directory portion of
    the path: that collapses case-aliased directories, symlinked directories,
    `..` segments and relative spellings in one step, without assuming anything
    about the platform.

    NO BLANKET LOWERCASING.  Codex's direction was explicit that it would
    conflate genuinely distinct files on a case-sensitive volume.  The final
    NAME is therefore casefolded only when this directory is OBSERVED to treat
    two spellings of it as one file -- which is checkable exactly when the file
    exists, and when it exists is when replacement has something to lose.

    TWO RESIDUAL GAPS, stated rather than papered over.

    1. When the parent cannot be stat'ed -- it does not exist yet, or is
       unreadable -- this falls back to the raw string key.  Two aliasing
       SPELLINGS of a not-yet-existing directory would each get a string key
       and both proceed.

       CORRECTED (Fable F2 on `58123dc7`): an earlier version of this note
       claimed the gap was narrow because "the publisher opens its parent
       before acquiring".  That is true of the FINITE publisher and false of
       the two callers that matter most -- ordinary Run's `acquire_lease` and
       Average's `hold_target` both run BEFORE the destination directory is
       created (`_copy_stream_seed` makes it), so they routinely acquire
       against a missing parent.  The same-pathname exclusion in `_acquire`
       is what actually covers them; do not rely on the parent existing.
    2. Two spellings of the FILE NAME in one directory, when the file does
       not exist yet, cannot be told apart without creating a probe file in
       the operator's output directory.  When the file DOES exist the check
       below settles it -- and an existing file is precisely when
       replacement has something to lose.
    """
    directory, name = os.path.split(target)
    try:
        state = os.stat(directory or ".")
    except OSError:
        return target
    folded = name.casefold()
    if folded != name:
        try:
            here = os.stat(os.path.join(directory, name))
            there = os.stat(os.path.join(directory, folded))
            if (here.st_dev, here.st_ino) == (there.st_dev, there.st_ino):
                name = folded
        except OSError:
            # Not both present: the directory's case rule is unobservable
            # without creating a file, and creating one here would be a side
            # effect on the operator's output directory.  Keep the exact name.
            pass
    return f"{state.st_dev}:{state.st_ino}:{os.path.normcase(name)}"


def _held_open_message(target: str, verb: str) -> str:
    """The ONE operator-facing account of a rename some holder is blocking.

    Spelled once so the streamed staging path and the finite publication path
    cannot drift into two different explanations of the same condition (DIR-3).
    """
    return (
        f"could not {verb} '{target}' for replacement: the file is held open by "
        "another program (on Windows an open handle blocks the rename, "
        "including antivirus, search indexing, a preview pane or another SMB "
        "client). The exact existing file was left untouched; stop the other "
        "program and retry."
    )


def replace_into_place(
    source,
    target,
    *,
    verb: str = "publish",
    src_dir_fd: int | None = None,
    dst_dir_fd: int | None = None,
) -> None:
    """Atomically install *source* at *target*, retrying a held-open rename.

    ONE replacement primitive for the finite operations.  `os.replace` is atomic
    on POSIX and on Windows, so a reader sees either the whole prior file or the
    whole new one and never a partial write -- which is what lets ADR-0010 say
    "the prior slot remains visible and untouched until that publication step".

    The retry is for Windows/SMB only (DIR-3): an open handle -- antivirus,
    search indexing, a preview pane, another SMB client -- makes the rename fail
    with a PermissionError that clears once the holder lets go.  Any OTHER error
    is raised immediately rather than retried.

    Pass *src_dir_fd* / *dst_dir_fd* to rename by NAME relative to an already
    open directory descriptor, as the finite publisher does.  That keeps the
    rename immune to the parent directory being moved or swapped underneath it
    between validation and publication.

    NOT used by `OutputTransaction._stage_prior`, deliberately.  That is the
    streamed Run path, which ADR-0010 leaves unchanged, and its loop interleaves
    reservation and receipt capture between attempts rather than performing a
    bare rename.  Converting it would put Run's hot output path at risk for a
    cosmetic saving.  The POLICY is shared instead: both use `_REPLACE_RETRIES`,
    `_REPLACE_RETRY_DELAY_S` and `_held_open_message`.
    """
    attempts = max(1, int(_REPLACE_RETRIES))
    for attempt in range(1, attempts + 1):
        try:
            _replace(
                source, target,
                src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd,
            )
            return
        except PermissionError as error:
            if attempt == attempts:
                raise PermissionError(
                    _held_open_message(_normalize_target(target), verb)
                ) from error
            time.sleep(max(0.0, float(_REPLACE_RETRY_DELAY_S)))


class OutputTransactionError(RuntimeError):
    """Base class for shared output-transaction refusals."""


class OwnershipRefused(OutputTransactionError):
    """An equal-valued or foreign object tried to act for an exact owner."""


class TargetChanged(OutputTransactionError):
    """The admitted target presence or content identity no longer matches."""


class LeaseUnavailable(OutputTransactionError):
    """Another live lease owns the normalized target."""


class TransactionStateError(OutputTransactionError):
    """The requested operation is invalid in the current truthful phase."""


class CleanupIncomplete(OutputTransactionError):
    """A fallible cleanup remains owned and retryable."""

    def __init__(self, snapshot):
        super().__init__(f"output cleanup is incomplete: {snapshot!r}")
        self.snapshot = snapshot


class OutputReceiptCapability(Enum):
    """Public durable-receipt capabilities supplied by output owners."""

    DURABLE_XYE = "durable-xye-v1"


@runtime_checkable
class OutputReceiptCapabilityProvider(Protocol):
    """Public structural seam for transaction-qualified receipt owners."""

    @property
    def output_receipt_capabilities(self) -> frozenset[OutputReceiptCapability]: ...


class LeaseOwner(str, Enum):
    """Independent owners that must all release one target lease."""

    RUN = "run"
    SESSION = "session"
    SOURCE = "source"
    CLEANUP = "cleanup"


class TransactionPhase(str, Enum):
    """Publicly meaningful phases of one admitted output transaction."""

    ADMITTED = "admitted"
    LEASED = "leased"
    READY_TO_RETRY = "ready-to-retry"
    EXECUTING = "executing"
    EPOCH_COMMITTED = "epoch-committed"
    ROLLBACK_PENDING = "rollback-pending"
    CLEANUP_PENDING = "cleanup-pending"
    INTEGRITY_HOLD = "integrity-hold"
    COMMITTED = "committed"
    ABORTED = "aborted"


class StreamSeedMode(str, Enum):
    """Initial bytes installed for one final-path streaming writer."""

    PRESERVE_BASE = "preserve-base"
    EMPTY_REPLACEMENT = "empty-replacement"


class RetryAction(str, Enum):
    """Exact fallible transition still owned by a transaction."""

    POOL_PAUSE = "pool-pause"
    BACKUP_RESERVATION = "backup-reservation"
    STAGE_PRIOR = "stage-prior"
    CANDIDATE_RESERVATION = "candidate-reservation"
    WRITER_CAPTURE = "writer-capture"
    PUBLICATION = "publication"
    ROLLBACK = "rollback"
    BACKUP_UNLINK = "backup-unlink"
    CANDIDATE_UNLINK = "candidate-unlink"
    CANDIDATE_RETIRE = "candidate-retire"
    STREAM_RETIRE = "stream-retire"
    POOL_RESUME = "pool-resume"


_RETRY_ORDER = (
    RetryAction.PUBLICATION,
    RetryAction.STAGE_PRIOR,
    RetryAction.BACKUP_RESERVATION,
    RetryAction.CANDIDATE_RESERVATION,
    RetryAction.WRITER_CAPTURE,
    RetryAction.ROLLBACK,
    RetryAction.BACKUP_UNLINK,
    RetryAction.CANDIDATE_UNLINK,
    RetryAction.CANDIDATE_RETIRE,
    RetryAction.STREAM_RETIRE,
    RetryAction.POOL_RESUME,
    RetryAction.POOL_PAUSE,
)


@dataclass(frozen=True)
class OwnerToken:
    """Immutable value token whose authority is nevertheless identity-based."""

    label: str


@dataclass(frozen=True)
class TargetSnapshot:
    """Content-sensitive snapshot of one normalized output target."""

    exists: bool
    size: int | None
    mtime_ns: int | None
    device: int | None
    inode: int | None
    digest: str | None
    # A full read can be revalidated against this exact filesystem revision.
    # Keep content equality unchanged (e.g. a rename changes ctime, not bytes).
    ctime_ns: int | None = field(default=None, compare=False)


@dataclass(frozen=True)
class _FileIdentity:
    """Immutable filesystem-object identity, separate from content truth."""

    device: int
    inode: int


@dataclass(frozen=True)
class _ObjectReceipt:
    """Positively verified identity and full integrity for one pathname role."""

    path: str
    identity: _FileIdentity
    snapshot: TargetSnapshot
    role: str


@dataclass(frozen=True)
class _StreamStatReceipt:
    """Post-fsync inode facts plus bounded semantic readback evidence."""

    path: str
    identity: _FileIdentity
    size: int
    mtime_ns: int
    ctime_ns: int
    evidence_digest: str
    evidence_bytes: int
    ordinal: int


@dataclass(frozen=True)
class _StageReceipt:
    """Authority installed before a target-to-backup replacement."""

    target: str
    backup: str
    admitted_prior: _ObjectReceipt
    placeholder: _ObjectReceipt


@dataclass(frozen=True)
class _WriterReceipt:
    """Candidate namespace authority and, after verification, writer result."""

    reservation: _ObjectReceipt
    writer_returned: bool
    result: _ObjectReceipt | None


@dataclass(frozen=True)
class _LinkReceipt:
    """Authority installed before a publication or restoration hard link."""

    source: _ObjectReceipt
    destination: str
    role: str


@dataclass(frozen=True)
class _LinkResolutionReceipt:
    """Positive proof that both names matched one immutable link receipt."""

    link: _LinkReceipt
    source_observed: TargetSnapshot
    destination_observed: TargetSnapshot
    role: str


@dataclass(frozen=True)
class _SourceCleanupReceipt:
    """Authority installed before removing a positively resolved source."""

    resolution: _LinkResolutionReceipt
    source: str
    observed: TargetSnapshot
    role: str


@dataclass(frozen=True)
class _TerminalReceipt:
    """Positive proof for a public ready or committed terminal state."""

    target: str
    expected: TargetSnapshot
    observed: TargetSnapshot
    role: str


@dataclass(frozen=True)
class TargetAdmission:
    """The exact admission object accepted by one transaction."""

    target: str
    snapshot: TargetSnapshot
    ordinal: int


@dataclass(frozen=True)
class TargetLease:
    """Opaque target lease; callers must return this exact instance."""

    target: str
    ordinal: int


@dataclass(frozen=True)
class StreamAttempt:
    """Opaque identity for one transaction-owned final-path writer attempt."""

    target: str
    ordinal: int


@dataclass(frozen=True)
class _StreamCloseAttempt:
    """Exact authority for one controlled close after a durable checkpoint."""

    target: str
    ordinal: int


@dataclass(frozen=True)
class StreamCheckpoint:
    """Post-fsync identity and positive dirty-row evidence at one boundary."""

    target: str
    size: int
    mtime_ns: int
    ctime_ns: int
    evidence_digest: str
    evidence_bytes: int
    ordinal: int


@dataclass(frozen=True)
class StreamTerminal:
    """Writer terminal receipt, optionally bound to an exact file object."""

    target: str
    size: int
    digest: str
    ordinal: int
    device: int | None = None
    inode: int | None = None
    mtime_ns: int | None = None
    ctime_ns: int | None = None

    def __post_init__(self) -> None:
        if (
            type(self.target) is not str
            or not self.target
            or type(self.size) is not int
            or self.size < 0
            or type(self.digest) is not str
            or len(self.digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.digest
            )
            or type(self.ordinal) is not int
            or self.ordinal < 1
            or not (
                all(value is None for value in (
                    self.device, self.inode, self.mtime_ns, self.ctime_ns,
                ))
                or all(
                    type(value) is int and value >= 0
                    for value in (
                        self.device,
                        self.inode,
                        self.mtime_ns,
                        self.ctime_ns,
                    )
                )
            )
        ):
            raise TypeError("stream terminal seal is invalid")


def stream_terminal_object_revision(
    value: object,
) -> tuple[int, int, int, int, int] | None:
    """Return the exact stat identity carried by a modern terminal seal."""

    if type(value) is not StreamTerminal or not all(
        type(item) is int and item >= 0
        for item in (
            value.device,
            value.inode,
            value.mtime_ns,
            value.ctime_ns,
        )
    ):
        return None
    return (
        value.device,
        value.inode,
        value.size,
        value.mtime_ns,
        value.ctime_ns,
    )


@dataclass(frozen=True)
class TargetHold:
    """An exclusive claim on one target for a writer that publishes ITSELF.

    `OutputTransaction` assumes it performs the write, so `abandon` refuses once
    the target has changed and `release_lease_owner` blocks the CLEANUP role
    outside a terminal phase.  A finite operation publishes out of band -- it
    builds a private candidate and renames it into place -- so it can never
    reach that terminal phase, and driving the transaction would leave the
    target leased forever.

    This is the SAME lease registry, keyed the same normalized way, so a hold
    and an ordinary Run/Append/Average lease exclude each other exactly as two
    Runs would.  It is only the state machine that is skipped.
    """

    lease: TargetLease
    owners: "Mapping[LeaseOwner, OwnerToken]"


@dataclass(frozen=True)
class CleanupToken:
    """Opaque retry owner; equal-valued copies have no authority."""

    target: str
    ordinal: int
    domain: str


@dataclass(frozen=True)
class LeaseSnapshot:
    """Detached immutable lease state."""

    target: str
    ordinal: int
    active: bool
    remaining_owners: tuple[LeaseOwner, ...]


@dataclass(frozen=True)
class TransactionSnapshot:
    """Detached immutable transaction state for composition and evidence."""

    target: str
    phase: TransactionPhase
    admission: TargetSnapshot
    pending_actions: tuple[RetryAction, ...]
    writer_succeeded: bool
    retryable: bool
    cleanup_token: CleanupToken | None
    partial_path: str | None = None
    durable_floor: StreamCheckpoint | None = None


@dataclass(frozen=True)
class XyeSnapshot:
    """Detached immutable state of one run-owned XYE publication."""

    directory: str
    staged_indices: tuple[int, ...]
    pending_stale: tuple[str, ...]
    complete: bool
    retryable: bool
    cleanup_token: CleanupToken | None


@dataclass
class _LeaseState:
    lease: TargetLease
    owners: dict[LeaseOwner, OwnerToken]


def _require_identity(actual, expected, name: str) -> None:
    # Contract-bearing: authority is object identity, even when values compare
    # equal.  This single seam is intentionally covered by a restored mutant.
    if actual is not expected:
        raise OwnershipRefused(f"foreign {name}; exact owner object required")


# The tree-wide win32 ctime seam (xrd_tools.io.stat_identity) applies to
# every comparison that has a PATHNAME view on one side: win32 fills a
# pathname stat's st_ctime from the creation time, so against a descriptor
# view (or a descriptor-recorded receipt) the slot is neutral there and
# identity is (dev, ino, size, mtime_ns).  A descriptor view compared with
# a descriptor-recorded receipt keeps ctime on every platform: win32 fills
# fstat's st_ctime from NTFS ChangeTime, which every write and every utime
# advance, so a same-size same-mtime in-place rewrite still changes it and
# the stat-only revalidators below cannot hand out a stale digest for it.
# Receipts always record the descriptor's observed ctime.
_identity_ctime_ns = identity_ctime_ns


def _comparable_identity(
    identity: tuple[int, int, int, int, int],
) -> tuple[int, int, int, int, int]:
    """Normalise a sealed (dev, ino, size, mtime_ns, ctime_ns) for a
    comparison that involves a pathname view."""
    return (*identity[:4], _identity_ctime_ns(identity[4]))


def _stat_identity(stat_result) -> tuple[int, int, int, int, int]:
    """The seamed identity of one stat view (pathname or descriptor)."""
    return (*_descriptor_identity(stat_result)[:4],
            _identity_ctime_ns(stat_result.st_ctime_ns))


def _descriptor_identity(stat_result) -> tuple[int, int, int, int, int]:
    """The exact (dev, ino, size, mtime_ns, ctime_ns) of one DESCRIPTOR view.

    Compare it only with another descriptor view or a descriptor-recorded
    receipt (``StreamTerminal``, ``TargetSnapshot.ctime_ns``); a pathname
    view goes through ``_stat_identity``.
    """
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
    )


# Windows only flushes a handle that was opened with write access
# (``os.fsync`` is ``_commit`` -> ``FlushFileBuffers``), so the read-only
# descriptors this module seals through raise ``EBADF`` there; POSIX flushes
# any descriptor.
_FSYNC_REQUIRES_WRITE_ACCESS = sys.platform == "win32"


def _fsync_descriptor(descriptor: int, path: Path) -> None:
    """Flush ``descriptor`` to stable storage.

    Where the platform refuses to flush a read-only descriptor, flush the
    same file through a second, writable descriptor of ``path`` instead —
    after proving that the reopened pathname names the descriptor's exact
    inode, the guard every seal in this module already applies.
    """
    try:
        os.fsync(descriptor)
        return
    except OSError as error:
        if not _FSYNC_REQUIRES_WRITE_ACCESS or error.errno != errno.EBADF:
            raise
    expected = _stat_identity(os.fstat(descriptor))
    writable = os.open(path, os.O_RDWR)
    try:
        if _stat_identity(os.fstat(writable)) != expected:
            raise TargetChanged(
                f"durable flush reopened a foreign inode: {path}"
            )
        os.fsync(writable)
    finally:
        os.close(writable)


def _sha256_handle(handle) -> str:
    digest = hashlib.sha256()
    while True:
        block = handle.read(_HASH_CHUNK_BYTES)
        if not block:
            return digest.hexdigest()
        digest.update(block)


def _capture_target(target: str, *, hash_content: bool = True) -> TargetSnapshot:
    """Capture one stable pathname/inode/content observation.

    The public value intentionally stores size/mtime/inode plus a digest.
    Device/inode detect replacement and the digest detects a deliberately
    same-stat in-place mutation.  Internal ctime checks only prove that the
    hash itself was not sampled across a concurrent write.
    """
    if type(hash_content) is not bool:
        raise TypeError("hash_content must be an exact bool")
    try:
        before = os.stat(target)
    except FileNotFoundError:
        try:
            os.stat(target)
        except FileNotFoundError:
            return TargetSnapshot(False, None, None, None, None, None)
        raise TargetChanged(f"target appeared while admitting {target}")

    if not hash_content:
        try:
            after = os.stat(target)
        except FileNotFoundError as exc:
            raise TargetChanged(
                f"target disappeared while admitting {target}"
            ) from exc
        if _stat_identity(before) != _stat_identity(after):
            raise TargetChanged(f"target mutated while fingerprinting {target}")
        return TargetSnapshot(
            True, int(after.st_size), int(after.st_mtime_ns),
            int(after.st_dev), int(after.st_ino), hashlib.sha256(b"").hexdigest() if after.st_size == 0 else None,
        )

    try:
        with open(target, "rb") as handle:
            opened = os.fstat(handle.fileno())
            digest = _sha256_handle(handle)
            finished = os.fstat(handle.fileno())
    except FileNotFoundError as exc:
        raise TargetChanged(f"target disappeared while admitting {target}") from exc
    try:
        after = os.stat(target)
    except FileNotFoundError as exc:
        raise TargetChanged(f"target disappeared while admitting {target}") from exc
    identities = {
        _stat_identity(before),
        _stat_identity(opened),
        _stat_identity(finished),
        _stat_identity(after),
    }
    # The two descriptor views bracket the hash: exact, ctime included, so a
    # same-size same-mtime rewrite inside the window is refused where the
    # pathname compare is neutral (win32) rather than digested torn.
    if len(identities) != 1 or (
        _descriptor_identity(opened) != _descriptor_identity(finished)
    ):
        raise TargetChanged(f"target mutated while fingerprinting {target}")
    # The four views agree on identity; the descriptor's closing view is the
    # recorded one (identical to ``after`` wherever ctime is part of
    # identity; the change time rather than the creation time on win32), so
    # ``revalidate_target_snapshot`` can hold its descriptor views to it.
    return TargetSnapshot(
        True,
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_dev),
        int(after.st_ino),
        digest,
        int(finished.st_ctime_ns),
    )


def capture_target_snapshot(path: Path | str) -> TargetSnapshot:
    """Return one stable, content-sensitive observation without ownership."""
    return _capture_target(os.path.realpath(os.fspath(path)))


def revalidate_target_snapshot(
    path: Path | str, snapshot: TargetSnapshot,
) -> TargetSnapshot:
    """Reuse a content digest only while its exact file revision is unchanged.

    This uses the same filesystem revision as stream-terminal revalidation;
    ctime also detects in-place edits that restore the previous size/mtime.
    Snapshots without that observation retain the full content check.
    """
    if type(snapshot) is not TargetSnapshot or not snapshot.exists:
        raise TypeError("target revalidation requires an existing TargetSnapshot")
    if snapshot.ctime_ns is None or snapshot.digest is None:
        current = capture_target_snapshot(path)
        if current != snapshot:
            raise TargetChanged("target changed since its content snapshot")
        return current
    expected = (
        snapshot.device, snapshot.inode, snapshot.size,
        snapshot.mtime_ns, int(snapshot.ctime_ns),
    )
    target = os.path.realpath(os.fspath(path))
    try:
        descriptor = os.open(target, os.O_RDONLY)
    except OSError as error:
        raise TargetChanged(f"target is unavailable: {target}") from error
    try:
        before = os.fstat(descriptor)
        try:
            named = os.stat(target)
        except OSError as error:
            raise TargetChanged(f"target pathname is unavailable: {target}") from error
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not _revision_holds(expected, before, named, after):
        raise TargetChanged("target changed since its content snapshot")
    return snapshot


def _revision_holds(
    expected: tuple[int, int, int, int, int],
    before: os.stat_result,
    named: os.stat_result,
    after: os.stat_result,
) -> bool:
    """Whether a descriptor-recorded revision still names the open object.

    The two descriptor views must match the recorded revision exactly (ctime
    included, on every platform); the pathname view between them is held to
    the seamed identity only.  Every revalidator that hands out a recorded
    digest without rereading the bytes goes through this.
    """
    return (
        _descriptor_identity(before) == expected
        and _descriptor_identity(after) == expected
        and _stat_identity(named) == _comparable_identity(expected)
    )


def revalidate_stream_terminal(
    path: Path | str,
    terminal: StreamTerminal,
) -> TargetSnapshot:
    """Revalidate one writer-bound terminal object without rehashing it."""

    if type(terminal) is not StreamTerminal:
        raise TypeError("terminal revalidation requires exact StreamTerminal")
    target = _normalize_target(path)
    if target != terminal.target:
        raise TargetChanged("stream terminal target does not match browse path")
    revision = stream_terminal_object_revision(terminal)
    if revision is None:
        raise TargetChanged("stream terminal lacks an exact object revision")
    try:
        descriptor = os.open(target, os.O_RDONLY)
    except OSError as exc:
        raise TargetChanged(
            f"stream terminal target is unavailable: {target}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        try:
            named = os.stat(target)
        except OSError as exc:
            raise TargetChanged(
                f"stream terminal pathname is unavailable: {target}"
            ) from exc
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not _revision_holds(revision, before, named, after):
        raise TargetChanged("stream terminal object changed before browse")
    return TargetSnapshot(
        True,
        terminal.size,
        terminal.mtime_ns,
        terminal.device,
        terminal.inode,
        terminal.digest,
    )


def _identity(snapshot: TargetSnapshot) -> _FileIdentity | None:
    if not snapshot.exists or snapshot.device is None or snapshot.inode is None:
        return None
    return _FileIdentity(snapshot.device, snapshot.inode)


def _descriptor_receipt(descriptor: int, path: Path, role: str) -> _ObjectReceipt:
    """Bind a newly created empty reservation before close or path lookup.

    ``O_CREAT|O_EXCL`` establishes that this descriptor names the newly
    reserved object.  The reservation is intentionally empty, so its complete
    content digest is known without adding an independent durability policy.
    """
    stat_result = os.fstat(descriptor)
    if int(stat_result.st_size) != 0:
        raise TargetChanged(f"new {role} reservation was not empty")
    snapshot = TargetSnapshot(
        True,
        0,
        int(stat_result.st_mtime_ns),
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        hashlib.sha256(b"").hexdigest(),
    )
    return _ObjectReceipt(
        _normalize_target(path),
        _FileIdentity(int(stat_result.st_dev), int(stat_result.st_ino)),
        snapshot,
        role,
    )


def _bind_descriptor_to_receipt(
    descriptor: int,
    path: Path,
    receipt: _ObjectReceipt,
) -> tuple[int, int, int, int, int]:
    """Bind an open pathname to prior exact authority without rereading it.

    The receipt already owns the complete digest from admission/staging.  This
    seam binds the newly opened descriptor and its pathname to that receipt's
    immutable object/stat identity.  The caller must still supply content
    evidence gathered while performing its required read and seal against the
    returned full stat tuple.
    """
    normalized = _normalize_target(path)
    if receipt.path != normalized:
        raise TargetChanged(
            f"{receipt.role} receipt does not own descriptor path {path}"
        )
    snapshot = receipt.snapshot
    if (
        not snapshot.exists
        or snapshot.size is None
        or snapshot.mtime_ns is None
        or snapshot.device is None
        or snapshot.inode is None
        or snapshot.digest is None
        or receipt.identity != _identity(snapshot)
    ):
        raise TransactionStateError(
            f"{receipt.role} receipt lacks exact immutable authority"
        )
    before = os.fstat(descriptor)
    try:
        named = os.stat(path)
    except FileNotFoundError as exc:
        raise TargetChanged(
            f"{receipt.role} pathname disappeared before descriptor binding: {path}"
        ) from exc
    after = os.fstat(descriptor)
    identities = {
        _stat_identity(before),
        _stat_identity(named),
        _stat_identity(after),
    }
    expected_prefix = (
        receipt.identity.device,
        receipt.identity.inode,
        int(snapshot.size),
        int(snapshot.mtime_ns),
    )
    observed = _stat_identity(after)
    if len(identities) != 1 or observed[:4] != expected_prefix:
        raise TargetChanged(
            f"{receipt.role} descriptor does not match its exact receipt: {path}"
        )
    return observed


def _descriptor_content_receipt(
    descriptor: int,
    path: Path,
    role: str,
    *,
    durable_fsync: bool = True,
    evidence_digest: str | None = None,
    expected_stat: tuple[int, int, int, int, int] | None = None,
) -> _ObjectReceipt:
    """Seal exact descriptor identity, hashing unless evidence is supplied."""
    if type(durable_fsync) is not bool:
        raise TypeError("durable_fsync must be an exact bool")
    if expected_stat is not None:
        expected_stat = _comparable_identity(expected_stat)
    if durable_fsync:
        _fsync_descriptor(descriptor, path)
    before = os.fstat(descriptor)
    if expected_stat is not None and _stat_identity(before) != expected_stat:
        raise TargetChanged(
            f"{role} descriptor changed before content verification: {path}"
        )
    if evidence_digest is None:
        offset = os.lseek(descriptor, 0, os.SEEK_CUR)
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as handle:
            digest = _sha256_handle(handle)
        os.lseek(descriptor, offset, os.SEEK_SET)
    else:
        digest = _require_evidence_digest(evidence_digest)
    after = os.fstat(descriptor)
    # Two descriptor views bracketing the hash: exact, ctime included.  The
    # caller's expected identity was normalised to the seamed form above and
    # stays a seamed compare.
    if (
        _descriptor_identity(before) != _descriptor_identity(after)
        or (
            expected_stat is not None
            and _stat_identity(after) != expected_stat
        )
    ):
        raise TargetChanged(f"{role} mutated while sealing descriptor {path}")
    snapshot = TargetSnapshot(
        True,
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_dev),
        int(after.st_ino),
        digest,
    )
    receipt = _ObjectReceipt(
        _normalize_target(path),
        _FileIdentity(int(after.st_dev), int(after.st_ino)),
        snapshot,
        role,
    )
    try:
        observed = os.stat(path)
    except FileNotFoundError as exc:
        raise TargetChanged(f"{role} pathname disappeared: {path}") from exc
    if (
        _stat_identity(observed) != _stat_identity(after)
        or (
            expected_stat is not None
            and _stat_identity(observed) != expected_stat
        )
    ):
        raise TargetChanged(f"{role} pathname did not match sealed descriptor {path}")
    return receipt


def _require_evidence_digest(value: str) -> str:
    digest = str(value)
    if len(digest) != 64:
        raise TransactionStateError("stream evidence digest must be SHA-256 hex")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise TransactionStateError(
            "stream evidence digest must be SHA-256 hex"
        ) from exc
    return digest.lower()


def _descriptor_stream_stat_receipt(
    descriptor: int,
    path: Path,
    *,
    evidence_digest: str,
    evidence_bytes: int,
    ordinal: int,
    role: str,
    expected_stat: tuple[int, int, int, int, int] | None = None,
    durable_fsync: bool = True,
) -> _StreamStatReceipt:
    """Seal bounded evidence to the exact descriptor identity.

    This helper intentionally performs no content read.  The caller owns the
    semantic expected-vs-readback comparison and passes the descriptor of the
    still-open canonical HDF5 owner.
    """
    if type(durable_fsync) is not bool:
        raise TypeError("durable_fsync must be an exact bool")
    digest = _require_evidence_digest(evidence_digest)
    byte_count = int(evidence_bytes)
    if byte_count < 0:
        raise TransactionStateError("stream evidence byte count is negative")
    if expected_stat is not None:
        expected_stat = _comparable_identity(expected_stat)
    before = os.fstat(descriptor)
    if expected_stat is not None and _stat_identity(before) != expected_stat:
        raise TargetChanged(
            f"{role} descriptor changed after semantic verification: {path}"
        )
    if durable_fsync:
        _fsync_descriptor(descriptor, path)
    first = os.fstat(descriptor)
    try:
        named = os.stat(path)
    except FileNotFoundError as exc:
        raise TargetChanged(f"{role} pathname disappeared: {path}") from exc
    last = os.fstat(descriptor)
    identities = {
        _stat_identity(first),
        _stat_identity(named),
        _stat_identity(last),
    }
    if len(identities) != 1 or (
        expected_stat is not None and identities != {expected_stat}
    ):
        # Name the (dev, ino, size, mtime_ns, ctime_ns) views so a platform
        # disagreement (Windows: named stat vs the open descriptor) is
        # diagnosable from the refusal alone.
        raise TargetChanged(
            f"{role} descriptor/path identity changed during seal: {path} "
            f"(descriptor={_stat_identity(first)} named={_stat_identity(named)} "
            f"descriptor-after={_stat_identity(last)} expected={expected_stat})"
        )
    return _StreamStatReceipt(
        _normalize_target(path),
        _FileIdentity(int(last.st_dev), int(last.st_ino)),
        int(last.st_size),
        int(last.st_mtime_ns),
        int(last.st_ctime_ns),
        digest,
        byte_count,
        int(ordinal),
    )


def _stream_receipt_identity(
    receipt: _StreamStatReceipt,
) -> tuple[int, int, int, int, int]:
    """The sealed (dev, ino, size, mtime_ns, ctime_ns) view of a receipt,
    normalised the way ``_stat_identity`` observes it."""
    return (
        receipt.identity.device,
        receipt.identity.inode,
        receipt.size,
        receipt.mtime_ns,
        _identity_ctime_ns(receipt.ctime_ns),
    )


def _stream_stat_matches(path: Path | str, receipt: _StreamStatReceipt) -> bool:
    try:
        observed = os.stat(path)
    except FileNotFoundError:
        return False
    return _stat_identity(observed) == _stream_receipt_identity(receipt)


def _receipt_for_snapshot(path: Path | str, snapshot: TargetSnapshot, role: str) -> _ObjectReceipt:
    identity = _identity(snapshot)
    if identity is None:
        raise TransactionStateError(f"{role} receipt requires an existing object")
    return _ObjectReceipt(_normalize_target(path), identity, snapshot, role)


def _same_identity(snapshot: TargetSnapshot, receipt: _ObjectReceipt) -> bool:
    return _identity(snapshot) == receipt.identity


def _exact_receipt(snapshot: TargetSnapshot, receipt: _ObjectReceipt) -> bool:
    return _same_identity(snapshot, receipt) and snapshot == receipt.snapshot


class OutputTransactionCoordinator:
    """Shared lease registry and factory for unmounted transaction values."""

    def __init__(self):
        self._lock = threading.RLock()
        self._leases: dict[str, _LeaseState] = {}
        self._ordinal = 0

    def _next_ordinal(self) -> int:
        with self._lock:
            self._ordinal += 1
            return self._ordinal

    def _cleanup_token(self, target: str, domain: str) -> CleanupToken:
        return CleanupToken(target, self._next_ordinal(), domain)

    def admit(
        self,
        target,
        *,
        transaction_owner: OwnerToken,
        target_owner: OwnerToken,
        durable_fsync: bool = True,
        fast_regenerable: bool = False,
    ) -> "OutputTransaction":
        """Bind one exact transaction and target owner to a stable snapshot."""
        if not isinstance(transaction_owner, OwnerToken):
            raise TypeError("transaction_owner must be an OwnerToken")
        if not isinstance(target_owner, OwnerToken):
            raise TypeError("target_owner must be an OwnerToken")
        if type(durable_fsync) is not bool:
            raise TypeError("durable_fsync must be an exact bool")
        if type(fast_regenerable) is not bool:
            raise TypeError("fast_regenerable must be an exact bool")
        normalized = _normalize_target(target)
        admission = TargetAdmission(
            normalized,
            _capture_target(normalized, hash_content=not fast_regenerable),
            self._next_ordinal(),
        )
        return OutputTransaction(
            coordinator=self,
            admission=admission,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            durable_fsync=durable_fsync,
            fast_regenerable=fast_regenerable,
        )

    def hold_target(self, target, *, label: str) -> TargetHold:
        """Claim *target* exclusively for a caller that publishes it itself.

        Raises `LeaseUnavailable` when anything already holds the target,
        whether that is another hold or an ordinary transaction lease.
        """
        if type(label) is not str or not label:
            raise ValueError("a target hold requires a non-empty label")
        owners = {
            role: OwnerToken(f"{label}-{role.value}") for role in LeaseOwner
        }
        lease = self._acquire(_normalize_target(target), owners)
        return TargetHold(lease, owners)

    def release_target(self, hold: TargetHold) -> None:
        """Give a hold back.  Idempotent only in the sense that it is one-shot."""
        if type(hold) is not TargetHold:
            raise TypeError("release_target requires an exact TargetHold")
        for role in LeaseOwner:
            self._release(hold.lease, role, hold.owners[role])

    def prepare_xye(
        self,
        directory,
        *,
        run_owner: OwnerToken,
    ) -> "XyeOutputTransaction":
        """Create one run-owned, unmounted XYE publication transaction."""
        if not isinstance(run_owner, OwnerToken):
            raise TypeError("run_owner must be an OwnerToken")
        return XyeOutputTransaction(
            coordinator=self,
            directory=_normalize_target(directory),
            run_owner=run_owner,
        )

    def _acquire(
        self,
        target: str,
        owners: Mapping[LeaseOwner, OwnerToken],
    ) -> TargetLease:
        expected_roles = set(LeaseOwner)
        if set(owners) != expected_roles:
            raise ValueError(
                "a lease requires exact run/session/source/cleanup owners"
            )
        copied = dict(owners)
        if not all(isinstance(owner, OwnerToken) for owner in copied.values()):
            raise TypeError("every lease owner must be an OwnerToken")
        with self._lock:
            key = _lease_key(target)
            # TWO independent exclusions, because the identity key is computed
            # from MUTABLE filesystem state and can legitimately change between
            # two acquisitions of the same pathname (Codex F2 on `e06d6123`):
            # hold `Scan.nexus` before it exists, create it, then hold the
            # IDENTICAL string -- the second key folds to `scan.nexus` on a
            # case-insensitive volume and the old code admitted both.
            if key in self._leases:
                raise LeaseUnavailable(f"target already leased: {target}")
            if any(state.lease.target == target for state in self._leases.values()):
                raise LeaseUnavailable(f"target already leased: {target}")
            lease = TargetLease(target, self._next_ordinal())
            self._leases[key] = _LeaseState(lease=lease, owners=copied)
            return lease

    def _entry(self, lease: TargetLease) -> tuple[str, _LeaseState] | None:
        """Find a lease's registry entry BY IDENTITY, never by recomputation.

        The key cannot be rebuilt at release time: the file or its parent may
        have been created, removed or renamed since acquisition, and a key that
        no longer matches would strand the lease permanently.  A reverse map
        keyed by pathname was the first fix and was worse -- a second
        acquisition of the same string overwrote the first one's entry, so the
        original became unreleasable.  The registry is small (one entry per
        contended output), so an exact scan is both cheapest to reason about
        and impossible to corrupt.
        """
        for key, state in self._leases.items():
            if state.lease is lease:
                return key, state
        return None

    def _lease_snapshot(self, state: _LeaseState, *, active: bool) -> LeaseSnapshot:
        remaining = tuple(role for role in LeaseOwner if role in state.owners)
        return LeaseSnapshot(
            state.lease.target,
            state.lease.ordinal,
            active,
            remaining,
        )

    def _require_lease(self, lease: TargetLease) -> _LeaseState:
        with self._lock:
            found = self._entry(lease)
            if found is None:
                raise OwnershipRefused("foreign or retired target lease")
            return found[1]

    def _release(
        self,
        lease: TargetLease,
        role: LeaseOwner,
        owner: OwnerToken,
    ) -> LeaseSnapshot:
        with self._lock:
            found = self._entry(lease)
            if found is None:
                raise OwnershipRefused("foreign or retired target lease")
            key, state = found
            expected = state.owners.get(role)
            if expected is None:
                raise OwnershipRefused(f"lease owner {role.value} already released")
            _require_identity(owner, expected, f"{role.value} lease token")
            del state.owners[role]
            # Contract-bearing: the normalized lease remains live until every
            # independently registered cleanup owner has released.
            if state.owners:
                return self._lease_snapshot(state, active=True)
            del self._leases[key]
            return self._lease_snapshot(state, active=False)


# As with H10's public HDF5 pool, production composition has one eager,
# process-wide coordinator while tests may instantiate isolated registries.
_coordinator = OutputTransactionCoordinator()


def get_output_transaction_coordinator() -> OutputTransactionCoordinator:
    """Return the one process-wide output transaction/lease coordinator."""
    return _coordinator


class OutputTransaction:
    """One admitted target's lease, replacement, rollback, and retry owner."""

    def __init__(
        self,
        *,
        coordinator: OutputTransactionCoordinator,
        admission: TargetAdmission,
        transaction_owner: OwnerToken,
        target_owner: OwnerToken,
        durable_fsync: bool = True,
        fast_regenerable: bool = False,
    ):
        if type(durable_fsync) is not bool:
            raise TypeError("durable_fsync must be an exact bool")
        if type(fast_regenerable) is not bool:
            raise TypeError("fast_regenerable must be an exact bool")
        self._coordinator = coordinator
        self._admission = admission
        self._transaction_owner = transaction_owner
        self._target_owner = target_owner
        self._durable_fsync = durable_fsync
        self._fast_regenerable = fast_regenerable
        self._lease: TargetLease | None = None
        self._phase = TransactionPhase.ADMITTED
        self._pending: set[RetryAction] = set()
        self._cleanup_token: CleanupToken | None = None
        self._resume_pool = None
        self._writer_succeeded = False
        self._writer_returned = False
        self._published = False
        self._lock = threading.RLock()
        private = f"{admission.ordinal}-{secrets.token_hex(16)}"
        target = Path(admission.target)
        self.backup = target.with_name(
            f".{target.name}{_BACKUP_SUFFIX}-{private}"
        )
        self._candidate = target.with_name(
            f".{target.name}.xdart-candidate-{private}"
        )
        self._backup_placeholder: TargetSnapshot | None = None
        self._backup_owned = False
        self._captured_prior: TargetSnapshot | None = None
        self._candidate_snapshot: TargetSnapshot | None = None
        self._candidate_owned = False
        self._candidate_descriptor: int | None = None
        self._candidate_descriptor_identity: _FileIdentity | None = None
        self._candidate_reservation: _ObjectReceipt | None = None
        self._backup_reservation: _ObjectReceipt | None = None
        self._stage_receipt: _StageReceipt | None = None
        self._prior_receipt: _ObjectReceipt | None = None
        self._writer_receipt: _WriterReceipt | None = None
        self._publication_receipt: _LinkReceipt | None = None
        self._publication_resolution: _LinkResolutionReceipt | None = None
        self._publication_source_cleanup: _SourceCleanupReceipt | None = None
        self._rollback_receipt: _LinkReceipt | None = None
        self._rollback_resolution: _LinkResolutionReceipt | None = None
        self._rollback_source_cleanup: _SourceCleanupReceipt | None = None
        self._terminal_receipt: _TerminalReceipt | None = None
        self._stream_attempt: StreamAttempt | None = None
        self._stream_reservation: _ObjectReceipt | None = None
        self._stream_checkpoint: _StreamStatReceipt | None = None
        self._stream_checkpoint_token: StreamCheckpoint | None = None
        self._stream_checkpoint_fresh = False
        self._stream_durable_floor: StreamCheckpoint | None = None
        self._stream_close_owner: _StreamCloseAttempt | None = None
        self._stream_close_checkpoint: StreamCheckpoint | None = None
        self._stream_terminal_receipt: _ObjectReceipt | None = None
        self._stream_terminal_stat: _StreamStatReceipt | None = None
        self._stream_committed = False
        self._stream_epoch_receipt: _ObjectReceipt | None = None
        self._stream_file_lock = None
        self._stream_partial = target.with_name(
            f".{target.name}.xdart-partial-{private}"
        )
        self._stream_partial_receipt: _ObjectReceipt | None = None
        self._stream_partial_cleanup: _ObjectReceipt | None = None
        self._stream_preserved_partial: str | None = None
        self._stream_retain_partial = False

    def _capture(self, path: Path | str) -> TargetSnapshot:
        return _capture_target(
            str(path), hash_content=not self._fast_regenerable,
        )

    @property
    def admission(self) -> TargetAdmission:
        return self._admission

    @property
    def fast_regenerable(self) -> bool:
        """Whether this transaction admitted the finite regenerable route."""
        return self._fast_regenerable

    @property
    def stream_base_snapshot(self) -> TargetSnapshot:
        """Exact rollback base for the current owned stream epoch."""
        receipt = self._stream_epoch_receipt
        return self._admission.snapshot if receipt is None else receipt.snapshot

    def snapshot(self) -> TransactionSnapshot:
        with self._lock:
            pending = tuple(action for action in _RETRY_ORDER if action in self._pending)
            return TransactionSnapshot(
                target=self._admission.target,
                phase=self._phase,
                admission=self._admission.snapshot,
                pending_actions=pending,
                writer_succeeded=self._writer_succeeded,
                retryable=self._phase in {
                    TransactionPhase.READY_TO_RETRY,
                    TransactionPhase.ROLLBACK_PENDING,
                    TransactionPhase.CLEANUP_PENDING,
                },
                cleanup_token=self._cleanup_token,
                partial_path=(
                    str(self._stream_partial)
                    if self._stream_partial_receipt is not None
                    else self._stream_preserved_partial
                ),
                durable_floor=self._stream_durable_floor,
            )

    def _require_admission_owners(
        self,
        *,
        admission: TargetAdmission,
        transaction_owner: OwnerToken,
        target_owner: OwnerToken,
    ) -> None:
        _require_identity(admission, self._admission, "target admission")
        _require_identity(
            transaction_owner,
            self._transaction_owner,
            "transaction token",
        )
        _require_identity(target_owner, self._target_owner, "target token")

    def acquire_lease(
        self,
        *,
        admission: TargetAdmission,
        transaction_owner: OwnerToken,
        target_owner: OwnerToken,
        owners: Mapping[LeaseOwner, OwnerToken],
    ) -> TargetLease:
        with self._lock:
            self._require_admission_owners(
                admission=admission,
                transaction_owner=transaction_owner,
                target_owner=target_owner,
            )
            if self._lease is not None:
                raise TransactionStateError("transaction already acquired a lease")
            lease = self._coordinator._acquire(self._admission.target, owners)
            self._lease = lease
            self._phase = TransactionPhase.LEASED
            return lease

    def _require_lease(self, lease: TargetLease) -> None:
        if self._lease is None:
            raise TransactionStateError("transaction has no target lease")
        _require_identity(lease, self._lease, "target lease")
        self._coordinator._require_lease(lease)

    def release_lease_owner(
        self,
        lease: TargetLease,
        role: LeaseOwner,
        owner: OwnerToken,
    ) -> LeaseSnapshot:
        with self._lock:
            self._require_lease(lease)
            if role is LeaseOwner.CLEANUP and self._phase not in {
                TransactionPhase.COMMITTED,
                TransactionPhase.ABORTED,
            }:
                raise TransactionStateError(
                    "cleanup lease owner cannot release before a terminal phase"
                )
            return self._coordinator._release(lease, role, owner)

    def abandon(self, lease: TargetLease) -> TransactionSnapshot:
        with self._lock:
            self._require_lease(lease)
            untouched_integrity_hold = (
                self._phase is TransactionPhase.INTEGRITY_HOLD
                and not self._pending
                and not self._published
                and not self._writer_succeeded
                and not self._writer_returned
                and self._candidate_reservation is None
                and self._candidate_snapshot is None
                and self._backup_reservation is None
                and self._captured_prior is None
                and self._writer_receipt is None
                and self._publication_receipt is None
                and self._rollback_receipt is None
                and self._stream_attempt is None
                and self._stream_epoch_receipt is None
                and self._stream_durable_floor is None
            )
            if (
                self._phase not in {
                    TransactionPhase.LEASED,
                    TransactionPhase.READY_TO_RETRY,
                }
                and not untouched_integrity_hold
            ) or self._pending:
                raise TransactionStateError(
                    "only an untouched or exactly restored lease can be abandoned"
                )
            observed = self._capture(self._admission.target)
            if observed != self.stream_base_snapshot:
                self._phase = TransactionPhase.INTEGRITY_HOLD
                raise TargetChanged("preflight target changed before abandonment")
            self._terminal_receipt = _TerminalReceipt(
                self._admission.target,
                self.stream_base_snapshot,
                observed,
                "rollback-ready",
            )
            self._phase = TransactionPhase.ABORTED
            return self.snapshot()

    def _ensure_cleanup_token(self) -> CleanupToken:
        if self._cleanup_token is None:
            self._cleanup_token = self._coordinator._cleanup_token(
                self._admission.target,
                "output",
            )
        return self._cleanup_token

    def _validate_target(self) -> None:
        current = self._capture(self._admission.target)
        if current != self.stream_base_snapshot:
            raise TargetChanged(
                f"target changed after admission: {self._admission.target}"
            )

    def _install_terminal_receipt(
        self,
        *,
        expected: TargetSnapshot,
        role: str,
    ) -> _TerminalReceipt:
        observed = self._capture(self._admission.target)
        if observed != expected:
            self._terminal_receipt = None
            raise TargetChanged(
                f"{role} final target failed positive integrity verification"
            )
        receipt = _TerminalReceipt(
            self._admission.target,
            expected,
            observed,
            role,
        )
        self._terminal_receipt = receipt
        return receipt

    def _has_terminal_receipt(self, role: str) -> bool:
        receipt = self._terminal_receipt
        if receipt is None or receipt.role != role:
            return False
        return (
            receipt.target == self._admission.target
            and receipt.expected == receipt.observed
        )

    def _refresh_phase(self) -> None:
        if self._phase is TransactionPhase.ABORTED and not self._pending:
            return
        if (
            self._phase is TransactionPhase.INTEGRITY_HOLD
            and self._stream_durable_floor is not None
            and not self._published
            and not self._has_terminal_receipt("durable-floor-preserved")
        ):
            return
        if RetryAction.ROLLBACK in self._pending:
            self._phase = TransactionPhase.ROLLBACK_PENDING
        elif self._pending:
            self._phase = TransactionPhase.CLEANUP_PENDING
        elif self._published and self._has_terminal_receipt("committed-result"):
            self._phase = TransactionPhase.COMMITTED
        elif (
            not self._published
            and self._has_terminal_receipt("durable-floor-preserved")
        ):
            self._phase = TransactionPhase.ABORTED
        elif not self._published and self._has_terminal_receipt("rollback-ready"):
            self._phase = TransactionPhase.READY_TO_RETRY
        else:
            self._phase = TransactionPhase.INTEGRITY_HOLD

    @staticmethod
    def _require_owned_snapshot(
        path: Path,
        expected: TargetSnapshot,
        label: str,
    ) -> TargetSnapshot:
        current = _capture_target(str(path))
        if current != expected:
            raise TargetChanged(
                f"{label} changed outside its transaction owner: {path}"
            )
        return current

    def _resolve_candidate_reservation(self) -> None:
        if RetryAction.CANDIDATE_RESERVATION not in self._pending:
            return
        current = _capture_target(str(self._candidate))
        receipt = self._candidate_reservation
        if receipt is None:
            if current.exists:
                raise TransactionStateError(
                    f"unowned candidate occupies {self._candidate}; refusing cleanup"
                )
            self._pending.discard(RetryAction.CANDIDATE_RESERVATION)
            return
        if not current.exists:
            self._candidate_snapshot = None
            self._candidate_owned = False
            self._candidate_reservation = None
            self._pending.discard(RetryAction.CANDIDATE_RESERVATION)
            return
        if not _same_identity(current, receipt):
            self._candidate_owned = False
            raise TargetChanged(
                f"candidate reservation was replaced at {self._candidate}"
            )
        if not _exact_receipt(current, receipt):
            raise TargetChanged(
                f"candidate reservation integrity changed before writer at {self._candidate}"
            )
        self._candidate_snapshot = receipt.snapshot
        self._candidate_owned = True
        self._pending.discard(RetryAction.CANDIDATE_RESERVATION)

    def _reserve_candidate(self) -> None:
        self._ensure_cleanup_token()
        self._pending.add(RetryAction.ROLLBACK)
        self._pending.add(RetryAction.CANDIDATE_RESERVATION)
        if self._candidate_descriptor is not None:
            raise TransactionStateError("candidate descriptor has not been retired")
        if self._candidate.exists():
            self._pending.discard(RetryAction.CANDIDATE_RESERVATION)
            raise TransactionStateError(
                f"unowned candidate occupies {self._candidate}; refusing write"
            )
        try:
            descriptor = os.open(
                self._candidate,
                os.O_CREAT | os.O_EXCL | os.O_RDWR,
                0o600,
            )
        except FileExistsError as exc:
            self._pending.discard(RetryAction.CANDIDATE_RESERVATION)
            raise TransactionStateError(
                f"unowned candidate occupies {self._candidate}; refusing write"
            ) from exc
        # Install ownership before receipt capture can fail.  This descriptor
        # pins the original inode across execute and every cleanup retry.
        self._candidate_descriptor = descriptor
        receipt = _descriptor_receipt(
            descriptor,
            self._candidate,
            "candidate-reservation",
        )
        self._candidate_descriptor_identity = receipt.identity
        self._candidate_reservation = receipt
        self._candidate_owned = True
        self._candidate_snapshot = receipt.snapshot
        self._writer_receipt = _WriterReceipt(receipt, False, None)

    def _attempt_candidate_retire(self) -> BaseException | None:
        """Release the inode pin only after every candidate identity owner."""
        descriptor = self._candidate_descriptor
        if descriptor is None:
            return None
        if any(owner is not None for owner in (
            self._candidate_reservation,
            self._writer_receipt,
            self._publication_receipt,
        )):
            return None
        self._ensure_cleanup_token()
        self._pending.add(RetryAction.CANDIDATE_RETIRE)
        try:
            observed = os.fstat(descriptor)
        except OSError as exc:
            if exc.errno != errno.EBADF:
                return exc
            # A close may complete and then report an error.  Never retry
            # close against a descriptor number already proven closed.
        except BaseException as exc:
            return exc
        else:
            identity = _FileIdentity(int(observed.st_dev), int(observed.st_ino))
            expected = self._candidate_descriptor_identity
            if expected is not None and identity != expected:
                return OwnershipRefused("candidate descriptor now names a foreign object")
            self._candidate_descriptor_identity = identity
            try:
                os.close(descriptor)
            except BaseException as exc:
                return exc
        self._candidate_descriptor = None
        self._candidate_descriptor_identity = None
        self._pending.discard(RetryAction.CANDIDATE_RETIRE)
        return None

    def _resolve_backup_reservation(self) -> None:
        if RetryAction.BACKUP_RESERVATION not in self._pending:
            return
        current = self._capture(self.backup)
        receipt = self._backup_reservation
        if receipt is None:
            if current.exists:
                raise TransactionStateError(
                    f"unowned backup occupies {self.backup}; refusing cleanup"
                )
            self._pending.discard(RetryAction.BACKUP_RESERVATION)
            return
        if not current.exists:
            self._backup_placeholder = None
            self._backup_owned = False
            self._backup_reservation = None
            self._pending.discard(RetryAction.BACKUP_RESERVATION)
            return
        if not _same_identity(current, receipt):
            self._backup_owned = False
            raise TargetChanged(
                f"backup reservation was replaced at {self.backup}"
            )
        if not _exact_receipt(current, receipt):
            raise TargetChanged(
                f"backup reservation integrity changed before staging at {self.backup}"
            )
        self._backup_placeholder = receipt.snapshot
        self._backup_owned = True
        self._pending.discard(RetryAction.BACKUP_RESERVATION)

    def _reserve_backup(self) -> None:
        self._ensure_cleanup_token()
        self._pending.add(RetryAction.ROLLBACK)
        self._pending.add(RetryAction.BACKUP_RESERVATION)
        if self.backup.exists():
            self._pending.discard(RetryAction.BACKUP_RESERVATION)
            raise TransactionStateError(
                f"unowned backup occupies {self.backup}; refusing overwrite"
            )
        try:
            descriptor = os.open(
                self.backup,
                os.O_CREAT | os.O_EXCL | os.O_RDWR,
                0o600,
            )
        except FileExistsError as exc:
            self._pending.discard(RetryAction.BACKUP_RESERVATION)
            raise TransactionStateError(
                f"unowned backup occupies {self.backup}; refusing overwrite"
            ) from exc
        try:
            receipt = _descriptor_receipt(
                descriptor,
                self.backup,
                "backup-placeholder",
            )
        except BaseException:
            os.close(descriptor)
            raise
        self._backup_reservation = receipt
        self._backup_owned = True
        self._backup_placeholder = receipt.snapshot
        os.close(descriptor)
        self._resolve_backup_reservation()
        if self._backup_placeholder is None:
            raise TargetChanged("owned backup disappeared during reservation")

    def _resolve_staged_prior(self) -> bool:
        if RetryAction.STAGE_PRIOR not in self._pending:
            return self._captured_prior is not None
        receipt = self._stage_receipt
        if receipt is None:
            raise TransactionStateError("staging transition has no immutable receipt")
        backup = self._capture(self.backup)
        target = self._capture(self._admission.target)
        admitted = receipt.admitted_prior
        placeholder = receipt.placeholder
        target_exact = _exact_receipt(target, admitted)
        backup_exact_prior = _exact_receipt(backup, admitted)
        backup_exact_placeholder = _exact_receipt(backup, placeholder)

        if target_exact and not backup.exists:
            self._backup_owned = False
            self._backup_placeholder = None
            self._backup_reservation = None
            self._stage_receipt = None
            self._pending.discard(RetryAction.STAGE_PRIOR)
            return False

        if target_exact and backup_exact_placeholder:
            _unlink(self.backup)
            self._backup_owned = False
            self._backup_placeholder = None
            self._backup_reservation = None
            self._stage_receipt = None
            self._pending.discard(RetryAction.STAGE_PRIOR)
            return False

        if not target.exists and backup_exact_prior:
            self._captured_prior = admitted.snapshot
            self._prior_receipt = _ObjectReceipt(
                _normalize_target(self.backup),
                admitted.identity,
                admitted.snapshot,
                "staged-prior",
            )
            self._backup_placeholder = None
            self._backup_reservation = None
            self._backup_owned = True
            self._pending.discard(RetryAction.STAGE_PRIOR)
            return True

        if target_exact and backup_exact_prior:
            _unlink(self.backup)
            self._captured_prior = None
            self._prior_receipt = None
            self._backup_placeholder = None
            self._backup_reservation = None
            self._backup_owned = False
            self._stage_receipt = None
            self._pending.discard(RetryAction.STAGE_PRIOR)
            return False

        if target.exists:
            raise TargetChanged(
                "staging transition found a foreign final target; receipt retained"
            )
        if _same_identity(backup, admitted):
            raise TargetChanged(
                "staged prior has admitted identity but failed immutable integrity"
            )
        if backup.exists:
            raise TargetChanged(
                "staged prior pathname has foreign identity; receipt retained"
            )
        raise TargetChanged(
            "staging transition lost both its exact private backup and prior"
        )

    def _stage_prior(self) -> None:
        attempts = max(1, int(_REPLACE_RETRIES))
        for attempt in range(1, attempts + 1):
            self._reserve_backup()
            if self._backup_reservation is None:
                raise TransactionStateError(
                    "backup reservation lacks descriptor authority"
                )
            admitted = _receipt_for_snapshot(
                self._admission.target,
                self.stream_base_snapshot,
                "admitted-prior",
            )
            self._stage_receipt = _StageReceipt(
                self._admission.target,
                _normalize_target(self.backup),
                admitted,
                self._backup_reservation,
            )
            self._pending.add(RetryAction.STAGE_PRIOR)
            replace_error: BaseException | None = None
            try:
                _replace(Path(self._admission.target), self.backup)
            except BaseException as exc:
                replace_error = exc
            try:
                captured = self._resolve_staged_prior()
            except BaseException as observation_error:
                if replace_error is not None:
                    raise replace_error.with_traceback(
                        replace_error.__traceback__
                    ) from observation_error
                raise

            # The immutable receipt, not the system-call return, decides
            # whether the prior was staged.  A network filesystem may report
            # an error after the exact move became observable.
            if captured:
                if self._captured_prior is None:
                    raise TransactionStateError(
                        "captured target lacks immutable prior authority"
                    )
                if self._captured_prior != self.stream_base_snapshot:
                    raise TargetChanged(
                        "captured target differs from the exact admitted target"
                    )
                return

            if replace_error is None:
                raise TargetChanged(
                    "target was not captured by the staging transition"
                )
            if not isinstance(replace_error, PermissionError):
                raise replace_error.with_traceback(replace_error.__traceback__)
            if attempt == attempts:
                raise PermissionError(
                    _held_open_message(self._admission.target, "stage")
                ) from replace_error
            time.sleep(max(0.0, float(_REPLACE_RETRY_DELAY_S)))

    def _resolve_writer_capture(self) -> None:
        if RetryAction.WRITER_CAPTURE not in self._pending:
            return
        receipt = self._writer_receipt
        if receipt is None or self._candidate_reservation is None:
            raise TransactionStateError("writer candidate has no private owner")
        current = _capture_target(str(self._candidate))
        if not current.exists:
            self._candidate_snapshot = None
            self._candidate_owned = False
            raise OSError("writer did not preserve its owned candidate")
        if not _same_identity(current, receipt.reservation):
            self._candidate_owned = False
            raise TargetChanged(
                "writer replaced the reserved candidate namespace identity"
            )
        result = _receipt_for_snapshot(
            self._candidate,
            current,
            "writer-result",
        )
        self._candidate_snapshot = result.snapshot
        self._candidate_owned = True
        self._writer_receipt = _WriterReceipt(
            receipt.reservation,
            self._writer_returned,
            result,
        )
        self._writer_succeeded = self._writer_returned
        self._pending.discard(RetryAction.WRITER_CAPTURE)

    @staticmethod
    def _link_resolution_is_exact(
        resolution: _LinkResolutionReceipt | None,
        receipt: _LinkReceipt,
    ) -> bool:
        return bool(
            resolution is not None
            and resolution.link is receipt
            and resolution.role == receipt.role
            and _exact_receipt(resolution.source_observed, receipt.source)
            and _exact_receipt(resolution.destination_observed, receipt.source)
        )

    @staticmethod
    def _source_cleanup_is_authorized(
        cleanup: _SourceCleanupReceipt | None,
        resolution: _LinkResolutionReceipt,
        *,
        source: Path,
        role: str,
    ) -> bool:
        return bool(
            cleanup is not None
            and cleanup.resolution is resolution
            and cleanup.source == _normalize_target(source)
            and cleanup.role == role
            and _exact_receipt(
                cleanup.observed,
                resolution.link.source,
            )
        )

    @staticmethod
    def _new_link_resolution(
        receipt: _LinkReceipt,
        source: TargetSnapshot,
        destination: TargetSnapshot,
    ) -> _LinkResolutionReceipt:
        if not (
            _exact_receipt(source, receipt.source)
            and _exact_receipt(destination, receipt.source)
        ):
            raise TargetChanged(
                f"{receipt.role} requires exact source and destination observations"
            )
        return _LinkResolutionReceipt(
            receipt,
            source,
            destination,
            receipt.role,
        )

    @classmethod
    def _new_source_cleanup(
        cls,
        resolution: _LinkResolutionReceipt,
        *,
        source: Path,
        observed: TargetSnapshot,
        role: str,
    ) -> _SourceCleanupReceipt:
        if not cls._link_resolution_is_exact(resolution, resolution.link):
            raise TransactionStateError(
                f"{role} source cleanup lacks exact link resolution"
            )
        if not _exact_receipt(observed, resolution.link.source):
            raise TargetChanged(
                f"{role} source changed before authorized cleanup"
            )
        return _SourceCleanupReceipt(
            resolution,
            _normalize_target(source),
            observed,
            role,
        )

    def _clear_publication_authority(self) -> None:
        self._publication_receipt = None
        self._publication_resolution = None
        self._publication_source_cleanup = None

    def _clear_rollback_authority(self) -> None:
        self._rollback_receipt = None
        self._rollback_resolution = None
        self._rollback_source_cleanup = None

    def _resolve_publication(self) -> str:
        if RetryAction.PUBLICATION not in self._pending:
            return "published" if self._published else "resolved"
        receipt = self._publication_receipt
        if receipt is None:
            raise TransactionStateError("publication has no immutable link receipt")
        source = _capture_target(str(self._candidate))
        current = _capture_target(self._admission.target)
        source_exact = _exact_receipt(source, receipt.source)
        current_exact = _exact_receipt(current, receipt.source)
        if current_exact:
            resolution = self._publication_resolution
            if resolution is None:
                if not source_exact:
                    raise TargetChanged(
                        "publication final is exact but its private source is "
                        "not exact; link resolution remains pending"
                    )
                resolution = self._new_link_resolution(receipt, source, current)
                self._publication_resolution = resolution
            elif not self._link_resolution_is_exact(resolution, receipt):
                raise TransactionStateError(
                    "publication has an invalid link-resolution receipt"
                )
            elif not source_exact:
                cleanup = self._publication_source_cleanup
                if not (
                    not source.exists
                    and self._source_cleanup_is_authorized(
                        cleanup,
                        resolution,
                        source=self._candidate,
                        role="publication-source-cleanup",
                    )
                ):
                    raise TargetChanged(
                        "publication source changed after exact link resolution"
                    )
            self._published = True
            self._pending.discard(RetryAction.PUBLICATION)
            self._pending.discard(RetryAction.ROLLBACK)
            return "published"
        if not current.exists:
            self._clear_publication_authority()
            self._pending.discard(RetryAction.PUBLICATION)
            return "absent"
        if self._captured_prior is not None and current == self._captured_prior:
            self._clear_publication_authority()
            self._pending.discard(RetryAction.PUBLICATION)
            return "prior"
        if current == self._admission.snapshot:
            self._clear_publication_authority()
            self._pending.discard(RetryAction.PUBLICATION)
            return "prior"
        if _same_identity(current, receipt.source):
            _unlink(Path(self._admission.target))
            self._clear_publication_authority()
            self._pending.discard(RetryAction.PUBLICATION)
            return "owned-corrupt"
        self._clear_publication_authority()
        self._pending.discard(RetryAction.PUBLICATION)
        return "foreign"

    def _cleanup_candidate_for_rollback(self) -> None:
        current = _capture_target(str(self._candidate))
        if not current.exists:
            self._candidate_snapshot = None
            self._candidate_owned = False
            self._candidate_reservation = None
            self._writer_receipt = None
            self._pending.discard(RetryAction.CANDIDATE_RESERVATION)
            self._pending.discard(RetryAction.WRITER_CAPTURE)
            return
        receipt = self._candidate_reservation
        if receipt is None or not _same_identity(current, receipt):
            self._candidate_owned = False
            raise TargetChanged(
                f"foreign candidate occupies {self._candidate} during rollback"
            )
        _unlink(self._candidate)
        self._candidate_snapshot = None
        self._candidate_owned = False
        self._candidate_reservation = None
        self._writer_receipt = None
        self._pending.discard(RetryAction.CANDIDATE_RESERVATION)
        self._pending.discard(RetryAction.WRITER_CAPTURE)

    def _resolve_rollback_link(self) -> None:
        receipt = self._rollback_receipt
        if receipt is None:
            raise TransactionStateError("rollback link has no immutable receipt")
        source = self._capture(self.backup)
        destination = self._capture(self._admission.target)
        source_exact = _exact_receipt(source, receipt.source)
        destination_exact = _exact_receipt(destination, receipt.source)
        if destination_exact:
            resolution = self._rollback_resolution
            if resolution is None:
                if not source_exact:
                    raise TargetChanged(
                        "rollback final is exact but its private source is not "
                        "exact; link resolution remains pending"
                    )
                self._rollback_resolution = self._new_link_resolution(
                    receipt,
                    source,
                    destination,
                )
                return
            if not self._link_resolution_is_exact(resolution, receipt):
                raise TransactionStateError(
                    "rollback has an invalid link-resolution receipt"
                )
            if source_exact:
                return
            if (
                not source.exists
                and self._source_cleanup_is_authorized(
                    self._rollback_source_cleanup,
                    resolution,
                    source=self.backup,
                    role="rollback-source-cleanup",
                )
            ):
                return
            raise TargetChanged(
                "rollback source changed after exact link resolution"
            )
        if not destination.exists:
            raise TargetChanged("rollback link did not persist at the final target")
        if _same_identity(destination, receipt.source):
            if not source.exists:
                raise TargetChanged(
                    "rollback final is the sole surviving admitted-prior identity"
                )
            _unlink(Path(self._admission.target))
            raise TargetChanged("rollback linked an integrity-invalid prior")
        raise TargetChanged(
            f"rollback refused foreign occupant at {self._admission.target}"
        )

    def _restore_prior(self) -> None:
        receipt = self._prior_receipt
        if receipt is None:
            raise TransactionStateError("captured prior lacks immutable authority")
        source = self._capture(self.backup)
        target = self._capture(self._admission.target)
        if target.exists:
            if self._rollback_receipt is not None:
                self._resolve_rollback_link()
                return
            if _exact_receipt(target, receipt):
                return
            raise TargetChanged(
                f"rollback refused foreign occupant at {self._admission.target}"
            )
        if not _exact_receipt(source, receipt):
            if _same_identity(source, receipt):
                raise TargetChanged("staged prior failed immutable integrity")
            raise TargetChanged("staged prior identity is foreign")
        self._rollback_receipt = _LinkReceipt(
            receipt,
            self._admission.target,
            "rollback",
        )
        self._rollback_resolution = None
        self._rollback_source_cleanup = None
        link_error: BaseException | None = None
        try:
            _link(self.backup, Path(self._admission.target))
        except BaseException as exc:
            link_error = exc
        try:
            self._resolve_rollback_link()
        except BaseException as observation_error:
            if link_error is not None:
                raise link_error.with_traceback(
                    link_error.__traceback__
                ) from observation_error
            raise
        if link_error is not None:
            return

    def _settle_backup_after_restore(self) -> BaseException | None:
        current = self._capture(self.backup)
        if not current.exists:
            rollback = self._rollback_receipt
            if rollback is not None:
                resolution = self._rollback_resolution
                if (
                    resolution is None
                    or not self._link_resolution_is_exact(resolution, rollback)
                    or not self._source_cleanup_is_authorized(
                        self._rollback_source_cleanup,
                        resolution,
                        source=self.backup,
                        role="rollback-source-cleanup",
                    )
                ):
                    self._pending.add(RetryAction.BACKUP_UNLINK)
                    return TargetChanged(
                        "rollback private source disappeared without authorized cleanup"
                    )
            self._pending.discard(RetryAction.BACKUP_UNLINK)
            self._backup_placeholder = None
            self._backup_reservation = None
            self._backup_owned = False
            return None
        receipt = self._prior_receipt
        if receipt is None or not _exact_receipt(current, receipt):
            self._pending.add(RetryAction.BACKUP_UNLINK)
            return TargetChanged(
                f"restored prior left a non-exact private backup at {self.backup}"
            )
        return self._attempt_backup_unlink(retain_authority=True)

    def _rollback_once(self) -> None:
        self._ensure_cleanup_token()
        self._pending.add(RetryAction.ROLLBACK)
        target = Path(self._admission.target)
        private_failures: list[BaseException] = []

        if RetryAction.BACKUP_RESERVATION in self._pending:
            try:
                self._resolve_backup_reservation()
            except BaseException as exc:
                private_failures.append(exc)

        stage_failure: BaseException | None = None
        if RetryAction.STAGE_PRIOR in self._pending:
            try:
                self._resolve_staged_prior()
            except BaseException as exc:
                stage_failure = exc

        if RetryAction.PUBLICATION in self._pending:
            self._resolve_publication()
        if self._published:
            self._pending.discard(RetryAction.ROLLBACK)
            return

        if self.stream_base_snapshot.exists:
            if self._prior_receipt is not None:
                self._restore_prior()
            elif RetryAction.STAGE_PRIOR in self._pending:
                if stage_failure is not None:
                    raise stage_failure
                raise TargetChanged("staged prior remains unresolved")
            else:
                current = self._capture(target)
                if current != self.stream_base_snapshot:
                    raise TargetChanged(f"rollback refused foreign occupant at {target}")
        else:
            current = self._capture(target)
            if current.exists:
                raise TargetChanged(f"rollback refused foreign occupant at {target}")

        backup_failure: BaseException | None = None
        if self._prior_receipt is not None:
            backup_failure = self._settle_backup_after_restore()
        elif self._backup_reservation is not None:
            current_backup = self._capture(self.backup)
            if current_backup.exists and _exact_receipt(
                current_backup,
                self._backup_reservation,
            ):
                failure = self._attempt_backup_unlink()
                if failure is not None:
                    backup_failure = failure
            elif current_backup.exists:
                if RetryAction.BACKUP_RESERVATION not in self._pending:
                    self._pending.add(RetryAction.BACKUP_RESERVATION)
                private_failures.append(
                    TargetChanged(
                        f"foreign backup reservation retained at {self.backup}"
                    )
                )
            else:
                self._backup_reservation = None
                self._backup_placeholder = None
                self._backup_owned = False
                self._pending.discard(RetryAction.BACKUP_RESERVATION)

        try:
            self._cleanup_candidate_for_rollback()
        except BaseException as exc:
            if not (
                RetryAction.CANDIDATE_RESERVATION in self._pending
                or RetryAction.WRITER_CAPTURE in self._pending
            ):
                self._pending.add(RetryAction.CANDIDATE_UNLINK)
            private_failures.append(exc)

        if backup_failure is not None:
            raise backup_failure
        if private_failures:
            self._install_terminal_receipt(
                expected=self.stream_base_snapshot,
                role="rollback-ready",
            )
            self._pending.discard(RetryAction.ROLLBACK)
            self._pending.discard(RetryAction.STAGE_PRIOR)
            self._stage_receipt = None
            self._captured_prior = None
            self._prior_receipt = None
            self._clear_rollback_authority()
            self._writer_returned = False
            self._writer_succeeded = False
            self._refresh_phase()
            raise private_failures[0]

        self._install_terminal_receipt(
            expected=self.stream_base_snapshot,
            role="rollback-ready",
        )
        self._writer_returned = False
        self._writer_succeeded = False
        self._pending.discard(RetryAction.ROLLBACK)
        self._pending.discard(RetryAction.STAGE_PRIOR)
        self._stage_receipt = None
        self._captured_prior = None
        self._prior_receipt = None
        self._clear_rollback_authority()
        self._refresh_phase()

    def _attempt_rollback(self) -> BaseException | None:
        try:
            self._rollback_once()
        except BaseException as exc:
            if RetryAction.ROLLBACK in self._pending:
                self._phase = TransactionPhase.ROLLBACK_PENDING
            else:
                self._refresh_phase()
            return exc
        return None

    def _attempt_backup_unlink(
        self,
        *,
        retain_authority: bool = False,
    ) -> BaseException | None:
        self._ensure_cleanup_token()
        self._pending.add(RetryAction.BACKUP_UNLINK)
        try:
            current = self._capture(self.backup)
        except BaseException as exc:
            return exc

        if not current.exists:
            rollback = self._rollback_receipt
            if retain_authority and rollback is not None:
                resolution = self._rollback_resolution
                if (
                    resolution is None
                    or not self._link_resolution_is_exact(resolution, rollback)
                    or not self._source_cleanup_is_authorized(
                        self._rollback_source_cleanup,
                        resolution,
                        source=self.backup,
                        role="rollback-source-cleanup",
                    )
                ):
                    return TargetChanged(
                        "rollback backup disappeared before authorized cleanup"
                    )
        else:
            receipt = self._prior_receipt or self._backup_reservation
            if receipt is None or not _exact_receipt(current, receipt):
                return TargetChanged(
                    f"backup cleanup refused non-exact object at {self.backup}"
                )
            rollback = self._rollback_receipt
            if retain_authority and rollback is not None:
                resolution = self._rollback_resolution
                if (
                    resolution is None
                    or not self._link_resolution_is_exact(resolution, rollback)
                ):
                    return TransactionStateError(
                        "rollback backup cleanup lacks exact link resolution"
                    )
                try:
                    self._rollback_source_cleanup = self._new_source_cleanup(
                        resolution,
                        source=self.backup,
                        observed=current,
                        role="rollback-source-cleanup",
                    )
                except BaseException as exc:
                    return exc
            try:
                _unlink(self.backup)
            except FileNotFoundError:
                # The exact cleanup receipt was installed before this race.
                # Absence is therefore compatible with the authorized unlink.
                pass
            except BaseException as exc:
                # Contract-bearing: a failed unlink keeps both the prior-result
                # backup and the exact retry phase.
                return exc

        self._pending.discard(RetryAction.BACKUP_UNLINK)
        self._backup_placeholder = None
        self._backup_reservation = None
        self._backup_owned = False
        if not retain_authority:
            self._captured_prior = None
            self._prior_receipt = None
            self._stage_receipt = None
            self._clear_rollback_authority()
        return None

    def _attempt_candidate_unlink(self) -> BaseException | None:
        self._ensure_cleanup_token()
        self._pending.add(RetryAction.CANDIDATE_UNLINK)
        try:
            current = _capture_target(str(self._candidate))
        except BaseException as exc:
            return exc

        publication = self._publication_receipt if self._published else None
        if not current.exists:
            if publication is not None:
                resolution = self._publication_resolution
                if (
                    resolution is None
                    or not self._link_resolution_is_exact(
                        resolution,
                        publication,
                    )
                    or not self._source_cleanup_is_authorized(
                        self._publication_source_cleanup,
                        resolution,
                        source=self._candidate,
                        role="publication-source-cleanup",
                    )
                ):
                    return TargetChanged(
                        "published candidate disappeared before authorized cleanup"
                    )
        else:
            receipt = self._candidate_reservation
            if receipt is None or not _same_identity(current, receipt):
                return TargetChanged(
                    f"candidate cleanup refused foreign object at {self._candidate}"
                )
            if publication is not None:
                resolution = self._publication_resolution
                if (
                    resolution is None
                    or not self._link_resolution_is_exact(
                        resolution,
                        publication,
                    )
                ):
                    return TransactionStateError(
                        "publication candidate cleanup lacks exact link resolution"
                    )
                try:
                    self._publication_source_cleanup = self._new_source_cleanup(
                        resolution,
                        source=self._candidate,
                        observed=current,
                        role="publication-source-cleanup",
                    )
                except BaseException as exc:
                    return exc
            try:
                _unlink(self._candidate)
            except FileNotFoundError:
                # An exact cleanup receipt, when publication is live, was
                # installed before this namespace race.
                pass
            except BaseException as exc:
                return exc

        self._pending.discard(RetryAction.CANDIDATE_UNLINK)
        self._candidate_snapshot = None
        self._candidate_owned = False
        self._candidate_reservation = None
        self._writer_receipt = None
        return None

    def _publish_candidate(self) -> None:
        writer_receipt = self._writer_receipt
        if writer_receipt is None or writer_receipt.result is None:
            raise TransactionStateError("writer candidate has no stable fingerprint")
        current = _capture_target(str(self._candidate))
        if not _exact_receipt(current, writer_receipt.result):
            raise TargetChanged("writer result changed before publication")
        self._publication_receipt = _LinkReceipt(
            writer_receipt.result,
            self._admission.target,
            "publication",
        )
        self._publication_resolution = None
        self._publication_source_cleanup = None
        self._pending.add(RetryAction.PUBLICATION)
        link_error: BaseException | None = None
        try:
            _link(self._candidate, Path(self._admission.target))
        except BaseException as exc:
            link_error = exc
        try:
            outcome = self._resolve_publication()
        except BaseException as observation_error:
            if link_error is not None:
                raise link_error.with_traceback(
                    link_error.__traceback__
                ) from observation_error
            raise
        if outcome == "published":
            return
        if outcome == "absent" and link_error is not None:
            raise link_error.with_traceback(link_error.__traceback__)
        if outcome == "absent":
            raise TargetChanged("publication target disappeared after no-clobber link")
        if outcome == "owned-corrupt":
            raise TargetChanged(
                "publication linked the owned candidate with invalid integrity"
            )
        raise TargetChanged(
            f"publication refused {outcome} occupant at {self._admission.target}"
        ) from link_error

    def _own_pool_resume(self, pool) -> None:
        self._ensure_cleanup_token()
        self._resume_pool = pool
        self._pending.add(RetryAction.POOL_RESUME)

    def _attempt_pool_resume(self, pool) -> BaseException | None:
        self._own_pool_resume(pool)
        try:
            pool.resume(self._admission.target)
        except BaseException as exc:
            # Contract-bearing: never clear the retained pool owner in a
            # finally block when resume itself did not complete.
            return exc
        self._pending.discard(RetryAction.POOL_RESUME)
        self._resume_pool = None
        return None

    def _verify_published_terminal(self) -> BaseException | None:
        receipt = self._publication_receipt
        if receipt is None:
            return TransactionStateError(
                "published result lacks its immutable link receipt"
            )
        resolution = self._publication_resolution
        if (
            resolution is None
            or not self._link_resolution_is_exact(resolution, receipt)
        ):
            self._terminal_receipt = None
            self._pending.add(RetryAction.PUBLICATION)
            return TransactionStateError(
                "published result lacks exact source/destination resolution"
            )
        try:
            source = _capture_target(str(self._candidate))
        except BaseException as exc:
            self._terminal_receipt = None
            self._pending.add(RetryAction.PUBLICATION)
            return exc
        if source.exists:
            self._terminal_receipt = None
            self._pending.add(RetryAction.CANDIDATE_UNLINK)
            return TargetChanged(
                "published result still has an uncleaned private source"
            )
        if not self._source_cleanup_is_authorized(
            self._publication_source_cleanup,
            resolution,
            source=self._candidate,
            role="publication-source-cleanup",
        ):
            self._terminal_receipt = None
            self._pending.add(RetryAction.PUBLICATION)
            return TargetChanged(
                "published private source disappeared without authorized cleanup"
            )
        try:
            current = _capture_target(self._admission.target)
        except BaseException as exc:
            self._terminal_receipt = None
            self._pending.add(RetryAction.PUBLICATION)
            return exc
        if _exact_receipt(current, receipt.source):
            try:
                self._install_terminal_receipt(
                    expected=receipt.source.snapshot,
                    role="committed-result",
                )
            except BaseException as exc:
                self._pending.add(RetryAction.PUBLICATION)
                return exc
            self._pending.discard(RetryAction.PUBLICATION)
            self._clear_publication_authority()
            return None

        self._terminal_receipt = None
        self._published = False
        self._pending.add(RetryAction.ROLLBACK)
        if _same_identity(current, receipt.source):
            self._pending.add(RetryAction.PUBLICATION)
            try:
                _unlink(Path(self._admission.target))
            except BaseException as exc:
                # The exact link receipt remains live so cleanup retry can
                # identify and remove only this owned invalid final link.
                return exc
            self._pending.discard(RetryAction.PUBLICATION)
            self._clear_publication_authority()
        else:
            # Absent or foreign final state is never inferred as publication.
            # A foreign namespace object is preserved for rollback resolution.
            self._pending.discard(RetryAction.PUBLICATION)
            self._clear_publication_authority()

        rollback_failure = self._attempt_rollback()
        if rollback_failure is not None:
            return rollback_failure
        return TargetChanged(
            "published result failed post-candidate-cleanup integrity verification"
        )

    def _finish_published_cleanup(self) -> list[BaseException]:
        """Verify the final result before discarding its admitted-prior backup."""
        failures: list[BaseException] = []
        if (
            self._candidate_owned
            or self._candidate_snapshot is not None
            or RetryAction.CANDIDATE_UNLINK in self._pending
        ):
            failure = self._attempt_candidate_unlink()
            if failure is not None:
                failures.append(failure)
                return failures

        if not self._has_terminal_receipt("committed-result"):
            failure = self._verify_published_terminal()
            if failure is not None:
                failures.append(failure)
                return failures

        if (
            self._backup_owned
            or self._captured_prior is not None
            or self._backup_placeholder is not None
        ):
            failure = self._attempt_backup_unlink()
            if failure is not None:
                failures.append(failure)
        return failures

    def _target_transition_pending(self) -> bool:
        if (
            self._phase is TransactionPhase.INTEGRITY_HOLD
            and self._stream_durable_floor is not None
            and not self._published
            and not self._has_terminal_receipt("durable-floor-preserved")
        ):
            return True
        if (
            RetryAction.CANDIDATE_UNLINK in self._pending
            and self._published
            and not self._has_terminal_receipt("committed-result")
        ):
            return True
        return bool(
            self._pending
            & {
                RetryAction.STAGE_PRIOR,
                RetryAction.PUBLICATION,
                RetryAction.ROLLBACK,
                RetryAction.STREAM_RETIRE,
            }
        )

    def _copy_stream_seed(
        self,
        *,
        seed_mode: StreamSeedMode = StreamSeedMode.PRESERVE_BASE,
    ) -> tuple[_ObjectReceipt, _StreamStatReceipt]:
        if type(seed_mode) is not StreamSeedMode:
            raise TypeError("seed_mode must be a StreamSeedMode")
        target = Path(self._admission.target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise TargetChanged(f"stream target appeared before seed: {target}")
        source_descriptor: int | None = None
        destination = os.open(
            target,
            os.O_CREAT | os.O_EXCL | os.O_RDWR,
            0o600,
        )
        try:
            self._stream_reservation = _descriptor_receipt(
                destination,
                target,
                "stream-reservation",
            )
            if (
                seed_mode is StreamSeedMode.PRESERVE_BASE
                and self._prior_receipt is not None
            ):
                prior = self._prior_receipt
                source_descriptor = os.open(self.backup, os.O_RDONLY)
                source_expected_stat = _bind_descriptor_to_receipt(
                    source_descriptor,
                    self.backup,
                    prior,
                )
                source_digest = hashlib.sha256()
                while True:
                    block = os.read(source_descriptor, _HASH_CHUNK_BYTES)
                    if not block:
                        break
                    source_digest.update(block)
                    view = memoryview(block)
                    while view:
                        written = os.write(destination, view)
                        view = view[written:]
                source_after = _descriptor_content_receipt(
                    source_descriptor,
                    self.backup,
                    "stream-seed-source",
                    durable_fsync=self._durable_fsync,
                    evidence_digest=source_digest.hexdigest(),
                    expected_stat=source_expected_stat,
                )
                if (
                    source_after.identity != prior.identity
                    or source_after.snapshot != prior.snapshot
                ):
                    raise TargetChanged("staged prior changed while seeding stream")
            if seed_mode is StreamSeedMode.EMPTY_REPLACEMENT:
                receipt = _descriptor_receipt(
                    destination,
                    target,
                    "stream-seed",
                )
            else:
                receipt = _descriptor_content_receipt(
                    destination,
                    target,
                    "stream-seed",
                    durable_fsync=self._durable_fsync,
                )
            checkpoint = _descriptor_stream_stat_receipt(
                destination,
                target,
                evidence_digest=str(receipt.snapshot.digest),
                evidence_bytes=int(receipt.snapshot.size or 0),
                ordinal=self._coordinator._next_ordinal(),
                role="stream-seed",
                durable_fsync=self._durable_fsync,
            )
        finally:
            if source_descriptor is not None:
                os.close(source_descriptor)
            os.close(destination)
        return receipt, checkpoint

    def _require_stream(
        self,
        attempt: StreamAttempt,
        lease: TargetLease,
    ) -> None:
        self._require_lease(lease)
        if self._stream_attempt is None:
            raise TransactionStateError("transaction has no streaming attempt")
        _require_identity(attempt, self._stream_attempt, "stream attempt")

    def begin_stream(
        self,
        *,
        admission: TargetAdmission,
        transaction_owner: OwnerToken,
        target_owner: OwnerToken,
        lease: TargetLease,
        pool=None,
        file_lock=None,
        seed_mode: StreamSeedMode = StreamSeedMode.PRESERVE_BASE,
    ) -> StreamAttempt:
        if type(seed_mode) is not StreamSeedMode:
            raise TypeError("seed_mode must be a StreamSeedMode")
        with self._lock:
            self._require_admission_owners(
                admission=admission,
                transaction_owner=transaction_owner,
                target_owner=target_owner,
            )
            self._require_lease(lease)
            if self._stream_attempt is not None:
                raise TransactionStateError("streaming attempt already exists")
            if self._phase not in {
                TransactionPhase.LEASED,
                TransactionPhase.READY_TO_RETRY,
            }:
                raise TransactionStateError(
                    f"transaction cannot stream in phase {self._phase.value}"
                )
            if self._pending:
                raise TransactionStateError(
                    "transaction cleanup must finish before a streaming writer"
                )
            if self._phase is TransactionPhase.READY_TO_RETRY:
                if not self._has_terminal_receipt("rollback-ready"):
                    self._phase = TransactionPhase.INTEGRITY_HOLD
                    raise TransactionStateError(
                        "retry-ready phase lacks positive rollback receipt"
                    )
                self._terminal_receipt = None
            if self.backup.exists() or self._candidate.exists():
                raise TransactionStateError(
                    "unowned private transaction path blocks streaming attempt"
                )

            selected_pool = get_pool() if pool is None else pool
            self._phase = TransactionPhase.EXECUTING
            self._stream_file_lock = file_lock
            self._own_pool_resume(selected_pool)
            self._pending.add(RetryAction.POOL_PAUSE)
            try:
                selected_pool.pause(self._admission.target)
            except BaseException:
                self._pending.discard(RetryAction.POOL_PAUSE)
                self._refresh_phase()
                raise
            self._pending.discard(RetryAction.POOL_PAUSE)

            primary: BaseException | None = None
            cleanup: list[BaseException] = []
            try:
                with (nullcontext() if file_lock is None else file_lock):
                    self._validate_target()
                    self._ensure_cleanup_token()
                    self._pending.add(RetryAction.ROLLBACK)
                    if self.stream_base_snapshot.exists:
                        self._stage_prior()
                    attempt = StreamAttempt(
                        self._admission.target,
                        self._coordinator._next_ordinal(),
                    )
                    reservation, checkpoint = self._copy_stream_seed(
                        seed_mode=seed_mode,
                    )
                    self._stream_attempt = attempt
                    self._stream_reservation = reservation
                    self._stream_checkpoint = checkpoint
                    self._stream_checkpoint_fresh = True
                    return attempt
            except BaseException as exc:
                primary = exc
                if self._stream_reservation is not None:
                    failure = self._attempt_stream_retire()
                    if failure is not None:
                        cleanup.append(failure)
                if RetryAction.ROLLBACK in self._pending:
                    failure = self._attempt_rollback()
                    if failure is not None:
                        cleanup.append(failure)
                if not self._target_transition_pending():
                    failure = self._attempt_pool_resume(selected_pool)
                    if failure is not None:
                        cleanup.append(failure)
                self._refresh_phase()
                if cleanup:
                    raise primary.with_traceback(primary.__traceback__) from cleanup[0]
                raise

    def authorize_stream_mutation(
        self,
        attempt: StreamAttempt,
        *,
        lease: TargetLease,
    ) -> None:
        with self._lock:
            self._require_stream(attempt, lease)
            if self._phase is not TransactionPhase.EXECUTING:
                raise TransactionStateError(
                    f"stream mutation requires EXECUTING, got {self._phase.value}"
                )
            reservation = self._stream_reservation
            if reservation is None:
                raise TransactionStateError("stream has no descriptor reservation")
            if self._stream_close_owner is not None:
                raise TransactionStateError(
                    "stream mutation cannot follow an authorized close"
                )
            checkpoint = self._stream_checkpoint
            try:
                stat_result = os.stat(self._admission.target)
            except FileNotFoundError as exc:
                self._phase = TransactionPhase.INTEGRITY_HOLD
                raise TargetChanged("stream working target disappeared") from exc
            if _FileIdentity(int(stat_result.st_dev), int(stat_result.st_ino)) != (
                reservation.identity
            ):
                self._phase = TransactionPhase.INTEGRITY_HOLD
                raise TargetChanged("stream working target identity changed")
            if self._stream_checkpoint_fresh:
                observed = _stat_identity(stat_result)
                sealed = (
                    None if checkpoint is None
                    else _stream_receipt_identity(checkpoint)
                )
                if observed != sealed:
                    self._phase = TransactionPhase.INTEGRITY_HOLD
                    # Name both (dev, ino, size, mtime_ns, ctime_ns) views:
                    # this refusal is the only evidence a live run leaves.
                    raise TargetChanged(
                        "stream checkpoint failed identity verification "
                        f"(observed={observed} checkpoint={sealed})"
                    )
                self._stream_checkpoint_fresh = False
                self._stream_terminal_receipt = None
                self._stream_terminal_stat = None

    def seal_stream_checkpoint(
        self,
        attempt: StreamAttempt,
        *,
        lease: TargetLease,
        descriptor: int,
        evidence_digest: str,
        evidence_bytes: int,
    ) -> StreamCheckpoint:
        with self._lock:
            self._require_stream(attempt, lease)
            if self._phase is not TransactionPhase.EXECUTING:
                raise TransactionStateError(
                    f"checkpoint requires EXECUTING, got {self._phase.value}"
                )
            try:
                receipt = _descriptor_stream_stat_receipt(
                    descriptor,
                    Path(self._admission.target),
                    evidence_digest=evidence_digest,
                    evidence_bytes=evidence_bytes,
                    ordinal=self._coordinator._next_ordinal(),
                    role="stream-checkpoint",
                    durable_fsync=self._durable_fsync,
                )
            except (OSError, TransactionStateError, TargetChanged) as exc:
                self._phase = TransactionPhase.INTEGRITY_HOLD
                if isinstance(exc, TargetChanged):
                    raise
                raise TargetChanged(
                    "stream checkpoint descriptor/evidence seal failed"
                ) from exc
            reservation = self._stream_reservation
            if reservation is None or receipt.identity != reservation.identity:
                self._phase = TransactionPhase.INTEGRITY_HOLD
                raise TargetChanged("stream checkpoint names a foreign inode")
            checkpoint = StreamCheckpoint(
                receipt.path,
                receipt.size,
                receipt.mtime_ns,
                receipt.ctime_ns,
                receipt.evidence_digest,
                receipt.evidence_bytes,
                receipt.ordinal,
            )
            self._stream_checkpoint = receipt
            self._stream_checkpoint_token = checkpoint
            self._stream_checkpoint_fresh = True
            self._stream_terminal_receipt = None
            self._stream_terminal_stat = None
            return checkpoint

    def begin_stream_close(
        self,
        attempt: StreamAttempt,
        *,
        lease: TargetLease,
    ) -> _StreamCloseAttempt | None:
        """Authorize the controlled HDF5 close after an exact durable floor.

        A close is allowed to update HDF5 metadata, but only after the prior
        post-fsync checkpoint still matches exactly.  The returned token must
        then be resolved by :meth:`seal_stream_close` after ``h5py`` reports a
        successful close.  Without a durable floor there is no irreversible
        receipt to preserve, so no close token is needed.
        """
        with self._lock:
            self._require_stream(attempt, lease)
            if self._phase is not TransactionPhase.EXECUTING:
                raise TransactionStateError(
                    "stream close authorization requires EXECUTING"
                )
            if (
                self._stream_durable_floor is None
                and not self._fast_regenerable
            ):
                return None
            if self._stream_close_owner is not None:
                return self._stream_close_owner
            checkpoint = self._stream_checkpoint
            floor = (
                self._stream_durable_floor
                if self._stream_durable_floor is not None
                else self._stream_checkpoint_token
            )
            if (
                not self._stream_checkpoint_fresh
                or self._stream_checkpoint_token is not floor
                or checkpoint is None
                or not _stream_stat_matches(self._admission.target, checkpoint)
            ):
                self._phase = TransactionPhase.INTEGRITY_HOLD
                raise TargetChanged(
                    "durability floor changed before the controlled close"
                )
            owner = _StreamCloseAttempt(
                self._admission.target,
                self._coordinator._next_ordinal(),
            )
            self._stream_close_owner = owner
            self._stream_checkpoint_fresh = False
            return owner

    def seal_stream_close(
        self,
        attempt: StreamAttempt,
        close_owner: _StreamCloseAttempt,
        *,
        lease: TargetLease,
        descriptor: int | None = None,
        expected_stat: tuple[int, int, int, int, int] | None = None,
        evidence_digest: str | None = None,
        evidence_bytes: int | None = None,
    ) -> StreamCheckpoint:
        """Resolve a successful controlled close to a new exact durable floor."""
        with self._lock:
            self._require_stream(attempt, lease)
            owner = self._stream_close_owner
            _require_identity(close_owner, owner, "stream close")
            resolved = self._stream_close_checkpoint
            if resolved is not None:
                receipt = self._stream_checkpoint
                if receipt is None or not _stream_stat_matches(
                    self._admission.target, receipt,
                ):
                    self._phase = TransactionPhase.INTEGRITY_HOLD
                    raise TargetChanged(
                        "resolved stream close changed before reuse"
                    )
                return resolved
            floor_receipt = self._stream_checkpoint
            if floor_receipt is None:
                self._phase = TransactionPhase.INTEGRITY_HOLD
                raise TransactionStateError(
                    "controlled close has no preceding durable checkpoint"
                )
            owned_descriptor = descriptor is None
            if descriptor is None:
                descriptor = os.open(self._admission.target, os.O_RDONLY)
            try:
                try:
                    receipt = _descriptor_stream_stat_receipt(
                        descriptor,
                        Path(self._admission.target),
                        evidence_digest=(
                            floor_receipt.evidence_digest
                            if evidence_digest is None
                            else evidence_digest
                        ),
                        evidence_bytes=(
                            floor_receipt.evidence_bytes
                            if evidence_bytes is None
                            else evidence_bytes
                        ),
                        ordinal=self._coordinator._next_ordinal(),
                        role="stream-close",
                        expected_stat=expected_stat,
                        durable_fsync=self._durable_fsync,
                    )
                except TargetChanged:
                    self._phase = TransactionPhase.INTEGRITY_HOLD
                    raise
            finally:
                if owned_descriptor:
                    os.close(descriptor)
            reservation = self._stream_reservation
            if reservation is None or receipt.identity != reservation.identity:
                self._phase = TransactionPhase.INTEGRITY_HOLD
                raise TargetChanged("controlled close names a foreign inode")
            checkpoint = StreamCheckpoint(
                receipt.path,
                receipt.size,
                receipt.mtime_ns,
                receipt.ctime_ns,
                receipt.evidence_digest,
                receipt.evidence_bytes,
                receipt.ordinal,
            )
            self._stream_checkpoint = receipt
            self._stream_checkpoint_token = checkpoint
            self._stream_checkpoint_fresh = True
            self._stream_durable_floor = checkpoint
            self._stream_close_checkpoint = checkpoint
            self._phase = TransactionPhase.EXECUTING
            return checkpoint

    def hold_stream_close(
        self,
        attempt: StreamAttempt,
        close_owner: _StreamCloseAttempt,
        *,
        lease: TargetLease,
    ) -> None:
        """Retain the exact close owner after semantic proof cannot complete."""
        with self._lock:
            self._require_stream(attempt, lease)
            _require_identity(
                close_owner,
                self._stream_close_owner,
                "stream close",
            )
            self._phase = TransactionPhase.INTEGRITY_HOLD

    def promote_stream_checkpoint(
        self,
        attempt: StreamAttempt,
        checkpoint: StreamCheckpoint,
        *,
        lease: TargetLease,
    ) -> StreamCheckpoint:
        """Make a sealed checkpoint the irreversible H10 durability floor.

        H10 durable receipts are monotonic.  They may be published only after
        the older rollback action has been retired, so no later abort can
        restore bytes older than this exact checkpoint.  The private backup
        remains owned until terminal cleanup; routine O(K) checkpoints never
        rescan the whole prior file merely to unlink it.
        """
        with self._lock:
            self._require_stream(attempt, lease)
            if self._phase is not TransactionPhase.EXECUTING:
                raise TransactionStateError(
                    "durability-floor promotion requires EXECUTING"
                )
            token = self._stream_checkpoint_token
            _require_identity(checkpoint, token, "stream checkpoint")
            if not self._stream_checkpoint_fresh:
                raise TransactionStateError(
                    "only the fresh sealed checkpoint can become durable"
                )
            receipt = self._stream_checkpoint
            if receipt is None or not _stream_stat_matches(
                self._admission.target, receipt,
            ):
                self._phase = TransactionPhase.INTEGRITY_HOLD
                raise TargetChanged(
                    "durability-floor checkpoint changed before promotion"
                )
            self._stream_durable_floor = checkpoint
            self._pending.discard(RetryAction.ROLLBACK)
            return checkpoint

    def seal_stream_terminal(
        self,
        attempt: StreamAttempt,
        *,
        lease: TargetLease,
    ) -> StreamTerminal:
        with self._lock:
            self._require_stream(attempt, lease)
            if self._phase is not TransactionPhase.EXECUTING:
                raise TransactionStateError(
                    f"terminal seal requires EXECUTING, got {self._phase.value}"
                )
            descriptor: int | None = None
            receipt: _ObjectReceipt | None = None
            terminal_stat: _StreamStatReceipt | None = None
            terminal: StreamTerminal | None = None
            failure: BaseException | None = None
            try:
                descriptor = os.open(self._admission.target, os.O_RDONLY)
                expected_stat = _stat_identity(os.fstat(descriptor))
                if self._durable_fsync:
                    _fsync_descriptor(descriptor, Path(self._admission.target))
                if _stat_identity(os.fstat(descriptor)) != expected_stat:
                    raise TargetChanged(
                        "stream terminal descriptor changed during fsync"
                    )
                checkpoint = self._stream_checkpoint if self._fast_regenerable else None
                if self._fast_regenerable and (
                    checkpoint is None or not self._stream_checkpoint_fresh
                    or expected_stat != _stream_receipt_identity(checkpoint)
                ):
                    raise TargetChanged("fast stream terminal lost its close checkpoint")
                receipt = _descriptor_content_receipt(
                    descriptor, Path(self._admission.target), "stream-terminal",
                    durable_fsync=False,
                    evidence_digest=(
                        None if checkpoint is None else checkpoint.evidence_digest
                    ),
                    expected_stat=expected_stat,
                )
                terminal_stat = _descriptor_stream_stat_receipt(
                    descriptor,
                    Path(self._admission.target),
                    evidence_digest=str(receipt.snapshot.digest),
                    evidence_bytes=(
                        int(receipt.snapshot.size or 0)
                        if checkpoint is None else checkpoint.evidence_bytes
                    ),
                    ordinal=self._coordinator._next_ordinal(),
                    role="stream-terminal",
                    expected_stat=expected_stat,
                    durable_fsync=False,
                )
                reservation = self._stream_reservation
                if reservation is None or receipt.identity != reservation.identity:
                    raise TargetChanged("stream terminal seal names a foreign inode")
                terminal = StreamTerminal(
                    receipt.path,
                    int(receipt.snapshot.size or 0),
                    str(receipt.snapshot.digest),
                    terminal_stat.ordinal,
                    terminal_stat.identity.device,
                    terminal_stat.identity.inode,
                    terminal_stat.mtime_ns,
                    terminal_stat.ctime_ns,
                )
            except BaseException as exc:
                failure = exc
            finally:
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except BaseException as exc:
                        if failure is None or (
                            isinstance(failure, Exception)
                            and not isinstance(exc, Exception)
                        ):
                            failure = exc
            if failure is not None:
                self._phase = TransactionPhase.INTEGRITY_HOLD
                if isinstance(failure, TargetChanged):
                    raise failure
                if not isinstance(failure, Exception):
                    raise failure
                raise TargetChanged("stream terminal content seal failed") from failure
            if receipt is None or terminal_stat is None or terminal is None:
                self._phase = TransactionPhase.INTEGRITY_HOLD
                raise TargetChanged("stream terminal content seal failed")
            self._stream_terminal_receipt = receipt
            self._stream_terminal_stat = terminal_stat
            return terminal

    def commit_stream_epoch(
        self,
        attempt: StreamAttempt,
        *,
        lease: TargetLease,
    ) -> TransactionSnapshot:
        with self._lock:
            self._require_stream(attempt, lease)
            if self._phase is TransactionPhase.EPOCH_COMMITTED:
                return self.snapshot()
            if self._phase not in {
                TransactionPhase.EXECUTING,
                TransactionPhase.CLEANUP_PENDING,
            }:
                raise TransactionStateError(
                    f"epoch commit requires EXECUTING, got {self._phase.value}"
                )
            with (
                nullcontext()
                if self._stream_file_lock is None
                else self._stream_file_lock
            ):
                failure = self._verify_stream_terminal(role="epoch-committed")
                if failure is not None:
                    self._phase = TransactionPhase.INTEGRITY_HOLD
                    raise failure
                receipt = self._stream_terminal_receipt
                if receipt is None:
                    self._phase = TransactionPhase.INTEGRITY_HOLD
                    raise TransactionStateError(
                        "epoch commit lacks an exact terminal receipt")
                self._stream_epoch_receipt = receipt
                self._pending.discard(RetryAction.ROLLBACK)
                if (
                    self._backup_owned
                    or self._captured_prior is not None
                    or RetryAction.BACKUP_UNLINK in self._pending
                ):
                    failure = self._attempt_backup_unlink()
                    if failure is not None:
                        self._phase = TransactionPhase.CLEANUP_PENDING
                        raise CleanupIncomplete(self.snapshot()) from failure
            self._phase = TransactionPhase.EPOCH_COMMITTED
            return self.snapshot()

    def begin_stream_epoch(
        self,
        previous: StreamAttempt,
        *,
        admission: TargetAdmission,
        transaction_owner: OwnerToken,
        target_owner: OwnerToken,
        lease: TargetLease,
    ) -> StreamAttempt:
        with self._lock:
            self._require_admission_owners(
                admission=admission,
                transaction_owner=transaction_owner,
                target_owner=target_owner,
            )
            self._require_stream(previous, lease)
            if self._phase is not TransactionPhase.EPOCH_COMMITTED:
                raise TransactionStateError(
                    "owned epoch extension requires a committed epoch"
                )
            if self._pending - {RetryAction.POOL_RESUME}:
                raise TransactionStateError(
                    "epoch cleanup must finish before extension"
                )
            self._validate_target()
            if self.backup.exists() or self._candidate.exists():
                raise TransactionStateError(
                    "private transaction path blocks owned epoch extension"
                )
            prior_stream = (
                self._stream_attempt, self._stream_reservation,
                self._stream_checkpoint, self._stream_checkpoint_token,
                self._stream_checkpoint_fresh, self._stream_durable_floor,
                self._stream_close_owner, self._stream_close_checkpoint,
                self._stream_terminal_receipt, self._stream_terminal_stat,
                self._stream_committed, self._stream_preserved_partial,
                self._stream_retain_partial,
            )
            self._stream_attempt = None
            self._stream_reservation = None
            self._stream_checkpoint = None
            self._stream_checkpoint_token = None
            self._stream_durable_floor = None
            self._stream_close_owner = None
            self._stream_close_checkpoint = None
            self._stream_preserved_partial = None
            self._stream_retain_partial = False
            self._published = False
            self._phase = TransactionPhase.EXECUTING
            try:
                with (
                    nullcontext()
                    if self._stream_file_lock is None
                    else self._stream_file_lock
                ):
                    self._pending.add(RetryAction.ROLLBACK)
                    self._stage_prior()
                    attempt = StreamAttempt(
                        self._admission.target,
                        self._coordinator._next_ordinal(),
                    )
                    reservation, checkpoint = self._copy_stream_seed()
                    self._stream_attempt = attempt
                    self._stream_reservation = reservation
                    self._stream_checkpoint = checkpoint
                    self._stream_checkpoint_token = None
                    self._stream_checkpoint_fresh = True
                    self._stream_terminal_receipt = None
                    self._stream_terminal_stat = None
                    self._stream_committed = False
                    self._terminal_receipt = None
                    return attempt
            except BaseException as primary:
                failures = []
                with (nullcontext() if self._stream_file_lock is None
                      else self._stream_file_lock):
                    if self._stream_reservation is not None:
                        failure = self._attempt_stream_retire()
                        if failure is not None:
                            failures.append(failure)
                    if RetryAction.ROLLBACK in self._pending:
                        failure = self._attempt_rollback()
                        if failure is not None:
                            failures.append(failure)
                    if (RetryAction.STREAM_RETIRE in self._pending
                            and RetryAction.ROLLBACK not in self._pending):
                        failure = self._attempt_stream_retire()
                        if failure is not None:
                            failures.append(failure)
                    (self._stream_attempt, self._stream_reservation,
                     self._stream_checkpoint, self._stream_checkpoint_token,
                     self._stream_checkpoint_fresh, self._stream_durable_floor,
                     self._stream_close_owner, self._stream_close_checkpoint,
                     self._stream_terminal_receipt, self._stream_terminal_stat,
                     self._stream_committed, self._stream_preserved_partial,
                     self._stream_retain_partial) = prior_stream
                    try:
                        self._phase = TransactionPhase.EXECUTING
                        self.seal_stream_terminal(previous, lease=lease)
                        self._install_terminal_receipt(
                            expected=self.stream_base_snapshot,
                            role="epoch-committed",
                        )
                    except BaseException as failure:
                        failures.append(failure)
                if failures:
                    self._refresh_phase()
                    raise primary.with_traceback(primary.__traceback__) from failures[0]
                self._phase = TransactionPhase.EPOCH_COMMITTED
                raise

    def _attempt_stream_retire(self) -> BaseException | None:
        self._ensure_cleanup_token()
        self._pending.add(RetryAction.STREAM_RETIRE)
        reservation = self._stream_reservation
        if reservation is None:
            self._pending.discard(RetryAction.STREAM_RETIRE)
            return None
        target = self._capture(self._admission.target)
        partial = self._capture(self._stream_partial)
        receipt = self._stream_partial_receipt
        if partial.exists:
            if receipt is None or not _exact_receipt(partial, receipt):
                return TargetChanged(
                    f"foreign stream partial occupies {self._stream_partial}"
                )
            if not target.exists and RetryAction.ROLLBACK in self._pending:
                return None
            if target != self.stream_base_snapshot:
                return TargetChanged("stream partial cleanup found a foreign target")
            if self._stream_retain_partial:
                self._pending.discard(RetryAction.STREAM_RETIRE)
                return None
            self._stream_partial_cleanup = receipt
            try:
                _unlink(self._stream_partial)
            except FileNotFoundError:
                pass
            except BaseException as exc:
                return exc
            partial = self._capture(self._stream_partial)
            if partial.exists:
                return TargetChanged("owned stream partial survived exact cleanup")
            self._stream_partial_receipt = self._stream_partial_cleanup = None
            self._pending.discard(RetryAction.STREAM_RETIRE)
            return None
        if receipt is not None:
            if self._stream_partial_cleanup is receipt:
                if target != self.stream_base_snapshot:
                    return TargetChanged(
                        "stream cleanup completion found a foreign target"
                    )
                self._stream_partial_receipt = self._stream_partial_cleanup = None
                self._pending.discard(RetryAction.STREAM_RETIRE)
                return None
            if _exact_receipt(target, reservation):
                # A failed/uncertain rename resolved to the exact unchanged
                # source object.  Retire may be attempted again without
                # promoting an absent destination to owned authority.
                self._stream_partial_receipt = None
            else:
                return TargetChanged("owned stream partial disappeared without cleanup")
        if not target.exists:
            self._pending.discard(RetryAction.STREAM_RETIRE)
            return None
        if not _same_identity(target, reservation):
            return TargetChanged("stream retirement refused foreign final target")
        self._stream_partial_receipt = _receipt_for_snapshot(
            self._stream_partial,
            target,
            "stream-partial",
        )
        replace_error: BaseException | None = None
        try:
            _replace(Path(self._admission.target), self._stream_partial)
        except BaseException as exc:
            replace_error = exc
        try:
            observed = self._capture(self._stream_partial)
            if not _exact_receipt(observed, self._stream_partial_receipt):
                current = self._capture(self._admission.target)
                if replace_error is not None and _exact_receipt(current, reservation):
                    self._stream_partial_receipt = None
                    return replace_error
                return TargetChanged("stream partial lost its exact owned bytes")
        except BaseException as exc:
            return replace_error if replace_error is not None else exc
        return replace_error

    def _verify_stream_terminal(
        self, *, role: str = "committed-result",
    ) -> BaseException | None:
        receipt = self._stream_terminal_receipt
        terminal_stat = self._stream_terminal_stat
        reservation = self._stream_reservation
        target = _normalize_target(self._admission.target)
        snapshot = None if receipt is None else receipt.snapshot
        sealed = None if terminal_stat is None else TargetSnapshot(
            True,
            terminal_stat.size,
            terminal_stat.mtime_ns,
            terminal_stat.identity.device,
            terminal_stat.identity.inode,
            terminal_stat.evidence_digest,
        )
        if (
            receipt is None
            or terminal_stat is None
            or reservation is None
            or receipt.path != target
            or terminal_stat.path != target
            or receipt.identity != reservation.identity
            or terminal_stat.identity != reservation.identity
            or snapshot != sealed
            or (
                not self._fast_regenerable
                and terminal_stat.evidence_bytes != terminal_stat.size
            )
            or not _stream_stat_matches(target, terminal_stat)
        ):
            self._terminal_receipt = None
            return TargetChanged("stream terminal seal changed before commit")
        self._terminal_receipt = _TerminalReceipt(
            self._admission.target,
            snapshot,
            snapshot,
            role,
        )
        return None

    def commit_stream(
        self,
        attempt: StreamAttempt,
        *,
        lease: TargetLease,
    ) -> TransactionSnapshot:
        with self._lock:
            self._require_stream(attempt, lease)
            if self._stream_committed:
                return self.retry_cleanup(self._ensure_cleanup_token())
            if self._phase is TransactionPhase.INTEGRITY_HOLD:
                raise TransactionStateError(
                    "stream commit refused while integrity hold is active"
                )
            failures: list[BaseException] = []
            with (
                nullcontext()
                if self._stream_file_lock is None
                else self._stream_file_lock
            ):
                failure = self._verify_stream_terminal()
                if failure is not None:
                    self._phase = TransactionPhase.INTEGRITY_HOLD
                    raise failure
                self._published = True
                self._writer_succeeded = True
                self._writer_returned = True
                self._stream_committed = True
                self._pending.discard(RetryAction.ROLLBACK)
                if self._backup_owned or self._captured_prior is not None:
                    failure = self._attempt_backup_unlink()
                    if failure is not None:
                        failures.append(failure)
            if not self._target_transition_pending():
                pool = self._resume_pool
                if pool is not None:
                    failure = self._attempt_pool_resume(pool)
                    if failure is not None:
                        failures.append(failure)
            self._refresh_phase()
            if failures:
                raise CleanupIncomplete(self.snapshot()) from failures[0]
            return self.snapshot()

    def abort_stream(
        self,
        attempt: StreamAttempt,
        *,
        lease: TargetLease,
        retain_partial: bool = False,
    ) -> TransactionSnapshot:
        with self._lock:
            self._require_stream(attempt, lease)
            self._stream_retain_partial |= bool(retain_partial)
            if self._stream_committed:
                raise TransactionStateError("committed stream cannot be aborted")
            if self._phase is TransactionPhase.ABORTED:
                return self.snapshot()
            failures: list[BaseException] = []
            with (
                nullcontext()
                if self._stream_file_lock is None
                else self._stream_file_lock
            ):
                floor = self._stream_durable_floor
                if floor is not None:
                    checkpoint = self._stream_checkpoint
                    if (
                        not self._stream_checkpoint_fresh
                        or self._stream_checkpoint_token is not floor
                        or checkpoint is None
                    ):
                        self._phase = TransactionPhase.INTEGRITY_HOLD
                        raise TransactionStateError(
                            "unsealed mutation follows the irreversible durable "
                            "floor; preserving the canonical target under its "
                            "live owner"
                        )
                    descriptor = os.open(self._admission.target, os.O_RDONLY)
                    try:
                        preserved = _descriptor_content_receipt(
                            descriptor,
                            Path(self._admission.target),
                            "durable-floor-preserved",
                            durable_fsync=self._durable_fsync,
                        )
                    finally:
                        os.close(descriptor)
                    if (
                        preserved.identity != checkpoint.identity
                        or not _stream_stat_matches(
                            self._admission.target, checkpoint,
                        )
                    ):
                        self._phase = TransactionPhase.INTEGRITY_HOLD
                        raise TargetChanged(
                            "durability floor changed after its exact seal"
                        )
                    self._terminal_receipt = _TerminalReceipt(
                        self._admission.target,
                        preserved.snapshot,
                        preserved.snapshot,
                        "durable-floor-preserved",
                    )
                    self._stream_preserved_partial = self._admission.target
                    self._pending.discard(RetryAction.ROLLBACK)
                    if (
                        self._backup_owned
                        or self._captured_prior is not None
                        or RetryAction.BACKUP_UNLINK in self._pending
                    ):
                        failure = self._attempt_backup_unlink()
                        if failure is not None:
                            failures.append(failure)
                    if not failures and RetryAction.POOL_RESUME in self._pending:
                        pool = self._resume_pool
                        if pool is None:
                            failures.append(TransactionStateError(
                                "durable-floor abort has no pool-resume owner"
                            ))
                        else:
                            failure = self._attempt_pool_resume(pool)
                            if failure is not None:
                                failures.append(failure)
                    if failures:
                        raise CleanupIncomplete(self.snapshot()) from failures[0]
                    if self._pending:
                        raise CleanupIncomplete(self.snapshot())
                    self._phase = TransactionPhase.ABORTED
                    return self.snapshot()
                if RetryAction.STREAM_RETIRE in self._pending or (
                    self._stream_partial_receipt is None
                    and RetryAction.ROLLBACK in self._pending
                    and Path(self._admission.target).exists()
                ):
                    failure = self._attempt_stream_retire()
                    if failure is not None:
                        failures.append(failure)
                if not failures and RetryAction.ROLLBACK in self._pending:
                    failure = self._attempt_rollback()
                    if failure is not None:
                        failures.append(failure)
                if not failures and RetryAction.STREAM_RETIRE in self._pending:
                    failure = self._attempt_stream_retire()
                    if failure is not None:
                        failures.append(failure)
            if not self._target_transition_pending():
                pool = self._resume_pool
                if pool is not None:
                    failure = self._attempt_pool_resume(pool)
                    if failure is not None:
                        failures.append(failure)
            self._refresh_phase()
            if failures:
                raise CleanupIncomplete(self.snapshot()) from failures[0]
            if self._pending:
                raise CleanupIncomplete(self.snapshot())
            if not self._has_terminal_receipt("rollback-ready"):
                self._phase = TransactionPhase.INTEGRITY_HOLD
                raise TransactionStateError(
                    "stream abort lacks positive rollback-ready proof"
                )
            self._phase = TransactionPhase.ABORTED
            return self.snapshot()

    def execute(
        self,
        write: Callable[[Path], object],
        *,
        admission: TargetAdmission,
        transaction_owner: OwnerToken,
        target_owner: OwnerToken,
        lease: TargetLease,
        pool=None,
        validate_writer_result: Callable[[Path, TargetSnapshot], object] | None = None,
    ) -> TransactionSnapshot:
        """Execute one writer under the admitted target and H10 pool boundary.

        A successful writer is never invoked again by cleanup retry.  Failed
        pre-commit work is rolled back to the admitted prior target (or to
        absence) before this transaction can accept another explicit execute.
        """
        if validate_writer_result is not None and not callable(validate_writer_result):
            raise TypeError("writer-result validator must be callable")
        with self._lock:
            self._require_admission_owners(
                admission=admission,
                transaction_owner=transaction_owner,
                target_owner=target_owner,
            )
            self._require_lease(lease)
            if self._writer_succeeded:
                raise TransactionStateError(
                    "writer already succeeded; retry cleanup instead"
                )
            if self._phase not in {
                TransactionPhase.LEASED,
                TransactionPhase.READY_TO_RETRY,
            }:
                raise TransactionStateError(
                    f"transaction cannot execute in phase {self._phase.value}"
                )
            if self._pending:
                raise TransactionStateError(
                    "transaction cleanup must finish before another writer"
                )
            if self._phase is TransactionPhase.READY_TO_RETRY:
                if not self._has_terminal_receipt("rollback-ready"):
                    self._phase = TransactionPhase.INTEGRITY_HOLD
                    raise TransactionStateError(
                        "retry-ready phase lacks positive rollback receipt"
                    )
                # A fresh attempt consumes the proof before its first
                # fallible filesystem boundary.  Failure cannot silently
                # reuse a receipt from the preceding attempt.
                self._terminal_receipt = None
                self._phase = TransactionPhase.INTEGRITY_HOLD
            self._validate_target()
            if self.backup.exists():
                raise TransactionStateError(
                    f"unowned backup occupies {self.backup}; refusing overwrite"
                )
            if self._candidate.exists():
                raise TransactionStateError(
                    f"unowned candidate occupies {self._candidate}; refusing write"
                )

            selected_pool = get_pool() if pool is None else pool
            self._phase = TransactionPhase.EXECUTING
            # A pause may acquire reader exclusion and then raise.  Install the
            # exact conservative resume owner before entering that boundary.
            self._own_pool_resume(selected_pool)
            self._pending.add(RetryAction.POOL_PAUSE)
            try:
                selected_pool.pause(self._admission.target)
            except BaseException:
                self._pending.discard(RetryAction.POOL_PAUSE)
                self._refresh_phase()
                raise
            self._pending.discard(RetryAction.POOL_PAUSE)

            primary: BaseException | None = None
            cleanup_failures: list[BaseException] = []
            try:
                # Recheck after reader exclusion.  No validation result is
                # trusted across a later clobber: the prior is captured and
                # fingerprinted, and publication itself is atomic no-clobber.
                self._validate_target()
                self._ensure_cleanup_token()
                self._pending.add(RetryAction.ROLLBACK)
                try:
                    if self._admission.snapshot.exists:
                        self._stage_prior()
                    self._reserve_candidate()
                    self._resolve_candidate_reservation()
                    if self._candidate_snapshot is None:
                        raise TargetChanged(
                            "owned candidate disappeared during reservation"
                        )

                    # Install the post-writer observation owner before entering
                    # user code.  An authorized mutation of the private path is
                    # resolved independently from the callback's primary error.
                    self._pending.add(RetryAction.WRITER_CAPTURE)
                    self._writer_returned = False
                    writer_error: BaseException | None = None
                    try:
                        write(self._candidate)
                    except BaseException as exc:
                        writer_error = exc
                    else:
                        self._writer_returned = True
                    observation_error: BaseException | None = None
                    try:
                        self._resolve_writer_capture()
                    except BaseException as exc:
                        observation_error = exc
                    if writer_error is not None:
                        if observation_error is not None:
                            cleanup_failures.append(observation_error)
                        raise writer_error.with_traceback(writer_error.__traceback__)
                    if observation_error is not None:
                        raise observation_error.with_traceback(
                            observation_error.__traceback__
                        )

                    if validate_writer_result is not None:
                        writer_receipt = self._writer_receipt
                        if writer_receipt is None or writer_receipt.result is None:
                            raise TransactionStateError(
                                "writer-result validation lacks an exact receipt"
                            )
                        validate_writer_result(
                            self._candidate,
                            writer_receipt.result.snapshot,
                        )
                        self._require_owned_snapshot(
                            self._candidate,
                            writer_receipt.result.snapshot,
                            "validated writer result",
                        )

                    if self._captured_prior is not None:
                        self._require_owned_snapshot(
                            self.backup,
                            self._captured_prior,
                            "captured prior",
                        )
                    self._publish_candidate()
                except BaseException as exc:
                    primary = exc
                    if RetryAction.ROLLBACK in self._pending:
                        rollback_failure = self._attempt_rollback()
                        if rollback_failure is not None:
                            cleanup_failures.append(rollback_failure)
                    if self._published:
                        # A later exact observation resolved the uncertain
                        # no-clobber boundary.  Its transient observation error
                        # is no longer a truthful operation failure.
                        primary = None
                        cleanup_failures.extend(self._finish_published_cleanup())
                else:
                    cleanup_failures.extend(self._finish_published_cleanup())
            except BaseException as exc:
                if primary is None:
                    primary = exc
                    if RetryAction.ROLLBACK in self._pending:
                        rollback_failure = self._attempt_rollback()
                        if rollback_failure is not None:
                            cleanup_failures.append(rollback_failure)
                    if self._published:
                        primary = None
                        cleanup_failures.extend(self._finish_published_cleanup())
            finally:
                retirement_failure = self._attempt_candidate_retire()
                if retirement_failure is not None:
                    cleanup_failures.append(retirement_failure)
                # The exact pause owner remains live while any later retry may
                # publish or restore the admitted final pathname.
                if not self._target_transition_pending():
                    resume_failure = self._attempt_pool_resume(selected_pool)
                    if resume_failure is not None:
                        cleanup_failures.append(resume_failure)

            self._refresh_phase()
            if primary is not None:
                if cleanup_failures:
                    raise primary.with_traceback(
                        primary.__traceback__
                    ) from cleanup_failures[0]
                raise primary.with_traceback(primary.__traceback__)
            if cleanup_failures:
                raise CleanupIncomplete(self.snapshot()) from cleanup_failures[0]
            return self.snapshot()

    def retry_cleanup(self, cleanup_token: CleanupToken) -> TransactionSnapshot:
        """Retry only retained cleanup actions, idempotently and totally."""
        with self._lock:
            if self._cleanup_token is None:
                raise OwnershipRefused("transaction has no cleanup token")
            _require_identity(cleanup_token, self._cleanup_token, "cleanup token")
            failures: list[BaseException] = []
            if RetryAction.STREAM_RETIRE in self._pending:
                with (
                    nullcontext()
                    if self._stream_file_lock is None
                    else self._stream_file_lock
                ):
                    failure = self._attempt_stream_retire()
                if failure is not None:
                    failures.append(failure)
            if RetryAction.ROLLBACK in self._pending:
                with (
                    nullcontext()
                    if self._stream_attempt is None
                    or self._stream_file_lock is None
                    else self._stream_file_lock
                ):
                    failure = self._attempt_rollback()
                if failure is not None:
                    failures.append(failure)
            if (RetryAction.STREAM_RETIRE in self._pending
                    and RetryAction.ROLLBACK not in self._pending):
                with (nullcontext() if self._stream_file_lock is None
                      else self._stream_file_lock):
                    failure = self._attempt_stream_retire()
                if failure is not None:
                    failures.append(failure)
            if (
                RetryAction.ROLLBACK not in self._pending
                and RetryAction.BACKUP_RESERVATION in self._pending
            ):
                try:
                    self._resolve_backup_reservation()
                except BaseException as exc:
                    failures.append(exc)
                else:
                    failure = self._attempt_backup_unlink()
                    if failure is not None:
                        failures.append(failure)
            if (
                RetryAction.ROLLBACK not in self._pending
                and RetryAction.CANDIDATE_RESERVATION in self._pending
            ):
                try:
                    self._resolve_candidate_reservation()
                except BaseException as exc:
                    failures.append(exc)
                else:
                    failure = self._attempt_candidate_unlink()
                    if failure is not None:
                        failures.append(failure)
            if (
                RetryAction.ROLLBACK not in self._pending
                and RetryAction.WRITER_CAPTURE in self._pending
            ):
                try:
                    self._resolve_writer_capture()
                except BaseException as exc:
                    failures.append(exc)
                else:
                    failure = self._attempt_candidate_unlink()
                    if failure is not None:
                        failures.append(failure)
            if self._published and self._stream_committed:
                with (nullcontext() if self._stream_file_lock is None
                      else self._stream_file_lock):
                    failure = self._verify_stream_terminal()
                    if failure is not None:
                        failures.append(failure)
                    elif (
                        self._backup_owned
                        or self._captured_prior is not None
                        or RetryAction.BACKUP_UNLINK in self._pending
                    ):
                        failure = self._attempt_backup_unlink()
                        if failure is not None:
                            failures.append(failure)
            elif self._published:
                failures.extend(self._finish_published_cleanup())
            elif RetryAction.BACKUP_UNLINK in self._pending:
                with (nullcontext() if self._stream_file_lock is None
                      else self._stream_file_lock):
                    failure = self._attempt_backup_unlink()
                if failure is not None:
                    failures.append(failure)
            if not self._published and RetryAction.CANDIDATE_UNLINK in self._pending:
                failure = self._attempt_candidate_unlink()
                if failure is not None:
                    failures.append(failure)
            retirement_failure = self._attempt_candidate_retire()
            if retirement_failure is not None:
                failures.append(retirement_failure)
            epoch_boundary = (
                self._stream_epoch_receipt is not None
                and not self._published
                and not self._stream_committed
                and self._has_terminal_receipt("epoch-committed")
            )
            if (
                RetryAction.POOL_RESUME in self._pending
                and not self._target_transition_pending()
                and not epoch_boundary
            ):
                pool = self._resume_pool
                if pool is None:
                    failures.append(
                        TransactionStateError("retained pool resume has no pool owner")
                    )
                else:
                    failure = self._attempt_pool_resume(pool)
                    if failure is not None:
                        failures.append(failure)
            if (
                not self._pending
                and not self._published
                and self._stream_epoch_receipt is None
                and self._stream_durable_floor is None
                and not self._has_terminal_receipt("rollback-ready")
            ):
                try:
                    self._install_terminal_receipt(
                        expected=self.stream_base_snapshot,
                        role="rollback-ready",
                    )
                except BaseException as exc:
                    failures.append(exc)
            if epoch_boundary and self._pending <= {RetryAction.POOL_RESUME}:
                try:
                    with (nullcontext() if self._stream_file_lock is None
                          else self._stream_file_lock):
                        failure = self._verify_stream_terminal(
                            role="epoch-committed",
                        )
                        if failure is not None:
                            raise failure
                except BaseException as exc:
                    failures.append(exc)
                    self._phase = TransactionPhase.INTEGRITY_HOLD
                else:
                    self._phase = TransactionPhase.EPOCH_COMMITTED
            else:
                self._refresh_phase()
            if failures:
                raise CleanupIncomplete(self.snapshot()) from failures[0]
            return self.snapshot()

    def abort(
        self,
        *,
        admission: TargetAdmission,
        transaction_owner: OwnerToken,
        target_owner: OwnerToken,
        lease: TargetLease,
    ) -> TransactionSnapshot:
        """Truthfully retire an uncommitted, fully rolled-back transaction."""
        with self._lock:
            self._require_admission_owners(
                admission=admission,
                transaction_owner=transaction_owner,
                target_owner=target_owner,
            )
            self._require_lease(lease)
            if self._phase is TransactionPhase.INTEGRITY_HOLD:
                raise TransactionStateError(
                    "cannot abort without positive terminal integrity proof"
                )
            if self._pending:
                raise TransactionStateError("cannot abort with pending cleanup")
            if self._published:
                raise TransactionStateError("committed writer cannot be aborted")
            self._phase = TransactionPhase.ABORTED
            return self.snapshot()


T = TypeVar("T")


class XyeOutputTransaction:
    """One run's staged XYE set and stale-tail publication transaction."""

    output_receipt_capabilities = frozenset({OutputReceiptCapability.DURABLE_XYE})

    def __init__(
        self,
        *,
        coordinator: OutputTransactionCoordinator,
        directory: str,
        run_owner: OwnerToken,
    ):
        self._coordinator = coordinator
        self._directory = directory
        self._run_owner = run_owner
        # Keyed by label, insertion-ordered.  A list rebuild per stage was
        # O(staged) per frame and therefore quadratic over a run; XYE is about
        # a third of a long Run's wall clock.
        self._staged: dict[int, object] = {}
        self._attempted: tuple[tuple[int, object], ...] | None = None
        self._attempted_final: bool | None = None
        self._pending_stale: list[Path] | None = None
        self._cleanup_token: CleanupToken | None = None
        self._complete = False
        self._retryable = False
        self._lock = threading.RLock()

    def _require_run(self, run_owner: OwnerToken) -> None:
        _require_identity(run_owner, self._run_owner, "XYE run token")

    def _ensure_cleanup_token(self) -> CleanupToken:
        if self._cleanup_token is None:
            self._cleanup_token = self._coordinator._cleanup_token(
                self._directory,
                "xye",
            )
        return self._cleanup_token

    def stage(self, run_owner: OwnerToken, index: int, value: T) -> None:
        with self._lock:
            self._require_run(run_owner)
            if self._complete or self._attempted is not None:
                raise TransactionStateError(
                    "cannot stage during an XYE publication attempt"
                )
            label = int(index)
            # pop-then-set keeps the previous semantics exactly: re-staging a
            # label moves it to the END of the publication order, as the old
            # filter-and-append did.  Both operations are O(1).
            self._staged.pop(label, None)
            self._staged[label] = value

    def abandon(self, *, run_owner: OwnerToken) -> XyeSnapshot:
        """Discard a mutable run set without touching any durable XYE file."""
        with self._lock:
            self._require_run(run_owner)
            if self._complete:
                return self.snapshot()
            if self._attempted is not None or self._retryable:
                raise TransactionStateError(
                    "cannot abandon an attempted XYE publication; retry it"
                )
            self._staged.clear()
            self._complete = True
            return self.snapshot()

    def snapshot(self) -> XyeSnapshot:
        with self._lock:
            pending = tuple(
                str(path) for path in (self._pending_stale or ())
            )
            staged = (
                self._attempted if self._attempted is not None
                else tuple(self._staged.items())
            )
            return XyeSnapshot(
                directory=self._directory,
                staged_indices=tuple(index for index, _value in staged),
                pending_stale=pending,
                complete=self._complete,
                retryable=self._retryable,
                cleanup_token=self._cleanup_token,
            )

    def _clear_stale(self) -> BaseException | None:
        remaining: list[Path] = []
        first_failure: BaseException | None = None
        for path in self._pending_stale or ():
            try:
                _unlink(path)
            except FileNotFoundError:
                continue
            except BaseException as exc:
                # Contract-bearing: retain every failed stale path.  New XYE
                # must remain withheld until this exact cleanup owner succeeds.
                remaining.append(path)
                if first_failure is None:
                    first_failure = exc
        self._pending_stale = remaining
        return first_failure

    def _publish_after_cleanup(
        self,
        publisher: Callable[[tuple], object],
    ) -> XyeSnapshot:
        failure = self._clear_stale()
        if failure is not None:
            self._retryable = True
            raise CleanupIncomplete(self.snapshot()) from failure
        try:
            if self._attempted is None:
                raise TransactionStateError("XYE attempted set is not frozen")
            publisher(self._attempted)
        except BaseException:
            self._retryable = True
            raise
        self._staged.clear()
        terminal = bool(self._attempted_final)
        self._attempted = None
        self._attempted_final = None
        self._complete = terminal
        self._retryable = False
        return self.snapshot()

    @staticmethod
    def _normalized_stale_paths(directory: str, stale_paths) -> tuple[str, ...]:
        normalized = tuple(
            dict.fromkeys(_normalize_target(path) for path in stale_paths)
        )
        outside = [
            path for path in normalized if os.path.dirname(path) != directory
        ]
        if outside:
            raise TransactionStateError(
                "stale XYE path is outside the normalized run directory: "
                + ", ".join(outside)
            )
        return normalized

    def _publish(
        self,
        *,
        run_owner: OwnerToken,
        stale_paths,
        publisher: Callable[[tuple], object],
        final: bool,
    ) -> XyeSnapshot:
        self._require_run(run_owner)
        if self._complete:
            return self.snapshot()
        if self._attempted is not None or self._retryable:
            raise TransactionStateError("retry the existing XYE publication")

        # A run with no new publication stake must never sweep a prior run's
        # durable tail.  A terminal empty epoch only closes this run owner.
        if not self._staged:
            if final:
                self._complete = True
            return self.snapshot()

        # Stale-tail cleanup belongs only to the first durable epoch.  Later
        # epochs must not reinterpret the run's own already-published files as
        # stale merely because the caller observes the directory again.
        if self._pending_stale is None:
            normalized = self._normalized_stale_paths(
                self._directory, stale_paths)
            self._ensure_cleanup_token()
            self._pending_stale = [Path(path) for path in normalized]

        self._attempted = tuple(self._staged.items())
        self._attempted_final = bool(final)
        return self._publish_after_cleanup(publisher)

    def publish_epoch(
        self,
        *,
        run_owner: OwnerToken,
        stale_paths,
        publisher: Callable[[tuple], object],
    ) -> XyeSnapshot:
        """Publish one retryable nonterminal epoch and retain the run owner."""
        with self._lock:
            return self._publish(
                run_owner=run_owner,
                stale_paths=stale_paths,
                publisher=publisher,
                final=False,
            )

    def publish(
        self,
        *,
        run_owner: OwnerToken,
        stale_paths,
        publisher: Callable[[tuple], object],
    ) -> XyeSnapshot:
        """Publish the terminal epoch and retire this run's mutable set."""
        with self._lock:
            return self._publish(
                run_owner=run_owner,
                stale_paths=stale_paths,
                publisher=publisher,
                final=True,
            )

    def retry_publication(
        self,
        cleanup_token: CleanupToken,
        *,
        publisher: Callable[[tuple], object],
    ) -> XyeSnapshot:
        """Retry only the withheld stale tail/publication, never a success."""
        with self._lock:
            if self._cleanup_token is None:
                raise OwnershipRefused("XYE transaction has no cleanup token")
            _require_identity(cleanup_token, self._cleanup_token, "XYE cleanup token")
            if self._complete:
                return self.snapshot()
            if self._pending_stale is None or self._attempted is None:
                raise TransactionStateError("XYE publication was never prepared")
            return self._publish_after_cleanup(publisher)


__all__ = [
    "CleanupIncomplete",
    "CleanupToken",
    "LeaseOwner",
    "LeaseSnapshot",
    "LeaseUnavailable",
    "OutputTransaction",
    "OutputTransactionCoordinator",
    "OutputTransactionError",
    "OutputReceiptCapability",
    "OutputReceiptCapabilityProvider",
    "OwnerToken",
    "OwnershipRefused",
    "RetryAction",
    "StreamAttempt",
    "StreamCheckpoint",
    "StreamTerminal",
    "TargetAdmission",
    "TargetChanged",
    "TargetLease",
    "TargetSnapshot",
    "capture_target_snapshot",
    "revalidate_target_snapshot",
    "revalidate_stream_terminal",
    "stream_terminal_object_revision",
    "TransactionPhase",
    "TransactionSnapshot",
    "TransactionStateError",
    "XyeOutputTransaction",
    "XyeSnapshot",
    "get_output_transaction_coordinator",
]
