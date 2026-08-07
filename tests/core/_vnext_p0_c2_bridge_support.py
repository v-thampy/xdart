"""Test-only public seam helpers for the vNext P0 C2 composition packet.

This module deliberately knows nothing about the H23 transaction, writer,
lease, or physical-target implementation.  It turns immutable E6 directory
observations into H10 identities and drives the public high-level LiveScan
session surface used by the bridge tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np


@dataclass(frozen=True, slots=True)
class SourceFact:
    source_identity: str
    logical_identity: int
    source_revision: int


class DeterministicIntegrator:
    """Small real ``integrate1d`` provider for the Qt-free bridge oracle."""

    detector = None

    def __init__(self, *, all_nan: bool = False) -> None:
        self.all_nan = bool(all_nan)

    def integrate1d(self, image, npt, *, unit="q_A^-1", **_kwargs):
        value = np.nan if self.all_nan else float(np.asarray(image).sum())
        return SimpleNamespace(
            radial=np.linspace(0.0, 1.0, int(npt)),
            intensity=np.full(int(npt), value),
            sigma=None,
            unit=unit,
        )


def observe_source_fact(
    root: Path,
    filename: str,
    *,
    logical_identity: int,
) -> SourceFact:
    """Obtain one bridge key/revision from the public E6 observe call."""
    from xrd_tools.sources.directory_session import DirectoryIndexSession

    session = DirectoryIndexSession(probe_candidates=False)
    try:
        session.configure(root, suffixes=(Path(filename).suffix,))
        observation = session.observe()
        candidate = next(
            value
            for value in observation.discovered_snapshot.candidates
            if value.path.name == filename
        )
        return SourceFact(
            source_identity=str(candidate.path.resolve()),
            logical_identity=int(logical_identity),
            source_revision=max(1, int(candidate.mtime_ns)),
        )
    finally:
        session.close()


def dynamic_api():
    from xrd_tools.session import (
        DynamicAccountingLimits,
        DynamicFrameIdentity,
        DynamicRunAccounting,
        ResultMode,
        StageLedger,
    )

    return SimpleNamespace(
        Limits=DynamicAccountingLimits,
        Key=DynamicFrameIdentity,
        Accounting=DynamicRunAccounting,
        Mode=ResultMode,
        Ledger=StageLedger,
    )


def accounting_for(
    target: Path,
    *,
    generation: int = 1,
    max_groups: int = 4,
    max_attempts: int = 4,
    max_outstanding: int = 8,
):
    api = dynamic_api()
    mode = api.Mode.one_d()
    target_name = f"nexus:{target}"
    ledger = api.Ledger(
        required_modes=(mode,),
        targets_by_mode={mode: (target_name,)},
    )
    accounting = api.Accounting(
        ledger,
        run_generation=int(generation),
        limits=api.Limits(max_groups, max_attempts, max_outstanding),
    )
    return ledger, accounting, mode, target_name


def discover(accounting, fact: SourceFact, *, group, ordinal: int, label: int):
    key = dynamic_api().Key(fact.source_identity, fact.logical_identity)
    accounting.discover(
        key,
        group=group,
        ordinal=int(ordinal),
        output_label=int(label),
    )
    return key


def successful_attempt(accounting, key, source_revision: int, mode):
    attempt = accounting.begin_attempt(key, source_revision=int(source_revision))
    accounting.record_enqueued(attempt)
    accounting.record_accepted(attempt)
    accounting.record_completed(attempt, produced=(mode,))
    accounting.record_written(attempt, modes=(mode,))
    return attempt


def submission_attempt(accounting, key, source_revision: int):
    attempt = accounting.begin_attempt(key, source_revision=int(source_revision))
    accounting.record_enqueued(attempt)
    return attempt


def retryable_attempt(accounting, key, source_revision: int, reason: str):
    attempt = accounting.begin_attempt(key, source_revision=int(source_revision))
    accounting.record_enqueued(attempt)
    accounting.record_failed(attempt, error=str(reason), retryable=True)
    return attempt


def integration_1d(value: float):
    from xrd_tools.core.containers import IntegrationResult1D

    return IntegrationResult1D(
        radial=np.linspace(0.1, 1.0, 8),
        intensity=np.full(8, float(value)),
        sigma=np.full(8, float(value) / 10.0),
        unit="q_A^-1",
    )


def append_intent(
    root: Path,
    *,
    extent: int,
    labels,
    generation: int,
    source_identity: str = "bridge/source",
):
    from xrd_tools.io.append import (
        AppendExternalMember,
        AppendIntent,
        AppendSource,
    )

    member = AppendExternalMember(
        path=str(root / "source-member.h5"),
        dataset_path="/entry/data/data",
        size=100 + int(extent),
        mtime_ns=1_000 + int(extent),
        source_start=0,
        source_stop=int(extent),
        ordinal=0,
    )
    source = AppendSource(
        path=str(root / "source-master.h5"),
        adapter_id="nexus_hdf5",
        size=200 + int(extent),
        mtime_ns=2_000 + int(extent),
        extent=int(extent),
        external_members=(member,),
        generation=int(generation),
    )
    return AppendIntent(
        entry="entry",
        source_base=str(root),
        source_identity=str(source_identity),
        science_fingerprint="c2-bridge-science-v1",
        modes=("1d:default",),
        source=source,
        labels=tuple(int(value) for value in labels),
    )


class LiveFrames:
    def __init__(self, labels=()):
        self._values = {}
        self._persisted = set()
        for label in labels:
            self.add(label)

    @property
    def index(self):
        return sorted(self._values)

    def __getitem__(self, label):
        return self._values[int(label)]

    def add(self, label, *, value=None):
        label = int(label)
        self._values[label] = SimpleNamespace(
            idx=label,
            int_1d=integration_1d(label if value is None else value),
            int_2d=None,
            gi_1d={},
            gi_2d={},
            scan_info={},
            map_raw=None,
            bg_raw=None,
            source_file="",
            source_frame_idx=0,
            mask=None,
            poni=None,
            thumbnail=None,
        )

    def mark_persisted(self, labels):
        self._persisted.update(int(label) for label in labels)


def live_scan(target: Path, intent, labels=()):
    return SimpleNamespace(
        name="c2-bridge",
        data_file=str(target),
        source_base=target.parent,
        file_lock=threading.RLock(),
        frames=LiveFrames(labels),
        skip_2d=True,
        bai_1d_args={},
        bai_2d_args={},
        gi=False,
        gi_config={},
        scan_data=None,
        geometry=None,
        global_mask=None,
        detector_shape=None,
        mg_args={"wavelength": 1e-10},
        _same_run_intent=intent,
    )


def open_session(live, accounting, *, replace=True):
    from xdart.modules.reduction import open_live_scan_nexus_session

    return open_live_scan_nexus_session(
        live,
        replace=bool(replace),
        accounting=accounting.writer_boundary,
    )
