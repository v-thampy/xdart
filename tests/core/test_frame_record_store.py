"""Headless :class:`FrameRecordStore` tests.

These are the Phase-B foundation tests: no Qt, no xdart, and no live display
flip.  They lock the store invariants before xdart projects onto it.
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

from xrd_tools.core import (
    Axis,
    FrameRecord,
    FrameView,
    TwoDKind,
    assert_framerecord_equivalent,
    axis_from_unit,
)
from xrd_tools.session import FrameHydrationResult, FrameRecordStore

@pytest.mark.parametrize("method", ("clear", "clear_checkpoint_recoverable"))
def test_checkpoint_clear_revokes_outside_record_lock(monkeypatch, method):
    store = FrameRecordStore(); gate_type = type(store._checkpoint_hydration)
    original = gate_type.revoke
    def checked(gate):
        assert not store._lock._is_owned(); original(gate)
    monkeypatch.setattr(gate_type, "revoke", checked)
    getattr(store, method)()


def _set_certified_hydrator(store, hydrate):
    def certified(request):
        record = hydrate(request.label)
        return None if record is None else FrameHydrationResult(request, record)

    store.set_hydrator(certified)


def _view(label=0, *, source: "str | None" = "/data/scan_0001.tif",
          source_frame: "int | None" = 0, scale=1.0):
    return FrameView(
        label=label,
        axis_1d=Axis("Q", "q_A^-1", values=np.array([1.0, 2.0, 3.0])),
        intensity_1d=np.array([10.0, 20.0, 30.0]) * scale,
        metadata_raw={"i0": 1.0, "sample": "A"},
        source_path=source,
        source_frame_index=source_frame,
    )


def _record(
    label=0,
    *,
    mode="q_total",
    source: "str | None" = "/data/scan_0001.tif",
    source_frame: "int | None" = 0,
    scale=1.0,
):
    return FrameRecord.from_view(
        _view(label, source=source, source_frame=source_frame, scale=scale),
        mode_1d=mode,
    )


def _multi_mode_record(label=0, *, scale=1.0):
    nq = 4
    nchi = 3
    base = FrameView(
        label=label,
        axis_1d=axis_from_unit("q_A^-1", np.linspace(1.0, 2.0, nq)),
        intensity_1d=np.arange(nq, dtype=float) * scale + label,
        axis_2d_x=axis_from_unit("qip_A^-1", np.linspace(0.1, 0.4, nq)),
        axis_2d_y=axis_from_unit("qoop_A^-1", np.linspace(-0.2, 0.2, nchi)),
        intensity_2d=(
            np.arange(nq * nchi, dtype=float).reshape(nchi, nq) * scale + label
        ),
        two_d_kind=TwoDKind.QIP_QOOP,
    )
    rec = FrameRecord.from_view(base, mode_1d="q_total", mode_2d="qip_qoop")
    rec = rec.with_result_1d(
        "q_ip",
        FrameView(
            label=label,
            axis_1d=axis_from_unit("qip_A^-1", np.linspace(0.1, 0.4, nq + 1)),
            intensity_1d=np.arange(nq + 1, dtype=float) * scale + 10 + label,
        ),
        make_active=False,
    )
    rec = rec.with_result_2d(
        "q_chi",
        FrameView(
            label=label,
            axis_2d_x=axis_from_unit("q_A^-1", np.linspace(1.0, 2.0, nq)),
            axis_2d_y=axis_from_unit("chi_deg", np.linspace(-45.0, 45.0, nchi + 1)),
            intensity_2d=(
                np.arange(nq * (nchi + 1), dtype=float).reshape(nchi + 1, nq)
                * scale + 20 + label
            ),
            two_d_kind=TwoDKind.Q_CHI,
        ),
        make_active=False,
    )
    return rec


def test_store_accumulates_modes_for_same_source():
    store = FrameRecordStore(max_heavy_items=None)
    store.upsert(_record(mode="q_total"))
    store.upsert(_record(mode="q_ip", scale=2.0))

    rec = store.get(0)
    assert rec is not None
    assert set(rec.modes_1d) == {"q_total", "q_ip"}
    np.testing.assert_allclose(rec.view_1d("q_ip").intensity_1d, [20.0, 40.0, 60.0])


def test_store_replaces_instead_of_merging_conflicting_sources():
    store = FrameRecordStore(max_heavy_items=None)
    store.upsert(_record(mode="q_total", source="/run/a/frame_0001.tif"))
    store.upsert(_record(mode="q_ip", source="/run/b/frame_0001.tif", scale=3.0))

    rec = store.get(0)
    assert rec is not None
    assert rec.modes_1d == ("q_ip",)
    assert store.source_identity(0) == "/run/b/frame_0001.tif#0"


def test_store_replaces_known_source_with_missing_source_instead_of_splicing():
    store = FrameRecordStore(max_heavy_items=None)
    store.upsert(_record(mode="q_total", source="/run/a/frame_0001.tif"))
    store.upsert(_record(mode="q_ip", source=None, source_frame=None, scale=2.0))

    rec = store.get(0)
    assert rec is not None
    assert rec.modes_1d == ("q_ip",)
    assert store.source_identity(0) == ""


def test_heavy_eviction_waits_until_frame_is_persisted():
    store = FrameRecordStore(max_heavy_items=1)
    store.upsert(_record(label=1))
    store.upsert(_record(label=2, source="/data/scan_0002.tif"))

    assert store.has_heavy_payload(1)
    assert store.has_heavy_payload(2)

    store.mark_persisted(1)

    assert not store.has_heavy_payload(1)
    assert store.has_heavy_payload(2)
    assert store.get(1).view_1d("q_total").intensity_1d is None
    assert store.get(1).view_1d("q_total").metadata_raw["sample"] == "A"


def test_new_unsaved_mode_does_not_inherit_label_persistence():
    store = FrameRecordStore(max_heavy_items=1)
    store.upsert(_record(label=1, mode="q_total"), persisted=True)
    assert store.is_persisted(1)

    store.upsert(_record(label=1, mode="q_ip", scale=2.0), persisted=False)
    assert not store.is_persisted(1)

    store.upsert(_record(label=2, source="/data/scan_0002.tif"), persisted=True)

    assert store.has_heavy_payload(1)
    assert not store.has_heavy_payload(2)
    rec = store.get(1)
    assert rec is not None
    assert set(rec.modes_1d) == {"q_total", "q_ip"}
    np.testing.assert_allclose(rec.view_1d("q_ip").intensity_1d, [20.0, 40.0, 60.0])


def test_mark_persisted_marks_all_current_modes_for_eviction():
    store = FrameRecordStore(max_heavy_items=1)
    store.upsert(_record(label=1, mode="q_total"), persisted=True)
    store.upsert(_record(label=1, mode="q_ip", scale=2.0), persisted=False)

    store.mark_persisted(1)
    assert store.is_persisted(1)

    store.upsert(_record(label=2, source="/data/scan_0002.tif"), persisted=True)

    assert not store.has_heavy_payload(1)
    assert store.has_heavy_payload(2)
    rec = store.get(1)
    assert rec is not None
    assert rec.view_1d("q_total").intensity_1d is None
    assert rec.view_1d("q_ip").intensity_1d is None


def test_mark_persisted_with_modes_does_not_mark_unwritten_modes():
    store = FrameRecordStore(max_heavy_items=1)
    store.upsert(_record(label=1, mode="q_total"), persisted=True)
    store.upsert(_record(label=1, mode="q_ip", scale=2.0), persisted=False)

    store.mark_persisted(1, modes=[("1d", "q_total")])
    assert not store.is_persisted(1)

    store.upsert(_record(label=2, source="/data/scan_0002.tif"), persisted=True)

    assert store.has_heavy_payload(1)
    assert not store.has_heavy_payload(2)
    rec = store.get(1)
    assert rec is not None
    np.testing.assert_allclose(rec.view_1d("q_ip").intensity_1d, [20.0, 40.0, 60.0])


def test_get_or_hydrate_restores_thinned_record():
    store = FrameRecordStore(max_heavy_items=1)
    store.upsert(_record(label=1), persisted=True)
    store.upsert(_record(label=2, source="/data/scan_0002.tif"), persisted=True)
    assert not store.has_heavy_payload(1)
    assert store.has_heavy_payload(2)

    calls = []

    def hydrate(label):
        calls.append(label)
        return _record(label=label, scale=5.0)

    _set_certified_hydrator(store, hydrate)
    rec = store.get_or_hydrate(1)

    assert calls == [1]
    assert store.has_heavy_payload(1)
    assert not store.has_heavy_payload(2)
    np.testing.assert_allclose(rec.view_1d("q_total").intensity_1d, [50.0, 100.0, 150.0])


def test_get_or_hydrate_restores_every_written_mode_from_nexus(tmp_path):
    import h5py

    from xrd_tools.io import read_frame_record, write_frame_records
    from xrd_tools.io.schema import (
        PROCESSED_SCHEMA_NAME,
        PROCESSED_SCHEMA_VERSION,
        SCHEMA_NAME_ATTR,
        SCHEMA_VERSION_ATTR,
    )

    records = [
        _multi_mode_record(0, scale=1.0),
        _multi_mode_record(1, scale=2.0),
    ]
    path = tmp_path / "multi_mode.nexus"
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs[SCHEMA_NAME_ATTR] = PROCESSED_SCHEMA_NAME
        entry.attrs[SCHEMA_VERSION_ATTR] = PROCESSED_SCHEMA_VERSION
        write_frame_records(entry, records)

    store = FrameRecordStore(max_heavy_items=1)
    store.upsert(records[0], persisted=True)
    store.upsert(records[1], persisted=True)
    assert not store.has_heavy_payload(0)

    _set_certified_hydrator(
        store, lambda label: read_frame_record(path, int(label)),
    )
    hydrated = store.get_or_hydrate(0)

    assert hydrated is not None
    assert store.has_heavy_payload(0)
    assert store.is_persisted(0)
    assert set(hydrated.modes_1d) == {"q_total", "q_ip"}
    assert set(hydrated.modes_2d) == {"qip_qoop", "q_chi"}
    assert_framerecord_equivalent(records[0], hydrated)


def test_get_or_hydrate_refuses_source_identity_when_captured_row_had_none():
    store = FrameRecordStore(max_heavy_items=0, require_persisted_for_eviction=False)
    store.upsert(_record(label=1, source=None, source_frame=None))
    assert store.source_identity(1) == ""
    assert not store.has_heavy_payload(1)

    def hydrate(label):
        return _record(label=label, source="/data/scan_0001.tif", source_frame=12)

    _set_certified_hydrator(store, hydrate)
    rec = store.get_or_hydrate(1)

    assert rec is not None
    assert store.source_identity(1) == ""
    assert not store.has_heavy_payload(1)


def test_get_or_hydrate_refuses_uncertified_raw_record_without_source():
    store = FrameRecordStore(
        max_heavy_items=0,
        require_persisted_for_eviction=False,
    )
    installed = store.upsert(
        _record(label=1, source=None, source_frame=None)
    )
    assert not store.has_heavy_payload(1)

    store.set_hydrator(
        lambda _request: _record(
            label=1,
            source=None,
            source_frame=None,
            scale=9.0,
        )
    )
    returned = store.get_or_hydrate(1)

    assert returned is installed
    assert returned is store.get(1)
    assert not store.has_heavy_payload(1)


def test_hydrator_none_returns_locked_current_replacement():
    store = FrameRecordStore(max_heavy_items=1)
    store.upsert(_record(label=1, source="/data/a.tif"), persisted=True)
    assert store.release_heavy(1)
    replacements = []

    def replace_then_refuse(_request):
        replacements.append(
            store.upsert(
                _record(label=1, source="/data/b.tif", scale=7.0),
                persisted=True,
            )
        )
        return None

    store.set_hydrator(replace_then_refuse)
    returned = store.get_or_hydrate(1)

    assert len(replacements) == 1
    assert returned is replacements[0]
    assert returned is store.get(1)
    assert store.source_identity(1) == "/data/b.tif#0"


def test_get_or_hydrate_refuses_conflicting_source_without_replacement():
    store = FrameRecordStore(max_heavy_items=0, require_persisted_for_eviction=False)
    store.upsert(_record(label=1, mode="q_total", source="/data/a.tif"))
    assert store.source_identity(1) == "/data/a.tif#0"

    def hydrate(label):
        return _record(label=label, mode="q_ip", source="/data/b.tif", scale=4.0)

    _set_certified_hydrator(store, hydrate)
    rec = store.get_or_hydrate(1)

    assert rec is not None
    assert rec.modes_1d == ("q_total",)
    assert store.source_identity(1) == "/data/a.tif#0"


@pytest.mark.parametrize("authority_mutation", ("label", "source", "revision", "generation"))
def test_get_or_hydrate_requires_exact_returned_revision_authority(
    authority_mutation,
):
    from xrd_tools.session import FrameHydrationResult

    store = FrameRecordStore(max_heavy_items=1)
    store.upsert(_record(label=1, source="/data/a.tif"), persisted=True)
    assert store.release_heavy(1)

    def hydrate(request):
        changes = {
            "label": {"label": 2},
            "source": {"source_identity": "/data/b.tif#0"},
            "revision": {"revision": request.revision + 1},
            "generation": {"generation": request.generation + 1},
        }[authority_mutation]
        foreign = replace(request, **changes)
        return FrameHydrationResult(
            foreign, _record(label=1, source="/data/a.tif", scale=9.0),
        )

    store.set_hydrator(hydrate)
    returned = store.get_or_hydrate(1)
    assert returned is store.get(1)
    assert not store.has_heavy_payload(1)


def test_get_or_hydrate_requires_the_exact_request_object_not_equal_replay():
    store = FrameRecordStore(max_heavy_items=1)
    store.upsert(_record(label=1, source="/data/a.tif"), persisted=True)
    assert store.release_heavy(1)

    def hydrate(request):
        return FrameHydrationResult(
            replace(request),
            _record(label=1, source="/data/a.tif", scale=9.0),
        )

    store.set_hydrator(hydrate)
    returned = store.get_or_hydrate(1)
    assert returned is store.get(1)
    assert not store.has_heavy_payload(1)


def test_certified_source_less_result_cannot_inherit_qualified_source():
    store = FrameRecordStore(max_heavy_items=1)
    store.upsert(_record(label=1, source="/data/a.tif"), persisted=True)
    assert store.release_heavy(1)

    def hydrate(request):
        return FrameHydrationResult(
            request,
            _record(label=1, source=None, source_frame=None, scale=9.0),
        )

    store.set_hydrator(hydrate)
    returned = store.get_or_hydrate(1)
    assert returned is store.get(1)
    assert store.source_identity(1) == "/data/a.tif#0"
    assert not store.has_heavy_payload(1)


def test_hydration_rechecks_projected_membership_not_only_projection_sets():
    store = FrameRecordStore(max_heavy_items=1)
    store.upsert(_record(label=1, source="/data/a.tif"), persisted=True)
    assert store.release_heavy(1)
    mode = ("1d", "q_total")

    def hydrate(request):
        store.replace_projection(1, hydratable=(mode,))
        return FrameHydrationResult(
            request,
            _record(label=1, source="/data/a.tif", scale=9.0),
        )

    store.set_hydrator(hydrate)
    returned = store.get_or_hydrate(1)
    assert returned is store.get(1)
    assert not store.has_heavy_payload(1)


def test_release_record_is_durability_qualified_and_preserves_revision_fence():
    store = FrameRecordStore(max_items=None, max_heavy_items=None)
    mode = ("1d", "q_total")
    record = _record(label=1, mode=mode[1], source="/data/a.tif")
    store.upsert(record)
    initial_revision = store._revisions[1]

    assert store.release_record(1) is False
    assert store.get(1) is record
    assert store._revisions[1] == initial_revision

    store.replace_projection(1, hydratable=(mode,))
    assert store.release_record(1) is False
    assert store.get(1) is record

    store.replace_projection(1, hydratable=(mode,), durable=(mode,))
    assert store.can_release_record(1) is True
    assert store.release_record(1) is True
    assert store.get(1) is None
    assert store.labels() == ()
    assert store._revisions[1] == initial_revision
    assert store.release_record(1) is False


def test_exchange_releasable_record_is_atomic_identity_qualified_cas():
    store = FrameRecordStore(max_items=None, max_heavy_items=None)
    prior = _record(label=1, mode="prior", source="/data/a.tif")
    candidate = _record(label=1, mode="candidate", source="/data/a.tif")
    store.upsert(prior, source_identity="/data/a.tif#0")

    assert store.exchange_releasable_record(
        1,
        expected=prior,
        replacement=candidate,
        source_identity="/data/a.tif#0",
        persisted=True,
    ) is False
    assert store.get(1) is prior

    store.mark_persisted(1)
    revision = store._revisions[1]
    assert store.exchange_releasable_record(
        1,
        expected=prior,
        replacement=candidate,
        source_identity="/data/a.tif#0",
        persisted=True,
    ) is True
    assert store.get(1) is candidate
    assert store.is_persisted(1)
    assert store._revisions[1] == revision + 1
    assert store.exchange_releasable_record(
        1,
        expected=prior,
        replacement=None,
    ) is False
    assert store.get(1) is candidate


def test_exchange_releasable_record_refuses_bounds_before_mutation():
    store = FrameRecordStore(
        max_items=1,
        max_heavy_items=0,
        require_persisted_for_eviction=False,
    )
    installed = store.upsert(
        _record(label=1, mode="prior", source="/data/a.tif"),
        source_identity="/data/a.tif#0",
        persisted=True,
    )
    revision = store._revisions[1]
    source = store.source_identity(1)
    persisted = store.persisted_modes(1)
    heavy_candidate = _record(
        label=1,
        mode="candidate",
        source="/data/a.tif",
    )

    assert store.exchange_releasable_record(
        1,
        expected=installed,
        replacement=heavy_candidate,
        source_identity="/data/a.tif#0",
        persisted=True,
    ) is False
    assert store.get(1) is installed
    assert store._revisions[1] == revision
    assert store.source_identity(1) == source
    assert store.persisted_modes(1) == persisted

    assert store.exchange_releasable_record(
        2,
        expected=None,
        replacement=_record(label=2, source="/data/b.tif"),
        source_identity="/data/b.tif#0",
        persisted=True,
    ) is False
    assert store.labels() == (1,)


def test_hydration_request_cannot_replay_across_commit_epoch_aba():
    store = FrameRecordStore(max_heavy_items=1)
    store.upsert(_record(label=1, source="/data/a.tif"), persisted=True)
    assert store.release_heavy(1)
    captured = []

    def capture(request):
        captured.append(request)
        return None

    class Gate:
        def enter(self, _epoch):
            return True

        def leave(self):
            pass

    gate = Gate()
    store.set_hydrator(capture)
    store.get_or_hydrate(1, commit_gate=gate, commit_epoch=1)
    assert len(captured) == 1
    store.set_hydrator(
        lambda _request: FrameHydrationResult(
            captured[0], _record(label=1, source="/data/a.tif", scale=9.0),
        ),
    )
    returned = store.get_or_hydrate(1, commit_gate=gate, commit_epoch=2)
    assert returned is store.get(1)
    assert not store.has_heavy_payload(1)


def test_snapshot_is_read_only_copy():
    store = FrameRecordStore(max_heavy_items=None)
    store.upsert(_record(label=1))
    snap = store.snapshot()

    try:
        snap[2] = _record(label=2)  # type: ignore[index]
    except TypeError:
        pass
    assert store.labels() == (1,)


def test_get_or_hydrate_does_not_persist_extra_unsaved_mode():
    # P2 regression: a hydrator that returns the persisted disk mode PLUS an
    # extra freshly-computed (unsaved) mode must NOT mark the extra mode
    # persisted — else it could be heavy-evicted before it is written (the
    # 748fcac persist-before-evict bug, re-introduced via get_or_hydrate).
    store = FrameRecordStore(max_heavy_items=1)
    store.upsert(_record(label=1, mode="q_total"), persisted=True)
    store.upsert(_record(label=2, source="/data/scan_0002.tif"), persisted=True)
    assert not store.has_heavy_payload(1)            # thinned (fully persisted)

    def hydrate(label):
        rec = FrameRecord.from_view(_view(label), mode_1d="q_total")  # on-disk mode
        return rec.with_result_1d("q_ip", _view(label, scale=2.0))   # extra unsaved

    _set_certified_hydrator(store, hydrate)
    rec = store.get_or_hydrate(1)
    assert set(rec.modes_1d) == {"q_total", "q_ip"}
    assert not store.is_persisted(1)                 # q_ip unsaved -> not fully persisted

    # Heavy pressure must thin the fully-persisted frame, NOT label 1 (q_ip unsaved).
    store.upsert(_record(label=3, source="/data/scan_0003.tif"), persisted=True)
    assert store.has_heavy_payload(1)
    assert store.get(1).view_1d("q_ip").intensity_1d is not None


def test_max_items_evicts_persisted_records_not_unpersisted():
    # max_items full-record eviction (audit gap): evicts a PERSISTED record, never
    # an unpersisted one (persist-before-evict also gates whole-record eviction).
    store = FrameRecordStore(max_items=2, max_heavy_items=None)
    store.upsert(_record(label=1, source="/d/1.tif"), persisted=True)
    store.upsert(_record(label=2, source="/d/2.tif"), persisted=False)
    store.upsert(_record(label=3, source="/d/3.tif"), persisted=True)
    assert len(store) == 2
    assert set(store.labels()) == {2, 3}            # persisted 1 evicted; unpersisted 2 kept

    # All-unpersisted past the cap: nothing is evictable, so the store keeps both.
    store2 = FrameRecordStore(max_items=1, max_heavy_items=None)
    store2.upsert(_record(label=1, source="/d/1.tif"), persisted=False)
    store2.upsert(_record(label=2, source="/d/2.tif"), persisted=False)
    assert len(store2) == 2


def test_concurrent_upsert_mark_hydrate_is_thread_safe():
    # The RLock is the store's headline safety feature; exercise it under
    # contention (upsert / mark_persisted / get_or_hydrate / snapshot from many
    # threads).  Assert no exception, no deadlock, and the bounds hold.
    import threading

    store = FrameRecordStore(max_heavy_items=8, max_items=20)
    _set_certified_hydrator(
        store, lambda label: _record(label=label, scale=3.0),
    )
    errors: list[BaseException] = []

    def worker(base):
        try:
            for i in range(base, base + 80):
                lbl = i % 20
                store.upsert(_record(label=lbl, source=f"/d/{lbl}.tif"),
                             persisted=(i % 2 == 0))
                store.mark_persisted(lbl)
                store.get_or_hydrate(lbl)
                store.snapshot()
        except BaseException as exc:                 # pragma: no cover - diagnostic
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(b,))
               for b in (0, 100, 200, 300)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)

    assert not errors, errors
    assert not any(t.is_alive() for t in threads)   # no deadlock
    assert len(store) <= 20                          # max_items bound held


# --------------------------------------------------------------------------- #
# A-prep2: freeze the config the LIVE store (D3 one-store collapse) will use.
#
# The bounded acquisition-store fixture pins the 64-item compatibility ceiling
# and the persist-before-evict invariant. Production derives its granted bound
# from the session allocation; this direct store contract remains xdart-free.
# --------------------------------------------------------------------------- #

LIVE_STORE_HEAVY_CAP = 64
"""Compatibility ceiling for the direct acquisition-store contract."""


def _live_store() -> FrameRecordStore:
    """The exact config the live FrameRecordStore path will use."""
    return FrameRecordStore(
        max_heavy_items=LIVE_STORE_HEAVY_CAP,
        require_persisted_for_eviction=True,
    )


def test_live_store_config_evicts_only_persisted_under_heavy_pressure():
    # (a) With max_heavy_items=64 and require-persisted eviction, pushing past
    # the cap thins ONLY persisted records; unpersisted heavy frames survive.
    store = _live_store()

    # Fill the cap with persisted frames (eligible for eviction)...
    for i in range(LIVE_STORE_HEAVY_CAP):
        store.upsert(_record(label=i, source=f"/d/{i}.tif"), persisted=True)
    # ...then add one MORE persisted frame: exactly one persisted frame is thinned.
    store.upsert(
        _record(label=LIVE_STORE_HEAVY_CAP, source="/d/cap.tif"), persisted=True
    )

    thinned = [
        i
        for i in range(LIVE_STORE_HEAVY_CAP + 1)
        if not store.has_heavy_payload(i)
    ]
    assert len(thinned) == 1                         # one over the cap -> one thinned
    # The thinned frame keeps its labels/axes/metadata; only arrays are dropped.
    rec = store.get(thinned[0])
    assert rec is not None
    assert rec.view_1d("q_total").intensity_1d is None
    assert rec.view_1d("q_total").metadata_raw["sample"] == "A"


def test_live_store_never_evicts_an_unsaved_extra_mode_under_heavy_pressure():
    # (b) persist-before-evict with an EXTRA UNSAVED record present (simulating a
    # freshly-computed GI sub-mode not yet on disk): the unpersisted frame is
    # NEVER thinned, even when the store is at/over the heavy cap.
    store = _live_store()

    # One frame carries a persisted primary mode PLUS an unsaved extra GI mode.
    store.upsert(_record(label=0, mode="q_total", source="/d/0.tif"), persisted=True)
    store.upsert(_record(label=0, mode="q_ip", source="/d/0.tif", scale=2.0),
                 persisted=False)
    assert not store.is_persisted(0)                 # extra mode unsaved

    # Flood the rest of the cap with fully-persisted frames, then overflow.
    for i in range(1, LIVE_STORE_HEAVY_CAP + 4):
        store.upsert(_record(label=i, source=f"/d/{i}.tif"), persisted=True)

    # The unsaved frame must still hold its heavy arrays (never evicted).
    assert store.has_heavy_payload(0)
    rec = store.get(0)
    assert rec is not None
    assert rec.view_1d("q_ip").intensity_1d is not None
    # Eviction happened among the persisted frames instead.
    persisted_thinned = [
        i
        for i in range(1, LIVE_STORE_HEAVY_CAP + 4)
        if not store.has_heavy_payload(i)
    ]
    assert persisted_thinned                         # some persisted frames thinned


def test_live_store_all_unpersisted_overflow_keeps_everything():
    # Corollary of (b): if EVERY heavy frame is unpersisted, nothing is
    # evictable, so the store exceeds the cap rather than dropping unsaved data.
    store = _live_store()
    for i in range(LIVE_STORE_HEAVY_CAP + 5):
        store.upsert(_record(label=i, source=f"/d/{i}.tif"), persisted=False)
    assert all(
        store.has_heavy_payload(i) for i in range(LIVE_STORE_HEAVY_CAP + 5)
    )


def test_live_store_mark_persisted_only_marks_passed_labels():
    # (c) mark_persisted marks ONLY the labels passed (mirrors "mark from
    # flush() only"): an unmentioned frame stays unpersisted and un-evictable.
    store = _live_store()
    store.upsert(_record(label=1, source="/d/1.tif"), persisted=False)
    store.upsert(_record(label=2, source="/d/2.tif"), persisted=False)
    assert not store.is_persisted(1)
    assert not store.is_persisted(2)

    store.mark_persisted([1])                         # only label 1

    assert store.is_persisted(1)
    assert not store.is_persisted(2)                  # 2 untouched

    # And it only marks the modes that exist on the passed label: marking a
    # missing label is a no-op (does not raise, marks nothing).
    store.mark_persisted([999])
    assert not store.is_persisted(999)


def test_live_store_mark_persisted_marks_all_current_modes_of_passed_label():
    # mark_persisted marks EVERY current mode of the passed label (so a frame
    # with a primary + extra GI mode becomes fully persisted, hence evictable).
    store = _live_store()
    store.upsert(_record(label=1, mode="q_total", source="/d/1.tif"), persisted=True)
    store.upsert(_record(label=1, mode="q_ip", source="/d/1.tif", scale=2.0),
                 persisted=False)
    assert not store.is_persisted(1)                  # q_ip unsaved

    store.mark_persisted(1)                            # flush wrote both modes
    assert store.is_persisted(1)
    rec = store.get(1)
    assert set(rec.modes_1d) == {"q_total", "q_ip"}


def test_live_store_set_hydrator_rehydrates_an_evicted_record_on_access():
    # (d) set_hydrator re-hydrates a thinned (evicted-arrays) record on access:
    # get_or_hydrate calls the hydrator exactly once and restores heavy arrays.
    store = _live_store()
    # Fill + overflow with persisted frames so the oldest is thinned.
    for i in range(LIVE_STORE_HEAVY_CAP + 1):
        store.upsert(_record(label=i, source=f"/d/{i}.tif"), persisted=True)

    thinned = next(
        i for i in range(LIVE_STORE_HEAVY_CAP + 1) if not store.has_heavy_payload(i)
    )

    calls: list[int] = []

    def hydrate(label):
        calls.append(label)
        return _record(label=label, source=f"/d/{label}.tif", scale=5.0)

    _set_certified_hydrator(store, hydrate)
    rec = store.get_or_hydrate(thinned)

    assert calls == [thinned]                          # hydrator called once
    assert store.has_heavy_payload(thinned)            # arrays restored
    assert rec is not None
    np.testing.assert_allclose(
        rec.view_1d("q_total").intensity_1d, [50.0, 100.0, 150.0]
    )

    # A frame whose arrays are still resident is returned WITHOUT re-hydrating.
    resident = next(
        i for i in range(LIVE_STORE_HEAVY_CAP + 1) if store.has_heavy_payload(i)
        and i != thinned
    )
    calls.clear()
    store.get_or_hydrate(resident)
    assert calls == []                                 # no hydrator call for resident



def test_projected_heavy_release_needs_durable_or_checkpoint_recoverable():
    """A live projection must not pin every heavy payload for the whole run.

    ``_releasable_modes_locked`` answers a projected label with the DURABLE set
    alone, and a fast-regenerable Overwrite publishes no durability receipt until
    close.  Before the writer also published checkpoint recoverability, that
    combination licensed no release at all and the display retained every frame's
    2-D array -- about 7 GB on a 3621-frame Run against a 64-frame cap.  Drive the
    release path production uses (``display_runtime`` registers
    ``can_release_heavy`` as the heavy-evictable probe and ``display_residency``
    calls ``release_heavy``), not the automatic cap enforcer, which consults the
    stricter predicate and would pass while the real leak stayed open.
    """
    from xrd_tools.core import Axis, FrameRecord, FrameView
    from xrd_tools.session import FrameRecordStore

    keys = [("1d", "default"), ("2d", "cake")]

    def project(store, label, *, durable, recoverable):
        view = FrameView(
            label=label,
            axis_1d=Axis("Q", "q_A^-1", values=np.arange(4.0)),
            intensity_1d=np.full(4, float(label)),
            axis_2d_x=Axis("Q", "q_A^-1", values=np.arange(4.0)),
            axis_2d_y=Axis("chi", "chi_deg", values=np.arange(3.0)),
            intensity_2d=np.full((3, 4), float(label)),
            mask_baked=True,
        )
        store.upsert(
            FrameRecord(
                label=label,
                results_1d={"default": view},
                results_2d={"cake": view},
                active_mode_1d="default",
                active_mode_2d="cake",
            ),
            source_identity="src",
        )
        store.replace_projection(
            label, hydratable=keys, durable=keys if durable else [],
        )
        if recoverable:
            resident = store.get(label)
            revisions = {key: 1 for key in keys}
            assert store._bind_checkpoint_revisions(
                label, expected=resident, revisions=revisions,
            )
            assert store._mark_checkpoint_recoverable(
                label, expected=resident, revisions=revisions,
                frame_verified=True, thumbnail_verified=False,
            )

    # Durability alone licenses release, as it always did.
    durable_store = FrameRecordStore(max_heavy_items=2)
    project(durable_store, 1, durable=True, recoverable=False)
    assert durable_store.can_release_heavy(1)

    # The fast path's projection on its own licenses nothing: this is the leak.
    leaking = FrameRecordStore(max_heavy_items=2)
    project(leaking, 1, durable=False, recoverable=False)
    assert not leaking.can_release_heavy(1)
    assert not leaking.release_heavy(1)
    assert leaking.has_heavy_payload(1)

    # Checkpoint recoverability licenses the release without asserting durable.
    recovered = FrameRecordStore(max_heavy_items=2)
    project(recovered, 1, durable=False, recoverable=True)
    assert recovered.durable_modes(1) == frozenset()
    assert recovered.can_release_heavy(1)
    assert recovered.release_heavy(1)
    assert not recovered.has_heavy_payload(1)
    # Thinning keeps the label resident and still recorded on disk, never
    # forgets it.  (hydratable_modes() would need a registered hydrator, which a
    # bare store has none of; persisted_modes is the on-disk fact itself.)
    assert recovered.get(1) is not None
    assert recovered.persisted_modes(1) == frozenset(keys)
