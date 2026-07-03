"""MEM-1b — a consciously-dropped 2D mode must release its heavy cake without
ever being promised as persisted/hydratable.

The leak: a publication-gate drop (e.g. an all-dummy GI 2D cake below the
critical angle) leaves the frame's ``('2d', mode)`` heavy key neither persisted
nor written, so ``_label_heavy_payload_persisted_locked`` can never clear it and
the full cake is pinned in memory forever.  ``mark_dropped`` fixes this by
thinning that mode's payload immediately, while keeping ``is_persisted`` honest.
"""

from __future__ import annotations

import numpy as np

from xrd_tools.core import FrameRecord, FrameView, TwoDKind, axis_from_unit
from xrd_tools.session import FrameRecordStore


def _cake_view(label, *, mode_x="qip_A^-1", mode_y="qoop_A^-1", n=(3, 4)):
    nchi, nq = n
    return FrameView(
        label=label,
        axis_2d_x=axis_from_unit(mode_x, np.linspace(0.1, 0.4, nq)),
        axis_2d_y=axis_from_unit(mode_y, np.linspace(-0.2, 0.2, nchi)),
        intensity_2d=np.arange(nq * nchi, dtype=float).reshape(nchi, nq),
        two_d_kind=TwoDKind.QIP_QOOP,
    )


def _record_1d_and_2d(label=0):
    """A frame with a 1D trace AND a 2D cake (both heavy)."""
    base = FrameView(
        label=label,
        axis_1d=axis_from_unit("q_A^-1", np.linspace(1.0, 2.0, 4)),
        intensity_1d=np.arange(4, dtype=float),
        axis_2d_x=axis_from_unit("qip_A^-1", np.linspace(0.1, 0.4, 4)),
        axis_2d_y=axis_from_unit("qoop_A^-1", np.linspace(-0.2, 0.2, 3)),
        intensity_2d=np.arange(12, dtype=float).reshape(3, 4),
        two_d_kind=TwoDKind.QIP_QOOP,
    )
    return FrameRecord.from_view(base, mode_1d="q_total", mode_2d="qip_qoop")


def test_mark_dropped_releases_only_the_dropped_mode():
    store = FrameRecordStore(max_heavy_items=8)
    store.upsert(_record_1d_and_2d(label=5))

    store.mark_dropped(5, modes=("2d", "qip_qoop"))

    got = store.get(5)
    # the dropped 2D cake is released...
    assert got.results_2d["qip_qoop"].intensity_2d is None
    # ...the mode still EXISTS (a typed read returns ABSENT, not KeyError)...
    assert "qip_qoop" in got.results_2d
    # ...and the 1D trace is untouched.
    assert got.results_1d["q_total"].intensity_1d is not None


def test_dropped_mode_is_never_reported_persisted():
    store = FrameRecordStore(max_heavy_items=8)
    store.upsert(_record_1d_and_2d(label=5))

    # everything EXCEPT the dropped 2D is genuinely on disk
    store.mark_persisted(5, modes=[("1d", "q_total")])
    store.mark_dropped(5, modes=("2d", "qip_qoop"))

    # honest: the frame is NOT fully persisted (the dropped 2D was never written,
    # so hydration must not be promised for it).
    assert store.is_persisted(5) is False


def test_record_with_only_dropped_2d_is_no_longer_heavy():
    """The leak reproducer: a 2D-only frame whose cake is dropped must stop
    being heavy (else it pins forever), yet must not be marked persisted."""
    store = FrameRecordStore(max_heavy_items=8)
    rec = FrameRecord.from_view(_cake_view(7), mode_2d="qip_qoop")
    store.upsert(rec)
    assert store.has_heavy_payload(7) is True

    store.mark_dropped(7, modes=("2d", "qip_qoop"))

    assert store.has_heavy_payload(7) is False   # cake released, not pinned
    assert store.is_persisted(7) is False        # never promised as on-disk


def test_dropped_2d_does_not_pin_the_heavy_tier():
    """With the cap at 1, a dropped-2D frame must not wedge heavy eviction."""
    store = FrameRecordStore(max_heavy_items=1)
    store.upsert(_record_1d_and_2d(label=1))
    # frame 1's 2D is consciously dropped; its 1D is persisted
    store.mark_dropped(1, modes=("2d", "qip_qoop"))
    store.mark_persisted(1, modes=[("1d", "q_total")])
    # a second fully-persisted heavy frame arrives
    store.upsert(_record_1d_and_2d(label=2))
    store.mark_persisted(2)

    # the heavy tier stayed bounded (nothing pinned): at most one heavy record
    heavy = [lbl for lbl in (1, 2) if store.has_heavy_payload(lbl)]
    assert len(heavy) <= 1


def test_mark_dropped_unknown_label_is_noop():
    store = FrameRecordStore(max_heavy_items=8)
    store.mark_dropped(999, modes=("2d", "qip_qoop"))   # must not raise
    assert store.get(999) is None
