"""Multi-scan frame records for grouped Stitch/RSM — the raw-popup enabler.

Grouped scans (5, 7, 8) must not collide on the flat frame index: each is stored
under ``/entry/frames/scan_<N>/frame_NNNN`` and read back by a ``(scan, frame)``
address. Single-scan stays flat (``frame_NNNN``, backward-compatible).
"""
from __future__ import annotations

import numpy as np

from xrd_tools.core.containers import IntegrationResult1D
from xrd_tools.io.nexus import write_rsm, write_stitched
from xrd_tools.io.nexus_record import (
    frame_record_key,
    frame_record_labels,
    iter_frame_record_groups,
)
from xrd_tools.io.read import get_raw_frame
from xrd_tools.rsm.volume import RSMVolume


def test_frame_record_key():
    assert frame_record_key(None, 1) == "frame_0001"
    assert frame_record_key(5, 1) == "scan_5/frame_0001"
    assert frame_record_key(7, 12) == "scan_7/frame_0012"


def _peaked(n: int, where: tuple[int, int]) -> np.ndarray:
    """A small image with one bright pixel — its argmax survives the thumbnail
    quantize/dequantize, so we can prove which record was resolved."""
    img = np.zeros((n, n), dtype=float)
    img[where] = 1000.0
    return img


def test_stitched_multiscan_frames_no_collision(tmp_path):
    import h5py
    s1 = IntegrationResult1D(radial=np.linspace(0.5, 5.0, 10),
                             intensity=np.ones(10), unit="q_A^-1")
    # scan 5 frame 1 → peak top-left; scan 7 frame 1 → peak bottom-right.
    records = [
        {"scan_label": 5, "frame_index": 1, "thumbnail": _peaked(8, (0, 0))},
        {"scan_label": 5, "frame_index": 2, "thumbnail": _peaked(8, (0, 7))},
        {"scan_label": 7, "frame_index": 1, "thumbnail": _peaked(8, (7, 7))},
    ]
    p = tmp_path / "grouped.nxs"
    with h5py.File(p, "w") as f:
        write_stitched(f.create_group("entry"), stitched_1d=s1,
                       frame_records=records)

    img_5_1 = get_raw_frame(p, 1, scan=5)
    img_7_1 = get_raw_frame(p, 1, scan=7)
    # same flat index (1) but different scans → different records (no collision)
    assert np.unravel_index(int(np.argmax(img_5_1)), img_5_1.shape) == (0, 0)
    assert np.unravel_index(int(np.argmax(img_7_1)), img_7_1.shape) == (7, 7)
    # and within a scan, the frame index addresses the right record
    img_5_2 = get_raw_frame(p, 2, scan=5)
    assert np.unravel_index(int(np.argmax(img_5_2)), img_5_2.shape) == (0, 7)


def test_single_scan_stays_flat(tmp_path):
    import h5py
    import pytest
    s1 = IntegrationResult1D(radial=np.linspace(0.5, 5.0, 10),
                             intensity=np.ones(10), unit="q_A^-1")
    records = [{"frame_index": 1, "thumbnail": _peaked(8, (3, 3))}]   # no scan_label
    p = tmp_path / "single.nxs"
    with h5py.File(p, "w") as f:
        write_stitched(f.create_group("entry"), stitched_1d=s1,
                       frame_records=records)
    # flat addressing (scan=None) resolves
    img = get_raw_frame(p, 1)
    assert np.unravel_index(int(np.argmax(img)), img.shape) == (3, 3)
    # asking for a scan that isn't there → KeyError
    with pytest.raises(KeyError):
        get_raw_frame(p, 1, scan=5)


def test_write_rsm_carries_frame_records(tmp_path):
    """write_rsm carries frame records too (RSM has the raw popup as well)."""
    import h5py
    vol = RSMVolume(h=np.linspace(0, 1, 3), k=np.linspace(0, 1, 3),
                    l=np.linspace(0, 1, 3),
                    intensity=np.random.default_rng(0).random((3, 3, 3)))
    records = [
        {"scan_label": 8, "frame_index": 1, "thumbnail": _peaked(8, (2, 2))},
        {"scan_label": 8, "frame_index": 2, "thumbnail": _peaked(8, (5, 5))},
    ]
    p = tmp_path / "rsm_grouped.nxs"
    with h5py.File(p, "w") as f:
        write_rsm(f.create_group("entry"), vol, frame_records=records)
    img = get_raw_frame(p, 2, scan=8)
    assert np.unravel_index(int(np.argmax(img)), img.shape) == (5, 5)


def test_iter_frame_record_groups_descends_flat_and_nested(tmp_path):
    import h5py
    s1 = IntegrationResult1D(radial=np.linspace(0.5, 5.0, 4),
                             intensity=np.ones(4), unit="q_A^-1")
    records = [
        {"frame_index": 1, "thumbnail": _peaked(8, (0, 0))},            # flat
        {"scan_label": 5, "frame_index": 1, "thumbnail": _peaked(8, (1, 1))},
        {"scan_label": 5, "frame_index": 2, "thumbnail": _peaked(8, (2, 2))},
        {"scan_label": 7, "frame_index": 1, "thumbnail": _peaked(8, (3, 3))},
    ]
    p = tmp_path / "mixed.nxs"
    with h5py.File(p, "w") as f:
        write_stitched(f.create_group("entry"), stitched_1d=s1, frame_records=records)

    with h5py.File(p, "r") as f:
        frames = f["entry/frames"]
        # the intermediate scan_<N> containers are NOT yielded — only leaf frames
        got = {(sl, fl) for sl, fl, _grp in iter_frame_record_groups(frames)}
        assert got == {(None, 1), ("5", 1), ("5", 2), ("7", 1)}
        # display labels: flat ints first ("1"), then "<scan>-<frame>"
        labels = frame_record_labels(frames)
        assert [t[0] for t in labels] == ["1", "5-1", "5-2", "7-1"]
        # each label carries its (scan, frame) resolution address
        assert labels[0] == ("1", None, 1)
        assert labels[1] == ("5-1", "5", 1)


def test_frame_record_labels_skips_unloadable(tmp_path):
    """A frame group with neither thumbnail nor source pointer is not a usable
    Frames-panel entry."""
    import h5py
    from xrd_tools.io.nexus_record import ensure_frames_container, write_frame_record
    p = tmp_path / "bare.nxs"
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        frames = ensure_frames_container(e)
        write_frame_record(frames, "frame_0001", thumbnail=_peaked(8, (0, 0)))
        write_frame_record(frames, "frame_0002")   # no thumbnail, no source → skip
        labels = frame_record_labels(e["frames"])
    assert [t[0] for t in labels] == ["1"]


def test_partition_bare_multimember_with_selection_raises():
    """_partition_selected_local_labels: a bare multi-member sequence (no
    global→local _map) WITH a selection is ambiguous — raise rather than silently
    widening to ALL frames.  A composite or single source resolves normally; no
    selection is a no-op.  (No current caller hits the bare-multi case — run_stitch
    wraps groups in a CompositeFrameSource — this pins the fail-loud contract.)"""
    import pytest
    from types import SimpleNamespace
    from xrd_tools.io.nexus_record import _partition_selected_local_labels

    bare = SimpleNamespace()                       # no _map attribute
    members = [SimpleNamespace(), SimpleNamespace()]
    with pytest.raises(ValueError, match="CompositeFrameSource"):
        _partition_selected_local_labels(bare, members, [0, 1])
    # no selection → no restriction
    assert _partition_selected_local_labels(bare, members, None) is None
    # single member → the selection passes through as that member's own labels
    assert _partition_selected_local_labels(
        SimpleNamespace(), [SimpleNamespace()], [3, 4]) == [{3, 4}]
    # composite (carries _map) → resolves global → (member, local)
    comp = SimpleNamespace(_map=[(0, 10), (1, 20)])
    assert _partition_selected_local_labels(comp, members, [0, 1]) == [{10}, {20}]
