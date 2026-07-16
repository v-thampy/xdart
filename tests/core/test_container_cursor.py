# -*- coding: utf-8 -*-
"""R2 Commit 2 — ContainerCursor lifecycle, single-open, and native reads.

Proves one open handle supplies descriptor + wavelength + metadata + frame
count + all reads, that close is idempotent on every path (normal, exception,
generator abandonment, Stop), that reads after close fail clearly, that a 2-D
dataset exposes only frame 0, that block frames share memory with their owner
block, and that a stale R1 candidate is rejected BEFORE the file is opened.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from xrd_tools.sources.cursor import ContainerCursor, CursorClosedError, ReadBlock
from xrd_tools.sources.discover import Candidate
from xrd_tools.sources.probe import ProbeState


def _stack(path, *, shape=(8, 16, 16), chunks=(2, 16, 16), dtype=np.uint16,
           wavelength=1.0):
    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        det = entry.create_group("instrument/detector")
        det.create_dataset("data", data=data, chunks=chunks)
        mono = entry.create_group("instrument/monochromator")
        mono.create_dataset("wavelength", data=np.array(wavelength))
    return path, data


def _lone_2d(path, *, shape=(20, 24), dtype=np.uint16):
    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        det = entry.create_group("instrument/detector")
        det.create_dataset("data", data=data)
    return path, data


def _imageless(path):
    with h5py.File(path, "w") as h5:
        e = h5.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        e.create_group("instrument")
    return path


def _count_h5_opens(monkeypatch) -> list:
    opens: list = []
    orig = h5py.File.__init__

    def counting(self, name, *a, **k):
        opens.append(str(name))
        return orig(self, name, *a, **k)

    monkeypatch.setattr(h5py.File, "__init__", counting)
    return opens


# --------------------------------------------------------------------------- #
# single open supplies everything
# --------------------------------------------------------------------------- #
def test_one_open_supplies_descriptor_metadata_count_and_reads(tmp_path, monkeypatch):
    p, data = _stack(tmp_path / "one_open.nxs")
    opens = _count_h5_opens(monkeypatch)
    with ContainerCursor(p) as cur:
        assert cur.descriptor.state is ProbeState.READY
        assert cur.wavelength == pytest.approx(1.0)
        assert cur.frame_count == 8
        _ = cur.metadata_for(0)             # plain NeXus -> {}
        for i in range(cur.frame_count):    # read every frame
            np.testing.assert_array_equal(cur.read_frame(i), data[i])
    # exactly one source-master open serviced the entire cursor lifecycle
    assert opens.count(str(p)) == 1


def test_frame_count_matches_descriptor(tmp_path):
    p, _ = _stack(tmp_path / "count.nxs")
    with ContainerCursor(p) as cur:
        assert cur.frame_count == cur.descriptor.frame_count == 8


# --------------------------------------------------------------------------- #
# native dtype + owner-block memory sharing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [np.uint16, np.uint32])
def test_reads_preserve_native_dtype(tmp_path, dtype):
    p, data = _stack(tmp_path / f"native_{np.dtype(dtype).name}.nxs", dtype=dtype)
    with ContainerCursor(p) as cur:
        assert cur.read_frame(0).dtype == np.dtype(dtype)
        block = cur.read_block(0, 4)
        assert block.array.dtype == np.dtype(dtype)


def test_block_frame_views_share_memory_with_owner(tmp_path):
    p, data = _stack(tmp_path / "share.nxs")
    with ContainerCursor(p) as cur:
        block = cur.read_block(2, 6)
        assert isinstance(block, ReadBlock)
        assert block.n_frames == 4
        assert block.nbytes == block.array.nbytes
        view = block.frame(3)  # absolute index 3 -> offset 1
        assert np.shares_memory(view, block.array)
        np.testing.assert_array_equal(view, data[3])


def test_iter_blocks_follows_plan_ranges(tmp_path):
    from xrd_tools.sources.read_plan import plan_reads

    p, data = _stack(tmp_path / "iter.nxs")  # (8,16,16) uint16, chunks (2,..)
    with ContainerCursor(p) as cur:
        d = cur.descriptor
        plan = plan_reads(
            d.frame_count, d.frame_shape, d.dtype, d.chunks,
            max_block_bytes=2 * (16 * 16 * 2))  # 2-frame budget
        seen = []
        for block in cur.iter_blocks(plan):
            assert block.nbytes <= plan.max_owner_bytes
            seen.extend(range(block.start, block.stop))
        assert seen == list(range(8))


# --------------------------------------------------------------------------- #
# 2-D dataset -> only frame 0
# --------------------------------------------------------------------------- #
def test_two_d_exposes_only_frame_zero(tmp_path):
    p, data = _lone_2d(tmp_path / "single.nxs")
    with ContainerCursor(p) as cur:
        assert cur.is_2d is True
        assert cur.frame_count == 1
        np.testing.assert_array_equal(cur.read_frame(0), data)
        with pytest.raises(IndexError):
            cur.read_frame(1)


# --------------------------------------------------------------------------- #
# close on all paths; reads after close fail
# --------------------------------------------------------------------------- #
def test_close_after_normal_exit(tmp_path):
    p, _ = _stack(tmp_path / "normal.nxs")
    cur = ContainerCursor(p)
    with cur:
        cur.read_frame(0)
    assert cur.closed is True


def test_close_on_exception(tmp_path):
    p, _ = _stack(tmp_path / "exc.nxs")
    cur = ContainerCursor(p)
    with pytest.raises(RuntimeError):
        with cur:
            cur.read_frame(0)
            raise RuntimeError("boom")
    assert cur.closed is True


def test_close_is_idempotent(tmp_path):
    p, _ = _stack(tmp_path / "idem.nxs")
    cur = ContainerCursor(p).open()
    cur.close()
    cur.close()  # no error on second close
    assert cur.closed is True


def test_generator_abandonment_closes_via_context(tmp_path):
    from xrd_tools.sources.read_plan import plan_reads

    p, _ = _stack(tmp_path / "abandon.nxs")
    cur = ContainerCursor(p)
    with cur:
        d = cur.descriptor
        plan = plan_reads(d.frame_count, d.frame_shape, d.dtype, d.chunks,
                          max_block_bytes=2 * (16 * 16 * 2))
        gen = cur.iter_blocks(plan)
        next(gen)          # consume one block, then abandon the generator
        gen.close()
    assert cur.closed is True


def test_reads_after_close_raise_clearly(tmp_path):
    p, _ = _stack(tmp_path / "closed.nxs")
    cur = ContainerCursor(p).open()
    cur.close()
    with pytest.raises(CursorClosedError):
        cur.read_frame(0)
    with pytest.raises(CursorClosedError):
        cur.read_block(0, 2)
    with pytest.raises(CursorClosedError):
        _ = cur.descriptor


def test_underlying_handle_closed_after_close(tmp_path):
    p, _ = _stack(tmp_path / "handle.nxs")
    cur = ContainerCursor(p).open()
    h5 = cur._h5  # noqa: SLF001 — white-box handle census
    assert bool(h5)  # open handle is truthy
    cur.close()
    assert not h5  # h5py.File is falsy once closed


def test_reopen_after_close_is_refused(tmp_path):
    p, _ = _stack(tmp_path / "reopen.nxs")
    cur = ContainerCursor(p).open()
    cur.close()
    with pytest.raises(CursorClosedError):
        cur.open()


# --------------------------------------------------------------------------- #
# imageless / processed -> reads fail clearly, descriptor still available
# --------------------------------------------------------------------------- #
def test_imageless_cursor_reads_fail_clearly(tmp_path):
    p = _imageless(tmp_path / "imageless.nxs")
    with ContainerCursor(p) as cur:
        assert cur.descriptor.state is ProbeState.IMAGELESS
        with pytest.raises(ValueError):
            cur.read_frame(0)


# --------------------------------------------------------------------------- #
# stale candidate rejected BEFORE opening the file
# --------------------------------------------------------------------------- #
def test_stale_candidate_rejected_before_open(tmp_path, monkeypatch):
    from xrd_tools.sources.directory_index import StaleCandidateError

    p, _ = _stack(tmp_path / "stale.nxs")
    st = p.stat()
    stale = Candidate(p, "nexus_hdf5", st.st_size + 999, st.st_mtime_ns)  # wrong stamp

    opens = _count_h5_opens(monkeypatch)
    with pytest.raises(StaleCandidateError):
        with ContainerCursor(p, candidate=stale):
            pass
    assert opens.count(str(p)) == 0  # never opened the file


def test_matching_candidate_opens_normally(tmp_path):
    p, data = _stack(tmp_path / "fresh.nxs")
    st = p.stat()
    fresh = Candidate(p, "nexus_hdf5", st.st_size, st.st_mtime_ns)
    with ContainerCursor(p, candidate=fresh) as cur:
        assert cur.descriptor.adapter_id == "nexus_hdf5"
        np.testing.assert_array_equal(cur.read_frame(0), data[0])
