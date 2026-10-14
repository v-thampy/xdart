"""C7 terminal review: persisted support must remain bounded and bound to its axis."""
import h5py
import numpy as np
import pytest

from tests.core.test_angular_support import _saved_int


@pytest.mark.parametrize("length", (None, 8192))
def test_support_metadata_requires_the_bounded_fixed_width_format(tmp_path, length):
    from xrd_tools.io.angular_support import AngularSupportFormatError, SupportTableReader
    target, _captured = _saved_int(tmp_path, frames=1)
    with h5py.File(target, "r+") as handle:
        group = handle["entry/integrated_1d"]
        values = group["angular_support_meta"].asstr()[()].tolist()
        del group["angular_support_meta"]
        group.create_dataset("angular_support_meta", data=values,
                             dtype=h5py.string_dtype("utf-8", length))
        with pytest.raises(AngularSupportFormatError, match="4096"):
            SupportTableReader(group)


def test_support_is_stale_when_only_the_axis_unit_changes(tmp_path):
    from xrd_tools.core.angular_support import SupportReason
    from xrd_tools.io.frame_view import FrameViewReader
    target, _captured = _saved_int(tmp_path, frames=1)
    with h5py.File(target, "r+") as handle:
        handle["entry/integrated_1d/axis_1"].attrs["units"] = "2th_deg"
    with FrameViewReader(target) as reader:
        assert reader.read(1).angular_support.reason is SupportReason.STALE_SAVED_SUPPORT


def test_dropping_many_rows_preserves_the_surviving_rows_support(tmp_path):
    from xrd_tools.io.frame_view import FrameViewReader
    from xrd_tools.io.nexus_record import drop_integrated_rows
    target, captured = _saved_int(tmp_path, frames=69)
    with h5py.File(target, "r+") as handle:
        drop_integrated_rows(handle, "entry/integrated_1d", list(range(1, 69)))
    with FrameViewReader(target) as reader:
        saved = reader.read(69).angular_support
        assert saved.available, saved.text
        np.testing.assert_array_equal(saved.supported_deg, captured[69].supported_deg.astype(np.float32))


def test_public_row_drop_keeps_support_appendable(tmp_path):
    from dataclasses import replace
    from xrd_tools.io.angular_support import SupportTableWriter
    from xrd_tools.io.nexus_record import drop_integrated_rows
    target, captured = _saved_int(tmp_path)
    with h5py.File(target, "r+") as handle:
        drop_integrated_rows(handle, "entry/integrated_1d", [2])
        group = handle["entry/integrated_1d"]
        changed = replace(captured[1], detail="a replacement profile")
        SupportTableWriter().write(group, {1: changed}, intensity_rows={1: group["intensity"][0]})


def test_saved_support_is_accounted_by_the_readers_memory_authority(tmp_path):
    from xrd_tools.core.physical_memory import physical_root_fact
    from xrd_tools.io.frame_view import FrameViewReader
    target, _captured = _saved_int(tmp_path, frames=1)
    with FrameViewReader(target) as reader:
        profile = reader.read(1).angular_support
        assert profile.available
        root = physical_root_fact(profile.supported_deg).root
        assert any(root is retained for retained in reader._memory_authority.retained_roots), \
            "saved coverage must participate in the reader's existing memory budget"

