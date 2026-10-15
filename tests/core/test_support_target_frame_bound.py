"""COVERAGE-EXPORT-FOLLOWUP-20261014 Stage A, frozen before the fix: the target reader's saved-support bound.

ANGULAR-SUPPORT-TARGET-FRAME-BOUND: a reader opened for one target frame bounded the primary 1-D group's support table
by its one-label frame map (``4 * 1 + 64`` rows), so a valid table of more rows read ``malformed_saved_support`` in the
target reader and in the preview built on it. The bound is the primary group's own qualified row count: the same for
the full reader, the target reader and the preview, whatever the labels are and whatever other modes hold.

Real writers (a headless Run through ``NexusSink``; ``NexusRecordWriter``) and public readers only. Malformed tables
are made only after a real write. The count observer is a pass-through subclass of the real ``SupportTableReader``,
as ``test_frame_preview`` wraps the real ``_frame_map``: it records the arguments and changes nothing. The reader and
the preview own no support reconstruction path, so an Off output reads its marker and nothing else.
"""
import h5py
import numpy as np
import pytest

from tests.core.test_angular_support import _saved_int


@pytest.fixture
def support_counts(monkeypatch):
    """Each real SupportTableReader construction's (group, frame_count, recording_off), passed through unchanged."""
    from xrd_tools.io import angular_support
    seen = []
    real = angular_support.SupportTableReader

    class Observed(real):
        def __init__(self, group, *, frame_count=None, recording_off=False):
            seen.append((group.name, frame_count, recording_off))
            super().__init__(group, frame_count=frame_count, recording_off=recording_off)

    monkeypatch.setattr(angular_support, "SupportTableReader", Observed)
    return seen


def _preview(target, label):
    from xrd_tools.io.frame_preview import read_frame_preview
    from xrd_tools.session import hydration
    key = hydration.HydrationReadKey(hydration.HydrationScope("context-a", "scan-7", "/raw/source", 3), str(target),
                                     int(label), hydration.HydrationPurpose.PREVIEW, None)
    return read_frame_preview(key).view


def _views(target, label):
    """The full reader's, the target reader's and the preview's view of one label."""
    from xrd_tools.io.frame_view import FrameViewReader
    with FrameViewReader(target, resolve_source=False) as reader:
        full = reader.read(label)
    with FrameViewReader(target, resolve_source=False, target_frame=label) as reader:
        one = reader.read(label)
    return {"full": full, "target": one, "preview": _preview(target, label)}


def _rows(target, group="entry/integrated_1d"):
    with h5py.File(target, "r") as handle:
        return int(handle[f"{group}/frame_index"].shape[0])


def test_a1_target_reader_and_preview_read_a_valid_table_beyond_the_one_label_bound(tmp_path, support_counts):
    target, captured = _saved_int(tmp_path, frames=69)
    rows = _rows(target)
    with h5py.File(target, "r") as handle:
        mappings = int(handle["entry/integrated_1d/angular_support_rows"].shape[0])
    assert rows == mappings == 69 > 4 * 1 + 64, "the real Run's table exceeds a one-label bound"
    for label in (1, 35, 69):
        run = captured[label]
        views = _views(target, label)
        for name, view in views.items():
            saved = view.angular_support
            assert saved.available, (name, label, saved.reason, saved.text)
            np.testing.assert_array_equal(saved.supported_deg, run.supported_deg.astype(np.float32))
            assert (saved.window_deg, saved.radial_unit, saved.bin_edges, saved.convention) == (
                run.window_deg, run.radial_unit, run.bin_edges, run.convention), (name, label)
            np.testing.assert_array_equal(view.intensity_1d, views["full"].intensity_1d)
    # every reader bounds the table by the group's own row count, read here from the file
    assert support_counts and {count for _, count, _ in support_counts} == {rows}, support_counts


def test_a2_sparse_labels_and_a_partial_secondary_mode_bound_by_the_primary_rows(tmp_path, support_counts):
    from tests.core.test_h23_record_writer import _Facade, _r1
    from xrd_tools.core.angular_support import AngularSupportProfile, SupportOrigin
    from xrd_tools.io import record_writer as rw
    from xrd_tools.io.frame_view import FrameViewReader
    from xrd_tools.session.stage_accounting import ResultMode

    def profile(value):
        return AngularSupportProfile(origin=SupportOrigin.RUN, supported_deg=np.full(4, value), window_deg=360.0,
                                     convention="detector_chi", radial_unit="q_A^-1", bin_edges=(0.05, 1.05))

    labels = tuple(range(5, 5 + 7 * 69, 7))  # 69 sparse labels; the largest is 481
    target = tmp_path / "sparse.nexus"
    facade = _Facade(target)
    primary, secondary = ResultMode.one_d("q_total"), ResultMode.one_d("q_oop")
    writer = rw.NexusRecordWriter(target, atomic=False, flush_every=None)
    writer.bind_session(facade)
    writer.begin(primary_mode_1d="q_total")
    for number, label in enumerate(labels):
        facade.set_revision(label, primary, 1)
        writer.write(rw.RecordWrite(label=label, result_1d=_r1(label + 1.0), mode_1d="q_total",
                                    angular_support=profile(float(number))))
        if number % 3 == 0:  # the secondary mode holds only every third frame
            facade.set_revision(label, secondary, 1)
            writer.write(rw.RecordWrite(label=label, result_1d=_r1(label + 2.0), mode_1d="q_oop"))
    writer.flush(force=True)
    writer.finish()
    rows, partial = _rows(target), _rows(target, "entry/integrated_1d/q_oop")
    assert (rows, partial, max(labels)) == (69, 23, 481), "sparse labels and a partial secondary mode"
    with FrameViewReader(target, resolve_source=False) as reader:
        assert reader.primary_mode_1d() == "q_total" and reader.modes_1d() == ("q_total", "q_oop")
    for number in (0, 34, 68):
        label = labels[number]
        for name, view in _views(target, label).items():
            saved = view.angular_support
            assert saved.available, (name, label, saved.reason, saved.text)
            np.testing.assert_array_equal(saved.supported_deg, np.full(4, float(number)))
            np.testing.assert_array_equal(view.intensity_1d, np.full(4, label + 1.0))
        with FrameViewReader(target, resolve_source=False, target_frame=label) as reader:
            record = reader.read_record(label)
            rows_view = reader.read_1d_rows((label,))
        np.testing.assert_array_equal(record.view_1d("q_total").intensity_1d, np.full(4, label + 1.0))
        if number % 3 == 0:
            np.testing.assert_array_equal(record.view_1d("q_oop").intensity_1d, np.full(4, label + 2.0))
        (primary_rows,) = [mode for mode in rows_view.modes if mode.mode == rows_view.primary_mode]
        assert primary_rows.support_rows is not None and primary_rows.support_rows[0].available
        np.testing.assert_array_equal(primary_rows.support_rows[0].supported_deg, np.full(4, float(number)))
    # the primary group's row count: not the largest label, not the partial mode, not their union
    assert support_counts and {count for _, count, _ in support_counts} == {rows}, support_counts


def test_a3_a_genuinely_oversized_table_stays_malformed_whatever_else_the_output_holds(tmp_path, support_counts):
    from xrd_tools.core.angular_support import SupportReason
    target, _captured = _saved_int(tmp_path, frames=1)
    with h5py.File(target, "r+") as handle:
        group = handle["entry/integrated_1d"]
        rows_node, digests_node = group["angular_support_rows"], group["angular_support_row_digest"]
        entries, digests = np.asarray(rows_node[()]), np.asarray(digests_node[()])
        attrs = {name: dict(node.attrs) for name, node in (("rows", rows_node), ("digests", digests_node))}
        assert entries.shape == (1, 2) and digests.shape == (1, 16)
        # 69 entries: unique labels, every row index valid, matching digests: only the size bound is wrong
        extra_labels = np.arange(1000, 1068, dtype=np.int64)
        grown = np.vstack([entries, np.column_stack([extra_labels, np.zeros_like(extra_labels)])])
        grown_digests = np.vstack([digests, np.tile(digests[0], (68, 1))])
        del group["angular_support_rows"], group["angular_support_row_digest"]
        group.create_dataset("angular_support_rows", data=grown.astype(entries.dtype))
        group.create_dataset("angular_support_row_digest", data=grown_digests.astype(digests.dtype))
        for name, key in (("angular_support_rows", "rows"), ("angular_support_row_digest", "digests")):
            group[name].attrs.update(attrs[key])
        # an entry-level inventory far larger than the group: a bound taken from it would excuse the table
        scan_data = handle["entry"].require_group("scan_data")
        scan_data["frame_index"] = np.arange(4096, dtype=np.int64)
    assert _rows(target) == 1
    with h5py.File(target, "r") as handle:
        stored = np.asarray(handle["entry/integrated_1d/intensity"][0])
    for name, view in _views(target, 1).items():
        assert view.angular_support.reason is SupportReason.MALFORMED_SAVED_SUPPORT, (name, view.angular_support)
        np.testing.assert_array_equal(view.intensity_1d, stored)  # the intensity stays readable, exactly as stored
    assert support_counts and {count for _, count, _ in support_counts} == {1}, support_counts


def test_a4_an_off_output_reads_recording_disabled_in_every_reader(tmp_path, support_counts):
    from xrd_tools.core.angular_support import SupportReason
    target, _captured = _saved_int(tmp_path, frames=3, enabled=False)
    for label in (1, 3):
        views = _views(target, label)
        for name, view in views.items():
            assert view.angular_support.reason is SupportReason.RECORDING_DISABLED, (name, view.angular_support)
            np.testing.assert_array_equal(view.intensity_1d, views["full"].intensity_1d)
    assert support_counts and all(off for _, _, off in support_counts), support_counts
