"""Item 11 review: scientific source identity, batch custody and admitted wide rows."""

import h5py
import numpy as np
import pytest

from tests.core.test_int_fit_source import _result
from tests.core.v2_fixture_factory import current_entry
from xrd_tools.analysis.fit_sources import IntResultRows, int_frame_snapshot
from xrd_tools.notebooks import load_snapshot
from xrd_tools.notebooks.inputs import NotebookInputError, int_frame_reference


def test_a_nondefault_entry_survives_the_notebook_reference(tmp_path):
    path = _result(tmp_path / "sample.nexus")
    with h5py.File(path, "r+") as handle:
        handle.move("entry", "entry_b")
    snapshot = int_frame_snapshot(path, 2, mode="default", entry="entry_b")
    reference = int_frame_reference(snapshot)
    assert dict(reference.selection)["entry"] == "entry_b"
    replayed = load_snapshot(reference, bundle_root=tmp_path, data_root=tmp_path)
    np.testing.assert_array_equal(replayed.values, snapshot.values)


def test_wavelength_is_part_of_the_row_scientific_identity(tmp_path):
    first = int_frame_snapshot(_result(tmp_path / "a.nexus", wavelength=1.), 0, mode="default")
    second = int_frame_snapshot(_result(tmp_path / "b.nexus", wavelength=2.), 0, mode="default")
    np.testing.assert_array_equal(first.values, second.values)
    assert first.key.fingerprint != second.key.fingerprint


def test_a_relocated_notebook_refuses_a_changed_wavelength(tmp_path):
    original, relocated = tmp_path / "original", tmp_path / "relocated"
    original.mkdir()
    relocated.mkdir()
    path = _result(original / "sample.nexus", wavelength=1.)
    reference = int_frame_reference(int_frame_snapshot(path, 0, mode="default"), data_root=original)
    _result(relocated / path.name, wavelength=2.)
    with pytest.raises(NotebookInputError, match="INPUT_CHANGED"):
        load_snapshot(reference, bundle_root=tmp_path, data_root=relocated)


@pytest.mark.parametrize("operation", ("inventory", "held", "next_chunk"))
def test_a_batch_refuses_a_replaced_file_including_held_rows(tmp_path, operation):
    path = _result(tmp_path / "sample.nexus", labels=(0, 1), wavelength=1.)
    rows = IntResultRows(path, mode="default", chunk=1)
    rows.snapshot(0)
    replacement = _result(tmp_path / "replacement.nexus", labels=(0, 1), shift=.2, wavelength=2.)
    replacement.replace(path)
    with pytest.raises(ValueError, match="FIT_SOURCE_CHANGED"):
        rows.labels() if operation == "inventory" else rows.snapshot(0 if operation == "held" else 1)


def test_rows_and_wavelength_cannot_come_from_two_file_generations(tmp_path, monkeypatch):
    from xrd_tools.io.frame_view import FrameViewReader
    path = _result(tmp_path / "sample.nexus", wavelength=1.)
    replacement = _result(tmp_path / "replacement.nexus", shift=.2, wavelength=2.)
    read = FrameViewReader.read_1d_rows

    def replace_after_read(reader, *args, **kwargs):
        result = read(reader, *args, **kwargs)
        replacement.replace(path)
        return result

    monkeypatch.setattr(FrameViewReader, "read_1d_rows", replace_after_read)
    with pytest.raises(ValueError, match="FIT_SOURCE_CHANGED"):
        int_frame_snapshot(path, 0, mode="default")


def test_default_batch_chunk_adapts_to_individually_admitted_wide_rows(tmp_path):
    from xrd_tools.core import FrameRecord, FrameView, axis_from_unit
    from xrd_tools.io import write_frame_records
    path = tmp_path / "wide.nexus"
    axis = axis_from_unit("q_A^-1", np.linspace(1., 4., 65_536))
    records = [FrameRecord.from_view(FrameView(label=i, axis_1d=axis,
               intensity_1d=np.full(axis.values.size, float(i + 1)), sigma_1d=np.ones(axis.values.size)))
               for i in range(64)]
    with h5py.File(path, "w") as handle:
        write_frame_records(current_entry(handle), records)
    rows = IntResultRows(path, mode="default")
    assert rows.labels() == tuple(range(64))
    for label in (0, 31, 32, 63):
        snapshot = rows.snapshot(label)
        assert snapshot.values.shape == (65_536,)
        np.testing.assert_array_equal(snapshot.values, float(label + 1))
