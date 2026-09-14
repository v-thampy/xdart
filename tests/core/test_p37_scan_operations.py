"""P3-7A headless Metadata, Scan Plot, and ROI operation contracts.

These tests intentionally exercise the public blocking runners.  Fakes stop at
the public source boundary; they do not replace table alignment, plotting, ROI
reduction, detachment, charging, fingerprinting, cancellation, or result truth.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
import os
import struct
import subprocess
import sys
import threading
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import tifffile

import xrd_tools.analysis as analysis
from xrd_tools.analysis.plans import RoiSignal
from xrd_tools.core.roi import RoiSpec
from xrd_tools.core.scan import SourceCapabilities, SourceKind, SourceSpec
from xrd_tools.io.metadata import ImageMetadataRead


_PUBLIC_RUNNERS = {
    "run_metadata_table",
    "run_metadata_table_requalification",
    "run_scan_plot",
    "run_roi_preview",
    "run_roi_scan",
    "run_displayed_peak_fit",
    "run_displayed_phase_fit",
}


class _CanonicalMode(Enum):
    SELECTED = 2


def _public(name: str):
    value = getattr(analysis, name, None)
    assert value is not None, f"missing frozen P3-7A public value {name}"
    return value


def _scan_ops():
    return importlib.import_module("xrd_tools.analysis.scan_operations")


def _word(value: object) -> str:
    return str(getattr(value, "value", value)).upper()


def _assert_terminal(result, disposition: str, code: str | None = None):
    assert _word(result.disposition) == disposition
    if code is not None:
        assert result.code == code
    return result


def _column(result, name: str):
    matches = [column for column in result.columns if column.name == name]
    assert len(matches) == 1
    return matches[0]


def _column_values(column):
    if column.kind == "numeric":
        assert column.text is None
        return column.numeric
    assert column.numeric is None
    return column.text


def _assert_owned_readonly(array: np.ndarray) -> None:
    assert isinstance(array, np.ndarray)
    assert array.dtype.kind != "O"
    assert array.flags.c_contiguous
    assert array.flags.owndata
    assert not array.flags.writeable


def _all_arrays(value: object, *, _seen: set[int] | None = None):
    seen = set() if _seen is None else _seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        yield value
        return
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from _all_arrays(getattr(value, field.name), _seen=seen)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _all_arrays(key, _seen=seen)
            yield from _all_arrays(item, _seen=seen)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _all_arrays(item, _seen=seen)


def _contains_identity(
    value: object, target: object, *, _seen: set[int] | None = None,
) -> bool:
    if value is target:
        return True
    seen = set() if _seen is None else _seen
    if id(value) in seen:
        return False
    seen.add(id(value))
    if dataclasses.is_dataclass(value):
        return any(
            _contains_identity(getattr(value, field.name), target, _seen=seen)
            for field in dataclasses.fields(value)
        )
    if isinstance(value, Mapping):
        return any(
            _contains_identity(item, target, _seen=seen)
            for pair in value.items() for item in pair
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_identity(item, target, _seen=seen) for item in value)
    return False


class _FakeSource:
    """Small public FrameSource-shaped boundary fake."""

    kind = SourceKind.IMAGE_FILE

    def __init__(
        self,
        path: Path,
        rows: Mapping[int, Mapping[str, object]],
        *,
        frames: Mapping[int, np.ndarray | BaseException] | None = None,
        scanned_axes: tuple[str, ...] = ("theta",),
    ) -> None:
        self.path = Path(path)
        self.spec = SourceSpec(self.path, self.kind)
        self._rows = {int(key): dict(value) for key, value in rows.items()}
        self._labels = tuple(self._rows)
        self._frames = dict(frames or {})
        self.scanned_axes = tuple(scanned_axes)
        self.capabilities = SourceCapabilities(
            supports_random_access=True,
            has_metadata=True,
            has_raw_references=True,
            has_scan_manifest=True,
        )
        self.metadata_calls: list[int] = []
        self.load_calls: list[int] = []
        self.fallback_calls: list[int] = []
        self.close_calls = 0
        self.call_threads: list[int] = []

    @property
    def frame_indices(self) -> list[int]:
        return list(self._labels)

    def metadata_for(self, label: int, *, max_input_bytes: int | None = None):
        assert max_input_bytes is None or (
            type(max_input_bytes) is int and max_input_bytes > 0
        )
        self.call_threads.append(threading.get_ident())
        self.metadata_calls.append(int(label))
        return dict(self._rows[int(label)])

    def scan_manifest(self):
        return [(label, dict(self._rows[label])) for label in self._labels]

    def load_frame(self, label: int) -> np.ndarray:
        self.call_threads.append(threading.get_ident())
        self.load_calls.append(int(label))
        value = self._frames.get(int(label), np.full((3, 4), float(label)))
        if isinstance(value, BaseException):
            raise value
        return np.asarray(value)

    def read_view(self, label: int, *, include_thumbnail: bool = True):
        """Observable thumbnail route used only by the M10 discriminator."""
        assert include_thumbnail
        self.fallback_calls.append(int(label))
        return SimpleNamespace(thumbnail=np.full((3, 4), -1.0))

    def close(self) -> None:
        self.close_calls += 1

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _install_source(monkeypatch, source_or_factory) -> list[SourceSpec]:
    ops = _scan_ops()
    opened: list[SourceSpec] = []

    def _open(spec, **_kwargs):
        opened.append(spec)
        return source_or_factory() if callable(source_or_factory) else source_or_factory

    monkeypatch.setattr(ops, "open_source", _open)
    monkeypatch.setattr(ops, "guess_source_kind", lambda _uri: SourceKind.IMAGE_FILE)
    return opened


def _metadata_plan(path: Path, **kwargs):
    plan_type = _public("MetadataTablePlan")
    source = SourceSpec(path, SourceKind.IMAGE_FILE, options={"metadata_format": "txt"})
    return plan_type(source=source, **kwargs)


def _run_table(monkeypatch, tmp_path: Path, rows, *, frames=None, source=None):
    image = tmp_path / "scan_0001.tif"
    image.write_bytes(b"not-read-by-metadata")
    owned = source or _FakeSource(image, rows, frames=frames)
    _install_source(monkeypatch, owned)
    result = _public("run_metadata_table")(_metadata_plan(image))
    return result, owned


def _preview_plan(table, label: int):
    return _public("RoiPreviewPlan").from_table(table, label=label)


def _roi_plan(table, signals, *, selected_labels=None, **kwargs):
    return _public("RoiScanPlan").from_table(
        table,
        signals=tuple(signals),
        selected_labels=None if selected_labels is None else tuple(selected_labels),
        **kwargs,
    )


def _simple_signal(name: str = "signal") -> RoiSignal:
    return RoiSignal(
        roi=RoiSpec(center_x=1.5, center_y=1.5, width_x=2, width_y=2),
        reducer="mean",
        name=name,
    )


def _expected_frame(tag: int, payload: bytes) -> bytes:
    return bytes((tag,)) + struct.pack(">Q", len(payload)) + payload


def test_metadata_table_runner_is_qt_free_ordered_bounded_and_direct(
    monkeypatch, tmp_path,
):
    ops = _scan_ops()
    assert ops._MAX_TABLE_ROWS == 100_000
    assert ops._MAX_TABLE_COLUMNS == 256
    assert ops._MAX_METADATA_PROJECTION_COLUMNS == 128
    assert ops._MAX_TABLE_BYTES == 64 * 1024 * 1024

    before_qt = {name for name in sys.modules if name.startswith(("PyQt", "PySide"))}
    caller = threading.get_ident()
    rows = {
        7: {"theta": 0.1, "Photod": 10.0, "State": "warm"},
        3: {"theta": 0.2, "Photod": 20.0},
    }
    result, source = _run_table(monkeypatch, tmp_path, rows)
    _assert_terminal(result, "COMPLETED", "OK")
    assert result.labels == (7, 3)
    assert tuple(column.name for column in result.columns) == (
        "frame_index", "theta", "Photod", "State",
    )
    np.testing.assert_array_equal(_column_values(_column(result, "frame_index")), [7, 3])
    assert _column_values(_column(result, "State")) == ("warm", None)
    assert source.call_threads and set(source.call_threads) == {caller}
    assert source.load_calls == []
    assert source.close_calls == 1
    assert before_qt == {name for name in sys.modules if name.startswith(("PyQt", "PySide"))}

    row_boundary, row_boundary_source = _run_table(
        monkeypatch, tmp_path, {index: {} for index in range(100_000)},
    )
    _assert_terminal(row_boundary, "COMPLETED", "OK")
    assert len(row_boundary.labels) == 100_000
    assert len(row_boundary_source.metadata_calls) == 100_000

    too_many_rows = _FakeSource(
        tmp_path / "rows.tif", {index: {} for index in range(100_001)}
    )
    (tmp_path / "rows.tif").write_bytes(b"x")
    _install_source(monkeypatch, too_many_rows)
    refused = _public("run_metadata_table")(_metadata_plan(tmp_path / "rows.tif"))
    _assert_terminal(refused, "REFUSED", "METADATA_ROW_LIMIT_EXCEEDED")
    assert too_many_rows.metadata_calls == []

    column_boundary, _ = _run_table(
        monkeypatch, tmp_path,
        {0: {f"c{index}": index for index in range(255)}},
    )
    _assert_terminal(column_boundary, "COMPLETED", "OK")
    assert len(column_boundary.columns) == 256

    too_many_columns = {0: {f"c{index}": index for index in range(256)}}
    result, _ = _run_table(monkeypatch, tmp_path, too_many_columns)
    _assert_terminal(result, "REFUSED", "METADATA_COLUMN_LIMIT_EXCEEDED")

    boundary, _ = _run_table(monkeypatch, tmp_path, {0: {"x": 1.0}})
    _assert_terminal(boundary, "COMPLETED", "OK")
    monkeypatch.setattr(ops, "_MAX_TABLE_BYTES", boundary.storage_bytes)
    exact_boundary, _ = _run_table(monkeypatch, tmp_path, {0: {"x": 1.0}})
    _assert_terminal(exact_boundary, "COMPLETED", "OK")
    assert exact_boundary.storage_bytes == boundary.storage_bytes
    monkeypatch.setattr(ops, "_MAX_TABLE_BYTES", boundary.storage_bytes - 1)
    one_over, _ = _run_table(monkeypatch, tmp_path, {0: {"x": 1.0}})
    _assert_terminal(one_over, "REFUSED", "METADATA_TABLE_LIMIT_EXCEEDED")

    monkeypatch.setattr(ops, "_MAX_TABLE_BYTES", 256)
    early_source = _FakeSource(
        tmp_path / "early-stop.tif",
        {1: {"large": "x" * 1024}, 2: {"late": 2.0}, 3: {"later": 3.0}},
    )
    (tmp_path / "early-stop.tif").write_bytes(b"x")
    _install_source(monkeypatch, early_source)
    early = _public("run_metadata_table")(
        _metadata_plan(tmp_path / "early-stop.tif")
    )
    _assert_terminal(early, "REFUSED", "METADATA_TABLE_LIMIT_EXCEEDED")
    assert early_source.metadata_calls == [1]


def test_scan_plot_runner_resolves_defaults_normalization_overlay_and_roi_kind(
    monkeypatch, tmp_path,
):
    rows = {
        1: {"theta": 1.0, "Photod": 10.0, "mon": 2.0, "ROI1": 99.0},
        2: {"theta": 2.0, "Photod": 20.0, "mon": 0.0, "ROI1": 98.0},
        3: {"theta": 3.0, "Photod": 30.0, "mon": np.nan, "ROI1": 97.0},
    }
    table, _ = _run_table(
        monkeypatch,
        tmp_path,
        rows,
    )
    opens = []
    monkeypatch.setattr(_scan_ops(), "open_source", lambda *_a, **_k: opens.append(1))
    default = _public("run_scan_plot")(_public("ScanPlotPlan")(), table)
    _assert_terminal(default, "COMPLETED", "OK")
    assert default.x_name == "theta"
    assert default.trace_names == ("Photod",)
    np.testing.assert_allclose(default.x, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(default.traces[0], [10.0, 20.0, 30.0])

    priority_root = tmp_path / "priority"
    priority_root.mkdir()
    priority, _ = _run_table(
        monkeypatch,
        priority_root,
        {1: {"theta": 1.0, "bs": 4.0, "mon": 8.0}},
    )
    priority_plot = _public("run_scan_plot")(_public("ScanPlotPlan")(), priority)
    _assert_terminal(priority_plot, "COMPLETED", "OK")
    assert priority_plot.trace_names == ("bs",)

    normalized = _public("run_scan_plot")(
        _public("ScanPlotPlan")(x="frame_index", y=("Photod",), normalization="mon"),
        table,
    )
    _assert_terminal(normalized, "COMPLETED", "OK")
    np.testing.assert_allclose(normalized.traces[0][:1], [5.0])
    assert np.isnan(normalized.traces[0][1:]).all()
    assert normalized.normalization_invalid_count == 2
    assert normalized.trace_origins == ("metadata",)
    assert normalized.original_identities == ("Photod",)
    assert opens == []

    frames = {label: np.full((3, 4), label, dtype=float) for label in table.labels}
    roi_source = _FakeSource(
        tmp_path / "scan_0001.tif", rows, frames=frames,
    )
    _install_source(monkeypatch, roi_source)
    roi = _public("run_roi_scan")(_roi_plan(table, (_simple_signal("ROI1"),)))
    _assert_terminal(roi, "COMPLETED", "OK")
    combined = _public("run_scan_plot")(
        _public("ScanPlotPlan")(y=("ROI1",)), table, roi_result=roi
    )
    assert combined.trace_names == ("ROI1", "ROI1_2")
    assert combined.trace_origins == ("metadata", "derived_roi")
    assert combined.original_identities == ("ROI1", "ROI1")

    plot_rows = {
        1: {f"c{index}": float(index) for index in range(17)},
    }
    plot_table, _ = _run_table(monkeypatch, tmp_path, plot_rows)
    sixteen = tuple(f"c{index}" for index in range(16))
    boundary = _public("run_scan_plot")(
        _public("ScanPlotPlan")(x="frame_index", y=sixteen), plot_table,
    )
    _assert_terminal(boundary, "COMPLETED", "OK")
    assert boundary.trace_names == sixteen
    too_many = _public("run_scan_plot")(
        _public("ScanPlotPlan")(
            x="frame_index", y=tuple(f"c{index}" for index in range(17)),
        ),
        plot_table,
    )
    _assert_terminal(too_many, "REFUSED", "SCAN_PLOT_TRACE_LIMIT_EXCEEDED")

    collision_roi = _public("RoiScanResult")(
        _scan_ops().AnalysisDisposition.COMPLETED,
        "OK",
        receipt=plot_table.receipt,
        table_fingerprint=plot_table.table_fingerprint,
        requested_labels=plot_table.labels,
        completed_labels=plot_table.labels,
        signal_names=("c0",),
        signal_values=(np.array([100.0]),),
        valid_counts=(np.array([1]),),
        result_fingerprint="collision",
    )
    emitted_sixteen = _public("run_scan_plot")(
        _public("ScanPlotPlan")(
            x="frame_index", y=tuple(f"c{index}" for index in range(15)),
        ),
        plot_table,
        roi_result=collision_roi,
    )
    _assert_terminal(emitted_sixteen, "COMPLETED", "OK")
    assert len(emitted_sixteen.traces) == 16
    emitted_seventeen = _public("run_scan_plot")(
        _public("ScanPlotPlan")(
            x="frame_index", y=tuple(f"c{index}" for index in range(16)),
        ),
        plot_table,
        roi_result=collision_roi,
    )
    _assert_terminal(
        emitted_seventeen, "REFUSED", "SCAN_PLOT_TRACE_LIMIT_EXCEEDED",
    )
    partial_roi = dataclasses.replace(
        collision_roi, requested_labels=(), completed_labels=(),
    )
    partial = _public("run_scan_plot")(
        _public("ScanPlotPlan")(x="frame_index", y=("c0",)),
        plot_table,
        roi_result=partial_roi,
    )
    _assert_terminal(partial, "REFUSED", "ROI_IDENTITY_MISMATCH")
    monkeypatch.setattr(_scan_ops(), "_MAX_PLOT_BYTES", boundary.storage_bytes)
    exact_bytes = _public("run_scan_plot")(
        _public("ScanPlotPlan")(x="frame_index", y=sixteen), plot_table,
    )
    _assert_terminal(exact_bytes, "COMPLETED", "OK")
    monkeypatch.setattr(_scan_ops(), "_MAX_PLOT_BYTES", boundary.storage_bytes - 1)
    byte_over = _public("run_scan_plot")(
        _public("ScanPlotPlan")(x="frame_index", y=sixteen), plot_table,
    )
    _assert_terminal(byte_over, "REFUSED", "SCAN_PLOT_LIMIT_EXCEEDED")


def test_roi_preview_and_scan_use_strict_raw_and_one_source_owner(monkeypatch, tmp_path):
    rows = {1: {"theta": 1.0}, 2: {"theta": 2.0}, 3: {"theta": 3.0}}
    table, _ = _run_table(monkeypatch, tmp_path, rows)
    made: list[_FakeSource] = []

    def factory():
        source = _FakeSource(
            tmp_path / "scan_0001.tif",
            rows,
            frames={label: np.full((3, 4), label, dtype=float) for label in rows},
        )
        made.append(source)
        return source

    _install_source(monkeypatch, factory)
    preview = _public("run_roi_preview")(_preview_plan(table, 2))
    _assert_terminal(preview, "COMPLETED", "OK")
    assert made[0].load_calls == [2]
    assert made[0].close_calls == 1
    _assert_owned_readonly(preview.image)

    scan = _public("run_roi_scan")(_roi_plan(table, (_simple_signal(),)))
    _assert_terminal(scan, "COMPLETED", "OK")
    assert made[1].load_calls == [1, 2, 3]
    assert made[1].close_calls == 1
    assert len(made) == 2


def test_roi_total_raw_absence_completes_all_rows_nan_zero_count_with_diagnostic(
    monkeypatch, tmp_path,
):
    rows = {1: {"theta": 1.0}, 2: {"theta": 2.0}}
    table, _ = _run_table(monkeypatch, tmp_path, rows)
    source = _FakeSource(
        tmp_path / "scan_0001.tif",
        rows,
        frames={label: OSError("raw archived") for label in rows},
    )
    _install_source(monkeypatch, source)
    result = _public("run_roi_scan")(_roi_plan(table, (_simple_signal(),)))
    _assert_terminal(result, "COMPLETED", "OK")
    assert result.requested_labels == (1, 2)
    assert result.completed_labels == (1, 2)
    assert result.no_raw_labels == (1, 2)
    assert np.isnan(result.signal_values[0]).all()
    np.testing.assert_array_equal(result.valid_counts[0], [0, 0])
    assert "ROI_RAW_UNAVAILABLE_ALL" in result.diagnostics


def test_roi_invalid_mask_warns_once_ignores_and_reports_machine_diagnostic(
    monkeypatch, tmp_path, caplog,
):
    rows = {1: {}, 2: {}}
    table, _ = _run_table(monkeypatch, tmp_path, rows)
    source = _FakeSource(
        tmp_path / "scan_0001.tif",
        rows,
        frames={label: np.arange(12.0).reshape(3, 4) for label in rows},
    )
    _install_source(monkeypatch, source)
    result = _public("run_roi_scan")(
        _roi_plan(table, (_simple_signal(),), mask=np.zeros((2, 2), dtype=bool))
    )
    _assert_terminal(result, "COMPLETED", "OK")
    assert "ROI_INVALID_MASK_IGNORED" in result.diagnostics
    messages = [record.message for record in caplog.records if "static mask" in record.message]
    assert len(messages) == 1
    assert source.load_calls == [1, 2]


def test_roi_loaded_all_nan_is_not_mislabeled_raw_absent(monkeypatch, tmp_path):
    rows = {1: {}}
    table, _ = _run_table(monkeypatch, tmp_path, rows)
    source = _FakeSource(
        tmp_path / "scan_0001.tif", rows, frames={1: np.full((3, 4), np.nan)}
    )
    _install_source(monkeypatch, source)
    result = _public("run_roi_scan")(_roi_plan(table, (_simple_signal(),)))
    _assert_terminal(result, "COMPLETED", "OK")
    assert result.no_raw_labels == ()
    assert np.isnan(result.signal_values[0][0])
    assert result.valid_counts[0][0] == 0
    assert "ROI_RAW_UNAVAILABLE_ALL" not in result.diagnostics


def test_p37_metadata_selection_required_returns_bounded_candidates_for_resubmit(
    monkeypatch, tmp_path,
):
    ops = _scan_ops()
    assert ops._MAX_CANDIDATES == 256
    assert ops._MAX_CANDIDATE_BYTES == 1024 * 1024
    opened: list[object] = []
    monkeypatch.setattr(ops, "open_source", lambda value, **_kw: opened.append(value))
    candidates = [
        SourceSpec(tmp_path / f"scan_{index}.tif", SourceKind.IMAGE_FILE)
        for index in range(2)
    ]
    monkeypatch.setattr(ops, "discover_scans", lambda *_a, **_k: list(candidates))
    plan = _public("MetadataTablePlan")(
        source=tmp_path,
        selection="directory",
        kind=SourceKind.IMAGE_FILE,
    )
    monkeypatch.setattr(ops, "discover_scans", lambda *_a, **_k: [candidates[0]])
    single = _public("run_metadata_table")(plan)
    _assert_terminal(single, "REFUSED", "SOURCE_SELECTION_REQUIRED")
    assert tuple(item.source_spec for item in single.candidates) == (candidates[0],)
    assert opened == []

    monkeypatch.setattr(ops, "discover_scans", lambda *_a, **_k: list(candidates))
    result = _public("run_metadata_table")(plan)
    _assert_terminal(result, "REFUSED", "SOURCE_SELECTION_REQUIRED")
    assert tuple(candidate.source_spec for candidate in result.candidates) == tuple(candidates)
    assert opened == []

    with_metadata_uri = SourceSpec(
        candidates[0].uri, candidates[0].kind,
        metadata_uri=tmp_path / "metadata-a",
    )
    without_metadata_uri = SourceSpec(candidates[0].uri, candidates[0].kind)
    assert ops._candidate(with_metadata_uri).fingerprint != ops._candidate(
        without_metadata_uri
    ).fingerprint

    candidate_charge = sum(candidate.storage_bytes for candidate in result.candidates)
    monkeypatch.setattr(ops, "_MAX_CANDIDATE_BYTES", candidate_charge)
    exact_bytes = _public("run_metadata_table")(plan)
    _assert_terminal(exact_bytes, "REFUSED", "SOURCE_SELECTION_REQUIRED")
    assert len(exact_bytes.candidates) == 2
    monkeypatch.setattr(ops, "_MAX_CANDIDATE_BYTES", candidate_charge - 1)
    byte_over = _public("run_metadata_table")(plan)
    _assert_terminal(byte_over, "REFUSED", "SOURCE_CANDIDATE_LIMIT_EXCEEDED")
    assert byte_over.candidates == ()
    monkeypatch.setattr(ops, "_MAX_CANDIDATE_BYTES", 1024 * 1024)

    monkeypatch.setattr(
        ops,
        "discover_scans",
        lambda *_a, **_k: [
            SourceSpec(tmp_path / f"accepted_{index}.tif", SourceKind.IMAGE_FILE)
            for index in range(256)
        ],
    )
    exact_count = _public("run_metadata_table")(plan)
    _assert_terminal(exact_count, "REFUSED", "SOURCE_SELECTION_REQUIRED")
    assert len(exact_count.candidates) == 256

    selected = result.candidates[0].source_spec
    Path(selected.uri).write_bytes(b"selected")
    selected_source = _FakeSource(Path(selected.uri), {3: {"theta": 1.0}})
    _install_source(monkeypatch, selected_source)
    resubmitted = _public("run_metadata_table")(
        _public("MetadataTablePlan")(source=selected)
    )
    _assert_terminal(resubmitted, "COMPLETED", "OK")
    assert resubmitted.labels == (3,)

    monkeypatch.setattr(
        ops,
        "discover_scans",
        lambda *_a, **_k: [
            SourceSpec(tmp_path / f"scan_{index}.tif", SourceKind.IMAGE_FILE)
            for index in range(257)
        ],
    )
    over = _public("run_metadata_table")(plan)
    _assert_terminal(over, "REFUSED", "SOURCE_CANDIDATE_LIMIT_EXCEEDED")
    assert over.candidates == ()

    with monkeypatch.context() as context:
        context.setattr(ops, "_MAX_CANDIDATE_BYTES", 0)
        context.setattr(
            ops, "_digest",
            lambda *_a, **_k: pytest.fail("over-budget candidate was hashed"),
        )
        context.setattr(ops, "discover_scans", lambda *_a, **_k: [candidates[0]])
        prehash_refusal = _public("run_metadata_table")(plan)
    _assert_terminal(
        prehash_refusal, "REFUSED", "SOURCE_CANDIDATE_LIMIT_EXCEEDED",
    )

    preopen_calls = []
    monkeypatch.setattr(
        ops, "open_source", lambda value, **_kw: preopen_calls.append(value),
    )
    for kind in (SourceKind.MEMORY, SourceKind.TILED, SourceKind.LIVE, SourceKind.UNKNOWN):
        refused = _public("run_metadata_table")(
            _public("MetadataTablePlan")(source=SourceSpec("virtual", kind))
        )
        _assert_terminal(refused, "REFUSED", "SOURCE_KIND_UNSUPPORTED")
    assert preopen_calls == []

    missing = _public("run_metadata_table")(
        _public("MetadataTablePlan")(
            source=SourceSpec(tmp_path / "absent.tif", SourceKind.IMAGE_FILE),
        )
    )
    _assert_terminal(missing, "REFUSED", "SOURCE_NOT_FOUND")
    assert preopen_calls == []
    for changes in ({"selection": "other"}, {"recursive": 1}):
        invalid_selection = _public("run_metadata_table")(
            _public("MetadataTablePlan")(
                source=tmp_path, kind=SourceKind.IMAGE_FILE, **changes,
            )
        )
        _assert_terminal(
            invalid_selection, "REFUSED", "INVALID_SOURCE_SELECTION",
        )
    assert preopen_calls == []

    import xrd_tools.io.nexus as nexus_io

    container = tmp_path / "container.h5"
    container.write_bytes(b"container")
    monkeypatch.setattr(
        ops,
        "image_series_spec",
        lambda *_a, **_k: SourceSpec(container, SourceKind.NEXUS_STACK),
    )
    monkeypatch.setattr(nexus_io, "list_entries", lambda _path: ["entry"])

    claimed_processed_opens = []
    monkeypatch.setattr(ops, "guess_source_kind", lambda _path: SourceKind.NEXUS_STACK)
    monkeypatch.setattr(
        ops, "open_source",
        lambda spec, **_kw: claimed_processed_opens.append(spec),
    )
    claimed_processed = _public("run_metadata_table")(
        _public("MetadataTablePlan")(
            source=SourceSpec(
                container, SourceKind.PROCESSED_NEXUS, entry="entry",
            )
        )
    )
    _assert_terminal(
        claimed_processed, "REFUSED", "SOURCE_IDENTITY_MISMATCH",
    )
    assert claimed_processed_opens == []

    class _BrokenCursor:
        descriptor = SimpleNamespace(frame_count=1)
        close_calls = 0

        def metadata_provider(self):
            raise RuntimeError("provider construction failed")

        def close(self):
            self.close_calls += 1

    broken_cursor = _BrokenCursor()
    with pytest.raises(RuntimeError, match="provider construction failed"):
        ops._CursorSource(broken_cursor)
    assert broken_cursor.close_calls == 1

    for routed_kind in (
        SourceKind.PROCESSED_NEXUS,
        SourceKind.NEXUS_STACK,
        SourceKind.EIGER_MASTER,
    ):
        routed = _FakeSource(container, {0: {"theta": 1.0}})
        routed.kind = routed_kind
        routed.spec = SourceSpec(container, routed_kind, entry="entry")
        routed.descriptor = SimpleNamespace(
            kind=routed_kind, resolved_entry="entry",
        )
        routed.frame_for = lambda label, source=routed: SimpleNamespace(
            metadata=dict(source._rows[label]), source_path=None,
            source_frame_index=None,
        )
        routed_opens = _install_source(monkeypatch, routed)
        monkeypatch.setattr(
            ops,
            "guess_source_kind",
            lambda _path, k=routed_kind: (
                SourceKind.PROCESSED_NEXUS
                if k is SourceKind.PROCESSED_NEXUS else SourceKind.NEXUS_STACK
            ),
        )
        routed_result = _public("run_metadata_table")(
            _public("MetadataTablePlan")(
                source=container,
                selection="image_series",
                entry="entry",
            )
        )
        _assert_terminal(routed_result, "COMPLETED", "OK")
        assert routed_result.receipt.resolved_kind is routed_kind
        assert routed_result.receipt.resolved_entry == "entry"
        assert len(routed_opens) == 1

    drifted = _FakeSource(container, {0: {"theta": 1.0}})
    drifted.kind = SourceKind.PROCESSED_NEXUS
    drifted.descriptor = SimpleNamespace(
        kind=SourceKind.PROCESSED_NEXUS,
        resolved_entry="other",
    )
    _install_source(monkeypatch, drifted)
    entry_drift = _public("run_metadata_table")(
        _public("MetadataTablePlan")(
            source=SourceSpec(
                container, SourceKind.PROCESSED_NEXUS, entry="entry",
            )
        )
    )
    _assert_terminal(entry_drift, "REFUSED", "SOURCE_IDENTITY_MISMATCH")

    class _DescriptorFailure(_FakeSource):
        @property
        def descriptor(self):
            raise RuntimeError("descriptor projection failed")

    descriptor_failure = _DescriptorFailure(container, {0: {"theta": 1.0}})
    descriptor_failure.kind = SourceKind.NEXUS_STACK
    _install_source(monkeypatch, descriptor_failure)
    with pytest.raises(RuntimeError, match="descriptor projection failed"):
        _public("run_metadata_table")(
            _public("MetadataTablePlan")(
                source=SourceSpec(
                    container, SourceKind.NEXUS_STACK, entry="entry",
                )
            )
        )
    assert descriptor_failure.close_calls == 1

    assert ops._resolve_spec_scan("7.2", ("7.1", "7.2")) == "7.2"
    assert ops._resolve_spec_scan("7", ("7.2",)) == "7.2"
    assert ops._resolve_spec_scan("7", ("7.1", "7.2")) is None
    assert ops._resolve_nexus_entry("entry", ("entry",)) == "entry"
    assert ops._resolve_nexus_entry("entry", ("entry", "entry")) is None


def test_p37_metadata_uses_public_generic_source_not_average_closed_reader(
    monkeypatch, tmp_path,
):
    # This is the single genuine parent RED: normal package import, then an
    # in-test lookup.  Collection remains healthy on the exact parent.
    runner = getattr(analysis, "run_metadata_table", None)
    assert callable(runner), "P3-7A public generic-source runner is absent"
    plan_type = getattr(analysis, "MetadataTablePlan", None)
    assert plan_type is not None
    ops = importlib.import_module("xrd_tools.analysis.scan_operations")

    spec_path = tmp_path / "real_spec"
    spec_path.write_text(
        "#F real_spec\n#E 1\n#O0 th  chi\n\n"
        "#S 5 ascan th 0 1 1 1\n#P0 0 5\n#N 3\n"
        "#L th  i0  det\n0 100 10\n1 110 20\n",
        encoding="utf-8",
    )
    real_spec = runner(plan_type(
        source=SourceSpec(
            spec_path, SourceKind.SPEC, options={"scan": "5.1"},
        )
    ))
    _assert_terminal(real_spec, "COMPLETED", "OK")
    np.testing.assert_array_equal(_column(real_spec, "th").numeric, [0.0, 1.0])
    np.testing.assert_array_equal(_column(real_spec, "i0").numeric, [100.0, 110.0])
    np.testing.assert_array_equal(_column(real_spec, "chi").numeric, [5.0, 5.0])

    import h5py

    stack = tmp_path / "bluesky.nxs"
    hy = np.array([11.0, 11.3, 11.6])
    with h5py.File(stack, "w") as handle:
        handle.attrs["creator"] = "NXWriter"
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        instrument = entry.create_group("instrument")
        bluesky = instrument.create_group("bluesky")
        bluesky.attrs["NX_class"] = "NXnote"
        metadata = bluesky.create_group("metadata")
        metadata.create_dataset("motors", data=b"!!python/tuple\n- hy\n")
        positioners = instrument.create_group("positioners")
        motor = positioners.create_group("hy")
        motor.create_dataset("value", data=hy)
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data.attrs["signal"] = "eiger_image"
        data.create_dataset("hy", data=hy)
        data.create_dataset("i0", data=np.array([100.0, 105.0, 110.0]))
        data.create_dataset("EPOCH", data=np.array([1.0, 2.0, 3.0]))
        images = data.create_dataset(
            "eiger_image", data=np.zeros((3, 2, 2), dtype=np.uint16),
        )
        images.attrs["signal_type"] = "detector"
        entry.create_dataset("end_time", data=np.bytes_("2026-08-24T00:00:00"))
    bluesky_result = runner(plan_type(
        source=stack, selection="image_series", entry="entry",
    ))
    _assert_terminal(bluesky_result, "COMPLETED", "OK")
    assert bluesky_result.declared_scanned_positioner == "hy"
    assert bluesky_result.selected_scanned_positioner == "hy"
    np.testing.assert_array_equal(_column(bluesky_result, "hy").numeric, hy)
    np.testing.assert_array_equal(
        _column(bluesky_result, "i0").numeric, [100.0, 105.0, 110.0],
    )

    image = tmp_path / "generic.tif"
    image.write_bytes(b"metadata-only")
    source = _FakeSource(image, {4: {"theta": 1.25, "Photod": 8.0}})
    opened = _install_source(monkeypatch, source)
    forbidden = {
        "xrd_tools.reduction.average",
        "xrd_tools.sources.execution_graph",
    }
    before = set(sys.modules)
    result = runner(plan_type(source=SourceSpec(image, SourceKind.IMAGE_FILE)))
    _assert_terminal(result, "COMPLETED", "OK")
    assert len(opened) == 1 and isinstance(opened[0], SourceSpec)
    assert source.metadata_calls == [4]
    assert source.load_calls == []
    assert forbidden.isdisjoint(set(sys.modules) - before)
    source_text = inspect.getsource(ops)
    assert "average_closed_v1" not in source_text
    assert "open_source_execution_graph" not in source_text


def test_p37_metadata_performs_zero_detector_pixel_reads(monkeypatch, tmp_path):
    rows = {1: {"theta": 1.0}, 2: {"theta": 2.0}}
    source = _FakeSource(tmp_path / "scan_0001.tif", rows)
    (tmp_path / "scan_0001.tif").write_bytes(b"x")
    source.load_frame = lambda *_a, **_k: pytest.fail("metadata read detector pixels")
    source.read_frame = lambda *_a, **_k: pytest.fail("metadata read cursor pixels")
    _install_source(monkeypatch, source)
    result = _public("run_metadata_table")(_metadata_plan(tmp_path / "scan_0001.tif"))
    _assert_terminal(result, "COMPLETED", "OK")
    assert source.metadata_calls == [1, 2]

    # Reach the built-in ImageFileSource observation arm as well: M09 replaces
    # this metadata-only call with a detector-pixel read.
    from xrd_tools.sources.image import ImageFileSource

    image = tmp_path / "built-in.raw"
    sidecar = tmp_path / "built-in.metadata"
    image.write_bytes(b"raw-not-read")
    sidecar.write_text("motor=1\nlabel=ok\nother=2\n", encoding="utf-8")
    built_in = ImageFileSource(
        image, metadata_format="metadata", frame_indices=(0, 1, 2),
    )
    built_in.load_frame = lambda *_a, **_k: pytest.fail(
        "built-in metadata arm read detector pixels"
    )
    observed_calls = []

    def observed(*args, **kwargs):
        observed_calls.append((args, kwargs))
        state = _scan_ops()._path_revision(sidecar)
        return ImageMetadataRead(
            {"motor": 1, "label": "ok", "other": 2}, sidecar, state,
        )

    _install_source(monkeypatch, built_in)
    monkeypatch.setattr(_scan_ops(), "read_image_metadata_observed", observed)
    built_in_result = _public("run_metadata_table")((
        _public("MetadataTablePlan")(
            source=SourceSpec(
                image, SourceKind.IMAGE_FILE,
                options={"metadata_format": "metadata"},
            )
        )
    ))
    _assert_terminal(built_in_result, "COMPLETED", "OK")
    assert built_in_result.labels == (0, 1, 2)
    assert len(observed_calls) == 1
    assert observed_calls[0][1]["max_input_bytes"] > 0


def test_p37_roi_preview_is_the_only_strict_raw_probe(monkeypatch, tmp_path):
    rows = {1: {"Photod": 1.0}, 2: {"Photod": 2.0}}
    table, _ = _run_table(monkeypatch, tmp_path, rows)
    plot = _public("run_scan_plot")(_public("ScanPlotPlan")(), table)
    _assert_terminal(plot, "COMPLETED", "OK")

    sources: list[_FakeSource] = []

    def factory():
        source = _FakeSource(
            tmp_path / "scan_0001.tif",
            rows,
            frames={1: np.ones((3, 4)), 2: np.ones((3, 4)) * 2},
        )
        source.probe_first_frame = lambda: pytest.fail("hidden raw probe")
        sources.append(source)
        return source

    _install_source(monkeypatch, factory)
    preview = _public("run_roi_preview")(_preview_plan(table, 2))
    _assert_terminal(preview, "COMPLETED", "OK")
    assert sources[0].load_calls == [2]
    assert sources[0].fallback_calls == []
    scan = _public("run_roi_scan")(_roi_plan(table, (_simple_signal(),)))
    _assert_terminal(scan, "COMPLETED", "OK")
    assert sources[1].load_calls == [1, 2]
    assert sources[1].fallback_calls == []


def test_p37_table_roi_source_label_order_and_fingerprint_mismatch_refuses(
    monkeypatch, tmp_path,
):
    ops = _scan_ops()
    rows = {1: {"theta": 1.0}, 2: {"theta": 2.0}}
    table, _ = _run_table(monkeypatch, tmp_path, rows)
    variants = (
        {2: {"theta": 2.0}, 1: {"theta": 1.0}},
        {1: {"theta": 99.0}, 2: {"theta": 2.0}},
        {1: {"theta": 1.0}, 3: {"theta": 3.0}},
    )
    for changed in variants:
        source = _FakeSource(
            tmp_path / "scan_0001.tif",
            changed,
            frames={label: np.ones((3, 4)) for label in changed},
        )
        _install_source(monkeypatch, source)
        result = _public("run_roi_preview")(_preview_plan(table, 1))
        _assert_terminal(result, "REFUSED", "SOURCE_IDENTITY_MISMATCH")
        assert source.load_calls == []

    axis_drift = _FakeSource(
        tmp_path / "scan_0001.tif", rows, scanned_axes=("phi",),
    )
    _install_source(monkeypatch, axis_drift)
    drifted_axis = _public("run_roi_preview")(_preview_plan(table, 1))
    _assert_terminal(
        drifted_axis, "REFUSED", "SOURCE_IDENTITY_MISMATCH",
    )
    assert axis_drift.load_calls == []

    wrong_receipt = dataclasses.replace(table.receipt, resolved_kind=SourceKind.TIFF_SERIES)
    wrong_plan = dataclasses.replace(_preview_plan(table, 1), receipt=wrong_receipt)
    source = _FakeSource(tmp_path / "scan_0001.tif", rows)
    _install_source(monkeypatch, source)
    refused = _public("run_roi_preview")(wrong_plan)
    _assert_terminal(refused, "REFUSED", "SOURCE_IDENTITY_MISMATCH")
    assert source.load_calls == []

    # Isolate ordered-label identity from catalog values: the same value rows
    # under different labels must retain the same catalog digest but a distinct
    # labels digest and source fingerprint (the M11 discriminator).
    path = tmp_path / "scan_0001.tif"
    first = _FakeSource(path, {1: {"theta": 1.0}, 2: {"theta": 2.0}})
    _install_source(monkeypatch, first)
    first_table = _public("run_metadata_table")(_metadata_plan(path))
    _assert_terminal(first_table, "COMPLETED", "OK")
    second = _FakeSource(path, {10: {"theta": 1.0}, 11: {"theta": 2.0}})
    _install_source(monkeypatch, second)
    second_table = _public("run_metadata_table")(_metadata_plan(path))
    _assert_terminal(second_table, "COMPLETED", "OK")
    assert first_table.receipt.catalog_digest == second_table.receipt.catalog_digest
    assert first_table.receipt.labels_digest != second_table.receipt.labels_digest
    assert first_table.receipt.source_fingerprint != second_table.receipt.source_fingerprint

    for selected_labels in ((2, 1), (1, 1)):
        selected_source = _FakeSource(
            path, rows,
            frames={label: np.ones((3, 4)) for label in rows},
        )
        _install_source(monkeypatch, selected_source)
        selected_result = _public("run_roi_scan")(
            _roi_plan(
                table, (_simple_signal(),),
                selected_labels=selected_labels,
            )
        )
        _assert_terminal(
            selected_result, "REFUSED", "SOURCE_IDENTITY_MISMATCH",
        )
        assert selected_source.load_calls == []

    subset_source = _FakeSource(
        path, rows, frames={2: np.ones((3, 4))},
    )
    _install_source(monkeypatch, subset_source)
    subset = _public("run_roi_scan")(
        _roi_plan(table, (_simple_signal(),), selected_labels=(2,))
    )
    _assert_terminal(subset, "COMPLETED", "OK")
    assert subset.requested_labels == subset.completed_labels == (2,)

    # A raw member admitted through the public SPEC frame projection must stay
    # exact through the strict read and the terminal publication fence.
    import xrd_tools.io.spec as spec_io

    spec_path = tmp_path / "scan.spec"
    raw_path = tmp_path / "raw_0001.tif"
    spec_path.write_text("#F scan.spec\n", encoding="utf-8")
    raw_path.write_bytes(b"raw-before")
    monkeypatch.setattr(spec_io, "list_spec_scans", lambda _path: ["7.1"])

    def spec_source(*, mutate_raw=False):
        source = _FakeSource(
            spec_path, {1: {"theta": 1.0}},
            frames={1: np.ones((3, 4))},
        )
        source.kind = SourceKind.SPEC
        source.scan_key = "7.1"
        source.frame_for = lambda label: SimpleNamespace(
            metadata={"theta": 1.0}, source_path=raw_path,
            source_frame_index=0,
        )
        if mutate_raw:
            original_load = source.load_frame

            def mutating_load(label):
                value = original_load(label)
                replacement_raw = tmp_path / "raw-replacement.tif"
                replacement_raw.write_bytes(b"raw-after-is-different")
                os.replace(replacement_raw, raw_path)
                return value

            source.load_frame = mutating_load
        return source

    spec_plan = _public("MetadataTablePlan")(
        source=SourceSpec(
            spec_path, SourceKind.SPEC, options={"scan": "7.1"},
        )
    )
    _install_source(monkeypatch, spec_source())
    spec_table = _public("run_metadata_table")(spec_plan)
    _assert_terminal(spec_table, "COMPLETED", "OK")
    mutating_source = spec_source(mutate_raw=True)
    _install_source(monkeypatch, mutating_source)
    raw_race = _public("run_roi_preview")(_preview_plan(spec_table, 1))
    _assert_terminal(raw_race, "REFUSED", "SOURCE_REVISION_CHANGED")
    assert mutating_source.load_calls == [1]

    repeated = _FakeSource(
        spec_path, {1: {"theta": 1.0}, 2: {"theta": 2.0}},
    )
    repeated.kind = SourceKind.SPEC
    repeated.scan_key = "7.1"
    relative_raw = os.path.relpath(raw_path, Path.cwd())
    repeated.frame_for = lambda label: SimpleNamespace(
        metadata={"theta": float(label)}, source_path=relative_raw,
        source_frame_index=0,
    )
    real_revision = ops._path_revision
    raw_revision_calls = []

    def revision_spy(path):
        if Path(path).resolve(strict=False) == raw_path.resolve(strict=False):
            raw_revision_calls.append(Path(path))
        return real_revision(path)

    _install_source(monkeypatch, repeated)
    monkeypatch.setattr(ops, "_path_revision", revision_spy)
    repeated_table = _public("run_metadata_table")(spec_plan)
    _assert_terminal(repeated_table, "COMPLETED", "OK")
    assert repeated_table.labels == (1, 2)
    assert len(raw_revision_calls) == 3
    monkeypatch.setattr(ops, "_path_revision", real_revision)

    scan_drift = spec_source()
    scan_drift.scan_key = "7.2"
    _install_source(monkeypatch, scan_drift)
    drifted_scan = _public("run_metadata_table")(spec_plan)
    _assert_terminal(drifted_scan, "REFUSED", "SOURCE_IDENTITY_MISMATCH")
    assert scan_drift.metadata_calls == []

    import xrd_tools.io.nexus as nexus_io

    container = tmp_path / "replace-after-container-open.h5"
    container.write_bytes(b"before")
    monkeypatch.setattr(nexus_io, "list_entries", lambda _path: ["entry"])
    container_source = _FakeSource(container, {0: {"theta": 1.0}})
    container_source.kind = SourceKind.NEXUS_STACK
    container_source.descriptor = SimpleNamespace(
        kind=SourceKind.NEXUS_STACK, resolved_entry="entry",
    )

    def replace_during_open(_spec, **_kwargs):
        replacement = tmp_path / "replacement-container.h5"
        replacement.write_bytes(b"replacement-is-different")
        os.replace(replacement, container)
        return container_source

    monkeypatch.setattr(ops, "open_source", replace_during_open)
    container_race = _public("run_metadata_table")(
        _public("MetadataTablePlan")(
            source=SourceSpec(
                container, SourceKind.NEXUS_STACK, entry="entry",
            )
        )
    )
    _assert_terminal(container_race, "REFUSED", "SOURCE_REVISION_CHANGED")
    assert container_source.metadata_calls == []
    assert container_source.close_calls == 1


def test_p37_metadata_text_numeric_storage_is_detached_and_exactly_charged(
    monkeypatch, tmp_path,
):
    ops = _scan_ops()
    assert ops._CANONICAL_PREFIX == b"xrd_tools.analysis.canonical.v1\x00"
    assert ops._CANONICAL_TAGS == {
        type(None): 0x00,
        bool: 0x01,
        int: 0x02,
        float: 0x03,
        str: 0x04,
        bytes: 0x05,
        Path: 0x06,
        Enum: 0x07,
        tuple: 0x08,
        Mapping: 0x09,
        np.ndarray: 0x0A,
    }
    encode = ops._canonical_bytes
    prefix = b"xrd_tools.analysis.canonical.v1\x00"
    assert encode(None) == prefix + _expected_frame(0x00, b"")
    assert encode(False) == prefix + _expected_frame(0x01, b"\x00")
    assert encode(True) == prefix + _expected_frame(0x01, b"\x01")
    assert encode(0) == prefix + _expected_frame(0x02, b"\x00")
    assert encode(256) == prefix + _expected_frame(0x02, b"\x00\x01\x00")
    assert encode(-256) == prefix + _expected_frame(0x02, b"\x01\x01\x00")
    assert encode(1.5) == prefix + _expected_frame(0x03, struct.pack(">d", 1.5))
    assert encode("x") == prefix + _expected_frame(0x04, b"x")
    assert encode(b"x") == prefix + _expected_frame(0x05, b"x")
    assert encode(Path("x")) == prefix + _expected_frame(0x06, b"x")

    enum_identity = (
        f"{_CanonicalMode.__module__}.{_CanonicalMode.__qualname__}".encode("utf-8")
    )
    enum_payload = (
        _expected_frame(0x04, enum_identity)
        + _expected_frame(0x02, b"\x00\x02")
    )
    assert encode(_CanonicalMode.SELECTED) == prefix + _expected_frame(0x07, enum_payload)

    tuple_payload = struct.pack(">Q", 2) + _expected_frame(0x01, b"\x01") + _expected_frame(0x02, b"\x00\x02")
    assert encode([True, 2]) == prefix + _expected_frame(0x08, tuple_payload)
    map_payload = (
        struct.pack(">Q", 2)
        + _expected_frame(0x04, b"a") + _expected_frame(0x02, b"\x00\x01")
        + _expected_frame(0x04, b"b") + _expected_frame(0x02, b"\x00\x02")
    )
    assert encode({"b": 2, "a": 1}) == prefix + _expected_frame(0x09, map_payload)
    array = np.array([[1, 2]], dtype="<i2")
    array_payload = (
        _expected_frame(0x04, array.dtype.str.encode())
        + struct.pack(">Q", 2)
        + struct.pack(">Q", 1)
        + struct.pack(">Q", 2)
        + array.tobytes(order="C")
    )
    assert encode(array) == prefix + _expected_frame(0x0A, array_payload)
    assert ops._OUTPUT_QNAN_BITS == 0x7FF8000000000000
    missing = ops._missing_numeric_array(1)
    assert missing.dtype.str == "<f8"
    assert struct.unpack("<Q", missing.tobytes())[0] == 0x7FF8000000000000
    with pytest.raises(ops.InvalidCanonicalValue):
        encode(missing)
    assert encode(missing, allow_missing=True).startswith(prefix)
    other_nan = np.array([0x7FF8000000000001], dtype="<u8").view("<f8")
    with pytest.raises(ops.InvalidCanonicalValue):
        encode(other_nan, allow_missing=True)
    assert ops._canonical_charge({"a": 1}) == len(encode({"a": 1}))
    assert hashlib.sha256(encode(True)).hexdigest() != hashlib.sha256(encode(1)).hexdigest()

    for invalid in (
        object(), {"bad": object()}, {1: "bad"}, {1, 2}, iter((1, 2)),
        np.array([object()], dtype=object), np.array(["x"]), float("nan"),
        np.array([np.nan]), np.array([np.inf]),
        np.array([1.0 + complex(0.0, np.inf)]), np.float64(1.0),
    ):
        with pytest.raises(ops.InvalidCanonicalValue):
            encode(invalid)

    expected_charge = ops._canonical_charge(("stream", array))
    expected_digest = ops._digest(("stream", array))
    monkeypatch.setattr(
        ops, "_canonical_bytes",
        lambda *_a, **_k: pytest.fail("charge/hash rebuilt the full document"),
    )
    assert ops._canonical_charge(("stream", array)) == expected_charge
    assert ops._digest(("stream", array)) == expected_digest

    image = tmp_path / "observed.raw"
    sidecar = tmp_path / "observed.metadata"
    image.write_bytes(b"image")
    sidecar.write_text("motor=1\nlabel=old\nother=2\n", encoding="utf-8")
    from xrd_tools.io.metadata import read_image_metadata_observed

    real_observed = read_image_metadata_observed(
        image,
        meta_format="metadata",
        max_input_bytes=1024,
    )
    sidecar_stat = sidecar.stat()
    assert dict(real_observed.values) == {"motor": 1, "label": "old", "other": 2}
    assert real_observed.source_path == sidecar
    assert real_observed.source_revision == (
        sidecar_stat.st_mode,
        sidecar_stat.st_dev,
        sidecar_stat.st_ino,
        sidecar_stat.st_size,
        sidecar_stat.st_mtime_ns,
        sidecar_stat.st_ctime_ns,
    )
    revision = ops._path_revision(sidecar)
    observed_calls = []
    real_scan_observed = ops.read_image_metadata_observed

    def observed_spy(*args, **kwargs):
        observed_calls.append((args, kwargs))
        return real_scan_observed(*args, **kwargs)

    monkeypatch.setattr(
        ops,
        "read_image_metadata_observed",
        observed_spy,
    )
    observed_plan = _public("MetadataTablePlan")(
        source=SourceSpec(
            image,
            SourceKind.IMAGE_FILE,
            options={"metadata_format": "metadata"},
        )
    )
    result = _public("run_metadata_table")(observed_plan)
    _assert_terminal(result, "COMPLETED", "OK")
    assert len(observed_calls) == 1
    assert observed_calls[0][1]["max_input_bytes"] > 0
    assert _column_values(_column(result, "label")) == ("old",)
    assert result.receipt.schema_version == "analysis-source-v2"
    assert result.receipt.dependency_revisions == (
        (sidecar, sidecar.resolve(), revision),
        (tmp_path, tmp_path.resolve(), ops._path_revision(tmp_path)),
    )
    stable_requalification = _public("run_metadata_table_requalification")(
        _public("MetadataTableRequalificationPlan")(
            result.receipt, result.table_fingerprint,
        )
    )
    _assert_terminal(stable_requalification, "COMPLETED", "OK")
    assert len(result.receipt.source_fingerprint) == 64
    assert result.receipt.source_fingerprint == result.receipt.source_fingerprint.lower()
    assert result.storage_bytes == ops._canonical_charge(
        ops._metadata_table_storage_value(result)
    )

    detector_shape = [1, 2]
    nested_input = {"metadata_format": "metadata", "detector_shape": detector_shape}
    nested_plan = _public("MetadataTablePlan")(
        source=SourceSpec(image, SourceKind.IMAGE_FILE, options=nested_input)
    )
    nested_result = _public("run_metadata_table")(nested_plan)
    _assert_terminal(nested_result, "COMPLETED", "OK")
    detector_shape.append(3)
    nested_input["metadata_format"] = "txt"
    assert nested_result.receipt.source_spec.options["detector_shape"] == (1, 2)
    assert nested_result.receipt.source_spec.options["metadata_format"] == "metadata"
    with pytest.raises(TypeError):
        nested_result.receipt.source_spec.options["new"] = 1
    assert ImageMetadataRead({"x": 1}, sidecar).source_revision is None

    plot_plan = _public("ScanPlotPlan")(y=["motor"])
    assert plot_plan.y == ("motor",)
    preview_plan = _public("RoiPreviewPlan")(
        result.receipt, result.table_fingerprint, list(result.labels), result.labels[0],
    )
    assert preview_plan.labels == result.labels
    roi_plan = _public("RoiScanPlan")(
        result.receipt, result.table_fingerprint, list(result.labels),
        list(result.labels), [_simple_signal()],
    )
    assert roi_plan.labels == roi_plan.selected_labels == result.labels
    assert len(roi_plan.signals) == 1

    class _MustNotIterate:
        def __iter__(self):
            pytest.fail("forbidden outer iterable was consumed")

    for invalid in ({"motor"}, (value for value in ("motor",)), _MustNotIterate()):
        with pytest.raises(ops.InvalidCanonicalValue):
            _public("ScanPlotPlan")(y=invalid)
    for invalid in ({1}, (value for value in result.labels), _MustNotIterate()):
        with pytest.raises(ops.InvalidCanonicalValue):
            _public("RoiPreviewPlan")(
                result.receipt, result.table_fingerprint, invalid, result.labels[0],
            )
    for invalid in (True, np.int64(result.labels[0])):
        with pytest.raises(ops.InvalidCanonicalValue):
            _public("RoiPreviewPlan")(
                result.receipt, result.table_fingerprint, result.labels, invalid,
            )
    for field, invalid in (
        ("labels", _MustNotIterate()),
        ("selected_labels", (value for value in result.labels)),
        ("signals", _MustNotIterate()),
    ):
        values = dict(
            receipt=result.receipt, table_fingerprint=result.table_fingerprint,
            labels=result.labels, selected_labels=result.labels,
            signals=(_simple_signal(),),
        )
        values[field] = invalid
        with pytest.raises(ops.InvalidCanonicalValue):
            _public("RoiScanPlan")(**values)

    monkeypatch.setattr(
        ops,
        "read_image_metadata_observed",
        lambda *_a, **_k: SimpleNamespace(
            values=MappingProxyType({"motor": 1.0}),
            source_path=sidecar,
            source_revision=None,
        ),
    )
    absent_revision = _public("run_metadata_table")(observed_plan)
    _assert_terminal(absent_revision, "REFUSED", "SOURCE_REVISION_CHANGED")

    replacement = (
        revision[0], revision[1], revision[2] + 1, *revision[3:],
    )
    real_path_revision = ops._path_revision
    monkeypatch.setattr(
        ops,
        "read_image_metadata_observed",
        lambda *_a, **_k: SimpleNamespace(
            values=MappingProxyType({"motor": 1.0, "label": "old", "other": 2.0}),
            source_path=sidecar,
            source_revision=revision,
        ),
    )
    monkeypatch.setattr(
        ops,
        "_path_revision",
        lambda path: replacement if Path(path) == sidecar else real_path_revision(path),
    )
    raced = _public("run_metadata_table")(observed_plan)
    _assert_terminal(raced, "REFUSED", "SOURCE_REVISION_CHANGED")
    assert raced.receipt is None and raced.columns == ()

    monkeypatch.setattr(
        ops,
        "read_image_metadata_observed",
        lambda *_a, **_k: SimpleNamespace(
            values=MappingProxyType({"motor": 1.0, "label": "old", "other": 2.0}),
            source_path=sidecar,
            source_revision=revision,
        ),
    )
    monkeypatch.setattr(ops, "_path_revision", real_path_revision)
    replacement_sidecar = tmp_path / "replacement.metadata"
    replacement_sidecar.write_text(
        "motor=9\nlabel=new\nother=8\n", encoding="utf-8",
    )

    def replace_from_callback(*_args):
        os.replace(replacement_sidecar, sidecar)

    terminal_race = _public("run_metadata_table")(
        observed_plan, progress_callback=replace_from_callback,
    )
    _assert_terminal(terminal_race, "REFUSED", "SOURCE_REVISION_CHANGED")
    sidecar_drift = _public("run_metadata_table_requalification")(
        _public("MetadataTableRequalificationPlan")(
            result.receipt, result.table_fingerprint,
        )
    )
    _assert_terminal(sidecar_drift, "REFUSED", "SOURCE_REVISION_CHANGED")

    provider = _FakeSource(
        image,
        {0: {"motor": 1.0, "label": "old", "other": 2.0}},
        scanned_axes=("motor",),
    )
    _install_source(monkeypatch, provider)
    provider_result = _public("run_metadata_table")(observed_plan)
    _assert_terminal(provider_result, "COMPLETED", "OK")
    assert provider.metadata_calls == [0]
    assert provider_result.receipt.source_fingerprint != result.receipt.source_fingerprint
    assert provider_result.receipt.metadata_observation_modes == ("provider_only",)
    assert len(observed_calls) == 2

    recipe = np.array([1, 2], dtype="<i8")
    array_source = _FakeSource(image, {0: {"motor": 1.0}}, frames={0: np.ones((2, 2))})
    _install_source(monkeypatch, array_source)
    array_table = _public("run_metadata_table")(
        _public("MetadataTablePlan")(
            source=SourceSpec(
                image, SourceKind.IMAGE_FILE,
                options={"detector_shape": recipe},
            )
        )
    )
    _assert_terminal(array_table, "COMPLETED", "OK")
    recipe[:] = 99
    array_preview_source = _FakeSource(
        image, {0: {"motor": 1.0}}, frames={0: np.ones((2, 2))},
    )
    _install_source(monkeypatch, array_preview_source)
    array_preview = _public("run_roi_preview")(_preview_plan(array_table, 0))
    _assert_terminal(array_preview, "COMPLETED", "OK")
    np.testing.assert_array_equal(
        array_table.receipt.source_spec.options["detector_shape"], [1, 2],
    )

    from xrd_tools.sources.image import ImageFileSource, TiffSeriesSource

    off_image = ImageFileSource(
        image, metadata_format=None, frame_indices=(0,),
    )
    _install_source(monkeypatch, off_image)
    monkeypatch.setattr(
        ops, "read_image_metadata_observed",
        lambda *_a, **_k: pytest.fail("metadata-off Image observed a sidecar"),
    )
    off_image_result = _public("run_metadata_table")(
        _public("MetadataTablePlan")(
            source=SourceSpec(
                image, SourceKind.IMAGE_FILE,
                options={"metadata_format": None},
            )
        )
    )
    _assert_terminal(off_image_result, "COMPLETED", "OK")
    assert off_image_result.receipt.metadata_observation_modes == ("metadata_off",)

    members = (tmp_path / "series_0001.tif", tmp_path / "series_0002.tif")
    for member in members:
        member.write_bytes(b"member")
    admitted = tuple(
        (str(member.resolve()), "th", float(index))
        for index, member in enumerate(members, start=1)
    )
    off_tiff_source = TiffSeriesSource(
        members, metadata_format=None, admitted_motor_values=admitted,
    )
    _install_source(monkeypatch, off_tiff_source)
    off_tiff = _public("run_metadata_table")(
        _public("MetadataTablePlan")(
            source=SourceSpec(
                tmp_path, SourceKind.TIFF_SERIES,
                options={
                    "files": tuple(str(member) for member in members),
                    "metadata_format": None,
                    "admitted_motor_values": admitted,
                },
            )
        )
    )
    _assert_terminal(off_tiff, "COMPLETED", "OK")
    assert off_tiff.receipt.metadata_observation_modes == (
        "metadata_off", "metadata_off",
    )
    np.testing.assert_array_equal(_column(off_tiff, "th").numeric, [1.0, 2.0])
    tiff_calls = []

    def tiff_observed(path, *_args, **_kwargs):
        tiff_calls.append(Path(path))
        return ImageMetadataRead({"Th": 99.0, "other": 1.0}, None, None)

    tiff_source = TiffSeriesSource(
        members, metadata_format="txt", admitted_motor_values=admitted,
    )
    _install_source(monkeypatch, tiff_source)
    monkeypatch.setattr(ops, "read_image_metadata_observed", tiff_observed)
    tiff_result = _public("run_metadata_table")(
        _public("MetadataTablePlan")(
            source=SourceSpec(
                tmp_path, SourceKind.TIFF_SERIES,
                options={
                    "files": tuple(str(member) for member in members),
                    "admitted_motor_values": admitted,
                },
            )
        )
    )
    _assert_terminal(tiff_result, "COMPLETED", "OK")
    assert tiff_calls == list(members)
    assert tiff_result.receipt.metadata_observation_modes == (
        "same_read", "same_read",
    )
    assert "Th" not in {column.name for column in tiff_result.columns}
    np.testing.assert_array_equal(_column(tiff_result, "th").numeric, [1.0, 2.0])
    assert tuple(resolved for _lexical, resolved, _revision in
                 tiff_result.receipt.dependency_revisions) == tuple(
                     member.resolve() for member in members
                 )
    members[1].write_bytes(b"changed member")
    member_drift = _public("run_metadata_table_requalification")(
        _public("MetadataTableRequalificationPlan")(
            tiff_result.receipt, tiff_result.table_fingerprint,
        )
    )
    _assert_terminal(member_drift, "REFUSED", "SOURCE_REVISION_CHANGED")


def test_p37_metadata_receipt_fences_alias_absence_and_two_pass_drift(
    monkeypatch, tmp_path,
):
    ops = _scan_ops()
    image = tmp_path / "frame.raw"
    replacement = tmp_path / "replacement.raw"
    image.write_bytes(b"image")
    replacement.write_bytes(b"other")
    alias = tmp_path / "selected.raw"
    alias.symlink_to(image)

    alias_result = _public("run_metadata_table")(
        _public("MetadataTablePlan")(
            SourceSpec(
                alias, SourceKind.IMAGE_FILE,
                options={"metadata_format": None},
            )
        )
    )
    _assert_terminal(alias_result, "COMPLETED", "OK")
    assert alias_result.receipt.lexical_root == alias
    assert alias_result.receipt.resolved_root == image.resolve()
    alias.unlink()
    alias.symlink_to(replacement)
    alias_drift = _public("run_metadata_table_requalification")(
        _public("MetadataTableRequalificationPlan")(
            alias_result.receipt, alias_result.table_fingerprint,
        )
    )
    _assert_terminal(alias_drift, "REFUSED", "SOURCE_REVISION_CHANGED")

    absent = _public("run_metadata_table")(
        _public("MetadataTablePlan")(
            SourceSpec(
                image, SourceKind.IMAGE_FILE,
                options={"metadata_format": "txt"},
            )
        )
    )
    _assert_terminal(absent, "COMPLETED", "OK")
    assert any(
        lexical == tmp_path and resolved == tmp_path.resolve()
        for lexical, resolved, _revision in absent.receipt.dependency_revisions
    )
    sidecar = image.with_suffix(".txt")
    sidecar.write_text(
        "# Counters\nI0 = 3\n# Motors\nth = 4\n",
        encoding="utf-8",
    )
    appeared = _public("run_metadata_table_requalification")(
        _public("MetadataTableRequalificationPlan")(
            absent.receipt, absent.table_fingerprint,
        )
    )
    _assert_terminal(appeared, "REFUSED", "SOURCE_REVISION_CHANGED")

    present = _public("run_metadata_table")(
        _public("MetadataTablePlan")(
            SourceSpec(
                image, SourceKind.IMAGE_FILE,
                options={"metadata_format": "txt"},
            )
        )
    )
    _assert_terminal(present, "COMPLETED", "OK")
    dependency_count = 1 + len(present.receipt.dependency_revisions)

    def mutate_after_first_sweep(completed, total):
        assert total == 2 * dependency_count
        if completed == dependency_count:
            sidecar.write_text(
                "# Counters\nI0 = 30\n# Motors\nth = 40\n",
                encoding="utf-8",
            )

    terminal_drift = _public("run_metadata_table_requalification")(
        _public("MetadataTableRequalificationPlan")(
            present.receipt, present.table_fingerprint,
        ),
        progress_callback=mutate_after_first_sweep,
    )
    _assert_terminal(
        terminal_drift, "REFUSED", "SOURCE_REVISION_CHANGED",
    )


def test_p37_real_tiff_txt_series_populates_complete_metadata(tmp_path):
    members = tuple(tmp_path / f"scan_{index:04d}.tif" for index in (1, 2))
    for index, member in enumerate(members, 1):
        tifffile.imwrite(member, np.full((3, 4), index, dtype=np.uint16))
        member.with_suffix(".txt").write_text(
            f"# Counters\nI0 = {10 * index}\n"
            f"# Motors\nth = {index / 10}\n"
            "User: p37, time: Mon Jan 15 10:30:00 2024  # Temp\n",
            encoding="utf-8",
        )
    from xrd_tools.sources.selection import image_series_spec
    result = _public("run_metadata_table")(
        _public("MetadataTablePlan")(
            image_series_spec(members[0], metadata_format="txt")
        )
    )
    _assert_terminal(result, "COMPLETED", "OK")
    np.testing.assert_allclose(_column(result, "I0").numeric, [10.0, 20.0])
    np.testing.assert_allclose(_column(result, "th").numeric, [0.1, 0.2])


def test_p37_metadata_requalification_rejects_hidden_sidecar_rewrite(tmp_path):
    member = tmp_path / "scan_0001.tif"
    sidecar = member.with_suffix(".txt")
    tifffile.imwrite(member, np.ones((3, 4), dtype=np.uint16))
    original = (
        "# Counters\nI0 = 10\n# Motors\nth = 1\n"
        "User: p37, time: Mon Jan 15 10:30:00 2024  # Temp\n"
    )
    replacement = (
        "# Counters\nI0 = 90\n# Motors\nth = 9\n"
        "User: p37, time: Mon Jan 15 10:30:00 2024  # Temp\n"
    )
    assert len(original.encode("utf-8")) == len(replacement.encode("utf-8"))
    sidecar.write_text(original, encoding="utf-8")

    from xrd_tools.sources.selection import image_series_spec
    result = _public("run_metadata_table")(
        _public("MetadataTablePlan")(
            image_series_spec(member, metadata_format="txt")
        )
    )
    _assert_terminal(result, "COMPLETED", "OK")
    admitted = sidecar.stat()

    sidecar.write_text(replacement, encoding="utf-8")
    os.utime(sidecar, ns=(admitted.st_atime_ns, admitted.st_mtime_ns))
    changed = sidecar.stat()
    assert (
        changed.st_dev, changed.st_ino, changed.st_size, changed.st_mtime_ns,
    ) == (
        admitted.st_dev, admitted.st_ino, admitted.st_size,
        admitted.st_mtime_ns,
    )
    assert changed.st_ctime_ns != admitted.st_ctime_ns

    requalified = _public("run_metadata_table_requalification")(
        _public("MetadataTableRequalificationPlan")(
            result.receipt, result.table_fingerprint,
        )
    )
    _assert_terminal(requalified, "REFUSED", "SOURCE_REVISION_CHANGED")


def test_p37_metadata_requalification_rejects_hidden_member_rewrite(tmp_path):
    members = tuple(tmp_path / f"scan_{index:04d}.tif" for index in (1, 2))
    for index, member in enumerate(members, 1):
        tifffile.imwrite(member, np.full((3, 4), index, dtype=np.uint16))

    from xrd_tools.sources.selection import image_series_spec
    result = _public("run_metadata_table")(
        _public("MetadataTablePlan")(
            image_series_spec(members[0], metadata_format=None)
        )
    )
    _assert_terminal(result, "COMPLETED", "OK")
    member = members[1]
    admitted = member.stat()
    payload = member.read_bytes()
    replacement = payload[:-1] + bytes((payload[-1] ^ 1,))

    member.write_bytes(replacement)
    os.utime(member, ns=(admitted.st_atime_ns, admitted.st_mtime_ns))
    changed = member.stat()
    assert (
        changed.st_dev, changed.st_ino, changed.st_size, changed.st_mtime_ns,
    ) == (
        admitted.st_dev, admitted.st_ino, admitted.st_size,
        admitted.st_mtime_ns,
    )
    assert changed.st_ctime_ns != admitted.st_ctime_ns

    requalified = _public("run_metadata_table_requalification")(
        _public("MetadataTableRequalificationPlan")(
            result.receipt, result.table_fingerprint,
        )
    )
    _assert_terminal(requalified, "REFUSED", "SOURCE_REVISION_CHANGED")


def test_p37_auto_metadata_receipt_tracks_rejected_candidates_and_atomic_absence(
    monkeypatch, tmp_path,
):
    ops = _scan_ops()
    image = tmp_path / "auto.raw"
    image.write_bytes(b"pixels")
    candidate = image.with_suffix(".txt")
    candidate.write_text("not_metadata=1\n", encoding="utf-8")
    plan = _public("MetadataTablePlan")(
        SourceSpec(
            image, SourceKind.IMAGE_FILE,
            options={"metadata_format": "auto"},
        )
    )

    empty = _public("run_metadata_table")(plan)
    _assert_terminal(empty, "COMPLETED", "OK")
    assert candidate in {
        lexical
        for lexical, _resolved, _revision
        in empty.receipt.dependency_revisions
    }
    candidate.write_text(
        "motor=1\ncounter=2\nsample=valid\n", encoding="utf-8",
    )
    stale = _public("run_metadata_table_requalification")(
        _public("MetadataTableRequalificationPlan")(
            empty.receipt, empty.table_fingerprint,
        )
    )
    _assert_terminal(stale, "REFUSED", "SOURCE_REVISION_CHANGED")

    race_root = tmp_path / "absence-race"
    race_root.mkdir()
    raced_image = race_root / "frame.raw"
    raced_image.write_bytes(b"pixels")
    raced_sidecar = raced_image.with_suffix(".txt")
    directory_revision = ops._path_revision(race_root)
    assert directory_revision is not None

    def create_after_absence(*_args, **_kwargs):
        raced_sidecar.write_text(
            "# Counters\nI0 = 3\n# Motors\nth = 4\n",
            encoding="utf-8",
        )
        return ImageMetadataRead(
            {}, None,
            discovery_revisions=((race_root, directory_revision),),
        )

    real_revision = ops._path_revision
    changed_directory_revision = (
        *directory_revision[:-1], directory_revision[-1] + 1,
    )
    monkeypatch.setattr(
        ops, "read_image_metadata_observed", create_after_absence,
    )
    monkeypatch.setattr(
        ops, "_path_revision",
        lambda path: (
            changed_directory_revision
            if Path(path) == race_root else real_revision(path)
        ),
    )
    raced = _public("run_metadata_table")(
        _public("MetadataTablePlan")(
            SourceSpec(
                raced_image, SourceKind.IMAGE_FILE,
                options={"metadata_format": "txt"},
            )
        )
    )
    _assert_terminal(raced, "REFUSED", "SOURCE_REVISION_CHANGED")


def test_p37_roi_preview_and_total_result_obey_byte_and_identity_caps(
    monkeypatch, tmp_path,
):
    ops = _scan_ops()
    assert ops._MAX_PREVIEW_BYTES == 64 * 1024 * 1024
    assert ops._MAX_ROI_BYTES == 64 * 1024 * 1024
    rows = {1: {}}
    table, _ = _run_table(monkeypatch, tmp_path, rows)
    source = _FakeSource(tmp_path / "scan_0001.tif", rows, frames={1: np.ones((3, 4))})
    _install_source(monkeypatch, source)
    preview = _public("run_roi_preview")(_preview_plan(table, 1))
    _assert_terminal(preview, "COMPLETED", "OK")
    assert preview.table_fingerprint == table.table_fingerprint
    assert preview.receipt.source_fingerprint == table.receipt.source_fingerprint
    monkeypatch.setattr(ops, "_MAX_PREVIEW_BYTES", preview.storage_bytes)
    source = _FakeSource(tmp_path / "scan_0001.tif", rows, frames={1: np.ones((3, 4))})
    _install_source(monkeypatch, source)
    exact_preview = _public("run_roi_preview")(_preview_plan(table, 1))
    _assert_terminal(exact_preview, "COMPLETED", "OK")
    assert exact_preview.storage_bytes == preview.storage_bytes
    monkeypatch.setattr(ops, "_MAX_PREVIEW_BYTES", preview.storage_bytes - 1)
    source = _FakeSource(tmp_path / "scan_0001.tif", rows, frames={1: np.ones((3, 4))})
    _install_source(monkeypatch, source)
    over = _public("run_roi_preview")(_preview_plan(table, 1))
    _assert_terminal(over, "REFUSED", "ROI_PREVIEW_LIMIT_EXCEEDED")
    assert over.image is None

    for count, code in ((0, "ROI_SIGNAL_COUNT_INVALID"), (6, "ROI_SIGNAL_COUNT_INVALID")):
        source = _FakeSource(tmp_path / "scan_0001.tif", rows, frames={1: np.ones((3, 4))})
        _install_source(monkeypatch, source)
        result = _public("run_roi_scan")(_roi_plan(table, tuple(_simple_signal(str(i)) for i in range(count))))
        _assert_terminal(result, "REFUSED", code)
        assert source.load_calls == []
    five = tuple(_simple_signal(str(index)) for index in range(5))
    source = _FakeSource(tmp_path / "scan_0001.tif", rows, frames={1: np.ones((3, 4))})
    _install_source(monkeypatch, source)
    boundary = _public("run_roi_scan")(_roi_plan(table, five))
    _assert_terminal(boundary, "COMPLETED", "OK")
    monkeypatch.setattr(ops, "_MAX_ROI_BYTES", boundary.storage_bytes)
    source = _FakeSource(tmp_path / "scan_0001.tif", rows, frames={1: np.ones((3, 4))})
    _install_source(monkeypatch, source)
    exact_boundary = _public("run_roi_scan")(_roi_plan(table, five))
    _assert_terminal(exact_boundary, "COMPLETED", "OK")
    assert exact_boundary.storage_bytes == boundary.storage_bytes
    monkeypatch.setattr(ops, "_MAX_ROI_BYTES", boundary.storage_bytes - 1)
    source = _FakeSource(tmp_path / "scan_0001.tif", rows, frames={1: np.ones((3, 4))})
    _install_source(monkeypatch, source)
    one_over = _public("run_roi_scan")(_roi_plan(table, five))
    _assert_terminal(one_over, "REFUSED", "ROI_RESULT_LIMIT_EXCEEDED")
    assert one_over.signal_values == ()


def test_p37_public_cancel_is_exact_threading_event_and_progress_failure_is_inert(
    monkeypatch, tmp_path,
):
    rows = {
        1: {"Photod": 1.0},
        2: {"Photod": 2.0},
        3: {"Photod": 3.0},
    }
    table, _ = _run_table(monkeypatch, tmp_path, rows)
    plan = _roi_plan(table, (_simple_signal(),))
    for invalid in (False, lambda: False, SimpleNamespace(is_set=lambda: False)):
        terminals = (
            _public("run_metadata_table")(
                _metadata_plan(tmp_path / "scan_0001.tif"),
                cancel_token=invalid,
            ),
            _public("run_scan_plot")(
                _public("ScanPlotPlan")(), table, cancel_token=invalid,
            ),
            _public("run_roi_preview")(
                _preview_plan(table, 1), cancel_token=invalid,
            ),
            _public("run_roi_scan")(plan, cancel_token=invalid),
        )
        for result in terminals:
            _assert_terminal(result, "REFUSED", "INVALID_CANCEL_TOKEN")

    already = threading.Event()
    already.set()
    result = _public("run_roi_scan")(plan, cancel_token=already)
    _assert_terminal(result, "CANCELLED", "CANCELLED")

    token = threading.Event()
    source = _FakeSource(
        tmp_path / "scan_0001.tif",
        rows,
        frames={label: np.ones((3, 4)) for label in rows},
    )
    _install_source(monkeypatch, source)

    def progress(done, _total):
        if done == 1:
            token.set()
        raise RuntimeError("presentation callback is inert")

    cancelled = _public("run_roi_scan")(plan, cancel_token=token, progress_callback=progress)
    _assert_terminal(cancelled, "CANCELLED", "CANCELLED")
    assert cancelled.completed_labels == (1,)
    assert source.load_calls == [1]
    assert source.close_calls == 1

    final_token = threading.Event()
    final_source = _FakeSource(
        tmp_path / "scan_0001.tif",
        rows,
        frames={label: np.ones((3, 4)) for label in rows},
    )
    _install_source(monkeypatch, final_source)

    def final_progress(done, total):
        if done == total:
            final_token.set()

    final_cancel = _public("run_roi_scan")(
        plan,
        cancel_token=final_token,
        progress_callback=final_progress,
    )
    _assert_terminal(final_cancel, "CANCELLED", "CANCELLED")
    assert final_cancel.completed_labels == (1, 2, 3)
    assert final_source.close_calls == 1

    metadata_token = threading.Event()
    metadata_source = _FakeSource(tmp_path / "scan_0001.tif", rows)
    _install_source(monkeypatch, metadata_source)
    metadata_cancel = _public("run_metadata_table")(
        _metadata_plan(tmp_path / "scan_0001.tif"),
        cancel_token=metadata_token,
        progress_callback=lambda *_a: metadata_token.set(),
    )
    _assert_terminal(metadata_cancel, "CANCELLED", "CANCELLED")
    assert metadata_source.close_calls == 1

    plot_token = threading.Event()
    plot_cancel = _public("run_scan_plot")(
        _public("ScanPlotPlan")(), table, cancel_token=plot_token,
        progress_callback=lambda *_a: plot_token.set(),
    )
    _assert_terminal(plot_cancel, "CANCELLED", "CANCELLED")

    preview_token = threading.Event()
    preview_source = _FakeSource(
        tmp_path / "scan_0001.tif", rows,
        frames={label: np.ones((3, 4)) for label in rows},
    )
    _install_source(monkeypatch, preview_source)
    preview_cancel = _public("run_roi_preview")(
        _preview_plan(table, 1), cancel_token=preview_token,
        progress_callback=lambda *_a: preview_token.set(),
    )
    _assert_terminal(preview_cancel, "CANCELLED", "CANCELLED")
    assert preview_source.load_calls == [1]
    assert preview_source.close_calls == 1


def test_p37_results_recursively_detach_without_object_arrays_or_backend_handles(
    monkeypatch, tmp_path,
):
    ops = _scan_ops()
    input_buffer = np.arange(6.0)
    detached_buffer = ops._owned(input_buffer)
    _assert_owned_readonly(detached_buffer)
    assert detached_buffer is not input_buffer
    assert not np.shares_memory(detached_buffer, input_buffer)
    input_buffer[:] = -1.0
    np.testing.assert_array_equal(detached_buffer, np.arange(6.0))

    rows = {1: {"text": "a", "numeric": 1.0}, 2: {"text": "b", "numeric": 2.0}}
    table, source = _run_table(monkeypatch, tmp_path, rows)
    source._rows[1]["text"] = "mutated"
    assert _column_values(_column(table, "text")) == ("a", "b")
    for array in _all_arrays(table):
        _assert_owned_readonly(array)

    roi_source = _FakeSource(
        tmp_path / "scan_0001.tif",
        rows,
        frames={1: np.ones((3, 4)), 2: np.ones((3, 4)) * 2},
    )
    _install_source(monkeypatch, roi_source)
    roi = _public("run_roi_scan")(_roi_plan(table, (_simple_signal(),)))
    for array in _all_arrays(roi):
        _assert_owned_readonly(array)
    forbidden_types = (threading.Event, BaseException)
    for field in dataclasses.fields(roi):
        assert not isinstance(getattr(roi, field.name), forbidden_types)
    assert not _contains_identity(roi, roi_source)


def test_p37_has_zero_writer_h23_mutation_or_persistence_reachability(
    monkeypatch, tmp_path,
):
    ops = _scan_ops()
    source_text = inspect.getsource(ops)
    for forbidden in (
        "NexusRecordWriter", "OutputTransaction", "for_existing_replacement",
        "h5py.File", "export_", "write_batch", "average_closed_v1",
        "OperationSlot", "PyQt", "PySide",
    ):
        assert forbidden not in source_text
    assert {
        name for name in analysis.__all__
        if name in _PUBLIC_RUNNERS
    } == _PUBLIC_RUNNERS
    assert not any(
        name.startswith(("run_displayed_peak_", "run_displayed_phase_"))
        and name not in _PUBLIC_RUNNERS
        for name in analysis.__all__
    )

    result, source = _run_table(monkeypatch, tmp_path, {1: {"Photod": 1.0}})
    _assert_terminal(result, "COMPLETED", "OK")
    assert source.load_calls == []
    # Other tests legitimately import writers. Prove this read-only operation
    # remains writer-free in a fresh interpreter, through a real TIFF source.
    probe = '''
import sys
from pathlib import Path
import numpy as np
import tifffile
from xrd_tools.analysis import MetadataTablePlan, run_metadata_table
source = Path(sys.argv[1]) / "metadata_0001.tif"
tifffile.imwrite(source, np.ones((2, 2), dtype=np.uint16))
result = run_metadata_table(MetadataTablePlan(source=source, selection="image_series"))
assert result.disposition.value == "completed", result
assert not any(name.startswith(("xrd_tools.io.record_writer", "xrd_tools.io.output_transaction"))
               for name in sys.modules)
'''
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(tmp_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
