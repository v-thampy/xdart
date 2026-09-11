"""Static owner/port guards for the parallel E3 context packet."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCATTERING = ROOT / "src/xdart/gui/tabs/scattering"
NEW_PRODUCTION = (
    SCATTERING / "browse_values.py",
    SCATTERING / "context_controller.py",
    SCATTERING / "context_projection.py",
    SCATTERING / "context_runtime.py",
    SCATTERING / "context_values.py",
    SCATTERING / "adapters/browse_loader.py",
)


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"))


def test_context_packet_exists_and_does_not_import_legacy_gui_owners():
    forbidden = {
        "staticWidget",
        "H5Viewer",
        "DisplayContextOperation",
        "FileTask",
        "BrowseLoadTask",
        "ParameterTree",
    }
    for path in NEW_PRODUCTION:
        tree = _tree(path)
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        attrs = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }
        assert not forbidden.intersection(names | attrs), path


def test_context_events_and_outcomes_are_values_only():
    forbidden = {
        "ndarray",
        "FrameRecordStore",
        "PublicationStore",
        "AcquisitionContext",
        "BrowseContext",
        "RunDisplayState",
        "QObject",
    }
    for path in (
        SCATTERING / "browse_values.py",
        SCATTERING / "events.py",
    ):
        for node in _tree(path).body:
            if not isinstance(node, ast.ClassDef):
                continue
            # This synchronous owner capture is deliberately not an event or
            # worker outcome. Its exact context is required for reintegration.
            if path.name == "browse_values.py" and node.name == "LoadedBrowseCapture":
                continue
            annotations = {
                name.id
                for field in node.body if isinstance(field, ast.AnnAssign)
                for name in ast.walk(field.annotation) if isinstance(name, ast.Name)
            }
            assert not forbidden.intersection(annotations), (path, node.name)


def test_projection_has_no_file_or_source_io_route():
    source = (SCATTERING / "context_projection.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    forbidden_calls = {
        "open",
        "read_frame_record",
        "read_frame_records",
        "load_processed_raw_or_thumbnail",
    }
    calls = {
        (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else ""
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not (calls & forbidden_calls)


def test_controller_has_no_store_provider_or_value_cache():
    source = (SCATTERING / "context_controller.py").read_text(
        encoding="utf-8"
    )
    assert "store_provider" not in source
    assert "publication_provider" not in source
    assert "payload_cache" not in source


def test_executor_never_reads_browse_or_selection_state():
    source = (
        SCATTERING / "adapters/run_executor.py"
    ).read_text(encoding="utf-8")
    assert "BrowseContext" not in source
    assert "DisplaySelection" not in source
