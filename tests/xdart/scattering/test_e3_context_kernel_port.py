"""Exact-port oracle for the accepted canonical display-context kernel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from xdart.modules.display_context import (
    AcquisitionContext,
    BrowseContext,
    ContextKind,
    DisplayBindings,
    DisplayContextError,
    DisplaySelection,
    new_context_token,
)


# The E3-accepted kernel (7c379710 / blob 73449efe) was superseded by the
# accepted E4-S canonical-first revision that composes the typed shared
# hydration purpose/token; the pinned object below is the byte-exact accepted
# E4-S kernel ported at E4-R.
_CANONICAL_COMMIT = "72b7b08efd4d24df0f74aa462a4b1766a97be6b2"
_CANONICAL_BLOB = "3329391554910eb2e062da0a6c2afe3415db398c"
_CANONICAL_SHA256 = (
    "7664a4c0979336015a0779b568461e2953ec04295a503c04cc711521594da825"
)
_MODULE = (
    Path(__file__).parents[3]
    / "src"
    / "xdart"
    / "modules"
    / "display_context.py"
)
_INVENTORY = Path(__file__).with_name("e3_c0_dependency_inventory.json")


class _Store:
    def __init__(self) -> None:
        self.clear_calls = 0

    def clear(self) -> None:
        self.clear_calls += 1


def _acquisition() -> AcquisitionContext:
    return AcquisitionContext(
        context_token=new_context_token(ContextKind.ACQUISITION),
        run_configuration=object(),
        config_generation=1,
        config_fingerprint="frozen",
        run_scan_key="scan.a",
        source_path="/data/scan.a.nxs",
        scan=object(),
        frame=object(),
        frame_ids=[],
        frames={},
        viewer_rows_1d={},
        viewer_rows_2d={},
        publication_store=_Store(),
    )


def test_port_is_byte_identical_to_the_accepted_canonical_blob() -> None:
    payload = _MODULE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == _CANONICAL_SHA256
    blob = subprocess.run(
        ["git", "hash-object", str(_MODULE)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert blob == _CANONICAL_BLOB
    canonical = subprocess.run(
        [
            "git",
            "rev-parse",
            f"{_CANONICAL_COMMIT}:src/xdart/modules/display_context.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert canonical == blob


def test_dependency_inventory_names_the_accepted_canonical_object() -> None:
    inventory = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    assert inventory["canonical_context_commit"] == _CANONICAL_COMMIT
    assert inventory["canonical_context_blob"] == _CANONICAL_BLOB
    assert inventory["canonical_context_sha256"] == _CANONICAL_SHA256


def test_port_import_is_headless_in_a_fresh_interpreter() -> None:
    script = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("e3_context", r"{_MODULE}")
module = importlib.util.module_from_spec(spec)
sys.modules["e3_context"] = module
spec.loader.exec_module(module)
forbidden = (
    "PySide6", "PyQt5", "PyQt6", "qtpy", "pyqtgraph", "h5py",
    "pyFAI", "fabio", "numpy", "pandas", "xdart",
)
assert not any(
    name == root or name.startswith(root + ".")
    for name in sys.modules
    for root in forbidden
), sorted(name for name in sys.modules if name.split(".", 1)[0] in forbidden)
loaded_xrd = {{
    name for name in sys.modules
    if name == "xrd_tools" or name.startswith("xrd_tools.")
}}
assert loaded_xrd <= {{
    "xrd_tools", "xrd_tools.session", "xrd_tools.session.hydration"
}}, sorted(loaded_xrd)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_canonical_context_and_selection_contract_is_directly_usable() -> None:
    acquisition = _acquisition()
    store = _Store()
    acquisition.adopt_record_store(store)
    selection = DisplaySelection.for_context(acquisition, 7)

    assert selection.names(acquisition)
    assert selection.owner == acquisition.hydration_owner
    assert acquisition.display_bindings() == DisplayBindings(
        acquisition.scan,
        acquisition.frame,
        acquisition.frame_ids,
        acquisition.frames,
        acquisition.viewer_rows_1d,
        acquisition.viewer_rows_2d,
        store,
        acquisition.publication_store,
    )
    with pytest.raises(DisplayContextError):
        acquisition.scan = object()


def test_browse_request_identity_and_release_are_exact() -> None:
    token = new_context_token(ContextKind.BROWSE)

    class Request:
        context_token = token
        load_generation = 2
        scan = object()
        fname = "/data/browse.b.nxs"
        scan_name = "browse.b"
        operation = None

    publication = _Store()
    browse = BrowseContext(
        context_token=token,
        load_generation=2,
        operation=None,
        requested_path=Request.fname,
        scan_key=Request.scan_name,
        scan=Request.scan,
        frame=object(),
        frame_ids=[],
        frames={},
        viewer_rows_1d={},
        viewer_rows_2d={},
        publication_store=publication,
        record_store=_Store(),
    )
    request = Request()
    browse.adopt_load_request(request)
    assert browse.admits(request)
    assert not browse.admits(Request())

    browse.release()
    assert browse.released
    assert publication.clear_calls == 1
    assert not browse.admits(request)
