"""Fresh worker-owned run-scan and GUI-admission ownership oracles."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pyqtgraph")

from tests.xdart.test_o3n_execution_owner import (  # noqa: E402,F401
    _select_nexus,
    _started,
    qapp,
    widget,
)

from xrd_tools.core.containers import IntegrationResult1D  # noqa: E402


def _result_1d(n=16, scale=1.0):
    radial = np.linspace(1.0, 5.0, n)
    return IntegrationResult1D(
        radial=radial,
        intensity=np.full(n, scale, dtype=float),
        sigma=np.sqrt(np.full(n, scale, dtype=float)),
        unit="q_A^-1",
    )


def _seed_display_scan(widget, labels=(3, 99)):
    """Put REAL prior state on the display scan the worker used to reuse."""
    scan = widget.scan
    for label in labels:
        scan.scan_data.loc[label, "seed"] = float(label)
    scan._seeded_cache_marker = "A-run-cache"
    return scan


# --------------------------------------------------------------------------- #
# §16.1 — a fresh per-run LiveScan, not a renamed display singleton
# --------------------------------------------------------------------------- #

def test_the_run_scan_is_not_the_display_scan(widget, tmp_path, monkeypatch):
    wrangler = _select_nexus(widget)
    display = _seed_display_scan(widget)
    src, out, thread = _started(wrangler, tmp_path, monkeypatch)

    run_scan = thread._initialize_scan(Path(src).stem)

    assert run_scan is not display, (
        "the worker renamed the display singleton instead of owning a run scan")
    assert thread._active_scan is run_scan


def test_the_run_scan_carries_no_prior_frames_or_scan_data(
        widget, tmp_path, monkeypatch):
    """§16.1: seeded labels 3 and 99 must not survive into the new run."""
    wrangler = _select_nexus(widget)
    display = _seed_display_scan(widget)
    src, out, thread = _started(wrangler, tmp_path, monkeypatch)

    run_scan = thread._initialize_scan(Path(src).stem)

    assert list(run_scan.scan_data.index) == []
    assert list(run_scan.frames.index) == []
    assert not hasattr(run_scan, "_seeded_cache_marker")
    # the display scan is left ALONE -- the worker neither executes on it nor
    # resets it.
    assert sorted(display.scan_data.index) == [3, 99]
    assert display._seeded_cache_marker == "A-run-cache"


def test_a_second_overwrite_run_starts_from_an_empty_run_scan(
        widget, tmp_path, monkeypatch):
    """Two-run row: A's frames/scan_data/caches/writer cursor must not reach B."""
    wrangler = _select_nexus(widget)
    widget.controls.set_write_mode("Overwrite")
    src, out, thread = _started(wrangler, tmp_path, monkeypatch)

    scan_a = thread._initialize_scan(Path(src).stem)
    scan_a.add_frame(frame=_fake_frame(7), calculate=False, update=True,
                     get_sd=True, set_mg=False, static=True, batch_save=True)
    scan_a._run_a_marker = True

    # a second accepted Start -> a second run.  The first run finishes through
    # the production run-end owner; without that, Start is the case-6
    # zero-delta refusal and the row would pass against the SAME thread.
    widget._exit_run_state(widget._new_projection_receipt())
    wrangler.start()
    thread_b = wrangler.thread
    assert thread_b is not thread, "the second Start was refused, not admitted"
    scan_b = thread_b._initialize_scan(Path(src).stem)

    assert scan_b is not scan_a
    assert list(scan_b.frames.index) == []
    assert list(scan_b.scan_data.index) == []
    assert not hasattr(scan_b, "_run_a_marker")


def _fake_frame(idx, scale=1.0):
    from xdart.modules.live import LiveFrame

    frame = LiveFrame(idx, np.zeros((4, 4), dtype=np.float32), static=True)
    frame.int_1d = _result_1d(scale=scale)
    frame.skip_map_raw = True
    return frame


# --------------------------------------------------------------------------- #
# §16.7 mutation row 13 — a bounded AST pin, not a taint analyzer
# --------------------------------------------------------------------------- #

_O3N_MODULES = (
    "test_o3n_nexus_freeze_identity.py",
    "test_o3n_depth.py",
    "test_o3n_malformed_inputs.py",
    "test_o3n_execution_owner.py",
    "test_o3nr1_run_scan.py",
    "test_o3nr1_exact_review.py",
    "test_o3nr1_totality.py",
    "test_o3nr2_prepared_execution.py",
    "test_o3nr3_execution_transaction.py",
)


def test_no_o3n_row_assigns_scan_data_file_by_hand():
    """The manual repoint hid §15.1 once; it may not come back."""
    here = Path(__file__).resolve().parent
    offenders = []
    for name in _O3N_MODULES:
        path = here / name
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (isinstance(target, ast.Attribute)
                        and target.attr == "data_file"):
                    offenders.append(f"{name}:{node.lineno}")
    assert offenders == [], (
        f"a test-side `.data_file =` assignment reappeared: {offenders}")


def test_gui_admission_asks_no_filesystem_or_hdf5_questions():
    """§16.6, as a bounded fact about the production tree (rule 9).

    GUI admission owns pure shape/value validation only.  Filesystem, link and
    HDF5 facts belong to the worker: on a beamline network path a "cheap" stat
    is not cheap, and a refusal that needs I/O must not run on the GUI thread.
    """
    import inspect
    import textwrap

    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler import (
        nexusWrangler,
    )

    tree = ast.parse(textwrap.dedent(
        inspect.getsource(nexusWrangler._validate_admissible_source)))
    body = tree.body[0].body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)):
        body = body[1:]                      # the docstring EXPLAINS the rule
    source = "\n".join(ast.unparse(node) for node in body)
    forbidden = ("is_file(", "is_dir(", "exists(", "os.stat", "h5py",
                 "check_output_not_source", "open_nexus_image_stack")
    found = [token for token in forbidden if token in source]
    assert found == [], (
        f"GUI admission asks filesystem/HDF5 questions: {found}")
