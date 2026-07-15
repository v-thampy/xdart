"""Validate and execute the public notebook set in deterministic smoke mode."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
EXPECTED = [
    "01_batch_integration.ipynb",
    "02_multigeometry_stitching.ipynb",
    "03_phase_and_peak_fitting.ipynb",
    "04_batch_phase_fitting.ipynb",
    "05_sin2psi_analysis.ipynb",
    "06_headless_reduction_pipeline.ipynb",
    "07_reading_processed_nxs.ipynb",
    "08_time_resolved_xrd_analysis.ipynb",
    "09_reciprocal_space_mapping.ipynb",
]
FORBIDDEN_IMPORT = re.compile(
    r"^\s*(?:from\s+(?:xdart|ssrl_xrd_tools)(?:\.|\s)|import\s+(?:xdart|ssrl_xrd_tools)(?:\.|\s|$))",
    re.MULTILINE,
)
PRIVATE_PATH = re.compile(r"/(?:Users|home)/")


def validate_notebook(path: Path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    for number, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        compile(cell.source, f"{path.name}:cell-{number}", "exec")
        if cell.execution_count is not None or cell.outputs:
            raise AssertionError(f"{path.name}:cell-{number} is not clean")
        if FORBIDDEN_IMPORT.search(cell.source):
            raise AssertionError(f"{path.name}:cell-{number} imports xdart or legacy code")
        if PRIVATE_PATH.search(cell.source):
            raise AssertionError(f"{path.name}:cell-{number} embeds a private path")
    return nb


def regenerate_and_check() -> None:
    """Build from source and reject generated JSON drift before execution."""
    repository = ROOT.parents[1]
    subprocess.run([sys.executable, str(ROOT / "_build_notebooks.py")], check=True)
    result = subprocess.run(
        ["git", "diff", "--exit-code", "--", "examples/notebooks"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise AssertionError("generated notebooks drifted from _build_notebooks.py\n" + result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--only", choices=EXPECTED, action="append", help="Execute one named notebook; validation still covers the full public set.")
    parser.add_argument("--real-data", action="store_true", help="Execute selected notebook(s) with XDART_TEST_DATA instead of synthetic smoke inputs.")
    args = parser.parse_args()

    regenerate_and_check()
    actual = sorted(path.name for path in ROOT.glob("*.ipynb"))
    if actual != EXPECTED:
        raise AssertionError(f"Expected {EXPECTED}; found {actual}")
    notebooks = [(ROOT / name, validate_notebook(ROOT / name)) for name in EXPECTED]
    if args.validate_only:
        print(f"Validated {len(notebooks)} clean public notebooks")
        return 0

    if args.real_data:
        if not os.environ.get("XDART_TEST_DATA"):
            raise AssertionError("--real-data requires XDART_TEST_DATA to name the public test-data root")
        os.environ["XDART_NOTEBOOK_SMOKE"] = "0"
        os.environ["XDART_NOTEBOOK_AUTORUN"] = "1"
    else:
        os.environ["XDART_NOTEBOOK_SMOKE"] = "1"
        os.environ.pop("XDART_TEST_DATA", None)
    selected = args.only or EXPECTED
    with tempfile.TemporaryDirectory(prefix="xdart-notebook-smoke-") as tmp:
        for path, nb in notebooks:
            if path.name not in selected:
                continue
            started = time.perf_counter()
            # Execute an in-memory notebook so checked-in JSON remains clean.
            NotebookClient(
                nb,
                timeout=args.timeout,
                kernel_name="python3",
                resources={"metadata": {"path": tmp}},
            ).execute()
            elapsed = time.perf_counter() - started
            print(f"{path.name}: {elapsed:.1f}s")
    print(f"Executed {len(selected)} {'real-data' if args.real_data else 'deterministic public'} notebook(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
