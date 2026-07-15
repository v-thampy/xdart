"""Validate and execute the public notebook set in deterministic smoke mode."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    actual = sorted(path.name for path in ROOT.glob("*.ipynb"))
    if actual != EXPECTED:
        raise AssertionError(f"Expected {EXPECTED}; found {actual}")
    notebooks = [(ROOT / name, validate_notebook(ROOT / name)) for name in EXPECTED]
    if args.validate_only:
        print(f"Validated {len(notebooks)} clean public notebooks")
        return 0

    os.environ["XDART_NOTEBOOK_SMOKE"] = "1"
    os.environ.pop("XDART_TEST_DATA", None)
    with tempfile.TemporaryDirectory(prefix="xdart-notebook-smoke-") as tmp:
        for path, nb in notebooks:
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
    print(f"Executed {len(notebooks)} deterministic public notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
