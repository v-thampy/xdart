#!/usr/bin/env python3
"""Release pre-flight for xdart (Phase 1b of the greenfield plan).

The two-repo era enforced publish order and version floors by runbook;
the monorepo enforces them by script.  Run everything before tagging:

    python scripts/release.py check          # all pre-flight checks
    python scripts/release.py check vX.Y.Z   # + tag consistency
    python scripts/release.py build          # checks, then build + twine

Checks (each prints PASS/FAIL; any FAIL exits 1):

  version    pyproject version == importlib version of the installed
             editable dist == xdart.__version__; if a tag is given (or
             HEAD is tagged v*), it must match.
  clean      no uncommitted changes (warn-only unless --strict-tree).
  schema     the persisted-format pins + byte-compat gate test files run
             green (tests/core/test_schema_as_code.py,
             tests/core/test_v2_record_compat.py).
  gui        offscreen smoke of the run-end reload/select-last path
             (tests/xdart/test_batch_finish_select_last.py); skipped where
             Qt is absent.  NOTE: this is a smoke, not the full GUI suite —
             the complete tests/xdart offscreen run is the CI gate (pr.yml).
  deps       known ceilings intact (pyFAI<2025.12 — 2025.12 ships a
             broken pyFAI-calib2 on Windows).

There is intentionally NO publish subcommand: the maintainer uploads
manually (see .github/workflows/release.yml, which runs `check` before
building the artifacts).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _fail(msg: str) -> bool:
    print(f"  FAIL  {msg}")
    return False


def _ok(msg: str) -> bool:
    print(f"  PASS  {msg}")
    return True


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.M)
    if not m:
        raise SystemExit("pyproject.toml has no version field")
    return m.group(1)


def check_version(tag: str | None) -> bool:
    ok = True
    pp = _pyproject_version()

    try:
        from importlib.metadata import version as pkg_version
        dist = pkg_version("xdart")
        ok &= (_ok if dist == pp else _fail)(
            f"installed dist {dist} == pyproject {pp}")
    except Exception as exc:  # not installed — CI installs before checking
        ok &= _fail(f"xdart dist not importable for version check: {exc}")

    try:
        import xdart
        ok &= (_ok if xdart.__version__ == pp else _fail)(
            f"xdart.__version__ {xdart.__version__} == pyproject {pp}")
    except Exception as exc:
        ok &= _fail(f"xdart import failed: {exc}")

    if tag is None:
        head_tags = subprocess.run(
            ["git", "tag", "--points-at", "HEAD"],
            capture_output=True, text=True, cwd=ROOT,
        ).stdout.split()
        vtags = [t for t in head_tags if t.startswith("v")]
        tag = vtags[0] if vtags else None
    if tag is not None:
        want = tag.lstrip("v")
        ok &= (_ok if want == pp else _fail)(f"tag {tag} == pyproject {pp}")
    else:
        print("  note  no v* tag on HEAD; tag check skipped")
    return ok


def check_clean(strict: bool) -> bool:
    out = subprocess.run(["git", "status", "--porcelain"],
                         capture_output=True, text=True, cwd=ROOT).stdout
    if not out.strip():
        return _ok("working tree clean")
    if strict:
        return _fail(f"uncommitted changes:\n{out}")
    print(f"  warn  uncommitted changes (use --strict-tree to fail):\n{out}")
    return True


def check_schema() -> bool:
    """The persisted-format pins and the byte-compat gate must be green."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/core/test_schema_as_code.py",
         "tests/core/test_v2_record_compat.py"],
        cwd=ROOT,
    )
    return (_ok if proc.returncode == 0 else _fail)(
        "schema pins + byte-compat gate")


def check_gui_smoke() -> bool:
    """Offscreen GUI smoke for the run-end reload / select-last path — the class
    of failure the schema-only checks miss (a stale mock there once left preflight
    green while CI was red).  Skipped (not failed) where Qt isn't installed; the
    full ``tests/xdart`` offscreen suite remains the CI gate (pr.yml)."""
    try:
        import PySide6  # noqa: F401
    except Exception:
        print("  note  PySide6 not installed; GUI smoke skipped "
              "(full tests/xdart offscreen is CI-gated in pr.yml)")
        return True
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/xdart/test_batch_finish_select_last.py"],
        cwd=ROOT, env=env,
    )
    return (_ok if proc.returncode == 0 else _fail)("GUI run-end smoke (offscreen)")


def check_deps() -> bool:
    text = (ROOT / "pyproject.toml").read_text()
    ok = True
    # pyFAI ceiling: 2025.12 ships a broken pyFAI-calib2 on Windows.
    ok &= (_ok if re.search(r'"pyFAI>=[\d.]+,<2025\.12"', text) else _fail)(
        "pyFAI ceiling <2025.12 present")
    return ok


def run_checks(tag: str | None, strict_tree: bool) -> bool:
    print("version:")
    ok = check_version(tag)
    print("tree:")
    ok &= check_clean(strict_tree)
    print("deps:")
    ok &= check_deps()
    print("schema (runs two test files):")
    ok &= check_schema()
    print("gui smoke (offscreen; skipped if no Qt — full suite is CI-gated):")
    ok &= check_gui_smoke()
    return ok


def build() -> bool:
    # Clean dist/ first: `python -m build` does not, so a stale same-version
    # artifact (e.g. a weeks-old 1.0.0 wheel) would linger and could be uploaded
    # by mistake.  CI is safe (fresh checkout) but a local `release.py build` is not.
    shutil.rmtree(ROOT / "dist", ignore_errors=True)
    if subprocess.run([sys.executable, "-m", "build"], cwd=ROOT).returncode:
        return _fail("build failed")
    artifacts = [str(p) for p in (ROOT / "dist").glob("*")
                 if p.suffix in (".whl", ".gz")]
    if not artifacts:
        return _fail("no artifacts in dist/")
    if subprocess.run([sys.executable, "-m", "twine", "check", *artifacts],
                      cwd=ROOT).returncode:
        return _fail("twine check failed")
    return _ok("build + twine check")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["check", "build"])
    ap.add_argument("tag", nargs="?", default=None,
                    help="expected vX.Y.Z tag (default: tag on HEAD if any)")
    ap.add_argument("--strict-tree", action="store_true",
                    help="fail (not warn) on uncommitted changes")
    args = ap.parse_args(argv)

    ok = run_checks(args.tag, args.strict_tree)
    if ok and args.command == "build":
        ok = build()
    print("\nrelease pre-flight:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
