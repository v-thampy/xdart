#!/usr/bin/env python3
"""Release pre-flight for xdart (Phase 1b of the greenfield plan).

The two-repo era enforced publish order and version floors by runbook;
the monorepo enforces them by script.  Run everything before tagging:

    python scripts/release.py check          # all pre-flight checks
    python scripts/release.py check vX.Y.Z   # + tag consistency
    python scripts/release.py build          # checks, then build + twine
    XDART_TEST_DATA=/absolute/corpus python scripts/release.py promotion \
        --policy tests/promotion/real_data_gate_v1.json \
        --report /absolute/evidence/promotion-report.json

Checks (each prints PASS/FAIL; any FAIL exits 1):

  version    pyproject version == importlib version of the installed
             editable dist == xdart.__version__; if a tag is given (or
             HEAD is tagged v*), it must match.
  clean      no uncommitted changes (warn-only unless --strict-tree).
  schema     the persisted-format pins + byte-compat gate test files run
             green (tests/core/test_schema_as_code.py,
             tests/core/test_v2_record_compat.py).
  gui        offscreen smoke of XYE run-end selection and NeXus viewer adoption
             (tests/xdart/scattering/test_xye_run_completion.py); skipped where
             Qt is absent.  NOTE: this is a smoke, not the full GUI suite —
             the complete tests/xdart offscreen run is the CI gate (pr.yml).
  deps       pyFAI's audited project, Pixi, conda-recipe, and uv pins agree.
  promotion  authenticate the finite private corpus, run the exact real-data
             science selection, and fail on any skip or collection drift.

There is intentionally NO publish subcommand: the maintainer pushes a release
tag to trigger .github/workflows/release.yml, which checks and builds before
publishing the artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

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
         "tests/xdart/scattering/test_xye_run_completion.py::"
         "test_completed_xye_run_opens_terminal_file_and_output_folder[hdf-True-2th_deg-itth]",
         "tests/xdart/scattering/test_xye_run_completion.py::"
         "test_completed_run_selected_artifact_enters_2d_viewer[Int 1D]"],
        cwd=ROOT, env=env,
    )
    return (_ok if proc.returncode == 0 else _fail)("GUI run-end smoke (offscreen)")


def check_deps() -> bool:
    project = (ROOT / "pyproject.toml").read_text()
    recipe = (ROOT / "recipe" / "recipe.yaml").read_text()
    uv_lock = (ROOT / "uv.lock").read_text()
    ok = True
    window = r">=2026\.5,<2026\.6"
    checks = (
        (re.search(rf'"pyFAI{window}"', project), "project pyFAI window"),
        (re.search(rf'^pyfai\s*=\s*"{window}"$', project, re.M),
         "Pixi pyFAI window"),
        (re.search(rf'^\s*- pyfai {window}(?:\s|$)', recipe, re.M),
         "conda recipe pyFAI window"),
        (re.search(rf'^\s*\{{ name = "pyfai", specifier = "{window}" \}},$',
                   uv_lock, re.M), "uv requirement pyFAI window"),
        (re.search(r'^name = "pyfai"\nversion = "2026\.5\.0"$',
                   uv_lock, re.M), "uv resolves pyFAI 2026.5.0"),
    )
    for matched, label in checks:
        ok &= (_ok if matched else _fail)(label)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _corpus_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"corpus path is not a safe relative path: {relative!r}")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"corpus path escapes its root: {relative!r}")
    return resolved


def _verify_promotion_corpus(root: Path, policy: dict) -> dict:
    verified_files: list[str] = []
    verified_groups: list[dict] = []
    for expected in policy.get("files", ()):
        relative = str(expected["path"])
        path = _corpus_path(root, relative)
        if not path.is_file():
            raise ValueError(f"required corpus file is missing: {relative}")
        size = path.stat().st_size
        if size != int(expected["size"]):
            raise ValueError(
                f"corpus size mismatch for {relative}: {size} != {expected['size']}"
            )
        actual = _sha256(path)
        if actual != str(expected["sha256"]):
            raise ValueError(f"corpus SHA-256 mismatch for {relative}")
        verified_files.append(relative)

    for expected in policy.get("file_groups", ()):
        pattern = str(expected["glob"])
        pattern_path = Path(pattern)
        if pattern_path.is_absolute() or ".." in pattern_path.parts:
            raise ValueError(f"corpus glob is not safe: {pattern!r}")
        paths = tuple(
            sorted(
                (path.resolve() for path in root.glob(pattern) if path.is_file()),
                key=lambda path: path.relative_to(root).as_posix(),
            )
        )
        if any(not path.is_relative_to(root) for path in paths):
            raise ValueError(f"corpus glob escapes its root: {pattern!r}")
        rows: list[str] = []
        total_bytes = 0
        for path in paths:
            relative = path.relative_to(root).as_posix()
            size = path.stat().st_size
            total_bytes += size
            rows.append(f"{relative}\0{size}\0{_sha256(path)}")
        manifest = hashlib.sha256("\n".join(rows).encode()).hexdigest()
        if len(paths) != int(expected["count"]):
            raise ValueError(
                f"corpus count mismatch for {pattern}: "
                f"{len(paths)} != {expected['count']}"
            )
        if total_bytes != int(expected["total_bytes"]):
            raise ValueError(f"corpus byte total mismatch for {pattern}")
        if manifest != str(expected["manifest_sha256"]):
            raise ValueError(f"corpus manifest mismatch for {pattern}")
        verified_groups.append(
            {
                "glob": pattern,
                "count": len(paths),
                "total_bytes": total_bytes,
                "manifest_sha256": manifest,
            }
        )
    return {"files": verified_files, "file_groups": verified_groups}


def _junit_summary(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    counts = {
        name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    node_ids: list[str] = []
    for case in root.iter("testcase"):
        classname = case.attrib.get("classname", "")
        name = case.attrib.get("name", "")
        module = classname.replace(".", "/")
        if module and not module.endswith(".py"):
            module += ".py"
        node_ids.append(f"{module}::{name}" if module else name)
    return {**counts, "node_ids": node_ids}


def run_promotion(policy_path: Path, report_path: Path) -> bool:
    started = datetime.now(timezone.utc)
    policy_path = policy_path.resolve()
    report_path = report_path.resolve()
    junit_path = report_path.with_name(f"{report_path.stem}.junit.xml")
    if report_path.exists() or junit_path.exists():
        return _fail("promotion report target already exists")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "started_at": started.isoformat(),
        "policy_path": str(policy_path),
        "report_path": str(report_path),
        "junit_path": str(junit_path),
        "python": sys.version,
        "platform": platform.platform(),
        "verdict": "failed",
    }
    ok = False
    try:
        if not policy_path.is_file():
            raise ValueError(f"promotion policy is missing: {policy_path}")
        policy = json.loads(policy_path.read_text())
        if int(policy.get("version", 0)) != 1:
            raise ValueError("unsupported promotion policy version")
        nodes = tuple(str(item) for item in policy.get("pytest_nodes", ()))
        expected_tests = int(policy.get("expected_tests", 0))
        if not nodes or expected_tests <= 0:
            raise ValueError("promotion policy has no finite pytest selection")

        configured = os.environ.get("XDART_TEST_DATA")
        if not configured:
            raise ValueError("XDART_TEST_DATA must be set explicitly")
        configured_path = Path(configured)
        if not configured_path.is_absolute():
            raise ValueError("XDART_TEST_DATA must be an absolute path")
        corpus_root = configured_path.resolve()
        if not corpus_root.is_dir():
            raise ValueError(f"XDART_TEST_DATA is not a directory: {corpus_root}")

        report.update(
            {
                "corpus_id": str(policy["corpus_id"]),
                "corpus_root": str(corpus_root),
                "policy_sha256": _sha256(policy_path),
                "git_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT,
                    check=True, capture_output=True, text=True,
                ).stdout.strip(),
                "git_tree": subprocess.run(
                    ["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT,
                    check=True, capture_output=True, text=True,
                ).stdout.strip(),
            }
        )
        print(f"corpus: {policy['corpus_id']}")
        report["corpus_verification"] = _verify_promotion_corpus(
            corpus_root, policy,
        )
        _ok("authenticated promotion corpus")

        argv = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--strict-markers",
            f"--junitxml={junit_path}",
            *nodes,
        ]
        env = dict(os.environ)
        prior_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(ROOT / "src") + (
            os.pathsep + prior_pythonpath if prior_pythonpath else ""
        )
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["TMPDIR"] = str(report_path.parent)
        report["pytest_argv"] = argv
        process = subprocess.run(argv, cwd=ROOT, env=env)
        report["pytest_exit_code"] = process.returncode
        if not junit_path.is_file():
            raise ValueError("pytest did not write the promotion JUnit report")
        summary = _junit_summary(junit_path)
        report["pytest"] = summary
        ok = (
            process.returncode == 0
            and summary["tests"] == expected_tests
            and summary["failures"] == 0
            and summary["errors"] == 0
            and summary["skipped"] == 0
            and len(summary["node_ids"]) == expected_tests
        )
        if not ok:
            raise ValueError(
                "promotion tests require exact count, zero failures/errors/skips, "
                "and a zero process exit"
            )
        report["verdict"] = "passed"
        _ok(f"authenticated real-data science gate ({expected_tests} tests)")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        _fail(str(error))
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["check", "build", "promotion"])
    ap.add_argument("tag", nargs="?", default=None,
                    help="expected vX.Y.Z tag (default: tag on HEAD if any)")
    ap.add_argument("--strict-tree", action="store_true",
                    help="fail (not warn) on uncommitted changes")
    ap.add_argument(
        "--policy", type=Path,
        help="checked-in real-data policy (required for promotion)",
    )
    ap.add_argument(
        "--report", type=Path,
        help="new JSON evidence path (required for promotion)",
    )
    args = ap.parse_args(argv)

    if args.command == "promotion":
        if args.tag is not None:
            ap.error("promotion does not accept a tag")
        if args.policy is None or args.report is None:
            ap.error("promotion requires --policy and --report")
        ok = run_promotion(args.policy, args.report)
        print("\npromotion gate:", "OK" if ok else "FAILED")
        return 0 if ok else 1

    ok = run_checks(args.tag, args.strict_tree)
    if ok and args.command == "build":
        ok = build()
    print("\nrelease pre-flight:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
