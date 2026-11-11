"""``scripts/release.py check`` compares the README's version lines with the version being released (1.2.10: the
README still described 1.2.3 at 1.2.9, his addition to the review)."""
from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path

import pytest
import xdart

ROOT = Path(__file__).resolve().parents[2]
VERSION = "1.2.10"


def _release():
    spec = importlib.util.spec_from_file_location("xdart_release_script", ROOT / "scripts" / "release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(root: Path, intro_version: str, checkout_version: str) -> None:
    (root / "pyproject.toml").write_text(f'[project]\nname = "xdart"\nversion = "{VERSION}"\n', encoding="utf-8")
    (root / "README.md").write_text(
        f"# xdart\n\nThis README describes **xdart {intro_version}**.\n\n"
        f"To use the exact {checkout_version}\nrelease, run `git switch --detach v{checkout_version}` before installing.\n",
        encoding="utf-8")


@pytest.fixture
def release_check(tmp_path, monkeypatch):
    release = _release()
    monkeypatch.setattr(release, "ROOT", tmp_path)
    # The temporary release's installed-version facts agree. The actual
    # release check reads both files and decides the status.
    monkeypatch.setattr(xdart, "__version__", VERSION)
    version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version", lambda name: VERSION if name == "xdart" else version(name))
    return release.check_version


@pytest.mark.parametrize(("intro", "checkout"), [("1.2.3", VERSION), (VERSION, "1.2.3")])
def test_check_fails_when_either_readme_version_is_stale(tmp_path, capsys, release_check, intro, checkout):
    _tree(tmp_path, intro, checkout)
    assert release_check(f"v{VERSION}") is False
    out = capsys.readouterr().out
    assert any(line.strip().startswith("FAIL") and "README" in line and "1.2.3" in line
               for line in out.splitlines()), out


def test_check_passes_the_readme_when_it_names_the_release(tmp_path, capsys, release_check):
    _tree(tmp_path, VERSION, VERSION)
    assert release_check(f"v{VERSION}") is True
    out = capsys.readouterr().out
    readme = [line for line in out.splitlines() if "README" in line]
    assert readme and all(line.strip().startswith("PASS") for line in readme), out
