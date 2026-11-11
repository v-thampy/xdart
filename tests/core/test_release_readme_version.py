"""``scripts/release.py check`` compares the README's version lines with the version being released (1.2.10: the
README still described 1.2.3 at 1.2.9, his addition to the review)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _release():
    spec = importlib.util.spec_from_file_location("xdart_release_script", ROOT / "scripts" / "release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(root: Path, version: str, readme_version: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(f'[project]\nname = "xdart"\nversion = "{version}"\n', encoding="utf-8")
    (root / "README.md").write_text(
        f"# xdart\n\nThis README describes **xdart {readme_version}**.\n\n"
        f"To use the exact {readme_version}\nrelease, run `git switch --detach v{readme_version}` before installing.\n",
        encoding="utf-8")


def test_check_fails_when_the_readme_describes_another_release(tmp_path, monkeypatch, capsys):
    release = _release()
    _tree(tmp_path / "stale", "1.2.10", "1.2.3")
    monkeypatch.setattr(release, "ROOT", tmp_path / "stale")
    release.check_version("v1.2.10")
    out = capsys.readouterr().out
    assert any(line.strip().startswith("FAIL") and "README" in line and "1.2.3" in line
               for line in out.splitlines()), out


def test_check_passes_the_readme_when_it_names_the_release(tmp_path, monkeypatch, capsys):
    release = _release()
    _tree(tmp_path / "current", "1.2.10", "1.2.10")
    monkeypatch.setattr(release, "ROOT", tmp_path / "current")
    release.check_version("v1.2.10")
    out = capsys.readouterr().out
    readme = [line for line in out.splitlines() if "README" in line]
    assert readme and all(line.strip().startswith("PASS") for line in readme), out
