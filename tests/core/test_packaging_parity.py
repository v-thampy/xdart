"""The conda package must offer what the project metadata promises.

``pixi global install xdart`` is the documented one-line installation. It
exposes only the commands the conda recipe declares for xdart itself, and it
installs only the recipe's run requirements, so both must follow
``pyproject.toml`` or a console script or the notebook stack silently goes
missing for exactly the users who never see a Python environment.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def _recipe() -> dict:
    return yaml.safe_load((ROOT / "recipe" / "recipe.yaml").read_text(encoding="utf-8"))


def _name(requirement: str) -> str:
    return re.split(r"[\s<>=!~\[;]", requirement.strip(), maxsplit=1)[0].lower().replace("_", "-")


def test_conda_recipe_declares_every_console_script():
    declared = {
        name.strip(): target.strip()
        for name, target in (
            entry.split("=", 1) for entry in _recipe()["build"]["python"]["entry_points"]
        )
    }
    assert declared == dict(_project()["scripts"])


def test_conda_recipe_installs_the_notebook_stack():
    """The one-line GUI installation must be able to open its own notebooks."""
    run = {_name(requirement) for requirement in _recipe()["requirements"]["run"]}
    wanted = {_name(requirement) for requirement in _project()["optional-dependencies"]["notebook"]}
    assert wanted and wanted <= run, sorted(wanted - run)


def test_conda_recipe_tests_import_the_notebook_stack():
    """A solved-but-broken notebook stack must fail the package build, not a user."""
    imports = {
        name
        for test in _recipe()["tests"]
        for name in test.get("python", {}).get("imports", ())
    }
    assert {"ipykernel", "ipympl", "ipywidgets", "jupyterlab"} <= imports
