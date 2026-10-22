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
from packaging.requirements import Requirement

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
    assert {"ipykernel", "ipympl", "ipywidgets", "jupyterlab", "marimo"} <= imports


def test_gui_extra_includes_the_complete_notebook_stack():
    extras = _project()["optional-dependencies"]
    bundled = set().union(*(Requirement(item).extras for item in extras["gui"]
                            if _name(item) == "xdart"))
    assert {"fitting", "rsm", "notebook"} <= bundled
    assert {"ipywidgets", "anywidget", "ipyfilechooser", "ipykernel", "ipympl", "jupyterlab", "nbformat", "marimo"} <= {
        _name(item) for item in extras["notebook"]
    }


def test_marimo_uses_the_selected_compatible_release_line_in_both_installs():
    pip = next(Requirement(item) for item in _project()["optional-dependencies"]["notebook"]
               if _name(item) == "marimo")
    conda = next(item for item in _recipe()["requirements"]["run"] if _name(item) == "marimo")
    assert str(pip.specifier) == str(Requirement(conda).specifier)
    assert "0.25.0" in pip.specifier and "0.24.9" not in pip.specifier and "0.26.0" not in pip.specifier


def test_the_runtime_tiff_header_reader_is_a_dependency_of_both_installs():
    """1.2.0 imported tifffile at Run admission but declared it only for development.

    A `pixi global install` on Windows then refused every TIFF scan (maintainer
    report, 2026-10-22); editable and CI environments hid it through the dev extra.
    """
    assert "tifffile" in {_name(item) for item in _project()["dependencies"]}
    assert "tifffile" in {_name(item) for item in _recipe()["requirements"]["run"]}
    imports = {name for test in _recipe()["tests"] for name in test.get("python", {}).get("imports", ())}
    assert "tifffile" in imports


def test_conda_recipe_offers_only_the_python_versions_the_project_claims():
    """An open-ended range hands users a Python that no test has ever run.

    `pixi global install` takes the newest interpreter the recipe allows, so the
    recipe's upper bound is what decides which Python users actually get.
    """
    minors = sorted(
        int(classifier.rsplit(".", 1)[1])
        for classifier in _project()["classifiers"]
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", classifier)
    )
    assert minors, "pyproject declares no supported Python minor versions"
    spec = next(r for r in _recipe()["requirements"]["run"] if _name(r) == "python")
    assert f"<3.{minors[-1] + 1}" in spec.replace(" ", ""), spec
