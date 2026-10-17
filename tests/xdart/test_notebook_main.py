"""``xdart-notebook`` drives this installation's real notebook frontends and ipykernel.

Every row runs the real command in a child process: the point of the launcher
is which Python and which JupyterLab it reaches, and a stub would hide that.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

pytest.importorskip("jupyterlab")


def _run(*arguments: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "xdart.notebook_main", *arguments],
        capture_output=True, text=True, timeout=180, env=env,
    )


def test_launcher_reaches_this_installations_jupyterlab():
    done = _run("--version")
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip().splitlines()[-1] == metadata.version("jupyterlab")


def test_launcher_registers_a_kernel_for_this_python(tmp_path):
    # JUPYTER_DATA_DIR is where `ipykernel install --user` writes; the user's
    # real kernel directory is never touched.
    environment = {**os.environ, "JUPYTER_DATA_DIR": str(tmp_path)}
    done = _run("--register-kernel", env=environment)
    assert done.returncode == 0, done.stdout + done.stderr
    spec = json.loads((tmp_path / "kernels" / "xdart" / "kernel.json").read_text("utf-8"))
    assert spec["display_name"] == "Python (xdart)"
    assert Path(spec["argv"][0]).resolve() == Path(sys.executable).resolve()


def test_launcher_keeps_a_kernel_name_the_user_chose(tmp_path):
    environment = {**os.environ, "JUPYTER_DATA_DIR": str(tmp_path)}
    done = _run("--register-kernel", "--name", "beamline", "--display-name", "Beamline",
                env=environment)
    assert done.returncode == 0, done.stdout + done.stderr
    spec = json.loads((tmp_path / "kernels" / "beamline" / "kernel.json").read_text("utf-8"))
    assert spec["display_name"] == "Beamline"
    assert not (tmp_path / "kernels" / "xdart").exists()


def test_launcher_never_silently_repoints_someone_elses_xdart_kernel(tmp_path):
    """`ipykernel install` overwrites without asking; an older kernel may be in use."""
    other = tmp_path / "kernels" / "xdart"
    other.mkdir(parents=True)
    stale = {"argv": ["/opt/other-env/bin/python", "-m", "ipykernel_launcher", "-f",
                      "{connection_file}"], "display_name": "xdart", "language": "python"}
    (other / "kernel.json").write_text(json.dumps(stale), encoding="utf-8")
    environment = {**os.environ, "JUPYTER_DATA_DIR": str(tmp_path)}

    refused = _run("--register-kernel", env=environment)
    assert refused.returncode == 2
    assert "/opt/other-env/bin/python" in refused.stderr and "--replace" in refused.stderr
    assert json.loads((other / "kernel.json").read_text("utf-8")) == stale

    replaced = _run("--register-kernel", "--replace", env=environment)
    assert replaced.returncode == 0, replaced.stdout + replaced.stderr
    spec = json.loads((other / "kernel.json").read_text("utf-8"))
    assert Path(spec["argv"][0]).resolve() == Path(sys.executable).resolve()

    # Registering again is idempotent: the kernel is already this Python.
    assert _run("--register-kernel", env=environment).returncode == 0


def test_launcher_explains_a_missing_notebook_stack():
    """`pip install xdart` without the extra still installs the command."""
    driver = (
        "import sys\n"
        "sys.modules['jupyterlab'] = None  # find_spec() now reports it absent\n"
        "from xdart.notebook_main import main\n"
        "raise SystemExit(main(['--version']))\n"
    )
    done = subprocess.run([sys.executable, "-c", driver], capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 1
    assert 'pip install "xdart[notebook]"' in done.stderr


def test_launcher_describes_itself_before_jupyterlab_does():
    done = _run("--help")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "--register-kernel" in done.stdout


def test_launcher_reaches_this_installations_marimo_without_starting_a_server(tmp_path):
    pytest.importorskip("marimo")
    notebook = tmp_path / "fit 'α'.py"
    content = b"import marimo\napp = marimo.App()\n"
    notebook.write_bytes(content)
    done = _run("--marimo", str(notebook), "--help")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "Usage: marimo edit" in done.stdout and "--no-sandbox" in done.stdout
    assert notebook.read_bytes() == content


@pytest.mark.parametrize("filename", (None, "result.ipynb", "missing.py", "directory.py"))
def test_marimo_launch_requires_an_explicit_existing_python_notebook(tmp_path, filename):
    if filename == "result.ipynb":
        (tmp_path / filename).write_text("{}", encoding="utf-8")
    elif filename == "directory.py":
        (tmp_path / filename).mkdir()
    arguments = [] if filename is None else [str(tmp_path / filename)]
    done = _run("--marimo", *arguments, "--help")
    assert done.returncode == 2, done.stdout + done.stderr
    assert "existing .py notebook" in done.stderr


def test_python_filename_does_not_guess_the_marimo_frontend(tmp_path):
    notebook = tmp_path / "workflow.py"
    notebook.write_text("", encoding="utf-8")
    done = _run(str(notebook), "--version")
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip().splitlines()[-1] == metadata.version("jupyterlab")


def test_marimo_mode_cannot_register_a_jupyter_kernel(tmp_path):
    notebook = tmp_path / "workflow.py"
    notebook.write_text("", encoding="utf-8")
    done = _run("--marimo", str(notebook), "--register-kernel", "--help")
    assert done.returncode == 2, done.stdout + done.stderr
    assert "--register-kernel" in done.stderr


def test_launcher_explains_a_missing_marimo_package(tmp_path):
    notebook = tmp_path / "workflow.py"
    notebook.write_text("", encoding="utf-8")
    driver = (
        "import sys\n"
        "sys.modules['marimo'] = None\n"
        "from xdart.notebook_main import main\n"
        "raise SystemExit(main(['--marimo', sys.argv[1], '--help']))\n"
    )
    done = subprocess.run([sys.executable, "-c", driver, str(notebook)],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 1
    assert 'pip install "xdart[notebook]"' in done.stderr
