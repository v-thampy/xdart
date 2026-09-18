"""Start JupyterLab, or register a kernel, with this installation's Python.

``pixi global install xdart`` puts only xdart's own commands on PATH, so the
JupyterLab installed beside it has no command of its own.  ``xdart-notebook``
is that command: it runs the JupyterLab of the environment xdart lives in, so
``xrd_tools``, the notebook widgets, and the compression filters are all the
ones the GUI uses.  Nothing here imports Qt.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

_USAGE = """\
usage: xdart-notebook [JupyterLab arguments ...]
       xdart-notebook --register-kernel [--name NAME] [--display-name TEXT] [--replace]

Start JupyterLab from the xdart installation, for the example notebooks and
the notebooks exported by Help -> Export Analyze Results Notebook.

  xdart-notebook                     open JupyterLab in the current folder
  xdart-notebook results.ipynb       open one notebook
  xdart-notebook --register-kernel   offer this Python to VS Code or another
                                     Jupyter as the kernel "Python (xdart)"

Every other argument goes to JupyterLab unchanged; `xdart-notebook --help-all`
lists them.
"""

_TAKEN = """\
A Jupyter kernel named "xdart" already exists for another Python:

    {python}

Keep both by choosing a name:   xdart-notebook --register-kernel --name xdart-app
Or repoint "xdart" to this one: xdart-notebook --register-kernel --replace
"""

_MISSING = """\
xdart-notebook needs the notebook packages, which this installation lacks.
Install them into the same environment:

    pip install "xdart[notebook]"

The conda package (pixi global install ... xdart) already includes them.
"""


def _child_environment() -> dict[str, str]:
    # The kernel JupyterLab starts is found through PATH on some platforms.
    environment = dict(os.environ)
    here = os.path.dirname(sys.executable)
    environment["PATH"] = here + os.pathsep + environment.get("PATH", "")
    return environment


def _run(module: str, arguments: list[str]) -> int:
    try:
        return subprocess.call(
            [sys.executable, "-m", module, *arguments], env=_child_environment(),
        )
    except KeyboardInterrupt:
        return 130


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] in (["-h"], ["--help"]):
        print(_USAGE, end="")
        return 0
    register = "--register-kernel" in arguments
    needed = "ipykernel" if register else "jupyterlab"
    if importlib.util.find_spec(needed) is None:
        print(_MISSING, end="", file=sys.stderr)
        return 1
    if not register:
        return _run("jupyterlab", arguments)
    arguments.remove("--register-kernel")

    def given(option: str) -> bool:
        return any(item == option or item.startswith(option + "=") for item in arguments)

    replace = "--replace" in arguments
    if replace:
        arguments.remove("--replace")
    if not given("--name"):
        other = _kernel_python("xdart")
        if other is not None and not replace:
            # `ipykernel install` overwrites silently, and an older "xdart"
            # kernel may still serve another environment the user relies on.
            print(_TAKEN.format(python=other), end="", file=sys.stderr)
            return 2
        arguments += ["--name", "xdart"]
        if not given("--display-name"):
            arguments += ["--display-name", "Python (xdart)"]
    if not (given("--user") or given("--prefix") or given("--sys-prefix")):
        arguments.insert(0, "--user")
    return _run("ipykernel", ["install", *arguments])


def _kernel_python(name: str) -> str | None:
    """The interpreter of an existing kernel *name* that is not this one."""
    from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel

    try:
        argv = KernelSpecManager().get_kernel_spec(name).argv
    except NoSuchKernel:
        return None
    if not argv or os.path.realpath(argv[0]) == os.path.realpath(sys.executable):
        return None
    return argv[0]


if __name__ == "__main__":
    raise SystemExit(main())
