"""xdart — PySide6 GUI for XRD data reduction."""

import os as _os

# Generated UI modules in this package are PySide6-based.  pyqtgraph defaults
# to the first available Qt binding, which can be PyQt5 in the xrd_edit
# environment; mixing PyQt5 and PySide6 in one process leads to aborts in GUI
# tests and occasional app startup crashes.  Pin pyqtgraph before any module can
# import ``pyqtgraph.Qt``.
_os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
_os.environ.setdefault("QT_API", "PySide6")

# Single-source the version from installed package metadata so we don't
# have to keep a hard-coded string in sync with pyproject.toml on every
# release bump.
from importlib.metadata import version as _pkg_version, PackageNotFoundError

try:
    # the shipped monorepo distribution is "xdart".  Fall back to the
    # pre-1.0 legacy dist name so an old install still reports something
    # truthful.
    __version__ = _pkg_version("xdart")
except PackageNotFoundError:
    try:
        __version__ = _pkg_version("xrd-tools")
    except PackageNotFoundError:  # pragma: no cover — source checkout
        __version__ = "0.0.0+unknown"

del _pkg_version, PackageNotFoundError
del _os

from . import modules
from . import utils
from . import gui
