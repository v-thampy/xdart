"""xdart — PySide6 GUI for XRD data reduction."""

# Single-source the version from installed package metadata so we don't
# have to keep a hard-coded string in sync with pyproject.toml on every
# release bump.
from importlib.metadata import version as _pkg_version, PackageNotFoundError

try:
    __version__ = _pkg_version("xdart")
except PackageNotFoundError:  # pragma: no cover — source checkout without install
    __version__ = "0.0.0+unknown"

del _pkg_version, PackageNotFoundError

from . import modules
from . import utils
from . import gui
