"""
xrd_tools — SSRL X-ray diffraction data processing tools.

Import from submodules directly:
    from xrd_tools.rsm import ExperimentConfig, RSMVolume
    from xrd_tools.analysis.fitting import fit_line_cut
    from xrd_tools.io import read_image_stack
"""
# Single-source the version from installed package metadata so we don't
# have to keep a hard-coded string in sync with pyproject.toml on every
# release bump.
from importlib.metadata import version as _pkg_version, PackageNotFoundError

try:
    # the shipped distribution is "xdart"; "xrd-tools" is the pre-1.0 legacy
    # dist name (some conda builds still install under it).
    __version__ = _pkg_version("xdart")
except PackageNotFoundError:
    try:
        __version__ = _pkg_version("xrd-tools")
    except PackageNotFoundError:  # pragma: no cover — source checkout without install
        __version__ = "0.0.0+unknown"

del _pkg_version, PackageNotFoundError
