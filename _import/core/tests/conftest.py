"""
Shared pytest fixtures for the ssrl_xrd_tools test suite.

Notes
-----
Eiger 4M actual shape from pyFAI detector registry: ``(2167, 2070)``.
All full-size fixtures use this shape so they are compatible with the
session-scoped ``ai_fixture``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ssrl_xrd_tools.core.containers import PONI
from ssrl_xrd_tools.integrate.calibration import poni_to_integrator, save_poni

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Actual Eiger 4M detector shape as reported by pyFAI.
_EIGER4M_SHAPE: tuple[int, int] = (2167, 2070)

#: Pixel coordinates of the detector centre (row, col).
_EIGER4M_CENTER: tuple[int, int] = (
    _EIGER4M_SHAPE[0] // 2,  # 1083
    _EIGER4M_SHAPE[1] // 2,  # 1035
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def poni_fixture() -> PONI:
    """
    Realistic PONI geometry for an Eiger 4M at ~200 mm sample-detector distance.

    Returns
    -------
    PONI
        Calibration geometry with Cu Kα wavelength (1.0 Å).
    """
    return PONI(
        dist=0.2,
        poni1=0.081,
        poni2=0.0775,
        rot1=0.0,
        rot2=0.0,
        rot3=0.0,
        wavelength=1.0e-10,
        detector="eiger4m",
    )


@pytest.fixture(scope="session")
def ai_fixture(poni_fixture: PONI):
    """
    pyFAI ``AzimuthalIntegrator`` built from ``poni_fixture``.

    Session-scoped so the integrator (and its lookup-tables once computed)
    are shared across all tests that use it.
    """
    return poni_to_integrator(poni_fixture)


@pytest.fixture(scope="session")
def synthetic_image() -> np.ndarray:
    """
    Full-size Eiger 4M detector image simulating a powder diffraction ring.

    A Gaussian ring at ~500 px radius centred on the detector midpoint is
    added to Poisson-distributed background noise.

    Returns
    -------
    np.ndarray
        Shape ``(2167, 2070)``, dtype ``float64``.
    """
    ny, nx = _EIGER4M_SHAPE
    cy, cx = _EIGER4M_CENTER
    rng = np.random.default_rng(42)
    y, x = np.mgrid[:ny, :nx]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    image = 1000.0 * np.exp(-((r - 500) / 50) ** 2) + rng.poisson(10, (ny, nx))
    return image.astype(np.float64)


@pytest.fixture(scope="session")
def synthetic_image_small() -> np.ndarray:
    """
    Small 100×100 image with a central Gaussian peak and Poisson noise.

    Intended for fast unit tests that do not require realistic detector
    geometry.

    Returns
    -------
    np.ndarray
        Shape ``(100, 100)``, dtype ``float64``.
    """
    rng = np.random.default_rng(0)
    y, x = np.mgrid[:100, :100]
    r = np.sqrt((y - 50) ** 2 + (x - 50) ** 2)
    image = 500.0 * np.exp(-(r / 15) ** 2) + rng.poisson(5, (100, 100))
    return image.astype(np.float64)


@pytest.fixture(scope="session")
def tmp_poni_file(poni_fixture: PONI, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Save ``poni_fixture`` to a temporary ``.poni`` file and return its path.

    Uses ``tmp_path_factory`` (session-scoped built-in) so the file outlives
    individual tests and is shared within the session.

    Returns
    -------
    Path
        Absolute path to the written ``.poni`` file.
    """
    tmp_dir: Path = tmp_path_factory.mktemp("poni")
    path = tmp_dir / "test.poni"
    save_poni(poni_fixture, path)
    return path


@pytest.fixture(scope="session")
def synthetic_mask() -> np.ndarray:
    """
    Boolean bad-pixel mask for the Eiger 4M shape.

    The 10-pixel border on all four edges is marked ``True`` (bad).

    Returns
    -------
    np.ndarray
        Shape ``(2167, 2070)``, dtype ``bool``.
    """
    ny, nx = _EIGER4M_SHAPE
    mask = np.zeros((ny, nx), dtype=bool)
    mask[:10, :] = True
    mask[-10:, :] = True
    mask[:, :10] = True
    mask[:, -10:] = True
    return mask
