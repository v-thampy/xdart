"""Tests for ssrl_xrd_tools.integrate.gid."""

from __future__ import annotations

import numpy as np
import pytest

from ssrl_xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from ssrl_xrd_tools.integrate.gid import (
    create_fiber_integrator,
    integrate_gi_1d,
    integrate_gi_2d,
    integrate_gi_exitangles,
    integrate_gi_exitangles_1d,
    integrate_gi_polar,
    integrate_gi_polar_1d,
)

try:
    from pyFAI.integrator.fiber import FiberIntegrator

    _HAS_FIBER = True
except ImportError:
    _HAS_FIBER = False


pytestmark = pytest.mark.skipif(
    not _HAS_FIBER,
    reason="FiberIntegrator requires pyFAI >= 2025.01",
)


def test_create_fiber_integrator(poni_fixture):
    fi = create_fiber_integrator(poni_fixture, incident_angle=0.3)

    assert "FiberIntegrator" in type(fi).__name__
    assert hasattr(fi, "_gi_incident_angle")
    np.testing.assert_allclose(fi._gi_incident_angle, np.deg2rad(0.3), rtol=1e-8, atol=1e-12)


def test_create_fiber_integrator_radians(poni_fixture):
    fi = create_fiber_integrator(
        poni_fixture,
        incident_angle=0.005,
        angle_unit="rad",
    )

    assert hasattr(fi, "_gi_incident_angle")
    np.testing.assert_allclose(fi._gi_incident_angle, 0.005, rtol=1e-10, atol=1e-12)


@pytest.mark.slow
def test_integrate_gi_1d(poni_fixture, synthetic_image):
    fi = create_fiber_integrator(poni_fixture, incident_angle=0.2)
    result = integrate_gi_1d(synthetic_image, fi, npt=500)

    assert isinstance(result, IntegrationResult1D)
    assert result.radial.shape == (500,)
    assert result.intensity.shape == (500,)


@pytest.mark.slow
def test_integrate_gi_2d(poni_fixture, synthetic_image):
    fi = create_fiber_integrator(poni_fixture, incident_angle=0.2)
    result = integrate_gi_2d(synthetic_image, fi, npt_rad=200, npt_azim=100)

    assert isinstance(result, IntegrationResult2D)
    assert result.radial.shape == (200,)
    assert result.azimuthal.shape == (100,)
    assert result.intensity.shape == (200, 100)


@pytest.mark.slow
def test_integrate_gi_polar(poni_fixture, synthetic_image):
    fi = create_fiber_integrator(poni_fixture, incident_angle=0.2)
    result = integrate_gi_polar(synthetic_image, fi, npt_rad=200, npt_azim=100)

    assert isinstance(result, IntegrationResult2D)
    assert result.radial.shape == (200,)
    assert result.azimuthal.shape == (100,)
    assert result.intensity.shape == (200, 100)


@pytest.mark.slow
def test_integrate_gi_exitangles(poni_fixture, synthetic_image):
    fi = create_fiber_integrator(poni_fixture, incident_angle=0.2)
    result = integrate_gi_exitangles(synthetic_image, fi, npt_rad=200, npt_azim=100)

    assert isinstance(result, IntegrationResult2D)
    assert result.radial.shape == (200,)
    assert result.azimuthal.shape == (100,)
    assert result.intensity.shape == (200, 100)


@pytest.mark.slow
def test_integrate_gi_1d_angle_override(poni_fixture, synthetic_image):
    fi = create_fiber_integrator(poni_fixture, incident_angle=0.2)

    _ = integrate_gi_1d(synthetic_image, fi, npt=500, incident_angle=0.5)

    np.testing.assert_allclose(fi._gi_incident_angle, np.deg2rad(0.5), rtol=1e-8, atol=1e-12)


@pytest.mark.slow
def test_integrate_gi_polar_1d(poni_fixture, synthetic_image):
    fi = create_fiber_integrator(poni_fixture, incident_angle=0.2)
    result = integrate_gi_polar_1d(synthetic_image, fi, npt=500)

    assert isinstance(result, IntegrationResult1D)
    assert result.radial.shape == (500,)
    assert result.intensity.shape == (500,)


@pytest.mark.slow
def test_integrate_gi_exitangles_1d(poni_fixture, synthetic_image):
    fi = create_fiber_integrator(poni_fixture, incident_angle=0.2)
    result = integrate_gi_exitangles_1d(synthetic_image, fi, npt=500)

    assert isinstance(result, IntegrationResult1D)
    assert result.radial.shape == (500,)
    assert result.intensity.shape == (500,)
