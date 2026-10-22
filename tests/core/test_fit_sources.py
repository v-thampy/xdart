"""One semantic radial conversion for GUI and notebook fits."""
from dataclasses import replace
import numpy as np
import pytest
from xrd_tools.analysis.fit_snapshot import FitAxis, FitInputKey, FitInputSnapshot
from xrd_tools.analysis.fit_sources import phase_fit_snapshot
from xrd_tools.core.containers import convert_radial_axis


def test_radial_phase_conversion_keeps_nonuniform_coordinates_intensity_and_sigma():
    q = np.linspace(1, 6, 301)
    angles = convert_radial_axis(q, 'q_A^-1', '2th_deg', 1.)
    key = FitInputKey('scan', '/data.nexus', 'admitted', 1, selection=(('meaning', 'radial'),))
    s = FitInputSnapshot(key, (FitAxis('2θ', '2th_deg', angles),), q**2,
                         wavelength_angstrom=1., sigma=np.sqrt(q))
    fitted = phase_fit_snapshot(s)
    np.testing.assert_allclose(fitted.axes[0].values, q, atol=1e-14)
    np.testing.assert_array_equal(fitted.values, s.values)
    np.testing.assert_array_equal(fitted.sigma, s.sigma)
    with pytest.raises(ValueError, match='PHASE_WAVELENGTH_REQUIRED'):
        phase_fit_snapshot(replace(s, wavelength_angstrom=None))
    for meaning in ('q_ip', 'q_oop', 'coverage', 'normalization', 'slice'):
        with pytest.raises(ValueError, match='PHASE_RADIAL_INTENSITY_REQUIRED'):
            phase_fit_snapshot(replace(s, key=replace(key, selection=(('meaning', meaning),))))
