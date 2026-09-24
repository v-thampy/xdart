"""Independent ray oracles; these do not certify the affine MG calibration model."""

import json

import numpy as np
import pytest

from xrd_tools.analysis.xu_stitch_calibration import (
    XuStitchCalibrationInput,
    authored_surface_calibration,
    canonical_surface_resource_bytes,
    capture_xu_stitch_calibration,
)
from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
from xrd_tools.core.scan import ScanFrame
from xrd_tools.integrate.multi import stitch_1d_streaming
from xrd_tools.integrate.xu_stitch import (
    XuPowderQProvider,
    resolve_xu_stitch_effective_geometry,
    run_xu_hist_stitch_1d,
)
from xrd_tools.sources.memory import MemoryFrameSource


def _stacked_q(shape, pitches, center, distance, wavelength, delta, nu):
    """Lab beam +Y; camera row -X, column +Z; outer nu carries inner del."""
    row, col = np.indices(shape, dtype=float)
    ray = np.stack((-(row - center[0]) * pitches[0],
                    np.full(shape, distance), (col - center[1]) * pitches[1]))
    d, n = np.deg2rad([delta, nu])
    rz = np.array([[np.cos(d), np.sin(d), 0],
                   [-np.sin(d), np.cos(d), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(n), -np.sin(n)],
                   [0, np.sin(n), np.cos(n)]])
    outgoing = np.einsum('ij,jhw->ihw', rx @ rz, ray)
    outgoing /= np.linalg.norm(ray, axis=0)
    outgoing[1] -= 1
    return 2 * np.pi / wavelength * np.linalg.norm(outgoing, axis=0)


def _receipt(tmp_path, canonical=False):
    if canonical:
        content = canonical_surface_resource_bytes()
    else:
        from pyFAI.detectors import Detector
        with xu_runtime_session() as session:
            wavelength = float(session.xu.en2lam(17000.004))
        value = authored_surface_calibration(
            detector=Detector(pixel1=75e-6, pixel2=172e-6, max_shape=(9, 13)),
            cch1=3.2, cch2=5.4, distance_m=.39,
            del_offset_deg=1.7, nu_offset_deg=-.8, energy_eV=17000.004,
            xu_wavelength_A=wavelength, source_motors={'del': 'del', 'nu': 'nu'},
            rotation=0, control_domain={'del': [-50., 50.], 'nu': [-35., 35.]},
            domain={'del': [-50., 50.], 'nu': [-35., 35.]},
            rms_q=.001, condition=10., n_frames=5, n_points=40,
        )
        content = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    (tmp_path / 'geometry.json').write_bytes(content)
    return capture_xu_stitch_calibration(
        XuStitchCalibrationInput('geometry.json'), project_root=tmp_path,
    )


@pytest.mark.parametrize('canonical', [False, True])
def test_xu_stacked_arm_matches_independent_ray_at_every_pixel(tmp_path, canonical):
    receipt = _receipt(tmp_path, canonical)
    asset = receipt.projection.value
    det, acq = asset['detector'], asset['acquisition']
    with xu_runtime_session() as session:
        geometry = resolve_xu_stitch_effective_geometry(receipt, session)
        provider = XuPowderQProvider(geometry, acq)
        for delta in (-40., -20., 0., 20., 40.):
            for nu in (-29., -10., 0., 10., 29.):
                expected = _stacked_q(
                    tuple(det['shape']), (det['pixel1_m'], det['pixel2_m']),
                    (det['cch1_px'], det['cch2_px']), det['distance_m'],
                    acq['xu_mapping_wavelength_A'],
                    delta - acq['detector_offsets_deg']['del'],
                    nu - acq['detector_offsets_deg']['nu'],
                )
                with provider.frame(nu=nu, del_=delta) as lease:
                    actual = lease.q_magnitude()
                    np.testing.assert_allclose(actual, expected, atol=8e-14, rtol=0)
                    del actual


def _exact_pose_ai(delta, nu):
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    d, n = np.deg2rad([delta, nu])
    # pyFAI basis (row, column, beam). Compose the physical arm first,
    # then decompose into pyFAI's R3 @ R2 @ R1 Euler convention.
    outer = np.array([[1, 0, 0], [0, np.cos(n), np.sin(n)],
                      [0, -np.sin(n), np.cos(n)]])
    inner = np.array([[np.cos(d), 0, -np.sin(d)], [0, 1, 0],
                      [np.sin(d), 0, np.cos(d)]])
    pose = outer @ inner
    rotations = (np.arctan2(-pose[2, 1], pose[2, 2]),
                 np.arctan2(pose[2, 0], np.hypot(pose[2, 1], pose[2, 2])),
                 np.arctan2(pose[1, 0], pose[0, 0]))
    return AzimuthalIntegrator(
        dist=.39, poni1=(3.2 + .5) * 75e-6, poni2=(5.4 + .5) * 172e-6,
        rot1=rotations[0], rot2=rotations[1], rot3=rotations[2],
        detector=Detector(pixel1=75e-6, pixel2=172e-6, max_shape=(9, 13)),
        wavelength=.729e-10,
    )


def test_pyfai_exact_stacked_pose_q_and_streamed_constant_radiance():
    angles = [(0., 0.), (30., 0.), (0., 20.), (30., 20.),
              (-30., 20.), (30., -20.), (-30., -20.), (45., 29.)]
    monitors = np.arange(1., len(angles) + 1.)
    images = []
    row, col = np.indices((9, 13), dtype=float)
    radius = np.sqrt(.39**2 + ((row - 3.2) * 75e-6)**2
                     + ((col - 5.4) * 172e-6)**2)
    solid_angle = 75e-6 * 172e-6 * .39 / radius**3
    for (delta, nu), monitor in zip(angles, monitors):
        ai = _exact_pose_ai(delta, nu)
        expected = _stacked_q((9, 13), (75e-6, 172e-6), (3.2, 5.4),
                              .39, .729, delta, nu)
        np.testing.assert_allclose(ai.qFunction(row, col) / 10, expected,
                                   atol=8e-14, rtol=0)
        # qArray builds float32 pixel-index grids upstream; the direct float64
        # coordinate check above isolates the rotation arithmetic itself.
        np.testing.assert_allclose(ai.qArray() / 10, expected, atol=5e-8, rtol=0)
        images.append(7.5 * monitor * solid_angle)
        ai.reset(collect_garbage=False)
    result, diagnostics = stitch_1d_streaming(
        images, len(images), lambda: (_exact_pose_ai(*a) for a in angles),
        npt=80, radial_range=(0., 9.), normalization=monitors, method='full',
    )
    occupied = diagnostics.normalization > 0
    assert occupied.any()
    np.testing.assert_allclose(result.intensity[occupied], 7.5, rtol=2e-6)


@pytest.mark.parametrize('backend', ['multigeometry', 'xu_hist'])
def test_overlap_uses_the_declared_backend_monitor_weighting(tmp_path, backend):
    # Two overlapping views with deliberately different signal/monitor ratios
    # distinguish sum(signal)/sum(monitor) from mean(signal/monitor).
    angles = ((20., 10.), (30., 20.))
    monitors, signals = (1., 3.), (10., 90.)
    if backend == 'multigeometry':
        images = [np.full((9, 13), signal) for signal in signals]
        result, diagnostics = stitch_1d_streaming(
            images, 2, lambda: (_exact_pose_ai(*a) for a in angles),
            npt=2, radial_range=(0., 18.), normalization=monitors,
            correct_solid_angle=False, method='full',
        )
        np.testing.assert_allclose(result.intensity[:1], [25.], rtol=2e-6)
        np.testing.assert_allclose(diagnostics.normalization, [117 * 4, 0], rtol=2e-6)
    else:
        receipt = _receipt(tmp_path)
        with xu_runtime_session() as session:
            geometry = resolve_xu_stitch_effective_geometry(receipt, session)
            solid = geometry.solid_angle.copy()
        source = MemoryFrameSource([
            ScanFrame(index=i, image=signal * solid,
                      metadata={'del': delta, 'nu': nu, 'monitor': monitor})
            for i, ((delta, nu), signal, monitor)
            in enumerate(zip(angles, signals, monitors))
        ])
        result = run_xu_hist_stitch_1d(
            receipt, source, frame_indices=(0, 1), npt=1, radial_range=(0., 9.),
            monitor_key='monitor', max_frame_bytes=1024 * 1024,
        )
        np.testing.assert_allclose(result.payload.intensity, [20.], rtol=2e-12)
        np.testing.assert_allclose(result.diagnostics.normalization, [2 * solid.sum()],
                                   rtol=2e-12)
