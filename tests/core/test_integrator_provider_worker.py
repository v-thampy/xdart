"""Regression: the executor-backed integrator provider must preserve a generic
(unnamed) detector's pixel size on worker threads.

A generic pyFAI ``Detector`` carries its pixel size only on the live
``AzimuthalIntegrator`` — the ``PONI`` dataclass stores a detector *name*, not
``pixel1``/``pixel2``.  ``_ReductionIntegratorProvider.standard()`` builds a
per-worker AI (pyFAI AIs aren't thread-safe to share); it must DEEPCOPY the base
AI rather than rebuild from ``scan.poni`` via ``poni_to_integrator`` — the rebuild
drops a generic detector's pixel size to ``None`` and ``integrate1d`` then crashes
(``TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'`` in
``calc_cartesian_positions``).

This is the fresh-Run-on-a-reloaded-``.nxs`` crash (multi-core): a processed scan
with a generic detector seeds ``scan.poni`` but the reduction ran on worker
threads and rebuilt a pixel-less AI.  Reintegrate survived only because it ran
single-worker (owner thread, which returns the base AI directly).
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pyFAI")


def test_provider_worker_preserves_generic_detector_pixel_size():
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    from xrd_tools.core.containers import PONI
    from xrd_tools.reduction.core import _ReductionIntegratorProvider

    # A GENERIC detector: real pixel size, but no resolvable pyFAI name.
    det = Detector(pixel1=100e-6, pixel2=100e-6)
    ai = AzimuthalIntegrator(
        dist=0.1, poni1=5e-3, poni2=5e-3, detector=det, wavelength=1e-10)
    assert ai.detector._pixel1 == 100e-6

    # PONI carries only a detector NAME — empty/"" => nothing to rebuild from.
    poni = PONI(dist=0.1, poni1=5e-3, poni2=5e-3, rot1=0.0, rot2=0.0, rot3=0.0,
                wavelength=1e-10, detector="")

    prov = _ReductionIntegratorProvider(
        scan=SimpleNamespace(poni=poni),
        plan=SimpleNamespace(gi=None),
        ai=ai,
        fi=None,
        initial_incident_angle=None,
    )

    # Owner thread returns the base AI directly (pixel size intact).
    assert prov.standard().detector._pixel1 == 100e-6

    # A WORKER thread must get a thread-isolated DEEPCOPY whose detector still
    # has the pixel size — NOT a poni rebuild that yields _pixel1 = None.
    out: dict = {}
    with ThreadPoolExecutor(max_workers=1) as ex:
        ex.submit(lambda: out.__setitem__("ai", prov.standard())).result()

    worker_ai = out["ai"]
    assert worker_ai is not None
    assert worker_ai is not ai                       # thread-isolated copy
    assert worker_ai.detector._pixel1 == 100e-6      # the fix: pixel size kept
    assert worker_ai.detector._pixel2 == 100e-6


def test_provider_binds_private_run_mask_once_and_resets_warmed_engines(
    monkeypatch: pytest.MonkeyPatch,
):
    """A borrowed AI is untouched and stale safe=False LUTs cannot survive."""
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    from xrd_tools.integrate.single import integrate_1d, integrate_2d
    from xrd_tools.reduction.core import _ReductionIntegratorProvider

    shape = (32, 32)
    detector = Detector(pixel1=100e-6, pixel2=100e-6, max_shape=shape)
    geometric = np.zeros(shape, dtype=bool)
    geometric[:, 0] = True
    detector.mask = geometric
    accepted = AzimuthalIntegrator(
        dist=0.1,
        poni1=1.6e-3,
        poni2=1.6e-3,
        detector=detector,
        wavelength=1e-10,
    )
    candidate = copy.deepcopy(accepted)
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1.0
    candidate.integrate1d(
        image,
        16,
        method="csr",
        mask=np.zeros(shape, dtype=bool),
        safe=False,
    )
    assert candidate.engines

    provider = _ReductionIntegratorProvider(
        scan=SimpleNamespace(poni=object()),
        plan=SimpleNamespace(gi=None),
        ai=accepted,
        fi=None,
        initial_incident_angle=None,
    )
    builds: list[object] = []

    def private_candidate():
        builds.append(object())
        return candidate

    monkeypatch.setattr(provider, "_new_standard", private_candidate)
    run_mask = np.zeros(shape, dtype=bool)
    run_mask[1, :] = True
    bound, admitted = provider.standard_with_run_mask(run_mask, shape)

    assert admitted is True
    assert bound is candidate
    assert provider.standard() is candidate
    assert len(builds) == 1
    assert not candidate.engines
    np.testing.assert_array_equal(accepted.detector.mask, geometric)
    np.testing.assert_array_equal(candidate.detector.mask.astype(bool), geometric | run_mask)
    again, admitted_again = provider.standard_with_run_mask(run_mask, shape)
    assert admitted_again is True and again is candidate
    assert len(builds) == 1

    reference = copy.deepcopy(accepted)
    explicit = geometric | run_mask
    expected_1d = integrate_1d(
        image,
        reference,
        npt=16,
        method="csr",
        mask=run_mask,
        error_model="poisson",
        correctSolidAngle=False,
        safe=False,
    )
    actual_1d = integrate_1d(
        image,
        candidate,
        npt=16,
        method="csr",
        mask=None,
        error_model="poisson",
        correctSolidAngle=False,
        safe=False,
        _detector_mask_is_bound=True,
    )
    expected_2d = integrate_2d(
        image,
        reference,
        npt_rad=16,
        npt_azim=8,
        method="csr",
        mask=run_mask,
        error_model="poisson",
        correctSolidAngle=False,
        safe=False,
    )
    actual_2d = integrate_2d(
        image,
        candidate,
        npt_rad=16,
        npt_azim=8,
        method="csr",
        mask=None,
        error_model="poisson",
        correctSolidAngle=False,
        safe=False,
        _detector_mask_is_bound=True,
    )
    np.testing.assert_array_equal(actual_1d.radial, expected_1d.radial)
    np.testing.assert_allclose(
        actual_1d.intensity,
        expected_1d.intensity,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    np.testing.assert_array_equal(actual_2d.radial, expected_2d.radial)
    np.testing.assert_array_equal(actual_2d.azimuthal, expected_2d.azimuthal)
    np.testing.assert_allclose(
        actual_2d.intensity,
        expected_2d.intensity,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    assert explicit.any()  # both the plan and geometric rows were exercised

    fallback, fallback_admitted = provider.standard_with_run_mask(
        np.zeros((16, 16), dtype=bool),
        (16, 16),
    )
    assert fallback_admitted is False
    assert fallback is accepted


def test_provider_refuses_duck_typed_detector_mask_semantics(
    monkeypatch: pytest.MonkeyPatch,
):
    """A custom AI may store detector.mask but ignore it during integration."""
    from xrd_tools.reduction.core import _ReductionIntegratorProvider

    class DetectorLike:
        shape = (2, 2)
        mask = np.zeros((2, 2), dtype=np.int8)

    class CustomAI:
        detector = DetectorLike()

        def reset_engines(self, **_kwargs):
            return None

        def integrate1d(self, _image, _npt, **kwargs):
            return kwargs["mask"]

        def integrate2d(self, _image, _npt_rad, _npt_azim, **kwargs):
            return kwargs["mask"]

    accepted = CustomAI()
    provider = _ReductionIntegratorProvider(
        scan=SimpleNamespace(poni=object()),
        plan=SimpleNamespace(gi=None),
        ai=accepted,
        fi=None,
        initial_incident_angle=None,
    )
    monkeypatch.setattr(
        provider,
        "_new_standard",
        lambda: (_ for _ in ()).throw(AssertionError("owner AI was cloned")),
    )

    run_mask = np.ones((2, 2), dtype=bool)
    returned, admitted = provider.standard_with_run_mask(run_mask, (2, 2))
    assert admitted is False
    assert returned is accepted
    again, admitted_again = provider.standard_with_run_mask(run_mask, (2, 2))
    assert admitted_again is False
    assert again is accepted


def test_numpy_false_safe_transition_stays_on_explicit_mask_path(
    monkeypatch: pytest.MonkeyPatch,
):
    """A safe=False mask transition must retain the legacy explicit sequence."""
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    from xrd_tools.integrate.single import integrate_1d
    from xrd_tools.reduction import Frame, Integration1DPlan, ReductionPlan
    from xrd_tools.reduction.core import (
        _ReductionIntegratorProvider,
        _reduce_frame,
    )

    shape = (32, 32)

    def new_ai():
        detector = Detector(pixel1=100e-6, pixel2=100e-6, max_shape=shape)
        detector_mask = np.zeros(shape, dtype=bool)
        detector_mask[:, 0] = True
        detector.mask = detector_mask
        return AzimuthalIntegrator(
            dist=0.1,
            poni1=1.6e-3,
            poni2=1.6e-3,
            detector=detector,
            wavelength=1e-10,
        )

    accepted = new_ai()
    reference = new_ai()
    provider = _ReductionIntegratorProvider(
        scan=SimpleNamespace(poni=object()),
        plan=SimpleNamespace(gi=None),
        ai=accepted,
        fi=None,
        initial_incident_angle=None,
    )
    monkeypatch.setattr(
        provider,
        "standard_with_run_mask",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("safe=False path attempted detector binding")
        ),
    )
    plan_mask = np.zeros(shape, dtype=bool)
    plan_mask[1, :] = True
    dynamic = np.zeros(shape, dtype=bool)
    dynamic[2, :] = True
    plan = ReductionPlan(
        integration_1d=Integration1DPlan(
            npt=16,
            method="csr",
            extra={"safe": np.bool_(False), "correctSolidAngle": False},
        ),
        integration_2d=None,
        mask=plan_mask,
    )
    image = np.arange(np.prod(shape), dtype=float).reshape(shape) + 1.0
    frames = (
        Frame(0, image=image),
        Frame(1, image=image, mask=dynamic),
        Frame(2, image=image),
    )
    actual = [
        _reduce_frame(
            frame,
            None,
            plan,
            provider,
            {},
        ).result_1d
        for frame in frames
    ]
    expected = [
        integrate_1d(
            image,
            reference,
            npt=16,
            method="csr",
            mask=mask,
            safe=np.bool_(False),
            correctSolidAngle=False,
        )
        for mask in (plan_mask, plan_mask | dynamic, plan_mask)
    ]
    for observed, baseline in zip(actual, expected):
        np.testing.assert_array_equal(observed.radial, baseline.radial)
        np.testing.assert_allclose(
            observed.intensity,
            baseline.intensity,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
