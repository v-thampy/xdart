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

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

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
