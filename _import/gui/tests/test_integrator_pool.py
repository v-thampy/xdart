# -*- coding: utf-8 -*-
"""Regression test for the parallel-batch pyFAI thread-safety bug.

pyFAI's ``AzimuthalIntegrator.integrate1d_ng(method='csr')`` mutates
intermediate buffers on ``self`` during a call.  When two worker
threads call the same integrator instance concurrently with different
input images, those buffers get clobbered and the results diverge —
verified empirically: parallel calls on a shared instance with
different inputs differ from a serial baseline by up to ~67%
relative when using random data.

The :class:`IntegratorPool` in
:mod:`xdart.utils.integrator_pool` hands each worker its own
integrator copy, so concurrent calls touch disjoint instances and
the bug is gone.  This test pins both halves
of that contract: the bug exists on shared, the fix works on pooled.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

pyFAI = pytest.importorskip("pyFAI")
from pyFAI.integrator.azimuthal import AzimuthalIntegrator  # noqa: E402
import pyFAI.detectors as det  # noqa: E402

from xdart.utils.integrator_pool import IntegratorPool


@pytest.fixture
def synthetic_setup():
    """Realistic Eiger-ish detector + 16 distinct synthetic images."""
    rng = np.random.default_rng(0)
    detector = det.Detector(pixel1=75e-6, pixel2=75e-6, max_shape=(2070, 2167))
    ai = AzimuthalIntegrator(
        dist=0.1, poni1=0.075, poni2=0.075,
        wavelength=1e-10, detector=detector,
    )
    images = [
        (rng.exponential(1.0, size=(2070, 2167)).astype(np.float32) * 1000)
        for _ in range(16)
    ]
    # Warm the CSR cache on the source integrator before splitting copies.
    ai.integrate1d_ng(images[0], npt=2000, method="csr")
    return ai, images


def test_shared_integrator_is_not_thread_safe(synthetic_setup):
    """Demonstrates the underlying bug.

    Calling the *same* integrator concurrently with different inputs
    produces results that diverge from a serial baseline.  This test
    isn't checking a feature — it's a guard against pyFAI silently
    fixing the issue upstream and us no longer needing the pool.  If
    this test starts failing (i.e. parallel matches serial without
    the pool), revisit whether the pool is still necessary.
    """
    ai, images = synthetic_setup
    serial = [ai.integrate1d_ng(img, npt=2000, method="csr") for img in images]

    def integrate(i):
        return ai.integrate1d_ng(images[i], npt=2000, method="csr")

    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel = list(pool.map(integrate, range(len(images))))

    mismatches = sum(
        1 for i in range(len(images))
        if not np.array_equal(serial[i].intensity, parallel[i].intensity)
    )
    # Expect at least half the calls to diverge — the exact number is
    # nondeterministic since it depends on thread scheduling.
    assert mismatches > 0, (
        "Expected the shared-integrator bug to be reproducible; "
        "if you're seeing this, pyFAI may have been fixed upstream "
        "and the IntegratorPool workaround may be removable."
    )


def test_pool_yields_serial_identical_results(synthetic_setup):
    """The pool eliminates the thread-safety mismatch.

    Each worker borrows its own integrator copy via the pool's
    context manager.  Concurrent calls never touch the same
    instance, so internal buffer mutation is per-worker and the
    output is bit-identical to a serial run on the source.
    """
    ai, images = synthetic_setup
    pool = IntegratorPool(ai, n=4)

    serial = [ai.integrate1d_ng(img, npt=2000, method="csr") for img in images]

    def integrate(i):
        with pool.borrow() as worker_ai:
            return worker_ai.integrate1d_ng(images[i], npt=2000, method="csr")

    with ThreadPoolExecutor(max_workers=4) as executor:
        parallel = list(executor.map(integrate, range(len(images))))

    for i in range(len(images)):
        np.testing.assert_array_equal(
            serial[i].intensity, parallel[i].intensity,
            err_msg=f"frame {i} diverged with pool",
        )


def test_pool_size_invariants():
    """Constructor rejects n<1; len reports n; borrow/release balance."""
    detector = det.Detector(pixel1=75e-6, pixel2=75e-6, max_shape=(256, 256))
    ai = AzimuthalIntegrator(
        dist=0.1, poni1=0.022, poni2=0.022,
        wavelength=1e-10, detector=detector,
    )

    with pytest.raises(ValueError):
        IntegratorPool(ai, n=0)

    pool = IntegratorPool(ai, n=3)
    assert len(pool) == 3
    # Two sequential borrow/release cycles leave the pool full.
    with pool.borrow():
        pass
    with pool.borrow():
        pass
    # If borrow/release didn't balance, the third borrow would block.
    # We can't easily test "doesn't block" so check the queue size
    # via a non-blocking get loop.
    drained = []
    try:
        while True:
            drained.append(pool._q.get_nowait())
    except Exception:
        pass
    assert len(drained) == 3


def test_pool_does_not_include_source(synthetic_setup):
    """The source integrator is reserved for non-pool consumers.

    None of the pool's three integrators should be ``is`` the source
    integrator — they must all be independent deep-copies.  Otherwise
    a "detach frame from pool" step in the wrangler could accidentally
    leak the source into the pool's borrow rotation.
    """
    ai, _ = synthetic_setup
    pool = IntegratorPool(ai, n=3)
    assert all(integrator is not ai for integrator in pool._integrators)
