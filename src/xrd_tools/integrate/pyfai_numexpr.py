"""One process-wide lock for the pyFAI work that reaches numexpr.

numexpr 2.14 no longer serializes concurrent ``evaluate`` calls (``necompiler.evaluate_lock`` is
commented out), and four of Int's reduction workers under ``OMP_NUM_THREADS=1`` corrupted the heap
inside numexpr while building their integrators' geometry (2026-09-02). pyFAI reaches numexpr in a
unit's formula (a centre array), the polarization array, a GI fiber unit and its window, and a merge
of results (2026.9 also recomputes a single frame's means); its integration engines run in C,
without the GIL and without numexpr, so threads share those.

Every thread that does such work while others may hold this lock for it: Int's reduction workers
while they build an integrator's geometry, and Stitch's frame workers, its merge and its scouting
passes (``xrd_tools.integrate.multi``).
"""
from __future__ import annotations

import functools
import threading

PYFAI_NUMEXPR_LOCK = threading.RLock()


def guard_result_means():
    """Guard 2026.9's shared compiled expressions, keeping its integration engines parallel.

    Install lazily before Stitch starts workers. Union and GI paths already hold the same lock;
    reentrancy lets their result merge use this guard too. No upstream package files are edited.
    """
    with PYFAI_NUMEXPR_LOCK:
        import pyFAI
        if pyFAI.version != "2026.9.0":
            return
        from pyFAI.containers import IntegrateResult
        original = IntegrateResult.__recalculate_means__
        if getattr(original, "_xdart_numexpr_guard", False):
            return

        @functools.wraps(original)
        def guarded(result):
            with PYFAI_NUMEXPR_LOCK:
                return original(result)

        guarded._xdart_numexpr_guard = True
        IntegrateResult.__recalculate_means__ = guarded


__all__ = ["PYFAI_NUMEXPR_LOCK", "guard_result_means"]
