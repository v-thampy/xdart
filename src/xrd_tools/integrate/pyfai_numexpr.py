"""One process-wide lock for the pyFAI work that reaches numexpr.

numexpr 2.14 no longer serializes concurrent ``evaluate`` calls (``necompiler.evaluate_lock`` is
commented out), and four of Int's reduction workers under ``OMP_NUM_THREADS=1`` corrupted the heap
inside numexpr while building their integrators' geometry (2026-09-02). pyFAI reaches numexpr in a
unit's formula (a centre array), the polarization array, a GI fiber unit and its window, and a merge
of two results (``IntegrateResult.union`` recomputes the means); its integration engines run in C,
without the GIL and without numexpr, so threads share those.

Every thread that does such work while others may hold this lock for it: Int's reduction workers
while they build an integrator's geometry, and Stitch's frame workers, its merge and its scouting
passes (``xrd_tools.integrate.multi``).
"""
from __future__ import annotations

import threading

PYFAI_NUMEXPR_LOCK = threading.Lock()

__all__ = ["PYFAI_NUMEXPR_LOCK"]
