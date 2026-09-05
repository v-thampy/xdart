# -*- coding: utf-8 -*-
"""One place to make a RECOGNIZED xdart processed result for a fixture.

OWNER-GATE-RAW-TARGET-20260905 made an ordinary Replace refuse an existing file
that is not a current xdart processed result. Several older fixtures stood in a
prior target with arbitrary bytes -- `b"immutable prior bytes"` and the like --
because their subject was the transaction's backup/restore machinery and the
prior's content was irrelevant to it. Under the new rule those bytes are exactly
the unrecognized occupant that must be refused, so the fixtures describe a
scenario the product now forbids.

Hand-building a conforming file is a trap: the recognizer wants current schema
identity AND real integrated content, and a hand-built near-miss would either
fail confusingly or, worse, pass for the wrong reason. This writes one through
the REAL writer instead, so whatever the recognizer requires is satisfied by
construction rather than by my reading of it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_recognized_result(target: Path | str, *, labels=(0,)) -> bytes:
    """Write a genuine processed result at *target*; return its bytes."""
    from xrd_tools.core.containers import IntegrationResult1D
    from xrd_tools.core.scan import Scan, ScanFrame
    from xrd_tools.reduction import (
        FrameReduction, NexusSink, ReductionPlan, ReductionResult,
    )

    target = Path(target)
    sink = NexusSink(
        target, overwrite=True, atomic=False, flush_every=None,
        source_base=str(target.parent),
    )
    sink.begin(Scan("recognized-prior", []), ReductionPlan(integration_2d=None))
    for label in labels:
        sink.write(
            ScanFrame(label),
            FrameReduction(label, result_1d=IntegrationResult1D(
                radial=np.linspace(0.1, 1.0, 8),
                intensity=np.full(8, float(label)),
                sigma=np.full(8, float(label) / 10 + 0.1),
                unit="q_A^-1",
            )),
        )
    sink.finish(ReductionResult("recognized-prior", {}, len(tuple(labels))))
    return target.read_bytes()
