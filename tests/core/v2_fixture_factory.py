"""2e — schema-driven v2 fixture factory.

Test files are built FROM the schema (and the real writer where it
exists), so fixtures cannot lag the layout: a schema change that the
writer honors flows into every fixture automatically.
"""
from __future__ import annotations

import numpy as np

from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.io.nexus import write_integrated_stack
from xrd_tools.io.schema import SCHEMA


def make_v2_entry(h5f, *, frame_indices=(0, 1, 2), n_q=8, n_chi=4,
                  with_2d=True, with_sigma=True, entry="entry", seed=3):
    """Populate ``h5f`` with a schema-conformant v2 entry via the REAL
    writer (write_integrated_stack), returning the entry group.

    The values are deterministic (seeded); the layout is whatever the
    writer+schema produce — that is the point.
    """
    rng = np.random.default_rng(seed)
    e = h5f.require_group(entry)
    fis = [int(i) for i in frame_indices]
    r1d = [
        IntegrationResult1D(
            radial=np.linspace(0.1, 4.0, n_q),
            intensity=rng.random(n_q),
            sigma=(rng.random(n_q) if with_sigma else None),
            unit="q_A^-1",
        )
        for _ in fis
    ]
    r2d = None
    if with_2d:
        r2d = [
            IntegrationResult2D(
                radial=np.linspace(0.1, 4.0, n_q),
                azimuthal=np.linspace(-90.0, 90.0, n_chi),
                intensity=rng.random((n_q, n_chi)),
                sigma=None,
                unit="q_A^-1",
            )
            for _ in fis
        ]
    write_integrated_stack(e, frame_indices=fis, results_1d=r1d,
                           results_2d=r2d)
    return e


def assert_v2_conformant(entry_grp) -> None:
    """Assert every schema-declared group present in the entry conforms."""
    from xrd_tools.io.nexus import validate_group_against_schema

    problems = []
    for gname in SCHEMA.groups:
        if gname in entry_grp:
            problems += validate_group_against_schema(entry_grp[gname], gname)
    assert problems == [], problems
