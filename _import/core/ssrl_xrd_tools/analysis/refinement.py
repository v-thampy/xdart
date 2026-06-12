"""Stub for Rietveld/LeBail refinement. Planned: GSAS-II, FullProf, lmfit."""
from __future__ import annotations

_MSG = (
    "Refinement is not yet implemented. "
    "Planned backends: GSAS-II, FullProf, lmfit."
)


def lebail_fit(*args, **kwargs):
    raise NotImplementedError(_MSG)


def rietveld_fit(*args, **kwargs):
    raise NotImplementedError(_MSG)


def lattice_params(*args, **kwargs):
    raise NotImplementedError(_MSG)


__all__: list[str] = []
