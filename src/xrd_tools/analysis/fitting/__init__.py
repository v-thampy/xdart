"""
Peak fitting, background subtraction, and peak finding for XRD data.

The fitting stack depends on optional packages such as ``lmfit`` and
``pymatgen``.  This package keeps its public import surface lazy so light
notebook helpers like :class:`FitConfig` remain importable without installing
``xrd-tools[fitting]``.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    # Structure-agnostic fitting
    "fit_line_cut": ".fit",
    "fit_peaks": ".fit",
    "fit_2d_slice": ".fit",
    "PeakFitResult1D": ".fit",
    "get_peak_model": ".fit",
    # lmfit models
    "LorentzianSquaredModel": ".models",
    "AsymmetricRectangleModel": ".models",
    "AssymetricRectangleModel": ".models",
    "Gaussian2DModel": ".models",
    "LorentzianSquared2DModel": ".models",
    "PseudoVoigt2DModel": ".models",
    "Pvoigt2DModel": ".models",
    "PlaneModel": ".models",
    "lorentzian_squared": ".models",
    "gauss_2D": ".models",
    "lor2_2D": ".models",
    "pvoigt_2D": ".models",
    # Background / peak utilities
    "snip_1d": ".background",
    "chebyshev_background": ".background",
    "fit_background": ".background",
    "subtract_background": ".background",
    "extract_peaks": ".peaks",
    "peak_table": ".peaks",
    # Structure-informed fitting
    "PhaseFitter": ".phase_fitting",
    "MultiPhaseResult": ".phase_fitting",
    # Batch helpers
    "FitConfig": ".batch",
    "FitResultStore": ".batch",
    "fit_sequence": ".batch",
    "fit_nexus": ".batch",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    try:
        module = import_module(module_name, __name__)
        value = getattr(module, name)
    except ModuleNotFoundError as exc:
        if exc.name not in {"lmfit", "pymatgen"}:
            raise
        value = _missing_optional_symbol(name, exc.name)
    globals()[name] = value
    return value


def _missing_optional_symbol(name: str, missing: str) -> Any:
    msg = (
        f"{name} requires optional dependency {missing!r}. "
        "Install it with `pip install xrd-tools[fitting]`."
    )
    if name[:1].isupper():
        class MissingOptional:
            def __init__(self, *args, **kwargs):
                raise ImportError(msg)

            def __repr__(self) -> str:
                return f"<missing optional fitting symbol {name}>"

        MissingOptional.__name__ = name
        MissingOptional.__qualname__ = name
        return MissingOptional

    def _missing(*args, **kwargs):
        raise ImportError(msg)

    _missing.__name__ = name
    _missing.__qualname__ = name
    _missing.__doc__ = msg
    return _missing
