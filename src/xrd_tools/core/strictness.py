# -*- coding: utf-8 -*-
"""Strictness policy for the headless reduction + reader seams (D7).

A :class:`StrictPolicy` says, per degradation, whether the pipeline RAISES
(loud) or silently / with-a-warning DEGRADES (graceful).  Headless callers
(``run_reduction``, the io readers) default to :meth:`StrictPolicy.loud`: a
scripted / batch reduction now RAISES on a degradation instead of writing bad
data.  The xdart GUI opts INTO :meth:`StrictPolicy.graceful` (it must never
abort a whole-scan save — it drops a bad frame and keeps going).

Three independent per-degradation switches (not one ``strict: bool``) so an
analysis caller can, say, tolerate a thumbnail substitution while still
demanding a usable normalization:

* ``missing_normalization`` — a configured monitor is missing / zero /
  non-finite, so the frame would be written UN-normalized;
* ``gi_all_dummy`` — the 2D integration is entirely dummy (no usable data);
* ``thumbnail_fallback`` — the full-resolution raw is unavailable, so a
  dequantized thumbnail would be substituted.

Import-light (a frozen dataclass + ``ValueError`` subclasses; no numpy / h5py /
pyFAI / Qt) so both ``reduction`` and the ``io`` reader seam can depend on it
without an upward import — see the ``test_architecture_guards`` core guard.
"""
from __future__ import annotations

from dataclasses import dataclass


class StrictnessError(ValueError):
    """Base for a strictness violation raised under a loud :class:`StrictPolicy`.

    Subclasses :class:`ValueError` (like :class:`~xrd_tools.reduction.core.GIFreezeError`)
    so existing broad ``except ValueError`` callers still catch it, while
    ``except StrictnessError`` catches the whole family and the specific
    subclasses stay individually catchable.
    """


class MissingNormalizationError(StrictnessError):
    """A configured monitor is unusable, so the frame would be UN-normalized."""


class GIAllDummyError(StrictnessError):
    """The 2D integration is entirely dummy — it would persist no usable data."""


@dataclass(frozen=True, slots=True)
class StrictPolicy:
    """Per-degradation loud/graceful switches for the reduction + reader seams.

    ``True`` = loud (RAISE on the degradation); ``False`` = graceful (warn-once
    or silently degrade).  The bare ``StrictPolicy()`` is loud (all-True), so a
    caller that forgets to choose gets the safe, fail-loud default.
    """

    missing_normalization: bool = True
    gi_all_dummy: bool = True
    thumbnail_fallback: bool = True

    @classmethod
    def loud(cls) -> "StrictPolicy":
        """Every degradation raises — the headless default."""
        return cls(missing_normalization=True, gi_all_dummy=True,
                   thumbnail_fallback=True)

    @classmethod
    def graceful(cls) -> "StrictPolicy":
        """No degradation raises — the GUI's explicit opt-in (never abort)."""
        return cls(missing_normalization=False, gi_all_dummy=False,
                   thumbnail_fallback=False)
