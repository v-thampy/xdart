# -*- coding: utf-8 -*-
"""Pure, Qt-free GI (grazing-incidence) theta-motor default policy.

Single source of truth for the incidence-motor default rule, shared by the
headless run-configuration freeze (``RunIntent.freeze`` resolves the effective
motor from a supplied choices list) and the xdart GUI dropdowns (which import
this module; the former ``xdart...gi_motor_defaults`` is now a thin re-export).
No Qt/pyqtgraph dependency — keeps ``xdart`` thin and the policy reusable.

"""

from __future__ import annotations

import re

#: Named incidence-motor preference, highest first (case-insensitive equality).
GI_MOTOR_PREFERENCE = (
    "th", "eta", "halpha", "gonth", "theta", "alpha_i", "mu", "incidence",
)

#: Rotation hints with PER-HINT affix rules (review wf_3614041c: a single
#: equals/starts/ends rule still fired through unrelated words at token EDGES —
#: **chi**ller, hexa**gon**, **ang**strom, m**eta** — re-creating the F3 leak).
#:
#: Both affixes — no realistic English-word collision at either token edge:
#: two**theta**/``thetaz``, sam**omega**, sam**phi``/``phiz``, x**rot**/``rotz``,
#: ``angle(s)``.
_ROTATION_HINTS_AFFIX = ("theta", "omega", "phi", "rot", "angle")
#: PREFIX-only: ``gonio``/``goniometer`` yes, hexa**gon** no.
_ROTATION_HINTS_PREFIX = ("gon",)
#: SUFFIX-only: ``samchi`` yes, **chi**ller no.
_ROTATION_HINTS_SUFFIX = ("chi",)

#: Incidence/rotation names matched only as a WHOLE token (``sample_mu`` yes,
#: ``muffin_x`` no).  ``th``/``om``/``mu`` are REAL, common beamline axes
#: (maintainer clarification 2026-07-12: they were never banned — F3 only
#: stopped mid-word SUBSTRING matching, wid**th**/h**om**e); whole-token
#: matching keeps them while staying leak-free, and catches decorated forms
#: (``sam_th``, ``th2``, ``sample_om``).  ``alpha``/``eta``/``ang`` as affixes
#: would over-match (m**eta**/b**eta** are NOT incidence axes — beta is
#: conventionally the EXIT angle, and Manual is safer).  ``halpha`` (the
#: bl11-3 incidence axis) is listed so decorated forms are caught — bare
#: names in :data:`GI_MOTOR_PREFERENCE` already win via the preference pass.
_ROTATION_TOKEN_ALIASES = (
    "th", "om", "mu", "eta", "ang", "alpha", "alphai", "halpha", "incidence",
)

#: camelCase boundary (lower/digit → upper), applied before lowercasing.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

#: Token separators after lowercasing: any run of non-alphanumerics.
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _tokens(name: str) -> list[str]:
    """Split a motor name into lowercase word tokens.

    Boundaries are non-alphanumeric runs AND camelCase transitions; trailing
    digits are stripped per token so ``theta2``/``phi1`` read as their axis.
    """
    spaced = _CAMEL_BOUNDARY.sub(" ", str(name))
    out: list[str] = []
    for tok in _NON_ALNUM.split(spaced.lower()):
        tok = tok.rstrip("0123456789")
        if tok:
            out.append(tok)
    return out


def _looks_like_rotation(name: str) -> bool:
    """True when *name* reads as a rotation / incidence axis, token-aware."""
    for tok in _tokens(name):
        if tok in _ROTATION_TOKEN_ALIASES:
            return True
        if any(tok.startswith(h) for h in _ROTATION_HINTS_PREFIX):
            return True
        if any(tok.endswith(h) for h in _ROTATION_HINTS_SUFFIX):
            return True
        for hint in _ROTATION_HINTS_AFFIX:
            if tok.startswith(hint) or tok.endswith(hint):
                return True
    return False


def pick_default_gi_motor(motors) -> str:
    """Choose the default GI incidence motor from *motors*.

    Returns a motor NAME from *motors* (the named-preference match, else the
    first rotation-sounding motor) or the literal ``'Manual'`` when nothing
    looks like an incidence axis.  Never returns a name that is not in *motors*.
    """
    names = [str(m) for m in (motors or []) if str(m)]
    if not names:
        return "Manual"
    lower = {m.lower(): m for m in names}
    for pref in GI_MOTOR_PREFERENCE:
        if pref in lower:
            return lower[pref]
    for m in names:
        if _looks_like_rotation(m):
            return m
    return "Manual"
