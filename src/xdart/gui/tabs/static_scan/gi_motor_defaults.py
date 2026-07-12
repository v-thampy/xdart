# -*- coding: utf-8 -*-
"""Shared policy for the GI (grazing-incidence) θ-motor dropdown default.

Both the image-wrangler ``GI/th_motor`` param and the integrator panel's
``gi_motor`` combo must SHOW every real motor of the source and DEFAULT-select
the same incidence axis, so the two never disagree.  The rule (maintainer spec,
recurring; extended 2026-07-12 for Codex review F3):

1. Show every real motor (not counters) + ``Manual``.
2. Default-select, case-insensitively, the first *named* preference that is
   actually present: ``th, eta, halpha, gonth, theta, alpha_i, mu, incidence``.
3. Else the first motor whose name *sounds like* a rotation / incidence axis —
   matched TOKEN-AWARE (word boundaries), never substring-anywhere: the 2-char
   substrings ``th``/``om`` wrongly claimed ``slit_wid**th**`` and
   ``sample_h**om**e`` (the F3 leak), silently defaulting to a translation
   stage.  A genuine bare ``th`` axis is still caught by rule 2's exact match.
4. Else ``Manual`` (no motor looks like an incidence axis — safer to make the
   user type the angle than to silently pick a translation stage).

Never inject a preference/heuristic name that is not a real motor of the source
(that is the ``th``-leak trap the legacy hard-coded default caused).
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
#: ``muffin_x`` no).  ``mu``/``om`` are 2 chars and ``alpha``/``eta``/``ang``
#: as affixes would over-match (h**alpha** is fine but alph**a**bet is not;
#: m**eta**/b**eta** are NOT incidence axes — beta is conventionally the EXIT
#: angle, and Manual is safer).  ``om`` here is the exact SPEC omega name: the
#: F3 ban covers the mid-word ``om`` SUBSTRING (h**om**e), not the real axis.
#: ``halpha`` (the bl11-3 incidence axis) is listed so DECORATED forms
#: (``sam_halpha``, ``halpha2``) are caught — bare names in
#: :data:`GI_MOTOR_PREFERENCE` already win via the preference pass.
_ROTATION_TOKEN_ALIASES = (
    "mu", "om", "eta", "ang", "alpha", "alphai", "halpha", "incidence",
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
