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

#: Rotation / incidence hints matched as token AFFIXES (a name token equal to,
#: starting with, or ending with one of these reads as a rotation axis).  All
#: are >= 3 chars — the 2-char ``th``/``om`` hints were dropped (F3): they only
#: ever matched through the middle of unrelated words (wid**th**, h**om**e),
#: and every genuine ``th``-family axis is covered by :data:`GI_MOTOR_PREFERENCE`
#: or by ``theta``/``gon`` here.
_ROTATION_AFFIX_HINTS = (
    "eta", "theta", "omega", "phi", "chi", "gon", "rot", "ang",
)

#: Incidence-axis names matched only as a WHOLE token (``sample_mu`` yes,
#: ``muffin_x`` no).  Kept equality-only because ``mu`` is 2 chars and
#: ``alpha`` as an affix would over-match.
_ROTATION_TOKEN_ALIASES = ("mu", "alpha", "alphai", "incidence")

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
        for hint in _ROTATION_AFFIX_HINTS:
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
