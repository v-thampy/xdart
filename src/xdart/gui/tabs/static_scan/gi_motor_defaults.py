# -*- coding: utf-8 -*-
"""Shared policy for the GI (grazing-incidence) θ-motor dropdown default.

Both the image-wrangler ``GI/th_motor`` param and the integrator panel's
``gi_motor`` combo must SHOW every real motor of the source and DEFAULT-select
the same incidence axis, so the two never disagree.  The rule (maintainer spec,
recurring):

1. Show every real motor (not counters) + ``Manual``.
2. Default-select, case-insensitively, the first *named* preference that is
   actually present: ``th, eta, halpha, gonth, theta``.
3. Else the first motor whose name *sounds like* a rotation / incidence axis.
4. Else ``Manual`` (no motor looks like an incidence axis — safer to make the
   user type the angle than to silently pick a translation stage).

Never inject a preference/heuristic name that is not a real motor of the source
(that is the ``th``-leak trap the legacy hard-coded default caused).
"""

from __future__ import annotations

#: Named incidence-motor preference, highest first (case-insensitive equality).
GI_MOTOR_PREFERENCE = ("th", "eta", "halpha", "gonth", "theta")

#: Substrings that mark a motor name as a rotation / incidence axis (fallback
#: when none of :data:`GI_MOTOR_PREFERENCE` is present).  Case-insensitive.
_ROTATION_HINTS = (
    "th", "eta", "theta", "omega", "om", "phi", "chi", "gon", "rot", "ang",
)


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
        ml = m.lower()
        if any(hint in ml for hint in _ROTATION_HINTS):
            return m
    return "Manual"
