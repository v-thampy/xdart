# -*- coding: utf-8 -*-
"""Pure helpers for the parameter-vs-frame trend (analyzer framework, Qt-free).

The flat ``params`` an analyzer emits (``center_0`` / ``fwhm_1`` / …) are grouped
into per-peak *families* for the trend plot (one curve per peak), and the
``{frame_index: params}`` accumulator is flattened to a ``(labels, columns)``
table for CSV export.  No Qt / pyqtgraph here so it is unit-testable headlessly
and shared by the embedded third-row plot."""

#: param-family key -> friendly label.  Families a peak analyzer emits are ``center`` / ``center_err`` / ``fwhm`` /
#: ``amplitude`` (see _peak_params), and the displayed fits' every other peak parameter and background parameter
#: (NEXT-128-WINDOW-POLISH item 14; a background's are named by ``family_label``).
FAMILY_LABELS = {
    "center": "Peak center",
    "center_err": "Center uncertainty",
    "fwhm": "FWHM",
    "amplitude": "Amplitude",
    "height": "Height",
    "sigma": "σ",
    "sigma_r": "σ (right side)",
    "fraction": "Lorentzian fraction",
    "expon": "Pearson VII exponent",
    "voigt_gamma": "Voigt γ",
    # a phase fit's kinds (NEXT-128-WINDOW-POLISH), one member per phase: ``frac_<phase>`` and ``<phase>_<cell>``
    "frac": "Phase fraction",
    "a": "Lattice a", "b": "Lattice b", "c": "Lattice c",
    "alpha": "Lattice α", "beta": "Lattice β", "gamma": "Lattice γ",
}
#: families whose VALUES are in the pattern's x-unit (vs. intensity).
X_UNIT_FAMILIES = {"center", "center_err", "fwhm", "sigma", "sigma_r", "voigt_gamma"}
#: families with a unit of their own: a fraction has none; cell lengths are in Å, cell angles in degrees
FAMILY_UNITS = {"frac": "", "a": "Å", "b": "Å", "c": "Å", "alpha": "°", "beta": "°", "gamma": "°",
                "fraction": "", "expon": ""}
#: a background's own parameters, families of their own with no peak (NEXT-128-WINDOW-POLISH item 14, "the peak
#: independent bg pars"): ``bg_`` a fitted background's or a template's scale (``bg_A``), ``bgx_`` the part added to
#: a template
BACKGROUND_PREFIXES = ("bg_", "bgx_")
_BACKGROUND_LABELS = {"slope": "Background slope", "intercept": "Background intercept", "c": "Background constant",
                      "A": "Template scale"}


def is_background(key):
    """Whether ``key`` is a background's own parameter (peak-independent)."""
    return str(key).startswith(BACKGROUND_PREFIXES)


def family_label(family):
    """A family as Track names it: its label, a background parameter by what it is ("Background c0"), else its key."""
    if family in FAMILY_LABELS:
        return FAMILY_LABELS[family]
    if is_background(family):
        tail = str(family).split("_", 1)[1]
        return _BACKGROUND_LABELS.get(tail, f"Background {tail}")
    return family
_CELL_KEYS = ("a", "b", "c", "alpha", "beta", "gamma")

#: per-curve colors (one per peak within a family).
CURVE_PENS = [(189, 147, 249), (80, 250, 123), (255, 184, 108),
              (139, 233, 253), (255, 121, 198), (241, 250, 140)]


def split_family(key):
    """``'center_0' -> ('center', 0)``; ``'center_err_2' -> ('center_err', 2)``;
    a key with no trailing ``_<int>`` -> ``(key, 0)``."""
    head, _, tail = key.rpartition("_")
    if head and tail.isdigit():
        return head, int(tail)
    return key, 0


def _phase_family(key, phases):
    """A phase fit's key as ``(kind, phase)``: ``frac_<phase>`` -> ``("frac", phase)``, ``<phase>_<cell>`` ->
    ``(cell, phase)`` for a listed phase (longest name first); None for any other key."""
    # The analyzer publishes cells after fractions: when flat keys collide, the value is a cell.
    for phase in sorted(phases, key=len, reverse=True):
        if key.startswith(f"{phase}_") and key[len(phase) + 1:] in _CELL_KEYS:
            return key[len(phase) + 1:], phase
    if key.startswith("frac_") and key[5:] in phases:
        return "frac", key[5:]
    return None


def group_families(keys, *, phase=False, phase_names=()):
    """An iterable of param keys -> ordered ``{family: [(key, member), ...]}`` so a family's curves plot together:
    a peak fit's members are its peak indices (sorted); a phase fit's keys (``frac_<phase>``, ``<phase>_<cell>``)
    group by kind, their members the phases' names in the order of their fractions (NEXT-128-WINDOW-POLISH).
    With ``phase=True``, cell suffixes also identify phases when lattice-only fits withhold every fraction.
    A phase host supplies its request's ``phase_names`` so names starting with ``frac_`` are not mistaken for
    fractions. Other trend consumers retain their peak and bare-key grouping."""
    keys = list(keys)
    phases = list(phase_names) or [key[5:] for key in keys if key.startswith("frac_") and len(key) > 5]
    if phase and not phases:
        phases = list(dict.fromkeys(head for key in keys for head, _, tail in (key.rpartition("_"),)
                                    if head and tail in _CELL_KEYS))
    fam = {}
    for key in keys:
        # a background's parameter is a family of its own with no peak or phase: the chooser never hides it
        f, member = (key, None) if is_background(key) else (_phase_family(key, phases) or split_family(key))
        fam.setdefault(f, []).append((key, member))
    order = {phase: position for position, phase in enumerate(phases)}
    for f in fam:
        fam[f].sort(key=lambda km: (0, km[1], "") if isinstance(km[1], int) else (1, order.get(km[1], len(order)),
                                                                                   str(km[1])))
    return fam


def member_label(member):
    """A family member as the trend's chooser names it: "Peak 0" for a peak index, a phase by its name."""
    return f"Peak {member}" if isinstance(member, int) else str(member)


def accumulator_to_table(accumulator):
    """``{frame_index: params_dict}`` -> ``(labels, columns)`` sorted by frame
    index, where ``columns`` is an order-preserving ``{param: [value per frame]}``
    (a param missing from a frame is ``nan``) — the CSV / plot table."""
    idxs = sorted(accumulator)
    labels = [str(i) for i in idxs]
    keys = []
    seen = set()
    for i in idxs:
        for k in accumulator[i]:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    columns = {k: [float(accumulator[i].get(k, float("nan"))) for i in idxs]
               for k in keys}
    return labels, columns
