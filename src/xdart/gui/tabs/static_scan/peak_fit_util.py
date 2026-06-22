# -*- coding: utf-8 -*-
"""Pure helpers for the parameter-vs-frame trend (analyzer framework, Qt-free).

The flat ``params`` an analyzer emits (``center_0`` / ``fwhm_1`` / …) are grouped
into per-peak *families* for the trend plot (one curve per peak), and the
``{frame_index: params}`` accumulator is flattened to a ``(labels, columns)``
table for CSV export.  No Qt / pyqtgraph here so it is unit-testable headlessly
and shared by the embedded third-row plot."""

#: param-family key -> friendly label.  Families a peak analyzer emits are
#: ``center`` / ``center_err`` / ``fwhm`` / ``amplitude`` (see _peak_params).
FAMILY_LABELS = {
    "center": "Peak center",
    "center_err": "Center uncertainty",
    "fwhm": "FWHM",
    "amplitude": "Amplitude",
}
#: families whose VALUES are in the pattern's x-unit (vs. intensity).
X_UNIT_FAMILIES = {"center", "center_err", "fwhm"}

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


def group_families(keys):
    """An iterable of param keys -> ordered ``{family: [(key, peak_index), ...]}``
    (each family's peaks sorted by index) so a family's curves plot together."""
    fam = {}
    for key in keys:
        f, i = split_family(key)
        fam.setdefault(f, []).append((key, i))
    for f in fam:
        fam[f].sort(key=lambda ki: ki[1])
    return fam


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
