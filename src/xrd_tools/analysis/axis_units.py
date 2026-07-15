"""Small, explicit unit contracts for q-based analysis."""

from __future__ import annotations

__all__ = ["canonical_q_unit", "require_inverse_angstrom"]


_INVERSE_ANGSTROM_UNITS = {
    "q_a^-1",
    "q_a-1",
    "a^-1",
    "a-1",
    "1/a",
    "1/angstrom",
    "angstrom^-1",
    "angstrom-1",
    "inverse_angstrom",
    # q_total is the magnitude required by d = 2 pi / q in GI polar maps.
    "qtot_a^-1",
    "qtot_a-1",
    "qtotal_a^-1",
    "qtotal_a-1",
    "q_total_a^-1",
    "q_total_a-1",
}


def canonical_q_unit(q_unit: str | None) -> str:
    """Return a stable radial q-unit label without converting axis values.

    Documented inverse-angstrom q aliases, including GI ``qtot`` labels, are
    canonicalized to ``q_A^-1``. Unknown or incompatible labels remain
    distinguishable so callers can preserve provenance or reject them.
    """
    raw = str(q_unit or "").strip()
    if not raw:
        return ""
    normalized = (
        raw.lower()
        .replace(" ", "")
        .replace("\u00e5", "angstrom")
        .replace("\u2212", "-")
    )
    if normalized in _INVERSE_ANGSTROM_UNITS:
        return "q_A^-1"
    return normalized


def require_inverse_angstrom(q_unit: str | None, *, operation: str = "q-based analysis") -> str:
    """Require a radial axis proven to be inverse angstrom.

    This validates only the supplied provenance. It intentionally does not
    convert 2-theta or inverse-nanometre axes: callers must convert the values
    explicitly before requesting a result in angstrom.
    """
    canonical = canonical_q_unit(q_unit)
    if canonical != "q_A^-1":
        raise ValueError(
            f"{operation} requires an explicit inverse-angstrom q unit; "
            f"got {q_unit or 'unspecified'!r}. Convert 2-theta or "
            "inverse-nanometre axes to q_A^-1 before fitting."
        )
    return canonical
