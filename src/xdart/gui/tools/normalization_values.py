"""One Normalization row group for Stitch and RSM (AUD-9 stage 2b, decision 4).

The maintainer, 2026-10-01: "There are monitors both before and after the foil. So add the 4th choice".
Four choices, one divisor per frame (``xrd_tools.corrections.normalization.frame_divisors``):

- None.
- Monitor: a monitor after the foils, whose counts already include their attenuation.
- Foil transmission × exposure: the foils' transmission and the counting time.
- Monitor × foil transmission: a monitor before the foils, which does not see their attenuation.

Exposure is never combined with a monitor, which already counts over the exposure. Qt-free: the rows are
the shared ``FormRow`` s Int's processing composer lays out; a page maps the selection onto its plan.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
import math

from xrd_tools.session.readiness import ControlFieldKind, ControlFormField

NONE, MONITOR, FOIL_EXPOSURE, MONITOR_FOIL = (
    "None", "Monitor", "Foil transmission × exposure", "Monitor × foil transmission")
NORMALIZATION_CHOICES = (NONE, MONITOR, FOIL_EXPOSURE, MONITOR_FOIL)
#: The group's keys, in their rows' order (each page prefixes its own path).
NORMALIZATION_KEYS = ("normalization", "monitor", "foil_column", "exposure_column", "foil_lengths")
#: RSM's preset (the example notebook's second set): the lengths depend on the X-ray energy.
DEFAULT_FOIL_LENGTHS = "1.06, 3.04, 4.65, 9.5"
DEFAULT_FOIL_COLUMN, DEFAULT_EXPOSURE_COLUMN = "foil status", "Seconds"
LABELS = {"normalization": "Normalization", "monitor": "Monitor", "foil_column": "Foil column",
          "exposure_column": "Exposure column", "foil_lengths": "Foil absorption"}
TIPS = {
    "normalization": (
        "Each frame's divisor. Monitor: a monitor after the foils (its counts include their attenuation). "
        "Foil transmission × exposure: the foils' transmission and the counting time. Monitor × foil "
        "transmission: a monitor before the foils, which does not see them."),
    "monitor": "The beam monitor column (for a monitor before the foils, choose Monitor × foil transmission).",
    "foil_column": ("The column of foil codes: up to four digits, each the number of that foil in the beam "
                    "(the example notebook's foil_to_factor)."),
    "exposure_column": "The counting time column (seconds), for Foil transmission × exposure.",
    "foil_lengths": (
        "Each foil's absorption (μ·t), four numbers, first foil first: the transmission is "
        "exp(−Σ count × length). They depend on the energy; the example notebook uses "
        "(1.06, 3.04, 4.64, 13.22) and (1.06, 3.04, 4.65, 9.5)."),
}


def rows_in_use(choice: str) -> dict[str, bool]:
    """Which of the group's rows the choice reads (the others are greyed, their values kept)."""
    foil = choice in (FOIL_EXPOSURE, MONITOR_FOIL)
    return {"normalization": True, "monitor": choice in (MONITOR, MONITOR_FOIL), "foil_column": foil,
            "exposure_column": choice == FOIL_EXPOSURE, "foil_lengths": foil}


def parse_foil_lengths(text) -> tuple[float, float, float, float]:
    """Four finite, non-negative absorption lengths, not all zero, from "a, b, c, d" (or spaces)."""
    parts = [part for part in str(text or "").replace(",", " ").split() if part]
    if len(parts) != 4:
        raise ValueError("four numbers are needed, first foil first")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError:
        raise ValueError("four numbers are needed, first foil first") from None
    if not all(math.isfinite(value) and value >= 0 for value in values) or not any(values):
        raise ValueError("each must be a number ≥ 0, and not all zero")
    return values  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class NormalizationSelection:
    """A complete choice: the monitor, foil and exposure columns it reads (None when unused) and the lengths."""

    choice: str
    monitor: str | None = None
    foil: str | None = None
    exposure: str | None = None
    lengths: tuple[float, float, float, float] | None = None


def _column(value) -> str | None:
    text = "" if value is None else str(value).strip()
    return None if text in ("", NONE) else text


def normalization_selection(choice, monitor=None, foil=None, exposure=None, lengths="") -> NormalizationSelection:
    """The selection a page's plan takes; ``ValueError`` naming what the choice still needs."""
    choice = str(choice or NONE)
    if choice not in NORMALIZATION_CHOICES:
        raise ValueError(f"Normalization must be one of {', '.join(NORMALIZATION_CHOICES)}")
    use = rows_in_use(choice)
    picked = {"monitor": _column(monitor) if use["monitor"] else None,
              "foil": _column(foil) if use["foil_column"] else None,
              "exposure": _column(exposure) if use["exposure_column"] else None}
    for key, label in (("monitor", "a monitor column"), ("foil", "a foil column"),
                       ("exposure", "an exposure column")):
        if use[{"monitor": "monitor", "foil": "foil_column", "exposure": "exposure_column"}[key]] \
                and picked[key] is None:
            raise ValueError(label)
    if picked["foil"] is not None and picked["foil"] == picked["exposure"]:
        raise ValueError("distinct foil and exposure columns")
    parsed = None
    if use["foil_lengths"]:
        try:
            parsed = parse_foil_lengths(lengths)
        except ValueError as error:
            raise ValueError(f"Foil absorption: {error}") from None
    return NormalizationSelection(choice, picked["monitor"], picked["foil"], picked["exposure"], parsed)


def choice_from(monitor, foil=None, exposure=None) -> str:
    """The choice a stored selection holds (a form or older settings: a monitor alone is Monitor)."""
    monitor, foil, exposure = _column(monitor), _column(foil), _column(exposure)
    if foil is not None:
        return FOIL_EXPOSURE if exposure is not None else MONITOR_FOIL
    return MONITOR if monitor is not None else NONE


def normalization_fields(fields: Mapping[str, ControlFormField], values: Mapping[str, object],
                         columns: Iterable[str]) -> dict[str, ControlFormField]:
    """The group's fields projected for Int's composer: the labels, the choices (the scan's columns, the
    current value kept), and the rows the choice does not read greyed with a reason."""
    columns = tuple(name for name in columns if str(name).strip())
    choice = str(values.get("normalization") or NONE)
    use = rows_in_use(choice)
    projected = {}
    for key in NORMALIZATION_KEYS:
        field, value = fields[key], values.get(key, fields[key].value)
        choices = field.choices
        if key == "normalization":
            choices = NORMALIZATION_CHOICES
        elif key in ("foil_column", "exposure_column"):
            keep = str(value or "").strip()
            choices = (NONE, *columns) + ((keep,) if keep and keep != NONE and keep not in columns else ())
        projected[key] = replace(
            field, label=LABELS[key], value=value, choices=choices,
            kind=ControlFieldKind.LINE if key == "foil_lengths" else ControlFieldKind.COMBO,
            enabled=use[key], reason="" if use[key] else f"{LABELS[key]} is not read by {choice}.")
    return projected


__all__ = [
    "DEFAULT_EXPOSURE_COLUMN", "DEFAULT_FOIL_COLUMN", "DEFAULT_FOIL_LENGTHS", "FOIL_EXPOSURE", "LABELS",
    "MONITOR", "MONITOR_FOIL", "NONE", "NORMALIZATION_CHOICES", "NORMALIZATION_KEYS", "NormalizationSelection",
    "TIPS", "choice_from", "normalization_fields", "normalization_selection", "parse_foil_lengths",
    "rows_in_use",
]
