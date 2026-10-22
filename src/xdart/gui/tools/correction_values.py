"""One Corrections row group for Stitch and RSM, in their Advanced dialogs (G8c unit 4c): Solid angle,
Polarization, Dark, Flat field and Flat cutoff.

Qt-free, as ``normalization_values``: each page lays the rows out with its own row builder under its own path
(``("Stitch", key)``, ``("RSM", key)``) and parses them into its own plan; their labels, defaults, choices and tips
are declared here once. Both pages' dark and flat pass Int's admission (``stitch_native.capture_detector_image``);
a blank Polarization is none, a blank Flat cutoff Int's 10 %.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

#: The group's keys, in their rows' order (each page prefixes its own path).
CORRECTION_KEYS = ("solid_angle", "polarization", "dark_path", "flat_path", "flat_cutoff")
LABELS = {"solid_angle": "Solid angle", "polarization": "Polarization", "dark_path": "Dark",
          "flat_path": "Flat field", "flat_cutoff": "Flat cutoff (% of median)"}
#: Solid angle and polarization on (Stitch's, and RSM's decision 5), the flat cutoff Int's 10 % of the median.
DEFAULTS = {"solid_angle": "On", "polarization": "0.99", "dark_path": "", "flat_path": "", "flat_cutoff": "10"}
SOLID_ANGLE_CHOICES = ("On", "Off")
#: The rows that browse for a correction image.
IMAGE_KEYS = ("dark_path", "flat_path")
TIPS = {
    "solid_angle": "Each pixel's solid angle, as pyFAI's: on by default.",
    "polarization": "Polarization factor from -1 to 1; leave blank for none.",
    "dark_path": "A dark image subtracted from every frame (Int's dark check admits it).",
    "flat_path": "A flat field dividing every frame (Int's flat check admits it).",
    "flat_cutoff": (
        "Flat pixels below this share of the flat's median are treated as dead and masked. Raise it to mask a "
        "region shadowed in the flat; lower it to keep weak but working pixels. 0 masks only zero, negative and "
        "non-finite pixels."),
}


@dataclass(frozen=True, slots=True)
class CorrectionRow:
    """One row as a page builds it: a combo when it has choices, a file row when it browses."""

    key: str
    label: str
    value: str
    choices: tuple[str, ...] = ()
    browse: bool = False
    tip: str = ""


def correction_rows(notes: Mapping[str, str] | None = None) -> tuple[CorrectionRow, ...]:
    """The group's rows in order; ``notes`` adds a page's own sentence to a row's tip (Stitch's stacked arm)."""
    notes = dict(notes or {})
    unknown = sorted(set(notes) - set(CORRECTION_KEYS))
    if unknown:
        raise KeyError(f"no Corrections row {', '.join(unknown)}")
    return tuple(
        CorrectionRow(key, LABELS[key], DEFAULTS[key], SOLID_ANGLE_CHOICES if key == "solid_angle" else (),
                      key in IMAGE_KEYS, " ".join(text for text in (TIPS[key], notes.get(key, "")) if text))
        for key in CORRECTION_KEYS)


__all__ = [
    "CORRECTION_KEYS", "CorrectionRow", "DEFAULTS", "IMAGE_KEYS", "LABELS", "SOLID_ANGLE_CHOICES", "TIPS",
    "correction_rows",
]
