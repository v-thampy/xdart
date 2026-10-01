"""Value-only editing transitions for the shared detector threshold control."""

from dataclasses import dataclass, replace
import math


@dataclass(frozen=True, slots=True)
class ThresholdState:
    apply_threshold: bool = False
    threshold_min: float | None = None
    threshold_max: float | None = None
    mask_saturation: bool = True


@dataclass(frozen=True, slots=True)
class ThresholdEditRefusal:
    reason: str


def auto_ceiling_hint(ceilings):
    """Threshold Auto's ceiling (decision 39) as the High box's grey hint, never its value, with the transitions'
    default maximum (the largest value it keeps): the one value every member shares, ``(value - 1, "Auto · value
    (source)")``; else ``(None, "Auto · per scan")`` when members differ or one is unknown, and ``(None, "Auto")``
    when none is known. ``ceilings``: each member's ``(value, "saturation" | "data type")``, or None (Stitch's and
    RSM's Threshold rows, G8c)."""
    values = {ceiling[0] for ceiling in ceilings if ceiling is not None}
    if len(ceilings) and None not in ceilings and len(values) == 1:
        value = values.pop()
        source = ("saturation" if any(ceiling[1] == "saturation" for ceiling in ceilings)
                  else "data type maximum")
        return float(value - 1), f"Auto · {value} ({source})"
    return None, "Auto · per scan" if values else "Auto"


def threshold_transition(
    state: ThresholdState, field: str, value: object, default_max: float | None,
) -> ThresholdState | ThresholdEditRefusal | None:
    """Edit ``min``, ``max`` or ``enabled``; None means an unchanged intent.

    The default is supplied by native source metadata: the largest value the
    automatic ceiling keeps (one below it), which a minimum typed under Auto
    keeps. The automatic ceiling is a hint, never a value (decision 39): under
    Auto the maximum box is empty, so clearing it keeps Auto and typing any
    number, even the hint's own, is manual (Auto leaves out values at or above
    its ceiling, a manual maximum only those above it). Disabled manual bounds
    survive a toggle. Legacy manual states may carry an implicit zero minimum;
    a touch materializes it.
    """
    if field == "enabled":
        if type(value) is not bool:
            return ThresholdEditRefusal("A boolean control requires true or false.")
        parsed = value
    elif field in {"min", "max"}:
        if type(value) is str and not value.strip():
            parsed = None
        else:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return ThresholdEditRefusal("A finite numeric value is required.")
            if type(value) is bool or not math.isfinite(parsed):
                return ThresholdEditRefusal("A finite numeric value is required.")
    else:
        return ThresholdEditRefusal("Unknown control.")

    automatic = state.mask_saturation and not state.apply_threshold
    current = (state.apply_threshold or state.mask_saturation) if field == "enabled" else (
        (0.0 if automatic else state.threshold_min) if field == "min" else
        (None if automatic else state.threshold_max))
    noncanonical_touch = state.apply_threshold and state.threshold_min is None
    if parsed == current and not noncanonical_touch:
        return None

    if field in {"min", "max"}:
        low = parsed if field == "min" else (0.0 if automatic else state.threshold_min)
        high = parsed if field == "max" else (default_max if automatic else state.threshold_max)
        if low is not None and high is not None and low > high:
            return ThresholdEditRefusal("Threshold minimum cannot exceed maximum.")
        if field == "max" and parsed is None:
            return ThresholdState()
        if high is None:
            high = default_max
            if high is None:
                return ThresholdEditRefusal("Select an integer detector input or enter an upper threshold first.")
        low = 0.0 if low is None else low
        if low > high:
            return ThresholdEditRefusal("Threshold minimum cannot exceed maximum.")
        return ThresholdState(True, low, high, False)

    candidate = replace(state, threshold_min=None, threshold_max=None) if not parsed and automatic else state
    manual = bool(parsed and candidate.threshold_max is not None)
    return replace(candidate, apply_threshold=manual, mask_saturation=bool(parsed and not manual),
                   threshold_min=0.0 if manual and candidate.threshold_min is None else candidate.threshold_min)
