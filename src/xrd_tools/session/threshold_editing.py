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


def threshold_transition(
    state: ThresholdState, field: str, value: object, default_max: float | None,
) -> ThresholdState | ThresholdEditRefusal | None:
    """Edit ``min``, ``max`` or ``enabled``; None means an unchanged intent.

    The default is supplied by native source metadata. Automatic display values
    are not stored bounds, while disabled manual bounds survive a toggle. Legacy
    manual states may carry an implicit zero minimum; a touch materializes it.
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
        (default_max if automatic else state.threshold_max))
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
