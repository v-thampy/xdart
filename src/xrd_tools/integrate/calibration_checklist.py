"""Read-only calibration progress from facts supplied by the existing owners."""
from dataclasses import dataclass

from xrd_tools.help_topics import CALIBRATION_STEPS


@dataclass(frozen=True)
class ChecklistFacts:
    instrument_ready: bool = False
    instrument_message: str = "Describe the instrument, detector and camera."
    rough_mode: str = "scan"
    rough_ready: bool = False
    rough_stale: bool = False
    rough_message: str = "Measure or enter a rough geometry, or explicitly skip it."
    source_ready: bool = False
    source_message: str = "Select the calibrant scan and load its metadata."
    frame_count: int = 0
    complete_count: int = 0
    frame_message: str = "Select frames and calibrate their rings."
    fit_present: bool = False
    fit_outdated: bool = False
    fit_message: str = "Fit the calibrated frames or refine the scan."
    #: the full fit summary, shown as the tooltip of the one-line fit and inspect cells (empty: the message)
    fit_detail: str = ""
    saved_current_fit: bool = False
    orientation_message: str = "Optional: inspect orientation evidence; no camera or sense is applied."
    #: the Orientation check's one readiness fact (LT-008): whether it can run with the window's current inputs, or
    #: (message) why not and (target) the step whose controls fix it; waiting while another operation runs
    orientation_ready: bool = True
    orientation_waiting: bool = False
    orientation_target: str = "orientation"


@dataclass(frozen=True)
class ChecklistRow:
    step: str
    text: str
    state: str
    message: str
    target: str
    #: the row's full detail, its cell tooltip (empty: the message)
    detail: str = ""


_ORIENTATION_STEP = ("Orientation check", "Orientation check: inspect the consistent camera and motor-sense hypotheses.")


def calibration_checklist(facts: ChecklistFacts) -> tuple[ChecklistRow, ...]:
    """Project readiness and provenance, without judging scientific fit quality."""
    rough = ("Not applicable" if facts.rough_mode == "skip" else "Out of date" if facts.rough_stale
             else "Done" if facts.rough_ready else "Needs input")
    fit = "Out of date" if facts.fit_outdated else "Done" if facts.fit_present else "Needs input"
    states = (
        "Done" if facts.instrument_ready else "Needs input", rough,
        "Done" if facts.source_ready else "Needs input",
        "Done" if facts.frame_count >= 3 and facts.complete_count == facts.frame_count else "Needs input",
        fit, "Optional" if facts.fit_present else "Needs input",
        "Done" if facts.fit_present and facts.saved_current_fit else "Needs input",
        "Waiting" if facts.orientation_waiting else "Optional" if facts.orientation_ready else "Needs input",
    )
    frames = facts.frame_message
    if 0 < facts.frame_count < 3:
        frames += "; the fit needs at least three Complete frames"
    messages = (facts.instrument_message, facts.rough_message, facts.source_message, frames,
                facts.fit_message, facts.fit_message,
                "Current ring fit saved; validate it on an informative frame." if states[6] == "Done"
                else "Save the current ring fit, then validate it on an informative frame.",
                facts.orientation_message)
    details = ("", "", "", "", facts.fit_detail, facts.fit_detail, "", "")
    targets = ("instrument", "rough", "select", "calibrate", "fit", "inspect", "save", facts.orientation_target)
    return tuple(ChecklistRow(label, text, state, message, target, detail)
                 for (label, text), state, message, target, detail
                 in zip((*CALIBRATION_STEPS, _ORIENTATION_STEP), states, messages, targets, details, strict=True))
