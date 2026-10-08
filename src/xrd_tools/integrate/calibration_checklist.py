"""Read-only calibration progress from facts supplied by the existing owners."""
from dataclasses import dataclass
import re

from xrd_tools.help_topics import get_topic


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
    saved_current_fit: bool = False
    orientation_message: str = "Optional: inspect orientation evidence; no camera or sense is applied."


@dataclass(frozen=True)
class ChecklistRow:
    step: str
    text: str
    state: str
    message: str
    target: str


# The teaching topic is the single text authority for the seven numbered steps.
_STEPS = tuple(" ".join(text.split()) for text in re.findall(
    r"^\d\. (.*?)(?=^\d\. |\n\nUnits)", get_topic("calibration.multi-step").text, re.M | re.S))


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
        "Done" if facts.fit_present and facts.saved_current_fit else "Needs input", "Optional",
    )
    messages = (facts.instrument_message, facts.rough_message, facts.source_message, facts.frame_message,
                facts.fit_message, facts.fit_message,
                "Current ring fit saved; validate it on an informative frame." if states[6] == "Done"
                else "Save the current ring fit, then validate it on an informative frame.",
                facts.orientation_message)
    texts = (*_STEPS, "Orientation check: inspect the consistent camera and motor-sense hypotheses.")
    targets = ("instrument", "rough", "select", "calibrate", "fit", "inspect", "save", "orientation")
    return tuple(ChecklistRow(re.split(r"[:.]", text, maxsplit=1)[0], text, state, message, target)
                 for text, state, message, target in zip(texts, states, messages, targets, strict=True))
