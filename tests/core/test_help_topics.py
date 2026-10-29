"""Frozen H1/H2/H7: public offline catalogue and literal diagnostic presentation."""
import subprocess
import sys

import pytest

IDS = (
    "calibration.multi-step", "calibration.frame-choice", "geometry.orientation",
    "calibration.diagnostics", "corrections.existing", "analysis.plot-roi",
    # PHASE-ANALYSIS-128 (2026-10-29): one topic per section of the Phase Analysis window, appended.
    "phase-analysis.phases", "phase-analysis.peak-shape", "phase-analysis.background",
    "phase-analysis.constraints", "phase-analysis.batch",
)
CODES = (
    "ROI_RAW_UNAVAILABLE", "ROI_RAW_SHAPE_INVALID", "ROI_PREVIEW_LIMIT_EXCEEDED",
    "ROI_IDENTITY_MISMATCH", "ROI_SIGNAL_COUNT_INVALID", "ROI_RAW_UNAVAILABLE_ALL",
    "ROI_INVALID_MASK_IGNORED", "ROI_RESULT_LIMIT_EXCEEDED",
)


def test_six_immutable_topics_cover_the_frozen_scientific_questions():
    """The six frozen topics, then PHASE-ANALYSIS-128's five, each in the same ten-heading shape."""
    from xrd_tools.help_topics import TOPICS, get_topic
    from dataclasses import FrozenInstanceError
    assert tuple(t.identifier for t in TOPICS) == IDS
    concepts = (
        ("Stitch", "RSM", "global", "refine", "overwrite"),
        ("three to five", "coverage", "completed", "identifiability"),
        ("row", "column", "(x, y)", "corner", "once", "azimuth"),
        ("RMS", "Å", "condition", "correlation", "residual"),
        ("dark", "flat", "solid angle", "polarization", "parallax", "normalization"),
        ("mean", "sum", "max", "min", "std", "subtract", "divide", "stale", "Log"),
        ("include", "calibrant", "library", "minimum", "legend", "colour", "fitted"),
        ("profile", "Caglioti", "Scherrer", "March", "free", "σ"),
        ("SNIP", "Chebyshev", "substrate", "template", "amorphous", "flat"),
        ("lattice", "q-shift", "cross-phase", "fit range", "iterations", "uncertainties"),
        ("Batch", "Sequential", "seed", "trend", "settings", "notebook"),
    )
    for topic, words in zip(TOPICS, concepts, strict=True):
        assert get_topic(topic.identifier) is topic
        for heading in ("Purpose", "Required inputs", "Steps", "Units and conventions",
                        "Assumptions", "Expected observations", "Diagnostics",
                        "Common mistakes", "Worked example", "References"):
            assert heading in topic.text
        for word in words:
            assert word.casefold() in topic.text.casefold(), (topic.identifier, word)
        assert "/Users/" not in topic.text and "H26" not in topic.text
        with pytest.raises(FrozenInstanceError):
            topic.title = "changed"
    assert get_topic("unpublished.β") is None


def test_search_is_ordered_casefolded_and_matches_every_term_in_id_title_or_body():
    from xrd_tools.help_topics import TOPICS, search_topics
    assert search_topics("  \t") == TOPICS
    assert [t.identifier for t in search_topics("GEOMETRY.ORIENTATION")] == [IDS[2]]
    assert IDS[4] in [t.identifier for t in search_topics("POLARIZATION flat")]
    assert search_topics("unfindable_calibrant") == ()
    assert search_topics("polarization unfindable_calibrant") == ()
    for query in ("geometry", "ROI", "calibration", "frame"):
        selected = search_topics(query)
        assert selected == tuple(t for t in TOPICS if query.casefold() in
                                 f"{t.identifier} {t.title} {t.text}".casefold())


def test_import_is_headless_and_import_light():
    result = subprocess.run([sys.executable, "-c",
        "import sys; import xrd_tools.help_topics; "
        "assert not any(k.split('.')[0] in "
        "{'PySide6','PyQt6','pyqtgraph','numpy','pyFAI','h5py','matplotlib'} for k in sys.modules)"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("code", CODES)
def test_each_existing_diagnostic_has_additive_guidance(code):
    from xrd_tools.help_topics import diagnostic_text
    text = diagnostic_text(code)
    assert text.startswith(code)
    assert "Next action:" in text and len(text) > len(code) + 80
    if "LIMIT" in code:
        assert "64 MiB" in text
    if "PREVIEW_LIMIT" in code:
        assert "rectangle" in text and "whole" in text


def test_literal_codes_and_details_precede_guidance_without_reclassification():
    from xrd_tools.help_topics import diagnostic_text
    details = ("ROI_INVALID_MASK_IGNORED", "unknown_β\nline two", "ROI_RAW_UNAVAILABLE_ALL")
    text = diagnostic_text("OK", details)
    assert text.startswith("OK\nROI_INVALID_MASK_IGNORED\nunknown_β\nline two\nROI_RAW_UNAVAILABLE_ALL")
    assert text.index("unknown_β") < text.index("Next action:")
    assert diagnostic_text("UNKNOWN\nβ", ("raw detail",)) == "UNKNOWN\nβ\nraw detail"
