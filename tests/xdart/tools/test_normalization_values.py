"""The one Normalization row group (AUD-9 stage 2b, decision 4), Qt-free: which rows each choice reads, the
selection a page's plan takes, and the projected fields Int's composer lays out."""
import pytest

from xdart.gui.tools import normalization_values as normalization
from xrd_tools.session.readiness import ControlFieldKind, ControlFormField, SectionId


def test_each_choice_reads_its_rows_and_refuses_what_it_still_needs():
    assert normalization.NORMALIZATION_CHOICES == (
        "None", "Monitor", "Foil transmission × exposure", "Monitor × foil transmission")
    reads = {choice: {key for key, used in normalization.rows_in_use(choice).items() if used}
             for choice in normalization.NORMALIZATION_CHOICES}
    assert reads == {
        "None": {"normalization"}, "Monitor": {"normalization", "monitor"},
        "Foil transmission × exposure": {"normalization", "foil_column", "exposure_column", "foil_lengths"},
        "Monitor × foil transmission": {"normalization", "monitor", "foil_column", "foil_lengths"}}
    select = normalization.normalization_selection
    lengths = "1.06, 3.04, 4.65, 9.5"
    assert select("None", "I0", "foil status", "Seconds", lengths) == normalization.NormalizationSelection("None")
    assert select("Monitor", "I0", "foil status", "Seconds", "x") == normalization.NormalizationSelection(
        "Monitor", monitor="I0")  # a row the choice does not read is not checked
    assert select("Foil transmission × exposure", "I0", "foil status", "Seconds", lengths) == \
        normalization.NormalizationSelection("Foil transmission × exposure", None, "foil status", "Seconds",
                                             (1.06, 3.04, 4.65, 9.5))
    assert select("Monitor × foil transmission", "I0", "foil status", "Seconds", "1 2 3 4") == \
        normalization.NormalizationSelection("Monitor × foil transmission", "I0", "foil status", None,
                                             (1.0, 2.0, 3.0, 4.0))
    for args, need in ((("Monitor", "None"), "a monitor column"),
                       (("Monitor × foil transmission", "I0", "None", "", lengths), "a foil column"),
                       (("Foil transmission × exposure", "", "foil status", "None", lengths), "an exposure column"),
                       (("Foil transmission × exposure", "", "Seconds", "Seconds", lengths),
                        "distinct foil and exposure columns"),
                       (("Monitor × foil transmission", "I0", "foil status", "", "1, 2, 3"),
                        "Foil absorption: four numbers are needed, first foil first"),
                       (("Monitor × foil transmission", "I0", "foil status", "", "0 0 0 0"),
                        "Foil absorption: each must be a number ≥ 0, and not all zero"),
                       (("Sum",), "Normalization must be one of None, Monitor, Foil transmission × exposure, "
                                  "Monitor × foil transmission")):
        with pytest.raises(ValueError, match="^" + need.replace("(", r"\(").replace(")", r"\)") + "$"):
            select(*args)
    assert normalization.choice_from("I0") == "Monitor" and normalization.choice_from("None") == "None"
    assert normalization.choice_from("None", "foil status", "Seconds") == "Foil transmission × exposure"
    assert normalization.choice_from("I0", "foil status") == "Monitor × foil transmission"


def test_the_projected_fields_grey_what_the_choice_does_not_read_and_list_the_scans_columns():
    fields = {key: ControlFormField(SectionId.PROCESSING, key, ("Page", key), "",
                                    ControlFieldKind.COMBO, ("None",)) for key in normalization.NORMALIZATION_KEYS}
    values = {"normalization": "Foil transmission × exposure", "monitor": "None", "foil_column": "foil status",
              "exposure_column": "seconds_old", "foil_lengths": "1.06, 3.04, 4.65, 9.5"}
    projected = normalization.normalization_fields(fields, values, ("I0", "foil status", "Seconds", " "))
    assert projected["normalization"].choices == normalization.NORMALIZATION_CHOICES
    assert projected["foil_column"].choices == ("None", "I0", "foil status", "Seconds")
    # a column the scan does not record (restored from settings) stays a choice
    assert projected["exposure_column"].choices == ("None", "I0", "foil status", "Seconds", "seconds_old")
    assert projected["foil_lengths"].kind is ControlFieldKind.LINE
    assert [key for key, field in projected.items() if not field.enabled] == ["monitor"]
    assert projected["monitor"].reason == "Monitor is not read by Foil transmission × exposure."
    assert {key: field.label for key, field in projected.items()} == normalization.LABELS
