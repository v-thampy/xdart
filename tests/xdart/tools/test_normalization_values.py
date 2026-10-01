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


def test_the_groups_columns_are_the_scans_counters_from_its_header_cached_by_the_files_state(tmp_path):
    """The group's columns for Stitch and RSM alike (G8b moved them from Stitch's page): a scan's ``#L``
    counters but its ``#O`` motors, from the header alone; the file's (mtime, size) keys the cache, so an edited
    file is read again; a file that cannot be read gives None, and Monitor's choices are then "None" alone."""
    spec = tmp_path / "scans.spec"
    header = "#F scans.spec\n#O0 eta  nu\n\n#S 1  a2scan eta 1 2 nu 3 4  1 1\n#P0 1 3\n"
    spec.write_text(header + "#L eta  nu  I0  Seconds\n1 3 100 1\n")
    state = spec.stat().st_mtime_ns, spec.stat().st_size
    assert normalization.spec_counter_names(str(spec), state, None) == ("I0", "Seconds")
    assert normalization.spec_counter_names(str(spec), state, "1.1") == ("I0", "Seconds")
    spec.write_text(header + "#L eta  nu  mon  Seconds  det\n1 3 100 1 5\n")  # edited: a new state, read again
    edited = spec.stat().st_mtime_ns, spec.stat().st_size
    assert normalization.spec_counter_names(str(spec), edited, None) == ("mon", "Seconds", "det")
    assert normalization.spec_counter_names(str(spec), state, None) == ("I0", "Seconds")  # the old key's
    missing = tmp_path / "absent.spec"
    assert normalization.spec_counter_names(str(missing), None, None) is None
    assert normalization.spec_counter_names(str(spec), edited, "7.1") is None  # a scan the file lacks
    assert normalization.monitor_choices(()) == ("None",)
    assert normalization.monitor_choices(("Seconds", " ", "mon", "H")) == ("None", "mon", "Seconds")
