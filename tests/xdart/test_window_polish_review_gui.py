"""Review discriminators for WINDOW-POLISH R3, frozen before their corrections."""

import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering.test_fit_tools_preservation import _peaks, _trace


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_lattice_only_phase_trend_groups_cells_without_fractions(qapp):
    """The phase analyzer withholds fractions for lattice-only phases but still reports their cells."""
    from xdart.gui.analysis.phase_fit_dialog import PhaseFitDialog
    dialog = PhaseFitDialog(vnext_submit=lambda *_args: None)
    try:
        trend = dialog.trend
        for frame in (1, 2):
            trend.set_row(str(frame), frame, {"Film_A_a": 4.2 + frame / 100, "Film_B_a": 5.1,
                                              "Film_A_alpha": 90., "Film_B_alpha": 91.})
        assert {trend.family.itemData(i) for i in range(trend.family.count())} == {"a", "alpha"}
        assert trend.members.text() == "Phases: all"
        actions = {action.text(): action for action in trend.members.menu().actions() if action.text()}
        assert set(actions) == {"All", "Film_A", "Film_B"}
        trend.family.setCurrentIndex(trend.family.findData("a"))
        trend.right_family.setCurrentIndex(trend.right_family.findData("alpha"))
        actions["Film_A"].trigger()
        assert [item.name() for item in trend.plot.listDataItems()] == ["Film_B_a"]
        assert [item.name() for item in trend.right_view_box.addedItems] == ["Film_B_alpha"]
        assert trend.plot.getPlotItem().getAxis("left").labelText == "Lattice a (Å)"
        assert trend.right_axis.labelText == "Lattice α (°)"
    finally:
        dialog.close()


def test_each_template_file_gets_its_message_even_when_the_text_matches(qapp, tmp_path):
    """Two different selected files can return the same scientific refusal, even sharing a basename."""
    from xdart.gui.analysis.background_template_row import BackgroundTemplateRow
    from xrd_tools.analysis.display_fit_operations import DisplayedPeakFitPlan, run_displayed_peak_fit
    row = BackgroundTemplateRow()
    trace, _anchor = _trace(*_peaks())
    boxes = []
    try:
        for folder in ("first", "second"):
            target = tmp_path / folder / "empty.xye"
            target.parent.mkdir()
            target.write_text("")
            row._file_chooser = lambda target=target: str(target)
            row._ask_unit = lambda _target, _hint: "q_A^-1"
            row.choose_btn.click()
            result = run_displayed_peak_fit(DisplayedPeakFitPlan(
                trace, background="template", background_template_source=row.source.to_pairs()))
            assert row.refuse(result)
            assert row.message_box is not None and row.message_box.isVisible()
            box = row.message_box
            assert row.refuse(result) and row.message_box is box, "the same refusal does not repeat"
            assert all(box is not prior for prior in boxes), "a different file gets its own message"
            boxes.append(box)
    finally:
        row.close()
