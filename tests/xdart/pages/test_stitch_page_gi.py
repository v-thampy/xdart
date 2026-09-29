"""Grazing incidence on the Stitch page, set as Int sets it in the Experiment section (maintainer,
2026-09-29): Int's own rows (Standard|Grazing, the incidence motor, the Manual angle, '…' for the
orientation and tilt), with Stitch's default order, and the GI cut beside the 1-D axis."""
from pathlib import Path

import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.scattering.controls_inventory import GI_MOTOR, GI_THETA
from xdart.gui.tabs.stitch.page import StitchPage
from xdart.gui.tabs.stitch.settings import apply_page_settings, page_settings
from xdart.gui.widgets.controls_panel import FormRow
from xrd_tools.integrate.stitch_gi import StitchGI
from tests.xdart.pages.test_stitch_page import _Status, qapp  # noqa: F401  (the Qt application)
from tests.xdart.tools.test_stitch_native_gi import _SAMPLE, _gi_form

#: The fixture scan's real motors (#O), in order.
_MOTORS = ("del", "nu", "eta", "mu", "chi", "phi")


def _page(tmp_path, backend, gi=None):
    page = StitchPage(empty_host_services(_Status()))
    page.load_form(_gi_form(tmp_path, backend, gi))
    return page


def _row(page, path):
    rows = [row for row in page._gi_host.findChildren(FormRow) if row._path == path and row.isVisibleTo(page._gi_host)]
    return rows[0] if rows else None


def _click(page, label):
    button = next(button for button in page._gi_host.findChildren(QtWidgets.QPushButton)
                  if button.text() == label and button.isVisibleTo(page._gi_host))
    button.click()


def _close(page, qapp):
    assert page.shutdown().status is PageCleanup.CLEAN
    page.deleteLater()
    qapp.processEvents()


def test_grazing_shows_ints_rows_with_the_scans_motors_and_eta_first(qapp, tmp_path):
    page = _page(tmp_path, "multigeometry")
    try:
        assert page.build_form().gi is None and _row(page, GI_MOTOR) is None
        assert all(row.isHidden() for row in page._gi_rows)
        _click(page, "Grazing")
        motor = _row(page, GI_MOTOR).editor
        assert [motor.itemText(index) for index in range(motor.count())] == ["Manual", *_MOTORS]
        assert motor.currentText() == "eta"  # Stitch's order: eta, gonth, halpha, …
        assert _row(page, GI_THETA) is None  # a motor gives the angle
        assert any(button.isVisibleTo(page._gi_host)  # '…': Orientation, Tilt
                   for button in page._gi_host.findChildren(QtWidgets.QToolButton, "controlsMoreButton"))
        assert not any(row.isHidden() for row in page._gi_rows)
        assert page.build_form().gi == StitchGI("q_ip", incidence_motor="eta")  # pyFAI reads no sample circles
        _click(page, "Standard")
        assert page.build_form().gi is None and all(row.isHidden() for row in page._gi_rows)
    finally:
        _close(page, qapp)


def test_a_manual_angle_cut_and_ranges_reach_the_form_and_round_trip(qapp, tmp_path):
    page = _page(tmp_path, "multigeometry")
    other = None
    try:
        _click(page, "Grazing")
        motor = _row(page, GI_MOTOR).editor
        motor.setCurrentText("Manual")
        motor.textActivated.emit("Manual")  # the user's choice; the row commits it next turn
        qapp.processEvents()
        angle = _row(page, GI_THETA).editor
        angle.setText("0.3")
        angle.editingFinished.emit()
        page.set_field("gi_cut", "Qₒₒₚ")
        page.set_field("gi_ip_auto", False)
        page.set_field("gi_ip_min", -1.)
        page.set_field("gi_ip_max", 1.5)
        expected = StitchGI("q_oop", incident_angle_deg=.3, window=(-1., 1.5))
        assert page.build_form().gi == expected
        (tmp_path / "other").mkdir()
        other = StitchPage(empty_host_services(_Status()))
        other.load_form(_gi_form(tmp_path / "other", "multigeometry", None))
        apply_page_settings(other, page_settings(page))
        assert other.build_form().gi == expected
        other.load_form(page.build_form())  # a prepared form restores it too
        assert other.build_form().gi == expected and _row(other, GI_THETA).editor.text() == "0.3"
    finally:
        _close(page, qapp)
        if other is not None:
            _close(other, qapp)


@pytest.mark.parametrize("recorded", (True, False))
def test_xrayutilities_reads_the_geometrys_sample_circles_or_says_one_reads_zero(qapp, tmp_path, recorded):
    page = _page(tmp_path, "xu_hist")
    try:
        if not recorded:  # a scan whose header has no phi
            spec = Path(page._source_paths[0])
            spec.write_text(spec.read_text().replace("  phi", "").replace(" -1.7", ""))
            page.set_sources((str(spec),))
        _click(page, "Grazing")
        gi = page.build_form().gi
        expected = _SAMPLE if recorded else (("mu", "mu"), ("chi", "chi"), ("phi", None))
        assert gi.incidence_motor == "eta" and gi.sample_motors == expected
        assert ("phi read as 0" in _status(page)) is not recorded
    finally:
        _close(page, qapp)


def _two_files(tmp_path, backend, drop=()):
    """The GI fixture's SPEC file and a second one; ``drop`` removes (name, value) motors from the
    first file's header."""
    page = _page(tmp_path, backend)
    path = Path(page._source_paths[0])
    other = path.with_name("other")
    other.write_text(path.read_text())
    text = path.read_text()
    for name, value in drop:
        text = text.replace("  " + name, "").replace(" " + value, "")
    path.write_text(text)
    page.set_sources((str(path), str(other)))
    page.set_field("image_stem", "sample_scan{scan}_")
    return page


def _status(page):
    return " ".join(label.text() for label in page._gi_host.findChildren(QtWidgets.QLabel)
                    if label.isVisibleTo(page._gi_host))


@pytest.mark.parametrize("dropped", [(), (("chi", "90"),), "both"])
def test_a_sample_circle_is_decided_by_every_selected_scan(qapp, tmp_path, dropped):
    """Codex review F3: the first scan's motors never decide for the others. Recorded by every scan:
    bound (and each scan's own value frozen); by none: read as 0, said; by only some: refused."""
    from xdart.gui.tools.stitch_values import prepare_stitch_tool
    page = _two_files(tmp_path, "xu_hist", (("chi", "90"),) if dropped == "both" else dropped)
    try:
        if dropped == "both":  # neither file records chi
            other = Path(page._source_paths[1])
            other.write_text(other.read_text().replace("  chi", "").replace(" 90", ""))
            page.set_field("image_stem", "sample_scan{scan}_")
        _click(page, "Grazing")
        if dropped == (("chi", "90"),):
            assert "chi recorded by only some scans" in _status(page)
            with pytest.raises(ValueError, match="chi is recorded by some selected scans and not others"):
                page.build_form()
            return
        gi = page.build_form().gi
        if dropped == "both":
            assert dict(gi.sample_motors)["chi"] is None and "chi read as 0" in _status(page)
            return
        assert gi.sample_motors == _SAMPLE
        prepared = prepare_stitch_tool(page.build_form())
        bound = [{name: value for name, _occurrence, value in item.values}
                 for request in prepared.requests for item in request.manifest.contributions]
        assert len(bound) == 4 and all(item["chi"] == 90. for item in bound)
    finally:
        _close(page, qapp)


def test_the_status_follows_a_backend_switch(qapp, tmp_path):
    """Codex review P3: with Grazing on under pyFAI, switching to xrayutilities (which reads the
    sample circles) rebuilds the status."""
    page = _page(tmp_path, "multigeometry")
    try:
        path = Path(page._source_paths[0])
        path.write_text(path.read_text().replace("  phi", "").replace(" -1.7", ""))
        page.set_sources((str(path),))
        _click(page, "Grazing")
        assert "read as 0" not in _status(page)
        page.set_field("backend", "xrayutilities histogram")
        assert "phi read as 0" in _status(page)
    finally:
        _close(page, qapp)
