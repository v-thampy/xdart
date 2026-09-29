"""Grazing incidence on the Stitch page, set as Int sets it in the Experiment section (maintainer,
2026-09-29): Int's own rows (Standard|Grazing, the incidence motor, the Manual angle, '…' for the
orientation), with Stitch's default order; Grazing's Axis adds Qip, Qoop and Qip + Qoop; the tilt motor
(chi, gonchi, else Manual) is in Advanced."""
from pathlib import Path

import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.scattering.controls_inventory import GI_MOTOR, GI_THETA, GI_TILT
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


def _axes(page):
    editor = page._rows["unit"].editor
    return [editor.itemText(index) for index in range(editor.count())]


def _choose(qapp, row, text):
    """A combo choice as the user makes it: FormRow commits on the next turn."""
    row.editor.setCurrentText(text)
    row.editor.textActivated.emit(text)
    qapp.processEvents()


def test_grazing_shows_ints_rows_with_the_scans_motors_and_eta_first(qapp, tmp_path):
    page = _page(tmp_path, "multigeometry")
    try:
        assert page.build_form().gi is None and _row(page, GI_MOTOR) is None
        assert all(row.isHidden() for row in page._gi_rows) and _axes(page) == ["Q (Å⁻¹)", "2θ (°)"]
        _click(page, "Grazing")
        motor = _row(page, GI_MOTOR).editor
        assert [motor.itemText(index) for index in range(motor.count())] == ["Manual", *_MOTORS]
        assert motor.currentText() == "eta"  # Stitch's order: eta, gonth, halpha, …
        assert _row(page, GI_THETA) is None  # a motor gives the angle
        assert any(button.isVisibleTo(page._gi_host)  # '…': Orientation
                   for button in page._gi_host.findChildren(QtWidgets.QToolButton, "controlsMoreButton"))
        assert GI_TILT not in {row._path for row in page._gi_host.findChildren(FormRow)}  # the tilt is in Advanced
        tilt, chi = page._tilt_rows
        assert [tilt.editor.itemText(index) for index in range(tilt.editor.count())] == ["Manual", *_MOTORS]
        assert tilt.editor.currentText() == "chi" and chi.isHidden()  # Stitch's tilt order: chi, gonchi, …
        # the maintainer's axes (2026-09-29): Q and 2θ stay plain; a GI axis writes its cut(s) beside I(q)
        assert _axes(page) == ["Q (Å⁻¹)", "2θ (°)", "Qip (Å⁻¹)", "Qoop (Å⁻¹)", "Qip + Qoop (Å⁻¹)"]
        assert page.build_form().gi is None and "choose Qip, Qoop or Qip + Qoop" in _status(page)
        assert all(row.isHidden() for row in page._gi_rows)
        page.set_field("unit", "Qip (Å⁻¹)")
        assert not any(row.isHidden() for row in page._gi_rows) and "choose" not in _status(page)
        # both backends read psic's circles: the incidence as eta, the tilt as chi, and mu and phi
        assert page.build_form().gi == StitchGI(("q_ip",), incidence_motor="eta", tilt_motor="chi", sample_motors=_SAMPLE)
        page.set_field("unit", "Qip + Qoop (Å⁻¹)")
        assert page.build_form().gi == StitchGI(("q_ip", "q_oop"), incidence_motor="eta", tilt_motor="chi",
                                                sample_motors=_SAMPLE)
        assert page.build_form().unit == "q_A^-1"
        _click(page, "Standard")
        assert page.build_form().gi is None and all(row.isHidden() for row in page._gi_rows)
        assert _axes(page) == ["Q (Å⁻¹)", "2θ (°)"] and page._value("unit") == "Q (Å⁻¹)"
    finally:
        _close(page, qapp)


@pytest.mark.parametrize("axis, cuts", [("Qoop (Å⁻¹)", ("q_oop",)), ("Qip + Qoop (Å⁻¹)", ("q_ip", "q_oop"))])
def test_a_manual_angle_axis_and_ranges_reach_the_form_and_round_trip(qapp, tmp_path, axis, cuts):
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
        page.set_field("unit", axis)
        page.set_field("gi_ip_auto", False)
        page.set_field("gi_ip_min", -1.)
        page.set_field("gi_ip_max", 1.5)
        tilt, chi = page._tilt_rows  # Advanced: a Manual tilt takes one χ for every frame
        _choose(qapp, tilt, "Manual")
        assert not chi.isHidden()
        chi.editor.setText("89.5")
        chi.editor.editingFinished.emit()
        expected = StitchGI(cuts, incident_angle_deg=.3, tilt_chi_deg=89.5, sample_motors=_SAMPLE, ip_range=(-1., 1.5))
        assert page.build_form().gi == expected
        (tmp_path / "other").mkdir()
        other = StitchPage(empty_host_services(_Status()))
        other.load_form(_gi_form(tmp_path / "other", "multigeometry", None))
        apply_page_settings(other, page_settings(page))
        assert other.build_form().gi == expected
        other.load_form(page.build_form())  # a prepared form restores it too
        assert other.build_form().gi == expected and _row(other, GI_THETA).editor.text() == "0.3"
        assert other._value("unit") == axis and other._tilt_rows[1].editor.text() == "89.5"
    finally:
        _close(page, qapp)
        if other is not None:
            _close(other, qapp)


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
@pytest.mark.parametrize("recorded", (True, False))
def test_both_backends_read_the_geometrys_sample_circles_or_say_one_reads_zero(qapp, tmp_path, recorded, backend):
    page = _page(tmp_path, backend)
    try:
        if not recorded:  # a scan whose header has no phi
            spec = Path(page._source_paths[0])
            spec.write_text(spec.read_text().replace("  phi", "").replace(" -1.7", ""))
            page.set_sources((str(spec),))
        _click(page, "Grazing")
        page.set_field("unit", "Qip (Å⁻¹)")
        gi = page.build_form().gi
        expected = _SAMPLE if recorded else (("mu", "mu"), ("phi", None))
        assert gi.incidence_motor == "eta" and gi.tilt_motor == "chi" and gi.sample_motors == expected
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
    # both files name the same images, so Int's automatic names would collide: name the outputs
    page.set_field("output_family", "first; second")
    return page


def _status(page):
    return " ".join(label.text() for label in page._gi_host.findChildren(QtWidgets.QLabel)
                    if label.isVisibleTo(page._gi_host))


@pytest.mark.parametrize("dropped", [(), (("phi", "-1.7"),), "both"])
def test_a_sample_circle_is_decided_by_every_selected_scan(qapp, tmp_path, dropped):
    """Codex review F3: the first scan's motors never decide for the others. Recorded by every scan:
    bound (and each scan's own value frozen); by none: read as 0, said; by only some: refused."""
    from xdart.gui.tools.stitch_values import prepare_stitch_tool
    page = _two_files(tmp_path, "xu_hist", (("phi", "-1.7"),) if dropped == "both" else dropped)
    try:
        if dropped == "both":  # neither file records phi
            other = Path(page._source_paths[1])
            other.write_text(other.read_text().replace("  phi", "").replace(" -1.7", ""))
            page.set_field("image_stem", "sample_scan{scan}_")
        _click(page, "Grazing")
        page.set_field("unit", "Qip (Å⁻¹)")
        if dropped == (("phi", "-1.7"),):
            assert "phi recorded by only some scans" in _status(page)
            with pytest.raises(ValueError, match="phi is recorded by some selected scans and not others"):
                page.build_form()
            return
        gi = page.build_form().gi
        if dropped == "both":
            assert dict(gi.sample_motors)["phi"] is None and "phi read as 0" in _status(page)
            return
        assert gi.sample_motors == _SAMPLE
        prepared = prepare_stitch_tool(page.build_form())
        bound = [{name: value for name, _occurrence, value in item.values}
                 for request in prepared.requests for item in request.manifest.contributions]
        assert len(bound) == 4 and all(item["chi"] == 90. and item["phi"] == -1.7 for item in bound)
    finally:
        _close(page, qapp)


def test_the_gi_axes_keep_the_typed_q_range_and_only_2theta_resets_it(qapp, tmp_path):
    """Every GI axis bins I(q) too, so moving among Q, Qip, Qoop and Qip + Qoop keeps a typed Q range;
    2θ changes the unit and resets it, as before."""
    page = _page(tmp_path, "multigeometry")
    try:
        _click(page, "Grazing")
        page.set_field("radial_auto", False)
        page.set_field("q_min", 1.)
        page.set_field("q_max", 3.)
        for axis in ("Qip (Å⁻¹)", "Qip + Qoop (Å⁻¹)", "Qoop (Å⁻¹)", "Q (Å⁻¹)", "Qip (Å⁻¹)"):
            page.set_field("unit", axis)
            assert page.build_form().q_range == (1., 3.)
        _click(page, "Standard")  # the GI axis becomes Q: the same unit, the same range
        assert page._value("unit") == "Q (Å⁻¹)" and page.build_form().q_range == (1., 3.)
        page.set_field("unit", "2θ (°)")
        assert page.build_form().q_range == (0., 90.)
    finally:
        _close(page, qapp)
