"""Grazing incidence on the Stitch page, set as Int sets it in the Experiment section (maintainer,
2026-09-29): Int's own rows (Standard|Grazing, the incidence motor, the Manual angle, '…' for the
orientation), with Stitch's default order; Grazing's Axis adds Qip and Qoop, each writing I(q) and both
cuts; the tilt motor (chi, gonchi, else Manual) is in Advanced."""
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
        # the maintainer's axes (2026-09-29): Q and 2θ stay plain; a GI axis writes I(q) and both cuts
        # ("it will be good to save both by default along with I(q)"), opening on the one chosen
        assert _axes(page) == ["Q (Å⁻¹)", "2θ (°)", "Qip (Å⁻¹)", "Qoop (Å⁻¹)"]
        # Int's fixed statuses (a longer one widened the panel); the hint is on the eliding readiness line
        assert page.build_form().gi is None and page._gi_card.status.text() == "grazing"
        assert page.controls.experiment_card.status.text() == "grazing"
        assert "choose Qip or Qoop" in page.run_bar.readinessLabel.full_text()
        assert all(row.isHidden() for row in page._gi_rows)
        page.set_field("unit", "Qip (Å⁻¹)")
        assert not any(row.isHidden() for row in page._gi_rows)
        assert "choose" not in page.run_bar.readinessLabel.full_text() + _status(page)
        # both backends read psic's circles: the incidence as eta, the tilt as chi, and mu and phi
        both = StitchGI(("q_ip", "q_oop"), incidence_motor="eta", tilt_motor="chi", sample_motors=_SAMPLE)
        assert page.build_form().gi == both
        page.set_field("unit", "Qoop (Å⁻¹)")
        assert page.build_form().gi == both
        assert page.build_form().unit == "q_A^-1"
        _click(page, "Standard")
        assert page.build_form().gi is None and all(row.isHidden() for row in page._gi_rows)
        assert _axes(page) == ["Q (Å⁻¹)", "2θ (°)"] and page._value("unit") == "Q (Å⁻¹)"
        assert page._gi_card.status.text() == page.controls.experiment_card.status.text() == "standard"
    finally:
        _close(page, qapp)


def test_grazing_keeps_the_controls_width_as_ints(qapp, tmp_path):
    """Maintainer, 2026-09-30: "when grazing is toggled, the horizontal size of the right control panel changes
    and expands ... this was also fixed in int 2d". Measured: the long status ("grazing · choose Qip or Qoop as
    the Axis") raised the panel's minimum width from 288 to 434 px; Int's fixed "grazing" keeps it."""
    page = _page(tmp_path, "multigeometry")
    try:
        page.show()
        qapp.processEvents()
        standard = page.controls.minimumSizeHint().width()
        _click(page, "Grazing")
        qapp.processEvents()
        assert page.controls.minimumSizeHint().width() == standard
        assert page.controls.experiment_title == "Configuration"  # Int's title (workspace_shell)
    finally:
        page.close()
        _close(page, qapp)


@pytest.mark.parametrize("axis", ["Qoop (Å⁻¹)", "Qip (Å⁻¹)"])
def test_a_manual_angle_axis_and_ranges_reach_the_form_and_round_trip(qapp, tmp_path, axis):
    cuts = ("q_ip", "q_oop")  # either GI axis writes both cuts
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
    """The page binds the Geometry's circles; Preview (on the worker) reads a circle no scan records
    as 0 and says so (Codex audit G-F2: the page reads no scan's table)."""
    from xdart.gui.tools.stitch_values import prepare_stitch_tool
    page = _page(tmp_path, backend)
    try:
        if not recorded:  # a scan whose header has no phi
            spec = Path(page._source_paths[0])
            spec.write_text(spec.read_text().replace("  phi", "").replace(" -1.7", ""))
            page.set_sources((str(spec),))
        _click(page, "Grazing")
        page.set_field("unit", "Qip (Å⁻¹)")
        gi = page.build_form().gi
        assert gi.incidence_motor == "eta" and gi.tilt_motor == "chi" and gi.sample_motors == _SAMPLE
        prepared = prepare_stitch_tool(page.build_form())
        expected = _SAMPLE if recorded else (("mu", "mu"), ("phi", None))
        assert all(request.plan.gi.sample_motors == expected for request in prepared.requests)
        said = [item for item in prepared.warnings if "read as 0" in item]
        assert said == ([] if recorded else ["Grazing: no selected scan records phi; read as 0"])
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
    bound (and each scan's own value frozen); by none: read as 0, said; by only some: refused, naming
    a scan without it. Preview decides, not the page (Codex audit G-F2)."""
    from xdart.gui.tools.stitch_values import StitchToolPreflightRefused, prepare_stitch_tool
    page = _two_files(tmp_path, "xu_hist", (("phi", "-1.7"),) if dropped == "both" else dropped)
    try:
        if dropped == "both":  # neither file records phi
            other = Path(page._source_paths[1])
            other.write_text(other.read_text().replace("  phi", "").replace(" -1.7", ""))
            page.set_field("image_stem", "sample_scan{scan}_")
        _click(page, "Grazing")
        page.set_field("unit", "Qip (Å⁻¹)")
        assert page.build_form().gi.sample_motors == _SAMPLE  # the page binds the Geometry's names
        if dropped == (("phi", "-1.7"),):
            with pytest.raises(StitchToolPreflightRefused) as refused:
                prepare_stitch_tool(page.build_form())
            assert refused.value.code == "GI_SAMPLE_CIRCLE_MIXED"
            assert f"phi (phi) is recorded by some selected scans and not others, not by {page._source_paths[0]}" in str(refused.value)
            return
        prepared = prepare_stitch_tool(page.build_form())
        if dropped == "both":
            assert all(dict(request.plan.gi.sample_motors)["phi"] is None for request in prepared.requests)
            assert "Grazing: no selected scan records phi; read as 0" in prepared.warnings
            return
        assert all(request.plan.gi.sample_motors == _SAMPLE for request in prepared.requests)
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
        for axis in ("Qip (Å⁻¹)", "Qoop (Å⁻¹)", "Q (Å⁻¹)", "Qip (Å⁻¹)"):
            page.set_field("unit", axis)
            assert page.build_form().q_range == (1., 3.)
        _click(page, "Standard")  # the GI axis becomes Q: the same unit, the same range
        assert page._value("unit") == "Q (Å⁻¹)" and page.build_form().q_range == (1., 3.)
        page.set_field("unit", "2θ (°)")
        assert page.build_form().q_range == (0., 90.)
    finally:
        _close(page, qapp)


def _mu_per_point(spec, values=(.5, 1.5)):
    """Move mu from the scans' #O/#P headers to a per-point #L column with ``values``."""
    lines, rows = [], None
    for line in spec.read_text().splitlines():
        if line.startswith("#O0"):
            line = line.replace("  mu", "")
        elif line.startswith("#P0"):  # the header order is del nu eta mu chi phi: drop mu's 0
            parts = line.split()
            line = " ".join(parts[:4] + parts[5:])
        elif line.startswith("#N "):
            line = f"#N {int(line.split()[1]) + 1}"
        elif line.startswith("#L"):
            line, rows = line + "  mu", 0
        elif rows is not None and line.strip() and not line.startswith("#"):
            line, rows = f"{line} {values[rows]}", rows + 1
        else:
            rows = None
        lines.append(line)
    spec.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
def test_a_sample_circle_recorded_per_point_is_bound_not_read_as_zero(qapp, tmp_path, backend):
    """Codex re-review R2-F2: a circle a scan records only per point (an #L column, not #O/#P) is
    recorded: it is bound, each frame's own value reaches the plan, and nothing says it reads 0."""
    from xdart.gui.tools.stitch_values import prepare_stitch_tool
    page = _page(tmp_path, backend)
    try:
        spec = Path(page._source_paths[0])
        _mu_per_point(spec)
        page.set_sources((str(spec),))
        _click(page, "Grazing")
        page.set_field("unit", "Qip (Å⁻¹)")
        prepared = prepare_stitch_tool(page.build_form())
        assert all(dict(request.plan.gi.sample_motors)["mu"] == "mu" for request in prepared.requests)
        assert not any("read as 0" in item for item in prepared.warnings)
        bound = [{name: value for name, _occurrence, value in item.values}
                 for request in prepared.requests for item in request.manifest.contributions]
        assert [item["mu"] for item in bound] == [.5, 1.5]
    finally:
        _close(page, qapp)


def test_an_edit_reads_no_scan_table_and_lists_no_image_folder(qapp, tmp_path, monkeypatch):
    """Codex audit G-F2: the page's edits (drafts and commits, Grazing on, every scan, the form) read
    no SPEC data table and list no image folder; Preview does that on the worker. The page's one
    source read is the first scan's header (the incidence and tilt choices), on a committed source
    or scan edit only."""
    import xrd_tools.io.image as image
    import xrd_tools.io.spec as spec_io
    from xdart.gui.tabs.stitch import page as page_module
    calls = []
    for module, name in ((spec_io, "read_spec_scan_columns"), (spec_io, "list_spec_scans"),
                         (image, "find_image_files"), (spec_io, "read_spec_scan_motor_names")):
        real = getattr(module, name)
        monkeypatch.setattr(module, name, lambda *args, real=real, name=name, **kwargs:
                            (calls.append(name), real(*args, **kwargs))[1])
    page_module._spec_motor_names.cache_clear()
    page = _page(tmp_path, "multigeometry")
    try:
        _click(page, "Grazing")
        assert page._gi_catalog == _MOTORS
        calls.clear()
        page_module._spec_motor_names.cache_clear()  # a warm cache would hide a read on a draft
        page._rows["scan"].draftChanged.emit(("Stitch", "scan"), "1, 2")  # a draft: no read
        page.set_field("image_stem", "sample_scan{scan}_")
        page.set_field("unit", "Qip (Å⁻¹)")
        page.build_form()
        assert calls == []
        page.set_field("scan_mode", "Stitch All")  # a commit: the first scan's header, nothing else
        page.build_form()
        assert calls == ["read_spec_scan_motor_names"] and page._gi_catalog == _MOTORS
        assert page._rows["output_family"].editor.placeholderText() == "auto · as Int names the scan's images, with _gi"
    finally:
        _close(page, qapp)


def test_append_commits_a_manual_angle_being_typed_before_it_skips(qapp, tmp_path):
    """Codex testing-fixes review F1: after Preview found the output already stitched, a manual incidence
    angle typed without Return (a click on Run leaves the focus in the field on macOS) was ignored:
    Append reused Preview's check and skipped. The typing is committed first, so the Run checks again."""
    from PySide6.QtTest import QTest
    from tests.xdart.pages.test_stitch_page import _wait
    page = _page(tmp_path, "multigeometry", StitchGI(("q_ip", "q_oop"), incident_angle_deg=.3))
    try:
        page.show()
        qapp.processEvents()
        page.begin_run()
        _wait(page, qapp)
        page.begin_preview()
        _wait(page, qapp)
        assert page._owner.prepared.unchanged == (0,)
        page.preview_dialog.close()
        page.activateWindow()
        qapp.processEvents()
        angle = _row(page, GI_THETA).editor
        angle.setFocus()
        qapp.processEvents()
        angle.selectAll()
        QTest.keyClicks(angle, "0.6")
        assert QtWidgets.QApplication.focusWidget() is angle
        page.begin_run()
        _wait(page, qapp)
        assert page.build_form().gi.incident_angle_deg == .6
        assert page.run_bar.readinessLabel.full_text().startswith("Complete · "), (
            page.run_bar.readinessLabel.full_text())
    finally:
        _close(page, qapp)


def test_a_geometry_edit_leaves_the_gi_card_and_a_rebuild_makes_no_window(qapp, tmp_path, monkeypatch):
    """Maintainer, 2026-09-29: typing in the calibration window's Diffractometer panel, or Read metadata,
    closed that window (macOS). Each geometry edit rebuilt the page's GI card: the old one deleted and a
    new one created with no parent, a top-level window until laid out. The card shows nothing of the
    geometry, so a geometry edit leaves it; a rebuild makes the card inside its host."""
    from xdart.gui.widgets.controls_panel import SubsectionCard
    parents = []
    born = SubsectionCard.__init__

    def recording(card, *args, **kwargs):
        born(card, *args, **kwargs)
        parents.append(card.parent())

    monkeypatch.setattr(SubsectionCard, "__init__", recording)
    page = _page(tmp_path, "multigeometry")
    try:
        _click(page, "Grazing")
        page._calibrate()
        dialog = page._calibration_dialog
        card, parents[:] = page._gi_card, []
        for circle, text in (("del", "2.3"), ("nu", "-0.5")):
            dialog.geometry_editor.zero_edit(circle).setText(text)
        dialog.geometry_editor.set_value(dialog.geometry_editor.value())  # as Read metadata sets it
        qapp.processEvents()
        assert page._gi_card is card and parents == []
        _click(page, "Standard")  # a rebuild: the card is born in its host
        assert parents and all(parent is page._gi_host for parent in parents)
    finally:
        if page._calibration_dialog is not None:
            page._calibration_dialog.close()
        _close(page, qapp)


def test_the_diffractometer_panels_sample_zeros_reach_grazing_incidence(qapp, tmp_path):
    """Deferred item 7: the Diffractometer panel's zero beside each sample circle (blank: 0) is
    subtracted from its motor's reading wherever grazing incidence reads it."""
    from xrd_tools.session.instrument import InstrumentGeometry
    from xdart.gui.widgets.geometry_editor import GeometryEditor
    editor = GeometryEditor()
    try:
        for circle in ("mu", "eta", "chi", "phi", "del", "nu"):
            assert editor.zero_edit(circle).placeholderText() == "0"
        editor.zero_edit("eta").setText("0.5")
        assert editor.value().zero("eta") == 0.5 and editor.value().zero("phi") is None
    finally:
        editor.deleteLater()
    page = _page(tmp_path, "multigeometry")
    try:
        geometry = page._instrument()
        page._set_instrument(InstrumentGeometry(geometry.geometry, geometry.motors, geometry.plane,
                                                (("eta", 0.5), ("phi", -1.0))))
        _click(page, "Grazing")
        page.set_field("unit", "Qip (Å⁻¹)")
        gi = page.build_form().gi
        assert dict(gi.motor_zeros) == {"eta": 0.5, "phi": -1.0}
    finally:
        _close(page, qapp)
