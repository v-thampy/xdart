"""BrowserView's plain keyed list (FIT-TOOLS-2026-10, Codex's design review F7: "Extend BrowserView for a plain keyed
list profile without importing scan/frame semantics into CIF rows"): the Phase Fit's phases are the shared browser's
list, which the window reconciles from its own collection after each Add or Remove. Int's, Stitch's and RSM's
browsers keep their own profiles (``test_browser_results_profile.py``, ``test_stitch_browser.py``,
``test_rsm_page.py``)."""
from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.tabs.scattering.browser_view import BrowserResult, BrowserView, keyed_list_profile
from xdart.gui.tabs.scattering.shell_values import ShellCommandKind


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _listed(view):
    return [(view.scans.item(row).data(256), view.scans.item(row).text(), view.scans.item(row).toolTip())
            for row in range(view.scans.count())]


def test_a_keyed_list_is_one_list_of_a_windows_entries_without_a_scans_or_frames_meaning() -> None:
    _app()
    view = BrowserView(profile=keyed_list_profile("Phases"))
    commands = []
    view.commandRequested.connect(commands.append)
    try:
        view.show()
        assert not view.menus and not view.heading.isVisible()
        assert view.scans_label.text() == "Phases" and view.scans_label.isVisible()
        assert not view.directory_label.isVisible() and not view.frames.isVisible()
        assert not any(button.isVisible() for button in view._action_buttons.values())
        assert not view.date_sort.isVisible() and not view.refresh.isVisible()
        alpha = BrowserResult("/cif/alpha.cif", "alpha", "/cif/alpha.cif")
        beta = BrowserResult("/cif/beta.cif", "beta", "/cif/beta.cif")
        view.reconcile_results((alpha,), None)  # Add
        view.reconcile_results((alpha, beta), beta.identifier)  # Add, the new one current
        assert _listed(view) == [("/cif/alpha.cif", "alpha", "/cif/alpha.cif"),
                                 ("/cif/beta.cif", "beta", "/cif/beta.cif")]
        assert view.current_result() == "/cif/beta.cif" and not commands  # reconciling emits nothing
        view.set_current_result("/cif/alpha.cif", keep_selection=False)  # the user picks alpha
        assert commands[-1].kind is ShellCommandKind.SELECT_RESULT and commands[-1].value == "/cif/alpha.cif"
        receipt = BrowserResult("/cif/beta.cif", "beta", "sha256 0123")  # a fit's receipt for beta
        view.reconcile_results((receipt,), None)  # Remove alpha
        assert _listed(view) == [("/cif/beta.cif", "beta", "sha256 0123")] and view.current_result() is None
    finally:
        view.close()
        view.deleteLater()


def test_ints_browser_keeps_its_heading() -> None:
    _app()
    view = BrowserView()
    try:
        view.show()
        assert view.heading.text() == "DATA BROWSER" and view.heading.isVisible()
        assert view.profile.heading
    finally:
        view.close()
        view.deleteLater()


def test_independent_column_checks_reconcile_silently_and_emit_identities_in_list_order() -> None:
    _app()
    view = BrowserView(profile=keyed_list_profile("Columns", checkable=True))
    changed = []
    view.checksChanged.connect(changed.append)
    try:
        a, b, c = (BrowserResult(name, name.upper()) for name in ("a", "b", "c"))
        view.reconcile_checked_results((a, b), ("b",))
        assert view.checked_results() == ("b",) and not changed
        view.scans.item(0).setCheckState(QtCore.Qt.CheckState.Checked)
        assert changed == [("a", "b")] and view.current_result() is None
        view.reconcile_checked_results((c, a, b), view.checked_results())
        assert view.checked_results() == ("a", "b") and len(changed) == 1
        view.reconcile_checked_results((c, b), ("a", "b"))
        assert view.checked_results() == ("b",) and len(changed) == 1
    finally:
        view.close()
        view.deleteLater()
