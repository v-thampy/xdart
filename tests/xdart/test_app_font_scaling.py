"""Frozen acceptance oracle for the application-wide five-tier font scale.

Config ▸ **Font Size** owns ONE preference —
``extra_small / small / default / large / extra_large`` — that applies to the
whole xdart application: ordinary Qt widgets (via ``QApplication.setFont``),
the theme's dense Controls QSS tokens, and pyqtgraph tick / axis / legend
fonts on plots that already exist.  The shared theme layer is the only owner:
pages never read ``QSettings`` and never hold a second scale value.

Every row here maps to a required mutation red (see the packet's mutation
table), so read a failure as "the contract moved", not "the test is fussy":

===========================================  =================================
production mutation                          row that must go red
===========================================  =================================
derive the tier from the *current* font      ``test_default_large_default_restores_exact_metrics``
                                             ``test_reapplying_the_same_tier_is_idempotent``
scale only the Controls selectors            ``test_tier_reaches_ordinary_widgets_and_dialogs``
omit plot restyling                          ``test_plot_built_before_the_change_converges_with_one_built_after``
retain plots strongly                        ``test_closed_plots_release_from_the_registry``
apply the saved tier only after widgets      ``test_saved_tier_is_applied_before_widget_construction``
drop the legacy-key migration                ``test_legacy_control_panel_key_migrates_without_drift``
remove Extra Small or Extra Large            ``test_menu_exposes_exactly_five_exclusive_tiers``
hard-code one fixed toolbar height at XL     ``test_extreme_tiers_do_not_clip_the_three_column_shell``
reapply the theme at Default regardless      ``test_theme_switch_preserves_the_selected_tier``
===========================================  =================================

Settings isolation: the theme layer reads ``XDART_SETTINGS_FILE`` (the same
shape as ``XDART_SESSION_FILE``), so no test — and no standalone probe — can
touch the maintainer's real ``com.xdart.xdart`` preferences.  ``conftest.py``
points it at a scratch ``.ini`` for the whole session; the ``settings``
fixture below narrows it to one file per test.
"""
import gc
import os
import weakref
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pyqtgraph")
import pyqtgraph as pg
from pyqtgraph import QtGui, QtWidgets

from xdart.gui.themes import apply_theme, render_qss
from xdart.gui.themes import typography as typo


# ── harness ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    """The session QApplication, restored to how this module found it.

    This is the only module in the suite that installs an application
    stylesheet or changes the application font, and both are process-global:
    left behind, they would silently move every sizeHint and font metric in
    whatever module pytest runs next.  Snapshot and put them back.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # Capture the pristine platform baseline BEFORE any row applies a tier, so
    # every row derives from the same immutable origin (contract rule 1).
    typo.capture_application_baseline(app)
    original_qss = app.styleSheet()
    original_font = QtGui.QFont(app.font())
    try:
        yield app
    finally:
        app.setStyleSheet(original_qss)
        app.setFont(original_font)
        # setFont clears the per-class hash; put the platform's own sizes back
        # at offset 0 so a later module sees exactly what it would have seen.
        typo.restore_platform_class_fonts(app, typo.DEFAULT_FONT_SCALE)
        for _ in range(3):
            app.processEvents()


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """A real QSettings on a scratch .ini — the production accessor, redirected."""
    monkeypatch.setenv("XDART_SETTINGS_FILE", str(tmp_path / "xdart.ini"))
    return typo.application_settings()


@pytest.fixture(autouse=True)
def _restore_default_tier(qapp):
    """Leave the process at Default so row order cannot matter."""
    yield
    apply_theme(qapp, "dark", font_scale=typo.DEFAULT_FONT_SCALE)


class _FakeStaticWidget(QtWidgets.QWidget):
    """The host page stub used by the existing main-window menu sentinel.

    The seam under test is the menu/preference owner inside ``Main`` — the page
    is only the surface the Config menu is hung on, so the real (multi-second)
    staticWidget is not needed for the menu rows.  Geometry rows below use the
    real page.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.h5viewer = SimpleNamespace(
            paramMenu=QtWidgets.QMenu(self), helpMenu=QtWidgets.QMenu(self))
        self.ui = SimpleNamespace(
            leftFrame=QtWidgets.QFrame(self),
            middleFrame=QtWidgets.QFrame(self),
            rightFrame=QtWidgets.QFrame(self),
        )

    def enable_async_hydration(self):
        pass


def _main_window(monkeypatch):
    from xdart import _gui_main
    from xdart.gui.pages.descriptors import PageDescriptor
    from xdart.gui.pages.handle import PageHandle
    from xdart.gui.pages.values import (
        CloseReceipt, PageCleanup, PageKey, PageLifecycle,
    )
    key = PageKey("font-test")
    clean = CloseReceipt(PageCleanup.CLEAN, "verified")

    def build(_services, parent):
        return PageHandle(
            key=key, widget=_FakeStaticWidget(parent), close=lambda: clean)

    descriptor = PageDescriptor(
        key=key, label="Font Test", order=0, build=build,
        lifecycle=PageLifecycle.SWITCHABLE, capabilities=frozenset(),
    )
    return _gui_main.Main(
        page_descriptors=(descriptor,), selected_page_key=key)


def _submenu(window, title):
    """Find a Config submenu, keeping its owning QAction alive.

    Returning the bare QMenu is a trap: ``QAction.menu()`` hands back a wrapper
    whose validity shiboken ties to the QAction wrapper it came from, and the
    QActions here are temporaries from ``actions()``.  Once they are collected
    the menu wrapper raises "Internal C++ object already deleted" even though
    Qt is still happily showing the menu.  Hold the pair.
    """
    for action in window.host_config_menu.actions():
        menu = action.menu()
        if menu is not None and menu.title() == title:
            return action, menu
    return None, None


def _font_actions(window):
    owner, menu = _submenu(window, "Font Size")
    assert menu is not None, "Config ▸ Font Size submenu is missing"
    actions = menu.actions()
    assert owner is not None
    return actions


def _trigger_tier(window, label):
    for action in _font_actions(window):
        if action.text() == label:
            action.trigger()
            return action
    raise AssertionError(f"no {label!r} action in Config ▸ Font Size")


def _probe_metrics(qapp):
    """Metrics that must restore EXACTLY across a tier round-trip."""
    label = QtWidgets.QLabel("Refresh")
    button = QtWidgets.QPushButton("Reintegrate")
    combo = QtWidgets.QComboBox()
    combo.addItem("q (A-1)")
    return {
        "app_pt": qapp.font().pointSize(),
        "label_hint": (label.sizeHint().width(), label.sizeHint().height()),
        "button_hint": (button.sizeHint().width(), button.sizeHint().height()),
        "combo_hint": (combo.sizeHint().width(), combo.sizeHint().height()),
        "qss": render_qss("dark", font_scale=typo.current_font_scale()),
        "plot_pt": typo.plot_font().pointSize(),
    }


def _styled_plot():
    """A production-shaped pyqtgraph plot: built, labelled, legended, styled."""
    from xdart.gui.themes import apply_seaborn_plot_style
    widget = pg.GraphicsLayoutWidget()
    plot = widget.addPlot()
    plot.setLabel("bottom", "q")
    plot.setLabel("left", "Intensity")
    legend = plot.addLegend()
    plot.plot([0, 1, 2], [1, 2, 3], name="frame 0")
    apply_seaborn_plot_style(plot)
    return widget, plot, legend


def _plot_font_metrics(plot, legend):
    bottom = plot.getAxis("bottom")
    left = plot.getAxis("left")
    return {
        "tick_bottom": bottom.style["tickFont"].pointSize(),
        "tick_left": left.style["tickFont"].pointSize(),
        "label_bottom": bottom.label.font().pointSize(),
        "label_left": left.label.font().pointSize(),
        "legend": legend.labelTextSize(),
    }


# ── 1. menu inventory and default selection ──────────────────────────────

def test_menu_exposes_exactly_five_exclusive_tiers(qapp, settings, monkeypatch):
    window = _main_window(monkeypatch)
    actions = _font_actions(window)

    assert [a.text() for a in actions] == [
        "Extra Small", "Small", "Default", "Large", "Extra Large"]
    assert all(a.isCheckable() for a in actions)

    groups = {a.actionGroup() for a in actions}
    assert len(groups) == 1, "the five tiers must share one action group"
    group = groups.pop()
    assert group is not None and group.isExclusive()

    assert [a.isChecked() for a in actions] == [
        False, False, True, False, False], "Default is the default selection"

    assert _submenu(window, "Control Panel Font Size")[1] is None, (
        "the superseded Controls-only submenu must be gone")


def test_saved_tier_is_preselected_in_the_menu(qapp, settings, monkeypatch):
    settings.setValue(typo.FONT_SCALE_SETTINGS_KEY, "extra_large")
    settings.sync()
    window = _main_window(monkeypatch)
    checked = [a.text() for a in _font_actions(window) if a.isChecked()]
    assert checked == ["Extra Large"]


# ── 2 + 12. settings migration and malformed-value fallback ──────────────

@pytest.mark.parametrize("legacy", ["small", "default", "large"])
def test_legacy_control_panel_key_migrates_without_drift(settings, legacy):
    settings.setValue(typo.LEGACY_FONT_SCALE_SETTINGS_KEY, legacy)
    settings.sync()
    assert typo.resolve_font_scale(settings) == legacy


def test_new_key_wins_over_the_legacy_key(settings):
    settings.setValue(typo.FONT_SCALE_SETTINGS_KEY, "extra_small")
    settings.setValue(typo.LEGACY_FONT_SCALE_SETTINGS_KEY, "large")
    settings.sync()
    assert typo.resolve_font_scale(settings) == "extra_small"


def test_absent_keys_resolve_to_default(settings):
    assert typo.resolve_font_scale(settings) == typo.DEFAULT_FONT_SCALE


@pytest.mark.parametrize("junk", [
    "", "  ", "HUGE", "12", "Large", "extra large", "extra-large", None, 3.5, [],
])
def test_malformed_values_fall_back_to_default(settings, junk):
    settings.setValue(typo.FONT_SCALE_SETTINGS_KEY, junk)
    settings.sync()
    assert typo.resolve_font_scale(settings) == typo.DEFAULT_FONT_SCALE
    assert typo.normalize_font_scale(junk) == typo.DEFAULT_FONT_SCALE


def test_only_exact_known_values_are_persisted(qapp, settings, monkeypatch):
    window = _main_window(monkeypatch)
    window._set_application_font_size("not-a-tier")
    assert settings.value(typo.FONT_SCALE_SETTINGS_KEY) == typo.DEFAULT_FONT_SCALE


def test_the_five_tiers_are_the_whole_contract():
    assert typo.FONT_SCALES == (
        "extra_small", "small", "default", "large", "extra_large")
    assert typo.DEFAULT_FONT_SCALE == "default"
    assert [label for _key, label in typo.FONT_SCALE_MENU] == [
        "Extra Small", "Small", "Default", "Large", "Extra Large"]
    assert tuple(key for key, _label in typo.FONT_SCALE_MENU) == typo.FONT_SCALES
    assert set(typo.FONT_SCALE_TOKENS) == set(typo.FONT_SCALES)


def test_token_table_is_monotonic_and_centred_on_default():
    offsets = [typo.FONT_SCALE_TOKENS[s].app_offset_pt for s in typo.FONT_SCALES]
    assert offsets == sorted(offsets) and len(set(offsets)) == 5
    assert typo.FONT_SCALE_TOKENS["default"].app_offset_pt == 0
    for field in ("control_px", "browse_px", "plot_pt", "title_pt"):
        values = [getattr(typo.FONT_SCALE_TOKENS[s], field)
                  for s in typo.FONT_SCALES]
        assert values == sorted(values) and len(set(values)) == 5, field


def test_default_tier_preserves_the_pre_change_dense_tokens():
    """Default must render byte-identically to the superseded 'default' preset."""
    tokens = typo.qss_font_tokens("default")
    assert tokens["control_panel_font"] == "12px"
    assert tokens["control_panel_status_font"] == "12px"
    assert tokens["control_panel_tick_font"] == "12px"
    assert tokens["control_panel_browse_font"] == "13px"
    assert tokens["control_panel_run_font"] == "13px"
    assert typo.FONT_SCALE_TOKENS["default"].plot_pt == 11


# ── 3. app-wide effect: ordinary widget, Controls field, dialog, plot ────

def test_tier_reaches_ordinary_widgets_and_dialogs(qapp):
    apply_theme(qapp, "dark", font_scale="default")
    label = QtWidgets.QLabel("DATA BROWSER")
    button = QtWidgets.QPushButton("Refresh")
    dialog = QtWidgets.QDialog()
    dialog_label = QtWidgets.QLabel("Update available", dialog)
    small = {
        "app": qapp.font().pointSize(),
        "label": label.sizeHint().height(),
        "button": button.sizeHint().height(),
        "dialog": dialog_label.sizeHint().height(),
    }

    apply_theme(qapp, "dark", font_scale="extra_large")
    big = {
        "app": qapp.font().pointSize(),
        "label": label.sizeHint().height(),
        "button": button.sizeHint().height(),
        "dialog": dialog_label.sizeHint().height(),
    }

    assert big["app"] > small["app"]
    for key in ("label", "button", "dialog"):
        assert big[key] > small[key], (
            f"{key} did not grow: a Controls-only scale is not application-wide")


def test_tier_reaches_the_controls_panel_fields(qapp, shell):
    """The dense Controls tokens are a density variant of the ONE tier.

    Measured on the real panel inside the real page: a bare ControlsPanelV2
    has only its section cards, the fields arrive with the page's wiring.
    """
    _window, widget = shell
    apply_theme(qapp, "dark", font_scale="default")
    qapp.processEvents()
    fields = [w for w in widget.findChildren(QtWidgets.QWidget)
              if w.objectName().startswith("controlsV2")
              and w.font().pixelSize() > 0]
    assert fields, "no ControlsPanel V2 widget is carrying the dense token"
    probe = fields[0]
    small = probe.font().pixelSize()

    apply_theme(qapp, "dark", font_scale="extra_large")
    qapp.processEvents()
    big = probe.font().pixelSize()

    assert big > small, (
        f"{probe.objectName()} stayed at {small}px: the dense Controls token "
        "is not following the application tier")


def test_tier_reaches_pyqtgraph_axis_tick_and_legend(qapp):
    apply_theme(qapp, "dark", font_scale="default")
    widget, plot, legend = _styled_plot()
    small = _plot_font_metrics(plot, legend)

    apply_theme(qapp, "dark", font_scale="extra_large")
    big = _plot_font_metrics(plot, legend)

    assert big["tick_bottom"] > small["tick_bottom"]
    assert big["tick_left"] > small["tick_left"]
    assert big["label_bottom"] > small["label_bottom"]
    assert big["label_left"] > small["label_left"]
    assert big["legend"] != small["legend"]
    widget.close()


# ── 4. live change now, and persistence across a restart ────────────────

def test_live_change_applies_once_and_persists_once(
        qapp, settings, monkeypatch):
    window = _main_window(monkeypatch)
    baseline = typo.application_baseline_point_size()

    calls = []
    real_apply = apply_theme
    import xdart.gui.themes as themes
    monkeypatch.setattr(
        themes, "apply_theme",
        lambda app, name="dark", *, font_scale=typo.DEFAULT_FONT_SCALE,
        **appearance: (
            calls.append((name, font_scale)),
            real_apply(
                app,
                name,
                font_scale=font_scale,
                **appearance,
            ),
        )[1],
    )

    _trigger_tier(window, "Large")

    assert calls == [("dark", "large")], "one apply per live change"
    assert qapp.font().pointSize() == baseline + 1
    assert settings.value(typo.FONT_SCALE_SETTINGS_KEY) == "large"

    # "restart": a fresh accessor over the same persisted file.
    assert typo.resolve_font_scale(typo.application_settings()) == "large"


def test_saved_tier_is_applied_before_widget_construction(
        qapp, settings, monkeypatch):
    """Startup must scale the app BEFORE any widget or plot is constructed.

    Driven through the real ordering owner (``_gui_main._start_gui``) rather
    than ``run()``: run() also claims process-global state (the QtAgg
    matplotlib flip, faulthandler, the rotating log file) that a test must not
    hijack — see the module banner in ``_gui_main``.  The guard below pins that
    ``run()`` still delegates here.
    """
    from xdart import _gui_main
    settings.setValue(typo.FONT_SCALE_SETTINGS_KEY, "extra_large")
    settings.sync()
    apply_theme(qapp, "dark", font_scale="default")

    baseline = typo.application_baseline_point_size()
    seen = []

    class _RecordingWindow(QtWidgets.QWidget):
        def __init__(self):
            # Point size AT CONSTRUCTION TIME — the whole assertion.
            seen.append(qapp.font().pointSize())
            super().__init__()

    _gui_main._start_gui(qapp, window_factory=_RecordingWindow)

    assert seen == [baseline + 2], (
        "the window was constructed before the saved tier was applied")


def test_run_delegates_startup_to_the_ordering_owner():
    main = (Path(__file__).resolve().parents[2] / "src" / "xdart"
            / "_gui_main.py").read_text(encoding="utf-8")
    run_body = main.split("\ndef run():", 1)[1].split("\nmain = run", 1)[0]
    assert "_start_gui(app)" in run_body
    assert "Main()" not in run_body, (
        "run() must construct the window through _start_gui so the saved tier "
        "is always applied first")


# ── 5 + 6. exact restoration and idempotence ────────────────────────────

def test_default_large_default_restores_exact_metrics(qapp):
    apply_theme(qapp, "dark", font_scale="default")
    before = _probe_metrics(qapp)

    apply_theme(qapp, "dark", font_scale="large")
    during = _probe_metrics(qapp)
    assert during != before

    apply_theme(qapp, "dark", font_scale="default")
    after = _probe_metrics(qapp)

    assert after == before, (
        "Default → Large → Default drifted: the tier is being derived from the "
        "CURRENT font instead of the captured platform baseline")


def test_every_tier_round_trips_through_default(qapp):
    apply_theme(qapp, "dark", font_scale="default")
    origin = _probe_metrics(qapp)
    for scale in typo.FONT_SCALES:
        apply_theme(qapp, "dark", font_scale=scale)
        apply_theme(qapp, "dark", font_scale="default")
        assert _probe_metrics(qapp) == origin, f"drift after visiting {scale}"


def test_reapplying_the_same_tier_is_idempotent(qapp):
    apply_theme(qapp, "dark", font_scale="large")
    once = _probe_metrics(qapp)
    for _ in range(4):
        apply_theme(qapp, "dark", font_scale="large")
    assert _probe_metrics(qapp) == once


def test_tiers_are_derived_from_the_captured_baseline(qapp):
    baseline = typo.application_baseline_point_size()
    assert baseline > 0
    for scale in typo.FONT_SCALES:
        apply_theme(qapp, "dark", font_scale=scale)
        expected = baseline + typo.FONT_SCALE_TOKENS[scale].app_offset_pt
        assert qapp.font().pointSize() == expected, scale
        assert typo.application_baseline_point_size() == baseline, (
            "the baseline moved: it must be captured once, never re-read")


def test_platform_per_class_fonts_ride_the_tier(qapp):
    """A platform that gives a widget class its own font keeps that relationship.

    macOS does, and the gaps are large — measured on cocoa, ``QToolButton`` is
    10 pt against a 13 pt application font, and ``QTipLabel``/``QHeaderView``/
    ``QSmallFont`` are 11 pt.  Both ``QApplication.setFont`` and the FIRST
    ``setStyleSheet`` wipe that hash, so without an explicit restore the
    *Default* tier would enlarge every tool button by 30% — a change to the
    shipped look that no tier asked for.

    The offscreen platform used by this suite defines no per-class fonts at
    all, so a plain assertion here would be structurally blind and pass on a
    branch that regressed macOS.  Seed one and re-arm the one-shot capture.
    """
    saved = (typo._BASELINE_FONT, typo._BASELINE_POINT_SIZE,
             typo._BASELINE_CLASS_FONTS, typo._CURRENT_SCALE)
    try:
        apply_theme(qapp, "dark", font_scale="default")
        seeded = max(1, qapp.font().pointSize() - 3)
        probe = QtGui.QFont(qapp.font())
        probe.setPointSize(seeded)
        # QMiniFont deliberately: it is a Qt pseudo-class no widget in the
        # shell resolves against, so seeding it exercises the capture/shift/
        # restore path without leaving a real control pinned to one tier for
        # every later row in the file.
        qapp.setFont(probe, "QMiniFont")

        typo._BASELINE_FONT = None
        typo._BASELINE_POINT_SIZE = 0
        typo._BASELINE_CLASS_FONTS = {}
        typo.capture_application_baseline(qapp)
        assert typo.platform_class_font_baseline().get("QMiniFont") == seeded

        for scale in typo.FONT_SCALES:
            apply_theme(qapp, "dark", font_scale=scale)
            offset = typo.FONT_SCALE_TOKENS[scale].app_offset_pt
            actual = QtWidgets.QApplication.font("QMiniFont").pointSize()
            assert actual == seeded + offset, (
                f"{scale}: QMiniFont is {actual}pt, expected "
                f"{seeded + offset}pt — the platform's own sizing was lost")

        apply_theme(qapp, "dark", font_scale="default")
        assert QtWidgets.QApplication.font("QMiniFont").pointSize() == seeded, (
            "Default is not a no-op for the platform's per-class fonts")
    finally:
        (typo._BASELINE_FONT, typo._BASELINE_POINT_SIZE,
         typo._BASELINE_CLASS_FONTS, typo._CURRENT_SCALE) = saved
        apply_theme(qapp, "dark", font_scale=typo.DEFAULT_FONT_SCALE)


# ── 7 + 8. live plots: convergence and weak release ─────────────────────

def test_plot_built_before_the_change_converges_with_one_built_after(qapp):
    apply_theme(qapp, "dark", font_scale="default")
    old_widget, old_plot, old_legend = _styled_plot()

    apply_theme(qapp, "dark", font_scale="extra_large")
    new_widget, new_plot, new_legend = _styled_plot()

    assert _plot_font_metrics(old_plot, old_legend) == \
        _plot_font_metrics(new_plot, new_legend), (
            "a plot that existed before the change did not follow the tier")
    old_widget.close()
    new_widget.close()


def test_closed_plots_are_not_retained_by_a_scale_change(qapp):
    """A scale change must never be what keeps a closed plot alive."""
    apply_theme(qapp, "dark", font_scale="default")
    before = typo.live_plot_item_count()

    widget, plot, _legend = _styled_plot()
    assert typo.live_plot_item_count() == before + 1
    ref = weakref.ref(plot)

    widget.close()
    del widget, plot, _legend
    gc.collect()
    qapp.processEvents()
    gc.collect()

    # The scale change must neither resurrect the dead plot nor trip over it.
    apply_theme(qapp, "dark", font_scale="large")

    assert ref() is None, "a closed plot was retained across a scale change"
    assert typo.live_plot_item_count() == before


def test_the_theme_layer_keeps_no_container_of_plots(qapp):
    """The restyle walks live Qt scenes and holds nothing.

    Stronger than a weak registry, and the reason the dialogs' plots (peak fit,
    phase fit, scan plot, ROI select, the parameter-trend right-hand axis) and
    every colour bar follow the tier without opting in one call site at a time.
    """
    containers = [
        name for name, value in vars(typo).items()
        if not name.startswith("__")
        and isinstance(value, (list, set, dict, frozenset, tuple))
        and any(hasattr(v, "getAxis") for v in
                (value.values() if isinstance(value, dict) else value))
    ]
    assert containers == [], (
        f"the theme layer is holding plots in {containers}")

    widget, plot, legend = _styled_plot()
    try:
        # A plot NOT created through the theme helper must still follow the
        # tier -- that is what proves the walk, not a registration list.
        bare_widget = pg.GraphicsLayoutWidget()
        bare = bare_widget.addPlot()
        bare.setLabel("bottom", "frame")
        bare_legend = bare.addLegend()
        bare.plot([0, 1], [1, 2], name="unregistered")
        bare_widget.show()

        apply_theme(qapp, "dark", font_scale="extra_large")
        assert _plot_font_metrics(bare, bare_legend) == \
            _plot_font_metrics(plot, legend), (
                "a plot that never called the theme helper did not follow the "
                "tier -- the restyle is registration-based, not a live walk")
        bare_widget.close()
    finally:
        widget.close()


# ── 9. geometry: extreme tiers against the three-column shell ───────────

SHELL_SIZES = [(1920, 1080), (1440, 900), (1024, 900)]


def _settle(qapp, window, widget, size):
    """Resize, then drain until the layout stops moving.

    The refits that follow a tier change are deferred one event-loop turn on
    purpose (a widget's cached size hint still answers for the previous tier
    when FontChange arrives), and each refit can itself invalidate a parent's
    hint.  A fixed number of ``processEvents`` rounds therefore measures a
    layout that is still converging, which shows up as an intermittent
    failure rather than an honest one.  Iterate to a fixed point instead, with
    a bound so a genuinely unstable layout still fails the caller's assertion.
    """
    window.resize(*size)
    previous = None
    for _ in range(12):
        for child in widget.findChildren(QtWidgets.QWidget):
            try:
                child.ensurePolished()
                child.updateGeometry()
            except Exception:
                pass
        for _ in range(3):
            qapp.processEvents()
        current = _clipped_widgets(widget)
        if current == previous:
            return
        previous = current


def _clipped_widgets(root):
    """Visible widgets whose hard ceiling is below what their content needs —
    i.e. text the user cannot read at this tier."""
    bad = set()
    for w in root.findChildren(QtWidgets.QWidget):
        if not w.isVisibleTo(root):
            continue
        hint = w.minimumSizeHint()
        name = w.objectName() or type(w).__name__
        if 0 < w.maximumHeight() < hint.height():
            bad.add(f"{name}:H")
        if 0 < w.maximumWidth() < hint.width():
            bad.add(f"{name}:W")
    return bad


@pytest.fixture(scope="module")
def shell(qapp):
    """The real three-column page, built once (it is expensive)."""
    previous = os.environ.get("XDART_CONTROLS_PANEL_V2")
    os.environ["XDART_CONTROLS_PANEL_V2"] = "1"
    from xdart.gui.tabs.static_scan import staticWidget
    window = QtWidgets.QMainWindow()
    widget = staticWidget()
    window.setCentralWidget(widget)
    window.show()
    try:
        yield window, widget
    finally:
        window.close()
        if previous is None:
            os.environ.pop("XDART_CONTROLS_PANEL_V2", None)
        else:
            os.environ["XDART_CONTROLS_PANEL_V2"] = previous


@pytest.mark.parametrize("scale", ["extra_small", "extra_large"])
@pytest.mark.parametrize("size", SHELL_SIZES)
def test_extreme_tiers_do_not_clip_the_three_column_shell(
        qapp, shell, scale, size):
    """No control becomes unreadable at an extreme tier that was readable at
    Default.

    A *regression* bar, deliberately.  Seven controls are already clipped at
    Default on this branch's parent (``cmap``, ``controlsFrame``,
    ``controlsV2BrowseButton``, ``controlsV2Chevron``, ``maxCoresSpinBox``,
    ``slice_center``, ``slice_width``) — pre-existing bugs this packet does not
    own.  What it does own is that turning the preference application-wide adds
    none of its own.
    """
    window, widget = shell

    apply_theme(qapp, "dark", font_scale="default")
    _settle(qapp, window, widget, size)
    baseline = _clipped_widgets(widget)

    apply_theme(qapp, "dark", font_scale=scale)
    _settle(qapp, window, widget, size)

    for name in ("leftFrame", "middleFrame", "rightFrame"):
        column = getattr(widget.ui, name)
        assert column.width() > 0 and column.height() > 0, (
            f"{name} collapsed at {scale} / {size[0]}x{size[1]}")

    new = _clipped_widgets(widget) - baseline
    assert new == set(), (
        f"{scale} at {size[0]}x{size[1]} newly clips {sorted(new)}")


def test_tier_change_restores_the_shell_geometry_exactly(qapp, shell):
    """Default -> Extra Large -> Default must land on the same layout.

    The refits are the risk here: one that grows a cap from the *current* cap
    instead of recomputing it ratchets up and never comes back down.
    """
    window, widget = shell
    apply_theme(qapp, "dark", font_scale="default")
    _settle(qapp, window, widget, (1920, 1080))
    before = (widget.minimumSizeHint().width(), _clipped_widgets(widget))

    apply_theme(qapp, "dark", font_scale="extra_large")
    _settle(qapp, window, widget, (1920, 1080))

    apply_theme(qapp, "dark", font_scale="default")
    _settle(qapp, window, widget, (1920, 1080))
    after = (widget.minimumSizeHint().width(), _clipped_widgets(widget))

    assert after == before, "the shell did not return to its Default geometry"


def test_narrow_shell_keeps_controls_scroll_reachable(qapp, shell):
    """At 1024x900 the shell is at its own minimum width — nothing is hidden.

    The page has a hard floor well above 1024 (measured ~1448 px at Default),
    which is pre-existing and not this packet's to move.  What matters here is
    that the floor grows only modestly with the tier and that the controls
    column stays inside a scroll area at every tier, so nothing becomes
    unreachable.
    """
    window, widget = shell
    widths = {}
    for scale in typo.FONT_SCALES:
        apply_theme(qapp, "dark", font_scale=scale)
        _settle(qapp, window, widget, (1024, 900))
        widths[scale] = widget.minimumSizeHint().width()

        areas = [a for a in widget.findChildren(QtWidgets.QScrollArea)
                 if a.isVisibleTo(widget)]
        assert areas, f"the controls column lost its scroll area at {scale}"
        for area in areas:
            inner = area.widget()
            if inner is None:
                continue
            reachable = (area.verticalScrollBar().maximum()
                         + area.viewport().height())
            assert reachable >= min(inner.sizeHint().height(), inner.height()), (
                f"{area.objectName()} content is past the scrollable range "
                f"at {scale}")

    assert widths["extra_small"] <= widths["default"] <= widths["extra_large"], (
        f"the shell floor is not monotonic in the tier: {widths}")
    growth = widths["extra_large"] - widths["extra_small"]
    assert growth <= 200, (
        f"Extra Small -> Extra Large widened the shell by {growth}px: "
        f"{widths}")


# ── 10. theme switching preserves the tier ─────────────────────────────

def test_theme_switch_preserves_the_selected_tier(qapp, settings, monkeypatch):
    window = _main_window(monkeypatch)
    _trigger_tier(window, "Extra Large")
    baseline = typo.application_baseline_point_size()
    expected = baseline + typo.FONT_SCALE_TOKENS["extra_large"].app_offset_pt
    assert qapp.font().pointSize() == expected

    window._set_theme("light")

    assert qapp.font().pointSize() == expected, (
        "switching theme reset the font tier to Default")
    assert typo.current_font_scale() == "extra_large"
    assert settings.value(typo.FONT_SCALE_SETTINGS_KEY) == "extra_large"
    assert settings.value("theme") == "light"

    window._set_theme("dark")
    assert qapp.font().pointSize() == expected


def test_font_change_preserves_the_selected_theme(qapp, settings, monkeypatch):
    window = _main_window(monkeypatch)
    window._set_theme("light")
    _trigger_tier(window, "Small")
    assert settings.value("theme") == "light"
    assert "#ffffff" in qapp.styleSheet() or "#f5f6fa" in qapp.styleSheet()


# ── 11. one owner: no page reads QSettings or holds a second scale ─────

_GUI_ROOT = Path(__file__).resolve().parents[2] / "src" / "xdart" / "gui"


def test_no_page_reads_qsettings_or_owns_a_second_scale():
    offenders = []
    for path in sorted(_GUI_ROOT.rglob("*.py")):
        rel = path.relative_to(_GUI_ROOT)
        if rel.parts[0] == "themes":
            continue                      # the theme layer IS the owner
        text = path.read_text(encoding="utf-8")
        if "QSettings" in text:
            offenders.append(f"{rel}: reads QSettings")
        for token in (
            "control_panel_font_size",
            "application_font_size",
            "appearance/toggle_accent",
            "appearance/spacing",
        ):
            if token in text:
                offenders.append(f"{rel}: holds the preference key {token!r}")
    assert offenders == [], (
        "pages must inherit the shared preference, not own one: %s" % offenders)


def test_the_main_window_reaches_settings_only_through_the_theme_owner():
    main = (Path(__file__).resolve().parents[2] / "src" / "xdart"
            / "_gui_main.py").read_text(encoding="utf-8")
    assert "QSettings(" not in main, (
        "_gui_main must use themes.typography.application_settings(), so a "
        "scratch XDART_SETTINGS_FILE protects the real user preferences")


def test_settings_accessor_honours_the_scratch_override(tmp_path, monkeypatch):
    target = tmp_path / "scratch.ini"
    monkeypatch.setenv("XDART_SETTINGS_FILE", str(target))
    handle = typo.application_settings()
    handle.setValue(typo.FONT_SCALE_SETTINGS_KEY, "small")
    handle.sync()
    assert target.exists()
    assert Path(handle.fileName()) == target
