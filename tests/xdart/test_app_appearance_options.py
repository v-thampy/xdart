"""Frozen contract for selectable application appearance options.

The application owns theme, font size, selected-control accent, and spacing as
one appearance.  Pages may consume the resolved visual tokens, but they do not
read preferences or install their own stylesheet.
"""

from __future__ import annotations

import os
import re
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import QtGui, QtWidgets

import xdart.gui.themes as themes
from xdart.gui.themes import apply_theme, render_qss
from xdart.gui.themes import typography


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    typography.capture_application_baseline(app)
    original_qss = app.styleSheet()
    original_font = QtGui.QFont(app.font())
    try:
        yield app
    finally:
        app.setStyleSheet(original_qss)
        app.setFont(original_font)
        typography.restore_platform_class_fonts(
            app, typography.DEFAULT_FONT_SCALE
        )
        app.processEvents()


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("XDART_SETTINGS_FILE", str(tmp_path / "xdart.ini"))
    return typography.application_settings()


class _FakeStaticWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.h5viewer = SimpleNamespace(
            paramMenu=QtWidgets.QMenu(self),
            helpMenu=QtWidgets.QMenu(self),
        )
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
    key = PageKey("appearance-test")
    clean = CloseReceipt(PageCleanup.CLEAN, "verified")

    def build(_services, parent):
        return PageHandle(
            key=key, widget=_FakeStaticWidget(parent), close=lambda: clean)

    descriptor = PageDescriptor(
        key=key, label="Appearance Test", order=0, build=build,
        lifecycle=PageLifecycle.SWITCHABLE, capabilities=frozenset(),
    )
    return _gui_main.Main(
        page_descriptors=(descriptor,), selected_page_key=key)


def _submenu(window, title):
    for action in window.host_config_menu.actions():
        menu = action.menu()
        if menu is not None and menu.title() == title:
            return action, menu
    return None, None


def _trigger(window, menu_title, label):
    owner, menu = _submenu(window, menu_title)
    assert owner is not None and menu is not None
    for action in menu.actions():
        if action.text() == label:
            action.trigger()
            return
    raise AssertionError(f"{menu_title!r} has no {label!r} action")


def _selector_body(qss: str, selector: str) -> str:
    match = re.search(
        rf"{re.escape(selector)}\s*\{{(?P<body>.*?)\}}",
        qss,
        flags=re.DOTALL,
    )
    assert match is not None, selector
    return match.group("body")


def test_accent_and_spacing_tables_are_exact_and_total():
    accent_owner = themes.accent
    spacing_owner = themes.spacing
    assert accent_owner.ACCENT_COLOR_MENU == (
        ("theme_default", "Theme Default"),
        ("mauve_grey", "Mauve Grey"),
        ("periwinkle_light", "Periwinkle Light"),
        ("periwinkle_mid", "Periwinkle Mid"),
        ("periwinkle_muted", "Periwinkle Muted"),
    )
    assert accent_owner.ACCENT_COLORS["mauve_grey"] == "#a49bb0"
    assert spacing_owner.SPACING_MENU == (
        ("extra_tight", "Extra Tight"),
        ("tight", "Tight"),
        ("normal", "Normal"),
        ("spacious", "Spacious"),
        ("extra_spacious", "Extra Spacious"),
    )
    assert accent_owner.normalize_accent_color("unknown") == "theme_default"
    assert spacing_owner.normalize_spacing("unknown") == "normal"


def test_omitted_and_explicit_defaults_match_and_candidates_only_recolor_selection(
):
    baseline = render_qss("dark", font_scale="default")
    explicit = render_qss(
        "dark",
        font_scale="default",
        accent_color="theme_default",
        spacing="normal",
    )
    assert explicit == baseline
    assert render_qss(
        "light", accent_color="theme_default", spacing="normal"
    ) == render_qss("light")

    mauve = render_qss(
        "dark",
        font_scale="default",
        accent_color="mauve_grey",
        spacing="normal",
    )
    assert "background-color: #a49bb0;" in _selector_body(
        mauve, "QPushButton:checked"
    )
    assert "background-color: #bd93f9;" in _selector_body(
        baseline, "QPushButton:checked"
    )
    for selector in (
        "QToolButton:pressed",
        "QLineEdit#BrowsePathEdit:focus",
        "QPushButton#BrowseButton",
        "QPushButton#startButton",
        'QFrame#controlsV2SectionHeader[accent="project"]',
        "QProgressBar::chunk",
    ):
        assert _selector_body(mauve, selector) == _selector_body(
            baseline, selector
        ), f"selected-control colour leaked into {selector}"


@pytest.mark.parametrize("theme_name", ["dark", "light"])
@pytest.mark.parametrize(
    "choice, expected",
    [
        ("mauve_grey", "#a49bb0"),
        ("periwinkle_light", "#b9bee3"),
        ("periwinkle_mid", "#a7afd6"),
        ("periwinkle_muted", "#8f98b8"),
    ],
)
def test_each_candidate_reaches_every_selected_control_family(
    theme_name, choice, expected
):
    qss = render_qss(theme_name, accent_color=choice)
    for selector in (
        "QPushButton:checked",
        "QToolButton:checked",
        "QCheckBox::indicator:checked, QRadioButton::indicator:checked",
        "QPushButton#controlsV2ToggleButton:checked,\n"
        "QPushButton#controlsV2PillButton:checked",
        "QToolButton#controlsV2AutoButton:checked",
    ):
        assert f"background-color: {expected};" in _selector_body(
            qss, selector
        ), selector
    assert "color: #1a1a1a;" in _selector_body(qss, "QPushButton:checked")


def test_buttons_are_square_while_panel_cards_remain_rounded():
    qss = render_qss("dark")
    for selector in (
        "QPushButton",
        "QToolButton",
        "QPushButton#BrowseButton",
        "QPushButton#toolButton",
        "QPushButton#controlsV2ActionButton",
        "QToolButton#controlsV2BrowseButton,\nQToolButton#controlsV2MoreButton",
        "QPushButton#controlsV2ToggleButton,\nQPushButton#controlsV2PillButton",
        "QToolButton#controlsV2AutoButton",
    ):
        assert "border-radius: 0px;" in _selector_body(qss, selector), selector

    assert "border-radius: 7px;" in _selector_body(
        qss, "QFrame#controlsV2SubsectionCard"
    )
    assert "border-top-left-radius: 7px;" in _selector_body(
        qss, "QFrame#controlsV2SectionHeader"
    )


def test_spacing_changes_real_qss_padding_without_changing_font_or_color():
    spacing_owner = themes.spacing
    ordered = [
        spacing_owner.spacing_tokens(name)
        for name, _label in spacing_owner.SPACING_MENU
    ]
    for attribute in (
        "button_y",
        "button_x",
        "layout_gap",
        "panel_margin",
        "browser_gap",
        "tools_gap",
        "tools_vertical_margin",
    ):
        values = [getattr(tokens, attribute) for tokens in ordered]
        assert values == sorted(values)
        assert len(set(values)) == 5

    tight = render_qss("dark", spacing="extra_tight")
    roomy = render_qss("dark", spacing="extra_spacious")
    tight_button = _selector_body(tight, "QPushButton")
    roomy_button = _selector_body(roomy, "QPushButton")
    assert "padding: 1px 6px;" in tight_button
    assert "padding: 8px 18px;" in roomy_button
    assert tight_button.replace("1px 6px", "8px 18px") == roomy_button
    compact_selector = (
        "QPushButton#e3BrowserCompactButton,\n"
        "QToolButton#e3RefreshBrowser"
    )
    assert "padding: 1px 2px;" in _selector_body(tight, compact_selector)
    assert "padding: 5px 8px;" in _selector_body(roomy, compact_selector)
    assert spacing_owner.spacing_tokens("extra_tight").layout_gap < (
        spacing_owner.spacing_tokens("extra_spacious").layout_gap
    )


def test_apply_theme_publishes_the_exact_spacing_for_responsive_layouts(qapp):
    try:
        apply_theme(qapp, "dark", spacing="spacious")
        assert themes.spacing.current_spacing() == "spacious"
    finally:
        apply_theme(qapp, "dark", spacing="normal")


def test_controls_cards_follow_all_five_spacing_tiers_live(qapp):
    from xdart.gui.tabs.static_scan.ui.controls_panel_v2 import (
        ControlsPanelV2,
        SubsectionCard,
    )

    panel = ControlsPanelV2()
    subsection = SubsectionCard("Example")
    panel.show()
    subsection.show()
    qapp.processEvents()
    try:
        observed = []
        for name, _label in themes.spacing.SPACING_MENU:
            apply_theme(qapp, "dark", spacing=name)
            qapp.processEvents()
            observed.append(
                (
                    panel.layout().spacing(),
                    panel.layout().contentsMargins().left(),
                    panel.project_card.body_layout.spacing(),
                    panel.project_card.body_layout.contentsMargins().top(),
                    subsection.body_layout.spacing(),
                )
            )
        for values in zip(*observed):
            assert tuple(values) == tuple(sorted(values))
            assert len(set(values)) == 5
        assert observed[2] == (12, 5, 5, 7, 4)
    finally:
        apply_theme(qapp, "dark", spacing="normal")
        panel.close()
        subsection.close()
        panel.deleteLater()
        subsection.deleteLater()
        qapp.processEvents()


def test_config_menus_are_exclusive_persisted_and_apply_one_complete_appearance(
    qapp, settings, monkeypatch
):
    from xdart import _gui_main
    import xdart.gui.themes as themes

    calls = []
    real_apply = themes.apply_theme

    def record(app, name="dark", **kwargs):
        calls.append((name, kwargs))
        return real_apply(app, name, **kwargs)

    monkeypatch.setattr(themes, "apply_theme", record)
    window = _main_window(monkeypatch)
    try:
        for title, labels in (
            ("Accent Color", [
                label for _key, label in themes.accent.ACCENT_COLOR_MENU
            ]),
            ("Spacing", [
                label for _key, label in themes.spacing.SPACING_MENU
            ]),
        ):
            owner, menu = _submenu(window, title)
            assert owner is not None and menu is not None
            assert [action.text() for action in menu.actions()] == labels
            assert all(action.isCheckable() for action in menu.actions())
            assert len({action.actionGroup() for action in menu.actions()}) == 1

        _trigger(window, "Accent Color", "Mauve Grey")
        assert calls[-1] == (
            "dark",
            {
                "font_scale": "default",
                "accent_color": "mauve_grey",
                "spacing": "normal",
            },
        )
        _trigger(window, "Spacing", "Spacious")
        assert calls[-1] == (
            "dark",
            {
                "font_scale": "default",
                "accent_color": "mauve_grey",
                "spacing": "spacious",
            },
        )
        window._set_theme("light")
        window._set_application_font_size("large")
        assert calls[-1] == (
            "light",
            {
                "font_scale": "large",
                "accent_color": "mauve_grey",
                "spacing": "spacious",
            },
        )
        assert settings.value(themes.accent.ACCENT_COLOR_SETTINGS_KEY) == (
            "mauve_grey"
        )
        assert settings.value(themes.spacing.SPACING_SETTINGS_KEY) == "spacious"
        assert settings.value("theme") == "light"
        assert settings.value(typography.FONT_SCALE_SETTINGS_KEY) == "large"
    finally:
        window.close()
        qapp.processEvents()


def test_malformed_saved_options_fail_closed_to_defaults(settings):
    accent_owner = themes.accent
    spacing_owner = themes.spacing
    settings.setValue(accent_owner.ACCENT_COLOR_SETTINGS_KEY, "purple-ish")
    settings.setValue(spacing_owner.SPACING_SETTINGS_KEY, -4)
    settings.sync()
    assert accent_owner.resolve_accent_color(settings) == "theme_default"
    assert spacing_owner.resolve_spacing(settings) == "normal"
