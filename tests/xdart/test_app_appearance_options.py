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
    def __init__(self):
        super().__init__()
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

    monkeypatch.setattr(
        _gui_main.tabs.static_scan, "staticWidget", _FakeStaticWidget
    )
    return _gui_main.Main()


def _submenu(window, title):
    for action in window.main_widget.h5viewer.paramMenu.actions():
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


def test_theme_default_is_byte_compatible_and_candidates_only_recolor_selection():
    baseline = render_qss("dark", font_scale="default")
    explicit = render_qss(
        "dark",
        font_scale="default",
        accent_color="theme_default",
        spacing="normal",
    )
    assert explicit == baseline

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
    assert _selector_body(mauve, "QLineEdit:focus") == _selector_body(
        baseline, "QLineEdit:focus"
    ), "the selected-control picker must not recolor focus/policy accents"


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
    tight = render_qss("dark", spacing="extra_tight")
    roomy = render_qss("dark", spacing="extra_spacious")
    tight_button = _selector_body(tight, "QPushButton")
    roomy_button = _selector_body(roomy, "QPushButton")
    assert "padding: 1px 6px;" in tight_button
    assert "padding: 8px 18px;" in roomy_button
    assert tight_button.replace("1px 6px", "8px 18px") == roomy_button
    assert spacing_owner.spacing_tokens("extra_tight").layout_gap < (
        spacing_owner.spacing_tokens("extra_spacious").layout_gap
    )


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
