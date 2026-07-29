"""One five-tier spacing preference shared by QSS and responsive layouts."""

from __future__ import annotations

from dataclasses import dataclass


SPACING_SETTINGS_KEY = "appearance/spacing"
DEFAULT_SPACING = "normal"

SPACING_MENU = (
    ("extra_tight", "Extra Tight"),
    ("tight", "Tight"),
    ("normal", "Normal"),
    ("spacious", "Spacious"),
    ("extra_spacious", "Extra Spacious"),
)
SPACING_NAMES = tuple(key for key, _label in SPACING_MENU)


@dataclass(frozen=True, slots=True)
class SpacingTokens:
    button_y: int
    button_x: int
    tool_y: int
    tool_x: int
    field_y: int
    field_x: int
    control_field_y: int
    control_field_x: int
    control_action_y: int
    control_action_x: int
    toggle_y: int
    toggle_x: int
    pill_y: int
    pill_x: int
    auto_y: int
    auto_x: int
    browse_button_y: int
    browse_button_x: int
    control_browse_y: int
    control_browse_x: int
    compact_action_y: int
    compact_action_x: int
    layout_gap: int
    panel_margin: int
    browser_gap: int
    tools_gap: int
    tools_vertical_margin: int

    def qss_tokens(self) -> dict[str, str]:
        return {
            "button_padding": f"{self.button_y}px {self.button_x}px",
            "tool_button_padding": f"{self.tool_y}px {self.tool_x}px",
            "field_padding": f"{self.field_y}px {self.field_x}px",
            "control_field_padding": (
                f"{self.control_field_y}px {self.control_field_x}px"
            ),
            "control_action_padding": (
                f"{self.control_action_y}px {self.control_action_x}px"
            ),
            "toggle_padding": f"{self.toggle_y}px {self.toggle_x}px",
            "pill_padding": f"{self.pill_y}px {self.pill_x}px",
            "auto_padding": f"{self.auto_y}px {self.auto_x}px",
            "browse_button_padding": (
                f"{self.browse_button_y}px {self.browse_button_x}px"
            ),
            "control_browse_padding": (
                f"{self.control_browse_y}px {self.control_browse_x}px"
            ),
            "compact_action_padding": (
                f"{self.compact_action_y}px {self.compact_action_x}px"
            ),
        }


SPACING_TOKENS = {
    "extra_tight": SpacingTokens(
        1, 6, 1, 5, 1, 2, 1, 4, 3, 6, 2, 5, 2, 7, 2, 4,
        0, 4, 1, 2, 0, 6,
        3, 4, 4, 4, 6,
    ),
    "tight": SpacingTokens(
        3, 9, 2, 7, 1, 3, 2, 5, 4, 8, 3, 6, 3, 10, 3, 5,
        1, 5, 1, 3, 1, 8,
        5, 6, 6, 6, 9,
    ),
    "normal": SpacingTokens(
        4, 12, 3, 9, 2, 4, 3, 7, 6, 10, 5, 8, 4, 13, 4, 7,
        1, 6, 2, 4, 1, 10,
        8, 8, 8, 8, 12,
    ),
    "spacious": SpacingTokens(
        6, 15, 5, 12, 4, 6, 5, 9, 8, 13, 7, 11, 6, 16, 6, 10,
        3, 9, 3, 6, 3, 12,
        11, 12, 11, 11, 15,
    ),
    "extra_spacious": SpacingTokens(
        8, 18, 7, 15, 6, 8, 7, 11, 10, 16, 9, 14, 8, 19, 8, 13,
        5, 12, 5, 8, 5, 15,
        14, 16, 14, 14, 18,
    ),
}

_current_spacing = DEFAULT_SPACING


def normalize_spacing(value) -> str:
    return value if type(value) is str and value in SPACING_NAMES else (
        DEFAULT_SPACING
    )


def resolve_spacing(settings) -> str:
    return normalize_spacing(
        settings.value(SPACING_SETTINGS_KEY, DEFAULT_SPACING)
    )


def spacing_tokens(name: str = DEFAULT_SPACING) -> SpacingTokens:
    return SPACING_TOKENS[normalize_spacing(name)]


def set_current_spacing(name: str) -> str:
    global _current_spacing
    _current_spacing = normalize_spacing(name)
    return _current_spacing


def current_spacing() -> str:
    return _current_spacing


def current_spacing_tokens() -> SpacingTokens:
    return spacing_tokens(_current_spacing)


__all__ = [
    "DEFAULT_SPACING",
    "SPACING_MENU",
    "SPACING_NAMES",
    "SPACING_SETTINGS_KEY",
    "SPACING_TOKENS",
    "SpacingTokens",
    "current_spacing",
    "current_spacing_tokens",
    "normalize_spacing",
    "resolve_spacing",
    "set_current_spacing",
    "spacing_tokens",
]
