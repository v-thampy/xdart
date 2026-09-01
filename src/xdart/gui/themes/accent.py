"""Values-only owner for the selected-control accent preference.

The ordinary theme accent still owns focus rings, progress, tabs, and tool
hover.  This preference deliberately affects selected toggles only.
"""

from __future__ import annotations


ACCENT_COLOR_SETTINGS_KEY = "appearance/toggle_accent"
DEFAULT_ACCENT_COLOR = "periwinkle_muted"

ACCENT_COLOR_MENU = (
    ("theme_default", "Theme Default"),
    ("mauve_grey", "Mauve Grey"),
    ("periwinkle_light", "Periwinkle Light"),
    ("periwinkle_mid", "Periwinkle Mid"),
    ("periwinkle_muted", "Periwinkle Muted"),
)

ACCENT_COLORS = {
    "mauve_grey": "#a49bb0",
    "periwinkle_light": "#b9bee3",
    "periwinkle_mid": "#a7afd6",
    "periwinkle_muted": "#8f98b8",
}

ACCENT_COLOR_NAMES = tuple(key for key, _label in ACCENT_COLOR_MENU)


def normalize_accent_color(value) -> str:
    """Return an exact known choice; malformed settings use the app default."""
    return value if type(value) is str and value in ACCENT_COLOR_NAMES else (
        DEFAULT_ACCENT_COLOR
    )


def resolve_accent_color(settings) -> str:
    return normalize_accent_color(
        settings.value(ACCENT_COLOR_SETTINGS_KEY, DEFAULT_ACCENT_COLOR)
    )


def selected_color(name: str, *, theme_default: str) -> str:
    normalized = normalize_accent_color(name)
    if normalized == "theme_default":
        return theme_default
    return ACCENT_COLORS[normalized]


__all__ = [
    "ACCENT_COLORS",
    "ACCENT_COLOR_MENU",
    "ACCENT_COLOR_NAMES",
    "ACCENT_COLOR_SETTINGS_KEY",
    "DEFAULT_ACCENT_COLOR",
    "normalize_accent_color",
    "resolve_accent_color",
    "selected_color",
]
