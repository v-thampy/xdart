"""Values-only owner for the live panel/container corner preference.

The preference deliberately excludes buttons, fields, indicators, chips,
scrollbars, and progress bars.  It owns only the remaining visual containers
which otherwise read as rounded cards or bounding panels.
"""

from __future__ import annotations


CONTROLS_CARD_CORNERS_SETTINGS_KEY = "appearance/controls_card_corners"
DEFAULT_CONTROLS_CARD_CORNERS = True

_current_controls_card_corners = DEFAULT_CONTROLS_CARD_CORNERS


def normalize_controls_card_corners(value) -> bool:
    """Resolve QSettings/native values to one explicit on/off choice."""
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    if type(value) is str:
        normalized = value.strip().lower()
        if normalized in {"1", "true", "on", "yes", "rounded"}:
            return True
        if normalized in {"0", "false", "off", "no", "square"}:
            return False
    return DEFAULT_CONTROLS_CARD_CORNERS


def resolve_controls_card_corners(settings) -> bool:
    return normalize_controls_card_corners(
        settings.value(
            CONTROLS_CARD_CORNERS_SETTINGS_KEY,
            DEFAULT_CONTROLS_CARD_CORNERS,
        )
    )


def set_current_controls_card_corners(value) -> bool:
    global _current_controls_card_corners
    _current_controls_card_corners = normalize_controls_card_corners(value)
    return _current_controls_card_corners


def current_controls_card_corners() -> bool:
    return _current_controls_card_corners


def rounded_container_radius(
    value=DEFAULT_CONTROLS_CARD_CORNERS,
) -> str:
    return "7px" if normalize_controls_card_corners(value) else "0px"


# Compatibility name for the persisted setting and callers predating the
# broader label.  There remains one boolean owner and one stored key.
controls_card_radius = rounded_container_radius


__all__ = [
    "CONTROLS_CARD_CORNERS_SETTINGS_KEY",
    "DEFAULT_CONTROLS_CARD_CORNERS",
    "controls_card_radius",
    "rounded_container_radius",
    "current_controls_card_corners",
    "normalize_controls_card_corners",
    "resolve_controls_card_corners",
    "set_current_controls_card_corners",
]
