"""xdart UI themes.

The small values modules own font, selected-control accent, spacing, and
Controls-card corner choices; :mod:`.dark` owns the palettes and QSS that
consume them.
``apply_theme`` remains the only live appearance entry point.
"""

from . import accent, corners, spacing
from .accent import (
    ACCENT_COLOR_MENU,
    ACCENT_COLOR_SETTINGS_KEY,
    DEFAULT_ACCENT_COLOR,
    normalize_accent_color,
    resolve_accent_color,
)

from .dark import (
    DARK,
    DARK_QSS,
    LIGHT,
    apply_dark_theme,
    apply_seaborn_plot_style,
    apply_theme,
    render_qss,
)
from .typography import (
    DEFAULT_FONT_SCALE,
    FONT_SCALE_MENU,
    FONT_SCALE_SETTINGS_KEY,
    FONT_SCALE_TOKENS,
    FONT_SCALES,
    LEGACY_FONT_SCALE_SETTINGS_KEY,
    application_baseline_point_size,
    application_settings,
    capture_application_baseline,
    current_font_scale,
    live_plot_item_count,
    normalize_font_scale,
    platform_class_font_baseline,
    plot_font,
    resolve_font_scale,
    restore_platform_class_fonts,
    restyle_live_plots,
    style_plot_fonts,
)
from .spacing import (
    DEFAULT_SPACING,
    SPACING_MENU,
    SPACING_SETTINGS_KEY,
    current_spacing,
    current_spacing_tokens,
    normalize_spacing,
    resolve_spacing,
    spacing_tokens,
)
from .corners import (
    CONTROLS_CARD_CORNERS_SETTINGS_KEY,
    DEFAULT_CONTROLS_CARD_CORNERS,
    controls_card_radius,
    current_controls_card_corners,
    normalize_controls_card_corners,
    resolve_controls_card_corners,
    set_current_controls_card_corners,
)

__all__ = [
    "DARK",
    "LIGHT",
    "DARK_QSS",
    "accent",
    "corners",
    "spacing",
    "apply_dark_theme",
    "apply_theme",
    "render_qss",
    "apply_seaborn_plot_style",
    # Application-wide font scale (see .typography for the contract).
    "FONT_SCALES",
    "FONT_SCALE_MENU",
    "FONT_SCALE_TOKENS",
    "DEFAULT_FONT_SCALE",
    "FONT_SCALE_SETTINGS_KEY",
    "LEGACY_FONT_SCALE_SETTINGS_KEY",
    "application_settings",
    "application_baseline_point_size",
    "capture_application_baseline",
    "current_font_scale",
    "normalize_font_scale",
    "resolve_font_scale",
    "plot_font",
    "platform_class_font_baseline",
    "restore_platform_class_fonts",
    "live_plot_item_count",
    "restyle_live_plots",
    "style_plot_fonts",
    "ACCENT_COLOR_MENU",
    "ACCENT_COLOR_SETTINGS_KEY",
    "DEFAULT_ACCENT_COLOR",
    "normalize_accent_color",
    "resolve_accent_color",
    "DEFAULT_SPACING",
    "SPACING_MENU",
    "SPACING_SETTINGS_KEY",
    "current_spacing",
    "current_spacing_tokens",
    "normalize_spacing",
    "resolve_spacing",
    "spacing_tokens",
    "CONTROLS_CARD_CORNERS_SETTINGS_KEY",
    "DEFAULT_CONTROLS_CARD_CORNERS",
    "controls_card_radius",
    "current_controls_card_corners",
    "normalize_controls_card_corners",
    "resolve_controls_card_corners",
    "set_current_controls_card_corners",
]
