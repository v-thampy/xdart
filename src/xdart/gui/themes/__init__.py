"""xdart UI themes.

Two owners, one preference.  :mod:`.typography` owns the application-wide font
scale (the five ``Font Size`` tiers, the captured platform baseline, the token
table, and the weak plot registry); :mod:`.dark` owns the palettes and the QSS
that consumes those tokens.  ``apply_theme`` is the single entry point that
applies both together, in the order Qt requires.
"""

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
    plot_font,
    resolve_font_scale,
    restyle_live_plots,
    style_plot_fonts,
)

__all__ = [
    "DARK",
    "LIGHT",
    "DARK_QSS",
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
    "live_plot_item_count",
    "restyle_live_plots",
    "style_plot_fonts",
]
