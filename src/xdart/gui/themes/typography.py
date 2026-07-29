"""The single owner of xdart's application-wide font scale.

Config ▸ **Font Size** is ONE preference with five tiers:

    extra_small · small · default · large · extra_large

and it applies to the entire application — ordinary Qt widgets, dialogs, the
theme's dense Controls QSS tokens, and pyqtgraph plot text.  Pages inherit it.
A page that reads ``QSettings``, keeps its own scale value, or multiplies its
current font has forked the preference and is a bug.

Three rules make the contract hold:

1. **One captured baseline.**  ``capture_application_baseline`` snapshots the
   platform's ``QApplication.font()`` ONCE, before any tier is applied.  Every
   tier is ``baseline ± n`` points.  Deriving from the *current* font would
   compound on each change, so ``Default → Large → Default`` would drift.
2. **One immutable token table.**  ``FONT_SCALE_TOKENS`` is the whole numeric
   contract.  The dense Controls tokens and the plot font are absolute values
   per tier, not a second setting — they are density variants of the same
   choice (the ``default`` row reproduces the superseded preset byte for byte).
3. **Order matters when applying.**  ``QApplication.setFont()`` alone does NOT
   reach widgets while an application stylesheet is installed — Qt's stylesheet
   style caches the resolved font and only re-resolves on a re-polish.
   ``apply_theme`` therefore always sets the font and THEN sets the stylesheet
   (re-setting an identical stylesheet still re-polishes, so idempotence is
   safe).  Existing pyqtgraph plots follow neither, so they are restyled
   explicitly through a weak registry.

Settings live behind :func:`application_settings`, which honours
``XDART_SETTINGS_FILE`` exactly the way session state honours
``XDART_SESSION_FILE``.  Tests and probes point it at scratch space so
automated work can never read or write the maintainer's real preferences.
"""

from __future__ import annotations

import logging
import os
import weakref
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ── the contract ─────────────────────────────────────────────────────────

#: Every tier, smallest first.  This tuple IS the contract; nothing else may
#: define a scale value.
FONT_SCALES = ("extra_small", "small", "default", "large", "extra_large")

DEFAULT_FONT_SCALE = "default"

#: ``(key, visible label)`` in menu order.
FONT_SCALE_MENU = (
    ("extra_small", "Extra Small"),
    ("small", "Small"),
    ("default", "Default"),
    ("large", "Large"),
    ("extra_large", "Extra Large"),
)

#: The application preference.
FONT_SCALE_SETTINGS_KEY = "application_font_size"

#: The superseded Controls-panel-only preference, read once for migration.
LEGACY_FONT_SCALE_SETTINGS_KEY = "control_panel_font_size"

#: Legacy value → tier.  The three old values keep their exact appearance for
#: the Controls panel and simply gain application-wide reach.
LEGACY_FONT_SCALE_MAP = {
    "small": "small",
    "default": "default",
    "large": "large",
}


@dataclass(frozen=True)
class FontScaleTokens:
    """The complete numeric definition of one tier.

    ``app_offset_pt``
        Points added to the captured platform baseline for
        ``QApplication.setFont`` — this drives every ordinary widget, dialog
        and menu.
    ``control_px`` / ``browse_px``
        Absolute pixel sizes for the dense Controls QSS tokens.  Kept in **px**
        because that is what the superseded preset used, so ``default`` renders
        byte-identically to the shipped appearance.
    ``plot_pt``
        Point size for pyqtgraph tick labels, axis labels and legends.
    ``title_pt`` / ``action_pt`` / ``calib_pt``
        Point sizes for the three legacy generated-UI fonts that Qt Designer
        hard-set on individual widgets (the display top-bar title, the
        integrator action buttons, and the Windows-only calibration buttons).
        Kept in **pt** because the generated code used ``setPointSize``, so
        ``default`` again reproduces today's rendering on every platform.
        A QSS ``font-size`` outranks a widget's own ``setFont``, which is how
        these follow the tier without touching generated files.
    """

    app_offset_pt: int
    control_px: int
    browse_px: int
    plot_pt: int
    title_pt: int
    action_pt: int
    calib_pt: float


#: THE token table.  Deterministic ``-2 … +2`` steps from each stable absolute
#: baseline; the ``default`` row is exactly what shipped before this preference
#: became application-wide.
FONT_SCALE_TOKENS = {
    "extra_small": FontScaleTokens(-2, 10, 11, 9, 13, 12, 6.5),
    "small":       FontScaleTokens(-1, 11, 12, 10, 14, 13, 7.5),
    "default":     FontScaleTokens(0, 12, 13, 11, 15, 14, 8.5),
    "large":       FontScaleTokens(+1, 13, 14, 12, 16, 15, 9.5),
    "extra_large": FontScaleTokens(+2, 14, 15, 13, 17, 16, 10.5),
}


def normalize_font_scale(value) -> str:
    """Map anything to an exact tier; unknown/malformed becomes ``default``.

    Deliberately strict: only the exact lowercase keys are accepted, so a
    truncated write, a stale value, or a hand-edited settings file degrades to
    a usable application instead of a half-applied scale.
    """
    if isinstance(value, str) and value in FONT_SCALES:
        return value
    return DEFAULT_FONT_SCALE


def migrate_legacy_font_scale(value):
    """Tier for a superseded ``control_panel_font_size`` value, else ``None``."""
    if isinstance(value, str):
        return LEGACY_FONT_SCALE_MAP.get(value)
    return None


def resolve_font_scale(settings) -> str:
    """The saved tier: new key first, then the legacy key, then ``default``."""
    try:
        if settings.contains(FONT_SCALE_SETTINGS_KEY):
            return normalize_font_scale(settings.value(FONT_SCALE_SETTINGS_KEY))
        legacy = migrate_legacy_font_scale(
            settings.value(LEGACY_FONT_SCALE_SETTINGS_KEY))
        if legacy is not None:
            return legacy
    except Exception:                        # pragma: no cover - defensive
        logger.debug("could not read the font scale preference", exc_info=True)
    return DEFAULT_FONT_SCALE


# ── settings access ──────────────────────────────────────────────────────

#: Env override for the preferences file.  Mirrors ``XDART_SESSION_FILE``:
#: automated runs point it at scratch space so they can never read or write
#: the maintainer's real preferences.
SETTINGS_FILE_ENV = "XDART_SETTINGS_FILE"


def application_settings():
    """The one handle on xdart's application preferences.

    Every reader and writer in the GUI goes through here, so a single env var
    redirects the whole application to a scratch file.
    """
    from pyqtgraph.Qt import QtCore
    override = os.environ.get(SETTINGS_FILE_ENV)
    if override:
        return QtCore.QSettings(override, QtCore.QSettings.Format.IniFormat)
    return QtCore.QSettings("xdart", "xdart")


# ── the captured platform baseline ───────────────────────────────────────

_BASELINE_FONT = None
_BASELINE_POINT_SIZE = 0
_CURRENT_SCALE = DEFAULT_FONT_SCALE


def capture_application_baseline(app) -> int:
    """Snapshot the platform's default application font, once.

    Idempotent, and deliberately never refreshed: after the first tier is
    applied ``app.font()`` is a *scaled* font, and re-capturing it would make
    every later tier relative to the last one.
    """
    global _BASELINE_FONT, _BASELINE_POINT_SIZE
    if _BASELINE_FONT is not None:
        return _BASELINE_POINT_SIZE
    from pyqtgraph.Qt import QtGui
    font = QtGui.QFont(app.font())
    size = font.pointSize()
    if size <= 0:
        # Platforms that express the default font in pixels: resolve the
        # equivalent point size so the offsets stay meaningful.
        size = QtGui.QFontInfo(font).pointSize()
    _BASELINE_FONT = font
    _BASELINE_POINT_SIZE = int(size)
    return _BASELINE_POINT_SIZE


def application_baseline_point_size():
    """Point size of the captured baseline (``None`` before capture)."""
    return _BASELINE_POINT_SIZE if _BASELINE_FONT is not None else None


def current_font_scale() -> str:
    """The tier most recently applied to this process."""
    return _CURRENT_SCALE


def scaled_application_font(app, scale=DEFAULT_FONT_SCALE):
    """The baseline font shifted by ``scale``'s offset.

    Copies the whole baseline QFont (family, weight, style) and changes only
    the point size, so the platform's font choice survives every tier.
    """
    from pyqtgraph.Qt import QtGui
    scale = normalize_font_scale(scale)
    baseline_pt = capture_application_baseline(app)
    font = QtGui.QFont(_BASELINE_FONT)
    font.setPointSize(baseline_pt + FONT_SCALE_TOKENS[scale].app_offset_pt)
    return font


def apply_application_font(app, scale=DEFAULT_FONT_SCALE) -> str:
    """Set the application font for ``scale`` and record the tier.

    The caller MUST re-apply the stylesheet afterwards: with an application
    stylesheet installed, Qt does not re-resolve widget fonts on an
    application-font change alone.  :func:`xdart.gui.themes.apply_theme` owns
    that order.
    """
    global _CURRENT_SCALE
    scale = normalize_font_scale(scale)
    app.setFont(scaled_application_font(app, scale))
    _CURRENT_SCALE = scale
    return scale


def qss_font_tokens(scale=DEFAULT_FONT_SCALE) -> dict:
    """The ``$token`` values the theme template substitutes for ``scale``."""
    tokens = FONT_SCALE_TOKENS[normalize_font_scale(scale)]
    return {
        "control_panel_font": f"{tokens.control_px}px",
        "control_panel_status_font": f"{tokens.control_px}px",
        "control_panel_tick_font": f"{tokens.control_px}px",
        "control_panel_browse_font": f"{tokens.browse_px}px",
        "control_panel_run_font": f"{tokens.browse_px}px",
        "display_title_font": f"{tokens.title_pt}pt",
        "legacy_action_font": f"{tokens.action_pt}pt",
    }


def calibration_button_font(scale=DEFAULT_FONT_SCALE) -> str:
    """Windows-only QSS size for the two narrow calibration buttons."""
    return f"{FONT_SCALE_TOKENS[normalize_font_scale(scale)].calib_pt}pt"


# ── pyqtgraph: one tier for new AND existing plots ───────────────────────

#: Weak by construction: a closed plot must not be kept alive by the fact that
#: it once wanted a font.  ``WeakSet`` drops entries as soon as the PlotItem is
#: collected, so a scale change never touches a dead QObject.
_PLOT_REGISTRY: "weakref.WeakSet" = weakref.WeakSet()


def plot_font(scale=None):
    """Point-sized QFont for pyqtgraph ticks, axis labels and legends."""
    from pyqtgraph.Qt import QtGui
    scale = normalize_font_scale(scale if scale is not None else _CURRENT_SCALE)
    font = QtGui.QFont(_BASELINE_FONT) if _BASELINE_FONT is not None \
        else QtGui.QFont()
    font.setPointSize(FONT_SCALE_TOKENS[scale].plot_pt)
    return font


def register_plot(plot) -> None:
    """Track ``plot`` weakly so a later tier change restyles it."""
    try:
        _PLOT_REGISTRY.add(plot)
    except TypeError:                        # pragma: no cover - defensive
        logger.debug("plot is not weak-referenceable; not registered")


def registered_plot_count() -> int:
    """How many live plots the registry currently tracks."""
    return len(_PLOT_REGISTRY)


def style_plot_fonts(plot, scale=None) -> None:
    """Apply the tier's font to one PlotItem's ticks, labels and legend.

    Registers the plot as a side effect, so every plot that is styled once
    keeps following the preference.  Idempotent, and every step is best-effort:
    a plot that lacks an axis or a legend must not break a settings change.
    """
    font = plot_font(scale)
    register_plot(plot)
    for axis_name in ("bottom", "left", "top", "right"):
        try:
            axis = plot.getAxis(axis_name)
        except Exception:
            continue
        if axis is None:
            continue
        try:
            axis.setStyle(tickFont=font)
        except Exception:
            pass
        label = getattr(axis, "label", None)
        if label is not None:
            try:
                label.setFont(font)
            except Exception:
                pass
    # pyqtgraph hard-defaults every legend to 9pt regardless of the
    # application font, so the legend needs the tier explicitly.
    legend = getattr(plot, "legend", None)
    if legend is not None:
        try:
            legend.setLabelTextSize(f"{font.pointSize()}pt")
        except Exception:
            pass


def restyle_registered_plots(scale=None) -> int:
    """Re-apply the tier to every live registered plot; returns how many.

    Iterates a snapshot: ``style_plot_fonts`` re-adds to the same WeakSet, and
    a collection during iteration would otherwise mutate it mid-walk.
    """
    plots = list(_PLOT_REGISTRY)
    for plot in plots:
        try:
            style_plot_fonts(plot, scale)
        except Exception:
            logger.debug("could not restyle a registered plot", exc_info=True)
    return len(plots)
