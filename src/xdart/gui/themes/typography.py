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
   safe).  Existing pyqtgraph text follows neither, so it is restyled
   explicitly by walking the live graphics scenes — which retains nothing at
   all, so a closed plot is simply absent from the next pass.

Settings live behind :func:`application_settings`, which honours
``XDART_SETTINGS_FILE`` exactly the way session state honours
``XDART_SESSION_FILE``.  Tests and probes point it at scratch space so
automated work can never read or write the maintainer's real preferences.
"""

from __future__ import annotations

import logging
import os
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
    ``title_pt`` / ``action_pt`` / ``pill_pt`` / ``calib_pt``
        Point sizes for the legacy generated-UI fonts that Qt Designer hard-set
        on individual widgets (the display top-bar title at 15 pt, the
        integrator action buttons at 14 pt, the 1-D/2-D pills at 11 pt, and the
        Windows-only calibration buttons at 8.5 pt).
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
    pill_pt: int
    calib_pt: float


#: THE token table.  Deterministic ``-2 … +2`` steps from each stable absolute
#: baseline; the ``default`` row is exactly what shipped before this preference
#: became application-wide.
FONT_SCALE_TOKENS = {
    "extra_small": FontScaleTokens(-2, 10, 11, 9, 13, 12, 9, 6.5),
    "small":       FontScaleTokens(-1, 11, 12, 10, 14, 13, 10, 7.5),
    "default":     FontScaleTokens(0, 12, 13, 11, 15, 14, 11, 8.5),
    "large":       FontScaleTokens(+1, 13, 14, 12, 16, 15, 12, 9.5),
    "extra_large": FontScaleTokens(+2, 14, 15, 13, 17, 16, 13, 10.5),
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


def font_scale_ratio(scale=None) -> float:
    """Current tier's point size divided by the Default tier's.

    For the handful of layout budgets that are genuinely a pixel allowance
    ("never let this combo grow past ~130 px") tuned at the Default tier.
    Multiplying such a literal by this ratio keeps it byte-identical at Default
    on every platform while letting it breathe at the larger tiers — which is
    what stops a metric-derived width from being truncated by a budget that
    was not derived from metrics.

    A caller using this is inheriting the one preference, not owning a second
    one: it holds no value and reads no settings.
    """
    scale = normalize_font_scale(scale if scale is not None else _CURRENT_SCALE)
    if _BASELINE_POINT_SIZE <= 0:
        return 1.0
    return ((_BASELINE_POINT_SIZE + FONT_SCALE_TOKENS[scale].app_offset_pt)
            / float(_BASELINE_POINT_SIZE))


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
        "legacy_pill_font": f"{tokens.pill_pt}pt",
    }


def calibration_button_font(scale=DEFAULT_FONT_SCALE) -> str:
    """Windows-only QSS size for the two narrow calibration buttons."""
    return f"{FONT_SCALE_TOKENS[normalize_font_scale(scale)].calib_pt}pt"


# ── pyqtgraph: one tier for new AND existing plots ───────────────────────

def plot_font(scale=None):
    """Point-sized QFont for pyqtgraph ticks, axis labels and legends."""
    from pyqtgraph.Qt import QtGui
    scale = normalize_font_scale(scale if scale is not None else _CURRENT_SCALE)
    font = QtGui.QFont(_BASELINE_FONT) if _BASELINE_FONT is not None \
        else QtGui.QFont()
    font.setPointSize(FONT_SCALE_TOKENS[scale].plot_pt)
    return font


def _restyle_axis(axis, font) -> None:
    """Tick font + axis-label font, then invalidate the cached tick picture.

    Without the invalidation the axis keeps its cached ``QPicture`` and the
    tick-text space it reserved at the old size, so the labels redraw but the
    geometry does not — the trick ``display_frame_widget._set_raw_pixel_axes``
    already uses.
    """
    try:
        axis.setStyle(tickFont=font)
    except Exception:
        return
    label = getattr(axis, "label", None)
    if label is not None:
        try:
            label.setFont(font)
        except Exception:
            pass
    try:
        axis.picture = None
        axis.update()
    except Exception:
        pass


def _restyle_legend(legend, font) -> None:
    """``LegendItem.setFont`` has no visual effect — entry labels carry inline
    ``font-size`` CSS from ``opts['labelTextSize']``, which pyqtgraph hard-
    defaults to 9pt regardless of the application font.  This is the only
    setter that both restyles existing entries and is inherited by later ones.
    """
    try:
        legend.setLabelTextSize(f"{font.pointSize()}pt")
    except Exception:
        pass


def _restyle_label_item(item, font) -> None:
    """``LabelItem.setFont`` is likewise inert; re-setting the text with an
    explicit ``size`` is the working recipe, and ``opts['size']`` persists so
    later ``setText`` calls (hover readouts) keep the tier."""
    try:
        if item.text is None:
            return
        item.setText(item.text, size=f"{font.pointSize()}pt")
    except Exception:
        pass


def _live_graphics_scenes():
    """Every live pyqtgraph scene, found through Qt — nothing is retained.

    ``QApplication.allWidgets()`` is deliberate: every plot host in xdart
    (``PlotWidget``, ``GraphicsLayoutWidget``, ``HistogramLUTWidget``)
    subclasses ``pyqtgraph.GraphicsView``, and ``allWidgets`` includes the
    display frame's ``setParent(None)``-detached waterfall/1-D views, which a
    walk down the visible widget tree would miss.  Holding no reference at all
    is strictly stronger than holding weak ones: a closed plot is simply not
    found on the next pass.
    """
    try:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtWidgets
    except Exception:                        # pragma: no cover - Qt missing
        return []
    app = QtWidgets.QApplication.instance()
    if app is None:
        return []
    scenes = []
    seen = set()
    for widget in app.allWidgets():
        if not isinstance(widget, pg.GraphicsView):
            continue
        try:
            scene = widget.scene()
        except Exception:
            continue
        if scene is None or id(scene) in seen:
            continue
        seen.add(id(scene))
        scenes.append(scene)
    return scenes


def restyle_live_plots(scale=None) -> int:
    """Re-apply the tier to every plot that already exists; returns items hit.

    Existing pyqtgraph text follows neither ``QApplication.setFont`` nor the
    stylesheet: ``AxisItem.label`` and ``LabelItem`` are ``QGraphicsTextItem``s
    pinned at their construction-time font, tick fonts are pinned by
    :func:`~xdart.gui.themes.dark.apply_seaborn_plot_style`, and legend entries
    carry inline CSS.  So a scale change has to reach them explicitly.

    Dispatching over scene items rather than over plots covers the axes that
    have no PlotItem of their own (the right-hand parameter-trend axis, the
    colour-bar axis) and the standalone position-readout labels, none of which
    any per-plot registry would have seen.
    """
    try:
        import pyqtgraph as pg
    except Exception:                        # pragma: no cover - Qt missing
        return 0
    font = plot_font(scale)
    touched = 0
    for scene in _live_graphics_scenes():
        try:
            items = scene.items()
        except Exception:
            continue
        for item in items:
            if isinstance(item, pg.AxisItem):
                _restyle_axis(item, font)
            elif isinstance(item, pg.LegendItem):
                _restyle_legend(item, font)
            elif isinstance(item, pg.LabelItem):
                _restyle_label_item(item, font)
            else:
                continue
            touched += 1
    return touched


def live_plot_item_count() -> int:
    """How many live PlotItems exist right now (colour bars included).

    Purely observational — it walks the same live scenes and keeps nothing, so
    the count drops as soon as a plot is closed and collected.
    """
    try:
        import pyqtgraph as pg
    except Exception:                        # pragma: no cover - Qt missing
        return 0
    total = 0
    for scene in _live_graphics_scenes():
        try:
            total += sum(1 for it in scene.items()
                         if isinstance(it, pg.PlotItem))
        except Exception:
            continue
    return total


def style_plot_fonts(plot, scale=None) -> None:
    """Apply the tier to one PlotItem's ticks, axis labels and legend.

    Used at construction (through ``apply_seaborn_plot_style``) so a new plot
    is born at the right size; :func:`restyle_live_plots` handles it from then
    on.  Idempotent, and every step is best-effort — a plot missing an axis or
    a legend must never break a settings change.
    """
    font = plot_font(scale)
    for axis_name in ("bottom", "left", "top", "right"):
        try:
            axis = plot.getAxis(axis_name)
        except Exception:
            continue
        if axis is not None:
            _restyle_axis(axis, font)
    legend = getattr(plot, "legend", None)
    if legend is not None:
        _restyle_legend(legend, font)
