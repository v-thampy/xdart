"""Figures for the illustrated physics guides, drawn by xdart's own code (NEXT-128-HELP-GUIDES).

The maintainer chose "Generate from xdart's code" (2026-10-31): each computed figure in a guide
(``src/xrd_tools/assets/help/*.html``) is drawn here with xrd_tools' real models and written as static SVG between
``<!-- figure:name -->`` and ``<!-- /figure -->``. The models are the peak profiles (``fit.get_peak_model``), the
fitter (``fit.fit_peaks``, ``fit.peak_component_curves``), the template background, Phase Analysis's width models
and texture correction, and its phase engine. The shared stylesheet sits between ``<!-- style -->`` and
``<!-- /style -->`` in every guide. The pages run no script; their CSS colours the SVG through classes, in light and
dark.

``render_guide(text)`` regenerates every figure and the stylesheet of a guide's text. tests/core/test_help_guides.py
fails when a shipped guide no longer matches the code. ``python -m xrd_tools.guide_figures`` rewrites the shipped
guides in place. Noise comes from seeded legacy generators (stable across numpy versions) and coordinates are rounded
to 0.1 px, so a figure is the same on every platform.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from html import escape
from pathlib import Path
import re

import numpy as np

from xrd_tools.help_topics import GUIDE_DIR

__all__ = ["GUIDE_CSS", "FIGURES", "render_guide", "template_fit", "williamson_hall", "write_guides"]

#: the stylesheet every guide carries (one source, so the guides never drift apart): xdart's plot colours, light and
#: dark, offline system fonts
GUIDE_CSS = """
:root {
  --bg: #f3f5f8; --panel: #ffffff; --fg: #1b2230; --muted: #566273; --rule: #d6dce5; --soft: #eceff5;
  --accent: #555f99; --data: #3f4859; --fit: #7446d0; --bgc: #23845a; --tmpl: #8a8f99;
  --c1: #c2571a; --c2: #1f72ad; --c3: #b02f72; --c4: #7a6a12; --resid: #c0405f; --range: rgba(116, 70, 208, .09);
  --good: #23845a; --bad: #b3262f;
  --display: "Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif;
  --body: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #12161d; --panel: #19202a; --fg: #e3e8ef; --muted: #9aa6b4; --rule: #2b3543; --soft: #202a36;
    --accent: #a7afd6; --data: #d2d2dc; --fit: #bd93f9; --bgc: #82c8a0; --tmpl: #8d96a3;
    --c1: #f0a35e; --c2: #6fb7e8; --c3: #f08cb8; --c4: #e3cf6e; --resid: #e68597; --range: rgba(189, 147, 249, .11);
    --good: #82c8a0; --bad: #f07a7a; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #12161d; --panel: #19202a; --fg: #e3e8ef; --muted: #9aa6b4; --rule: #2b3543; --soft: #202a36;
  --accent: #a7afd6; --data: #d2d2dc; --fit: #bd93f9; --bgc: #82c8a0; --tmpl: #8d96a3;
  --c1: #f0a35e; --c2: #6fb7e8; --c3: #f08cb8; --c4: #e3cf6e; --resid: #e68597; --range: rgba(189, 147, 249, .11);
  --good: #82c8a0; --bad: #f07a7a; color-scheme: dark;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.6 var(--body);
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  padding-block: 28px 56px; }
main { max-width: 860px; margin: 0 auto; display: flex; flex-direction: column; gap: 30px; }
h1, h2 { font-family: var(--display); font-weight: 650; line-height: 1.2; margin: 0; text-wrap: balance; }
h1 { font-size: clamp(27px, 4.6vw, 38px); }
h2 { font-size: 22px; }
p { margin: 0; max-width: 68ch; }
.eyebrow { font: 500 12px var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.lead { font-size: 17px; }
.group { font: 600 12px var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent);
  border-top: 1px solid var(--rule); padding-top: 14px; margin-bottom: -16px; }
header, section { display: flex; flex-direction: column; gap: 12px; }
figure { margin: 0; background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 14px;
  display: flex; flex-direction: column; gap: 10px; }
.plot { overflow-x: auto; }
svg { display: block; width: 100%; height: auto; min-width: 340px; }
svg.wide { min-width: 640px; }
figcaption { font-size: 13.5px; color: var(--muted); max-width: 74ch; }
code, .mono { font-family: var(--mono); font-size: .9em; }
.key { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 13px; color: var(--muted); }
.key span { display: inline-flex; align-items: center; gap: 6px; }
.key i { display: inline-block; width: 18px; height: 0; border-top: 2.5px solid currentColor; }
.key i.dash { border-top-style: dashed; }
.key i.dot { width: 7px; height: 7px; border: 0; border-radius: 50%; background: currentColor; }
.k-data { color: var(--data); } .k-fit { color: var(--fit); } .k-bg { color: var(--bgc); } .k-res { color: var(--resid); }
.k-c1 { color: var(--c1); } .k-c2 { color: var(--c2); } .k-c3 { color: var(--c3); } .k-c4 { color: var(--c4); }
.k-tm { color: var(--tmpl); }
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; }
.fact { background: var(--soft); border-radius: 5px; padding: 9px 11px; font-size: 13px; color: var(--muted); min-width: 0; }
.fact b { display: block; color: var(--fg); font: 600 14px var(--body); }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 14px; min-width: 520px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font: 600 12px var(--mono); letter-spacing: .05em; text-transform: uppercase; color: var(--muted); }
td.yes { color: var(--good); font-weight: 600; white-space: nowrap; } td.no { color: var(--bad); font-weight: 600; white-space: nowrap; }
ul { margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 4px; max-width: 70ch; }
footer { font-size: 13px; color: var(--muted); border-top: 1px solid var(--rule); padding-top: 14px; }
svg text { font: 12px var(--body); fill: var(--muted); }
svg text.axis { font-size: 12.5px; fill: var(--fg); }
svg text.note { font-style: italic; }
svg text.lab { font-size: 13px; fill: var(--fg); }
svg text.eq { font: 16px var(--display); fill: var(--fg); }
svg text.strong { fill: var(--bgc); font-weight: 600; }
svg text.bt { font-weight: 600; fill: var(--fg); font-size: 13px; }
svg .plotbox { fill: none; stroke: var(--rule); }
svg .grid { stroke: var(--rule); stroke-width: .6; opacity: .7; }
svg .tick { stroke: var(--muted); }
svg .range, svg .area { fill: var(--range); stroke: none; }
svg .pt { fill: var(--data); }
svg .fit { fill: none; stroke: var(--fit); stroke-width: 2.2; }
svg .bgc { fill: none; stroke: var(--bgc); stroke-width: 1.6; stroke-dasharray: 7 5; }
svg .tmpl { fill: none; stroke: var(--tmpl); stroke-width: 1.4; stroke-dasharray: 2 4; }
svg .c1, svg .c2, svg .c3, svg .c4 { fill: none; stroke-width: 1.5; }
svg .c1 { stroke: var(--c1); } svg .c2 { stroke: var(--c2); } svg .c3 { stroke: var(--c3); } svg .c4 { stroke: var(--c4); }
svg .stick { stroke: var(--accent); stroke-width: 1.6; }
svg .marker { stroke: var(--accent); stroke-width: 1; stroke-dasharray: 2 3; }
svg .resid { fill: none; stroke: var(--resid); stroke-width: 1.2; }
svg .zero, svg .half { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 4 4; }
svg .err { stroke: var(--data); stroke-width: 1; opacity: .6; }
svg .ann { stroke: var(--fg); stroke-width: 1.1; fill: none; }
svg .mk { fill: var(--fg); } svg .mk-beam { fill: var(--c2); } svg .mk-q { fill: var(--fit); } svg .mk-flow { fill: var(--muted); }
svg .plane { stroke: var(--muted); stroke-width: 1.4; }
svg .beam { stroke: var(--c2); stroke-width: 2.2; fill: none; }
svg .qvec { stroke: var(--fit); stroke-width: 3; fill: none; }
svg .box { fill: var(--panel); stroke: var(--rule); stroke-width: 1.2; }
svg .box.good { stroke: var(--good); } svg .box.bad { stroke: var(--bad); } svg .box.quiet { stroke-dasharray: 4 3; }
svg .flow { stroke: var(--muted); stroke-width: 1.4; fill: none; }
"""


def _f(value: float) -> str:
    """A coordinate as SVG text: 0.1 px, never "-0.0" (the same digits on every platform)."""
    text = f"{round(float(value), 1):.1f}"
    return "0.0" if text == "-0.0" else text


def _num(value: float) -> str:
    """A tick label: an integer above 100, else at most two decimals."""
    value = float(value)
    if abs(value) < 1e-9:
        return "0"
    return f"{value:.0f}" if abs(value) >= 100 else f"{value:.2f}".rstrip("0").rstrip(".")


def _ticks(lo: float, hi: float, count: int) -> list[float]:
    raw = (hi - lo) / count
    magnitude = 10.0 ** np.floor(np.log10(raw))
    step = next(s * magnitude for s in (1, 2, 2.5, 5, 10) if s * magnitude >= raw * (1 - 1e-12))
    first = np.ceil(lo / step - 1e-9) * step
    return [round(float(t), 9) for t in np.arange(first, hi + step * 1e-6, step)]


class _Svg:
    """One figure's SVG: elements in drawing order, styled by class (the guide's stylesheet colours them)."""

    def __init__(self, width: int, height: int, label: str, *, wide: bool = False):
        self.width, self.height, self.label, self.wide = width, height, label, wide
        self.parts: list[str] = []

    def add(self, tag: str, attrs: dict, text: str | None = None) -> None:
        rendered = " ".join(f'{key}="{escape(str(value), quote=True)}"' for key, value in attrs.items())
        self.parts.append(f"<{tag} {rendered}></{tag}>" if text is None else
                          f"<{tag} {rendered}>{escape(text, quote=False)}</{tag}>")

    def text(self, x, y, text, cls=None, anchor=None, **attrs):
        values = {"x": _f(x), "y": _f(y), **attrs}
        if cls:
            values["class"] = cls
        if anchor:
            values["text-anchor"] = anchor
        self.add("text", values, text)

    def line(self, x1, y1, x2, y2, cls, **attrs):
        self.add("line", {"x1": _f(x1), "y1": _f(y1), "x2": _f(x2), "y2": _f(y2), "class": cls, **attrs})

    def path(self, d: str, cls: str, **attrs):
        self.add("path", {"d": d, "class": cls, **attrs})

    def arrow_marker(self, name: str, cls: str = "mk") -> str:
        """A one-figure arrowhead (ids are page-wide, so each figure names its own); returns its url()."""
        self.parts.append(f'<defs><marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
                          f'markerHeight="6" orient="auto-start-reverse"><path class="{cls}" d="M0 0 L10 5 L0 10 z">'
                          f'</path></marker></defs>')
        return f"url(#{name})"

    def render(self) -> str:
        cls = ' class="wide"' if self.wide else ""
        return (f'<svg{cls} viewBox="0 0 {self.width} {self.height}" role="img" '
                f'aria-label="{escape(self.label, quote=True)}">' + "".join(self.parts) + "</svg>")


@dataclass(frozen=True)
class _Frame:
    box: tuple[float, float, float, float]
    xr: tuple[float, float]
    yr: tuple[float, float]

    def X(self, x):
        bx, _by, bw, _bh = self.box
        return bx + (float(x) - self.xr[0]) / (self.xr[1] - self.xr[0]) * bw

    def Y(self, y):
        _bx, by, _bw, bh = self.box
        return by + bh - (float(y) - self.yr[0]) / (self.yr[1] - self.yr[0]) * bh

    def d(self, xs, ys) -> str:
        return " ".join(f"{'L' if i else 'M'}{_f(self.X(x))} {_f(self.Y(y))}" for i, (x, y) in enumerate(zip(xs, ys)))


def _frame(svg: _Svg, box, xr, yr, *, nx=6, ny=4, xlabel="", ylabel="", xlabels=True, ylabels=True) -> _Frame:
    """Axes with a light grid, ticks and labels; returns the panel's scale."""
    frame = _Frame(tuple(box), tuple(xr), tuple(yr))
    bx, by, bw, bh = box
    for t in _ticks(xr[0], xr[1], nx):
        svg.line(frame.X(t), by, frame.X(t), by + bh, "grid")
        svg.line(frame.X(t), by + bh, frame.X(t), by + bh + 4, "tick")
        if xlabels:
            svg.text(frame.X(t), by + bh + 17, _num(t), anchor="middle")
    for t in _ticks(yr[0], yr[1], ny):
        svg.line(bx, frame.Y(t), bx + bw, frame.Y(t), "grid")
        svg.line(bx - 4, frame.Y(t), bx, frame.Y(t), "tick")
        if ylabels:
            svg.text(bx - 7, frame.Y(t) + 4, _num(t), anchor="end")
    svg.add("rect", {"x": _f(bx), "y": _f(by), "width": _f(bw), "height": _f(bh), "class": "plotbox"})
    if xlabel:
        svg.text(bx + bw / 2, by + bh + 35, xlabel, "axis", "middle")
    if ylabel:
        cy = by + bh / 2
        svg.text(15, cy, ylabel, "axis", "middle", transform=f"rotate(-90 15 {_f(cy)})")
    return frame


def _dots(svg: _Svg, frame: _Frame, xs, ys, r=2.0):
    for x, y in zip(xs, ys):
        svg.add("circle", {"cx": _f(frame.X(x)), "cy": _f(frame.Y(y)), "r": _f(r), "class": "pt"})


def _noise(seed: int, size: int) -> np.ndarray:
    return np.random.RandomState(seed).standard_normal(size)  # the legacy generator's stream never changes


def _pseudo_voigt(x, center, fwhm, height, fraction=0.5):
    """xdart's pseudo-Voigt profile (lmfit's, through fit.get_peak_model; FWHM = 2σ) scaled to ``height``."""
    from xrd_tools.analysis.fitting.fit import get_peak_model
    y = get_peak_model("pseudovoigt").eval(x=np.asarray(x, dtype=float), amplitude=1.0, center=center,
                                           sigma=fwhm / 2.0, fraction=fraction)
    peak = get_peak_model("pseudovoigt").eval(x=np.array([center]), amplitude=1.0, center=center, sigma=fwhm / 2.0,
                                              fraction=fraction)[0]
    return height * y / peak


def _settled(fit_result):
    """A fit's parameters with each free value rounded to four significant figures. The figures draw from these, so
    a fit's last digits, which differ between platforms and library builds, never move a drawn coordinate."""
    params = fit_result.params.copy()
    for parameter in params.values():
        if parameter.expr is None:
            parameter.set(value=float(f"{parameter.value:.4g}"))
    return params


# ---- the Peak Fitter's figures ------------------------------------------------------------------------------

def _measures() -> str:
    """One peak and the three numbers that carry its physics: position, width at half maximum, area."""
    svg = _Svg(680, 290, "One peak annotated with its position, width at half maximum and area")
    head = svg.arrow_marker("ga-measures")
    q0, fwhm, height = 2.0, 0.05, 600.0
    xs = np.linspace(1.8, 2.2, 401)
    base = 80.0 + 20.0 * xs
    peak = _pseudo_voigt(xs, q0, fwhm, height)
    frame = _frame(svg, (64, 16, 596, 222), (1.8, 2.2), (0, 760), nx=8, xlabel="q (Å⁻¹)", ylabel="Intensity")
    upper = frame.d(xs, base + peak)
    lower = " ".join(f"L{_f(frame.X(x))} {_f(frame.Y(y))}" for x, y in zip(xs[::-1], base[::-1]))
    svg.path(f"{upper} {lower} Z", "area")
    svg.path(frame.d(xs, base), "bgc")
    svg.path(frame.d(xs, base + peak), "fit")
    top = 80.0 + 20.0 * q0 + height
    svg.line(frame.X(q0), frame.Y(top), frame.X(q0), 16 + 222, "marker")
    svg.text(frame.X(q0) + 8, frame.Y(top) + 4, "position q₀ → d = 2π/q₀", "lab")
    half = 80.0 + 20.0 * q0 + height / 2
    left, right = q0 - fwhm / 2, q0 + fwhm / 2  # the profile's own width at half maximum
    svg.path(f"M{_f(frame.X(left))} {_f(frame.Y(half))} L{_f(frame.X(right))} {_f(frame.Y(half))}", "ann",
             **{"marker-start": head, "marker-end": head})
    svg.text(frame.X(right) + 8, frame.Y(half) + 4, "width Δq → size and microstrain", "lab")
    svg.text(frame.X(q0 - 0.125), frame.Y(80.0 + 20.0 * (q0 - 0.125)) - 34, "area → amount, |F|², texture", "lab",
             "middle")
    return svg.render()


@dataclass(frozen=True)
class WilliamsonHall:
    """The Williamson-Hall figure's numbers, from Phase Analysis's own Scherrer width model."""

    k: float
    size_angstrom: float
    strain: float
    svg: str


def williamson_hall() -> WilliamsonHall:
    """Six reflections' widths from ``phase_fitting._scherrer_sigma`` (size 40 nm, microstrain 0.2 %, 2 % scatter);
    xdart adds size and strain in quadrature, FWHM² = (2πK/D)² + (2εq)², so FWHM² against q² is a line whose
    intercept gives the size and whose slope gives the strain."""
    from xrd_tools.analysis.fitting.phase_fitting import _SCHERRER_K, _scherrer_sigma
    size, strain = 400.0, 0.002
    qs = np.array([1.8, 2.6, 3.1, 3.7, 4.5, 5.2])
    widths = 2.0 * _scherrer_sigma(qs, size, strain) * (1.0 + 0.02 * _noise(31, qs.size))
    slope, intercept = np.polyfit(qs ** 2, widths ** 2, 1)
    size_fit, strain_fit = 2 * np.pi * _SCHERRER_K / np.sqrt(intercept), np.sqrt(slope) / 2
    svg = _Svg(680, 270, "Williamson-Hall plot: width squared against q squared for six reflections, a line whose "
                         "intercept gives the crystallite size and slope the microstrain")
    scale = 1e4  # FWHM² in 10⁻⁴ Å⁻²
    frame = _frame(svg, (76, 16, 584, 196), (0, 30), (0, 8), nx=6, ny=4, xlabel="q² (Å⁻²)",
                   ylabel="FWHM² (10⁻⁴ Å⁻²)")
    line_x = np.array([0.0, 30.0])
    svg.path(frame.d(line_x, scale * (intercept + slope * line_x)), "fit")
    _dots(svg, frame, qs ** 2, scale * widths ** 2, r=4)
    svg.line(frame.X(0), frame.Y(scale * intercept), frame.X(5), frame.Y(scale * intercept), "half")
    svg.text(frame.X(0.6), frame.Y(scale * intercept) + 20,
             f"intercept (2πK/D)²: size D = {size_fit / 10:.0f} nm", "lab")
    svg.text(frame.X(29.5), frame.Y(3.0), f"slope (2ε)²: microstrain ε = {100 * strain_fit:.2f} %", "lab", "end")
    return WilliamsonHall(float(_SCHERRER_K), float(size_fit), float(strain_fit), svg.render())


def _anatomy() -> str:
    """A real fit by xdart's fitter: three pseudo-Voigt peaks on a sloped background, inside a fit range."""
    from xrd_tools.analysis.fitting.fit import fit_peaks, peak_component_curves
    centres, heights, fwhms = (1.9, 2.45, 3.1), (520.0, 360.0, 260.0), (0.06, 0.07, 0.08)
    xs = np.linspace(1.4, 3.6, 221)
    truth = 120.0 + 40.0 * xs + sum(_pseudo_voigt(xs, c, w, h) for c, h, w in zip(centres, heights, fwhms))
    ys = truth + np.sqrt(truth) * 1.1 * _noise(7, xs.size)
    lo, hi = 1.605, 3.445  # between samples, so no point sits on the range's edge
    inside = (xs >= lo) & (xs <= hi)
    x_in, y_in = xs[inside], ys[inside]
    result = fit_peaks(x_in, y_in, positions=list(centres), model="pseudovoigt", background="linear")
    fitted = _settled(result.fit_result)
    names = tuple(fitted)
    values = tuple(float(fitted[name].value) for name in names)
    components = peak_component_curves("pseudovoigt", names, values, x_in)
    model = result.fit_result.model
    background = np.asarray(model.eval_components(params=fitted, x=x_in)["bg_"], dtype=float)
    best = np.asarray(model.eval(fitted, x=x_in), dtype=float)
    svg = _Svg(680, 360, "A fit by xdart's fitter: data points, the total fit, a sloped background and three peak "
                         "components over the fit range, with the residual below")
    upper = _frame(svg, (64, 14, 596, 214), (1.4, 3.6), (0, 900), nx=11, ylabel="Intensity", xlabels=False)
    svg.add("rect", {"x": _f(upper.X(lo)), "y": "14.0", "width": _f(upper.X(hi) - upper.X(lo)), "height": "214.0",
                     "class": "range"})
    svg.text((upper.X(lo) + upper.X(hi)) / 2, 29, "fit range: drag it on the plot, or type it", "note", "middle")
    for index, (_name, curve) in enumerate(components):
        svg.path(upper.d(x_in, curve), f"c{index + 1}")
    svg.path(upper.d(x_in, background), "bgc")
    _dots(svg, upper, xs, ys)
    svg.path(upper.d(x_in, best), "fit")
    for index in range(len(centres)):
        centre = fitted[f"p{index}_center"].value
        svg.line(upper.X(centre), 36, upper.X(centre), 228, "marker")
    lower = _frame(svg, (64, 256, 596, 62), (1.4, 3.6), (-90, 90), nx=11, ny=4, xlabel="q (Å⁻¹)")
    svg.line(64, lower.Y(0), 660, lower.Y(0), "zero")
    svg.path(lower.d(x_in, y_in - best), "resid")
    svg.text(15, 287, "residual", "axis", "middle", transform="rotate(-90 15 287.0)")
    return svg.render()


def _profiles() -> str:
    """Five of xdart's peak models, each scaled to height 1 and to its own width at half maximum (the model's
    ``fwhm``)."""
    from xrd_tools.analysis.fitting.fit import get_peak_model
    specs = (("gaussian", {"sigma": 1.0}, "c2"), ("lorentzian", {"sigma": 1.0}, "c1"),
             ("pseudovoigt", {"sigma": 1.0, "fraction": 0.5}, "fit"), ("pearson7", {"sigma": 1.0, "expon": 1.5}, "c4"),
             ("split_pseudovoigt", {"sigma": 0.7, "sigma_r": 1.4, "fraction": 0.5}, "c3"))
    svg = _Svg(680, 300, "Five peak profiles with the same height and the same full width at half maximum")
    frame = _frame(svg, (60, 14, 600, 222), (-2.5, 2.5), (0, 1.05), nx=10, ny=4,
                   xlabel="distance from the centre, in widths at half maximum", ylabel="height, scaled to 1")
    svg.line(60, frame.Y(0.5), 660, frame.Y(0.5), "half")
    svg.text(654, frame.Y(0.5) - 6, "half maximum", "note", "end")
    xs = np.linspace(-2.5, 2.5, 501)
    for name, values, cls in specs:
        model = get_peak_model(name)
        params = model.make_params(amplitude=1.0, center=0.0, **values)
        width = float(params["fwhm"].value)  # the model's own width at half maximum
        height = float(model.eval(params, x=np.array([0.0]))[0])  # each profile peaks at its centre
        svg.path(frame.d(xs, np.asarray(model.eval(params, x=xs * width), dtype=float) / height), cls)
    return svg.render()


@dataclass(frozen=True)
class TemplateFit:
    """The template figure's fit: the scale xdart's fitter found for the substrate's pattern."""

    amplitude: float
    svg: str


def template_fit() -> TemplateFit:
    """A glass-like substrate pattern scaled by 1.6 under three sharp peaks, fitted with background="template"."""
    from xrd_tools.analysis.fitting.fit import fit_peaks
    xs = np.linspace(1.0, 5.0, 241)
    template = 30.0 + 55.0 * np.exp(-(((xs - 2.2) / 0.9) ** 2)) + 8.0 * np.sin(3.0 * xs)
    centres, heights, fwhms = (1.8, 3.0, 4.1), (180.0, 140.0, 90.0), (0.05, 0.06, 0.07)
    truth = 1.6 * template + sum(_pseudo_voigt(xs, c, w, h) for c, h, w in zip(centres, heights, fwhms))
    ys = truth + np.sqrt(truth) * 0.9 * _noise(11, xs.size)
    result = fit_peaks(xs, ys, positions=list(centres), model="pseudovoigt", background="template",
                       background_template=(xs, template))
    fitted = _settled(result.fit_result)
    amplitude = float(fitted["bg_A"].value)
    svg = _Svg(680, 300, "A measured substrate pattern, the same pattern scaled by the fitted bg_A, and the sample's "
                         "data and fit")
    frame = _frame(svg, (64, 14, 596, 222), (1, 5), (0, 360), nx=8, xlabel="q (Å⁻¹)", ylabel="Intensity")
    svg.path(frame.d(xs, template), "tmpl")
    svg.path(frame.d(xs, amplitude * template), "bgc")
    _dots(svg, frame, xs, ys, r=1.9)
    svg.path(frame.d(xs, np.asarray(result.fit_result.model.eval(fitted, x=xs), dtype=float)), "fit")
    at = int(np.argmin(np.abs(xs - 4.7)))
    svg.text(frame.X(4.95), frame.Y(template[at]) + 18, "measured substrate (the template)", "note", "end")
    svg.text(frame.X(2.4), frame.Y(185), f"template × bg_A = {amplitude:.2f}", "note strong", "middle")
    return TemplateFit(amplitude, svg.render())


def _weights() -> str:
    """Points with their measured σ around one peak, fitted with weights 1/σ by xdart's fitter."""
    from xrd_tools.analysis.fitting.fit import fit_peaks
    xs = np.linspace(1.6, 2.4, 33)
    truth = 40.0 + _pseudo_voigt(xs, 2.0, 0.12, 300.0, fraction=0.4)
    sigma = np.sqrt(truth) * 1.3
    ys = truth + sigma * _noise(23, xs.size)
    result = fit_peaks(xs, ys, positions=[2.0], model="pseudovoigt", background="constant", weights=1.0 / sigma)
    fine = np.linspace(1.6, 2.4, 161)
    svg = _Svg(680, 220, "Data points with their uncertainties around one peak, and the weighted fit")
    frame = _frame(svg, (64, 12, 596, 150), (1.6, 2.4), (0, 440), nx=8, ny=4, xlabel="q (Å⁻¹)", ylabel="Intensity")
    for x, y, s in zip(xs, ys, sigma):
        svg.line(frame.X(x), frame.Y(y - s), frame.X(x), frame.Y(y + s), "err")
    _dots(svg, frame, xs, ys, r=2.4)
    svg.path(frame.d(fine, np.asarray(result.fit_result.model.eval(_settled(result.fit_result), x=fine),
                                      dtype=float)), "fit")
    return svg.render()


# ---- Phase Analysis's figures -------------------------------------------------------------------------------

_CSCL = """data_CsCl
_symmetry_space_group_name_H-M   'P m -3 m'
_cell_length_a   4.12000000
_cell_length_b   4.12000000
_cell_length_c   4.12000000
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   221
loop_
 _symmetry_equiv_pos_as_xyz
  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Cs  Cs0  0.00000000  0.00000000  0.00000000  1
  Cl  Cl1  0.50000000  0.50000000  0.50000000  1
"""
#: rock salt written out site by site (P 1), so no symmetry expansion is involved
_NACL = """data_NaCl
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   5.64000000
_cell_length_b   5.64000000
_cell_length_c   5.64000000
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   1
loop_
 _symmetry_equiv_pos_as_xyz
  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Na  Na0  0.0  0.0  0.0  1
  Na  Na1  0.0  0.5  0.5  1
  Na  Na2  0.5  0.0  0.5  1
  Na  Na3  0.5  0.5  0.0  1
  Cl  Cl0  0.5  0.5  0.5  1
  Cl  Cl1  0.5  0.0  0.0  1
  Cl  Cl2  0.0  0.5  0.0  1
  Cl  Cl3  0.0  0.0  0.5  1
"""
_WAVELENGTH = 1.0  # Å; a pattern in q does not depend on it


def _phase(cif: str, name: str, q_range):
    from xrd_tools.analysis.phase_references import PhaseEntry, phase_model
    entry = PhaseEntry(hashlib.sha256(cif.encode()).hexdigest(), name, "cif", cif.encode())
    return phase_model(entry, wavelength_angstrom=_WAVELENGTH, q_range=q_range)


def _phase_patterns(models, q, scales, sigma=0.012):
    """Each phase's pattern on ``q`` from xdart's phase engine (a fixed Gaussian width), and their sum."""
    from xrd_tools.analysis.fitting.phase_fitting import PhaseFitter
    fitter = PhaseFitter(q, np.zeros_like(q))
    extent = (float(q.min()), float(q.max()))
    for model in models:
        fitter.add_phase(model, q_range=extent, min_intensity=0.0)
    params = fitter.build_parameters(phase_profile="gaussian", width_model="fixed")
    params["q_shift"].set(value=0.0)
    for index, scale in enumerate(scales):
        params[f"p{index}_scale"].set(value=scale)
        params[f"p{index}_sigma"].set(value=sigma)
    parts = [np.asarray(fitter.eval_phase(index, params), dtype=float) for index in range(len(models))]
    return parts, np.asarray(fitter.eval_model(params), dtype=float)


def _pattern() -> str:
    """CsCl's pattern from its structure: positions from the lattice, heights from the structure factor."""
    q = np.linspace(1.0, 6.0, 1001)
    model = _phase(_CSCL, "CsCl", (1.0, 6.0))
    (part,), _total = _phase_patterns((model,), q, (1.0,))
    top = float(part.max())
    svg = _Svg(680, 290, "CsCl's diffraction pattern computed from its crystal structure, with each reflection's "
                         "indices")
    frame = _frame(svg, (64, 14, 596, 222), (1.0, 6.0), (0, 1.15), nx=10, ny=4, xlabel="q (Å⁻¹)",
                   ylabel="Intensity, scaled to 1")
    svg.path(frame.d(q, part / top), "fit")
    heights = {peak.q: float(np.interp(peak.q, q, part)) / top for peak in model.peaks if 1.0 <= peak.q <= 6.0}
    placed: list[float] = []
    for peak in sorted(model.peaks, key=lambda p: -heights.get(p.q, 0.0)):  # the strongest first; none overlap
        height = heights.get(peak.q, 0.0)
        if height < 0.08 or any(abs(frame.X(peak.q) - x) < 40 for x in placed):
            continue
        placed.append(frame.X(peak.q))
        label = "(" + "".join(str(i) for i in sorted((abs(i) for i in peak.hkl), reverse=True)) + ")"  # cubic family
        svg.text(frame.X(peak.q), frame.Y(height) - 6, label, "lab", "middle")
    return svg.render()


def _widths() -> str:
    """Peak width against q from Phase Analysis's two width models: Caglioti (the instrument) and Scherrer (size and
    microstrain), xdart's own functions."""
    from xrd_tools.analysis.fitting.phase_fitting import _caglioti_sigma, _scherrer_sigma
    q = np.linspace(0.5, 6.0, 221)
    caglioti = 1e3 * np.asarray(_caglioti_sigma(q, 4e-6, 0.0, 2.5e-5), dtype=float)
    scherrer = 1e3 * np.asarray(_scherrer_sigma(q, 300.0, 0.0015), dtype=float)
    svg = _Svg(680, 270, "Peak width against q from the Caglioti instrument model and the Scherrer size and strain "
                         "model")
    frame = _frame(svg, (70, 14, 590, 200), (0.5, 6.0), (0, 16), nx=11, ny=4, xlabel="q (Å⁻¹)",
                   ylabel="width σ (10⁻³ Å⁻¹)")
    svg.path(frame.d(q, caglioti), "c2")
    svg.path(frame.d(q, scherrer), "c1")
    svg.text(frame.X(0.6), frame.Y(3.0), "Caglioti: the instrument", "lab")
    svg.text(frame.X(0.6), frame.Y(14.0), "Scherrer: crystallite size and microstrain", "lab")
    return svg.render()


def _texture() -> str:
    """The March-Dollase factor from Phase Analysis's own correction, against the angle between a reflection's normal
    and the texture axis."""
    from xrd_tools.analysis.fitting.phase_fitting import _march_dollase
    angle = np.linspace(0.0, 90.0, 181)
    directions = np.column_stack([np.sin(np.radians(angle)), np.zeros_like(angle), np.cos(np.radians(angle))])
    svg = _Svg(680, 270, "March-Dollase intensity factor against the angle between a reflection's normal and the "
                         "texture axis, for plate-like, random and needle-like texture")
    frame = _frame(svg, (64, 14, 596, 200), (0, 90), (0, 4.8), nx=9, ny=4,
                   xlabel="angle between the reflection's normal and the texture axis (°)",
                   ylabel="intensity factor")
    for r, cls in ((0.6, "c1"), (1.0, "zero"), (1.6, "c2")):
        factor = np.asarray(_march_dollase(directions, np.eye(3), np.array([0.0, 0.0, 1.0]), r), dtype=float)
        svg.path(frame.d(angle, factor), cls)
    svg.text(frame.X(26), frame.Y(2.5), "r = 0.6: plate-like", "lab")
    svg.text(frame.X(3), frame.Y(1.0) - 8, "r = 1: random powder", "lab")
    svg.text(frame.X(88), frame.Y(1.6 ** 1.5) - 8, "r = 1.6: needle-like", "lab", "end")
    return svg.render()


def _mixture() -> str:
    """Two phases' patterns from xdart's phase engine and their sum, shifted up so each phase's own curve shows: what
    Phase Analysis fits, with one scale per phase."""
    q = np.linspace(1.0, 6.0, 1001)
    models = (_phase(_CSCL, "CsCl", (1.0, 6.0)), _phase(_NACL, "NaCl", (1.0, 6.0)))
    parts, total = _phase_patterns(models, q, (1.0, 0.6))
    top = float(total.max())
    shift = 1.2
    svg = _Svg(680, 320, "A two-phase pattern: the CsCl and NaCl patterns from their structures, and their sum drawn "
                         "above them")
    frame = _frame(svg, (64, 14, 596, 252), (1.0, 6.0), (0, 2.3), nx=10, ny=4, xlabel="q (Å⁻¹)",
                   ylabel="Intensity", ylabels=False)
    svg.path(frame.d(q, parts[0] / top), "c2")
    svg.path(frame.d(q, parts[1] / top), "c1")
    svg.path(frame.d(q, shift + total / top), "fit")
    svg.text(frame.X(5.95), frame.Y(shift + 0.62), "their sum, what is fitted", "note", "end")
    svg.text(frame.X(5.95), frame.Y(0.62), "each phase's pattern × its scale", "note", "end")
    return svg.render()


#: each figure's builder, by the name its marker carries
FIGURES = {
    "measures": _measures, "williamson-hall": lambda: williamson_hall().svg, "anatomy": _anatomy,
    "profiles": _profiles, "template": lambda: template_fit().svg, "weights": _weights,
    "pattern": _pattern, "widths": _widths, "texture": _texture, "mixture": _mixture,
}

_FIGURE = re.compile(r"(<!-- figure:([a-z0-9-]+) -->)(.*?)(<!-- /figure -->)", re.S)
_STYLE = re.compile(r"(<!-- style -->)(.*?)(<!-- /style -->)", re.S)


def render_guide(text: str) -> str:
    """``text`` with every figure redrawn by the code and the shared stylesheet restored."""
    cache: dict[str, str] = {}

    def figure(match):
        name = match.group(2)
        if name not in cache:
            cache[name] = FIGURES[name]()
        return f"{match.group(1)}\n{cache[name]}\n{match.group(4)}"

    text = _FIGURE.sub(figure, text)
    return _STYLE.sub(lambda match: f"{match.group(1)}\n<style>{GUIDE_CSS}</style>\n{match.group(3)}", text)


def write_guides(folder: Path = GUIDE_DIR) -> list[Path]:
    """Rewrite every shipped guide with the code's figures; returns those that changed."""
    changed = []
    for path in sorted(Path(folder).glob("*.html")):
        text = path.read_text(encoding="utf-8")
        fresh = render_guide(text)
        if fresh != text:
            path.write_text(fresh, encoding="utf-8")
            changed.append(path)
    return changed


if __name__ == "__main__":  # python -m xrd_tools.guide_figures
    for path in write_guides():
        print(f"updated {path}")
