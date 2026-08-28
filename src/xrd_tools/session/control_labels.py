"""Shared Controls range-axis label projection."""

from __future__ import annotations

from collections.abc import Mapping


_AA_INV = "Å⁻¹"
_DEG = "°"
_CHI = "χ"


def _unit_radial_label(unit: object) -> str:
    """Return the radial range label for a pyFAI unit."""

    text = str(unit or "").lower()
    if text == "2th_deg" or "2θ" in text or "2th" in text or "2theta" in text:
        return f"2θ ({_DEG})"
    # The 1D chi/azimuthal-profile mode still edits a Q band, matching
    # integratorTree._update_standard_1d_label.
    return f"Q ({_AA_INV})"


def _axis_text_to_gi_1d_mode(axis: object) -> str:
    """Interpret a displayed GI axis label."""

    text = str(axis or "")
    lower = text.lower()
    if "exit" in lower:
        return "exit_angle"
    if "χgi" in lower or "χ_gi" in lower or "chigi" in lower or "chi_gi" in lower:
        return "chi_gi"
    if "qₒₒₚ" in lower or "q_oop" in lower or "qoop" in lower:
        return "q_oop"
    if "qᵢₚ" in lower or "q_ip" in lower or "qip" in lower:
        return "q_ip"
    return "q_total"


def _axis_text_to_gi_2d_mode(axis: object) -> str:
    """Interpret a displayed GI axis label."""

    text = str(axis or "")
    lower = text.lower()
    if "exit" in lower:
        return "exit_angles"
    if (
        "qᵢₚ" in lower
        or "qₒₒₚ" in lower
        or "q_ip" in lower
        or "q_oop" in lower
        or "qip" in lower
        or "qoop" in lower
        or ("ip" in lower and "oop" in lower)
    ):
        return "qip_qoop"
    return "q_chi"


def range_axis_labels_1d(values: Mapping[tuple[str, ...], object]) -> tuple[str, str]:
    # The selected GI mode owns the labels. A hidden radial-label widget can
    # retain stale text after switching to polar Q, so it is only read in
    # standard mode.
    gi_mode = values.get(("Int1D", "gi_mode"))
    if gi_mode is not None:
        mode = str(gi_mode)
        if mode in {"q_ip", "q_oop"}:
            return f"Qip ({_AA_INV})", f"Qoop ({_AA_INV})"
        if mode == "exit_angle":
            return f"Qip ({_AA_INV})", f"Exit ({_DEG})"
        if mode == "chi_gi":
            return f"Q ({_AA_INV})", f"{_CHI}GI ({_DEG})"
        return (
            _unit_radial_label(values.get(("Int1D", "unit"), "q_A^-1")),
            f"{_CHI} ({_DEG})",
        )

    live_radial = values.get(("Int1D", "radial_label"))
    live_azim = values.get(("Int1D", "azim_label"))
    if live_radial and live_azim:
        return str(live_radial), str(live_azim)

    axis = values.get(("Int1D", "axis"), "")
    mode = _axis_text_to_gi_1d_mode(axis)
    axis_text = str(axis or "").lower()
    if mode != "q_total" or "gi" in axis_text:
        return range_axis_labels_1d({
            ("Int1D", "gi_mode"): mode,
            ("Int1D", "unit"): values.get(("Int1D", "unit"), "q_A^-1"),
        })
    return _unit_radial_label(axis), f"{_CHI} ({_DEG})"


def range_axis_labels_2d(values: Mapping[tuple[str, ...], object]) -> tuple[str, str]:
    # As above, the selected GI mode owns labels while GI controls are active.
    gi_mode = values.get(("Int2D", "gi_mode"))
    if gi_mode is not None:
        mode = str(gi_mode)
        if mode == "qip_qoop":
            return f"Qip ({_AA_INV})", f"Qoop ({_AA_INV})"
        if mode == "exit_angles":
            return f"Qip ({_AA_INV})", f"Exit ({_DEG})"
        return (
            _unit_radial_label(values.get(("Int2D", "unit"), "q_A^-1")),
            f"{_CHI} ({_DEG})",
        )

    live_radial = values.get(("Int2D", "radial_label"))
    live_azim = values.get(("Int2D", "azim_label"))
    if live_radial and live_azim:
        return str(live_radial), str(live_azim)

    axis = values.get(("Int2D", "axis"), "")
    mode = _axis_text_to_gi_2d_mode(axis)
    axis_text = str(axis or "").lower()
    if mode != "q_chi" or "gi" in axis_text:
        return range_axis_labels_2d({
            ("Int2D", "gi_mode"): mode,
            ("Int2D", "unit"): values.get(("Int2D", "unit"), "q_A^-1"),
        })
    return _unit_radial_label(axis), f"{_CHI} ({_DEG})"
