"""S-5 — GI mode / unit change must re-key the output-axis ranges.

The GI 1D modes q_oop/exit_angle/chi_gi share ``azimuth_range`` in DIFFERENT
units (q_ip uses ``radial_range``).  A frozen/hydrated explicit range from a
prior mode survived _controls_v2_axis_to_native (only gi_mode_1d changed), so the
next run silently clipped e.g. χGI to a ~4° wedge and WROTE it.  These drive the
REAL _controls_v2_axis_to_native (no monkeypatch on the fixed seam).
"""

import os
from types import MethodType, SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
from xdart.gui.tabs.static_scan.integrator import GI_LABELS_1D, GI_LABELS_2D


def _host(a1, a2, *, gi):
    host = SimpleNamespace(
        scan=SimpleNamespace(gi=gi),
        _controls_v2_ensure_native_int_defaults=lambda: None,
        _controls_v2_scan_int_args=lambda: (a1, a2),
        _controls_v2_npts_oop_visible=lambda: False,
        _controls_v2_unit_code=lambda text, dim: (
            "2th_deg" if str(text).startswith("2") else "q_A^-1"),
    )
    host._controls_v2_axis_to_native = MethodType(
        staticWidget._controls_v2_axis_to_native, host)
    return host


def test_gi_mode_change_clears_stale_output_range():
    a1 = {"gi_mode_1d": "q_oop",
          "azimuth_range": (0.0, 4.0), "radial_range": (0.0, 5.0)}
    host = _host(a1, {"gi_mode_2d": "qip_qoop"}, gi=True)

    host._controls_v2_axis_to_native("Int1D", GI_LABELS_1D[4])  # q_oop -> chi_gi
    assert a1["gi_mode_1d"] == "chi_gi"
    assert "azimuth_range" not in a1
    assert "radial_range" not in a1


def test_gi_same_mode_reselect_keeps_range():
    # Hydration / re-selecting the SAME mode must NOT wipe a restored range.
    a1 = {"gi_mode_1d": "chi_gi", "azimuth_range": (-90.0, 90.0)}
    host = _host(a1, {"gi_mode_2d": "qip_qoop"}, gi=True)

    host._controls_v2_axis_to_native("Int1D", GI_LABELS_1D[4])  # chi_gi -> chi_gi
    assert a1["gi_mode_1d"] == "chi_gi"
    assert a1["azimuth_range"] == (-90.0, 90.0)


def test_gi_2d_mode_change_clears_range():
    a2 = {"gi_mode_2d": "qip_qoop", "azimuth_range": (0.0, 4.0)}
    host = _host({"gi_mode_1d": "q_total"}, a2, gi=True)

    host._controls_v2_axis_to_native("Int2D", GI_LABELS_2D[-1])  # different 2D mode
    assert a2["gi_mode_2d"] != "qip_qoop"
    assert "azimuth_range" not in a2


def test_standard_unit_change_clears_radial_range():
    a1 = {"unit": "q_A^-1", "radial_range": (0.0, 5.0)}
    host = _host(a1, {}, gi=False)

    host._controls_v2_axis_to_native("Int1D", "2θ (°)")   # q -> 2theta
    assert a1["unit"] == "2th_deg"
    assert "radial_range" not in a1


def test_standard_same_unit_keeps_range():
    a1 = {"unit": "q_A^-1", "radial_range": (0.0, 5.0)}
    host = _host(a1, {}, gi=False)

    host._controls_v2_axis_to_native("Int1D", "q (Å⁻¹)")   # stays q
    assert a1["unit"] == "q_A^-1"
    assert a1["radial_range"] == (0.0, 5.0)
