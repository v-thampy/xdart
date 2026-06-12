"""2D axis-identity round-trip guards (minimal TwoDKind seam).

Covers the data-source-unification plan's Stage-1 contract for the batch
GI cake blocker:

* the q/χ cake and the GI qip/qoop map both survive write→read with their
  axes, units and intensity intact (the identity is persisted in the
  ``q``/``chi`` dataset ``units`` attrs);
* the kind is reconstructable from those units
  (:func:`display_logic.two_d_kind_from_units`);
* a reloaded qip/qoop cake is NOT pushed through the display's Q↔2θ
  conversion (which would arcsin out-of-range qip values into a
  collapsed/blank cake) — the actual reload-blank mechanism.

All headless: synthetic results, the real ssrl writer + xdart reader, no
pyFAI / detector data needed.
"""
import tempfile
import os
from types import SimpleNamespace, MethodType

import numpy as np
import pytest
import h5py

from ssrl_xrd_tools.core.containers import IntegrationResult2D
from ssrl_xrd_tools.io.nexus import write_integrated_stack
from xdart.modules.ewald.frame_series import _load_frame_v2
from xdart.gui.tabs.static_scan.display_logic import (
    two_d_kind_from_units, is_gi_2d_units,
)


# --- pure classifier --------------------------------------------------------

@pytest.mark.display_logic
def test_two_d_kind_from_units():
    assert two_d_kind_from_units("qip_A^-1", "qoop_A^-1") == "qip_qoop"
    assert two_d_kind_from_units("q_A^-1", "chi_deg") == "standard"
    assert two_d_kind_from_units("2th_deg", "chi_deg") == "standard"
    assert two_d_kind_from_units("horiz_exit", "vert_exit") == "exit_angles"
    assert two_d_kind_from_units("", "") == "standard"        # back-compat default
    assert is_gi_2d_units("qip_A^-1", "qoop_A^-1") is True
    assert is_gi_2d_units("q_A^-1", "chi_deg") is False


# --- writer -> reader round-trip --------------------------------------------

def _roundtrip(result):
    tmp = tempfile.mktemp(suffix=".h5")
    try:
        with h5py.File(tmp, "w") as f:
            write_integrated_stack(f.require_group("entry"),
                                   frame_indices=[1], results_2d=[result])
        with h5py.File(tmp, "r") as f:
            return _load_frame_v2(f, 1, static=True, gi=False).int_2d
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def test_roundtrip_standard_q_chi_2d():
    res = IntegrationResult2D(
        radial=np.linspace(0.5, 6.0, 40),
        azimuthal=np.linspace(-180.0, 180.0, 36),
        intensity=np.random.default_rng(0).random((40, 36)),
        unit="q_A^-1", azimuthal_unit="chi_deg",
    )
    back = _roundtrip(res)
    assert np.allclose(back.radial, res.radial)
    assert np.allclose(back.azimuthal, res.azimuthal)
    assert back.unit == "q_A^-1" and back.azimuthal_unit == "chi_deg"
    assert two_d_kind_from_units(back.unit, back.azimuthal_unit) == "standard"


def test_roundtrip_gi_qip_qoop_2d():
    # The batch GI cake: qip/qoop axes + units must survive write->read.
    res = IntegrationResult2D(
        radial=np.linspace(-3.5, 3.5, 50),       # qip (can be negative)
        azimuthal=np.linspace(0.0, 5.0, 40),     # qoop
        intensity=np.random.default_rng(1).random((50, 40)),
        unit="qip_A^-1", azimuthal_unit="qoop_A^-1",
    )
    back = _roundtrip(res)
    assert np.allclose(back.radial, res.radial)
    assert np.allclose(back.azimuthal, res.azimuthal)
    assert back.unit == "qip_A^-1" and back.azimuthal_unit == "qoop_A^-1"
    assert two_d_kind_from_units(back.unit, back.azimuthal_unit) == "qip_qoop"
    # Reconstructable as GI without any persisted scan.gi flag.
    assert is_gi_2d_units(back.unit, back.azimuthal_unit) is True


# --- the reload-blank mechanism: qip must not be Q->2θ converted -----------

def test_get_xydata_skips_conversion_for_reloaded_qip():
    from xdart.gui.tabs.static_scan.display_data import DisplayDataMixin

    qip = np.linspace(-3.5, 3.5, 50)
    qoop = np.linspace(0.0, 5.0, 40)
    int_2d = IntegrationResult2D(
        radial=qip, azimuthal=qoop,
        intensity=np.ones((50, 40)),
        unit="qip_A^-1", azimuthal_unit="qoop_A^-1",
    )
    # scan.gi False (saved file reloaded without the flag) and the image
    # unit set to 2θ — the pre-fix path would arcsin-convert qip -> garbage.
    host = SimpleNamespace(
        scan=SimpleNamespace(gi=False),
        ui=SimpleNamespace(imageUnit=SimpleNamespace(currentText=lambda: "2θ (°)")),
    )
    host.get_xydata = MethodType(DisplayDataMixin.get_xydata, host)
    radial, azimuthal = host.get_xydata(int_2d)
    assert np.allclose(radial, qip)        # returned verbatim, NOT converted
    assert np.allclose(azimuthal, qoop)
