"""AUD-9 stage 2b decision 2: a SPEC member's UB recomputed from its ``#G1`` orientation reflections in the model
RSM maps with (the instrument's circle directions, the calibration's zero offsets), with ``#G3`` as the cross-check.

The maintainer, 2026-10-01: "Where does spec store the standard psic circle directions? Xu provides a way to
specify the actual circle directions on the instrument. Shouldn't that be used?" SPEC stores none; ``#G3`` was
computed in SPEC's own psic convention from ``#G1``'s two reflections, so the UB must be recomputed in the model
the frames are mapped with. The ``#G1``/``#G3`` below are scan 43's own (test_data/RSM/STO_align): the reflections
(1 1 4) and (-1 0 4), measured at mu -4.54 and -0.39 degrees and chi 195.78 and 179.68 degrees.

A record reversing mu, chi and nu is psic's mirror image (17-2; approved 2026-10-01): RSM's lowering sets
``PixelQMap.mirror`` and the recompute is made through the mirror, where SPEC's labels live, so a pure mirror keeps
``#G3``. The rows built with ``mirror=False`` are reversed circles mapped as recorded (a record that is not a mirror).
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from xrd_tools.analysis.rsm_operation import RSMOperationRefused, _rsm_orientation_ub
from xrd_tools.core.geometry.diffractometer import Diffractometer
from xrd_tools.core.geometry.pixel_q import DetectorHeader, PixelQMap
from xrd_tools.io.spec import get_orientation_header
from xrd_tools.rsm.orientation import ROUNDING, relative_difference, spec_orientation

G1 = ("3.906354655 3.906280405 3.906280405 90 90 90 1.608452345 1.608482919 1.608482919 90 90 90 1 1 4 -1 0 4 "
      "52.58521484 40.4010746 195.782618 0 -5.960464455e-09 -4.54167967 50.99613281 11.14921913 179.6754891 0 "
      "-5.960464455e-09 -0.3932421696 0.8156862134 0.8156862134 0 0")
G3 = ("0.01216395927 -0.006505909096 -1.608423764 1.60834664 0.01390876234 0.01210757889 0.01385901527 "
      "-1.608409624 0.006610666882")
G3_ARRAY = np.array(G3.split(), dtype=float).reshape(3, 3)
STANDARD = (("x+", "z-", "y+", "z-"), ("x+", "z-"))
REVERSED_MU_CHI_NU = (("x-", "z-", "y-", "z-"), ("x-", "z-"))  # 17-2's record: psic's mirror image
REVERSED_MU_CHI_NU_PHI = (("x-", "z-", "y-", "z+"), ("x-", "z-"))  # 17-2 for now: the mirror, with phi reversed


def _spec(tmp_path: Path, *, g1: str | None = G1) -> SimpleNamespace:
    path = tmp_path / "orientation.spec"
    lines = ["#F orientation.spec", "#E 1", "#D Fri May 22 17:43:40 2026", "",
             "#S 43  ascan  eta 42.3411 42.4411  1 1", "#D Fri May 22 17:43:40 2026"]
    if g1 is not None:
        lines.append(f"#G1 {g1}")
    lines += [f"#G3 {G3}", "#N 2", "#L eta  Seconds", "42.3411 1", "42.4411 1", ""]
    path.write_text("\n".join(lines))
    return SimpleNamespace(resolved_root=str(path), resolved_scan="43.1")


def _mapper(circles=STANDARD, offsets=(), mirror=False) -> PixelQMap:
    diffractometer = replace(Diffractometer.psic(), sample_circles=circles[0], detector_circles=circles[1])
    header = DetectorHeader(cch1=97.0, cch2=243.0, pwidth1=0.172, pwidth2=0.172, distance=1014.7173,
                            Nch1=195, Nch2=487)
    return PixelQMap(diffractometer, header, angle_offsets=offsets, **({"mirror": True} if mirror else {}))


def test_under_psic_standard_directions_the_recomputed_ub_is_g3_itself(tmp_path):
    """The proof the recompute is right: in psic's standard directions, scan 43's two reflections give #G3 to the
    header's rounding, and #G3's own bytes are kept, so no identity moves."""
    analysis = _spec(tmp_path)
    orientation = spec_orientation(get_orientation_header(analysis.resolved_root, analysis.resolved_scan))
    difference = relative_difference(_mapper().orientation_ub(orientation), G3_ARRAY)
    assert difference < 1e-9  # measured 5.84e-10: #G1 prints about ten significant digits
    ub, origin = _rsm_orientation_ub(analysis, G3_ARRAY, _mapper(), None)
    assert ub.tobytes() == G3_ARRAY.tobytes()
    assert origin.startswith("#G3; recomputed from (1 1 4) and (-1 0 4) in the instrument's directions it agrees to")


def test_reversed_circles_mapped_as_recorded_get_their_own_ub_off_zero(tmp_path):
    """Circles reversed and mapped as recorded (no mirror): scan 43's reflections sit at mu and chi away from 0 and
    180 degrees, so #G3 is not this model's UB; the recompute is, and Preview names the circles. (SPEC's labels
    then fit only a model of SPEC's handedness: for 17-2's whole record RSM maps through the mirror instead.)"""
    ub, origin = _rsm_orientation_ub(_spec(tmp_path), G3_ARRAY, _mapper(REVERSED_MU_CHI_NU), None)
    assert relative_difference(ub, G3_ARRAY) > 0.5  # measured 0.778
    assert origin.startswith("recomputed from (1 1 4) and (-1 0 4) in the instrument's directions; #G3 differs by")
    assert origin.endswith("reversed circles away from 0° or 180° there: chi, mu")


def test_a_reversed_circle_the_reflections_hold_at_zero_keeps_g3(tmp_path):
    """phi is 0 at both reflections: reversing it changes no reflection's direction, so #G3 is still the UB."""
    phi_reversed = (("x+", "z-", "y+", "z+"), ("x+", "z-"))
    ub, origin = _rsm_orientation_ub(_spec(tmp_path), G3_ARRAY, _mapper(phi_reversed), None)
    assert ub.tobytes() == G3_ARRAY.tobytes() and origin.startswith("#G3; recomputed")


def test_the_calibrations_zero_offsets_enter_the_reflections_as_every_frame(tmp_path):
    """A del zero offset is subtracted from the reflections' angles as from every frame's: the UB moves with it."""
    offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.5)  # mu, eta, chi, phi, nu, del
    ub, origin = _rsm_orientation_ub(_spec(tmp_path), G3_ARRAY, _mapper(offsets=offsets), None)
    assert relative_difference(ub, G3_ARRAY) > ROUNDING and origin.startswith("recomputed")


def test_without_orientation_reflections_only_the_standard_model_keeps_g3(tmp_path):
    analysis = _spec(tmp_path, g1=None)
    ub, origin = _rsm_orientation_ub(analysis, G3_ARRAY, _mapper(), None)
    assert ub.tobytes() == G3_ARRAY.tobytes() and origin.startswith("#G3 (the header holds no orientation")
    with pytest.raises(RSMOperationRefused) as refused:
        _rsm_orientation_ub(analysis, G3_ARRAY, _mapper(REVERSED_MU_CHI_NU), None)
    assert refused.value.code == "RSM_ORIENTATION_REFLECTIONS_MISSING"


def test_g1_without_two_usable_reflections_is_none():
    values = np.array(G1.split(), dtype=float)
    assert spec_orientation(values) is not None
    assert spec_orientation(values[:20]) is None  # too few values
    parallel = values.copy()
    parallel[15:18] = parallel[12:15] * 2  # (2 2 8) parallel to (1 1 4)
    assert spec_orientation(parallel) is None


def test_reflections_that_do_not_reproduce_g3_are_not_trusted(tmp_path):
    """The self-check: a #G1 whose reflections do not give its own #G3 in psic's standard model (here or1's eta
    moved by 5 degrees; in practice a non-psic layout or an edited lattice) keeps #G3 in the standard model, with
    a warning, and refuses another model rather than swap a valid #G3 for a recompute of untrusted reflections."""
    values = G1.split()
    values[25] = str(float(values[25]) + 5.0)  # or1's eta (#G1 angles: del, eta, chi, phi, nu, mu)
    analysis = _spec(tmp_path, g1=" ".join(values))
    ub, origin = _rsm_orientation_ub(analysis, G3_ARRAY, _mapper(), None)
    assert ub.tobytes() == G3_ARRAY.tobytes() and origin.startswith("#G3, kept; ⚠ its own reflections")
    with pytest.raises(RSMOperationRefused) as refused:
        _rsm_orientation_ub(analysis, G3_ARRAY, _mapper(REVERSED_MU_CHI_NU), None)
    assert refused.value.code == "RSM_ORIENTATION_REFLECTIONS_MISSING"


def test_through_the_mirror_17_2s_record_keeps_g3(tmp_path):
    """17-2's record mapped through the mirror (RSM's lowering sets it): SPEC's psic model is the mirror image's,
    so scan 43's reflections give #G3 there, whatever mu and chi are, and #G3's own bytes are kept."""
    ub, origin = _rsm_orientation_ub(_spec(tmp_path), G3_ARRAY, _mapper(REVERSED_MU_CHI_NU, mirror=True), None)
    assert ub.tobytes() == G3_ARRAY.tobytes()
    assert origin.startswith("#G3; recomputed from (1 1 4) and (-1 0 4) in the instrument's directions through the "
                             "mirror it agrees to")


def test_through_the_mirror_phi_still_reversed_keeps_g3_where_the_reflections_hold_phi_at_zero(tmp_path):
    """17-2 now: the mirror with phi reversed too. Phi is 0 at both reflections, so #G3 is still the UB; only a
    reflection taken at another phi could move it."""
    mapper = _mapper(REVERSED_MU_CHI_NU_PHI, mirror=True)
    ub, origin = _rsm_orientation_ub(_spec(tmp_path), G3_ARRAY, mapper, None)
    assert ub.tobytes() == G3_ARRAY.tobytes() and "through the mirror it agrees to" in origin


def test_through_the_mirror_without_orientation_reflections_only_a_pure_mirror_keeps_g3(tmp_path):
    """No #G1: a pure mirror maps in SPEC's own model, so #G3 stands; with phi still reversed it cannot be
    recomputed, and is refused as any other model is."""
    analysis = _spec(tmp_path, g1=None)
    ub, origin = _rsm_orientation_ub(analysis, G3_ARRAY, _mapper(REVERSED_MU_CHI_NU, mirror=True), None)
    assert ub.tobytes() == G3_ARRAY.tobytes() and origin.startswith("#G3 (the header holds no orientation")
    with pytest.raises(RSMOperationRefused) as refused:
        _rsm_orientation_ub(analysis, G3_ARRAY, _mapper(REVERSED_MU_CHI_NU_PHI, mirror=True), None)
    assert refused.value.code == "RSM_ORIENTATION_REFLECTIONS_MISSING"


def test_through_the_mirror_reflections_that_do_not_reproduce_g3_keep_it_with_the_warning(tmp_path):
    """The standard self-check is made without the mirror. Reflections that fail it keep #G3, with the warning,
    for a pure mirror (its model is SPEC's), and are refused with phi still reversed."""
    values = G1.split()
    values[25] = str(float(values[25]) + 5.0)  # or1's eta
    analysis = _spec(tmp_path, g1=" ".join(values))
    ub, origin = _rsm_orientation_ub(analysis, G3_ARRAY, _mapper(REVERSED_MU_CHI_NU, mirror=True), None)
    assert ub.tobytes() == G3_ARRAY.tobytes() and origin.startswith("#G3, kept; ⚠ its own reflections")
    with pytest.raises(RSMOperationRefused) as refused:
        _rsm_orientation_ub(analysis, G3_ARRAY, _mapper(REVERSED_MU_CHI_NU_PHI, mirror=True), None)
    assert refused.value.code == "RSM_ORIENTATION_REFLECTIONS_MISSING"
