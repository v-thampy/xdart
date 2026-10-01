"""AUD-9 stage 2b decision 2: a SPEC member's UB recomputed from its ``#G1`` orientation reflections in the model
RSM maps with (the instrument's circle directions, the calibration's zero offsets), with ``#G3`` as the cross-check.

The maintainer, 2026-10-01: "Where does spec store the standard psic circle directions? Xu provides a way to
specify the actual circle directions on the instrument. Shouldn't that be used?" SPEC stores none; ``#G3`` was
computed in SPEC's own psic convention from ``#G1``'s two reflections, so the UB must be recomputed in the model
the frames are mapped with. The ``#G1``/``#G3`` below are scan 43's own (test_data/RSM/STO_align): the reflections
(1 1 4) and (-1 0 4), measured at mu -4.54 and -0.39 degrees and chi 195.78 and 179.68 degrees.
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
RECORD_17_2 = (("x-", "z-", "y-", "z-"), ("x-", "z-"))  # mu, chi and nu reversed (session/instrument.py)


def _spec(tmp_path: Path, *, g1: str | None = G1) -> SimpleNamespace:
    path = tmp_path / "orientation.spec"
    lines = ["#F orientation.spec", "#E 1", "#D Fri May 22 17:43:40 2026", "",
             "#S 43  ascan  eta 42.3411 42.4411  1 1", "#D Fri May 22 17:43:40 2026"]
    if g1 is not None:
        lines.append(f"#G1 {g1}")
    lines += [f"#G3 {G3}", "#N 2", "#L eta  Seconds", "42.3411 1", "42.4411 1", ""]
    path.write_text("\n".join(lines))
    return SimpleNamespace(resolved_root=str(path), resolved_scan="43.1")


def _mapper(circles=STANDARD, offsets=()) -> PixelQMap:
    diffractometer = replace(Diffractometer.psic(), sample_circles=circles[0], detector_circles=circles[1])
    header = DetectorHeader(cch1=97.0, cch2=243.0, pwidth1=0.172, pwidth2=0.172, distance=1014.7173,
                            Nch1=195, Nch2=487)
    return PixelQMap(diffractometer, header, angle_offsets=offsets)


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


def test_a_record_that_reverses_circles_off_zero_gets_its_own_ub(tmp_path):
    """The 17-2 record reverses mu, chi and nu. Scan 43's reflections sit at mu and chi away from 0 and 180
    degrees, so #G3 is not this model's UB; the recompute is, and Preview names the circles."""
    ub, origin = _rsm_orientation_ub(_spec(tmp_path), G3_ARRAY, _mapper(RECORD_17_2), None)
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
        _rsm_orientation_ub(analysis, G3_ARRAY, _mapper(RECORD_17_2), None)
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
        _rsm_orientation_ub(analysis, G3_ARRAY, _mapper(RECORD_17_2), None)
    assert refused.value.code == "RSM_ORIENTATION_REFLECTIONS_MISSING"
