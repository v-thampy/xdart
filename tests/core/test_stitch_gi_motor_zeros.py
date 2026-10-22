"""Deferred item 7 (maintainer, 2026-09-30: "a place to enter a zero offset for each angle ... in the
Diffractometer panel ... set to 0 by default"): a sample circle's angle is its motor's reading minus
its zero (the ARM-ZERO-BOUNDS convention), for both backends' grazing incidence."""
import numpy as np
import pytest

from xrd_tools.integrate.stitch_gi import StitchGI, psic_normal

_SAMPLE = (("mu", "mu"), ("phi", "phi"))


def test_a_sample_circles_angle_is_its_reading_minus_its_zero():
    gi = StitchGI(("q_ip",), incidence_motor="eta", tilt_motor="chi", sample_motors=_SAMPLE,
                  motor_zeros=(("eta", 0.5), ("chi", 0.25), ("phi", -2.0)))
    metadata = {"mu": 0.2, "eta": 1.0, "chi": 90.25, "phi": 8.0}
    assert gi.circles(metadata) == (0.2, 0.5, 90.0, 10.0)
    np.testing.assert_allclose(gi.normal(metadata), psic_normal(0.2, 0.5, 90.0, 10.0))
    # a Manual angle is already physical
    manual = StitchGI(("q_ip",), incident_angle_deg=0.4, tilt_motor="chi", sample_motors=_SAMPLE,
                      motor_zeros=(("chi", 0.25),))
    assert manual.circles(metadata)[1:3] == (0.4, 90.0)


def test_zeros_are_recorded_only_when_set_so_earlier_records_keep_their_identity():
    plain = StitchGI(("q_ip",), incidence_motor="eta", tilt_motor="chi", sample_motors=_SAMPLE)
    zero = StitchGI(("q_ip",), incidence_motor="eta", tilt_motor="chi", sample_motors=_SAMPLE,
                    motor_zeros=(("eta", 0.0),))  # a zero of 0 is no zero
    assert zero.motor_zeros == () and zero.identity() == plain.identity()
    assert "motor_zeros" not in plain.to_provenance()
    offset = StitchGI(("q_ip",), incidence_motor="eta", tilt_motor="chi", sample_motors=_SAMPLE,
                      motor_zeros=(("chi", 0.25),))
    assert offset.identity() != plain.identity()
    assert offset.to_provenance()["motor_zeros"] == {"chi": 0.25}


@pytest.mark.parametrize("zeros", [(("th", 1.0),), (("eta", float("nan")),)])
def test_a_zero_for_a_motor_it_does_not_read_or_not_finite_is_refused(zeros):
    with pytest.raises(ValueError):
        StitchGI(("q_ip",), incidence_motor="eta", tilt_motor="chi", sample_motors=_SAMPLE, motor_zeros=zeros)
