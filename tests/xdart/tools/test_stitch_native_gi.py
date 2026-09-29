"""Grazing incidence for a native Stitch 1D (maintainer, 2026-09-29/30): the GI value, and the
incidence (and xrayutilities' other sample circles) bound per frame like the arm's motors."""
from dataclasses import replace
from pathlib import Path
import re

import pytest

from xdart.gui.tools.stitch_values import prepare_stitch_tool
from xrd_tools.integrate.stitch_gi import StitchGI
from tests.xdart.tools.test_stitch_native_values import _native_form

_SAMPLE = (("mu", "mu"), ("chi", "chi"), ("phi", "phi"))


def _gi_form(tmp_path, backend, gi):
    """The native form's SPEC with psic's sample circles in its header, as 17-2's files have them
    (#P: eta 0.3, mu 0, chi 90, phi -1.7), and the GI value."""
    form = _native_form(tmp_path, backend, scans="1")
    spec = Path(form.spec_path)
    text = re.sub(r"^(#O0 .*)$", r"\1  eta  mu  chi  phi", spec.read_text(), flags=re.M)
    spec.write_text(re.sub(r"^(#P0 .*)$", r"\1 0.3 0 90 -1.7", text, flags=re.M))
    return replace(form, gi=gi)


def test_the_gi_value_takes_one_incidence_and_names_the_motors_it_reads():
    motor = StitchGI("q_ip", incidence_motor="eta", sample_motors=_SAMPLE, window=(0., .5))
    assert motor.unit == "qip_A^-1" and StitchGI("q_oop", incident_angle_deg=.2).unit == "qoop_A^-1"
    assert motor.metadata_names("multigeometry") == ("eta",)  # pyFAI: the incidence
    assert motor.metadata_names("xu_hist") == ("eta", "mu", "chi", "phi")  # xrayutilities: every circle
    assert StitchGI("q_ip", incident_angle_deg=.2).metadata_names("xu_hist") == ()  # Manual, circles at zero
    assert motor.incidence({"eta": .3}) == .3 and StitchGI("q_ip", incident_angle_deg=.2).incidence({}) == .2
    for bad, message in ((dict(cut="q_total", incident_angle_deg=.2), "cut"),
                         (dict(cut="q_ip"), "exactly one"),
                         (dict(cut="q_ip", incidence_motor="eta", incident_angle_deg=.2), "exactly one"),
                         (dict(cut="q_ip", incident_angle_deg=.2, sample_orientation=9), "orientation"),
                         (dict(cut="q_ip", incident_angle_deg=.2, sample_motors=(("chi", "chi"),)), "sample motors"),
                         (dict(cut="q_ip", incident_angle_deg=.2, window=(.5, .5)), "min < max")):
        with pytest.raises(ValueError, match=message):
            StitchGI(**bad)


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
def test_a_gi_stitch_binds_every_frames_incidence_and_changes_the_plan(tmp_path, backend):
    gi = StitchGI("q_ip", incidence_motor="eta", sample_motors=_SAMPLE, window=(0., .5))
    form = _gi_form(tmp_path, backend, gi)
    request = prepare_stitch_tool(form).request
    assert request.plan.gi == gi
    bound = {name: value for item in request.manifest.contributions for name, _occurrence, value in item.values}
    expected = {"eta": .3} if backend == "multigeometry" else {"eta": .3, "mu": 0., "chi": 90., "phi": -1.7}
    assert {name: bound[name] for name in expected} == expected
    plain = prepare_stitch_tool(replace(form, gi=None)).request.plan
    assert plain.fingerprint != request.plan.fingerprint and replace(form, gi=None).fingerprint != form.fingerprint
    # a motor the scan does not record is refused by name, as an arm motor is
    with pytest.raises(Exception, match="missing required column.*halpha"):
        prepare_stitch_tool(replace(form, gi=replace(gi, incidence_motor="halpha")))


def test_a_gi_cut_needs_a_native_stitch_1d_in_q(tmp_path):
    form = _gi_form(tmp_path, "multigeometry", StitchGI("q_oop", incident_angle_deg=.2))
    with pytest.raises(ValueError, match="native Stitch 1D in Q"):
        replace(form, unit="2th_deg")


def test_stitch_picks_its_incidence_motor_as_int_does_with_eta_first():
    """Maintainer, 2026-09-29: "Defaulting to eta if it is there, gonth, halpha and so on"."""
    from xrd_tools.session.gi_motor import STITCH_GI_MOTOR_PREFERENCE, pick_default_gi_motor
    assert pick_default_gi_motor(["halpha", "gonth", "Eta", "del"], STITCH_GI_MOTOR_PREFERENCE) == "Eta"
    assert pick_default_gi_motor(["halpha", "gonth", "del"], STITCH_GI_MOTOR_PREFERENCE) == "gonth"
    assert pick_default_gi_motor(["halpha", "del"], STITCH_GI_MOTOR_PREFERENCE) == "halpha"
    assert pick_default_gi_motor(["del", "nu"], STITCH_GI_MOTOR_PREFERENCE) == "Manual"
    assert pick_default_gi_motor(["th", "eta"]) == "th"  # Int's own order is unchanged
