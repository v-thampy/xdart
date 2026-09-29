"""Grazing incidence for a native Stitch 1D (maintainer, 2026-09-29): the GI value, and the
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


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
def test_a_gi_stitch_publishes_i_of_q_and_its_cut_as_one_int_record(tmp_path, backend, monkeypatch):
    """The committed file is an Int record (maintainer, 2026-09-29): Int's reader reads I(q) and the cut
    as the exact float32 cast of what the kernel computed; the cut's coverage, normalization and the
    contributing frames are in entry/stitch."""
    import json

    import numpy as np

    from xrd_tools.analysis import stitch_operation
    from xrd_tools.analysis.module_transaction import ModuleDisposition
    from xrd_tools.io.frame_view import read_frame_view
    from xrd_tools.io.processed_scan_id import is_current_processed_xdart_path
    from xrd_tools.io.stitch_record import read_stitch_record

    request = prepare_stitch_tool(_gi_form(tmp_path, backend, StitchGI(
        "q_ip", incidence_motor="eta", sample_motors=_SAMPLE))).request
    kernel = "run_stitch" if backend == "multigeometry" else "run_xu_hist_stitch_1d"
    real, science = getattr(stitch_operation, kernel), []

    def spy(*args, **kwargs):
        science.append(real(*args, **kwargs))
        return science[-1]

    monkeypatch.setattr(stitch_operation, kernel, spy)
    result = stitch_operation.run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    target = request.module.output.target
    assert is_current_processed_xdart_path(target)
    if backend == "multigeometry":
        iq, (cut, diagnostics) = science[0].payload, science[0].gi_cut
    else:
        iq, cut, diagnostics = science[0].payload, science[0].gi_payload, science[0].gi_diagnostics

    def cast(values):
        return np.asarray(values, dtype="<f4").astype(float)

    np.testing.assert_array_equal(read_frame_view(target, 1, mode_1d="q_total").intensity_1d, cast(iq.intensity))
    view = read_frame_view(target, 1, mode_1d="q_ip")
    assert view.axis_1d.unit == "qip_A^-1"
    np.testing.assert_array_equal(view.axis_1d.values, cast(cut.radial))
    np.testing.assert_array_equal(view.intensity_1d, cast(cut.intensity))
    stored = read_stitch_record(target)
    assert stored.gi[0] == "q_ip" and stored.gi[1].coverage.sum() > 0
    np.testing.assert_array_equal(stored.gi[1].coverage, cast(diagnostics.coverage))
    np.testing.assert_array_equal(stored.gi[1].normalization, cast(diagnostics.normalization))
    assert len(json.loads(stored.frames_json)["records"]) == len(request.module.source.selected_labels)
    assert result.payload.inspection.result_fingerprint == result.terminal.commit.result_fingerprint


@pytest.mark.parametrize("tamper", ["gi-sigma", "other-cut", "cut-extra", "cut-class", "stitch-class", "plain-cut"])
def test_record_admission_takes_exactly_the_stitch_graph_the_record_calls_for(tmp_path, tamper):
    """Codex review F6: a GI cut has no sigma (none is fingerprinted), and entry/stitch holds exactly
    what the record's modes call for, each an NXcollection; anything else is refused."""
    import h5py
    import numpy as np

    from xrd_tools.analysis.stitch_operation import run_stitch_operation
    from xrd_tools.io.analysis_artifact import AnalysisArtifactInvalid, inspect_analysis_artifact

    gi = None if tamper == "plain-cut" else StitchGI("q_ip", incidence_motor="eta", sample_motors=_SAMPLE)
    request = prepare_stitch_tool(_gi_form(tmp_path, "xu_hist", gi)).request
    assert run_stitch_operation(request).terminal.disposition.name == "COMMITTED"
    target = request.module.output.target
    inspect_analysis_artifact(target)  # as written: admitted
    with h5py.File(target, "r+") as handle:
        stitch = handle["entry/stitch"]
        if tamper == "gi-sigma":
            cut = handle["entry/integrated_1d/q_ip"]  # a valid float32 row, as Int would write one
            cut.create_dataset("sigma", data=np.full(cut["intensity"].shape, 123, "<f4"), chunks=True,
                               maxshape=(None, cut["intensity"].shape[1]))
        elif tamper == "other-cut":
            stitch.create_group("q_oop").attrs["NX_class"] = "NXcollection"
        elif tamper == "cut-extra":
            stitch["q_ip"].create_dataset("garbage", data=np.ones(3))
        elif tamper == "cut-class":
            stitch["q_ip"].attrs["NX_class"] = np.bytes_(b"NXdata")  # well-formed, wrong class
        elif tamper == "stitch-class":
            stitch.attrs["NX_class"] = np.bytes_(b"NXdata")
        else:  # a plain record given a cut group
            stitch.create_group("q_ip").attrs["NX_class"] = "NXcollection"
    reason = {"gi-sigma": "carries no sigma", "cut-extra": "GI cut diagnostics are not exact",
              "cut-class": "GI cut diagnostics are not exact", "stitch-class": "not an NXcollection"
              }.get(tamper, "unknown stitch graph")
    with pytest.raises(AnalysisArtifactInvalid, match=reason):
        inspect_analysis_artifact(target)
