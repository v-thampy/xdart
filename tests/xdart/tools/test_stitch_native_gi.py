"""Grazing incidence for a native Stitch 1D (maintainer, 2026-09-29): the GI value, and psic's sample
circles (the incidence as eta, the tilt as chi, mu and phi) bound per frame like the arm's motors, for
both backends."""
from dataclasses import replace
from pathlib import Path
import re

import pytest

from xdart.gui.tools.stitch_values import prepare_stitch_tool
from xrd_tools.integrate.stitch_gi import GI_UNITS, StitchGI
from tests.xdart.tools.test_stitch_native_values import _native_form

_SAMPLE = (("mu", "mu"), ("phi", "phi"))


def _gi_form(tmp_path, backend, gi):
    """The native form's SPEC with psic's sample circles in its header, as 17-2's files have them
    (#P: eta 0.3, mu 0, chi 90, phi -1.7), and the GI value."""
    form = _native_form(tmp_path, backend, scans="1")
    spec = Path(form.spec_path)
    text = re.sub(r"^(#O0 .*)$", r"\1  eta  mu  chi  phi", spec.read_text(), flags=re.M)
    spec.write_text(re.sub(r"^(#P0 .*)$", r"\1 0.3 0 90 -1.7", text, flags=re.M))
    return replace(form, gi=gi)


def test_the_gi_value_takes_one_incidence_and_one_tilt_and_names_the_motors_it_reads():
    motor = StitchGI(("q_ip",), incidence_motor="eta", tilt_motor="chi", sample_motors=_SAMPLE, oop_range=(0., .5))
    assert GI_UNITS == {"q_ip": "qip_A^-1", "q_oop": "qoop_A^-1"}
    assert motor.metadata_names() == ("eta", "chi", "mu", "phi")  # both backends: every circle
    manual = StitchGI(("q_ip",), incident_angle_deg=.2)
    assert manual.metadata_names() == () and manual.tilt_chi_deg == 90.  # Manual: level at a chi-90 mounting
    assert motor.circles({"eta": .3, "chi": 88., "mu": .5, "phi": -1.7}) == (.5, .3, 88., -1.7)
    assert manual.circles({}) == (0., .2, 90., 0.)
    # pyFAI's incidence and tilt come from the circles' surface normal: chi 90 is no tilt
    assert manual.pyfai_angles(manual.normal({})) == pytest.approx((.2, 0.), abs=1e-12)
    assert motor.pyfai_angles(motor.normal({"eta": .4, "chi": 88., "mu": 0., "phi": 0.})) == pytest.approx((.4, -2.), abs=1e-9)
    # Int's two ranges serve both cuts: each is binned over its own, windowed by the other
    both = StitchGI(("q_oop", "q_ip"), incident_angle_deg=.2, ip_range=(-1., 1.), oop_range=(0., .5))
    assert both.cuts == ("q_ip", "q_oop")  # one order
    assert (both.bins("q_ip"), both.window("q_ip"), both.bins("q_oop"), both.window("q_oop")) == (
        (-1., 1.), (0., .5), (0., .5), (-1., 1.))
    for bad, message in ((dict(cuts=("q_ip",), incident_angle_deg=.2, tilt_motor="chi", tilt_chi_deg=90.), "exactly one"),
                         (dict(cuts=("q_total",), incident_angle_deg=.2), "cuts"),
                         (dict(cuts=(), incident_angle_deg=.2), "cuts"),
                         (dict(cuts=("q_ip", "q_ip"), incident_angle_deg=.2), "cuts"),
                         (dict(cuts="q_ip", incident_angle_deg=.2), "cuts"),
                         (dict(cuts=("q_ip",)), "exactly one"),
                         (dict(cuts=("q_ip",), incidence_motor="eta", incident_angle_deg=.2), "exactly one"),
                         (dict(cuts=("q_ip",), incident_angle_deg=.2, sample_orientation=9), "orientation"),
                         (dict(cuts=("q_ip",), incident_angle_deg=.2, sample_motors=(("mu", "mu"), ("chi", "chi"), ("phi", "phi"))), "sample motors"),
                         (dict(cuts=("q_ip",), incident_angle_deg=.2, oop_range=(.5, .5)), "min < max")):
        with pytest.raises(ValueError, match=message):
            StitchGI(**bad)


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
def test_a_gi_stitch_binds_every_frames_sample_circles_and_changes_the_plan(tmp_path, backend):
    gi = StitchGI(("q_ip",), incidence_motor="eta", tilt_motor="chi", sample_motors=_SAMPLE, oop_range=(0., .5))
    form = _gi_form(tmp_path, backend, gi)
    request = prepare_stitch_tool(form).request
    assert request.plan.gi == gi
    bound = {name: value for item in request.manifest.contributions for name, _occurrence, value in item.values}
    expected = {"eta": .3, "mu": 0., "chi": 90., "phi": -1.7}  # both backends: psic's four circles
    assert {name: bound[name] for name in expected} == expected
    plain_request = prepare_stitch_tool(replace(form, gi=None)).request
    plain = plain_request.plan
    assert plain.fingerprint != request.plan.fingerprint and replace(form, gi=None).fingerprint != form.fingerprint
    # the provenance says how the cuts are made; a plain Stitch's has no GI
    assert request.provenance["plan"]["gi"] == gi.to_provenance() and "gi" not in plain_request.provenance["plan"]
    assert "gi-stitch-on-xu" not in request.provenance.get("holds", ())
    both = prepare_stitch_tool(replace(form, gi=replace(gi, cuts=("q_ip", "q_oop")))).request.plan
    assert both.fingerprint not in {plain.fingerprint, request.plan.fingerprint}
    # a motor the scan does not record is refused by name, as an arm motor is
    with pytest.raises(Exception, match="missing required column.*halpha"):
        prepare_stitch_tool(replace(form, gi=replace(gi, incidence_motor="halpha")))
    with pytest.raises(Exception, match="missing required column.*gonchi"):
        prepare_stitch_tool(replace(form, gi=replace(gi, tilt_motor="gonchi")))
    # a surface far from the pyFAI sample orientation's level is refused at Preview, before a run
    with pytest.raises(Exception, match="choose the orientation of this mounting"):
        prepare_stitch_tool(replace(form, gi=replace(gi, sample_orientation=7)))


def test_a_gi_cut_needs_a_native_stitch_in_q(tmp_path):
    form = _gi_form(tmp_path, "multigeometry", StitchGI(("q_oop",), incident_angle_deg=.2))
    with pytest.raises(ValueError, match="native Stitch in Q"):
        replace(form, unit="2th_deg")


def test_stitch_picks_its_incidence_motor_as_int_does_with_eta_first():
    """Maintainer, 2026-09-29: "Defaulting to eta if it is there, gonth, halpha and so on"."""
    from xrd_tools.session.gi_motor import STITCH_GI_MOTOR_PREFERENCE, pick_default_gi_motor
    assert pick_default_gi_motor(["halpha", "gonth", "Eta", "del"], STITCH_GI_MOTOR_PREFERENCE) == "Eta"
    assert pick_default_gi_motor(["halpha", "gonth", "del"], STITCH_GI_MOTOR_PREFERENCE) == "gonth"
    assert pick_default_gi_motor(["halpha", "del"], STITCH_GI_MOTOR_PREFERENCE) == "halpha"
    assert pick_default_gi_motor(["del", "nu"], STITCH_GI_MOTOR_PREFERENCE) == "Manual"
    assert pick_default_gi_motor(["th", "eta"]) == "th"  # Int's own order is unchanged


@pytest.mark.parametrize("cuts", (("q_ip",), ("q_ip", "q_oop")))
@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
def test_a_gi_stitch_publishes_i_of_q_and_its_cuts_as_one_int_record(tmp_path, backend, cuts, monkeypatch):
    """The committed file is an Int record (maintainer, 2026-09-29): Int's reader reads I(q) and each cut
    (Qip, or Qip + Qoop) as the exact float32 cast of what the kernel computed; each cut's coverage and
    normalization, and the contributing frames, are in entry/stitch."""
    import json

    import numpy as np

    from xrd_tools.analysis import stitch_operation
    from xrd_tools.analysis.module_transaction import ModuleDisposition
    from xrd_tools.io.frame_view import read_frame_view
    from xrd_tools.io.processed_scan_id import is_current_processed_xdart_path
    from xrd_tools.io.stitch_record import read_stitch_record

    request = prepare_stitch_tool(_gi_form(tmp_path, backend, StitchGI(
        cuts, incidence_motor="eta", tilt_motor="chi", sample_motors=_SAMPLE))).request
    kernel = "run_stitch" if backend == "multigeometry" else "run_xu_hist_stitch_1d"
    real, science = getattr(stitch_operation, kernel), []

    def spy(*args, **kwargs):
        science.append(real(*args, **kwargs))
        return science[-1]

    monkeypatch.setattr(stitch_operation, kernel, spy)
    result = stitch_operation.run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    target = request.module.output.target
    assert Path(target).name == "sample_scan1_gi_stitch1d.nexus"  # as Int names a GI result
    assert is_current_processed_xdart_path(target)
    iq, computed = science[0].payload, science[0].gi_cuts

    def cast(values):
        return np.asarray(values, dtype="<f4").astype(float)

    np.testing.assert_array_equal(read_frame_view(target, 1, mode_1d="q_total").intensity_1d, cast(iq.intensity))
    stored = read_stitch_record(target)
    assert [mode for mode, _value in stored.gi] == list(cuts) and len(computed) == len(cuts)
    assert [mode for mode, _cut in result.payload.gi] == list(cuts)
    for (mode, value), (cut, diagnostics) in zip(stored.gi, computed):
        view = read_frame_view(target, 1, mode_1d=mode)
        assert view.axis_1d.unit == GI_UNITS[mode] == cut.unit
        np.testing.assert_array_equal(view.axis_1d.values, cast(cut.radial))
        np.testing.assert_array_equal(view.intensity_1d, cast(cut.intensity))
        assert value.coverage.sum() > 0
        np.testing.assert_array_equal(value.coverage, cast(diagnostics.coverage))
        np.testing.assert_array_equal(value.normalization, cast(diagnostics.normalization))
    assert len(json.loads(stored.frames_json)["records"]) == len(request.module.source.selected_labels)
    import h5py
    with h5py.File(target, "r") as handle:  # the record's own provenance names the cuts
        recorded = json.loads(handle["entry/stitch/provenance_json"][()])
    assert recorded["plan"]["gi"]["cuts"] == list(cuts)
    assert result.payload.inspection.result_fingerprint == result.terminal.commit.result_fingerprint


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
def test_a_gi_stitch_2d_publishes_both_maps_i_of_q_and_the_cuts_as_one_int_record(tmp_path, backend):
    """GI Stitch 2D (STITCH-2D-20260930 item 3): one Int record holds I(q) and the cuts (the 1-D modes
    q_total, q_ip, q_oop) and the (chi, q) map with the q_ip-q_oop map (the 2-D modes q_chi, qip_qoop),
    Int's reader reading each map as the exact float32 cast of the committed payload's."""
    import numpy as np

    from xrd_tools.analysis.module_transaction import ModuleDisposition
    from xrd_tools.analysis.stitch_operation import run_stitch_operation
    from xrd_tools.core.frame_view import TwoDKind
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, read_analysis_artifact
    from xrd_tools.io.frame_view import read_frame_view

    form = replace(_gi_form(tmp_path, backend, StitchGI(("q_ip", "q_oop"), incidence_motor="eta", tilt_motor="chi",
                                                        sample_motors=_SAMPLE)),
                   mode="2d", npt_rad_2d=48, npt_azim_2d=32)
    request = prepare_stitch_tool(form).request
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    target = request.module.output.target
    assert Path(target).name == "sample_scan1_gi_stitch2d.nexus"
    payload = read_analysis_artifact(target, expected_kind=AnalysisArtifactKind.STITCH_2D)
    assert payload.intensity.shape == (48, 32) and payload.profile is not None
    assert [mode for mode, _cut in payload.gi] == ["q_ip", "q_oop"]
    gi_map = payload.gi_map
    assert gi_map.intensity.shape == gi_map.coverage.shape == (48, 32) and gi_map.coverage.sum() > 0
    assert [unit for _axis, unit in gi_map.axis_units] == [GI_UNITS["q_ip"], GI_UNITS["q_oop"]]
    for mode, kind, shown in (("q_chi", TwoDKind.Q_CHI, payload), ("qip_qoop", TwoDKind.QIP_QOOP, gi_map)):
        view = read_frame_view(target, 1, include_thumbnail=False, mode_2d=mode)
        assert view.two_d_kind is kind
        np.testing.assert_array_equal(view.intensity_2d, np.asarray(shown.intensity, dtype=float).T)
    for mode, cut in payload.gi:
        np.testing.assert_array_equal(read_frame_view(target, 1, include_thumbnail=False, mode_1d=mode).intensity_1d,
                                      np.asarray(cut.intensity, dtype=float))
    assert result.payload.inspection.result_fingerprint == result.terminal.commit.result_fingerprint


@pytest.mark.parametrize("tamper", ["gi-sigma", "other-cut", "cut-extra", "cut-class", "stitch-class", "plain-cut",
                                    "second-cut-extra", "second-cut-sigma"])
def test_record_admission_takes_exactly_the_stitch_graph_the_record_calls_for(tmp_path, tamper):
    """Codex review F6: a GI cut has no sigma (none is fingerprinted), and entry/stitch holds exactly
    what the record's modes call for, each an NXcollection; anything else is refused."""
    import h5py
    import numpy as np

    from xrd_tools.analysis.stitch_operation import run_stitch_operation
    from xrd_tools.io.analysis_artifact import AnalysisArtifactInvalid, inspect_analysis_artifact

    cuts = ("q_ip", "q_oop") if tamper.startswith("second-cut") else ("q_ip",)
    gi = None if tamper == "plain-cut" else StitchGI(cuts, incidence_motor="eta", tilt_motor="chi",
                                                      sample_motors=_SAMPLE)
    request = prepare_stitch_tool(_gi_form(tmp_path, "xu_hist", gi)).request
    assert run_stitch_operation(request).terminal.disposition.name == "COMMITTED"
    target = request.module.output.target
    inspect_analysis_artifact(target)  # as written: admitted
    with h5py.File(target, "r+") as handle:
        stitch = handle["entry/stitch"]
        if tamper in {"gi-sigma", "second-cut-sigma"}:
            cut = handle["entry/integrated_1d/" + cuts[-1]]  # a valid float32 row, as Int would write one
            cut.create_dataset("sigma", data=np.full(cut["intensity"].shape, 123, "<f4"), chunks=True,
                               maxshape=(None, cut["intensity"].shape[1]))
        elif tamper == "other-cut":
            stitch.create_group("q_oop").attrs["NX_class"] = "NXcollection"
        elif tamper in {"cut-extra", "second-cut-extra"}:
            stitch[cuts[-1]].create_dataset("garbage", data=np.ones(3))
        elif tamper == "cut-class":
            stitch["q_ip"].attrs["NX_class"] = np.bytes_(b"NXdata")  # well-formed, wrong class
        elif tamper == "stitch-class":
            stitch.attrs["NX_class"] = np.bytes_(b"NXdata")
        else:  # a plain record given a cut group
            stitch.create_group("q_ip").attrs["NX_class"] = "NXcollection"
    reason = {"gi-sigma": "carries no sigma", "second-cut-sigma": "carries no sigma",
              "cut-extra": "GI cut diagnostics are not exact", "second-cut-extra": "GI cut diagnostics are not exact",
              "cut-class": "GI cut diagnostics are not exact", "stitch-class": "not an NXcollection"
              }.get(tamper, "unknown stitch graph")
    with pytest.raises(AnalysisArtifactInvalid, match=reason):
        inspect_analysis_artifact(target)
