"""COVERAGE-SUPPORT-20261013 decision-6 frozen headless rows (Claude, October 13, before the setting is implemented).

The maintainer's addition at the step-3 disposition: “Record angular coverage”, default on, carried by the shared
AngularSupportPolicy (SessionPolicy.angular_support for Int, the Stitch operation's policy argument). Off skips the
computation and its coverage arrays and records an explicit disabled reason; the default is one constant. Spies wrap
the real kernel; nothing is faked. New APIs are imported inside each row.

INT-PERF-REGRESSION-20261014 Stage A: the maintainer switched the default Off (“Switch the default to Off”); R1 and R3
now pin Off, and A4 pins an omitted policy against explicit On and Off for Int and Stitch.
"""
from unittest import mock

import h5py
import numpy as np

from tests.core.test_angular_support import _planar, _saved_int, _stitch


def test_r1_the_default_is_one_constant_and_it_is_off():
    from xrd_tools.analysis.angular_support import AngularSupportPolicy, DEFAULT_RECORD_ANGULAR_SUPPORT
    from xrd_tools.core.angular_support import REASON_TEXT, SupportReason
    from xrd_tools.session.policy import SessionPolicy
    assert DEFAULT_RECORD_ANGULAR_SUPPORT is False, "Off by default (INT-PERF-REGRESSION-20261014 Stage A)"
    assert AngularSupportPolicy().enabled is DEFAULT_RECORD_ANGULAR_SUPPORT
    assert SessionPolicy().angular_support is None, "None means the default policy"
    assert SupportReason.RECORDING_DISABLED.value == "recording_disabled"
    assert "Record angular coverage" in REASON_TEXT[SupportReason.RECORDING_DISABLED]


def test_r2_int_off_skips_computation_and_records_its_reason(tmp_path):
    from xrd_tools.analysis import angular_support as kernel
    from xrd_tools.core.angular_support import SupportOrigin, SupportReason
    from xrd_tools.io.frame_view import FrameViewReader
    from xrd_tools.io.schema import ANGULAR_SUPPORT_DATASETS
    with mock.patch.object(kernel, "support_footprint", wraps=kernel.support_footprint) as footprints, \
            mock.patch.object(kernel.RunSupportCapture, "applied_mask", autospec=True,
                              side_effect=kernel.RunSupportCapture.applied_mask) as masks:
        off_target, off = _saved_int(tmp_path / "off", enabled=False)
    assert footprints.call_count == 0 and masks.call_count == 0, "off computes nothing"
    assert all(profile.reason is SupportReason.RECORDING_DISABLED and profile.origin is SupportOrigin.RUN
               for profile in off.values())
    on_target, on = _saved_int(tmp_path / "on")
    assert all(profile.available for profile in on.values())
    # round-2 successor edit (maintainer's contract B, October 14): the reason is the output's one marker, with no
    # per-frame coverage; it superseded the one all-NaN table row frozen here on October 13
    with h5py.File(off_target, "r") as handle, h5py.File(on_target, "r") as other:
        group = handle["entry/integrated_1d"]
        assert not any(name in group for name in ANGULAR_SUPPORT_DATASETS), "off writes nothing per frame"
        from xrd_tools.io.schema import ANGULAR_SUPPORT_RECORDING_ATTR
        assert handle["entry"].attrs[ANGULAR_SUPPORT_RECORDING_ATTR] == "off", "the reason is recorded once"
        assert ANGULAR_SUPPORT_RECORDING_ATTR not in other["entry"].attrs
        np.testing.assert_array_equal(group["intensity"][()], other["entry/integrated_1d/intensity"][()])
    with FrameViewReader(off_target) as reader:
        for label in reader.labels():
            saved = reader.read(label).angular_support
            assert saved.reason is SupportReason.RECORDING_DISABLED and saved.origin is SupportOrigin.SAVED
            assert "Record angular coverage" in saved.text


def test_r3_session_policy_carries_the_int_policy():
    from xrd_tools.analysis.angular_support import AngularSupportPolicy
    from xrd_tools.core.angular_support import SupportReason
    from xrd_tools.reduction import Frame, Integration1DPlan, MemorySink, ReductionPlan, Scan
    from xrd_tools.session import FrameRecordStore
    from xrd_tools.session.headless_scan import open_headless_scan_session
    from xrd_tools.session.policy import SessionPolicy

    def published(angular_support):
        """A real headless session under an exact, validated policy (``requirements_from`` + ``resolve_session_policy``,
        as open_headless_scan_session re-derives it) carrying ``angular_support``."""
        from dataclasses import replace
        from types import SimpleNamespace
        from xrd_tools.session.policy import requirements_from, resolve_session_policy
        ai = _planar(shape=(120, 160), poni=(0.006013, 0.008007))
        frames = [Frame(label, image=np.full((120, 160), 10.0 + label, np.float32)) for label in (1, 2)]
        plan = ReductionPlan(integration_1d=Integration1DPlan(npt=40, unit="2th_deg", method="csr"))
        image = frames[0].image
        policy = resolve_session_policy(
            requirements_from(SimpleNamespace(frame_shape=image.shape, dtype=image.dtype), plan),
            envelope_bytes=8 * 1024 ** 3, env={})
        assert type(policy) is SessionPolicy and policy.angular_support is None
        if angular_support is not None:
            policy = replace(policy, angular_support=angular_support)
        store = FrameRecordStore()
        session = open_headless_scan_session(Scan("r3", frames, integrator=ai), plan, sink=MemorySink(),
                                             record_store=store, executor=1, policy=policy)
        for frame in frames:
            session.submit(frame)
        session.finish()
        return [next(iter(store.get(frame.index).results_1d.values())).angular_support for frame in frames]

    assert all(item.reason is SupportReason.RECORDING_DISABLED for item in published(None)), \
        "no Int policy is the default policy, which records nothing"
    assert all(item.available for item in published(AngularSupportPolicy(enabled=True)))
    off = published(AngularSupportPolicy(enabled=False))
    assert all(item.reason is SupportReason.RECORDING_DISABLED for item in off)


def test_r4_stitch_off_skips_computation_and_records_its_reason(tmp_path):
    from xrd_tools.analysis import angular_support as kernel
    from xrd_tools.analysis.angular_support import AngularSupportPolicy
    from xrd_tools.core.angular_support import SupportOrigin, SupportReason
    from xrd_tools.io.analysis_artifact import read_analysis_artifact
    from xrd_tools.io.stitch_record import read_stitch_record
    shape = (195, 487)
    ones = np.full(shape, 200, np.uint16)
    poses = ((0.0, 0.0), (8.0, 0.0))
    with mock.patch.object(kernel, "support_footprint", wraps=kernel.support_footprint) as footprints:
        request, off = _stitch(tmp_path / "off", poses, [ones] * 2, policy=AngularSupportPolicy(enabled=False))
    assert footprints.call_count == 0, "off computes nothing"
    assert off.angular_support.reason is SupportReason.RECORDING_DISABLED
    target = request.module.output.target
    saved = read_analysis_artifact(target).angular_support
    assert saved.reason is SupportReason.RECORDING_DISABLED and saved.origin is SupportOrigin.SAVED
    assert read_stitch_record(target).angular_support.reason is SupportReason.RECORDING_DISABLED


def test_a4_an_omitted_policy_is_the_default_and_explicit_policies_win(tmp_path):
    """INT-PERF-REGRESSION-20261014 A4: headless Int and Stitch given no policy record nothing, since the default is
    Off; explicit AngularSupportPolicy(enabled=True/False) is honored either way, and recording never changes
    intensity."""
    from xrd_tools.analysis.angular_support import AngularSupportPolicy
    from xrd_tools.core.angular_support import SupportReason
    from xrd_tools.reduction import Integration1DPlan, ReductionPlan
    from tests.core.test_angular_support import _OMITTED, _int_scan, _run
    ai = _planar(shape=(120, 160), poni=(0.006013, 0.008007))
    images = [np.full((120, 160), 50.0, np.float32)]
    plan = ReductionPlan(integration_1d=Integration1DPlan(npt=40, unit="2th_deg", method="csr"))
    omitted, on, off = (_run(plan, _int_scan(ai, images), policy).frames[0] for policy in
                        (_OMITTED, AngularSupportPolicy(enabled=True), AngularSupportPolicy(enabled=False)))
    assert omitted.angular_support_1d.reason is SupportReason.RECORDING_DISABLED, "no Int policy: Off"
    assert on.angular_support_1d.available, "explicit On records"
    assert off.angular_support_1d.reason is SupportReason.RECORDING_DISABLED
    np.testing.assert_array_equal(omitted.result_1d.intensity, on.result_1d.intensity)
    ones = np.full((195, 487), 200, np.uint16)
    poses = ((0.0, 0.0), (8.0, 0.0))
    _request, stitched = _stitch(tmp_path / "omitted", poses, [ones] * 2, policy=_OMITTED)
    assert stitched.angular_support.reason is SupportReason.RECORDING_DISABLED, "no Stitch policy: Off"
    _request, stitched = _stitch(tmp_path / "on", poses, [ones] * 2, policy=AngularSupportPolicy(enabled=True))
    assert stitched.angular_support.available, "explicit On records"
