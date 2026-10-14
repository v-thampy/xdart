"""COVERAGE-SUPPORT-20261013 correction round 2 (Claude, October 14): rows frozen before implementation.

The maintainer chose contract B: an output's “Record angular coverage” choice is fixed when the output is created.
Append and Reintegrate inherit it, a new output or Overwrite takes the request, an off output stores one marker and
writes no per-frame coverage, and an older unmarked output stays distinct from deliberate off. Codex's event-2 B5 and
B6 are pinned beside it: saved vectors read within the reader's budget and owned by the caller, and dropped labels'
entries retired with their rows. Real reduction, writer and reader paths; spies wrap the real kernel. New names are
imported after each row's first behavioral check where possible, so the parent fails on behavior.
"""
import gc
import shutil
import weakref
from unittest import mock

import h5py
import numpy as np
import pytest

from tests.core.test_angular_support import _planar, _saved_int


def _scan(labels):
    from xrd_tools.reduction import Frame, Scan
    ai = _planar(shape=(120, 160), poni=(0.006013, 0.008007))
    return Scan("k6", [Frame(label, image=np.full((120, 160), 10.0 + label, np.float32)) for label in labels],
                integrator=ai)


def _plan():
    """_saved_int's plan: the masked band, 40 q bins."""
    from xrd_tools.reduction import Integration1DPlan, ReductionPlan
    band = np.zeros((120, 160), bool)
    band[:, 100:110] = True
    return ReductionPlan(integration_1d=Integration1DPlan(npt=40, unit="q_A^-1", method="csr"), mask=band)


def _reduce(target, labels, *, overwrite, enabled=True, **kwargs):
    """A real headless Run into ``target``: Overwrite creates the output, otherwise it appends new frames."""
    from xrd_tools.analysis.angular_support import AngularSupportPolicy
    from xrd_tools.reduction import NexusSink, run_reduction
    run_reduction(_plan(), _scan(labels), NexusSink(target, overwrite=overwrite),
                  angular_support=AngularSupportPolicy(enabled=enabled), **{"execution": "chunked",
                                                                              "executor": False, **kwargs})


def _computed():
    """Spy on the real kernel: how many footprints the wrapped work computed."""
    from xrd_tools.analysis import angular_support as kernel
    return mock.patch.object(kernel, "support_footprint", wraps=kernel.support_footprint)


def _saved(target):
    """Every label's saved support, as FrameViewReader reads it."""
    from xrd_tools.io.frame_view import FrameViewReader
    with FrameViewReader(target) as reader:
        return {label: reader.read(label).angular_support for label in reader.labels()}


def _tables(target):
    """The per-frame support datasets present in the primary 1-D group."""
    from xrd_tools.io.schema import ANGULAR_SUPPORT_DATASETS
    with h5py.File(target, "r") as handle:
        group = handle.get("entry/integrated_1d")
        return () if group is None else tuple(name for name in ANGULAR_SUPPORT_DATASETS if name in group)


def _marker(target):
    """The output's recorded choice (the entry attribute), or None when unmarked."""
    from xrd_tools.io.schema import ANGULAR_SUPPORT_RECORDING_ATTR
    with h5py.File(target, "r") as handle:
        value = handle["entry"].attrs.get(ANGULAR_SUPPORT_RECORDING_ATTR)
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _older(source, target):
    """An older output's form: the same file with its saved support removed and no marker."""
    from xrd_tools.io.schema import ANGULAR_SUPPORT_DATASETS
    shutil.copy2(source, target)
    with h5py.File(target, "r+") as handle:
        for name in ANGULAR_SUPPORT_DATASETS:
            del handle["entry/integrated_1d"][name]
    return target


# ------------------------------------------------------------------------------------------------ contract B

def test_c1_off_is_one_marker_and_older_outputs_stay_distinct(tmp_path):
    from xrd_tools.core.angular_support import SupportOrigin, SupportReason
    from xrd_tools.io.frame_view import FrameViewReader
    off, _ = _saved_int(tmp_path / "off", enabled=False)
    assert _tables(off) == (), "an off output writes no per-frame coverage"
    assert _marker(off) == "off", "one marker records the output's choice"
    on, _ = _saved_int(tmp_path / "on")
    assert _marker(on) is None and len(_tables(on)) == 4, "on is the unmarked, recorded form"
    older = _older(on, tmp_path / "older.nexus")
    with FrameViewReader(off) as reader:
        labels = tuple(reader.labels())
        rows = reader.read_1d_rows(labels)
        (primary,) = [mode for mode in rows.modes if mode.mode == rows.primary_mode]
        for label, profile in zip(primary.labels, primary.support_rows, strict=True):
            assert profile.reason is SupportReason.RECORDING_DISABLED and profile.origin is SupportOrigin.SAVED
            assert reader.read(label).angular_support.reason is SupportReason.RECORDING_DISABLED
    assert all(profile.reason is SupportReason.NO_SAVED_SUPPORT for profile in _saved(older).values()), \
        "an older output is never read as deliberately off (Browse may rebuild it)"


def test_c2_append_to_an_off_output_stays_off_and_computes_nothing(tmp_path):
    from xrd_tools.core.angular_support import SupportReason
    off, _ = _saved_int(tmp_path, enabled=False)
    with _computed() as footprints:
        _reduce(off, [4, 5], overwrite=False)  # the request is the default, on
    assert footprints.call_count == 0, "Append inherits off: nothing is computed"
    assert _tables(off) == () and _marker(off) == "off"
    saved = _saved(off)
    assert sorted(saved) == [1, 2, 3, 4, 5]
    assert all(profile.reason is SupportReason.RECORDING_DISABLED for profile in saved.values())


def test_c3_append_to_a_recorded_output_keeps_recording(tmp_path):
    on, _ = _saved_int(tmp_path)
    with _computed() as footprints:
        _reduce(on, [4, 5], overwrite=False, enabled=False)
    assert footprints.call_count > 0, "Append inherits on whatever the request"
    assert _marker(on) is None
    saved = _saved(on)
    assert sorted(saved) == [1, 2, 3, 4, 5] and all(profile.available for profile in saved.values())


def test_c4_an_older_output_appends_as_on_and_stays_rebuildable(tmp_path):
    from xrd_tools.core.angular_support import SupportReason
    on, _ = _saved_int(tmp_path / "source")
    older = _older(on, tmp_path / "older.nexus")
    with _computed() as footprints:
        _reduce(older, [4, 5], overwrite=False, enabled=False)
    assert footprints.call_count > 0, "an unmarked older output appends as on"
    assert _marker(older) is None
    saved = _saved(older)
    assert all(saved[label].reason is SupportReason.NO_SAVED_SUPPORT for label in (1, 2, 3)), \
        "the older frames stay rebuildable, never marked off"
    assert all(saved[label].available for label in (4, 5))


def test_c5_overwrite_and_new_outputs_take_the_request(tmp_path):
    from xrd_tools.core.angular_support import SupportReason
    on, _ = _saved_int(tmp_path / "b")
    _reduce(on, [1, 2], overwrite=True, enabled=False)
    assert _tables(on) == () and _marker(on) == "off", "Overwrite creates the output again under the request"
    assert all(profile.reason is SupportReason.RECORDING_DISABLED for profile in _saved(on).values())
    off, _ = _saved_int(tmp_path / "a", enabled=False)
    _reduce(off, [1, 2, 3], overwrite=True, enabled=True)
    assert _marker(off) is None and all(profile.available for profile in _saved(off).values())


def test_c6_a_stopped_off_run_keeps_its_marker_and_a_resume_inherits_it(tmp_path):
    from xrd_tools.core.angular_support import SupportReason
    from xrd_tools.reduction import CancelToken
    target = tmp_path / "stop.nexus"
    token, seen = CancelToken(), []

    def progress(*args):
        seen.append(args)
        if len(seen) >= 3:
            token.cancel()

    _reduce(target, [1, 2, 3, 4, 5, 6], overwrite=True, enabled=False, execution="streaming", executor=1,
            cancel_token=token, progress_cb=progress)
    with h5py.File(target, "r") as handle:
        stopped = sorted(int(label) for label in handle["entry/integrated_1d/frame_index"][()])
    assert 0 < len(stopped) < 6, stopped
    assert _tables(target) == (), "a stopped off Run wrote no per-frame coverage"
    assert _marker(target) == "off", "the marker is written when the output is created, not at the end"
    resumed = [label for label in range(1, 9) if label not in stopped][-2:]
    with _computed() as footprints:
        _reduce(target, resumed, overwrite=False)  # the request is on
    assert footprints.call_count == 0
    assert all(profile.reason is SupportReason.RECORDING_DISABLED for profile in _saved(target).values())


def test_c7_removing_every_row_keeps_an_off_outputs_choice(tmp_path):
    from xrd_tools.io.nexus_record import drop_integrated_rows
    off, _ = _saved_int(tmp_path, enabled=False)
    with h5py.File(off, "r+") as handle:
        drop_integrated_rows(handle, "entry/integrated_1d", [1, 2, 3])
        assert "integrated_1d" not in handle["entry"], "the helper removes an emptied group"
    assert _marker(off) == "off", "the choice belongs to the output, not to its rows"


# ------------------------------------------------------------------------------------------------ B6

def test_b6_a_writer_drop_retires_its_entry_and_later_writes_stay_bound(tmp_path):
    from tests.core.test_h23_record_writer import _Facade, _r1
    from xrd_tools.core.angular_support import AngularSupportProfile, SupportOrigin
    from xrd_tools.io import record_writer as rw
    from xrd_tools.session.stage_accounting import ResultMode

    def profile(value):
        return AngularSupportProfile(origin=SupportOrigin.RUN, supported_deg=np.full(4, value), window_deg=360.0,
                                     convention="detector_chi", radial_unit="q_A^-1", bin_edges=(0.05, 1.05))

    target = tmp_path / "drop.nexus"
    facade, mode = _Facade(target), ResultMode.one_d("default")
    for label in (0, 1, 2):
        facade.set_revision(label, mode, 1)
    writer = rw.NexusRecordWriter(target, atomic=False, flush_every=None)
    writer.bind_session(facade)
    writer.begin()
    for label in (0, 1, 2):
        writer.write(rw.RecordWrite(label=label, result_1d=_r1(label + 1.0),
                                    angular_support=profile(10.0 * (label + 1))))
    facade.set_revision(1, mode, 2)
    writer.mark_publication_dropped(1, mode, expected_revision=2)
    facade.set_revision(3, mode, 1)
    writer.write(rw.RecordWrite(label=3, result_1d=_r1(4.0), angular_support=profile(40.0)))  # a later frame
    facade.set_revision(0, mode, 2)
    writer.write(rw.RecordWrite(label=0, result_1d=_r1(5.0), angular_support=profile(50.0)))  # a replacement
    writer.flush(force=True)
    writer.finish()
    from xrd_tools.io.angular_support import read_angular_support
    with h5py.File(target, "r") as handle:
        group = handle["entry/integrated_1d"]
        entries = sorted(int(label) for label in group["angular_support_rows"][()][:, 0])
        assert entries == [0, 2, 3], "the dropped label's entry is retired with its row"
        saved = read_angular_support(group, labels=(0, 2, 3), intensity_rows=group["intensity"][()])
    for label, value in ((0, 50.0), (2, 30.0), (3, 40.0)):
        assert saved[label].available and np.allclose(saved[label].supported_deg, value), (label, saved[label].text)


def test_b6_a_helper_drop_then_append_keeps_survivors_and_new_rows(tmp_path):
    from xrd_tools.io.nexus_record import drop_integrated_rows
    on, captured = _saved_int(tmp_path, frames=69)
    with h5py.File(on, "r+") as handle:
        drop_integrated_rows(handle, "entry/integrated_1d", list(range(1, 69)))
        entries = handle["entry/integrated_1d/angular_support_rows"][()][:, 0].tolist()
    assert entries == [69], "the helper retires dropped labels' entries inside its rebuild"
    _reduce(on, [70, 71], overwrite=False)
    saved = _saved(on)
    assert sorted(saved) == [69, 70, 71]
    assert all(profile.available for profile in saved.values()), {label: p.text for label, p in saved.items()}
    np.testing.assert_array_equal(saved[69].supported_deg, captured[69].supported_deg.astype(np.float32))


# ------------------------------------------------------------------------------------------------ B5

def test_b5_a_cancelled_or_closed_read_keeps_nothing_and_the_caller_keeps_its_vector(tmp_path):
    from xrd_tools.io.frame_view import FrameViewReader
    target, captured = _saved_int(tmp_path, frames=2)
    with FrameViewReader(target) as reader:
        baseline = reader.retained_bytes
        calls = []

        def cancelled():
            calls.append(None)
            return len(calls) > 6

        with pytest.raises(InterruptedError):
            reader.read_1d_rows((1, 2), cancelled=cancelled)
        assert reader.retained_bytes == baseline, "a cancelled read admits nothing"
        rows = reader.read_1d_rows((1, 2))
        (primary,) = [mode for mode in rows.modes if mode.mode == rows.primary_mode]
        ref = weakref.ref(primary.support_rows[1].supported_deg)
        del rows, primary
        gc.collect()
        assert ref() is None, "no reader or codec reference outlives the read, while the reader stays open"
        rows = reader.read_1d_rows((2,))  # read again: from the file, equal
        (primary,) = [mode for mode in rows.modes if mode.mode == rows.primary_mode]
        vector = primary.support_rows[0].supported_deg
    assert reader.retained_bytes == 0
    np.testing.assert_array_equal(vector, captured[2].supported_deg.astype(np.float32),
                                  err_msg="the caller owns its vector after close")
