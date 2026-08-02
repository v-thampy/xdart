from __future__ import annotations

import inspect
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import get_type_hints

import numpy as np
import pytest

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.session.intent_store import (
    IntentCommitAccepted,
    IntentFreezeAccepted,
    IntentRecaptureRequired,
    RunIntentSnapshot,
    RunIntentStore,
)
from xrd_tools.session.run_configuration import GIIntent, RunIntent, ThresholdIntent
from xrd_tools.sources.selection import DirectorySourceSpec


def _intent(*, generation: int = 0, output_mode: str = "Append") -> RunIntent:
    return RunIntent(
        source_spec=SourceSpec(
            Path("/raw/frame_0001.tif"),
            SourceKind.TIFF_SERIES,
            options={"files": ["frame_0001.tif"], "nested": {"values": [1]}},
        ),
        output_mode=output_mode,
        bai_1d_args={"radial_range": [0.1, 5.0]},
        bai_2d_args={"nested": {"weights": [1.0]}},
        gi=GIIntent(enabled=True, incidence_motor="th", th_val=0.2),
        threshold=ThresholdIntent(threshold_min=1.0, threshold_max=10.0),
        poni_values={"detector": {"pixel": 75e-6}},
        run_options={"writer": {"flush_every": 5}},
        generation=generation,
    )


def _generation(store: RunIntentStore) -> int:
    return store.snapshot().thaw().generation


def test_constructor_and_snapshot_thaws_are_deeply_independent():
    initial = _intent()
    store = RunIntentStore(initial)
    initial.source_spec.options["nested"]["values"].append(2)
    initial.bai_1d_args["radial_range"].append(9.0)
    initial.poni_values["detector"]["pixel"] = 1.0

    first = store.snapshot()
    first_thaw = first.thaw()
    second_thaw = first.thaw()
    later = store.snapshot().thaw()
    first_thaw.source_spec.options["nested"]["values"].append(3)
    first_thaw.bai_2d_args["nested"]["weights"].append(2.0)
    first_thaw.run_options["writer"]["flush_every"] = 9

    assert first.revision == 0
    assert second_thaw.source_spec.options["nested"]["values"] == [1]
    assert later.source_spec.options["nested"]["values"] == [1]
    assert later.bai_1d_args["radial_range"] == [0.1, 5.0]
    assert later.bai_2d_args["nested"]["weights"] == [1.0]
    assert later.poni_values["detector"]["pixel"] == 75e-6
    assert later.run_options["writer"]["flush_every"] == 5


def test_constructor_defensively_retains_mutable_gi_and_threshold_scalars():
    initial = _intent()
    gi_value = np.array(0.2)
    threshold_value = np.array(1.0)
    initial.gi.th_val = gi_value
    initial.threshold.threshold_min = threshold_value

    store = RunIntentStore(initial)
    gi_value[...] = 1.3
    threshold_value[...] = 9.0

    thawed = store.snapshot().thaw()
    frozen = store.freeze(expected_revision=0)

    assert thawed.gi.th_val == pytest.approx(0.2)
    assert thawed.threshold.threshold_min == pytest.approx(1.0)
    assert store.revision == 0
    assert isinstance(frozen, IntentFreezeAccepted)
    assert frozen.configuration.gi.th_val == pytest.approx(0.2)
    assert frozen.configuration.threshold.threshold_min == pytest.approx(1.0)


def test_commit_defensively_retains_mutable_gi_and_threshold_scalars():
    store = RunIntentStore(_intent())
    candidate = _intent()
    gi_value = np.array(0.2)
    threshold_value = np.array(1.0)
    candidate.gi.th_val = gi_value
    candidate.threshold.threshold_min = threshold_value

    accepted = store.commit(candidate, expected_revision=0)
    gi_value[...] = 1.3
    threshold_value[...] = 9.0

    thawed = store.snapshot().thaw()
    frozen = store.freeze(expected_revision=1)

    assert isinstance(accepted, IntentCommitAccepted)
    assert thawed.gi.th_val == pytest.approx(0.2)
    assert thawed.threshold.threshold_min == pytest.approx(1.0)
    assert store.revision == 1
    assert isinstance(frozen, IntentFreezeAccepted)
    assert frozen.configuration.gi.th_val == pytest.approx(0.2)
    assert frozen.configuration.threshold.threshold_min == pytest.approx(1.0)


def test_thaw_keeps_mutable_scalar_values_independent():
    initial = _intent()
    initial.gi.th_val = np.array(0.2)
    initial.threshold.threshold_min = np.array(1.0)
    store = RunIntentStore(initial)

    first = store.snapshot().thaw()
    second = store.snapshot().thaw()
    first.gi.th_val[...] = 1.3
    first.threshold.threshold_min[...] = 9.0

    current = store.snapshot().thaw()
    assert second.gi.th_val == pytest.approx(0.2)
    assert second.threshold.threshold_min == pytest.approx(1.0)
    assert current.gi.th_val == pytest.approx(0.2)
    assert current.threshold.threshold_min == pytest.approx(1.0)


def test_public_snapshot_constructor_defensively_copies_its_intent():
    intent = _intent()
    gi_value = np.array(0.2)
    intent.gi.th_val = gi_value

    snapshot = RunIntentSnapshot(revision=7, _intent=intent)
    gi_value[...] = 1.3

    assert snapshot.thaw().gi.th_val == pytest.approx(0.2)


def test_commit_installs_independent_candidate_and_advances_revision_once():
    store = RunIntentStore(_intent())
    candidate = _intent(output_mode="Overwrite")

    result = store.commit(candidate, expected_revision=0)
    candidate.run_options["writer"]["flush_every"] = 99

    assert isinstance(result, IntentCommitAccepted)
    assert result.revision == store.revision == 1
    assert result.snapshot.revision == 1
    assert result.snapshot.thaw().output_mode == "Overwrite"
    assert store.snapshot().thaw().run_options["writer"]["flush_every"] == 5


def test_identical_commit_is_an_accepted_edit_and_advances_once():
    store = RunIntentStore(_intent())
    candidate = store.snapshot().thaw()

    accepted = store.commit(candidate, expected_revision=0)

    assert isinstance(accepted, IntentCommitAccepted)
    assert accepted.revision == 1
    assert store.revision == 1


def test_stale_commit_and_intervening_edit_require_recapture_without_mutation():
    store = RunIntentStore(_intent())
    old_modal_candidate = store.snapshot().thaw()
    user_edit = _intent(output_mode="Overwrite")
    accepted = store.commit(user_edit, expected_revision=0)
    before = store.snapshot().thaw()

    stale = store.commit(old_modal_candidate, expected_revision=0)

    assert isinstance(accepted, IntentCommitAccepted)
    assert isinstance(stale, IntentRecaptureRequired)
    assert stale.expected_revision == 0
    assert stale.snapshot.revision == 1
    assert stale.snapshot.thaw().output_mode == "Overwrite"
    assert store.revision == 1
    assert store.snapshot().thaw().output_mode == before.output_mode


@pytest.mark.parametrize("forged_generation", [0, 99])
def test_commit_preserves_store_generation_against_forged_candidate_generation(
    forged_generation: int,
):
    store = RunIntentStore(_intent())
    frozen = store.freeze(expected_revision=0)
    assert isinstance(frozen, IntentFreezeAccepted)
    assert frozen.configuration.generation == 1
    candidate = store.snapshot().thaw()
    candidate.generation = forged_generation

    accepted = store.commit(candidate, expected_revision=0)

    assert isinstance(accepted, IntentCommitAccepted)
    assert _generation(store) == 1


def test_compare_and_freeze_preserves_revision_and_advances_generation_once():
    store = RunIntentStore(_intent())

    accepted = store.freeze(expected_revision=0, gi_motor_choices=["th"])

    assert isinstance(accepted, IntentFreezeAccepted)
    assert accepted.revision == store.revision == 0
    assert accepted.configuration.generation == 1
    assert accepted.configuration.gi.effective_motor == "th"
    assert _generation(store) == 1

    later = store.freeze(expected_revision=0, gi_motor_choices=["th", "eta"])
    assert isinstance(later, IntentFreezeAccepted)
    assert later.configuration.generation == 2
    assert later.revision == store.revision == 0
    assert later.configuration.identity != accepted.configuration.identity


def test_stale_and_invalid_freeze_consume_no_generation():
    store = RunIntentStore(_intent())
    stale = store.freeze(expected_revision=1)

    assert isinstance(stale, IntentRecaptureRequired)
    assert stale.snapshot.revision == 0
    assert _generation(store) == 0

    invalid = RunIntentStore(RunIntent(live_mode=True, batch_mode=True))
    with pytest.raises(ValueError, match="cannot both be enabled"):
        invalid.freeze(expected_revision=0)
    assert _generation(invalid) == 0


def test_generation_and_revision_are_orthogonal_across_freeze_then_commit():
    store = RunIntentStore(_intent())
    pre_freeze_candidate = store.snapshot().thaw()
    first = store.freeze(expected_revision=0)
    assert isinstance(first, IntentFreezeAccepted)
    assert first.configuration.generation == 1
    pre_freeze_candidate.output_mode = "Overwrite"

    commit = store.commit(pre_freeze_candidate, expected_revision=0)
    assert isinstance(commit, IntentCommitAccepted)
    assert commit.revision == 1
    assert _generation(store) == 1

    second = store.freeze(expected_revision=1, gi_motor_choices=["th", "eta"])

    assert isinstance(second, IntentFreezeAccepted)
    assert second.configuration.generation == 2
    assert second.configuration.gi.effective_motor == "th"
    assert store.revision == 1


def test_directory_source_snapshot_remains_typed_and_independent(tmp_path):
    source = DirectorySourceSpec(
        root=tmp_path / "raw",
        recursive=True,
        generation=3,
        metadata_format=None,
    )
    store = RunIntentStore(RunIntent(source_spec=source))

    thawed = store.snapshot().thaw()

    assert isinstance(thawed.source_spec, DirectorySourceSpec)
    assert thawed.source_spec is not source
    assert thawed.source_spec.root == source.root
    assert thawed.source_spec.generation == 3
    assert thawed.source_spec.metadata_format is None


@pytest.mark.parametrize("revision", [True, -1, "0"])
def test_invalid_expected_revision_fails_clearly(revision):
    store = RunIntentStore(_intent())
    with pytest.raises((TypeError, ValueError)):
        store.commit(_intent(), expected_revision=revision)
    with pytest.raises((TypeError, ValueError)):
        store.freeze(expected_revision=revision)


def test_public_values_hints_and_lazy_session_exports_are_runtime_resolvable():
    import xrd_tools.session as session
    import xrd_tools.session.intent_store as module

    assert session.RunIntentStore is RunIntentStore
    assert session.RunIntentSnapshot is RunIntentSnapshot
    assert session.IntentCommitAccepted is IntentCommitAccepted
    assert session.IntentFreezeAccepted is IntentFreezeAccepted
    assert session.IntentRecaptureRequired is IntentRecaptureRequired
    assert get_type_hints(RunIntentStore.revision.fget)
    for name in module.__all__:
        value = getattr(module, name)
        if getattr(value, "__annotations__", None):
            assert get_type_hints(value), name
        if inspect.isclass(value):
            for method_name, method in inspect.getmembers(value, inspect.isfunction):
                if not method_name.startswith("_"):
                    assert get_type_hints(method), f"{name}.{method_name}"


def test_store_imports_without_gui_or_heavy_io_stack():
    code = """
import sys
from xrd_tools.session import RunIntentStore
import xrd_tools.session.intent_store
forbidden = ("xdart", "PySide6", "PyQt5", "PyQt6", "pyqtgraph", "h5py", "fabio")
leaked = sorted(name for name in sys.modules if any(name == item or name.startswith(item + ".") for item in forbidden))
assert not leaked, leaked
assert RunIntentStore is not None
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[2] / "src"))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr


def test_same_revision_concurrent_commits_allow_exactly_one_winner():
    store = RunIntentStore(_intent())
    barrier = threading.Barrier(3)
    results: list[IntentCommitAccepted | IntentRecaptureRequired] = []

    def commit_candidate(output_mode: str) -> None:
        candidate = _intent(output_mode=output_mode)
        barrier.wait()
        results.append(store.commit(candidate, expected_revision=0))

    first = threading.Thread(target=commit_candidate, args=("Append",))
    second = threading.Thread(target=commit_candidate, args=("Overwrite",))
    first.start()
    second.start()
    barrier.wait()
    first.join()
    second.join()

    assert sum(isinstance(result, IntentCommitAccepted) for result in results) == 1
    assert sum(isinstance(result, IntentRecaptureRequired) for result in results) == 1
    assert store.revision == 1
