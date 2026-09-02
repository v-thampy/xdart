"""Focused operation oracle for immutable Reintegration v4."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
from types import MappingProxyType, SimpleNamespace

import h5py
import numpy as np
import pytest

from tests.core.test_vnext_p34_existing_replacement import (
    _plans,
    _seed_existing,
    _stub_integrators,
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


_DEFAULT_TERMINAL = object()


def _dimension_preparation(seeded, dimension):
    preparation = copy.deepcopy(seeded.preparation)
    if dimension == "2d":
        from xrd_tools.reduction.provenance_config import _integration_2d_args

        args = _integration_2d_args(_plans()[1], None)
        args.pop("gi_mode_2d", None)
        preparation["selected_plan"] = {
            "version": 1,
            "dimension": "2d",
            "bai_args": args,
            "gi_mode": None,
        }
    return preparation


def _plan(
    seeded,
    *,
    dimension="1d",
    explicit_output=None,
    expected_terminal=_DEFAULT_TERMINAL,
    preparation=None,
    **values,
):
    from xrd_tools.reduction import ReintegrateSuccessorPlan

    return ReintegrateSuccessorPlan.from_artifact(
        seeded.target,
        entry="entry",
        dimension=dimension,
        preparation=(
            _dimension_preparation(seeded, dimension)
            if preparation is None else preparation
        ),
        expected_target_snapshot=None,
        expected_terminal_identity=(
            seeded.terminal.commit_identity
            if expected_terminal is _DEFAULT_TERMINAL
            else expected_terminal
        ),
        expected_labels=seeded.labels,
        explicit_output=(
            seeded.target.parent / "immutable-successor.nexus"
            if explicit_output is None else explicit_output
        ),
        **values,
    )


def _prepared_eligible_source(
    seeded,
    *,
    master_name=None,
    member_name="raw-member.h5",
):
    """Rewrite only the raw-source topology into the strict v1 fast class."""

    from xdart.gui.tabs.scattering.contracts import (
        ExternalSourceState,
        SourceExecutionStamp,
        SourceFileState,
    )
    from xrd_tools.io.finite_artifact import capture_finite_source

    original_source = seeded.source
    source = (
        original_source
        if master_name is None else original_source.with_name(master_name)
    )
    member = source.with_name(member_name)
    original_source.rename(member)
    with h5py.File(source, "w") as document:
        entry = document.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data["data_000001"] = h5py.ExternalLink(
            member.name, "/entry/instrument/detector/data",
        )
    source_state = SourceFileState.capture(source)
    member_state = SourceFileState.capture(member)
    frame_count = seeded.raw.shape[0]
    execution = SourceExecutionStamp(
        source_state,
        "nexus_hdf5",
        frame_count,
        0,
        external_members=(ExternalSourceState(
            member_state,
            "/entry/instrument/detector/data",
            0,
            frame_count,
            0,
        ),),
    )
    encoded = json.dumps(
        execution.as_dict(), sort_keys=True, separators=(",", ":"),
    )
    with h5py.File(seeded.target, "r+") as document:
        detector = document["entry/instrument/detector"]
        if "mask" in detector:
            del detector["mask"]
        config = document["entry/reduction/config"]
        del config["source_execution"]
        config.create_dataset("source_execution", data=encoded)
        for label in seeded.labels:
            source_group = document[f"entry/frames/frame_{label:04d}/source"]
            if master_name is not None:
                del source_group["path"]
                source_group.create_dataset("path", data=source.name)
            for key, value in {
                "adapter_id": "nexus_hdf5",
                "file_size": source_state.size,
                "file_mtime_ns": source_state.mtime_ns,
                "frame_count": frame_count,
                "dataset_path": "/entry/data/data_000001",
                "self_contained": False,
            }.items():
                source_group.attrs.modify(key, value)
    return capture_finite_source(seeded.target)


def _prepared_plan(
    seeded,
    *,
    dimension="1d",
    explicit_output=None,
    offer=None,
    preparation=None,
    **values,
):
    from xrd_tools.reduction import (
        ReintegrateSuccessorPlan,
        prepare_reintegrate_bundle,
    )

    if offer is None:
        source = _prepared_eligible_source(seeded)
        offer = prepare_reintegrate_bundle(
            source,
            entry="entry",
            labels=seeded.labels,
            source_root=str(seeded.target.parent),
        )
    assert offer.disposition == "READY", offer.miss_code
    assert offer.bundle is not None
    assert offer.bundle.one_d.disposition == "READY"
    assert offer.bundle.two_d.disposition == "READY"
    plan = ReintegrateSuccessorPlan.from_prepared_capsule(
        offer,
        dimension=dimension,
        preparation=(
            _dimension_preparation(seeded, dimension)
            if preparation is None else preparation
        ),
        expected_labels=seeded.labels,
        explicit_output=(
            seeded.target.parent / "immutable-prepared.nexus"
            if explicit_output is None else explicit_output
        ),
        **values,
    )
    return offer, plan


def _install_prepared_append_lineage(seeded, *, multi_epoch=False):
    from xrd_tools.io.append import (
        AppendExternalMember,
        AppendIntent,
        AppendSource,
        _json as canonical_append_json,
        begin_same_run_lineage,
        commit_append_lineage,
    )
    from xrd_tools.io.finite_artifact import capture_finite_source

    _prepared_eligible_source(seeded)
    with h5py.File(seeded.target, "r") as document:
        execution = json.loads(
            document["entry/reduction/config/source_execution"].asstr()[()]
        )
    external = execution["external_members"][0]

    def source(stop, generation):
        member = AppendExternalMember(
            external["file"]["path"],
            external["dataset"],
            external["file"]["size"],
            external["file"]["mtime_ns"],
            external["first"],
            stop,
            external["epoch"],
        )
        return AppendSource(
            execution["path"],
            execution["adapter_id"],
            execution["size"],
            execution["mtime_ns"],
            stop,
            dataset_paths=("/entry/data/data_000001",),
            external_members=(member,),
            generation=generation,
        )

    def intent(labels, selected_source):
        return AppendIntent(
            "entry",
            str(seeded.target.parent),
            "unicode/source-id",
            "unicode/science-id",
            ("1d:default", "2d:default"),
            selected_source,
            labels,
        )

    if multi_epoch:
        current = intent((0, 1), source(2, 1))
        decision = begin_same_run_lineage(current)
    else:
        current = intent(seeded.labels, source(len(seeded.labels), 0))
        decision = begin_same_run_lineage(current)
    with h5py.File(seeded.target, "r+") as document:
        commit_append_lineage(
            document["entry"], decision, written_labels=decision.labels,
        )
        if multi_epoch:
            config = document["entry/reduction/config"]
            lineage = json.loads(config["append_lineage"].asstr()[()])
            current_source = lineage["epochs"][0]["source"]
            prior_source = copy.deepcopy(current_source)
            prior_source["extent"] = 1
            prior_source["generation"] = 0
            prior_source["external_members"][0]["source_stop"] = 1
            lineage["epochs"] = [
                {"source": prior_source, "labels": [0]},
                {"source": current_source, "labels": [1]},
            ]
            del config["append_lineage"]
            config.create_dataset(
                "append_lineage", data=canonical_append_json(lineage),
            )
    with h5py.File(seeded.target, "r") as document:
        raw = document["entry/reduction/config/append_lineage"][()]
    return capture_finite_source(seeded.target), raw


def _drop_dimension_labels(target, dimension, labels):
    from xrd_tools.io.nexus_record import drop_integrated_rows

    with h5py.File(target, "r+") as document:
        drop_integrated_rows(
            document,
            f"entry/integrated_{dimension}",
            tuple(labels),
        )


def _stat_terminal(snapshot, digest):
    from xrd_tools.io.output_transaction import StreamTerminal

    return StreamTerminal(
        snapshot.path,
        snapshot.size,
        digest,
        1,
        snapshot.device,
        snapshot.inode,
        snapshot.mtime_ns,
        snapshot.ctime_ns,
    )


def test_immutable_successor_preserves_source_and_publishes_distinct(
    tmp_path, monkeypatch,
):
    from xrd_tools.io.finite_artifact import require_finite_artifact_lineage
    from xrd_tools.reduction import run_reintegrate_successor

    seeded = _seed_existing(tmp_path, labels=(2, 5), name="immutable")
    before = seeded.target.read_bytes()
    before_digest = _digest(seeded.target)
    _stub_integrators(monkeypatch)

    plan = _plan(seeded)
    result = run_reintegrate_successor(plan)

    assert result.disposition == "COMMITTED"
    assert result.source_artifact == str(seeded.target.resolve())
    assert result.output_artifact == plan.output_artifact
    assert result.output_artifact != result.source_artifact
    assert result.committed_labels == seeded.labels
    assert result.publication_dropped_labels == ()
    assert result.terminal is not None
    assert result.commit_identity is not None
    assert seeded.target.read_bytes() == before
    assert _digest(seeded.target) == before_digest

    with h5py.File(result.output_artifact, "r") as document:
        lineage = require_finite_artifact_lineage(document)
        assert lineage.lineage_identity == plan.lineage_identity
        assert document.attrs["file_name"] == result.output_artifact
        np.testing.assert_array_equal(
            document["entry/integrated_2d/frame_index"][...],
            np.asarray(seeded.labels),
        )
        assert document["entry/integrated_1d/frame_index"][...].tolist() == [2, 5]
        assert document["entry/integrated_1d/intensity"][0, 0] == pytest.approx(102)


def test_v4_recipe_round_trip_is_file_io_free_and_v3_is_refused(
    tmp_path, monkeypatch,
):
    from xrd_tools.reduction import (
        ReintegrateRecipeMigrationRequired,
        ReintegrateSuccessorPlan,
    )
    import xrd_tools.reduction.reintegrate_successor as module

    seeded = _seed_existing(tmp_path, labels=(2,), name="recipe")
    plan = _plan(seeded)
    recipe = plan.as_recipe()

    monkeypatch.setattr(module, "capture_finite_source", lambda *_: pytest.fail("I/O"))
    replay = ReintegrateSuccessorPlan.from_recipe(recipe)
    assert replay.as_recipe() == recipe

    old = seeded.preparation.copy()
    with pytest.raises(ReintegrateRecipeMigrationRequired) as captured:
        ReintegrateSuccessorPlan.from_recipe({
            "schema": "xrd_tools.reintegrate.plan",
            "version": 3,
            "plan": old,
        })
    assert captured.value.code == "IMMUTABLE_SUCCESSOR_V4_REQUIRED"
    assert captured.value.version == 3

    cyclic = {}
    cyclic["plan"] = cyclic
    with pytest.raises(
        ValueError, match="RECIPE_BOUNDED_JSON_UNSUPPORTED",
    ) as captured:
        ReintegrateSuccessorPlan.from_recipe(cyclic)
    assert type(captured.value) is ValueError


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("version",), 4.0),
        (("plan", "api_version"), 4.0),
        (("plan", "qualification", "api_version"), 3.0),
        (("plan", "qualification", "api_version"), 99),
        (
            ("plan", "finite", "output_artifact"),
            "NORMALIZATION_EQUIVALENT",
        ),
    ),
    ids=(
        "recipe-version-float",
        "plan-api-float",
        "qualification-api-float",
        "qualification-api-unknown",
        "output-path-noncanonical",
    ),
)
def test_recipe_literals_and_paths_are_exact_before_file_io(
    tmp_path, monkeypatch, path, value,
):
    from xrd_tools.reduction import ReintegrateSuccessorPlan
    import xrd_tools.reduction.reintegrate_successor as module

    seeded = _seed_existing(tmp_path, labels=(2,), name="recipe-literals")
    recipe = _plan(seeded).as_recipe()
    selected = recipe
    for key in path[:-1]:
        selected = selected[key]
    if value == "NORMALIZATION_EQUIVALENT":
        output = Path(selected[path[-1]])
        value = str(output.parent / "folder" / ".." / output.name)
    selected[path[-1]] = value
    monkeypatch.setattr(
        module,
        "capture_finite_source",
        lambda *_args, **_kwargs: pytest.fail("recipe admission touched a file"),
    )

    with pytest.raises(ValueError):
        ReintegrateSuccessorPlan.from_recipe(recipe)


def test_click_preparation_api_requires_an_exact_integer_before_file_io(
    tmp_path, monkeypatch,
):
    from xrd_tools.reduction import ReintegrateSuccessorPlan
    import xrd_tools.reduction.reintegrate_successor as module

    seeded = _seed_existing(tmp_path, labels=(2,), name="click-api-literal")
    offer, _plan_value = _prepared_plan(seeded)
    preparation = _dimension_preparation(seeded, "1d")
    preparation["api_version"] = 1.0
    monkeypatch.setattr(
        module,
        "capture_finite_source",
        lambda *_args, **_kwargs: pytest.fail("click admission touched a file"),
    )

    with pytest.raises(module.PreparedCapsuleMiss) as captured:
        ReintegrateSuccessorPlan.from_prepared_capsule(
            offer,
            dimension="1d",
            preparation=preparation,
            expected_labels=seeded.labels,
        )
    assert captured.value.code.value == "REQUESTED_SCIENCE_UNSUPPORTED"


@pytest.mark.parametrize("constructor", ("direct", "fallback"))
@pytest.mark.parametrize("fault", ("cycle", "byte-limit"))
def test_bounded_legacy_preparation_refuses_before_artifact_inspection(
    tmp_path, monkeypatch, constructor, fault,
):
    from xrd_tools.reduction import ReintegrateSuccessorPlan
    import xrd_tools.reduction.reintegrate_successor as module

    seeded = _seed_existing(
        tmp_path, labels=(2,), name=f"legacy-boundary-{constructor}-{fault}",
    )
    preparation = _dimension_preparation(seeded, "1d")
    if fault == "cycle":
        cycle = []
        cycle.append(cycle)
        preparation["cycle"] = cycle
    else:
        monkeypatch.setattr(module, "_MAX_RECIPE_BYTES", 256)
    monkeypatch.setattr(
        module._legacy.ReintegratePlan,
        "from_artifact",
        lambda *_args, **_kwargs: pytest.fail("legacy artifact inspection ran"),
    )
    monkeypatch.setattr(
        module,
        "capture_finite_source",
        lambda *_args, **_kwargs: pytest.fail("finite capture ran"),
    )
    kwargs = {
        "entry": "entry",
        "dimension": "1d",
        "preparation": preparation,
        "expected_labels": seeded.labels,
    }

    with pytest.raises(ValueError, match="REINTEGRATE_PREPARATION_UNSUPPORTED"):
        if constructor == "direct":
            ReintegrateSuccessorPlan.from_artifact(seeded.target, **kwargs)
        else:
            ReintegrateSuccessorPlan.from_prepared_or_artifact(
                None, seeded.target, **kwargs,
            )


def test_recipe_replay_uses_only_the_detached_boundary_snapshot(
    tmp_path, monkeypatch,
):
    from xrd_tools.reduction import ReintegrateSuccessorPlan
    import xrd_tools.reduction.reintegrate_successor as module

    seeded = _seed_existing(tmp_path, labels=(2,), name="recipe-detach")
    recipe = _plan(seeded).as_recipe()
    expected = copy.deepcopy(recipe)
    boundary = module._bounded_recipe_snapshot

    def detach_then_mutate(value):
        snapshot = boundary(value)
        value["plan"]["qualification"]["entry"] = "mutated-after-boundary"
        value.clear()
        return snapshot

    monkeypatch.setattr(module, "_bounded_recipe_snapshot", detach_then_mutate)
    replay = ReintegrateSuccessorPlan.from_recipe(recipe)
    assert replay.as_recipe() == expected


def test_cancelled_successor_never_publishes_or_mutates_source(
    tmp_path, monkeypatch,
):
    from xrd_tools.reduction import run_reintegrate_successor

    seeded = _seed_existing(tmp_path, labels=(2, 5), name="cancel")
    before = seeded.target.read_bytes()
    plan = _plan(seeded)
    token = threading.Event()
    token.set()
    result = run_reintegrate_successor(plan, cancel_token=token)

    assert result.disposition == "ABORTED"
    assert result.committed_labels == ()
    assert result.terminal is None
    assert result.commit_identity is None
    assert not (seeded.target.parent / "immutable-successor.nexus").exists()
    assert seeded.target.read_bytes() == before


def test_candidate_owned_qualify_cancellation_returns_clean_abort(
    tmp_path, monkeypatch,
):
    from xrd_tools.reduction import run_reintegrate_successor

    seeded = _seed_existing(tmp_path, labels=(2, 5), name="qualify-cancel")
    before = seeded.target.read_bytes()
    plan = _plan(seeded)
    token = threading.Event()

    def cancel_at_qualify(progress):
        if progress.stage == "qualify" and progress.completed == 0:
            token.set()

    result = run_reintegrate_successor(
        plan,
        cancel_token=token,
        progress_cb=cancel_at_qualify,
    )
    assert result.disposition == "ABORTED"
    assert result.terminal is None
    assert result.commit_identity is None
    assert seeded.target.read_bytes() == before
    assert not Path(plan.output_artifact).exists()
    assert not tuple(tmp_path.glob(".xdart-finite-*.candidate"))


@pytest.mark.parametrize("recipe", (False, True), ids=("direct", "recipe"))
@pytest.mark.parametrize("preflight", ("return", "raise"))
def test_cancellation_immediately_after_preflight_precedes_source_expectations(
    tmp_path, monkeypatch, recipe, preflight,
):
    from xrd_tools.reduction import (
        ReintegrateSuccessorPlan,
        run_reintegrate_successor,
    )
    import xrd_tools.reduction.reintegrate_successor as module

    seeded = _seed_existing(
        tmp_path,
        labels=(2, 5),
        name=f"post-preflight-{recipe}-{preflight}",
    )
    _offer, plan = _prepared_plan(
        seeded,
        explicit_output=(
            seeded.target.parent / f"post-preflight-{recipe}-{preflight}.nexus"
        ),
    )
    if recipe:
        plan = ReintegrateSuccessorPlan.from_recipe(plan.as_recipe())
    before = seeded.target.read_bytes()
    token = threading.Event()

    def cancel(*_args, **_kwargs):
        token.set()
        if preflight == "raise":
            raise module._legacy.ReintegrateCancelled()

    monkeypatch.setattr(module, "preflight_prepared_execution", cancel)
    monkeypatch.setattr(
        module,
        "_source_expectations",
        lambda *_args, **_kwargs: pytest.fail(
            "source expectations ran after cancellation"
        ),
    )
    result = run_reintegrate_successor(plan, cancel_token=token)

    assert result.disposition == "ABORTED"
    assert result.terminal is None
    assert not Path(plan.output_artifact).exists()
    assert seeded.target.read_bytes() == before
    assert not tuple(seeded.target.parent.glob(".xdart-finite-*.candidate"))


@pytest.mark.parametrize("recipe", (False, True), ids=("direct", "recipe"))
def test_preset_cancellation_precedes_missing_source_revalidation(
    tmp_path, recipe,
):
    from xrd_tools.reduction import (
        ReintegrateSuccessorPlan,
        run_reintegrate_successor,
    )

    seeded = _seed_existing(
        tmp_path, labels=(2, 5), name=f"preset-missing-{recipe}",
    )
    plan = _plan(seeded)
    if recipe:
        plan = ReintegrateSuccessorPlan.from_recipe(plan.as_recipe())
    seeded.target.unlink()
    token = threading.Event()
    token.set()
    result = run_reintegrate_successor(plan, cancel_token=token)
    assert result.disposition == "ABORTED"
    assert result.terminal is None
    assert not Path(plan.output_artifact).exists()
    assert not tuple(tmp_path.glob(".xdart-finite-*.candidate"))


def test_integrator_failure_is_primary_and_never_masquerades_as_abort(
    tmp_path, monkeypatch,
):
    from xrd_tools.reduction import run_reintegrate_successor
    import xrd_tools.reduction.core as core

    seeded = _seed_existing(tmp_path, labels=(2, 5), name="integrator-failure")
    before = seeded.target.read_bytes()
    plan = _plan(seeded)
    failure = RuntimeError("injected pyFAI failure")

    def fail_integration(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(core, "integrate_1d", fail_integration)
    monkeypatch.setattr(core, "integrate_2d", fail_integration)

    with pytest.raises(RuntimeError, match="injected pyFAI failure") as raised:
        run_reintegrate_successor(plan)

    assert raised.value is failure
    assert not (seeded.target.parent / "immutable-successor.nexus").exists()
    assert seeded.target.read_bytes() == before


def test_postlink_hold_releases_old_run_and_exact_replay_reuses_successor(
    tmp_path, monkeypatch,
):
    from xrd_tools.io.finite_artifact import (
        FiniteArtifactIntegrityError,
        FiniteArtifactPublicationHeld,
    )
    from xrd_tools.reduction import (
        ReintegrateSuccessorPlan,
        run_reintegrate_successor,
    )
    import xrd_tools.reduction.reintegrate_successor as module

    seeded = _seed_existing(tmp_path, labels=(2, 5), name="held-replay")
    before = seeded.target.read_bytes()
    plan = _plan(seeded)
    recipe = plan.as_recipe()
    _stub_integrators(monkeypatch)
    inspect = module._inspect_committed
    attempts = 0

    def fail_first_inspection(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FiniteArtifactIntegrityError("injected post-link observation")
        return inspect(*args, **kwargs)

    monkeypatch.setattr(module, "_inspect_committed", fail_first_inspection)
    with pytest.raises(FiniteArtifactPublicationHeld):
        run_reintegrate_successor(plan)

    output = seeded.target.parent / "immutable-successor.nexus"
    assert output.exists()
    assert seeded.target.read_bytes() == before

    def no_second_write(*_args, **_kwargs):
        pytest.fail("exact ALREADY_COMMITTED replay performed candidate work")

    monkeypatch.setattr(module._SuccessorRuntime, "execute_candidate", no_second_write)
    direct = run_reintegrate_successor(plan)
    replay = run_reintegrate_successor(ReintegrateSuccessorPlan.from_recipe(recipe))

    assert direct.disposition == "ALREADY_COMMITTED"
    assert replay.disposition == "ALREADY_COMMITTED"
    assert direct.output_artifact == replay.output_artifact == str(output)
    assert seeded.target.read_bytes() == before


def test_prepared_manifest_receipt_is_factory_owned_and_dimension_bound(
    tmp_path,
):
    from xrd_tools.io.finite_artifact import capture_finite_source
    from xrd_tools.io.record_writer import (
        ReplacementManifestReceipt,
        prepare_replacement_manifest_receipt,
    )

    seeded = _seed_existing(tmp_path, labels=(2, 5), name="manifest-receipt")
    source = capture_finite_source(seeded.target).snapshot
    facts_digest = hashlib.sha256(b"prepared facts").hexdigest()
    with h5py.File(seeded.target, "r") as document:
        receipt = prepare_replacement_manifest_receipt(
            document,
            source,
            entry="entry",
            dimension="1d",
            facts_digest=facts_digest,
        )

    assert type(receipt) is ReplacementManifestReceipt
    assert receipt.source_snapshot is source
    assert receipt.dimension == "1d"
    assert receipt.facts_digest == facts_digest
    assert receipt.gi_name == "gi_mode_1d"
    assert len(receipt.exclusions) == 8
    with pytest.raises(TypeError, match="factory-owned"):
        replace(receipt)


def test_prepared_successor_uses_detached_facts_and_one_click_manifest(
    tmp_path, monkeypatch,
):
    from xrd_tools.io.record_writer import NexusRecordWriter
    from xrd_tools.reduction import run_reintegrate_successor
    import xrd_tools.reduction.reintegrate_successor as module

    seeded = _seed_existing(tmp_path, labels=(2, 5), name="prepared-run")
    before = seeded.target.read_bytes()
    _offer, plan = _prepared_plan(seeded)
    before = seeded.target.read_bytes()
    recipe = plan.as_recipe()
    _stub_integrators(monkeypatch)
    monkeypatch.setattr(
        module._legacy,
        "_inspect_artifact",
        lambda *_args, **_kwargs: pytest.fail("prepared click inspected artifact"),
    )
    monkeypatch.setattr(
        NexusRecordWriter,
        "_detach_replacement_fact",
        lambda *_args, **_kwargs: pytest.fail("prepared click detached a fact"),
    )
    manifest = NexusRecordWriter._replacement_manifest_digest
    calls = []

    def counted(writer, *args, **kwargs):
        calls.append(args[0])
        return manifest(writer, *args, **kwargs)

    monkeypatch.setattr(
        NexusRecordWriter, "_replacement_manifest_digest", counted,
    )
    result = run_reintegrate_successor(plan)

    assert result.disposition == "COMMITTED"
    assert result.committed_labels == seeded.labels
    assert len(calls) == 1
    assert seeded.target.read_bytes() == before
    assert recipe["plan"]["route"] == "prepared"


@pytest.mark.parametrize("recipe", (False, True), ids=("direct", "recipe"))
@pytest.mark.parametrize("position", (0, 1, 2), ids=("first", "middle", "last"))
def test_phase_b_prepared_fact_fault_is_transaction_integrity_not_route_rejection(
    tmp_path, monkeypatch, recipe, position,
):
    import xrd_tools.reduction.reintegrate_successor as module
    from xrd_tools.io.finite_artifact import FiniteArtifactIntegrityError
    from xrd_tools.reduction import (
        ReintegrateSuccessorPlan,
        run_reintegrate_successor,
    )

    seeded = _seed_existing(
        tmp_path, labels=(2, 5, 8), name=f"phase-b-fact-{recipe}-{position}",
    )
    _offer, plan = _prepared_plan(seeded)
    before = seeded.target.read_bytes()
    if recipe:
        plan = ReintegrateSuccessorPlan.from_recipe(plan.as_recipe())
    facts = dict(plan._prepared._facts_by_label)
    facts.pop(plan.labels[position])
    object.__setattr__(
        plan._prepared,
        "_facts_by_label",
        MappingProxyType(facts),
    )
    monkeypatch.setattr(
        module,
        "_legacy_successor_plan",
        lambda *_args, **_kwargs: pytest.fail(
            "phase-B evidence fault entered route fallback"
        ),
    )
    _stub_integrators(monkeypatch)
    with pytest.raises(
        FiniteArtifactIntegrityError,
        match="ARTIFACT_FACTS_CHANGED",
    ):
        run_reintegrate_successor(plan)
    assert seeded.target.read_bytes() == before
    assert not Path(plan.output_artifact).exists()
    assert not tuple(tmp_path.glob(".xdart-finite-*.candidate"))


@pytest.mark.parametrize("seam", ("boundary", "underlying", "finalize"))
@pytest.mark.parametrize("outcome", ("commit", "abort", "held"))
def test_external_publication_truth_survives_persistent_settlement_faults(
    tmp_path, monkeypatch, outcome, seam,
):
    import xrd_tools.reduction.reintegrate_successor as module
    from xrd_tools.io.finite_artifact import FiniteArtifactPublicationHeld
    from xrd_tools.reduction import run_reintegrate_successor
    from xrd_tools.session.dynamic_accounting import (
        DynamicRunAccounting,
        DynamicRunState,
        _DynamicWriterBoundary,
    )
    from xrd_tools.session.scan_session import ScanSession

    seeded = _seed_existing(
        tmp_path, labels=(2, 5), name=f"settlement-{outcome}-{seam}",
    )
    before = seeded.target.read_bytes()
    plan = _plan(seeded)
    token = threading.Event()
    captured = {}
    original_init = module._SuccessorRuntime.__init__

    def capture_runtime(runtime, *args, **kwargs):
        original_init(runtime, *args, **kwargs)
        captured["runtime"] = runtime

    monkeypatch.setattr(module._SuccessorRuntime, "__init__", capture_runtime)
    if seam == "boundary":
        method = {
            "commit": "session_finished",
            "abort": "epoch_aborted",
            "held": "publication_held",
        }[outcome]

        def persistent_boundary_fault(*_args, **_kwargs):
            raise OSError(f"persistent {outcome} accounting boundary fault")

        monkeypatch.setattr(
            _DynamicWriterBoundary, method, persistent_boundary_fault,
        )
    elif seam == "underlying":
        method = {
            "commit": "_finish_bound",
            "abort": "_abort_bound",
            "held": "_hold_bound",
        }[outcome]

        def persistent_underlying_fault(*_args, **_kwargs):
            raise OSError(f"persistent {outcome} underlying accounting fault")

        monkeypatch.setattr(
            DynamicRunAccounting, method, persistent_underlying_fault,
        )
    else:
        monkeypatch.setattr(
            ScanSession,
            "_final_sweep",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError(f"persistent {outcome} finalization fault")
            ),
        )
    if outcome == "abort":
        original_execute = module._SuccessorRuntime.execute_candidate

        def execute_then_cancel(runtime, *args, **kwargs):
            seal = original_execute(runtime, *args, **kwargs)
            token.set()
            return seal

        monkeypatch.setattr(
            module._SuccessorRuntime,
            "execute_candidate",
            execute_then_cancel,
        )
    elif outcome == "held":
        monkeypatch.setattr(
            module,
            "_inspect_committed",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ValueError("post-link terminal inspection is unavailable")
            ),
        )
    _stub_integrators(monkeypatch)
    if outcome == "held":
        with pytest.raises(FiniteArtifactPublicationHeld) as observed:
            run_reintegrate_successor(plan, cancel_token=token)
        assert observed.value.request.output_artifact == plan.output_artifact
        result = None
    else:
        result = run_reintegrate_successor(plan, cancel_token=token)
        assert result.disposition == (
            "COMMITTED" if outcome == "commit" else "ABORTED"
        )
        assert any("persistent" in item for item in result.diagnostics)
    runtime = captured["runtime"]
    expected_state = {
        "commit": DynamicRunState.FINISHED,
        "abort": DynamicRunState.ABORTED,
        "held": DynamicRunState.HELD,
    }[outcome]
    assert runtime.accounting.snapshot().state is expected_state
    assert runtime.session._dynamic_terminal_settled
    assert not runtime.session._dynamic_external_publication_pending
    assert runtime.session not in runtime.accounting.owner_census()
    assert runtime.source._topology is None
    assert seeded.target.read_bytes() == before
    assert Path(plan.output_artifact).exists() is (outcome != "abort")
    assert not tuple(tmp_path.glob(".xdart-finite-*.candidate"))


@pytest.mark.parametrize("dimension", ("1d", "2d"))
def test_unicode_single_epoch_append_is_prepared_and_preserved(
    tmp_path, monkeypatch, dimension,
):
    from xrd_tools.reduction import (
        prepare_reintegrate_bundle,
        run_reintegrate_successor,
    )

    seeded = _seed_existing(
        tmp_path,
        labels=(0, 1),
        name=f"unicodé-append-{dimension}",
    )
    source, expected_lineage = _install_prepared_append_lineage(seeded)
    offer = prepare_reintegrate_bundle(
        source,
        entry="entry",
        labels=seeded.labels,
        source_root=str(seeded.target.parent),
    )
    assert offer.disposition == "READY", offer.miss_code
    assert offer.bundle is not None
    assert offer.bundle.one_d.disposition == "READY"
    assert offer.bundle.two_d.disposition == "READY"
    output = seeded.target.parent / f"unicode-{dimension}.nexus"
    _offer, plan = _prepared_plan(
        seeded,
        offer=offer,
        dimension=dimension,
        explicit_output=output,
    )
    _stub_integrators(monkeypatch)
    result = run_reintegrate_successor(plan)
    assert result.disposition == "COMMITTED"
    with h5py.File(output, "r") as document:
        assert (
            document["entry/reduction/config/append_lineage"][()]
            == expected_lineage
        )


def test_multi_epoch_append_is_one_typed_prepared_miss(tmp_path):
    from xrd_tools.reduction import prepare_reintegrate_bundle

    seeded = _seed_existing(
        tmp_path, labels=(0, 1), name="multi-epoch-append",
    )
    source, raw = _install_prepared_append_lineage(
        seeded, multi_epoch=True,
    )
    assert len(json.loads(raw)["epochs"]) == 2
    offer = prepare_reintegrate_bundle(
        source,
        entry="entry",
        labels=seeded.labels,
        source_root=str(seeded.target.parent),
    )
    codes = (
        (offer.miss_code,)
        if offer.disposition == "MISS"
        else (offer.bundle.one_d.miss_code, offer.bundle.two_d.miss_code)
    )
    assert {code.value for code in codes} == {"APPEND_LINEAGE_UNSUPPORTED"}


def test_malformed_or_unaligned_append_prefix_is_one_whole_offer_miss(
    tmp_path,
):
    from xrd_tools.io.finite_artifact import capture_finite_source
    from xrd_tools.reduction import prepare_reintegrate_bundle

    for seam in (
        "empty-identity",
        "duplicate-mode",
        "unknown-mode",
        "disk-label-mismatch",
        "wrong-epoch-labels",
    ):
        seeded = _seed_existing(
            tmp_path, labels=(0, 1), name=f"append-refusal-{seam}",
        )
        _source, _raw = _install_prepared_append_lineage(seeded)
        with h5py.File(seeded.target, "r+") as document:
            config = document["entry/reduction/config"]
            node = config["append_lineage"]
            lineage = json.loads(node.asstr()[()])
            if seam == "empty-identity":
                lineage["source_identity"] = ""
            elif seam == "duplicate-mode":
                lineage["modes"] = [
                    lineage["modes"][0], lineage["modes"][0],
                ]
            elif seam == "unknown-mode":
                lineage["modes"] = ["unknown"]
            elif seam == "disk-label-mismatch":
                index = document["entry/integrated_1d/frame_index"]
                index[0] = int(index[0]) + 1
            else:
                lineage["epochs"][0]["labels"] = [0]
            if seam != "disk-label-mismatch":
                node[()] = json.dumps(
                    lineage, sort_keys=True, separators=(",", ":"),
                )
        source = capture_finite_source(seeded.target)
        offer = prepare_reintegrate_bundle(
            source,
            entry="entry",
            labels=seeded.labels,
            source_root=str(seeded.target.parent),
        )
        assert offer.disposition == "MISS", seam
        assert offer.miss_code.value == "APPEND_LINEAGE_UNSUPPORTED", seam


def test_prepared_recipe_replay_authenticates_without_file_io(
    tmp_path, monkeypatch,
):
    from xrd_tools.reduction import (
        PreparedRouteRejected,
        ReintegrateSuccessorPlan,
    )
    import xrd_tools.reduction.reintegrate_successor as module

    seeded = _seed_existing(tmp_path, labels=(2,), name="prepared-recipe")
    _offer, plan = _prepared_plan(seeded)
    recipe = plan.as_recipe()
    monkeypatch.setattr(
        module,
        "capture_finite_source",
        lambda *_args, **_kwargs: pytest.fail("prepared recipe touched a file"),
    )
    replay = ReintegrateSuccessorPlan.from_recipe(recipe)
    assert replay.as_recipe() == recipe

    mutated = copy.deepcopy(recipe)
    mutated["plan"]["execution"]["prepared"]["facts"][0]["path"] = str(
        (seeded.target.parent / "foreign.h5").resolve()
    )
    with pytest.raises(PreparedRouteRejected) as captured:
        ReintegrateSuccessorPlan.from_recipe(mutated)
    assert captured.value.code.value == "CAPSULE_SCHEMA_UNSUPPORTED"

    rerouted = copy.deepcopy(recipe)
    prepared = rerouted["plan"]["execution"]["prepared"]
    topology = prepared["topology"]
    for row in topology["payload"]["frame_routes"]:
        row["hdf"]["start"] += 1
        row["hdf"]["stop"] += 1
    topology["topology_digest"] = module._sha(topology["payload"])
    preimage = copy.deepcopy(prepared)
    preimage.pop("execution_digest")
    prepared["execution_digest"] = module._sha(preimage)
    with pytest.raises(PreparedRouteRejected) as captured:
        ReintegrateSuccessorPlan.from_recipe(rerouted)
    assert captured.value.code.value == "CAPSULE_SCHEMA_UNSUPPORTED"


def test_prepared_topology_accepts_factory_equivalent_parent_relative_link(
    tmp_path,
):
    import xrd_tools.reduction.reintegrate_prepared as module

    seeded = _seed_existing(tmp_path, labels=(2,), name="parent-link")
    _offer, plan = _prepared_plan(seeded)
    prepared = module.prepared_execution_mapping(plan._prepared)
    topology = prepared["topology"]
    link = topology["payload"]["external_signature"][0]
    link["lexical_filename"] = str(
        Path("unused-parent") / ".." / Path(link["lexical_filename"]).name
    )
    topology["topology_digest"] = module._digest(topology["payload"])

    _receipt, admitted = module._admit_topology(topology)
    assert admitted.external_signature[0].lexical_filename == (
        link["lexical_filename"]
    )


def test_self_external_hdf_graph_is_one_typed_preparation_miss(tmp_path):
    from xdart.gui.tabs.scattering.contracts import (
        ExternalSourceState,
        SourceExecutionStamp,
        SourceFileState,
    )
    from xrd_tools.io.finite_artifact import capture_finite_source
    from xrd_tools.reduction import prepare_reintegrate_bundle

    seeded = _seed_existing(tmp_path, labels=(0,), name="self-external")
    with h5py.File(seeded.source, "r+") as document:
        data = document["entry"].create_group("data")
        data["data_000001"] = h5py.ExternalLink(
            seeded.source.name,
            "/entry/instrument/detector/data",
        )
    state = SourceFileState.capture(seeded.source)
    execution = SourceExecutionStamp(
        state,
        "nexus_hdf5",
        1,
        0,
        external_members=(ExternalSourceState(
            state,
            "/entry/instrument/detector/data",
            0,
            1,
            0,
        ),),
    )
    encoded = json.dumps(
        execution.as_dict(), sort_keys=True, separators=(",", ":"),
    )
    with h5py.File(seeded.target, "r+") as document:
        detector = document["entry/instrument/detector"]
        if "mask" in detector:
            del detector["mask"]
        config = document["entry/reduction/config"]
        del config["source_execution"]
        config.create_dataset("source_execution", data=encoded)
        source = document["entry/frames/frame_0000/source"]
        for key, value in {
            "adapter_id": "nexus_hdf5",
            "file_size": state.size,
            "file_mtime_ns": state.mtime_ns,
            "frame_count": 1,
            "dataset_path": "/entry/data/data_000001",
            "self_contained": False,
        }.items():
            source.attrs.modify(key, value)

    artifact = capture_finite_source(seeded.target)
    offer = prepare_reintegrate_bundle(
        artifact,
        entry="entry",
        labels=(0,),
        source_root=str(seeded.target.parent),
    )
    assert offer.disposition == "MISS"
    assert offer.miss_code.value == "SOURCE_TOPOLOGY_UNSUPPORTED"


@pytest.mark.parametrize("regression", ("source", "member"))
def test_prepared_append_recipe_refuses_revision_regression(
    tmp_path, regression,
):
    from xrd_tools.reduction import PreparedRouteRejected
    from xrd_tools.reduction import (
        ReintegrateSuccessorPlan,
        prepare_reintegrate_bundle,
    )
    import xrd_tools.reduction.reintegrate_prepared as module

    seeded = _seed_existing(
        tmp_path, labels=(0, 1), name=f"append-regression-{regression}",
    )
    source, _raw = _install_prepared_append_lineage(seeded)
    offer = prepare_reintegrate_bundle(
        source,
        entry="entry",
        labels=seeded.labels,
        source_root=str(seeded.target.parent),
    )
    plan = ReintegrateSuccessorPlan.from_prepared_capsule(
        offer,
        dimension="1d",
        preparation=_dimension_preparation(seeded, "1d"),
        expected_labels=seeded.labels,
        explicit_output=seeded.target.parent / "append-regression.nexus",
    )
    prepared = module.prepared_execution_mapping(plan._prepared)
    topology = prepared["topology"]
    payload = topology["payload"]
    if regression == "source":
        payload["final_source"]["size"] = payload["execution"]["size"] - 1
        payload["lineage"]["epochs"][0]["source"]["size"] = (
            payload["final_source"]["size"]
        )
    else:
        payload["final_source"]["external_members"][0]["size"] = (
            payload["execution"]["external_members"][0]["file"]["size"] - 1
        )
        payload["lineage"]["epochs"][0]["source"][
            "external_members"
        ][0]["size"] = payload["final_source"]["external_members"][0][
            "size"
        ]
    payload["lineage_digest"] = module._legacy._digest(payload["lineage"])
    topology["topology_digest"] = module._digest(payload)

    with pytest.raises(PreparedRouteRejected) as captured:
        module._admit_topology(topology)
    assert captured.value.code.value == "CAPSULE_SCHEMA_UNSUPPORTED"


@pytest.mark.parametrize(
    ("dimension", "browse_labels", "ready", "miss"),
    (
        ("1d", (2, 5), "1d", "2d"),
        ("2d", (2, 5, 9), "2d", "1d"),
    ),
)
def test_prepared_dimensions_admit_independent_inventories(
    tmp_path, monkeypatch, dimension, browse_labels, ready, miss,
):
    from xrd_tools.reduction import (
        ReintegrateSuccessorPlan,
        prepare_reintegrate_bundle,
        run_reintegrate_successor,
    )

    seeded = _seed_existing(
        tmp_path, labels=(2, 5, 9), name=f"independent-{dimension}",
    )
    _drop_dimension_labels(seeded.target, "1d", (9,))
    source = _prepared_eligible_source(seeded)
    before = seeded.target.read_bytes()
    offer = prepare_reintegrate_bundle(
        source,
        entry="entry",
        labels=browse_labels,
        source_root=str(seeded.target.parent),
    )

    assert offer.disposition == "READY"
    admissions = {"1d": offer.bundle.one_d, "2d": offer.bundle.two_d}
    assert admissions[ready].disposition == "READY"
    assert admissions[miss].disposition == "MISS"
    assert admissions[miss].miss_code.value == "DIMENSION_LABELS_INCOMPATIBLE"

    output = (
        seeded.target.parent / f"independent-{dimension}-successor.nexus"
    )
    plan = ReintegrateSuccessorPlan.from_prepared_capsule(
        offer,
        dimension=dimension,
        preparation=_dimension_preparation(seeded, dimension),
        expected_labels=browse_labels,
        explicit_output=output,
    )
    _stub_integrators(monkeypatch)
    result = run_reintegrate_successor(plan)
    assert result.disposition == "COMMITTED"
    assert result.committed_labels == browse_labels
    assert seeded.target.read_bytes() == before
    sibling = "2d" if dimension == "1d" else "1d"
    expected_sibling = (2, 5, 9) if sibling == "2d" else (2, 5)
    with h5py.File(output, "r") as document:
        assert tuple(document[
            f"entry/integrated_{dimension}/frame_index"
        ][...]) == browse_labels
        assert tuple(document[
            f"entry/integrated_{sibling}/frame_index"
        ][...]) == expected_sibling


@pytest.mark.parametrize(
    ("one_d", "two_d", "expected"),
    (
        (
            (7, 9),
            (8, 9),
            ("DIMENSION_LABELS_INCOMPATIBLE",) * 2,
        ),
        (
            (7, 9),
            None,
            ("DIMENSION_LABELS_INCOMPATIBLE", "DIMENSION_UNAVAILABLE"),
        ),
        (
            None,
            (8, 9),
            ("DIMENSION_UNAVAILABLE", "DIMENSION_LABELS_INCOMPATIBLE"),
        ),
    ),
)
def test_prepared_fixed_bundle_retains_two_local_dimension_misses(
    tmp_path, monkeypatch, one_d, two_d, expected,
):
    from xrd_tools.reduction import prepare_reintegrate_bundle
    import xrd_tools.reduction.reintegrate as legacy

    seeded = _seed_existing(
        tmp_path,
        labels=(2, 5, 7, 8, 9),
        name="local-misses-both-mismatch",
    )
    browse_labels = (2, 5, 9)
    with h5py.File(seeded.target, "r+") as document:
        for dimension, labels in (("1d", one_d), ("2d", two_d)):
            if labels is None:
                del document[f"entry/integrated_{dimension}"]
    if one_d is not None:
        _drop_dimension_labels(seeded.target, "1d", one_d)
    if two_d is not None:
        _drop_dimension_labels(seeded.target, "2d", two_d)
    monkeypatch.setattr(
        legacy,
        "_canonical_acquisition_selected",
        lambda *_args, **_kwargs: pytest.fail(
            "dimension-neutral inspection selected an arbitrary BAI"
        ),
    )
    source = _prepared_eligible_source(seeded)
    offer = prepare_reintegrate_bundle(
        source,
        entry="entry",
        labels=browse_labels,
        source_root=str(seeded.target.parent),
    )

    assert offer.disposition == "READY"
    assert offer.bundle is not None
    for admission, code in zip(
        (offer.bundle.one_d, offer.bundle.two_d), expected,
    ):
        assert admission.disposition == "MISS"
        assert admission.miss_code.value == code


def test_noncurrent_missing_dimensions_are_not_relabelled_as_local_misses(
    tmp_path,
):
    from xrd_tools.reduction import prepare_reintegrate_bundle

    seeded = _seed_existing(
        tmp_path, labels=(2, 5, 9), name="noncurrent-missing-dimensions",
    )
    with h5py.File(seeded.target, "r+") as document:
        del document["entry/integrated_1d"]
        del document["entry/integrated_2d"]
    source = _prepared_eligible_source(seeded)
    offer = prepare_reintegrate_bundle(
        source,
        entry="entry",
        labels=seeded.labels,
        source_root=str(seeded.target.parent),
    )

    assert offer.disposition == "MISS"
    assert offer.bundle is None
    assert offer.miss_code.value == "TARGET_SCHEMA_UNSUPPORTED"


def test_irrelevant_mismatched_inventory_cannot_contaminate_ready_sibling(
    tmp_path,
):
    from xrd_tools.reduction import prepare_reintegrate_bundle

    seeded = _seed_existing(
        tmp_path, labels=(2, 5, 9), name="irrelevant-inventory",
    )
    with h5py.File(seeded.target, "r+") as document:
        document["entry/integrated_1d/frame_index"][-1] = 999
    source = _prepared_eligible_source(seeded)
    offer = prepare_reintegrate_bundle(
        source,
        entry="entry",
        labels=seeded.labels,
        source_root=str(seeded.target.parent),
    )

    assert offer.disposition == "READY"
    assert offer.bundle.one_d.disposition == "MISS"
    assert offer.bundle.one_d.miss_code.value == "DIMENSION_LABELS_INCOMPATIBLE"
    assert offer.bundle.two_d.disposition == "READY"


@pytest.mark.parametrize("route", ("prepared", "bounded-legacy"))
@pytest.mark.parametrize("terminal_kind", ("fast", "full"))
def test_fast_and_full_terminals_preserve_custody_across_recipe_replay(
    tmp_path, monkeypatch, route, terminal_kind,
):
    from xrd_tools.io.finite_artifact import capture_finite_source
    from xrd_tools.reduction import (
        ReintegrateSuccessorPlan,
        prepare_reintegrate_bundle,
        run_reintegrate_successor,
    )
    import xrd_tools.reduction.reintegrate_successor as module

    seeded = _seed_existing(
        tmp_path, labels=(2, 5), name=f"terminal-{route}-{terminal_kind}",
    )
    if route == "prepared":
        source = _prepared_eligible_source(seeded)
    else:
        source = capture_finite_source(seeded.target)
    digest = (
        source.snapshot.digest if terminal_kind == "full" else "a" * 64
    )
    terminal = _stat_terminal(source.snapshot, digest)
    output = seeded.target.parent / f"terminal-{route}-{terminal_kind}.nexus"
    if route == "prepared":
        offer = prepare_reintegrate_bundle(
            source,
            entry="entry",
            labels=seeded.labels,
            expected_terminal=terminal,
            source_root=str(seeded.target.parent),
        )
        assert offer.disposition == "READY"
        plan = ReintegrateSuccessorPlan.from_prepared_capsule(
            offer,
            dimension="1d",
            preparation=_dimension_preparation(seeded, "1d"),
            expected_labels=seeded.labels,
            explicit_output=output,
        )
    else:
        plan = ReintegrateSuccessorPlan.from_artifact(
            seeded.target,
            entry="entry",
            dimension="1d",
            preparation=_dimension_preparation(seeded, "1d"),
            expected_terminal_identity=terminal,
            expected_labels=seeded.labels,
            explicit_output=output,
        )

    before = seeded.target.read_bytes()
    replay = ReintegrateSuccessorPlan.from_recipe(plan.as_recipe())
    assert plan.expected_terminal == replay.expected_terminal == terminal
    direct_request = module._request_for_plan(plan)[1]
    replay_request = module._request_for_plan(replay)[1]
    assert direct_request == replay_request
    expected_lineage_terminal = terminal if terminal_kind == "full" else None
    assert direct_request.predecessor.terminal == expected_lineage_terminal
    assert replay_request.predecessor.terminal == expected_lineage_terminal

    _stub_integrators(monkeypatch)
    first = run_reintegrate_successor(plan)
    second = run_reintegrate_successor(replay)
    assert first.disposition == "COMMITTED"
    assert second.disposition == "ALREADY_COMMITTED"
    assert seeded.target.read_bytes() == before


@pytest.mark.parametrize("dimension", ("1d", "2d"))
def test_prepared_recipe_keeps_changed_click_science_separate_from_evidence(
    tmp_path, monkeypatch, dimension,
):
    from xrd_tools.reduction import (
        ReintegrateSuccessorPlan,
        run_reintegrate_successor,
    )
    import xrd_tools.reduction.reintegrate_successor as module

    seeded = _seed_existing(
        tmp_path, labels=(2, 5), name=f"changed-science-{dimension}",
    )
    preparation = _dimension_preparation(seeded, dimension)
    key = "npt" if dimension == "1d" else "npt_rad"
    preparation["selected_plan"]["bai_args"][key] += 1
    _offer, plan = _prepared_plan(
        seeded,
        dimension=dimension,
        preparation=preparation,
        explicit_output=(
            seeded.target.parent / f"changed-science-{dimension}.nexus"
        ),
    )
    replay = ReintegrateSuccessorPlan.from_recipe(plan.as_recipe())

    assert replay.as_recipe() == plan.as_recipe()
    assert replay.selected_plan["bai_args"][key] == (
        plan.selected_plan["bai_args"][key]
    )
    assert module._request_for_plan(replay)[1] == module._request_for_plan(plan)[1]
    _stub_integrators(monkeypatch)
    first = run_reintegrate_successor(plan)
    second = run_reintegrate_successor(replay)
    assert first.disposition == "COMMITTED"
    assert second.disposition == "ALREADY_COMMITTED"


@pytest.mark.parametrize("boundary", ["bundle", "execution"])
@pytest.mark.parametrize("fault", ["cycle", "byte-limit"])
def test_direct_prepared_admission_refuses_before_canonicalization(
    tmp_path, monkeypatch, boundary, fault,
):
    import xrd_tools.reduction.reintegrate_prepared as module

    seeded = _seed_existing(
        tmp_path, labels=(2,), name=f"prepared-{boundary}-{fault}",
    )
    offer, plan = _prepared_plan(seeded)
    if boundary == "bundle":
        mapping = module.prepared_bundle_mapping(offer.bundle)
        admit = module.admit_prepared_bundle
        limit = "MAX_PREPARED_BUNDLE_BYTES"
    else:
        mapping = module.prepared_execution_mapping(plan._prepared)
        admit = module.admit_prepared_execution
        limit = "MAX_PREPARED_EXECUTION_BYTES"
    if fault == "cycle":
        cycle = []
        cycle.append(cycle)
        mapping["facts"][0]["snapshot"]["cycle"] = cycle
        expected = module.PreparedCapsuleMissCode.CAPSULE_SCHEMA_UNSUPPORTED
    else:
        monkeypatch.setattr(module, limit, 256)
        expected = module.PreparedCapsuleMissCode.CAPSULE_BYTE_LIMIT
    monkeypatch.setattr(
        module,
        "_canonical_bytes",
        lambda *_args, **_kwargs: pytest.fail(
            "prepared admission canonicalized before its bounded boundary"
        ),
    )
    with pytest.raises(module.PreparedRouteRejected) as captured:
        admit(mapping)
    assert captured.value.code is expected


def test_prepared_and_manifest_versions_require_exact_integer_literals(
    tmp_path,
):
    import xrd_tools.io.record_writer as writer
    import xrd_tools.reduction.reintegrate_prepared as module

    seeded = _seed_existing(tmp_path, labels=(2,), name="prepared-versions")
    offer, plan = _prepared_plan(seeded)
    execution = module.prepared_execution_mapping(plan._prepared)
    bundle = module.prepared_bundle_mapping(offer.bundle)

    cases = [
        ("execution", execution, module.admit_prepared_execution),
        ("bundle", bundle, module.admit_prepared_bundle),
        ("target", execution["target"], module._admit_target),
        ("artifact", execution["artifact"], module._admit_artifact),
        (
            "selected-admission",
            execution["selected_admission"],
            module._admit_admission,
        ),
        (
            "selected-payload",
            execution["selected_admission"]["payload"],
            module._admit_payload,
        ),
        ("bundle-one-d", bundle["one_d"], module._admit_admission),
        ("bundle-two-d", bundle["two_d"], module._admit_admission),
    ]
    for role, original, admit in cases:
        admit(copy.deepcopy(original))
        mutated = copy.deepcopy(original)
        mutated["version"] = 1.0
        with pytest.raises(
            (module.PreparedRouteRejected, TypeError, ValueError),
        ) as captured:
            admit(mutated)
        assert captured.value is not None, role

    topology = copy.deepcopy(execution["topology"])
    module._admit_topology(copy.deepcopy(topology))
    topology["payload"]["execution"]["version"] = 1.0
    topology["topology_digest"] = module._digest(topology["payload"])
    with pytest.raises(module.PreparedRouteRejected):
        module._admit_topology(topology)

    malformed_execution = copy.deepcopy(execution["topology"])
    payload = malformed_execution["payload"]
    del payload["execution"]["adapter_id"]
    payload["execution_digest"] = module._legacy._digest(payload["execution"])
    malformed_execution["topology_digest"] = module._digest(payload)
    with pytest.raises(module.PreparedRouteRejected) as captured:
        module._admit_topology(malformed_execution)
    assert captured.value.code.value == "CAPSULE_SCHEMA_UNSUPPORTED"

    receipt = copy.deepcopy(
        execution["selected_admission"]["payload"]["manifest_receipt"]
    )
    writer.admit_replacement_manifest_receipt(copy.deepcopy(receipt))
    for field, value in (("version", 1.0), ("manifest_algorithm_version", 2.0)):
        mutated = copy.deepcopy(receipt)
        mutated[field] = value
        with pytest.raises(TypeError):
            writer.admit_replacement_manifest_receipt(mutated)
    node_names = (
        "gi_node_signature",
        "gi_structure_signature",
        "selected_bai_signature",
        "selected_audit_signature",
        "source_execution_signature",
        "append_lineage_signature",
    )
    for name in node_names:
        node = copy.deepcopy(receipt[name])
        writer._admit_node_digest(copy.deepcopy(node))
        node["version"] = 1.0
        with pytest.raises(TypeError):
            writer._admit_node_digest(node)


@pytest.mark.parametrize(
    "boundary",
    (
        "artifact-retained-mask",
        "artifact-mask-decode",
        "fact-label",
        "route-label",
        "route-snapshot-count",
        "route-self-contained",
        "route-hdf-start",
        "route-hdf-stop",
        "route-source-state",
        "revision-state",
    ),
)
def test_prepared_admission_refuses_bool_integer_aliases(tmp_path, boundary):
    import xrd_tools.reduction.reintegrate_prepared as module

    labels = (1,) if boundary == "fact-label" else (0,)
    seeded = _seed_existing(
        tmp_path, labels=labels, name=f"bool-alias-{boundary}",
    )
    _offer, plan = _prepared_plan(seeded)
    mapping = module.prepared_execution_mapping(plan._prepared)

    if boundary.startswith("artifact-"):
        artifact = mapping["artifact"]
        field = (
            "retained_mask_bytes"
            if boundary == "artifact-retained-mask"
            else "mask_decode_bytes"
        )
        artifact[field] = False
        preimage = copy.deepcopy(artifact)
        preimage.pop("artifact_digest")
        artifact["artifact_digest"] = module._digest(preimage)
        admit = lambda: module._admit_artifact(artifact)
    elif boundary == "fact-label":
        mapping["facts"][0]["label"] = True
        admit = lambda: module._admit_facts(
            mapping["facts"], labels, plan._prepared._inspection.topology,
        )
    else:
        topology = mapping["topology"]
        payload = topology["payload"]
        route = payload["frame_routes"][0]
        if boundary == "route-label":
            route["label"] = False
        elif boundary == "route-snapshot-count":
            assert route["snapshot_count"] == 1
            route["snapshot_count"] = True
        elif boundary == "route-self-contained":
            assert route["self_contained"] is False
            route["self_contained"] = 0
        elif boundary == "route-hdf-start":
            assert route["hdf"]["start"] == 0
            route["hdf"]["start"] = False
        elif boundary == "route-hdf-stop":
            assert route["hdf"]["stop"] == 1
            route["hdf"]["stop"] = True
        elif boundary == "route-source-state":
            assert route["source_state"]["first_label"] == 0
            route["source_state"]["first_label"] = False
        else:
            revision = next(
                row for row in payload["revisions"]
                if row["value"][2] == "source_file"
            )
            assert revision["value"][1]["first_label"] == 0
            revision["value"][1]["first_label"] = False
        topology["topology_digest"] = module._digest(payload)
        admit = lambda: module._admit_topology(topology)

    with pytest.raises(module.PreparedRouteRejected) as captured:
        admit()
    assert captured.value.code.value == "CAPSULE_SCHEMA_UNSUPPORTED"


def test_prepared_target_joins_canonical_snapshot_and_terminal_revision(
    tmp_path,
):
    import stat
    from xrd_tools.reduction import prepare_reintegrate_bundle
    import xrd_tools.reduction.reintegrate_prepared as module

    seeded = _seed_existing(tmp_path, labels=(2,), name="prepared-target-join")
    source = _prepared_eligible_source(seeded)
    terminal = _stat_terminal(source.snapshot, "a" * 64)
    offer = prepare_reintegrate_bundle(
        source,
        entry="entry",
        labels=seeded.labels,
        expected_terminal=terminal,
        source_root=str(seeded.target.parent),
    )
    target = module.prepared_bundle_mapping(offer.bundle)["target"]

    def rehash(value):
        preimage = copy.deepcopy(value)
        preimage.pop("target_digest")
        value["target_digest"] = module._digest(preimage)

    changed_digest = copy.deepcopy(target)
    changed_digest["terminal"]["digest"] = "b" * 64
    rehash(changed_digest)
    assert module._admit_target(changed_digest).terminal.digest == "b" * 64

    terminal_mutations = (
        ("target", str(Path(target["snapshot"]["path"]).parent / "x" / ".." / Path(target["snapshot"]["path"]).name)),
        ("size", target["terminal"]["size"] + 1),
        ("device", target["terminal"]["device"] + 1),
        ("inode", target["terminal"]["inode"] + 1),
        ("mtime_ns", target["terminal"]["mtime_ns"] + 1),
        ("ctime_ns", target["terminal"]["ctime_ns"] + 1),
    )
    for field, value in terminal_mutations:
        mutated = copy.deepcopy(target)
        mutated["terminal"][field] = value
        rehash(mutated)
        with pytest.raises(module.PreparedRouteRejected) as captured:
            module._admit_target(mutated)
        assert captured.value.code.value == "CAPSULE_DIGEST_MISMATCH", field

    snapshot_mutations = (
        ("path", "relative.nexus"),
        (
            "path",
            str(
                Path(target["snapshot"]["path"]).parent
                / "x" / ".." / Path(target["snapshot"]["path"]).name
            ),
        ),
        ("mode", stat.S_IFDIR | 0o755),
    )
    for field, value in snapshot_mutations:
        mutated = copy.deepcopy(target)
        mutated["snapshot"][field] = value
        if field == "path":
            mutated["terminal"]["target"] = value
        rehash(mutated)
        with pytest.raises(module.PreparedRouteRejected) as captured:
            module._admit_target(mutated)
        assert captured.value.code.value == "CAPSULE_SCHEMA_UNSUPPORTED", field


@pytest.mark.parametrize("fault", ("snapshot-schema", "foreign-path"))
def test_prepared_fact_schema_and_route_are_admitted_before_execution(
    tmp_path, fault,
):
    import xrd_tools.reduction.reintegrate_prepared as module

    seeded = _seed_existing(
        tmp_path, labels=(2,), name=f"prepared-fact-{fault}",
    )
    _offer, plan = _prepared_plan(seeded)
    mapping = module.prepared_execution_mapping(plan._prepared)
    if fault == "snapshot-schema":
        mapping["facts"][0]["snapshot"]["foreign"] = 1
    else:
        mapping["facts"][0]["path"] = str(
            (seeded.target.parent / "foreign.h5").resolve()
        )

    with pytest.raises(module.PreparedRouteRejected) as captured:
        module.admit_prepared_execution(mapping)
    assert captured.value.code.value == "CAPSULE_SCHEMA_UNSUPPORTED"


def test_prepared_nested_byte_ceilings_admit_exact_and_refuse_boundary_plus_one(
    tmp_path, monkeypatch,
):
    import xrd_tools.reduction.reintegrate_prepared as module

    seeded = _seed_existing(tmp_path, labels=(2, 5), name="prepared-ceilings")
    _offer, plan = _prepared_plan(seeded)
    mapping = module.prepared_execution_mapping(plan._prepared)
    topology = plan._prepared._inspection.topology

    facts_size = len(module._canonical_bytes(mapping["facts"]))
    monkeypatch.setattr(module, "MAX_PREPARED_FACTS_BYTES", facts_size)
    module._admit_facts(mapping["facts"], plan.labels, topology)
    monkeypatch.setattr(module, "MAX_PREPARED_FACTS_BYTES", facts_size - 1)
    with pytest.raises(module.PreparedRouteRejected) as captured:
        module._admit_facts(mapping["facts"], plan.labels, topology)
    assert captured.value.code.value == "CAPSULE_BYTE_LIMIT"

    admission = mapping["selected_admission"]
    admission_size = len(module._canonical_bytes(admission))
    monkeypatch.setattr(
        module, "MAX_PREPARED_DIMENSION_BYTES", admission_size,
    )
    module._admit_admission(copy.deepcopy(admission))
    module._ready_admission(plan._prepared.selected_admission.payload)
    monkeypatch.setattr(
        module, "MAX_PREPARED_DIMENSION_BYTES", admission_size - 1,
    )
    with pytest.raises(module.PreparedRouteRejected) as captured:
        module._admit_admission(copy.deepcopy(admission))
    assert captured.value.code.value == "CAPSULE_BYTE_LIMIT"
    with pytest.raises(module.PreparedCapsuleMiss) as captured:
        module._ready_admission(plan._prepared.selected_admission.payload)
    assert captured.value.code.value == "CAPSULE_BYTE_LIMIT"


@pytest.mark.parametrize("payload", ("execution", "lineage"))
def test_prepared_topology_factory_enforces_each_nested_ceiling(
    tmp_path, monkeypatch, payload,
):
    from xrd_tools.reduction import (
        ReintegrateSuccessorPlan,
        prepare_reintegrate_bundle,
    )
    import xrd_tools.reduction.reintegrate_prepared as module

    seeded = _seed_existing(
        tmp_path,
        labels=((0, 1) if payload == "lineage" else (2, 5)),
        name=f"topology-ceiling-{payload}",
    )
    if payload == "lineage":
        source, _raw = _install_prepared_append_lineage(seeded)
        offer = prepare_reintegrate_bundle(
            source,
            entry="entry",
            labels=seeded.labels,
            source_root=str(seeded.target.parent),
        )
        plan = ReintegrateSuccessorPlan.from_prepared_capsule(
            offer,
            dimension="1d",
            preparation=_dimension_preparation(seeded, "1d"),
            expected_labels=seeded.labels,
            explicit_output=(
                seeded.target.parent / "topology-lineage-successor.nexus"
            ),
        )
    else:
        _offer, plan = _prepared_plan(seeded)
    topology = plan._prepared._inspection.topology
    value = (
        topology.execution if payload == "execution" else topology.lineage
    )
    assert value is not None
    size = len(module._canonical_bytes(value))
    limit = (
        "MAX_PREPARED_SOURCE_EXECUTION_BYTES"
        if payload == "execution" else "MAX_PREPARED_APPEND_LINEAGE_BYTES"
    )
    monkeypatch.setattr(module, limit, size)
    module._topology_receipt(topology)
    monkeypatch.setattr(module, limit, size - 1)
    with pytest.raises(module.PreparedCapsuleMiss) as captured:
        module._topology_receipt(topology)
    assert captured.value.code.value == "CAPSULE_BYTE_LIMIT"


def test_decoded_source_path_has_an_exact_preparation_boundary(
    tmp_path, monkeypatch,
):
    from xrd_tools.reduction import prepare_reintegrate_bundle
    import xrd_tools.reduction.reintegrate_prepared as module

    seeded = _seed_existing(tmp_path, labels=(2,), name="prepared-path-limit")
    source = _prepared_eligible_source(
        seeded,
        master_name=f"raw-master-{'x' * 64}.h5",
    )
    before = seeded.target.read_bytes()
    baseline = prepare_reintegrate_bundle(
        source,
        entry="entry",
        labels=seeded.labels,
        source_root=str(seeded.target.parent),
    )
    fact_path = baseline.bundle.facts[0]["path"]
    size = len(fact_path.encode("utf-8"))
    assert size > len(str(seeded.target.parent).encode("utf-8"))

    monkeypatch.setattr(module, "MAX_PREPARED_TEXT_BYTES", size)
    exact = prepare_reintegrate_bundle(
        source,
        entry="entry",
        labels=seeded.labels,
        source_root=str(seeded.target.parent),
    )
    assert exact.disposition == "READY"
    monkeypatch.setattr(module, "MAX_PREPARED_TEXT_BYTES", size - 1)
    refused = prepare_reintegrate_bundle(
        source,
        entry="entry",
        labels=seeded.labels,
        source_root=str(seeded.target.parent),
    )
    assert refused.disposition == "MISS"
    assert refused.miss_code.value == "CAPSULE_BYTE_LIMIT"
    assert seeded.target.read_bytes() == before


def test_manifest_exclusions_are_a_deterministic_semantic_join(
    tmp_path,
):
    import xrd_tools.reduction.reintegrate_prepared as module

    seeded = _seed_existing(tmp_path, labels=(2,), name="exclusion-join")
    _offer, plan = _prepared_plan(seeded)
    mapping = module.prepared_execution_mapping(plan._prepared)
    admission = mapping["selected_admission"]
    payload = admission["payload"]
    receipt = payload["manifest_receipt"]
    receipt["exclusions"].append("/entry/foreign")
    receipt_preimage = copy.deepcopy(receipt)
    receipt_preimage.pop("receipt_digest")
    receipt["receipt_digest"] = module._digest(receipt_preimage)
    payload_preimage = copy.deepcopy(payload)
    payload_preimage.pop("dimension_payload_digest")
    payload["dimension_payload_digest"] = module._digest(payload_preimage)
    admission["dimension_payload_digest"] = payload[
        "dimension_payload_digest"
    ]
    admission["admission_digest"] = module._digest(
        module._admission_preimage(
            admission["dimension"],
            "READY",
            None,
            admission["dimension_payload_digest"],
        )
    )
    one_digest, two_digest = (
        (admission["admission_digest"], mapping["sibling_commitment"]["admission_digest"])
        if admission["dimension"] == "1d"
        else (mapping["sibling_commitment"]["admission_digest"], admission["admission_digest"])
    )
    mapping["bundle_digest"] = module._bundle_root(
        tuple(mapping["labels"]),
        mapping["facts_digest"],
        one_digest,
        two_digest,
    )
    execution_preimage = copy.deepcopy(mapping)
    execution_preimage.pop("execution_digest")
    mapping["execution_digest"] = module._digest(execution_preimage)

    with pytest.raises(module.PreparedRouteRejected) as captured:
        module.admit_prepared_execution(mapping)
    assert captured.value.code.value == "CAPSULE_DIGEST_MISMATCH"


def test_manifest_receipt_refuses_a_foreign_document_snapshot(tmp_path):
    from xrd_tools.io.finite_artifact import capture_finite_source
    from xrd_tools.io.record_writer import (
        WriterStateError,
        prepare_replacement_manifest_receipt,
    )

    first = _seed_existing(tmp_path, labels=(2,), name="receipt-first")
    second = _seed_existing(tmp_path, labels=(2,), name="receipt-second")
    foreign = capture_finite_source(second.target).snapshot
    with h5py.File(first.target, "r") as document:
        with pytest.raises(WriterStateError, match="differs"):
            prepare_replacement_manifest_receipt(
                document,
                foreign,
                entry="entry",
                dimension="1d",
                facts_digest=hashlib.sha256(b"facts").hexdigest(),
            )


@pytest.mark.parametrize(
    ("dimension", "first_route"),
    [
        ("1d", "bounded-legacy"),
        ("1d", "prepared"),
        ("2d", "bounded-legacy"),
        ("2d", "prepared"),
    ],
)
def test_cross_route_identity_content_and_exact_reuse(
    tmp_path, monkeypatch, dimension, first_route,
):
    import xrd_tools.reduction.reintegrate_successor as module
    from xrd_tools.reduction import run_reintegrate_successor

    seeded = _seed_existing(
        tmp_path, labels=(2, 5), name=f"parity-{dimension}-{first_route}",
    )
    output = seeded.target.parent / f"parity-{dimension}.nexus"
    _offer, prepared = _prepared_plan(
        seeded, dimension=dimension, explicit_output=output,
    )
    legacy = _plan(
        seeded,
        dimension=dimension,
        explicit_output=output,
        expected_terminal=None,
    )
    assert prepared.version_identity == legacy.version_identity
    assert prepared.publication_identity == legacy.publication_identity
    assert prepared.lineage_identity == legacy.lineage_identity
    assert prepared.science_identity == legacy.science_identity
    assert prepared.route_identity != legacy.route_identity
    assert prepared.operation_context_identity != legacy.operation_context_identity
    assert prepared.operation_identity != legacy.operation_identity

    plans = {"prepared": prepared, "bounded-legacy": legacy}
    _stub_integrators(monkeypatch)
    first = run_reintegrate_successor(plans[first_route])
    first_bytes = output.read_bytes()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("exact cross-route reuse rebuilt a candidate")

    monkeypatch.setattr(module._SuccessorRuntime, "execute_candidate", forbidden)
    second_route = (
        "prepared" if first_route == "bounded-legacy" else "bounded-legacy"
    )
    second = run_reintegrate_successor(plans[second_route])
    assert first.disposition == "COMMITTED"
    assert second.disposition == "ALREADY_COMMITTED"
    assert second.commit_identity == first.commit_identity
    assert output.read_bytes() == first_bytes


@pytest.mark.parametrize("append", (False, True), ids=("plain", "append"))
@pytest.mark.parametrize("dimension", ("1d", "2d"))
def test_cross_route_science_and_preservation_are_exactly_equal(
    tmp_path, monkeypatch, dimension, append,
):
    from xrd_tools.io.record_writer import (
        _replacement_exclusions_for,
        _replacement_manifest_digest_for,
    )
    from xrd_tools.reduction import (
        prepare_reintegrate_bundle,
        run_reintegrate_successor,
    )

    seeded = _seed_existing(
        tmp_path,
        labels=(0, 1) if append else (2, 5),
        name=f"science-parity-{dimension}-{append}",
    )
    prepared_output = seeded.target.parent / f"prepared-{dimension}.nexus"
    legacy_output = seeded.target.parent / f"legacy-{dimension}.nexus"
    offer = None
    if append:
        source, _lineage = _install_prepared_append_lineage(seeded)
        offer = prepare_reintegrate_bundle(
            source,
            entry="entry",
            labels=seeded.labels,
            source_root=str(seeded.target.parent),
        )
        assert offer.disposition == "READY", offer.miss_code
    _offer, prepared = _prepared_plan(
        seeded,
        dimension=dimension,
        explicit_output=prepared_output,
        offer=offer,
    )
    legacy = _plan(
        seeded,
        dimension=dimension,
        explicit_output=legacy_output,
        expected_terminal=None,
    )
    _stub_integrators(monkeypatch)
    prepared_result = run_reintegrate_successor(prepared)
    legacy_result = run_reintegrate_successor(legacy)
    assert prepared_result.committed_labels == legacy_result.committed_labels
    assert prepared_result.science_identity == legacy_result.science_identity
    assert prepared_result.version_identity == legacy_result.version_identity

    observations = []
    for output in (prepared_output, legacy_output):
        with h5py.File(output, "r") as document:
            group = document[f"entry/integrated_{dimension}"]
            arrays = {
                name: (
                    np.asarray(group[name][...]).dtype.str,
                    np.asarray(group[name][...]).shape,
                    hashlib.sha256(
                        np.asarray(group[name][...]).tobytes()
                    ).hexdigest(),
                )
                for name in ("frame_index", "q", "chi", "intensity", "sigma")
                if name in group
            }
            entry = document["entry"]
            exclusions = _replacement_exclusions_for(
                entry, dimension, include_finite_lineage=True,
            )
            preserved = _replacement_manifest_digest_for(
                document,
                "entry",
                exclusions,
                ignore_source_base=True,
                ignore_file_name=True,
            )
            config = document["entry/reduction/config"]
            observations.append((
                arrays,
                preserved,
                config[f"dimension_replacement_{dimension}"].asstr()[()],
                config["source_execution"].asstr()[()],
                None
                if "append_lineage" not in config
                else config["append_lineage"].asstr()[()],
                config[
                    f"dimension_replacement_{dimension}_result_seal"
                ].asstr()[()],
            ))
    assert observations[0] == observations[1]


def test_candidate_readonly_validation_recomputes_preservation(
    tmp_path, monkeypatch,
):
    import xrd_tools.io.record_writer as writer_module
    from xrd_tools.io.finite_artifact import FiniteArtifactIntegrityError
    from xrd_tools.reduction import run_reintegrate_successor

    seeded = _seed_existing(
        tmp_path, labels=(2, 5), name="candidate-preservation",
    )
    before = seeded.target.read_bytes()
    plan = _plan(seeded)
    _stub_integrators(monkeypatch)
    original = writer_module._replacement_manifest_digest_for
    candidate_reads = 0

    def mismatched(document, *args, **kwargs):
        nonlocal candidate_reads
        observed = original(document, *args, **kwargs)
        if document.mode != "r":
            return observed
        try:
            descriptor = document.id.get_vfd_handle()
        except RuntimeError:
            candidate = True
        else:
            candidate = (
                type(descriptor) is int
                and os.fstat(descriptor).st_ino != plan.source_snapshot.inode
            )
        if candidate:
            candidate_reads += 1
            return "0" * 64
        return observed

    monkeypatch.setattr(
        writer_module, "_replacement_manifest_digest_for", mismatched,
    )
    with pytest.raises(FiniteArtifactIntegrityError, match="preserved"):
        run_reintegrate_successor(plan)
    assert candidate_reads == 1
    assert seeded.target.read_bytes() == before
    assert not (seeded.target.parent / "immutable-successor.nexus").exists()
    assert not tuple(seeded.target.parent.glob(".xdart-finite-*.candidate"))


@pytest.mark.parametrize(
    "seam", ("prepublish", "postlink", "existing-reuse"),
)
def test_raw_source_drift_has_exact_publication_outcome(
    tmp_path, monkeypatch, seam,
):
    import xrd_tools.io.finite_artifact as finite_module
    import xrd_tools.reduction.reintegrate_successor as module
    from xrd_tools.io.finite_artifact import (
        FiniteArtifactCollision,
        FiniteArtifactIntegrityError,
    )
    from xrd_tools.reduction import run_reintegrate_successor

    seeded = _seed_existing(
        tmp_path, labels=(2, 5), name=f"raw-drift-{seam}",
    )
    output = seeded.target.parent / "raw-drift-successor.nexus"
    _offer, plan = _prepared_plan(seeded, explicit_output=output)
    member = seeded.source.with_name("raw-member.h5")
    _stub_integrators(monkeypatch)

    def mutate_member():
        with h5py.File(member, "r+") as document:
            data = document["entry/instrument/detector/data"]
            data[0, 0, 0] = int(data[0, 0, 0]) + 1

    if seam == "prepublish":
        original = module._require_terminal_source_topology

        def drift_then_validate(inspected, cancel_token=None):
            mutate_member()
            return original(inspected, cancel_token)

        monkeypatch.setattr(
            module, "_require_terminal_source_topology", drift_then_validate,
        )
        with pytest.raises(FiniteArtifactIntegrityError, match="topology"):
            run_reintegrate_successor(plan)
        assert not output.exists()
    elif seam == "postlink":
        original = finite_module._link

        def link_then_drift(*args, **kwargs):
            original(*args, **kwargs)
            mutate_member()

        monkeypatch.setattr(finite_module, "_link", link_then_drift)
        result = run_reintegrate_successor(plan)
        assert result.disposition == "COMMITTED"
        assert output.exists()
    else:
        first = run_reintegrate_successor(plan)
        first_bytes = output.read_bytes()
        original = module._inspect_committed

        def inspect_then_drift(*args, **kwargs):
            observed = original(*args, **kwargs)
            mutate_member()
            return observed

        def forbidden(*_args, **_kwargs):
            raise AssertionError("existing replay rebuilt a candidate")

        monkeypatch.setattr(module, "_inspect_committed", inspect_then_drift)
        monkeypatch.setattr(
            module._SuccessorRuntime, "execute_candidate", forbidden,
        )
        with pytest.raises((FiniteArtifactCollision, FiniteArtifactIntegrityError)):
            run_reintegrate_successor(plan)
        assert output.exists()
        assert first.output_artifact == str(output)
        assert output.read_bytes() == first_bytes


@pytest.mark.parametrize(
    ("seam", "expected_code"),
    [
        ("none", "CAPSULE_NOT_SUPPLIED"),
        ("admission", "CAPSULE_DIGEST_MISMATCH"),
        ("target", "TARGET_CHANGED"),
        ("target-disappeared", "TARGET_CHANGED"),
        ("member-disappeared", "SOURCE_REVISION_CHANGED"),
        ("terminal", "TERMINAL_CHANGED"),
        ("entry", "ENTRY_CHANGED"),
        ("labels", "LABELS_CHANGED"),
        ("root", "SOURCE_ROOT_CHANGED"),
        ("science", "REQUESTED_SCIENCE_UNSUPPORTED"),
    ],
)
def test_common_dispatcher_has_one_exact_phase_a_fallback(
    tmp_path, monkeypatch, seam, expected_code,
):
    import xrd_tools.reduction.reintegrate_successor as module
    from xrd_tools.reduction import (
        ReintegrateSuccessorPlan,
        prepare_reintegrate_bundle,
    )
    from xrd_tools.reduction.reintegrate_prepared import (
        PreparedCapsuleMissCode,
        PreparedRouteRejected,
    )

    seeded = _seed_existing(tmp_path, labels=(2, 5), name=f"fallback-{seam}")
    source = _prepared_eligible_source(seeded)
    offer = prepare_reintegrate_bundle(
        source,
        entry="entry",
        labels=seeded.labels,
        source_root=str(seeded.target.parent),
    )
    assert offer.disposition == "READY"
    preparation = _dimension_preparation(seeded, "1d")
    values = {
        "entry": "entry",
        "dimension": "1d",
        "preparation": preparation,
        "source_root": str(seeded.target.parent),
        "expected_labels": seeded.labels,
    }
    offered = offer
    source_artifact = seeded.target
    if seam == "none":
        offered = None
    elif seam == "admission":
        monkeypatch.setattr(
            module,
            "admit_prepared_bundle",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                PreparedRouteRejected(
                    PreparedCapsuleMissCode.CAPSULE_DIGEST_MISMATCH
                )
            ),
        )
    elif seam == "target":
        source_artifact = seeded.target.with_name("other.nexus")
    elif seam == "target-disappeared":
        seeded.target.unlink()
    elif seam == "member-disappeared":
        seeded.source.with_name("raw-member.h5").unlink()
    elif seam == "terminal":
        values["expected_terminal_identity"] = seeded.terminal.commit_identity
    elif seam == "entry":
        values["entry"] = "other"
    elif seam == "labels":
        values["expected_labels"] = (2,)
    elif seam == "root":
        values["source_root"] = str(seeded.target.parent / "other")
    elif seam == "science":
        preparation["api_version"] = 99

    calls = []
    sentinel = object()

    def fallback(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(module, "_legacy_successor_plan", fallback)
    result = ReintegrateSuccessorPlan.from_prepared_or_artifact(
        offered, source_artifact, **values,
    )
    assert result is sentinel
    assert len(calls) == 1
    assert calls[0][1]["miss_code"].value == expected_code
    assert calls[0][1]["legacy_reason"].value == "CAPSULE_MISS"


def test_capsule_miss_is_one_visible_result_diagnostic_across_recipe_reuse(
    tmp_path, monkeypatch,
):
    from xrd_tools.reduction import (
        ReintegrateSuccessorPlan,
        run_reintegrate_successor,
    )

    seeded = _seed_existing(tmp_path, labels=(2, 5), name="miss-diagnostic")
    output = seeded.target.parent / "miss-diagnostic-output.nexus"
    plan = ReintegrateSuccessorPlan.from_prepared_or_artifact(
        None,
        seeded.target,
        entry="entry",
        dimension="1d",
        preparation=_dimension_preparation(seeded, "1d"),
        source_root=str(seeded.target.parent),
        expected_labels=seeded.labels,
        explicit_output=output,
    )
    _stub_integrators(monkeypatch)
    first = run_reintegrate_successor(plan)
    replay = ReintegrateSuccessorPlan.from_recipe(plan.as_recipe())
    second = run_reintegrate_successor(replay)
    expected = ("PREPARED_CAPSULE_MISS:CAPSULE_NOT_SUPPLIED",)
    assert first.diagnostics == expected
    assert second.diagnostics == expected
    assert second.disposition == "ALREADY_COMMITTED"

    direct = _plan(
        seeded,
        explicit_output=seeded.target.parent / "direct-diagnostic.nexus",
        expected_terminal=None,
    )
    assert run_reintegrate_successor(direct).diagnostics == ()


@pytest.mark.parametrize(
    "seam",
    (
        "inspection",
        "inspection-known-error",
        "manifest-receipt",
        "manifest-writer-error",
    ),
)
def test_browse_target_drift_is_never_a_fallback_authorizing_miss(
    tmp_path, monkeypatch, seam,
):
    import xrd_tools.reduction.reintegrate_prepared as prepared_module
    from xrd_tools.reduction import prepare_reintegrate_bundle

    seeded = _seed_existing(
        tmp_path, labels=(2, 5), name=f"browse-target-drift-{seam}",
    )
    source = _prepared_eligible_source(seeded)
    replaced = False

    def replace_target_once():
        nonlocal replaced
        if replaced:
            return
        replaced = True
        staged = seeded.target.with_name(f"{seeded.target.name}.replacement")
        shutil.copy2(seeded.target, staged)
        os.replace(staged, seeded.target)

    if seam in ("inspection", "inspection-known-error"):
        original = prepared_module._legacy._inspect_artifact

        def drift_then_inspect(*args, **kwargs):
            if seam == "inspection":
                replace_target_once()
                return original(*args, **kwargs)
            original(*args, **kwargs)
            replace_target_once()
            raise ValueError("selected frame inventory is not exact")

        monkeypatch.setattr(
            prepared_module._legacy, "_inspect_artifact", drift_then_inspect,
        )
    else:
        original = prepared_module.prepare_replacement_manifest_receipt

        def drift_then_receipt(*args, **kwargs):
            replace_target_once()
            if seam == "manifest-receipt":
                return original(*args, **kwargs)
            raise prepared_module.WriterStateError(
                "replacement manifest domain is unsupported"
            )

        monkeypatch.setattr(
            prepared_module,
            "prepare_replacement_manifest_receipt",
            drift_then_receipt,
        )
    with pytest.raises(ValueError, match="TARGET_SNAPSHOT_CHANGED"):
        prepare_reintegrate_bundle(
            source,
            entry="entry",
            labels=seeded.labels,
            source_root=str(seeded.target.parent),
        )
    assert replaced
    assert not tuple(tmp_path.glob(".xdart-finite-*.candidate"))


@pytest.mark.parametrize("seam", ("inspection", "manifest-receipt"))
def test_browse_cancellation_is_never_a_fallback_authorizing_miss(
    tmp_path, monkeypatch, seam,
):
    import xrd_tools.reduction.reintegrate_prepared as prepared_module
    from xrd_tools.reduction import prepare_reintegrate_bundle

    seeded = _seed_existing(
        tmp_path, labels=(2, 5), name=f"browse-cancel-miss-{seam}",
    )
    source = _prepared_eligible_source(seeded)
    token = threading.Event()
    if seam == "inspection":
        original = prepared_module._legacy._inspect_artifact

        def cancel_then_known_error(*args, **kwargs):
            original(*args, **kwargs)
            token.set()
            raise ValueError("selected frame inventory is not exact")

        monkeypatch.setattr(
            prepared_module._legacy,
            "_inspect_artifact",
            cancel_then_known_error,
        )
    else:
        def cancel_then_writer_error(*_args, **_kwargs):
            token.set()
            raise prepared_module.WriterStateError(
                "replacement manifest domain is unsupported"
            )

        monkeypatch.setattr(
            prepared_module,
            "prepare_replacement_manifest_receipt",
            cancel_then_writer_error,
        )
    with pytest.raises(prepared_module._legacy.ReintegrateCancelled):
        prepare_reintegrate_bundle(
            source,
            entry="entry",
            labels=seeded.labels,
            source_root=str(seeded.target.parent),
            cancel_token=token,
        )
    assert not tuple(tmp_path.glob(".xdart-finite-*.candidate"))


@pytest.mark.parametrize("seam", ("swap", "disappear"))
def test_phase_a_predecessor_drift_falls_back_once_before_route_seal(
    tmp_path, monkeypatch, seam,
):
    import xrd_tools.reduction.reintegrate_successor as module
    from xrd_tools.reduction import (
        ReintegrateSuccessorPlan,
        prepare_reintegrate_bundle,
    )

    seeded = _seed_existing(
        tmp_path, labels=(2, 5), name=f"predecessor-drift-{seam}",
    )
    source = _prepared_eligible_source(seeded)
    offer = prepare_reintegrate_bundle(
        source,
        entry="entry",
        labels=seeded.labels,
        source_root=str(seeded.target.parent),
    )
    original_predecessor = module._predecessor
    drifted = False

    def drift_at_predecessor(*args, **kwargs):
        nonlocal drifted
        if drifted:
            return original_predecessor(*args, **kwargs)
        drifted = True
        staged = seeded.target.with_name(f"{seeded.target.name}.saved")
        shutil.copy2(seeded.target, staged)
        if seam == "swap":
            os.replace(staged, seeded.target)
            return original_predecessor(*args, **kwargs)
        seeded.target.unlink()
        try:
            return original_predecessor(*args, **kwargs)
        finally:
            shutil.copy2(staged, seeded.target)
            staged.unlink()

    fallback_calls = 0
    original_fallback = module._legacy_successor_plan

    def counted_fallback(*args, **kwargs):
        nonlocal fallback_calls
        fallback_calls += 1
        return original_fallback(*args, **kwargs)

    monkeypatch.setattr(module, "_predecessor", drift_at_predecessor)
    monkeypatch.setattr(module, "_legacy_successor_plan", counted_fallback)
    plan = ReintegrateSuccessorPlan.from_prepared_or_artifact(
        offer,
        seeded.target,
        entry="entry",
        dimension="1d",
        preparation=_dimension_preparation(seeded, "1d"),
        source_root=str(seeded.target.parent),
        expected_labels=seeded.labels,
    )
    assert fallback_calls == 1
    assert plan.legacy_reason.value == "CAPSULE_MISS"
    assert plan.miss_code.value == "TARGET_CHANGED"
    assert not tuple(tmp_path.glob(".xdart-finite-*.candidate"))


def test_phase_a_cancellation_wins_over_translated_target_drift(
    tmp_path, monkeypatch,
):
    import xrd_tools.reduction.reintegrate_successor as module
    from xrd_tools.io.finite_artifact import FiniteArtifactIntegrityError
    from xrd_tools.reduction import (
        ReintegrateSuccessorPlan,
        prepare_reintegrate_bundle,
    )

    seeded = _seed_existing(tmp_path, labels=(2, 5), name="phase-a-cancel")
    source = _prepared_eligible_source(seeded)
    offer = prepare_reintegrate_bundle(
        source,
        entry="entry",
        labels=seeded.labels,
        source_root=str(seeded.target.parent),
    )
    token = threading.Event()
    fallback_calls = 0

    def cancel_then_drift(*_args, **_kwargs):
        token.set()
        raise FiniteArtifactIntegrityError("target changed during capture")

    def forbidden_fallback(*_args, **_kwargs):
        nonlocal fallback_calls
        fallback_calls += 1
        raise AssertionError("cancellation entered legacy fallback")

    monkeypatch.setattr(module, "capture_finite_source", cancel_then_drift)
    monkeypatch.setattr(module, "_legacy_successor_plan", forbidden_fallback)
    with pytest.raises(module._legacy.ReintegrateCancelled):
        ReintegrateSuccessorPlan.from_prepared_or_artifact(
            offer,
            seeded.target,
            entry="entry",
            dimension="1d",
            preparation=_dimension_preparation(seeded, "1d"),
            source_root=str(seeded.target.parent),
            expected_labels=seeded.labels,
            cancel_token=token,
        )
    assert fallback_calls == 0


@pytest.mark.parametrize(
    "seam",
    (
        "generic",
        "cancel",
        "post-plan-drift",
        "post-plan-target-direct",
        "post-plan-target-recipe",
    ),
)
def test_common_dispatcher_never_falls_back_after_nonmiss_or_route_seal(
    tmp_path, monkeypatch, seam,
):
    import xrd_tools.reduction.reintegrate_successor as module
    from xrd_tools.reduction import (
        ReintegrateSuccessorPlan,
        prepare_reintegrate_bundle,
        run_reintegrate_successor,
    )
    from xrd_tools.reduction.reintegrate_prepared import (
        PreparedRouteChanged,
        PreparedRouteRejected,
    )

    seeded = _seed_existing(tmp_path, labels=(2, 5), name=f"no-fallback-{seam}")
    source = _prepared_eligible_source(seeded)
    offer = prepare_reintegrate_bundle(
        source,
        entry="entry",
        labels=seeded.labels,
        source_root=str(seeded.target.parent),
    )
    calls = 0
    original_fallback = module._legacy_successor_plan

    def counted_fallback(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_fallback(*args, **kwargs)

    monkeypatch.setattr(module, "_legacy_successor_plan", counted_fallback)
    token = threading.Event()
    if seam == "generic":
        monkeypatch.setattr(
            module,
            "admit_prepared_bundle",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                LookupError("not a capsule miss")
            ),
        )
        expected = LookupError
    elif seam == "cancel":
        token.set()
        expected = module._legacy.ReintegrateCancelled
    else:
        expected = PreparedRouteChanged

    if not seam.startswith("post-plan"):
        with pytest.raises(expected):
            ReintegrateSuccessorPlan.from_prepared_or_artifact(
                offer,
                seeded.target,
                entry="entry",
                dimension="1d",
                preparation=_dimension_preparation(seeded, "1d"),
                source_root=str(seeded.target.parent),
                expected_labels=seeded.labels,
                cancel_token=token,
            )
    else:
        plan = ReintegrateSuccessorPlan.from_prepared_or_artifact(
            offer,
            seeded.target,
            entry="entry",
            dimension="1d",
            preparation=_dimension_preparation(seeded, "1d"),
            source_root=str(seeded.target.parent),
            expected_labels=seeded.labels,
        )
        if seam == "post-plan-target-recipe":
            plan = ReintegrateSuccessorPlan.from_recipe(plan.as_recipe())
        if seam == "post-plan-drift":
            member = seeded.source.with_name("raw-member.h5")
            with h5py.File(member, "r+") as document:
                data = document["entry/instrument/detector/data"]
                data[0, 0, 0] = int(data[0, 0, 0]) + 1
        else:
            with h5py.File(seeded.target, "r+") as document:
                document.attrs["post_plan_drift"] = 1
        with pytest.raises(PreparedRouteChanged) as captured:
            run_reintegrate_successor(plan)
        assert captured.value.code.value == (
            "SOURCE_REVISION_CHANGED"
            if seam == "post-plan-drift" else "TARGET_CHANGED"
        )
        assert not Path(plan.output_artifact).exists()
    assert calls == 0


@pytest.mark.parametrize(
    "tamper",
    [
        "selected-result",
        "frame-index",
        "preserved-sibling-data",
        "preserved-metadata-data",
        "selected-bai",
        "gi-config",
        "source-execution",
        "append-lineage",
        "source-base",
        "result-seal-missing",
        "result-seal-payload",
    ],
)
def test_exact_replay_refuses_committed_tamper(
    tmp_path, monkeypatch, tamper,
):
    from xrd_tools.io.finite_artifact import FiniteArtifactCollision
    from xrd_tools.reduction import run_reintegrate_successor

    seeded = _seed_existing(
        tmp_path,
        labels=(0, 1) if tamper == "append-lineage" else (2, 5),
        name=f"seal-{tamper}",
        append=tamper == "append-lineage",
    )
    plan = _plan(seeded)
    _stub_integrators(monkeypatch)
    result = run_reintegrate_successor(plan)
    with h5py.File(result.output_artifact, "r+") as document:
        config = document["entry/reduction/config"]

        def rewrite_json(node, change):
            value = json.loads(node.asstr()[()])
            change(value)
            node[()] = json.dumps(
                value, sort_keys=True, separators=(",", ":"),
            )

        if tamper == "selected-result":
            dataset = document["entry/integrated_1d/intensity"]
            dataset[0, 0] = float(dataset[0, 0]) + 1.0
        elif tamper == "frame-index":
            dataset = document["entry/integrated_1d/frame_index"]
            dataset[0] = int(dataset[0]) + 1
        elif tamper == "preserved-sibling-data":
            dataset = document["entry/integrated_2d/intensity"]
            dataset[0, 0, 0] = float(dataset[0, 0, 0]) + 1.0
        elif tamper == "preserved-metadata-data":
            dataset = document["entry/data/theta"]
            dataset[0] = float(dataset[0]) + 1.0
        elif tamper == "selected-bai":
            rewrite_json(
                config["bai_1d_args"],
                lambda value: value.__setitem__("npt", value["npt"] + 1),
            )
        elif tamper == "gi-config":
            if "gi_config" not in config:
                config.create_dataset("gi_config", data='{"foreign":1}')
            else:
                rewrite_json(
                    config["gi_config"],
                    lambda value: value.__setitem__("foreign", 1),
                )
        elif tamper == "source-execution":
            rewrite_json(
                config["source_execution"],
                lambda value: value.__setitem__(
                    "frame_count", value["frame_count"] + 1,
                ),
            )
        elif tamper == "append-lineage":
            rewrite_json(
                config["append_lineage"],
                lambda value: value.__setitem__("foreign", 1),
            )
        elif tamper == "source-base":
            document["entry"].attrs.modify("source_base", "/foreign")
        elif tamper == "result-seal-missing":
            del config["dimension_replacement_1d_result_seal"]
        elif tamper == "result-seal-payload":
            rewrite_json(
                config["dimension_replacement_1d_result_seal"],
                lambda value: value.__setitem__("result_sha256", "0" * 64),
            )
    with pytest.raises(FiniteArtifactCollision):
        run_reintegrate_successor(plan)


@pytest.mark.parametrize("dimension", ("1d", "2d"))
def test_exact_replay_distinguishes_absent_gi_from_present_empty_gi(
    tmp_path, monkeypatch, dimension,
):
    from xrd_tools.io.finite_artifact import FiniteArtifactCollision
    from xrd_tools.reduction import run_reintegrate_successor

    seeded = _seed_existing(
        tmp_path, labels=(2, 5), name=f"empty-gi-{dimension}",
    )
    plan = _plan(seeded, dimension=dimension)
    _stub_integrators(monkeypatch)
    result = run_reintegrate_successor(plan)
    with h5py.File(result.output_artifact, "r+") as document:
        config = document["entry/reduction/config"]
        assert "gi_config" not in config
        config.create_dataset("gi_config", data="{}")
    with pytest.raises(FiniteArtifactCollision):
        run_reintegrate_successor(plan)


@pytest.mark.parametrize(
    "topology_change",
    ("alias-to-clone", "clone-to-alias"),
)
def test_candidate_preservation_binds_hard_link_alias_equivalence(
    tmp_path, monkeypatch, topology_change,
):
    import xrd_tools.reduction.reintegrate_successor as module
    from xrd_tools.io.finite_artifact import FiniteArtifactIntegrityError
    from xrd_tools.reduction import run_reintegrate_successor

    seeded = _seed_existing(
        tmp_path, labels=(2, 5), name=f"alias-preservation-{topology_change}",
    )
    with h5py.File(seeded.target, "r+") as document:
        entry = document["entry"]
        first = entry.create_dataset(
            "preserved_alias_a", data=np.asarray([1, 2, 3], dtype="i4"),
        )
        if topology_change == "alias-to-clone":
            entry["preserved_alias_b"] = first
        else:
            entry.create_dataset(
                "preserved_alias_b",
                data=np.asarray([1, 2, 3], dtype="i4"),
            )
    before = seeded.target.read_bytes()
    plan = _plan(seeded, expected_terminal=None)
    original_execute = module._SuccessorRuntime.execute_candidate

    def change_alias_topology(runtime, document, *args, **kwargs):
        seal = original_execute(runtime, document, *args, **kwargs)
        entry = document["entry"]
        if topology_change == "alias-to-clone":
            values = np.asarray(entry["preserved_alias_b"][...])
            del entry["preserved_alias_b"]
            entry.create_dataset("preserved_alias_b", data=values)
        else:
            del entry["preserved_alias_b"]
            entry["preserved_alias_b"] = entry["preserved_alias_a"]
        document.flush()
        return seal

    monkeypatch.setattr(
        module._SuccessorRuntime,
        "execute_candidate",
        change_alias_topology,
    )
    _stub_integrators(monkeypatch)
    with pytest.raises(
        FiniteArtifactIntegrityError,
        match="preserved source content",
    ):
        run_reintegrate_successor(plan)
    assert seeded.target.read_bytes() == before
    assert not Path(plan.output_artifact).exists()
    assert not tuple(tmp_path.glob(".xdart-finite-*.candidate"))


@pytest.mark.parametrize(
    ("dropped", "disposition", "committed"),
    [
        ((5,), "COMMITTED", (2,)),
        ((2, 5), "ABORTED", ()),
    ],
    ids=("partial", "all"),
)
def test_publication_drop_matrix_has_one_finite_outcome(
    tmp_path, monkeypatch, dropped, disposition, committed,
):
    from xrd_tools.reduction import run_reintegrate_successor

    seeded = _seed_existing(tmp_path, labels=(2, 5), name=f"drop-{disposition}")
    before = seeded.target.read_bytes()
    plan = _plan(seeded)
    _stub_integrators(monkeypatch, dropped=dropped)
    result = run_reintegrate_successor(plan)

    assert result.disposition == disposition
    assert result.committed_labels == committed
    assert result.publication_dropped_labels == dropped
    assert seeded.target.read_bytes() == before
    assert (seeded.target.parent / "immutable-successor.nexus").exists() is (
        disposition == "COMMITTED"
    )


@pytest.mark.parametrize(
    "projection",
    (
        "source-graph-evidence",
        "qualification-target",
        "source-root",
        "gi-bootstrap",
        "artifact-family",
        "lineage-source-artifact",
    ),
)
def test_recipe_redundant_projection_is_not_caller_selected(
    tmp_path, monkeypatch, projection,
):
    from xrd_tools.reduction import ReintegrateSuccessorPlan
    import xrd_tools.reduction.reintegrate_successor as module

    seeded = _seed_existing(
        tmp_path, labels=(2,), name=f"recipe-projection-{projection}",
    )
    _offer, plan = _prepared_plan(seeded)
    recipe = plan.as_recipe()
    qualification = recipe["plan"]["qualification"]
    finite = recipe["plan"]["finite"]
    if projection == "source-graph-evidence":
        finite["source_graph_evidence"]["facts_digest"] = "0" * 64
    elif projection == "qualification-target":
        qualification["target"] = str(seeded.target.with_name("other.nexus"))
    elif projection == "source-root":
        qualification["source_root"] = str(seeded.target.parent / "other")
    elif projection == "gi-bootstrap":
        qualification["gi_bootstrap_incidence"] = 0.125
    elif projection == "artifact-family":
        finite["artifact_family"] = "other-family"
    else:
        lineage = json.loads(finite["lineage_json"])
        lineage["predecessor"]["source_artifact"] = str(
            seeded.target.with_name("other.nexus")
        )
        finite["lineage_json"] = json.dumps(
            lineage, sort_keys=True, separators=(",", ":"),
        )
        finite["lineage_identity"] = hashlib.sha256(
            finite["lineage_json"].encode("utf-8")
        ).hexdigest()
    monkeypatch.setattr(
        module,
        "capture_finite_source",
        lambda *_args, **_kwargs: pytest.fail("recipe replay performed file I/O"),
    )
    with pytest.raises(ValueError, match="identity|qualification|plan facts"):
        ReintegrateSuccessorPlan.from_recipe(recipe)


def test_recipe_predecessor_identity_is_joined_to_the_admitted_parent(
    tmp_path, monkeypatch,
):
    import xrd_tools.reduction.reintegrate_successor as module
    from xrd_tools.reduction import (
        ReintegrateSuccessorPlan,
        run_reintegrate_successor,
    )

    seeded = _seed_existing(tmp_path, labels=(2,), name="recipe-predecessor")
    _stub_integrators(monkeypatch)
    first_plan = _plan(
        seeded,
        explicit_output=seeded.target.parent / "generation-one.nexus",
    )
    first = run_reintegrate_successor(first_plan)
    successor_source = SimpleNamespace(
        target=first_plan.output_artifact,
        labels=seeded.labels,
        preparation=seeded.preparation,
        terminal=SimpleNamespace(commit_identity=first.terminal),
    )
    second_plan = _plan(
        successor_source,
        explicit_output=seeded.target.parent / "generation-two.nexus",
    )
    recipe = second_plan.as_recipe()
    finite = recipe["plan"]["finite"]
    lineage = json.loads(finite["lineage_json"])
    predecessor = lineage["predecessor"]
    for name in (
        "version_identity", "publication_identity", "lineage_identity",
    ):
        predecessor[name] = "0" * 64
    finite["lineage_json"] = json.dumps(
        lineage, sort_keys=True, separators=(",", ":"),
    )
    finite["lineage_identity"] = hashlib.sha256(
        finite["lineage_json"].encode("utf-8")
    ).hexdigest()
    with monkeypatch.context() as replay_patch:
        replay_patch.setattr(
            module,
            "capture_finite_source",
            lambda *_args, **_kwargs: pytest.fail(
                "recipe replay performed file I/O"
            ),
        )
        replay = ReintegrateSuccessorPlan.from_recipe(recipe)
    with pytest.raises(ValueError, match="FINITE_REQUEST_CHANGED"):
        run_reintegrate_successor(replay)
    assert not (seeded.target.parent / "generation-two.nexus").exists()
    assert not tuple(seeded.target.parent.glob(".xdart-finite-*.candidate"))


@pytest.mark.parametrize(
    "seam", ("soft-link", "oversized", "malformed", "source-swap"),
)
def test_finite_predecessor_lineage_is_bounded_local_and_bracketed(
    tmp_path, monkeypatch, seam,
):
    import xrd_tools.reduction.reintegrate_successor as module
    from xrd_tools.io.finite_artifact import (
        FINITE_LINEAGE_MAX_BYTES,
        FiniteArtifactIntegrityError,
    )
    from xrd_tools.reduction import (
        ReintegrateSuccessorPlan,
        run_reintegrate_successor,
    )

    seeded = _seed_existing(tmp_path, labels=(2,), name=f"parent-{seam}")
    _stub_integrators(monkeypatch)
    first_plan = _plan(
        seeded,
        explicit_output=seeded.target.parent / "finite-parent.nexus",
    )
    first = run_reintegrate_successor(first_plan)
    parent = first_plan.output_artifact
    if seam != "source-swap":
        with h5py.File(parent, "r+") as document:
            config = document["entry/reduction/config"]
            del config["finite_artifact"]
            if seam == "soft-link":
                config.create_dataset(
                    "foreign_lineage", data=np.asarray([1], "u1"),
                )
                config["finite_artifact"] = h5py.SoftLink(
                    "/entry/reduction/config/foreign_lineage"
                )
            elif seam == "oversized":
                config.create_dataset(
                    "finite_artifact",
                    shape=(FINITE_LINEAGE_MAX_BYTES + 1,),
                    dtype="u1",
                )
            else:
                config.create_dataset(
                    "finite_artifact", data=np.frombuffer(b"{", dtype="u1"),
                )
    else:
        original = module.require_finite_artifact_lineage

        def swap_after_read(document, *args, **kwargs):
            lineage = original(document, *args, **kwargs)
            shown = os.fspath(document.filename)
            moved = shown + ".moved"
            os.replace(shown, moved)
            shutil.copyfile(moved, shown)
            return lineage

        monkeypatch.setattr(
            module, "require_finite_artifact_lineage", swap_after_read,
        )
    expected_terminal = first.terminal if seam == "source-swap" else None
    with pytest.raises((FiniteArtifactIntegrityError, ValueError)):
        ReintegrateSuccessorPlan.from_artifact(
            parent,
            entry="entry",
            dimension="1d",
            preparation=_dimension_preparation(seeded, "1d"),
            expected_terminal_identity=expected_terminal,
            expected_labels=seeded.labels,
            explicit_output=seeded.target.parent / "finite-child.nexus",
        )
    assert not (seeded.target.parent / "finite-child.nexus").exists()
