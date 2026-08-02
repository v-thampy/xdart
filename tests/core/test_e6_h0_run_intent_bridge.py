"""Frozen compatibility oracle for the E6 shared-headless H0 bridge."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.session.run_configuration import (
    GIIntent,
    RunIntent,
    ThresholdIntent,
)
from xrd_tools.sources.selection import DirectorySourceSpec


def test_directory_metadata_policy_survives_clone_freeze_thaw_and_fingerprint(
    tmp_path,
):
    explicit = RunIntent(source_spec=DirectorySourceSpec(
        root=tmp_path / "raw",
        recursive=True,
        suffixes=(".tif",),
        generation=7,
        metadata_format="pdi",
    ))

    candidate = explicit.clone_candidate()
    frozen = candidate.freeze()
    thawed = frozen.thaw_source_spec()

    assert candidate.source_spec.metadata_format == "pdi"
    assert frozen.source.metadata_format == "pdi"
    assert frozen.source.as_dict()["metadata_format"] == "pdi"
    assert thawed.metadata_format == "pdi"

    automatic = RunIntent(source_spec=DirectorySourceSpec(
        root=tmp_path / "raw",
        recursive=True,
        suffixes=(".tif",),
        generation=7,
        metadata_format="auto",
    )).freeze()
    assert automatic.fingerprint != frozen.fingerprint


def test_run_intent_normalizes_legacy_image_metadata_but_preserves_explicit_off():
    legacy = SourceSpec(
        Path("/raw/frame.tif"),
        SourceKind.IMAGE_FILE,
        options={"detector": "eiger"},
    )
    explicit_off = SourceSpec(
        Path("/raw/frame.tif"),
        SourceKind.IMAGE_FILE,
        options={"metadata_format": None},
    )

    normalized = RunIntent(source_spec=legacy)
    disabled = RunIntent(source_spec=explicit_off)

    assert normalized.source_spec.options["metadata_format"] == "auto"
    assert (
        normalized.clone_candidate().freeze().thaw_source_spec()
        .options["metadata_format"]
        == "auto"
    )
    assert disabled.source_spec.options["metadata_format"] is None


def test_direct_freeze_normalizes_legacy_image_assigned_after_construction():
    intent = RunIntent()
    intent.source_spec = SourceSpec(
        Path("/raw/late-frame.tif"),
        SourceKind.IMAGE_FILE,
        options={},
    )

    thawed = intent.freeze().thaw_source_spec()

    assert thawed.options["metadata_format"] == "auto"


def test_h0_preserves_db79_source_family_fingerprints():
    non_image = RunIntent(source_spec=SourceSpec(
        "spec://scan",
        SourceKind.SPEC,
    )).freeze()
    exact_auto_image = RunIntent(source_spec=SourceSpec(
        Path("/raw/frame.tif"),
        SourceKind.IMAGE_FILE,
        options={"metadata_format": "auto"},
    )).freeze()

    assert non_image.fingerprint == (
        "7c418abb0231d84ae069221b432a31fc07624756caa4add5a"
        "4106e0e797f7797"
    )
    assert exact_auto_image.fingerprint == (
        "759f96b10212f4177a2af8d12cba5c4287232232cb10fc38"
        "6b8c8cc8f209b8c8"
    )


def test_clone_candidate_detaches_mutable_gi_and_threshold_values():
    gi_value = np.array(0.2)
    threshold_value = np.array(1.0)
    original = RunIntent(
        gi=GIIntent(enabled=True, th_val=gi_value),
        threshold=ThresholdIntent(threshold_min=threshold_value),
    )

    candidate = original.clone_candidate()
    gi_value[...] = 1.3
    threshold_value[...] = 9.0

    assert candidate.gi.th_val == 0.2
    assert candidate.threshold.threshold_min == 1.0
    candidate.gi.th_val[...] = 2.4
    candidate.threshold.threshold_min[...] = 8.0
    assert original.gi.th_val == 1.3
    assert original.threshold.threshold_min == 9.0


class _MutableNumber:
    def __init__(self, value):
        self.value = float(value)

    def __float__(self):
        return self.value

    def __deepcopy__(self, _memo):
        raise AssertionError("canonical clone invoked arbitrary deepcopy")


def test_clone_and_store_capture_float_coercible_values_without_aliasing():
    from xrd_tools.session.intent_store import (
        IntentCommitAccepted,
        IntentFreezeAccepted,
        RunIntentStore,
    )

    gi_value = _MutableNumber(0.2)
    threshold_value = _MutableNumber(1.0)
    original = RunIntent(
        gi=GIIntent(enabled=True, th_val=gi_value),
        threshold=ThresholdIntent(threshold_min=threshold_value),
    )
    candidate = original.clone_candidate()
    store = RunIntentStore(original)

    gi_value.value = 1.3
    threshold_value.value = 9.0

    assert candidate.freeze().gi.th_val == 0.2
    stored = store.snapshot().thaw()
    assert float(stored.gi.th_val) == 0.2
    assert float(stored.threshold.threshold_min) == 1.0

    committed_gi = _MutableNumber(0.4)
    committed_threshold = _MutableNumber(2.0)
    committed = store.commit(
        RunIntent(
            gi=GIIntent(enabled=True, th_val=committed_gi),
            threshold=ThresholdIntent(threshold_min=committed_threshold),
        ),
        expected_revision=0,
    )
    assert isinstance(committed, IntentCommitAccepted)
    committed_gi.value = 3.4
    committed_threshold.value = 8.0

    frozen = store.freeze(expected_revision=1)
    assert isinstance(frozen, IntentFreezeAccepted)
    assert frozen.configuration.gi.th_val == 0.4
    assert frozen.configuration.threshold.threshold_min == 2.0


def test_intent_store_delegates_to_canonical_clone_without_copy_hooks():
    class HostileDeepcopyValue:
        def tolist(self):
            return [1, {"value": 2}]

        def __deepcopy__(self, _memo):
            raise AssertionError("intent store invoked an open-ended copy hook")

    from xrd_tools.session import RunIntentStore as PublicRunIntentStore
    from xrd_tools.session.intent_store import RunIntentStore

    store = RunIntentStore(RunIntent(
        run_options={"value": HostileDeepcopyValue()},
    ))

    assert PublicRunIntentStore is RunIntentStore
    assert store.snapshot().thaw().run_options == {
        "value": [1, {"value": 2}],
    }
