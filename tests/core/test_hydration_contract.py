"""Shared enum-native hydration value contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest


def _api():
    return import_module("xrd_tools.session.hydration")


def _scope(api):
    return api.HydrationScope("context-a", "scan-7", "/raw/source", 4)


def test_purpose_values_are_enum_only():
    api = _api()
    assert [str(value) for value in api.HydrationPurpose] == ["1d", "2d", "full"]
    assert not hasattr(api, "normalize_hydration_purpose")
    assert not hasattr(import_module("xrd_tools.session"), "normalize_hydration_purpose")


def test_outcomes_are_complete_and_frozen_values_are_total():
    api = _api()
    assert {value.name for value in api.HydrationOutcome} == {
        "HYDRATED",
        "ALREADY_RESIDENT",
        "SUPERSEDED",
        "OWNER_MISMATCH",
        "FAILED",
        "CANCELLED",
    }
    scope = _scope(api)
    assert hash(scope)
    with pytest.raises(FrozenInstanceError):
        scope.source = "/other"


def test_read_identity_and_presentation_identity_are_distinct():
    api = _api()
    scope = _scope(api)
    preview = api.HydrationReadKey(
        scope, "/processed/scan.nxs", 17, api.HydrationPurpose.PREVIEW
    )
    full = api.HydrationReadKey(
        scope, "/processed/scan.nxs", 17, api.HydrationPurpose.FULL
    )
    assert preview != full
    first = api.HydrationToken(preview, 8)
    later = api.HydrationToken(preview, 9)
    assert first.read_key == later.read_key
    assert first != later
    assert api.HydrationCompletion(
        first, api.HydrationOutcome.HYDRATED
    ).token is first
    assert api.HydrationCompletion(
        later, api.HydrationOutcome.FAILED, "source unavailable"
    ).diagnostic == "source unavailable"


def test_contract_values_cannot_carry_arrays_handles_stores_or_callbacks(tmp_path):
    api = _api()
    assert {
        value.__name__: tuple(field.name for field in fields(value))
        for value in (
            api.HydrationScope,
            api.HydrationReadKey,
            api.HydrationToken,
            api.HydrationCompletion,
        )
    } == {
        "HydrationScope": ("context_token", "scan_key", "source", "epoch"),
        "HydrationReadKey": (
            "scope",
            "artifact_identity",
            "frame_identity",
            "purpose",
        ),
        "HydrationToken": ("read_key", "presentation_generation"),
        "HydrationCompletion": ("token", "outcome", "diagnostic"),
    }
    assert api.HydrationScope.__annotations__ == {
        "context_token": "str",
        "scan_key": "str",
        "source": "str",
        "epoch": "int",
    }
    assert api.HydrationReadKey.__annotations__ == {
        "scope": "HydrationScope",
        "artifact_identity": "str",
        "frame_identity": "int | str",
        "purpose": "HydrationPurpose",
    }
    assert api.HydrationToken.__annotations__ == {
        "read_key": "HydrationReadKey",
        "presentation_generation": "int",
    }
    assert api.HydrationCompletion.__annotations__ == {
        "token": "HydrationToken",
        "outcome": "HydrationOutcome",
        "diagnostic": "str | None",
    }
    scope = _scope(api)
    with pytest.raises(TypeError):
        api.HydrationReadKey(
            scope, np.zeros(2), 1, api.HydrationPurpose.PREVIEW
        )
    with Path(tmp_path / "open.txt").open("w") as handle:
        with pytest.raises(TypeError):
            api.HydrationReadKey(
                scope, handle, 1, api.HydrationPurpose.PREVIEW
            )


def test_context_request_accepts_only_the_current_enum():
    api = _api()
    context = import_module("xdart.modules.display_context")
    owner = context.HydrationOwner("context-a", "scan-7", "/raw/source", 4)
    request = context.HydrationRequest(
        label=17,
        purpose=api.HydrationPurpose.PREVIEW,
        generation=9,
        owner=owner,
        stores=(object(),),
        commit_gate=object(),
    )
    assert request.purpose is api.HydrationPurpose.PREVIEW
    assert request.scope == _scope(api)
    assert request.read_key is None
    assert request.token is None
    with pytest.raises(TypeError):
        context.HydrationRequest(
            label=np.zeros(1),
            purpose=api.HydrationPurpose.PREVIEW,
            generation=9,
            owner=owner,
            stores=(),
            commit_gate=None,
        )
    for spelling in ("1d", "2d", "preview", "full", "raw"):
        with pytest.raises(TypeError):
            context.HydrationRequest(
                label=17,
                purpose=spelling,
                generation=9,
                owner=owner,
                stores=(),
                commit_gate=None,
            )


def test_typed_context_request_requires_one_consistent_identity():
    api = _api()
    context = import_module("xdart.modules.display_context")
    owner = context.HydrationOwner("context-a", "scan-7", "/raw/source", 4)
    key = api.HydrationReadKey(
        _scope(api), "/processed/scan.nxs", 17, api.HydrationPurpose.PREVIEW
    )
    token = api.HydrationToken(key, 9)
    request = context.HydrationRequest(
        label=17,
        purpose=api.HydrationPurpose.PREVIEW,
        generation=9,
        owner=owner,
        stores=(object(),),
        commit_gate=object(),
        read_key=key,
        token=token,
    )
    assert request.token is token
    wrong = api.HydrationToken(key, 10)
    with pytest.raises(ValueError):
        context.HydrationRequest(
            label=17,
            purpose=api.HydrationPurpose.PREVIEW,
            generation=9,
            owner=owner,
            stores=(),
            commit_gate=None,
            read_key=key,
            token=wrong,
        )


def test_typed_values_and_requests_reject_malformed_or_subclassed_envelopes():
    api = _api()
    context = import_module("xdart.modules.display_context")
    owner = context.HydrationOwner("context-a", "scan-7", "/raw/source", 4)
    key = api.HydrationReadKey(
        _scope(api), "/processed/scan.nxs", 17, api.HydrationPurpose.PREVIEW
    )
    token = api.HydrationToken(key, 9)

    class ScopeChild(api.HydrationScope):
        pass

    class KeyChild(api.HydrationReadKey):
        pass

    class TokenChild(api.HydrationToken):
        pass

    class OwnerChild(context.HydrationOwner):
        pass

    with pytest.raises(TypeError):
        api.HydrationReadKey(
            ScopeChild("context-a", "scan-7", "/raw/source", 4),
            "/processed/scan.nxs",
            17,
            api.HydrationPurpose.PREVIEW,
        )
    with pytest.raises(TypeError):
        api.HydrationToken(
            KeyChild(
                key.scope,
                key.artifact_identity,
                key.frame_identity,
                key.purpose,
            ),
            9,
        )
    with pytest.raises(TypeError):
        api.HydrationCompletion(
            TokenChild(token.read_key, token.presentation_generation),
            api.HydrationOutcome.HYDRATED,
        )
    for label, generation in (("", 9), (17, -1)):
        with pytest.raises((TypeError, ValueError)):
            context.HydrationRequest(
                label=label,
                purpose=api.HydrationPurpose.PREVIEW,
                generation=generation,
                owner=owner,
                stores=(),
                commit_gate=None,
            )
    with pytest.raises(TypeError):
        context.HydrationRequest(
            label=17,
            purpose=api.HydrationPurpose.PREVIEW,
            generation=9,
            owner=OwnerChild(*owner.as_tuple()),
            stores=(),
            commit_gate=None,
        )
    with pytest.raises(TypeError):
        context.HydrationRequest(
            label=17,
            purpose=api.HydrationPurpose.PREVIEW,
            generation=9,
            owner=owner,
            stores=(),
            commit_gate=None,
            read_key=object(),
            token=object(),
        )


def test_context_request_fields_are_the_complete_reference_only_shape():
    context = import_module("xdart.modules.display_context")
    assert tuple(field.name for field in fields(context.HydrationRequest)) == (
        "label",
        "purpose",
        "generation",
        "owner",
        "stores",
        "commit_gate",
        "read_key",
        "token",
        "checkpoint_token",
        "checkpoint_gate",
        "scope",
    )
    assert context.HydrationRequest.__annotations__ == {
        "label": "int | str",
        "purpose": "HydrationPurpose",
        "generation": "int",
        "owner": "HydrationOwner",
        "stores": "tuple",
        "commit_gate": "object",
        "read_key": "HydrationReadKey | None",
        "token": "HydrationToken | None",
        "checkpoint_token": "_CheckpointHydrationToken | None",
        "checkpoint_gate": "_CheckpointHydrationGate | None",
        "scope": "HydrationScope",
    }


def test_scope_is_derived_once_from_the_accepted_owner():
    source = Path(
        import_module("xdart.modules.display_context").__file__
    ).read_text(encoding="utf-8")
    assert source.count("HydrationScope(*self.owner.as_tuple())") == 1
    assert "HydrationScope(" not in source.replace(
        "HydrationScope(*self.owner.as_tuple())", ""
    )
