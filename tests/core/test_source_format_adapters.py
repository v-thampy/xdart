# -*- coding: utf-8 -*-
"""R1 — the source-format adapter seam.

Pins the extension policy the R1 mission requires: a new source format is
declared through ONE :func:`register_adapter` call, never an edit to
``open_source``'s if-chain or to ``discover.py``'s per-kind branches.  The
worked example uses ``SourceKind.TILED`` — reserved but unused by any
built-in adapter — as the stand-in "synthetic new format" (the same pattern
``test_source_registry_seam.py`` already uses for the legacy
``register_source`` seam).
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.sources.adapters import (
    SourceFormatAdapter,
    _ADAPTERS,
    _KIND_INDEX,
    adapter_for_kind,
    all_adapters,
    get_adapter,
    register_adapter,
)
from xrd_tools.sources.probe import ProbeResult, ProbeState
from xrd_tools.sources.registry import open_source


@contextlib.contextmanager
def _isolated_adapter_registry():
    """Save/restore the process-global adapter registry so a test's
    ``register_adapter`` never leaks into the rest of the suite (mirrors
    ``test_source_registry_seam.py``'s ``_isolated_registry``)."""
    saved_adapters = dict(_ADAPTERS)
    saved_kinds = dict(_KIND_INDEX)
    try:
        yield
    finally:
        _ADAPTERS.clear()
        _ADAPTERS.update(saved_adapters)
        _KIND_INDEX.clear()
        _KIND_INDEX.update(saved_kinds)


class _SyntheticSource:
    """Minimal FrameSource duck for the synthetic-format adapter (no I/O)."""

    def __init__(self, spec):
        self.spec = spec
        self.frame_indices = [0]

    def load_frame(self, index):
        return np.zeros((2, 2))


def _synthetic_adapter() -> SourceFormatAdapter:
    return SourceFormatAdapter(
        id="synthetic_widget",
        kinds=(SourceKind.TILED,),
        is_candidate=lambda path: path.suffix.lower() == ".widget",
        scan_name=lambda path: path.stem,
        probe=lambda path: ProbeResult(ProbeState.READY, reason="widget file",
                                       kind=SourceKind.TILED),
        open=lambda spec: _SyntheticSource(spec),
    )


# ---- registration + lookup ---------------------------------------------


def test_register_adapter_is_discoverable_by_id_and_kind():
    with _isolated_adapter_registry():
        adapter = _synthetic_adapter()
        register_adapter(adapter)
        assert get_adapter("synthetic_widget") is adapter
        assert adapter_for_kind(SourceKind.TILED) is adapter
        assert adapter in all_adapters()


def test_register_adapter_does_not_disturb_other_adapters():
    """Registering one adapter does not disturb another kind's dispatch —
    the built-in nexus/image/spec/live adapters stay intact."""
    with _isolated_adapter_registry():
        before = {a.id for a in all_adapters()}
        register_adapter(_synthetic_adapter())
        after = {a.id for a in all_adapters()}
        assert before <= after
        assert adapter_for_kind(SourceKind.IMAGE_FILE) is not None
        assert adapter_for_kind(SourceKind.NEXUS_STACK) is not None


# ---- discovered, named, probed, and opened WITHOUT editing existing branches


def test_synthetic_format_is_named_probed_and_opened_through_the_seam(tmp_path):
    with _isolated_adapter_registry():
        register_adapter(_synthetic_adapter())
        adapter = adapter_for_kind(SourceKind.TILED)

        candidate = tmp_path / "run_0007.widget"
        candidate.write_text("not real data")

        # name-only candidate rule
        assert adapter.is_candidate(candidate) is True
        assert adapter.is_candidate(tmp_path / "run_0007.txt") is False

        # canonical scan-name function
        assert adapter.scan_name(candidate) == "run_0007"

        # explicit content probe/classifier
        result = adapter.probe(candidate)
        assert result.state is ProbeState.READY
        assert result.kind is SourceKind.TILED

        # open_source() dispatches to the adapter without ANY new branch in
        # registry.py's if-chain or a call to register_source().
        source = open_source(SourceSpec(candidate, SourceKind.TILED))
        assert isinstance(source, _SyntheticSource)
        assert source.frame_indices == [0]
        assert np.array_equal(source.load_frame(0), np.zeros((2, 2)))


def test_synthetic_format_is_discovered_through_the_real_enumeration_and_index_path(tmp_path):
    """Same synthetic format, but driven through enumerate_candidates() and
    DirectoryIndex.poll() end-to-end -- not just the is_candidate predicate
    in isolation -- proving a new format needs zero changes to either."""
    from xrd_tools.sources.directory_index import DirectoryIndex
    from xrd_tools.sources.discover import enumerate_candidates

    with _isolated_adapter_registry():
        register_adapter(_synthetic_adapter())
        (tmp_path / "run_0007.widget").write_text("not real data")
        (tmp_path / "unrelated.txt").write_text("ignore me")

        candidates = enumerate_candidates(tmp_path)
        assert [c.path.name for c in candidates] == ["run_0007.widget"]
        assert candidates[0].adapter_id == "synthetic_widget"

        index = DirectoryIndex(tmp_path)
        snapshot = index.poll()
        assert [c.path.name for c in snapshot.candidates] == ["run_0007.widget"]

        result = index.probe_candidate(snapshot.candidates[0])
        assert result.state is ProbeState.READY
        assert result.kind is SourceKind.TILED


def test_synthetic_adapter_overrides_legacy_register_source_kind_symmetrically():
    """The adapter seam and the legacy register_source() seam both work for
    the SAME kind; register_source (checked first in open_source) still wins
    when both are registered — preserving the documented override order."""
    from xrd_tools.sources.registry import _REGISTRY, register_source

    with _isolated_adapter_registry():
        register_adapter(_synthetic_adapter())

        class _LegacySource(_SyntheticSource):
            pass

        saved = dict(_REGISTRY)
        try:
            register_source(SourceKind.TILED, lambda spec: _LegacySource(spec))
            source = open_source(SourceSpec("tiled://x", SourceKind.TILED))
            assert isinstance(source, _LegacySource)
        finally:
            _REGISTRY.clear()
            _REGISTRY.update(saved)


# ---- legacy register_source(kind, factory) / open_source unchanged ------


def test_legacy_register_source_and_open_source_are_unaffected_by_adapters():
    """A kind with no adapter and no legacy factory still raises the
    built-in clean error — the adapter seam did not swallow dispatch for
    unrelated kinds."""
    with pytest.raises(ValueError):
        open_source(SourceSpec("/x.weird", SourceKind.UNKNOWN))


def test_builtin_kinds_are_all_covered_by_an_adapter():
    """Every kind open_source's if-chain used to construct directly is now
    reachable through a registered built-in adapter (the if-chain itself is
    an untouched, now-dead fallback for anything unadapted)."""
    for kind in (
        SourceKind.NEXUS_STACK,
        SourceKind.EIGER_MASTER,
        SourceKind.PROCESSED_NEXUS,
        SourceKind.IMAGE_FILE,
        SourceKind.TIFF_SERIES,
        SourceKind.SPEC,
        SourceKind.LIVE,
    ):
        assert adapter_for_kind(kind) is not None, kind
