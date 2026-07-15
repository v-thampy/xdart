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
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.sources.adapters import (
    SourceFormatAdapter,
    _ADAPTERS,
    adapter_for_kind,
    all_adapters,
    candidate_owner,
    get_adapter,
    register_adapter,
)
from xrd_tools.sources.probe import ProbeResult, ProbeState
from xrd_tools.sources.registry import open_source


@contextlib.contextmanager
def _isolated_adapter_registry():
    """Save/restore the process-global adapter registry so a test's
    ``register_adapter`` never leaks into the rest of the suite (mirrors
    ``test_source_registry_seam.py``'s ``_isolated_registry``).  Kind/candidate
    ownership is resolved live from ``_ADAPTERS`` (there is no separate kind
    index to restore), so saving that one dict is sufficient."""
    saved_adapters = dict(_ADAPTERS)
    try:
        yield
    finally:
        _ADAPTERS.clear()
        _ADAPTERS.update(saved_adapters)


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


# ===========================================================================
# R1-R2 — ONE precedence rule; import-order-safe bootstrap; same-id cleanup
# ===========================================================================

def _overlapping_adapter(idn: str) -> SourceFormatAdapter:
    """Two of these share BOTH the TILED kind and the .widget predicate, so
    they contend for the same file AND the same kind."""
    return SourceFormatAdapter(
        id=idn, kinds=(SourceKind.TILED,),
        is_candidate=lambda p: p.suffix == ".widget",
        scan_name=lambda p: p.stem,
        probe=lambda p: ProbeResult(ProbeState.READY, reason=idn, kind=SourceKind.TILED),
        open=lambda spec: _SyntheticSource(spec))


def test_r1r2_precedence_is_identical_for_discovery_kind_probe_and_open(tmp_path):
    """Gate 3: with two overlapping adapters, candidate discovery,
    adapter_for_kind, probe, and open all resolve to the SAME adapter —
    the last-registered one (the held defect discovered through first_widget
    but resolved kind/open through second_widget)."""
    from xrd_tools.sources.directory_index import DirectoryIndex

    with _isolated_adapter_registry():
        register_adapter(_overlapping_adapter("first_widget"))
        register_adapter(_overlapping_adapter("second_widget"))

        (tmp_path / "x.widget").write_bytes(b"x")

        # discovery owner
        cand = candidate_owner(tmp_path / "x.widget")
        assert cand.id == "second_widget"
        # kind lookup owner
        assert adapter_for_kind(SourceKind.TILED).id == "second_widget"
        # discovered candidate carries the same owner
        index = DirectoryIndex(tmp_path)
        c = index.poll().candidates[0]
        assert c.adapter_id == "second_widget"
        # probe goes through the same adapter (reason carries the adapter id)
        assert index.probe_candidate(c).reason == "second_widget"
        # open goes through the same adapter
        opened = open_source(SourceSpec(tmp_path / "x.widget", SourceKind.TILED))
        assert isinstance(opened, _SyntheticSource)


def test_r1r2_external_registration_survives_later_builtin_bootstrap():
    """Gate 4: in a FRESH process, an external adapter registered BEFORE the
    lazy built-in bootstrap still owns its kind/candidate afterwards; and the
    reverse order (external registered AFTER built-ins) also yields the
    external.  External always outranks built-in, regardless of import order."""
    root = str(Path(__file__).resolve().parents[2] / "src")

    def _run(order: str) -> dict:
        code = textwrap.dedent(f"""
            import json, sys, tempfile
            from pathlib import Path
            from xrd_tools.core.scan import SourceKind
            from xrd_tools.sources.adapters import (
                SourceFormatAdapter, register_adapter, adapter_for_kind)
            from xrd_tools.sources.probe import ProbeResult, ProbeState

            def plugin():
                return SourceFormatAdapter(
                    id="plugin_image", kinds=(SourceKind.IMAGE_FILE,),
                    is_candidate=lambda p: p.suffix.lower() == ".tif",
                    scan_name=lambda p: p.stem,
                    probe=lambda p: ProbeResult(ProbeState.READY, kind=SourceKind.IMAGE_FILE),
                    open=lambda spec: None)

            order = {order!r}
            if order == "before":
                register_adapter(plugin())
                assert "xrd_tools.sources.registry" not in sys.modules
                from xrd_tools.sources.discover import enumerate_candidates
            else:
                import xrd_tools.sources.registry  # built-ins first
                register_adapter(plugin())
                from xrd_tools.sources.discover import enumerate_candidates

            with tempfile.TemporaryDirectory() as d:
                d = Path(d); (d / "f.tif").write_bytes(b"x")
                disc = enumerate_candidates(d)[0].adapter_id
            kind_owner = adapter_for_kind(SourceKind.IMAGE_FILE).id
            print(json.dumps({{"discovery": disc, "kind": kind_owner}}))
        """)
        env = dict(os.environ)
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stdout + out.stderr
        import json
        return json.loads(out.stdout.strip())

    before = _run("before")
    after = _run("after")
    assert before == {"discovery": "plugin_image", "kind": "plugin_image"}
    assert after == {"discovery": "plugin_image", "kind": "plugin_image"}


def test_r1r2_same_id_replacement_removes_obsolete_kind_mappings():
    """Gate 5: re-registering an id with FEWER kinds leaves no stale kind
    mapping — the held defect resolved LIVE to a rebound adapter that only
    declared TILED."""
    with _isolated_adapter_registry():
        register_adapter(SourceFormatAdapter(
            id="rebound", kinds=(SourceKind.LIVE,),
            is_candidate=lambda p: False, scan_name=lambda p: p.stem,
            probe=lambda p: ProbeResult(ProbeState.READY), open=lambda s: None))
        # rebind the SAME id, now declaring only TILED
        register_adapter(SourceFormatAdapter(
            id="rebound", kinds=(SourceKind.TILED,),
            is_candidate=lambda p: False, scan_name=lambda p: p.stem,
            probe=lambda p: ProbeResult(ProbeState.READY), open=lambda s: None))

        # LIVE must NOT resolve to 'rebound' (it no longer declares LIVE) —
        # it falls back to the built-in 'live' adapter.
        live_owner = adapter_for_kind(SourceKind.LIVE)
        assert live_owner is None or live_owner.id != "rebound"
        if live_owner is not None:
            assert SourceKind.LIVE in live_owner.kinds
        # TILED now resolves to the rebound adapter.
        assert adapter_for_kind(SourceKind.TILED).id == "rebound"


def test_r1r2_candidate_owner_and_kind_owner_answer_different_questions(tmp_path):
    """A plugin that declares an EXISTING kind without claiming a given file
    exposes the precise contract: candidate_owner (who claims the filename)
    and adapter_for_kind (who owns the kind) apply the same precedence but can
    legitimately differ.  The load-bearing guarantee is that a DISCOVERED
    candidate is probed through its RECORDED owning adapter (never re-resolved
    by kind), so discovery and probe never disagree — the structural fact that
    keeps a future opener (R2) safe from a kind-hijack."""
    from xrd_tools.sources.directory_index import DirectoryIndex

    with _isolated_adapter_registry():
        # external plugin: owns IMAGE_FILE kind, but only claims *.xyz files
        plugin = SourceFormatAdapter(
            id="xyz_plugin", kinds=(SourceKind.IMAGE_FILE,),
            is_candidate=lambda p: p.suffix.lower() == ".xyz",
            scan_name=lambda p: p.stem,
            probe=lambda p: ProbeResult(ProbeState.READY, reason="xyz_plugin",
                                        kind=SourceKind.IMAGE_FILE),
            open=lambda spec: _SyntheticSource(spec))
        register_adapter(plugin)

        (tmp_path / "a.tif").write_bytes(b"x")

        # candidate ownership of a .tif goes to the built-in image_file
        # (the plugin does not claim .tif) ...
        assert candidate_owner(tmp_path / "a.tif").id == "image_file"
        # ... while kind ownership of IMAGE_FILE goes to the external plugin
        # (external outranks built-in).  Same precedence, different question.
        assert adapter_for_kind(SourceKind.IMAGE_FILE).id == "xyz_plugin"

        # The discovered candidate records image_file and is PROBED through
        # image_file — never through the kind owner — so discovery == probe.
        index = DirectoryIndex(tmp_path)
        c = index.poll().candidates[0]
        assert c.adapter_id == "image_file"
        # image_file's real probe on a bogus tif is INVALID (not the plugin's
        # READY); the point is the reason/verdict comes from image_file, the
        # recorded owner, not from xyz_plugin.
        assert index.probe_candidate(c).reason != "xyz_plugin"
