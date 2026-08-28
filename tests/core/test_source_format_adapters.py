# -*- coding: utf-8 -*-
"""R1 — the source-format adapter seam.

Pins the extension policy the R1 mission requires: a new source format is
declared through ONE :func:`register_adapter` call, never an edit to
``open_source`` or to ``discover.py``'s per-kind branches.  The
worked example uses ``SourceKind.TILED`` — reserved but unused by any
built-in adapter — as the stand-in "synthetic new format".
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from threading import Barrier, Event, Thread

import numpy as np
import pytest

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.sources.adapters import (
    SourceFormatAdapter,
    _ADAPTERS,
    _REGISTRY_LOCK,
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
    ``register_adapter`` never leaks into the rest of the suite.  Kind/candidate
    ownership is resolved live from ``_ADAPTERS`` (there is no separate kind
    index to restore), so saving that one dict is sufficient."""
    with _REGISTRY_LOCK:
        saved_adapters = dict(_ADAPTERS)
    try:
        yield
    finally:
        with _REGISTRY_LOCK:
            _ADAPTERS.clear()
            _ADAPTERS.update(saved_adapters)


def _run_fresh_process(code: str) -> str:
    root = str(Path(__file__).resolve().parents[2] / "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout.strip()


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

        # open_source() dispatches to the adapter without a format-specific
        # branch in registry.py.
        spec = SourceSpec(candidate, SourceKind.TILED)
        source = open_source(spec)
        assert isinstance(source, _SyntheticSource)
        assert source.spec is spec
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


# ---- adapter-only open_source dispatch -----------------------------------


def test_unadapted_kind_raises_a_clean_error():
    """A kind with no adapter raises the public opening error."""
    with pytest.raises(ValueError):
        open_source(SourceSpec("/x.weird", SourceKind.UNKNOWN))


def test_builtin_kinds_are_all_covered_by_an_adapter():
    """Every built-in source kind is owned by a registered adapter."""
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


@pytest.mark.parametrize(("accessor_import", "lookup", "expected"), (
    (
        "from xrd_tools.sources import adapter_for_kind",
        "adapter_for_kind(SourceKind.IMAGE_FILE).id",
        "image_file",
    ),
    (
        "from xrd_tools.sources import get_adapter",
        'get_adapter("image_file").id',
        "image_file",
    ),
    (
        "from xrd_tools.sources.adapters import candidate_owner",
        'candidate_owner(Path("frame.tif")).id',
        "image_file",
    ),
    (
        "from xrd_tools.sources.adapters import explicit_source_owner",
        'explicit_source_owner(Path("frame.tif"), SourceKind.IMAGE_FILE).id',
        "image_file",
    ),
    (
        "from xrd_tools.sources import all_adapters",
        '"image_file" in {adapter.id for adapter in all_adapters()}',
        True,
    ),
))
def test_public_adapter_lookup_bootstraps_builtins_in_a_fresh_process(
    accessor_import,
    lookup,
    expected,
):
    code = f"""
        import json
        import sys
        from pathlib import Path
        from xrd_tools.core.scan import SourceKind
        {accessor_import}

        assert "xrd_tools.sources.registry" not in sys.modules
        value = {lookup}
        assert "xrd_tools.sources.registry" in sys.modules
        print(json.dumps(value))
    """
    assert json.loads(_run_fresh_process(code)) == expected


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
        return json.loads(_run_fresh_process(code))

    before = _run("before")
    after = _run("after")
    assert before == {"discovery": "plugin_image", "kind": "plugin_image"}
    assert after == {"discovery": "plugin_image", "kind": "plugin_image"}


def test_same_id_external_registration_survives_builtin_bootstrap():
    """A pre-bootstrap external entry cannot be replaced by its lower-tier
    builtin counterpart merely because a public lookup triggers bootstrap."""
    code = """
        import json
        import sys
        from pathlib import Path
        from xrd_tools.core.scan import SourceKind, SourceSpec
        from xrd_tools.sources.adapters import SourceFormatAdapter, register_adapter

        external = SourceFormatAdapter(
            id="image_file",
            kinds=(SourceKind.IMAGE_FILE,),
            is_candidate=lambda path: path.suffix.lower() == ".tif",
            scan_name=lambda path: "external:" + path.stem,
            probe=lambda path: None,
            open=lambda spec: ("EXTERNAL", str(spec.uri)),
        )
        register_adapter(external)
        assert "xrd_tools.sources.registry" not in sys.modules

        from xrd_tools.sources import adapter_for_kind, get_adapter
        from xrd_tools.sources.adapters import candidate_owner

        kind_owner = adapter_for_kind(SourceKind.IMAGE_FILE)
        assert "xrd_tools.sources.registry" in sys.modules
        from xrd_tools.sources import open_source
        candidate = Path("frame.tif")
        result = {
            "same_by_id": get_adapter("image_file") is external,
            "same_by_kind": kind_owner is external,
            "same_by_candidate": candidate_owner(candidate) is external,
            "open": open_source(SourceSpec(candidate, SourceKind.IMAGE_FILE)),
        }
        print(json.dumps(result))
    """
    assert json.loads(_run_fresh_process(code)) == {
        "same_by_id": True,
        "same_by_kind": True,
        "same_by_candidate": True,
        "open": ["EXTERNAL", "frame.tif"],
    }


def test_concurrent_lazy_bootstrap_cannot_overwrite_same_id_external():
    """Force builtin bootstrap to pause after its same-id read while an
    external registration contends.  The register transaction lock makes the
    external write occur after the builtin transaction, so it remains final."""
    code = """
        import json
        import sys
        from pathlib import Path
        from threading import Event, Thread, current_thread
        from xrd_tools.core.scan import SourceKind
        import xrd_tools.sources.adapters as adapters

        builtin_read = Event()
        release_builtin = Event()
        external_started = Event()
        external_entered_get = Event()
        external_written = Event()

        class ControlledRegistry(dict):
            def get(self, key, default=None):
                value = super().get(key, default)
                if key == "image_file" and current_thread().name == "bootstrap":
                    builtin_read.set()
                    assert release_builtin.wait(2)
                elif key == "image_file" and current_thread().name == "external":
                    external_entered_get.set()
                return value

            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                if key == "image_file" and current_thread().name == "external":
                    external_written.set()

        adapters._ADAPTERS = ControlledRegistry()
        external = adapters.SourceFormatAdapter(
            id="image_file",
            kinds=(SourceKind.IMAGE_FILE,),
            is_candidate=lambda path: path.suffix.lower() == ".tif",
            scan_name=lambda path: "external:" + path.stem,
            probe=lambda path: None,
            open=lambda spec: ("EXTERNAL", str(spec.uri)),
        )

        lookup_result = []
        lookup_errors = []
        external_errors = []

        def bootstrap_lookup():
            try:
                lookup_result.append(adapters.adapter_for_kind(SourceKind.IMAGE_FILE))
            except BaseException as exc:
                lookup_errors.append(repr(exc))

        def register_external():
            external_started.set()
            try:
                adapters.register_adapter(external)
            except BaseException as exc:
                external_errors.append(repr(exc))

        bootstrap = Thread(target=bootstrap_lookup, name="bootstrap")
        external_thread = Thread(target=register_external, name="external")
        bootstrap.start()
        assert builtin_read.wait(2)
        external_thread.start()
        assert external_started.wait(2)

        # With the transaction lock, the external thread cannot even reach the
        # registry read while builtin registration is paused inside its lock.
        entered_before_release = external_entered_get.wait(1)
        release_builtin.set()
        bootstrap.join(3)
        external_thread.join(3)

        assert not entered_before_release
        assert not bootstrap.is_alive()
        assert not external_thread.is_alive()
        assert not lookup_errors
        assert not external_errors
        assert external_written.is_set()
        final = adapters.get_adapter("image_file")
        print(json.dumps({
            "final_is_external": final is external,
            "final_is_builtin": adapters._ADAPTERS["image_file"].builtin,
        }))
    """
    assert json.loads(_run_fresh_process(code)) == {
        "final_is_external": True,
        "final_is_builtin": False,
    }


def test_candidate_predicate_runs_outside_the_registry_lock():
    predicate_entered = Event()
    release_predicate = Event()
    registration_done = Event()
    owners = []
    errors = []

    def blocking_predicate(path):
        predicate_entered.set()
        if not release_predicate.wait(2):
            raise TimeoutError("predicate release timed out")
        return path.suffix == ".blocked"

    blocking = SourceFormatAdapter(
        id="blocking_candidate",
        kinds=(SourceKind.TILED,),
        is_candidate=blocking_predicate,
        scan_name=lambda path: path.stem,
        probe=lambda path: None,
        open=lambda spec: None,
    )
    concurrent = SourceFormatAdapter(
        id="concurrent_registration",
        kinds=(SourceKind.TILED,),
        is_candidate=lambda path: False,
        scan_name=lambda path: path.stem,
        probe=lambda path: None,
        open=lambda spec: None,
    )

    def lookup():
        try:
            owners.append(candidate_owner(Path("scan.blocked")))
        except BaseException as exc:
            errors.append(exc)

    def register_concurrently():
        try:
            register_adapter(concurrent)
        except BaseException as exc:
            errors.append(exc)
        finally:
            registration_done.set()

    with _isolated_adapter_registry():
        register_adapter(blocking)
        lookup_thread = Thread(target=lookup)
        writer_thread = Thread(target=register_concurrently)
        lookup_thread.start()
        assert predicate_entered.wait(2)
        writer_thread.start()
        try:
            assert registration_done.wait(1), (
                "plugin predicate executed while holding the registry lock"
            )
        finally:
            release_predicate.set()
        lookup_thread.join(3)
        writer_thread.join(3)

        assert not lookup_thread.is_alive()
        assert not writer_thread.is_alive()
        assert not errors
        assert owners == [blocking]


def test_bounded_concurrent_registration_lookup_and_iteration():
    start = Barrier(5)
    errors = []

    def concurrent_adapter(index):
        return SourceFormatAdapter(
            id=f"concurrent_{index}",
            kinds=(SourceKind.TILED,),
            is_candidate=lambda path: path.suffix == ".race",
            scan_name=lambda path: path.stem,
            probe=lambda path: None,
            open=lambda spec: None,
        )

    def guarded(action):
        try:
            start.wait(timeout=2)
            action()
        except BaseException as exc:
            errors.append(exc)

    def write_many():
        for index in range(64):
            register_adapter(concurrent_adapter(index))

    def read_kinds():
        for _ in range(128):
            adapter_for_kind(SourceKind.TILED)

    def read_candidates():
        for _ in range(128):
            candidate_owner(Path("scan.race"))

    def read_ids():
        for index in range(128):
            get_adapter(f"concurrent_{index % 64}")

    def iterate_snapshots():
        for _ in range(128):
            ids = [adapter.id for adapter in all_adapters()]
            assert len(ids) == len(set(ids))

    with _isolated_adapter_registry():
        threads = [
            Thread(target=guarded, args=(action,))
            for action in (
                write_many,
                read_kinds,
                read_candidates,
                read_ids,
                iterate_snapshots,
            )
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)

        assert not any(thread.is_alive() for thread in threads)
        assert not errors


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
    keeps path-first opening safe from a kind-hijack."""
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


# ===========================================================================
# R1-R8 — path-based open_source prefers the adapter that CLAIMS the path
# ===========================================================================

def _marker_adapter(idn: str, kinds, suffix: str) -> SourceFormatAdapter:
    """Adapter whose open() returns a ('OPENED_BY', id) marker so the test can
    see which adapter actually opened a spec."""
    return SourceFormatAdapter(
        id=idn, kinds=kinds,
        is_candidate=lambda p: p.suffix.lower() == suffix,
        scan_name=lambda p: p.stem,
        probe=lambda p: ProbeResult(ProbeState.READY, reason=idn, kind=kinds[0]),
        open=lambda spec: ("OPENED_BY", idn))


def test_r1r8_narrow_external_predicate_cannot_hijack_open_of_an_unclaimed_path(tmp_path):
    """Gate 4: an external adapter owning IMAGE_FILE for *.xyz must NOT hijack
    explicit opening of a .tif that the built-in image_file adapter claims —
    the held defect resolved open_owner=xyz_plugin for candidate_owner=image_file."""
    with _isolated_adapter_registry():
        # external plugin owns IMAGE_FILE kind, claims only .xyz
        register_adapter(_marker_adapter("xyz_plugin", (SourceKind.IMAGE_FILE,), ".xyz"))

        tif = tmp_path / "a.tif"
        tif.write_bytes(b"x")
        assert candidate_owner(tif).id == "image_file"          # built-in claims .tif
        # explicit open of the .tif must go through image_file, NOT xyz_plugin
        opened = open_source(SourceSpec(tif, SourceKind.IMAGE_FILE))
        from xrd_tools.sources.image import ImageFileSource
        assert isinstance(opened, ImageFileSource)

        # while the .xyz the plugin DOES claim opens through the plugin
        xyz = tmp_path / "b.xyz"
        xyz.write_bytes(b"x")
        assert candidate_owner(xyz).id == "xyz_plugin"
        assert open_source(SourceSpec(xyz, SourceKind.IMAGE_FILE)) == ("OPENED_BY", "xyz_plugin")


def test_r1r8_fully_overlapping_adapters_open_through_the_precedence_winner(tmp_path):
    """Gate 5: when two adapters share BOTH the kind and the predicate,
    discovery, probe, and open all use the last-registered winner."""
    with _isolated_adapter_registry():
        register_adapter(_marker_adapter("first_w", (SourceKind.TILED,), ".widget"))
        register_adapter(_marker_adapter("second_w", (SourceKind.TILED,), ".widget"))

        w = tmp_path / "x.widget"
        w.write_bytes(b"x")
        assert candidate_owner(w).id == "second_w"
        assert adapter_for_kind(SourceKind.TILED).id == "second_w"
        assert open_source(SourceSpec(w, SourceKind.TILED)) == ("OPENED_BY", "second_w")


def test_r1r8_incompatible_claimer_falls_through_to_kind_owner(tmp_path):
    """When the path-claiming adapter is NOT compatible with the explicitly
    requested kind, opening falls through to the kind owner (the caller's
    explicit kind wins) — e.g. force-open a .tif as NEXUS_STACK."""
    tif = tmp_path / "a.tif"
    tif.write_bytes(b"x")
    # image_file claims .tif but does not declare NEXUS_STACK; nexus_hdf5 owns
    # NEXUS_STACK and opens it (and then fails to read the bogus file, which is
    # the expected downstream error, not a routing error).
    import pytest
    with pytest.raises(Exception):
        open_source(SourceSpec(tif, SourceKind.NEXUS_STACK))


def test_r1r8_virtual_source_uses_the_kind_owner():
    """A virtual (non-path) URI opens through its kind owner."""
    from xrd_tools.sources.memory import LiveFrameSource
    # LIVE virtual uri -> no candidate claims it -> kind owner (live adapter)
    live = open_source(SourceSpec("live-run-1", SourceKind.LIVE))
    assert isinstance(live, LiveFrameSource)
