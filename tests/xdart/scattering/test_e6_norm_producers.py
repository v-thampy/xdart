"""Frozen E6-NORM-N1 producer oracle (handoff §25.2; prompt §3.3).

Frozen BEFORE any production edit and run on exact parent
``38815d73eca60916ab074f0d6871d4de610a93d1``: every producer row was RED
there (the per-artifact acquisition slot, the exact-frame read, the Browse
aggregate field and the fold-site census did not exist), while the retained
Q2 kernel sentinel, the Browse publish-nothing legs and the scattering
architecture file stayed green.  Producers must reuse the accepted Q2
kernel exactly; the two identity shapes are frozen by §25.2:

* acquisition — ``(run_identity.generation, run_identity.fingerprint,
  str(artifact), source_scan)``;
* Browse — ``(context_token, scan_key, requested_path)``.

Every fold consumes the already-admitted ``view.metadata_numeric``; the
helper views carry a deliberately different ``metadata_raw`` decoy channel
so any raw-metadata substitution changes every channel assertion.
"""

from __future__ import annotations

import ast
from pathlib import Path
from threading import Event

import h5py
import numpy as np
import pytest

from xdart.gui.tabs.scattering.adapters import browse_loader as loader_module
from xdart.gui.tabs.scattering.adapters.browse_loader import BrowseLoader
from xdart.gui.tabs.scattering.browse_values import (
    BrowseLoadRequest,
    BrowseLoadStatus,
)
from xdart.gui.tabs.scattering.display_runtime import RunDisplayState
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
from xdart.gui.tabs.scattering.events import CleanupStatus, RunIdentity
from xdart.modules.display_context import BrowseContext, DisplayContextError
from xdart.modules.frame_publication import FramePublication, PublicationStore
from xrd_tools.core import Axis, FrameRecord, FrameView
from xrd_tools.io import FrameScalarCatalog, FrameScalarRow
from xrd_tools.session.frame_record_store import FrameRecordStore
from xrd_tools.session.scan_norm import (
    ScanNormAggregate,
    channel_is_partial,
    empty_norm_aggregate,
    fold_norm_metadata,
    next_norm_revision,
)


_ROOT = Path(__file__).parents[3] / "src"
_SCATTERING = _ROOT / "xdart" / "gui" / "tabs" / "scattering"
PRODUCTION = {
    "display_runtime": _SCATTERING / "display_runtime.py",
    "run_executor": _SCATTERING / "adapters" / "run_executor.py",
    "browse_loader": _SCATTERING / "adapters" / "browse_loader.py",
    "display_context": _ROOT / "xdart" / "modules" / "display_context.py",
}

IDENTITY = RunIdentity(generation=7, fingerprint="run-fp-n1")
ARTIFACT = "/data/run/scan_a.nxs"
SCAN = "scan_a"

#: The decoy raw-metadata channel: NEVER a valid fold source.  Folding
#: ``metadata_raw`` instead of the admitted ``metadata_numeric`` makes every
#: channel-dict assertion in this module fail.
_RAW_DECOY = {"raw_only_monitor": 999.0}


def _view(label: int, metadata_numeric) -> FrameView:
    return FrameView(
        label,
        axis_1d=Axis("q", "A^-1", values=np.array([0.0, 1.0])),
        intensity_1d=np.array([1.0, 2.0]),
        source_path=f"/data/raw/frame_{label}.tif",
        source_frame_index=label,
        metadata_numeric=metadata_numeric,
        metadata_raw=_RAW_DECOY,
    )


def _display(
    identity: RunIdentity = IDENTITY,
    *,
    artifact: str = ARTIFACT,
    scan_key: str = SCAN,
    partitions: int = 1,
    publication_store_factory=PublicationStore,
    measurement_mode: str = "Standard",
    gi: bool = False,
):
    display = RunDisplayState(identity, max_payload_items=8)
    display.set_factories(FrameRecordStore, publication_store_factory)
    display.configure(partition_count=partitions, npt=2, frame_bytes=48)
    owner = display.add_artifact(
        Path(artifact),
        scan_key,
        mask=None,
        mask_saturation=True,
        measurement_mode=measurement_mode,
        gi_incidence_motor="samth" if gi else "",
        gi_resolved_motor="samth" if gi else "",
        gi_mode_1d="qip" if gi else "",
        gi_mode_2d="qip_qoop" if gi else "",
    )
    return display, owner


def _retain(display: RunDisplayState, owner, label: int, metadata_numeric):
    view = _view(label, metadata_numeric)
    record = FrameRecord.from_view(view)
    publication = FramePublication(
        view,
        record=record,
        source_identity=f"{view.source_path}#{label}",
        scan_key=owner.source_scan,
    )
    key = display.append_navigation(
        owner.source_scan, str(owner.artifact), label
    ).appended
    display.retain_frame(
        owner,
        key,
        record,
        publication,
        source_identity=f"{view.source_path}#{label}",
        frame_mask_qualified=False,
    )
    return key


def _expected(rows):
    """Independent test-local ABSENT-rule accumulator (not the kernel)."""
    channels: dict[str, tuple[float, int]] = {}
    for row in rows:
        for key, value in row.items():
            try:
                array = np.asarray(value)
                if array.shape != ():
                    continue
                numeric = float(array)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(numeric) or numeric <= 0.0:
                continue
            total, count = channels.get(key.lower(), (0.0, 0))
            channels[key.lower()] = (total + numeric, count + 1)
    return len(rows), channels


class _BrowseScan:
    """Minimal open-scan stand-in for the loader's injected seam."""

    entry = "entry"
    metadata: dict = {}


def _artifact_file(tmp_path: Path, name: str) -> str:
    path = tmp_path / name
    with h5py.File(path, "w"):
        pass
    return str(path)


def _catalog(source, rows):
    return FrameScalarCatalog(
        str(Path(source).resolve()),
        "entry",
        tuple(
            FrameScalarRow(index, metadata_raw=row)
            for index, row in enumerate(rows, start=1)
        ),
    )


class _CatalogReader:
    def __init__(self, source, *, resolve_source, read_catalog):
        assert resolve_source is False
        self._source = str(source)
        self._read_catalog = read_catalog

    def __enter__(self):
        return self

    def read_scalar_catalog(self, *, cancelled):
        return self._read_catalog(self._source, cancelled)

    def __exit__(self, _exc_type, _exc, _tb):
        return None


def _reader_factory(read_catalog):
    def open_reader(source, *, resolve_source):
        return _CatalogReader(
            source,
            resolve_source=resolve_source,
            read_catalog=read_catalog,
        )

    return open_reader


def _rows_reader(rows):
    def read_catalog(source, cancelled):
        if cancelled():
            raise InterruptedError("Browse scalar catalog read cancelled")
        return _catalog(source, rows)

    return _reader_factory(read_catalog)


def _load_via_loader(
    monkeypatch,
    tmp_path: Path,
    rows,
    *,
    token: str,
    generation: int,
    name: str | None = None,
):
    source = _artifact_file(
        tmp_path,
        name or f"{token}.nexus",
    )
    monkeypatch.setattr(
        loader_module,
        "canonical_browse_scan_key",
        lambda path: Path(path).stem,
    )
    monkeypatch.setattr(
        loader_module,
        "read_browse_presentation",
        lambda _path: ({}, None),
    )
    request = BrowseLoadRequest(token, generation, source)
    loader = BrowseLoader(
        open_scan=lambda _path: _BrowseScan(),
        open_reader=_rows_reader(rows),
    )
    loader.begin(request)
    worker = loader._worker
    if worker is not None:
        worker.join(timeout=10.0)
    outcome = loader.poll(request)
    assert outcome is not None and outcome.status is BrowseLoadStatus.READY
    operation = loader._active
    context = loader.consume(outcome)
    assert context is not None
    return request, context, loader, operation, outcome


# --------------------------------------------------------------------- #
# Retained sentinel (green on parent): the accepted Q2 kernel itself.
# --------------------------------------------------------------------- #


def test_q2_kernel_sentinel_is_present_and_functional():
    aggregate = empty_norm_aggregate(("token", "scan", "/p.nxs"))
    assert (aggregate.revision, aggregate.row_count) == (0, 0)
    folded = fold_norm_metadata(aggregate, {"MON": 2.5, "bad": float("nan")})
    published = next_norm_revision(folded)
    assert (published.revision, published.row_count) == (1, 1)
    assert dict(published.channels) == {"mon": (2.5, 1)}
    assert not channel_is_partial(published, "mon")


# --------------------------------------------------------------------- #
# Acquisition producer rows (red on parent).
# --------------------------------------------------------------------- #


def test_acquisition_aggregate_is_absent_before_the_first_successful_retain():
    display, owner = _display()
    assert owner.norm_aggregate is None
    key = DisplayFrameKey(IDENTITY, SCAN, ARTIFACT, 1, 1)
    assert display.frame_norm_aggregate(key) is None


def test_acquisition_identity_and_revision_row_count_sequence():
    display, owner = _display()
    key = _retain(display, owner, 1, {"i0": 2.0})
    first = owner.norm_aggregate
    assert type(first) is ScanNormAggregate
    assert first.identity == (
        IDENTITY.generation,
        IDENTITY.fingerprint,
        ARTIFACT,
        SCAN,
    )
    assert (first.revision, first.row_count) == (1, 1)
    assert dict(first.channels) == {"i0": (2.0, 1)}
    _retain(display, owner, 2, {"I0": 4.0})
    second = owner.norm_aggregate
    assert (second.revision, second.row_count) == (2, 2)
    assert dict(second.channels) == {"i0": (6.0, 2)}
    assert display.frame_norm_aggregate(key) is second


def test_standard_and_gi_folds_match_an_independent_oracle():
    rows = [
        {"i0": 1.5, "bstop": 3.0},
        {"I0": 2.5},
        {"i0": 4.0, "bstop": 0.0},
    ]
    expected_rows, expected_channels = _expected(rows)
    for gi in (False, True):
        identity = RunIdentity(
            generation=4 if gi else 3,
            fingerprint="fp-gi" if gi else "fp-std",
        )
        display, owner = _display(
            identity,
            artifact=f"/data/{'gi' if gi else 'std'}.nxs",
            scan_key="scan_x",
            measurement_mode="GI" if gi else "Standard",
            gi=gi,
        )
        for label, row in enumerate(rows, start=1):
            _retain(display, owner, label, row)
        aggregate = owner.norm_aggregate
        assert aggregate.row_count == expected_rows
        assert dict(aggregate.channels) == expected_channels
        assert aggregate.revision == len(rows)


def test_absent_rule_rows_increment_row_count_but_not_channel_count():
    rows = [
        {"mon": 5.0},
        {},
        {"mon": float("nan")},
        {"mon": "no-number"},
        {"mon": 0.0},
        {"mon": -3.0},
        {"MON": 7.0},
    ]
    display, owner = _display()
    for label, row in enumerate(rows, start=1):
        _retain(display, owner, label, row)
    aggregate = owner.norm_aggregate
    assert (aggregate.revision, aggregate.row_count) == (7, 7)
    assert dict(aggregate.channels) == {"mon": (12.0, 2)}
    assert channel_is_partial(aggregate, "mon")


class _InjectedRetainFailure(RuntimeError):
    pass


def test_injected_retain_failure_leaves_the_exact_prior_aggregate():
    armed = {"on": False}

    class _FailingPublicationStore(PublicationStore):
        def upsert(self, publication, *, protected=()):
            if armed["on"]:
                raise _InjectedRetainFailure("armed retain failure")
            return super().upsert(publication, protected=protected)

    display, owner = _display(
        publication_store_factory=_FailingPublicationStore
    )
    _retain(display, owner, 1, {"mon": 2.0})
    prior = owner.norm_aggregate
    armed["on"] = True
    with pytest.raises(_InjectedRetainFailure):
        _retain(display, owner, 2, {"mon": 100.0})
    assert owner.norm_aggregate is prior
    assert (prior.revision, prior.row_count) == (1, 1)
    assert dict(prior.channels) == {"mon": (2.0, 1)}
    armed["on"] = False
    _retain(display, owner, 3, {"mon": 3.0})
    recovered = owner.norm_aggregate
    assert (recovered.revision, recovered.row_count) == (2, 2)
    assert dict(recovered.channels) == {"mon": (5.0, 2)}


def test_late_residency_failure_leaves_the_exact_prior_aggregate(monkeypatch):
    display, owner = _display()
    _retain(display, owner, 1, {"mon": 2.0})
    prior = owner.norm_aggregate

    def fail_after_publication(*, protected):
        del protected
        raise _InjectedRetainFailure("armed late residency failure")

    monkeypatch.setattr(display._residency, "enforce", fail_after_publication)
    with pytest.raises(_InjectedRetainFailure):
        _retain(display, owner, 2, {"mon": 100.0})
    assert owner.norm_aggregate is prior
    assert (prior.revision, prior.row_count) == (1, 1)
    assert dict(prior.channels) == {"mon": (2.0, 1)}


def test_projection_and_redraw_paths_do_not_change_count_or_revision():
    display, owner = _display()
    key = _retain(display, owner, 1, {"mon": 2.0})
    aggregate = owner.norm_aggregate
    payload = display.project(key, 3, closed=True, require_complete=False)
    assert payload is not None
    display.put_payload(payload)
    display.catalog_snapshot()
    display.residency_snapshot()
    assert display.resolve_frame(key) is key
    assert display.frame_norm_aggregate(key) is aggregate
    assert owner.norm_aggregate is aggregate
    assert (aggregate.revision, aggregate.row_count) == (1, 1)


def test_second_directory_artifact_starts_absent_and_accumulates_independently():
    display, owner_a = _display(partitions=2)
    _retain(display, owner_a, 1, {"mon": 2.0})
    frozen_a = owner_a.norm_aggregate
    owner_b = display.add_artifact(
        Path("/data/run/scan_b.nxs"),
        "scan_b",
        mask=None,
        mask_saturation=True,
        measurement_mode="Standard",
    )
    assert owner_b.norm_aggregate is None
    _retain(display, owner_b, 1, {"flux": 8.0})
    assert owner_a.norm_aggregate is frozen_a
    second = owner_b.norm_aggregate
    assert type(second) is ScanNormAggregate
    assert second.identity == (
        IDENTITY.generation,
        IDENTITY.fingerprint,
        "/data/run/scan_b.nxs",
        "scan_b",
    )
    assert (second.revision, second.row_count) == (1, 1)
    assert dict(second.channels) == {"flux": (8.0, 1)}
    assert dict(frozen_a.channels) == {"mon": (2.0, 1)}


def test_exact_frame_read_rejects_foreign_run_artifact_and_source():
    display, owner = _display()
    key = _retain(display, owner, 1, {"mon": 2.0})
    aggregate = owner.norm_aggregate
    assert display.frame_norm_aggregate(key) is aggregate
    foreign_run = DisplayFrameKey(
        RunIdentity(generation=99, fingerprint="foreign"),
        SCAN,
        ARTIFACT,
        1,
        1,
    )
    foreign_artifact = DisplayFrameKey(
        IDENTITY, SCAN, "/data/run/other.nxs", 1, 1
    )
    foreign_source = DisplayFrameKey(IDENTITY, "scan_other", ARTIFACT, 1, 1)
    assert display.frame_norm_aggregate(foreign_run) is None
    assert display.frame_norm_aggregate(foreign_artifact) is None
    assert display.frame_norm_aggregate(foreign_source) is None


# --------------------------------------------------------------------- #
# Browse/reload producer rows.
# --------------------------------------------------------------------- #


def test_browse_and_reload_publish_revision_one_after_one_full_pass(
    tmp_path, monkeypatch,
):
    rows = [{"i0": 1.0}, {"I0": 2.0}, {"i0": 4.0}]
    _, context, loader, _, _ = _load_via_loader(
        monkeypatch, tmp_path, rows, token="browse-first", generation=1
    )
    aggregate = context.norm_aggregate
    assert type(aggregate) is ScanNormAggregate
    assert (aggregate.revision, aggregate.row_count) == (1, 3)
    assert dict(aggregate.channels) == {"i0": (7.0, 3)}
    _, reloaded, reload_loader, _, _ = _load_via_loader(
        monkeypatch,
        tmp_path,
        rows,
        token="browse-reload",
        generation=2,
        name="scan_0002.nexus",
    )
    again = reloaded.norm_aggregate
    assert (again.revision, again.row_count) == (1, 3)
    assert dict(again.channels) == dict(aggregate.channels)
    assert again.identity != aggregate.identity
    assert loader.release_context(context).cleanup_status.value == "cleaned"
    assert reload_loader.release_context(reloaded).cleanup_status.value == "cleaned"
    loader.close()
    reload_loader.close()


def test_browse_identity_is_exactly_token_scan_key_and_requested_path(
    tmp_path,
    monkeypatch,
):
    request, context, loader, _, _ = _load_via_loader(
        monkeypatch,
        tmp_path,
        [{"mon": 2.0}],
        token="browse-ident",
        generation=4,
    )
    aggregate = context.norm_aggregate
    scan_key = Path(request.source_path).stem
    assert scan_key
    assert aggregate.identity == (
        request.token,
        scan_key,
        request.source_path,
    )
    assert context.scan_key == scan_key
    assert context.requested_path == request.source_path
    assert loader.release_context(context).cleanup_status.value == "cleaned"
    loader.close()


def test_browse_cancellation_failure_and_empty_input_publish_nothing(
    tmp_path,
    monkeypatch,
):
    rows = [{"mon": 2.0}, {"mon": 3.0}]
    source = _artifact_file(tmp_path, "scan_0003.nexus")
    monkeypatch.setattr(
        loader_module,
        "canonical_browse_scan_key",
        lambda path: Path(path).stem,
    )
    monkeypatch.setattr(
        loader_module,
        "read_browse_presentation",
        lambda _path: ({}, None),
    )

    for token, generation, build_before_wait in (
        ("browse-cancel", 6, False),
        ("browse-tail-cancel", 7, True),
    ):
        entered = Event()
        release = Event()

        def blocked_catalog(path, cancelled):
            catalog = _catalog(path, rows) if build_before_wait else None
            entered.set()
            assert release.wait(timeout=5.0)
            if cancelled():
                raise InterruptedError("Browse scalar catalog read cancelled")
            return catalog if catalog is not None else _catalog(path, rows)

        request = BrowseLoadRequest(token, generation, source)
        loader = BrowseLoader(
            join_timeout=0.001,
            open_scan=lambda _path: _BrowseScan(),
            open_reader=_reader_factory(blocked_catalog),
        )
        loader.begin(request)
        assert entered.wait(timeout=2.0)
        pending = loader.cancel(request)
        assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
        release.set()
        worker = loader._worker
        assert worker is not None
        worker.join(timeout=5.0)
        assert loader.cancel(request).cleanup_status.value == "cleaned"
        assert loader.poll(request) is None
        assert loader._active is None
        loader.close()

    def fail_catalog(_path, _cancelled):
        raise RuntimeError("reader failure")

    failing_request = BrowseLoadRequest("browse-fail", 8, source)
    failing_loader = BrowseLoader(
        open_scan=lambda _path: _BrowseScan(),
        open_reader=_reader_factory(fail_catalog),
    )
    failing_loader.begin(failing_request)
    failing_worker = failing_loader._worker
    assert failing_worker is not None
    failing_worker.join(timeout=5.0)
    failed = failing_loader.poll(failing_request)
    assert failed is not None and failed.status is BrowseLoadStatus.FAILED
    assert failed.detail == "reader failure"
    assert failing_loader.consume(failed) is None
    failing_loader.close()

    empty_request = BrowseLoadRequest("browse-empty", 9, source)
    empty_loader = BrowseLoader(
        open_scan=lambda _path: _BrowseScan(),
        open_reader=_rows_reader(()),
    )
    empty_loader.begin(empty_request)
    empty_worker = empty_loader._worker
    assert empty_worker is not None
    empty_worker.join(timeout=5.0)
    empty = empty_loader.poll(empty_request)
    assert empty is not None and empty.status is BrowseLoadStatus.FAILED
    assert empty.detail == "processed browse artifact has no frames"
    assert empty_loader.consume(empty) is None
    empty_loader.close()


def test_browse_aggregate_reassignment_is_refused(tmp_path, monkeypatch):
    _, context, loader, _, _ = _load_via_loader(
        monkeypatch,
        tmp_path,
        [{"mon": 2.0}],
        token="browse-once",
        generation=9,
    )
    held = context.norm_aggregate
    assert type(held) is ScanNormAggregate
    with pytest.raises(DisplayContextError):
        context.norm_aggregate = next_norm_revision(held)
    assert context.norm_aggregate is held
    assert loader.release_context(context).cleanup_status.value == "cleaned"
    loader.close()


def test_acquisition_browse_and_reload_agree_on_row_count_and_channels(
    tmp_path,
    monkeypatch,
):
    rows = [
        {"i0": 1.5, "bstop": 3.0},
        {"I0": 2.5},
        {"i0": 4.0, "bstop": 0.0},
    ]
    display, owner = _display()
    for label, row in enumerate(rows, start=1):
        _retain(display, owner, label, row)
    acquisition = owner.norm_aggregate
    _, browse, browse_loader, _, _ = _load_via_loader(
        monkeypatch,
        tmp_path,
        rows,
        token="browse-parity",
        generation=11,
        name="scan_0004.nexus",
    )
    _, reloaded, reload_loader, _, _ = _load_via_loader(
        monkeypatch,
        tmp_path,
        rows,
        token="browse-parity-reload",
        generation=12,
        name="scan_0004.nexus",
    )
    first, second, third = (
        acquisition,
        browse.norm_aggregate,
        reloaded.norm_aggregate,
    )
    assert (
        first.row_count == second.row_count == third.row_count == len(rows)
    )
    assert dict(first.channels) == dict(second.channels) == dict(
        third.channels
    )
    assert first.revision == len(rows)
    assert second.revision == third.revision == 1
    assert len({first.identity, second.identity, third.identity}) == 3
    assert browse_loader.release_context(browse).cleanup_status.value == "cleaned"
    assert reload_loader.release_context(reloaded).cleanup_status.value == "cleaned"
    browse_loader.close()
    reload_loader.close()


# --------------------------------------------------------------------- #
# Structural census and attachment boundary.
# --------------------------------------------------------------------- #


def test_no_attachment_beyond_the_two_retention_owners(tmp_path, monkeypatch):
    display, owner = _display()
    _retain(display, owner, 1, {"mon": 2.0})
    for carrier in (
        owner.records,
        owner.publications,
        display,
    ):
        assert not hasattr(carrier, "norm_aggregate")
    request, context, loader, operation, outcome = _load_via_loader(
        monkeypatch,
        tmp_path,
        [{"mon": 2.0}],
        token="browse-attach",
        generation=13,
    )
    for carrier in (
        loader,
        operation,
        outcome,
        request,
        context.record_store,
        context.publication_store,
    ):
        assert not hasattr(carrier, "norm_aggregate")
    assert type(context.norm_aggregate) is ScanNormAggregate
    assert loader.release_context(context).cleanup_status.value == "cleaned"
    loader.close()


def test_browse_scalar_reader_is_invoked_once(tmp_path, monkeypatch):
    source = _artifact_file(tmp_path, "scan_0005.nexus")
    monkeypatch.setattr(
        loader_module,
        "canonical_browse_scan_key",
        lambda path: Path(path).stem,
    )
    monkeypatch.setattr(
        loader_module,
        "read_browse_presentation",
        lambda _path: ({}, None),
    )
    request = BrowseLoadRequest("browse-one-reader", 14, source)
    calls = []

    def read_catalog(path, cancelled):
        calls.append(path)
        assert not cancelled()
        return _catalog(path, [{"mon": 2.0}, {"mon": 3.0}])

    loader = BrowseLoader(
        open_scan=lambda _path: _BrowseScan(),
        open_reader=_reader_factory(read_catalog),
    )
    loader.begin(request)
    worker = loader._worker
    assert worker is not None
    worker.join(timeout=5.0)
    outcome = loader.poll(request)
    assert outcome is not None and outcome.status is BrowseLoadStatus.READY
    context = loader.consume(outcome)
    assert type(context) is BrowseContext
    assert calls == [source]
    assert loader.release_context(context).cleanup_status.value == "cleaned"
    loader.close()


def test_browse_loader_has_one_exact_scalar_catalog_route():
    source = PRODUCTION["browse_loader"].read_text()
    for token in (
        "iter_frame_records",
        "FrameScalarRow",
        "_iter_browse_records",
        "_DEFAULT_RECORD_READER",
        "read_records",
        "_read_records",
        "_catalog_from_legacy_records",
    ):
        assert token not in source, token

    tree = ast.parse(source)
    loader_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BrowseLoader"
    )
    methods = {
        node.name: node
        for node in loader_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    constructor = methods["__init__"]
    constructor_parameters = {
        argument.arg
        for argument in (
            *constructor.args.args,
            *constructor.args.kwonlyargs,
        )
    }
    assert "open_reader" in constructor_parameters
    assert "read_records" not in constructor_parameters

    read_context = methods["_read_context"]
    keyword_names = [
        argument.arg for argument in read_context.args.kwonlyargs
    ]
    operation_index = keyword_names.index("_operation")
    assert read_context.args.kw_defaults[operation_index] is None

    scalar_catalog_calls = [
        node
        for node in ast.walk(read_context)
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "_catalog_from_reader"
        )
    ]
    assert len(scalar_catalog_calls) == 1


def test_census_single_fold_sites_one_browse_pass_no_forbidden_routes():
    sources = {name: path.read_text() for name, path in PRODUCTION.items()}
    for name, source in sources.items():
        for token in (
            "scan_data",
            "read_scan_data",
            "scan_aggregate",
            "pandas",
            "DataFrame",
            "iterrows",
            "metadata_raw",
        ):
            assert token not in source, (name, token)

    context_tree = ast.parse(sources["display_context"])
    for node in ast.walk(context_tree):
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported = [node.module or ""]
        else:
            continue
        for name in imported:
            assert "scan_norm" not in name, name
            assert name.split(".")[0] not in {"numpy", "pandas"}, name

    def _fold_calls(tree: ast.AST) -> list[ast.Call]:
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else getattr(func, "attr", "")
                )
                if name == "fold_norm_metadata":
                    calls.append(node)
        return calls

    def _enclosing_functions(tree: ast.AST, targets) -> list[str]:
        owners = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for target in targets:
                    if any(inner is target for inner in ast.walk(node)):
                        owners.append(node.name)
        return owners

    display_tree = ast.parse(sources["display_runtime"])
    browse_tree = ast.parse(sources["browse_loader"])
    display_folds = _fold_calls(display_tree)
    browse_folds = _fold_calls(browse_tree)
    assert len(display_folds) == 1
    assert _enclosing_functions(display_tree, display_folds) == [
        "retain_frame"
    ]
    assert len(browse_folds) == 1
    assert _enclosing_functions(browse_tree, browse_folds) == [
        "_read_context"
    ]
    for tree in (display_tree, browse_tree):
        fold_symbol_loads = [
            node
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Name)
                and node.id == "fold_norm_metadata"
                and isinstance(node.ctx, ast.Load)
            )
            or (
                isinstance(node, ast.Attribute)
                and node.attr == "fold_norm_metadata"
                and isinstance(node.ctx, ast.Load)
            )
        ]
        assert len(fold_symbol_loads) == 1
    assert "fold_norm_metadata" not in sources["run_executor"]
    assert "norm_aggregate" not in sources["run_executor"]
    assert "metadata_numeric" in sources["display_runtime"]
