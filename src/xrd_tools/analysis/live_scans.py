"""Readiness and detached custody for native per-scan Live."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import h5py
from silx.io.specfile import SfError

from xrd_tools.analysis import stitch_sources
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.io import image, nexus, spec, stat_identity
from xrd_tools.io.bluesky_nexus import is_bluesky_nxwriter
from xrd_tools.io.metadata import SpecOccurrence, SpecOccurrenceUnsettled
from xrd_tools.io.processed_scan_id import has_processed_output_markers_file
from xrd_tools.sources import descriptor, execution_graph, open_source
from xrd_tools.sources.discover import _walk_files
from xrd_tools.sources.execution_graph import SourceFileState
from xrd_tools.sources.point_images import (
    PointImagePolicy, PointImagesRequired, _nexus_point_count, detect_point_images, inspect_point_images,
)
from xrd_tools.sources.registry import _nexus_is_candidate
from xrd_tools.sources.spec import find_spec_image_files

_WAIT_ERRORS = (OSError, ValueError, LookupError, SfError, SpecOccurrenceUnsettled)


class _Skipped(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LiveMemberWitness:
    member: stitch_sources.PlannedStitchMember
    source: SourceSpec
    occurrence_identity: tuple
    occurrence: SpecOccurrence | None = field(compare=False)
    successor_required: bool
    labels: tuple[int, ...]
    mapping: tuple
    files: tuple[SourceFileState, ...]
    graphs: tuple
    raw_paths: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class LiveContainerWitness:
    root: SourceFileState
    entries: tuple[tuple[str, str, LiveMemberWitness], ...]


@dataclass(frozen=True, slots=True)
class LiveEvidence:
    members: tuple[LiveMemberWitness, ...]
    containers: tuple[LiveContainerWitness, ...]
    labels: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class LiveScanRow:
    group: stitch_sources.PlannedStitchGroup
    status: str
    reason: str = ""
    evidence: LiveEvidence | None = None


@dataclass(frozen=True, slots=True)
class LiveScanObservation:
    groups: tuple[LiveScanRow, ...]
    late_entries: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class LiveConsumedGroup:
    identity: tuple
    containers: tuple[tuple[SourceFileState, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class LiveScanHandoff:
    group: stitch_sources.PlannedStitchGroup
    evidence: LiveEvidence
    session_id: str
    attempt_id: str

    @classmethod
    def from_ready(cls, row, *, session_id, attempt_id):
        if row.status != "ready" or row.evidence is None:
            raise ValueError("Live can hand off only a ready group")
        return cls(row.group, row.evidence, session_id, attempt_id)

    def consumed(self):
        return LiveConsumedGroup(self.group.identity, tuple(
            (item.root, tuple(key for key, *_ in item.entries)) for item in self.evidence.containers))


def _planned_points(command):
    """Interpret only known SPEC macros with complete, numeric arguments."""
    tokens = command.split()
    if not tokens:
        return None
    macro, *args = tokens
    match = re.fullmatch(r"[ad](\d*)scan", macro)
    if match:
        motors = int(match[1] or 1) if len(match[1]) < 5 else 0
        if not motors or len(args) != 3 * motors + 2:
            return None
        coordinates = [i for i in range(3 * motors) if i % 3]
        intervals = (len(args) - 2,)
    elif macro in {"mesh", "dmesh"} and len(args) == 9:
        coordinates, intervals = (1, 2, 5, 6), (3, 7)
    elif macro == "hklscan" and len(args) == 8:
        coordinates, intervals = range(6), (6,)
    elif macro == "loopscan" and len(args) == 2:
        coordinates, intervals = (), (0,)
    else:
        return None
    try:
        if not all(math.isfinite(float(args[i])) for i in (*coordinates, len(args) - 1)):
            return None
        if not all(re.fullmatch(r"[+]?[0-9]{1,18}", args[i]) for i in intervals):
            return None
        counts = tuple(int(args[i]) for i in intervals)
    except ValueError:
        return None
    if macro == "loopscan":
        return counts[0] or None
    return math.prod(value + 1 for value in counts)


def _end_marker(entry):
    if type(entry.get("end_time", getlink=True)) is not h5py.HardLink:
        return ""
    dataset = entry.get("end_time")
    if not isinstance(dataset, h5py.Dataset) or dataset.shape not in {(), (1,)}:
        return ""
    if h5py.check_string_dtype(dataset.dtype) is None:
        return ""
    value = dataset.asstr()[()] if dataset.shape == () else dataset.asstr()[0]
    return value.strip()


def _spec_identity(proof):
    return (proof.path, proof.resolved, proof.device, proof.inode, proof.content)


def _same_files(files, state):
    for expected in files:
        current = state(Path(expected.path))
        if current != expected or execution_graph._resolved(current) != execution_graph._resolved(expected):
            return False
    return True


def _graph_files(graph):
    stamp = graph.stamp
    return tuple(dict.fromkeys((stamp.file, *stamp.members,
                                *(member.file for member in stamp.external_members),
                                *stamp.dependency_files)))


@dataclass(frozen=True, slots=True)
class _WaitingSpecMember:
    state: SourceFileState
    raw: tuple[SourceFileState, ...]
    scan_key: tuple | None
    reason: str


class _ObservationSweep:
    def __init__(self, observer, cancelled, retired):
        self.observer = observer
        self.selection = observer.selection
        self.cancelled = cancelled
        self.retired = retired
        self.states = {}
        self.keys = {}
        self.listings = {}
        self.members = {}
        self.roots = {}
        self.graphs = {}
        self.next_spec = {}
        self.next_classification = {}
        self.next_listings = {}
        self.next_members = {}
        self.next_waiting = {}
        self.waiting_keys = {}
        self.next_roots = {}
        self.next_late = {}

    def check(self):
        if self.cancelled is not None and self.cancelled():
            raise stitch_sources.SourceSearchCancelled("Live scan observation stopped")

    def state(self, path):
        if path not in self.states:
            self.states[path] = SourceFileState.capture(path)
        return self.states[path]

    def spec_keys(self, path):
        if path not in self.keys:
            try:
                state = self.state(path)
                saved = self.observer._spec_keys.get(path)
                if stat_identity.IDENTITY_CARRIES_CTIME and saved is not None and saved[0] == state:
                    keys = saved[1]
                else:
                    keys = tuple(spec.list_spec_scans(path))
                self.next_spec[path] = state, keys
                self.keys[path] = keys
            except _WAIT_ERRORS:
                self.keys[path] = ()
        return self.keys[path]

    def inventory(self, directory, recursive, name_ok, check):
        files = _walk_files(directory, recursive, check=check)
        image.share_image_inventory(files)
        spec_paths, nexus_paths, eiger_paths = [], [], []
        for path in files:
            check()
            if _nexus_is_candidate(path) and name_ok(path.stem):
                kind, _, _ = self.listing(path)
                if kind is SourceKind.NEXUS_STACK:
                    nexus_paths.append(path)
                elif kind is SourceKind.EIGER_MASTER:
                    eiger_paths.append(path)
            elif path.suffix.lower() not in stitch_sources._NOT_SPEC and name_ok(path.name):
                try:
                    state = self.state(path)
                    previous = self.observer._classification.get(path)
                    is_spec = (previous[1] if previous is not None and previous[0] == state
                               else spec.is_spec_file(path, raise_access_errors=True))
                    self.next_classification[path] = state, is_spec
                    if is_spec:
                        spec_paths.append(path)
                except OSError:
                    continue
        return spec_paths, (*nexus_paths, *eiger_paths)

    def nexus_listing(self, path):
        kind, entries, _ = self.listing(path)
        return kind, entries

    def listing(self, path):
        if path in self.listings:
            return self.listings[path]
        try:
            state = self.state(path)
            baseline = self.retired.get(path)
            saved = self.observer._late.get(path) if baseline is not None else self.observer._listings.get(path)
            if baseline is not None and state == baseline[0]:
                value = SourceKind.NEXUS_STACK, baseline[1], ""
            elif saved is not None and saved[0] == state:
                value = saved[1]
            else:
                value = self.read_listing(path, state, entries_only=baseline is not None)
            if baseline is not None:
                self.next_late[path] = state, value
            else:
                self.next_listings[path] = state, value
        except _WAIT_ERRORS as error:
            value = SourceKind.NEXUS_STACK, (), str(error)
        self.listings[path] = value
        return value

    @staticmethod
    def read_listing(path, state, *, entries_only):
        with h5py.File(path, "r") as stream:
            if any(value is None for _, value in stream.items()):
                raise ValueError(f"{path.name}: the root has an unresolved entry link")
            entries = tuple(nexus.list_entries_from_open(stream))
            if entries_only:
                result = SourceKind.NEXUS_STACK, entries, ""
            elif has_processed_output_markers_file(stream):
                result = SourceKind.PROCESSED_NEXUS, (), "processed output"
            else:
                is_eiger = any(image.eiger_vendor_master(stream[key]) for key in entries)
                kind = SourceKind.EIGER_MASTER if is_eiger else SourceKind.NEXUS_STACK
                result = kind, entries, ""
        if not state.matches_disk():
            raise ValueError(f"{path.name} changed while listing its entries")
        return result

    def graph(self, path, key, kind):
        identity = path, key
        if identity not in self.graphs:
            graph = execution_graph.qualify_source_execution_graph(
                SourceSpec(path, kind, entry=key),
                cancelled=self.cancelled, infer_point_images=False)
            self.graphs[identity] = graph
        return self.graphs[identity]

    def container(self, path):
        if path in self.roots:
            return self.roots[path]
        kind, entries, reason = self.listing(path)
        state = self.state(path)
        saved = self.observer._roots.get(path)
        if reason:
            raise ValueError(reason)
        if kind is SourceKind.PROCESSED_NEXUS:
            raise _Skipped(f"{path.name} is a processed output")
        if not entries:
            raise ValueError(f"{path.name}: waiting for NeXus entries")
        if saved is not None and saved[0] == state and all(
                _same_files(member.files, self.state) for _, _, member in saved[2].entries):
            witness = saved[2]
        else:
            members = []
            with h5py.File(path, "r") as stream:
                if kind is SourceKind.EIGER_MASTER and not is_bluesky_nxwriter(stream):
                    raise _Skipped(f"{path.name} is a detector-only Eiger master")
                for key in entries:
                    self.check()
                    entry = stream[key]
                    if not is_bluesky_nxwriter(entry):
                        raise ValueError(f"{path.name}:{key} is waiting for NXWriter completion")
                    marker = _end_marker(entry)
                    if not marker:
                        raise ValueError(f"{path.name}:{key} has no entry-local end_time")
                    paths = tuple(nexus._exact_entry_stack_paths(stream, path, key))
                    member = stitch_sources.PlannedStitchMember(path, key, kind)
                    graph = self.graph(path, key, kind)
                    metadata_paths = nexus.live_bluesky_metadata_selectors(entry, key, source_ub=self.observer.source_ub)
                    metadata_files = execution_graph.linked_metadata_dependencies(
                        state, metadata_paths, cancelled=self.cancelled)
                    files = tuple(dict.fromkeys((*_graph_files(graph), *metadata_files)))
                    if tuple(graph.dataset_paths) != paths or not _same_files(files, self.state):
                        raise ValueError(f"{path.name}:{key} detector graph is changing")
                    source = SourceSpec(path, kind, entry=key)
                    count = graph.stamp.frame_count
                    points = _nexus_point_count(source, count, entry=entry)
                    detection = detect_point_images(source, graph.group_key, points, count) if points else None
                    if not self.selection.nexus_entry or key == self.selection.nexus_entry:
                        source = _confirmed_source(self.selection, source, detection=detection)
                    grouping = source.options.get("point_images")
                    if grouping is None:
                        labels = tuple(range(count))
                        mapping = paths, count, graph.stamp.external_members
                    else:
                        policy = PointImagePolicy.from_dict(grouping)
                        if not points or points * policy.images_per_point != count:
                            raise ValueError(f"{path.name}:{key} no longer has the confirmed point grouping")
                        labels = tuple(range(points))
                        mapping = (paths, count, graph.stamp.external_members,
                                   tuple(tuple(policy.raw_indices(label)) for label in labels))
                    members.append((key, marker, LiveMemberWitness(
                        member, source, (), None, False, labels, mapping, files, (graph,))))
            if not state.matches_disk():
                raise ValueError(f"{path.name} changed while checking completion")
            witness = LiveContainerWitness(state, tuple(members))
        self.roots[path] = witness
        self.next_roots[path] = state, (kind, entries, ""), witness
        return witness

    def waiting_scan_key(self, path, key):
        if path not in self.waiting_keys:
            self.waiting_keys[path] = spec.spec_waiting_scan_keys(path)
        return self.waiting_keys[path].get(key)

    def spec_member(self, member):
        if member in self.members:
            return self.members[member]
        path, key = member.path, member.key
        if key not in self.spec_keys(path):
            raise ValueError(f"{path.name}: waiting for SPEC scan {key}")
        source = stitch_sources.planned_member_source(self.selection, self.observer.read_options, member)
        options = source.options
        paths = tuple(find_spec_image_files(options["image_dir"], path.stem, key, options["image_stem"]))
        raw = tuple(self.state(path) for path in paths)
        state = self.state(path)
        waiting = self.observer._waiting.get(member)
        can_reuse_wait = stat_identity.IDENTITY_CARRIES_CTIME
        if can_reuse_wait and waiting is not None and waiting.raw == raw:
            if waiting.state == state or (
                    waiting.scan_key is not None and waiting.scan_key == self.waiting_scan_key(path, key)):
                self.next_waiting[member] = replace(waiting, state=state)
                raise ValueError(waiting.reason)

        def wait(reason, *, proved=True):
            if can_reuse_wait:
                scan_key = self.waiting_scan_key(path, key) if proved else None
                self.next_waiting[member] = _WaitingSpecMember(state, raw, scan_key, reason)
            raise ValueError(reason)

        saved = self.observer._members.get(member)
        same_media = (saved is not None and saved[1] == raw and
                      _same_files(saved[2].files, self.state))
        if same_media and saved[0] == state and stat_identity.IDENTITY_CARRIES_CTIME:
            witness = saved[2]
        else:
            proof = spec.spec_occurrence(path, key)
            if proof is None:
                wait(f"{path.name}: waiting for complete SPEC scan {key}", proved=False)
            command, successor = spec.spec_scan_completion_header(path, key)
            planned = _planned_points(command)
            needs_successor = planned is None or proof.rows < planned
            if needs_successor and not successor:
                wait(f"{path.name}: waiting for completion of SPEC scan {key}")
            if not proof.rows:
                raise _Skipped(f"{path.name}: SPEC scan {key} has no recorded points")
            identity = _spec_identity(proof)
            if same_media and saved[2].occurrence_identity == identity:
                witness = replace(saved[2], occurrence=proof, successor_required=needs_successor)
            else:
                if not paths:  # no detector files yet: do not open the scan
                    wait(f"{path.name}: waiting for detector coverage of scan {key}")
                source = _confirmed_source(self.selection, source)
                opened = open_source(source)
                try:
                    labels = tuple(opened.frame_indices)
                    if len(labels) != proof.rows or not opened.capabilities.has_raw_references:
                        # Not cached: coverage of an opened source can depend on linked files outside the key.
                        raise ValueError(f"{path.name}: waiting for detector coverage of scan {key}")
                    graphs = []
                    for detector in paths:
                        self.check()
                        if detector.suffix.lower() not in stitch_sources._CONTAINER_SUFFIXES:
                            continue
                        entry = options.get("nexus_entry") or "entry"
                        native = descriptor.describe_container(detector, entry=entry)
                        if native.resolved_entry is None:
                            raise ValueError(f"{detector.name} has no detector entry")
                        graphs.append(self.graph(detector, native.resolved_entry, native.kind))
                    files = tuple(dict.fromkeys((*raw, *(value for graph in graphs for value in _graph_files(graph)))))
                    if not _same_files(files, SourceFileState.capture):
                        raise ValueError(f"{path.name}: detector inputs changed while observing scan {key}")
                    mapping = _spec_mapping(opened, labels)
                    witness = LiveMemberWitness(member, source, identity, proof, needs_successor,
                                                labels, mapping, files, tuple(graphs), paths)
                finally:
                    _close(opened)
        self.members[member] = witness
        self.next_members[member] = state, raw, witness
        return witness

    def evidence(self, group):
        containers = tuple(self.container(path) for path in group.paths
                           if group.identity[0] in {"nexus", "directory-nexus"})
        entries = {(value.member.path, key): value for root in containers
                   for key, _, value in root.entries}
        if not group.members:
            raise ValueError("Waiting for the selected exact NeXus entry")
        if self.selection.nexus_entry and containers and any(
                self.selection.nexus_entry not in {key for key, _, _ in root.entries}
                for root in containers):
            raise ValueError(f"Waiting for exact NeXus entry {self.selection.nexus_entry!r}")
        members = tuple(self.spec_member(member) if member.kind is SourceKind.SPEC
                        else entries[(member.path, member.key)] for member in group.members)
        select = self.observer.select_labels
        labels = tuple(member.labels if select is None else tuple(select(member.labels)) for member in members)
        if not all(labels):
            raise _Skipped("No frames selected for this scan group")
        return LiveEvidence(members, containers, labels)


def _close(source):
    close = getattr(source, "close", None)
    if close is not None:
        close()


_INSPECT_GROUPING = object()


def _confirmed_source(selection, source, *, detection=_INSPECT_GROUPING):
    key = source.options.get("scan") or source.entry
    policy = next((choice for uri, scan, choice in selection.point_image_choices
                   if uri == str(source.uri) and scan == key), None)
    if source.kind is SourceKind.SPEC and policy is None:
        return source
    if detection is _INSPECT_GROUPING:
        detection = inspect_point_images(source)
    if detection is None and source.kind is SourceKind.SPEC:
        return source
    if detection is not None and (policy is None or policy.images_per_point != detection.images_per_point):
        raise _Skipped(f"{Path(source.uri).name}:{key} needs confirmed point-image grouping")
    if policy is None:
        return source
    return replace(source, options={**source.options, "point_images": policy.to_dict()})


class LiveScanObserver:
    """One frozen selection's poll state. Use on the poll worker only."""

    def __init__(self, selection, read_image_kwargs, *, select_labels=None, stillness=2.0, source_ub=False):
        self.selection = selection
        self.read_options = dict(read_image_kwargs)
        self.select_labels = select_labels
        self.stillness = stillness
        self.source_ub = source_ub
        self._spec_keys = {}
        self._classification = {}
        self._listings = {}
        self._members = {}
        self._roots = {}
        self._late = {}
        self._still = {}
        # Pre-open waits retain the verified scan key and the current image revisions.
        self._waiting = {}

    def invalidate(self, identity):
        self._still.pop(identity, None)
        self._spec_keys.clear()
        self._members.clear()
        self._waiting.clear()
        self._roots.clear()

    def poll(self, *, now, consumed=(), cancelled=None):
        consumed = tuple(item.consumed() if isinstance(item, LiveScanHandoff) else item for item in consumed)
        already = {item.identity for item in consumed}
        retired = {Path(root.path): (root, entries) for item in consumed for root, entries in item.containers}
        sweep = _ObservationSweep(self, cancelled, retired)
        rows, late, still = [], [], {}
        with image.shared_image_listings(all_stems=True), spec.spec_capture_scope():
            groups = stitch_sources.list_planned_stitch_groups(
                self.selection, check=sweep.check, allow_pending=True, spec_keys=sweep.spec_keys,
                nexus_listing=sweep.nexus_listing, directory_inventory=sweep.inventory)
            for group in groups:
                if group.identity in already:
                    continue
                try:
                    evidence = sweep.evidence(group)
                except (_Skipped, PointImagesRequired) as error:
                    rows.append(LiveScanRow(group, "skipped", str(error)))
                except _WAIT_ERRORS as error:
                    rows.append(LiveScanRow(group, "waiting", str(error) or "Waiting for recorded data"))
                else:
                    previous = self._still.get(group.identity)
                    since = previous[1] if previous is not None and previous[0] == evidence else now
                    still[group.identity] = evidence, since
                    if now - since >= self.stillness:
                        rows.append(LiveScanRow(group, "ready", evidence=evidence))
                    else:
                        rows.append(LiveScanRow(group, "waiting", "Waiting for source stillness"))
            for path, (_, baseline) in retired.items():
                sweep.check()
                known = set(baseline)
                _, entries, _ = sweep.listing(path)
                late.extend((str(path), key) for key in entries if key not in known)
            sweep.check()
        self._spec_keys = sweep.next_spec
        self._classification = sweep.next_classification
        self._listings = sweep.next_listings
        self._members = sweep.next_members
        self._waiting = sweep.next_waiting
        self._roots = sweep.next_roots
        self._late = sweep.next_late
        self._still = still
        return LiveScanObservation(tuple(rows), tuple(late))


class LiveHandoffChanged(ValueError):
    code = "LIVE_HANDOFF_CHANGED"


class LiveOutputChanged(ValueError):
    code = "LIVE_OUTPUT_CHANGED"

    def __init__(self, target, message="Output changed after Live preflight"):
        super().__init__(message)
        self.live_blocked_target = target


def require_current_handoff(handoff, *, cancelled=None, readiness=True):
    """Check this handoff against disk without re-resolving its group."""
    from xrd_tools.io.metadata import SpecOccurrenceVerification

    if handoff is None:
        return
    members = handoff.evidence.members
    proofs = tuple(member.occurrence for member in members if member.occurrence is not None)
    successors = tuple(member.occurrence for member in members if member.successor_required)
    verification = SpecOccurrenceVerification(
        proofs, cancelled=cancelled, successor_proofs=successors if readiness else ())
    states = {}

    def state(path):
        if cancelled is not None and cancelled():
            raise InterruptedError("Live scan admission stopped")
        if path not in states:
            states[path] = SourceFileState.capture(path)
        return states[path]

    try:
        with image.shared_image_listings(all_stems=True):
            for member in members:
                if readiness and member.occurrence is not None:
                    source = member.source
                    options = source.options
                    current = tuple(find_spec_image_files(options["image_dir"], Path(source.uri).stem,
                                                         options["scan"], options["image_stem"]))
                    if current != member.raw_paths:
                        raise LiveHandoffChanged(f"{member.member.path.name}:{member.member.key} detector mapping changed")
        for member in members:
            if member.occurrence is not None and not verification.holds(member.occurrence):
                raise LiveHandoffChanged(f"{member.member.path.name}:{member.member.key} changed after Live observation")
            if not _same_files(member.files, state):
                raise LiveHandoffChanged(f"{member.member.path.name}:{member.member.key} detector inputs changed")
        if readiness:
            for container in handoff.evidence.containers:
                if not _same_files((container.root,), state) or any(
                        not _same_files(member.files, state) for _, _, member in container.entries):
                    raise LiveHandoffChanged(f"{Path(container.root.path).name} present entries or linked inputs changed")
    except InterruptedError:
        raise
    except LiveHandoffChanged:
        raise
    except _WAIT_ERRORS as error:
        raise LiveHandoffChanged(str(error) or "Live source became unavailable") from error


def _spec_mapping(source, labels):
    return tuple(tuple((str(path), index, source._container_dataset_paths.get(path))
                       for path, index in source.point_raw_locators(label))
                 for label in labels)


def require_snapshot(expected, requested, source, proof, labels, graph):
    """Require that the actual native metadata reader answered the detached observation."""
    if expected is None:
        return
    seen = expected.source
    if (Path(requested.uri) != Path(seen.uri) or requested.kind is not seen.kind
            or requested.entry != seen.entry or labels != expected.labels):
        raise LiveHandoffChanged("Native source identity or frame labels differ from the Live observation")
    for key in ("point_images", "scan", "nexus_entry"):
        if seen.options.get(key) != requested.options.get(key):
            raise LiveHandoffChanged(f"Native {key} differs from the Live observation")
    if proof is not None:
        if (_spec_identity(proof) != expected.occurrence_identity
                or _spec_mapping(source, labels) != expected.mapping):
            raise LiveHandoffChanged(f"SPEC scan {proof.key} or its detector mapping changed")
    elif expected.occurrence is not None:
        raise LiveHandoffChanged("Native SPEC source has no bound scan occurrence")
    elif graph is None or _graph_projection(graph) != _graph_projection(expected.graphs[0]):
        raise LiveHandoffChanged("Native detector selector or dependency graph changed")


def _graph_projection(graph):
    stamp = graph.stamp
    return (graph.execution_source.entry, graph.dataset_paths, graph.detector_shape, graph.native_dtype,
            stamp.frame_count, stamp.external_members, stamp.dependency_files, stamp.source_aliases)


def require_native_members(handoff, receipts):
    if handoff is not None and tuple(receipt.selected_labels for receipt in receipts) != handoff.evidence.labels:
        raise LiveHandoffChanged("Native selected frames differ from the Live observation")


def snapshot_member(handoff, ordinal):
    if handoff is None:
        return None
    if ordinal >= len(handoff.evidence.members):
        raise LiveHandoffChanged("Native group gained a member")
    return handoff.evidence.members[ordinal]


def live_output_current(handoff, target, check):
    """Check Append currency and the exact target revision in the same worker."""
    if handoff is None:
        return check(), None
    before = observe_live_target(target)
    current = check()
    after = observe_live_target(target)
    if before != after:
        raise LiveOutputChanged(before, "Output changed while checking Append currency")
    return current, after


@dataclass(frozen=True, slots=True)
class LiveTarget:
    name: str
    alias: str
    parent: str
    revision: tuple

    @property
    def keys(self):
        device, inode = self.revision[1:3]
        values = {("path", self.name), ("parent", self.alias)}
        if device is not None and inode:
            values.add(("object", device, inode))
        return frozenset(values)


def observe_live_target(path):
    from xrd_tools.io.output_transaction import (
        TargetChanged, _lease_key, _normalize_target, capture_target_revision, target_revision_key,
    )

    normalized = _normalize_target(path)
    parent = os.path.normcase(os.path.realpath(Path(path).parent))
    before = _lease_key(normalized)
    try:
        revision = target_revision_key(capture_target_revision(path))
    except TargetChanged as error:
        raise OSError(str(error)) from error
    after = _lease_key(normalized)
    if before != after or parent != os.path.normcase(os.path.realpath(Path(path).parent)):
        raise OSError(f"Output parent changed while checking {path}")
    return LiveTarget(normalized, after, parent, revision)


def require_live_target(request, *, allow_created_parent=False):
    target = request.live_target_revision
    try:
        current = observe_live_target(request.module.output.target)
    except OSError as error:
        unavailable = LiveOutputChanged(target, str(error))
        unavailable.code = "LIVE_OUTPUT_UNAVAILABLE"
        raise unavailable from error
    if current == target:
        return
    if (allow_created_parent and target is not None and not target.revision[0]
            and target.alias == target.name and current.name == target.name
            and current.parent == target.parent and current.revision == target.revision):
        # Normal native Run can create a previously absent Save Path.
        return
    raise LiveOutputChanged(target)


def live_asset_revisions(paths):
    revisions = []
    with spec.spec_capture_scope():
        for path in paths:
            try:
                if isinstance(path, tuple):
                    proof = spec.spec_occurrence(*path)
                    revisions.append(None if proof is None else _spec_identity(proof))
                else:
                    state = SourceFileState.capture(path)
                    revisions.append((state, execution_graph._resolved(state)))
            except OSError as error:
                revisions.append(error.errno)
            except (ValueError, LookupError, SfError, SpecOccurrenceUnsettled):
                revisions.append(None)
    return tuple(revisions)


def capture_live_assets(form, handoff):
    """Observe frozen per-job asset references for a relevant refusal retry."""
    if handoff is None:
        return None
    from xrd_tools.integrate.stitch_calibration import xu_partner_path

    project = Path(form.project_root)
    geometry = getattr(form, "geometry_path", None)
    if geometry is None:
        geometry = form.geometry_asset.locator
    correction = getattr(form, "corrections", form)
    ub = getattr(form, "ub", None)
    paths = [project / geometry, *(project / value for key in ("mask_path", "dark_path", "flat_path")
                                  if (value := getattr(correction, key, None)))]
    if getattr(form, "backend", "xu_hist") == "xu_hist":
        partner = xu_partner_path(paths[0])
        paths.extend((partner, partner.with_suffix(partner.suffix + ".provenance.json")))
    if ub is not None and ub.spec_path is not None:
        paths.append((Path(ub.spec_path), ub.scan))
    values = tuple(dict.fromkeys(paths))
    return values, live_asset_revisions(values)
