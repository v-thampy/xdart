from __future__ import annotations; import hashlib, json, math, os, tempfile, threading; from bisect import bisect_right; from collections.abc import Mapping; from contextlib import contextmanager; from dataclasses import dataclass, fields; from pathlib import Path, PurePosixPath; from types import MappingProxyType, SimpleNamespace; from typing import Any, Callable, Literal, NamedTuple; from xrd_tools.io.append import _replacement_hard_group, decode_replacement_lineage, science_fingerprint; from xrd_tools.io.output_transaction import StreamTerminal, TargetSnapshot, capture_target_snapshot, revalidate_stream_terminal, stream_terminal_object_revision; from xrd_tools.session.policy import FlushPolicy, SessionPolicy, SessionResourceAllocation, SessionResourceRequirements, requirements_from, resolve_session_policy
_REQUESTS = {"workers", "reduction_inflight", "queue_depth", "owner_block_bytes", "staging_items", "record_heavy_items", "publication_heavy_items", "thumbnail_items", "record_items", "publication_items"}; _STAGES = {"qualify", "read", "reduce", "write", "settle"}
# Keep headless replacement admission aligned with the GUI's 256 MiB decoded
# scientific-mask ceiling without importing GUI policy into xrd_tools.  Both
# the persisted int64 index vector and its expanded bool mask must fit the cap.
_MAX_PERSISTED_MASK_BYTES = 256 * 1024 ** 2
_MAX_HDF_OWNER_CHILDREN = 1_000_000
_MAX_HDF_OWNER_NAME_BYTES = 4096
_MAX_HDF_OWNER_NAMES_BYTES = 64 << 20
_MAX_HDF_LINK_VALUE_BYTES = 4096
_MAX_HDF_LINK_VALUES_BYTES = 64 << 20
_MAX_HDF_EXTERNAL_SLOTS = 65_536
_MAX_HDF_EXTERNAL_PATH_BYTES = 4096
_MAX_HDF_EXTERNAL_PATHS_BYTES = 64 << 20
_MAX_HDF_EXTERNAL_ADDRESS = (1 << 63) - 1
_REINTEGRATE_PLAN_API_VERSION = 3
_REINTEGRATE_RECIPE_VERSION = 3
_REINTEGRATE_SCIENCE_API_VERSION = 1
class ReintegrateCancelled(RuntimeError): pass
class _PersistedMaskSpec(NamedTuple): retained_bytes: int; decode_bytes: int
class _ArtifactInspection(NamedTuple): labels: tuple[int, ...]; detector_shape: tuple[int, int]; native_dtype: str; persisted_shared_science: Mapping[str, Any]; persisted_selected_plan: Mapping[str, Any]; acquisition_fingerprint: str; source_base: str; append_lineage: bytes | None; gi_values: Mapping[int, float]; mask_spec: _PersistedMaskSpec; mask: Any | None; raw_options: Mapping[str, Any] | None; topology: _SourceTopology
class _RuntimeOutcome(NamedTuple): disposition: str; committed: tuple[int, ...]; dropped: tuple[int, ...]; diagnostics: tuple[str, ...]; audit: str | None; terminal: Any | None
def _reject(condition: Any, message: str) -> None: return None if not condition else (_ for _ in ()).throw(ValueError(message))
def _diagnostic(value: Any) -> str:
    try: return str(value).encode("utf-8")[:1024].decode("utf-8", "ignore")
    except BaseException: return type(value).__name__
def _keys(value: Mapping[str, Any], expected: set[str], role: str) -> None: _reject(type(value) is not dict or set(value) != expected or any(type(k) is not str for k in value), f"{role} has a noncanonical keyset")
def _plain(value: Any) -> Any: return {k: _plain(v) for k, v in value.items()} if isinstance(value, Mapping) else [_plain(v) for v in value] if isinstance(value, tuple) else value
def _freeze(value: Any) -> Any:
    if type(value) is dict: _reject(any(type(k) is not str for k in value), "noncanonical JSON object key"); return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if type(value) is list: return tuple(_freeze(v) for v in value)
    if value is None or type(value) in {str, bool, int} or type(value) is float and math.isfinite(value): return value
    raise ValueError(f"noncanonical JSON value {type(value).__name__}")
def _canonical(value: Any) -> bytes: return json.dumps(_plain(value), sort_keys=True, separators=(",", ":")).encode()
def _digest(value: Any) -> str: return hashlib.sha256(_canonical(value)).hexdigest()
def _event(token: threading.Event | None, *, honor: bool = True) -> None: (None if token is None or type(token) is threading.Event else (_ for _ in ()).throw(TypeError("cancellation token must be an exact threading.Event or None"))); (None if not honor or token is None or not token.is_set() else (_ for _ in ()).throw(ReintegrateCancelled("reintegration cancelled")))


def _relocated_path(value: str, stored_root: str, selected_root: str) -> str:
    """Lexically relocate only paths owned by the persisted Project root."""
    current = _normalized_absolute_path(
        value, "REPLACEMENT_SOURCE_RELOCATION_UNSUPPORTED")
    if not stored_root:
        return current
    old = _normalized_absolute_path(
        stored_root, "REPLACEMENT_SOURCE_RELOCATION_UNSUPPORTED")
    new = _normalized_absolute_path(
        selected_root, "REPLACEMENT_SOURCE_RELOCATION_UNSUPPORTED")
    try:
        owned = os.path.commonpath((old, current)) == old
    except ValueError:
        owned = False
    if not owned:
        return current
    relative = os.path.relpath(current, old)
    _reject(relative == os.pardir or relative.startswith(os.pardir + os.sep),
            "REPLACEMENT_SOURCE_RELOCATION_UNSUPPORTED")
    return os.path.normcase(os.path.abspath(os.path.join(new, relative)))


def _refresh_state(state: dict[str, Any], stored_root: str,
                   selected_root: str) -> None:
    path = _relocated_path(state["path"], stored_root, selected_root)
    try:
        observed = os.stat(path)
    except OSError as error:
        raise ValueError("REPLACEMENT_SOURCE_REVISION_UNAVAILABLE") from error
    _reject(
        int(observed.st_size) != state["size"]
        or int(observed.st_mtime_ns) != state["mtime_ns"],
        "REPLACEMENT_SOURCE_REVISION_CHANGED",
    )
    state.update({
        "path": path,
        "ctime_ns": int(observed.st_ctime_ns),
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
    })


def _refresh_append_member(member: dict[str, Any], stored_root: str,
                           selected_root: str) -> None:
    path = _relocated_path(member["path"], stored_root, selected_root)
    try:
        observed = os.stat(path)
    except OSError as error:
        raise ValueError("REPLACEMENT_SOURCE_REVISION_UNAVAILABLE") from error
    _reject(
        int(observed.st_size) != member["size"]
        or int(observed.st_mtime_ns) != member["mtime_ns"],
        "REPLACEMENT_SOURCE_REVISION_CHANGED",
    )
    member["path"] = path


def _relocate_source_context(
    stored_root: str,
    selected_root: str | None,
    execution: Mapping[str, Any],
    lineage: Mapping[str, Any] | None,
) -> tuple[str, Mapping[str, Any], Mapping[str, Any] | None, bytes | None]:
    """Freeze a moved raw graph under one exact selected Project root."""
    if selected_root is None:
        selected_root = stored_root
    selected = _normalized_absolute_path(
        selected_root, "REPLACEMENT_SOURCE_RELOCATION_UNSUPPORTED")
    if stored_root:
        _normalized_absolute_path(
            stored_root, "REPLACEMENT_SOURCE_RELOCATION_UNSUPPORTED")
    relocated = _plain(execution)
    _reject(type(relocated) is not dict,
            "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
    _refresh_state(relocated, stored_root, selected)
    for key in ("member_stamps", "dependency_files"):
        for state in relocated[key]:
            _refresh_state(state, stored_root, selected)
    for member in relocated["external_members"]:
        _refresh_state(member["file"], stored_root, selected)
    for value in relocated["admitted_motor_values"]:
        value["source_path"] = _relocated_path(
            value["source_path"], stored_root, selected)
    for value in relocated["metadata_sources"]:
        value["source_path"] = _relocated_path(
            value["source_path"], stored_root, selected)
        if value["metadata_file"] is not None:
            _refresh_state(value["metadata_file"], stored_root, selected)
    from xrd_tools.io.record_writer import (
        WriterStateError, _validate_replacement_execution,
    )
    try:
        relocated = _validate_replacement_execution(relocated)
    except WriterStateError as error:
        raise ValueError("REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED") from error

    relocated_lineage = None
    lineage_bytes = None
    if lineage is not None:
        relocated_lineage = _plain(lineage)
        _reject(type(relocated_lineage) is not dict,
                "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
        relocated_lineage["source_base"] = selected
        for epoch in relocated_lineage["epochs"]:
            source = epoch["source"]
            _refresh_append_member(source, stored_root, selected)
            for key in ("image_members", "external_members"):
                for member in source[key]:
                    _refresh_append_member(member, stored_root, selected)
        from xrd_tools.io.append import _lineage_labels
        try:
            _lineage_labels(relocated_lineage)
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError("REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED") from error
        lineage_bytes = _canonical(relocated_lineage)
    return selected, relocated, relocated_lineage, lineage_bytes
def _mask_spec(retained, decode, shape, role="persisted-mask resources"):
    pixels = int(shape[0]) * int(shape[1]) if type(shape) in {tuple, list} and len(shape) == 2 and all(type(value) is int and value > 0 for value in shape) else -1
    _reject(any(type(value) is not int or value < 0 for value in (retained, decode)) or bool(retained) != bool(decode) or retained > _MAX_PERSISTED_MASK_BYTES or decode > _MAX_PERSISTED_MASK_BYTES or retained and (retained != pixels or decode < 8 or decode % 8 or decode > 8 * pixels), f"{role} are malformed")
    return _PersistedMaskSpec(retained, decode)
def _normalized_absolute_path(value, role: str) -> str:
    _reject(type(value) is not str or not value or not os.path.isabs(value), role)
    normalized = os.path.normcase(os.path.abspath(os.path.normpath(value)))
    _reject(normalized != value, role)
    return value


def _resolve_source_locator(locator: str, source_root: str) -> Path:
    _reject(type(locator) is not str or not locator or "\\" in locator,
            "REPLACEMENT_SOURCE_RELOCATION_UNSUPPORTED")
    shown = PurePosixPath(locator)
    if shown.is_absolute():
        _reject(str(shown) != locator or "." in shown.parts or ".." in shown.parts,
                "REPLACEMENT_SOURCE_RELOCATION_UNSUPPORTED")
        return Path(str(shown))
    from xrd_tools.io.read import resolve_project_source_path
    try:
        return resolve_project_source_path(
            locator, source_root, must_exist=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("REPLACEMENT_SOURCE_RELOCATION_UNSUPPORTED") from error


def _source_route(path: Path) -> str:
    suffix = path.suffix.lower()
    route = (
        "fabio" if suffix in {".tif", ".tiff", ".edf", ".cbf", ".img"}
        or path.name.lower().endswith(".mar3450")
        else "raw" if suffix == ".raw"
        else "hdf5" if suffix in {".h5", ".hdf5", ".nxs", ".nexus", ".cxi"}
        else None
    )
    _reject(route is None, "REPLACEMENT_SOURCE_FORMAT_UNSUPPORTED")
    return route
def _decoder_input_paths(fact):
    execution = fact["source_execution"]; adapter = execution["adapter_id"]
    paths = [fact["path"], execution["path"]]
    if adapter == "tiff_series":
        paths.extend(member["path"] for member in execution["member_stamps"])
    lineage = fact["append_lineage"]
    if lineage is not None:
        for epoch in lineage["epochs"]:
            source = epoch["source"]
            if source["adapter_id"] == "tiff_series":
                paths.extend(member["path"] for member in source["image_members"])
            elif source["adapter_id"] == "image_file":
                paths.append(source["path"])
    return tuple(dict.fromkeys(str(path) for path in paths))
def _native_dtype(value): import numpy as np; allowed = {"|i1", "|u1"} | {f"{end}{kind}{size}" for end in "<>" for kind, sizes in (("i", "248"), ("u", "248"), ("f", "248")) for size in sizes}; _reject(type(value) is not str or value not in allowed, "plan facts are malformed"); return np.dtype(value)
def _validate_external_intervals(intervals, count: int):
    _reject(any(type(row) not in {tuple, list} or len(row) != 3 or any(type(x) is not int for x in row) for row in intervals), "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED"); value = tuple(tuple(row) for row in intervals); cursor = 0
    for position, (start, stop, epoch) in enumerate(value): _reject(start != cursor or stop <= start or epoch != position, "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED"); cursor = stop
    _reject(cursor != count, "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED"); return value
def _revision(path: Path):
    selected = Path(os.path.abspath(path.expanduser()))
    try: before = selected.resolve(strict=True); stat = selected.stat(); target = before.stat(); after = selected.resolve(strict=True)
    except OSError as error: raise ValueError("REPLACEMENT_SOURCE_REVISION_UNAVAILABLE") from error
    value = (int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns), int(stat.st_dev), int(stat.st_ino)); _reject(os.path.normcase(os.path.normpath(before)) != os.path.normcase(os.path.normpath(after)) or value != (int(target.st_size), int(target.st_mtime_ns), int(target.st_ctime_ns), int(target.st_dev), int(target.st_ino)), "REPLACEMENT_SOURCE_REVISION_CHANGED"); return (str(before), *value)
def _revision_inventory(states, current):
    authority = {os.path.normcase(os.path.normpath(v["path"])): v for v, _role in current}; raws, targets = {}, {}
    for state, role in states: observed = _revision(Path(state["path"])); raw = os.path.normcase(os.path.normpath(state["path"])); expected = authority.get(raw); _reject(observed[1:3] != (expected["size"], expected["mtime_ns"]) if expected is not None else observed[1:] != tuple(state[k] for k in ("size", "mtime_ns", "ctime_ns", "device", "inode")), "REPLACEMENT_SOURCE_REVISION_CHANGED"); target = os.path.normcase(os.path.normpath(observed[0])); _reject(raw in raws and raws[raw][0] != target or target in targets and targets[target] != observed[1:], "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); raws[raw] = (target, expected or state, role, observed); targets[target] = observed[1:]
    for state, role in current: raw = os.path.normcase(os.path.normpath(state["path"])); observed = _revision(Path(state["path"])); _reject(observed[1:3] != (state["size"], state["mtime_ns"]), "REPLACEMENT_SOURCE_REVISION_CHANGED"); target = os.path.normcase(os.path.normpath(observed[0])); _reject(raw in raws and raws[raw][0] != target or target in targets and targets[target] != observed[1:], "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); raws[raw] = (target, state, role, observed); targets[target] = observed[1:]
    return raws, targets


def _execution_revisions(execution, final=None, label=None, *,
                         validated=False, full=False,
                         external_selection=None):
    from xrd_tools.io.record_writer import (
        WriterStateError, _validate_replacement_execution,
    )

    if not validated:
        try:
            execution = _validate_replacement_execution(execution)
        except WriterStateError as error:
            raise ValueError("REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED") from error
    adapter = execution["adapter_id"]
    ordinal = label - execution["first_label"] if type(label) is int else -1
    if adapter == "tiff_series":
        if full:
            states = [
                (value, "source_member")
                for value in execution["member_stamps"]
            ]
            states.extend(
                (value["metadata_file"], "image_metadata")
                for value in execution["metadata_sources"]
                if value["metadata_file"] is not None
            )
            current = ([] if final is None else [
                (value, "final_member") for value in final["image_members"]
            ])
        else:
            _reject(
                not 0 <= ordinal < len(execution["member_stamps"])
                or final is not None
                and not 0 <= ordinal < len(final["image_members"]),
                "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED",
            )
            states = [(execution["member_stamps"][ordinal], "source_member")]
            metadata = (execution["metadata_sources"][ordinal]["metadata_file"]
                        if execution["metadata_sources"] else None)
            states += [] if metadata is None else [(metadata, "image_metadata")]
            current = ([] if final is None else
                       [(final["image_members"][ordinal], "final_member")])
    elif adapter == "nexus_hdf5":
        if full:
            selected = execution["external_members"]
            final_selected = () if final is None else final["external_members"]
        elif external_selection is not None:
            execution_member, final_member = external_selection
            selected = () if execution_member is None else (execution_member,)
            final_selected = () if final_member is None else (final_member,)
        else:
            selected = tuple(
                value for value in execution["external_members"]
                if value["first"] <= ordinal < value["stop"])
            final_selected = (() if final is None else tuple(
                value for value in final["external_members"]
                if value["source_start"] <= ordinal < value["source_stop"]))
            _reject(
                len(selected) > 1 or len(final_selected) > 1,
                "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
            )
        states = ([(execution, "source_file")]
                  + [(value["file"], "external_member") for value in selected]
                  + [(value, "detector_dependency")
                     for value in execution["dependency_files"]])
        current = ([] if final is None else [(final, "final_source")]
                   + [(value, "final_external") for value in final_selected])
    else:
        states = [(execution, "source_file")]
        current = [] if final is None else [(final, "final_source")]
    if adapter != "nexus_hdf5":
        states += [(value, "detector_dependency")
                   for value in execution["dependency_files"]]
    raws, targets = _revision_inventory(states, current)
    return execution, raws, targets


def _revision_signature(revisions):
    return tuple(sorted(
        (raw, value[0], tuple(value[3]))
        for raw, value in revisions.items()
    ))


def _revision_lookup(revisions):
    """Index every admitted lexical and resolved spelling without scanning."""

    lookup = dict(revisions)
    for value in revisions.values():
        resolved = os.path.normcase(os.path.normpath(value[0]))
        prior = lookup.get(resolved)
        _reject(prior is not None and prior[3] != value[3],
                "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
        lookup.setdefault(resolved, value)
    return MappingProxyType(lookup)


class _HdfStorageSlot(NamedTuple):
    lexical_path: str
    resolved_path: str
    offset: int
    logical_start: int
    logical_stop: int


class _HdfStorageSlice(NamedTuple):
    lexical_path: str
    resolved_path: str
    offset: int
    size: int
    frame_offset: int


class _HdfDatasetRoute(NamedTuple):
    dataset_path: str
    start: int
    stop: int
    member_path: str
    frame_bytes: int
    storage_slots: tuple[_HdfStorageSlot, ...]
    storage_stops: tuple[int, ...]


class _HdfFrameRoute(NamedTuple):
    dataset_path: str
    start: int
    stop: int
    member_path: str
    storage_slices: tuple[_HdfStorageSlice, ...]


class _FrameRoute(NamedTuple):
    source_path: str
    source_state: Mapping[str, Any]
    snapshot_count: int
    snapshot_dataset_path: str | None
    self_contained: bool | None
    revision_paths: tuple[str, ...]
    hdf: _HdfFrameRoute | None


class _HdfOwnerLink(NamedTuple):
    local_path: str
    name: str
    kind: str
    lexical_filename: str | None
    resolved_target: str | None
    remote_path: str | None
    local_object: str | None


def _local_hdf_group(handle, path):
    """Walk one HDF group only after proving every ancestor is a hard link."""

    import h5py

    shown = PurePosixPath(path)
    _reject(not shown.is_absolute() or str(shown) != path,
            "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
    current = handle
    for name in shown.parts[1:]:
        _reject(type(current.get(name, getlink=True)) is not h5py.HardLink,
                "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
        current = current.get(name)
        _reject(not isinstance(current, h5py.Group)
                or current.file.id != handle.id,
                "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
    return current


def _bounded_hdf_owner_links(handle, parent, source_path, revisions):
    import h5py

    owner = _local_hdf_group(handle, parent)
    _reject(
        len(owner) > _MAX_HDF_OWNER_CHILDREN,
        "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
    )
    links = []
    name_bytes = 0
    encoded_value_bytes = 0
    for name in owner:
        _reject(type(name) is not str,
                "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
        try:
            encoded = name.encode("utf-8")
            info = owner.id.links.get_info(encoded)
        except (UnicodeError, TypeError, ValueError, RuntimeError) as error:
            raise ValueError(
                "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED"
            ) from error
        name_bytes += len(encoded)
        _reject(
            len(encoded) > _MAX_HDF_OWNER_NAME_BYTES
            or name_bytes > _MAX_HDF_OWNER_NAMES_BYTES
            or info.type not in {
                h5py.h5l.TYPE_HARD,
                h5py.h5l.TYPE_SOFT,
                h5py.h5l.TYPE_EXTERNAL,
            },
            "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
        )
        if info.type != h5py.h5l.TYPE_HARD:
            value_size = int(info.u)
            record_limit = (
                2 * _MAX_HDF_LINK_VALUE_BYTES + 3
                if info.type == h5py.h5l.TYPE_EXTERNAL
                else _MAX_HDF_LINK_VALUE_BYTES + 1
            )
            encoded_value_bytes += value_size
            _reject(
                value_size <= 0 or value_size > record_limit
                or encoded_value_bytes > _MAX_HDF_LINK_VALUES_BYTES,
                "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
            )
        links.append((name, info.type))
    signature = []
    external = []
    materialized_value_bytes = 0
    master = str(source_path.resolve(strict=True))
    for name, low_level_type in sorted(links):
        local_path = f"{parent.rstrip('/')}/{name}"
        link = owner.get(name, getlink=True)
        if type(link) is h5py.ExternalLink:
            filename, remote = link.filename, link.path
            try:
                filename_bytes = filename.encode("utf-8")
                remote_bytes = remote.encode("utf-8")
            except (AttributeError, UnicodeError) as error:
                raise ValueError(
                    "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED"
                ) from error
            materialized_value_bytes += (
                len(filename_bytes) + len(remote_bytes))
            _reject(
                low_level_type != h5py.h5l.TYPE_EXTERNAL
                or type(filename) is not str or not filename
                or type(remote) is not str or not remote
                or len(filename_bytes) > _MAX_HDF_LINK_VALUE_BYTES
                or len(remote_bytes) > _MAX_HDF_LINK_VALUE_BYTES
                or materialized_value_bytes > _MAX_HDF_LINK_VALUES_BYTES
                or not PurePosixPath(remote).is_absolute()
                or str(PurePosixPath(remote)) != remote,
                "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
            )
            lexical = str(Path(os.path.abspath(source_path.parent / filename)))
            try:
                resolved = str(Path(lexical).resolve(strict=True))
            except OSError as error:
                raise ValueError(
                    "REPLACEMENT_SOURCE_REVISION_UNAVAILABLE") from error
            _admitted_lexical_revision(lexical, resolved, revisions)
            value = _HdfOwnerLink(local_path, name, "external", filename,
                                  resolved, remote, None)
            external.append(value)
        elif type(link) is h5py.HardLink:
            _reject(low_level_type != h5py.h5l.TYPE_HARD,
                    "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
            child = owner.get(name)
            _reject(child is None or child.file.id != handle.id,
                    "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
            value = _HdfOwnerLink(
                local_path, name, "hard", None, master, None,
                f"{type(child).__name__}:{child.name}",
            )
        elif type(link) is h5py.SoftLink:
            try:
                target_bytes = link.path.encode("utf-8")
            except (AttributeError, UnicodeError) as error:
                raise ValueError(
                    "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED"
                ) from error
            materialized_value_bytes += len(target_bytes)
            _reject(
                low_level_type != h5py.h5l.TYPE_SOFT
                or type(link.path) is not str
                or len(target_bytes) > _MAX_HDF_LINK_VALUE_BYTES
                or materialized_value_bytes > _MAX_HDF_LINK_VALUES_BYTES,
                "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
            )
            value = _HdfOwnerLink(local_path, name, "soft", None, None,
                                  link.path, None)
        else:
            raise ValueError(
                "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
        signature.append(value)
    return tuple(signature), tuple(external)


def _hdf_owner_census(path, parent, revisions):
    with __import__("h5py").File(path, "r") as handle:
        _hdf_handle_revision(handle, path, revisions)
        signature, external = _bounded_hdf_owner_links(
            handle, parent, Path(path), revisions)
        _hdf_handle_revision(handle, path, revisions)
    return signature, external


def _local_hdf_dataset(handle, path):
    import h5py

    shown = PurePosixPath(path)
    _reject(not shown.is_absolute() or str(shown) != path
            or shown.name in {"", ".", ".."},
            "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
    owner = _local_hdf_group(handle, str(shown.parent))
    _reject(type(owner.get(shown.name, getlink=True)) is not h5py.HardLink,
            "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
    dataset = owner.get(shown.name)
    _reject(not isinstance(dataset, h5py.Dataset)
            or dataset.file.id != handle.id,
            "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
    return dataset


@contextmanager
def _open_admitted_hdf_dataset(
    master, master_path, member_path, dataset_path, revisions,
):
    import h5py

    same = (
        os.path.normcase(os.path.normpath(os.path.abspath(master_path)))
        == os.path.normcase(os.path.normpath(os.path.abspath(member_path)))
    )
    if same:
        _hdf_handle_revision(master, master_path, revisions)
        dataset = _local_hdf_dataset(master, dataset_path)
        yield dataset
        _hdf_handle_revision(master, master_path, revisions)
        return
    with h5py.File(member_path, "r") as member:
        _hdf_handle_revision(member, member_path, revisions)
        dataset = _local_hdf_dataset(member, dataset_path)
        yield dataset
        _hdf_handle_revision(member, member_path, revisions)


def _admit_hdf_routes(execution, final, fact, revisions, external_links):
    """Freeze each HDF frame segment and its exact storage dependencies."""

    import h5py
    import numpy as np

    source_path = Path(execution["path"])
    authority = (execution["external_members"] if final is None
                 else final["external_members"])
    count = execution["frame_count"] if final is None else final["extent"]
    routes = []
    schema = None
    mapped_dependencies = set()
    with h5py.File(source_path, "r") as handle:
        _hdf_handle_revision(handle, source_path, revisions)
        if authority:
            _reject(len(authority) != len(external_links),
                    "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
            candidates = tuple(zip(external_links, authority))
        else:
            selectors = (() if final is None else tuple(final["dataset_paths"]))
            if not selectors:
                selector = fact["snapshot"].get("dataset_path")
                _reject(type(selector) is not str or not selector,
                        "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
                selectors = (selector,)
            candidates = tuple((selector, None) for selector in selectors)
        cursor = 0
        for selector, member in candidates:
            if member is not None:
                recorded_path = (member["file"]["path"] if final is None
                                 else member["path"])
                recorded_dataset = (member["dataset"] if final is None
                                    else member["dataset_path"])
                _reject(
                    type(selector) is not _HdfOwnerLink
                    or selector.kind != "external"
                    or selector.resolved_target is None
                    or selector.remote_path != recorded_dataset
                    or os.path.normcase(os.path.normpath(os.path.abspath(
                        source_path.parent / selector.lexical_filename)))
                    != os.path.normcase(os.path.normpath(os.path.abspath(
                        recorded_path))),
                    "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
                )
                member_path = selector.resolved_target
                dataset_path = selector.remote_path
                start = (member["first"] if final is None
                         else member["source_start"])
                stop = (member["stop"] if final is None
                         else member["source_stop"])
            else:
                dataset_path = selector
                start = cursor
                member_path = execution["path"]
            with _open_admitted_hdf_dataset(
                handle, source_path, member_path, dataset_path, revisions,
            ) as dataset:
                if member is not None:
                    _reject(start != cursor or dataset.ndim != 3
                            or int(dataset.shape[0]) != stop - start,
                            "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
                    frame_shape = tuple(
                        int(value) for value in dataset.shape[1:])
                else:
                    extent = int(dataset.shape[0]) if dataset.ndim == 3 else 1
                    stop = start + extent
                    _reject(dataset.ndim not in {2, 3},
                            "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
                    frame_shape = tuple(int(value) for value in (
                        dataset.shape[1:] if dataset.ndim == 3
                        else dataset.shape))
                dtype = np.dtype(dataset.dtype)
                frame_bytes = _bounded_hdf_byte_count(
                    frame_shape, dtype.itemsize)
                total_bytes = _bounded_hdf_byte_count(
                    tuple(int(value) for value in dataset.shape),
                    dtype.itemsize)
                current_schema = (frame_shape, dtype.str)
                _reject(bool(dataset.is_virtual)
                        or schema is not None and current_schema != schema,
                        "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
                schema = current_schema
                slots = _hdf_external_paths(
                    dataset, total_bytes=total_bytes)
                for slot in slots:
                    admitted = _admitted_lexical_revision(
                        slot.lexical_path, slot.resolved_path, revisions)
                    physical_stop = (
                        slot.offset + slot.logical_stop - slot.logical_start)
                    _reject(
                        physical_stop > int(admitted[3][1]),
                        "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
                    )
                    mapped_dependencies.add(os.path.normcase(os.path.normpath(
                        slot.lexical_path)))
                _reject(slots and (dataset.chunks is not None
                                   or dataset.ndim != 3),
                        "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
                routes.append((start, stop, _HdfDatasetRoute(
                    dataset_path, start, stop, member_path, frame_bytes, slots,
                    tuple(slot.logical_stop for slot in slots))))
                cursor = stop
        _hdf_handle_revision(handle, source_path, revisions)
    dependencies = {
        os.path.normcase(os.path.normpath(value["path"]))
        for value in execution["dependency_files"]
    }
    _reject(cursor != count or dependencies != mapped_dependencies,
            "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
    return tuple(routes)


def _ordered_revision_paths(*paths):
    ordered = []
    seen = set()
    for path in paths:
        if path is None:
            continue
        key = os.path.normcase(os.path.normpath(os.path.abspath(path)))
        if key not in seen:
            seen.add(key)
            ordered.append(str(path))
    return tuple(ordered)


def _admitted_hdf_route(routes, stops, ordinal):
    position = bisect_right(stops, ordinal)
    _reject(
        position >= len(routes)
        or not routes[position][0] <= ordinal < routes[position][1],
        "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
    )
    return routes[position][2]


def _prebound_hdf_frame(route, ordinal):
    """Retain only the external-storage slices needed by one frame."""

    _reject(
        not route.start <= ordinal < route.stop or route.frame_bytes <= 0,
        "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
    )
    if not route.storage_slots:
        return _HdfFrameRoute(
            route.dataset_path, route.start, route.stop, route.member_path, ())
    logical_start = (ordinal - route.start) * route.frame_bytes
    logical_stop = logical_start + route.frame_bytes
    position = bisect_right(route.storage_stops, logical_start)
    slices = []
    frame_cursor = 0
    while position < len(route.storage_slots):
        slot = route.storage_slots[position]
        if slot.logical_start >= logical_stop:
            break
        left = max(logical_start, slot.logical_start)
        right = min(logical_stop, slot.logical_stop)
        if left < right:
            frame_offset = left - logical_start
            _reject(
                frame_offset != frame_cursor,
                "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
            )
            size = right - left
            slices.append(_HdfStorageSlice(
                slot.lexical_path, slot.resolved_path,
                slot.offset + left - slot.logical_start,
                size, frame_offset,
            ))
            frame_cursor += size
        position += 1
    _reject(
        frame_cursor != route.frame_bytes,
        "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
    )
    return _HdfFrameRoute(
        route.dataset_path, route.start, route.stop, route.member_path,
        tuple(slices),
    )


@dataclass(frozen=True, slots=True)
class _SourceTopology:
    source_base: str
    execution: Mapping[str, Any]
    lineage: Mapping[str, Any] | None
    final_source: Mapping[str, Any] | None
    execution_digest: str
    lineage_digest: str | None
    revision_signature: tuple | None
    revisions: Mapping[str, Any]
    revision_lookup: Mapping[str, Any]
    frame_routes: Mapping[int, _FrameRoute]
    external_parent: str | None
    external_paths: tuple[str, ...]
    external_signature: tuple[_HdfOwnerLink, ...]


def _external_ranges(values, *, final):
    rows = tuple(
        ((value["source_start"], value["source_stop"], value)
         if final else (value["first"], value["stop"], value))
        for value in values
    )
    return tuple(stop for _start, stop, _value in rows), rows


def _external_member(stops, ranges, ordinal):
    if not ranges:
        return None
    position = bisect_right(stops, ordinal)
    _reject(
        position >= len(ranges)
        or not ranges[position][0] <= ordinal < ranges[position][1],
        "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
    )
    start, stop, value = ranges[position]
    return position, start, stop, value


def _admit_source_topology(fact, *, full_inventory, token=None,
                           selected_labels=None):
    from xrd_tools.io.append import _lineage_labels
    from xrd_tools.io.record_writer import (
        WriterStateError, _validate_replacement_execution,
    )

    _event(token)
    execution = fact["source_execution"]
    try:
        execution = _validate_replacement_execution(execution)
    except WriterStateError as error:
        raise ValueError("REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED") from error
    lineage = fact["append_lineage"]
    selected_labels = ((fact["label"],) if selected_labels is None
                       else tuple(selected_labels))
    _reject(not selected_labels
            or any(type(label) is not int for label in selected_labels)
            or selected_labels != tuple(sorted(set(selected_labels))),
            "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
    epoch_sources = {}
    if lineage is not None:
        try:
            _lineage_labels(lineage)
            epochs = lineage["epochs"]
            _reject(type(epochs) is not list or not epochs,
                    "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
            for epoch in epochs:
                labels = epoch["labels"]
                _reject(
                    type(epoch) is not dict or type(epoch["source"]) is not dict
                    or type(labels) is not list
                    or any(type(label) is not int for label in labels)
                    or labels != sorted(set(labels)),
                    "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED",
                )
                for label in labels:
                    _reject(label in epoch_sources,
                            "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
                    epoch_sources[label] = epoch["source"]
        except (TypeError, ValueError, KeyError) as error:
            if (type(error) is ValueError
                    and str(error).startswith("REPLACEMENT_")):
                raise
            raise ValueError("REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED") from error
    final = None if lineage is None else lineage["epochs"][-1]["source"]
    _reject(lineage is not None
            and any(label not in epoch_sources for label in selected_labels),
            "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
    adapter = execution["adapter_id"]
    _reject(
        final is not None and (
            final["adapter_id"] != adapter
            or os.path.normcase(os.path.normpath(final["path"]))
            != os.path.normcase(os.path.normpath(execution["path"]))
            or execution["frame_count"] > final["extent"]),
        "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED",
    )
    execution_stops, execution_ranges = _external_ranges(
        execution["external_members"], final=False)
    final_stops, final_ranges = _external_ranges(
        () if final is None else final["external_members"], final=True)
    execution_rows = tuple(
        (start, stop, value["epoch"])
        for start, stop, value in execution_ranges
    )
    final_rows = tuple(
        (start, stop, value["ordinal"])
        for start, stop, value in final_ranges
    )
    if execution_rows:
        _validate_external_intervals(
            execution_rows, execution["frame_count"],
        )
        _reject(
            adapter != "nexus_hdf5"
            or len(execution_rows) != len(set(execution_rows)),
            "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
        )
    if final_rows:
        _validate_external_intervals(final_rows, final["extent"])
        _reject(
            adapter != "nexus_hdf5"
            or len(final_rows) != len(set(final_rows)),
            "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
        )
    external = (execution["external_members"] if final is None
                else final["external_members"])
    # Admission freezes the whole immutable source graph once.  Frame reads
    # below use only their pre-bound route and exact revision closure.
    _execution, revisions, _targets = _execution_revisions(
        execution, final, fact["label"], validated=True, full=True,
    )
    revision_lookup = _revision_lookup(revisions)
    parent = None
    paths = ()
    links = ()
    owner_signature = ()
    if external:
        snapshot_path = fact["snapshot"].get("dataset_path")
        _reject(
            type(snapshot_path) is not str or not snapshot_path.startswith("/"),
            "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
        )
        parent = str(PurePosixPath(snapshot_path).parent)
        source_path = Path(execution["path"])
        owner_signature, links = _hdf_owner_census(
            source_path, parent, revision_lookup)
        paths = tuple(link.local_path for link in links)
        expected = tuple(
            (os.path.normcase(os.path.normpath(
                (value["file"] if final is None else value)["path"])),
             value["dataset"] if final is None else value["dataset_path"])
            for value in external
        )
        actual = tuple(
            (os.path.normcase(os.path.normpath(os.path.abspath(
                source_path.parent / link.lexical_filename))), link.remote_path)
            for link in links
        )
        recorded = () if final is None else tuple(final["dataset_paths"])
        _reject(
            actual != expected or recorded not in {(), paths},
            "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
        )
    hdf_routes = ()
    hdf_stops = ()
    if adapter == "nexus_hdf5":
        hdf_ranges = _admit_hdf_routes(
            execution, final, fact, revision_lookup, links,
        )
        hdf_stops = tuple(stop for _start, stop, _route in hdf_ranges)
        hdf_routes = hdf_ranges
    frame_routes = {}
    total = execution["frame_count"] if final is None else final["extent"]
    for label in selected_labels:
        ordinal = label - execution["first_label"]
        _reject(not 0 <= ordinal < total,
                "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
        epoch_source = None if lineage is None else epoch_sources[label]
        if adapter == "tiff_series":
            execution_member = (
                execution["member_stamps"][ordinal]
                if ordinal < execution["frame_count"] else None
            )
            final_member = (
                execution_member if final is None
                else final["image_members"][ordinal]
            )
            _reject(
                final_member is None
                or final is not None and (
                    final_member["source_start"] != ordinal
                    or final_member["source_stop"] != ordinal + 1
                ),
                "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED",
            )
            source_state = final_member
            if epoch_source is not None:
                _reject(
                    ordinal >= epoch_source["extent"]
                    or ordinal >= len(epoch_source["image_members"]),
                    "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED",
                )
                source_state = epoch_source["image_members"][ordinal]
                _reject(
                    source_state["source_start"] != ordinal
                    or source_state["source_stop"] != ordinal + 1
                    or os.path.normcase(os.path.normpath(source_state["path"]))
                    != os.path.normcase(os.path.normpath(final_member["path"])),
                    "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED",
                )
            metadata = (
                execution["metadata_sources"][ordinal]["metadata_file"]
                if ordinal < len(execution["metadata_sources"]) else None
            )
            revision_paths = _ordered_revision_paths(
                None if execution_member is None else execution_member["path"],
                final_member["path"],
                None if metadata is None else metadata["path"],
            )
            frame_routes[label] = _FrameRoute(
                final_member["path"], source_state, 1, None, None,
                revision_paths, None,
            )
        elif adapter == "image_file":
            _reject(total != 1 or ordinal != 0,
                    "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
            source_state = (execution if epoch_source is None
                            else epoch_source)
            current = execution if final is None else final
            _reject(
                os.path.normcase(os.path.normpath(source_state["path"]))
                != os.path.normcase(os.path.normpath(current["path"])),
                "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED",
            )
            frame_routes[label] = _FrameRoute(
                current["path"], source_state, 1, None, None,
                _ordered_revision_paths(
                    execution["path"], current["path"],
                    *(value["path"] for value in execution["dependency_files"]),
                ), None,
            )
        elif adapter == "nexus_hdf5":
            route = _prebound_hdf_frame(
                _admitted_hdf_route(hdf_routes, hdf_stops, ordinal), ordinal)
            source_state = (execution if epoch_source is None
                            else epoch_source)
            selected_execution = (
                None if ordinal >= execution["frame_count"] else
                _external_member(
                    execution_stops, execution_ranges, ordinal))
            selected_final = _external_member(
                final_stops, final_ranges, ordinal)
            execution_member = (
                None if selected_execution is None else selected_execution[3])
            final_member = (
                None if selected_final is None else selected_final[3])
            if external:
                _reject(final is not None and final_member is None,
                        "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
            revision_paths = _ordered_revision_paths(
                execution["path"],
                None if execution_member is None
                else execution_member["file"]["path"],
                None if final_member is None
                else final_member["path"],
                route.member_path,
                *(slot.lexical_path for slot in route.storage_slices),
            )
            frame_routes[label] = _FrameRoute(
                execution["path"], source_state, source_state["extent"]
                if epoch_source is not None else total,
                paths[0] if paths else fact["snapshot"]["dataset_path"],
                not bool(external), revision_paths, route,
            )
        else:
            raise ValueError("REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
    _event(token)
    return _SourceTopology(
        fact["source_base"], execution, lineage, final,
        _digest(execution), None if lineage is None else _digest(lineage),
        _revision_signature(revisions), MappingProxyType(dict(revisions)),
        revision_lookup, MappingProxyType(frame_routes),
        parent, paths, owner_signature,
    )


def _require_fact_topology(fact, topology):
    _reject(
        fact["source_execution"] is not topology.execution
        or fact["append_lineage"] is not topology.lineage
        or fact["source_base"] != topology.source_base,
        "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED",
    )


def _validate_terminal_topology(topology, token=None):
    _event(token)
    _reject(
        _digest(topology.execution) != topology.execution_digest
        or (None if topology.lineage is None else _digest(topology.lineage))
        != topology.lineage_digest,
        "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED",
    )
    _execution, revisions, _targets = _execution_revisions(
        topology.execution, topology.final_source, validated=True, full=True,
    )
    revision_lookup = _revision_lookup(revisions)
    _reject(
        topology.revision_signature is None
        or _revision_signature(revisions) != topology.revision_signature,
        "REPLACEMENT_SOURCE_REVISION_CHANGED",
    )
    if topology.external_parent is not None:
        signature, _links = _hdf_owner_census(
            Path(topology.execution["path"]), topology.external_parent,
            revision_lookup,
        )
        _reject(
            signature != topology.external_signature,
            "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
        )
    _event(token)


def _qualified_fact(fact, topology=None):
    topology = topology or _admit_source_topology(
        fact, full_inventory=False)
    _require_fact_topology(fact, topology)
    snapshot = fact["snapshot"]
    _reject(
        set(snapshot) != {
            "adapter_id", "size", "mtime_ns", "frame_count",
            "dataset_path", "self_contained",
        }
        or any(snapshot[key] is None for key in (
            "adapter_id", "size", "mtime_ns", "frame_count")),
        "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED",
    )
    label = fact["label"]
    route = topology.frame_routes.get(label)
    _reject(route is None, "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
    execution = topology.execution
    adapter = execution["adapter_id"]
    ordinal = label - execution["first_label"]
    index = fact["frame_index"]
    _reject(
        index != (ordinal if adapter == "nexus_hdf5" else 0),
        "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED",
    )
    path = _resolve_source_locator(
        fact["path"], fact["source_base"])
    _reject(
        os.path.normcase(os.path.normpath(os.path.abspath(path)))
        != os.path.normcase(os.path.normpath(os.path.abspath(
            route.source_path)))
        or (adapter == "nexus_hdf5") != (_source_route(path) == "hdf5")
        or adapter in {"tiff_series", "image_file"}
        and _source_route(path) not in {"fabio", "raw"},
        "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED",
    )
    state = route.source_state
    _reject(
        (
            snapshot["adapter_id"], snapshot["size"],
            snapshot["mtime_ns"], snapshot["frame_count"],
            snapshot["dataset_path"],
        ) != (
            adapter, state["size"], state["mtime_ns"],
            route.snapshot_count, route.snapshot_dataset_path,
        )
        or route.self_contained is not None
        and snapshot["self_contained"] is not route.self_contained
        or route.self_contained is None
        and snapshot["self_contained"] not in {None, True},
        "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED",
    )
    revisions = {}
    for revision_path in route.revision_paths:
        raw = os.path.normcase(os.path.normpath(
            os.path.abspath(revision_path)))
        admitted = topology.revision_lookup.get(raw)
        _reject(admitted is None,
                "REPLACEMENT_SOURCE_REVISION_CHANGED")
        observed = _revision(Path(revision_path))
        _reject(observed != admitted[3],
                "REPLACEMENT_SOURCE_REVISION_CHANGED")
        revisions[raw] = admitted
    return path, execution, revisions


def _admitted_revision(path, revisions):
    raw = os.path.normcase(os.path.normpath(os.path.abspath(path)))
    admitted = revisions.get(raw)
    _reject(admitted is None, "REPLACEMENT_SOURCE_REVISION_CHANGED")
    return admitted


def _admitted_lexical_revision(path, resolved, revisions):
    """Return the revision admitted for this exact external-storage slot."""

    raw = os.path.normcase(os.path.normpath(os.path.abspath(path)))
    admitted = revisions.get(raw)
    _reject(
        admitted is None
        or os.path.normcase(os.path.normpath(admitted[0]))
        != os.path.normcase(os.path.normpath(resolved)),
        "REPLACEMENT_SOURCE_REVISION_CHANGED",
    )
    return admitted


def _descriptor_revision(descriptor, admitted):
    try:
        observed = os.fstat(descriptor)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("REPLACEMENT_SOURCE_REVISION_CHANGED") from error
    _reject(
        type(descriptor) is not int
        or (
            int(observed.st_size),
            int(observed.st_mtime_ns),
            int(observed.st_ctime_ns),
            int(observed.st_dev),
            int(observed.st_ino),
        ) != tuple(int(value) for value in admitted[3][1:]),
        "REPLACEMENT_SOURCE_REVISION_CHANGED",
    )
    return admitted[3]


def _hdf_handle_revision(handle, path, revisions):
    admitted = _admitted_revision(path, revisions)
    try:
        descriptor = handle.id.get_vfd_handle()
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise ValueError(
            "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED"
        ) from error
    if type(descriptor) is not int:
        raise ValueError("REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
    return _descriptor_revision(descriptor, admitted)


def _bounded_hdf_byte_count(shape, itemsize):
    _reject(
        type(itemsize) is not int or itemsize <= 0
        or type(shape) not in {tuple, list} or not shape,
        "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
    )
    total = 1
    for value in shape:
        _reject(
            type(value) is not int or value <= 0
            or total > _MAX_HDF_EXTERNAL_ADDRESS // value,
            "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
        )
        total *= value
    _reject(
        total > _MAX_HDF_EXTERNAL_ADDRESS // itemsize,
        "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
    )
    return total * itemsize


def _hdf_external_paths(dataset, *, total_bytes=None):
    """Census one external-storage table once, with finite interval bounds."""

    import h5py

    origin = Path(os.path.abspath(os.fsdecode(dataset.file.filename)))
    total_bytes = (
        _bounded_hdf_byte_count(
            tuple(int(value) for value in dataset.shape),
            int(__import__("numpy").dtype(dataset.dtype).itemsize),
        ) if total_bytes is None else total_bytes
    )
    _reject(
        type(total_bytes) is not int or total_bytes <= 0
        or total_bytes > _MAX_HDF_EXTERNAL_ADDRESS,
        "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
    )
    try:
        properties = dataset.id.get_create_plist()
        count = int(properties.get_external_count())
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError(
            "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED"
        ) from error
    slots = []
    retained_path_bytes = 0
    logical_cursor = 0
    try:
        _reject(
            count < 0 or count > _MAX_HDF_EXTERNAL_SLOTS,
            "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
        )
        for index in range(count):
            try:
                filename, offset, size = properties.get_external(index)
                _reject(type(filename) is not bytes or not filename
                        or type(offset) is not int or type(size) is not int,
                        "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
                shown = filename.decode("utf-8", "strict")
            except (UnicodeError, TypeError, ValueError, RuntimeError) as error:
                if (type(error) is ValueError
                        and str(error).startswith("REPLACEMENT_")):
                    raise
                raise ValueError(
                    "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED"
                ) from error
            _reject(
                len(filename) > _MAX_HDF_EXTERNAL_PATH_BYTES,
                "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
            )
            lexical = str(Path(os.path.abspath(origin.parent / shown)))
            try:
                resolved = str(Path(lexical).resolve(strict=True))
                lexical_bytes = lexical.encode("utf-8")
                resolved_bytes = resolved.encode("utf-8")
            except (OSError, UnicodeError) as error:
                raise ValueError(
                    "REPLACEMENT_SOURCE_REVISION_UNAVAILABLE") from error
            retained_path_bytes += (
                len(filename) + len(lexical_bytes) + len(resolved_bytes))
            offset = int(offset)
            size = int(size)
            unlimited = size == int(h5py.h5f.UNLIMITED)
            remaining = total_bytes - logical_cursor
            _reject(
                len(lexical_bytes) > _MAX_HDF_EXTERNAL_PATH_BYTES
                or len(resolved_bytes) > _MAX_HDF_EXTERNAL_PATH_BYTES
                or retained_path_bytes > _MAX_HDF_EXTERNAL_PATHS_BYTES
                or offset < 0 or offset > _MAX_HDF_EXTERNAL_ADDRESS
                or size <= 0
                or not unlimited and size > _MAX_HDF_EXTERNAL_ADDRESS
                or remaining <= 0,
                "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
            )
            capacity = remaining if unlimited else min(size, remaining)
            _reject(
                capacity <= 0
                or offset > _MAX_HDF_EXTERNAL_ADDRESS - capacity
                or logical_cursor > _MAX_HDF_EXTERNAL_ADDRESS - capacity,
                "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
            )
            logical_stop = logical_cursor + capacity
            slots.append(_HdfStorageSlot(
                lexical, resolved, offset, logical_cursor, logical_stop))
            logical_cursor = logical_stop
    finally:
        try:
            properties.close()
        except (AttributeError, RuntimeError):
            pass
    _reject(
        bool(count) != bool(slots) or slots and logical_cursor != total_bytes,
        "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
    )
    previous = None
    for slot in sorted(
        slots,
        key=lambda value: (
            os.path.normcase(os.path.normpath(value.resolved_path)),
            value.offset,
        ),
    ):
        key = os.path.normcase(os.path.normpath(slot.resolved_path))
        physical_stop = slot.offset + slot.logical_stop - slot.logical_start
        _reject(
            previous is not None and previous[0] == key
            and slot.offset < previous[1],
            "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
        )
        previous = (key, physical_stop)
    return tuple(slots)


def _read_external_hdf_frame(dataset, index, revisions, slices):
    """Read one external-storage frame only from admitted file objects."""

    import numpy as np

    dtype = np.dtype(dataset.dtype)
    frame_shape = tuple(int(value) for value in dataset.shape[1:])
    frame_bytes = _bounded_hdf_byte_count(frame_shape, dtype.itemsize)
    _reject(
        dataset.ndim != 3
        or not 0 <= int(index) < int(dataset.shape[0])
        or dataset.chunks is not None
        or bool(dataset.is_virtual),
        "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
    )
    payload = bytearray()
    for slot in slices:
        _reject(
            slot.frame_offset != len(payload) or slot.size <= 0
            or slot.offset < 0
            or slot.offset > _MAX_HDF_EXTERNAL_ADDRESS - slot.size,
            "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
        )
        admitted = _admitted_lexical_revision(
            slot.lexical_path, slot.resolved_path, revisions)
        try:
            descriptor = os.open(slot.lexical_path, os.O_RDONLY)
        except OSError as error:
            raise ValueError(
                "REPLACEMENT_SOURCE_REVISION_CHANGED"
            ) from error
        try:
            _descriptor_revision(descriptor, admitted)
            with os.fdopen(os.dup(descriptor), "rb") as source:
                source.seek(slot.offset)
                chunk = source.read(slot.size)
            _descriptor_revision(descriptor, admitted)
        finally:
            os.close(descriptor)
        _reject(
            len(chunk) != slot.size,
            "REPLACEMENT_SOURCE_REVISION_CHANGED",
        )
        payload.extend(chunk)
    _reject(
        len(payload) != frame_bytes,
        "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
    )
    return np.frombuffer(payload, dtype=dtype).reshape(frame_shape).copy()


def _expected_source_schema(shape, dtype, expected_shape, expected_dtype):
    if expected_shape is None and expected_dtype is None:
        return
    _reject(
        expected_shape is None
        or expected_dtype is None
        or tuple(shape) != tuple(expected_shape)
        or dtype != expected_dtype,
        "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED",
    )
@contextmanager
def _immutable_nonhdf_source(path, revisions, token):
    raw = os.path.normcase(os.path.normpath(os.path.abspath(path)))
    admitted = revisions.get(raw)
    _reject(admitted is None, "REPLACEMENT_SOURCE_REVISION_CHANGED")
    observed = admitted[3]
    expected = tuple(int(value) for value in observed[1:])
    _event(token)
    try:
        source = Path(path).open("rb")
    except OSError as error:
        raise ValueError("REPLACEMENT_SOURCE_REVISION_CHANGED") from error
    try:
        before = os.fstat(source.fileno())
        opened = (
            int(before.st_size), int(before.st_mtime_ns),
            int(before.st_ctime_ns), int(before.st_dev), int(before.st_ino),
        )
        _reject(opened != expected, "REPLACEMENT_SOURCE_REVISION_CHANGED")
        with tempfile.TemporaryDirectory(prefix="xdart-reintegrate-source-") as root:
            snapshot = Path(root) / Path(path).name
            copied = 0
            with snapshot.open("wb") as target:
                while copied < expected[0]:
                    _event(token)
                    payload = source.read(min(1 << 20, expected[0] - copied))
                    _reject(not payload, "REPLACEMENT_SOURCE_REVISION_CHANGED")
                    target.write(payload)
                    copied += len(payload)
                _reject(bool(source.read(1)), "REPLACEMENT_SOURCE_REVISION_CHANGED")
            after = os.fstat(source.fileno())
            closed = (
                int(after.st_size), int(after.st_mtime_ns),
                int(after.st_ctime_ns), int(after.st_dev), int(after.st_ino),
            )
            _reject(closed != expected, "REPLACEMENT_SOURCE_REVISION_CHANGED")
            _event(token)
            yield snapshot
            _event(token)
    finally:
        source.close()
def _decode_nonhdf_source(
    path, route, snapshot, index, raw_options, read, token,
    expected_shape, expected_dtype, revisions,
):
    import numpy as np
    _reject(
        index != 0 or snapshot["dataset_path"] is not None
        or snapshot["frame_count"] != 1
        or snapshot["self_contained"] not in {None, True},
        "REPLACEMENT_RAW_DECODER_UNRECORDED"
        if route == "raw" else "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED",
    )
    def decode(decode_path):
        image = None
        if route == "raw":
            from xrd_tools.io.image import infer_raw_detector_shape, read_image
            options = {
                "raw_dtype": "int32", "raw_header_skip": 0,
                "detector_shape": None,
            } if raw_options is None else dict(raw_options)
            raw_dtype = options.get("raw_dtype")
            names = {
                f"{kind}{bits}" for kind in ("int", "uint", "float")
                for bits in ((8, 16, 32, 64) if kind != "float" else (16, 32, 64))
            }
            codes = {
                f"{end}{kind}{size}" for end in "<>="
                for kind, sizes in (("i", "1248"), ("u", "1248"), ("f", "248"))
                for size in sizes
            } | {"|i1", "|u1"}
            _reject(
                set(options) != {"raw_dtype", "raw_header_skip", "detector_shape"}
                or type(raw_dtype) is not str or raw_dtype not in names | codes
                or type(options["raw_header_skip"]) is not int
                or options["raw_header_skip"] < 0
                or options["detector_shape"] is not None and (
                    type(options["detector_shape"]) is not tuple
                    or len(options["detector_shape"]) != 2
                    or any(type(value) is not int or value <= 0
                           for value in options["detector_shape"])
                ),
                "REPLACEMENT_RAW_DECODER_UNRECORDED",
            )
            try:
                dtype = np.dtype(raw_dtype).str
                header = options["raw_header_skip"]
                shape = tuple(
                    options["detector_shape"]
                    or infer_raw_detector_shape(
                        decode_path, raw_dtype=dtype,
                        raw_header_skip=header,
                    )
                    or ()
                )
                _reject(
                    len(shape) != 2
                    or decode_path.stat().st_size
                    != header + int(shape[0]) * int(shape[1])
                    * np.dtype(dtype).itemsize,
                    "REPLACEMENT_RAW_DECODER_UNRECORDED",
                )
                _expected_source_schema(
                    shape, dtype, expected_shape, expected_dtype,
                )
                _event(token)
                if read:
                    image = read_image(
                        decode_path, detector_shape=shape,
                        raw_dtype=dtype, raw_header_skip=header,
                        preserve_dtype=True, exact_frame=True,
                    )
            except (TypeError, ValueError, OSError) as error:
                if type(error) is ValueError and str(error).startswith("REPLACEMENT_"):
                    raise
                raise ValueError("REPLACEMENT_RAW_DECODER_UNRECORDED") from error
            return tuple(shape), dtype, image
        if decode_path.suffix.lower() in {".tif", ".tiff"}:
            import tifffile
            with tifffile.TiffFile(decode_path) as handle:
                shapes = {tuple(page.shape) for page in handle.pages}
                dtypes = {np.dtype(page.dtype).str for page in handle.pages}
                _reject(
                    len(handle.pages) != 1 or len(shapes) != 1
                    or len(dtypes) != 1,
                    "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED",
                )
                shape, dtype = shapes.pop(), dtypes.pop()
        else:
            import fabio
            with fabio.openheader(str(decode_path)) as header:
                _reject(
                    int(getattr(header, "nframes", 0)) != 1,
                    "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED",
                )
                shape = tuple(header.shape)
                dtype = np.dtype(
                    getattr(header, "dtype", getattr(header, "_dtype", None))
                ).str
        _expected_source_schema(shape, dtype, expected_shape, expected_dtype)
        _event(token)
        if read:
            from xrd_tools.io.image import read_image
            image = read_image(
                decode_path, frame=0, preserve_dtype=True, exact_frame=True,
            )
        return tuple(shape), dtype, image
    if not read:
        return decode(path)
    with _immutable_nonhdf_source(path, revisions, token) as stable:
        return decode(stable)
def _source_fact_inner(fact, *, raw_options=None, read=False, token=None,
                       expected_shape=None, expected_dtype=None, topology=None,
                       qualification=None):
    import numpy as np
    topology = topology or _admit_source_topology(
        fact, full_inventory=False, token=token)
    _event(token)
    path, execution, before = qualification or _qualified_fact(fact, topology)
    source_route = _source_route(path)
    snapshot = fact["snapshot"]
    index = fact["frame_index"]
    frame_route = topology.frame_routes[fact["label"]]
    image = None
    if source_route == "hdf5":
        import h5py

        route = frame_route.hdf
        _reject(route is None,
                "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
        with h5py.File(path, "r") as handle:
            _hdf_handle_revision(handle, path, before)
            with _open_admitted_hdf_dataset(
                handle, path, route.member_path, route.dataset_path, before,
            ) as dataset:
                local_index = index - route.start
                _reject(
                    bool(dataset.is_virtual)
                    or dataset.ndim not in {2, 3}
                    or dataset.ndim == 2 and (
                        route.stop - route.start != 1 or local_index != 0)
                    or dataset.ndim == 3 and (
                        int(dataset.shape[0]) != route.stop - route.start
                        or not 0 <= local_index < int(dataset.shape[0])),
                    "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
                )
                shape = tuple(int(value) for value in (
                    dataset.shape[1:] if dataset.ndim == 3
                    else dataset.shape))
                dtype = np.dtype(dataset.dtype).str
                _expected_source_schema(
                    shape, dtype, expected_shape, expected_dtype)
                if read:
                    if route.storage_slices:
                        image = _read_external_hdf_frame(
                            dataset, local_index, before,
                            route.storage_slices)
                    else:
                        image = np.asarray(
                            dataset[local_index]
                            if dataset.ndim == 3 else dataset[()])
            _hdf_handle_revision(handle, path, before)
    else:
        shape, dtype, image = _decode_nonhdf_source(
            path, source_route, snapshot, index, raw_options, read, token,
            expected_shape, expected_dtype, before,
        )
    after = _qualified_fact(fact, topology)[2]
    _reject(
        tuple((key, value[3]) for key, value in before.items())
        != tuple((key, value[3]) for key, value in after.items()),
        "REPLACEMENT_SOURCE_REVISION_CHANGED",
    )
    _reject(
        len(shape) != 2 or image is not None and (
            tuple(image.shape) != tuple(shape)
            or np.dtype(image.dtype).str != dtype),
        "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED",
    )
    return tuple(shape), dtype, path, image
def _source_fact(fact, *, raw_options=None, read=False, token=None,
                 expected_shape=None, expected_dtype=None, topology=None,
                 qualification=None):
    try: return _source_fact_inner(fact, raw_options=raw_options, read=read, token=token, expected_shape=expected_shape, expected_dtype=expected_dtype, topology=topology, qualification=qualification)
    except Exception as error: raise (error if isinstance(error, ReintegrateCancelled) or type(error) is ValueError and str(error).startswith("REPLACEMENT_") else ValueError("REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED" if Path(str(fact.get("path", ""))).suffix.lower() in {".h5", ".hdf5", ".nxs", ".cxi"} else "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"))
def _background_fact(fact, shared, shape, path, token):
    from xrd_tools.reduction.background import FrameBackgroundPlan, _frame, _one, _tag, resolve_frame_background; plan = FrameBackgroundPlan.from_mapping(_plain(shared["background"])); pair = fact["background_dependency"]
    if plan.mode == "None": _reject(pair is not None, "replacement inactive Background differs"); return None, None
    _reject(pair is None, "replacement active Background is absent"); keys = tuple(sorted(set(key for key in (plan.metadata_key, plan.normalization_key) if key))); items = tuple((key, _tag(_one(fact["metadata"], key))) for key in keys); current = _frame((fact["label"], str(path.resolve(strict=True)), fact["snapshot"]["dataset_path"] or None, fact["frame_index"], tuple(shape), items)); result = resolve_frame_background(plan, current, cancelled=token)
    if result.disposition == "CANCELLED": _event(token); raise RuntimeError("Background returned CANCELLED without requested cancellation")
    _reject(result.disposition != "RESOLVED" or (result.descriptor_bytes, result.fingerprint) != pair, "replacement Background dependency differs"); return result.background, pair
def _fact_projection(selected, shared):
    from xrd_tools.reduction.background import FrameBackgroundPlan
    background = FrameBackgroundPlan.from_mapping(_plain(shared["background"]))
    gi = shared["gi"]
    reduction = None if selected is None else _core_plan(selected, shared)
    monitors = () if reduction is None else tuple(
        item.monitor_key
        for item in (reduction.integration_1d, reduction.integration_2d)
        if item is not None and item.monitor_key
    )
    keys = {
        gi["resolved_motor"]
        if gi["enabled"] and gi["resolved_motor"] != "Manual" else None,
        background.metadata_key,
        background.normalization_key,
        *monitors,
    }
    return tuple(sorted(key for key in keys if key)), shared["geometry"] is not None
def _load_fact(fact, shape, dtype, shared, token, raw_options=None,
               topology=None):
    topology = topology or _admit_source_topology(
        fact, full_inventory=False, token=token)
    _event(token); qualification = _qualified_fact(fact, topology); path = qualification[0]; _event(token); background, pair = _background_fact(fact, shared, shape, path, token); _event(token); final_shape, final_dtype, _path, image = _source_fact(fact, raw_options=raw_options, read=True, token=token, expected_shape=shape, expected_dtype=dtype, topology=topology, qualification=qualification); _event(token); _reject(final_shape != tuple(shape) or final_dtype != dtype, "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); return path, image, background, pair
def _geometry_fact(fact, shared):
    value = dict(fact["geometry"]); active = shared["geometry"] is not None; _reject(active != bool(value) or active and (set(value) != {"rot1", "rot2", "rot3", "incident_angle"} or any(type(v) is not float or not math.isfinite(v) for v in value.values())), "replacement geometry differs"); return None if not active else __import__("xrd_tools.core.scan", fromlist=["FrameGeometry"]).FrameGeometry(**value)
def _background_resource_terms(mode: str, shape: tuple[int, int]) -> tuple[int, int, int, int]: pixels = int(shape[0]) * int(shape[1]); _reject(mode not in {"None", "Series Average", "Single BG File", "BG Directory"}, "unsupported Background resource mode"); return (0, 0, 0, 0) if mode == "None" else (8*pixels, 25*pixels, 8*pixels, 64 << 20) if mode == "Series Average" else (8*pixels, 8*pixels, 8*pixels, 64 << 20)
def _calibration(shared, incidence=None):
    from xrd_tools.integrate.calibration import detector_calibration_from_projection, detector_calibration_to_integrator; values = shared["accepted_scientific_assets"]["poni_values"]
    if values is None: return None, None, None
    config = shared["accepted_scientific_assets"]["poni_detector_config_json"]
    try: parsed = json.loads(config); calibration = detector_calibration_from_projection(values, detector_config=parsed)
    except (TypeError, ValueError, json.JSONDecodeError) as error: raise ValueError("accepted PONI calibration is malformed") from error
    ai = detector_calibration_to_integrator(calibration); gi = shared["gi"]
    if not gi["enabled"]: return calibration, ai, None
    from xrd_tools.integrate.gid import _ATTR_INC, _ATTR_ORIENT, _ATTR_TILT, _xrd_fiber_integrator_type; fi = __import__("copy").deepcopy(ai).promote(_xrd_fiber_integrator_type()); inc, tilt, orient = math.radians(incidence if incidence is not None else gi["th_val"] if gi["resolved_motor"] == "Manual" else 0.0), math.radians(gi["tilt_angle"]), gi["sample_orientation"]; fi.reset_integrator(inc, tilt, orient); fi.USE_LEGACY_MASK_NORMALIZATION = False; setattr(fi, _ATTR_INC, inc); setattr(fi, _ATTR_TILT, tilt); setattr(fi, _ATTR_ORIENT, orient); return calibration, ai, fi
def _validated_shared_science(run: Mapping[str, Any], requested: Mapping[str, Any] | None = None, *, geometry=None):
    outer = dict(run); signed = outer.pop("scientific_signature", None); _reject(type(signed) is not dict, "persisted scientific signature is missing"); signed = dict(signed); assets = signed.pop("accepted_scientific_assets", None); _reject(signed != outer, "persisted duplicated scientific signature differs"); expected = {"schema_version", "generation", "fingerprint", "source", "processing_mode", "output_mode", "live_mode", "batch_mode", "max_cores", "gi", "threshold", "poni_file", "poni_values", "mask_file", "project_root", "save_path", "bai_1d_args", "bai_2d_args", "run_options"}; optional = ({"background"} if "background" in outer else set()) | ({"poni_v3_override"} if "poni_v3_override" in outer else set()); _reject(set(outer) != expected | optional, "persisted run configuration has a noncanonical keyset")
    if "poni_v3_override" in outer:
        try: override = __import__("xrd_tools.session.run_configuration", fromlist=["PoniV3OverrideIntent"]).PoniV3OverrideIntent.from_mapping(outer["poni_v3_override"])
        except (TypeError, ValueError) as error: raise ValueError("persisted PONI v3 override is malformed") from error
        _reject(type(assets) is not dict or type(assets.get("poni_values")) is not dict or assets["poni_values"].get("parallax") is not override.parallax or assets["poni_values"].get("detector_config", {}).get("sensor") != {"material": override.material, "thickness": override.thickness_m}, "persisted PONI v3 override differs from effective calibration")
    persisted_gi = outer.get("gi")
    from xrd_tools.session.run_configuration import FrozenGIConfiguration
    try: decoded_gi = FrozenGIConfiguration.from_dict(persisted_gi)
    except (TypeError, ValueError, KeyError) as error: raise ValueError("persisted GI is malformed") from error
    accepted_poni = assets.get("poni_values") if type(assets) is dict else None; persisted_poni = outer.get("poni_values"); _reject(persisted_poni is not None and persisted_poni != accepted_poni, "accepted scientific assets are malformed")
    gi_projection = {key: persisted_gi[key] for key in ("enabled", "incidence_motor", "resolved_motor", "th_val", "sample_orientation", "tilt_angle")}
    if decoded_gi.enabled: gi_projection["gi_exit_angle_convention"] = decoded_gi.gi_exit_angle_convention
    projection = {"version": 1, "gi": gi_projection, "threshold": outer.get("threshold"), "poni_values": accepted_poni if persisted_poni is None else persisted_poni, "accepted_scientific_assets": assets, "geometry": geometry, "background": outer.get("background", {"version": 1, "mode": "None"})}; _reject(requested is not None and projection != _plain(requested), "requested shared science differs from acquisition"); return projection
def _validate_science(selected, shared, dimension):
    from xrd_tools.reduction.background import FrameBackgroundPlan; _keys(selected, {"version", "dimension", "bai_args", "gi_mode"}, "selected_plan"); _keys(shared, {"version", "gi", "threshold", "poni_values", "accepted_scientific_assets", "geometry", "background"}, "requested_shared_science")
    _reject(type(dimension) is not str or dimension not in {"1d", "2d"} or type(selected["version"]) is not int or selected["version"] != 1 or type(selected["dimension"]) is not str or selected["dimension"] != dimension or type(selected["bai_args"]) is not dict or {"gi_mode_1d", "gi_mode_2d"} & set(selected["bai_args"]) or type(shared["version"]) is not int or shared["version"] != 1, "preparation version/dimension")
    gi, threshold, assets = shared["gi"], shared["threshold"], shared["accepted_scientific_assets"]; geometry = shared["geometry"]
    if geometry is not None:
        _reject(type(geometry) is not dict or set(geometry) != {"convention", "mapping_json", "motor_sources"} or type(geometry["convention"]) is not str or type(geometry["mapping_json"]) is not str or type(geometry["motor_sources"]) is not dict or any(type(k) is not str or type(v) is not str for k, v in geometry["motor_sources"].items()), "geometry science is malformed")
        try: mapping_json = geometry["mapping_json"]; payload = json.loads(mapping_json); module = __import__("xrd_tools.core.geometry", fromlist=["Diffractometer", "DiffractometerGeometry"]); kind = "Diffractometer" if type(payload) is dict and "preset" in payload and "convention" not in payload else "DiffractometerGeometry" if type(payload) is dict and "convention" in payload and "preset" not in payload else None; _reject(kind is None, "geometry science is malformed"); parsed_geometry = getattr(module, kind).from_json(mapping_json)
        except (TypeError, ValueError, KeyError) as error: raise ValueError("geometry science is malformed") from error
        _reject(parsed_geometry.to_json() != mapping_json or geometry["convention"] != getattr(parsed_geometry, "preset", getattr(parsed_geometry, "convention", None)) or geometry["motor_sources"] != {motor: motor for motor in parsed_geometry.all_referenced_motors()}, "geometry science is malformed")
    gi_keys = {"enabled", "incidence_motor", "resolved_motor", "th_val", "sample_orientation", "tilt_angle"}
    _keys(gi, gi_keys | ({"gi_exit_angle_convention"} if gi.get("enabled") else set()), "gi"); _keys(threshold, {"apply_threshold", "threshold_min", "threshold_max", "mask_saturation"}, "threshold"); _keys(assets, {"poni_values", "poni_detector_config_json", "poni_sha256", "mask_sha256"}, "accepted_scientific_assets")
    finite = lambda value: type(value) is float and math.isfinite(value)
    if gi["enabled"]:
        from xrd_tools.corrections.grazing import validate_gi_exit_angle_convention
        try: validate_gi_exit_angle_convention(gi["gi_exit_angle_convention"])
        except ValueError as error: raise ValueError("GI science is malformed") from error
    _reject(type(gi["enabled"]) is not bool or any(type(gi[k]) is not str or not gi[k] for k in ("incidence_motor", "resolved_motor")) or gi["enabled"] and gi["incidence_motor"] != gi["resolved_motor"] or not finite(gi["th_val"]) or type(gi["sample_orientation"]) is not int or not 1 <= gi["sample_orientation"] <= 8 or not finite(gi["tilt_angle"]), "GI science is malformed"); _reject(any(type(threshold[k]) is not bool for k in ("apply_threshold", "mask_saturation")) or any(v is not None and not finite(v) for v in (threshold["threshold_min"], threshold["threshold_max"])) or threshold["threshold_min"] is not None and threshold["threshold_max"] is not None and threshold["threshold_min"] > threshold["threshold_max"], "threshold science is malformed")
    digest = lambda value: value is None or type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    _reject(shared["poni_values"] != assets["poni_values"] or (assets["poni_values"] is None) != (assets["poni_detector_config_json"] is None) or not all(digest(assets[k]) for k in ("poni_sha256", "mask_sha256")), "accepted scientific assets are malformed")
    if assets["poni_detector_config_json"] is not None:
        try: parsed = json.loads(assets["poni_detector_config_json"]); canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError, json.JSONDecodeError) as error: raise ValueError("accepted detector config is malformed") from error
        _reject(type(parsed) is not dict or type(parsed.get("orientation")) is not int or not 1 <= parsed["orientation"] <= 4 or canonical != assets["poni_detector_config_json"], "accepted detector config is noncanonical")
        try: __import__("xrd_tools.integrate.calibration", fromlist=["detector_calibration_from_projection"]).detector_calibration_from_projection(assets["poni_values"], detector_config=parsed)
        except (TypeError, ValueError) as error: raise ValueError("accepted PONI values are malformed") from error
        _calibration(shared, gi["th_val"] if gi["enabled"] else None)
    FrameBackgroundPlan.from_mapping(shared["background"]); plan = _core_plan(selected, shared); from xrd_tools.reduction.provenance_config import _integration_1d_args, _integration_2d_args; normalized = (_integration_1d_args(plan.integration_1d, plan.gi) if dimension == "1d" else _integration_2d_args(plan.integration_2d, plan.gi)); normalized.pop(f"gi_mode_{dimension}", None)
    domains = {"1d": {"q_total", "q_ip", "q_oop", "exit_angle", "chi_gi"}, "2d": {"qip_qoop", "q_chi", "exit_angles"}}; _reject(gi["enabled"] != (selected["gi_mode"] is not None) or selected["gi_mode"] is not None and selected["gi_mode"] not in domains[dimension] or normalized != selected["bai_args"], "selected GI mode/BAI differs"); return plan
def _dimension_audit(*, dimension: str, operation_identity: str, science_identity: str, acquisition_fingerprint: str, requested_shared_science: Mapping[str, Any], selected_plan: Mapping[str, Any], append_lineage: bytes | None): return {"schema_version": 1, "operation": "existing_dimension_replacement", "dimension": dimension, "operation_identity": operation_identity, "science_identity": science_identity, "acquisition_fingerprint": acquisition_fingerprint, "shared_science_fingerprint": science_fingerprint(_plain(requested_shared_science)), "selected_plan": _plain(selected_plan), "selected_gi_mode": selected_plan.get("gi_mode"), "append_lineage_action": ("already_absent" if append_lineage is None else "preserved_append_disabled"), "append_lineage_sha256": (None if append_lineage is None else hashlib.sha256(append_lineage).hexdigest())}
def _audit_identity(audit: Mapping[str, Any]) -> str: return hashlib.sha256(_canonical(audit)).hexdigest()
def _scrub_frame(frame): frame.image = frame.background = frame.geometry = frame.source_identity = frame.source_path = frame.loader = frame.mask = None; frame.source_frame_index = frame.normalization_factor = None; frame.background_dependency_bytes = frame.background_dependency_fingerprint = None; frame.metadata.clear()


def _target_stat_revision(value):
    return (
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        int(value.st_dev),
        int(value.st_ino),
    )


def _target_object_revision(target: Path, snapshot: TargetSnapshot):
    """Bind a full-file snapshot to the exact object opened for HDF reads."""

    try:
        before = target.stat()
        descriptor = os.open(target, os.O_RDONLY)
        try:
            opened = os.fstat(descriptor)
            after = target.stat()
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ValueError("TARGET_SNAPSHOT_CHANGED") from error
    expected = (
        int(snapshot.size),
        int(snapshot.mtime_ns),
        int(snapshot.device),
        int(snapshot.inode),
    )
    revisions = tuple(_target_stat_revision(value) for value in (before, opened, after))
    _reject(
        not snapshot.exists
        or any(
            (revision[0], revision[1], revision[3], revision[4]) != expected
            for revision in revisions
        )
        or len(set(revisions)) != 1,
        "TARGET_SNAPSHOT_CHANGED",
    )
    return revisions[0]


def _target_hdf_fence(handle, target: Path, snapshot: TargetSnapshot, revision):
    try:
        descriptor = handle.id.get_vfd_handle()
        opened = os.fstat(descriptor)
        named = target.stat()
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise ValueError("TARGET_SNAPSHOT_CHANGED") from error
    _reject(
        type(descriptor) is not int
        or _target_stat_revision(opened) != revision
        or _target_stat_revision(named) != revision
        or (
            revision[0],
            revision[1],
            revision[3],
            revision[4],
        ) != (
            int(snapshot.size),
            int(snapshot.mtime_ns),
            int(snapshot.device),
            int(snapshot.inode),
        ),
        "TARGET_SNAPSHOT_CHANGED",
    )


def _open_target_hdf(target: Path):
    import h5py

    return h5py.File(target, "r")


def _canonical_acquisition_selected(run, shared, dimension):
    from xrd_tools.reduction.provenance_config import _integration_1d_args, _integration_2d_args
    mode_key = f"gi_mode_{dimension}"; bai_value = run.get(f"bai_{dimension}_args"); _reject(type(bai_value) is not dict, "selected dimension differs from acquisition provenance"); bai = dict(bai_value); recorded_mode = bai.pop(mode_key, None); run_gi = run.get("gi") or {}; active = shared["gi"]["enabled"]; mode = run_gi.get(f"mode_{dimension}") if active else None; _reject(active and recorded_mode != mode, "selected dimension differs from acquisition provenance")
    candidate = {"version": 1, "dimension": dimension, "bai_args": bai, "gi_mode": mode}
    try:
        plan = _core_plan(candidate, shared); normalized = (_integration_1d_args(plan.integration_1d, plan.gi) if dimension == "1d" else _integration_2d_args(plan.integration_2d, plan.gi)); normalized.pop(mode_key, None); candidate["bai_args"] = normalized; _validate_science(candidate, shared, dimension)
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("selected dimension differs from acquisition provenance") from error
    return candidate


def _selected_inventory_rows(rows, labels, message):
    """Validate one sorted inventory, then select rows in one linear pass."""

    prior = None
    for value in rows:
        _reject(type(value) is not int or value < 0
                or prior is not None and value <= prior, message)
        prior = value
    selected = []
    cursor = 0
    for label in labels:
        while cursor < len(rows) and rows[cursor] < label:
            cursor += 1
        _reject(cursor >= len(rows) or rows[cursor] != label, message)
        selected.append(cursor)
    return tuple(selected)


def _persisted_mask(group, shape, *, read: bool):
    """Validate the persisted mask layout before optionally materializing it."""

    import h5py
    import numpy as np

    from xrd_tools.io.append import (
        _MAX_REPLACEMENT_PATH_UTF8_BYTES,
        _replacement_utf8_attribute,
    )

    mask_node = _replacement_hard_group(
        group, "instrument/detector/mask", h5py.Dataset)
    if mask_node is None:
        return _PersistedMaskSpec(0, 0), None
    pixels = int(shape[0]) * int(shape[1])
    properties = None
    try:
        properties = mask_node.id.get_create_plist()
        layout = properties.get_layout()
        external_count = int(properties.get_external_count())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        layout = None
        external_count = -1
    finally:
        if properties is not None:
            try:
                properties.close()
            except (AttributeError, RuntimeError):
                pass
    try:
        description = (_replacement_utf8_attribute(
            mask_node, "description", "processed detector mask description",
            max_bytes=_MAX_REPLACEMENT_PATH_UTF8_BYTES,
        ) if len(mask_node.attrs) == 1
        and set(mask_node.attrs) == {"description"} else None)
    except ValueError:
        description = None
    _reject(
        mask_node.ndim != 1
        or mask_node.shape[0] < 1
        or mask_node.dtype != np.dtype(np.int64)
        or mask_node.maxshape != mask_node.shape
        or layout != h5py.h5d.CONTIGUOUS
        or mask_node.chunks is not None
        or mask_node.compression is not None
        or mask_node.compression_opts is not None
        or bool(mask_node.shuffle)
        or bool(mask_node.fletcher32)
        or mask_node.scaleoffset is not None
        or external_count != 0
        or bool(mask_node.is_virtual)
        or description != "flat pixel indices, shape (N,)"
        or pixels > _MAX_PERSISTED_MASK_BYTES
        or mask_node.shape[0] > pixels
        or mask_node.nbytes > _MAX_PERSISTED_MASK_BYTES,
        "processed detector mask is malformed",
    )
    spec = _PersistedMaskSpec(pixels, int(mask_node.nbytes))
    if not read:
        return spec, None
    flat = np.asarray(mask_node[()])
    _reject(
        flat.shape != mask_node.shape
        or flat.dtype != np.dtype(np.int64)
        or int(flat[0]) < 0
        or int(flat[-1]) >= pixels
        or bool(np.any(flat[1:] <= flat[:-1])),
        "processed detector mask is malformed",
    )
    mask = np.zeros(shape, dtype=bool)
    mask.ravel()[flat.astype(np.intp, copy=False)] = True
    return spec, mask


def _load_persisted_mask(target, entry, shape, snapshot, target_revision,
                         expected):
    """Materialize only after the mask-bearing resource floor is admitted."""

    with _open_target_hdf(target) as handle:
        _target_hdf_fence(handle, target, snapshot, target_revision)
        group = _replacement_hard_group(handle, entry)
        _reject(group is None, "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
        spec, mask = _persisted_mask(group, shape, read=True)
        _reject(spec != expected, "processed detector mask changed")
        _target_hdf_fence(handle, target, snapshot, target_revision)
    return mask


def _inspect_artifact(target: Path, entry: str, dimension: str,
                      snapshot: TargetSnapshot, target_revision,
                      source_root: str | None = None, *,
                      read_mask: bool = True) -> _ArtifactInspection:
    import h5py, numpy as np; from xrd_tools.io.append import _MAX_REPLACEMENT_PATH_UTF8_BYTES, _replacement_utf8_attribute, _replacement_utf8_scalar; from xrd_tools.io.record_writer import WriterStateError, _decode_replacement_fact, _read_replacement_frame_index, _replacement_json_node, _replacement_scalar
    def text(node, expected, role):
        try: observed = _replacement_utf8_scalar(node, role)
        except ValueError: observed = None
        _reject(observed != expected, f"{role} is noncanonical")
    with _open_target_hdf(target) as handle:
        _target_hdf_fence(handle, target, snapshot, target_revision)
        from xrd_tools.io.processed_scan_id import (
            is_current_processed_xdart_file,
        )
        _reject(
            not is_current_processed_xdart_file(handle, entry),
            "replacement target is not a current xdart .nexus record",
        )
        group = _replacement_hard_group(handle, entry); _reject(group is None, "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); top = _replacement_hard_group(group, f"integrated_{dimension}"); index = _replacement_hard_group(top, "frame_index", h5py.Dataset)
        try: values = _read_replacement_frame_index(index, "selected frame inventory", require_nonempty=True)
        except WriterStateError as error: raise ValueError("selected frame inventory is not exact") from error
        labels = tuple(int(x) for x in values); _reject(not labels or any(v < 0 for v in labels) or labels != tuple(sorted(set(labels))), "selected frame inventory is not exact")
        try: source_base, persisted_lineage, decoded_lineage = decode_replacement_lineage(handle, entry=entry)
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error: raise ValueError("REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED") from error
        config = _replacement_hard_group(group, "reduction/config"); execution_node = _replacement_hard_group(config, "source_execution", h5py.Dataset); _reject(execution_node is None, "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); execution = _replacement_json_node(config, "source_execution", "source execution"); source_base, execution, decoded_lineage, lineage = _relocate_source_context(source_base, source_root, execution, decoded_lineage); run_node = _replacement_hard_group(config, "run_configuration", h5py.Dataset); run = _replacement_json_node(config, "run_configuration", "run configuration"); text(run_node, _canonical(run).decode(), "run configuration")
        geom_node = _replacement_hard_group(config, "geometry"); geom_keys = {"convention", "mapping_json", "motor_sources"}
        _reject(geom_node is not None and len(geom_node) != 3, "selected BAI/GI/geometry is malformed")
        geom_leaves = () if geom_node is None else tuple(_replacement_hard_group(geom_node, name, h5py.Dataset) for name in ("convention", "mapping_json", "motor_sources")); _reject(geom_node is not None and (set(geom_node) != geom_keys or any(node is None for node in geom_leaves)), "selected BAI/GI/geometry is malformed"); geometry = None if geom_node is None else {"convention": _replacement_utf8_scalar(geom_leaves[0], "geometry convention"), "mapping_json": _replacement_utf8_scalar(geom_leaves[1], "geometry mapping"), "motor_sources": _replacement_json_node(geom_node, "motor_sources", "geometry motors")}
        shared = _validated_shared_science(run, geometry=geometry); gi_node = _replacement_hard_group(config, "gi_config", h5py.Dataset); gi_raw = _replacement_json_node(config, "gi_config", "selected GI config", required=False); _reject((gi_raw is None) == shared["gi"]["enabled"] or gi_raw is not None and type(gi_raw) is not dict, "selected GI config is malformed"); gi_cfg = {} if gi_raw is None else gi_raw; source = run.get("source"); _reject(source is not None and type(source) is not dict, "REPLACEMENT_RAW_DECODER_UNRECORDED"); options_missing = source is None or "options" not in source; options = {} if options_missing else source["options"]; _reject(type(options) is not dict, "REPLACEMENT_RAW_DECODER_UNRECORDED"); raw_keys = {"raw_dtype", "raw_header_skip", "detector_shape"}; present_raw_keys = set(options) & raw_keys; _reject(bool(present_raw_keys) and present_raw_keys != raw_keys, "REPLACEMENT_RAW_DECODER_UNRECORDED"); raw = {key: options[key] for key in raw_keys} if present_raw_keys else {}
        if raw: _reject(type(raw["raw_dtype"]) is not str or type(raw["raw_header_skip"]) is not int or raw["raw_header_skip"] < 0 or type(raw["detector_shape"]) is not list or len(raw["detector_shape"]) != 2 or any(type(v) is not int or v <= 0 for v in raw["detector_shape"]), "REPLACEMENT_RAW_DECODER_UNRECORDED")
        raw = None if not raw else MappingProxyType({**raw, "detector_shape": tuple(raw["detector_shape"])})
        mode_key = f"gi_mode_{dimension}"; bai_name = f"bai_{dimension}_args"; bai_node = _replacement_hard_group(config, bai_name, h5py.Dataset); bai_value = _replacement_json_node(config, bai_name, "selected BAI"); physical_node = _replacement_hard_group(config, "gi", h5py.Dataset); physical = _replacement_json_node(config, "gi", "persisted GI truth")
        physical_keysets = {frozenset()}
        if shared["gi"]["enabled"]:
            physical_keys = frozenset({"gi_mode_1d", "gi_mode_2d", "incidence_motor", "th_val", "sample_orientation", "tilt_angle"})
            if shared["gi"]["gi_exit_angle_convention"] == "xdart_reflection_qoop_v1":
                physical_keysets = {physical_keys | {"gi_exit_angle_convention"}}
            else:
                # Original historical records have the exact six-key nested
                # projection.  A bounded legacy reprocess writes the explicit
                # legacy marker so a subsequent reprocess cannot reinterpret
                # it as current science; both retain the historical outer
                # eight-key run identity.
                physical_keysets = {
                    physical_keys,
                    physical_keys | {"gi_exit_angle_convention"},
                }
        shared_physical_keys = ("incidence_motor", "th_val", "sample_orientation", "tilt_angle")
        _reject(type(bai_value) is not dict or type(physical) is not bool or physical != shared["gi"]["enabled"] or frozenset(gi_cfg) not in physical_keysets or physical and any(gi_cfg[key] != shared["gi"][key] for key in shared_physical_keys) or physical and "gi_exit_angle_convention" in gi_cfg and gi_cfg["gi_exit_angle_convention"] != shared["gi"]["gi_exit_angle_convention"], "selected BAI/GI/geometry is malformed"); text(bai_node, _canonical(bai_value).decode(), "selected BAI"); text(physical_node, _canonical(physical).decode(), "persisted GI truth"); (text(geom_leaves[0], geometry["convention"], "geometry convention"), text(geom_leaves[1], geometry["mapping_json"], "geometry mapping"), text(geom_leaves[2], _canonical(geometry["motor_sources"]).decode(), "geometry motors")) if geometry is not None else None
        bai = dict(bai_value); bai_mode = bai.pop(mode_key, None); gi_mode = gi_cfg.get(mode_key); prior_name = f"dimension_replacement_{dimension}"; prior_node = _replacement_hard_group(config, prior_name, h5py.Dataset); prior = _replacement_json_node(config, prior_name, "prior dimension audit", required=False); _reject(bool(gi_cfg) != (gi_node is not None) or (prior is None) != (prior_node is None), "selected config inventory differs"); text(gi_node, _canonical(gi_cfg).decode(), "selected GI config") if gi_node is not None else None; text(prior_node, _canonical(prior).decode(), "prior dimension audit") if prior_node is not None else None
        primary = ("default" if "primary_mode" not in top.attrs else _replacement_utf8_attribute(top, "primary_mode", "selected primary mode", max_bytes=_MAX_REPLACEMENT_PATH_UTF8_BYTES)); _reject(type(primary) is not str or primary != (gi_mode or "default") or ((bai_mode != gi_mode) if prior is None else bai_mode is not None), "selected GI mode/BAI differs"); selected = {"version": 1, "dimension": dimension, "bai_args": bai, "gi_mode": gi_mode}; _validate_science(selected, shared, dimension); acquisition = _canonical_acquisition_selected(run, shared, dimension)
        if prior is None: _reject(selected != acquisition, "selected dimension differs from acquisition provenance")
        else:
            _keys(prior, {"schema_version", "operation", "dimension", "operation_identity", "science_identity", "acquisition_fingerprint", "shared_science_fingerprint", "selected_plan", "selected_gi_mode", "append_lineage_action", "append_lineage_sha256"}, "prior dimension audit"); sha = lambda value: type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value); prior_science = _digest({"api_version": _REINTEGRATE_SCIENCE_API_VERSION, "dimension": dimension, "selected_plan": selected, "requested_shared_science": shared}); lineage_hash = None if persisted_lineage is None else hashlib.sha256(persisted_lineage).hexdigest()
            _reject(prior["schema_version"] != 1 or prior["operation"] != "existing_dimension_replacement" or prior["dimension"] != dimension or prior["selected_plan"] != selected or prior["selected_gi_mode"] != selected["gi_mode"] or not sha(prior["operation_identity"]) or prior["science_identity"] != prior_science or prior["acquisition_fingerprint"] != run.get("fingerprint") or prior["shared_science_fingerprint"] != science_fingerprint(shared) or prior["append_lineage_action"] != ("already_absent" if persisted_lineage is None else "preserved_append_disabled") or prior["append_lineage_sha256"] != lineage_hash, "prior dimension audit differs")
        try: first = _decode_replacement_fact(handle, labels[0], entry=entry, context=(source_base, decoded_lineage, execution))
        except WriterStateError as error: raise ValueError("REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED") from error
        _reject(raw is None and any(_source_route(Path(path)) == "raw" for path in _decoder_input_paths(first)), "REPLACEMENT_RAW_DECODER_UNRECORDED"); topology = _admit_source_topology(first, full_inventory=True, selected_labels=labels); described, dtype, _path, _image = _source_fact(first, raw_options=raw, topology=topology); detector = _replacement_hard_group(group, "instrument/detector/detector_shape", h5py.Dataset)
        _reject(detector is not None and (detector.shape != (2,) or detector.dtype != np.dtype(np.int64)), "processed detector descriptor is malformed")
        raw_shape = None if detector is None else np.asarray(detector[()]); _reject(raw_shape is not None and (raw_shape.shape != (2,) or raw_shape.dtype != np.dtype(np.int64) or any(int(v) <= 0 for v in raw_shape)), "processed detector descriptor is malformed"); shape = described if raw_shape is None else tuple(int(v) for v in raw_shape); _reject(shape != described, "processed detector descriptor is malformed")
        pfg = _replacement_hard_group(group, "per_frame_geometry"); _reject((shared["geometry"] is None) != (pfg is None), "replacement geometry differs")
        if pfg is not None:
            _reject(len(pfg) != 5, "replacement geometry differs")
            required = {"frame_index", "rot1", "rot2", "rot3", "incident_angle"}; nodes = {key: _replacement_hard_group(pfg, key, h5py.Dataset) for key in required}; rows_node = nodes["frame_index"]
            _reject(set(pfg) != required or any(node is None for node in nodes.values()) or any(nodes[key].ndim != 1 or nodes[key].dtype != np.dtype(np.float32) or nodes[key].shape != rows_node.shape for key in required - {"frame_index"}), "replacement geometry differs")
            try: geometry_rows = _read_replacement_frame_index(rows_node, "replacement geometry inventory", require_nonempty=True)
            except WriterStateError as error: raise ValueError("replacement geometry differs") from error
            geometry_values = {key: np.asarray(nodes[key][()]) for key in required - {"frame_index"}}; geometry_labels = tuple(int(v) for v in geometry_rows); selected_rows = _selected_inventory_rows(geometry_labels, labels, "replacement geometry differs"); _reject(any(value.dtype != np.dtype(np.float32) or not np.isfinite(value[list(selected_rows)]).all() for value in geometry_values.values()), "replacement geometry differs")
        values = {}; gi = shared.get("gi") or {}; motor = gi.get("resolved_motor") if gi.get("enabled") else None; scan_data = _replacement_hard_group(group, "scan_data")
        if motor not in {None, "Manual"} and scan_data is not None:
            rows_node = _replacement_hard_group(scan_data, "frame_index", h5py.Dataset); motor_node = _replacement_hard_group(scan_data, motor, h5py.Dataset)
            _reject(rows_node is None or motor_node is None or motor_node.ndim != 1 or motor_node.dtype != np.dtype(np.float32) or motor_node.shape != rows_node.shape, "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
            try: raw_rows = _read_replacement_frame_index(rows_node, "replacement scan inventory", require_nonempty=True)
            except WriterStateError as error: raise ValueError("REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED") from error
            data = np.asarray(motor_node[()]); _reject(data.dtype != np.dtype(np.float32), "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); rows = tuple(int(v) for v in raw_rows); _reject(len(data) != len(rows), "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); selected_rows = _selected_inventory_rows(rows, labels, "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); values = {label: float(data[row]) for label, row in zip(labels, selected_rows) if math.isfinite(float(data[row]))}
        _reject(motor not in {None, "Manual"} and set(values) != set(labels), "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
        if motor not in {None, "Manual"} and first["source_execution"]["adapter_id"] == "tiff_series": execution = first["source_execution"]; admitted = execution["admitted_motor_values"]; ordinals = tuple(label - execution["first_label"] for label in labels); _reject(len(admitted) != len(execution["member_stamps"]) or any(not 0 <= ordinal < len(admitted) for ordinal in ordinals) or any(admitted[ordinal]["motor"] != motor or admitted[ordinal]["value"] != values[label] for label, ordinal in zip(labels, ordinals)), "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
        mask_spec, mask = _persisted_mask(group, shape, read=read_mask)
        _target_hdf_fence(handle, target, snapshot, target_revision)
    fingerprint = run.get("fingerprint"); _reject(type(fingerprint) is not str or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint), "processed acquisition identity is malformed"); return _ArtifactInspection(labels, shape, dtype, shared, selected, fingerprint, source_base, lineage, MappingProxyType(values), mask_spec, mask, raw, topology)
def _core_plan(selected: Mapping[str, Any], shared: Mapping[str, Any], mask=None):
    from xrd_tools.session.readiness import build_native_int_reduction_plan_from_args; args = dict(selected["bai_args"]); dimension = selected["dimension"]; gi = shared["gi"]; threshold = shared["threshold"]
    if gi["enabled"]: args[f"gi_mode_{dimension}"] = selected["gi_mode"]
    plan = build_native_int_reduction_plan_from_args(args if dimension == "1d" else None, args if dimension == "2d" else None, gi_enabled=gi["enabled"], gi_incident_angle=(gi["th_val"] if gi["enabled"] and gi["resolved_motor"] == "Manual" else None), incidence_motor=(None if not gi["enabled"] or gi["resolved_motor"] == "Manual" else gi["resolved_motor"]), tilt_angle=gi["tilt_angle"], sample_orientation=gi["sample_orientation"], gi_exit_angle_convention=gi.get("gi_exit_angle_convention", "xdart_reflection_qoop_v1"), integrate_1d=dimension == "1d", integrate_2d=dimension == "2d", threshold_min=(threshold["threshold_min"] if threshold["apply_threshold"] else None), threshold_max=(threshold["threshold_max"] if threshold["apply_threshold"] else None), mask_saturation=threshold["mask_saturation"]); plan.mask = mask; return plan
def _prepare_gi_scouts(target, entry, observed, selected, shared, snapshot, target_revision, *, cancel_token=None):
    _event(cancel_token); gi = shared["gi"]
    if not gi["enabled"] or gi["resolved_motor"] == "Manual": return selected, None
    try: bootstrap = float(observed.gi_values[observed.labels[0]])
    except (KeyError, TypeError, ValueError) as error: raise ValueError("REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED") from error
    _reject(not math.isfinite(bootstrap), "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
    from xrd_tools.reduction.core import Frame, Scan, _apply_gi_freeze_policy, _gi_1d_freeze_key, _gi_2d_freeze_keys, prepare_gi_freeze; metadata_keys, include_geometry = _fact_projection(selected, shared)
    class Manifest: scan_manifest = lambda self: [(label, {gi["resolved_motor"]: value}) for label, value in observed.gi_values.items()]
    plan, diagnostic = prepare_gi_freeze(Manifest(), _core_plan(selected, shared, observed.mask), incidence_motor=gi["resolved_motor"])
    _reject(diagnostic.status == "unverifiable", "GI metadata extrema are unverifiable"); missing = _gi_1d_freeze_key(plan) is not None or bool(_gi_2d_freeze_keys(plan))
    if not missing: return selected, bootstrap
    scouts = diagnostic.scout_indices if diagnostic.status == "frozen" else (observed.labels[0],); plan = plan if diagnostic.status == "frozen" else __import__("dataclasses").replace(plan, extra={**plan.extra, "gi_freeze_scout_indices": list(scouts)})
    holder = {"frame": None}; frames = []
    for label in scouts:
        frame = Frame(label, metadata={gi["resolved_motor"]: observed.gi_values[label]}); frames.append(frame)
        def load(current, label=label):
            import h5py; from xrd_tools.io.record_writer import _decode_replacement_fact
            prior = holder["frame"]; _scrub_frame(prior) if prior is not None else None; _event(cancel_token)
            with _open_target_hdf(target) as handle:
                _target_hdf_fence(handle, target, snapshot, target_revision)
                fact = _decode_replacement_fact(handle, label, entry=entry, context=(observed.source_base, observed.topology.lineage, observed.topology.execution), metadata_keys=metadata_keys, include_geometry=include_geometry)
                _target_hdf_fence(handle, target, snapshot, target_revision)
            path, image, background, pair = _load_fact(fact, observed.detector_shape, observed.native_dtype, shared, cancel_token, observed.raw_options, observed.topology)
            current.source_path, current.source_frame_index, current.source_identity = path, fact["frame_index"], fact; current.metadata.update(fact["metadata"]); current.background = background; current.geometry = _geometry_fact(fact, shared)
            if pair is not None: current.background_dependency_bytes, current.background_dependency_fingerprint = pair
            holder["frame"] = current; return image
        frame.loader = load
    calibration, integrator, fi = _calibration(shared, bootstrap)
    try: frozen = _apply_gi_freeze_policy(plan, Scan("reintegrate-scout", frames, poni=None if calibration is None else calibration.poni, integrator=integrator), freeze_policy="scout_union", fi=fi, initial_incident_angle=bootstrap)
    except __import__("xrd_tools.io.record_writer", fromlist=["WriterStateError"]).WriterStateError as error: raise ValueError("REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED") from error
    finally: [_scrub_frame(frame) for frame in frames]
    from xrd_tools.reduction.provenance_config import _integration_1d_args, _integration_2d_args
    _reject(_gi_1d_freeze_key(frozen) is not None or _gi_2d_freeze_keys(frozen), "GI range freeze is incomplete"); args = (_integration_1d_args(frozen.integration_1d, frozen.gi) if selected["dimension"] == "1d" else _integration_2d_args(frozen.integration_2d, frozen.gi)); args.pop(f"gi_mode_{selected['dimension']}", None); return {**selected, "bai_args": args}, bootstrap
def _requirements(shape, dtype, selected, shared):
    terms = _background_resource_terms(shared["background"]["mode"], shape)
    descriptor = SimpleNamespace(frame_shape=shape, dtype=_native_dtype(dtype))
    return requirements_from(
        descriptor, _core_plan(selected, shared),
        background_bytes=terms[0], resolver_background_bytes=terms[1],
        worker_background_bytes=terms[2],
        background_binding_bytes=terms[3],
    )
def _allocation_recipe(allocation: SessionResourceAllocation) -> dict[str, Any]: return {"requirements": {f.name: getattr(allocation.requirements, f.name) for f in fields(SessionResourceRequirements)}, "envelope_bytes": allocation.envelope_bytes, "counts": dict(allocation.counts), "categories": dict(allocation.categories), "minimum_bytes": allocation.minimum_bytes, "floor_bytes": allocation.floor_bytes, "assigned_bytes": allocation.assigned_bytes, "origin": allocation.origin, "oversize_excess_bytes": allocation.oversize_excess_bytes}
def _mask_owner_block_bytes(req, mask_spec):
    """Return the exact frame-quantized owner needed at mask decode peak."""

    if not mask_spec.retained_bytes:
        return None
    peak = mask_spec.retained_bytes + mask_spec.decode_bytes
    frame = req.native_frame_bytes
    return max(frame, ((peak + frame - 1) // frame) * frame)


def _policy(req, spec: Mapping[str, Any], mask_spec=None) -> SessionPolicy:
    mask_spec = mask_spec or _PersistedMaskSpec(0, 0)
    owner = _mask_owner_block_bytes(req, mask_spec)
    _keys(spec, ({"version", "kind", "envelope_bytes", "requests"} if spec.get("kind") == "resolve" else {"version", "kind", "allocation"}), "resource_policy"); _reject(type(spec["version"]) is not int or spec["version"] != 1, "resource policy version")
    if spec["kind"] == "resolve":
        requests = spec["requests"]
        _reject(type(requests) is not dict or not set(requests) <= _REQUESTS or any(type(value) is not int or value < 0 for value in requests.values()) or spec["envelope_bytes"] is not None and (type(spec["envelope_bytes"]) is not int or spec["envelope_bytes"] <= 0), "resource requests")
        requests = dict(requests)
        _reject(owner is not None and "owner_block_bytes" in requests,
                "Reintegrate owns the persisted-mask owner_block_bytes request")
        if owner is not None:
            requests["owner_block_bytes"] = owner
        policy = resolve_session_policy(req, envelope_bytes=spec["envelope_bytes"], requests=requests, flush=FlushPolicy())
        _reject(owner is not None and policy.allocation.owner_block_bytes != owner,
                f"REINTEGRATE_MASK_OWNER_BLOCK_GRANT_INCOMPLETE(required={owner}, granted={policy.allocation.owner_block_bytes})")
        return policy
    _reject(spec["kind"] != "explicit", "resource policy kind")
    value = spec["allocation"]; _keys(value, {"requirements", "envelope_bytes", "counts", "categories", "minimum_bytes", "floor_bytes", "assigned_bytes", "origin", "oversize_excess_bytes"}, "allocation")
    _keys(value["requirements"], {f.name for f in fields(SessionResourceRequirements)}, "requirements"); _keys(value["counts"], _REQUESTS, "counts"); _keys(value["categories"], {"source_native", "staging", "records", "publication", "worker"}, "categories")
    _reject(any(type(v) is not int or v < 0 for v in (*value["requirements"].values(), *value["counts"].values(), *value["categories"].values(), value["envelope_bytes"], value["minimum_bytes"], value["floor_bytes"], value["assigned_bytes"], value["oversize_excess_bytes"])) or value["origin"] not in {"automatic", "explicit"}, "allocation leaves are malformed"); req2 = SessionResourceRequirements(**value["requirements"]); _reject(req2.fingerprint != req.fingerprint, "RESOURCE_REQUIREMENTS_IDENTITY"); alloc = SessionResourceAllocation(requirements=req2, **{k: value[k] for k in value if k != "requirements"}); _reject(alloc.origin not in {"automatic", "explicit"}, "allocation origin"); policy = resolve_session_policy(req, allocation=alloc, flush=FlushPolicy()); _reject(owner is not None and policy.allocation.owner_block_bytes != owner, "REINTEGRATE_MASK_OWNER_BLOCK_GRANT_CHANGED"); return policy
def _snapshot_mapping(value: TargetSnapshot) -> dict[str, Any]: return {f.name: getattr(value, f.name) for f in fields(TargetSnapshot)}
def _validate_persisted_selected(selected, dimension):
    _keys(selected, {"version", "dimension", "bai_args", "gi_mode"}, "selected_plan"); domains = {"1d": {"q_total", "q_ip", "q_oop", "exit_angle", "chi_gi"}, "2d": {"qip_qoop", "q_chi", "exit_angles"}}
    _reject(type(dimension) is not str or dimension not in domains or type(selected["version"]) is not int or selected["version"] != 1 or selected["dimension"] != dimension or type(selected["bai_args"]) is not dict or {"gi_mode_1d", "gi_mode_2d"} & set(selected["bai_args"]) or type(selected["gi_mode"]) is not str or selected["gi_mode"] not in domains[dimension], "preparation version/dimension")
def _resolve_persisted_selected(selected, shared, dimension, mask):
    _validate_persisted_selected(selected, dimension)
    mode = selected["gi_mode"] if shared["gi"]["enabled"] else None; requested = {**selected, "gi_mode": mode}; plan = _core_plan(requested, shared, mask)
    from xrd_tools.reduction.provenance_config import _integration_1d_args, _integration_2d_args
    args = _integration_1d_args(plan.integration_1d, plan.gi) if dimension == "1d" else _integration_2d_args(plan.integration_2d, plan.gi); args.pop(f"gi_mode_{dimension}", None); resolved = {**requested, "bai_args": args}; _validate_science(resolved, shared, dimension); return resolved
def _value(cls, *values): obj = object.__new__(cls); [object.__setattr__(obj, f.name, v) for f, v in zip(fields(cls), values)]; return obj
@dataclass(frozen=True, slots=True, init=False)
class ReintegratePlan:
    api_version: int; target: str; entry: str; source_root: str; expected_target_snapshot: TargetSnapshot; dimension: Literal["1d", "2d"]; labels: tuple[int, ...]; detector_shape: tuple[int, int]; native_dtype: str; selected_plan: Mapping[str, Any]; requested_shared_science: Mapping[str, Any]; gi_bootstrap_incidence: float | None; retained_mask_bytes: int; mask_decode_bytes: int; session_policy: SessionPolicy; rollback_policy: Literal["ROLLBACK_ON_STOP"]; science_identity: str; operation_identity: str
    def __new__(cls, *args, **kwargs): raise TypeError("ReintegratePlan is factory-constructed")
    @property
    def resource_allocation(self) -> SessionResourceAllocation: return self.session_policy.allocation
    @classmethod
    def from_artifact(cls, target: str | os.PathLike[str], *, entry: str, dimension: Literal["1d", "2d"], preparation: Mapping[str, object], source_root: str | os.PathLike[str] | None = None, expected_target_snapshot: TargetSnapshot | None = None, expected_terminal_identity: StreamTerminal | None = None, expected_labels: tuple[int, ...] | None = None, cancel_token: threading.Event | None = None) -> ReintegratePlan:
        _event(cancel_token); preparation = _plain(_freeze(preparation)); _keys(preparation, {"api_version", "selected_plan", "requested_shared_science", "resource_policy"}, "preparation"); selected, shared = preparation["selected_plan"], preparation["requested_shared_science"]; persisted = type(shared) is dict and set(shared) == {"version", "kind"} and type(shared["version"]) is int and shared["version"] == 1 and shared["kind"] == "persisted_target"; _reject(type(preparation["api_version"]) is not int or preparation["api_version"] != 1, "preparation version/dimension")
        if persisted: _validate_persisted_selected(selected, dimension); _reject(type(expected_target_snapshot) is not TargetSnapshot or not expected_target_snapshot.exists or type(expected_labels) is not tuple or not expected_labels or expected_labels != tuple(sorted(set(expected_labels))) or any(type(v) is not int or v < 0 for v in expected_labels), "expected target/labels are malformed")
        else: _validate_science(selected, shared, dimension); _reject(expected_target_snapshot is not None and type(expected_target_snapshot) is not TargetSnapshot or expected_labels is not None and (type(expected_labels) is not tuple or any(type(v) is not int or v < 0 for v in expected_labels)), "expected target/labels are malformed")
        _reject(expected_terminal_identity is not None and type(expected_terminal_identity) is not StreamTerminal, "expected terminal identity is malformed")
        path = Path(target).resolve()
        sealed = None
        if stream_terminal_object_revision(expected_terminal_identity) is not None:
            sealed = revalidate_stream_terminal(path, expected_terminal_identity)
            _reject(expected_target_snapshot != sealed, "TARGET_SNAPSHOT_CHANGED")
            _event(cancel_token)
        snapshot = capture_target_snapshot(path)
        if sealed is not None:
            _reject(revalidate_stream_terminal(path, expected_terminal_identity) != sealed, "TARGET_SNAPSHOT_CHANGED")
        if source_root is not None:
            source_root = os.fspath(source_root)
            _normalized_absolute_path(
                source_root, "REPLACEMENT_SOURCE_RELOCATION_UNSUPPORTED")
        _reject(not snapshot.exists, "replacement target must exist"); _reject(sealed is None and expected_target_snapshot is not None and snapshot != expected_target_snapshot, "TARGET_SNAPSHOT_CHANGED"); target_revision = _target_object_revision(path, snapshot); _event(cancel_token); observed = _inspect_artifact(path, entry, dimension, snapshot, target_revision, source_root, read_mask=False); _event(cancel_token); labels = tuple(observed.labels)
        _reject(expected_labels is not None and labels != expected_labels, "EXPECTED_LABELS_CHANGED")
        if persisted: shared = _plain(observed.persisted_shared_science); selected = _resolve_persisted_selected(selected, shared, dimension, None)
        else: _reject(_plain(observed.persisted_shared_science) != shared, "shared science differs")
        req = _requirements(observed.detector_shape, observed.native_dtype, selected, shared); policy = _policy(req, preparation["resource_policy"], observed.mask_spec)
        mask = _load_persisted_mask(path, entry, observed.detector_shape, snapshot, target_revision, observed.mask_spec) if observed.mask_spec.retained_bytes else None
        observed = observed._replace(mask=mask)
        selected, bootstrap = _prepare_gi_scouts(path, entry, observed, selected, shared, snapshot, target_revision, cancel_token=cancel_token); _event(cancel_token); _reject(capture_target_snapshot(path) != snapshot or _target_object_revision(path, snapshot) != target_revision, "TARGET_SNAPSHOT_CHANGED"); _event(cancel_token); return _make_plan(str(path), entry, observed.source_base, dimension, labels, observed.detector_shape, observed.native_dtype, selected, shared, bootstrap, observed.mask_spec.retained_bytes, observed.mask_spec.decode_bytes, policy, snapshot=snapshot)
    def as_recipe(self) -> dict[str, object]: return {"schema": "xrd_tools.reintegrate.plan", "version": _REINTEGRATE_RECIPE_VERSION, "plan": _plan_mapping(self)}
    @classmethod
    def from_recipe(cls, recipe: Mapping[str, object]) -> ReintegratePlan:
        recipe = _plain(_freeze(recipe)); _keys(recipe, {"schema", "version", "plan"}, "recipe")
        _reject(recipe["schema"] != "xrd_tools.reintegrate.plan" or type(recipe["version"]) is not int or recipe["version"] != _REINTEGRATE_RECIPE_VERSION, "recipe schema/version")
        value = recipe["plan"]; _keys(value, {"api_version", "target", "entry", "source_root", "expected_target_snapshot", "dimension", "labels", "detector_shape", "native_dtype", "selected_plan", "requested_shared_science", "gi_bootstrap_incidence", "retained_mask_bytes", "mask_decode_bytes", "session_policy", "rollback_policy", "science_identity", "operation_identity"}, "recipe plan"); _reject(type(value["api_version"]) is not int or value["api_version"] != _REINTEGRATE_PLAN_API_VERSION or value["rollback_policy"] != "ROLLBACK_ON_STOP" or any(type(value[key]) is not str or len(value[key]) != 64 or any(char not in "0123456789abcdef" for char in value[key]) for key in ("science_identity", "operation_identity")), "recipe plan version/rollback")
        snap = value["expected_target_snapshot"]; _keys(snap, {"exists", "size", "mtime_ns", "device", "inode", "digest"}, "target snapshot")
        _reject(snap["exists"] is not True or any(type(snap[k]) is not int or snap[k] < 0 for k in ("size", "mtime_ns", "device", "inode")) or type(snap["digest"]) is not str or len(snap["digest"]) != 64 or any(c not in "0123456789abcdef" for c in snap["digest"]), "recipe target snapshot")
        snapshot = TargetSnapshot(**snap); session = value["session_policy"]; _keys(session, {"flush", "allocation"}, "session_policy"); _keys(session["flush"], {"interval", "cap", "margin"}, "flush")
        _reject(session["flush"] != {"interval": 8, "cap": 64, "margin": 8}, "recipe flush policy")
        _reject(type(value["labels"]) is not list or type(value["detector_shape"]) is not list, "recipe tuple fields are not JSON arrays"); selected, shared = _plain(_freeze(value["selected_plan"])), _plain(_freeze(value["requested_shared_science"])); _validate_science(selected, shared, value["dimension"])
        mask_spec = _mask_spec(value["retained_mask_bytes"], value["mask_decode_bytes"], value["detector_shape"], "recipe persisted-mask resources")
        req = _requirements(tuple(value["detector_shape"]), value["native_dtype"], selected, shared); policy = _policy(req, {"version": 1, "kind": "explicit", "allocation": session["allocation"]}, mask_spec)
        return _make_plan(value["target"], value["entry"], value["source_root"], value["dimension"], tuple(value["labels"]), tuple(value["detector_shape"]), value["native_dtype"], selected, shared, value["gi_bootstrap_incidence"], value["retained_mask_bytes"], value["mask_decode_bytes"], policy, snapshot=snapshot, expected_science=value["science_identity"], expected_operation=value["operation_identity"])
def _plan_mapping(plan: ReintegratePlan) -> dict[str, Any]: return {"api_version": _REINTEGRATE_PLAN_API_VERSION, "target": plan.target, "entry": plan.entry, "source_root": plan.source_root, "expected_target_snapshot": _snapshot_mapping(plan.expected_target_snapshot), "dimension": plan.dimension, "labels": list(plan.labels), "detector_shape": list(plan.detector_shape), "native_dtype": plan.native_dtype, "selected_plan": _plain(plan.selected_plan), "requested_shared_science": _plain(plan.requested_shared_science), "gi_bootstrap_incidence": plan.gi_bootstrap_incidence, "retained_mask_bytes": plan.retained_mask_bytes, "mask_decode_bytes": plan.mask_decode_bytes, "session_policy": {"flush": {"interval": 8, "cap": 64, "margin": 8}, "allocation": _allocation_recipe(plan.resource_allocation)}, "rollback_policy": plan.rollback_policy, "science_identity": plan.science_identity, "operation_identity": plan.operation_identity}
def _make_plan(path, entry, source_root, dimension, labels, shape, dtype, selected, shared, bootstrap, retained_mask_bytes, mask_decode_bytes, policy, *, snapshot=None, expected_science=None, expected_operation=None):
    replay = snapshot is not None; target = str(path) if replay else str(Path(path).resolve()); _reject(replay and (type(path) is not str or not os.path.isabs(path) or os.path.abspath(os.path.normpath(path)) != path), "recipe target/dtype is noncanonical"); source_root = _normalized_absolute_path(source_root, "recipe source root is noncanonical"); snapshot = snapshot or capture_target_snapshot(target); native = _native_dtype(dtype); gi = shared["gi"]; _reject(type(entry) is not str or not entry or dimension not in {"1d", "2d"} or type(labels) is not tuple or not labels or labels != tuple(sorted(set(labels))) or any(type(v) is not int or v < 0 for v in labels) or type(shape) is not tuple or len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape) or native.kind not in "iuf" or native.str != dtype or (bootstrap is not None if not gi["enabled"] or gi["resolved_motor"] == "Manual" else type(bootstrap) is not float or not math.isfinite(bootstrap)), "plan facts are malformed")
    selected, shared = _freeze(selected), _freeze(shared); science = _digest({"api_version": _REINTEGRATE_SCIENCE_API_VERSION, "dimension": dimension, "selected_plan": selected, "requested_shared_science": shared})
    mask_spec = _mask_spec(retained_mask_bytes, mask_decode_bytes, shape,
                           "plan persisted-mask resources")
    expected_owner = _mask_owner_block_bytes(policy.allocation.requirements,
                                             mask_spec)
    _reject(expected_owner is not None
            and policy.allocation.owner_block_bytes != expected_owner,
            "REINTEGRATE_MASK_OWNER_BLOCK_GRANT_CHANGED")
    payload = {"target": target, "entry": entry, "source_root": source_root, "snapshot": _snapshot_mapping(snapshot), "labels": labels, "shape": shape, "dtype": dtype, "bootstrap": bootstrap, "retained_mask_bytes": retained_mask_bytes, "mask_decode_bytes": mask_decode_bytes, "rollback": "ROLLBACK_ON_STOP", "flush": {"interval": 8, "cap": 64, "margin": 8}, "allocation": _allocation_recipe(policy.allocation), "science": science}
    operation = _digest(payload); _reject(expected_science is not None and science != expected_science, "SCIENCE_IDENTITY"); _reject(expected_operation is not None and operation != expected_operation, "OPERATION_IDENTITY"); obj = object.__new__(ReintegratePlan)
    values = (_REINTEGRATE_PLAN_API_VERSION, target, entry, source_root, snapshot, dimension, labels, shape, str(dtype), selected, shared, bootstrap, retained_mask_bytes, mask_decode_bytes, policy, "ROLLBACK_ON_STOP", science, operation); [object.__setattr__(obj, field.name, value) for field, value in zip(fields(ReintegratePlan), values)]; return obj
@dataclass(frozen=True, slots=True, init=False)
class ReintegrateProgress:
    operation_identity: str; stage: str; completed: int; total: int; revision: int
    def __new__(cls, *args, **kwargs): raise TypeError("ReintegrateProgress is factory-constructed")
def _progress(identity, stage, completed, total, revision): _reject(stage not in _STAGES or any(type(v) is not int or v < 0 for v in (completed, total, revision)) or completed > total, "invalid reintegration progress"); return _value(ReintegrateProgress, identity, stage, completed, total, revision)
@dataclass(frozen=True, slots=True, init=False)
class ReintegrateResult:
    disposition: str; input_labels: tuple[int, ...]; committed_labels: tuple[int, ...]; publication_dropped_labels: tuple[int, ...]; diagnostics: tuple[str, ...]; science_identity: str; operation_identity: str; audit_identity: str | None; commit_identity: Any | None
    def __new__(cls, *args, **kwargs): raise TypeError("ReintegrateResult is factory-constructed")
class _ReintegrateFrameSource:
    def __init__(self, plan, token=None, raw_options=None, topology=None): self.plan, self.token, self.raw_options, self.bound_allocation, self._jit, self._fact_reader, self._frames, self._topology = plan, token, raw_options, None, {}, None, {}, topology; self._metadata_keys, self._include_geometry = _fact_projection(getattr(plan, "selected_plan", None), plan.requested_shared_science)
    @property
    def frame_indices(self): return list(self.plan.labels)
    @property
    def jit_roots(self): return tuple(v for v in self._jit.values() if v is not None)
    def bind_allocation(self, allocation): _reject(allocation is not self.plan.resource_allocation, "RESOURCE_ALLOCATION_IDENTITY"); self.bound_allocation = allocation
    def bind_fact_reader(self, reader): self._fact_reader = reader
    def container_descriptor(self):
        import numpy as np; terms = _background_resource_terms(self.plan.requested_shared_science["background"]["mode"], self.plan.detector_shape); return SimpleNamespace(frame_shape=self.plan.detector_shape, dtype=np.dtype(self.plan.native_dtype), background_bytes=terms[0], resolver_background_bytes=terms[1], worker_background_bytes=terms[2], background_binding_bytes=terms[3])
    def to_scan(self, **kwargs):
        from xrd_tools.reduction.core import Frame, Scan, _REINTEGRATE_SCAN_MARKER; calibration, integrator, fi = _calibration(self.plan.requested_shared_science, self.plan.gi_bootstrap_incidence); gi = self.plan.requested_shared_science["gi"]; frames = [Frame(label, metadata=({gi["resolved_motor"]: self.plan.gi_bootstrap_incidence} if label == self.plan.labels[0] and gi["enabled"] and gi["resolved_motor"] != "Manual" else {})) for label in self.plan.labels]; self._frames = {frame.index: frame for frame in frames}
        scan = Scan("reintegrate", frames, poni=None if calibration is None else calibration.poni, integrator=fi or integrator, **kwargs); scan.extra["_reintegrate_marker"] = _REINTEGRATE_SCAN_MARKER; return scan
    def prepare(self, frame):
        if self._fact_reader is None or self._jit: raise RuntimeError("replacement JIT fact ownership is invalid")
        try:
            _event(self.token); fact = self._fact_reader(int(frame.index), metadata_keys=self._metadata_keys, include_geometry=self._include_geometry); self._jit.update({"source_identity": fact, "loader": frame.loader})
            _reject(self._topology is None,
                    "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
            _require_fact_topology(fact, self._topology)
            path, image, background, pair = _load_fact(fact, self.plan.detector_shape, self.plan.native_dtype, self.plan.requested_shared_science, self.token, self.raw_options, self._topology)
            revision = int(fact["snapshot"]["mtime_ns"]); gi = self.plan.requested_shared_science["gi"]; incidence = fact["metadata"].get(gi["resolved_motor"]); _reject(gi["enabled"] and gi["resolved_motor"] != "Manual" and (type(incidence) is not float or not math.isfinite(incidence) or frame.index == self.plan.labels[0] and incidence != self.plan.gi_bootstrap_incidence), "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
            frame.source_path, frame.source_frame_index, frame.source_identity = path, fact["frame_index"], fact; frame.metadata.update(fact["metadata"]); frame.background = background
            frame.geometry = _geometry_fact(fact, self.plan.requested_shared_science)
            if pair is not None: frame.background_dependency_bytes, frame.background_dependency_fingerprint = pair
            geometry = None if frame.geometry is None else {key: getattr(frame.geometry, key) for key in ("rot1", "rot2", "rot3", "incident_angle")}; dependency = None if frame.background_dependency_bytes is None and frame.background_dependency_fingerprint is None else (frame.background_dependency_bytes, frame.background_dependency_fingerprint); _reject(frame.index != fact["label"] or frame.source_path != path or frame.source_frame_index != fact["frame_index"] or frame.source_identity is not fact or dict(frame.metadata) != dict(fact["metadata"]) or geometry != (None if not fact["geometry"] else dict(fact["geometry"])) or dependency != fact["background_dependency"], "replacement local JIT stub differs")
            self._jit.update({"source_identity": fact, "image": image, "background": background, "metadata": frame.metadata, "geometry": frame.geometry, "normalization": frame.normalization_factor, "mask": frame.mask, "dependency": pair}); return image, revision
        except BaseException as error: self.clear_label(frame.index); (None if type(error).__name__ != "WriterStateError" or type(error).__module__ != "xrd_tools.io.record_writer" else (_ for _ in ()).throw(ValueError("REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"))); raise
    def validate_terminal_topology(self):
        _reject(self._topology is None, "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
        _validate_terminal_topology(self._topology, self.token)
    def clear_label(self, label): frame = self._frames.get(int(label)); _scrub_frame(frame) if frame is not None else None; self._jit.clear()
    def clear_jit(self): [self.clear_label(label) for label in tuple(self._frames)]; self._jit.clear()
class _ExecutionRuntime:
    def __init__(self, plan, cancel_token, progress_cb): self.plan, self.token, self.progress_cb, self.session, self.source, self.sink, self.result, self.audit, self.revision, self.diagnostics, self.accounting, self.primary, self.dropped = plan, cancel_token, progress_cb, None, None, None, None, None, 0, [], None, None, ()
    def _report(self, stage, completed, total):
        self.revision += 1; value = _progress(self.plan.operation_identity, stage, completed, total, self.revision)
        if self.progress_cb is not None:
            try: self.progress_cb(value)
            except BaseException as error: self._note(f"progress callback: {_diagnostic(error)}")
    def _note(self, value): self.diagnostics.append(_diagnostic(value)) if len(self.diagnostics) < 16 else None
    def _outcome(self, terminal):
        from xrd_tools.reduction.core import NexusTerminalDisposition, NexusTerminalResult
        if type(terminal) is not NexusTerminalResult: raise RuntimeError("replacement settlement returned no typed terminal")
        if terminal.disposition is NexusTerminalDisposition.ABORTED: return _RuntimeOutcome("ROLLED_BACK", (), self.dropped, tuple(self.diagnostics), None, None)
        dropped = self.dropped; committed = tuple(v for v in self.plan.labels if v not in set(dropped)); (None if committed else (_ for _ in ()).throw(RuntimeError("replacement committed an empty survivor set")))
        return _RuntimeOutcome("COMMITTED", committed, dropped, tuple(self.diagnostics), self.audit, terminal.commit_identity)
    def _terminal(self, terminal): outcome = self._outcome(terminal); self.close(); return outcome
    def _has_custody(self):
        transaction = None if self.sink is None else self.sink._transaction
        if transaction is None: return False
        try: return self.sink._transaction_owners is not None or transaction.snapshot().phase.value not in {"committed", "aborted"}
        except BaseException: return True
    def _stop(self, error):
        self.primary = self.primary or error
        try: self.session._session._record_failure(error)
        except BaseException as cleanup: self._note(cleanup)
        try: self.session.stop()
        except BaseException as cleanup: self._note(cleanup)
    def _settle_construction(self):
        try: terminal = self.sink.abort(None)
        except BaseException as error:
            if self._has_custody(): self._note(error); return _RuntimeOutcome("SETTLEMENT_PENDING", (), (), tuple(self.diagnostics), None, None)
            raise self.primary or error
        if terminal is None: raise self.primary or RuntimeError("replacement construction lost terminal custody")
        return self._terminal(terminal)
    def _settle(self):
        if self.session is None: return self._settle_construction()
        self._report("settle", 0, 1)
        try:
            self.result = self.session.finish(raise_on_failure=False); terminal = self.session.terminal_result
        except BaseException as error:
            terminal = self.session.terminal_result
            if terminal is not None and self.session._dynamic_terminal_settled: self.primary = self.primary or error; return self._terminal(terminal)
            writer = None if self.sink is None else self.sink._writer
            if not (self._has_custody() or terminal is not None or self.session._dynamic_frozen_result is not None): raise self.primary or error
            self._note(error); pending_audit = self.audit if writer is not None and writer._finish_step >= 3 else None
            return _RuntimeOutcome("SETTLEMENT_PENDING", (), (), tuple(self.diagnostics), pending_audit, None)
        self._report("settle", 1, 1); self.primary = self.primary or self.session._session._current_failure(); return self._terminal(terminal)
    def run(self):
        if self.token is not None and self.token.is_set(): return _RuntimeOutcome("ROLLED_BACK", (), (), (), None, None)
        _event(self.token); self._report("qualify", 0, len(self.plan.labels)); _event(self.token); path = Path(self.plan.target)
        if capture_target_snapshot(path) != self.plan.expected_target_snapshot: raise ValueError("TARGET_SNAPSHOT_CHANGED")
        target_revision = _target_object_revision(path, self.plan.expected_target_snapshot)
        _event(self.token)
        from xrd_tools.reduction.core import NexusSink; from xrd_tools.session import DynamicAccountingLimits, DynamicFrameIdentity, DynamicRunAccounting, StageLedger, required_result_modes; from xrd_tools.session.scan_session import ScanSession; inspected = _inspect_artifact(path, self.plan.entry, self.plan.dimension, self.plan.expected_target_snapshot, target_revision, self.plan.source_root, read_mask=False); _event(self.token)
        if capture_target_snapshot(path) != self.plan.expected_target_snapshot or _target_object_revision(path, self.plan.expected_target_snapshot) != target_revision: raise ValueError("TARGET_SNAPSHOT_CHANGED")
        _event(self.token)
        expected_mask = _PersistedMaskSpec(
            self.plan.retained_mask_bytes, self.plan.mask_decode_bytes)
        if inspected.labels != self.plan.labels or inspected.detector_shape != self.plan.detector_shape or inspected.native_dtype != self.plan.native_dtype or inspected.mask_spec != expected_mask or _mask_owner_block_bytes(self.plan.resource_allocation.requirements, inspected.mask_spec) != (None if not inspected.mask_spec.retained_bytes else self.plan.resource_allocation.owner_block_bytes) or _plain(inspected.persisted_shared_science) != _plain(self.plan.requested_shared_science) or self.plan.gi_bootstrap_incidence is not None and inspected.gi_values.get(self.plan.labels[0]) != self.plan.gi_bootstrap_incidence: raise ValueError("RECIPE_ARTIFACT_FACTS_CHANGED")
        mask = (_load_persisted_mask(
            path, self.plan.entry, inspected.detector_shape,
            self.plan.expected_target_snapshot, target_revision,
            inspected.mask_spec,
        ) if inspected.mask_spec.retained_bytes else None)
        inspected = inspected._replace(mask=mask)
        _event(self.token)
        audit = _dimension_audit(dimension=self.plan.dimension, operation_identity=self.plan.operation_identity, science_identity=self.plan.science_identity, acquisition_fingerprint=inspected.acquisition_fingerprint, requested_shared_science=self.plan.requested_shared_science, selected_plan=self.plan.selected_plan, append_lineage=inspected.append_lineage); self.audit = _audit_identity(audit); background = self.plan.requested_shared_science["background"]; run = {} if background["mode"] == "None" else {"background": _plain(background)}
        lock = threading.RLock(); self.source = _ReintegrateFrameSource(self.plan, self.token, inspected.raw_options, inspected.topology); self.sink = NexusSink.for_existing_replacement(path, expected_target_snapshot=self.plan.expected_target_snapshot, dimension=self.plan.dimension, labels=self.plan.labels, audit_bytes=_canonical(audit), selected_plan=self.plan.selected_plan["bai_args"], selected_gi_mode=self.plan.selected_plan["gi_mode"], source_execution=dict(inspected.topology.execution), append_lineage=inspected.append_lineage, cancel_token=self.token, entry=self.plan.entry, source_base=inspected.source_base, run_configuration_provenance=run, write_thumbnails=False, flush_every=None, file_lock=lock)
        self.source.bind_fact_reader(lambda label, **kwargs: self.sink._writer._detach_replacement_fact(label, **kwargs)); core_plan = _core_plan(self.plan.selected_plan, self.plan.requested_shared_science, inspected.mask); modes = required_result_modes(core_plan); targets = {mode: (f"nexus:{path}",) for mode in modes}; ledger = StageLedger(required_modes=modes, targets_by_mode=targets); self.accounting = DynamicRunAccounting(ledger, run_generation=1, limits=DynamicAccountingLimits(1, 1, len(self.plan.labels)))
        try: self.session = ScanSession(core_plan, self.source, self.sink, policy=self.plan.session_policy, cancel_token=self.token, clear_frame_images=True, accounting=ledger, dynamic_accounting=self.accounting, targets_by_mode=targets)
        except BaseException as error: self.primary = error; return self._settle()
        total = len(self.plan.labels)
        try:
            self.session.start()
            self.sink._writer._replacement_read_context = (
                inspected.source_base, inspected.topology.lineage,
                inspected.topology.execution,
            )
            for completed, frame in enumerate(self.session.scan.frames):
                if self.token is not None and self.token.is_set(): self.session.stop(); break
                try:
                    self._report("read", completed, total); image, revision = self.source.prepare(frame); key = DynamicFrameIdentity(self.plan.operation_identity, int(frame.index)); self.accounting.discover(key, group=self.plan.operation_identity, ordinal=completed, output_label=int(frame.index)); attempt = self.accounting.begin_attempt(key, source_revision=revision); self.accounting.record_enqueued(attempt)
                    if not self.session.submit(frame, image, attempt_token=attempt): self.accounting.record_cancelled(attempt, reason="reintegration submission cancelled"); break
                    if not self.session._session.drain(): break
                    self._report("reduce", completed + 1, total); self._report("write", completed + 1, total)
                finally: self.source.clear_label(frame.index)
        except ReintegrateCancelled: self.session.stop()
        except BaseException as error: self._stop(error)
        try:
            if self.session._session._current_failure() is None:
                self.session.flush(force=True)
                self.source.validate_terminal_topology()
            snapshot = self.accounting.snapshot(); pairs = snapshot.publication_dropped | snapshot.pending_publication_dropped; self.dropped = tuple(label for label in self.plan.labels if any(key.logical_frame_identity == label for key, _mode in pairs))
        except BaseException as error: self._note(error); self._stop(error)
        return self._settle()
    def finish_current(self): self.session._mark_dynamic_failure(ReintegrateCancelled("reintegration cancelled before commit")) if self.token is not None and self.token.is_set() and self.session is not None and not self.sink._transaction.snapshot().writer_succeeded else None; return self._settle()
    def close(self): source, sink = self.source, self.sink; writer = None if sink is None else sink._writer; source.clear_jit() if source is not None else None; source._frames.clear() if source is not None else None; setattr(source, "_fact_reader", None) if source is not None else None; setattr(source, "_topology", None) if source is not None else None; [setattr(writer, name, None) for name in ("_replacement_configuration", "_replacement_read_context", "_replacement_manifest", "_replacement_expected")] if writer is not None else None; writer._row_cursors.clear() if writer is not None else None; setattr(writer, "_replacement_labels", ()) if writer is not None else None; self.session = self.source = self.sink = self.result = self.accounting = None
def _open_runtime(plan, cancel_token, progress_cb): return _ExecutionRuntime(plan, cancel_token, progress_cb)
class ReintegrateRunner:
    def __init__(self, plan: ReintegratePlan, *, cancel_token: threading.Event | None = None, progress_cb: Callable[[ReintegrateProgress], object] | None = None) -> None:
        if type(plan) is not ReintegratePlan: raise TypeError("runner requires an exact ReintegratePlan")
        _event(cancel_token, honor=False); self.plan, self.cancel_token, self.progress_cb = plan, cancel_token, progress_cb; self._thread, self._runtime, self._state = threading.get_ident(), None, "NEW"
    def __enter__(self) -> ReintegrateRunner: return self
    def __exit__(self, exc_type, exc, traceback) -> None: self.close()
    def _result(self, outcome):
        from xrd_tools.io.output_transaction import StreamTerminal
        if not (outcome.audit is None or type(outcome.audit) is str and len(outcome.audit) == 64 and all(c in "0123456789abcdef" for c in outcome.audit)) or outcome.disposition not in {"COMMITTED", "ROLLED_BACK", "SETTLEMENT_PENDING"} or outcome.disposition == "COMMITTED" and (not outcome.committed or type(outcome.terminal) is not StreamTerminal or outcome.audit is None or outcome.committed != tuple(v for v in self.plan.labels if v not in set(outcome.dropped))) or outcome.disposition != "COMMITTED" and (outcome.committed or outcome.terminal is not None) or outcome.disposition == "ROLLED_BACK" and outcome.audit is not None or set(outcome.committed) & set(outcome.dropped) or tuple(v for v in self.plan.labels if v in set(outcome.committed)) != outcome.committed or tuple(v for v in self.plan.labels if v in set(outcome.dropped)) != outcome.dropped: raise RuntimeError("replacement result contract is invalid")
        return _value(ReintegrateResult, outcome.disposition, self.plan.labels, outcome.committed, outcome.dropped, tuple(_diagnostic(x) for x in outcome.diagnostics[:16]), self.plan.science_identity, self.plan.operation_identity, outcome.audit, outcome.terminal)
    def _terminal_result(self, outcome):
        self._state = "SETTLEMENT_PENDING" if outcome.disposition == "SETTLEMENT_PENDING" else "TERMINAL"; value = self._result(outcome); runtime = self._runtime; self._runtime = runtime if self._state == "SETTLEMENT_PENDING" else None; (None if self._state != "TERMINAL" or runtime.primary is None else (_ for _ in ()).throw(runtime.primary)); return value
    def run(self) -> ReintegrateResult:
        if threading.get_ident() != self._thread or self._state != "NEW": raise RuntimeError("runner is one-shot and thread-affine")
        self._runtime = _open_runtime(self.plan, self.cancel_token, self.progress_cb); self._state = "ACTIVE"
        try: outcome = self._runtime.run()
        except ReintegrateCancelled: outcome = _RuntimeOutcome("ROLLED_BACK", (), (), (), None, None)
        except BaseException: self._state = "TERMINAL"; raise
        return self._terminal_result(outcome)
    def finish_current(self) -> ReintegrateResult:
        if threading.get_ident() != self._thread or self._state != "SETTLEMENT_PENDING": raise RuntimeError("no settlement is pending on this runner thread")
        try: outcome = self._runtime.finish_current()
        except BaseException: self._state = "TERMINAL"; raise
        return self._terminal_result(outcome)
    def close(self) -> None:
        if threading.get_ident() != self._thread: raise RuntimeError("runner is thread-affine")
        if self._state == "SETTLEMENT_PENDING" or self._runtime is not None and self._runtime._has_custody(): raise RuntimeError("reintegration settlement custody remains pending")
        self._runtime is not None and self._runtime.close(); self._state = "CLOSED"
def run_reintegrate(plan: ReintegratePlan, *, cancel_token: threading.Event | None = None, progress_cb: Callable[[ReintegrateProgress], object] | None = None) -> ReintegrateResult:
    with ReintegrateRunner(plan, cancel_token=cancel_token, progress_cb=progress_cb) as runner:
        result = runner.run()
        while result.disposition == "SETTLEMENT_PENDING": result = runner.finish_current()
        return result
