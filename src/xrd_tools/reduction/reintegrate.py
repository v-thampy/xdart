from __future__ import annotations; import hashlib, json, math, os, threading; from collections.abc import Mapping; from dataclasses import dataclass, fields; from pathlib import Path, PurePosixPath; from types import MappingProxyType, SimpleNamespace; from typing import Any, Callable, Literal, NamedTuple; from xrd_tools.io.append import _replacement_hard_group, decode_replacement_lineage, science_fingerprint; from xrd_tools.io.output_transaction import TargetSnapshot, capture_target_snapshot; from xrd_tools.session.policy import FlushPolicy, SessionPolicy, SessionResourceAllocation, SessionResourceRequirements, requirements_from, resolve_session_policy
_REQUESTS = {"workers", "reduction_inflight", "queue_depth", "owner_block_bytes", "staging_items", "record_heavy_items", "publication_heavy_items", "thumbnail_items", "record_items", "publication_items"}; _STAGES = {"qualify", "read", "reduce", "write", "settle"}
class ReintegrateCancelled(RuntimeError): pass
class _ArtifactInspection(NamedTuple): labels: tuple[int, ...]; detector_shape: tuple[int, int]; native_dtype: str; persisted_shared_science: Mapping[str, Any]; persisted_selected_plan: Mapping[str, Any]; acquisition_fingerprint: str; source_base: str; append_lineage: bytes | None; gi_values: Mapping[int, float]; mask: Any | None; raw_options: Mapping[str, Any] | None
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
def _resolve_source_locator(locator: str, source_base: str, root: Path) -> Path:
    _reject(type(locator) is not str or type(source_base) is not str or source_base and (not os.path.isabs(source_base) or os.path.abspath(os.path.normpath(source_base)) != source_base), "REPLACEMENT_SOURCE_RELOCATION_UNSUPPORTED"); shown = PurePosixPath(locator); relative = not shown.is_absolute(); _reject(relative and (not source_base or str(shown) != locator or ".." in shown.parts), "REPLACEMENT_SOURCE_RELOCATION_UNSUPPORTED"); return Path(str(shown)) if not relative else Path(source_base, *shown.parts)
def _source_route(path: Path) -> str: suffix = path.suffix.lower(); route = "fabio" if suffix in {".tif", ".tiff", ".edf", ".cbf", ".img"} or path.name.lower().endswith(".mar3450") else "raw" if suffix == ".raw" else "hdf5" if suffix in {".h5", ".hdf5", ".nxs", ".cxi"} else None; _reject(route is None, "REPLACEMENT_SOURCE_FORMAT_UNSUPPORTED"); return route
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
def _execution_revisions(execution, final=None, label=None):
    from xrd_tools.io.record_writer import WriterStateError, _validate_replacement_execution
    try: execution = _validate_replacement_execution(execution)
    except WriterStateError as error: raise ValueError("REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED") from error
    adapter, ordinal = execution["adapter_id"], label - execution["first_label"] if type(label) is int else -1
    if adapter == "tiff_series": _reject(not 0 <= ordinal < len(execution["member_stamps"]) or final is not None and not 0 <= ordinal < len(final["image_members"]), "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); states = [(execution["member_stamps"][ordinal], "source_member")]; metadata = execution["metadata_sources"][ordinal]["metadata_file"] if execution["metadata_sources"] else None; states += [] if metadata is None else [(metadata, "image_metadata")]; current = [] if final is None else [(final["image_members"][ordinal], "final_member")]
    elif adapter == "nexus_hdf5": selected = tuple(v for v in execution["external_members"] if v["first"] <= ordinal < v["stop"]); final_selected = () if final is None else tuple(v for v in final["external_members"] if v["source_start"] <= ordinal < v["source_stop"]); _reject(len(selected) > 1 or len(final_selected) > 1, "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED"); states = [(execution, "source_file")] + [(v["file"], "external_member") for v in selected] + [(v, "detector_dependency") for v in execution["dependency_files"]]; current = [] if final is None else [(final, "final_source")] + [(v, "final_external") for v in final_selected]
    else: states, current = [(execution, "source_file")], ([] if final is None else [(final, "final_source")])
    if adapter != "nexus_hdf5": states += [(v, "detector_dependency") for v in execution["dependency_files"]]
    authority = {os.path.normcase(os.path.normpath(v["path"])): v for v, _role in current}; raws, targets = {}, {}
    for state, role in states: observed = _revision(Path(state["path"])); raw = os.path.normcase(os.path.normpath(state["path"])); expected = authority.get(raw); _reject(observed[1:3] != (expected["size"], expected["mtime_ns"]) if expected is not None else observed[1:] != tuple(state[k] for k in ("size", "mtime_ns", "ctime_ns", "device", "inode")), "REPLACEMENT_SOURCE_REVISION_CHANGED"); target = os.path.normcase(os.path.normpath(observed[0])); _reject(raw in raws and raws[raw][0] != target or target in targets and targets[target] != observed[1:], "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); raws[raw] = (target, expected or state, role, observed); targets[target] = observed[1:]
    for state, role in current: raw = os.path.normcase(os.path.normpath(state["path"])); observed = _revision(Path(state["path"])); _reject(observed[1:3] != (state["size"], state["mtime_ns"]), "REPLACEMENT_SOURCE_REVISION_CHANGED"); target = os.path.normcase(os.path.normpath(observed[0])); _reject(raw in raws and raws[raw][0] != target or target in targets and targets[target] != observed[1:], "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); raws[raw] = (target, state, role, observed); targets[target] = observed[1:]
    return execution, raws, targets
def _qualified_fact(fact):
    snapshot = fact["snapshot"]; _reject(set(snapshot) != {"adapter_id", "size", "mtime_ns", "frame_count", "dataset_path", "self_contained"} or any(snapshot[k] is None for k in ("adapter_id", "size", "mtime_ns", "frame_count")), "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); path = _resolve_source_locator(fact["path"], fact["source_base"], Path("/")); lineage = fact["append_lineage"]; source = None if lineage is None else lineage["epochs"][-1]["source"]; label = fact["label"]; execution, raws, targets = _execution_revisions(fact["source_execution"], source, label); adapter, index = execution["adapter_id"], fact["frame_index"]; ordinal = label - execution["first_label"]; count = execution["frame_count"] if source is None else source["extent"]
    _reject(source is not None and (source["adapter_id"] != adapter or os.path.normcase(os.path.normpath(source["path"])) != os.path.normcase(os.path.normpath(execution["path"])) or execution["frame_count"] > source["extent"]), "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
    if adapter == "tiff_series": members = execution["member_stamps"] if source is None else source["image_members"]; _reject(not 0 <= ordinal < len(members) or count != len(members) or index != 0, "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); state, count = members[ordinal], 1
    elif adapter == "nexus_hdf5": _reject(index != ordinal or not 0 <= ordinal < count, "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); state = execution if source is None else source
    elif adapter == "image_file": _reject(count != 1 or index != 0 or label != execution["first_label"], "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); state, count = execution if source is None else source, 1
    else: raise ValueError("REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
    route = _source_route(path); _reject((adapter == "nexus_hdf5") != (route == "hdf5") or adapter == "image_file" and route not in {"fabio", "raw"} or adapter == "tiff_series" and route not in {"fabio", "raw"} or os.path.normcase(os.path.normpath(os.path.abspath(path))) != os.path.normcase(os.path.normpath(state["path"])), "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
    final_external = execution["external_members"] if source is None else source["external_members"]; external_rows = tuple((v["first"], v["stop"], v["epoch"]) for v in final_external) if source is None else tuple((v["source_start"], v["source_stop"], v["ordinal"]) for v in final_external)
    if external_rows: _reject(route != "hdf5", "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED"); _validate_external_intervals(external_rows, execution["frame_count"] if source is None else source["extent"]); _reject(len(external_rows) != len(set(external_rows)), "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
    if lineage is not None:
        matching = tuple(epoch for epoch in lineage["epochs"] if label in epoch["labels"]); _reject(len(matching) != 1, "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); epoch_source = matching[0]["source"]; _reject(adapter == "tiff_series" and not 0 <= ordinal < len(epoch_source["image_members"]), "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); epoch_state = epoch_source if adapter != "tiff_series" else epoch_source["image_members"][ordinal]; count = 1 if adapter == "tiff_series" else epoch_source["extent"]; _reject(os.path.normcase(os.path.normpath(epoch_state["path"])) != os.path.normcase(os.path.normpath(state["path"])), "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); state = epoch_state
    _reject((snapshot["adapter_id"], snapshot["size"], snapshot["mtime_ns"], snapshot["frame_count"]) != (adapter, state["size"], state["mtime_ns"], count), "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); return path, execution, raws
def _hdf_dependencies(datasets, shape, dtype):
    import h5py, numpy as np; seen, found = set(), set()
    def visit(dataset):
        origin = str(Path(dataset.file.filename).resolve(strict=True)); key = (origin, dataset.name)
        if key in seen: return
        seen.add(key); _reject(dataset.ndim != 3 or tuple(dataset.shape[1:]) != tuple(shape) or np.dtype(dataset.dtype).str != dtype, "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
        for filename, _offset, _size in dataset.external or (): found.add(str((Path(origin).parent / filename).resolve(strict=True)))
        for source in (dataset.virtual_sources() if dataset.is_virtual else ()):
            target = str((Path(origin).parent / source.file_name).resolve(strict=True)); found.add(target)
            with h5py.File(target, "r") as handle: visit(handle[source.dset_name])
    [visit(dataset) for dataset in datasets]; return found
def _source_fact_inner(fact, *, raw_options=None, read=False, token=None):
    import numpy as np; _event(token); path, execution, before = _qualified_fact(fact); route = _source_route(path); snapshot = fact["snapshot"]; index = fact["frame_index"]; image = None
    if route == "hdf5":
        import h5py; source = None if fact["append_lineage"] is None else fact["append_lineage"]["epochs"][-1]["source"]; count = execution["frame_count"] if source is None else source["extent"]; authority = execution["external_members"] if source is None else source["external_members"]; _reject((snapshot["self_contained"] is False) != bool(authority), "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED")
        if authority:
            with h5py.File(path, "r") as handle:
                parent = str(PurePosixPath(snapshot["dataset_path"]).parent); owner = handle.get(parent); paths = () if not isinstance(owner, h5py.Group) else tuple(f"{parent.rstrip('/')}/{name}" for name in sorted(owner) if type(owner.get(name, getlink=True)).__name__ == "ExternalLink"); links = tuple(handle.get(item, getlink=True) for item in paths); expected = tuple((os.path.normcase(os.path.normpath((v["file"] if source is None else v)["path"])), v["dataset"] if source is None else v["dataset_path"]) for v in authority); actual = tuple((os.path.normcase(os.path.normpath(os.path.abspath(path.parent / link.filename))), link.path) for link in links); ranges = tuple((v["first"], v["stop"]) if source is None else (v["source_start"], v["source_stop"]) for v in authority); selected = tuple(i for i, (start, stop) in enumerate(ranges) if start <= index < stop); recorded_paths = () if source is None else tuple(source["dataset_paths"]); _reject(actual != expected or source is not None and recorded_paths not in {(), paths} or len(selected) != 1 or snapshot["dataset_path"] != paths[0] or source is None and snapshot["frame_count"] != count or snapshot["self_contained"] is not False, "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED"); position = selected[0]; from xrd_tools.io.nexus import NexusImageStack
                with NexusImageStack(handle, [paths[position]]) as stack: dataset = stack._dsets[0]; start, stop = ranges[position]; shape, dtype = tuple(dataset.shape[1:]), np.dtype(dataset.dtype).str; _reject(dataset.ndim != 3 or dataset.shape[0] != stop - start, "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED"); chosen = (authority[position]["file"] if source is None else authority[position])["path"]; direct = {str(path.resolve(strict=True)), before[os.path.normcase(os.path.normpath(chosen))][0]}; dependencies = {before[os.path.normcase(os.path.normpath(v["path"]))][0] for v in execution["dependency_files"]}; _reject(not {os.path.normcase(os.path.normpath(v)) for v in _hdf_dependencies((dataset,), shape, dtype) - direct} <= {os.path.normcase(os.path.normpath(v)) for v in dependencies}, "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED"); image = np.asarray(stack[index - start]) if read else None
        else:
            from xrd_tools.sources.cursor import ContainerCursor; from xrd_tools.core.scan import SourceKind
            with ContainerCursor(path) as cursor:
                desc = cursor.descriptor; _reject(desc.kind not in {SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER} or desc.dataset_path != snapshot["dataset_path"] or source is None and desc.frame_count != snapshot["frame_count"] or desc.frame_count != count or desc.self_contained is not snapshot["self_contained"] or not 0 <= index < desc.frame_count, "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); shape, dtype = tuple(desc.frame_shape or ()), np.dtype(desc.dtype).str; selectors = tuple(desc.segment_paths) or ((desc.dataset_path,) if desc.dataset_path else ()); _reject(source is not None and tuple(source["dataset_paths"]) != selectors, "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED"); segments = tuple(cursor._h5[item] for item in selectors); offsets = np.cumsum((0, *(int(item.shape[0]) if item.ndim == 3 else 1 for item in segments))); selected = tuple(i for i in range(len(segments)) if offsets[i] <= index < offsets[i + 1]); _reject(len(selected) != 1, "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED"); dependencies = {before[os.path.normcase(os.path.normpath(v["path"]))][0] for v in execution["dependency_files"]}; discovered = _hdf_dependencies((segments[selected[0]],), shape, dtype) - {str(path.resolve(strict=True))}; _reject(bool(dependencies) or bool(discovered), "REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED"); image = cursor.read_frame(index) if read else None
    elif route == "raw":
        from xrd_tools.io.image import infer_raw_detector_shape, read_image
        _reject(index != 0 or snapshot["dataset_path"] is not None or snapshot["frame_count"] != 1 or snapshot["self_contained"] not in {None, True}, "REPLACEMENT_RAW_DECODER_UNRECORDED"); options = {"raw_dtype": "int32", "raw_header_skip": 0, "detector_shape": None} if raw_options is None else dict(raw_options); raw_dtype = options.get("raw_dtype"); names = {f"{kind}{bits}" for kind in ("int", "uint", "float") for bits in ((8, 16, 32, 64) if kind != "float" else (16, 32, 64))}; codes = {f"{end}{kind}{size}" for end in "<>=" for kind, sizes in (("i", "1248"), ("u", "1248"), ("f", "248")) for size in sizes} | {"|i1", "|u1"}; _reject(set(options) != {"raw_dtype", "raw_header_skip", "detector_shape"} or type(raw_dtype) is not str or raw_dtype not in names | codes or type(options["raw_header_skip"]) is not int or options["raw_header_skip"] < 0 or options["detector_shape"] is not None and (type(options["detector_shape"]) is not tuple or len(options["detector_shape"]) != 2 or any(type(v) is not int or v <= 0 for v in options["detector_shape"])), "REPLACEMENT_RAW_DECODER_UNRECORDED")
        try: dtype, header = np.dtype(raw_dtype).str, options["raw_header_skip"]; shape = tuple(options["detector_shape"] or infer_raw_detector_shape(path, raw_dtype=dtype, raw_header_skip=header) or ()); _reject(len(shape) != 2, "REPLACEMENT_RAW_DECODER_UNRECORDED"); image = read_image(path, detector_shape=shape, raw_dtype=dtype, raw_header_skip=header, preserve_dtype=True, exact_frame=True) if read else None
        except (TypeError, ValueError, OSError) as error: raise ValueError("REPLACEMENT_RAW_DECODER_UNRECORDED") from error
    else:
        _reject(index != 0 or snapshot["dataset_path"] is not None or snapshot["frame_count"] != 1 or snapshot["self_contained"] not in {None, True}, "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
        if path.suffix.lower() in {".tif", ".tiff"}:
            import tifffile
            with tifffile.TiffFile(path) as handle: shapes, dtypes = {tuple(v.shape) for v in handle.pages}, {np.dtype(v.dtype).str for v in handle.pages}; _reject(len(handle.pages) != 1 or len(shapes) != 1 or len(dtypes) != 1, "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); shape, dtype = shapes.pop(), dtypes.pop()
        else:
            import fabio
            with fabio.openheader(str(path)) as header: _reject(int(getattr(header, "nframes", 0)) != 1, "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); shape, dtype = tuple(header.shape), np.dtype(getattr(header, "dtype", getattr(header, "_dtype", None))).str
        if read: from xrd_tools.io.image import read_image; image = read_image(path, frame=0, preserve_dtype=True, exact_frame=True)
    after = _qualified_fact(fact)[2]; _reject(tuple((k, v[3]) for k, v in before.items()) != tuple((k, v[3]) for k, v in after.items()), "REPLACEMENT_SOURCE_REVISION_CHANGED"); _reject(len(shape) != 2 or image is not None and (tuple(image.shape) != tuple(shape) or np.dtype(image.dtype).str != dtype), "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); return tuple(shape), dtype, path, image
def _source_fact(fact, *, raw_options=None, read=False, token=None):
    try: return _source_fact_inner(fact, raw_options=raw_options, read=read, token=token)
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
def _load_fact(fact, shape, dtype, shared, token, raw_options=None): _event(token); path, _execution, _revisions = _qualified_fact(fact); _event(token); background, pair = _background_fact(fact, shared, shape, path, token); _event(token); final_shape, final_dtype, _path, image = _source_fact(fact, raw_options=raw_options, read=True, token=token); _event(token); _reject(final_shape != tuple(shape) or final_dtype != dtype, "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); return path, image, background, pair
def _geometry_fact(fact, shared):
    value = dict(fact["geometry"]); active = shared["geometry"] is not None; _reject(active != bool(value) or active and (set(value) != {"rot1", "rot2", "rot3", "incident_angle"} or any(type(v) is not float or not math.isfinite(v) for v in value.values())), "replacement geometry differs"); return None if not active else __import__("xrd_tools.core.scan", fromlist=["FrameGeometry"]).FrameGeometry(**value)
def _background_resource_terms(mode: str, shape: tuple[int, int]) -> tuple[int, int, int, int]: pixels = int(shape[0]) * int(shape[1]); _reject(mode not in {"None", "Series Average", "Single BG File", "BG Directory"}, "unsupported Background resource mode"); return (0, 0, 0, 0) if mode == "None" else (8*pixels, 25*pixels, 8*pixels, 64 << 20) if mode == "Series Average" else (8*pixels, 8*pixels, 8*pixels, 64 << 20)
def _calibration(shared, incidence=None):
    from xrd_tools.integrate.calibration import detector_calibration_to_integrator, load_detector_calibration; values = shared["accepted_scientific_assets"]["poni_values"]
    if values is None: return None, None, None
    config = shared["accepted_scientific_assets"]["poni_detector_config_json"]; payload = "\n".join(("poni_version: 2.1", f"Detector: {values['detector']}", f"Detector_config: {config}", f"Distance: {values['dist']}", f"Poni1: {values['poni1']}", f"Poni2: {values['poni2']}", f"Rot1: {values['rot1']}", f"Rot2: {values['rot2']}", f"Rot3: {values['rot3']}", f"Wavelength: {values['wavelength']}")); calibration = load_detector_calibration("accepted-reintegration.poni", data=payload.encode()); ai = detector_calibration_to_integrator(calibration); gi = shared["gi"]
    if not gi["enabled"]: return calibration, ai, None
    from xrd_tools.integrate.gid import _ATTR_INC, _ATTR_ORIENT, _ATTR_TILT, _xrd_fiber_integrator_type; fi = __import__("copy").deepcopy(ai).promote(_xrd_fiber_integrator_type()); inc, tilt, orient = math.radians(incidence if incidence is not None else gi["th_val"] if gi["resolved_motor"] == "Manual" else 0.0), math.radians(gi["tilt_angle"]), gi["sample_orientation"]; fi.reset_integrator(inc, tilt, orient); fi.USE_LEGACY_MASK_NORMALIZATION = False; setattr(fi, _ATTR_INC, inc); setattr(fi, _ATTR_TILT, tilt); setattr(fi, _ATTR_ORIENT, orient); return calibration, ai, fi
def _validated_shared_science(run: Mapping[str, Any], requested: Mapping[str, Any] | None = None, *, geometry=None):
    outer = dict(run); signed = outer.pop("scientific_signature", None); _reject(type(signed) is not dict, "persisted scientific signature is missing"); signed = dict(signed); assets = signed.pop("accepted_scientific_assets", None); _reject(signed != outer, "persisted duplicated scientific signature differs"); expected = {"schema_version", "generation", "fingerprint", "source", "processing_mode", "output_mode", "live_mode", "batch_mode", "max_cores", "gi", "threshold", "poni_file", "poni_values", "mask_file", "project_root", "save_path", "bai_1d_args", "bai_2d_args", "run_options"}; _reject(set(outer) != expected | ({"background"} if "background" in outer else set()), "persisted run configuration has a noncanonical keyset")
    persisted_gi = outer.get("gi"); _keys(persisted_gi, {"enabled", "incidence_motor", "resolved_motor", "th_val", "sample_orientation", "tilt_angle", "mode_1d", "mode_2d"}, "persisted GI"); accepted_poni = assets.get("poni_values") if type(assets) is dict else None; persisted_poni = outer.get("poni_values"); _reject(persisted_poni is not None and persisted_poni != accepted_poni, "accepted scientific assets are malformed"); projection = {"version": 1, "gi": {key: persisted_gi[key] for key in ("enabled", "incidence_motor", "resolved_motor", "th_val", "sample_orientation", "tilt_angle")}, "threshold": outer.get("threshold"), "poni_values": accepted_poni if persisted_poni is None else persisted_poni, "accepted_scientific_assets": assets, "geometry": geometry, "background": outer.get("background", {"version": 1, "mode": "None"})}; _reject(requested is not None and projection != _plain(requested), "requested shared science differs from acquisition"); return projection
def _validate_science(selected, shared, dimension):
    from xrd_tools.reduction.background import FrameBackgroundPlan; _keys(selected, {"version", "dimension", "bai_args", "gi_mode"}, "selected_plan"); _keys(shared, {"version", "gi", "threshold", "poni_values", "accepted_scientific_assets", "geometry", "background"}, "requested_shared_science")
    _reject(type(dimension) is not str or dimension not in {"1d", "2d"} or type(selected["version"]) is not int or selected["version"] != 1 or type(selected["dimension"]) is not str or selected["dimension"] != dimension or type(selected["bai_args"]) is not dict or {"gi_mode_1d", "gi_mode_2d"} & set(selected["bai_args"]) or type(shared["version"]) is not int or shared["version"] != 1, "preparation version/dimension")
    gi, threshold, assets = shared["gi"], shared["threshold"], shared["accepted_scientific_assets"]; geometry = shared["geometry"]
    if geometry is not None:
        _reject(type(geometry) is not dict or set(geometry) != {"convention", "mapping_json", "motor_sources"} or type(geometry["convention"]) is not str or type(geometry["mapping_json"]) is not str or type(geometry["motor_sources"]) is not dict or any(type(k) is not str or type(v) is not str for k, v in geometry["motor_sources"].items()), "geometry science is malformed")
        try: mapping_json = geometry["mapping_json"]; payload = json.loads(mapping_json); module = __import__("xrd_tools.core.geometry", fromlist=["Diffractometer", "DiffractometerGeometry"]); kind = "Diffractometer" if type(payload) is dict and "preset" in payload and "convention" not in payload else "DiffractometerGeometry" if type(payload) is dict and "convention" in payload and "preset" not in payload else None; _reject(kind is None, "geometry science is malformed"); parsed_geometry = getattr(module, kind).from_json(mapping_json)
        except (TypeError, ValueError, KeyError) as error: raise ValueError("geometry science is malformed") from error
        _reject(parsed_geometry.to_json() != mapping_json or geometry["convention"] != getattr(parsed_geometry, "preset", getattr(parsed_geometry, "convention", None)) or geometry["motor_sources"] != {motor: motor for motor in parsed_geometry.all_referenced_motors()}, "geometry science is malformed")
    _keys(gi, {"enabled", "incidence_motor", "resolved_motor", "th_val", "sample_orientation", "tilt_angle"}, "gi"); _keys(threshold, {"apply_threshold", "threshold_min", "threshold_max", "mask_saturation"}, "threshold"); _keys(assets, {"poni_values", "poni_detector_config_json", "poni_sha256", "mask_sha256"}, "accepted_scientific_assets")
    finite = lambda value: type(value) is float and math.isfinite(value)
    _reject(type(gi["enabled"]) is not bool or any(type(gi[k]) is not str or not gi[k] for k in ("incidence_motor", "resolved_motor")) or gi["enabled"] and gi["incidence_motor"] != gi["resolved_motor"] or not finite(gi["th_val"]) or type(gi["sample_orientation"]) is not int or not 1 <= gi["sample_orientation"] <= 8 or not finite(gi["tilt_angle"]), "GI science is malformed"); _reject(any(type(threshold[k]) is not bool for k in ("apply_threshold", "mask_saturation")) or any(v is not None and not finite(v) for v in (threshold["threshold_min"], threshold["threshold_max"])) or threshold["threshold_min"] is not None and threshold["threshold_max"] is not None and threshold["threshold_min"] > threshold["threshold_max"], "threshold science is malformed")
    digest = lambda value: value is None or type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    _reject(shared["poni_values"] != assets["poni_values"] or (assets["poni_values"] is None) != (assets["poni_detector_config_json"] is None) or not all(digest(assets[k]) for k in ("poni_sha256", "mask_sha256")), "accepted scientific assets are malformed")
    if assets["poni_values"] is not None: poni = assets["poni_values"]; _keys(poni, {"dist", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength", "detector"}, "poni_values"); _reject(any(not finite(poni[k]) for k in ("dist", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength")) or poni["dist"] <= 0 or poni["wavelength"] < 0 or type(poni["detector"]) is not str, "accepted PONI values are malformed")
    if assets["poni_detector_config_json"] is not None:
        try: parsed = json.loads(assets["poni_detector_config_json"]); canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError, json.JSONDecodeError) as error: raise ValueError("accepted detector config is malformed") from error
        _reject(type(parsed) is not dict or type(parsed.get("orientation")) is not int or not 1 <= parsed["orientation"] <= 4 or canonical != assets["poni_detector_config_json"], "accepted detector config is noncanonical"); _calibration(shared, gi["th_val"] if gi["enabled"] else None)
    FrameBackgroundPlan.from_mapping(shared["background"]); plan = _core_plan(selected, shared); from xrd_tools.reduction.provenance_config import _integration_1d_args, _integration_2d_args; normalized = (_integration_1d_args(plan.integration_1d, plan.gi) if dimension == "1d" else _integration_2d_args(plan.integration_2d, plan.gi)); normalized.pop(f"gi_mode_{dimension}", None)
    domains = {"1d": {"q_total", "q_ip", "q_oop", "exit_angle", "chi_gi"}, "2d": {"qip_qoop", "q_chi", "exit_angles"}}; _reject(gi["enabled"] != (selected["gi_mode"] is not None) or selected["gi_mode"] is not None and selected["gi_mode"] not in domains[dimension] or normalized != selected["bai_args"], "selected GI mode/BAI differs"); return plan
def _dimension_audit(*, dimension: str, operation_identity: str, science_identity: str, acquisition_fingerprint: str, requested_shared_science: Mapping[str, Any], selected_plan: Mapping[str, Any], append_lineage: bytes | None): return {"schema_version": 1, "operation": "existing_dimension_replacement", "dimension": dimension, "operation_identity": operation_identity, "science_identity": science_identity, "acquisition_fingerprint": acquisition_fingerprint, "shared_science_fingerprint": science_fingerprint(_plain(requested_shared_science)), "selected_plan": _plain(selected_plan), "selected_gi_mode": selected_plan.get("gi_mode"), "append_lineage_action": ("already_absent" if append_lineage is None else "preserved_append_disabled"), "append_lineage_sha256": (None if append_lineage is None else hashlib.sha256(append_lineage).hexdigest())}
def _audit_identity(audit: Mapping[str, Any]) -> str: return hashlib.sha256(_canonical(audit)).hexdigest()
def _scrub_frame(frame): frame.image = frame.background = frame.geometry = frame.source_identity = frame.source_path = frame.loader = frame.mask = None; frame.source_frame_index = frame.normalization_factor = None; frame.background_dependency_bytes = frame.background_dependency_fingerprint = None; frame.metadata.clear()
def _canonical_acquisition_selected(run, shared, dimension):
    from xrd_tools.reduction.provenance_config import _integration_1d_args, _integration_2d_args
    mode_key = f"gi_mode_{dimension}"; bai_value = run.get(f"bai_{dimension}_args"); _reject(type(bai_value) is not dict, "selected dimension differs from acquisition provenance"); bai = dict(bai_value); recorded_mode = bai.pop(mode_key, None); run_gi = run.get("gi") or {}; active = shared["gi"]["enabled"]; mode = run_gi.get(f"mode_{dimension}") if active else None; _reject(active and recorded_mode != mode, "selected dimension differs from acquisition provenance")
    candidate = {"version": 1, "dimension": dimension, "bai_args": bai, "gi_mode": mode}
    try:
        plan = _core_plan(candidate, shared); normalized = (_integration_1d_args(plan.integration_1d, plan.gi) if dimension == "1d" else _integration_2d_args(plan.integration_2d, plan.gi)); normalized.pop(mode_key, None); candidate["bai_args"] = normalized; _validate_science(candidate, shared, dimension)
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("selected dimension differs from acquisition provenance") from error
    return candidate
def _inspect_artifact(target: Path, entry: str, dimension: str) -> _ArtifactInspection:
    import h5py, numpy as np; from xrd_tools.io.record_writer import WriterStateError, _decode_replacement_fact, _replacement_json_node, _replacement_scalar
    def text(node, expected, role): info = None if not isinstance(node, h5py.Dataset) else h5py.check_string_dtype(node.dtype); _reject(not isinstance(node, h5py.Dataset) or node.shape != () or node.maxshape != () or node.chunks is not None or node.compression is not None or dict(node.attrs) or info is None or info.encoding != "utf-8" or info.length is not None or _replacement_scalar(node[()], role) != expected, f"{role} is noncanonical")
    with h5py.File(target, "r") as handle:
        group = _replacement_hard_group(handle, entry); _reject(group is None, "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); top = _replacement_hard_group(group, f"integrated_{dimension}"); index = _replacement_hard_group(top, "frame_index", h5py.Dataset); values = None if index is None else np.asarray(index[()])
        _reject(values is None or index.ndim != 1 or values.dtype.kind not in "iu", "selected frame inventory is not exact"); labels = tuple(int(x) for x in values); _reject(not labels or any(v < 0 for v in labels) or labels != tuple(sorted(set(labels))), "selected frame inventory is not exact")
        try: source_base, lineage, _decoded = decode_replacement_lineage(handle, entry=entry)
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error: raise ValueError("REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED") from error
        config = _replacement_hard_group(group, "reduction/config"); _reject(_replacement_hard_group(config, "source_execution", h5py.Dataset) is None, "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); run_node = _replacement_hard_group(config, "run_configuration", h5py.Dataset); run = _replacement_json_node(config, "run_configuration", "run configuration"); text(run_node, _canonical(run).decode(), "run configuration")
        geom_node = _replacement_hard_group(config, "geometry"); geom_leaves = () if geom_node is None else tuple(_replacement_hard_group(geom_node, name, h5py.Dataset) for name in ("convention", "mapping_json", "motor_sources")); _reject(geom_node is not None and (set(geom_node) != {"convention", "mapping_json", "motor_sources"} or any(node is None for node in geom_leaves)), "selected BAI/GI/geometry is malformed"); geometry = None if geom_node is None else {"convention": _replacement_scalar(geom_leaves[0][()], "geometry convention"), "mapping_json": _replacement_scalar(geom_leaves[1][()], "geometry mapping"), "motor_sources": _replacement_json_node(geom_node, "motor_sources", "geometry motors")}
        shared = _validated_shared_science(run, geometry=geometry); gi_node = _replacement_hard_group(config, "gi_config", h5py.Dataset); gi_raw = _replacement_json_node(config, "gi_config", "selected GI config", required=False); _reject((gi_raw is None) == shared["gi"]["enabled"] or gi_raw is not None and type(gi_raw) is not dict, "selected GI config is malformed"); gi_cfg = {} if gi_raw is None else gi_raw; source = run.get("source"); _reject(source is not None and type(source) is not dict, "REPLACEMENT_RAW_DECODER_UNRECORDED"); options_missing = source is None or "options" not in source; options = {} if options_missing else source["options"]; _reject(type(options) is not dict, "REPLACEMENT_RAW_DECODER_UNRECORDED"); raw_keys = {"raw_dtype", "raw_header_skip", "detector_shape"}; present_raw_keys = set(options) & raw_keys; _reject(bool(present_raw_keys) and present_raw_keys != raw_keys, "REPLACEMENT_RAW_DECODER_UNRECORDED"); raw = {key: options[key] for key in raw_keys} if present_raw_keys else {}
        if raw: _reject(type(raw["raw_dtype"]) is not str or type(raw["raw_header_skip"]) is not int or raw["raw_header_skip"] < 0 or type(raw["detector_shape"]) is not list or len(raw["detector_shape"]) != 2 or any(type(v) is not int or v <= 0 for v in raw["detector_shape"]), "REPLACEMENT_RAW_DECODER_UNRECORDED")
        raw = None if not raw else MappingProxyType({**raw, "detector_shape": tuple(raw["detector_shape"])})
        mode_key = f"gi_mode_{dimension}"; bai_name = f"bai_{dimension}_args"; bai_node = _replacement_hard_group(config, bai_name, h5py.Dataset); bai_value = _replacement_json_node(config, bai_name, "selected BAI"); physical_node = _replacement_hard_group(config, "gi", h5py.Dataset); physical = _replacement_json_node(config, "gi", "persisted GI truth"); physical_keys = set() if not shared["gi"]["enabled"] else {"gi_mode_1d", "gi_mode_2d", "incidence_motor", "th_val", "sample_orientation", "tilt_angle"}; _reject(type(bai_value) is not dict or type(physical) is not bool or physical != shared["gi"]["enabled"] or set(gi_cfg) != physical_keys or physical and any(gi_cfg[key] != shared["gi"][key] for key in ("incidence_motor", "th_val", "sample_orientation", "tilt_angle")), "selected BAI/GI/geometry is malformed"); text(bai_node, _canonical(bai_value).decode(), "selected BAI"); text(physical_node, _canonical(physical).decode(), "persisted GI truth"); (text(geom_leaves[0], geometry["convention"], "geometry convention"), text(geom_leaves[1], geometry["mapping_json"], "geometry mapping"), text(geom_leaves[2], _canonical(geometry["motor_sources"]).decode(), "geometry motors")) if geometry is not None else None
        bai = dict(bai_value); bai_mode = bai.pop(mode_key, None); gi_mode = gi_cfg.get(mode_key); prior_name = f"dimension_replacement_{dimension}"; prior_node = _replacement_hard_group(config, prior_name, h5py.Dataset); prior = _replacement_json_node(config, prior_name, "prior dimension audit", required=False); _reject(bool(gi_cfg) != (gi_node is not None) or (prior is None) != (prior_node is None), "selected config inventory differs"); text(gi_node, _canonical(gi_cfg).decode(), "selected GI config") if gi_node is not None else None; text(prior_node, _canonical(prior).decode(), "prior dimension audit") if prior_node is not None else None
        primary = _replacement_scalar(top.attrs.get("primary_mode", "default"), "selected primary mode"); _reject(type(primary) is not str or primary != (gi_mode or "default") or ((bai_mode != gi_mode) if prior is None else bai_mode is not None), "selected GI mode/BAI differs"); selected = {"version": 1, "dimension": dimension, "bai_args": bai, "gi_mode": gi_mode}; _validate_science(selected, shared, dimension); acquisition = _canonical_acquisition_selected(run, shared, dimension)
        if prior is None: _reject(selected != acquisition, "selected dimension differs from acquisition provenance")
        else:
            _keys(prior, {"schema_version", "operation", "dimension", "operation_identity", "science_identity", "acquisition_fingerprint", "shared_science_fingerprint", "selected_plan", "selected_gi_mode", "append_lineage_action", "append_lineage_sha256"}, "prior dimension audit"); sha = lambda value: type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value); prior_science = _digest({"api_version": 1, "dimension": dimension, "selected_plan": selected, "requested_shared_science": shared}); lineage_hash = None if lineage is None else hashlib.sha256(lineage).hexdigest()
            _reject(prior["schema_version"] != 1 or prior["operation"] != "existing_dimension_replacement" or prior["dimension"] != dimension or prior["selected_plan"] != selected or prior["selected_gi_mode"] != selected["gi_mode"] or not sha(prior["operation_identity"]) or prior["science_identity"] != prior_science or prior["acquisition_fingerprint"] != run.get("fingerprint") or prior["shared_science_fingerprint"] != science_fingerprint(shared) or prior["append_lineage_action"] != ("already_absent" if lineage is None else "preserved_append_disabled") or prior["append_lineage_sha256"] != lineage_hash, "prior dimension audit differs")
        try: first = _decode_replacement_fact(handle, labels[0], entry=entry)
        except WriterStateError as error: raise ValueError("REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED") from error
        _reject(raw is None and any(_source_route(Path(path)) == "raw" for path in _decoder_input_paths(first)), "REPLACEMENT_RAW_DECODER_UNRECORDED"); described, dtype, _path, _image = _source_fact(first, raw_options=raw); detector = _replacement_hard_group(group, "instrument/detector/detector_shape", h5py.Dataset); raw_shape = None if detector is None else np.asarray(detector[()]); _reject(raw_shape is not None and (raw_shape.dtype.kind not in "iu" or raw_shape.size != 2 or any(int(v) <= 0 for v in raw_shape.ravel())), "processed detector descriptor is malformed"); shape = described if raw_shape is None else tuple(int(v) for v in raw_shape.ravel()); _reject(shape != described, "processed detector descriptor is malformed")
        pfg = _replacement_hard_group(group, "per_frame_geometry"); _reject((shared["geometry"] is None) != (pfg is None), "replacement geometry differs")
        if pfg is not None: required = {"frame_index", "rot1", "rot2", "rot3", "incident_angle"}; nodes = {key: _replacement_hard_group(pfg, key, h5py.Dataset) for key in required}; rows_node = nodes["frame_index"]; geometry_rows = None if rows_node is None else np.asarray(rows_node[()]); geometry_labels = () if geometry_rows is None else tuple(int(v) for v in geometry_rows); _reject(set(pfg) != required or any(node is None for node in nodes.values()) or geometry_rows is None or rows_node.ndim != 1 or rows_node.dtype != np.dtype(np.int64) or any(v < 0 for v in geometry_labels) or geometry_labels != tuple(sorted(set(geometry_labels))) or not set(labels) <= set(geometry_labels) or any(nodes[key].ndim != 1 or nodes[key].dtype != np.dtype(np.float32) or len(nodes[key]) != len(geometry_labels) or not np.isfinite(np.asarray(nodes[key][()])[[geometry_labels.index(label) for label in labels]]).all() for key in required - {"frame_index"}), "replacement geometry differs")
        values = {}; gi = shared.get("gi") or {}; motor = gi.get("resolved_motor") if gi.get("enabled") else None; scan_data = _replacement_hard_group(group, "scan_data")
        if motor not in {None, "Manual"} and scan_data is not None: rows_node = _replacement_hard_group(scan_data, "frame_index", h5py.Dataset); motor_node = _replacement_hard_group(scan_data, motor, h5py.Dataset); raw_rows = None if rows_node is None else np.asarray(rows_node[()]); data = None if motor_node is None else np.asarray(motor_node[()]); _reject(raw_rows is None or data is None or rows_node.ndim != 1 or raw_rows.dtype.kind not in "iu" or data.ndim != 1 or data.dtype.kind != "f", "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); rows = tuple(int(v) for v in raw_rows); _reject(any(v < 0 for v in rows) or rows != tuple(sorted(set(rows))) or len(data) != len(rows) or not set(labels) <= set(rows), "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"); values = {label: float(data[rows.index(label)]) for label in labels if math.isfinite(float(data[rows.index(label)]))}
        _reject(motor not in {None, "Manual"} and set(values) != set(labels), "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
        if motor not in {None, "Manual"} and first["source_execution"]["adapter_id"] == "tiff_series": execution = first["source_execution"]; admitted = execution["admitted_motor_values"]; ordinals = tuple(label - execution["first_label"] for label in labels); _reject(len(admitted) != len(execution["member_stamps"]) or any(not 0 <= ordinal < len(admitted) for ordinal in ordinals) or any(admitted[ordinal]["motor"] != motor or admitted[ordinal]["value"] != values[label] for label, ordinal in zip(labels, ordinals)), "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
        mask_node = _replacement_hard_group(group, "instrument/detector/mask", h5py.Dataset); mask = None
        if mask_node is not None: flat = np.asarray(mask_node[()]); _reject(flat.ndim != 1 or not len(flat) or flat.dtype != np.dtype(np.int64) or mask_node.chunks is not None or mask_node.compression is not None or mask_node.maxshape != mask_node.shape or dict(mask_node.attrs) != {"description": "flat pixel indices, shape (N,)"} or any(int(v) < 0 or int(v) >= shape[0] * shape[1] for v in flat), "processed detector mask is malformed"); mask = np.zeros(shape, dtype=bool); mask.ravel()[flat.astype(np.intp)] = True
    fingerprint = run.get("fingerprint"); _reject(type(fingerprint) is not str or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint), "processed acquisition identity is malformed"); return _ArtifactInspection(labels, shape, dtype, shared, selected, fingerprint, source_base, lineage, MappingProxyType(values), mask, raw)
def _core_plan(selected: Mapping[str, Any], shared: Mapping[str, Any], mask=None):
    from xrd_tools.session.readiness import build_native_int_reduction_plan_from_args; args = dict(selected["bai_args"]); dimension = selected["dimension"]; gi = shared["gi"]; threshold = shared["threshold"]
    if gi["enabled"]: args[f"gi_mode_{dimension}"] = selected["gi_mode"]
    plan = build_native_int_reduction_plan_from_args(args if dimension == "1d" else None, args if dimension == "2d" else None, gi_enabled=gi["enabled"], gi_incident_angle=(gi["th_val"] if gi["enabled"] and gi["resolved_motor"] == "Manual" else None), incidence_motor=(None if not gi["enabled"] or gi["resolved_motor"] == "Manual" else gi["resolved_motor"]), tilt_angle=gi["tilt_angle"], sample_orientation=gi["sample_orientation"], integrate_1d=dimension == "1d", integrate_2d=dimension == "2d", threshold_min=(threshold["threshold_min"] if threshold["apply_threshold"] else None), threshold_max=(threshold["threshold_max"] if threshold["apply_threshold"] else None), mask_saturation=threshold["mask_saturation"]); plan.mask = mask; return plan
def _prepare_gi_scouts(target, entry, observed, selected, shared, *, cancel_token=None):
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
            with h5py.File(target, "r") as handle: fact = _decode_replacement_fact(handle, label, entry=entry, metadata_keys=metadata_keys, include_geometry=include_geometry)
            path, image, background, pair = _load_fact(fact, observed.detector_shape, observed.native_dtype, shared, cancel_token, observed.raw_options)
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
def _requirements(shape, dtype, selected, shared): terms = _background_resource_terms(shared["background"]["mode"], shape); descriptor = SimpleNamespace(frame_shape=shape, dtype=_native_dtype(dtype)); return requirements_from(descriptor, _core_plan(selected, shared), background_bytes=terms[0], resolver_background_bytes=terms[1], worker_background_bytes=terms[2], background_binding_bytes=terms[3])
def _allocation_recipe(allocation: SessionResourceAllocation) -> dict[str, Any]: return {"requirements": {f.name: getattr(allocation.requirements, f.name) for f in fields(SessionResourceRequirements)}, "envelope_bytes": allocation.envelope_bytes, "counts": dict(allocation.counts), "categories": dict(allocation.categories), "minimum_bytes": allocation.minimum_bytes, "floor_bytes": allocation.floor_bytes, "assigned_bytes": allocation.assigned_bytes, "origin": allocation.origin, "oversize_excess_bytes": allocation.oversize_excess_bytes}
def _policy(req, spec: Mapping[str, Any]) -> SessionPolicy:
    _keys(spec, ({"version", "kind", "envelope_bytes", "requests"} if spec.get("kind") == "resolve" else {"version", "kind", "allocation"}), "resource_policy"); _reject(type(spec["version"]) is not int or spec["version"] != 1, "resource policy version")
    if spec["kind"] == "resolve": requests = spec["requests"]; _reject(type(requests) is not dict or not set(requests) <= _REQUESTS or any(type(value) is not int or value < 0 for value in requests.values()) or spec["envelope_bytes"] is not None and (type(spec["envelope_bytes"]) is not int or spec["envelope_bytes"] <= 0), "resource requests"); return resolve_session_policy(req, envelope_bytes=spec["envelope_bytes"], requests=requests, flush=FlushPolicy())
    _reject(spec["kind"] != "explicit", "resource policy kind")
    value = spec["allocation"]; _keys(value, {"requirements", "envelope_bytes", "counts", "categories", "minimum_bytes", "floor_bytes", "assigned_bytes", "origin", "oversize_excess_bytes"}, "allocation")
    _keys(value["requirements"], {f.name for f in fields(SessionResourceRequirements)}, "requirements"); _keys(value["counts"], _REQUESTS, "counts"); _keys(value["categories"], {"source_native", "staging", "records", "publication", "worker"}, "categories")
    _reject(any(type(v) is not int or v < 0 for v in (*value["requirements"].values(), *value["counts"].values(), *value["categories"].values(), value["envelope_bytes"], value["minimum_bytes"], value["floor_bytes"], value["assigned_bytes"], value["oversize_excess_bytes"])) or value["origin"] not in {"automatic", "explicit"}, "allocation leaves are malformed"); req2 = SessionResourceRequirements(**value["requirements"]); _reject(req2.fingerprint != req.fingerprint, "RESOURCE_REQUIREMENTS_IDENTITY"); alloc = SessionResourceAllocation(requirements=req2, **{k: value[k] for k in value if k != "requirements"}); _reject(alloc.origin not in {"automatic", "explicit"}, "allocation origin"); return resolve_session_policy(req, allocation=alloc, flush=FlushPolicy())
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
    api_version: int; target: str; entry: str; expected_target_snapshot: TargetSnapshot; dimension: Literal["1d", "2d"]; labels: tuple[int, ...]; detector_shape: tuple[int, int]; native_dtype: str; selected_plan: Mapping[str, Any]; requested_shared_science: Mapping[str, Any]; gi_bootstrap_incidence: float | None; session_policy: SessionPolicy; rollback_policy: Literal["ROLLBACK_ON_STOP"]; science_identity: str; operation_identity: str
    def __new__(cls, *args, **kwargs): raise TypeError("ReintegratePlan is factory-constructed")
    @property
    def resource_allocation(self) -> SessionResourceAllocation: return self.session_policy.allocation
    @classmethod
    def from_artifact(cls, target: str | os.PathLike[str], *, entry: str, dimension: Literal["1d", "2d"], preparation: Mapping[str, object], expected_target_snapshot: TargetSnapshot | None = None, expected_labels: tuple[int, ...] | None = None, cancel_token: threading.Event | None = None) -> ReintegratePlan:
        _event(cancel_token); preparation = _plain(_freeze(preparation)); _keys(preparation, {"api_version", "selected_plan", "requested_shared_science", "resource_policy"}, "preparation"); selected, shared = preparation["selected_plan"], preparation["requested_shared_science"]; persisted = type(shared) is dict and set(shared) == {"version", "kind"} and type(shared["version"]) is int and shared["version"] == 1 and shared["kind"] == "persisted_target"; _reject(type(preparation["api_version"]) is not int or preparation["api_version"] != 1, "preparation version/dimension")
        if persisted: _validate_persisted_selected(selected, dimension); _reject(type(expected_target_snapshot) is not TargetSnapshot or not expected_target_snapshot.exists or type(expected_labels) is not tuple or not expected_labels or expected_labels != tuple(sorted(set(expected_labels))) or any(type(v) is not int or v < 0 for v in expected_labels), "expected target/labels are malformed")
        else: _validate_science(selected, shared, dimension); _reject(expected_target_snapshot is not None and type(expected_target_snapshot) is not TargetSnapshot or expected_labels is not None and (type(expected_labels) is not tuple or any(type(v) is not int or v < 0 for v in expected_labels)), "expected target/labels are malformed")
        path = Path(target).resolve(); snapshot = capture_target_snapshot(path)
        _reject(not snapshot.exists, "replacement target must exist"); _reject(expected_target_snapshot is not None and snapshot != expected_target_snapshot, "TARGET_SNAPSHOT_CHANGED"); _event(cancel_token); observed = _inspect_artifact(path, entry, dimension); _event(cancel_token); labels = tuple(observed.labels)
        _reject(expected_labels is not None and labels != expected_labels, "EXPECTED_LABELS_CHANGED")
        if persisted: shared = _plain(observed.persisted_shared_science); selected = _resolve_persisted_selected(selected, shared, dimension, observed.mask)
        else: _reject(_plain(observed.persisted_shared_science) != shared, "shared science differs")
        req = _requirements(observed.detector_shape, observed.native_dtype, selected, shared); policy = _policy(req, preparation["resource_policy"])
        selected, bootstrap = _prepare_gi_scouts(path, entry, observed, selected, shared, cancel_token=cancel_token); _event(cancel_token); _reject(capture_target_snapshot(path) != snapshot, "TARGET_SNAPSHOT_CHANGED"); _event(cancel_token); return _make_plan(str(path), entry, dimension, labels, observed.detector_shape, observed.native_dtype, selected, shared, bootstrap, policy, snapshot=snapshot)
    def as_recipe(self) -> dict[str, object]: return {"schema": "xrd_tools.reintegrate.plan", "version": 1, "plan": _plan_mapping(self)}
    @classmethod
    def from_recipe(cls, recipe: Mapping[str, object]) -> ReintegratePlan:
        recipe = _plain(_freeze(recipe)); _keys(recipe, {"schema", "version", "plan"}, "recipe")
        _reject(recipe["schema"] != "xrd_tools.reintegrate.plan" or type(recipe["version"]) is not int or recipe["version"] != 1, "recipe schema/version")
        value = recipe["plan"]; _keys(value, {"api_version", "target", "entry", "expected_target_snapshot", "dimension", "labels", "detector_shape", "native_dtype", "selected_plan", "requested_shared_science", "gi_bootstrap_incidence", "session_policy", "rollback_policy", "science_identity", "operation_identity"}, "recipe plan"); _reject(type(value["api_version"]) is not int or value["api_version"] != 1 or value["rollback_policy"] != "ROLLBACK_ON_STOP" or any(type(value[key]) is not str or len(value[key]) != 64 or any(char not in "0123456789abcdef" for char in value[key]) for key in ("science_identity", "operation_identity")), "recipe plan version/rollback")
        snap = value["expected_target_snapshot"]; _keys(snap, {"exists", "size", "mtime_ns", "device", "inode", "digest"}, "target snapshot")
        _reject(snap["exists"] is not True or any(type(snap[k]) is not int or snap[k] < 0 for k in ("size", "mtime_ns", "device", "inode")) or type(snap["digest"]) is not str or len(snap["digest"]) != 64 or any(c not in "0123456789abcdef" for c in snap["digest"]), "recipe target snapshot")
        snapshot = TargetSnapshot(**snap); session = value["session_policy"]; _keys(session, {"flush", "allocation"}, "session_policy"); _keys(session["flush"], {"interval", "cap", "margin"}, "flush")
        _reject(session["flush"] != {"interval": 8, "cap": 64, "margin": 8}, "recipe flush policy")
        _reject(type(value["labels"]) is not list or type(value["detector_shape"]) is not list, "recipe tuple fields are not JSON arrays"); selected, shared = _plain(_freeze(value["selected_plan"])), _plain(_freeze(value["requested_shared_science"])); _validate_science(selected, shared, value["dimension"])
        req = _requirements(tuple(value["detector_shape"]), value["native_dtype"], selected, shared); policy = _policy(req, {"version": 1, "kind": "explicit", "allocation": session["allocation"]})
        return _make_plan(value["target"], value["entry"], value["dimension"], tuple(value["labels"]), tuple(value["detector_shape"]), value["native_dtype"], selected, shared, value["gi_bootstrap_incidence"], policy, snapshot=snapshot, expected_science=value["science_identity"], expected_operation=value["operation_identity"])
def _plan_mapping(plan: ReintegratePlan) -> dict[str, Any]: return {"api_version": 1, "target": plan.target, "entry": plan.entry, "expected_target_snapshot": _snapshot_mapping(plan.expected_target_snapshot), "dimension": plan.dimension, "labels": list(plan.labels), "detector_shape": list(plan.detector_shape), "native_dtype": plan.native_dtype, "selected_plan": _plain(plan.selected_plan), "requested_shared_science": _plain(plan.requested_shared_science), "gi_bootstrap_incidence": plan.gi_bootstrap_incidence, "session_policy": {"flush": {"interval": 8, "cap": 64, "margin": 8}, "allocation": _allocation_recipe(plan.resource_allocation)}, "rollback_policy": plan.rollback_policy, "science_identity": plan.science_identity, "operation_identity": plan.operation_identity}
def _make_plan(path, entry, dimension, labels, shape, dtype, selected, shared, bootstrap, policy, *, snapshot=None, expected_science=None, expected_operation=None):
    replay = snapshot is not None; target = str(path) if replay else str(Path(path).resolve()); _reject(replay and (type(path) is not str or not os.path.isabs(path) or os.path.abspath(os.path.normpath(path)) != path), "recipe target/dtype is noncanonical"); snapshot = snapshot or capture_target_snapshot(target); native = _native_dtype(dtype); gi = shared["gi"]; _reject(type(entry) is not str or not entry or dimension not in {"1d", "2d"} or type(labels) is not tuple or not labels or labels != tuple(sorted(set(labels))) or any(type(v) is not int or v < 0 for v in labels) or type(shape) is not tuple or len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape) or native.kind not in "iuf" or native.str != dtype or (bootstrap is not None if not gi["enabled"] or gi["resolved_motor"] == "Manual" else type(bootstrap) is not float or not math.isfinite(bootstrap)), "plan facts are malformed")
    selected, shared = _freeze(selected), _freeze(shared); science = _digest({"api_version": 1, "dimension": dimension, "selected_plan": selected, "requested_shared_science": shared})
    payload = {"target": target, "entry": entry, "snapshot": _snapshot_mapping(snapshot), "labels": labels, "shape": shape, "dtype": dtype, "bootstrap": bootstrap, "rollback": "ROLLBACK_ON_STOP", "flush": {"interval": 8, "cap": 64, "margin": 8}, "allocation": _allocation_recipe(policy.allocation), "science": science}
    operation = _digest(payload); _reject(expected_science is not None and science != expected_science, "SCIENCE_IDENTITY"); _reject(expected_operation is not None and operation != expected_operation, "OPERATION_IDENTITY"); obj = object.__new__(ReintegratePlan)
    values = (1, target, entry, snapshot, dimension, labels, shape, str(dtype), selected, shared, bootstrap, policy, "ROLLBACK_ON_STOP", science, operation); [object.__setattr__(obj, field.name, value) for field, value in zip(fields(ReintegratePlan), values)]; return obj
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
    def __init__(self, plan, token=None, raw_options=None): self.plan, self.token, self.raw_options, self.bound_allocation, self._jit, self._fact_reader, self._frames = plan, token, raw_options, None, {}, None, {}; self._metadata_keys, self._include_geometry = _fact_projection(getattr(plan, "selected_plan", None), plan.requested_shared_science)
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
            path, image, background, pair = _load_fact(fact, self.plan.detector_shape, self.plan.native_dtype, self.plan.requested_shared_science, self.token, self.raw_options)
            revision = int(fact["snapshot"]["mtime_ns"]); gi = self.plan.requested_shared_science["gi"]; incidence = fact["metadata"].get(gi["resolved_motor"]); _reject(gi["enabled"] and gi["resolved_motor"] != "Manual" and (type(incidence) is not float or not math.isfinite(incidence) or frame.index == self.plan.labels[0] and incidence != self.plan.gi_bootstrap_incidence), "REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED")
            frame.source_path, frame.source_frame_index, frame.source_identity = path, fact["frame_index"], fact; frame.metadata.update(fact["metadata"]); frame.background = background
            frame.geometry = _geometry_fact(fact, self.plan.requested_shared_science)
            if pair is not None: frame.background_dependency_bytes, frame.background_dependency_fingerprint = pair
            geometry = None if frame.geometry is None else {key: getattr(frame.geometry, key) for key in ("rot1", "rot2", "rot3", "incident_angle")}; dependency = None if frame.background_dependency_bytes is None and frame.background_dependency_fingerprint is None else (frame.background_dependency_bytes, frame.background_dependency_fingerprint); _reject(frame.index != fact["label"] or frame.source_path != path or frame.source_frame_index != fact["frame_index"] or frame.source_identity is not fact or dict(frame.metadata) != dict(fact["metadata"]) or geometry != (None if not fact["geometry"] else dict(fact["geometry"])) or dependency != fact["background_dependency"], "replacement local JIT stub differs")
            self._jit.update({"source_identity": fact, "image": image, "background": background, "metadata": frame.metadata, "geometry": frame.geometry, "normalization": frame.normalization_factor, "mask": frame.mask, "dependency": pair}); return image, revision
        except BaseException as error: self.clear_label(frame.index); (None if type(error).__name__ != "WriterStateError" or type(error).__module__ != "xrd_tools.io.record_writer" else (_ for _ in ()).throw(ValueError("REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"))); raise
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
        _event(self.token)
        from xrd_tools.reduction.core import NexusSink; from xrd_tools.session import DynamicAccountingLimits, DynamicFrameIdentity, DynamicRunAccounting, StageLedger, required_result_modes; from xrd_tools.session.scan_session import ScanSession; inspected = _inspect_artifact(path, self.plan.entry, self.plan.dimension); _event(self.token)
        if capture_target_snapshot(path) != self.plan.expected_target_snapshot: raise ValueError("TARGET_SNAPSHOT_CHANGED")
        _event(self.token)
        if inspected.labels != self.plan.labels or inspected.detector_shape != self.plan.detector_shape or inspected.native_dtype != self.plan.native_dtype or _plain(inspected.persisted_shared_science) != _plain(self.plan.requested_shared_science) or self.plan.gi_bootstrap_incidence is not None and inspected.gi_values.get(self.plan.labels[0]) != self.plan.gi_bootstrap_incidence: raise ValueError("RECIPE_ARTIFACT_FACTS_CHANGED")
        audit = _dimension_audit(dimension=self.plan.dimension, operation_identity=self.plan.operation_identity, science_identity=self.plan.science_identity, acquisition_fingerprint=inspected.acquisition_fingerprint, requested_shared_science=self.plan.requested_shared_science, selected_plan=self.plan.selected_plan, append_lineage=inspected.append_lineage); self.audit = _audit_identity(audit); background = self.plan.requested_shared_science["background"]; run = {} if background["mode"] == "None" else {"background": _plain(background)}
        lock = threading.RLock(); self.source = _ReintegrateFrameSource(self.plan, self.token, inspected.raw_options); self.sink = NexusSink.for_existing_replacement(path, expected_target_snapshot=self.plan.expected_target_snapshot, dimension=self.plan.dimension, labels=self.plan.labels, audit_bytes=_canonical(audit), selected_plan=self.plan.selected_plan["bai_args"], selected_gi_mode=self.plan.selected_plan["gi_mode"], cancel_token=self.token, entry=self.plan.entry, source_base=inspected.source_base, run_configuration_provenance=run, write_thumbnails=False, flush_every=None, file_lock=lock)
        self.source.bind_fact_reader(lambda label, **kwargs: self.sink._writer._detach_replacement_fact(label, **kwargs)); core_plan = _core_plan(self.plan.selected_plan, self.plan.requested_shared_science, inspected.mask); modes = required_result_modes(core_plan); targets = {mode: (f"nexus:{path}",) for mode in modes}; ledger = StageLedger(required_modes=modes, targets_by_mode=targets); self.accounting = DynamicRunAccounting(ledger, run_generation=1, limits=DynamicAccountingLimits(1, 1, len(self.plan.labels)))
        try: self.session = ScanSession(core_plan, self.source, self.sink, policy=self.plan.session_policy, cancel_token=self.token, clear_frame_images=True, accounting=ledger, dynamic_accounting=self.accounting, targets_by_mode=targets)
        except BaseException as error: self.primary = error; return self._settle()
        total = len(self.plan.labels)
        try:
            self.session.start()
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
        try: self.session.flush(force=True) if self.session._session._current_failure() is None else None; snapshot = self.accounting.snapshot(); pairs = snapshot.publication_dropped | snapshot.pending_publication_dropped; self.dropped = tuple(label for label in self.plan.labels if any(key.logical_frame_identity == label for key, _mode in pairs))
        except BaseException as error: self._note(error); self._stop(error)
        return self._settle()
    def finish_current(self): self.session._mark_dynamic_failure(ReintegrateCancelled("reintegration cancelled before commit")) if self.token is not None and self.token.is_set() and self.session is not None and not self.sink._transaction.snapshot().writer_succeeded else None; return self._settle()
    def close(self): source, sink = self.source, self.sink; writer = None if sink is None else sink._writer; source.clear_jit() if source is not None else None; source._frames.clear() if source is not None else None; setattr(source, "_fact_reader", None) if source is not None else None; [setattr(writer, name, None) for name in ("_replacement_configuration", "_replacement_read_context", "_replacement_manifest", "_replacement_expected")] if writer is not None else None; writer._row_cursors.clear() if writer is not None else None; setattr(writer, "_replacement_labels", ()) if writer is not None else None; self.session = self.source = self.sink = self.result = self.accounting = None
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
