"""Replay a finite Int request through its shared admission and reduction owners.

The editable RunIntent profile is configuration, not a promise that raw files or
external calibration assets are immutable. Each explicit run admits them again.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.session.run_intent_profile import load_run_intent_profile
from xrd_tools.sources.selection import DirectorySourceSpec


def _relocated(value, previous, root):
    if not value or '://' in str(value):
        return value
    path = Path(value).expanduser()
    if not path.is_absolute():
        return str(root / path)
    if previous:
        try:
            return str(root / path.relative_to(previous))
        except ValueError:
            pass
    return str(path)


def _source_at(spec, previous, root):
    if type(spec) is DirectorySourceSpec:
        return replace(spec, root=Path(_relocated(spec.root, previous, root)))
    if type(spec) is not SourceSpec:
        raise ValueError('REPLAY_SOURCE_REQUIRED: select a raw input')
    options = dict(spec.options)
    for key in ('image_dir', 'selected_file'):
        if options.get(key):
            options[key] = _relocated(options[key], previous, root)
    if options.get('files'):
        options['files'] = tuple(_relocated(item, previous, root) for item in options['files'])
    return SourceSpec(_relocated(spec.uri, previous, root), spec.kind,
        metadata_uri=_relocated(spec.metadata_uri, previous, root), entry=spec.entry, options=options)


def _directory_sources(spec):
    from xrd_tools.sources.adapters import get_adapter
    from xrd_tools.sources.directory_index import DirectoryIndex
    from xrd_tools.sources.run_plan import RunCandidatePlan, directory_candidate_groups
    from xrd_tools.sources.selection import image_series_spec
    from xrd_tools.sources.probe import ProbeState
    index = DirectoryIndex(spec.root, recursive=spec.recursive, name_filter=spec.name_filter)
    index.poll()
    snapshot = index.snapshot
    if spec.suffixes:
        snapshot = replace(snapshot, candidates=tuple(candidate for candidate in snapshot.candidates
                           if candidate.path.suffix.casefold() in spec.suffixes))
    for group in directory_candidate_groups(RunCandidatePlan.from_snapshot(snapshot)):
        candidate = group[0]
        adapter = get_adapter(candidate.adapter_id)
        relative = candidate.path.parent.relative_to(spec.root)
        if SourceKind.IMAGE_FILE in adapter.kinds:
            yield image_series_spec(candidate.path, metadata_format=spec.metadata_format), relative
        else:
            observed = adapter.probe(candidate.path)
            if observed.state is ProbeState.IMAGELESS:
                continue
            if observed.state is not ProbeState.READY or observed.descriptor is None:
                raise ValueError(f'REPLAY_SOURCE_NOT_READY: {candidate.path}: {observed.state.value}')
            descriptor = observed.descriptor
            if descriptor.kind is SourceKind.PROCESSED_NEXUS:
                continue  # ordinary Int Directory also excludes processed records
            yield SourceSpec(candidate.path, descriptor.kind, entry=descriptor.resolved_entry or
                             descriptor.requested_entry), relative


@dataclass
class _BackgroundState:
    background_capsule: object = None


def run_int_replay(configuration, *, data_root, output_root, on_progress=None, should_cancel=None):
    """Execute current Int science in a fresh Project subfolder; return committed paths.

    Live is intentionally refused. Append/Replace describes the original GUI
    target; replay creates a new namespace with CREATE_NEW transactions instead.
    One sustained source reader feeds the ordinary bounded streaming workers.
    """
    import numpy as np
    from xrd_tools.integrate.calibration import detector_calibration_to_integrator, detector_calibration_record
    from xrd_tools.io.output_path import generated_artifact_family, resolve_finite_output_target
    from xrd_tools.reduction import CancelToken, NexusSink, ReductionSession, XYESink
    from xrd_tools.reduction.background import background_frame_fact, resolve_background_before_submit, background_resource_terms
    from xrd_tools.session.policy import requirements_from, resolve_session_policy
    from xrd_tools.session.display_logic import xye_prefix_for_unit
    from types import SimpleNamespace
    from xrd_tools.session.readiness import native_int_reduction_plan
    from xrd_tools.session.scientific_assets import load_scientific_assets
    from xrd_tools.sources import open_source
    from xrd_tools.sources.execution_graph import (qualify_source_execution_graph, graph_saturation,
        source_execution_projection, source_snapshots_projection)
    intent = load_run_intent_profile(json.dumps(configuration, allow_nan=False))
    if intent.live_mode:
        raise ValueError('LIVE_REPLAY_UNAVAILABLE: export a finite request with Live off')
    if intent.processing_mode not in {'Int 1D', 'Int 2D', 'Int 1D (XYE)'}:
        raise ValueError('REPLAY_MODE_UNAVAILABLE: choose an Int processing mode')
    root, destination = Path(data_root).expanduser().resolve(), Path(output_root).expanduser().resolve()
    if destination == root or not destination.is_relative_to(root):
        raise ValueError('REPLAY_OUTPUT_ROOT: choose a fresh subfolder of DATA_ROOT')
    if destination.exists():
        raise FileExistsError(f'REPLAY_OUTPUT_EXISTS: choose a new OUTPUT_ROOT: {destination}')
    if not root.is_dir():
        raise NotADirectoryError(f'DATA_ROOT is not an existing project folder: {root}')
    previous = Path(intent.project_root).expanduser() if intent.project_root else None
    original_output = Path(intent.save_path)
    intent.source_spec = _source_at(intent.source_spec, previous, root)
    for key in ('poni_file', 'mask_file', 'flat_file', 'dark_file'):
        setattr(intent, key, _relocated(getattr(intent, key), previous, root))
    if intent.background.locator:
        intent.background = replace(intent.background,
            locator=_relocated(intent.background.locator, previous, root))
    intent.project_root, intent.save_path = str(root), str(destination)
    cancelled = should_cancel or (lambda: False)
    if cancelled():
        raise InterruptedError('Run replay cancelled')
    assets = load_scientific_assets(intent, cancelled=cancelled)
    if assets.detector_calibration is None:
        raise ValueError('REPLAY_CALIBRATION_REQUIRED: select an existing PONI')
    if intent.mask_file and assets.mask_bytes is None:
        raise ValueError('REPLAY_MASK_REQUIRED: selected mask is unavailable')
    for key in ('flat', 'dark'):
        if getattr(intent, key + '_file') and getattr(assets, key + '_bytes') is None:
            raise ValueError(f'REPLAY_CORRECTION_REQUIRED: selected {key} is unavailable')
    sources = _directory_sources(intent.source_spec) if type(intent.source_spec) is DirectorySourceSpec else (
        (intent.source_spec, Path()),)
    outputs, created, targets = [], False, set()
    mask, flat, dark = assets.mask, assets.flat, assets.dark
    for spec, relative in sources:
        if cancelled():
            raise InterruptedError('Run replay cancelled')
        graph = qualify_source_execution_graph(spec, cancelled=cancelled)
        automatic = intent.threshold.mask_saturation and not intent.threshold.apply_threshold
        saturation = graph_saturation(graph, detector=assets.poni_values[7]) if automatic else None
        candidate = intent.clone_candidate()
        candidate.source_spec = spec
        frozen = candidate.freeze(saturation=None if saturation is None or saturation.source == 'dtype'
                                  else saturation.value)
        source = open_source(spec)
        try:
            calibration = assets.detector_calibration
            integrator = detector_calibration_to_integrator(calibration)
            scan = source.to_scan(poni=calibration.poni, integrator=integrator)
            scan.gi_config = frozen.gi.scan_config().copy()
            scan.extra = dict(scan.extra or {})
            if calibration.parallax is not None:
                scan.extra['detector_calibration'] = detector_calibration_record(calibration, integrator=integrator)
            if mask is not None:
                scan.extra['global_mask'] = np.flatnonzero(mask)
            if flat is not None:
                scan.extra['flatfield'] = flat
            if dark is not None:
                scan.extra['darkcurrent'] = dark
            plan = replace(native_int_reduction_plan(frozen, companion_modes_2d=True), mask=mask, flat=flat, dark=dark)
            family = generated_artifact_family(original_output.stem if original_output.suffix else scan.name,
                                               grazing_incidence=frozen.gi.enabled)
            directory = destination / relative
            target = resolve_finite_output_target(directory, family,
                operation_token='int-2d' if plan.integration_2d is not None else 'int-1d')
            if target in targets:
                raise ValueError(f'REPLAY_OUTPUT_COLLISION: {target.name}; choose distinct input families')
            targets.add(target)
            if not created:
                destination.mkdir(parents=False)  # one exclusive namespace, never an existing output tree
                created = True
            directory.mkdir(parents=True, exist_ok=True)
            xye = frozen.processing_mode == 'Int 1D (XYE)'
            sink = XYESink(directory / family,
                pattern=xye_prefix_for_unit(plan.integration_1d.unit) + '_{scan}_{frame:04d}.xye') if xye else NexusSink(
                target, create_new=True, source_base=root, artifact_family=family,
                run_configuration_provenance=frozen.as_provenance(),
                source_execution_provenance=source_execution_projection(graph),
                source_snapshots_provenance=source_snapshots_projection(graph, writer=True))
            token = CancelToken()
            bgstate = _BackgroundState()
            frames = {int(frame.index): frame for frame in scan.frames}
            selector = graph.descriptor.dataset_path if graph.descriptor is not None else None
            descriptor = graph.descriptor or SimpleNamespace(
                frame_shape=graph.detector_shape, dtype=np.dtype(graph.native_dtype))
            pixels = int(np.prod(descriptor.frame_shape))
            G, R, B, bindings = background_resource_terms(frozen.background, pixels)
            policy = resolve_session_policy(requirements_from(descriptor, plan,
                background_bytes=G, resolver_background_bytes=R,
                worker_background_bytes=B, background_binding_bytes=bindings),
                requested_workers=max(1, int(frozen.max_cores)))
            def progress(event):
                if cancelled():
                    token.cancel()
                if on_progress is not None:
                    on_progress(event)
            with ReductionSession(plan, scan, sink, execution='streaming',
                    executor=policy.allocation.workers, inflight_max=policy.allocation.reduction_inflight,
                    retain_products=False, clear_frame_images=True,
                    progress_cb=progress, cancel_token=token,
                    gi_freeze_mode='scout_union' if frozen.gi.enabled else None) as session:
                for images, labels in source.iter_chunks(1):
                    if cancelled():
                        token.cancel()
                        break
                    image = np.asarray(images)[0] if np.asarray(images).ndim == 3 else np.asarray(images)
                    frame = frames[int(labels[0])]
                    if frozen.background.mode != 'None':
                        fact = background_frame_fact(frame, frozen.background, image.shape, selector)
                        resolve_background_before_submit(frame, frozen.background, None, cancelled=cancelled,
                            frame_fact=fact, capsule_owner=bgstate)
                    if not session.submit(frame, image):
                        break
                result = session.finish()
            if result.failed:
                raise RuntimeError(f'Int replay failed: {result.error}')
            if result.cancelled:
                raise InterruptedError('Run replay cancelled; any committed partial outputs remain in OUTPUT_ROOT')
            if xye:
                outputs.extend(sorted((directory / family).glob('*.xye')))
            else:
                outputs.append(target)
        finally:
            close = getattr(source, 'close', None)
            if close is not None:
                close()
    if not created:
        raise ValueError('REPLAY_SOURCE_EMPTY: no image scans were found')
    return tuple(outputs)
