"""Shared persisted Reintegration fixtures and scientific result builders."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

def _r1(value, n=4, q0=.1):
    from xrd_tools.core.containers import IntegrationResult1D
    return IntegrationResult1D(np.linspace(q0, q0 + .9, n), np.full(n, value, float), np.full(n, value / 10, float), "q_A^-1")
def _r2(value, nq=4, nchi=3):
    from xrd_tools.core.containers import IntegrationResult2D
    data = np.arange(nq * nchi, dtype=float).reshape(nq, nchi) + value
    return IntegrationResult2D(np.linspace(.1, 1., nq), np.linspace(-1., 1., nchi), data, data / 10, "q_A^-1", "chi_deg")
def _plans(gi=None, monitor=None):
    from xrd_tools.reduction import Integration1DPlan, Integration2DPlan, ReductionPlan
    one = Integration1DPlan(npt=4, method="numpy", monitor_key=monitor); two = Integration2DPlan(npt_rad=4, npt_azim=3, method="numpy", monitor_key=monitor)
    return one, two, ReductionPlan(one, two, gi=gi)
def _preparation(*, background=None, gi=None, resource_policy=None):
    from xrd_tools.reduction.provenance_config import _integration_1d_args
    if background is None:
        background = {"version": 1, "mode": "None"}
    if gi is None:
        gi = {
            "enabled": False, "incidence_motor": "Manual",
            "resolved_motor": "Manual", "th_val": 0.1,
            "sample_orientation": 4, "tilt_angle": 0.0,
        }
    if resource_policy is None:
        resource_policy = {
            "version": 1, "kind": "resolve", "envelope_bytes": 8 << 30,
            "requests": {"workers": 1, "reduction_inflight": 1},
        }
    return {
        "api_version": 1,
        "selected_plan": {
            "version": 1, "dimension": "1d",
            "bai_args": _integration_1d_args(_plans()[0], None),
            "gi_mode": None if not gi["enabled"] else "q_total",
        },
        "requested_shared_science": {
            "version": 1,
            "gi": gi,
            "threshold": {
                "apply_threshold": False, "threshold_min": None,
                "threshold_max": None, "mask_saturation": False,
            },
            "poni_values": None,
            "accepted_scientific_assets": {
                "poni_values": None, "poni_detector_config_json": None,
                "poni_sha256": None, "mask_sha256": None,
            },
            "geometry": None,
            "background": background,
        },
        "resource_policy": resource_policy,
    }
def _seed_existing(tmp_path, *, labels=(2, 5, 9), append=False, gi=None, monitor=None, monitor_metadata=None, name="fixture", background=False, disabled_motor=None, persisted_poni=True, legacy_bai=False, detector_descriptor=True, source_options=True, artifact_family=None):
    from xdart.gui.tabs.scattering.contracts import SourceExecutionStamp, SourceFileState
    from xrd_tools.core.containers import PONI
    from xrd_tools.core.geometry.diffractometer import DetectorCalibration
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.integrate.calibration import detector_calibration_to_integrator
    from xrd_tools.io.append import AppendIntent, AppendSource
    from xrd_tools.io.output_transaction import StreamTerminal
    from xrd_tools.reduction import Frame, FrameBackgroundPlan, FrameReduction, NexusSink, ReductionResult, Scan
    from xrd_tools.reduction.provenance_config import _integration_1d_args, _integration_2d_args
    from xrd_tools.session.run_configuration import GIIntent, RunIntent, ThresholdIntent
    root = tmp_path / name; root.mkdir()
    target, source = root / "existing.nexus", root / "raw.nxs"
    shape, count = (5, 7), max(labels) + 1; raw = np.arange(count * np.prod(shape), dtype=np.uint16).reshape(count, *shape)
    with h5py.File(source, "w") as handle:
        entry = handle.create_group("entry"); entry.attrs["NX_class"] = "NXentry"
        entry.create_group("instrument/detector").create_dataset(
            "data", data=raw, chunks=(min(2, count), *shape),
        )
    one, two, core_plan = _plans(gi, monitor)
    if legacy_bai:
        integration_extra = {
            "dummy": -1.0, "delta_dummy": 0.0,
            "correctSolidAngle": True, "safe": True,
        }
        one = replace(one, extra=integration_extra)
        two = replace(
            two, azimuth_offset=90.0, extra=integration_extra,
        )
        core_plan = replace(
            core_plan, integration_1d=one, integration_2d=two,
        )
    poni = PONI(.2, .0002, .0003, 0., 0., 0., 1e-10, "Pilatus300kw")
    detector_config = {"max_shape": [5, 7], "orientation": 3}; calibration = DetectorCalibration(poni, detector_config); integrator = detector_calibration_to_integrator(calibration); detector_values = {key: getattr(poni, key) for key in ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")}; detector_values.update(detector_name=poni.detector, x_pixel_size=integrator.detector.pixel2, y_pixel_size=integrator.detector.pixel1)
    poni_file, mask_file = root / "accepted.poni", root / "accepted.mask"
    poni_file.write_bytes(b"accepted-p3-1a-poni"); mask = np.zeros(shape, bool); mask[0, 0] = True; mask_file.write_bytes(mask.astype(np.uint8).tobytes())
    background_plan = FrameBackgroundPlan()
    if background: import fabio; background_path = root / "background.tif"; fabio.tifimage.TifImage(data=np.ones(shape, np.uint16)).write(str(background_path)); background_plan = FrameBackgroundPlan(mode="Single BG File", locator=str(background_path))
    gi_intent = GIIntent(incidence_motor=disabled_motor or "Manual") if gi is None else GIIntent(enabled=True, incidence_motor=gi.incidence_motor or "Manual", th_val=0. if gi.incident_angle is None else gi.incident_angle, sample_orientation=gi.sample_orientation, tilt_angle=gi.tilt_angle, mode_1d=gi.mode_1d.value, mode_2d=gi.mode_2d.value)
    bai_1d = _integration_1d_args(one, gi)
    bai_2d = _integration_2d_args(two, gi)
    if legacy_bai:
        bai_1d = {
            "numpoints": one.npt, "unit": one.unit, "method": one.method,
            "radial_range": None, "azimuth_range": None,
            "gi_mode_1d": "q_total", "npt_oop": 1000,
            "dummy": -1.0, "delta_dummy": 0.0,
            "polarization_factor": None, "correctSolidAngle": True,
            "safe": True, "chi_offset": 90.0,
        }
        bai_2d = {
            "npt_rad": two.npt_rad, "npt_azim": two.npt_azim,
            "unit": two.unit, "method": two.method,
            "radial_range": None, "azimuth_range": None,
            "gi_mode_2d": "q_chi", "dummy": -1.0,
            "delta_dummy": 0.0, "polarization_factor": None,
            "correctSolidAngle": True, "safe": True, "chi_offset": 90.0,
        }
    intent = RunIntent(
        source_spec=SourceSpec(source, SourceKind.NEXUS_STACK, entry="entry"),
        processing_mode="Int 1D + 2D", output_mode="Overwrite",
        bai_1d_args=bai_1d,
        bai_2d_args=bai_2d, gi=gi_intent,
        threshold=ThresholdIntent(mask_saturation=False),
        background=background_plan,
        poni_file=str(poni_file),
        poni_values=poni.to_dict() if persisted_poni else None,
        mask_file=str(mask_file), project_root=str(root), save_path=str(target),
    )
    frozen = intent.freeze(gi_motor_choices=(None if gi is None else [gi_intent.incidence_motor]))
    assets = {"poni_values": poni.to_dict(), "poni_detector_config_json": json.dumps(detector_config, sort_keys=True, separators=(",", ":")), "poni_sha256": hashlib.sha256(poni_file.read_bytes()).hexdigest(), "mask_sha256": hashlib.sha256(mask_file.read_bytes()).hexdigest()}
    provenance = frozen.as_provenance()
    if source_options is False:
        provenance["source"].pop("options")
    elif source_options is None:
        provenance["source"]["options"] = None
    signed = copy.deepcopy(provenance); signed["accepted_scientific_assets"] = assets
    provenance["scientific_signature"] = signed
    state = SourceFileState.capture(source)
    execution = SourceExecutionStamp(state, "nexus_hdf5", count, 0)
    snapshot = {"adapter_id": "nexus_hdf5", "size": state.size, "mtime_ns": state.mtime_ns, "frame_count": count, "dataset_path": "/entry/instrument/detector/data", "self_contained": True}
    same = None
    if append:
        same = AppendIntent(
            "entry", str(root), "fixture/source", frozen.fingerprint,
            (("1d:q_total", "2d:qip_qoop") if gi else
             ("1d:default", "2d:default")),
            AppendSource(str(source), "nexus_hdf5", state.size, state.mtime_ns,
                         count, dataset_paths=(snapshot["dataset_path"],)),
            tuple(labels),
        )
    values = [float(label) / 10 for label in labels]
    monitor_source = monitor_metadata or monitor
    frames = [Frame(label, image=raw[label].copy(), metadata={"theta": value, **({monitor_source: float(label + 1)} if monitor_source else {})}, source_path=source, source_frame_index=label) for label, value in zip(labels, values)]
    if background:
        from xrd_tools.reduction import resolve_frame_background
        for frame in frames: resolved = resolve_frame_background(background_plan, (frame.index, str(source.resolve()), snapshot["dataset_path"], frame.index, shape, ())); frame.background_dependency_bytes, frame.background_dependency_fingerprint = resolved.descriptor_bytes, resolved.fingerprint
    scan = Scan(
        "seed", frames, poni=poni, integrator=integrator,
        motors={"theta": np.asarray(values)}, extra={
            "detector_shape": shape, "detector_calibration": detector_values,
            "global_mask": np.flatnonzero(mask),
        },
    )
    sink = NexusSink(
        target, overwrite=True, atomic=False, flush_every=None, source_base=root,
        artifact_family=artifact_family,
        run_configuration_provenance=provenance,
        source_execution_provenance=execution.as_dict(),
        source_snapshots_provenance={str(source): snapshot}, same_run_intent=same,
    )
    sink.begin(scan, core_plan); products = {}
    for frame in frames:
        product = FrameReduction(frame.index, _r1(frame.index), _r2(frame.index),
                                 thumbnail=np.full((2, 2), frame.index, np.float32))
        products[frame.index] = product; sink.write(frame, product)
    terminal = sink.finish(ReductionResult("seed", products, len(products)))
    assert type(terminal.commit_identity) is StreamTerminal
    if not detector_descriptor:
        with h5py.File(target, "r+") as handle:
            del handle["entry/instrument/detector/detector_shape"]
    selected_args = _integration_1d_args(one, gi); selected_args.pop("gi_mode_1d", None)
    selected = {"version": 1, "dimension": "1d", "bai_args": selected_args,
                "gi_mode": None if gi is None else gi.mode_1d.value}
    shared_gi = {key: provenance["gi"][key] for key in (
        "enabled", "incidence_motor", "resolved_motor", "th_val",
        "sample_orientation", "tilt_angle")}
    if shared_gi["enabled"]:
        shared_gi["gi_exit_angle_convention"] = provenance["gi"][
            "gi_exit_angle_convention"
        ]
    shared = {
        "version": 1,
        "gi": shared_gi,
        "threshold": provenance["threshold"], "poni_values": poni.to_dict(),
        "accepted_scientific_assets": assets, "geometry": None,
        "background": provenance.get("background", {"version": 1, "mode": "None"}),
    }
    prep = {"api_version": 1, "selected_plan": selected,
            "requested_shared_science": shared, "resource_policy": {
                "version": 1, "kind": "resolve", "envelope_bytes": 8 << 30,
                "requests": {"workers": 1, "reduction_inflight": 1}}}
    return SimpleNamespace(target=target, source=source, raw=raw, labels=tuple(labels), preparation=prep, mask=mask, terminal=terminal, append_intent=same)
def _stub_integrators(monkeypatch, *, dropped=()):
    import xrd_tools.reduction.core as core
    seen = []
    def one(image, integrator, **_kwargs):
        label = int(np.asarray(image).flat[0] // 35); seen.append((label, int(integrator.detector.orientation)))
        return _r1(np.nan if label in dropped else label + 100, q0=.2)
    monkeypatch.setattr(core, "integrate_1d", one); monkeypatch.setattr(core, "integrate_2d", lambda image, _integrator, **_kwargs: _r2(int(np.asarray(image).flat[0] // 35) + 100))
    return seen
