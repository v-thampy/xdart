"""Finite native Stitch mask and raw-preview facts; no detector image decoding."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat

import numpy as np

from xrd_tools.core.geometry import ImageOrientation
from xrd_tools.corrections.detector import DEFAULT_MIN_FLAT_FRACTION, prepare_flatfield
from xrd_tools.io.stat_identity import identity_ctime_ns


def _refuse(code, message):
    from xrd_tools.analysis.stitch_operation import StitchOperationRefused
    raise StitchOperationRefused(code, message)


def _state(path):
    info = os.stat(path, follow_symlinks=False)
    return (info.st_mode, info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, identity_ctime_ns(info.st_ctime_ns))


@dataclass(frozen=True, slots=True)
class StitchMaskReceipt:
    lexical_path: str
    resolved_path: str
    revision: tuple[int, ...]
    sha256: str
    shape: tuple[int, int]
    pixels: bytes

    @property
    def fingerprint(self):
        from xrd_tools.analysis.module_transaction import module_plan_fingerprint, ModuleKind
        return module_plan_fingerprint(ModuleKind.STITCH, (
            "stitch-mask-v1", self.lexical_path, self.resolved_path, self.revision,
            self.sha256, self.shape, hashlib.sha256(self.pixels).hexdigest(),
        ))


def capture_mask(path):
    """Capture the shared mask decoder's bool result and its exact file revision."""
    from xrd_tools.io.image import load_mask, read_detector_image_layout
    lexical = Path(path).expanduser().absolute()
    try:
        if lexical.is_symlink():
            _refuse("MASK_SYMLINK_REFUSED", "mask cannot be a symlink")
        resolved = lexical.resolve(strict=True)
        before = _state(resolved)
        if not stat.S_ISREG(before[0]) or not 0 < before[3] <= 64 << 20:
            _refuse("MASK_SIZE_UNSUPPORTED", "mask must be a bounded regular file")
        if resolved.suffix.lower() == ".npy":
            header = np.load(resolved, mmap_mode="r", allow_pickle=False)
            shape, size = header.shape, header.nbytes
            del header
        else:
            layout = read_detector_image_layout(resolved)
            shape = layout.shape
            size = math.prod(shape) * np.dtype(layout.dtype).itemsize
        if len(shape) != 2 or min(shape) < 1 or size > 64 << 20:
            _refuse("MASK_SIZE_UNSUPPORTED", "decoded mask exceeds the bound")
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        mask = load_mask(resolved)
        after = _state(resolved)
        if before != after or lexical.resolve(strict=True) != resolved:
            _refuse("MASK_IDENTITY_MISMATCH", "mask changed during capture")
    except OSError as error:
        _refuse("MASK_UNAVAILABLE", str(error))
    return StitchMaskReceipt(str(lexical), str(resolved), after, digest,
                             tuple(mask.shape), mask.tobytes(order="C"))


def revalidate_mask(receipt):
    if receipt is not None and capture_mask(receipt.lexical_path) != receipt:
        _refuse("MASK_IDENTITY_MISMATCH", "mask no longer matches Preview")


_DETECTOR_IMAGE_SUFFIXES = frozenset({".npy", ".tif", ".tiff", ".edf", ".cbf"})


@dataclass(frozen=True, slots=True)
class StitchImageReceipt:
    """A dark or flat detector image bound to its exact file revision."""

    role: str
    lexical_path: str
    resolved_path: str
    revision: tuple[int, ...]
    sha256: str
    shape: tuple[int, int]
    pixels_sha256: str
    #: flats only (derived from the pixels): pixels prepare_flatfield masks, and the median
    flat_masked_pixels: int | None = None
    flat_median: float | None = None
    #: flats only: the cutoff those masked pixels were counted with
    flat_min_fraction: float | None = None

    @property
    def fingerprint(self):
        from xrd_tools.analysis.module_transaction import module_plan_fingerprint, ModuleKind
        return module_plan_fingerprint(ModuleKind.STITCH, (
            "stitch-detector-image-v1", self.role, self.lexical_path, self.resolved_path,
            self.revision, self.sha256, self.shape, self.pixels_sha256,
        ))

    def to_provenance(self, project_root):
        from xrd_tools.analysis.stitch_operation import _relative_locator
        return {"relative_path": _relative_locator(self.resolved_path, project_root, self.role),
                "lexical_relative_path": _relative_locator(self.lexical_path, project_root, self.role),
                "sha256": self.sha256, "revision": list(self.revision), "shape": list(self.shape),
                "receipt_fingerprint": self.fingerprint,
                **({} if self.role != "flat" else {
                    "min_fraction": (DEFAULT_MIN_FLAT_FRACTION if self.flat_min_fraction is None
                                     else self.flat_min_fraction),
                    "masked_pixels": self.flat_masked_pixels,
                    "median": self.flat_median})}


def capture_detector_image(path, role, max_bytes, min_fraction=DEFAULT_MIN_FLAT_FRACTION):
    """Decode a single-image dark or flat file and bind its exact revision.

    Returns the receipt and the decoded float64 pixels in file orientation.
    """
    from xrd_tools.io.image import read_detector_image_layout, read_image
    code = role.upper()
    lexical = Path(path).expanduser().absolute()
    try:
        if lexical.is_symlink():
            _refuse(f"{code}_SYMLINK_REFUSED", f"{role} cannot be a symlink")
        resolved = lexical.resolve(strict=True)
        if resolved.suffix.lower() not in _DETECTOR_IMAGE_SUFFIXES:
            _refuse(f"{code}_FORMAT_UNSUPPORTED",
                    f"{role} must be one .npy, .tif, .edf or .cbf detector image")
        before = _state(resolved)
        if not stat.S_ISREG(before[0]) or not 0 < before[3] <= max_bytes:
            _refuse(f"{code}_SIZE_UNSUPPORTED", f"{role} must be a bounded regular file")
        if (resolved.suffix.lower() != ".npy"
                and read_detector_image_layout(resolved).frame_count != 1):
            # read_image would silently take the first frame of a stack.
            _refuse(f"{code}_FORMAT_UNSUPPORTED", f"{role} must hold exactly one image")
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        pixels = (np.load(resolved, allow_pickle=False) if resolved.suffix.lower() == ".npy"
                  else read_image(resolved))
        pixels = np.array(pixels, dtype=np.float64, order="C", copy=True)
        if pixels.ndim != 2 or min(pixels.shape) < 1 or pixels.nbytes > max_bytes:
            _refuse(f"{code}_SIZE_UNSUPPORTED", f"{role} must decode to one bounded 2-D image")
        after = _state(resolved)
        if before != after or lexical.resolve(strict=True) != resolved:
            _refuse(f"{code}_IDENTITY_MISMATCH", f"{role} changed during capture")
    except (OSError, ValueError) as error:
        from xrd_tools.analysis.stitch_operation import StitchOperationRefused
        if isinstance(error, StitchOperationRefused):
            raise
        _refuse(f"{code}_UNAVAILABLE", str(error))
    pixels.setflags(write=False)
    masked = median = None
    if role == "flat":
        finite = pixels[np.isfinite(pixels)]
        median = float(np.median(finite)) if finite.size else None
        try:
            masked = int(np.isnan(prepare_flatfield(pixels, min_fraction)).sum())
        except ValueError:
            _refuse("FLAT_UNUSABLE", "the flat field has no usable pixels")
    receipt = StitchImageReceipt(role, str(lexical), str(resolved), after, digest,
                                 tuple(pixels.shape), hashlib.sha256(pixels.tobytes()).hexdigest(),
                                 masked, median,
                                 float(min_fraction) if role == "flat" else None)
    return receipt, pixels


def native_detector_image(plan, receipt):
    """Re-decode a bound dark or flat image and orient it into the calibrated frame."""
    if receipt is None:
        return None
    current, pixels = capture_detector_image(
        receipt.lexical_path, receipt.role, plan.max_frame_bytes,
        DEFAULT_MIN_FLAT_FRACTION if receipt.flat_min_fraction is None else receipt.flat_min_fraction)
    if current != receipt:
        _refuse(f"{receipt.role.upper()}_IDENTITY_MISMATCH", f"{receipt.role} no longer matches Preview")
    _wavelength, shape, orientation = geometry_facts(plan)
    value = np.array(orientation.apply(pixels), dtype=np.float64, order="C", copy=True)
    if value.shape != tuple(shape):
        _refuse(f"{receipt.role.upper()}_SHAPE_MISMATCH",
                f"{receipt.role} shape does not match the calibrated detector")
    value.setflags(write=False)
    return value


def geometry_facts(plan):
    from xrd_tools.analysis.stitch_operation import XuStitchOperationPlan, _runtime_geometry
    if type(plan) is XuStitchOperationPlan:
        asset = plan.calibration.projection.value
        rotation = int(asset["acquisition"]["image_rotation_quarter_turns"]) * 90
        return (float(asset["acquisition"]["xdart_wavelength_A"]),
                tuple(asset["detector"]["shape"]), ImageOrientation(rotation=rotation))
    from xrd_tools.integrate.calibration import detector_calibration_to_integrator
    diff, orientation = _runtime_geometry(plan.geometry, plan.geometry.content)
    ai = detector_calibration_to_integrator(diff.calibration)
    return float(ai.wavelength) * 1e10, tuple(ai.detector.shape), orientation


def native_mask(plan):
    from xrd_tools.integrate.stitch_conditioning import prepare_stitch_mask
    receipt = plan._mask_receipt
    if receipt is None:
        return None
    revalidate_mask(receipt)
    _wavelength, shape, orientation = geometry_facts(plan)
    raw = np.frombuffer(receipt.pixels, dtype=bool).reshape(receipt.shape)
    return prepare_stitch_mask(raw, orientation, shape)


def native_plan_identity(plan):
    return ("native-stitch-controls-v1", plan.native_records,
            None if plan.conditioning is None else plan.conditioning.to_dict(),
            None if plan._mask_receipt is None else plan._mask_receipt.fingerprint)


def _snapshot(path, manifest, rotation):
    from xrd_tools.io.image import read_detector_image_layout
    from xrd_tools.sources.descriptor import describe_container
    options = json.loads(manifest.source_options_json)
    read_options = options.get("read_image_kwargs", options)
    raw_options = {key: read_options[key] for key in (
        "detector_shape", "detector", "raw_dtype", "raw_header_skip",
    ) if key in read_options}
    before = _state(path)
    snapshot = {"size": before[3], "mtime_ns": before[4], "rotation": rotation}
    if path.suffix.lower() in {".h5", ".hdf5", ".nxs", ".nexus", ".cxi"}:
        desc = describe_container(path, entry=manifest.source_entry or "entry")
        if not desc.frame_shape or desc.dtype is None or desc.frame_count < 1:
            _refuse("RAW_LAYOUT_UNAVAILABLE", "detector container has no exact layout")
        shape, dtype, count = tuple(desc.frame_shape), np.dtype(desc.dtype), desc.frame_count
        snapshot.update(adapter_id=desc.adapter_id or "nexus_hdf5",
                        self_contained=bool(desc.self_contained))
        if len(desc.segment_paths) > 1:
            # Keep the master-global frame selector for the shared Eiger reader;
            # selecting only its first dataset would misaddress later segments.
            if path.suffix.lower() not in {".h5", ".hdf5"} or not path.stem.endswith("_master"):
                _refuse("RAW_LAYOUT_UNSUPPORTED", "segmented raw view requires its Eiger master")
        elif desc.dataset_path is not None:
            snapshot["dataset_path"] = desc.dataset_path
        elif not desc.segment_paths:
            _refuse("RAW_LOCATOR_MISSING", "detector container has no exact dataset")
    else:
        layout = read_detector_image_layout(path, **raw_options)
        shape, dtype, count = layout.shape, np.dtype(layout.dtype), layout.frame_count
        snapshot.update(adapter_id="image_file", self_contained=True)
    snapshot.update(frame_count=int(count), raw_shape=list(shape), raw_dtype=dtype.str,
                    raw_header_bytes=int(raw_options.get("raw_header_skip", 0)))
    if _state(path) != before:
        _refuse("SOURCE_IDENTITY_MISMATCH", "raw layout changed during Preview")
    return snapshot, before


def frame_seeds(source, manifest, plan):
    """Bind layout and original identities before science, without decoding pixels."""
    from xrd_tools.analysis.stitch_operation import StitchGroupManifestReceipt, _source_members
    members = manifest.members if type(manifest) is StitchGroupManifestReceipt else (manifest,)
    source_members = _source_members(source)
    _wavelength, expected_shape, orientation = geometry_facts(plan)
    records = []
    cache = {}
    for member_index, member in enumerate(members):
        original_scan = member.source_scan or member.source_entry or Path(member.source_relative_path).stem
        # Member ordinal prevents address collisions without discarding original scan identity.
        scan_label = f"{member_index + 1}-{original_scan}".replace("/", "_")
        for contribution in member.contributions:
            file = member.files[contribution.file_ordinal]
            path = Path(manifest.project_root) / file.relative_path
            cache_key = (str(path), member.source_options_json, member.source_entry)
            if cache_key not in cache:
                cache[cache_key] = _snapshot(path, member, orientation.rotation)
            snapshot, state = cache[cache_key]
            shape = tuple(snapshot["raw_shape"])
            if orientation.rotation in {90, 270}:
                shape = shape[::-1]
            if (state != file.revision or shape != expected_shape
                    or contribution.source_frame_index >= snapshot["frame_count"]
                    or math.prod(shape) * 8 > plan.max_frame_bytes):
                _refuse("RAW_LAYOUT_MISMATCH", "selected raw frame differs from calibration or receipt")
            records.append({"frame_index": contribution.label, "scan_label": scan_label,
                "source_path": str(path), "source_frame_index": contribution.source_frame_index,
                "source_snapshot": dict(snapshot), "identity": {
                    "source_id": source_members[member_index].fingerprint,
                    "scan_id": str(original_scan), "frame_index": contribution.label,
                    "contribution_index": len(records),
                }})
    return records


def native_provenance(value, source, output, plan, manifest):
    from xrd_tools.io.analysis_stitch_auxiliary import (
        analysis_contributing_frame_binding, analysis_contributing_frame_members,
    )
    from xrd_tools.analysis.stitch_operation import _relative_locator
    if not plan.native_records:
        return value
    if output.artifact_family is None:
        _refuse("OUTPUT_FAMILY_REQUIRED", "native Stitch requires its explicit artifact family")
    wavelength, _shape, _orientation = geometry_facts(plan)
    if not math.isfinite(wavelength) or wavelength <= 0:
        _refuse("STITCH_WAVELENGTH_INVALID", "calibration wavelength is unavailable")
    receipt = plan._mask_receipt
    value["plan"].update(native_records=True, conditioning=None if plan.conditioning is None
        else plan.conditioning.to_dict(), mask=None if receipt is None else {
            "relative_path": _relative_locator(receipt.resolved_path, Path(manifest.project_root), "mask"),
            "lexical_relative_path": _relative_locator(receipt.lexical_path, Path(manifest.project_root), "mask"),
            "sha256": receipt.sha256, "revision": list(receipt.revision),
            "shape": list(receipt.shape), "receipt_fingerprint": receipt.fingerprint,
        })
    if plan.gi is not None:  # how the GI cuts were made, as the record's reader sees it
        value["plan"]["gi"] = plan.gi.to_provenance()
    value["wavelength_A"] = wavelength
    value["source_energy_checks"] = source_energy_checks(source, wavelength, recorded_energy_eV(plan))
    value["frame_members"] = analysis_contributing_frame_binding(analysis_contributing_frame_members(
        frame_seeds(source, manifest, plan), source_base=manifest.project_root))
    value["output"]["artifact_family"] = output.artifact_family
    return value


def recorded_energy_eV(plan):
    """The energy the scans recorded when the calibration's energy was refined (``source_energy_eV``,
    saved with it), else None: the scans are checked against it rather than the refined one."""
    from xrd_tools.analysis.stitch_operation import XuStitchOperationPlan
    if type(plan) is XuStitchOperationPlan:
        value = plan.calibration.projection.value["acquisition"].get("source_energy_eV")
    else:
        try:  # a goniometer calibration's record; a PONI calibration has none
            record = json.loads(plan.geometry.content)
        except (TypeError, ValueError):
            record = None
        value = record.get("source_energy_eV") if isinstance(record, dict) else None
    return None if value is None else float(value)


def source_energy_checks(source, wavelength_A, recorded_eV=None):
    from xrd_tools.analysis.module_transaction import MetadataColumnSelector
    from xrd_tools.analysis.stitch_operation import _fresh_table, _source_members, _manifest_numeric_lookup
    from xrd_tools.core.energy import wavelength_m_to_energy_eV
    from xrd_tools.core.scan import SourceKind
    from xrd_tools.sources.descriptor import describe_container
    expected = wavelength_m_to_energy_eV(wavelength_A * 1e-10) if recorded_eV is None else float(recorded_eV)
    checks = []
    for member in _source_members(source):
        table = _fresh_table(member)
        names = [column.name for column in table.columns]
        energies = []
        if names.count("energy") > 1:
            _refuse("AMBIGUOUS_METADATA_SELECTOR", "energy must occur at most once per member")
        if "energy" in names:
            positions, columns = _manifest_numeric_lookup(table, (MetadataColumnSelector("energy"),))
            energies.extend(float(columns[0][1][positions[label]]) for label in member.selected_labels)
        if member.analysis.resolved_kind in {SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER}:
            import h5py
            from xrd_tools.io.nexus import _read_energy
            desc = describe_container(member.analysis.resolved_root, entry=member.analysis.resolved_entry or "entry")
            if desc.wavelength is not None:
                # ContainerDescriptor uses the shared NeXus reader's angstroms.
                energies.append(wavelength_m_to_energy_eV(desc.wavelength * 1e-10))
            with h5py.File(member.analysis.resolved_root, "r") as handle:
                energy_kev = _read_energy(handle[member.analysis.resolved_entry or "entry"])
                if math.isfinite(energy_kev):
                    energies.append(energy_kev * 1000)
        if any(not math.isfinite(energy) or energy <= 0 or abs(energy - expected) > 0.001 * expected
               for energy in energies):
            _refuse("STITCH_SOURCE_ENERGY_CONFLICT", "selected member energy differs from calibration")
        checks.append({"source_id": member.fingerprint, "sample_count": len(energies),
                       "range_eV": [min(energies), max(energies)] if energies else None})
    return checks


def _validated_native_members(request, seeds):
    from xrd_tools.io.analysis_stitch_auxiliary import (
        analysis_contributing_frame_binding_matches, analysis_contributing_frame_members,
    )
    members = analysis_contributing_frame_members(seeds, source_base=request.manifest.project_root)
    if not analysis_contributing_frame_binding_matches(request.provenance["frame_members"], members):
        _refuse("SOURCE_IDENTITY_MISMATCH", "raw preview facts changed since Preview")
    return members


def native_frame_members(request):
    """Detach display facts on the worker after checking the frozen Preview binding."""
    return _validated_native_members(request, frame_seeds(request.module.source, request.manifest, request.plan))


def capture_callback(request):
    """Bind the already-previewed seed to each borrowed native science frame."""
    from xrd_tools.io.nexus_record import make_thumbnail_array
    if not request.plan.native_records:
        return [], None
    seeds = frame_seeds(request.module.source, request.manifest, request.plan)
    _validated_native_members(request, seeds)
    by_label = dict(zip(request.module.source.selected_labels, seeds, strict=True))
    records = []
    def capture(label, raw):
        seed = by_label[label]
        if seed is not seeds[len(records)] or np.asarray(raw).dtype != np.dtype(seed["source_snapshot"]["raw_dtype"]):
            _refuse("RAW_LAYOUT_MISMATCH", "science frame order or native dtype differs from Preview")
        records.append({**seed, "thumbnail": make_thumbnail_array(raw),
                        "mask_baked": False, "thumbnail_mask_baked": False})
    return records, capture
