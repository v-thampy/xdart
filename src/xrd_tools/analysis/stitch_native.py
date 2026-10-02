"""Finite native Stitch mask and raw-preview facts; no detector image decoding."""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import stat

import numpy as np

from xrd_tools.core.geometry import ImageOrientation
from xrd_tools.corrections.detector import (
    DEFAULT_MIN_FLAT_FRACTION, check_detector_image, prepare_dark, prepare_flatfield,
)
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
    #: Int's check's warnings (``check_detector_image``): its counts leave out the detector's own mask when the
    #: capture is given it, and count its gaps otherwise
    warnings: tuple[str, ...] = ()

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


def capture_detector_image(path, role, max_bytes, min_fraction=DEFAULT_MIN_FLAT_FRACTION, *, detector_mask=None):
    """Decode a single-image dark or flat file and bind its exact revision.

    Returns the receipt and the decoded float64 pixels in file orientation. ``detector_mask``, the detector's
    own mask (module gaps) in file orientation, is left out of the usability check's counts, as Int's
    preflight passes it; absent, every pixel counts.
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
    # Int's admission (corrections/detector.py, as output_preflight does): one prepared copy serves the
    # usability check, an unusable dark or flat is refused, and the flat's median is the check's (its
    # positive finite pixels) (Codex Stitch/Int reuse audit R4: an all-NaN dark was accepted here)
    name = "flat field" if role == "flat" else "dark"
    try:
        prepared = prepare_flatfield(pixels, min_fraction) if role == "flat" else prepare_dark(pixels)
    except ValueError as error:
        _refuse(f"{code}_UNUSABLE", f"the {name} is unusable: {error}")
    check = check_detector_image(pixels, role, detector_mask=detector_mask, min_fraction=min_fraction,
                                 prepared=prepared)
    if check.errors:
        _refuse(f"{code}_UNUSABLE", check.errors[0])
    masked = median = None
    if role == "flat":
        unusable = np.isnan(prepared)
        if detector_mask is not None and np.shape(detector_mask) == unusable.shape:  # as the check counts
            unusable &= ~np.asarray(detector_mask, dtype=bool)
        masked, median = int(unusable.sum()), check.median
    del prepared
    receipt = StitchImageReceipt(role, str(lexical), str(resolved), after, digest,
                                 tuple(pixels.shape), hashlib.sha256(pixels.tobytes()).hexdigest(),
                                 masked, median,
                                 float(min_fraction) if role == "flat" else None, check.warnings)
    return receipt, pixels


def detector_image_warnings(*receipts):
    """The dark and flat checks' warnings (``check_detector_image``: a flat's median far from 1, many masked pixels,
    counted without the calibrated detector's own mask) of these receipts, each once and as Int shows them: what
    Stitch's and RSM's Previews list for their dark and flat (G8c)."""
    shown = []
    for receipt in receipts:
        for item in (() if receipt is None else receipt.warnings):
            line = item[:1].upper() + item[1:]
            if line not in shown:
                shown.append(line)
    return tuple(shown)


def calibration_detector_mask(value):
    """An xrayutilities calibration's detector mask (module gaps) in the calibration's own turned frame, as Stitch's
    XU backend rebuilds it: pyFAI's detector from the partner's factory name and config, its count and digest
    checked against the record. None: a detector without one. ValueError: not rebuildable, or not the record's."""
    from xrd_tools.integrate.turned_detector import partner_detector
    from xrd_tools.integrate.xu_stitch import _sha256_array

    detector = value["detector"]
    try:  # the named model, or a quarter-turned partner's model turned by its own turn (AUD-9 item 8)
        _rebuilt, mask = partner_detector(detector, value["acquisition"])
    except Exception as error:  # noqa: BLE001 - pyFAI's own failure, by name
        raise ValueError(f"the calibration's detector cannot be rebuilt: {error}") from None
    shape = tuple(int(item) for item in detector["shape"])
    mask = np.zeros(shape, dtype=bool) if mask is None else np.ascontiguousarray(mask, dtype=bool)
    if (mask.shape != shape or int(mask.sum()) != detector["mask_count"]
            or _sha256_array(mask, "|u1") != detector["mask_sha256"]):
        raise ValueError("the calibration's detector mask differs from its record")
    return mask if mask.any() else None


def native_detector_mask(plan):
    """The calibrated detector's own mask (module gaps) in file orientation, read-only, which a dark's and a flat's
    usability checks leave out of their counts, as Int's preflight passes it (AUD-9 stage 2b S3c): the XU
    partner's rebuilt detector, or the pyFAI geometry's when the plan masks with it. None: no gaps, or a plan that
    does not mask them (its run grids them, so its checks count them)."""
    from xrd_tools.analysis.stitch_operation import XuStitchOperationPlan, _detector_mask, _runtime_geometry
    if type(plan) is XuStitchOperationPlan:
        try:
            turned = calibration_detector_mask(plan.calibration.projection.value)
        except ValueError as error:
            _refuse("XU_DETECTOR_MASK_MISMATCH", str(error))
        _wavelength, _shape, orientation = geometry_facts(plan)
    else:
        if not plan.use_detector_mask:
            return None
        diffractometer, orientation = _runtime_geometry(plan.geometry, plan.geometry.content)
        turned = _detector_mask(diffractometer)
    if turned is None or not turned.any():
        return None
    raw = orientation.unapply(turned)
    raw.setflags(write=False)
    return raw


def native_detector_image(plan, receipt):
    """Re-decode a bound dark or flat image and orient it into the calibrated frame (its check given the
    detector's own mask bound at Preview, so the re-capture is Preview's)."""
    if receipt is None:
        return None
    current, pixels = capture_detector_image(
        receipt.lexical_path, receipt.role, plan.max_frame_bytes,
        DEFAULT_MIN_FLAT_FRACTION if receipt.flat_min_fraction is None else receipt.flat_min_fraction,
        detector_mask=plan._detector_mask)
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
        flip = asset["acquisition"].get("image_flip", False)  # a mirrored camera's (written only then)
        return (float(asset["acquisition"]["xdart_wavelength_A"]),
                tuple(asset["detector"]["shape"]), ImageOrientation(rotation=rotation, flip_horizontal=flip))
    from xrd_tools.integrate.calibration import detector_calibration_to_integrator
    diff, orientation = _runtime_geometry(plan.geometry, plan.geometry.content,
                                          metadata_energy_eV=plan.metadata_energy_eV)
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
    value = ("native-stitch-controls-v1", plan.native_records,
            None if plan.conditioning is None else plan.conditioning.to_dict(),
            None if plan._mask_receipt is None else plan._mask_receipt.fingerprint)
    energy = getattr(plan, 'metadata_energy_eV', None)
    return value if energy is None else value + ('metadata-energy-eV', energy)


def native_energy_selection(plan, metadata_energy_eV=None):
    """A pyFAI plan's acquisition term and the calibration's separate fitted offset."""
    from xrd_tools.analysis.stitch_operation import _runtime_geometry
    from xrd_tools.core.energy import resolve_acquisition_energy
    base, _orientation = _runtime_geometry(plan.geometry, plan.geometry.content)
    if metadata_energy_eV is None:
        metadata_energy_eV = () if plan.metadata_energy_eV is None else (plan.metadata_energy_eV,)
    return resolve_acquisition_energy(base.calibration.poni.wavelength, metadata_energy_eV,
                                      fitted_source_energy_eV=recorded_energy_eV(plan))


def _snapshot(path, manifest, rotation, flip=False):
    from xrd_tools.io.image import read_detector_image_layout
    from xrd_tools.sources.descriptor import describe_container
    options = json.loads(manifest.source_options_json)
    read_options = options.get("read_image_kwargs", options)
    raw_options = {key: read_options[key] for key in (
        "detector_shape", "detector", "raw_dtype", "raw_header_skip",
    ) if key in read_options}
    before = _state(path)
    snapshot = {"size": before[3], "mtime_ns": before[4], "rotation": rotation,
                **({"flip_horizontal": True} if flip else {})}  # a mirrored camera's, only then
    if path.suffix.lower() in {".h5", ".hdf5", ".nxs", ".nexus", ".cxi"}:
        desc = describe_container(path, entry=manifest.source_entry or "entry")
        if not desc.frame_shape or desc.dtype is None or desc.frame_count < 1:
            _refuse("RAW_LAYOUT_UNAVAILABLE", "detector container has no exact layout")
        shape, dtype, count = tuple(desc.frame_shape), np.dtype(desc.dtype), desc.frame_count
        recorded = desc.saturation
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
        recorded = layout.saturation
    snapshot.update(frame_count=int(count), raw_shape=list(shape), raw_dtype=dtype.str,
                    raw_header_bytes=int(raw_options.get("raw_header_skip", 0)))
    if _state(path) != before:
        _refuse("SOURCE_IDENTITY_MISMATCH", "raw layout changed during Preview")
    return snapshot, before, recorded


def calibration_detector(plan):
    """The calibration's detector name (pyFAI's), for decision 39's documented counter limits."""
    from xrd_tools.analysis.stitch_operation import XuStitchOperationPlan, _runtime_geometry
    if type(plan) is XuStitchOperationPlan:
        return plan.calibration.projection.value["detector"].get("detector_factory_name")
    diff, _orientation = _runtime_geometry(plan.geometry, plan.geometry.content)
    turned = getattr(diff.calibration, "turned_from", None)
    if turned is not None:  # a quarter-turned model: pyFAI's generic detector names none (AUD-9 item 8)
        return turned["detector"]
    return getattr(diff.calibration.poni, "detector", None)


def _manifest_members(manifest):
    from xrd_tools.analysis.stitch_operation import StitchGroupManifestReceipt
    return manifest.members if type(manifest) is StitchGroupManifestReceipt else (manifest,)


def frame_seeds(source, manifest, plan):
    """Bind layout and original identities before science, without decoding pixels."""
    return _frame_facts(source, manifest, plan)[0]


def _frame_facts(source, manifest, plan):
    """The frame seeds and each member's automatic ceiling (decision 39), from one layout read per file.

    A member's ceiling is the value its files record, else the calibration detector's documented counter
    limit, else its data type's maximum (None for float data with nothing recorded). The files of one
    member must agree on what they record.
    """
    from xrd_tools.analysis.stitch_operation import _source_members
    from xrd_tools.core.invalid import resolve_detector_saturation
    members = _manifest_members(manifest)
    source_members = _source_members(source)
    _wavelength, expected_shape, orientation = geometry_facts(plan)
    detector = calibration_detector(plan)
    records = []
    ceilings = []
    cache = {}
    for member_index, member in enumerate(members):
        recorded = set()
        dtypes = set()
        original_scan = member.source_scan or member.source_entry or Path(member.source_relative_path).stem
        # Member ordinal prevents address collisions without discarding original scan identity.
        scan_label = f"{member_index + 1}-{original_scan}".replace("/", "_")
        for contribution in member.contributions:
            file = member.files[contribution.file_ordinal]
            path = Path(manifest.project_root) / file.relative_path
            cache_key = (str(path), member.source_options_json, member.source_entry)
            if cache_key not in cache:
                cache[cache_key] = _snapshot(path, member, orientation.rotation, orientation.flip_horizontal)
            snapshot, state, file_recorded = cache[cache_key]
            recorded.add(file_recorded)
            dtypes.add(snapshot["raw_dtype"])
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
        if len(recorded) > 1:
            _refuse("STITCH_SATURATION_DIFFERS",
                    f"the files of scan {original_scan} record different detector saturations")
        # Files of mixed data types leave only a recorded or documented value; each frame keeps
        # its own data type's rule in the mask either way.
        ceilings.append(resolve_detector_saturation(recorded=recorded.pop(), detector=detector,
                                                    dtype=dtypes.pop() if len(dtypes) == 1 else None))
    return records, tuple(ceilings)


def member_saturation(ceilings):
    """The per-member values a record keeps: None for the data type's rule; None when all take it."""
    values = tuple(None if ceiling is None or ceiling.source == "dtype" else ceiling.value
                   for ceiling in ceilings)
    return None if all(value is None for value in values) else values


def native_ceilings(request, frames):
    """Each member's automatic ceiling, for the Stitch page's hint: ``(value, "saturation")`` for the
    resolved value the output records, else ``(value, "data type")`` for its frames' data type maximum,
    or None when neither is known (float data). ``frames`` are the request's preview members."""
    from xrd_tools.core.invalid import integer_saturation_ceiling
    recorded = request.provenance["plan"].get("conditioning") or {}
    members = _manifest_members(request.manifest)
    values = recorded.get("saturation") or [None] * len(members)
    ceilings, start = [], 0
    for member, value in zip(members, values, strict=True):
        count = len(member.contributions)
        if value is not None:
            ceilings.append((int(value), "saturation"))
        else:
            native = {integer_saturation_ceiling(np.empty(0, np.dtype(frame["source_snapshot"]["raw_dtype"])))
                      for frame in frames[start:start + count]}
            ceiling = native.pop() if len(native) == 1 else None
            ceilings.append(None if ceiling is None else (int(ceiling), "data type"))
        start += count
    return tuple(ceilings)


def label_saturation(request):
    """Each selected frame's resolved automatic ceiling, from the prepared request's conditioning (the
    values Preview resolved and the output records), by the frame labels the kernels read; None when
    every member takes its data type's rule."""
    conditioning = request.plan.conditioning
    recorded = request.provenance["plan"].get("conditioning") if request.plan.native_records else None
    values = None if conditioning is None or conditioning.mode != "auto" or not recorded else recorded.get("saturation")
    if values is None:
        return None
    members = _manifest_members(request.manifest)
    per_frame = [value for member, value in zip(members, values, strict=True)
                 for _contribution in member.contributions]
    return {label: value for label, value in zip(request.module.source.selected_labels, per_frame, strict=True)
            if value is not None}


def native_provenance(value, source, output, plan, manifest, *, preparation=None):
    """``preparation``: the calling preparation's admitted facts: its metadata tables (reused, not read
    again) and, filled here, the contributing frames' display facts from this one layout read."""
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
    from xrd_tools.analysis.stitch_operation import StitchOperationPlan
    pyfai = type(plan) is StitchOperationPlan
    if pyfai:
        value['plan']['energy_selection'] = native_energy_selection(plan)
        if 'effective_calibration' in value['geometry']:
            value['geometry']['effective_calibration']['wavelength'] = value['plan']['energy_selection']['wavelength_m']
    cached = None if preparation is None else preparation.get('energy_checks')
    value["source_energy_checks"] = cached if cached is not None else source_energy_checks(
        source, wavelength, (plan.metadata_energy_eV if pyfai and plan.metadata_energy_eV is not None
                            else recorded_energy_eV(plan)),
        tables=None if preparation is None else preparation.get("tables"))
    seeds, ceilings = _frame_facts(source, manifest, plan)
    saturation = member_saturation(ceilings)
    if plan.conditioning is not None and plan.conditioning.mode == "auto" and saturation is not None:
        # each member's resolved automatic ceiling (decision 39), recorded with the policy it serves
        value["plan"]["conditioning"] = replace(plan.conditioning, saturation=saturation).to_dict()
    members = analysis_contributing_frame_members(seeds, source_base=manifest.project_root)
    if preparation is not None:
        preparation["members"] = members
    value["frame_members"] = analysis_contributing_frame_binding(members)
    value["output"]["artifact_family"] = output.artifact_family
    return value


def recorded_energy_eV(plan):
    """The energy the scans recorded when the calibration's energy was refined (``source_energy_eV``,
    saved with it), else None: the scans are checked against it rather than the refined one."""
    from xrd_tools.analysis.stitch_operation import XuStitchOperationPlan
    if type(plan) is XuStitchOperationPlan:
        value = plan.calibration.projection.value["acquisition"].get("source_energy_eV")
    else:
        from xrd_tools.core.energy import calibration_recorded_energy_eV
        try:  # a goniometer calibration's record; a PONI calibration has none
            record = json.loads(plan.geometry.content)
        except (TypeError, ValueError):
            return None
        try:
            return calibration_recorded_energy_eV(record)
        except ValueError as error:  # never a reference no comparison can fail (Codex re-review R2-F1)
            _refuse("STITCH_CALIBRATION_ENERGY_INVALID", str(error))
    return None if value is None else float(value)


def source_energy_checks(source, wavelength_A, recorded_eV=None, *, tables=None,
                          allow_calibration_override=False):
    """``tables``: the members' metadata tables a preparation already admitted (in member order);
    absent, each is read."""
    from xrd_tools.analysis.module_transaction import MetadataColumnSelector
    from xrd_tools.analysis.stitch_operation import _fresh_table, _source_members, _manifest_numeric_lookup
    from xrd_tools.core.energy import wavelength_m_to_energy_eV
    from xrd_tools.core.scan import SourceKind
    from xrd_tools.sources.descriptor import describe_container
    expected = wavelength_m_to_energy_eV(wavelength_A * 1e-10) if recorded_eV is None else float(recorded_eV)
    checks = []
    members = _source_members(source)
    for index, member in enumerate(members):
        table = _fresh_table(member) if tables is None else tables[index]
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
        if any(not math.isfinite(energy) or energy <= 0 or
               (not allow_calibration_override and abs(energy - expected) > 0.001 * expected)
               for energy in energies):
            _refuse("STITCH_SOURCE_ENERGY_CONFLICT", ("selected member metadata energy must be finite and positive"
                if allow_calibration_override else "selected member energy differs from calibration"))
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
    """Detach display facts on the worker after checking the frozen Preview binding. A prepared request
    keeps the ones its preparation bound, from the same layout read (PERF-STITCH-1)."""
    if request._preview_members is not None:
        return copy.deepcopy(list(request._preview_members))  # detached, as a fresh read returns them
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
