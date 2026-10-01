"""Typed, bounded scientific contract for the standalone RSM operation.

This module deliberately separates notebook-equivalent scientific intent from
the legacy convenience :class:`~xrd_tools.analysis.plans.RSMPlan`.  The R1
operation requires an explicit normalization policy, records detector/image
conditioning, resolves exact full-detector q bounds without reading detector
images, and applies per-frame foil/exposure normalization to the numerator
before data enter the shared ``sum(raw) / sum(norm)`` grid accumulator.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import InitVar, dataclass, field
from enum import Enum
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import stat
import struct
import threading
import weakref

import numpy as np

from xrd_tools.corrections.normalization import BeamNormalizationRefused, frame_divisors
from xrd_tools.analysis.module_transaction import (
    MetadataColumnSelector,
    ModuleArtifactOutput,
    ModuleArtifactRefused,
    ModuleDisposition,
    ModuleKind,
    ModuleOperationRequest,
    ModuleOutputRequest,
    ModuleProgress,
    ModuleSourceGroupReceipt,
    ModuleSourceReceipt,
    ModuleTerminalResult,
    admit_module_artifact,
    module_artifact_request,
    module_plan_fingerprint,
    module_provenance_digest,
    _rsm_v2_module_request,
)
from xrd_tools.analysis.rsm_geometry_asset import (
    RSMCalibrationGeometryReceipt,
    RSMEffectiveGeometry,
    RSMGeometryAssetRefused,
    RSMGeometryAssetReceipt,
    RSMMemberGeometryBinding,
    bind_rsm_member_geometry,
    lower_rsm_effective_geometry,
    revalidate_rsm_geometry_asset,
    revalidate_rsm_geometry_source,
    rsm_calibration_detector_mask,
    rsm_calibration_solid_angle,
    rsm_effective_pixel_q_map,
)

#: RSM v2's geometry sources: the canonical asset, or a calibration window's xrayutilities partner
#: (AUD-9 stage 2b C7b). Both name the same custody facts (project root, paths, digests, receipt fingerprint).
_RSM_GEOMETRY_RECEIPTS = (RSMGeometryAssetReceipt, RSMCalibrationGeometryReceipt)
from xrd_tools.analysis.plans import RSMPlan, run_rsm
from xrd_tools.analysis.scan_operations import (
    AnalysisDisposition,
    AnalysisSourceLeaseRefused,
    MetadataTablePlan,
    analysis_canonical_fingerprint,
    requalified_analysis_source,
    run_metadata_table,
)
from xrd_tools.core.allocator_pressure import (
    AllocatorPressureCallFailed,
    AllocatorPressureUnavailable,
    bind_allocator_pressure_relief,
)
from xrd_tools.core.geometry import (
    DetectorHeader,
    Diffractometer,
    ImageOrientation,
    PixelQMap,
)
from xrd_tools.core.geometry.xu_runtime import XuRuntimeUnsupported
from xrd_tools.core.scan import SourceKind
from xrd_tools.io.analysis_artifact import (
    AnalysisArtifactCleanupPending,
    AnalysisArtifactKind,
    AnalysisArtifactOverwrite,
    AnalysisArtifactPayload,
    AnalysisArtifactProjectionInvalid,
    analysis_execution_attestation_digest,
    canonical_analysis_provenance,
    project_analysis_artifact_result,
    read_analysis_artifact,
    rsm_record_identity,
)
from xrd_tools.io.nexus import write_rsm
from xrd_tools.io.stat_identity import identity_ctime_ns
from xrd_tools.io.spec import get_UB, get_energy, get_energy_and_UB, get_orientation_header
from xrd_tools.rsm.gridding import (
    RSMGridChunkLease,
    RSMGridChunkReleaseError,
    StreamingGridder,
    _rsm_grid_chunk_release_facts,
)
from xrd_tools.rsm.coordinate_frame import (
    RSMCoordinateFrame,
    rsm_coordinate_matrix,
)
from xrd_tools.rsm.volume import RSMVolume


_MAX_RSM_VOXELS = 8_000_000
_MAX_CHUNK_SIZE = 1024
_MAX_FRAME_BYTES = 4 * 1024 * 1024 * 1024
_MAX_CHUNK_BYTES = 4 * 1024 * 1024 * 1024
_MAX_MANIFEST_BYTES = 512 * 1024
_MAX_MANIFEST_FILES = 8192
_MAX_CONTRIBUTIONS = 4096
_MAX_AXIS_POINTS = 1_000_000
_MAX_PROVENANCE_BYTES = 1024 * 1024
_MAX_JSON_DEPTH = 32
_MAX_RSM_RESIDENT_BYTES = 1024 * 1024 * 1024
_PSIC_ROLES = ("mu", "eta", "chi", "phi", "nu", "del")
#: The raw NeXus stacks an RSM member may be (AUD-9 stage 2b C5c); processed xdart members stay held.
_RSM_NEXUS_KINDS = frozenset({SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER})
_REQUEST_FACTORY = object()
_RSM_V2_FACTORY = object()
_RSM_V2_HOLDS = (
    "gi-q-coordinate-correction",
    "gi-intensity-corrections",
    "incidence-angle-convention-beyond-accepted-non-gi",
    "live-partial-volume-directory-live-and-tiled-streaming",
    "volume-rendering-isosurfaces-vtk-ui-and-gpu-opencl",
    "non-psic-custom-fourc-two-circle-and-sixc-geometry-assets",
    "non-identity-raw-image-orientation",
    "tiled-and-processed-xdart-scan-members",
    "heterogeneous-detector-geometry-or-roi",
    "windows-smb-unc-linux-and-intel-macos-validation",
    "broad-test-battery-launcher-replacement-canonical-merge-push-tag-and-release",
)


class RSMNormalizationMode(str, Enum):
    IDENTITY = "identity"
    FOIL_TRANSMISSION_EXPOSURE = "foil-transmission-exposure"


def _rsm_coordinate_matrix(
    coordinate_frame: RSMCoordinateFrame,
    source_ub: object,
    *,
    invalid_ub_code: str,
) -> np.ndarray:
    """Return the exact matrix xrayutilities must apply for one frame."""

    if type(coordinate_frame) is not RSMCoordinateFrame:
        raise TypeError("RSM coordinate frame must be exact")
    try:
        return rsm_coordinate_matrix(coordinate_frame, source_ub)
    except ValueError as error:
        raise RSMOperationRefused(invalid_ub_code) from error


class RSMOperationRefused(ValueError):
    """An RSM scientific input could not be admitted exactly."""

    def __init__(self, code: str, message: str | None = None):
        self.code = code
        super().__init__(message or code)


class RSMOperationVerificationError(RuntimeError):
    """A committed RSM output could not be strictly re-admitted."""

    def __init__(self, execution: "RSMOperationExecution", message: str):
        self.execution = execution
        super().__init__(message)


class RSMOperationCleanupPending(AnalysisArtifactCleanupPending):
    """Retryable cleanup retaining the exact RSM execution owner."""

    def __init__(self, execution: "RSMOperationExecution"):
        self.execution = execution
        super().__init__(execution.output_snapshot)

    def retry_cleanup(self) -> "RSMOperationResult":
        return self.execution.retry_cleanup()


class RSMOperationVerificationErrorV2(RuntimeError):
    """A committed grouped RSM output could not be strictly re-admitted."""

    def __init__(self, execution: "RSMOperationExecutionV2", message: str):
        self.execution = execution
        super().__init__(message)


class RSMOperationCleanupPendingV2(AnalysisArtifactCleanupPending):
    """Retryable cleanup retaining the exact grouped RSM execution owner."""

    def __init__(self, execution: "RSMOperationExecutionV2"):
        self.execution = execution
        super().__init__(execution.output_snapshot)

    def retry_cleanup(self) -> "RSMOperationResultV2":
        return self.execution.retry_cleanup()


def _failure_diagnostic(error: BaseException) -> str:
    try:
        message = str(error)
    except BaseException:
        message = "exception message unavailable"
    kind = type(error)
    return f"{kind.__module__}.{kind.__qualname__}: {message}"[:4096]


def _finite_float(value: object, name: str, *, positive: bool = False) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be an exact finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a finite {qualifier}number")
    return result


def _selector_value(
    selector: MetadataColumnSelector | None,
) -> tuple[str, int] | None:
    if selector is None:
        return None
    return selector.name, selector.occurrence


def _canonical_roi(
    value: object,
    header: DetectorHeader,
) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if (
        type(value) is not tuple
        or len(value) != 4
        or any(type(item) is not int for item in value)
    ):
        raise TypeError("RSM ROI must be one exact four-integer tuple")
    roi = tuple(value)
    if roi[0] < 0 or roi[2] < 0:
        raise ValueError("RSM ROI starts must be non-negative")
    row_stop = header.Nch1 + roi[1] if roi[1] < 0 else roi[1]
    column_stop = header.Nch2 + roi[3] if roi[3] < 0 else roi[3]
    if (
        not roi[0] < row_stop <= header.Nch1
        or not roi[2] < column_stop <= header.Nch2
    ):
        raise ValueError("RSM ROI must be contained by the detector")
    cropped = header.with_roi(roi)
    if cropped.Nch1 < 2 or cropped.Nch2 < 2:
        raise ValueError("RSM ROI must retain at least a 2 by 2 detector")
    return roi


@dataclass(frozen=True, slots=True)
class RSMNormalizationPolicy:
    """Mandatory per-frame numerator normalization for one RSM plan."""

    mode: RSMNormalizationMode
    foil_selector: MetadataColumnSelector | None
    exposure_selector: MetadataColumnSelector | None
    absorption_lengths: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if type(self.mode) is not RSMNormalizationMode:
            raise TypeError("RSM normalization mode must be exact")
        if (
            type(self.absorption_lengths) is not tuple
            or len(self.absorption_lengths) != 4
        ):
            raise TypeError("RSM absorption lengths must be an exact four-tuple")
        lengths = tuple(
            _finite_float(value, "RSM absorption length")
            for value in self.absorption_lengths
        )
        if any(value < 0 for value in lengths):
            raise ValueError("RSM absorption lengths must be non-negative")
        if self.mode is RSMNormalizationMode.IDENTITY:
            if (
                self.foil_selector is not None
                or self.exposure_selector is not None
                or lengths != (0.0, 0.0, 0.0, 0.0)
            ):
                raise ValueError(
                    "identity RSM normalization requires no selectors and zero lengths"
                )
        else:
            if (
                type(self.foil_selector) is not MetadataColumnSelector
                or type(self.exposure_selector) is not MetadataColumnSelector
            ):
                raise TypeError(
                    "foil/exposure RSM normalization requires exact selectors"
                )
            if self.foil_selector == self.exposure_selector:
                raise ValueError("foil and exposure selectors must be distinct")
            if not any(value > 0 for value in lengths):
                raise ValueError("foil absorption lengths cannot all be zero")
        object.__setattr__(self, "absorption_lengths", lengths)

    @classmethod
    def identity(cls) -> "RSMNormalizationPolicy":
        return cls(
            RSMNormalizationMode.IDENTITY,
            None,
            None,
            (0.0, 0.0, 0.0, 0.0),
        )

    def _canonical_value(self) -> tuple[object, ...]:
        return (
            self.mode,
            _selector_value(self.foil_selector),
            _selector_value(self.exposure_selector),
            self.absorption_lengths,
        )


@dataclass(frozen=True, slots=True)
class RSMFrameCorrections:
    """The per-frame corrections RSM shares with Stitch (design §4; the maintainer, 2026-10-01: "masks should
    be applied when specified. Other processing options like thresholding, monitor etc should also apply";
    decision 5: solid angle and polarization on by default, which the form sets).

    ``mask_path``: a Mask File (with the detector's own mask). ``threshold``: Int's threshold policy on the
    native values. ``dark_path`` and ``flat_path`` (with ``flat_min_fraction``): admitted by Int's owners.
    ``solid_angle`` and ``polarization_factor``: the normalization weight, as pyFAI's. Every field at its
    default is no correction, and a plan without corrections keeps the identity it had."""

    mask_path: str | None = None
    threshold: object = None
    dark_path: str | None = None
    flat_path: str | None = None
    flat_min_fraction: float = 0.1
    solid_angle: bool = False
    polarization_factor: float | None = None

    def __post_init__(self) -> None:
        from xrd_tools.integrate.stitch_conditioning import StitchConditioning
        for name in ("mask_path", "dark_path", "flat_path"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value):
                raise TypeError(f"RSM {name} must be a non-empty path string")
        if self.threshold is not None and type(self.threshold) is not StitchConditioning:
            raise TypeError("RSM threshold must be Int's exact StitchConditioning")
        if type(self.solid_angle) is not bool:
            raise TypeError("RSM solid angle must be an exact bool")
        fraction = _finite_float(self.flat_min_fraction, "RSM flat cutoff")
        if not 0 <= fraction < 1:
            raise ValueError("RSM flat cutoff must be a fraction of the flat's median")
        object.__setattr__(self, "flat_min_fraction", fraction)
        if self.polarization_factor is not None:
            factor = _finite_float(self.polarization_factor, "RSM polarization factor")
            if not -1 <= factor <= 1:
                raise ValueError("RSM polarization factor must lie in [-1, 1]")
            object.__setattr__(self, "polarization_factor", factor)


@dataclass(frozen=True, slots=True)
class RSMBoundCorrections:
    """``RSMFrameCorrections`` bound at Preview: the mask, dark and flat receipts (Stitch's own captures), and
    ``saturation``, each member's automatic ceiling from the calibrated detector (decision 39, Stitch's
    ``member_saturation``): None for the data type's rule, and None when every member takes it. ``detector_mask``:
    the calibrated detector's own mask in the raw frame, which the dark and flat captures leave out of their
    counts at Preview and again at the run (so the run's capture is Preview's); not identity, the geometry's."""

    corrections: RSMFrameCorrections
    mask_receipt: object
    dark_receipt: object
    flat_receipt: object
    saturation: tuple[int | None, ...] | None = None
    detector_mask: np.ndarray | None = field(default=None, compare=False, repr=False)

    def identity(self) -> tuple[object, ...]:
        value = self.corrections
        identity = ("frame-corrections-v1",
                None if self.mask_receipt is None else (self.mask_receipt.sha256, self.mask_receipt.shape),
                None if value.threshold is None else (value.threshold.mode, value.threshold.threshold_min,
                                                      value.threshold.threshold_max),
                None if self.dark_receipt is None else self.dark_receipt.fingerprint,
                None if self.flat_receipt is None else (self.flat_receipt.fingerprint, value.flat_min_fraction),
                value.solid_angle, value.polarization_factor)
        # a resolved ceiling only when one applies: an identity without it is the one it was
        return identity if self.saturation is None else (*identity, ("saturation", self.saturation))

    def to_provenance(self) -> dict[str, object]:
        value = self.corrections
        return {
            "mask": None if self.mask_receipt is None else {
                "name": Path(self.mask_receipt.lexical_path).name, "sha256": self.mask_receipt.sha256,
                "shape": list(self.mask_receipt.shape)},
            "threshold": None if value.threshold is None else {
                "mode": value.threshold.mode, "threshold_min": value.threshold.threshold_min,
                "threshold_max": value.threshold.threshold_max,
                **({} if self.saturation is None else {"saturation": list(self.saturation)})},
            "dark": None if self.dark_receipt is None else {
                "name": Path(self.dark_receipt.lexical_path).name, "fingerprint": self.dark_receipt.fingerprint},
            "flat": None if self.flat_receipt is None else {
                "name": Path(self.flat_receipt.lexical_path).name, "fingerprint": self.flat_receipt.fingerprint,
                "min_fraction": value.flat_min_fraction},
            "solid_angle": value.solid_angle,
            "polarization_factor": value.polarization_factor,
        }


@dataclass(frozen=True, slots=True)
class RSMUBOverride:
    """A UB matrix for an RSM in HKL, replacing the source's (the maintainer, 2026-10-01: "For files with no ub
    matrix, a warning with a chance to enter the ub matrix manually or through a spec file can be offered").

    Exactly one of ``matrix``, nine numbers row by row in SPEC's ``#G3`` convention (as SPEC writes the line), or
    ``spec_path`` with ``scan``, that SPEC scan's own ``#G3``. The numbers are checked when bound
    (``RSM_UB_INVALID``)."""

    matrix: tuple[float, ...] | None = None
    spec_path: str | None = None
    scan: str | None = None

    def __post_init__(self) -> None:
        manual = self.matrix is not None
        if manual == (self.spec_path is not None or self.scan is not None):
            raise TypeError("an RSM UB is nine numbers or a SPEC file and scan, not both")
        if manual:
            if type(self.matrix) is not tuple or len(self.matrix) != 9 or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) for value in self.matrix):
                raise TypeError("an RSM UB matrix is nine numbers, row by row")
            object.__setattr__(self, "matrix", tuple(float(value) for value in self.matrix))
        elif (type(self.spec_path) is not str or not self.spec_path or type(self.scan) is not str
              or not self.scan.strip()):
            raise TypeError("an RSM UB from SPEC needs a SPEC file and a scan")


@dataclass(frozen=True, slots=True)
class RSMBoundUB:
    """``RSMUBOverride`` bound at Preview: the nine numbers used, and for a SPEC source its file's revision."""

    override: RSMUBOverride
    matrix: tuple[float, ...]
    #: (name, resolved path, SHA-256, stat state, scan key) of the SPEC file; None for a typed matrix
    spec_receipt: tuple | None

    def array(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=np.float64).reshape(3, 3)

    def identity(self) -> tuple[object, ...]:
        return ("ub-override-v1", self.matrix,
                None if self.spec_receipt is None else (self.spec_receipt[4], self.spec_receipt[2]))

    def to_provenance(self) -> dict[str, object]:
        if self.spec_receipt is None:
            return {"kind": "manual", "matrix": list(self.matrix)}
        return {"kind": "spec", "name": self.spec_receipt[0], "scan": self.spec_receipt[4],
                "sha256": self.spec_receipt[2], "matrix": list(self.matrix)}


def _rsm_spec_scan_key(scan: str) -> str:
    """A SPEC scan as silx keys it: ``43`` is its first occurrence, ``43.1``; ``43.2`` as given."""
    value = scan.strip()
    return f"{value}.1" if value.isdigit() else value


def _rsm_ub_file_state(path: Path) -> tuple[int, ...]:
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _bind_rsm_ub(override, coordinate_frame) -> RSMBoundUB | None:
    """Bind a UB override at Preview: typed numbers as given, or a SPEC scan's ``#G3`` with its file's revision.
    Only an HKL map uses a UB (``RSM_UB_OVERRIDE_UNUSED`` in Q); a non-finite or singular matrix is
    ``RSM_UB_INVALID``, a SPEC scan without a ``#G3`` line ``RSM_UB_MISSING``."""
    if override is None:
        return None
    if type(override) is not RSMUBOverride:
        raise TypeError("RSM UB override must be exact RSMUBOverride")
    if coordinate_frame is not RSMCoordinateFrame.HKL:
        raise RSMOperationRefused("RSM_UB_OVERRIDE_UNUSED", "a UB applies to an HKL map; Q coordinates use none")
    receipt = None
    if override.matrix is not None:
        values = override.matrix
    else:
        scan = _rsm_spec_scan_key(override.scan)
        try:
            resolved = Path(override.spec_path).expanduser().resolve(strict=True)
            before = _rsm_ub_file_state(resolved)
            digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
            ub = get_UB(str(resolved), scan)
            if _rsm_ub_file_state(resolved) != before:
                raise RSMOperationRefused("RSM_UB_INVALID", "the UB's SPEC file changed while read")
        except KeyError as error:
            if error.args and error.args[0] == "G3":
                raise RSMOperationRefused("RSM_UB_MISSING",
                                          f"SPEC scan {scan} records no UB matrix (#G3)") from error
            raise RSMOperationRefused("RSM_UB_INVALID", f"that SPEC file has no scan {scan}") from error
        except (OSError, ValueError, TypeError) as error:
            if isinstance(error, RSMOperationRefused):
                raise
            raise RSMOperationRefused("RSM_UB_INVALID", f"the UB's SPEC scan is unreadable: {error}") from error
        values = tuple(float(value) for value in np.asarray(ub, dtype=np.float64).ravel())
        receipt = (resolved.name, str(resolved), digest, before, scan)
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (9,) or not np.all(np.isfinite(array)):
        raise RSMOperationRefused("RSM_UB_INVALID", "a UB matrix is nine finite numbers")
    _rsm_coordinate_matrix(RSMCoordinateFrame.HKL, array.reshape(3, 3), invalid_ub_code="RSM_UB_INVALID")
    return RSMBoundUB(override, tuple(float(value) for value in array), receipt)


def _revalidate_rsm_ub(bound) -> None:
    """At the run: a SPEC file the UB came from is unchanged since Preview."""
    if bound is None or bound.spec_receipt is None:
        return
    path = Path(bound.spec_receipt[1])
    try:
        unchanged = (_rsm_ub_file_state(path) == bound.spec_receipt[3]
                     and hashlib.sha256(path.read_bytes()).hexdigest() == bound.spec_receipt[2])
    except OSError:
        unchanged = False
    if not unchanged:
        raise RSMOperationRefused("SOURCE_IDENTITY_MISMATCH", "the UB's SPEC file changed since Preview")


@dataclass(frozen=True, slots=True)
class RSMImageConditioning:
    """Image-domain conditioning applied before per-frame normalization."""

    additive_offset: float
    high_threshold: float | None
    static_hot_threshold: float | None

    def __post_init__(self) -> None:
        offset = _finite_float(self.additive_offset, "RSM additive offset")
        if offset < 0:
            raise ValueError("RSM additive offset must be non-negative")
        high = (
            None
            if self.high_threshold is None
            else _finite_float(
                self.high_threshold,
                "RSM high threshold",
                positive=True,
            )
        )
        hot = (
            None
            if self.static_hot_threshold is None
            else _finite_float(
                self.static_hot_threshold,
                "RSM static-hot threshold",
                positive=True,
            )
        )
        object.__setattr__(self, "additive_offset", offset)
        object.__setattr__(self, "high_threshold", high)
        object.__setattr__(self, "static_hot_threshold", hot)

    def _canonical_value(self) -> tuple[object, ...]:
        return (
            self.additive_offset,
            self.high_threshold,
            self.static_hot_threshold,
            "exact-all-frames-static-hot-v1",
        )


@dataclass(frozen=True, slots=True)
class RSMDetectorGeometry:
    """The single canonical psic detector/angle description for R1."""

    header: DetectorHeader
    motor_selectors: tuple[tuple[str, MetadataColumnSelector], ...]
    image_orientation: ImageOrientation = field(default_factory=ImageOrientation)
    roi: tuple[int, int, int, int] | None = None
    preset: str = "psic"

    def __post_init__(self) -> None:
        if type(self.header) is not DetectorHeader:
            raise TypeError("RSM detector header must be exact DetectorHeader")
        header = self.header
        for name in ("cch1", "cch2"):
            _finite_float(getattr(header, name), f"detector {name}")
        for name in ("pwidth1", "pwidth2", "distance"):
            _finite_float(
                getattr(header, name),
                f"detector {name}",
                positive=True,
            )
        if (
            type(header.Nch1) is not int
            or type(header.Nch2) is not int
            or header.Nch1 < 2
            or header.Nch2 < 2
        ):
            raise ValueError("RSM detector dimensions must be exact and at least 2")
        if type(self.motor_selectors) is not tuple or len(self.motor_selectors) != 6:
            raise TypeError("RSM psic motor selectors must be one exact six-tuple")
        roles: list[str] = []
        selectors: list[MetadataColumnSelector] = []
        for item in self.motor_selectors:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not MetadataColumnSelector
            ):
                raise TypeError("RSM psic motor mapping entries are invalid")
            roles.append(item[0])
            selectors.append(item[1])
        if tuple(roles) != _PSIC_ROLES:
            raise ValueError(
                "RSM psic motor roles must be ordered mu, eta, chi, phi, nu, del"
            )
        if len({(item.name, item.occurrence) for item in selectors}) != 6:
            raise ValueError("RSM psic motor selectors must be unique")
        if type(self.image_orientation) is not ImageOrientation:
            raise TypeError("RSM image orientation must be exact ImageOrientation")
        if any(
            type(value) is not bool
            for value in (
                self.image_orientation.flip_vertical,
                self.image_orientation.flip_horizontal,
                self.image_orientation.transpose,
            )
        ):
            raise TypeError("RSM image orientation flags must be exact booleans")
        if not self.image_orientation.is_identity:
            raise ValueError("R1 RSM supports only the notebook identity orientation")
        if self.preset != "psic":
            raise ValueError("R1 RSM supports only the canonical psic preset")
        object.__setattr__(self, "roi", _canonical_roi(self.roi, header))

    @property
    def cropped_shape(self) -> tuple[int, int]:
        header = self.header if self.roi is None else self.header.with_roi(self.roi)
        return header.Nch1, header.Nch2

    def _canonical_value(self) -> tuple[object, ...]:
        return (
            self.preset,
            (
                self.header.cch1,
                self.header.cch2,
                self.header.pwidth1,
                self.header.pwidth2,
                self.header.distance,
                self.header.Nch1,
                self.header.Nch2,
            ),
            tuple(
                (role, selector.name, selector.occurrence)
                for role, selector in self.motor_selectors
            ),
            (
                self.image_orientation.rotation,
                self.image_orientation.flip_vertical,
                self.image_orientation.flip_horizontal,
                self.image_orientation.transpose,
            ),
            self.roi,
        )


@dataclass(frozen=True, slots=True)
class RSMOperationPlan:
    """Frozen R1 scientific intent; source facts are captured by preparation."""

    geometry: RSMDetectorGeometry
    conditioning: RSMImageConditioning
    normalization: RSMNormalizationPolicy
    bins: tuple[int, int, int] = (200, 200, 200)
    chunk_size: int = 8
    max_frame_bytes: int = 256 * 1024 * 1024
    max_chunk_bytes: int = 256 * 1024 * 1024
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.geometry) is not RSMDetectorGeometry
            or type(self.conditioning) is not RSMImageConditioning
            or type(self.normalization) is not RSMNormalizationPolicy
        ):
            raise TypeError("RSM operation plan requires exact typed policies")
        motor_keys = {
            (selector.name, selector.occurrence)
            for _role, selector in self.geometry.motor_selectors
        }
        normalization_keys = {
            (selector.name, selector.occurrence)
            for selector in (
                self.normalization.foil_selector,
                self.normalization.exposure_selector,
            )
            if selector is not None
        }
        if motor_keys & normalization_keys:
            raise ValueError(
                "RSM normalization selectors must be disjoint from motor selectors"
            )
        if (
            type(self.bins) is not tuple
            or len(self.bins) != 3
            or any(type(value) is not int or value < 2 for value in self.bins)
        ):
            raise TypeError("RSM bins must be one exact positive integer triple")
        if math.prod(self.bins) > _MAX_RSM_VOXELS:
            raise ValueError("RSM grid exceeds the 8,000,000 voxel R1 bound")
        if (
            type(self.chunk_size) is not int
            or not 1 <= self.chunk_size <= _MAX_CHUNK_SIZE
        ):
            raise ValueError("RSM chunk size is outside the R1 bound")
        if (
            type(self.max_frame_bytes) is not int
            or not 1 <= self.max_frame_bytes <= _MAX_FRAME_BYTES
        ):
            raise ValueError("RSM frame byte limit is outside the R1 bound")
        if (
            type(self.max_chunk_bytes) is not int
            or not 1 <= self.max_chunk_bytes <= _MAX_CHUNK_BYTES
        ):
            raise ValueError("RSM chunk byte limit is outside the R1 bound")
        object.__setattr__(
            self,
            "fingerprint",
            module_plan_fingerprint(ModuleKind.RSM, self._canonical_value()),
        )

    def _canonical_value(self) -> tuple[object, ...]:
        return (
            "rsm-operation-plan-v1",
            self.geometry._canonical_value(),
            self.conditioning._canonical_value(),
            self.normalization._canonical_value(),
            self.bins,
            self.chunk_size,
            self.max_frame_bytes,
            self.max_chunk_bytes,
            "exact-full-detector-q-bounds-v1",
            "numerator-first-normalization-v1",
            "bounded-chunk-working-set-v1",
        )


def required_rsm_selectors(
    plan: RSMOperationPlan,
) -> tuple[MetadataColumnSelector, ...]:
    """Return the exact metadata projection required to execute ``plan``."""

    if type(plan) is not RSMOperationPlan:
        raise TypeError("RSM selector projection requires exact plan")
    requested = [selector for _role, selector in plan.geometry.motor_selectors]
    if plan.normalization.foil_selector is not None:
        requested.append(plan.normalization.foil_selector)
    if plan.normalization.exposure_selector is not None:
        requested.append(plan.normalization.exposure_selector)
    unique = {
        (selector.name, selector.occurrence): selector for selector in requested
    }
    return tuple(
        sorted(unique.values(), key=lambda item: (item.name, item.occurrence))
    )


def _file_state(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_mode,
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        identity_ctime_ns(value.st_ctime_ns),
    )


def _revision(value: object, name: str) -> tuple[int, int, int, int, int, int]:
    if (
        type(value) is not tuple
        or len(value) != 6
        or any(type(item) is not int for item in value)
    ):
        raise TypeError(f"{name} must be one exact file revision")
    return value


def _relative_locator(
    path: str | Path,
    project_root: Path,
    name: str,
    *,
    must_exist: bool = True,
) -> str:
    try:
        resolved = Path(path).expanduser().resolve(strict=must_exist)
        relative = resolved.relative_to(project_root)
    except (OSError, ValueError) as error:
        raise RSMOperationRefused(
            "INPUT_OUTSIDE_PROJECT",
            f"{name} must resolve inside the selected Project root",
        ) from error
    value = relative.as_posix()
    if not value or value == "." or len(value.encode("utf-8")) > 4096:
        raise RSMOperationRefused(
            "INPUT_LOCATOR_INVALID",
            f"{name} has no bounded Project-relative locator",
        )
    return value


def _project_root(value: str | Path) -> Path:
    try:
        root = Path(value).expanduser().resolve(strict=True)
    except (OSError, TypeError) as error:
        raise RSMOperationRefused(
            "PROJECT_UNAVAILABLE", "selected Project root is unavailable"
        ) from error
    if not root.is_dir():
        raise RSMOperationRefused(
            "PROJECT_UNAVAILABLE", "selected Project root is not a directory"
        )
    return root


def _bind_project_output(
    output: ModuleOutputRequest,
    project_root: str | Path,
) -> ModuleOutputRequest:
    """Bind lexical output intent to one canonical path inside Project."""

    if type(output) is not ModuleOutputRequest:
        raise TypeError("RSM output binding requires exact ModuleOutputRequest")
    root = _project_root(project_root)
    lexical = Path(output.target)
    if lexical.is_symlink():
        raise RSMOperationRefused(
            "OUTPUT_SYMLINK_UNSUPPORTED",
            "RSM output target must not be a symbolic link",
        )
    try:
        resolved = lexical.resolve(strict=False)
        relative = resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise RSMOperationRefused(
            "OUTPUT_OUTSIDE_PROJECT",
            "RSM output must resolve inside the selected Project root",
        ) from error
    if (
        not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or len(relative.as_posix().encode("utf-8")) > 4096
    ):
        raise RSMOperationRefused(
            "OUTPUT_PATH_INVALID", "RSM output has no bounded Project path"
        )
    return ModuleOutputRequest(
        resolved, output.kind, output.overwrite, output.artifact_family
    )


@dataclass(frozen=True, slots=True)
class RSMOutputAuthorityReceipt:
    """Preview-time identity of the canonical directory owning RSM output.

    ``created_relative_path`` (RSM v2 only) is set when the output's folder did
    not exist at Preview: the identity is then the deepest existing directory on
    its path, and the run creates the folders below it (as Stitch's run creates
    its output folder, AUD-9 stage 2), so Preview creates nothing.
    """

    parent_relative_path: str
    parent_identity: tuple[int, int, int]
    created_relative_path: str | None = None

    def __post_init__(self) -> None:
        relative = self.parent_relative_path
        if relative != ".":
            relative = _manifest_relative_path(
                relative,
                "RSM output parent path",
            )
        created = self.created_relative_path
        if created is not None:
            created = _manifest_relative_path(created, "RSM output folder path")
        if (
            type(self.parent_relative_path) is not str
            or relative != self.parent_relative_path
            or created != self.created_relative_path
            or type(self.parent_identity) is not tuple
            or len(self.parent_identity) != 3
            or any(type(item) is not int for item in self.parent_identity)
            or not stat.S_ISDIR(self.parent_identity[0])
        ):
            raise TypeError("RSM output authority receipt is invalid")

    def to_provenance(self) -> dict[str, object]:
        mode, device, inode = self.parent_identity
        value: dict[str, object] = {
            "parent_relative_path": self.parent_relative_path,
            "parent_identity": {
                "mode": mode,
                "device": device,
                "inode": inode,
            },
        }
        if self.created_relative_path is not None:
            value["created_relative_path"] = self.created_relative_path
        return value


def _capture_output_authority(
    output: ModuleOutputRequest,
    project_root: str | Path,
    *,
    allow_absent_parent: bool = False,
) -> RSMOutputAuthorityReceipt:
    root = _project_root(project_root)
    parent = Path(output.target).parent
    authority, created = parent, ()
    if allow_absent_parent:  # the deepest existing directory on the folder's path
        while authority != root and not os.path.lexists(authority):
            authority, created = authority.parent, (authority.name, *created)
    try:
        relative = authority.relative_to(root)
        observed = os.stat(authority, follow_symlinks=False)
    except (OSError, ValueError) as error:
        raise RSMOperationRefused(
            "OUTPUT_PARENT_UNAVAILABLE",
            "RSM output parent directory is unavailable inside Project",
        ) from error
    if not stat.S_ISDIR(observed.st_mode):
        raise RSMOperationRefused(
            "OUTPUT_PARENT_UNAVAILABLE",
            "RSM output parent must be one ordinary directory",
        )
    relative_path = "." if not relative.parts else relative.as_posix()
    return RSMOutputAuthorityReceipt(
        relative_path,
        (int(observed.st_mode), int(observed.st_dev), int(observed.st_ino)),
        "/".join(created) if created else None,
    )


def _created_output_folders(
    output: ModuleOutputRequest,
    root: Path,
    authority: RSMOutputAuthorityReceipt,
) -> tuple[Path, ...]:
    """The folders below a Preview-time authority that the run creates, from the
    outermost; the authority must be the directory Preview qualified."""

    anchor = (
        root
        if authority.parent_relative_path == "."
        else root / authority.parent_relative_path
    )
    folders = []
    for part in Path(authority.created_relative_path).parts:
        folders.append((folders[-1] if folders else anchor) / part)
    observed = os.stat(anchor, follow_symlinks=False)
    if (
        folders[-1] != Path(output.target).parent
        or not stat.S_ISDIR(observed.st_mode)
        or (int(observed.st_mode), int(observed.st_dev), int(observed.st_ino))
        != authority.parent_identity
    ):
        raise RSMOperationRefused(
            "OUTPUT_IDENTITY_MISMATCH",
            "RSM output path authority changed after Preview",
        )
    return tuple(folders)


def _create_rsm_v2_output_folder(request: RSMOperationRequestV2) -> None:
    """Make the output's folder Preview qualified without creating it (at the
    start of a run, as Stitch's run makes its output folder): each folder below
    the unchanged authority is made, or must already be, one ordinary directory."""

    authority = request.output_authority
    if authority.created_relative_path is None:
        return
    root = _project_root(request.preflight.project_root)
    try:
        folders = _created_output_folders(request.module.output, root, authority)
    except OSError as error:
        raise RSMOperationRefused(
            "OUTPUT_IDENTITY_MISMATCH",
            "RSM output path authority changed after Preview",
        ) from error
    for folder in folders:
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass
        except OSError as error:
            raise RSMOperationRefused(
                "OUTPUT_PARENT_UNAVAILABLE",
                f"RSM output folder could not be made: {error}",
            ) from error
        try:
            ordinary = stat.S_ISDIR(os.lstat(folder).st_mode)
        except OSError:
            ordinary = False
        if not ordinary:
            raise RSMOperationRefused(
                "OUTPUT_IDENTITY_MISMATCH",
                "RSM output path authority changed after Preview",
            )


def _requalify_project_output(
    output: ModuleOutputRequest,
    project_root: str | Path,
    authority: RSMOutputAuthorityReceipt,
    *,
    folders_made: bool = True,
) -> None:
    """Refuse completed namespace changes under the trusted-Project boundary.

    This is deliberately not a descriptor-bound publication claim.  Concurrent
    mutation by an untrusted actor in a shared-writable Project remains held for
    the common Stitch/RSM output namespace owner.

    Below an authority captured without the output's folder, each folder must be
    one ordinary directory once the run has made them (``folders_made``), and
    absent or one before.
    """

    try:
        current = _bind_project_output(output, project_root)
        if (
            type(authority) is RSMOutputAuthorityReceipt
            and authority.created_relative_path is not None
        ):
            for folder in _created_output_folders(
                current, _project_root(project_root), authority
            ):
                if not folders_made and not os.path.lexists(folder):
                    break
                if not stat.S_ISDIR(os.lstat(folder).st_mode):
                    raise RSMOperationRefused("OUTPUT_IDENTITY_MISMATCH")
            current_authority = authority
        else:
            current_authority = _capture_output_authority(current, project_root)
    except (RSMOperationRefused, OSError) as error:
        raise RSMOperationRefused(
            "OUTPUT_IDENTITY_MISMATCH",
            "RSM output path authority changed after Preview",
        ) from error
    if (
        type(authority) is not RSMOutputAuthorityReceipt
        or current.fingerprint != output.fingerprint
        or current_authority != authority
    ):
        raise RSMOperationRefused(
            "OUTPUT_IDENTITY_MISMATCH",
            "RSM output path authority changed after Preview",
        )


def _manifest_relative_path(value: object, name: str) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > 4096:
        raise TypeError(f"{name} must be a bounded relative path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a normalized relative path")
    return path.as_posix()


def _portable_json_value(value: object, *, depth: int = 0) -> object:
    if depth > _MAX_JSON_DEPTH:
        raise RSMOperationRefused(
            "SOURCE_SPEC_INVALID",
            "source options exceed the nesting bound",
        )
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str or not key or len(key) > 4096:
                raise RSMOperationRefused(
                    "SOURCE_SPEC_INVALID",
                    "source option key is invalid",
                )
            result[key] = _portable_json_value(item, depth=depth + 1)
        return result
    if type(value) in {tuple, list}:
        return [_portable_json_value(item, depth=depth + 1) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _portable_json_value(value.value, depth=depth + 1)
    if isinstance(value, np.generic):
        return _portable_json_value(value.item(), depth=depth + 1)
    if type(value) in {str, bool, int} or value is None:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise RSMOperationRefused(
        "SOURCE_SPEC_INVALID",
        "source options contain an unsupported value",
    )


@dataclass(frozen=True, slots=True)
class RSMManifestFile:
    relative_path: str
    revision: tuple[int, int, int, int, int, int] | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _manifest_relative_path(self.relative_path, "RSM manifest file path"),
        )
        if self.revision is not None:
            _revision(self.revision, "RSM manifest file revision")


@dataclass(frozen=True, slots=True)
class RSMContribution:
    label: int
    file_ordinal: int
    source_frame_index: int
    values: tuple[tuple[str, int, float], ...]
    normalization_divisor: float

    def __post_init__(self) -> None:
        if (
            type(self.label) is not int
            or type(self.file_ordinal) is not int
            or self.file_ordinal < 0
            or type(self.source_frame_index) is not int
            or self.source_frame_index < 0
            or type(self.values) is not tuple
        ):
            raise TypeError("RSM contribution is invalid")
        parsed: list[tuple[str, int, float]] = []
        for item in self.values:
            if (
                type(item) is not tuple
                or len(item) != 3
                or type(item[0]) is not str
                or not item[0]
                or item[0].strip() != item[0]
                or type(item[1]) is not int
                or item[1] < 0
                or type(item[2]) is not float
                or not math.isfinite(item[2])
            ):
                raise TypeError("RSM contribution metadata value is invalid")
            parsed.append(item)
        if tuple(parsed) != tuple(
            sorted(parsed, key=lambda item: (item[0], item[1]))
        ):
            raise ValueError("RSM contribution metadata values must be sorted")
        if len({(name, occurrence) for name, occurrence, _value in parsed}) != len(
            parsed
        ):
            raise ValueError("RSM contribution metadata values must be unique")
        if type(self.normalization_divisor) is not float:
            raise TypeError("RSM normalization divisor must be an exact float")
        _finite_float(
            self.normalization_divisor,
            "RSM normalization divisor",
            positive=True,
        )


@dataclass(eq=False, frozen=True, slots=True)
class RSMPreflightReceipt:
    """Factory-owned, detector-image-free scientific preflight receipt."""

    project_root: str
    source_relative_path: str
    source_scan: str
    source_options_json: str = field(repr=False)
    primary_revision: tuple[int, int, int, int, int, int]
    files: tuple[RSMManifestFile, ...]
    contributions: tuple[RSMContribution, ...]
    energy_eV: float
    ub: tuple[tuple[float, float, float], ...]
    q_bounds: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ]
    detector_shape: tuple[int, int]
    cropped_shape: tuple[int, int]
    source_fingerprint: str
    module_source_fingerprint: str
    table_fingerprint: str
    plan_fingerprint: str
    fingerprint: str = field(init=False)
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        try:
            root = Path(self.project_root)
            options = json.loads(self.source_options_json)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise TypeError("RSM preflight receipt is invalid") from error
        if (
            _claim is not _REQUEST_FACTORY
            or type(self.project_root) is not str
            or not root.is_absolute()
            or type(self.source_scan) is not str
            or not self.source_scan
            or type(options) is not dict
            or type(self.files) is not tuple
            or not self.files
            or len(self.files) > _MAX_MANIFEST_FILES
            or any(type(item) is not RSMManifestFile for item in self.files)
            or len({item.relative_path for item in self.files}) != len(self.files)
            or type(self.contributions) is not tuple
            or not self.contributions
            or len(self.contributions) > _MAX_CONTRIBUTIONS
            or any(type(item) is not RSMContribution for item in self.contributions)
            or any(item.file_ordinal >= len(self.files) for item in self.contributions)
            or len({item.label for item in self.contributions})
            != len(self.contributions)
        ):
            raise TypeError("RSM preflight receipt is invalid")
        canonical_options = json.dumps(
            options,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        if canonical_options != self.source_options_json:
            raise ValueError("RSM preflight source options are not canonical")
        object.__setattr__(
            self,
            "source_relative_path",
            _manifest_relative_path(self.source_relative_path, "RSM source path"),
        )
        _revision(self.primary_revision, "RSM source primary revision")
        energy = _finite_float(self.energy_eV, "RSM energy", positive=True)
        if (
            type(self.ub) is not tuple
            or len(self.ub) != 3
            or any(type(row) is not tuple or len(row) != 3 for row in self.ub)
            or any(
                type(value) is not float or not math.isfinite(value)
                for row in self.ub
                for value in row
            )
        ):
            raise TypeError("RSM preflight UB must be an exact finite 3 by 3 tuple")
        if (
            type(self.q_bounds) is not tuple
            or len(self.q_bounds) != 3
            or any(
                type(item) is not tuple
                or len(item) != 2
                or any(type(value) is not float for value in item)
                or not all(math.isfinite(value) for value in item)
                or not item[1] > item[0]
                for item in self.q_bounds
            )
        ):
            raise TypeError("RSM preflight q bounds are invalid")
        for shape, name in (
            (self.detector_shape, "detector shape"),
            (self.cropped_shape, "cropped detector shape"),
        ):
            if (
                type(shape) is not tuple
                or len(shape) != 2
                or any(type(value) is not int or value < 2 for value in shape)
            ):
                raise TypeError(f"RSM preflight {name} is invalid")
        for digest, name in (
            (self.source_fingerprint, "source fingerprint"),
            (self.module_source_fingerprint, "module source fingerprint"),
            (self.table_fingerprint, "table fingerprint"),
            (self.plan_fingerprint, "plan fingerprint"),
        ):
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise TypeError(f"RSM {name} must be lowercase SHA-256")
        object.__setattr__(self, "energy_eV", energy)
        object.__setattr__(
            self,
            "fingerprint",
            module_plan_fingerprint(ModuleKind.RSM, self._canonical_value()),
        )
        encoded = json.dumps(
            self.to_provenance(),
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        if len(encoded) > _MAX_MANIFEST_BYTES:
            raise RSMOperationRefused(
                "PREFLIGHT_BYTE_LIMIT_EXCEEDED",
                "RSM persisted preflight exceeds 512 KiB",
            )

    def _canonical_value(self) -> tuple[object, ...]:
        return (
            "rsm-preflight-v1",
            self.source_relative_path,
            self.source_scan,
            json.loads(self.source_options_json),
            self.primary_revision,
            tuple((item.relative_path, item.revision) for item in self.files),
            tuple(
                (
                    item.label,
                    item.file_ordinal,
                    item.source_frame_index,
                    item.values,
                    item.normalization_divisor,
                )
                for item in self.contributions
            ),
            self.energy_eV,
            self.ub,
            self.q_bounds,
            self.detector_shape,
            self.cropped_shape,
            self.source_fingerprint,
            self.module_source_fingerprint,
            self.table_fingerprint,
            self.plan_fingerprint,
        )

    def to_provenance(self) -> dict[str, object]:
        return {
            "schema_version": "rsm-preflight-v1",
            "source": {
                "relative_path": self.source_relative_path,
                "kind": "spec",
                "scan": self.source_scan,
                "options": json.loads(self.source_options_json),
                "primary_revision": list(self.primary_revision),
                "source_fingerprint": self.source_fingerprint,
                "module_source_fingerprint": self.module_source_fingerprint,
                "table_fingerprint": self.table_fingerprint,
            },
            "files": [
                {
                    "relative_path": item.relative_path,
                    "revision": (
                        None if item.revision is None else list(item.revision)
                    ),
                }
                for item in self.files
            ],
            "contributions": [
                {
                    "label": item.label,
                    "file_ordinal": item.file_ordinal,
                    "source_frame_index": item.source_frame_index,
                    "values": [
                        {
                            "name": name,
                            "occurrence": occurrence,
                            "value": value,
                        }
                        for name, occurrence, value in item.values
                    ],
                    "normalization_divisor": item.normalization_divisor,
                }
                for item in self.contributions
            ],
            "energy_eV": self.energy_eV,
            "ub": [list(row) for row in self.ub],
            "q_bounds": [list(item) for item in self.q_bounds],
            "detector_shape": list(self.detector_shape),
            "cropped_shape": list(self.cropped_shape),
            "plan_fingerprint": self.plan_fingerprint,
            "fingerprint": self.fingerprint,
        }

    def __copy__(self):
        raise TypeError("RSM preflight receipt is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("RSM preflight receipt is not copyable")

    def __reduce__(self):
        raise TypeError("RSM preflight receipt is not serializable")

    def __reduce_ex__(self, _protocol):
        raise TypeError("RSM preflight receipt is not serializable")


def _fresh_table(
    source: ModuleSourceReceipt,
    *,
    cancel_token: threading.Event | None = None,
):
    table = run_metadata_table(
        MetadataTablePlan(source.analysis.source_spec),
        cancel_token=cancel_token,
    )
    if table.disposition is AnalysisDisposition.CANCELLED:
        raise RSMOperationRefused("CANCELLED")
    if (
        table.disposition is not AnalysisDisposition.COMPLETED
        or table.receipt != source.analysis
        or table.table_fingerprint != source.table_fingerprint
        or table.labels != source.analysis.labels
    ):
        raise RSMOperationRefused(
            "SOURCE_IDENTITY_MISMATCH",
            "source metadata no longer matches its receipt",
        )
    return table


def _project_source_options(
    analysis,
    root: Path,
    selectors: tuple[MetadataColumnSelector, ...],
) -> str:
    """Return the closed, Project-relative R1 SPEC source projection."""

    if analysis.source_spec.metadata_uri is not None:
        raise RSMOperationRefused(
            "SOURCE_SPEC_UNSUPPORTED",
            "R1 RSM does not accept SourceSpec.metadata_uri",
        )
    if analysis.resolved_kind in _RSM_NEXUS_KINDS:
        # a raw NeXus stack (AUD-9 stage 2b C5c): one exact entry, no scan and no reader overrides, as Stitch
        # admits it; its frames, energy, motors and UB come from the entry, the UB read in SPEC's #G3
        # convention (2π included), which xdart's own NeXus writer stores there
        if (
            analysis.source_spec.options
            or not analysis.resolved_entry
            or analysis.resolved_scan
        ):
            raise RSMOperationRefused(
                "SOURCE_SPEC_UNSUPPORTED",
                "a NeXus member needs one exact entry and no reader overrides",
            )
        return json.dumps(
            {"entry": analysis.resolved_entry, "kind": analysis.resolved_kind.value,
             "ub_convention": "spec-g3"},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
    if analysis.resolved_kind is not SourceKind.SPEC:
        raise RSMOperationRefused("SOURCE_KIND_UNSUPPORTED")
    options = dict(analysis.source_spec.options)
    allowed = {
        "scan",
        "image_dir",
        "image_stem",
        "read_image_kwargs",
        "metadata_column_projection",
    }
    if (
        analysis.resolved_entry is not None
        or not analysis.resolved_scan
        or options.get("scan") != analysis.resolved_scan
        or not options.get("image_dir")
    ):
        raise RSMOperationRefused(
            "SOURCE_SPEC_UNSUPPORTED",
            "R1 RSM requires one exact SPEC scan and image directory",
        )
    unexpected = set(options) - allowed
    if unexpected:
        raise RSMOperationRefused(
            "SOURCE_SPEC_UNSUPPORTED",
            f"R1 SPEC source has unsupported options {sorted(unexpected)}",
        )
    projection = options.get("metadata_column_projection")
    expected_projection = tuple(
        (selector.name, selector.occurrence) for selector in selectors
    )
    if type(projection) is not tuple or tuple(projection) != expected_projection:
        raise RSMOperationRefused(
            "SOURCE_PROJECTION_MISMATCH",
            "R1 source projection must exactly match the RSM selectors",
        )
    projected = dict(options)
    projected["image_dir"] = _relative_locator(
        options["image_dir"], root, "SPEC image directory"
    )
    read_options = options.get("read_image_kwargs", {})
    if type(read_options) is not dict:
        read_options = dict(read_options)
    allowed_read = {
        "detector_shape",
        "detector",
        "raw_dtype",
        "raw_header_skip",
        "threshold",
        "rotation",
    }
    unexpected_read = set(read_options) - allowed_read
    if unexpected_read:
        raise RSMOperationRefused(
            "SOURCE_SPEC_UNSUPPORTED",
            "R1 SPEC image reader has unsupported options "
            f"{sorted(unexpected_read)}",
        )
    if read_options.get("rotation", 0) != 0:
        raise RSMOperationRefused(
            "SOURCE_ORIENTATION_CONFLICT",
            "source rotation must be zero; RSM owns image orientation",
        )
    if read_options.get("threshold") is not None:
        raise RSMOperationRefused(
            "SOURCE_CONDITIONING_CONFLICT",
            "source thresholding must be disabled; RSM owns conditioning",
        )
    projected["read_image_kwargs"] = read_options
    portable = _portable_json_value(projected)
    return json.dumps(
        portable,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _rsm_nexus_energy_and_ub(analysis, coordinate_frame, ub_override):
    """A raw NeXus member's energy (eV) and UB, from its entry through xdart's NeXus reader (AUD-9 stage 2b
    C5c): the energy in keV there; the UB from ``sample/ub_matrix`` or ``orientation_matrix``, read in SPEC's
    #G3 convention (2π included, as xdart's writer stores it), else a UB given at Preview, else, in HKL,
    ``RSM_UB_MISSING``. Q coordinates need no UB."""
    from xrd_tools.io.nexus import read_nexus
    metadata = read_nexus(analysis.resolved_root, entry=analysis.resolved_entry)
    energy = float(metadata.energy) * 1000.0
    if coordinate_frame is not RSMCoordinateFrame.HKL:
        return energy, None
    if ub_override is not None:
        return energy, ub_override.array()
    if metadata.ub_matrix is None:
        raise RSMOperationRefused(
            "RSM_UB_MISSING",
            "this NeXus entry records no UB matrix (sample/ub_matrix or orientation_matrix); enter one, take it "
            "from a SPEC scan, or choose Q coordinates",
        )
    return energy, np.asarray(metadata.ub_matrix, dtype=np.float64)


def _numeric_lookup(
    table,
    selectors: tuple[MetadataColumnSelector, ...],
) -> tuple[dict[int, int], tuple[tuple[MetadataColumnSelector, np.ndarray], ...]]:
    positions: dict[int, int] = {}
    for index, label in enumerate(table.labels):
        if type(label) is not int or label in positions:
            raise RSMOperationRefused(
                "SOURCE_IDENTITY_MISMATCH",
                "metadata labels are not exact and unique",
            )
        positions[label] = index
    columns: dict[str, list[object]] = {}
    for column in table.columns:
        columns.setdefault(column.name, []).append(column)
    selected: list[tuple[MetadataColumnSelector, np.ndarray]] = []
    for selector in selectors:
        matches = columns.get(selector.name, ())
        if (
            selector.occurrence >= len(matches)
            or matches[selector.occurrence].numeric is None
        ):
            raise RSMOperationRefused(
                "INVALID_METADATA_SELECTOR",
                f"metadata column {selector.name!r} occurrence "
                f"{selector.occurrence} is not numeric",
            )
        values = matches[selector.occurrence].numeric
        if values.shape != (len(positions),):
            raise RSMOperationRefused(
                "SOURCE_IDENTITY_MISMATCH",
                "metadata column length changed",
            )
        selected.append((selector, values))
    return positions, tuple(selected)


@dataclass(frozen=True, slots=True)
class _RSMCapturedMemberFacts:
    project_root: str
    source_relative_path: str
    source_scan: str
    source_options_json: str
    primary_revision: tuple[int, int, int, int, int, int]
    files: tuple[RSMManifestFile, ...]
    contributions: tuple[RSMContribution, ...]
    coordinate_frame: RSMCoordinateFrame
    energy_eV: float
    ub: tuple[tuple[float, float, float], ...]
    q_bounds: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ]
    detector_shape: tuple[int, int]
    cropped_shape: tuple[int, int]
    source_fingerprint: str
    module_source_fingerprint: str
    table_fingerprint: str
    #: where the UB came from, for Preview only (decision 2): never part of an identity
    ub_origin: str | None = None


_STANDARD_PSIC_AXES = ("x+", "z-", "y+", "z-", "x+", "z-")  # mu, eta, chi, phi, nu, del


def _rsm_reversed_circles(mapper: PixelQMap) -> tuple[str, ...]:
    """The psic circles a map turns other than psic's standard way (by role)."""
    from xrd_tools.rsm.orientation import ROLE_ORDER
    axes = tuple(mapper.diff_config.sample_circles) + tuple(mapper.diff_config.detector_circles)
    if len(axes) != len(_STANDARD_PSIC_AXES):
        return ()
    return tuple(role for role, axis, plain in zip(ROLE_ORDER, axes, _STANDARD_PSIC_AXES) if axis != plain)


def _rsm_orientation_ub(analysis, g3, mapper: PixelQMap, runtime_session) -> tuple[np.ndarray, str]:
    """A SPEC member's UB in the model RSM maps with (AUD-9 stage 2b decision 2; the maintainer: "Xu provides a
    way to specify the actual circle directions on the instrument. Shouldn't that be used?"). SPEC stores no circle
    directions, and ``#G3`` was computed in its own psic convention from the header's two orientation reflections
    (``#G1``); the UB is recomputed from them in the map's directions and zero offsets.

    ``#G3`` is the cross-check, twice. The header's reflections must reproduce ``#G3`` in psic's standard model to
    its rounding, or they cannot be trusted in another: psic's standard model then keeps ``#G3`` (with a warning),
    any other refuses. And a recompute within the rounding of ``#G3`` is ``#G3`` itself, so its bytes and every
    identity are unchanged. A model other than psic's standard needs the reflections
    (``RSM_ORIENTATION_REFLECTIONS_MISSING``)."""
    from dataclasses import replace

    from xrd_tools.rsm.orientation import ROLE_ORDER, ROUNDING, relative_difference, spec_orientation

    g3 = np.asarray(g3, dtype=np.float64)
    reversed_roles = _rsm_reversed_circles(mapper)
    plain = not reversed_roles and not any(mapper.angle_offsets)
    try:
        orientation = spec_orientation(
            get_orientation_header(analysis.resolved_root, analysis.resolved_scan))
    except KeyError:
        orientation = None
    if orientation is None:
        if plain:
            return g3, "#G3 (the header holds no orientation reflections to recompute it from)"
        raise RSMOperationRefused(
            "RSM_ORIENTATION_REFLECTIONS_MISSING",
            "this scan's header holds no orientation reflections (#G1) to recompute its UB in the instrument's "
            "directions; give a UB, or map in Q coordinates",
        )
    named = " and ".join(
        "(" + " ".join(f"{value:g}" for value in item.hkl) + ")" for item in orientation.reflections)
    standard = mapper if plain else replace(
        mapper,
        diff_config=replace(
            mapper.diff_config,
            sample_circles=_STANDARD_PSIC_AXES[:4],
            detector_circles=_STANDARD_PSIC_AXES[4:],
        ),
        angle_offsets=(),
    )
    self_check = relative_difference(standard.orientation_ub(orientation, runtime_session=runtime_session), g3)
    if self_check > ROUNDING:
        if plain:
            return g3, (f"#G3, kept; ⚠ its own reflections {named} give it only to {self_check:.3g} (relative) in "
                        "psic's standard directions, so they are not a psic #G1 this RSM can check it by")
        raise RSMOperationRefused(
            "RSM_ORIENTATION_REFLECTIONS_MISSING",
            f"this scan's reflections {named} do not reproduce its own #G3 in psic's standard directions "
            f"({self_check:.3g}, relative), so its UB cannot be recomputed in the instrument's; give a UB",
        )
    if plain:
        return g3, f"#G3; recomputed from {named} in the instrument's directions it agrees to {self_check:.2g}"
    recomputed = mapper.orientation_ub(orientation, runtime_session=runtime_session)
    difference = relative_difference(recomputed, g3)
    if difference <= ROUNDING:
        return g3, f"#G3; recomputed from {named} in the instrument's directions it agrees to {difference:.2g}"
    moved = sorted({role for role in reversed_roles for item in orientation.reflections
                    if abs(math.remainder(item.angles[ROLE_ORDER.index(role)], 180.0)) > 1e-6})
    note = "" if not moved else "; reversed circles away from 0° or 180° there: " + ", ".join(moved)
    return recomputed, (f"recomputed from {named} in the instrument's directions; #G3 differs by "
                        f"{difference:.3g} (relative){note}")


def _capture_preflight_facts(
    source: ModuleSourceReceipt,
    table,
    project_root: str | Path,
    *,
    selectors: tuple[MetadataColumnSelector, ...],
    motor_selectors: tuple[tuple[str, MetadataColumnSelector], ...],
    detector_header: DetectorHeader,
    roi: tuple[int, int, int, int] | None,
    conditioning: RSMImageConditioning,
    normalization: RSMNormalizationPolicy,
    chunk_size: int,
    max_frame_bytes: int,
    max_chunk_bytes: int,
    expected_detector_shape: tuple[int, int] | None = None,
    invalid_energy_code: str | None = None,
    invalid_ub_code: str = "UB_INVALID",
    normalize_source_fact_errors: bool = False,
    normalize_ub_errors: bool = False,
    mapper: PixelQMap | None = None,
    active_runtime_session: object | None = None,
    allow_single_static_hot: bool = False,
    cancel_token: threading.Event | None = None,
    coordinate_frame: RSMCoordinateFrame = RSMCoordinateFrame.HKL,
    ub_override: RSMBoundUB | None = None,
) -> _RSMCapturedMemberFacts:
    if type(coordinate_frame) is not RSMCoordinateFrame:
        raise TypeError("RSM coordinate frame must be exact")
    root = _project_root(project_root)
    analysis = source.analysis
    if (
        conditioning.static_hot_threshold is not None
        and len(source.selected_labels) < 2
        and not allow_single_static_hot
    ):
        raise RSMOperationRefused(
            "STATIC_MASK_REQUIRES_MULTIPLE_FRAMES",
            "exact all-frame static-hot masking requires at least two frames",
        )
    source_relative = _relative_locator(analysis.resolved_root, root, "source")
    source_options_json = _project_source_options(analysis, root, selectors)
    nexus = analysis.resolved_kind in _RSM_NEXUS_KINDS
    if expected_detector_shape is not None and not nexus:
        try:
            read_options = json.loads(source_options_json)["read_image_kwargs"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise RSMOperationRefused("RSM_MEMBER_IDENTITY_MISMATCH") from error
        if (
            type(read_options) is not dict
            or tuple(read_options.get("detector_shape", ()))
            != expected_detector_shape
        ):
            raise RSMOperationRefused(
                "RSM_MEMBER_IDENTITY_MISMATCH",
                "member detector shape differs from effective geometry",
            )
    if analysis.primary_post_state is None or not (
        analysis.resolved_entry if nexus else analysis.resolved_scan
    ):
        raise RSMOperationRefused(
            "SOURCE_IDENTITY_MISMATCH",
            "source has no exact primary revision, scan or entry",
        )
    dependency_rows = (
        (analysis.lexical_root, analysis.resolved_root, analysis.primary_post_state),
        *analysis.dependency_revisions,
    )
    files: list[RSMManifestFile] = []
    file_ordinals: dict[Path, int] = {}
    dependency_by_resolved: dict[
        Path, tuple[int, int, int, int, int, int] | None
    ] = {}
    charge = len(source_options_json.encode("utf-8")) + 2048

    def charge_piece(value: object) -> None:
        nonlocal charge
        charge += len(
            json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        ) + 1
        if charge > _MAX_MANIFEST_BYTES:
            raise RSMOperationRefused(
                "PREFLIGHT_BYTE_LIMIT_EXCEEDED",
                "RSM preflight exceeds 512 KiB while being captured",
            )

    for _lexical, raw_resolved, revision in dependency_rows:
        resolved = Path(raw_resolved)
        previous = dependency_by_resolved.get(resolved, ...)
        if previous is not ...:
            if previous != revision:
                raise RSMOperationRefused(
                    "SOURCE_IDENTITY_MISMATCH",
                    "source dependency revisions disagree",
                )
            continue
        if len(files) >= _MAX_MANIFEST_FILES:
            raise RSMOperationRefused(
                "PREFLIGHT_FILE_LIMIT_EXCEEDED",
                "RSM preflight exceeds 8192 dependency files",
            )
        relative = _relative_locator(
            resolved,
            root,
            "source dependency",
            must_exist=revision is not None,
        )
        item = RSMManifestFile(relative, revision)
        files.append(item)
        file_ordinals[resolved] = len(files) - 1
        dependency_by_resolved[resolved] = revision
        charge_piece(
            {
                "relative_path": item.relative_path,
                "revision": None if revision is None else list(revision),
            }
        )

    label_positions, selected_columns = _numeric_lookup(table, selectors)
    selected_by_key = {
        (selector.name, selector.occurrence): values
        for selector, values in selected_columns
    }
    try:
        rows = np.asarray(
            [label_positions[label] for label in source.selected_labels],
            dtype=np.int64,
        )
    except KeyError as error:
        raise RSMOperationRefused(
            "SOURCE_IDENTITY_MISMATCH",
            "selected RSM label is outside the exact metadata table",
        ) from error
    foil = exposure = None
    if normalization.foil_selector is not None:
        selector = normalization.foil_selector
        foil = selected_by_key[(selector.name, selector.occurrence)][rows]
    if normalization.exposure_selector is not None:
        selector = normalization.exposure_selector
        exposure = selected_by_key[(selector.name, selector.occurrence)][rows]
    divisors = rsm_normalization_divisors(
        normalization,
        len(source.selected_labels),
        foil_status=foil,
        exposure_seconds=exposure,
    )
    angles = tuple(
        np.asarray(
            selected_by_key[(selector.name, selector.occurrence)][rows],
            dtype=np.float64,
        )
        for _role, selector in motor_selectors
    )
    contributions: list[RSMContribution] = []
    try:
        with requalified_analysis_source(
            analysis,
            cancel_token=cancel_token,
        ) as opened:
            if nexus and expected_detector_shape is not None:
                # the open entry's pixel-free descriptor (the lease's cursor source, or the source's own)
                descriptor = getattr(opened, "descriptor", None)
                if descriptor is None and callable(getattr(opened, "container_descriptor", None)):
                    descriptor = opened.container_descriptor()
                frame_shape = None if descriptor is None else descriptor.frame_shape
                if tuple(frame_shape or ()) != tuple(expected_detector_shape):
                    raise RSMOperationRefused(
                        "RSM_MEMBER_IDENTITY_MISMATCH",
                        "member detector shape differs from effective geometry",
                    )
            ub_origin = None
            try:
                if nexus:
                    energy, ub_value = _rsm_nexus_energy_and_ub(
                        analysis, coordinate_frame, ub_override)
                elif coordinate_frame is RSMCoordinateFrame.HKL and ub_override is not None:
                    # a UB given at Preview replaces the source's (AUD-9 stage 2b C5); the energy is the source's
                    energy = get_energy(
                        analysis.resolved_root,
                        analysis.resolved_scan,
                    )
                    ub_value = ub_override.array()
                    ub_origin = "given (typed, or from a named SPEC scan), used as given"
                elif coordinate_frame is RSMCoordinateFrame.HKL:
                    energy, ub_value = get_energy_and_UB(
                        analysis.resolved_root,
                        analysis.resolved_scan,
                    )
                    if mapper is not None:  # decision 2: the UB in the model RSM maps with
                        ub_value, ub_origin = _rsm_orientation_ub(
                            analysis, ub_value, mapper, active_runtime_session)
                else:
                    energy = get_energy(
                        analysis.resolved_root,
                        analysis.resolved_scan,
                    )
                    ub_value = None
            except RSMOperationRefused:  # already named (a NeXus member's RSM_UB_MISSING, C5c)
                raise
            except KeyError as error:
                if normalize_source_fact_errors:
                    missing_ub = (
                        coordinate_frame is RSMCoordinateFrame.HKL
                        and error.args
                        and error.args[0] == "G3"
                    )
                    if missing_ub:  # no #G3 line: give a UB, or map in Q (AUD-9 stage 2b C5)
                        raise RSMOperationRefused(
                            "RSM_UB_MISSING",
                            "this source records no UB matrix (#G3); enter one, take it from a SPEC "
                            "scan, or choose Q coordinates",
                        ) from error
                    raise RSMOperationRefused(
                        invalid_energy_code or "RSM_MEMBER_ENERGY_INVALID"
                    ) from error
                raise
            except ValueError as error:
                if normalize_source_fact_errors:
                    code = (
                        invalid_ub_code
                        if coordinate_frame is RSMCoordinateFrame.HKL
                        else invalid_energy_code
                    )
                    raise RSMOperationRefused(
                        code or "RSM_MEMBER_ENERGY_INVALID"
                    ) from error
                raise
            try:
                energy = _finite_float(energy, "RSM energy", positive=True)
            except (TypeError, ValueError) as error:
                if invalid_energy_code is None:
                    raise
                raise RSMOperationRefused(invalid_energy_code) from error
            if normalize_ub_errors or coordinate_frame is not RSMCoordinateFrame.HKL:
                ub = _rsm_coordinate_matrix(
                    coordinate_frame,
                    ub_value,
                    invalid_ub_code=invalid_ub_code,
                )
            else:
                ub = np.asarray(ub_value, dtype=np.float64)
                if ub.shape != (3, 3) or not np.all(np.isfinite(ub)):
                    raise RSMOperationRefused(invalid_ub_code)
            for contribution_index, label in enumerate(source.selected_labels):
                if cancel_token is not None and cancel_token.is_set():
                    raise RSMOperationRefused("CANCELLED")
                if len(contributions) >= _MAX_CONTRIBUTIONS:
                    raise RSMOperationRefused(
                        "PREFLIGHT_CONTRIBUTION_LIMIT_EXCEEDED",
                        "RSM selection exceeds 4096 contributions",
                    )
                raw_locator = getattr(opened, "raw_locator_for", None)
                if callable(raw_locator):
                    path, frame_index = raw_locator(label)
                else:
                    frame = opened.frame_for(label)
                    path, frame_index = frame.source_path, frame.source_frame_index
                if path is None or type(frame_index) is not int or frame_index < 0:
                    raise RSMOperationRefused(
                        "RAW_LOCATOR_MISSING",
                        f"selected frame {label} has no exact raw source locator",
                    )
                try:
                    resolved = Path(path).expanduser().resolve(strict=True)
                    state = _file_state(os.stat(resolved, follow_symlinks=False))
                except OSError as error:
                    raise RSMOperationRefused(
                        "RAW_SOURCE_UNAVAILABLE",
                        f"selected frame {label} raw source is unavailable",
                    ) from error
                if (
                    not stat.S_ISREG(state[0])
                    or dependency_by_resolved.get(resolved) != state
                ):
                    raise RSMOperationRefused(
                        "SOURCE_IDENTITY_MISMATCH",
                        f"selected frame {label} raw source is outside its receipt",
                    )
                try:
                    ordinal = file_ordinals[resolved]
                    position = label_positions[label]
                except KeyError as error:
                    raise RSMOperationRefused(
                        "SOURCE_IDENTITY_MISMATCH",
                        "selected raw source or label is outside its receipt",
                    ) from error
                values_list: list[tuple[str, int, float]] = []
                for selector, column in selected_columns:
                    value = float(column[position])
                    if not math.isfinite(value):
                        raise RSMOperationRefused(
                            "INVALID_METADATA_VALUE",
                            f"metadata column {selector.name!r} contains a "
                            "non-finite value",
                        )
                    values_list.append((selector.name, selector.occurrence, value))
                contribution = RSMContribution(
                    label,
                    ordinal,
                    frame_index,
                    tuple(sorted(values_list)),
                    float(divisors[contribution_index]),
                )
                charge_piece(
                    {
                        "label": contribution.label,
                        "file_ordinal": contribution.file_ordinal,
                        "source_frame_index": contribution.source_frame_index,
                        "values": contribution.values,
                        "normalization_divisor": contribution.normalization_divisor,
                    }
                )
                contributions.append(contribution)
            selected_mapper = (
                PixelQMap(Diffractometer.psic(), detector_header)
                if mapper is None
                else mapper
            )
            bounds = (
                resolve_exact_rsm_q_bounds(
                    selected_mapper,
                    angles,
                    energy,
                    ub,
                    roi=roi,
                    chunk_size=chunk_size,
                    max_frame_bytes=max_frame_bytes,
                    max_chunk_bytes=max_chunk_bytes,
                    cancel_token=cancel_token,
                )
                if active_runtime_session is None
                else _resolve_exact_rsm_q_bounds_active(
                    selected_mapper,
                    angles,
                    energy,
                    ub,
                    roi=roi,
                    chunk_size=chunk_size,
                    max_frame_bytes=max_frame_bytes,
                    max_chunk_bytes=max_chunk_bytes,
                    runtime_session=active_runtime_session,
                    cancel_token=cancel_token,
                )
            )
    except AnalysisSourceLeaseRefused as error:
        raise RSMOperationRefused(error.code) from error
    cropped_header = detector_header if roi is None else detector_header.with_roi(roi)
    return _RSMCapturedMemberFacts(
        str(root),
        source_relative,
        "" if nexus else analysis.resolved_scan,
        source_options_json,
        analysis.primary_post_state,
        tuple(files),
        tuple(contributions),
        coordinate_frame,
        float(energy),
        tuple(tuple(float(value) for value in row) for row in ub),
        bounds,
        (detector_header.Nch1, detector_header.Nch2),
        (cropped_header.Nch1, cropped_header.Nch2),
        analysis.source_fingerprint,
        source.fingerprint,
        source.table_fingerprint,
        ub_origin=ub_origin,
    )


def _capture_preflight(
    source: ModuleSourceReceipt,
    plan: RSMOperationPlan,
    table,
    project_root: str | Path,
    *,
    cancel_token: threading.Event | None = None,
) -> RSMPreflightReceipt:
    facts = _capture_preflight_facts(
        source,
        table,
        project_root,
        selectors=required_rsm_selectors(plan),
        motor_selectors=plan.geometry.motor_selectors,
        detector_header=plan.geometry.header,
        roi=plan.geometry.roi,
        conditioning=plan.conditioning,
        normalization=plan.normalization,
        chunk_size=plan.chunk_size,
        max_frame_bytes=plan.max_frame_bytes,
        max_chunk_bytes=plan.max_chunk_bytes,
        cancel_token=cancel_token,
    )
    return RSMPreflightReceipt(
        facts.project_root,
        facts.source_relative_path,
        facts.source_scan,
        facts.source_options_json,
        facts.primary_revision,
        facts.files,
        facts.contributions,
        facts.energy_eV,
        facts.ub,
        facts.q_bounds,
        facts.detector_shape,
        facts.cropped_shape,
        facts.source_fingerprint,
        facts.module_source_fingerprint,
        facts.table_fingerprint,
        plan.fingerprint,
        _REQUEST_FACTORY,
    )


def _require_rsm_v2_digest(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TypeError(f"{name} must be lowercase SHA-256")
    return value


def _rsm_v2_uncopyable(kind: str):
    def copy_value(self):
        raise TypeError(f"{kind} is not copyable")

    def deepcopy_value(self, _memo):
        raise TypeError(f"{kind} is not copyable")

    def reduce_value(self):
        raise TypeError(f"{kind} is not serializable")

    def reduce_ex_value(self, _protocol):
        raise TypeError(f"{kind} is not serializable")

    def replace_value(self, /, **_changes):
        raise TypeError(f"{kind} is not replaceable")

    return (
        copy_value,
        deepcopy_value,
        reduce_value,
        reduce_ex_value,
        replace_value,
    )


def _conditioning_provenance(
    conditioning: RSMImageConditioning,
) -> dict[str, object]:
    return {
        "additive_offset": conditioning.additive_offset,
        "high_threshold": conditioning.high_threshold,
        "static_hot_threshold": conditioning.static_hot_threshold,
        "static_rule": "exact-all-selected-frames-static-hot-v1",
    }


def _normalization_provenance(
    normalization: RSMNormalizationPolicy,
) -> dict[str, object]:
    return {
        "mode": normalization.mode.value,
        "foil_selector": (
            None
            if normalization.foil_selector is None
            else {
                "name": normalization.foil_selector.name,
                "occurrence": normalization.foil_selector.occurrence,
            }
        ),
        "exposure_selector": (
            None
            if normalization.exposure_selector is None
            else {
                "name": normalization.exposure_selector.name,
                "occurrence": normalization.exposure_selector.occurrence,
            }
        ),
        "absorption_lengths": list(normalization.absorption_lengths),
        "placement": "conditioned-numerator-before-grid",
    }


@dataclass(eq=False, frozen=True, slots=True)
class RSMCommonGrid:
    """One exact union grid fixed before any detector frame is decoded."""

    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    bins: tuple[int, int, int]
    coordinate_frame: RSMCoordinateFrame
    linspace_policy: str
    accumulation_policy: str
    empty_policy: str
    fingerprint: str
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _RSM_V2_FACTORY
            or type(self.bounds) is not tuple
            or len(self.bounds) != 3
            or any(
                type(item) is not tuple
                or len(item) != 2
                or any(type(value) is not float for value in item)
                or not all(math.isfinite(value) for value in item)
                or item[1] <= item[0]
                for item in self.bounds
            )
            or type(self.bins) is not tuple
            or len(self.bins) != 3
            or any(
                type(value) is not int or not 2 <= value <= _MAX_AXIS_POINTS
                for value in self.bins
            )
            or math.prod(self.bins) > _MAX_RSM_VOXELS
            or type(self.coordinate_frame) is not RSMCoordinateFrame
            or self.linspace_policy != "numpy-linspace-f8-v1"
            or self.accumulation_policy
            != "xrayutilities-gridder3d-sum-raw-sum-norm-v1"
            or self.empty_policy != "NaN-empty-v1"
        ):
            raise TypeError("RSM common grid is not factory-owned")
        _require_rsm_v2_digest(self.fingerprint, "RSM common-grid fingerprint")
        for bounds, count in zip(self.bounds, self.bins, strict=True):
            axis64 = np.linspace(bounds[0], bounds[1], count, dtype=np.float64)
            axis32 = np.ascontiguousarray(axis64, dtype="<f4")
            if (
                axis32.shape != (count,)
                or not np.all(np.isfinite(axis32))
                or not np.all(np.diff(axis32.astype(np.float64)) > 0.0)
            ):
                raise RSMOperationRefused(
                    "RSM_COMMON_GRID_INVALID",
                    "common grid collapses or becomes nonfinite in float32",
                )

    def to_provenance(self) -> dict[str, object]:
        return {
            "bounds": [list(item) for item in self.bounds],
            "bins": list(self.bins),
            "coordinate_frame": self.coordinate_frame.value,
            "axis_names": list(self.coordinate_frame.axis_names),
            "axis_units": list(self.coordinate_frame.axis_units),
            "linspace_policy": self.linspace_policy,
            "accumulation_policy": self.accumulation_policy,
            "empty_policy": self.empty_policy,
            "fingerprint": self.fingerprint,
        }

    (
        __copy__,
        __deepcopy__,
        __reduce__,
        __reduce_ex__,
        __replace__,
    ) = _rsm_v2_uncopyable("RSM common grid")


def make_rsm_common_grid(
    member_bounds: tuple[
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]], ...
    ],
    bins: tuple[int, int, int],
    *,
    coordinate_frame: RSMCoordinateFrame = RSMCoordinateFrame.HKL,
) -> RSMCommonGrid:
    if type(member_bounds) is not tuple or not member_bounds:
        raise TypeError("RSM common grid requires exact ordered member bounds")
    if type(bins) is not tuple:
        raise TypeError("RSM common grid bins must be an exact tuple")
    if type(coordinate_frame) is not RSMCoordinateFrame:
        raise TypeError("RSM common grid coordinate frame must be exact")
    try:
        bounds = tuple(
            (
                float(min(item[axis][0] for item in member_bounds)),
                float(max(item[axis][1] for item in member_bounds)),
            )
            for axis in range(3)
        )
    except (IndexError, TypeError, ValueError) as error:
        raise RSMOperationRefused(
            "RSM_MEMBER_Q_BOUNDS_INVALID",
            "member q bounds cannot form one exact union",
        ) from error
    policies = (
        "numpy-linspace-f8-v1",
        "xrayutilities-gridder3d-sum-raw-sum-norm-v1",
        "NaN-empty-v1",
    )
    fingerprint = analysis_canonical_fingerprint(
        "rsm-common-grid-v2",
        (
            coordinate_frame,
            coordinate_frame.axis_names,
            coordinate_frame.axis_units,
            bounds,
            bins,
            *policies,
        ),
    )
    return RSMCommonGrid(
        bounds,
        bins,
        coordinate_frame,
        *policies,
        fingerprint,
        _RSM_V2_FACTORY,
    )


@dataclass(eq=False, frozen=True, slots=True)
class RSMPreflightMemberV2:
    """One detector-image-free, source- and geometry-bound member receipt."""

    ordinal: int
    member_form_fingerprint: str
    module_source_fingerprint: str
    source_fingerprint: str
    table_fingerprint: str
    source_relative_path: str
    source_scan: str
    source_options_json: str = field(repr=False)
    primary_revision: tuple[int, int, int, int, int, int]
    dependency_files: tuple[RSMManifestFile, ...]
    contributions: tuple[RSMContribution, ...]
    coordinate_frame: RSMCoordinateFrame
    energy_eV: float
    ub: tuple[tuple[float, float, float], ...]
    member_q_bounds: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ]
    detector_shape: tuple[int, int]
    cropped_shape: tuple[int, int]
    geometry_binding: RSMMemberGeometryBinding
    normalization_policy: RSMNormalizationPolicy
    mask_policy_intent: tuple[object, ...]
    fingerprint: str
    _claim: InitVar[object] = None
    #: where the UB came from, for Preview only (decision 2): outside the fingerprint
    ub_origin: str | None = field(default=None, repr=False)

    def __post_init__(self, _claim: object) -> None:
        try:
            options = json.loads(self.source_options_json)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise TypeError("RSM v2 member source options are invalid") from error
        canonical_options = json.dumps(
            options,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        digests = (
            self.member_form_fingerprint,
            self.module_source_fingerprint,
            self.source_fingerprint,
            self.table_fingerprint,
            self.fingerprint,
        )
        if (
            _claim is not _RSM_V2_FACTORY
            or type(self.ordinal) is not int
            or not 0 <= self.ordinal < 16
            or any(
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in digests
            )
            or canonical_options != self.source_options_json
            or type(options) is not dict
            or type(self.source_scan) is not str
            or bool(self.source_scan) == ("entry" in options)  # a SPEC scan, or a NeXus entry (C5c)
            or type(self.dependency_files) is not tuple
            or not self.dependency_files
            or len(self.dependency_files) > _MAX_MANIFEST_FILES
            or any(type(item) is not RSMManifestFile for item in self.dependency_files)
            or len({item.relative_path for item in self.dependency_files})
            != len(self.dependency_files)
            or type(self.contributions) is not tuple
            or not self.contributions
            or len(self.contributions) > _MAX_CONTRIBUTIONS
            or any(type(item) is not RSMContribution for item in self.contributions)
            or any(
                item.file_ordinal >= len(self.dependency_files)
                for item in self.contributions
            )
            or len({item.label for item in self.contributions})
            != len(self.contributions)
            or type(self.coordinate_frame) is not RSMCoordinateFrame
            or type(self.energy_eV) is not float
            or not math.isfinite(self.energy_eV)
            or self.energy_eV <= 0.0
            or type(self.ub) is not tuple
            or len(self.ub) != 3
            or any(type(row) is not tuple or len(row) != 3 for row in self.ub)
            or any(
                type(value) is not float or not math.isfinite(value)
                for row in self.ub
                for value in row
            )
            or type(self.member_q_bounds) is not tuple
            or len(self.member_q_bounds) != 3
            or any(
                type(item) is not tuple
                or len(item) != 2
                or any(type(value) is not float for value in item)
                or not all(math.isfinite(value) for value in item)
                or item[1] <= item[0]
                for item in self.member_q_bounds
            )
            or type(self.detector_shape) is not tuple
            or len(self.detector_shape) != 2
            or any(type(value) is not int or value < 2 for value in self.detector_shape)
            or type(self.cropped_shape) is not tuple
            or len(self.cropped_shape) != 2
            or any(type(value) is not int or value < 2 for value in self.cropped_shape)
            or any(
                cropped > detector
                for cropped, detector in zip(
                    self.cropped_shape, self.detector_shape, strict=True
                )
            )
            or type(self.geometry_binding) is not RSMMemberGeometryBinding
            or self.geometry_binding.member_ordinal != self.ordinal
            or type(self.normalization_policy) is not RSMNormalizationPolicy
            or type(self.mask_policy_intent) is not tuple
            or not self.mask_policy_intent
            or self.mask_policy_intent[0]
            not in {"none", "exact-all-selected-frames-static-hot-v1"}
        ):
            raise TypeError("RSM v2 preflight member is not factory-owned")
        object.__setattr__(
            self,
            "source_relative_path",
            _manifest_relative_path(self.source_relative_path, "RSM v2 source path"),
        )
        _revision(self.primary_revision, "RSM v2 source primary revision")
        if self.mask_policy_intent[0] == "none":
            if self.mask_policy_intent != ("none",):
                raise ValueError("RSM none-mask intent has unexpected fields")
        elif (
            len(self.mask_policy_intent) != 3
            or type(self.mask_policy_intent[1]) is not float
            or not math.isfinite(self.mask_policy_intent[1])
            or self.mask_policy_intent[1] <= 0.0
        ):
            raise ValueError("RSM static-mask intent is invalid")
        else:
            _require_rsm_v2_digest(
                self.mask_policy_intent[2],
                "RSM conditioning fingerprint",
            )
        if (
            self.coordinate_frame is RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU
            and self.ub
            != ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        ):
            raise ValueError("Cartesian-Q RSM member must bind explicit identity")

    def _canonical_value(self) -> tuple[object, ...]:
        return (
            self.ordinal,
            self.member_form_fingerprint,
            self.module_source_fingerprint,
            self.source_fingerprint,
            self.table_fingerprint,
            self.source_relative_path,
            self.source_scan,
            json.loads(self.source_options_json),
            self.primary_revision,
            tuple(
                (item.relative_path, item.revision)
                for item in self.dependency_files
            ),
            tuple(
                (
                    item.label,
                    item.file_ordinal,
                    item.source_frame_index,
                    item.values,
                    item.normalization_divisor,
                )
                for item in self.contributions
            ),
            self.coordinate_frame,
            self.coordinate_frame.matrix_policy,
            self.energy_eV,
            self.ub,
            self.member_q_bounds,
            self.detector_shape,
            self.cropped_shape,
            self.geometry_binding.fingerprint,
            self.normalization_policy._canonical_value(),
            self.mask_policy_intent,
        )

    def to_provenance(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "member_form_fingerprint": self.member_form_fingerprint,
            "module_source_fingerprint": self.module_source_fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "table_fingerprint": self.table_fingerprint,
            "source_relative_path": self.source_relative_path,
            "source_scan": self.source_scan,
            "source_options": json.loads(self.source_options_json),
            "primary_revision": list(self.primary_revision),
            "dependency_files": [
                {
                    "relative_path": item.relative_path,
                    "revision": None if item.revision is None else list(item.revision),
                }
                for item in self.dependency_files
            ],
            "contributions": [
                {
                    "label": item.label,
                    "file_ordinal": item.file_ordinal,
                    "source_frame_index": item.source_frame_index,
                    "values": [
                        {
                            "name": name,
                            "occurrence": occurrence,
                            "value": value,
                        }
                        for name, occurrence, value in item.values
                    ],
                    "normalization_divisor": item.normalization_divisor,
                }
                for item in self.contributions
            ],
            "coordinate_frame": self.coordinate_frame.value,
            "axis_names": list(self.coordinate_frame.axis_names),
            "axis_units": list(self.coordinate_frame.axis_units),
            "matrix_policy": self.coordinate_frame.matrix_policy,
            "energy_eV": self.energy_eV,
            "coordinate_matrix": [list(row) for row in self.ub],
            "member_q_bounds": [list(item) for item in self.member_q_bounds],
            "detector_shape": list(self.detector_shape),
            "cropped_shape": list(self.cropped_shape),
            "geometry_binding": {
                "member_ordinal": self.geometry_binding.member_ordinal,
                "effective_geometry_fingerprint": (
                    self.geometry_binding.effective_geometry_fingerprint
                ),
                "motor_selectors": [
                    {
                        "role": role,
                        "name": selector.name,
                        "occurrence": selector.occurrence,
                    }
                    for role, selector in self.geometry_binding.motor_selectors
                ],
                "fingerprint": self.geometry_binding.fingerprint,
            },
            "normalization_policy": _normalization_provenance(
                self.normalization_policy
            ),
            "mask_policy_intent": list(self.mask_policy_intent),
            "fingerprint": self.fingerprint,
        }

    (
        __copy__,
        __deepcopy__,
        __reduce__,
        __reduce_ex__,
        __replace__,
    ) = _rsm_v2_uncopyable("RSM v2 preflight member")


def _make_rsm_preflight_member_v2(
    facts: _RSMCapturedMemberFacts,
    *,
    ordinal: int,
    member_form_fingerprint: str,
    binding: RSMMemberGeometryBinding,
    normalization: RSMNormalizationPolicy,
    conditioning: RSMImageConditioning,
) -> RSMPreflightMemberV2:
    conditioning_fingerprint = analysis_canonical_fingerprint(
        "rsm-conditioning-v2",
        conditioning._canonical_value(),
    )
    mask_intent = (
        ("none",)
        if conditioning.static_hot_threshold is None
        else (
            "exact-all-selected-frames-static-hot-v1",
            conditioning.static_hot_threshold,
            conditioning_fingerprint,
        )
    )
    canonical = (
        ordinal,
        member_form_fingerprint,
        facts.module_source_fingerprint,
        facts.source_fingerprint,
        facts.table_fingerprint,
        facts.source_relative_path,
        facts.source_scan,
        json.loads(facts.source_options_json),
        facts.primary_revision,
        tuple((item.relative_path, item.revision) for item in facts.files),
        tuple(
            (
                item.label,
                item.file_ordinal,
                item.source_frame_index,
                item.values,
                item.normalization_divisor,
            )
            for item in facts.contributions
        ),
        facts.coordinate_frame,
        facts.coordinate_frame.matrix_policy,
        facts.energy_eV,
        facts.ub,
        facts.q_bounds,
        facts.detector_shape,
        facts.cropped_shape,
        binding.fingerprint,
        normalization._canonical_value(),
        mask_intent,
    )
    fingerprint = analysis_canonical_fingerprint(
        "rsm-preflight-member-v2", canonical
    )
    return RSMPreflightMemberV2(
        ordinal,
        member_form_fingerprint,
        facts.module_source_fingerprint,
        facts.source_fingerprint,
        facts.table_fingerprint,
        facts.source_relative_path,
        facts.source_scan,
        facts.source_options_json,
        facts.primary_revision,
        facts.files,
        facts.contributions,
        facts.coordinate_frame,
        facts.energy_eV,
        facts.ub,
        facts.q_bounds,
        facts.detector_shape,
        facts.cropped_shape,
        binding,
        normalization,
        mask_intent,
        fingerprint,
        _RSM_V2_FACTORY,
        ub_origin=facts.ub_origin,
    )


def _require_unique_rsm_physical_contributions_v2(
    members: tuple[RSMPreflightMemberV2, ...],
) -> None:
    physical: set[tuple[tuple[int, int, int, int, int, int], int]] = set()
    for member in members:
        for contribution in member.contributions:
            revision = member.dependency_files[contribution.file_ordinal].revision
            if revision is None:
                raise RSMOperationRefused(
                    "RSM_MEMBER_IDENTITY_MISMATCH",
                    "selected contribution has no exact file revision",
                )
            identity = (revision, contribution.source_frame_index)
            if identity in physical:
                raise RSMOperationRefused(
                    "RSM_SOURCE_GROUP_INVALID",
                    "two members select the same physical contribution",
                )
            physical.add(identity)


@dataclass(eq=False, frozen=True, slots=True)
class RSMOperationPlanV2:
    """Effective-geometry and ordered-member-bound RSM v2 plan."""

    effective_geometry: RSMEffectiveGeometry
    ordered_geometry_bindings: tuple[RSMMemberGeometryBinding, ...]
    coordinate_frame: RSMCoordinateFrame
    common_grid: RSMCommonGrid
    conditioning: RSMImageConditioning
    normalization: RSMNormalizationPolicy
    chunk_size: int
    max_frame_bytes: int
    max_chunk_bytes: int
    fingerprint: str
    _claim: InitVar[object] = None
    #: The per-frame corrections bound at Preview (``RSMBoundCorrections``); None: none, as before.
    corrections: object = None
    #: A UB given at Preview (``RSMBoundUB``), replacing every member's source UB; None: the sources' own.
    ub_override: object = None
    #: The calibrated detector's own mask (module gaps) in the raw frame, read-only (``rsm_calibration_detector_mask``),
    #: masked in every frame whatever the corrections (S3c unit 4b); None: the canonical asset, or no gaps. Not
    #: identity: the geometry receipt carries its digest.
    detector_mask: object = None

    def __post_init__(self, _claim: object) -> None:
        header = getattr(self.effective_geometry, "detector_header", None)
        if (
            _claim is not _RSM_V2_FACTORY
            or (self.corrections is not None and type(self.corrections) is not RSMBoundCorrections)
            or (self.ub_override is not None and type(self.ub_override) is not RSMBoundUB)
            or (self.detector_mask is not None and (
                type(self.detector_mask) is not np.ndarray or self.detector_mask.dtype != np.dtype(bool)
                or self.detector_mask.flags.writeable or header is None
                or self.detector_mask.shape != (int(header.Nch1), int(header.Nch2))))
            or type(self.effective_geometry) is not RSMEffectiveGeometry
            or type(self.ordered_geometry_bindings) is not tuple
            or not 1 <= len(self.ordered_geometry_bindings) <= 16
            or any(
                type(item) is not RSMMemberGeometryBinding
                for item in self.ordered_geometry_bindings
            )
            or tuple(
                item.member_ordinal for item in self.ordered_geometry_bindings
            )
            != tuple(range(len(self.ordered_geometry_bindings)))
            or any(
                item.effective_geometry_fingerprint
                != self.effective_geometry.fingerprint
                for item in self.ordered_geometry_bindings
            )
            or type(self.coordinate_frame) is not RSMCoordinateFrame
            or type(self.common_grid) is not RSMCommonGrid
            or self.common_grid.coordinate_frame is not self.coordinate_frame
            or type(self.conditioning) is not RSMImageConditioning
            or type(self.normalization) is not RSMNormalizationPolicy
            or type(self.chunk_size) is not int
            or not 1 <= self.chunk_size <= _MAX_CHUNK_SIZE
            or type(self.max_frame_bytes) is not int
            or not 1 <= self.max_frame_bytes <= _MAX_FRAME_BYTES
            or type(self.max_chunk_bytes) is not int
            or not 1 <= self.max_chunk_bytes <= _MAX_CHUNK_BYTES
        ):
            raise TypeError("RSM v2 operation plan is not factory-owned")
        _require_rsm_v2_digest(self.fingerprint, "RSM v2 plan fingerprint")
        normalization_keys = {
            (selector.name, selector.occurrence)
            for selector in (
                self.normalization.foil_selector,
                self.normalization.exposure_selector,
            )
            if selector is not None
        }
        for binding in self.ordered_geometry_bindings:
            motor_keys = {
                (selector.name, selector.occurrence)
                for _role, selector in binding.motor_selectors
            }
            if motor_keys & normalization_keys:
                raise ValueError(
                    "RSM normalization selectors must be disjoint from motor selectors"
                )

    @property
    def bins(self) -> tuple[int, int, int]:
        return self.common_grid.bins

    def _canonical_value(self) -> tuple[object, ...]:
        return (
            "rsm-operation-plan-v2",
            self.coordinate_frame,
            self.coordinate_frame.axis_names,
            self.coordinate_frame.axis_units,
            self.coordinate_frame.matrix_policy,
            self.effective_geometry.fingerprint,
            tuple(item.fingerprint for item in self.ordered_geometry_bindings),
            self.common_grid.fingerprint,
            self.conditioning._canonical_value(),
            self.normalization._canonical_value(),
            self.common_grid.bins,
            self.chunk_size,
            self.max_frame_bytes,
            self.max_chunk_bytes,
            "exact-all-selected-pixel-q-bounds-v2",
            "ordered-one-common-grid-v1",
            "numerator-first-normalization-v1",
            "bounded-chunk-working-set-v1",
            None,
            None,
        ) + (() if self.corrections is None else (self.corrections.identity(),)) + (
            () if self.ub_override is None else (self.ub_override.identity(),))

    def to_provenance(self) -> dict[str, object]:
        return {
            "coordinate_frame": self.coordinate_frame.value,
            "axis_names": list(self.coordinate_frame.axis_names),
            "axis_units": list(self.coordinate_frame.axis_units),
            "matrix_policy": self.coordinate_frame.matrix_policy,
            "bins": list(self.bins),
            "chunk_size": self.chunk_size,
            "max_frame_bytes": self.max_frame_bytes,
            "max_chunk_bytes": self.max_chunk_bytes,
            "q_bounds_policy": "exact-all-selected-pixel-q-bounds-v2",
            "grid_policy": "ordered-one-common-grid-v1",
            "normalization_placement": "numerator-first-normalization-v1",
            "working_set_policy": "bounded-chunk-working-set-v1",
            "gi_q_correction": None,
            "gi_intensity_correction": None,
            # the per-frame corrections, only when the plan has them
            **({} if self.corrections is None else {"frame_corrections": self.corrections.to_provenance()}),
            # where the UB came from, only when given at Preview (the sources' own otherwise)
            **({} if self.ub_override is None else {"ub_source": self.ub_override.to_provenance()}),
            "fingerprint": self.fingerprint,
        }

    (
        __copy__,
        __deepcopy__,
        __reduce__,
        __reduce_ex__,
        __replace__,
    ) = _rsm_v2_uncopyable("RSM v2 operation plan")


def _make_rsm_operation_plan_v2(
    effective_geometry: RSMEffectiveGeometry,
    bindings: tuple[RSMMemberGeometryBinding, ...],
    coordinate_frame: RSMCoordinateFrame,
    common_grid: RSMCommonGrid,
    conditioning: RSMImageConditioning,
    normalization: RSMNormalizationPolicy,
    *,
    chunk_size: int,
    max_frame_bytes: int,
    max_chunk_bytes: int,
    corrections: RSMBoundCorrections | None = None,
    ub_override: RSMBoundUB | None = None,
    detector_mask: np.ndarray | None = None,
) -> RSMOperationPlanV2:
    canonical = (
        "rsm-operation-plan-v2",
        coordinate_frame,
        coordinate_frame.axis_names,
        coordinate_frame.axis_units,
        coordinate_frame.matrix_policy,
        effective_geometry.fingerprint,
        tuple(item.fingerprint for item in bindings),
        common_grid.fingerprint,
        conditioning._canonical_value(),
        normalization._canonical_value(),
        common_grid.bins,
        chunk_size,
        max_frame_bytes,
        max_chunk_bytes,
        "exact-all-selected-pixel-q-bounds-v2",
        "ordered-one-common-grid-v1",
        "numerator-first-normalization-v1",
        "bounded-chunk-working-set-v1",
        None,
        None,
    ) + (() if corrections is None else (corrections.identity(),)) + (
        () if ub_override is None else (ub_override.identity(),))
    fingerprint = module_plan_fingerprint(ModuleKind.RSM, canonical)
    return RSMOperationPlanV2(
        effective_geometry,
        bindings,
        coordinate_frame,
        common_grid,
        conditioning,
        normalization,
        chunk_size,
        max_frame_bytes,
        max_chunk_bytes,
        fingerprint,
        _RSM_V2_FACTORY,
        corrections,
        ub_override,
        detector_mask,
    )


@dataclass(eq=False, frozen=True, slots=True)
class RSMGroupPreflightReceiptV2:
    """Ordered group preflight with exact contribution and grid identity."""

    project_root: str
    geometry_asset_receipt: RSMGeometryAssetReceipt | RSMCalibrationGeometryReceipt
    effective_geometry: RSMEffectiveGeometry
    ordered_geometry_bindings: tuple[RSMMemberGeometryBinding, ...]
    ordered_members: tuple[RSMPreflightMemberV2, ...]
    group_source_fingerprint: str
    common_grid: RSMCommonGrid
    plan_fingerprint: str
    fingerprint: str
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _RSM_V2_FACTORY
            or type(self.project_root) is not str
            or not Path(self.project_root).is_absolute()
            or type(self.geometry_asset_receipt) not in _RSM_GEOMETRY_RECEIPTS
            or self.geometry_asset_receipt.project_root != self.project_root
            or type(self.effective_geometry) is not RSMEffectiveGeometry
            or self.effective_geometry.asset_receipt_fingerprint
            != self.geometry_asset_receipt.receipt_fingerprint
            or type(self.ordered_geometry_bindings) is not tuple
            or type(self.ordered_members) is not tuple
            or not 1 <= len(self.ordered_members) <= 16
            or len(self.ordered_geometry_bindings) != len(self.ordered_members)
            or any(
                type(item) is not RSMMemberGeometryBinding
                for item in self.ordered_geometry_bindings
            )
            or any(type(item) is not RSMPreflightMemberV2 for item in self.ordered_members)
            or tuple(item.ordinal for item in self.ordered_members)
            != tuple(range(len(self.ordered_members)))
            or any(
                member.geometry_binding is not binding
                for member, binding in zip(
                    self.ordered_members,
                    self.ordered_geometry_bindings,
                    strict=True,
                )
            )
            or type(self.common_grid) is not RSMCommonGrid
            or any(
                item.coordinate_frame is not self.common_grid.coordinate_frame
                for item in self.ordered_members
            )
        ):
            raise TypeError("RSM v2 group preflight is not factory-owned")
        _require_rsm_v2_digest(
            self.group_source_fingerprint,
            "RSM group source fingerprint",
        )
        _require_rsm_v2_digest(self.plan_fingerprint, "RSM v2 plan fingerprint")
        _require_rsm_v2_digest(self.fingerprint, "RSM v2 preflight fingerprint")
        if sum(len(item.dependency_files) for item in self.ordered_members) > _MAX_MANIFEST_FILES:
            raise RSMOperationRefused(
                "RSM_SOURCE_GROUP_LIMIT_EXCEEDED",
                "RSM group exceeds 8192 dependency files",
            )
        if sum(len(item.contributions) for item in self.ordered_members) > _MAX_CONTRIBUTIONS:
            raise RSMOperationRefused(
                "RSM_SOURCE_GROUP_LIMIT_EXCEEDED",
                "RSM group exceeds 4096 selected frames",
            )
        _require_unique_rsm_physical_contributions_v2(self.ordered_members)
        encoded = json.dumps(
            self.to_provenance(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > _MAX_MANIFEST_BYTES:
            raise RSMOperationRefused(
                "RSM_SOURCE_GROUP_LIMIT_EXCEEDED",
                "canonical group preflight exceeds 512 KiB",
            )

    @property
    def members(self) -> tuple[RSMPreflightMemberV2, ...]:
        return self.ordered_members

    def to_provenance(self) -> dict[str, object]:
        return {
            "schema_version": "rsm-preflight-v2",
            "coordinate_frame": self.common_grid.coordinate_frame.value,
            "axis_names": list(self.common_grid.coordinate_frame.axis_names),
            "axis_units": list(self.common_grid.coordinate_frame.axis_units),
            "project_root": self.project_root,
            "geometry_asset_receipt_fingerprint": (
                self.geometry_asset_receipt.receipt_fingerprint
            ),
            "effective_geometry_fingerprint": self.effective_geometry.fingerprint,
            "ordered_geometry_binding_fingerprints": [
                item.fingerprint for item in self.ordered_geometry_bindings
            ],
            "ordered_members": [item.to_provenance() for item in self.ordered_members],
            "group_source_fingerprint": self.group_source_fingerprint,
            "common_grid": self.common_grid.to_provenance(),
            "plan_fingerprint": self.plan_fingerprint,
            "fingerprint": self.fingerprint,
        }

    (
        __copy__,
        __deepcopy__,
        __reduce__,
        __reduce_ex__,
        __replace__,
    ) = _rsm_v2_uncopyable("RSM v2 group preflight")


def _make_rsm_group_preflight_v2(
    *,
    project_root: Path,
    asset: RSMGeometryAssetReceipt | RSMCalibrationGeometryReceipt,
    effective: RSMEffectiveGeometry,
    bindings: tuple[RSMMemberGeometryBinding, ...],
    members: tuple[RSMPreflightMemberV2, ...],
    group: ModuleSourceGroupReceipt,
    common_grid: RSMCommonGrid,
    plan: RSMOperationPlanV2,
) -> RSMGroupPreflightReceiptV2:
    expected_member_sources = tuple(item.fingerprint for item in group.members)
    observed_member_sources = tuple(
        item.module_source_fingerprint for item in members
    )
    expected_bounds = tuple(
        (
            float(min(item.member_q_bounds[axis][0] for item in members)),
            float(max(item.member_q_bounds[axis][1] for item in members)),
        )
        for axis in range(3)
    )
    expected_grid_fingerprint = analysis_canonical_fingerprint(
        "rsm-common-grid-v2",
        (
            common_grid.coordinate_frame,
            common_grid.coordinate_frame.axis_names,
            common_grid.coordinate_frame.axis_units,
            expected_bounds,
            common_grid.bins,
            common_grid.linspace_policy,
            common_grid.accumulation_policy,
            common_grid.empty_policy,
        ),
    )
    if (
        observed_member_sources != expected_member_sources
        or plan.effective_geometry is not effective
        or plan.ordered_geometry_bindings != bindings
        or plan.coordinate_frame is not common_grid.coordinate_frame
        or plan.common_grid is not common_grid
        or any(
            item.coordinate_frame is not common_grid.coordinate_frame
            for item in members
        )
        or common_grid.bounds != expected_bounds
        or common_grid.fingerprint != expected_grid_fingerprint
    ):
        raise RSMOperationRefused(
            "RSM_MEMBER_IDENTITY_MISMATCH",
            "RSM v2 group preflight inputs are not one exact ordered relation",
        )
    canonical = (
        str(project_root),
        asset.receipt_fingerprint,
        effective.fingerprint,
        tuple(item.fingerprint for item in bindings),
        tuple(item.fingerprint for item in members),
        group.fingerprint,
        common_grid.fingerprint,
        plan.fingerprint,
    )
    fingerprint = analysis_canonical_fingerprint("rsm-preflight-v2", canonical)
    return RSMGroupPreflightReceiptV2(
        str(project_root),
        asset,
        effective,
        bindings,
        members,
        group.fingerprint,
        common_grid,
        plan.fingerprint,
        fingerprint,
        _RSM_V2_FACTORY,
    )


def _effective_geometry_provenance(
    effective: RSMEffectiveGeometry,
) -> dict[str, object]:
    diffractometer = effective.diffractometer_projection

    def motor(value: object) -> dict[str, object]:
        return {
            "source_motor": value.source_motor,
            "sign": value.sign,
            "offset": value.offset,
        }

    from xrd_tools.core.geometry.xu_runtime import (
        xu_runtime_requirements_projection,
    )

    header = effective.detector_header
    orientation = effective.image_orientation
    return {
        "asset_receipt_fingerprint": effective.asset_receipt_fingerprint,
        "asset_semantic_fingerprint": effective.asset_semantic_fingerprint,
        "diffractometer": {
            "preset": diffractometer.preset,
            "rot1": motor(diffractometer.rot1),
            "rot2": motor(diffractometer.rot2),
            "rot3": motor(diffractometer.rot3),
            "incident_angle": motor(diffractometer.incident_angle),
            "sample_circles": list(diffractometer.sample_circles),
            "detector_circles": list(diffractometer.detector_circles),
            "r_i": list(diffractometer.r_i),
            "camera": list(diffractometer.camera),
            "hxrd_n": list(diffractometer.hxrd_n),
            "hxrd_q": list(diffractometer.hxrd_q),
            "hxrd_geometry": diffractometer.hxrd_geometry,
            "circle_motors": [motor(item) for item in diffractometer.circle_motors],
            "sample_motors": list(diffractometer.sample_motors),
            "detector_motors": list(diffractometer.detector_motors),
            "qconv_kwargs": dict(diffractometer.qconv_kwargs),
            "hxrd_kwargs": dict(diffractometer.hxrd_kwargs),
            "ang2q_kwargs": dict(diffractometer.ang2q_kwargs),
            "calibration": diffractometer.calibration,
        },
        "detector_header": {
            name: getattr(header, name)
            for name in (
                "cch1",
                "cch2",
                "pwidth1",
                "pwidth2",
                "distance",
                "Nch1",
                "Nch2",
            )
        },
        "image_orientation": {
            "rotation": orientation.rotation,
            "flip_vertical": orientation.flip_vertical,
            "flip_horizontal": orientation.flip_horizontal,
            "transpose": orientation.transpose,
        },
        "roi": list(effective.roi),
        "runtime_requirements": list(
            xu_runtime_requirements_projection(effective.runtime_requirements)
        ),
        **({} if effective.source == "asset" else {
            "source": effective.source,
            "refinement": {
                "applied": bool(effective.refinement),
                "angle_offsets_deg": list(effective.angle_offsets),
                "mount_deg": dict(zip(("detrot", "tilt", "tiltazimuth"), effective.area_mount)),
                "energy_offset_eV": effective.energy_offset_eV,
                "calibration_energy_eV": effective.calibration_energy_eV,
            },
        }),
        "fingerprint": effective.fingerprint,
    }


def _rsm_v2_provenance(
    group: ModuleSourceGroupReceipt,
    output: ModuleOutputRequest,
    output_authority: RSMOutputAuthorityReceipt,
    asset: RSMGeometryAssetReceipt | RSMCalibrationGeometryReceipt,
    effective: RSMEffectiveGeometry,
    plan: RSMOperationPlanV2,
    preflight: RSMGroupPreflightReceiptV2,
) -> dict[str, object]:
    from xrd_tools.core.geometry.xu_runtime import (
        xu_runtime_requirements_projection,
    )

    return {
        "schema_version": "rsm-operation-v3-intent",
        "kind": "rsm",
        "coordinate_frame": {
            "name": plan.coordinate_frame.value,
            "axis_names": list(plan.coordinate_frame.axis_names),
            "axis_units": list(plan.coordinate_frame.axis_units),
            "matrix_policy": plan.coordinate_frame.matrix_policy,
        },
        "source_group": {
            "member_fingerprints": [item.fingerprint for item in group.members],
            "group_fingerprint": group.fingerprint,
            "preflight_fingerprint": preflight.fingerprint,
        },
        "asset": {
            "lexical_relative_path": asset.lexical_relative_path,
            "resolved_relative_path": asset.resolved_relative_path,
            "byte_count": asset.byte_count,
            "raw_sha256": asset.raw_sha256,
            "semantic_fingerprint": asset.semantic_fingerprint,
            "receipt_fingerprint": asset.receipt_fingerprint,
        },
        "effective_geometry": _effective_geometry_provenance(effective),
        "members": [item.to_provenance() for item in preflight.members],
        "common_grid": preflight.common_grid.to_provenance(),
        "conditioning": _conditioning_provenance(plan.conditioning),
        "normalization": _normalization_provenance(plan.normalization),
        "plan": plan.to_provenance(),
        "runtime_requirements": list(
            xu_runtime_requirements_projection(effective.runtime_requirements)
        ),
        "output": {
            "kind": output.kind.value,
            "target": output.target,
            "overwrite": output.overwrite.value,
            "output_fingerprint": output.fingerprint,
            **output_authority.to_provenance(),
            # the stable slot's family, persisted as native Stitch persists its own
            **(
                {}
                if output.artifact_family is None
                else {"artifact_family": output.artifact_family}
            ),
        },
        "holds": list(_RSM_V2_HOLDS),
    }


@dataclass(eq=False, frozen=True, slots=True)
class RSMOperationRequestV2:
    """One immutable image-free RSM v2 Preview result."""

    module: ModuleOperationRequest
    plan: RSMOperationPlanV2
    preflight: RSMGroupPreflightReceiptV2
    output_authority: RSMOutputAuthorityReceipt
    provenance_json: str
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _RSM_V2_FACTORY
            or type(self.module) is not ModuleOperationRequest
            or self.module._rsm_v2_bound is not True
            or type(self.module.source) is not ModuleSourceGroupReceipt
            or self.module.kind is not ModuleKind.RSM
            or self.module.output.kind is not AnalysisArtifactKind.RSM
            or type(self.plan) is not RSMOperationPlanV2
            or type(self.preflight) is not RSMGroupPreflightReceiptV2
            or type(self.output_authority) is not RSMOutputAuthorityReceipt
            or type(self.provenance_json) is not str
            or self.module.plan_fingerprint != self.plan.fingerprint
            or self.preflight.plan_fingerprint != self.plan.fingerprint
            or self.preflight.group_source_fingerprint
            != self.module.source.fingerprint
            or self.preflight.effective_geometry is not self.plan.effective_geometry
            or self.preflight.common_grid is not self.plan.common_grid
            or self.preflight.ordered_geometry_bindings
            != self.plan.ordered_geometry_bindings
        ):
            raise TypeError("RSM v2 operation request is not factory-owned")
        try:
            provenance = json.loads(self.provenance_json)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise TypeError("RSM v2 operation provenance is invalid") from error
        expected = _rsm_v2_provenance(
            self.module.source,
            self.module.output,
            self.output_authority,
            self.preflight.geometry_asset_receipt,
            self.preflight.effective_geometry,
            self.plan,
            self.preflight,
        )
        if (
            type(provenance) is not dict
            or set(provenance)
            != {
                "schema_version",
                "kind",
                "coordinate_frame",
                "source_group",
                "asset",
                "effective_geometry",
                "members",
                "common_grid",
                "conditioning",
                "normalization",
                "plan",
                "runtime_requirements",
                "output",
                "holds",
            }
            or provenance != expected
            or canonical_analysis_provenance(provenance) != self.provenance_json
            or module_provenance_digest(ModuleKind.RSM, provenance)
            != self.module.provenance_digest
        ):
            raise ValueError("RSM v2 provenance does not match its exact request")

    @property
    def provenance(self) -> dict[str, object]:
        return json.loads(self.provenance_json)

    (
        __copy__,
        __deepcopy__,
        __reduce__,
        __reduce_ex__,
        __replace__,
    ) = _rsm_v2_uncopyable("RSM v2 operation request")


@dataclass(eq=False, frozen=True, slots=True, weakref_slot=True)
class RSMStaticMaskReceipt:
    """Scalar-only identity for one member's independently derived mask."""

    member_preflight_fingerprint: str
    mask_policy: str
    conditioning_fingerprint: str
    full_shape: tuple[int, int]
    full_raw_digest: str | None
    full_masked_pixel_count: int
    cropped_shape: tuple[int, int]
    cropped_raw_digest: str | None
    cropped_masked_pixel_count: int
    fingerprint: str
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _RSM_V2_FACTORY
            or self.mask_policy
            not in {"none", "exact-all-selected-frames-static-hot-v1"}
            or type(self.full_shape) is not tuple
            or type(self.cropped_shape) is not tuple
            or len(self.full_shape) != 2
            or len(self.cropped_shape) != 2
            or any(
                type(value) is not int or value < 2
                for value in (*self.full_shape, *self.cropped_shape)
            )
            or any(
                cropped > full
                for cropped, full in zip(
                    self.cropped_shape,
                    self.full_shape,
                    strict=True,
                )
            )
            or type(self.full_masked_pixel_count) is not int
            or not 0
            <= self.full_masked_pixel_count
            <= math.prod(self.full_shape)
            or type(self.cropped_masked_pixel_count) is not int
            or not 0
            <= self.cropped_masked_pixel_count
            <= math.prod(self.cropped_shape)
            or self.cropped_masked_pixel_count
            > self.full_masked_pixel_count
        ):
            raise TypeError("RSM static-mask receipt is not factory-owned")
        _require_rsm_v2_digest(
            self.member_preflight_fingerprint,
            "RSM member preflight fingerprint",
        )
        _require_rsm_v2_digest(
            self.conditioning_fingerprint,
            "RSM conditioning fingerprint",
        )
        _require_rsm_v2_digest(
            self.fingerprint,
            "RSM static-mask receipt fingerprint",
        )
        if self.mask_policy == "none":
            if (
                self.full_raw_digest is not None
                or self.cropped_raw_digest is not None
                or self.full_masked_pixel_count != 0
                or self.cropped_masked_pixel_count != 0
            ):
                raise ValueError("RSM none-mask receipt has unexpected facts")
        else:
            _require_rsm_v2_digest(
                self.full_raw_digest,
                "RSM full static-mask raw digest",
            )
            _require_rsm_v2_digest(
                self.cropped_raw_digest,
                "RSM cropped static-mask raw digest",
            )
        expected = analysis_canonical_fingerprint(
            "rsm-static-mask-v1",
            (
                self.member_preflight_fingerprint,
                self.mask_policy,
                self.conditioning_fingerprint,
                self.full_shape,
                self.full_raw_digest,
                self.full_masked_pixel_count,
                self.cropped_shape,
                self.cropped_raw_digest,
                self.cropped_masked_pixel_count,
            ),
        )
        if self.fingerprint != expected:
            raise ValueError("RSM static-mask receipt fingerprint changed")

    def to_attestation(self, ordinal: int) -> dict[str, object]:
        if type(ordinal) is not int or not 0 <= ordinal < 16:
            raise TypeError("RSM mask ordinal is invalid")
        return {
            "ordinal": ordinal,
            "member_preflight_fingerprint": self.member_preflight_fingerprint,
            "mask_policy": self.mask_policy,
            "full_shape": list(self.full_shape),
            "full_raw_digest": self.full_raw_digest,
            "full_masked_pixel_count": self.full_masked_pixel_count,
            "cropped_shape": list(self.cropped_shape),
            "cropped_raw_digest": self.cropped_raw_digest,
            "cropped_masked_pixel_count": self.cropped_masked_pixel_count,
            "mask_receipt_fingerprint": self.fingerprint,
        }

    (
        __copy__,
        __deepcopy__,
        __reduce__,
        __reduce_ex__,
        __replace__,
    ) = _rsm_v2_uncopyable("RSM static-mask receipt")


_RSM_V2_STATIC_MASK_ISSUANCE_LOCK = threading.RLock()
_RSM_V2_STATIC_MASK_ISSUANCE: weakref.WeakKeyDictionary[
    RSMStaticMaskReceipt, tuple[object, ...]
] = weakref.WeakKeyDictionary()


def _rsm_v2_static_mask_receipt_value(
    receipt: RSMStaticMaskReceipt,
) -> tuple[object, ...]:
    try:
        return (
            receipt.member_preflight_fingerprint,
            receipt.mask_policy,
            receipt.conditioning_fingerprint,
            receipt.full_shape,
            receipt.full_raw_digest,
            receipt.full_masked_pixel_count,
            receipt.cropped_shape,
            receipt.cropped_raw_digest,
            receipt.cropped_masked_pixel_count,
            receipt.fingerprint,
        )
    except AttributeError as error:
        raise TypeError("RSM static-mask receipt was not issued") from error


def _rsm_v2_static_mask_attestation(
    receipt: object,
    ordinal: int,
    *,
    expected_conditioning_fingerprint: str | None = None,
) -> dict[str, object]:
    """Revalidate one factory receipt before exposing its scalar projection."""

    if type(receipt) is not RSMStaticMaskReceipt:
        raise TypeError("RSM static-mask attestation requires an exact receipt")
    if type(ordinal) is not int or not 0 <= ordinal < 16:
        raise TypeError("RSM mask ordinal is invalid")
    with _RSM_V2_STATIC_MASK_ISSUANCE_LOCK:
        issued = _RSM_V2_STATIC_MASK_ISSUANCE.get(receipt)
    if issued is None:
        raise TypeError("RSM static-mask receipt was not issued")
    if _rsm_v2_static_mask_receipt_value(receipt) != issued:
        raise ValueError("RSM static-mask receipt changed after issuance")
    if (
        expected_conditioning_fingerprint is not None
        and issued[2] != expected_conditioning_fingerprint
    ):
        raise ValueError("RSM static-mask receipt conditioning changed")
    return {
        "ordinal": ordinal,
        "member_preflight_fingerprint": issued[0],
        "mask_policy": issued[1],
        "full_shape": list(issued[3]),
        "full_raw_digest": issued[4],
        "full_masked_pixel_count": issued[5],
        "cropped_shape": list(issued[6]),
        "cropped_raw_digest": issued[7],
        "cropped_masked_pixel_count": issued[8],
        "mask_receipt_fingerprint": issued[9],
    }


def _rsm_v2_static_mask_raw_digest(mask: np.ndarray) -> str:
    if (
        type(mask) is not np.ndarray
        or mask.dtype != np.dtype(bool)
        or mask.ndim != 2
        or not mask.flags.c_contiguous
    ):
        raise TypeError("RSM static mask must be one contiguous bool array")
    rows, columns = (int(value) for value in mask.shape)
    bits = np.packbits(
        mask.reshape(-1).astype(np.uint8, copy=False),
        bitorder="little",
    )
    return hashlib.sha256(
        b"xdart.rsm.static-mask.bits.v1\0"
        + struct.pack("<QQ", rows, columns)
        + bits.tobytes()
    ).hexdigest()


def _rsm_v2_cropped_static_mask(
    mask: np.ndarray,
    roi: tuple[int, int, int, int],
    expected_shape: tuple[int, int],
) -> np.ndarray:
    if (
        type(mask) is not np.ndarray
        or mask.dtype != np.dtype(bool)
        or mask.ndim != 2
        or type(roi) is not tuple
        or len(roi) != 4
        or any(type(value) is not int for value in roi)
    ):
        raise RSMOperationRefused("RSM_STATIC_MASK_INVALID")
    r0, r1, c0, c1 = roi
    cropped = np.ascontiguousarray(mask[r0:r1, c0:c1], dtype=bool)
    if cropped.shape != expected_shape:
        raise RSMOperationRefused("RSM_STATIC_MASK_INVALID")
    cropped.setflags(write=False)
    return cropped


def _make_rsm_static_mask_receipt(
    member: RSMPreflightMemberV2,
    conditioning: RSMImageConditioning,
    full_mask: np.ndarray | None,
    cropped_mask: np.ndarray | None,
) -> RSMStaticMaskReceipt:
    if (
        type(member) is not RSMPreflightMemberV2
        or type(conditioning) is not RSMImageConditioning
    ):
        raise TypeError("RSM static-mask receipt requires exact inputs")
    conditioning_fingerprint = analysis_canonical_fingerprint(
        "rsm-conditioning-v2",
        conditioning._canonical_value(),
    )
    policy = member.mask_policy_intent[0]
    if policy == "none":
        if (
            member.mask_policy_intent != ("none",)
            or full_mask is not None
            or cropped_mask is not None
        ):
            raise RSMOperationRefused("RSM_STATIC_MASK_IDENTITY_MISMATCH")
        full_digest = cropped_digest = None
        full_count = cropped_count = 0
    else:
        if (
            member.mask_policy_intent
            != (
                "exact-all-selected-frames-static-hot-v1",
                conditioning.static_hot_threshold,
                conditioning_fingerprint,
            )
            or type(full_mask) is not np.ndarray
            or type(cropped_mask) is not np.ndarray
            or full_mask.shape != member.detector_shape
            or cropped_mask.shape != member.cropped_shape
            or full_mask.dtype != np.dtype(bool)
            or cropped_mask.dtype != np.dtype(bool)
            or full_mask.flags.writeable
            or cropped_mask.flags.writeable
        ):
            raise RSMOperationRefused("RSM_STATIC_MASK_IDENTITY_MISMATCH")
        full_digest = _rsm_v2_static_mask_raw_digest(full_mask)
        cropped_digest = _rsm_v2_static_mask_raw_digest(cropped_mask)
        full_count = int(np.count_nonzero(full_mask))
        cropped_count = int(np.count_nonzero(cropped_mask))
    canonical = (
        member.fingerprint,
        policy,
        conditioning_fingerprint,
        member.detector_shape,
        full_digest,
        full_count,
        member.cropped_shape,
        cropped_digest,
        cropped_count,
    )
    fingerprint = analysis_canonical_fingerprint(
        "rsm-static-mask-v1",
        canonical,
    )
    receipt = RSMStaticMaskReceipt(
        member.fingerprint,
        policy,
        conditioning_fingerprint,
        member.detector_shape,
        full_digest,
        full_count,
        member.cropped_shape,
        cropped_digest,
        cropped_count,
        fingerprint,
        _RSM_V2_FACTORY,
    )
    with _RSM_V2_STATIC_MASK_ISSUANCE_LOCK:
        _RSM_V2_STATIC_MASK_ISSUANCE[receipt] = (
            _rsm_v2_static_mask_receipt_value(receipt)
        )
    return receipt


def _rsm_v2_chunk_owned_bytes(
    *,
    raw_bytes: int,
    float_bytes: int,
    pixel_count: int,
    mask_state_bytes: int,
    science: bool,
) -> int:
    if science:
        # Decoded copies + stacked raw, conditioned numerator, three q roots,
        # two feed payloads, the temporary finite mask, and the retained
        # cropped static mask for this member.
        return 2 * raw_bytes + 6 * float_bytes + pixel_count + mask_state_bytes
    # Mask pass: decoded copies + stacked raw + conditioned chunk plus the
    # retained first/same/above/cropped state for this member.
    return 2 * raw_bytes + float_bytes + mask_state_bytes


def _rsm_v2_correction_bytes(plan, frame_count: int, member) -> int:
    """A plan's per-frame corrections, charged with each science chunk (AUD-9 stage 2b S2).

    Measured 2026-10-01 with tracemalloc, every correction on: 72.7 B per detector pixel at Pilatus 100K, one
    pose (S2.2, ``logs/s2-2-weights-peak.log``: the polarization's lab-frame q and terms over the whole crop, and
    the weights). S2.3 bands the polarization (``_rsm_weight_band_rows``), so its working set is one band:
    charged 80 B per band pixel, or Int's threshold's 9 B per detector pixel (a float64 frame and its mask)
    when larger (they run at different moments). Also charged: one weight array of 8 B per cropped pixel (one
    per frame when the polarization follows the pose), and the run's decoded mask, dark, flat and solid angle.
    S2.3 measured (``logs/s23-weights-peak.log``): Pilatus 100K 80.6 B/px at one pose (one band covers that
    crop); a 2167 x 2070 detector 12.3 B/px with its weight array. None: 0, the charge unchanged."""
    bound = plan.corrections
    if bound is None:
        return 0
    return _rsm_v2_correction_charge(bound.corrections, bound.dark_receipt is not None,
                                     bound.flat_receipt is not None, member.detector_shape,
                                     member.cropped_shape, frame_count)


def _rsm_v2_correction_charge(value, dark: bool, flat: bool, detector_shape, cropped_shape,
                              frame_count: int) -> int:
    """``_rsm_v2_correction_bytes`` in shapes and flags, so the Preview charges a form before its plan exists
    (AUD-9 stage 2b S3a): ``value`` the ``RSMFrameCorrections``, ``dark``/``flat`` whether each is bound."""
    full = math.prod(detector_shape)
    cropped = math.prod(cropped_shape)
    resident = full * (1 + 8 * (bool(dark) + bool(flat) + int(value.solid_angle)))
    polarized = value.polarization_factor is not None
    # S2.3: the polarization's working set is one band of rows (``_rsm_weight_band_rows``), not the detector;
    # Int's threshold works on whole frames at another moment, so the larger of the two is charged
    rows, columns = cropped_shape
    band = min(int(rows), _rsm_weight_band_rows(columns)) * int(columns)
    working = max(80 * band if polarized else 0, 9 * full if value.threshold is not None else 0)
    weighted = value.solid_angle or polarized or bool(flat)
    # the weights: one detector array, or one per frame when the polarization follows the pose
    return resident + working + (8 * cropped * (frame_count if polarized else 1) if weighted else 0)


def rsm_v2_chunk_bytes(detector_shape, cropped_shape, frame_count: int, *, static_hot: bool,
                       corrections=None, dark: bool = False, flat: bool = False, detector_mask: bool = False) -> int:
    """The bytes a run charges one chunk of ``frame_count`` frames of one member against Max chunk MiB, in
    shapes and flags (AUD-9 stage 2b S3a): the larger of the science pass (decoded and conditioned frames, q,
    the feed and the per-frame corrections) and, with all-frame hot masking, the mask pass. The memory plan
    (``_require_rsm_v2_memory_plan``) and the Preview's Auto cap both charge through it. ``detector_mask``: the
    plan holds the calibrated detector's own mask (one bool a detector pixel) through both passes (S3c)."""
    full = math.prod(detector_shape)
    cropped = math.prod(cropped_shape)
    held = full * np.dtype(bool).itemsize if detector_mask else 0
    pixels = frame_count * full
    raw_bytes = pixels * np.dtype(np.float64).itemsize
    float_bytes = pixels * np.dtype(np.float64).itemsize
    science = _rsm_v2_chunk_owned_bytes(
        raw_bytes=raw_bytes,
        float_bytes=float_bytes,
        pixel_count=pixels,
        mask_state_bytes=(cropped * np.dtype(bool).itemsize if static_hot else 0) + held,
        science=True,
    ) + (0 if corrections is None
         else _rsm_v2_correction_charge(corrections, dark, flat, detector_shape, cropped_shape, frame_count))
    if not static_hot:
        return science
    mask_state_bytes = max(
        full * (np.dtype(np.float64).itemsize + 3 * np.dtype(bool).itemsize),
        full * np.dtype(bool).itemsize + cropped * np.dtype(bool).itemsize,
    ) + held
    return max(science, _rsm_v2_chunk_owned_bytes(
        raw_bytes=raw_bytes,
        float_bytes=float_bytes,
        pixel_count=pixels,
        mask_state_bytes=mask_state_bytes,
        science=False,
    ))


def _rsm_v2_correction_state_bytes(plan, members) -> int:
    """The run's one-off correction state, before the chunks (``_rsm_correction_state``, with the grid
    resident): the decoded mask, dark and flat, and pyFAI's solid angle with its transient, which is not a
    chunk's but bounds the run's peak too (AUD-9 stage 2b S3a). 0 with no corrections."""
    bound = plan.corrections
    if bound is None:
        return 0
    full = max(math.prod(member.detector_shape) for member in members)
    held = full * (1 + 8 * ((bound.dark_receipt is not None) + (bound.flat_receipt is not None)))
    return held + (full * _RSM_SOLID_ANGLE_TRANSIENT_BYTES_PER_PIXEL if bound.corrections.solid_angle else 0)


#: Auto Max chunk MiB's floor: the preset's 256 MiB (AUD-9 stage 2b S3a).
_RSM_AUTO_CHUNK_FLOOR_BYTES = 256 * 1024 * 1024


def rsm_v2_auto_chunking(detector_shape, cropped_shape, chunk_size: int, *, static_hot: bool,
                         corrections=None, detector_mask: bool = False) -> tuple[int, int]:
    """Auto Max chunk MiB (AUD-9 stage 2b S3a): the preset's 256 MiB, or one frame's charge rounded up to the
    next MiB when that is larger, so a default run never refuses for memory on a large detector (an Eiger 4M's
    plain frame alone charges 282.3 MiB); and the chunk holds as many frames as that cap fits, at most
    ``chunk_size`` and at least one. ``corrections``: the form's ``RSMFrameCorrections`` (its dark and flat
    charged when their paths are set); ``detector_mask``: the calibration's detector has its own mask, which the plan
    holds. Returns (chunk_size, max_chunk_bytes)."""
    flags = {
        "static_hot": static_hot,
        "corrections": corrections,
        "dark": corrections is not None and corrections.dark_path is not None,
        "flat": corrections is not None and corrections.flat_path is not None,
        "detector_mask": detector_mask,
    }
    mib = 1024 * 1024
    one = rsm_v2_chunk_bytes(detector_shape, cropped_shape, 1, **flags)
    cap = max(_RSM_AUTO_CHUNK_FLOOR_BYTES, -(-one // mib) * mib)
    frames = int(chunk_size)
    while frames > 1 and rsm_v2_chunk_bytes(detector_shape, cropped_shape, frames, **flags) > cap:
        frames -= 1
    return frames, cap


#: pyFAI's solid angle, computed once a run before the chunks (``_rsm_correction_state``) and not banded,
#: since banding would change its weights: its traced peak, output included, measured 2026-10-01
#: (``logs/s3a-solid-angle-transient.log``): 48.1 B/px at Pilatus 100K, 48.0 B/px at 2167 x 2070. Charged at 49.
#: A refined version-3 calibration's (``rsm_calibration_solid_angle``, Stitch's arm then the inverse turn) peaks
#: the same: 48.0 B/px at 195 x 1475, turned and not (``logs/s3c-solid-angle-transient.log``).
_RSM_SOLID_ANGLE_TRANSIENT_BYTES_PER_PIXEL = 49


def _load_rsm_v2_raw_chunk(
    source: object,
    labels: tuple[int, ...],
    member: RSMPreflightMemberV2,
    plan: RSMOperationPlanV2,
    *,
    science: bool,
    mask_state_bytes: int,
    cancel_token: threading.Event | None,
) -> np.ndarray:
    if not labels or len(labels) > plan.chunk_size:
        raise TypeError("RSM v2 chunk labels are invalid")
    # SpecSource's admitted decoder contract returns owned float64 frames even
    # when the authenticated on-disk raw encoding is an integer dtype.
    expected_decoded_dtype = np.dtype(np.float64)
    try:
        nexus = "entry" in json.loads(member.source_options_json)
    except (TypeError, json.JSONDecodeError) as error:
        raise RSMOperationRefused("RSM_MEMBER_IDENTITY_MISMATCH") from error
    native_dtype = None
    frames: list[np.ndarray] = []
    raw_bytes = 0
    pixels_per_frame = math.prod(member.detector_shape)
    try:
        for label in labels:
            if cancel_token is not None and cancel_token.is_set():
                raise RSMOperationRefused("CANCELLED")
            observed = np.asarray(source.load_frame(label))
            if (
                observed.dtype.kind not in "biuf"
                or observed.shape != member.detector_shape
            ):
                raise RSMOperationRefused("FRAME_LAYOUT_INVALID")
            if nexus:
                # a NeXus stack hands its dataset's native dtype, the same for every frame (C5c): decoded to
                # float64 here, exactly for integers
                if native_dtype is None:
                    native_dtype = observed.dtype
                elif observed.dtype != native_dtype:
                    raise RSMOperationRefused("FRAME_LAYOUT_INVALID")
                owned = np.array(observed, dtype=expected_decoded_dtype, copy=True, order="C")
            else:
                owned = np.array(observed, copy=True, order="C")
                if owned.dtype != expected_decoded_dtype:  # SPEC's decoder contract: float64 every frame
                    raise RSMOperationRefused("FRAME_LAYOUT_INVALID")
            observed = None
            float_frame_bytes = pixels_per_frame * np.dtype(np.float64).itemsize
            if (
                owned.nbytes > plan.max_frame_bytes
                or float_frame_bytes > plan.max_frame_bytes
            ):
                raise RSMOperationRefused("FRAME_MEMORY_LIMIT_EXCEEDED")
            frames.append(owned)
            raw_bytes += owned.nbytes
            float_bytes = len(frames) * float_frame_bytes
            required = _rsm_v2_chunk_owned_bytes(
                raw_bytes=raw_bytes,
                float_bytes=float_bytes,
                pixel_count=len(frames) * pixels_per_frame,
                mask_state_bytes=mask_state_bytes,
                science=science,
            ) + (_rsm_v2_correction_bytes(plan, len(frames), member) if science else 0)
            if required > plan.max_chunk_bytes:
                raise RSMOperationRefused("Q_MEMORY_LIMIT_EXCEEDED")
        if cancel_token is not None and cancel_token.is_set():
            raise RSMOperationRefused("CANCELLED")
        stacked = np.stack(frames, axis=0)
        if (
            type(stacked) is not np.ndarray
            or not stacked.flags.c_contiguous
            or stacked.shape != (len(labels), *member.detector_shape)
        ):
            raise RSMOperationRefused("FRAME_LAYOUT_INVALID")
        return stacked
    finally:
        frames.clear()


def _rsm_v2_contribution_angles(
    member: RSMPreflightMemberV2,
    contributions: tuple[RSMContribution, ...],
) -> tuple[np.ndarray, ...]:
    if (
        type(contributions) is not tuple
        or not contributions
        or tuple(role for role, _selector in member.geometry_binding.motor_selectors)
        != _PSIC_ROLES
    ):
        raise RSMOperationRefused("RSM_MEMBER_IDENTITY_MISMATCH")
    values = []
    for _role, selector in member.geometry_binding.motor_selectors:
        key = (selector.name, selector.occurrence)
        try:
            series = [
                _contribution_values(contribution)[key]
                for contribution in contributions
            ]
        except KeyError as error:
            raise RSMOperationRefused(
                "RSM_MEMBER_IDENTITY_MISMATCH"
            ) from error
        array = np.ascontiguousarray(series, dtype=np.float64)
        array.setflags(write=False)
        values.append(array)
    return tuple(values)


def _rsm_calibration_detector(receipt) -> str | None:
    """The calibration's detector name (pyFAI's), as Stitch's ``calibration_detector``; None for the
    canonical asset, which names none."""
    if type(receipt) is not RSMCalibrationGeometryReceipt:
        return None
    return receipt.calibration.projection.value["detector"].get("detector_factory_name")


def _rsm_member_saturation(threshold, members, detector):
    """Each member's automatic ceiling (decision 39), resolved as Stitch's ``_frame_facts`` does without a
    recorded value: the calibrated detector's documented counter limit for integer data, else the data type's
    rule (None; ``member_saturation``). A SPEC member's data type is its authenticated on-disk encoding; a
    NeXus stack's is learned only at the run (its declared float64 is a byte charge), so it is unknown here
    and the documented limit applies, as the resolver does for an unknown type."""
    if threshold is None or threshold.mode != "auto" or detector is None:
        return None
    from xrd_tools.analysis.stitch_native import member_saturation
    from xrd_tools.core.invalid import resolve_detector_saturation
    ceilings = []
    for member in members:
        try:
            options = json.loads(member.source_options_json)
            dtype = None if "entry" in options else np.dtype(options["read_image_kwargs"]["raw_dtype"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            dtype = None
        ceilings.append(resolve_detector_saturation(detector=detector, dtype=dtype))
    return member_saturation(ceilings)


def _bind_rsm_corrections(corrections, members, max_frame_bytes, *, detector=None, detector_mask=None):
    """Bind a plan's corrections at Preview: Stitch's mask capture and Int-admitted dark and flat
    (``capture_detector_image``), each the members' full detector shape, and an Auto threshold's ceilings
    from ``detector``, the calibrated detector's name (``_rsm_member_saturation``). ``detector_mask``, its own
    mask in the raw frame, is left out of the dark's and flat's usability counts, as Int passes it. None: None."""
    if type(corrections) not in {RSMFrameCorrections, type(None)}:
        raise TypeError("RSM corrections must be exact RSMFrameCorrections")
    if corrections is None or corrections == RSMFrameCorrections():  # every field at its default: none
        return None
    from xrd_tools.analysis.stitch_native import capture_detector_image, capture_mask
    shapes = {tuple(member.detector_shape) for member in members}
    if len(shapes) != 1:
        raise RSMOperationRefused("RSM_CORRECTION_IMAGE_INVALID", "the members' detectors differ")
    shape, = shapes
    if detector_mask is not None and tuple(detector_mask.shape) != shape:
        raise RSMOperationRefused("RSM_CORRECTION_IMAGE_INVALID",
                                  f"the calibration's detector is {detector_mask.shape}, the members' {shape}")
    mask = dark = flat = None
    try:
        if corrections.mask_path is not None:
            mask = capture_mask(corrections.mask_path)
            if tuple(mask.shape) != shape:
                raise RSMOperationRefused("RSM_MASK_INVALID", f"the mask is {mask.shape}, the detector {shape}")
    except ValueError as error:
        if type(error) is RSMOperationRefused:
            raise
        raise RSMOperationRefused("RSM_MASK_INVALID", str(error)) from None
    for role in ("dark", "flat"):
        path = getattr(corrections, f"{role}_path")
        if path is None:
            continue
        try:
            receipt, pixels = capture_detector_image(path, role, max_frame_bytes, corrections.flat_min_fraction,
                                                     detector_mask=detector_mask)
        except ValueError as error:
            raise RSMOperationRefused("RSM_CORRECTION_IMAGE_INVALID", str(error)) from None
        if tuple(pixels.shape) != shape:
            raise RSMOperationRefused("RSM_CORRECTION_IMAGE_INVALID",
                                      f"the {role} is {pixels.shape}, the detector {shape}")
        if role == "dark":
            dark = receipt
        else:
            flat = receipt
    return RSMBoundCorrections(corrections, mask, dark, flat,
                               _rsm_member_saturation(corrections.threshold, members, detector),
                               None if dark is None and flat is None else detector_mask)


@dataclass(frozen=True, slots=True)
class _RSMCorrectionState:
    """A run's corrections, decoded once: the static invalid pixels (Mask File, unusable dark or flat
    pixels), the neutral-filled dark and flat, the solid angle, the polarization axes."""

    bound: RSMBoundCorrections
    invalid: np.ndarray | None
    dark: np.ndarray | None
    flat: np.ndarray | None
    solid: np.ndarray | None
    axes: tuple | None


def _rsm_correction_state(plan, header, diffractometer, asset=None) -> _RSMCorrectionState | None:
    """Decode a plan's corrections for its run (each image verified against its Preview receipt). The solid
    angle: a refined version-3 calibration's (``asset``; its mount, as Stitch's arm), else the untilted formula
    about the beam pixel."""
    bound = plan.corrections
    if bound is None:
        return None
    from xrd_tools.analysis.stitch_native import capture_detector_image, revalidate_mask
    from xrd_tools.corrections.frame_preparation import (
        correction_image, flat_detector_solid_angle, polarization_axes,
    )
    shape = (int(header.Nch1), int(header.Nch2))
    invalid = None
    if bound.mask_receipt is not None:
        revalidate_mask(bound.mask_receipt)
        invalid = np.frombuffer(bound.mask_receipt.pixels, dtype=bool).reshape(shape).copy()
    images = {}
    for role in ("dark", "flat"):
        receipt = getattr(bound, f"{role}_receipt")
        if receipt is None:
            images[role] = None
            continue
        current, pixels = capture_detector_image(receipt.lexical_path, role, plan.max_frame_bytes,
                                                 bound.corrections.flat_min_fraction,
                                                 detector_mask=bound.detector_mask)
        if current != receipt:
            raise RSMOperationRefused("RSM_CORRECTION_IMAGE_INVALID", f"the {role} no longer matches Preview")
        values, unusable = correction_image(pixels, role, min_fraction=bound.corrections.flat_min_fraction)
        images[role] = values
        if unusable is not None:
            invalid = unusable if invalid is None else np.logical_or(invalid, unusable)
    solid = None
    if bound.corrections.solid_angle:
        try:
            solid = rsm_calibration_solid_angle(asset)
        except RSMGeometryAssetRefused as error:
            raise RSMOperationRefused(error.code, str(error)) from None
        if solid is None:
            solid = flat_detector_solid_angle(shape, distance=float(header.distance),
                                              cch=(float(header.cch1), float(header.cch2)),
                                              pixel_size=(float(header.pwidth1), float(header.pwidth2)))
        elif tuple(solid.shape) != shape:
            raise RSMOperationRefused("RSM_CALIBRATION_UNSUPPORTED",
                                      f"the calibration's solid angle is {solid.shape}, the detector {shape}")
    axes = None
    if bound.corrections.polarization_factor is not None:
        # the outer detector circle (nu) is the vertical; the Diffractometer's or a legacy config's stack
        circles = getattr(diffractometer, "detector_circles", None) or diffractometer.detector_rot
        axes = polarization_axes(diffractometer.r_i, circles[0])
    return _RSMCorrectionState(bound, invalid, images["dark"], images["flat"], solid, axes)


#: The rows of one band of the polarization's lab-frame q and terms, in detector pixels: its working set (about
#: 64 B a pixel, measured) is bounded by this whatever the detector, where an unbanded Eiger 4M needed 359 MB
#: (AUD-9 stage 2b S2.3).
_RSM_WEIGHT_BAND_PIXELS = 1 << 18


def _rsm_weight_band_rows(cropped_columns: int) -> int:
    return max(1, _RSM_WEIGHT_BAND_PIXELS // max(1, int(cropped_columns)))


def _rsm_chunk_weights(state, mapper, session, angles, energy, roi):
    """One chunk's normalization weights (solid angle × polarization × flat), cropped to the ROI: one detector
    array for every frame when they do not depend on the detector pose (no polarization, or one pose), one per
    frame otherwise; None: no weight (1, as before).

    The polarization is computed in bands of rows (``_rsm_weight_band_rows``), each through ``pixel_q``'s
    ``rows`` with the crop's own beam centre, so every weight is bit-identical to one whole-detector call;
    ``frame_weights``' order (solid × polarization, then × flat) is kept band by band, and each distinct pose
    is computed once."""
    from xrd_tools.corrections.frame_preparation import frame_weights, polarization_weight
    if state is None or (state.solid is None and state.axes is None and state.flat is None):
        return None
    rows, columns = int(mapper.header.Nch1), int(mapper.header.Nch2)
    r0, r1, c0, c1 = roi if roi is not None else (0, rows, 0, columns)
    r1, c1 = rows + r1 if r1 < 0 else r1, columns + c1 if c1 < 0 else c1  # as DetectorHeader.with_roi
    shape = (r1 - r0, c1 - c0)
    base = None if state.solid is None else state.solid[r0:r1, c0:c1]
    flat = None if state.flat is None else state.flat[r0:r1, c0:c1]
    if state.axes is None:  # no polarization: the same weights at every pose
        weights = frame_weights(np.ones(shape) if base is None else base, flat=flat)
        if weights is base or weights.base is not None or weights is state.solid:
            weights = np.array(weights, copy=True)
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        weights.setflags(write=False)
        return weights
    nu, delta = (np.asarray(angles[index], dtype=np.float64) for index in (4, 5))
    poses = [(float(a), float(b)) for a, b in zip(nu, delta)]
    weights = np.empty(shape if len(set(poses)) == 1 else (len(poses), *shape), dtype=np.float64)
    xu_module, _numpy = session.active_modules()
    # the energy q is mapped with: a refined calibration's offset included (C7b), so the scatter is elastic
    wavelength = float(xu_module.en2lam(float(mapper.mapped_energy(energy))))
    factor = state.bound.corrections.polarization_factor
    band = _rsm_weight_band_rows(shape[1])
    first = {}
    for index, pose in enumerate(poses):
        target = weights if weights.ndim == 2 else weights[index]
        if pose in first:
            if weights.ndim == 3:
                target[...] = weights[first[pose]]
            continue
        first[pose] = index
        # the detector pose alone, in the lab frame (the sample circles at zero, an identity UB): the elastic
        # scattered direction of each pixel
        lab = tuple(np.zeros(1) for _ in range(4)) + (np.array([pose[0]]), np.array([pose[1]]))
        for start in range(0, shape[0], band):
            stop = min(start + band, shape[0])
            qx, qy, qz = mapper.pixel_q(lab, float(energy), UB=np.eye(3), roi=roi, runtime_session=session,
                                        rows=(start, stop))
            polarization = polarization_weight(
                np.asarray(qx)[0], np.asarray(qy)[0], np.asarray(qz)[0], factor=factor,
                wavelength_A=wavelength, beam=state.axes[0], vertical=state.axes[1])
            qx = qy = qz = None
            block = target[start:stop]
            if base is None:
                block[...] = polarization  # frame_weights' 1 × polarization
            else:
                np.multiply(base[start:stop], polarization, out=block)
            if flat is not None:
                np.multiply(block, flat[start:stop], out=block)
            polarization = None
    weights.setflags(write=False)
    return weights


def _apply_rsm_frame_corrections(raw, conditioned, state, divisors, saturation=None):
    """A chunk's per-frame corrections, in place on its conditioned float64 frames: Int's threshold on the
    native values (Auto with the member's resolved ``saturation``, as Stitch passes it), the static invalid
    pixels, then the dark and the frame's divisor (the shared ``prepare_frame_signal``, Stitch's)."""
    from xrd_tools.corrections.frame_preparation import FramePreparationRefused, prepare_frame_signal
    from xrd_tools.integrate.stitch_conditioning import condition_stitch_frame
    threshold = state.bound.corrections.threshold
    for index in range(conditioned.shape[0]):
        frame = conditioned[index]
        if threshold is not None:
            frame[np.isnan(condition_stitch_frame(raw[index], threshold, saturation))] = np.nan
        if state.invalid is not None:
            frame[state.invalid] = np.nan
        try:
            prepare_frame_signal(frame, dark=state.dark, divisor=float(divisors[index]))
        except FramePreparationRefused:
            raise RSMOperationRefused("NORMALIZATION_RESULT_INVALID") from None
    if np.any(np.isinf(conditioned)):
        raise RSMOperationRefused("NORMALIZATION_RESULT_INVALID")


def _make_rsm_v2_chunk_lease(
    source: object,
    member: RSMPreflightMemberV2,
    plan: RSMOperationPlanV2,
    contributions: tuple[RSMContribution, ...],
    *,
    cropped_static_mask: np.ndarray | None,
    cancel_token: threading.Event | None,
    corrections: _RSMCorrectionState | None = None,
) -> RSMGridChunkLease:
    """Build one non-generator chunk whose image roots are owned by its lease."""

    labels = tuple(item.label for item in contributions)
    raw = _load_rsm_v2_raw_chunk(
        source,
        labels,
        member,
        plan,
        science=True,
        mask_state_bytes=(
            0 if cropped_static_mask is None else cropped_static_mask.nbytes
        ) + (0 if plan.detector_mask is None else plan.detector_mask.nbytes),
        cancel_token=cancel_token,
    )
    conditioned = condition_rsm_images(raw, plan.conditioning)
    if plan.detector_mask is not None:  # the calibrated detector's own gaps, whatever the corrections (S3c)
        np.copyto(conditioned, np.nan, where=plan.detector_mask)
    if cropped_static_mask is not None:
        roi = plan.effective_geometry.roi
        if (
            type(cropped_static_mask) is not np.ndarray
            or cropped_static_mask.dtype != np.dtype(bool)
            or cropped_static_mask.shape != member.cropped_shape
            or cropped_static_mask.flags.writeable
            or type(roi) is not tuple
            or len(roi) != 4
        ):
            raise RSMOperationRefused("RSM_STATIC_MASK_IDENTITY_MISMATCH")
        r0, r1, c0, c1 = roi
        cropped_view = conditioned[:, r0:r1, c0:c1]
        if cropped_view.shape[1:] != cropped_static_mask.shape:
            raise RSMOperationRefused("RSM_STATIC_MASK_IDENTITY_MISMATCH")
        cropped_view[:, cropped_static_mask] = np.nan
        cropped_view = None
    divisors = np.ascontiguousarray(
        [item.normalization_divisor for item in contributions],
        dtype=np.float64,
    )
    if corrections is not None:  # the shared per-frame corrections (AUD-9 stage 2b)
        saturation = corrections.bound.saturation
        _apply_rsm_frame_corrections(raw, conditioned, corrections, divisors,
                                     None if saturation is None else saturation[member.ordinal])
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            conditioned /= divisors[:, None, None]
    divisors = None
    if np.any(np.isinf(conditioned)):
        raise RSMOperationRefused("NORMALIZATION_RESULT_INVALID")
    lease = RSMGridChunkLease.from_arrays(raw, conditioned, len(labels))
    raw = conditioned = None
    return lease


def _derive_rsm_v2_static_mask(
    source: object,
    member: RSMPreflightMemberV2,
    plan: RSMOperationPlanV2,
    *,
    cancel_token: threading.Event | None,
    frame_callback: Callable[[int], object] | None,
) -> tuple[np.ndarray | None, np.ndarray | None, RSMStaticMaskReceipt]:
    threshold = plan.conditioning.static_hot_threshold
    if threshold is None:
        receipt = _make_rsm_static_mask_receipt(
            member,
            plan.conditioning,
            None,
            None,
        )
        return None, None, receipt
    first: np.ndarray | None = None
    same: np.ndarray | None = None
    above: np.ndarray | None = None
    completed = 0
    full_pixels = math.prod(member.detector_shape)
    cropped_pixels = math.prod(member.cropped_shape)
    mask_state_bytes = max(
        full_pixels
        * (
            np.dtype(np.float64).itemsize
            + 3 * np.dtype(bool).itemsize
        ),
        full_pixels * np.dtype(bool).itemsize
        + cropped_pixels * np.dtype(bool).itemsize,
    ) + (0 if plan.detector_mask is None else plan.detector_mask.nbytes)
    contributions = member.contributions
    for start in range(0, len(contributions), plan.chunk_size):
        if cancel_token is not None and cancel_token.is_set():
            raise RSMOperationRefused("CANCELLED")
        selected = contributions[start : start + plan.chunk_size]
        raw = _load_rsm_v2_raw_chunk(
            source,
            tuple(item.label for item in selected),
            member,
            plan,
            science=False,
            mask_state_bytes=mask_state_bytes,
            cancel_token=cancel_token,
        )
        conditioned = condition_rsm_images(raw, plan.conditioning)
        raw = None
        for frame in conditioned:
            if cancel_token is not None and cancel_token.is_set():
                raise RSMOperationRefused("CANCELLED")
            if first is None:
                first = np.array(frame, dtype=np.float64, copy=True, order="C")
                same = np.ones(member.detector_shape, dtype=bool)
                above = first > threshold
            else:
                assert same is not None and above is not None
                same &= frame == first
                above &= frame > threshold
            completed += 1
            if frame_callback is not None:
                try:
                    frame_callback(completed)
                except Exception:
                    pass
        conditioned = None
    if (
        completed != len(contributions)
        or first is None
        or same is None
        or above is None
    ):
        raise RSMOperationRefused("RSM_STATIC_MASK_INVALID")
    full_mask = np.ascontiguousarray(same & above, dtype=bool)
    full_mask.setflags(write=False)
    first = same = above = None
    cropped_mask = _rsm_v2_cropped_static_mask(
        full_mask,
        plan.effective_geometry.roi,
        member.cropped_shape,
    )
    receipt = _make_rsm_static_mask_receipt(
        member,
        plan.conditioning,
        full_mask,
        cropped_mask,
    )
    return full_mask, cropped_mask, receipt


def _require_rsm_v2_memory_plan(request: RSMOperationRequestV2) -> None:
    plan = request.plan
    voxels = math.prod(plan.bins)
    fixed_grid_bytes = 4 * voxels * np.dtype(np.float64).itemsize
    stored_projection_bytes = (
        fixed_grid_bytes
        + voxels * np.dtype(np.float64).itemsize
        + 2 * voxels * np.dtype(np.float32).itemsize
    )
    public_grid_finalization_bytes = (
        fixed_grid_bytes
        + 2 * voxels * np.dtype(np.float64).itemsize
        + voxels * np.dtype(bool).itemsize
    )
    finalization_bytes = max(
        stored_projection_bytes,
        public_grid_finalization_bytes,
    )
    peak_chunk_bytes = 0
    for member in request.preflight.members:
        try:
            options = json.loads(member.source_options_json)
            declared_raw_dtype = (
                np.dtype(np.float64) if "entry" in options  # a NeXus member (C5c)
                else np.dtype(options["read_image_kwargs"]["raw_dtype"])
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RSMOperationRefused("RSM_MEMBER_IDENTITY_MISMATCH") from error
        if declared_raw_dtype.kind not in "biuf":
            raise RSMOperationRefused("RSM_MEMBER_IDENTITY_MISMATCH")
        frame_count = min(plan.chunk_size, len(member.contributions))
        pixels = frame_count * math.prod(member.detector_shape)
        raw_bytes = pixels * np.dtype(np.float64).itemsize
        float_bytes = pixels * np.dtype(np.float64).itemsize
        if (
            raw_bytes // frame_count > plan.max_frame_bytes
            or float_bytes // frame_count > plan.max_frame_bytes
        ):
            raise RSMOperationRefused("FRAME_MEMORY_LIMIT_EXCEEDED")
        bound = plan.corrections
        required = rsm_v2_chunk_bytes(
            member.detector_shape,
            member.cropped_shape,
            frame_count,
            static_hot=plan.conditioning.static_hot_threshold is not None,
            corrections=None if bound is None else bound.corrections,
            dark=bound is not None and bound.dark_receipt is not None,
            flat=bound is not None and bound.flat_receipt is not None,
            detector_mask=plan.detector_mask is not None,
        )
        if required > plan.max_chunk_bytes:
            mib = 1024 * 1024
            raise RSMOperationRefused(
                "Q_MEMORY_LIMIT_EXCEEDED",
                f"one chunk of {frame_count} frame{'s' if frame_count != 1 else ''} needs "
                f"{-(-required // mib)} MiB; Max chunk MiB is {plan.max_chunk_bytes // mib}: raise it, lower "
                "Chunk, or leave Max chunk MiB on Auto",
            )
        peak_chunk_bytes = max(peak_chunk_bytes, required)
    transient_bytes = (fixed_grid_bytes + _rsm_v2_correction_state_bytes(plan, request.preflight.members)
                       + (0 if plan.detector_mask is None else plan.detector_mask.nbytes))
    resident_bytes = max(
        fixed_grid_bytes + peak_chunk_bytes,
        finalization_bytes,
        transient_bytes,
    )
    if resident_bytes >= _MAX_RSM_RESIDENT_BYTES:
        raise RSMOperationRefused("Q_MEMORY_LIMIT_EXCEEDED")


def required_rsm_selectors_v2(
    motor_selectors: tuple[tuple[str, MetadataColumnSelector], ...],
    normalization: RSMNormalizationPolicy,
) -> tuple[MetadataColumnSelector, ...]:
    if type(motor_selectors) is not tuple or type(normalization) is not RSMNormalizationPolicy:
        raise TypeError("RSM v2 selector projection requires exact values")
    requested = [selector for _role, selector in motor_selectors]
    if normalization.foil_selector is not None:
        requested.append(normalization.foil_selector)
    if normalization.exposure_selector is not None:
        requested.append(normalization.exposure_selector)
    unique = {
        (selector.name, selector.occurrence): selector for selector in requested
    }
    return tuple(sorted(unique.values(), key=lambda item: (item.name, item.occurrence)))


def _resolve_exact_rsm_q_bounds_active(
    mapper: PixelQMap,
    angles: tuple[np.ndarray, ...] | list[np.ndarray],
    energy_eV: float,
    ub: np.ndarray,
    *,
    roi: tuple[int, int, int, int] | None,
    chunk_size: int,
    max_frame_bytes: int,
    max_chunk_bytes: int,
    runtime_session: object,
    cancel_token: threading.Event | None = None,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Resolve q bounds under one already-active group runtime owner."""

    from xrd_tools.core.geometry.xu_runtime import (
        require_active_xu_runtime_session,
    )

    session = require_active_xu_runtime_session(runtime_session)
    if type(mapper) is not PixelQMap:
        raise TypeError("RSM v2 q-bound resolution requires exact PixelQMap")
    energy = _finite_float(energy_eV, "RSM energy", positive=True)
    matrix = np.asarray(ub, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise RSMOperationRefused("RSM_MEMBER_UB_INVALID")
    if type(chunk_size) is not int or not 1 <= chunk_size <= _MAX_CHUNK_SIZE:
        raise ValueError("RSM q-bound chunk size is outside the R2 bound")
    if (
        type(max_frame_bytes) is not int
        or not 1 <= max_frame_bytes <= _MAX_FRAME_BYTES
        or type(max_chunk_bytes) is not int
        or not 1 <= max_chunk_bytes <= _MAX_CHUNK_BYTES
    ):
        raise ValueError("RSM q-bound memory limit is outside the R2 bound")
    if cancel_token is not None and type(cancel_token) is not threading.Event:
        raise TypeError("RSM cancellation token must be exact threading.Event")
    if type(angles) not in {tuple, list} or len(angles) != 6:
        raise RSMOperationRefused("RSM_MEMBER_Q_BOUNDS_INVALID")
    values = tuple(np.asarray(item, dtype=np.float64) for item in angles)
    if not values or values[0].ndim != 1 or len(values[0]) < 1:
        raise RSMOperationRefused("RSM_MEMBER_Q_BOUNDS_INVALID")
    frame_count = len(values[0])
    if any(
        item.ndim != 1
        or len(item) != frame_count
        or not np.all(np.isfinite(item))
        for item in values
    ):
        raise RSMOperationRefused("RSM_MEMBER_Q_BOUNDS_INVALID")
    header = mapper.header if roi is None else mapper.header.with_roi(roi)
    if header.Nch1 < 2 or header.Nch2 < 2:
        raise RSMOperationRefused("RSM_MEMBER_Q_BOUNDS_INVALID")
    conditioned_frame_bytes = (
        mapper.header.Nch1
        * mapper.header.Nch2
        * np.dtype(np.float64).itemsize
    )
    if conditioned_frame_bytes > max_frame_bytes:
        raise RSMOperationRefused(
            "FRAME_MEMORY_LIMIT_EXCEEDED",
            "RSM full conditioned detector frame exceeds max_frame_bytes",
        )
    q_plane_bytes = header.Nch1 * header.Nch2 * np.dtype(np.float64).itemsize
    q_chunk_bytes = 3 * q_plane_bytes * min(frame_count, chunk_size)
    if q_chunk_bytes > max_chunk_bytes:
        raise RSMOperationRefused(
            "Q_MEMORY_LIMIT_EXCEEDED",
            "RSM exact q preflight exceeds max_chunk_bytes",
        )
    session.require_active()
    try:
        # a refined calibration's energy offset, mount and zero offsets (AUD-9 stage 2b C7b), as pixel_q takes them
        hxrd = mapper.diff_config.make_hxrd(mapper.mapped_energy(energy))
    except (RSMOperationRefused, XuRuntimeUnsupported):
        raise
    except Exception:
        raise RSMOperationRefused("RSM_MEMBER_Q_BOUNDS_INVALID") from None
    session.require_active()
    try:
        hxrd.Ang2Q.init_area(
            mapper.diff_config.init_area_detrot,
            mapper.diff_config.init_area_tiltazimuth,
            cch1=float(header.cch1),
            cch2=float(header.cch2),
            pwidth1=float(header.pwidth1),
            pwidth2=float(header.pwidth2),
            distance=float(header.distance),
            Nch1=int(header.Nch1),
            Nch2=int(header.Nch2),
            **mapper.init_area_extra(),
        )
    except (RSMOperationRefused, XuRuntimeUnsupported):
        raise
    except Exception:
        raise RSMOperationRefused("RSM_MEMBER_Q_BOUNDS_INVALID") from None
    lows = np.full(3, np.inf, dtype=np.float64)
    highs = np.full(3, -np.inf, dtype=np.float64)
    values = mapper.area_angles(values)
    for start in range(0, frame_count, chunk_size):
        if cancel_token is not None and cancel_token.is_set():
            raise RSMOperationRefused("CANCELLED")
        session.require_active()
        stop = min(start + chunk_size, frame_count)
        try:
            q_values = hxrd.Ang2Q.area(
                *(item[start:stop] for item in values),
                UB=matrix,
                **mapper.diff_config.ang2q_kwargs,
            )
            expected = (stop - start, header.Nch1, header.Nch2)
            admitted: list[np.ndarray] = []
            for value in q_values:
                array = np.asarray(value)
                if expected[0] == 1 and array.shape == expected[1:]:
                    array = array.reshape(expected)
                if array.shape != expected or not np.all(np.isfinite(array)):
                    raise RSMOperationRefused("RSM_MEMBER_Q_BOUNDS_INVALID")
                admitted.append(array)
            if len(admitted) != 3:
                raise RSMOperationRefused("RSM_MEMBER_Q_BOUNDS_INVALID")
        except (RSMOperationRefused, XuRuntimeUnsupported):
            raise
        except Exception:
            raise RSMOperationRefused("RSM_MEMBER_Q_BOUNDS_INVALID") from None
        for axis, item in enumerate(admitted):
            lows[axis] = min(lows[axis], float(np.min(item)))
            highs[axis] = max(highs[axis], float(np.max(item)))
    if any(
        not math.isfinite(float(lo))
        or not math.isfinite(float(hi))
        or hi <= lo
        for lo, hi in zip(lows, highs, strict=True)
    ):
        raise RSMOperationRefused("RSM_MEMBER_Q_BOUNDS_INVALID")
    return tuple(
        (float(lo), float(hi)) for lo, hi in zip(lows, highs, strict=True)
    )  # type: ignore[return-value]


def prepare_rsm_operation_v2(
    source_group: ModuleSourceGroupReceipt,
    output: ModuleOutputRequest,
    geometry_asset_receipt: RSMGeometryAssetReceipt | RSMCalibrationGeometryReceipt,
    member_form_fingerprints: tuple[str, ...],
    member_motor_selectors: tuple[
        tuple[tuple[str, MetadataColumnSelector], ...], ...
    ],
    conditioning: RSMImageConditioning,
    normalization: RSMNormalizationPolicy,
    bins: tuple[int, int, int],
    *,
    chunk_size: int,
    max_frame_bytes: int,
    max_chunk_bytes: int,
    project_root: str | Path,
    coordinate_frame: RSMCoordinateFrame = RSMCoordinateFrame.HKL,
    cancel_token: threading.Event | None = None,
    corrections: RSMFrameCorrections | None = None,
    ub_override: RSMUBOverride | None = None,
) -> RSMOperationRequestV2:
    """Build one ordered, image-free RSM v2 Preview transaction. ``corrections`` (``RSMFrameCorrections``)
    binds the Mask File, dark and flat here, as Stitch's Preview does. ``ub_override`` (``RSMUBOverride``, HKL
    only) replaces every member's source UB, typed or from a named SPEC scan."""

    if (
        type(source_group) is not ModuleSourceGroupReceipt
        or source_group.kind is not ModuleKind.RSM
        or type(output) is not ModuleOutputRequest
        or output.kind is not AnalysisArtifactKind.RSM
        or type(geometry_asset_receipt) not in _RSM_GEOMETRY_RECEIPTS
        or type(member_form_fingerprints) is not tuple
        or type(member_motor_selectors) is not tuple
        or len(member_form_fingerprints) != len(source_group.members)
        or len(member_motor_selectors) != len(source_group.members)
        or any(
            type(item) is not str
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in member_form_fingerprints
        )
        or type(conditioning) is not RSMImageConditioning
        or type(normalization) is not RSMNormalizationPolicy
        or type(bins) is not tuple
        or type(chunk_size) is not int
        or type(max_frame_bytes) is not int
        or (max_chunk_bytes is not None and type(max_chunk_bytes) is not int)
        or type(coordinate_frame) is not RSMCoordinateFrame
    ):
        raise TypeError("RSM v2 preparation requires exact grouped values")
    # REPLACE regenerates a same-kind RSM result in place; the publisher refuses a
    # foreign or other-kind file at the target (AUD-9 stage 2, Stitch's Replace).
    if len(set(member_form_fingerprints)) != len(member_form_fingerprints):
        raise RSMOperationRefused(
            "RSM_SOURCE_GROUP_INVALID",
            "RSM v2 member form fingerprints must be unique",
        )
    if (
        len(bins) != 3
        or any(
            type(value) is not int or not 2 <= value <= _MAX_AXIS_POINTS
            for value in bins
        )
        or math.prod(bins) > _MAX_RSM_VOXELS
    ):
        raise RSMOperationRefused(
            "RSM_COMMON_GRID_INVALID",
            "RSM v2 bins are outside the common-grid bounds",
        )
    if not 1 <= chunk_size <= _MAX_CHUNK_SIZE:
        raise RSMOperationRefused(
            "Q_MEMORY_LIMIT_EXCEEDED",
            "RSM v2 chunk size is outside the q-memory bound",
        )
    if not 1 <= max_frame_bytes <= _MAX_FRAME_BYTES:
        raise RSMOperationRefused(
            "FRAME_MEMORY_LIMIT_EXCEEDED",
            "RSM v2 frame byte limit is outside the absolute bound",
        )
    if max_chunk_bytes is not None and not 1 <= max_chunk_bytes <= _MAX_CHUNK_BYTES:
        raise RSMOperationRefused(
            "Q_MEMORY_LIMIT_EXCEEDED",
            "RSM v2 chunk byte limit is outside the absolute bound",
        )
    if cancel_token is not None and type(cancel_token) is not threading.Event:
        raise TypeError("RSM cancellation token must be exact threading.Event")
    root = _project_root(project_root)
    if geometry_asset_receipt.project_root != str(root):
        raise RSMOperationRefused(
            "RSM_GEOMETRY_ASSET_IDENTITY_MISMATCH",
            "geometry asset is not bound to the selected Project",
        )
    output = _bind_project_output(output, root)
    # Preview creates no folder: the run makes it (_create_rsm_v2_output_folder)
    output_authority = _capture_output_authority(output, root, allow_absent_parent=True)
    bound_ub = _bind_rsm_ub(ub_override, coordinate_frame)  # after the cheap checks: it reads a file

    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session

    # the corrections the plan will bind (every field at its default is none, as _bind_rsm_corrections says),
    # so the Preview charges a chunk exactly as the run's memory plan will
    charged = None if corrections is None or corrections == RSMFrameCorrections() else corrections
    with xu_runtime_session() as runtime_session:
        effective = lower_rsm_effective_geometry(geometry_asset_receipt)
        # the calibrated detector's own mask (module gaps), built once (S3c unit 4b): every frame's mask whatever
        # the corrections, the dark and flat checks' exclusion, and memory the plan holds and charges
        detector_mask = rsm_calibration_detector_mask(geometry_asset_receipt)
        if max_chunk_bytes is None:  # Auto: sized from the detector before any member is charged (S3a)
            header = effective.detector_header
            cropped = header if effective.roi is None else header.with_roi(effective.roi)
            chunk_size, max_chunk_bytes = rsm_v2_auto_chunking(
                (int(header.Nch1), int(header.Nch2)), (int(cropped.Nch1), int(cropped.Nch2)), chunk_size,
                static_hot=conditioning.static_hot_threshold is not None, corrections=charged,
                detector_mask=detector_mask is not None)
            if max_chunk_bytes > _MAX_CHUNK_BYTES:
                raise RSMOperationRefused(
                    "Q_MEMORY_LIMIT_EXCEEDED",
                    "one frame of this detector exceeds the absolute chunk bound",
                )
        else:  # the user's cap: one frame must fit it, said with what it needs before any member is read (S3a)
            header = effective.detector_header
            cropped = header if effective.roi is None else header.with_roi(effective.roi)
            one = rsm_v2_chunk_bytes(
                (int(header.Nch1), int(header.Nch2)), (int(cropped.Nch1), int(cropped.Nch2)), 1,
                static_hot=conditioning.static_hot_threshold is not None, corrections=charged,
                dark=charged is not None and charged.dark_path is not None,
                flat=charged is not None and charged.flat_path is not None,
                detector_mask=detector_mask is not None)
            if one > max_chunk_bytes:
                mib = 1024 * 1024
                raise RSMOperationRefused(
                    "Q_MEMORY_LIMIT_EXCEEDED",
                    f"one frame needs {-(-one // mib)} MiB; Max chunk MiB is {max_chunk_bytes // mib}: raise it, "
                    "or leave Max chunk MiB on Auto",
                )
        mapper = rsm_effective_pixel_q_map(effective)
        bindings = tuple(
            bind_rsm_member_geometry(
                effective,
                member_ordinal=ordinal,
                motor_selectors=motor_selectors,
            )
            for ordinal, motor_selectors in enumerate(member_motor_selectors)
        )
        members: list[RSMPreflightMemberV2] = []
        for ordinal, (source, binding, form_fingerprint) in enumerate(
            zip(
                source_group.members,
                bindings,
                member_form_fingerprints,
                strict=True,
            )
        ):
            if cancel_token is not None and cancel_token.is_set():
                raise RSMOperationRefused("CANCELLED")
            required = required_rsm_selectors_v2(
                binding.motor_selectors,
                normalization,
            )
            if source.resolved_selectors != required:
                raise RSMOperationRefused(
                    "SOURCE_PROJECTION_MISMATCH",
                    "member source selectors do not match the RSM v2 policies",
                )
            expected_shape = (
                effective.detector_header.Nch1,
                effective.detector_header.Nch2,
            )
            facts = _capture_preflight_facts(
                source,
                _fresh_table(source, cancel_token=cancel_token),
                root,
                selectors=required,
                motor_selectors=binding.motor_selectors,
                detector_header=effective.detector_header,
                roi=effective.roi,
                conditioning=conditioning,
                normalization=normalization,
                chunk_size=chunk_size,
                max_frame_bytes=max_frame_bytes,
                max_chunk_bytes=max_chunk_bytes,
                expected_detector_shape=expected_shape,
                invalid_energy_code="RSM_MEMBER_ENERGY_INVALID",
                invalid_ub_code="RSM_MEMBER_UB_INVALID",
                normalize_source_fact_errors=True,
                normalize_ub_errors=True,
                mapper=mapper,
                active_runtime_session=runtime_session,
                allow_single_static_hot=True,
                cancel_token=cancel_token,
                coordinate_frame=coordinate_frame,
                ub_override=bound_ub,
            )
            try:
                projected = json.loads(facts.source_options_json)
                read_options = (
                    {"detector_shape": facts.detector_shape} if "entry" in projected  # NeXus (C5c)
                    else projected["read_image_kwargs"]
                )
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise RSMOperationRefused("RSM_MEMBER_IDENTITY_MISMATCH") from error
            if (
                type(read_options) is not dict
                or tuple(read_options.get("detector_shape", ())) != expected_shape
                or facts.detector_shape != expected_shape
            ):
                raise RSMOperationRefused(
                    "RSM_MEMBER_IDENTITY_MISMATCH",
                    "member detector shape differs from effective geometry",
                )
            members.append(
                _make_rsm_preflight_member_v2(
                    facts,
                    ordinal=ordinal,
                    member_form_fingerprint=form_fingerprint,
                    binding=binding,
                    normalization=normalization,
                    conditioning=conditioning,
                )
            )
        member_tuple = tuple(members)
        _require_unique_rsm_physical_contributions_v2(member_tuple)
        common_grid = make_rsm_common_grid(
            tuple(item.member_q_bounds for item in member_tuple),
            bins,
            coordinate_frame=coordinate_frame,
        )
        plan = _make_rsm_operation_plan_v2(
            effective,
            bindings,
            coordinate_frame,
            common_grid,
            conditioning,
            normalization,
            chunk_size=chunk_size,
            max_frame_bytes=max_frame_bytes,
            max_chunk_bytes=max_chunk_bytes,
            corrections=_bind_rsm_corrections(
                corrections, member_tuple, max_frame_bytes,
                detector=_rsm_calibration_detector(geometry_asset_receipt), detector_mask=detector_mask),
            ub_override=bound_ub,
            detector_mask=detector_mask,
        )
        preflight = _make_rsm_group_preflight_v2(
            project_root=root,
            asset=geometry_asset_receipt,
            effective=effective,
            bindings=bindings,
            members=member_tuple,
            group=source_group,
            common_grid=common_grid,
            plan=plan,
        )

    revalidate_rsm_geometry_source(geometry_asset_receipt)
    for source in source_group.members:
        _fresh_table(source, cancel_token=cancel_token)
    _requalify_project_output(output, root, output_authority, folders_made=False)
    provenance = _rsm_v2_provenance(
        source_group,
        output,
        output_authority,
        geometry_asset_receipt,
        effective,
        plan,
        preflight,
    )
    canonical = canonical_analysis_provenance(provenance)
    if len(canonical.encode("utf-8")) > _MAX_PROVENANCE_BYTES:
        raise RSMOperationRefused(
            "RSM_SOURCE_GROUP_LIMIT_EXCEEDED",
            "canonical RSM v2 provenance exceeds 1 MiB",
        )
    module = _rsm_v2_module_request(
        source_group,
        output,
        plan.fingerprint,
        module_provenance_digest(ModuleKind.RSM, provenance),
        plan_owner=plan,
        preflight_owner=preflight,
        output_authority_owner=output_authority,
    )
    request = RSMOperationRequestV2(
        module,
        plan,
        preflight,
        output_authority,
        canonical,
        _RSM_V2_FACTORY,
    )
    # the run's memory plan, at Preview too: a chunk its cap cannot hold is refused here, naming what it needs,
    # not after the sources are opened (AUD-9 stage 2b S3a)
    _require_rsm_v2_memory_plan(request)
    return request


def _engine_version() -> str:
    try:
        return importlib.metadata.version("xrayutilities")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _provenance(
    source: ModuleSourceReceipt,
    output: ModuleOutputRequest,
    output_authority: RSMOutputAuthorityReceipt,
    plan: RSMOperationPlan,
    preflight: RSMPreflightReceipt,
) -> dict[str, object]:
    normalization = plan.normalization
    return {
        "schema_version": "rsm-operation-v1",
        "kind": "rsm",
        "source": {
            "source_fingerprint": source.analysis.source_fingerprint,
            "module_source_fingerprint": source.fingerprint,
            "metadata_table_fingerprint": source.table_fingerprint,
            "selected_labels": list(source.selected_labels),
            "preflight": preflight.to_provenance(),
        },
        "geometry": {
            "preset": plan.geometry.preset,
            "header": {
                name: getattr(plan.geometry.header, name)
                for name in (
                    "cch1",
                    "cch2",
                    "pwidth1",
                    "pwidth2",
                    "distance",
                    "Nch1",
                    "Nch2",
                )
            },
            "motor_selectors": [
                {
                    "role": role,
                    "name": selector.name,
                    "occurrence": selector.occurrence,
                }
                for role, selector in plan.geometry.motor_selectors
            ],
            "image_orientation": {
                "rotation": plan.geometry.image_orientation.rotation,
                "flip_vertical": plan.geometry.image_orientation.flip_vertical,
                "flip_horizontal": plan.geometry.image_orientation.flip_horizontal,
                "transpose": plan.geometry.image_orientation.transpose,
            },
            "roi": None if plan.geometry.roi is None else list(plan.geometry.roi),
            "q_bounds": [list(item) for item in preflight.q_bounds],
            "energy_eV": preflight.energy_eV,
            "ub": [list(row) for row in preflight.ub],
        },
        "conditioning": {
            "additive_offset": plan.conditioning.additive_offset,
            "high_threshold": plan.conditioning.high_threshold,
            "static_hot_threshold": plan.conditioning.static_hot_threshold,
            "static_rule": "exact-all-frames-static-hot-v1",
        },
        "normalization": {
            "mode": normalization.mode.value,
            "foil_selector": (
                None
                if normalization.foil_selector is None
                else {
                    "name": normalization.foil_selector.name,
                    "occurrence": normalization.foil_selector.occurrence,
                }
            ),
            "exposure_selector": (
                None
                if normalization.exposure_selector is None
                else {
                    "name": normalization.exposure_selector.name,
                    "occurrence": normalization.exposure_selector.occurrence,
                }
            ),
            "absorption_lengths": list(normalization.absorption_lengths),
            "placement": "conditioned-numerator-before-grid",
        },
        "plan": {
            "bins": list(plan.bins),
            "chunk_size": plan.chunk_size,
            "max_frame_bytes": plan.max_frame_bytes,
            "max_chunk_bytes": plan.max_chunk_bytes,
            "plan_fingerprint": plan.fingerprint,
        },
        "engine": {
            "xrayutilities": _engine_version(),
            "numpy": np.__version__,
        },
        "output": {
            "kind": output.kind.value,
            **output_authority.to_provenance(),
        },
        "holds": [
            "gi-and-refraction-corrections",
            "hostile-shared-project-output-namespace",
            "multi-scan-rsm",
            "volume-rendering",
        ],
    }


@dataclass(eq=False, frozen=True, slots=True)
class RSMOperationRequest:
    module: ModuleOperationRequest
    plan: RSMOperationPlan
    preflight: RSMPreflightReceipt
    output_authority: RSMOutputAuthorityReceipt
    provenance_json: str
    _claim: InitVar[object] = None
    _cancel_token: InitVar[threading.Event | None] = None

    def __post_init__(
        self,
        _claim: object,
        _cancel_token: threading.Event | None,
    ) -> None:
        if (
            _claim is not _REQUEST_FACTORY
            or (
                _cancel_token is not None
                and type(_cancel_token) is not threading.Event
            )
            or type(self.module) is not ModuleOperationRequest
            or self.module.kind is not ModuleKind.RSM
            or self.module.output.kind is not AnalysisArtifactKind.RSM
            or type(self.plan) is not RSMOperationPlan
            or type(self.preflight) is not RSMPreflightReceipt
            or type(self.output_authority) is not RSMOutputAuthorityReceipt
            or type(self.provenance_json) is not str
            or self.module.plan_fingerprint != self.plan.fingerprint
            or self.preflight.plan_fingerprint != self.plan.fingerprint
            or self.preflight.module_source_fingerprint
            != self.module.source.fingerprint
            or self.preflight.source_fingerprint
            != self.module.source.analysis.source_fingerprint
            or self.preflight.table_fingerprint
            != self.module.source.table_fingerprint
            or tuple(item.label for item in self.preflight.contributions)
            != self.module.source.selected_labels
        ):
            raise TypeError("RSM operation request is invalid")
        expected_preflight = _capture_preflight(
            self.module.source,
            self.plan,
            _fresh_table(
                self.module.source,
                cancel_token=_cancel_token,
            ),
            self.preflight.project_root,
            cancel_token=_cancel_token,
        )
        if expected_preflight.fingerprint != self.preflight.fingerprint:
            raise ValueError(
                "RSM preflight is not the exact bound source projection"
            )
        expected_output = _bind_project_output(
            self.module.output,
            self.preflight.project_root,
        )
        expected_authority = _capture_output_authority(
            expected_output,
            self.preflight.project_root,
        )
        if (
            expected_output.fingerprint != self.module.output.fingerprint
            or expected_authority != self.output_authority
        ):
            raise ValueError("RSM output is not canonically bound to Project")
        try:
            provenance = json.loads(self.provenance_json)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise TypeError("RSM operation provenance is invalid") from error
        if (
            type(provenance) is not dict
            or module_provenance_digest(ModuleKind.RSM, provenance)
            != self.module.provenance_digest
            or provenance
            != _provenance(
                self.module.source,
                self.module.output,
                self.output_authority,
                self.plan,
                self.preflight,
            )
        ):
            raise ValueError("RSM provenance does not match its exact request")

    @property
    def provenance(self) -> dict[str, object]:
        return json.loads(self.provenance_json)

    def __copy__(self):
        raise TypeError("RSM operation request is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("RSM operation request is not copyable")


def prepare_rsm_operation(
    source: ModuleSourceReceipt,
    output: ModuleOutputRequest,
    plan: RSMOperationPlan,
    *,
    project_root: str | Path,
    cancel_token: threading.Event | None = None,
) -> RSMOperationRequest:
    """Bind one exact SPEC scan, scientific preflight, plan, and output."""

    if (
        type(source) is not ModuleSourceReceipt
        or source.kind is not ModuleKind.RSM
        or type(output) is not ModuleOutputRequest
        or type(plan) is not RSMOperationPlan
    ):
        raise TypeError("RSM preparation requires exact RSM module values")
    if cancel_token is not None and type(cancel_token) is not threading.Event:
        raise TypeError("RSM cancellation token must be exact threading.Event")
    if source.analysis.resolved_kind is not SourceKind.SPEC:
        raise RSMOperationRefused(
            "SOURCE_KIND_UNSUPPORTED",
            "R1 RSM accepts only one extensionless SPEC scan",
        )
    if output.kind is not AnalysisArtifactKind.RSM:
        raise RSMOperationRefused(
            "OUTPUT_KIND_MISMATCH",
            "RSM requires an RSM analysis artifact output",
        )
    if output.overwrite is not AnalysisArtifactOverwrite.CREATE_NEW:
        raise RSMOperationRefused(
            "RSM_OUTPUT_POLICY_UNSUPPORTED",
            "RSM must create one new immutable artifact",
        )
    root = _project_root(project_root)
    output = _bind_project_output(output, root)
    output_authority = _capture_output_authority(output, root)
    required = required_rsm_selectors(plan)
    if tuple(source.resolved_selectors) != required:
        raise RSMOperationRefused(
            "SOURCE_PROJECTION_MISMATCH",
            "module source selectors must exactly match the RSM plan",
        )
    table = _fresh_table(source, cancel_token=cancel_token)
    preflight = _capture_preflight(
        source,
        plan,
        table,
        root,
        cancel_token=cancel_token,
    )
    provenance = _provenance(
        source,
        output,
        output_authority,
        plan,
        preflight,
    )
    module = ModuleOperationRequest(
        source,
        output,
        plan.fingerprint,
        module_provenance_digest(ModuleKind.RSM, provenance),
    )
    canonical = module_artifact_request(module, provenance).provenance_json
    return RSMOperationRequest(
        module,
        plan,
        preflight,
        output_authority,
        canonical,
        _REQUEST_FACTORY,
        cancel_token,
    )


def rsm_normalization_divisors(
    policy: RSMNormalizationPolicy,
    frame_count: int,
    *,
    foil_status: np.ndarray | None = None,
    exposure_seconds: np.ndarray | None = None,
) -> np.ndarray:
    """Compute finite positive per-frame divisors for numerator normalization."""

    if type(policy) is not RSMNormalizationPolicy:
        raise TypeError("RSM normalization policy must be exact")
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError("RSM normalization frame count must be positive")
    if policy.mode is RSMNormalizationMode.IDENTITY:
        if foil_status is not None or exposure_seconds is not None:
            raise ValueError("identity normalization does not accept metadata arrays")
        result = np.ones(frame_count, dtype=np.float64)
        result.setflags(write=False)
        return result
    if foil_status is None or exposure_seconds is None:
        raise RSMOperationRefused(
            "NORMALIZATION_METADATA_MISSING",
            "foil and exposure metadata are required",
        )
    # the shared beam divisor (Stitch's too): foil transmission × exposure, byte for byte as before
    try:
        return frame_divisors(
            frame_count,
            foil=foil_status,
            exposure=exposure_seconds,
            absorption_lengths=policy.absorption_lengths,
        )
    except BeamNormalizationRefused as error:
        raise RSMOperationRefused(error.code, str(error)) from None


def condition_rsm_images(
    images: np.ndarray,
    conditioning: RSMImageConditioning,
    *,
    static_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return one owned float64 chunk after the recorded conditioning policy."""

    if type(conditioning) is not RSMImageConditioning:
        raise TypeError("RSM image conditioning must be exact")
    source = np.asarray(images)
    if source.dtype.kind not in "biuf" or source.ndim not in {2, 3}:
        raise RSMOperationRefused(
            "FRAME_LAYOUT_INVALID",
            "RSM images must be numeric two- or three-dimensional arrays",
        )
    result = np.array(source, dtype=np.float64, copy=True, order="C")
    if result.ndim == 2:
        result = result[np.newaxis, :, :]
    result += conditioning.additive_offset
    if conditioning.high_threshold is not None:
        result[result > conditioning.high_threshold] = np.nan
    if static_mask is not None:
        mask = np.asarray(static_mask)
        if mask.dtype.kind != "b" or mask.shape != result.shape[1:]:
            raise RSMOperationRefused(
                "STATIC_MASK_LAYOUT_INVALID",
                "RSM static mask must match the full detector frame",
            )
        result[:, mask] = np.nan
    return result


def rsm_static_hot_mask(
    chunks: Iterable[np.ndarray],
    conditioning: RSMImageConditioning,
    *,
    cancel_token: threading.Event | None = None,
) -> np.ndarray | None:
    """Compute the notebook's exact all-frames, same-value hot-pixel mask."""

    if cancel_token is not None and type(cancel_token) is not threading.Event:
        raise TypeError("RSM cancellation token must be exact threading.Event")
    threshold = conditioning.static_hot_threshold
    if threshold is None:
        return None
    first: np.ndarray | None = None
    same: np.ndarray | None = None
    above: np.ndarray | None = None
    frame_count = 0
    for raw_chunk in chunks:
        if cancel_token is not None and cancel_token.is_set():
            raise RSMOperationRefused("CANCELLED")
        chunk = condition_rsm_images(raw_chunk, conditioning)
        for frame in chunk:
            if cancel_token is not None and cancel_token.is_set():
                raise RSMOperationRefused("CANCELLED")
            if first is None:
                first = np.array(frame, copy=True, order="C")
                same = np.ones(first.shape, dtype=bool)
                above = first > threshold
            else:
                assert same is not None and above is not None
                same &= frame == first
                above &= frame > threshold
            frame_count += 1
    if frame_count == 0 or first is None or same is None or above is None:
        raise RSMOperationRefused("SOURCE_EMPTY")
    result = np.ascontiguousarray(same & above)
    result.setflags(write=False)
    return result


def resolve_exact_rsm_q_bounds(
    mapper: PixelQMap,
    angles: tuple[np.ndarray, ...] | list[np.ndarray],
    energy_eV: float,
    ub: np.ndarray,
    *,
    roi: tuple[int, int, int, int] | None,
    chunk_size: int,
    max_frame_bytes: int,
    max_chunk_bytes: int,
    cancel_token: threading.Event | None = None,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Resolve exact q extrema over every selected detector pixel, image-free."""

    if type(mapper) is not PixelQMap:
        raise TypeError("RSM q-bound resolution requires exact PixelQMap")
    energy = _finite_float(energy_eV, "RSM energy", positive=True)
    matrix = np.asarray(ub, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("RSM UB must be one finite 3 by 3 matrix")
    if type(chunk_size) is not int or not 1 <= chunk_size <= _MAX_CHUNK_SIZE:
        raise ValueError("RSM q-bound chunk size is outside the R1 bound")
    if (
        type(max_frame_bytes) is not int
        or not 1 <= max_frame_bytes <= _MAX_FRAME_BYTES
    ):
        raise ValueError("RSM q-bound frame byte limit is outside the R1 bound")
    if (
        type(max_chunk_bytes) is not int
        or not 1 <= max_chunk_bytes <= _MAX_CHUNK_BYTES
    ):
        raise ValueError("RSM q-bound chunk byte limit is outside the R1 bound")
    if cancel_token is not None and type(cancel_token) is not threading.Event:
        raise TypeError("RSM cancellation token must be exact threading.Event")
    if type(angles) not in {tuple, list} or len(angles) != 6:
        raise ValueError("RSM psic q-bound resolution requires six angle arrays")
    values = tuple(np.asarray(item, dtype=np.float64) for item in angles)
    if not values or values[0].ndim != 1 or len(values[0]) < 1:
        raise ValueError("RSM angle arrays must be nonempty and one-dimensional")
    frame_count = len(values[0])
    if any(
        item.ndim != 1
        or len(item) != frame_count
        or not np.all(np.isfinite(item))
        for item in values
    ):
        raise ValueError("RSM angle arrays must be aligned and finite")
    cropped = mapper.header if roi is None else mapper.header.with_roi(roi)
    if cropped.Nch1 < 2 or cropped.Nch2 < 2:
        raise ValueError("RSM q-bound ROI has an invalid detector shape")
    if cancel_token is not None and cancel_token.is_set():
        raise RSMOperationRefused("CANCELLED")
    conditioned_frame_bytes = (
        mapper.header.Nch1
        * mapper.header.Nch2
        * np.dtype(np.float64).itemsize
    )
    if conditioned_frame_bytes > max_frame_bytes:
        raise RSMOperationRefused(
            "FRAME_MEMORY_LIMIT_EXCEEDED",
            "RSM full conditioned detector frame requires "
            f"{conditioned_frame_bytes} bytes, exceeding "
            f"max_frame_bytes={max_frame_bytes}",
        )
    q_plane_bytes = (
        cropped.Nch1
        * cropped.Nch2
        * np.dtype(np.float64).itemsize
    )
    q_chunk_bytes = 3 * q_plane_bytes * min(frame_count, chunk_size)
    if q_chunk_bytes > max_chunk_bytes:
        raise RSMOperationRefused(
            "Q_MEMORY_LIMIT_EXCEEDED",
            "RSM exact q preflight requires "
            f"{q_plane_bytes} bytes per coordinate plane and "
            f"{q_chunk_bytes} bytes per coordinate chunk, exceeding "
            f"max_chunk_bytes={max_chunk_bytes}",
        )
    lows = np.full(3, np.inf, dtype=np.float64)
    highs = np.full(3, -np.inf, dtype=np.float64)
    for start in range(0, frame_count, chunk_size):
        if cancel_token is not None and cancel_token.is_set():
            raise RSMOperationRefused("CANCELLED")
        stop = min(start + chunk_size, frame_count)
        q_values = mapper.pixel_q(
            [item[start:stop] for item in values],
            energy,
            UB=matrix,
            roi=roi,
            image_shape=(stop - start, cropped.Nch1, cropped.Nch2),
        )
        expected = (stop - start, cropped.Nch1, cropped.Nch2)
        if (
            type(q_values) is not tuple
            or len(q_values) != 3
            or any(np.asarray(item).shape != expected for item in q_values)
            or any(not np.all(np.isfinite(item)) for item in q_values)
        ):
            raise RSMOperationRefused(
                "Q_COORDINATES_INVALID",
                "RSM q-coordinate preflight returned an invalid detector grid",
            )
        for axis, item in enumerate(q_values):
            lows[axis] = min(lows[axis], float(np.min(item)))
            highs[axis] = max(highs[axis], float(np.max(item)))
    if any(
        not math.isfinite(float(lo))
        or not math.isfinite(float(hi))
        or not hi > lo
        for lo, hi in zip(lows, highs)
    ):
        raise RSMOperationRefused("Q_BOUNDS_INVALID")
    return tuple(
        (float(lo), float(hi)) for lo, hi in zip(lows, highs)
    )  # type: ignore[return-value]


def _raw_chunks(
    source,
    labels: tuple[int, ...],
    *,
    chunk_size: int,
    expected_shape: tuple[int, int],
    max_frame_bytes: int,
    max_chunk_bytes: int,
    cancel_token: threading.Event | None,
    frame_callback: Callable[[int], object] | None = None,
):
    completed = 0
    for start in range(0, len(labels), chunk_size):
        if cancel_token is not None and cancel_token.is_set():
            raise RSMOperationRefused("CANCELLED")
        chunk_labels = labels[start : start + chunk_size]
        frames: list[np.ndarray] = []
        working_bytes = 0
        for label in chunk_labels:
            if cancel_token is not None and cancel_token.is_set():
                raise RSMOperationRefused("CANCELLED")
            image = np.asarray(source.load_frame(label))
            if image.dtype.kind not in "biuf" or image.shape != expected_shape:
                raise RSMOperationRefused(
                    "FRAME_LAYOUT_INVALID",
                    f"RSM frame {label} does not match detector shape "
                    f"{expected_shape}",
                )
            float_bytes = image.size * np.dtype(np.float64).itemsize
            if image.nbytes > max_frame_bytes or float_bytes > max_frame_bytes:
                raise MemoryError(
                    f"RSM frame {label} requires {max(image.nbytes, float_bytes)} "
                    f"bytes, exceeding max_frame_bytes={max_frame_bytes}"
                )
            # Bound arrays owned by this operation while the downstream
            # gridder consumes the chunk: decoded frames, the stacked raw
            # copy, conditioned input, the gridder's float copy, and its
            # three per-pixel q-coordinate arrays. Backend-private temporary
            # allocations remain outside this enforceable ownership boundary.
            working_bytes += 2 * image.nbytes + 5 * float_bytes
            if working_bytes > max_chunk_bytes:
                raise MemoryError(
                    f"RSM chunk through frame {label} requires at least "
                    f"{working_bytes} bytes, exceeding "
                    f"max_chunk_bytes={max_chunk_bytes}"
                )
            frames.append(image)
            completed += 1
            if frame_callback is not None:
                try:
                    frame_callback(completed)
                except Exception:
                    pass
        if cancel_token is not None and cancel_token.is_set():
            raise RSMOperationRefused("CANCELLED")
        yield np.stack(frames, axis=0), list(chunk_labels)


def _contribution_values(
    contribution: RSMContribution,
) -> dict[tuple[str, int], float]:
    return {
        (name, occurrence): value
        for name, occurrence, value in contribution.values
    }


class _RSMSourceView:
    """Selected, conditioned, numerator-normalized view of one source lease."""

    def __init__(
        self,
        source,
        request: RSMOperationRequest,
        *,
        cancel_token: threading.Event | None,
        frame_callback: Callable[[int], object] | None,
    ) -> None:
        self._source = source
        self._request = request
        self._cancel_token = cancel_token
        self._frame_callback = frame_callback
        self.name = getattr(source, "name", type(source).__name__)
        self.capabilities = source.capabilities
        self._labels = request.module.source.selected_labels
        self._contributions = request.preflight.contributions
        self._by_label = {
            item.label: item for item in self._contributions
        }
        geometry = request.plan.geometry
        self._role_values = {
            role: np.asarray(
                [
                    _contribution_values(item)[
                        (selector.name, selector.occurrence)
                    ]
                    for item in self._contributions
                ],
                dtype=np.float64,
            )
            for role, selector in geometry.motor_selectors
        }

    @property
    def frame_indices(self) -> list[int]:
        return list(self._labels)

    @property
    def motors(self) -> dict[str, np.ndarray]:
        return {
            role: np.array(values, copy=True)
            for role, values in self._role_values.items()
        }

    def motor_series(self, name: str) -> np.ndarray:
        try:
            return np.array(self._role_values[name], copy=True)
        except KeyError as error:
            raise KeyError(name) from error

    def _condition(self, images: np.ndarray, labels: list[int]) -> np.ndarray:
        conditioned = condition_rsm_images(images, self._request.plan.conditioning)
        divisors = np.asarray(
            [self._by_label[label].normalization_divisor for label in labels],
            dtype=np.float64,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            conditioned /= divisors[:, None, None]
        if np.any(np.isinf(conditioned)):
            raise RSMOperationRefused(
                "NORMALIZATION_RESULT_INVALID",
                "RSM normalization produced an infinite numerator",
            )
        return conditioned

    def load_frame(self, index: int) -> np.ndarray:
        label = int(index)
        if label not in self._by_label:
            raise KeyError(label)
        chunks = _raw_chunks(
            self._source,
            (label,),
            chunk_size=1,
            expected_shape=self._request.preflight.detector_shape,
            max_frame_bytes=self._request.plan.max_frame_bytes,
            max_chunk_bytes=self._request.plan.max_chunk_bytes,
            cancel_token=self._cancel_token,
            frame_callback=self._frame_callback,
        )
        images, labels = next(chunks)
        return self._condition(images, labels)[0]

    def iter_chunks(self, chunk_size: int):
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("RSM chunk size must be positive")
        for images, labels in _raw_chunks(
            self._source,
            self._labels,
            chunk_size=chunk_size,
            expected_shape=self._request.preflight.detector_shape,
            max_frame_bytes=self._request.plan.max_frame_bytes,
            max_chunk_bytes=self._request.plan.max_chunk_bytes,
            cancel_token=self._cancel_token,
            frame_callback=self._frame_callback,
        ):
            yield self._condition(images, labels), labels

    def metadata_for(self, index: int) -> Mapping[str, object]:
        method = getattr(self._source, "metadata_for", None)
        return {} if not callable(method) else method(index)

    def frame_for(self, index: int):
        method = getattr(self._source, "frame_for", None)
        if not callable(method):
            raise AttributeError("source has no frame_for")
        return method(index)


def _crop_static_mask(
    mask: np.ndarray | None,
    roi: tuple[int, int, int, int] | None,
    expected_shape: tuple[int, int],
) -> np.ndarray | None:
    if mask is None:
        return None
    result = np.asarray(mask, dtype=bool)
    if roi is not None:
        r0, r1, c0, c1 = roi
        result = result[r0:r1, c0:c1]
    if result.shape != expected_shape:
        raise RSMOperationRefused(
            "STATIC_MASK_LAYOUT_INVALID",
            "RSM static mask does not match the cropped detector",
        )
    frozen = np.ascontiguousarray(result)
    frozen.setflags(write=False)
    return frozen


def _runtime_plan(
    request: RSMOperationRequest,
    static_mask: np.ndarray | None,
) -> RSMPlan:
    plan = request.plan
    preflight = request.preflight
    return RSMPlan(
        mapper=PixelQMap(Diffractometer.psic(), plan.geometry.header),
        diff_motors=_PSIC_ROLES,
        bins=plan.bins,
        UB=np.asarray(preflight.ub, dtype=np.float64),
        coordinate_frame=RSMCoordinateFrame.HKL,
        energy=preflight.energy_eV,
        chunk_size=plan.chunk_size,
        q_bounds=preflight.q_bounds,
        roi=plan.geometry.roi,
        static_mask=static_mask,
        scout_pad=0.0,
        corrections=None,
        gi=None,
    )


def _validate_science_volume(
    value: object,
    request: RSMOperationRequest,
) -> RSMVolume:
    if type(value) is not RSMVolume:
        raise RSMOperationRefused("INVALID_SCIENCE_RESULT")
    if value.coordinate_frame is not RSMCoordinateFrame.HKL:
        raise RSMOperationRefused("INVALID_SCIENCE_RESULT")
    axes = value.axis_values
    if value.intensity.shape != request.plan.bins:
        raise RSMOperationRefused("INVALID_SCIENCE_RESULT")
    for axis, size, bounds in zip(axes, request.plan.bins, request.preflight.q_bounds):
        values = np.asarray(axis)
        expected = np.linspace(bounds[0], bounds[1], size, dtype=np.float64)
        if (
            values.shape != (size,)
            or values.dtype.kind not in "fiu"
            or not np.all(np.isfinite(values))
            or not np.array_equal(values, expected)
        ):
            raise RSMOperationRefused("INVALID_SCIENCE_RESULT")
    intensity = np.asarray(value.intensity)
    if intensity.dtype.kind not in "fiu" or np.any(np.isinf(intensity)):
        raise RSMOperationRefused("INVALID_SCIENCE_RESULT")
    return value


@dataclass(eq=False, frozen=True, slots=True)
class RSMOperationResult:
    request: RSMOperationRequest
    terminal: ModuleTerminalResult
    payload: AnalysisArtifactPayload | None = None

    def __post_init__(self) -> None:
        committed = self.terminal.disposition is ModuleDisposition.COMMITTED
        if (
            type(self.request) is not RSMOperationRequest
            or type(self.terminal) is not ModuleTerminalResult
            or self.terminal.request is not self.request.module
            or (
                committed
                and type(self.payload) is not AnalysisArtifactPayload
            )
            or (not committed and self.payload is not None)
            or (
                committed
                and self.payload.kind is not AnalysisArtifactKind.RSM
            )
            or (
                committed
                and self.payload.result_fingerprint
                != self.terminal.commit.result_fingerprint
            )
        ):
            raise TypeError("RSM operation result is invalid")


class RSMOperationExecution:
    """One-shot execution owner retained across cleanup and reload retries."""

    def __init__(self, request: RSMOperationRequest, *, coordinator=None):
        if type(request) is not RSMOperationRequest:
            raise TypeError("RSM execution requires exact RSMOperationRequest")
        self.request = request
        self.coordinator = coordinator
        self._state = "new"
        self._output: ModuleArtifactOutput | None = None
        self._result: RSMOperationResult | None = None
        self._verification_terminal: ModuleTerminalResult | None = None
        self._revision = 0
        self._progress_callback: Callable[[ModuleProgress], object] | None = None

    def __copy__(self):
        raise TypeError("RSM operation execution is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("RSM operation execution is not copyable")

    @property
    def output_snapshot(self):
        return None if self._output is None else self._output.snapshot

    def _terminal(
        self,
        disposition: ModuleDisposition,
        code: str,
        diagnostic: str = "",
    ) -> RSMOperationResult:
        value = RSMOperationResult(
            self.request,
            ModuleTerminalResult(
                self.request.module,
                disposition,
                code,
                diagnostic=diagnostic,
            ),
        )
        self._result = value
        self._state = "done"
        return value

    def _emit(self, stage: str, completed: int, total: int) -> None:
        callback = self._progress_callback
        if callback is None:
            return
        self._revision += 1
        progress = ModuleProgress(
            self.request.module,
            self._revision,
            stage,
            completed,
            total,
        )
        try:
            callback(progress)
        except Exception:
            pass

    def _strict_result(
        self,
        terminal: ModuleTerminalResult,
        *,
        total: int,
    ) -> RSMOperationResult:
        if terminal.disposition is not ModuleDisposition.COMMITTED:
            value = RSMOperationResult(self.request, terminal)
            self._result = value
            self._state = "done"
            return value
        if self._output is None or self._output.snapshot.receipt is None:
            raise RSMOperationVerificationError(
                self, "committed RSM output has no exact artifact receipt"
            )
        try:
            payload = read_analysis_artifact(
                self.request.module.output.target,
                expected_receipt=self._output.snapshot.receipt,
            )
        except BaseException as error:
            self._verification_terminal = terminal
            self._state = "verification_failed"
            raise RSMOperationVerificationError(
                self, f"strict RSM reload failed: {_failure_diagnostic(error)}"
            ) from error
        if payload.result_fingerprint != terminal.commit.result_fingerprint:
            self._verification_terminal = terminal
            self._state = "verification_failed"
            raise RSMOperationVerificationError(
                self, "strict RSM reload result fingerprint changed"
            )
        try:
            provenance = json.loads(payload.provenance_json)
            source_provenance = provenance["source"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._verification_terminal = terminal
            self._state = "verification_failed"
            raise RSMOperationVerificationError(
                self, "strict RSM provenance is malformed"
            ) from error
        if (
            payload.kind is not AnalysisArtifactKind.RSM
            or payload.inspection.shape != self.request.plan.bins
            or tuple(name for name, _axis in payload.axes) != ("h", "k", "l")
            or any(
                not np.array_equal(
                    np.asarray(axis),
                    np.asarray(
                        np.linspace(bounds[0], bounds[1], size),
                        dtype=np.float32,
                    ),
                )
                for (_name, axis), bounds, size in zip(
                    payload.axes,
                    self.request.preflight.q_bounds,
                    self.request.plan.bins,
                )
            )
            or payload.provenance_json != self.request.provenance_json
            or source_provenance.get("preflight")
            != self.request.preflight.to_provenance()
            or source_provenance.get("selected_labels")
            != list(self.request.module.source.selected_labels)
        ):
            self._verification_terminal = terminal
            self._state = "verification_failed"
            raise RSMOperationVerificationError(
                self, "strict RSM payload no longer matches its request"
            )
        self._emit("reload", total, total)
        value = RSMOperationResult(self.request, terminal, payload)
        self._result = value
        self._verification_terminal = terminal
        self._state = "done"
        return value

    def run(
        self,
        *,
        cancel_token: threading.Event | None = None,
        progress_callback: Callable[[ModuleProgress], object] | None = None,
    ) -> RSMOperationResult:
        if self._state != "new":
            raise RuntimeError("RSM execution is one-shot")
        if cancel_token is not None and type(cancel_token) is not threading.Event:
            raise TypeError("RSM cancellation token must be exact threading.Event")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("RSM progress callback must be callable")
        self._state = "running"
        self._progress_callback = progress_callback
        labels = self.request.module.source.selected_labels
        hot_pass = self.request.plan.conditioning.static_hot_threshold is not None
        science_steps = len(labels) * (2 if hot_pass else 1)
        total = science_steps + 3
        self._emit("preflight", 0, total)
        if cancel_token is not None and cancel_token.is_set():
            return self._terminal(ModuleDisposition.CANCELLED, "CANCELLED")
        try:
            _requalify_project_output(
                self.request.module.output,
                self.request.preflight.project_root,
                self.request.output_authority,
            )
        except RSMOperationRefused as error:
            return self._terminal(ModuleDisposition.REFUSED, error.code)
        try:
            with requalified_analysis_source(
                self.request.module.source.analysis,
                cancel_token=cancel_token,
            ) as source:
                static_mask = None
                if hot_pass:
                    static_mask = rsm_static_hot_mask(
                        (
                            images
                            for images, _chunk_labels in _raw_chunks(
                                source,
                                labels,
                                chunk_size=self.request.plan.chunk_size,
                                expected_shape=self.request.preflight.detector_shape,
                                max_frame_bytes=self.request.plan.max_frame_bytes,
                                max_chunk_bytes=self.request.plan.max_chunk_bytes,
                                cancel_token=cancel_token,
                                frame_callback=lambda done: self._emit(
                                    "static-mask", done, total
                                ),
                            )
                        ),
                        self.request.plan.conditioning,
                        cancel_token=cancel_token,
                    )
                cropped_mask = _crop_static_mask(
                    static_mask,
                    self.request.plan.geometry.roi,
                    self.request.preflight.cropped_shape,
                )
                base = len(labels) if hot_pass else 0
                view = _RSMSourceView(
                    source,
                    self.request,
                    cancel_token=cancel_token,
                    frame_callback=lambda done: self._emit(
                        "science", base + done, total
                    ),
                )
                scientific = run_rsm(
                    _runtime_plan(self.request, cropped_mask),
                    view,
                    scan_labels=[self.request.preflight.source_scan],
                )
        except AnalysisSourceLeaseRefused as error:
            if error.code == "CANCELLED":
                return self._terminal(ModuleDisposition.CANCELLED, "CANCELLED")
            return self._terminal(ModuleDisposition.REFUSED, error.code)
        except RSMOperationRefused as error:
            disposition = (
                ModuleDisposition.CANCELLED
                if error.code == "CANCELLED"
                else ModuleDisposition.REFUSED
            )
            return self._terminal(disposition, error.code)
        except BaseException as error:
            return self._terminal(
                ModuleDisposition.FAILED,
                "SCIENCE_FAILED",
                _failure_diagnostic(error),
            )
        if cancel_token is not None and cancel_token.is_set():
            return self._terminal(ModuleDisposition.CANCELLED, "CANCELLED")
        try:
            volume = _validate_science_volume(scientific.payload, self.request)
        except RSMOperationRefused as error:
            return self._terminal(ModuleDisposition.FAILED, error.code)
        try:
            result_projection = project_analysis_artifact_result(
                kind=AnalysisArtifactKind.RSM,
                axes=(("h", volume.h), ("k", volume.k), ("l", volume.l)),
                axis_units=(("h", None), ("k", None), ("l", None)),
                intensity=volume.intensity,
                sigma=None,
                coverage=None,
                normalization=None,
            )
        except AnalysisArtifactProjectionInvalid:
            return self._terminal(
                ModuleDisposition.FAILED,
                "RSM_RESULT_STORAGE_PROJECTION_INVALID",
            )
        try:
            _requalify_project_output(
                self.request.module.output,
                self.request.preflight.project_root,
                self.request.output_authority,
            )
        except RSMOperationRefused as error:
            return self._terminal(ModuleDisposition.REFUSED, error.code)
        provenance = self.request.provenance
        bound = module_artifact_request(self.request.module, provenance)
        try:
            self._output = admit_module_artifact(
                self.request.module,
                provenance,
                cancel_token=cancel_token,
                coordinator=self.coordinator,
            )
        except ModuleArtifactRefused as error:
            disposition = (
                ModuleDisposition.CANCELLED
                if error.code == "CANCELLED"
                else ModuleDisposition.REFUSED
            )
            return self._terminal(disposition, error.code)
        except FileExistsError:
            return self._terminal(ModuleDisposition.REFUSED, "OUTPUT_EXISTS")
        except BaseException as error:
            return self._terminal(
                ModuleDisposition.FAILED,
                "OUTPUT_ADMISSION_FAILED",
                _failure_diagnostic(error),
            )

        def writer(entry) -> None:
            write_rsm(
                entry,
                result_projection=result_projection,
                provenance=bound.provenance_json,
                bounded_artifact=True,
            )

        def prepublish_check() -> None:
            try:
                _requalify_project_output(
                    self.request.module.output,
                    self.request.preflight.project_root,
                    self.request.output_authority,
                )
            except RSMOperationRefused as error:
                raise ModuleArtifactRefused(error.code) from error

        try:
            terminal = self._output.publish(
                writer,
                cancel_token=cancel_token,
                prepublish_check=prepublish_check,
            )
        except AnalysisArtifactCleanupPending as error:
            self._state = "cleanup_pending"
            raise RSMOperationCleanupPending(self) from error
        self._emit("publish", total - 1, total)
        return self._strict_result(terminal, total=total)

    def retry_cleanup(self) -> RSMOperationResult:
        if self._result is not None:
            return self._result
        if self._state != "cleanup_pending" or self._output is None:
            raise RuntimeError("RSM execution has no retryable cleanup")
        try:
            terminal = self._output.retry_cleanup()
        except AnalysisArtifactCleanupPending as error:
            raise RSMOperationCleanupPending(self) from error
        hot_pass = self.request.plan.conditioning.static_hot_threshold is not None
        total = len(self.request.module.source.selected_labels) * (
            2 if hot_pass else 1
        ) + 3
        self._emit("publish", total - 1, total)
        return self._strict_result(terminal, total=total)

    def retry_verification(self) -> RSMOperationResult:
        """Retry strict detached reload without replaying science or writing."""

        if self._result is not None:
            return self._result
        if (
            self._state != "verification_failed"
            or self._verification_terminal is None
            or self._verification_terminal.disposition
            is not ModuleDisposition.COMMITTED
        ):
            raise RuntimeError("RSM execution has no retryable verification")
        hot_pass = self.request.plan.conditioning.static_hot_threshold is not None
        total = len(self.request.module.source.selected_labels) * (
            2 if hot_pass else 1
        ) + 3
        return self._strict_result(self._verification_terminal, total=total)


def run_rsm_operation(
    request: RSMOperationRequest,
    *,
    cancel_token: threading.Event | None = None,
    progress_callback: Callable[[ModuleProgress], object] | None = None,
    coordinator=None,
) -> RSMOperationResult:
    """Run one prepared RSM operation, retaining cleanup in its exception."""

    execution = RSMOperationExecution(request, coordinator=coordinator)
    return execution.run(
        cancel_token=cancel_token,
        progress_callback=progress_callback,
    )


def _revalidate_rsm_v2_geometry(request: RSMOperationRequestV2) -> None:
    if type(request) is not RSMOperationRequestV2:
        raise TypeError("RSM v2 revalidation requires an exact request")
    try:
        revalidate_rsm_geometry_source(request.preflight.geometry_asset_receipt)
        current_effective = lower_rsm_effective_geometry(
            request.preflight.geometry_asset_receipt
        )
    except RSMGeometryAssetRefused as error:
        raise RSMOperationRefused(error.code) from error
    if (
        current_effective.fingerprint
        != request.plan.effective_geometry.fingerprint
        or _effective_geometry_provenance(current_effective)
        != _effective_geometry_provenance(request.plan.effective_geometry)
    ):
        raise RSMOperationRefused("RSM_EFFECTIVE_GEOMETRY_MISMATCH")


def _revalidate_rsm_v2_authorities(
    request: RSMOperationRequestV2,
    *,
    cancel_token: threading.Event | None,
) -> None:
    _revalidate_rsm_v2_geometry(request)
    _revalidate_rsm_ub(request.plan.ub_override)  # a UB taken from a SPEC file: that file unchanged
    for source, member in zip(
        request.module.source.members,
        request.preflight.members,
        strict=True,
    ):
        if (
            source.fingerprint != member.module_source_fingerprint
            or source.analysis.source_fingerprint != member.source_fingerprint
            or source.table_fingerprint != member.table_fingerprint
        ):
            raise RSMOperationRefused("RSM_MEMBER_IDENTITY_MISMATCH")
        _fresh_table(source, cancel_token=cancel_token)
    _requalify_project_output(
        request.module.output,
        request.preflight.project_root,
        request.output_authority,
    )


def _validate_rsm_v2_science_volume(
    value: object,
    request: RSMOperationRequestV2,
) -> RSMVolume:
    if type(value) is not RSMVolume:
        raise RSMOperationRefused("INVALID_SCIENCE_RESULT")
    if value.coordinate_frame is not request.plan.coordinate_frame:
        raise RSMOperationRefused("INVALID_SCIENCE_RESULT")
    if value.intensity.shape != request.plan.bins:
        raise RSMOperationRefused("INVALID_SCIENCE_RESULT")
    for axis, size, bounds in zip(
        value.axis_values,
        request.plan.bins,
        request.plan.common_grid.bounds,
        strict=True,
    ):
        observed = np.asarray(axis)
        expected = np.linspace(bounds[0], bounds[1], size, dtype=np.float64)
        if (
            observed.shape != (size,)
            or observed.dtype.kind not in "fiu"
            or not np.all(np.isfinite(observed))
            or not np.array_equal(observed, expected)
        ):
            raise RSMOperationRefused("INVALID_SCIENCE_RESULT")
    intensity = np.asarray(value.intensity)
    if intensity.dtype.kind not in "fiu" or np.any(np.isinf(intensity)):
        raise RSMOperationRefused("INVALID_SCIENCE_RESULT")
    return value


def _rsm_v2_execution_attestation(
    request: RSMOperationRequestV2,
    *,
    result_fingerprint: str,
    mask_receipts: tuple[RSMStaticMaskReceipt, ...],
    science_chunk_count: int,
    q_release_check_chunk_count: int,
    frame_release_check_frame_count: int,
    runtime: object,
) -> dict[str, object]:
    from xrd_tools.core.geometry.xu_runtime import XuRuntimeExecutionRecord

    selected_frames = request.module.source.selected_frame_count
    if (
        type(runtime) is not XuRuntimeExecutionRecord
        or type(mask_receipts) is not tuple
        or len(mask_receipts) != len(request.preflight.members)
        or type(science_chunk_count) is not int
        or type(q_release_check_chunk_count) is not int
        or type(frame_release_check_frame_count) is not int
        or not 1 <= science_chunk_count <= selected_frames
        or q_release_check_chunk_count != science_chunk_count
        or frame_release_check_frame_count != selected_frames
    ):
        raise RSMOperationRefused("RSM_EXECUTION_ATTESTATION_MISMATCH")
    attestation = {
        "schema_version": "rsm-execution-attestation-v2",
        "module_request_fingerprint": request.module.fingerprint,
        "result_projection_policy": "analysis_artifact_stored_le_f4_v1",
        "result_fingerprint": _require_rsm_v2_digest(
            result_fingerprint,
            "RSM stored-result fingerprint",
        ),
        "geometry_asset_receipt_fingerprint": (
            request.preflight.geometry_asset_receipt.receipt_fingerprint
        ),
        "effective_geometry_fingerprint": request.plan.effective_geometry.fingerprint,
        "common_grid_fingerprint": request.plan.common_grid.fingerprint,
        "selected_scan_count": len(request.preflight.members),
        "selected_frame_count": selected_frames,
        "science_chunk_count": science_chunk_count,
        "q_release_check_chunk_count": q_release_check_chunk_count,
        "frame_release_check_frame_count": frame_release_check_frame_count,
        "release_check_passed": True,
        "member_masks": [
            _rsm_v2_static_mask_attestation(receipt, ordinal)
            for ordinal, receipt in enumerate(mask_receipts)
        ],
        "xu_runtime": runtime.to_attestation(),
        "coordinate_frame": request.plan.coordinate_frame.value,
        "axis_names": list(request.plan.coordinate_frame.axis_names),
        "axis_units": list(request.plan.coordinate_frame.axis_units),
        "matrix_policy": request.plan.coordinate_frame.matrix_policy,
    }
    return attestation


@dataclass(eq=False, frozen=True, slots=True)
class RSMOperationResultV2:
    request: RSMOperationRequestV2
    terminal: ModuleTerminalResult
    payload: AnalysisArtifactPayload | None = None

    def __post_init__(self) -> None:
        committed = self.terminal.disposition is ModuleDisposition.COMMITTED
        if (
            type(self.request) is not RSMOperationRequestV2
            or type(self.terminal) is not ModuleTerminalResult
            or self.terminal.request is not self.request.module
            or (committed and type(self.payload) is not AnalysisArtifactPayload)
            or (not committed and self.payload is not None)
            or (
                committed
                and (
                    self.payload.kind is not AnalysisArtifactKind.RSM
                    or self.payload.schema_version
                    != (
                        3
                        if self.request.plan.coordinate_frame
                        is RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU
                        else 2
                    )
                    or self.payload.result_fingerprint
                    != self.terminal.commit.result_fingerprint
                    or self.payload.execution_attestation_digest
                    != self.terminal.commit.execution_attestation_digest
                )
            )
        ):
            raise TypeError("RSM v2 operation result is invalid")


class RSMOperationExecutionV2:
    """One-shot grouped owner retained across cleanup and reload retries."""

    def __init__(self, request: RSMOperationRequestV2, *, coordinator=None):
        if type(request) is not RSMOperationRequestV2:
            raise TypeError("RSM v2 execution requires an exact request")
        self.request = request
        self.coordinator = coordinator
        self._state = "new"
        self._output: ModuleArtifactOutput | None = None
        self._result: RSMOperationResultV2 | None = None
        self._verification_terminal: ModuleTerminalResult | None = None
        self._execution_attestation_json: str | None = None
        self._execution_attestation_digest: str | None = None
        self._mask_receipts: tuple[RSMStaticMaskReceipt, ...] = ()
        self._progress_total = 1
        self._revision = 0
        self._progress_callback: Callable[[ModuleProgress], object] | None = None

    def __copy__(self):
        raise TypeError("RSM v2 operation execution is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("RSM v2 operation execution is not copyable")

    @property
    def output_snapshot(self):
        return None if self._output is None else self._output.snapshot

    def _terminal(
        self,
        disposition: ModuleDisposition,
        code: str,
        diagnostic: str = "",
    ) -> RSMOperationResultV2:
        value = RSMOperationResultV2(
            self.request,
            ModuleTerminalResult(
                self.request.module,
                disposition,
                code,
                diagnostic=diagnostic,
            ),
        )
        self._result = value
        self._state = "done"
        return value

    def _emit(self, stage: str, completed: int) -> None:
        callback = self._progress_callback
        if callback is None:
            return
        self._revision += 1
        value = ModuleProgress(
            self.request.module,
            self._revision,
            stage,
            min(completed, self._progress_total),
            self._progress_total,
        )
        try:
            callback(value)
        except Exception:
            pass

    def _strict_result(
        self,
        terminal: ModuleTerminalResult,
    ) -> RSMOperationResultV2:
        if terminal.disposition is not ModuleDisposition.COMMITTED:
            value = RSMOperationResultV2(self.request, terminal)
            self._result = value
            self._state = "done"
            return value
        if self._output is None or self._output.snapshot.receipt is None:
            raise RSMOperationVerificationErrorV2(
                self,
                "committed RSM v2 output has no exact artifact receipt",
            )
        try:
            payload = read_analysis_artifact(
                self.request.module.output.target,
                expected_receipt=self._output.snapshot.receipt,
            )
        except BaseException as error:
            self._verification_terminal = terminal
            self._state = "verification_failed"
            raise RSMOperationVerificationErrorV2(
                self,
                f"strict RSM v2 reload failed: {_failure_diagnostic(error)}",
            ) from error
        expected_axes = tuple(
            np.ascontiguousarray(
                np.linspace(bounds[0], bounds[1], count, dtype=np.float64),
                dtype="<f4",
            )
            for bounds, count in zip(
                self.request.plan.common_grid.bounds,
                self.request.plan.bins,
                strict=True,
            )
        )
        invalid = (
            payload.kind is not AnalysisArtifactKind.RSM
            or payload.schema_version
            != (
                3
                if self.request.plan.coordinate_frame
                is RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU
                else 2
            )
            or payload.result_fingerprint != terminal.commit.result_fingerprint
            or payload.provenance_json != self.request.provenance_json
            or payload.execution_attestation_json
            != self._execution_attestation_json
            or payload.execution_attestation_digest
            != self._execution_attestation_digest
            or terminal.commit.execution_attestation_digest
            != self._execution_attestation_digest
            or payload.inspection.request_fingerprint
            != self.request.module.fingerprint
            or payload.inspection.source_fingerprint
            != self.request.module.source.fingerprint
            or payload.inspection.plan_fingerprint != self.request.plan.fingerprint
            or payload.inspection.provenance_digest
            != self.request.module.provenance_digest
            or payload.inspection.shape != self.request.plan.bins
            or payload.inspection.axis_units
            != tuple(
                zip(
                    self.request.plan.coordinate_frame.axis_names,
                    self.request.plan.coordinate_frame.axis_units,
                    strict=True,
                )
            )
            or tuple(name for name, _axis in payload.axes)
            != self.request.plan.coordinate_frame.axis_names
            or any(
                not np.array_equal(observed, expected)
                for (_name, observed), expected in zip(
                    payload.axes,
                    expected_axes,
                    strict=True,
                )
            )
            or np.any(np.isinf(payload.intensity))
            or payload.sigma is not None
            or payload.coverage is not None
            or payload.normalization is not None
        )
        if invalid:
            self._verification_terminal = terminal
            self._state = "verification_failed"
            raise RSMOperationVerificationErrorV2(
                self,
                "strict RSM v2 payload no longer matches its request",
            )
        try:
            provenance = json.loads(payload.provenance_json)
            attestation = json.loads(payload.execution_attestation_json)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            self._verification_terminal = terminal
            self._state = "verification_failed"
            raise RSMOperationVerificationErrorV2(
                self,
                "strict RSM v2 intent or attestation is malformed",
            ) from error
        try:
            expected_masks = [
                _rsm_v2_static_mask_attestation(receipt, ordinal)
                for ordinal, receipt in enumerate(self._mask_receipts)
            ]
        except (TypeError, ValueError) as error:
            self._verification_terminal = terminal
            self._state = "verification_failed"
            raise RSMOperationVerificationErrorV2(
                self,
                "strict RSM v2 identities changed after commit",
            ) from error
        if (
            provenance != self.request.provenance
            or attestation.get("member_masks") != expected_masks
        ):
            self._verification_terminal = terminal
            self._state = "verification_failed"
            raise RSMOperationVerificationErrorV2(
                self,
                "strict RSM v2 identities changed after commit",
            )
        self._emit("reload", self._progress_total)
        value = RSMOperationResultV2(self.request, terminal, payload)
        self._result = value
        self._verification_terminal = terminal
        self._state = "done"
        return value

    def run(
        self,
        *,
        cancel_token: threading.Event | None = None,
        progress_callback: Callable[[ModuleProgress], object] | None = None,
    ) -> RSMOperationResultV2:
        if self._state != "new":
            raise RuntimeError("RSM v2 execution is one-shot")
        if cancel_token is not None and type(cancel_token) is not threading.Event:
            raise TypeError("RSM v2 cancellation token must be threading.Event")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("RSM v2 progress callback must be callable")
        self._state = "running"
        self._progress_callback = progress_callback
        selected_frames = self.request.module.source.selected_frame_count
        mask_frames = sum(
            len(member.contributions)
            for member in self.request.preflight.members
            if member.mask_policy_intent[0]
            == "exact-all-selected-frames-static-hot-v1"
        )
        self._progress_total = selected_frames + mask_frames + 4
        self._emit("revalidate", 0)
        if cancel_token is not None and cancel_token.is_set():
            return self._terminal(ModuleDisposition.CANCELLED, "CANCELLED")
        try:
            _require_rsm_v2_memory_plan(self.request)
            _create_rsm_v2_output_folder(self.request)
            _revalidate_rsm_v2_authorities(
                self.request,
                cancel_token=cancel_token,
            )
        except RSMOperationRefused as error:
            disposition = (
                ModuleDisposition.CANCELLED
                if error.code == "CANCELLED"
                else ModuleDisposition.REFUSED
            )
            return self._terminal(disposition, error.code)
        self._emit("revalidate", 1)

        from xrd_tools.core.geometry.xu_runtime import xu_runtime_session

        runtime_owner = xu_runtime_session(
            self.request.plan.effective_geometry.runtime_requirements
        )
        allocator_pressure = None
        mapper = gridder = volume = None
        mask_receipts: list[RSMStaticMaskReceipt] = []
        science_chunk_count = 0
        q_release_chunk_count = 0
        frame_release_count = 0
        progress_completed = 1
        try:
            with runtime_owner as session:
                # Runtime admission precedes optional allocator optimization.
                # Darwin binding failures still refuse before science; other
                # platforms retain the same root-release proofs without it.
                allocator_pressure = bind_allocator_pressure_relief()
                mapper = rsm_effective_pixel_q_map(
                    self.request.plan.effective_geometry
                )
                gridder = StreamingGridder(
                    mapper,
                    self.request.plan.bins,
                    runtime_session=session,
                    coordinate_frame=self.request.plan.coordinate_frame,
                )
                gridder.set_bounds(*self.request.plan.common_grid.bounds)
                correction_state = _rsm_correction_state(
                    self.request.plan,
                    self.request.plan.effective_geometry.detector_header,
                    self.request.plan.effective_geometry.diffractometer_projection,
                    self.request.preflight.geometry_asset_receipt,
                )
                for source_receipt, member in zip(
                    self.request.module.source.members,
                    self.request.preflight.members,
                    strict=True,
                ):
                    if cancel_token is not None and cancel_token.is_set():
                        raise RSMOperationRefused("CANCELLED")
                    with requalified_analysis_source(
                        source_receipt.analysis,
                        cancel_token=cancel_token,
                    ) as source:
                        mask_base = progress_completed

                        def mask_progress(done: int) -> None:
                            self._emit("static-mask", mask_base + done)

                        full_mask, cropped_mask, mask_receipt = (
                            _derive_rsm_v2_static_mask(
                                source,
                                member,
                                self.request.plan,
                                cancel_token=cancel_token,
                                frame_callback=(
                                    mask_progress
                                    if member.mask_policy_intent[0]
                                    != "none"
                                    else None
                                ),
                            )
                        )
                        if member.mask_policy_intent[0] != "none":
                            progress_completed += len(member.contributions)
                        mask_receipts.append(mask_receipt)
                        full_mask = None
                        for start in range(
                            0,
                            len(member.contributions),
                            self.request.plan.chunk_size,
                        ):
                            if cancel_token is not None and cancel_token.is_set():
                                raise RSMOperationRefused("CANCELLED")
                            contributions = member.contributions[
                                start : start + self.request.plan.chunk_size
                            ]
                            angles = _rsm_v2_contribution_angles(
                                member,
                                contributions,
                            )
                            lease = _make_rsm_v2_chunk_lease(
                                source,
                                member,
                                self.request.plan,
                                contributions,
                                cropped_static_mask=cropped_mask,
                                cancel_token=cancel_token,
                                **({} if correction_state is None else {"corrections": correction_state}),
                            )
                            weights = _rsm_chunk_weights(
                                correction_state, mapper, session, angles, member.energy_eV,
                                self.request.plan.effective_geometry.roi,
                            )
                            release = gridder.add_leased(
                                lease,
                                angles,
                                member.energy_eV,
                                UB=np.ascontiguousarray(
                                    member.ub,
                                    dtype=np.float64,
                                ),
                                roi=self.request.plan.effective_geometry.roi,
                                weight=1.0 if weights is None else weights,
                            )
                            weights = None
                            release_facts = _rsm_grid_chunk_release_facts(
                                release,
                                expected_frame_count=len(contributions),
                            )
                            science_chunk_count += 1
                            q_release_chunk_count += 1
                            frame_release_count += release_facts[0]
                            progress_completed += release_facts[0]
                            self._emit("science", progress_completed)
                            if (
                                cancel_token is not None
                                and cancel_token.is_set()
                            ):
                                raise RSMOperationRefused("CANCELLED")
                            lease = None
                            angles = ()
                        cropped_mask = None
                    if cancel_token is not None and cancel_token.is_set():
                        raise RSMOperationRefused("CANCELLED")
                if gridder.n_frames_processed != selected_frames:
                    raise RSMOperationRefused(
                        "RSM_EXECUTION_ATTESTATION_MISMATCH"
                    )
                # Every source lease is closed and every issued chunk receipt
                # has proven its raw/q/feed roots dead.  One end-science call is
                # intentionally independent of member/chunk count.
                if allocator_pressure is not None:
                    allocator_pressure.relieve()
                volume = _validate_rsm_v2_science_volume(
                    gridder.to_volume(),
                    self.request,
                )
        except AnalysisSourceLeaseRefused as error:
            if error.code == "CANCELLED":
                return self._terminal(ModuleDisposition.CANCELLED, "CANCELLED")
            return self._terminal(ModuleDisposition.REFUSED, error.code)
        except RSMGridChunkReleaseError:
            return self._terminal(
                ModuleDisposition.REFUSED,
                "RSM_CHUNK_RELEASE_FAILED",
            )
        except XuRuntimeUnsupported as error:
            return self._terminal(ModuleDisposition.REFUSED, error.code)
        except AllocatorPressureUnavailable:
            return self._terminal(
                ModuleDisposition.REFUSED,
                "RSM_ALLOCATOR_PRESSURE_UNAVAILABLE",
            )
        except AllocatorPressureCallFailed:
            return self._terminal(
                ModuleDisposition.FAILED,
                "RSM_ALLOCATOR_PRESSURE_FAILED",
            )
        except RSMOperationRefused as error:
            disposition = (
                ModuleDisposition.CANCELLED
                if error.code == "CANCELLED"
                else ModuleDisposition.REFUSED
            )
            return self._terminal(disposition, error.code)
        except BaseException as error:
            return self._terminal(
                ModuleDisposition.FAILED,
                "SCIENCE_FAILED",
                _failure_diagnostic(error),
            )
        if cancel_token is not None and cancel_token.is_set():
            return self._terminal(ModuleDisposition.CANCELLED, "CANCELLED")
        if runtime_owner.execution_record is None or volume is None:
            return self._terminal(
                ModuleDisposition.REFUSED,
                "RSM_EXECUTION_ATTESTATION_MISMATCH",
            )
        try:
            axis_names = volume.coordinate_frame.axis_names
            axis_units = volume.coordinate_frame.axis_units
            result_projection = project_analysis_artifact_result(
                kind=AnalysisArtifactKind.RSM,
                axes=volume.axes,
                axis_units=tuple(zip(axis_names, axis_units, strict=True)),
                intensity=volume.intensity,
                sigma=None,
                coverage=None,
                normalization=None,
            )
        except AnalysisArtifactProjectionInvalid:
            return self._terminal(
                ModuleDisposition.FAILED,
                "RSM_RESULT_STORAGE_PROJECTION_INVALID",
            )
        volume = gridder = mapper = None
        gc.collect(0)
        try:
            # The stored projection is now the only large product owner needed
            # by publication; released grid/finalization pages must not stack
            # with HDF5 validation and strict detached reload.
            if allocator_pressure is not None:
                allocator_pressure.relieve()
        except AllocatorPressureUnavailable:
            return self._terminal(
                ModuleDisposition.REFUSED,
                "RSM_ALLOCATOR_PRESSURE_UNAVAILABLE",
            )
        except AllocatorPressureCallFailed:
            return self._terminal(
                ModuleDisposition.FAILED,
                "RSM_ALLOCATOR_PRESSURE_FAILED",
            )
        self._mask_receipts = tuple(mask_receipts)
        try:
            attestation = _rsm_v2_execution_attestation(
                self.request,
                result_fingerprint=result_projection.result_fingerprint,
                mask_receipts=self._mask_receipts,
                science_chunk_count=science_chunk_count,
                q_release_check_chunk_count=q_release_chunk_count,
                frame_release_check_frame_count=frame_release_count,
                runtime=runtime_owner.execution_record,
            )
            attestation_digest = analysis_execution_attestation_digest(
                AnalysisArtifactKind.RSM,
                attestation,
                request_fingerprint=self.request.module.fingerprint,
            )
            _revalidate_rsm_v2_authorities(
                self.request,
                cancel_token=cancel_token,
            )
            bound = module_artifact_request(
                self.request.module,
                self.request.provenance,
                execution_attestation=attestation,
                execution_attestation_digest=attestation_digest,
                rsm_mask_receipts=self._mask_receipts,
            )
            self._execution_attestation_json = bound.execution_attestation_json
            self._execution_attestation_digest = attestation_digest
            self._emit("projection", self._progress_total - 2)
            self._output = admit_module_artifact(
                self.request.module,
                self.request.provenance,
                execution_attestation=attestation,
                execution_attestation_digest=attestation_digest,
                rsm_mask_receipts=self._mask_receipts,
                cancel_token=cancel_token,
                coordinator=self.coordinator,
            )
        except (RSMOperationRefused, ModuleArtifactRefused) as error:
            code = error.code
            disposition = (
                ModuleDisposition.CANCELLED
                if code == "CANCELLED"
                else ModuleDisposition.REFUSED
            )
            return self._terminal(disposition, code)
        except FileExistsError:
            return self._terminal(ModuleDisposition.REFUSED, "OUTPUT_EXISTS")
        except BaseException as error:
            return self._terminal(
                ModuleDisposition.FAILED,
                "OUTPUT_ADMISSION_FAILED",
                _failure_diagnostic(error),
            )

        def writer(entry: object) -> None:
            write_rsm(
                entry,
                result_projection=result_projection,
                provenance=bound.provenance_json,
                bounded_artifact=True,
                coordinate_frame=(
                    self.request.plan.coordinate_frame
                    if self.request.plan.coordinate_frame
                    is RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU
                    else None
                ),
            )

        def prepublish_check() -> None:
            try:
                _revalidate_rsm_v2_authorities(
                    self.request,
                    cancel_token=cancel_token,
                )
            except RSMOperationRefused as error:
                raise ModuleArtifactRefused(error.code) from error

        try:
            terminal = self._output.publish(
                writer,
                cancel_token=cancel_token,
                prepublish_check=prepublish_check,
            )
        except AnalysisArtifactCleanupPending as error:
            self._state = "cleanup_pending"
            raise RSMOperationCleanupPendingV2(self) from error
        self._emit("publish", self._progress_total - 1)
        return self._strict_result(terminal)

    def retry_cleanup(self) -> RSMOperationResultV2:
        if self._result is not None:
            return self._result
        if self._state != "cleanup_pending" or self._output is None:
            raise RuntimeError("RSM v2 execution has no retryable cleanup")
        try:
            terminal = self._output.retry_cleanup()
        except AnalysisArtifactCleanupPending as error:
            raise RSMOperationCleanupPendingV2(self) from error
        self._emit("publish", self._progress_total - 1)
        return self._strict_result(terminal)

    def retry_verification(self) -> RSMOperationResultV2:
        """Retry strict detached reload without replaying science or writing."""

        if self._result is not None:
            return self._result
        if (
            self._state != "verification_failed"
            or self._verification_terminal is None
            or self._verification_terminal.disposition
            is not ModuleDisposition.COMMITTED
        ):
            raise RuntimeError("RSM v2 execution has no retryable verification")
        return self._strict_result(self._verification_terminal)


def run_rsm_operation_v2(
    request: RSMOperationRequestV2,
    *,
    cancel_token: threading.Event | None = None,
    progress_callback: Callable[[ModuleProgress], object] | None = None,
    coordinator=None,
) -> RSMOperationResultV2:
    """Run one exact grouped RSM v2 request."""

    execution = RSMOperationExecutionV2(request, coordinator=coordinator)
    return execution.run(
        cancel_token=cancel_token,
        progress_callback=progress_callback,
    )


def rsm_output_current(request: RSMOperationRequestV2) -> bool:
    """Whether ``request``'s output already holds its RSM: the same plan (geometry,
    grid, conditioning, normalization, frame) over the same members and frames
    (sources, scans, images and motor readings). A Run in Append mode skips such an
    output, as Stitch's skips one (``stitch_output_current``, AUD-9 stage 2).

    File revisions are left out as Stitch leaves them out, and with them the
    metadata table's fingerprint, which carries its SPEC file's revision (probe 2:
    a SPEC file grown by a later scan changes it and nothing else of the member):
    an earlier scan's RSM stays standing. The file's envelope alone is read
    (``rsm_record_identity``); a missing, unreadable or foreign file is not current.
    """

    from xrd_tools.analysis.stitch_operation import _REVISION_KEYS, _without_revisions

    if type(request) is not RSMOperationRequestV2:
        raise TypeError("RSM v2 output currency requires an exact request")
    target = Path(request.module.output.target)
    identity = rsm_record_identity(target) if target.is_file() else None
    if identity is None:
        return False
    plan_fingerprint, provenance = identity
    keys = _REVISION_KEYS | {"table_fingerprint"}
    try:
        stored = json.loads(provenance)["members"]
        planned = json.loads(json.dumps(request.provenance["members"]))
    except (ValueError, KeyError, TypeError):
        return False
    return plan_fingerprint == request.plan.fingerprint and _without_revisions(
        stored, keys
    ) == _without_revisions(planned, keys)


__all__ = [
    "RSMBoundCorrections",
    "RSMCommonGrid",
    "RSMContribution",
    "RSMDetectorGeometry",
    "RSMFrameCorrections",
    "RSMImageConditioning",
    "RSMManifestFile",
    "RSMNormalizationMode",
    "RSMNormalizationPolicy",
    "RSMOperationCleanupPending",
    "RSMOperationCleanupPendingV2",
    "RSMOperationExecution",
    "RSMOperationExecutionV2",
    "RSMOperationPlan",
    "RSMOperationPlanV2",
    "RSMOperationRefused",
    "RSMOperationRequest",
    "RSMOperationRequestV2",
    "RSMOperationResult",
    "RSMOperationResultV2",
    "RSMOperationVerificationError",
    "RSMOperationVerificationErrorV2",
    "RSMOutputAuthorityReceipt",
    "RSMPreflightReceipt",
    "RSMPreflightMemberV2",
    "RSMStaticMaskReceipt",
    "RSMGroupPreflightReceiptV2",
    "condition_rsm_images",
    "prepare_rsm_operation",
    "prepare_rsm_operation_v2",
    "required_rsm_selectors",
    "required_rsm_selectors_v2",
    "make_rsm_common_grid",
    "resolve_exact_rsm_q_bounds",
    "run_rsm_operation",
    "run_rsm_operation_v2",
    "rsm_normalization_divisors",
    "rsm_output_current",
    "rsm_static_hot_mask",
]
