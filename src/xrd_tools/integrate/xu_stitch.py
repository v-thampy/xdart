"""Production PSIC powder 1-D Stitch backend using xrayutilities q mapping."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
import hashlib
import importlib.metadata
import math
import threading
import weakref

import numpy as np

from xrd_tools.analysis.scan_operations import analysis_canonical_fingerprint
from xrd_tools.analysis.xu_stitch_calibration import (
    XuStitchCalibrationReceipt,
    revalidate_xu_stitch_calibration,
)
from xrd_tools.core.containers import IntegrationResult1D
from xrd_tools.core.energy import energy_eV_to_wavelength_m
from xrd_tools.core.geometry import DetectorHeader
from xrd_tools.core.geometry.xu_runtime import (
    XuRuntimeExecutionRecord,
    XuRuntimeSession,
    xu_runtime_session,
)
from xrd_tools.corrections.stack import CorrectionStack
from xrd_tools.integrate.multi import StitchDiagnostics
from xrd_tools.rsm.corrections import detector_header_to_ai


_MAX_COUNT = 2**24
_Q_ROOT_POLICY = "shared_ultimate_ndarray_root_weakref_v1"
_RUNTIME_GEOMETRY_FACTORY = object()
# SURFACE v1 pins the float64 solid-angle bytes the reference host (macOS
# arm64) computes from the asset's detector header.  Other libm/SIMD builds
# land within an ULP of the same array and hash differently, so the asset
# reference maps to every hash accepted as that projection.  Add a platform
# only from its own CI log (the refusal below prints the host hash); the
# projection carries the reference, provenance carries the host value.
_SOLID_ANGLE_SHA256_EQUIVALENTS: Mapping[str, frozenset[str]] = {
    "a8d6453bc56b99a0c3151b54999adff4684de5648b8f23c8deb7aa0799937034": frozenset({
        # macOS arm64 (reference host, == asset corrections.solid_angle_sha256)
        "a8d6453bc56b99a0c3151b54999adff4684de5648b8f23c8deb7aa0799937034",
    }),
}


class XuStitchScienceRefused(RuntimeError):
    def __init__(self, code: str, message: str | None = None):
        self.code = code
        super().__init__(message or code)


class XuStitchCancelled(RuntimeError):
    pass


def _sha256_array(values: np.ndarray, dtype: str) -> str:
    array = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


@dataclass(frozen=True, slots=True)
class XuStitchEffectiveGeometryProjection:
    asset_semantic_fingerprint: str
    detector_type: str
    detector_config: tuple[tuple[str, float | int], ...]
    shape: tuple[int, int]
    mask_count: int
    mask_sha256: str
    solid_angle_sha256: str
    energy_eV: float
    xdart_wavelength_A: float
    xu_mapping_wavelength_A: float
    sample_axes: tuple[str, ...]
    detector_axes: tuple[str, ...]
    camera: tuple[str, str]
    # What this host computed; equivalent to the reference, not fingerprinted.
    solid_angle_sha256_host: str
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.asset_semantic_fingerprint) is not str
            or len(self.asset_semantic_fingerprint) != 64
            or self.detector_type != "pyFAI.detectors._dectris.Pilatus300kw"
            or self.detector_config
            != (
                ("orientation", 3),
                ("pixel1", 0.000172),
                ("pixel2", 0.000172),
            )
            or self.shape != (195, 1475)
            or self.mask_count != 2730
            or self.mask_sha256
            != "0a48766039393e4b1ffc08e70e09b3af762b90a451eabfdad392ecc773b04b28"
            or self.solid_angle_sha256
            != "a8d6453bc56b99a0c3151b54999adff4684de5648b8f23c8deb7aa0799937034"
            or self.solid_angle_sha256_host
            not in _SOLID_ANGLE_SHA256_EQUIVALENTS[self.solid_angle_sha256]
            or self.energy_eV != 17000.018
            or self.xdart_wavelength_A != 0.7293180418985439
            or self.xu_mapping_wavelength_A != 0.7293180420938394
            or self.sample_axes != ("x+", "z-", "y+", "z-")
            or self.detector_axes != ("x+", "z-")
            or self.camera != ("x-", "z+")
        ):
            raise TypeError("XU effective geometry projection is invalid")
        object.__setattr__(
            self,
            "fingerprint",
            analysis_canonical_fingerprint(
                "xu-stitch-effective-geometry-v1",
                (
                    self.asset_semantic_fingerprint,
                    self.detector_type,
                    self.detector_config,
                    self.shape,
                    self.mask_count,
                    self.mask_sha256,
                    self.solid_angle_sha256,
                    self.energy_eV,
                    self.xdart_wavelength_A,
                    self.xu_mapping_wavelength_A,
                    self.sample_axes,
                    self.detector_axes,
                    self.camera,
                ),
            ),
        )

    def to_provenance(self) -> dict[str, object]:
        return {
            "fingerprint": self.fingerprint,
            "detector_type": self.detector_type,
            "detector_config": dict(self.detector_config),
            "shape": list(self.shape),
            "mask_count": self.mask_count,
            "mask_sha256": self.mask_sha256,
            "solid_angle_sha256": self.solid_angle_sha256,
            "solid_angle_sha256_host": self.solid_angle_sha256_host,
            "energy_eV": self.energy_eV,
            "xdart_wavelength_A": self.xdart_wavelength_A,
            "xu_mapping_wavelength_A": self.xu_mapping_wavelength_A,
            "sample_axes": list(self.sample_axes),
            "detector_axes": list(self.detector_axes),
            "camera": list(self.camera),
        }


@dataclass(eq=False, frozen=True, slots=True)
class XuStitchRuntimeGeometry:
    projection: XuStitchEffectiveGeometryProjection
    hxrd: object
    mask: np.ndarray
    solid_angle: np.ndarray
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _RUNTIME_GEOMETRY_FACTORY
            or type(self.projection) is not XuStitchEffectiveGeometryProjection
            or type(self.mask) is not np.ndarray
            or self.mask.shape != self.projection.shape
            or self.mask.dtype != np.dtype(bool)
            or self.mask.flags.writeable
            or type(self.solid_angle) is not np.ndarray
            or self.solid_angle.shape != self.projection.shape
            or self.solid_angle.dtype != np.dtype(np.float64)
            or self.solid_angle.flags.writeable
        ):
            raise TypeError("XU runtime geometry is invalid")


def resolve_xu_stitch_effective_geometry(
    receipt: XuStitchCalibrationReceipt,
    session: XuRuntimeSession,
) -> XuStitchRuntimeGeometry:
    if (
        type(receipt) is not XuStitchCalibrationReceipt
        or type(session) is not XuRuntimeSession
        or session.xu is None
        or session.numpy is None
    ):
        raise TypeError("effective XU geometry requires an active exact runtime")
    revalidate_xu_stitch_calibration(receipt)
    asset = receipt.projection.value
    detector_asset = asset["detector"]
    acquisition = asset["acquisition"]
    xu_asset = asset["xrayutilities"]
    corrections = asset["corrections"]
    try:
        import pyFAI
        from pyFAI.detectors import detector_factory

        if (
            importlib.metadata.version("pyFAI") != "2026.5.0"
            or pyFAI.version != "2026.5.0"
        ):
            raise XuStitchScienceRefused(
                "XU_RUNTIME_UNSUPPORTED", "pyFAI 2026.5.0 is required"
            )
        detector = detector_factory(
            detector_asset["detector_factory_name"],
            dict(detector_asset["detector_config"]),
        )
    except XuStitchScienceRefused:
        raise
    except BaseException as error:
        raise XuStitchScienceRefused(
            "XU_DETECTOR_RECONSTRUCTION_FAILED",
            "registered detector reconstruction failed",
        ) from error
    shape = tuple(int(value) for value in detector_asset["shape"])
    config = detector.get_config()
    normalized_config = (
        ("orientation", int(config["orientation"])),
        ("pixel1", float(config["pixel1"])),
        ("pixel2", float(config["pixel2"])),
    )
    if (
        f"{type(detector).__module__}.{type(detector).__qualname__}"
        != "pyFAI.detectors._dectris.Pilatus300kw"
        or tuple(detector.shape) != shape
        or tuple(detector.max_shape) != shape
        or normalized_config
        != (
            ("orientation", 3),
            ("pixel1", 0.000172),
            ("pixel2", 0.000172),
        )
        or detector.sensor is not None
        or any(
            detector_asset[name] is not None
            for name in ("sensor_material", "sensor_thickness_m", "parallax")
        )
    ):
        raise XuStitchScienceRefused(
            "XU_SENSOR_PARALLAX_UNSUPPORTED",
            "detector or sensor reconstruction differs from SURFACE v1",
        )
    mask = np.ascontiguousarray(detector.calc_mask(), dtype=bool)
    if (
        mask.shape != shape
        or int(mask.sum()) != detector_asset["mask_count"]
        or _sha256_array(mask, "|u1") != detector_asset["mask_sha256"]
    ):
        raise XuStitchScienceRefused(
            "XU_DETECTOR_MASK_MISMATCH",
            "registered detector mask differs from SURFACE v1",
        )
    header = DetectorHeader(
        cch1=float(detector_asset["cch1_px"]),
        cch2=float(detector_asset["cch2_px"]),
        pwidth1=float(detector_asset["pixel1_m"]) * 1e3,
        pwidth2=float(detector_asset["pixel2_m"]) * 1e3,
        distance=float(detector_asset["distance_m"]) * 1e3,
        Nch1=shape[0],
        Nch2=shape[1],
    )
    solid = np.asarray(
        CorrectionStack().normalization(detector_header_to_ai(header), shape),
        dtype=np.float64,
        order="C",
    )
    solid_sha256 = _sha256_array(solid, "<f8")
    accepted_solid_sha256 = _SOLID_ANGLE_SHA256_EQUIVALENTS.get(
        corrections["solid_angle_sha256"], frozenset()
    )
    if (
        solid.shape != shape
        or not np.isfinite(solid).all()
        or np.any(solid <= 0)
        or solid_sha256 not in accepted_solid_sha256
    ):
        raise XuStitchScienceRefused(
            "XU_SOLID_ANGLE_MISMATCH",
            "solid-angle projection differs from SURFACE v1 "
            f"(host sha256={solid_sha256}; accepted={sorted(accepted_solid_sha256)})",
        )
    energy = float(acquisition["energy_eV"])
    if energy_eV_to_wavelength_m(energy) * 1e10 != acquisition["xdart_wavelength_A"]:
        raise XuStitchScienceRefused(
            "XU_WAVELENGTH_MISMATCH", "xdart wavelength projection changed"
        )
    xu = session.xu
    qconversion = xu.QConversion(
        xu_asset["sample_axes"],
        xu_asset["detector_axes"],
        xu_asset["incident_beam"],
    )
    hxrd = xu.HXRD(
        xu_asset["inplane_reference"],
        xu_asset["surface_normal_reference"],
        geometry=xu_asset["geometry"],
        en=energy,
        qconv=qconversion,
    )
    if (
        float(hxrd._wl) != acquisition["xu_mapping_wavelength_A"]
        or float(xu.en2lam(energy)) != acquisition["xu_mapping_wavelength_A"]
    ):
        raise XuStitchScienceRefused(
            "XU_WAVELENGTH_MISMATCH", "XU mapping wavelength projection changed"
        )
    hxrd.Ang2Q.init_area(
        *xu_asset["camera"],
        cch1=float(detector_asset["cch1_px"]),
        cch2=float(detector_asset["cch2_px"]),
        Nch1=shape[0],
        Nch2=shape[1],
        pwidth1=float(detector_asset["pixel1_m"]) * 1e3,
        pwidth2=float(detector_asset["pixel2_m"]) * 1e3,
        distance=float(detector_asset["distance_m"]) * 1e3,
    )
    projection = XuStitchEffectiveGeometryProjection(
        receipt.semantic_fingerprint,
        f"{type(detector).__module__}.{type(detector).__qualname__}",
        normalized_config,
        shape,
        int(mask.sum()),
        _sha256_array(mask, "|u1"),
        str(corrections["solid_angle_sha256"]),
        energy,
        float(acquisition["xdart_wavelength_A"]),
        float(hxrd._wl),
        tuple(xu_asset["sample_axes"]),
        tuple(xu_asset["detector_axes"]),
        tuple(xu_asset["camera"]),
        solid_sha256,
    )
    frozen_mask = np.frombuffer(mask.tobytes(order="C"), dtype=bool).reshape(shape)
    frozen_solid = np.frombuffer(
        np.ascontiguousarray(solid, dtype="<f8").tobytes(order="C"),
        dtype="<f8",
    ).reshape(shape)
    return XuStitchRuntimeGeometry(
        projection,
        hxrd,
        frozen_mask,
        frozen_solid,
        _RUNTIME_GEOMETRY_FACTORY,
    )


def _ultimate_ndarray_root(value: np.ndarray) -> np.ndarray:
    current = value
    seen: set[int] = set()
    for _depth in range(16):
        marker = id(current)
        if marker in seen:
            raise XuStitchScienceRefused(
                "XU_Q_ROOT_UNSUPPORTED", "cyclic ndarray base chain"
            )
        seen.add(marker)
        parent = current.base
        if not isinstance(parent, np.ndarray):
            return current
        current = parent
    raise XuStitchScienceRefused(
        "XU_Q_ROOT_UNSUPPORTED", "ndarray base chain exceeds depth 16"
    )


class XuPowderQFrameLease:
    def __init__(self, area_result: object, shape: tuple[int, int]):
        if type(area_result) is not tuple or len(area_result) != 3:
            raise XuStitchScienceRefused(
                "XU_Q_ROOT_UNSUPPORTED", "XU area result is not an exact triple"
            )
        values = tuple(np.squeeze(np.asarray(value)) for value in area_result)
        roots = tuple(_ultimate_ndarray_root(value) for value in values)
        if (
            roots[0] is not roots[1]
            or roots[1] is not roots[2]
            or roots[0].shape != (shape[0], shape[1], 3)
            or roots[0].dtype != np.dtype(np.float64)
            or any(value.shape != shape for value in values)
        ):
            raise XuStitchScienceRefused(
                "XU_Q_ROOT_UNSUPPORTED",
                "XU q components do not share the exact float64 root",
            )
        self._values = values
        self._root = roots[0]
        self._root_ref = weakref.ref(self._root)
        self._released = False

    def q_magnitude(self) -> np.ndarray:
        if self._released:
            raise RuntimeError("XU q-frame lease is released")
        qx, qy, qz = self._values
        finite_components = np.isfinite(qx) & np.isfinite(qy) & np.isfinite(qz)
        with np.errstate(over="ignore", invalid="ignore"):
            q = np.hypot(np.hypot(qx, qy), qz)
        if np.any(finite_components & ~np.isfinite(q)):
            raise XuStitchScienceRefused(
                "XU_Q_NORM_NONFINITE",
                "pairwise q norm introduced a nonfinite value",
            )
        return q

    def release(self) -> None:
        if self._released:
            return
        root_ref = self._root_ref
        self._values = None
        self._root = None
        self._root_ref = None
        self._released = True
        if root_ref() is not None:
            raise XuStitchScienceRefused(
                "STITCH_FRAME_RELEASE_FAILED",
                "XU q root remains resident after frame",
            )

    def __enter__(self) -> "XuPowderQFrameLease":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.release()
        return False


class XuPowderQProvider:
    def __init__(self, geometry: XuStitchRuntimeGeometry, acquisition: Mapping[str, object]):
        if (
            type(geometry) is not XuStitchRuntimeGeometry
            or not isinstance(acquisition, Mapping)
        ):
            raise TypeError("XU q provider requires exact runtime geometry and acquisition")
        self.geometry = geometry
        self._nu_offset = float(acquisition["detector_offsets_deg"]["nu"])
        self._del_offset = float(acquisition["detector_offsets_deg"]["del"])

    def frame(self, *, nu: float, del_: float) -> XuPowderQFrameLease:
        if any(type(value) not in {int, float} or not math.isfinite(float(value)) for value in (nu, del_)):
            raise XuStitchScienceRefused(
                "INVALID_METADATA_VALUE", "XU detector angles must be finite"
            )
        area = self.geometry.hxrd.Ang2Q.area(
            0.0,
            0.0,
            0.0,
            0.0,
            float(nu) - self._nu_offset,
            float(del_) - self._del_offset,
        )
        return XuPowderQFrameLease(area, self.geometry.projection.shape)


def _checked_frame_histograms(
    source,
    label: int,
    *,
    provider: XuPowderQProvider,
    geometry: XuStitchRuntimeGeometry,
    q_edges: np.ndarray,
    nu_value: float,
    del_value: float,
    monitor: float,
    max_frame_bytes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce one frame's 1-D contributions and prove frame release."""

    raw_image = image = q = valid = None
    lease: XuPowderQFrameLease | None = None
    owned_refs: list[weakref.ReferenceType] = []
    pending: BaseException | None = None
    d_signal = d_normalization = d_count = None
    try:
        raw_image = np.asarray(source.load_frame(label))
        if raw_image.nbytes > max_frame_bytes:
            raise XuStitchScienceRefused(
                "XU_FRAME_TOO_LARGE", "source frame exceeds max_frame_bytes"
            )
        image = np.array(raw_image, dtype=np.float64, order="C", copy=True)
        owned_refs.append(weakref.ref(image))
        raw_image = None
        if (
            image.shape != geometry.projection.shape
            or image.nbytes > max_frame_bytes
        ):
            raise XuStitchScienceRefused(
                "XU_FRAME_SHAPE_UNSUPPORTED",
                "oriented float64 frame differs from SURFACE shape/envelope",
            )
        try:
            with np.errstate(divide="raise", invalid="ignore", over="raise"):
                np.divide(image, monitor, out=image)
        except FloatingPointError as error:
            raise XuStitchScienceRefused(
                "STITCH_NUMERIC_OVERFLOW",
                "monitor normalization overflowed a finite signal",
            ) from error
        lease = provider.frame(nu=nu_value, del_=del_value)
        q = lease.q_magnitude()
        owned_refs.append(weakref.ref(q))
        valid = np.logical_not(geometry.mask)
        owned_refs.append(weakref.ref(valid))
        np.logical_and(valid, np.isfinite(q), out=valid)
        np.logical_and(valid, np.isfinite(image), out=valid)
        np.logical_and(valid, np.isfinite(geometry.solid_angle), out=valid)
        np.logical_and(valid, geometry.solid_angle > 0, out=valid)
        d_signal = np.histogram(
            q[valid], q_edges, weights=image[valid]
        )[0]
        d_normalization = np.histogram(
            q[valid], q_edges, weights=geometry.solid_angle[valid]
        )[0]
        d_count = np.histogram(q[valid], q_edges)[0].astype(
            np.float64,
            copy=False,
        )
    except BaseException as error:
        pending = error.with_traceback(None)
    finally:
        raw_image = image = q = valid = None
        release_error: BaseException | None = None
        if lease is not None:
            try:
                lease.release()
            except BaseException as error:
                release_error = error.with_traceback(None)
        lease = None
        if any(reference() is not None for reference in owned_refs):
            release_error = XuStitchScienceRefused(
                "STITCH_FRAME_RELEASE_FAILED",
                "an owned XU frame array remains resident",
            )
        owned_refs.clear()
        if release_error is not None:
            raise XuStitchScienceRefused(
                "STITCH_FRAME_RELEASE_FAILED",
                "the XU one-frame release fence failed",
            ) from release_error
    if pending is not None:
        raise pending from None
    if (
        type(d_signal) is not np.ndarray
        or type(d_normalization) is not np.ndarray
        or type(d_count) is not np.ndarray
    ):
        raise XuStitchScienceRefused(
            "STITCH_NUMERIC_OVERFLOW", "frame histograms were not produced"
        )
    return d_signal, d_normalization, d_count


@dataclass(frozen=True, slots=True)
class XuStitchScienceObservations:
    source_del_range_deg: tuple[float, float]
    source_nu_range_deg: tuple[float, float]
    source_energy_range_eV: tuple[float, float] | None
    control_domain_extrapolation_frame_count: int
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        ranges = (self.source_del_range_deg, self.source_nu_range_deg)
        if (
            any(
                type(value) is not tuple
                or len(value) != 2
                or any(type(item) is not float or not math.isfinite(item) for item in value)
                or value[0] > value[1]
                for value in ranges
            )
            or (
                self.source_energy_range_eV is not None
                and (
                    type(self.source_energy_range_eV) is not tuple
                    or len(self.source_energy_range_eV) != 2
                    or any(
                        type(item) is not float
                        or not math.isfinite(item)
                        or item <= 0
                        for item in self.source_energy_range_eV
                    )
                    or self.source_energy_range_eV[0]
                    > self.source_energy_range_eV[1]
                )
            )
            or type(self.control_domain_extrapolation_frame_count) is not int
            or self.control_domain_extrapolation_frame_count < 0
            or type(self.warnings) is not tuple
            or self.warnings
            not in {
                (),
                ("XU_CALIBRATION_EXTRAPOLATION_WITHIN_VALIDATED_SCAN",),
            }
            or bool(self.control_domain_extrapolation_frame_count)
            != bool(self.warnings)
        ):
            raise TypeError("XU Stitch observations are invalid")

    def to_provenance(self) -> dict[str, object]:
        return {
            "source_del_range_deg": list(self.source_del_range_deg),
            "source_nu_range_deg": list(self.source_nu_range_deg),
            "source_energy_range_eV": (
                None
                if self.source_energy_range_eV is None
                else list(self.source_energy_range_eV)
            ),
            "control_domain_extrapolation_frame_count": (
                self.control_domain_extrapolation_frame_count
            ),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class XuStitchScienceResult:
    payload: IntegrationResult1D
    diagnostics: StitchDiagnostics
    effective_geometry: XuStitchEffectiveGeometryProjection
    runtime: XuRuntimeExecutionRecord
    observations: XuStitchScienceObservations
    selected_frame_count: int
    release_check_frame_count: int

    def __post_init__(self) -> None:
        if (
            type(self.payload) is not IntegrationResult1D
            or type(self.diagnostics) is not StitchDiagnostics
            or type(self.effective_geometry)
            is not XuStitchEffectiveGeometryProjection
            or type(self.runtime) is not XuRuntimeExecutionRecord
            or type(self.observations) is not XuStitchScienceObservations
            or type(self.selected_frame_count) is not int
            or self.selected_frame_count < 1
            or self.release_check_frame_count != self.selected_frame_count
        ):
            raise TypeError("XU Stitch science result is invalid")


def run_xu_hist_stitch_1d(
    receipt: XuStitchCalibrationReceipt,
    source,
    *,
    frame_indices: Sequence[int],
    q_min_A_inverse: float,
    q_max_A_inverse: float,
    npt: int,
    monitor_key: str | None,
    source_energy_key: str | None = None,
    max_frame_bytes: int,
    cancel_token: threading.Event | None = None,
    progress_callback: Callable[[int, int], object] | None = None,
) -> XuStitchScienceResult:
    if type(receipt) is not XuStitchCalibrationReceipt:
        raise TypeError("XU Stitch science requires exact calibration receipt")
    labels = tuple(frame_indices)
    if (
        not labels
        or any(type(label) is not int for label in labels)
        or type(q_min_A_inverse) is not float
        or type(q_max_A_inverse) is not float
        or not math.isfinite(q_min_A_inverse)
        or not math.isfinite(q_max_A_inverse)
        or q_min_A_inverse >= q_max_A_inverse
        or type(npt) is not int
        or not 1 <= npt <= 1_000_000
        or (monitor_key is not None and (type(monitor_key) is not str or not monitor_key))
        or (
            source_energy_key is not None
            and source_energy_key != "energy"
        )
        or type(max_frame_bytes) is not int
        or not 1 <= max_frame_bytes <= 4 * 1024 * 1024 * 1024
    ):
        raise TypeError("XU Stitch science inputs are invalid")
    if cancel_token is not None and type(cancel_token) is not threading.Event:
        raise TypeError("XU Stitch cancel token must be exact Event")
    try:
        available_labels = tuple(source.frame_indices)
    except (AttributeError, TypeError) as error:
        raise TypeError("XU Stitch source must expose exact frame indices") from error
    if (
        any(type(label) is not int for label in available_labels)
        or len(set(available_labels)) != len(available_labels)
    ):
        raise TypeError("XU Stitch source frame indices are invalid")
    available_positions = {
        label: position for position, label in enumerate(available_labels)
    }
    if (
        len(set(labels)) != len(labels)
        or any(label not in available_positions for label in labels)
        or tuple(available_positions[label] for label in labels)
        != tuple(sorted(available_positions[label] for label in labels))
    ):
        raise XuStitchScienceRefused(
            "SOURCE_SELECTION_IDENTITY_MISMATCH",
            "XU Stitch labels must be a unique in-source ordered selection",
        )
    asset = receipt.projection.value
    acquisition = asset["acquisition"]
    validation = asset["validation"]
    frame_inputs: list[tuple[int, float, float, float]] = []
    del_values: list[float] = []
    nu_values: list[float] = []
    energy_values: list[float] = []
    extrapolation_count = 0
    validated_domain = validation["validated_scan_domain_source_deg"]
    control_domain = validation["control_domain_source_deg"]
    for label in labels:
        if cancel_token is not None and cancel_token.is_set():
            raise XuStitchCancelled("CANCELLED")
        metadata = source.metadata_for(label)
        if not isinstance(metadata, Mapping):
            raise XuStitchScienceRefused(
                "INVALID_METADATA_VALUE", "frame metadata is not a mapping"
            )
        try:
            del_value = float(metadata[acquisition["source_motors"]["del"]])
            nu_value = float(metadata[acquisition["source_motors"]["nu"]])
        except (KeyError, TypeError, ValueError) as error:
            raise XuStitchScienceRefused(
                "INVALID_METADATA_VALUE", "required XU angle is unavailable"
            ) from error
        if not math.isfinite(del_value) or not math.isfinite(nu_value):
            raise XuStitchScienceRefused(
                "INVALID_METADATA_VALUE", "required XU angle is nonfinite"
            )
        if not (
            float(validated_domain["del"][0])
            <= del_value
            <= float(validated_domain["del"][1])
            and float(validated_domain["nu"][0])
            <= nu_value
            <= float(validated_domain["nu"][1])
        ):
            raise XuStitchScienceRefused(
                "XU_CALIBRATION_DOMAIN_EXCEEDED",
                "frame lies outside the validated SURFACE angle domain",
            )
        if not (
            float(control_domain["del"][0])
            <= del_value
            <= float(control_domain["del"][1])
            and float(control_domain["nu"][0])
            <= nu_value
            <= float(control_domain["nu"][1])
        ):
            extrapolation_count += 1
        monitor = 1.0
        if monitor_key is not None:
            try:
                monitor = float(metadata[monitor_key])
            except (KeyError, TypeError, ValueError) as error:
                raise XuStitchScienceRefused(
                    "INVALID_MONITOR_VALUE", "monitor value is unavailable"
                ) from error
            if not math.isfinite(monitor) or monitor <= 0:
                raise XuStitchScienceRefused(
                    "INVALID_MONITOR_VALUE", "monitor must be finite and positive"
                )
        if source_energy_key is not None:
            try:
                energy = float(metadata[source_energy_key])
            except (KeyError, TypeError, ValueError) as error:
                raise XuStitchScienceRefused(
                    "XU_SOURCE_ENERGY_CONFLICT",
                    "source energy is unavailable",
                ) from error
            expected_energy = float(acquisition["energy_eV"])
            if (
                not math.isfinite(energy)
                or energy <= 0
                or abs(energy - expected_energy) > 0.001 * expected_energy
            ):
                raise XuStitchScienceRefused(
                    "XU_SOURCE_ENERGY_CONFLICT",
                    "source energy conflicts with the calibrated XU energy",
                )
            energy_values.append(energy)
        del_values.append(del_value)
        nu_values.append(nu_value)
        frame_inputs.append((label, del_value, nu_value, monitor))
    observations = XuStitchScienceObservations(
        (float(min(del_values)), float(max(del_values))),
        (float(min(nu_values)), float(max(nu_values))),
        (
            None
            if not energy_values
            else (float(min(energy_values)), float(max(energy_values)))
        ),
        extrapolation_count,
        (
            ()
            if extrapolation_count == 0
            else ("XU_CALIBRATION_EXTRAPOLATION_WITHIN_VALIDATED_SCAN",)
        ),
    )
    q_edges = np.linspace(
        q_min_A_inverse,
        q_max_A_inverse,
        npt + 1,
        dtype=np.float64,
    )
    centres = (q_edges[:-1] + q_edges[1:]) * 0.5
    if (
        not np.isfinite(q_edges).all()
        or not np.isfinite(centres).all()
        or np.any(np.diff(q_edges) <= 0)
        or (centres.size > 1 and np.any(np.diff(centres) <= 0))
    ):
        raise XuStitchScienceRefused(
            "STITCH_Q_GRID_INVALID",
            "q edges or centers are not finite and strictly increasing",
        )
    signal = np.zeros(npt, dtype=np.float64)
    normalization = np.zeros(npt, dtype=np.float64)
    count = np.zeros(npt, dtype=np.float64)
    release_count = 0
    runtime_owner = xu_runtime_session()
    with runtime_owner as session:
        geometry = resolve_xu_stitch_effective_geometry(receipt, session)
        provider = XuPowderQProvider(geometry, acquisition)
        for index, (label, del_value, nu_value, monitor) in enumerate(frame_inputs):
            if cancel_token is not None and cancel_token.is_set():
                raise XuStitchCancelled("CANCELLED")
            d_signal, d_normalization, d_count = _checked_frame_histograms(
                source,
                label,
                provider=provider,
                geometry=geometry,
                q_edges=q_edges,
                nu_value=nu_value,
                del_value=del_value,
                monitor=monitor,
                max_frame_bytes=max_frame_bytes,
            )
            release_count += 1
            if (
                not np.isfinite(d_signal).all()
                or not np.isfinite(d_normalization).all()
                or not np.isfinite(d_count).all()
                or np.any(d_normalization < 0)
                or np.any(d_count < 0)
                or np.any(d_count != np.floor(d_count))
            ):
                raise XuStitchScienceRefused(
                    "STITCH_NUMERIC_OVERFLOW", "per-frame histogram is invalid"
                )
            try:
                with np.errstate(over="raise", invalid="raise"):
                    candidate_count = count + d_count
                    candidate_signal = signal + d_signal
                    candidate_normalization = normalization + d_normalization
            except FloatingPointError as error:
                raise XuStitchScienceRefused(
                    "STITCH_NUMERIC_OVERFLOW",
                    "cumulative histogram overflowed",
                ) from error
            if np.any(candidate_count > _MAX_COUNT):
                raise XuStitchScienceRefused(
                    "STITCH_COVERAGE_STORAGE_EXACTNESS_EXCEEDED",
                    "histogram count exceeds exact float32 storage",
                )
            if (
                not np.isfinite(candidate_signal).all()
                or not np.isfinite(candidate_normalization).all()
            ):
                raise XuStitchScienceRefused(
                    "STITCH_NUMERIC_OVERFLOW", "cumulative histogram is invalid"
                )
            signal, normalization, count = (
                candidate_signal,
                candidate_normalization,
                candidate_count,
            )
            if progress_callback is not None:
                progress_callback(index + 1, len(labels))
        revalidate_xu_stitch_calibration(receipt)
        intensity = np.full(npt, np.nan, dtype=np.float64)
        occupied = normalization > 0
        try:
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                intensity[occupied] = signal[occupied] / normalization[occupied]
        except FloatingPointError as error:
            raise XuStitchScienceRefused(
                "STITCH_NUMERIC_OVERFLOW",
                "final Stitch normalization overflowed",
            ) from error
        if not np.isfinite(intensity[occupied]).all():
            raise XuStitchScienceRefused(
                "STITCH_NUMERIC_OVERFLOW", "normalized histogram is invalid"
            )
        payload = IntegrationResult1D(centres, intensity, None, "q_A^-1")
        diagnostics = StitchDiagnostics(count, normalization)
        effective_projection = geometry.projection
    if runtime_owner.execution_record is None:
        raise XuStitchScienceRefused(
            "XU_RUNTIME_RESTORE_FAILED", "XU runtime restoration was not attested"
        )
    return XuStitchScienceResult(
        payload,
        diagnostics,
        effective_projection,
        runtime_owner.execution_record,
        observations,
        len(labels),
        release_count,
    )


__all__ = [
    "XuPowderQFrameLease",
    "XuPowderQProvider",
    "XuStitchCancelled",
    "XuStitchEffectiveGeometryProjection",
    "XuStitchRuntimeGeometry",
    "XuStitchScienceRefused",
    "XuStitchScienceObservations",
    "XuStitchScienceResult",
    "resolve_xu_stitch_effective_geometry",
    "run_xu_hist_stitch_1d",
]
