"""Production PSIC powder 1-D Stitch backend using xrayutilities q mapping."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
import hashlib
import importlib.metadata
from importlib import resources
import io
import math
import threading
import weakref

import numpy as np

from xrd_tools.analysis.scan_operations import analysis_canonical_fingerprint
from xrd_tools.analysis.xu_stitch_calibration import (
    XuStitchCalibrationReceipt,
    revalidate_xu_stitch_calibration,
)
from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D, convert_radial_axis
from xrd_tools.core.energy import energy_eV_to_wavelength_m
from xrd_tools.core.geometry import DetectorHeader
from xrd_tools.core.geometry.xu_runtime import (
    XuRuntimeExecutionRecord,
    XuRuntimeSession,
    xu_runtime_session,
)
from xrd_tools.corrections.stack import CorrectionStack
from xrd_tools.integrate.multi import StitchDiagnostics
from xrd_tools.integrate.stitch_conditioning import (
    StitchConditioning,
    condition_stitch_frame,
    load_stitch_frame,
)
from xrd_tools.rsm.corrections import detector_header_to_ai


_MAX_COUNT = 2**24
_Q_ROOT_POLICY = "shared_ultimate_ndarray_root_weakref_v1"
_RUNTIME_GEOMETRY_FACTORY = object()
# SURFACE v1 pins the float64 solid-angle bytes the reference host (macOS
# arm64) computes from the asset's detector header, and ships that array as
# the ``*_solid_angle.npy`` sidecar next to the asset.  The sidecar is
# authenticated by content only: its ``<f8`` C-order bytes must hash to the
# asset's ``corrections.solid_angle_sha256`` pin, so a stale or swapped file
# can be refused but never accepted.  Other libm/SIMD builds land within an
# ULP of the reference and hash differently — PR #1 saw ubuntu-24.04
# ``0faf8164…`` and ``2a973ee0…`` on ONE runner image plus macos-15-intel
# ``38626808…``, i.e. the spread is per runner CPU, not per platform — so a
# host hash is accepted exactly when it equals the pin, and otherwise only
# when every element agrees with the reference within _SOLID_ANGLE_RTOL
# (~4500 ULP at these magnitudes, eight orders below any geometry change).
# Computation always uses the reference bytes; provenance records the host
# hash, which path accepted it, and the measured deviation.
_SOLID_ANGLE_RESOURCE_PARTS = (
    "assets",
    "xu",
    "psic_powder_1d_surface_v1_solid_angle.npy",
)
_MAX_SOLID_ANGLE_RESOURCE_BYTES = 4 * 1024 * 1024
_SOLID_ANGLE_RTOL = 1e-12
_SOLID_ANGLE_MATCHES = frozenset({"exact", "tolerance"})


class XuStitchScienceRefused(RuntimeError):
    def __init__(self, code: str, message: str | None = None):
        self.code = code
        super().__init__(message or code)


class XuStitchCancelled(RuntimeError):
    pass


def _sha256_array(values: np.ndarray, dtype: str) -> str:
    array = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _solid_angle_reference_bytes() -> bytes:
    resource = resources.files("xrd_tools").joinpath(*_SOLID_ANGLE_RESOURCE_PARTS)
    with resource.open("rb") as handle:
        return handle.read(_MAX_SOLID_ANGLE_RESOURCE_BYTES + 1)


def _solid_angle_reference(
    expected_sha256: str, shape: tuple[int, int]
) -> np.ndarray:
    """Load the packaged SURFACE v1 solid-angle array, authenticated by its pin."""
    try:
        raw = _solid_angle_reference_bytes()
        if not 1 <= len(raw) <= _MAX_SOLID_ANGLE_RESOURCE_BYTES:
            raise ValueError("solid-angle reference size is out of range")
        reference = np.load(io.BytesIO(raw), allow_pickle=False)
    except (EOFError, OSError, TypeError, ValueError) as error:
        raise XuStitchScienceRefused(
            "XU_SOLID_ANGLE_REFERENCE_UNAVAILABLE",
            "SURFACE v1 solid-angle reference resource is unavailable",
        ) from error
    if (
        type(reference) is not np.ndarray
        or reference.dtype != np.dtype("<f8")
        or reference.shape != shape
        or not np.isfinite(reference).all()
        or np.any(reference <= 0)
        or _sha256_array(reference, "<f8") != expected_sha256
    ):
        raise XuStitchScienceRefused(
            "XU_SOLID_ANGLE_REFERENCE_UNAVAILABLE",
            "SURFACE v1 solid-angle reference resource does not match its pin",
        )
    return np.ascontiguousarray(reference, dtype="<f8")


@dataclass(frozen=True, slots=True)
class XuStitchEffectiveGeometryProjection:
    asset_semantic_fingerprint: str
    detector_type: str
    detector_config: tuple[tuple[str, float | int | tuple[int, int]], ...]
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
    # What this host computed and how it was accepted against the reference;
    # host-specific, not fingerprinted (computation uses the reference bytes).
    solid_angle_sha256_host: str
    solid_angle_match: str
    solid_angle_max_rel_dev: float
    asset_version: int = 1
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.asset_semantic_fingerprint) is not str
            or len(self.asset_semantic_fingerprint) != 64
            or type(self.solid_angle_sha256_host) is not str
            or len(self.solid_angle_sha256_host) != 64
            or self.solid_angle_match not in _SOLID_ANGLE_MATCHES
            or (self.solid_angle_match == "exact")
            != (self.solid_angle_sha256_host == self.solid_angle_sha256)
            or type(self.solid_angle_max_rel_dev) is not float
            or not math.isfinite(self.solid_angle_max_rel_dev)
            or not 0.0 <= self.solid_angle_max_rel_dev <= _SOLID_ANGLE_RTOL
            or (self.solid_angle_match == "exact")
            != (self.solid_angle_max_rel_dev == 0.0)
            or type(self.asset_version) is not int
            or self.asset_version not in {1, 2, 3}
            or type(self.detector_type) is not str
            or not self.detector_type.startswith("pyFAI.detectors.")
            or len(self.shape) != 2
            or any(type(value) is not int or value <= 0 for value in self.shape)
            or type(self.mask_count) is not int
            or not 0 <= self.mask_count <= math.prod(self.shape)
            or any(type(value) is not str or len(value) != 64
                   for value in (self.mask_sha256, self.solid_angle_sha256))
            or any(not math.isfinite(value) or value <= 0 for value in (
                self.energy_eV, self.xdart_wavelength_A, self.xu_mapping_wavelength_A,
            ))
            or self.sample_axes != ("x+", "z-", "y+", "z-")
            or self.detector_axes != ("x+", "z-")
            or self.camera != ("x-", "z+")
        ):
            raise TypeError("XU effective geometry projection is invalid")
        if self.asset_version == 1 and (
            self.detector_type != "pyFAI.detectors._dectris.Pilatus300kw"
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
            or self.energy_eV != 17000.018
            or self.xdart_wavelength_A != 0.7293180418985439
            or self.xu_mapping_wavelength_A != 0.7293180420938394
        ):
            raise TypeError("XU effective geometry projection is invalid")
        object.__setattr__(
            self,
            "fingerprint",
            analysis_canonical_fingerprint(
                f"xu-stitch-effective-geometry-v{self.asset_version}",
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
        value = {
            "fingerprint": self.fingerprint,
            "detector_type": self.detector_type,
            "detector_config": dict(self.detector_config),
            "shape": list(self.shape),
            "mask_count": self.mask_count,
            "mask_sha256": self.mask_sha256,
            "solid_angle_sha256": self.solid_angle_sha256,
            "solid_angle_sha256_host": self.solid_angle_sha256_host,
            "solid_angle_match": self.solid_angle_match,
            "solid_angle_max_rel_dev": self.solid_angle_max_rel_dev,
            "energy_eV": self.energy_eV,
            "xdart_wavelength_A": self.xdart_wavelength_A,
            "xu_mapping_wavelength_A": self.xu_mapping_wavelength_A,
            "sample_axes": list(self.sample_axes),
            "detector_axes": list(self.detector_axes),
            "camera": list(self.camera),
        }
        if self.asset_version != 1:
            value["asset_version"] = self.asset_version
        return value


@dataclass(eq=False, frozen=True, slots=True)
class XuStitchRuntimeGeometry:
    projection: XuStitchEffectiveGeometryProjection
    hxrd: object
    mask: np.ndarray
    #: The static solid angle (versions 1 and 2), or None for version 3,
    #: whose ``frame_solid_angle(nu, del)`` comes from the attested pyFAI arm.
    solid_angle: np.ndarray | None
    _claim: InitVar[object] = None
    frame_solid_angle: Callable[[float, float], np.ndarray] | None = None
    #: xrayutilities sample displacement in mm (the detector distance's units), version 3 only.
    sampledis: tuple[float, float, float] | None = None

    def __post_init__(self, _claim: object) -> None:
        static = self.solid_angle is not None
        if (
            _claim is not _RUNTIME_GEOMETRY_FACTORY
            or type(self.projection) is not XuStitchEffectiveGeometryProjection
            or type(self.mask) is not np.ndarray
            or self.mask.shape != self.projection.shape
            or self.mask.dtype != np.dtype(bool)
            or self.mask.flags.writeable
            or static == (self.frame_solid_angle is not None)
            or static != (self.projection.asset_version != 3)
            or (static and (
                type(self.solid_angle) is not np.ndarray
                or self.solid_angle.shape != self.projection.shape
                or self.solid_angle.dtype != np.dtype(np.float64)
                or self.solid_angle.flags.writeable))
            or (self.sampledis is not None and (
                type(self.sampledis) is not tuple or len(self.sampledis) != 3
                or not all(type(v) is float and math.isfinite(v) for v in self.sampledis)))
        ):
            raise TypeError("XU runtime geometry is invalid")


def _static_solid_angle(asset_version, corrections, header, shape):
    """Versions 1 and 2: one solid-angle array from the detector header, digest-bound."""
    if asset_version != 1 and corrections["solid_angle_policy"] != "calibrated_pyfai_detector_header_v1":
        raise XuStitchScienceRefused("XU_SOLID_ANGLE_MISMATCH", "unsupported calibrated solid-angle policy")
    solid_values = CorrectionStack().normalization(detector_header_to_ai(header), shape)
    solid = np.asarray(
        solid_values,
        dtype=np.float64,
        order="C",
    )
    solid_sha256 = _sha256_array(solid, "<f8")
    reference_solid_sha256 = str(corrections["solid_angle_sha256"])
    if solid.shape != shape or not np.isfinite(solid).all() or np.any(solid <= 0):
        raise XuStitchScienceRefused(
            "XU_SOLID_ANGLE_MISMATCH",
            "solid-angle projection is not a finite positive SURFACE v1 array "
            f"(host sha256={solid_sha256})",
        )
    if asset_version == 1:
        reference_solid = _solid_angle_reference(reference_solid_sha256, shape)
    else:
        if solid_sha256 != reference_solid_sha256:
            raise XuStitchScienceRefused(
                "XU_SOLID_ANGLE_MISMATCH", "calibrated solid-angle digest differs from authored asset",
            )
        reference_solid = solid
    if solid_sha256 == reference_solid_sha256:
        solid_match = "exact"
        solid_max_rel_dev = 0.0
    else:
        solid_max_rel_dev = float(
            np.max(np.abs(solid - reference_solid) / reference_solid)
        )
        if not solid_max_rel_dev <= _SOLID_ANGLE_RTOL:
            raise XuStitchScienceRefused(
                "XU_SOLID_ANGLE_MISMATCH",
                "solid-angle projection differs from SURFACE v1 "
                f"(host sha256={solid_sha256}; "
                f"reference sha256={reference_solid_sha256}; "
                f"max_rel_dev={solid_max_rel_dev:.3e}; "
                f"rtol={_SOLID_ANGLE_RTOL:.0e})",
            )
        solid_match = "tolerance"
    return solid_sha256, solid_match, solid_max_rel_dev, reference_solid


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_thaw(item) for item in value]
    return value


def _extended_solid_angle(corrections, shape):
    """Version 3: per-frame solid angle from the attested pyFAI extended arm.

    The same per-frame integrators MultiGeometry uses, so the two backends weight
    each frame alike. MultiGeometry weights a frame by its absolute solid angle,
    the relative one times pixel area / dist²; a sample offset changes dist frame
    by frame. XU keeps its relative units: dist² is taken relative to the arm's
    reference distance, so a frame at that distance is weighted as before.
    """
    from xrd_tools.analysis.xu_stitch_calibration import (
        EXTENDED_SOLID_ANGLE_POLICY, extended_arm_digest,
    )
    from xrd_tools.core.containers import PONI
    from xrd_tools.core.geometry import DetectorCalibration, Diffractometer
    from xrd_tools.core.geometry.stacked_arm import StackedDetectorArm
    from xrd_tools.integrate.multi import create_multigeometry_integrators_from_geometry
    arm = _thaw(corrections["solid_angle_arm"])
    if (corrections["solid_angle_policy"] != EXTENDED_SOLID_ANGLE_POLICY
            or extended_arm_digest(arm) != corrections["solid_angle_sha256"]):
        raise XuStitchScienceRefused("XU_SOLID_ANGLE_MISMATCH", "the extended solid-angle arm is not attested")
    calibration = DetectorCalibration(
        PONI(dist=arm["dist_m"], poni1=arm["poni1_m"], poni2=arm["poni2_m"],
             wavelength=arm["wavelength_m"], detector=arm["detector"]),
        detector_config=dict(arm["detector_config"]))
    stacked = StackedDetectorArm(
        nu_motor="nu", del_motor="del", nu_offset=arm["nu_offset"], nu_scale=arm["nu_scale"],
        del_offset=arm["del_offset"], del_scale=arm["del_scale"], beam_rotation=arm["beam_rotation"],
        mount_tilt=arm["mount_tilt"], mount_roll=arm["mount_roll"], sample_offset1=arm["sample_offset1"],
        sample_offset2=arm["sample_offset2"], sample_offset3=arm["sample_offset3"])
    diffractometer = Diffractometer.from_pyfai_goniometer(
        stacked.to_pyfai_goniometer(calibration), source_motors={"nu_angle": "nu", "del_angle": "del"})

    reference_dist = float(arm["dist_m"])

    def frame_solid_angle(nu: float, del_: float) -> np.ndarray:
        integrator, = create_multigeometry_integrators_from_geometry(
            diffractometer, {"del": np.array([float(del_)]), "nu": np.array([float(nu)])})
        relative = np.ascontiguousarray(CorrectionStack().normalization(integrator, shape), dtype=np.float64)
        value = relative * (reference_dist / float(integrator.dist)) ** 2
        if value.shape != shape or not np.isfinite(value).all() or np.any(value <= 0):
            raise XuStitchScienceRefused("XU_SOLID_ANGLE_MISMATCH", "per-frame solid angle is not finite and positive")
        return value
    return frame_solid_angle


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
    asset_version = asset["version"]
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
    expected_config = dict(detector_asset["detector_config"])
    if "max_shape" in expected_config:
        expected_config["max_shape"] = tuple(expected_config["max_shape"])
        normalized_config = tuple(sorted((
            *normalized_config, ("max_shape", tuple(detector.max_shape)),
        )))
    if (
        (asset_version == 1 and (
            f"{type(detector).__module__}.{type(detector).__qualname__}"
            != "pyFAI.detectors._dectris.Pilatus300kw"
            or normalized_config != (
                ("orientation", 3),
                ("pixel1", 0.000172),
                ("pixel2", 0.000172),
            )
        ))
        or tuple(detector.shape) != shape
        or tuple(detector.max_shape) != shape
        or normalized_config != tuple(sorted(expected_config.items()))
        or float(detector.pixel1) != detector_asset["pixel1_m"]
        or float(detector.pixel2) != detector_asset["pixel2_m"]
        or detector.sensor is not None
        or any(
            detector_asset[name] is not None
            for name in ("sensor_material", "sensor_thickness_m", "parallax")
        )
    ):
        raise XuStitchScienceRefused(
            "XU_SENSOR_PARALLAX_UNSUPPORTED",
            "detector or sensor reconstruction differs from calibration",
        )
    detector_mask = detector.calc_mask()
    mask = np.zeros(shape, dtype=bool) if detector_mask is None else np.ascontiguousarray(detector_mask, dtype=bool)
    if (
        mask.shape != shape
        or int(mask.sum()) != detector_asset["mask_count"]
        or _sha256_array(mask, "|u1") != detector_asset["mask_sha256"]
    ):
        raise XuStitchScienceRefused(
            "XU_DETECTOR_MASK_MISMATCH",
            "registered detector mask differs from calibration",
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
    if asset_version == 3:
        frame_solid_angle = _extended_solid_angle(corrections, shape)
        solid_sha256 = str(corrections["solid_angle_sha256"])
        solid_match, solid_max_rel_dev, reference_solid = "exact", 0.0, None
    else:
        frame_solid_angle = None
        solid_sha256, solid_match, solid_max_rel_dev, reference_solid = _static_solid_angle(
            asset_version, corrections, header, shape)
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
    extended = {} if asset_version != 3 else {
        "detrot": float(detector_asset["detrot_deg"]), "tilt": float(detector_asset["tilt_deg"]),
        "tiltazimuth": float(detector_asset["tiltazimuth_deg"])}
    hxrd.Ang2Q.init_area(
        *xu_asset["camera"],
        cch1=float(detector_asset["cch1_px"]),
        cch2=float(detector_asset["cch2_px"]),
        Nch1=shape[0],
        Nch2=shape[1],
        pwidth1=float(detector_asset["pixel1_m"]) * 1e3,
        pwidth2=float(detector_asset["pixel2_m"]) * 1e3,
        distance=float(detector_asset["distance_m"]) * 1e3,
        **extended,
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
        solid_match,
        solid_max_rel_dev,
        asset_version,
    )
    frozen_mask = np.frombuffer(mask.tobytes(order="C"), dtype=bool).reshape(shape)
    if asset_version == 3:
        # xrayutilities takes the displacement in the detector distance's units (mm).
        return XuStitchRuntimeGeometry(
            projection, hxrd, frozen_mask, None, _RUNTIME_GEOMETRY_FACTORY,
            frame_solid_angle, tuple(float(value) for value in acquisition["sampledis_mm"]),
        )
    # The reference bytes, not the host's: every host integrates with the
    # array the projection's solid_angle_sha256 names.
    frozen_solid = np.frombuffer(
        reference_solid.tobytes(order="C"),
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

    def polarization(self, factor: float, wavelength_A: float, beam, vertical) -> np.ndarray:
        """pyFAI's polarization factor per pixel, from the elastic scattered direction.

        ``P = ½(1 + cos²2θ − f·cos2χ·sin²2θ)`` with χ measured from the horizontal
        axis (perpendicular to the beam and the vertical), so ``f = 1`` is fully
        horizontal polarization, as in pyFAI.
        """
        if self._released:
            raise RuntimeError("XU q-frame lease is released")
        k = 2 * math.pi / wavelength_A
        horizontal = np.cross(beam, vertical)
        qx, qy, qz = self._values
        # k_f = q + k·beam; beam, vertical and horizontal are orthonormal.
        along = qx * beam[0] + qy * beam[1] + qz * beam[2] + k
        up = qx * vertical[0] + qy * vertical[1] + qz * vertical[2]
        side = qx * horizontal[0] + qy * horizontal[1] + qz * horizontal[2]
        along *= along
        up *= up
        side *= side
        norm = along + up + side
        finite = np.isfinite(norm)
        if np.any(np.abs(norm[finite] / (k * k) - 1.0) > 1e-9):
            raise XuStitchScienceRefused(
                "XU_POLARIZATION_GEOMETRY", "XU q is not elastic for the mapping wavelength",
            )
        side -= up
        side *= -factor
        side += along
        side /= norm
        side += 1.0
        side *= 0.5
        return side

    def chi(self) -> np.ndarray:
        """Each pixel's azimuth about the beam (degrees, −180 to 180), pyFAI's χ: −atan2(qx, qz) in the
        SURFACE asset's frame (x vertical, the beam along y), the asset's own validation convention.
        Pinned to pyFAI's χ on scan 14's calibration pair (0.8° and 1.0° at single-circle poses; a
        mirrored or quarter-turned χ is off by 90° or more)."""
        if self._released:
            raise RuntimeError("XU q-frame lease is released")
        qx, _qy, qz = self._values
        chi = np.arctan2(qx, qz)
        np.negative(chi, out=chi)
        return np.degrees(chi, out=chi)

    def gi_coordinates(self, normal, axis) -> tuple[np.ndarray, np.ndarray]:
        """(q_ip, q_oop) for the sample normal ``normal`` in the lab and the q_ip sign ``axis``
        (see ``stitch_gi``)."""
        if self._released:
            raise RuntimeError("XU q-frame lease is released")
        from xrd_tools.integrate.stitch_gi import gi_coordinates
        return gi_coordinates(*self._values, normal, axis)

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
        # xrayutilities switches to its sample-displacement path whenever
        # ``sampledis`` is passed, so it is passed only when it is nonzero.
        displaced = ({} if self.geometry.sampledis is None or not any(self.geometry.sampledis)
                     else {"sampledis": list(self.geometry.sampledis)})
        area = self.geometry.hxrd.Ang2Q.area(
            0.0,
            0.0,
            0.0,
            0.0,
            float(nu) - self._nu_offset,
            float(del_) - self._del_offset,
            **displaced,
        )
        return XuPowderQFrameLease(area, self.geometry.projection.shape)


_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def _polarization_axes(asset) -> tuple[np.ndarray, np.ndarray]:
    """The beam and the lab vertical of an XU asset.

    The outer detector circle (nu) scatters in the horizontal plane, so its
    rotation axis is the vertical, whatever the asset's axis names.
    """
    xu_asset = asset["xrayutilities"]
    beam = np.asarray(xu_asset["incident_beam"], dtype=np.float64)
    axis = str(xu_asset["detector_axes"][0])
    if beam.shape != (3,) or not np.isfinite(beam).all() or axis[:1] not in _AXES:
        raise XuStitchScienceRefused("XU_POLARIZATION_GEOMETRY", "XU beam or detector axis is unsupported")
    beam = beam / np.linalg.norm(beam)
    vertical = np.asarray(_AXES[axis[0]])
    if abs(float(beam @ vertical)) > 1e-12:
        raise XuStitchScienceRefused("XU_POLARIZATION_GEOMETRY", "the nu circle axis is not perpendicular to the beam")
    return beam, vertical


def _radial_coordinates(q, unit, wavelength_A):
    if unit == "q_A^-1":
        return q
    argument = q * (wavelength_A / (4 * np.pi))
    if not np.isfinite(argument).all() or np.any(argument < 0) or np.any(argument > 1 + 1e-12):
        raise XuStitchScienceRefused("STITCH_RADIAL_DOMAIN_INVALID", "Q is outside the physical two-theta domain")
    return convert_radial_axis(q, "q_A^-1", "2th_deg", wavelength_A)


def _scout_radial_range(provider, frame_inputs, unit, wavelength_A, cancel_token):
    """Full geometry extent over every selected angle, without image rereads."""
    low, high = math.inf, -math.inf
    for _label, del_value, nu_value, _monitor in frame_inputs:
        if cancel_token is not None and cancel_token.is_set():
            raise XuStitchCancelled("CANCELLED")
        lease = provider.frame(nu=nu_value, del_=del_value)
        values = None
        try:
            values = _radial_coordinates(lease.q_magnitude(), unit, wavelength_A)
            if not np.isfinite(values).all():
                raise XuStitchScienceRefused("STITCH_Q_GRID_INVALID", "nonfinite detector coordinates")
            low = min(low, float(values.min()))
            high = max(high, float(values.max()))
        finally:
            values = None
            lease.release()
    return low, high


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
    unit: str = "q_A^-1",
    wavelength_A: float | None = None,
    conditioning: StitchConditioning | None = None,
    mask: np.ndarray | None = None,
    frame_record_callback: Callable[[int, np.ndarray], None] | None = None,
    polarization: tuple | None = None,
    dark: np.ndarray | None = None,
    flat: np.ndarray | None = None,
    gi_cuts: tuple | None = None,
    map_edges: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce one frame's 1-D contributions and prove frame release.

    As in pyFAI, a dark is subtracted from the conditioned frame, and the flat
    and polarization weight the normalization next to the solid angle.
    ``gi_cuts`` (normal, axis, ((cut, window, edges), …)) also bins the frame's GI
    cuts from the same pixels and weights, the GI coordinates computed once; their
    histograms, one triple per cut, are the fifth value; with ``gi_cuts``'
    optional fourth item, the (q_ip edges, q_oop edges) of a GI Stitch 2D's map,
    the frame's q_ip-q_oop map (q_ip, q_oop) is one more triple after the cuts'.
    ``map_edges`` (radial, χ) also bins the frame's pixels over the (radial, χ)
    map from the same pixels and weights; its triple (radial, azimuthal) is the
    sixth value, else None.
    """

    raw_image = image = q = valid = weights = factor = solid = None
    cut = other = inside = q_ip = q_oop = chi = None
    gi_histograms = ()
    map_histograms = None
    lease: XuPowderQFrameLease | None = None
    owned_refs: list[weakref.ReferenceType] = []
    pending: BaseException | None = None
    d_signal = d_normalization = d_count = None
    try:
        raw_image = load_stitch_frame(source, label, conditioning)
        if raw_image.nbytes > max_frame_bytes:
            raise XuStitchScienceRefused(
                "XU_FRAME_TOO_LARGE", "source frame exceeds max_frame_bytes"
            )
        if frame_record_callback is not None:
            frame_record_callback(label, raw_image)
        image = condition_stitch_frame(raw_image, conditioning)
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
        if dark is not None:
            np.subtract(image, dark, out=image)
        try:
            with np.errstate(divide="raise", invalid="ignore", over="raise"):
                np.divide(image, monitor, out=image)
        except FloatingPointError as error:
            raise XuStitchScienceRefused(
                "STITCH_NUMERIC_OVERFLOW",
                "monitor normalization overflowed a finite signal",
            ) from error
        lease = provider.frame(nu=nu_value, del_=del_value)
        q = _radial_coordinates(lease.q_magnitude(), unit, wavelength_A)
        owned_refs.append(weakref.ref(q))
        valid = np.logical_not(geometry.mask)
        owned_refs.append(weakref.ref(valid))
        if mask is not None:
            np.logical_and(valid, np.logical_not(mask), out=valid)
        np.logical_and(valid, np.isfinite(q), out=valid)
        np.logical_and(valid, np.isfinite(image), out=valid)
        solid = geometry.solid_angle
        if solid is None:
            solid = geometry.frame_solid_angle(nu_value, del_value)
            owned_refs.append(weakref.ref(solid))
        np.logical_and(valid, np.isfinite(solid), out=valid)
        np.logical_and(valid, solid > 0, out=valid)
        d_signal = np.histogram(
            q[valid], q_edges, weights=image[valid]
        )[0]
        weights = solid
        if polarization is not None or flat is not None:
            weights = np.array(solid, dtype=np.float64, copy=True)
            owned_refs.append(weakref.ref(weights))
            if polarization is not None:
                factor = lease.polarization(*polarization)
                owned_refs.append(weakref.ref(factor))
                np.multiply(weights, factor, out=weights)
                factor = None
            if flat is not None:
                np.multiply(weights, flat, out=weights)
        d_normalization = np.histogram(
            q[valid], q_edges, weights=weights[valid]
        )[0]
        d_count = np.histogram(q[valid], q_edges)[0].astype(
            np.float64,
            copy=False,
        )
        if gi_cuts is not None:
            normal, axis, cuts, *gi_map_edges = gi_cuts
            q_ip, q_oop = lease.gi_coordinates(normal, axis)
            owned_refs.extend((weakref.ref(q_ip), weakref.ref(q_oop)))
            histograms = []
            for axis, window, gi_edges in cuts:
                cut, other = (q_ip, q_oop) if axis == "q_ip" else (q_oop, q_ip)
                inside = np.logical_and(valid, np.isfinite(cut))
                if window is not None:
                    np.logical_and(inside, (other >= window[0]) & (other <= window[1]), out=inside)
                histograms.append((np.histogram(cut[inside], gi_edges, weights=image[inside])[0],
                                   np.histogram(cut[inside], gi_edges, weights=weights[inside])[0],
                                   np.histogram(cut[inside], gi_edges)[0].astype(np.float64, copy=False)))
                inside = None
            if gi_map_edges:  # inside both ranges, as each cut keeps the pixels in the other's
                inside = np.logical_and(valid, np.isfinite(q_ip))
                np.logical_and(inside, np.isfinite(q_oop), out=inside)
                histograms.append(tuple(
                    np.histogram2d(q_ip[inside], q_oop[inside], gi_map_edges[0], weights=values)[0]
                    for values in (image[inside], weights[inside], None)))
                inside = None
            gi_histograms = tuple(histograms)
            cut = other = q_ip = q_oop = None
        if map_edges is not None:
            chi = lease.chi()
            owned_refs.append(weakref.ref(chi))
            inside = np.logical_and(valid, np.isfinite(chi))
            map_histograms = (
                np.histogram2d(q[inside], chi[inside], map_edges, weights=image[inside])[0],
                np.histogram2d(q[inside], chi[inside], map_edges, weights=weights[inside])[0],
                np.histogram2d(q[inside], chi[inside], map_edges)[0],
            )
            inside = chi = None
        # Whole-frame monitor-normalised total: the dropped-frame statistic. Not
        # the in-range histogram, which legitimately shrinks at the range edges.
        d_total = float(image[valid].sum())
    except BaseException as error:
        pending = error.with_traceback(None)
    finally:
        raw_image = image = q = valid = weights = factor = solid = None
        cut = other = inside = q_ip = q_oop = chi = None
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
    return d_signal, d_normalization, d_count, d_total, gi_histograms, map_histograms


def _scout_gi_bins(provider, frame_inputs, normals, axes, gi, cancel_token):
    """Both GI coordinates' ranges, as the pyFAI kernel resolves them (``multi._gi_bins``): each its
    own range, else its full geometry extent inside the other's range over every selected frame (one
    pass); a cut is binned over its own and windowed by the other's."""
    from xrd_tools.integrate.stitch_gi import GI_CUTS
    bins = {cut: gi.bins(cut) for cut in GI_CUTS}
    scouted = [cut for cut in GI_CUTS if bins[cut] is None]
    if not scouted:
        return bins
    low, high = dict.fromkeys(scouted, math.inf), dict.fromkeys(scouted, -math.inf)
    for (_label, del_value, nu_value, _monitor), normal, axis in zip(frame_inputs, normals, axes):
        if cancel_token is not None and cancel_token.is_set():
            raise XuStitchCancelled("CANCELLED")
        lease = provider.frame(nu=nu_value, del_=del_value)
        try:
            q_ip, q_oop = lease.gi_coordinates(normal, axis)
            for cut in scouted:
                values, other = (q_ip, q_oop) if cut == "q_ip" else (q_oop, q_ip)
                inside = np.isfinite(values)
                window = gi.window(cut)
                if window is not None:
                    inside &= (other >= window[0]) & (other <= window[1])
                if inside.any():
                    low[cut] = min(low[cut], float(values[inside].min()))
                    high[cut] = max(high[cut], float(values[inside].max()))
        finally:
            q_ip = q_oop = values = other = inside = None
            lease.release()
    for cut in scouted:
        if not low[cut] < high[cut]:
            raise XuStitchScienceRefused("STITCH_GI_RANGE_EMPTY", "no pixel lies inside the GI window")
        bins[cut] = (low[cut], high[cut])
    return bins


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
    #: A GI Stitch's cuts beside I(q) (``stitch_gi``), each (result, diagnostics) in the plan's
    #: order; () otherwise.
    gi_cuts: tuple[tuple[IntegrationResult1D, StitchDiagnostics], ...] = ()
    #: A Stitch 2D's (radial, χ) map beside I(q), (result, diagnostics); None otherwise.
    map: tuple[IntegrationResult2D, StitchDiagnostics] | None = None
    #: A GI Stitch 2D's q_ip-q_oop map, (result, diagnostics); None otherwise.
    gi_map: tuple[IntegrationResult2D, StitchDiagnostics] | None = None

    def __post_init__(self) -> None:
        for value in (self.map, self.gi_map):
            if value is not None and (type(value) is not tuple or len(value) != 2
                                      or type(value[0]) is not IntegrationResult2D
                                      or type(value[1]) is not StitchDiagnostics):
                raise TypeError("XU Stitch map is invalid")
        if type(self.gi_cuts) is not tuple or any(
                type(cut) is not tuple or len(cut) != 2 or type(cut[0]) is not IntegrationResult1D
                or type(cut[1]) is not StitchDiagnostics for cut in self.gi_cuts):
            raise TypeError("XU Stitch GI result is invalid")
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
    q_min_A_inverse: float | None = None,
    q_max_A_inverse: float | None = None,
    npt: int,
    monitor_key: str | None,
    source_energy_key: str | None = None,
    max_frame_bytes: int,
    cancel_token: threading.Event | None = None,
    progress_callback: Callable[[int, int], object] | None = None,
    unit: str = "q_A^-1",
    radial_range: tuple[float, float] | None = None,
    conditioning: StitchConditioning | None = None,
    mask: np.ndarray | None = None,
    frame_record_callback: Callable[[int, np.ndarray], None] | None = None,
    frame_screen: object = None,
    polarization_factor: float | None = None,
    dark: np.ndarray | None = None,
    flat: np.ndarray | None = None,
    flat_min_fraction: float = 0.1,
    gi=None,
    map_grid: tuple[int, int, tuple[float, float]] | None = None,
) -> XuStitchScienceResult:
    """Run native Q or 2theta histogram bins; omitted bounds scout all geometry.

    Legacy q_min/q_max calls and an absent conditioning policy retain the
    canonical input-reader behavior. ``mask`` is already in calibrated image
    coordinates and is combined with, never substituted for, the detector mask.
    The caller must supply frames already oriented by the calibration asset's
    ``image_rotation_quarter_turns``. This kernel does not rotate them;
    ``StitchOperationExecution`` applies the rotation for native plans.
    ``frame_record_callback`` borrows each raw frame before conditioning.
    ``polarization_factor``, ``dark`` and ``flat`` (oriented like the frames)
    apply pyFAI's corrections; an unusable dark or flat pixel joins the mask.
    ``gi`` (a ``StitchGI``) also bins its GI cuts beside I(q), with ``npt`` points:
    every frame's sample normal comes from psic's four sample circles (eta from the
    incidence, chi from the tilt; ``StitchGI.normal``, xrayutilities' own transform
    for the SURFACE asset), and the result carries them as ``gi_cuts``.
    ``map_grid`` (radial points, χ points, χ range in degrees) also bins a Stitch 2D's
    (radial, χ) map over the radial range from the same pixels and weights as I(q),
    χ as pyFAI's (``XuPowderQFrameLease.chi``); the result carries it as ``map``. With
    ``gi`` too, the q_ip-q_oop map (the same points over the cuts' ranges) is ``gi_map``.
    """
    from xrd_tools.integrate.stitch_gi import GI_UNITS, StitchGI
    if map_grid is not None and (
            type(map_grid) is not tuple or len(map_grid) != 3
            or any(type(points) is not int or not 1 <= points <= 1_000_000 for points in map_grid[:2])
            or type(map_grid[2]) is not tuple or len(map_grid[2]) != 2
            or any(type(value) not in {int, float} or not math.isfinite(value) for value in map_grid[2])
            or map_grid[2][0] >= map_grid[2][1]):
        raise ValueError("a Stitch 2D map needs its radial and χ points, a finite increasing χ range")
    if gi is not None and (type(gi) is not StitchGI or unit != "q_A^-1"):
        raise TypeError("an XU GI cut needs an exact StitchGI beside I(q) in Q")
    if type(receipt) is not XuStitchCalibrationReceipt:
        raise TypeError("XU Stitch science requires exact calibration receipt")
    if unit not in {"q_A^-1", "2th_deg"}:
        raise ValueError("XU Stitch supports Q or two-theta native bins")
    if q_min_A_inverse is not None or q_max_A_inverse is not None:
        if (type(q_min_A_inverse) is not float or type(q_max_A_inverse) is not float
                or unit != "q_A^-1" or radial_range is not None):
            raise TypeError("legacy Q bounds require two floats and cannot be combined with another axis/range")
        radial_range = (q_min_A_inverse, q_max_A_inverse)
    if radial_range is not None and (
        type(radial_range) is not tuple or len(radial_range) != 2
        or any(type(v) not in {int, float} or not math.isfinite(v) for v in radial_range)
        or radial_range[0] >= radial_range[1]
    ):
        raise ValueError("Stitch radial range must be finite and increasing")
    if conditioning is not None and type(conditioning) is not StitchConditioning:
        raise TypeError("conditioning must be exact StitchConditioning or absent")
    if polarization_factor is not None and (
            isinstance(polarization_factor, bool) or not isinstance(polarization_factor, (int, float))
            or not math.isfinite(polarization_factor) or not -1.0 <= polarization_factor <= 1.0):
        raise ValueError("polarization factor must be a finite number from -1 to 1, or None")
    if frame_record_callback is not None and not callable(frame_record_callback):
        raise TypeError("frame-record callback must be callable")
    labels = tuple(frame_indices)
    if (
        not labels
        or any(type(label) is not int for label in labels)
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
    normals: list[np.ndarray] = []  # GI: each frame's sample normal, from psic's circles
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
            # None: the circle a single 2θ arm holds at zero
            del_name, nu_name = (acquisition["source_motors"][circle] for circle in ("del", "nu"))
            del_value = 0.0 if del_name is None else float(metadata[del_name])
            nu_value = 0.0 if nu_name is None else float(metadata[nu_name])
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
            # a refined calibration's scans are checked against the energy they recorded
            expected_energy = float(acquisition.get("source_energy_eV", acquisition["energy_eV"]))
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
        if gi is not None:
            try:
                normals.append(gi.normal(metadata))
            except (KeyError, TypeError, ValueError) as error:
                raise XuStitchScienceRefused(
                    "INVALID_METADATA_VALUE", "a GI sample angle is unavailable or nonfinite"
                ) from error
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
    # [signal, normalization, count] of I(q) and, for GI, of the cut, and of a Stitch 2D's map
    totals = [np.zeros(npt, dtype=np.float64) for _ in range(3)]
    gi_totals = () if gi is None else tuple([np.zeros(npt, dtype=np.float64) for _ in range(3)]
                                            for _cut in gi.cuts)
    if gi is not None and map_grid is not None:  # the GI map, after the cuts
        gi_totals += ([np.zeros(map_grid[:2], dtype=np.float64) for _ in range(3)],)
    map_totals = None if map_grid is None else [np.zeros(map_grid[:2], dtype=np.float64) for _ in range(3)]
    release_count = 0

    def add(target, d_signal, d_normalization, d_count):
        try:
            with np.errstate(over="raise", invalid="raise"):
                candidate = [target[0] + d_signal, target[1] + d_normalization, target[2] + d_count]
        except FloatingPointError as error:
            raise XuStitchScienceRefused(
                "STITCH_NUMERIC_OVERFLOW",
                "cumulative histogram overflowed",
            ) from error
        if np.any(candidate[2] > _MAX_COUNT):
            raise XuStitchScienceRefused(
                "STITCH_COVERAGE_STORAGE_EXACTNESS_EXCEEDED",
                "histogram count exceeds exact float32 storage",
            )
        if not np.isfinite(candidate[0]).all() or not np.isfinite(candidate[1]).all():
            raise XuStitchScienceRefused(
                "STITCH_NUMERIC_OVERFLOW", "cumulative histogram is invalid"
            )
        target[:] = candidate

    def accumulate(d_signal, d_normalization, d_count, gi_histograms=(), map_histograms=None):
        add(totals, d_signal, d_normalization, d_count)
        for target, histograms in zip(gi_totals, gi_histograms, strict=True):
            add(target, *histograms)
        if map_histograms is not None:
            add(map_totals, *map_histograms)
    runtime_owner = xu_runtime_session()
    with runtime_owner as session:
        geometry = resolve_xu_stitch_effective_geometry(receipt, session)
        provider = XuPowderQProvider(geometry, acquisition)
        wavelength_A = energy_eV_to_wavelength_m(geometry.projection.energy_eV) * 1e10
        if mask is not None:
            if type(mask) is not np.ndarray or mask.dtype != np.dtype(bool) or mask.shape != geometry.projection.shape:
                raise ValueError("external mask must be boolean with the calibrated detector shape")
        from xrd_tools.integrate.multi import _detector_correction
        dark, dark_invalid = _detector_correction(dark, "dark", 0.0)
        flat, flat_invalid = _detector_correction(flat, "flat", 1.0, flat_min_fraction)
        for name, value, invalid in (("dark", dark, dark_invalid), ("flat", flat, flat_invalid)):
            if value is not None and value.shape != geometry.projection.shape:
                raise ValueError(f"Stitch {name} must have the calibrated detector shape")
            if invalid is not None:
                mask = invalid if mask is None else np.logical_or(mask, invalid)
        polarization = None
        if polarization_factor is not None:
            polarization = (float(polarization_factor), geometry.projection.xu_mapping_wavelength_A,
                            *_polarization_axes(asset))
        bounds = radial_range if radial_range is not None else _scout_radial_range(
            provider, frame_inputs, unit, wavelength_A, cancel_token,
        )
        q_edges = np.linspace(*bounds, npt + 1, dtype=np.float64)
        centres = (q_edges[:-1] + q_edges[1:]) * 0.5
        if (not np.isfinite(q_edges).all() or not np.isfinite(centres).all()
                or np.any(np.diff(q_edges) <= 0)
                or (centres.size > 1 and np.any(np.diff(centres) <= 0))):
            raise XuStitchScienceRefused("STITCH_Q_GRID_INVALID", "radial grid is not finite and strictly increasing")
        map_edges = None
        if map_grid is not None:
            if not str(asset["xrayutilities"].get("chi_validation_convention", "")).startswith("minus_atan2_qx_qz"):
                raise XuStitchScienceRefused("XU_CHI_CONVENTION_UNSUPPORTED",
                                             "a Stitch 2D map takes the SURFACE asset's χ convention")
            map_edges = (np.linspace(*bounds, map_grid[0] + 1, dtype=np.float64),
                         np.linspace(*map_grid[2], map_grid[1] + 1, dtype=np.float64))
        gi_cuts = [None] * len(frame_inputs)
        if gi is not None:
            frame = asset["xrayutilities"]
            if (list(frame["incident_beam"]) != [0.0, 1.0, 0.0]
                    or list(frame["surface_normal_reference"]) != [0.0, 0.0, 1.0]):
                raise XuStitchScienceRefused("XU_GI_FRAME_UNSUPPORTED", "GI takes psic's SURFACE frame")
            axes = [gi.in_plane_axis(normal) for normal in normals]
            gi_bins = _scout_gi_bins(provider, frame_inputs, normals, axes, gi, cancel_token)
            gi_edges = {cut: np.linspace(*gi_bins[cut], npt + 1, dtype=np.float64) for cut in gi.cuts}
            other = {"q_ip": "q_oop", "q_oop": "q_ip"}
            cuts = tuple((cut, gi_bins[other[cut]], gi_edges[cut]) for cut in gi.cuts)
            gi_map_edges = () if map_grid is None else ((
                np.linspace(*gi_bins["q_ip"], map_grid[0] + 1, dtype=np.float64),
                np.linspace(*gi_bins["q_oop"], map_grid[1] + 1, dtype=np.float64)),)
            gi_cuts = [(normal, axis, cuts, *gi_map_edges) for normal, axis in zip(normals, axes)]
        for index, (label, del_value, nu_value, monitor) in enumerate(frame_inputs):
            if cancel_token is not None and cancel_token.is_set():
                raise XuStitchCancelled("CANCELLED")
            d_signal, d_normalization, d_count, d_total, d_gi, d_map = _checked_frame_histograms(
                source,
                label,
                provider=provider,
                geometry=geometry,
                q_edges=q_edges,
                nu_value=nu_value,
                del_value=del_value,
                monitor=monitor,
                max_frame_bytes=max_frame_bytes,
                unit=unit,
                wavelength_A=wavelength_A,
                conditioning=conditioning,
                mask=mask,
                frame_record_callback=frame_record_callback,
                polarization=polarization,
                dark=dark,
                flat=flat,
                gi_cuts=gi_cuts[index],
                map_edges=map_edges,
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
            if frame_screen is None:
                accumulate(d_signal, d_normalization, d_count, d_gi, d_map)
            else:
                # Hold the frame until its row neighbours are known.
                for held, verdict in frame_screen.push(
                        d_total, (d_signal, d_normalization, d_count, d_gi, d_map)):
                    if not verdict.excluded:
                        accumulate(*held)
            if progress_callback is not None:
                progress_callback(index + 1, len(labels))
        if frame_screen is not None:
            for held, verdict in frame_screen.finish():
                if not verdict.excluded:
                    accumulate(*held)
            if not any(not verdict.excluded for verdict in frame_screen.verdicts):
                raise XuStitchScienceRefused(
                    "STITCH_ALL_FRAMES_DROPPED", "every frame was excluded as a dropped frame"
                )
        revalidate_xu_stitch_calibration(receipt)
        payload, diagnostics = _normalized(centres, *totals, unit)
        gi_results = tuple(
            _normalized((gi_edges[cut][:-1] + gi_edges[cut][1:]) * 0.5, *cut_totals, GI_UNITS[cut])
            for cut, cut_totals in zip(() if gi is None else gi.cuts, gi_totals))
        gi_map_result = None
        if gi is not None and map_grid is not None:
            (ip_edges, oop_edges), = gi_map_edges
            gi_map_result = _normalized_map((ip_edges[:-1] + ip_edges[1:]) * 0.5,
                                            (oop_edges[:-1] + oop_edges[1:]) * 0.5, *gi_totals[-1],
                                            GI_UNITS["q_ip"], GI_UNITS["q_oop"])
        map_result = None
        if map_grid is not None:
            radial_edges, chi_edges = map_edges
            map_result = _normalized_map((radial_edges[:-1] + radial_edges[1:]) * 0.5,
                                         (chi_edges[:-1] + chi_edges[1:]) * 0.5, *map_totals, unit)
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
        gi_results,
        map_result,
        gi_map_result,
    )


def _normalized_map(radial, azimuthal, signal, normalization, count, unit, azimuthal_unit="chi_deg"):
    """A Stitch 2D's map as ``_normalized`` makes I(q), in the (radial, azimuthal) orientation: the
    (radial, χ) map, or the (q_ip, q_oop) one of a GI Stitch 2D."""
    return IntegrationResult2D(radial, azimuthal, _normalized_intensity(signal, normalization), None, unit,
                               azimuthal_unit), StitchDiagnostics(count, normalization)


def _normalized(centres, signal, normalization, count, unit):
    """Σ signal / Σ normalization where a bin has any weight (NaN elsewhere), with its diagnostics."""
    return (IntegrationResult1D(centres, _normalized_intensity(signal, normalization), None, unit),
            StitchDiagnostics(count, normalization))


def _normalized_intensity(signal, normalization):
    intensity = np.full(normalization.shape, np.nan, dtype=np.float64)
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
    return intensity


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
