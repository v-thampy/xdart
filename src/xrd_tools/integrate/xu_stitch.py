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
from xrd_tools.corrections.frame_preparation import (
    FramePreparationRefused, correction_image, frame_valid, frame_weights, polarization_axes,
    polarization_weight, prepare_frame_signal,
)
from xrd_tools.corrections.normalization import BeamNormalizationRefused, frame_divisors
from xrd_tools.corrections.stack import CorrectionStack
from xrd_tools.integrate.frame_workers import parallel_frames
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
    # Acquisition energy is separate from the immutable calibration attestation.
    # None preserves the calibration's exact projection and historical fingerprint.
    runtime_energy: tuple[float, float, float] | None = None
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if self.runtime_energy is not None and (
            type(self.runtime_energy) is not tuple or len(self.runtime_energy) != 3
            or any(type(v) is not float or not math.isfinite(v) or v <= 0
                   for v in self.runtime_energy)
        ):
            raise TypeError("XU acquisition energy projection is invalid")
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
            # psic's nu x+ and del z-, either reversed for a circle turning the other way
            or self.detector_axes not in {(nu, delta) for nu in ("x+", "x-") for delta in ("z-", "z+")}
            or (self.asset_version != 3 and self.detector_axes != ("x+", "z-"))
            or self.camera not in ({("x-", "z+"), ("x+", "z-")} if self.asset_version == 3
                                   else {("x-", "z+")})
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
                ) + (() if self.runtime_energy is None else ("acquisition-energy", self.runtime_energy)),
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
        if self.runtime_energy is not None:
            value["acquisition_energy"] = dict(zip(
                ("energy_eV", "xdart_wavelength_A", "xu_mapping_wavelength_A"), self.runtime_energy,
                strict=True))
        return value

    @property
    def runtime_xdart_wavelength_A(self) -> float:
        return self.xdart_wavelength_A if self.runtime_energy is None else self.runtime_energy[1]

    @property
    def runtime_xu_wavelength_A(self) -> float:
        return self.xu_mapping_wavelength_A if self.runtime_energy is None else self.runtime_energy[2]


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


def _extended_solid_angle(corrections, shape, *, sample_offset=True):
    """Version 3: per-frame solid angle from the attested pyFAI extended arm.

    The same per-frame integrators MultiGeometry uses, so the two backends weight
    each frame alike. MultiGeometry weights a frame by its absolute solid angle,
    the relative one times pixel area / dist²; a sample offset changes dist frame
    by frame. XU keeps its relative units: dist² is taken relative to the arm's
    reference distance, so a frame at that distance is weighted as before.

    ``sample_offset`` False: the arm with the calibrant's sample offset at zero (RSM,
    whose q never takes it), after the attestation; the arm then turns about the
    sample, so the weight is the same at every pose.
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
    if not sample_offset:
        arm = {**arm, "sample_offset1": 0.0, "sample_offset2": 0.0, "sample_offset3": 0.0}
    from xrd_tools.integrate.turned_detector import frame_detector_record
    # the arm's model in the partner's frame: itself, or (a quarter-turned partner, AUD-9 item 8) pyFAI's generic
    # detector of the partner's shape and the arm's pitches
    name, config = frame_detector_record(arm["detector"], arm["detector_config"], shape)
    calibration = DetectorCalibration(
        PONI(dist=arm["dist_m"], poni1=arm["poni1_m"], poni2=arm["poni2_m"],
             wavelength=arm["wavelength_m"], detector=name),
        detector_config=dict(config))
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
    *,
    metadata_energy_eV: float | None = None,
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

        from xrd_tools.integrate.turned_detector import partner_detector

        if (
            pyFAI.version not in ("2026.5.0", "2026.9.0")
            or importlib.metadata.version("pyFAI") != pyFAI.version
        ):
            raise XuStitchScienceRefused(
                "XU_RUNTIME_UNSUPPORTED", "pyFAI 2026.5.0 or 2026.9.0 is required"
            )
        # the named model, or (a quarter-turned partner, AUD-9 item 8) pyFAI's generic detector of the partner's
        # shape with the model's own mask turned as the images are
        detector, detector_mask = partner_detector(detector_asset, acquisition)
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
    if float(xu.en2lam(energy)) != acquisition["xu_mapping_wavelength_A"]:
        raise XuStitchScienceRefused("XU_WAVELENGTH_MISMATCH", "XU mapping wavelength projection changed")
    runtime_energy = None
    if metadata_energy_eV is not None:
        from xrd_tools.core.energy import resolve_acquisition_energy
        selected = resolve_acquisition_energy(float(acquisition["xdart_wavelength_A"]) * 1e-10,
            (metadata_energy_eV,), fitted_source_energy_eV=acquisition.get("source_energy_eV"))
        if not math.isclose(selected["energy_eV"], energy, rel_tol=1e-14):
            runtime_energy = (float(selected["energy_eV"]), float(selected["wavelength_m"]) * 1e10,
                              float(xu.en2lam(selected["energy_eV"])))
    qconversion = xu.QConversion(
        xu_asset["sample_axes"],
        xu_asset["detector_axes"],
        xu_asset["incident_beam"],
    )
    hxrd = xu.HXRD(
        xu_asset["inplane_reference"],
        xu_asset["surface_normal_reference"],
        geometry=xu_asset["geometry"],
        en=energy if runtime_energy is None else runtime_energy[0],
        qconv=qconversion,
    )
    if (
        float(hxrd._wl) != (acquisition["xu_mapping_wavelength_A"] if runtime_energy is None else runtime_energy[2])
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
        float(acquisition["xu_mapping_wavelength_A"]),
        tuple(xu_asset["sample_axes"]),
        tuple(xu_asset["detector_axes"]),
        tuple(xu_asset["camera"]),
        solid_sha256,
        solid_match,
        solid_max_rel_dev,
        asset_version,
        runtime_energy,
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
    def __init__(self, area_result: object, shape: tuple[int, int], *, camera=("x+", "z-")):
        self._camera = camera
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
        """pyFAI's polarization factor per pixel, from the elastic scattered direction (the shared
        ``corrections.frame_preparation.polarization_weight``, RSM's too)."""
        if self._released:
            raise RuntimeError("XU q-frame lease is released")
        qx, qy, qz = self._values
        try:
            return polarization_weight(qx, qy, qz, factor=factor, wavelength_A=wavelength_A, beam=beam,
                                       vertical=vertical)
        except ValueError:
            raise XuStitchScienceRefused(
                "XU_POLARIZATION_GEOMETRY", "XU q is not elastic for the mapping wavelength",
            ) from None

    def chi(self) -> np.ndarray:
        """pyFAI χ in the asset's image frame: native atan2(qx,-qz), frozen old frame -atan2(qx,qz)."""
        if self._released:
            raise RuntimeError("XU q-frame lease is released")
        qx, _qy, qz = self._values
        if self._camera == ("x+", "z-"):
            chi = np.empty_like(qz)
            np.negative(qz, out=chi)
            np.arctan2(qx, chi, out=chi)
        else:
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
        return XuPowderQFrameLease(area, self.geometry.projection.shape, camera=self.geometry.projection.camera)


def _polarization_axes(asset) -> tuple[np.ndarray, np.ndarray]:
    """The beam and the lab vertical of an XU asset (``corrections.frame_preparation.polarization_axes``)."""
    xu_asset = asset["xrayutilities"]
    try:
        return polarization_axes(xu_asset["incident_beam"], xu_asset["detector_axes"][0])
    except ValueError as error:
        message = ("the nu circle axis is not perpendicular to the beam" if "perpendicular" in str(error)
                   else "XU beam or detector axis is unsupported")
        raise XuStitchScienceRefused("XU_POLARIZATION_GEOMETRY", message) from None


def _radial_coordinates(q, unit, wavelength_A):
    if unit == "q_A^-1":
        return q
    argument = q * (wavelength_A / (4 * np.pi))
    if not np.isfinite(argument).all() or np.any(argument < 0) or np.any(argument > 1 + 1e-12):
        raise XuStitchScienceRefused("STITCH_RADIAL_DOMAIN_INVALID", "Q is outside the physical two-theta domain")
    return convert_radial_axis(q, "q_A^-1", "2th_deg", wavelength_A)


def _scout_chi_range(provider, frame_inputs, cancel_token):
    """A Stitch 2D's Auto χ range: every selected angle's χ extent, without image rereads (as
    ``_scout_radial_range``); a single value widens by half a degree each way."""
    low, high = math.inf, -math.inf
    for _label, del_value, nu_value, _monitor in frame_inputs:
        if cancel_token is not None and cancel_token.is_set():
            raise XuStitchCancelled("CANCELLED")
        lease = provider.frame(nu=nu_value, del_=del_value)
        try:
            chi = lease.chi()
            finite = chi[np.isfinite(chi)]
            if finite.size:
                low, high = min(low, float(finite.min())), max(high, float(finite.max()))
        finally:
            lease.release()
    if not low <= high:
        raise XuStitchScienceRefused("STITCH_Q_GRID_INVALID", "no finite χ on the detector")
    return (low, high) if low < high else (low - 0.5, high + 0.5)


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


def _xu_frame_arrays(
    source,
    label: int,
    *,
    provider: XuPowderQProvider,
    geometry: XuStitchRuntimeGeometry,
    nu_value: float,
    del_value: float,
    monitor: float,
    max_frame_bytes: int,
    unit: str = "q_A^-1",
    wavelength_A: float | None = None,
    conditioning: StitchConditioning | None = None,
    saturation: int | None = None,
    mask: np.ndarray | None = None,
    frame_record_callback: Callable[[int, np.ndarray], None] | None = None,
    polarization: tuple | None = None,
    dark: np.ndarray | None = None,
    flat: np.ndarray | None = None,
    gi_cuts: tuple | None = None,
    map_edges: tuple[np.ndarray, np.ndarray] | None = None,
    map_unit: str | None = None,
    chi_window: tuple[float, float] | None = None,
    point_images: bool = False,
) -> tuple[dict, list]:
    """One frame's pixels and coordinates for its histograms (``_xu_frame_histograms``): every
    xrayutilities call of the frame, made on the thread its runtime session belongs to, whose lease is
    released here. Returns (the arrays by name, weak references to those the frame owns); the histograms
    empty the first and prove every one of the second released (the one-frame release fence).

    As in pyFAI, a dark is subtracted from the conditioned frame, and the flat and polarization weight
    the normalization next to the solid angle. With ``gi_cuts`` (normal, axis, …) the frame's GI
    coordinates, and with ``map_edges`` or ``chi_window`` its χ (and with ``map_edges`` the map's radial
    coordinate, in ``map_unit`` when it is not ``unit``), are computed once for the cuts', the maps' and
    I(q)'s histograms.
    """

    raw_image = image = q = valid = weights = factor = solid = None
    q_ip = q_oop = chi = radial = magnitude = None
    lease: XuPowderQFrameLease | None = None
    owned_refs: list[weakref.ReferenceType] = []
    pending: BaseException | None = None
    arrays = None
    try:
        if point_images:
            point_loader = getattr(source, "load_point_frame", None)
            if callable(point_loader):
                raw_image = np.asarray(point_loader(label, conditioning=conditioning, saturation=saturation))
            else:
                frame = source.frame_for(label)
                statistics = frame.point_statistics_loader
                if not callable(statistics):
                    raise XuStitchScienceRefused(
                        "POINT_IMAGE_LOADER_UNAVAILABLE", "grouped frame has no native point loader"
                    )
                raw_image = np.asarray(statistics(conditioning=conditioning, saturation=saturation).signal)
                frame = statistics = None
            point_loader = None
        else:
            raw_image = load_stitch_frame(source, label, conditioning)
        if raw_image.nbytes > max_frame_bytes:
            raise XuStitchScienceRefused(
                "XU_FRAME_TOO_LARGE", "source frame exceeds max_frame_bytes"
            )
        if frame_record_callback is not None:
            frame_record_callback(label, raw_image)
        # A grouped loader already conditions each native operand. Applying the
        # threshold to the result would discard legitimate signed differences.
        image = (np.array(raw_image, dtype=np.float64, order="C", copy=True) if point_images else
                 condition_stitch_frame(raw_image, conditioning, saturation))
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
        try:  # the dark, then the frame's divisor: the shared preparation (RSM's too)
            prepare_frame_signal(image, dark=dark, divisor=monitor)
        except FramePreparationRefused as error:
            raise XuStitchScienceRefused(
                "STITCH_NUMERIC_OVERFLOW",
                "monitor normalization overflowed a finite signal",
            ) from error
        lease = provider.frame(nu=nu_value, del_=del_value)
        magnitude = lease.q_magnitude()
        owned_refs.append(weakref.ref(magnitude))
        q = _radial_coordinates(magnitude, unit, wavelength_A)
        owned_refs.append(weakref.ref(q))
        if map_edges is not None and map_unit is not None and map_unit != unit:  # the map's own axis
            radial = _radial_coordinates(magnitude, map_unit, wavelength_A)
            owned_refs.append(weakref.ref(radial))
        magnitude = None
        solid = geometry.solid_angle
        if solid is None:
            solid = geometry.frame_solid_angle(nu_value, del_value)
            owned_refs.append(weakref.ref(solid))
        # the shared valid pixels and weights (solid angle × polarization × flat, as in pyFAI; RSM's too)
        valid = frame_valid(geometry.mask, mask=mask, finite=(q, image), positive=(solid,))
        owned_refs.append(weakref.ref(valid))
        if polarization is not None:
            factor = lease.polarization(*polarization)
            owned_refs.append(weakref.ref(factor))
        weights = frame_weights(solid, polarization=factor, flat=flat)
        if weights is not solid:
            owned_refs.append(weakref.ref(weights))
        factor = None
        if gi_cuts is not None:
            q_ip, q_oop = lease.gi_coordinates(gi_cuts[0], gi_cuts[1])
            owned_refs.extend((weakref.ref(q_ip), weakref.ref(q_oop)))
        if map_edges is not None or chi_window is not None:
            chi = lease.chi()
            owned_refs.append(weakref.ref(chi))
        arrays = dict(image=image, q=q, valid=valid, weights=weights, q_ip=q_ip, q_oop=q_oop, chi=chi,
                      radial=radial)
    except BaseException as error:
        pending = error.with_traceback(None)
        arrays = None
    finally:
        raw_image = image = q = valid = weights = factor = solid = None
        q_ip = q_oop = chi = radial = magnitude = None
        release_error: BaseException | None = None
        if lease is not None:
            try:
                lease.release()
            except BaseException as error:
                release_error = error.with_traceback(None)
        lease = None
        if pending is not None and any(reference() is not None for reference in owned_refs):
            release_error = XuStitchScienceRefused(
                "STITCH_FRAME_RELEASE_FAILED",
                "an owned XU frame array remains resident",
            )
        if release_error is not None:
            arrays = None
            owned_refs.clear()
            raise XuStitchScienceRefused(
                "STITCH_FRAME_RELEASE_FAILED",
                "the XU one-frame release fence failed",
            ) from release_error
    if pending is not None:
        raise pending from None
    return arrays, owned_refs


def _xu_frame_histograms(
    arrays: dict,
    owned_refs: list,
    *,
    q_edges: np.ndarray,
    gi_cuts: tuple | None = None,
    map_edges: tuple[np.ndarray, np.ndarray] | None = None,
    chi_window: tuple[float, float] | None = None,
    positive_total_screening: bool = True,
) -> tuple:
    """One frame's 1-D contributions from its ``_xu_frame_arrays`` (NumPy only: a frame worker's share
    with several workers), emptying ``arrays`` and proving the frame released before it returns.
    ``chi_window`` (°) bins I(q) from the pixels whose χ (their centre's) is inside it, ends included; the
    cuts and the map keep every pixel (their own windows and ranges).

    ``gi_cuts`` (normal, axis, ((cut, window, edges), …)) also bins the frame's GI cuts from the same pixels
    and weights; their histograms, one triple per cut, are the fifth value; with ``gi_cuts``' optional
    fourth item, the (q_ip edges, q_oop edges) of a GI Stitch 2D's map, the frame's q_ip-q_oop map (q_ip,
    q_oop) is one more triple after the cuts'. ``map_edges`` (radial, χ) also bins the frame's pixels over
    the (radial, χ) map from the same pixels and weights; its triple (radial, azimuthal) is the sixth
    value, else None.
    """
    image = q = valid = weights = None
    cut = other = inside = q_ip = q_oop = chi = radial = profile = None
    gi_histograms = ()
    map_histograms = None
    pending: BaseException | None = None
    d_signal = d_normalization = d_count = None
    try:
        image, q, valid, weights = arrays["image"], arrays["q"], arrays["valid"], arrays["weights"]
        profile = valid  # I(q)'s pixels: every valid one, or those inside its χ window
        if chi_window is not None:
            chi = arrays["chi"]
            profile = np.logical_and(valid, np.isfinite(chi))
            np.logical_and(profile, chi >= chi_window[0], out=profile)
            np.logical_and(profile, chi <= chi_window[1], out=profile)
            chi = None
        d_signal = np.histogram(
            q[profile], q_edges, weights=image[profile]
        )[0]
        d_normalization = np.histogram(
            q[profile], q_edges, weights=weights[profile]
        )[0]
        d_count = np.histogram(q[profile], q_edges)[0].astype(
            np.float64,
            copy=False,
        )
        profile = None
        if gi_cuts is not None:
            _normal, _axis, cuts, *gi_map_edges = gi_cuts
            q_ip, q_oop = arrays["q_ip"], arrays["q_oop"]
            histograms = []
            for axis, window, gi_edges in cuts:
                cut, other = (q_ip, q_oop) if axis == "q_ip" else (q_oop, q_ip)
                inside = np.logical_and(valid, np.isfinite(cut))
                if window is not None:
                    np.logical_and(inside, (other >= window[0]) & (other <= window[1]), out=inside)
                # Exclude unbinned weights before NumPy's cumulative weighted histogram:
                # an Auto and a covering window must sum the same pixels identically.
                np.logical_and(inside, (cut >= gi_edges[0]) & (cut <= gi_edges[-1]), out=inside)
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
            chi = arrays["chi"]
            radial = arrays["radial"]
            inside = np.logical_and(valid, np.isfinite(chi))
            if radial is None:
                radial = q
            else:
                np.logical_and(inside, np.isfinite(radial), out=inside)
            map_histograms = (
                np.histogram2d(radial[inside], chi[inside], map_edges, weights=image[inside])[0],
                np.histogram2d(radial[inside], chi[inside], map_edges, weights=weights[inside])[0],
                np.histogram2d(radial[inside], chi[inside], map_edges)[0],
            )
            inside = chi = radial = None
        # Whole-frame monitor-normalised total: the dropped-frame statistic. Not
        # the in-range histogram, which legitimately shrinks at the range edges.
        # Difference totals can be negative without a dropped exposure. NaN is
        # FrameScreen's existing per-label unscreened statistic; other labels
        # in a mixed source retain their positive-count screening.
        d_total = float(image[valid].sum()) if positive_total_screening else float("nan")
    except BaseException as error:
        pending = error.with_traceback(None)
    finally:
        image = q = valid = weights = None
        cut = other = inside = q_ip = q_oop = chi = radial = profile = None
        arrays.clear()
        released = not any(reference() is not None for reference in owned_refs)
        owned_refs.clear()
        if not released:
            raise XuStitchScienceRefused(
                "STITCH_FRAME_RELEASE_FAILED",
                "the XU one-frame release fence failed",
            ) from XuStitchScienceRefused(
                "STITCH_FRAME_RELEASE_FAILED",
                "an owned XU frame array remains resident",
            )
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
    validated_domain_extrapolation_frame_count: int = 0

    @staticmethod
    def extrapolation_warnings(control_count, validated_count):
        if validated_count:
            return ("XU_CALIBRATION_DOMAIN_EXCEEDED_ALLOWED",)
        return ("XU_CALIBRATION_EXTRAPOLATION_WITHIN_VALIDATED_SCAN",) if control_count else ()

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
            or type(self.validated_domain_extrapolation_frame_count) is not int
            or not 0 <= self.validated_domain_extrapolation_frame_count <= self.control_domain_extrapolation_frame_count
            or type(self.warnings) is not tuple
            or self.warnings != self.extrapolation_warnings(
                self.control_domain_extrapolation_frame_count, self.validated_domain_extrapolation_frame_count)
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
            **({"validated_domain_extrapolation_frame_count": self.validated_domain_extrapolation_frame_count}
               if self.validated_domain_extrapolation_frame_count else {}),
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
    foil_key: str | None = None,
    exposure_key: str | None = None,
    absorption_lengths: tuple[float, float, float, float] | None = None,
    source_energy_key: str | None = None,
    max_frame_bytes: int,
    cancel_token: threading.Event | None = None,
    progress_callback: Callable[[int, int], object] | None = None,
    unit: str = "q_A^-1",
    radial_range: tuple[float, float] | None = None,
    conditioning: StitchConditioning | None = None,
    saturation: Mapping[int, int] | None = None,
    mask: np.ndarray | None = None,
    frame_record_callback: Callable[[int, np.ndarray], None] | None = None,
    frame_screen: object = None,
    polarization_factor: float | None = None,
    dark: np.ndarray | None = None,
    flat: np.ndarray | None = None,
    flat_min_fraction: float = 0.1,
    gi=None,
    map_grid: tuple[int, int, tuple[float, float]] | None = None,
    map_unit: str | None = None,
    map_radial_range: tuple[float, float] | None = None,
    workers: int = 1,
    azimuth_range_1d: tuple[float, float] | None = None,
    allow_extrapolation: bool = False,
    metadata_energy_eV: float | None = None,
) -> XuStitchScienceResult:
    """Run native Q or 2theta histogram bins; omitted bounds scout all geometry.

    ``allow_extrapolation`` admits poses beyond the export validation envelope,
    recording their count and warning; all other scientific checks still apply.

    ``workers`` above 1 bins that many frames at once on threads (``frame_workers``): every
    xrayutilities call stays on this thread, the one its runtime session belongs to, and only the
    NumPy histograms run on the workers; frames merge in order, so the result is the same, bit for bit.

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
    ``map_unit`` gives the map its own radial unit and ``map_radial_range`` its own range (None:
    its frames' extent), as Int's 2-D integration beside its 1-D one; None shares I(q)'s.
    ``azimuth_range_1d`` (°) is I(q)'s χ window, as pyFAI's (None: every χ): the pixels whose χ is
    inside it, where pyFAI's pixel splitting also takes a pixel that only reaches into it. The GI cuts and
    the map keep their own windows and ranges.
    """
    from xrd_tools.integrate.stitch_gi import GI_UNITS, StitchGI
    if map_grid is not None and (
            type(map_grid) is not tuple or len(map_grid) != 3
            or any(type(points) is not int or not 1 <= points <= 1_000_000 for points in map_grid[:2])
            or map_grid[2] is not None and (
                type(map_grid[2]) is not tuple or len(map_grid[2]) != 2
                or any(type(value) not in {int, float} or not math.isfinite(value) for value in map_grid[2])
                or map_grid[2][0] >= map_grid[2][1])):
        raise ValueError("a Stitch 2D map needs its radial and χ points, a finite increasing χ range")
    if gi is not None and (type(gi) is not StitchGI or unit != "q_A^-1"):
        raise TypeError("an XU GI cut needs an exact StitchGI beside I(q) in Q")
    if map_unit is not None and (map_grid is None or map_unit not in {"q_A^-1", "2th_deg"}
                                 or (gi is not None and map_unit != "q_A^-1")):
        raise ValueError("a Stitch 2D map's own axis is Q or 2theta (Q with GI)")
    if map_radial_range is not None and (
            map_unit is None or type(map_radial_range) is not tuple or len(map_radial_range) != 2
            or any(type(v) not in {int, float} or not math.isfinite(v) for v in map_radial_range)
            or map_radial_range[0] >= map_radial_range[1]):
        raise ValueError("a Stitch 2D map's radial range must be finite and increasing, with its unit")
    if azimuth_range_1d is not None and (
            type(azimuth_range_1d) is not tuple or len(azimuth_range_1d) != 2
            or any(type(v) not in {int, float} or not math.isfinite(v) for v in azimuth_range_1d)
            or azimuth_range_1d[0] >= azimuth_range_1d[1]):
        raise ValueError("I(q)'s χ window must be finite and increasing")
    if type(allow_extrapolation) is not bool:
        raise TypeError("allow_extrapolation must be an exact bool")
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
        # decision 4's foil choices: a foil column with its lengths, with an exposure or a monitor, never both
        or (foil_key is not None and (type(foil_key) is not str or not foil_key or absorption_lengths is None
                                      or (exposure_key is None) is (monitor_key is None)))
        or (foil_key is None and (exposure_key is not None or absorption_lengths is not None))
        or (exposure_key is not None and (type(exposure_key) is not str or not exposure_key))
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
    point_images: list[bool] = []
    differences: list[bool] = []
    normals: list[np.ndarray] = []  # GI: each frame's sample normal, from psic's circles
    del_values: list[float] = []
    nu_values: list[float] = []
    energy_values: list[float] = []
    extrapolation_count = 0
    validated_extrapolation_count = 0
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
        point_policy = metadata.get("point_images")
        grouped = isinstance(point_policy, Mapping)
        point_images.append(grouped)
        differences.append(grouped and point_policy.get("operation") == "difference")
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
            if not allow_extrapolation:
                raise XuStitchScienceRefused(
                    "XU_CALIBRATION_DOMAIN_EXCEEDED",
                    "frame lies outside the validated SURFACE angle domain",
                )
            validated_extrapolation_count += 1
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
        if foil_key is not None:  # decision 4's foil choices: the shared divisor, RSM's too
            try:
                foil_value = float(metadata[foil_key])
            except (KeyError, TypeError, ValueError) as error:
                raise XuStitchScienceRefused(
                    "FOIL_STATUS_INVALID", "foil status is unavailable"
                ) from error
            try:
                exposure_value = None if exposure_key is None else float(metadata[exposure_key])
            except (KeyError, TypeError, ValueError) as error:
                raise XuStitchScienceRefused(
                    "EXPOSURE_INVALID", "exposure is unavailable"
                ) from error
            try:
                monitor = float(frame_divisors(
                    1, monitor=None if monitor_key is None else [monitor], foil=[foil_value],
                    exposure=None if exposure_value is None else [exposure_value],
                    absorption_lengths=absorption_lengths)[0])
            except BeamNormalizationRefused as error:
                raise XuStitchScienceRefused(error.code, str(error)) from None
        if source_energy_key is not None:
            try:
                energy = float(metadata[source_energy_key])
            except (KeyError, TypeError, ValueError) as error:
                raise XuStitchScienceRefused(
                    "XU_SOURCE_ENERGY_CONFLICT",
                    "source energy is unavailable",
                ) from error
            # a refined calibration's scans are checked against the energy they recorded
            expected_energy = (float(acquisition.get("source_energy_eV", acquisition["energy_eV"]))
                               if metadata_energy_eV is None else metadata_energy_eV)
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
        XuStitchScienceObservations.extrapolation_warnings(extrapolation_count, validated_extrapolation_count),
        validated_extrapolation_count,
    )
    # [signal, normalization, count] of I(q) and, for GI, of the cut, and of a Stitch 2D's map
    totals = [np.zeros(npt, dtype=np.float64) for _ in range(3)]
    gi_totals = () if gi is None else tuple([np.zeros(npt, dtype=np.float64) for _ in range(3)]
                                            for _cut in gi.cuts)
    if gi is not None and map_grid is not None:  # the GI map, after the cuts
        gi_totals += ([np.zeros(map_grid[:2], dtype=np.float64) for _ in range(3)],)
    map_totals = None if map_grid is None else [np.zeros(map_grid[:2], dtype=np.float64) for _ in range(3)]
    if type(workers) is not int or workers < 1:
        raise ValueError("Stitch frame workers must be a positive int")
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
        geometry = resolve_xu_stitch_effective_geometry(receipt, session, metadata_energy_eV=metadata_energy_eV)
        provider = XuPowderQProvider(geometry, acquisition)
        wavelength_A = geometry.projection.runtime_xdart_wavelength_A
        if mask is not None:
            if type(mask) is not np.ndarray or mask.dtype != np.dtype(bool) or mask.shape != geometry.projection.shape:
                raise ValueError("external mask must be boolean with the calibrated detector shape")
        # admitted by Int's owners (corrections.frame_preparation, shared with the pyFAI path and RSM)
        dark, dark_invalid = correction_image(dark, "dark")
        flat, flat_invalid = correction_image(flat, "flat", min_fraction=flat_min_fraction)
        for name, value in (("dark", dark), ("flat", flat)):
            if value is not None and value.shape != geometry.projection.shape:
                raise ValueError(f"Stitch {name} must have the calibrated detector shape")
        if flat_invalid is not None:
            mask = flat_invalid if mask is None else np.logical_or(mask, flat_invalid)
        mask_without_dark = mask
        if dark_invalid is not None:
            mask = dark_invalid if mask is None else np.logical_or(mask, dark_invalid)
        polarization = None
        if polarization_factor is not None:
            polarization = (float(polarization_factor), geometry.projection.runtime_xu_wavelength_A,
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
        if (map_grid is not None or azimuth_range_1d is not None) and not str(
                asset["xrayutilities"].get("chi_validation_convention", "")).startswith("minus_atan2_qx_qz"):
            raise XuStitchScienceRefused("XU_CHI_CONVENTION_UNSUPPORTED",
                                         "a Stitch 2D map or χ window takes the SURFACE asset's χ convention")
        if map_grid is not None:
            chi_range = map_grid[2] if map_grid[2] is not None else _scout_chi_range(
                provider, frame_inputs, cancel_token)  # Auto: the frames' own χ extent
            # the map's own radial axis (Auto: its frames' extent), else I(q)'s
            map_bounds = bounds if map_unit is None or (map_unit, map_radial_range) == (unit, radial_range) else (
                map_radial_range if map_radial_range is not None else _scout_radial_range(
                    provider, frame_inputs, map_unit, wavelength_A, cancel_token))
            map_edges = (np.linspace(*map_bounds, map_grid[0] + 1, dtype=np.float64),
                         np.linspace(*chi_range, map_grid[1] + 1, dtype=np.float64))
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
        merged = 0

        def read(index):
            """The frame's pixels and coordinates: its xrayutilities work, on this (the session's) thread."""
            if cancel_token is not None and cancel_token.is_set():
                raise XuStitchCancelled("CANCELLED")
            label, del_value, nu_value, monitor = frame_inputs[index]
            return _xu_frame_arrays(
                source,
                label,
                provider=provider,
                geometry=geometry,
                nu_value=nu_value,
                del_value=del_value,
                monitor=monitor,
                max_frame_bytes=max_frame_bytes,
                unit=unit,
                wavelength_A=wavelength_A,
                conditioning=conditioning,
                saturation=None if saturation is None else saturation.get(label),
                mask=mask_without_dark if differences[index] else mask,
                frame_record_callback=frame_record_callback,
                polarization=polarization,
                dark=None if differences[index] else dark,
                flat=flat,
                gi_cuts=gi_cuts[index],
                map_edges=map_edges,
                map_unit=map_unit,
                chi_window=azimuth_range_1d,
                point_images=point_images[index],
            )

        def compute(index, arrays, owned_refs):
            """The frame's histograms (NumPy only: a frame worker's share)."""
            return _xu_frame_histograms(arrays, owned_refs, q_edges=q_edges, gi_cuts=gi_cuts[index],
                                        map_edges=map_edges, chi_window=azimuth_range_1d,
                                        positive_total_screening=not differences[index])

        def merge(histograms):
            """Add one frame's histograms, in frame order (the screen holds a frame until its row
            neighbours are known)."""
            nonlocal release_count, merged
            d_signal, d_normalization, d_count, d_total, d_gi, d_map = histograms
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
                for held, verdict in frame_screen.push(
                        d_total, (d_signal, d_normalization, d_count, d_gi, d_map)):
                    if not verdict.excluded:
                        accumulate(*held)
            merged += 1
            if progress_callback is not None:
                progress_callback(merged, len(labels))

        if progress_callback is not None:  # the frames' stitching has begun: 0 merged
            progress_callback(0, len(labels))
        if workers == 1:
            for index in range(len(frame_inputs)):
                merge(compute(index, *read(index)))
        else:
            # a read frame holds its arrays (pyFAI's holds a frame and a cold integrator): one ahead of the
            # workers, not two each (scan 14 at 4 workers: the same 5.8 s, about 130 MiB less at its peak)
            parallel_frames(len(frame_inputs), read, compute, merge, workers, read_ahead=workers + 1)
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
                                         (chi_edges[:-1] + chi_edges[1:]) * 0.5, *map_totals,
                                         unit if map_unit is None else map_unit)
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
