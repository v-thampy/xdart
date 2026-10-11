"""Shared Qt-free admission of Int calibration, mask, flat and dark assets.

Relocated without scientific-policy changes from the ordinary Int preflight.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import tempfile
from typing import Callable
import numpy as np
from xrd_tools.corrections.detector import DEFAULT_MIN_FLAT_FRACTION, DetectorImageCheck, check_detector_image
from xrd_tools.integrate.calibration import apply_sensor_parallax, load_detector_calibration
from xrd_tools.io import load_mask
from xrd_tools.io.image import read_detector_image_layout, read_image
from xrd_tools.io.stat_identity import identity_ctime_ns
from xrd_tools.session.run_configuration import RunIntent
load_poni = load_detector_calibration
_LOG = logging.getLogger(__name__)
_SCIENTIFIC_ASSET_STREAM_CHUNK_BYTES = 1 << 16
_SCIENTIFIC_MASK_COMPATIBILITY_FLOOR_BYTES = 64 << 20
_SCIENTIFIC_MASK_HEADER_ALLOWANCE_BYTES = 1 << 20
_SCIENTIFIC_MASK_MAX_DECODED_BYTES = 256 << 20

def _not_cancelled():
    return False

@dataclass(frozen=True, slots=True)
class AcceptedScientificAssets:
    poni_values: tuple[float, float, float, float, float, float, float, str] | None
    mask_dtype: str | None
    mask_shape: tuple[int, ...] | None
    mask_bytes: bytes | None
    poni_sha256: str | None
    mask_sha256: str | None
    poni_detector_config_json: str | None = None
    poni_parallax: bool | None = None
    #: The admitted flat field and dark as prepared float32 (unusable pixels NaN).
    flat_shape: tuple[int, int] | None = None
    flat_bytes: bytes | None = None
    flat_sha256: str | None = None
    dark_shape: tuple[int, int] | None = None
    dark_bytes: bytes | None = None
    dark_sha256: str | None = None
    #: Readiness warnings about the flat or dark (see check_detector_image), shown
    #: when the run starts; they change nothing about the science.
    correction_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for role, name in (("flat", "flat field"), ("dark", "dark")):
            shape, payload, digest = (getattr(self, f"{role}_{part}")
                                      for part in ("shape", "bytes", "sha256"))
            if any(value is not None for value in (shape, payload, digest)) and not (
                    type(shape) is tuple and len(shape) == 2
                    and all(type(value) is int and value > 0 for value in shape)
                    and type(payload) is bytes and len(payload) == 4 * shape[0] * shape[1]
                    and type(digest) is str and len(digest) == 64):
                raise TypeError(f"accepted {name} is invalid")
        if type(self.correction_warnings) is not tuple or not all(
                type(item) is str for item in self.correction_warnings):
            raise TypeError("accepted correction warnings are invalid")
        values = self.poni_values
        if values is not None and not (
                type(values) is tuple and len(values) == 8
                and all(type(value) is float and math.isfinite(value) for value in values[:7])
                and values[0] > 0 and values[6] >= 0 and type(values[7]) is str):
            raise TypeError("accepted PONI is invalid")
        parts = self.mask_dtype, self.mask_shape, self.mask_bytes
        if any(value is not None for value in parts) and not (
                type(self.mask_dtype) is str and type(self.mask_shape) is tuple
                and all(type(value) is int and value >= 0 for value in self.mask_shape)
                and type(self.mask_bytes) is bytes):
            raise TypeError("accepted detector mask is invalid")
        if not all(value is None or type(value) is str
                   for value in (self.poni_sha256, self.mask_sha256)):
            raise TypeError("accepted scientific assets are invalid")
        config_text = self.poni_detector_config_json
        if (values is None) != (config_text is None):
            raise TypeError("accepted PONI and detector config must be paired")
        if self.poni_parallax is not None and type(self.poni_parallax) is not bool:
            raise TypeError("accepted PONI parallax is invalid")
        if values is None and self.poni_parallax is not None:
            raise TypeError("accepted PONI parallax has no calibration")
        if config_text is not None:
            try:
                config = json.loads(config_text)
                canonical = json.dumps(
                    config, sort_keys=True, separators=(",", ":"), allow_nan=False,
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise TypeError("accepted detector config is invalid") from exc
            orientation = config.get("orientation") if type(config) is dict else None
            if (canonical != config_text or type(orientation) is not int
                    or orientation not in range(1, 5)):
                raise TypeError("accepted detector config is not canonical")
            sensor = config.get("sensor")
            if self.poni_parallax is None:
                if sensor is not None:
                    raise TypeError("legacy accepted PONI cannot contain a sensor")
            else:
                try:
                    from xrd_tools.integrate.calibration import (
                        validate_sensor_parallax,
                    )
                    validate_sensor_parallax(
                        sensor.get("material") if type(sensor) is dict else None,
                        sensor.get("thickness") if type(sensor) is dict else None,
                        self.poni_parallax,
                        wavelength_m=values[6],
                    )
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        "accepted PONI sensor/parallax is invalid"
                    ) from exc

    @property
    def poni(self) -> object | None:
        from xrd_tools.core.containers import PONI
        return None if self.poni_values is None else PONI(*self.poni_values)

    @property
    def detector_calibration(self) -> object | None:
        if self.poni_values is None:
            return None
        from xrd_tools.integrate.calibration import (
            detector_calibration_from_projection,
        )

        projection = dict(self.poni.to_dict())
        config = json.loads(self.poni_detector_config_json)
        if self.poni_parallax is not None:
            projection["detector_config"] = config
            projection["parallax"] = self.poni_parallax
        return detector_calibration_from_projection(
            projection,
            detector_config=config,
        )

    @property
    def poni_projection(self) -> dict[str, object] | None:
        if self.poni_values is None:
            return None
        from xrd_tools.integrate.calibration import (
            detector_calibration_projection,
        )
        return detector_calibration_projection(self.detector_calibration)

    @property
    def signed_poni_values(self) -> dict[str, object] | None:
        """The projection a v3 (sensor and parallax) calibration signs into its run, else None.

        Admission's candidate and the start's freeze both take it, so the run they admit and the
        run that starts carry one calibration identity.
        """
        return self.poni_projection if self.poni_parallax is not None else None

    @property
    def mask(self) -> np.ndarray | None:
        return None if self.mask_bytes is None else np.frombuffer(
            self.mask_bytes, dtype=np.dtype(self.mask_dtype)).reshape(
                self.mask_shape).copy()

    @property
    def flat(self) -> np.ndarray | None:
        return None if self.flat_bytes is None else np.frombuffer(
            self.flat_bytes, dtype=np.float32).reshape(self.flat_shape).copy()

    @property
    def dark(self) -> np.ndarray | None:
        return None if self.dark_bytes is None else np.frombuffer(
            self.dark_bytes, dtype=np.float32).reshape(self.dark_shape).copy()


def load_scientific_assets(
    intent: RunIntent,
    *,
    cancelled: Callable[[], bool] = _not_cancelled,
    calibration_loader=None,
) -> AcceptedScientificAssets:
    if cancelled():
        raise RuntimeError("admission cancelled")
    read_calibration = load_poni if calibration_loader is None else calibration_loader
    calibration, poni_digest = _load_stable_asset(
        intent.poni_file,
        lambda path, data: read_calibration(path, data=data)
        if read_calibration is load_detector_calibration else read_calibration(path),
        max_bytes=1 << 20,
    )
    # The optional v3 override is absent from legacy intent projections and
    # helper-owned, RunIntent-compatible views.
    override = getattr(intent, "poni_v3_override", None)
    if override is not None:
        if calibration is None:
            raise ValueError(
                "an enabled PONI v3 override requires a selected PONI"
            )
        calibration = apply_sensor_parallax(
            calibration,
            material=override.material,
            thickness_m=override.thickness_m,
            parallax=override.parallax,
        )
    mask, mask_digest = _load_stable_mask_asset(
        intent.mask_file, calibration, cancelled=cancelled,
    )
    detector = (
        _detector_facts(calibration)
        if getattr(intent, "flat_file", "") or getattr(intent, "dark_file", "") else None
    )
    flat, flat_digest, flat_warnings = _load_stable_correction_asset(
        getattr(intent, "flat_file", ""), "flat", cancelled=cancelled, detector=detector,
        min_fraction=getattr(intent, "flat_min_fraction", DEFAULT_MIN_FLAT_FRACTION),
    )
    dark, dark_digest, dark_warnings = _load_stable_correction_asset(
        getattr(intent, "dark_file", ""), "dark", cancelled=cancelled, detector=detector,
    )
    poni = None if calibration is None else calibration.poni
    poni_values = None if poni is None else (
        float(poni.dist), float(poni.poni1), float(poni.poni2), float(poni.rot1),
        float(poni.rot2), float(poni.rot3), float(poni.wavelength),
        str(poni.detector),
    )
    accepted = (
        None if mask is None
        else np.ascontiguousarray(mask, dtype=bool)
    )
    config_json = None if calibration is None else json.dumps(
        dict(calibration.detector_config), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    return AcceptedScientificAssets(
        poni_values, None if accepted is None else accepted.dtype.str,
        None if accepted is None else tuple(accepted.shape),
        None if accepted is None else accepted.tobytes(),
        poni_digest, mask_digest, config_json,
        None if calibration is None else calibration.parallax,
        flat_shape=None if flat is None else tuple(int(v) for v in flat.shape),
        flat_bytes=None if flat is None else flat.tobytes(),
        flat_sha256=flat_digest,
        dark_shape=None if dark is None else tuple(int(v) for v in dark.shape),
        dark_bytes=None if dark is None else dark.tobytes(),
        dark_sha256=dark_digest,
        correction_warnings=flat_warnings + dark_warnings,
    )


def _scientific_mask_layout(path: Path) -> tuple[tuple[int, int], np.dtype]:
    if path.suffix.casefold() == ".npy":
        value = np.load(path, allow_pickle=False, mmap_mode="r")
        try:
            shape = tuple(value.shape)
            dtype = np.dtype(value.dtype)
        finally:
            mapping = getattr(value, "_mmap", None)
            if mapping is not None:
                mapping.close()
        frame_count = 1
    else:
        layout = read_detector_image_layout(path)
        shape, dtype, frame_count = (
            tuple(layout.shape), np.dtype(layout.dtype), layout.frame_count,
        )
    if (
        len(shape) != 2 or any(type(value) is not int or value <= 0 for value in shape)
        or frame_count != 1 or dtype.kind not in "biuf" or dtype.itemsize > 8
        or int(shape[0]) * int(shape[1]) * dtype.itemsize
        > _SCIENTIFIC_MASK_MAX_DECODED_BYTES
    ):
        raise ValueError(f"scientific mask schema is unsupported: {path}")
    return (int(shape[0]), int(shape[1])), dtype


def _scientific_mask_file_limit(
    calibration: object | None,
) -> int:
    shape = None
    config = getattr(calibration, "detector_config", None)
    if isinstance(config, dict) or hasattr(config, "get"):
        shape = config.get("max_shape")
        if shape is None:
            shape = config.get("shape")
    valid = (
        type(shape) in {tuple, list}
        and len(shape) == 2
        and all(type(value) is int and value > 0 for value in shape)
    )
    if valid:
        pixels = int(shape[0]) * int(shape[1])
        itemsize = 8
        if pixels * itemsize > _SCIENTIFIC_MASK_MAX_DECODED_BYTES:
            raise ValueError("calibration mask shape exceeds the decoded limit")
    else:
        pixels = _SCIENTIFIC_MASK_MAX_DECODED_BYTES
        itemsize = 1
    return max(
        _SCIENTIFIC_MASK_COMPATIBILITY_FLOOR_BYTES,
        itemsize * pixels + _SCIENTIFIC_MASK_HEADER_ALLOWANCE_BYTES,
    )


def _asset_state(path: Path) -> tuple[int, ...]:
    state = path.stat()
    return (
        state.st_size, state.st_mtime_ns, identity_ctime_ns(state.st_ctime_ns),
        state.st_dev, state.st_ino,
    )


def _asset_descriptor_state(stream: object) -> tuple[int, ...]:
    # Compared against the pathname view above: ctime on the win32 seam.
    state = os.fstat(stream.fileno())
    return (
        state.st_size, state.st_mtime_ns, identity_ctime_ns(state.st_ctime_ns),
        state.st_dev, state.st_ino,
    )


def _stream_asset(
    path: Path, *, target: Path | None = None, max_bytes: int,
    cancelled: Callable[[], bool] = _not_cancelled,
) -> tuple[tuple[int, ...], str]:
    if cancelled():
        raise RuntimeError("admission cancelled")
    digest = hashlib.sha256()
    total = 0
    output = None if target is None else target.open("xb")
    try:
        try:
            stream = path.open("rb")
        except FileNotFoundError:
            raise
        with stream:
            opened = _asset_descriptor_state(stream)
            try:
                before = _asset_state(path)
            except FileNotFoundError as exc:
                raise ValueError(
                    f"scientific asset changed while admitted: {path}"
                ) from exc
            if before != opened:
                raise ValueError(
                    f"scientific asset changed while admitted: {path}"
                )
            while True:
                if cancelled():
                    raise RuntimeError("admission cancelled")
                payload = stream.read(_SCIENTIFIC_ASSET_STREAM_CHUNK_BYTES)
                if not payload:
                    break
                total += len(payload)
                if total > max_bytes:
                    raise ValueError(
                        f"scientific asset exceeds {max_bytes} bytes: {path}"
                    )
                digest.update(payload)
                if output is not None:
                    output.write(payload)
            finished = _asset_descriptor_state(stream)
    finally:
        if output is not None:
            output.close()
    try:
        if cancelled():
            raise RuntimeError("admission cancelled")
        after = _asset_state(path)
    except FileNotFoundError as exc:
        raise ValueError(
            f"scientific asset changed while admitted: {path}"
        ) from exc
    if after != before or finished != opened:
        raise ValueError(f"scientific asset changed while admitted: {path}")
    return before, digest.hexdigest()


def _load_stable_mask_asset(
    path_text: object, calibration: object | None,
    *,
    cancelled: Callable[[], bool] = _not_cancelled,
) -> tuple[object | None, str | None]:
    if type(path_text) is not str or not path_text:
        return None, None
    path = Path(path_text)
    if cancelled():
        raise RuntimeError("admission cancelled")
    try:
        limit = _scientific_mask_file_limit(calibration)
    except FileNotFoundError:
        return None, None
    with tempfile.TemporaryDirectory(prefix="xdart-mask-preflight-") as root:
        snapshot = Path(root) / ("mask" + path.suffix)
        try:
            before_state, before_digest = _stream_asset(
                path, target=snapshot, max_bytes=limit,
                cancelled=cancelled,
            )
        except FileNotFoundError:
            return None, None
        if cancelled():
            raise RuntimeError("admission cancelled")
        shape, dtype = _scientific_mask_layout(snapshot)
        snapshot_limit = max(
            _SCIENTIFIC_MASK_COMPATIBILITY_FLOOR_BYTES,
            int(shape[0]) * int(shape[1]) * int(dtype.itemsize)
            + _SCIENTIFIC_MASK_HEADER_ALLOWANCE_BYTES,
        )
        if snapshot.stat().st_size > snapshot_limit:
            raise ValueError(f"scientific mask schema is unsupported: {path}")
        if cancelled():
            raise RuntimeError("admission cancelled")
        value = load_mask(snapshot)
        if cancelled():
            raise RuntimeError("admission cancelled")
        try:
            after_state, after_digest = _stream_asset(
                path, max_bytes=limit, cancelled=cancelled,
            )
        except FileNotFoundError as exc:
            raise ValueError(
                f"scientific asset changed while admitted: {path}"
            ) from exc
    if after_state != before_state or after_digest != before_digest:
        raise ValueError(f"scientific asset changed while admitted: {path}")
    if value.dtype != np.dtype(bool) or tuple(value.shape) != shape:
        raise ValueError(f"scientific mask schema is unsupported: {path}")
    return value, before_digest


def _detector_facts(calibration: object | None) -> tuple[tuple[int, int] | None, np.ndarray | None] | None:
    """The calibrated detector's shape and own mask (gaps), or None when unknown."""
    if calibration is None:
        return None
    try:
        from xrd_tools.integrate.calibration import detector_calibration_to_integrator
        detector = detector_calibration_to_integrator(calibration).detector
        shape = None if detector.shape is None else tuple(int(v) for v in detector.shape)
        mask = detector.mask
    except Exception:  # noqa: BLE001 - an unknown detector only skips the checks
        _LOG.debug("detector facts unavailable for the correction checks", exc_info=True)
        return None
    return shape, (None if mask is None else np.asarray(mask, dtype=bool))


def _read_correction_image(path: Path, name: str) -> np.ndarray:
    """Decode one dark or flat-field image, refusing anything but one 2-D image."""
    try:
        _scientific_mask_layout(path)
    except ValueError as exc:
        raise ValueError(f"the {name} must be one 2-D image: {path}") from exc
    return (np.load(path, allow_pickle=False) if path.suffix.casefold() == ".npy"
            else read_image(path))


def inspect_detector_correction(
    poni_file: str, path_text: str, role: str,
    min_fraction: float = DEFAULT_MIN_FLAT_FRACTION,
    cache: dict | None = None,
) -> DetectorImageCheck:
    """Check a dark or flat-field file before a run: for the Advanced dialog.

    Uses the selected PONI's detector for the shape and gap checks when one is set.
    ``cache`` (owned by the caller, e.g. for one dialog session) keeps the last
    decoded image per role, so a cutoff change does not re-read the file.
    """
    name = "flat field" if role == "flat" else "dark"
    path = Path(path_text)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return DetectorImageCheck(role, (), errors=(f"the {name} file does not exist",))
    except OSError as exc:
        return DetectorImageCheck(role, (), errors=(str(exc),))
    key = (str(path), stat.st_mtime_ns, stat.st_size, str(poni_file))
    loaded = None if cache is None else cache.get(role)
    if loaded is None or loaded[0] != key:
        try:
            raw = _read_correction_image(path, name)
        except FileNotFoundError:
            return DetectorImageCheck(role, (), errors=(f"the {name} file does not exist",))
        except (OSError, ValueError) as exc:
            return DetectorImageCheck(role, (), errors=(str(exc),))
        detector = None
        if poni_file:
            try:
                detector = _detector_facts(load_poni(Path(poni_file)))
            except Exception:  # noqa: BLE001 - an unreadable PONI only skips the checks
                detector = None
        loaded = (key, raw, detector)
        if cache is not None:
            cache[role] = loaded
    _key, raw, detector = loaded
    shape, mask = (None, None) if detector is None else detector
    return check_detector_image(raw, role, detector_shape=shape, detector_mask=mask,
                                min_fraction=min_fraction)


def _load_stable_correction_asset(
    path_text: object,
    role: str,
    *,
    cancelled: Callable[[], bool] = _not_cancelled,
    min_fraction: float = DEFAULT_MIN_FLAT_FRACTION,
    detector: tuple[tuple[int, int] | None, np.ndarray | None] | None = None,
) -> tuple[np.ndarray | None, str | None, tuple[str, ...]]:
    """Admit a flat field or dark: one bounded 2-D image, prepared, stable while read.

    A missing file is refused, not skipped: dropping a requested correction
    would change the science silently. So is an image that does not match the
    calibrated detector; warnings (see ``check_detector_image``) are returned.
    """
    if type(path_text) is not str or not path_text:
        return None, None, ()
    from xrd_tools.corrections import prepare_dark, prepare_flatfield
    name = "flat field" if role == "flat" else "dark"
    path = Path(path_text)
    if cancelled():
        raise RuntimeError("admission cancelled")
    with tempfile.TemporaryDirectory(prefix=f"xdart-{role}-preflight-") as root:
        snapshot = Path(root) / (role + path.suffix)
        try:
            before_state, before_digest = _stream_asset(
                path, target=snapshot, max_bytes=_SCIENTIFIC_MASK_MAX_DECODED_BYTES,
                cancelled=cancelled,
            )
        except FileNotFoundError as exc:
            raise ValueError(f"{name} file is unavailable: {path}") from exc
        if cancelled():
            raise RuntimeError("admission cancelled")
        try:
            raw = _read_correction_image(snapshot, name)
        except ValueError as exc:
            raise ValueError(f"the {name} must be one 2-D image: {path}") from exc
        try:
            prepared = (prepare_flatfield(raw, min_fraction) if role == "flat"
                        else prepare_dark(raw))
        except ValueError as exc:
            raise ValueError(f"the {name} is unusable: {exc}: {path}") from exc
        shape, mask = (None, None) if detector is None else detector
        # One prepared copy serves the check and the admitted value.
        check = check_detector_image(raw, role, detector_shape=shape, detector_mask=mask,
                                     min_fraction=min_fraction, prepared=prepared)
        if check.errors:
            raise ValueError(f"{check.errors[0]}: {path}")
        value = prepared.astype(np.float32)
        del prepared, raw
        try:
            after_state, after_digest = _stream_asset(
                path, max_bytes=_SCIENTIFIC_MASK_MAX_DECODED_BYTES, cancelled=cancelled,
            )
        except FileNotFoundError as exc:
            raise ValueError(f"scientific asset changed while admitted: {path}") from exc
    if after_state != before_state or after_digest != before_digest:
        raise ValueError(f"scientific asset changed while admitted: {path}")
    return np.ascontiguousarray(value), before_digest, check.warnings


def _load_stable_asset(
    path_text: object, loader: Callable[[Path, bytes], object],
    *, max_bytes: int | None = None,
) -> tuple[object | None, str | None]:
    if type(path_text) is not str or not path_text:
        return None, None
    path = Path(path_text)
    def observation() -> tuple[tuple[int, ...], bytes]:
        try:
            stream = path.open("rb")
        except FileNotFoundError:
            raise
        with stream:
            opened = _asset_descriptor_state(stream)
            try:
                state = _asset_state(path)
            except FileNotFoundError as exc:
                raise ValueError(
                    f"scientific asset changed while admitted: {path}"
                ) from exc
            if state != opened:
                raise ValueError(
                    f"scientific asset changed while admitted: {path}"
                )
            payload = stream.read() if max_bytes is None else stream.read(max_bytes + 1)
            finished = _asset_descriptor_state(stream)
        if max_bytes is not None and len(payload) > max_bytes:
            raise ValueError(f"scientific asset exceeds {max_bytes} bytes: {path}")
        try:
            after = _asset_state(path)
        except FileNotFoundError as exc:
            raise ValueError(
                f"scientific asset changed while admitted: {path}"
            ) from exc
        if after != state or finished != opened:
            raise ValueError(f"scientific asset changed while admitted: {path}")
        return state, payload
    try:
        before_state, before = observation()
    except FileNotFoundError:
        return None, None
    value = loader(path, before)
    try:
        after_state, after = observation()
    except FileNotFoundError as exc:
        raise ValueError(f"scientific asset changed while admitted: {path}") from exc
    if after_state != before_state or after != before:
        raise ValueError(f"scientific asset changed while admitted: {path}")
    return value, hashlib.sha256(before).hexdigest()

