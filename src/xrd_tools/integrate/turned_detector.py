"""A detector in the calibration window's frame (AUD-9 stage 2b item 8; the maintainer: "Yes for calibration window
fix").

The calibration window turns and mirrors every stored image into its frame (``stitch_orientation`` of the camera).
For a camera that keeps the image's axes (no turn, a half turn, or either mirrored) the detector there is the model
itself, as it always was. For a quarter turn the window's images are the stored ones transposed, so its detector is
the model turned with them: pyFAI's generic ``Detector`` of the transposed shape with the two pixel sizes swapped,
whose own mask (the module gaps) is the model's mask turned by the same ``ImageOrientation.apply`` as the images.

pyFAI can load that generic detector and integrate with it, but its configuration holds no mask, so every record of
a quarter-turned detector also names the model, and the readers rebuild the gaps from it:

- the pyFAI goniometer record carries ``TURNED_DETECTOR_KEY`` (the model and the turn), which
  ``DetectorCalibration.turned_from`` holds and every integrator built from it takes as its detector's mask
  (``detector_calibration_to_integrator``);
- the xrayutilities partner names the model as its detector (its ``shape``, pitches and mask digest the turned
  frame's, its turn its own ``image_rotation_quarter_turns``/``image_flip``), and its readers rebuild it through
  ``partner_detector`` (the mask) and ``frame_detector_record`` (the solid-angle arm's pyFAI detector).

A record of a model in its own shape is read exactly as before, so a pair the old window saved for a quarter-turned
camera (the model's own shape with the beam pixel already turned) stays as it was: refused where it was refused.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from xrd_tools.core.geometry.diffractometer import ImageOrientation
from xrd_tools.core.geometry.stacked_arm import TURNED_DETECTOR_KEY, checked_turned_detector

__all__ = [
    "TURNED_DETECTOR_KEY", "frame_detector_record", "is_turned_window_record", "partner_detector",
    "turned_detector_record", "turned_frame_shape", "turned_model_mask", "window_detector", "window_detector_record",
]


def turned_detector_record(model: str, orientation: ImageOrientation) -> dict:
    """``TURNED_DETECTOR_KEY``'s value for ``model`` under a quarter-turning ``orientation``."""
    return checked_turned_detector({"detector": str(model), "image_rotation": int(orientation.rotation),
                                    "image_flip": bool(orientation.flip_horizontal)})


def turned_model_mask(model: str, orientation: ImageOrientation) -> np.ndarray | None:
    """The model's own mask (its module gaps) turned as its images are, read-only; None for a model without
    gaps."""
    return _turned_model_mask(str(model), int(orientation.rotation), bool(orientation.flip_horizontal))


@lru_cache(maxsize=16)
def _turned_model_mask(model: str, rotation: int, flip: bool) -> np.ndarray | None:
    from pyFAI.detectors import detector_factory

    mask = detector_factory(model).calc_mask()
    if mask is None or not np.any(mask):
        return None
    turned = np.ascontiguousarray(
        ImageOrientation(rotation=rotation, flip_horizontal=flip).apply(np.asarray(mask, dtype=bool)), dtype=bool)
    turned.setflags(write=False)
    return turned


@lru_cache(maxsize=16)
def turned_frame_shape(model: str) -> tuple[int, int]:
    """The model's shape turned by a quarter: its (columns, rows)."""
    from pyFAI.detectors import detector_factory

    rows, columns = (int(value) for value in detector_factory(str(model)).shape)
    return columns, rows


def _require_uniform_flat(detector, model):
    """The generic quarter-turn record can retain only a uniform flat model's pixel geometry."""
    if not detector.uniform_pixel or not detector.IS_FLAT:
        raise ValueError(f"A quarter-turn camera cannot represent {model}'s nonuniform or curved pixels; "
                         "choose a camera that keeps its axes.")


def window_detector_record(model: str, orientation: ImageOrientation, *, pyfai_orientation: int = 3):
    """What a PONI or goniometer record writes for the window's detector: ``(name, config, turned)``, ``turned``
    being ``TURNED_DETECTOR_KEY``'s value (None when the model is not turned). A camera that keeps the image's axes:
    the model itself, its config ``{}`` (or its pyFAI orientation). A quarter turn: pyFAI's generic detector of the
    transposed shape, pixel sizes swapped, in the same pyFAI orientation."""
    from pyFAI.detectors import detector_factory

    config = {} if int(pyfai_orientation) == 3 else {"orientation": int(pyfai_orientation)}
    if not orientation.swaps_axes:
        return str(model), config, None
    native = detector_factory(model, config=config) if config else detector_factory(model)
    _require_uniform_flat(native, model)
    rows, columns = (int(value) for value in native.shape)
    turned_config = {"pixel1": float(native.pixel2), "pixel2": float(native.pixel1), "max_shape": [columns, rows],
                     "orientation": int(pyfai_orientation)}
    return "Detector", turned_config, turned_detector_record(model, orientation)


def window_detector(model: str, orientation: ImageOrientation, *, pyfai_orientation: int = 3):
    """The detector the window's frame sees, and its record: ``(detector, name, config, turned)``
    (``window_detector_record``'s). A quarter turn's detector holds the model's mask turned."""
    from pyFAI.detectors import detector_factory

    name, config, turned = window_detector_record(model, orientation, pyfai_orientation=pyfai_orientation)
    if turned is None:
        return (detector_factory(model, config=config) if config else detector_factory(model)), name, config, None
    detector = detector_factory(name, config=dict(config, max_shape=tuple(config["max_shape"])))
    mask = turned_model_mask(model, orientation)
    if mask is not None:
        detector.mask = np.array(mask, dtype=np.int8)
    return detector, name, config, turned


def is_turned_window_record(name, config, turned) -> bool:
    """Whether a record's detector ``(name, config)`` is exactly the window's for its ``TURNED_DETECTOR_KEY``
    value ``turned``: pyFAI's generic detector of the model's turned shape, its pixel sizes swapped, in the config's
    own pyFAI orientation (what ``window_detector_record`` writes); False for anything else."""
    try:
        turned = checked_turned_detector(turned)
        if turned is None or not hasattr(config, "keys"):
            return False
        orientation = ImageOrientation(rotation=turned["image_rotation"], flip_horizontal=turned["image_flip"])
        expected = window_detector_record(turned["detector"], orientation,
                                          pyfai_orientation=int(config.get("orientation", 3)))
    except (TypeError, ValueError, KeyError, RuntimeError):
        return False
    return (name, dict(config), turned) == expected


def frame_detector_record(name: str, config, shape) -> tuple[str, dict]:
    """The pyFAI detector ``(name, config)`` in a record's own frame of ``shape``: the record's own, unless it names
    a model in the raw frame of a quarter-turned record (the model's shape is ``shape`` transposed, and the config
    sets no ``max_shape``): then pyFAI's generic detector of ``shape`` with the record's pitches and orientation."""
    from pyFAI.detectors import detector_factory

    config = dict(config or {})
    shape = tuple(int(value) for value in shape)
    if name == "Detector" or "max_shape" in config:
        return name, config
    model = detector_factory(name, config=config) if config else detector_factory(name)
    if tuple(int(value) for value in model.shape) != shape and tuple(int(value) for value in model.shape)[::-1] == shape:
        _require_uniform_flat(model, name)
        return "Detector", {"pixel1": float(config.get("pixel1", model.pixel1)),
                            "pixel2": float(config.get("pixel2", model.pixel2)), "max_shape": list(shape),
                            "orientation": int(config.get("orientation", 3))}
    return name, config


def partner_detector(detector_asset, acquisition):
    """An xrayutilities partner's detector rebuilt in the partner's own frame, and its own mask (None: no gaps):
    the named model itself when it has the partner's shape (every partner the old window wrote), or, for a partner
    whose quarter turn names the model in the raw frame, pyFAI's generic detector of the partner's shape and pitches
    with the model's mask turned by the partner's own turn. A square model reads as itself: for every square model
    pyFAI lists, the two readings agree (square pixels, gaps the same turned), and the readers check the mask's count
    and digest against the partner's either way."""
    from pyFAI.detectors import detector_factory

    config = dict(detector_asset["detector_config"])
    if "max_shape" in config:
        config["max_shape"] = tuple(config["max_shape"])
    detector = detector_factory(detector_asset["detector_factory_name"], config)
    shape = tuple(int(value) for value in detector_asset["shape"])
    orientation = ImageOrientation(rotation=90 * int(acquisition["image_rotation_quarter_turns"]),
                                   flip_horizontal=bool(acquisition.get("image_flip", False)))
    native = tuple(int(value) for value in detector.shape)
    if native != shape and orientation.swaps_axes and "max_shape" not in config and native[::-1] == shape:
        name, turned_config = frame_detector_record(detector_asset["detector_factory_name"], config, shape)
        turned = detector_factory(name, config=dict(turned_config, max_shape=shape))
        return turned, turned_model_mask(detector_asset["detector_factory_name"], orientation)
    return detector, detector.calc_mask()
