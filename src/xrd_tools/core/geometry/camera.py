"""The detector camera: which lab directions the raw image's rows and columns run along.

The lab frame is psic's (You 1999): x up, y along the beam, z = x × y, to the left looking
downstream. With every circle at zero the detector faces the beam, and the raw array's first
index (rows) and second index (columns) each run along ±x or ±z: a camera is that pair of signed
axes, as xrayutilities' ``Ang2Q.init_area`` takes them (``detectorDir1``, ``detectorDir2``). Its 8
values are every flip, 90° turn and transpose of the image. The diffractometer stays the same when
the detector is mounted another way; only the camera changes.

pyFAI's frame (its ``Geometry`` docstring: dim1 up, dim2 horizontal, dim3 along the beam) is
t1 = x, t2 = -z, so the camera ``('x+', 'z-')`` is its own. Each camera has one image transform
bringing the raw array into that frame, found and pinned against xrayutilities and pyFAI on both
2θ and χ at every pixel (tests/core/test_detector_camera.py), for pyFAI orientation 3. pyFAI's
detector orientation (1–4) reindexes pixels. In pyFAI 2026.9, matching a new orientation with
mirrored indices preserves the physical ray and every derived coordinate, including χ. Changing
orientation for unchanged image indices instead changes their physical interpretation. Converting
the PONI's origin still reverses the corresponding arm rotations; camera mounting and the
first-pixel corner remain separate settings.

The legacy ``DiffractometerConfig.init_area_detrot`` / ``init_area_tiltazimuth`` fields hold these
two directions (not xrayutilities' detrot / tiltazimuth angles); the names stay for saved
experiment configurations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xrd_tools.core.geometry.diffractometer import ImageOrientation

#: The directions a camera axis may take while the detector faces the beam.
CAMERA_AXES = ('x+', 'x-', 'z+', 'z-')

#: (rows, columns) → the image transform into pyFAI's frame: (rotation °, flip horizontal).
_PYFAI_TRANSFORM = {
    ('x+', 'z-'): (0, False), ('x+', 'z+'): (0, True),
    ('x-', 'z+'): (180, False), ('x-', 'z-'): (180, True),
    ('z-', 'x-'): (90, False), ('z+', 'x-'): (90, True),
    ('z+', 'x+'): (270, False), ('z-', 'x+'): (270, True),
}


def _signature(orientation: ImageOrientation) -> bytes:
    """What a transform does to a small non-square array, so equal transforms compare equal."""
    return orientation.apply(np.arange(6).reshape(2, 3)).tobytes()


@dataclass(frozen=True)
class DetectorCamera:
    """The signed lab axes the raw image's rows and columns run along, at zero angles."""

    rows: str
    columns: str

    def __post_init__(self):
        if self.rows not in CAMERA_AXES or self.columns not in CAMERA_AXES or self.rows[0] == self.columns[0]:
            raise ValueError(f'a camera is one x and one z direction (x±, z±), not '
                             f'({self.rows!r}, {self.columns!r})')

    def to_xu(self) -> tuple[str, str]:
        """``Ang2Q.init_area``'s detectorDir1, detectorDir2."""
        return self.rows, self.columns

    def raw_vector(self, lab_axis: str) -> tuple[int, int]:
        """Unit lab direction as (column, row) in the stored image, at zero pose."""
        if lab_axis not in CAMERA_AXES:
            raise ValueError('a detector-plane direction is x± or z±')
        sign = 1 if lab_axis[1] == '+' else -1
        return tuple(sign * (1 if axis[1] == '+' else -1) if axis[0] == lab_axis[0] else 0
                     for axis in (self.columns, self.rows))

    def image_orientation(self) -> ImageOrientation:
        """The transform that brings the raw array into pyFAI's frame (its orientation 3)."""
        rotation, flip = _PYFAI_TRANSFORM[(self.rows, self.columns)]
        return ImageOrientation(rotation=rotation, flip_horizontal=flip)

    @classmethod
    def from_image_orientation(cls, orientation: ImageOrientation) -> DetectorCamera:
        """The camera a pyFAI path assumes when it applies ``orientation`` to the raw array."""
        signature = _signature(orientation)
        for (rows, columns), (rotation, flip) in _PYFAI_TRANSFORM.items():
            if _signature(ImageOrientation(rotation=rotation, flip_horizontal=flip)) == signature:
                return cls(rows, columns)
        raise ValueError('not an image transform')  # the eight above are every flip and turn


#: New Stitch/RSM calibrations share pyFAI's native frame. Frozen XU assets retain their recorded frame.
STITCH_CAMERA = DetectorCamera('x+', 'z-')

# Mounting descriptions are headless values; editors and figures share this table.
UPRIGHT_CAMERA = DetectorCamera('x-', 'z+')
_ROTATIONS = {0: ('x-', 'z+'), 90: ('z+', 'x+'), 180: ('x+', 'z-'), 270: ('z-', 'x-')}


def mounting_camera(rotation: int, flip: bool, flip_vertical: bool = False) -> DetectorCamera:
    """Clockwise mounting looking downstream, then stored-column/row reversals."""
    rows, columns = _ROTATIONS[rotation]
    if flip:
        columns = columns[0] + ('-' if columns[1] == '+' else '+')
    if flip_vertical:
        rows = rows[0] + ('-' if rows[1] == '+' else '+')
    return DetectorCamera(rows, columns)


def mounting_controls(camera: DetectorCamera) -> tuple[int, bool]:
    for rotation, (rows, columns) in _ROTATIONS.items():
        if camera.rows == rows:
            return rotation, camera.columns != columns
    raise ValueError('not a detector camera')


def mounting_description(camera: DetectorCamera) -> str:
    rotation, flip = mounting_controls(camera)
    return ('upright' if rotation == 0 else f'{rotation}° CW') + (
        ', horizontal flip' if flip else '')


def stitch_orientation(camera: DetectorCamera, *, frame: DetectorCamera = STITCH_CAMERA) -> ImageOrientation:
    """Bring raw images into the native calibration frame, or an asset's explicitly recorded frame."""
    native = camera.image_orientation()
    if frame == STITCH_CAMERA:
        return native
    if frame == DetectorCamera('x-', 'z+'):
        return ImageOrientation(rotation=(native.rotation + 180) % 360,
                                flip_horizontal=native.flip_horizontal)
    raise ValueError('unsupported calibration image frame')


def stitch_camera(orientation: ImageOrientation, *, frame: DetectorCamera = STITCH_CAMERA) -> DetectorCamera:
    """The raw camera whose transformed images have the calibration's recorded frame."""
    camera = DetectorCamera.from_image_orientation(orientation)
    if frame == STITCH_CAMERA:
        return camera
    if frame == DetectorCamera('x-', 'z+'):
        return DetectorCamera(*(axis[0] + ('-' if axis[1] == '+' else '+') for axis in camera.to_xu()))
    raise ValueError('unsupported calibration image frame')
