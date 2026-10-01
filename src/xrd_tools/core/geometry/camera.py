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
detector orientation (1–4) is not an image transform: with the detector untilted and the beam
pixel fixed it moves only the PONI's origin, but once the detector is tilted (rot1/rot2/rot3, as
every arm pose is) it changes 2θ and χ — pyFAI reflects the pixel coordinates before rotating and
the transverse output after, so the effective rotation is F·R·F (Codex R18-F1, pinned by a test).

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


#: The camera of the image Stitch calibrates and integrates in, its xrayutilities asset's: rows run down
#: (x-), columns to the left looking downstream (z+). A raw image from another camera is turned into it
#: first (:func:`stitch_orientation`); this one needs nothing (an image rotation of 0, as before). Scan 14
#: settled it (2026-09-30): 17-2's rows run down and its columns the way +nu swings the detector, to the
#: right, so its camera is ('x-', 'z-'), and with the maintainer's senses (nu x-, del z-) the arm fits only
#: with the columns mirrored, as this frame says. It is 180° from ``_PYFAI_TRANSFORM``'s ('x+', 'z-'): Stitch's
#: pyFAI χ is 180° from its xrayutilities χ (a pre-existing gap; 2θ and |q| agree).
STITCH_CAMERA = DetectorCamera('x-', 'z+')


def _in_stitch_frame(array: np.ndarray, camera: DetectorCamera) -> np.ndarray:
    """A raw ``camera`` array with its rows along x- and its columns along z+ (a view)."""
    rows, columns = camera.rows, camera.columns
    if rows[0] == 'z':  # its rows run along z: transposed, they run along x
        array, rows, columns = array.T, columns, rows
    if rows == 'x+':
        array = array[::-1, :]
    if columns == 'z-':
        array = array[:, ::-1]
    return array


def _effect(transform) -> tuple[tuple[int, ...], bytes]:
    """What a transform does to a small non-square array (its shape and values)."""
    out = np.ascontiguousarray(transform(np.arange(6).reshape(2, 3)))
    return out.shape, out.tobytes()


#: Every image transform Stitch applies: a counterclockwise turn, then a horizontal flip.
_STITCH_TRANSFORMS = tuple(ImageOrientation(rotation=rotation, flip_horizontal=flip)
                           for flip in (False, True) for rotation in (0, 90, 180, 270))


def stitch_orientation(camera: DetectorCamera) -> ImageOrientation:
    """The image transform (a turn, then a horizontal flip) bringing a raw image from ``camera`` into
    Stitch's frame (``STITCH_CAMERA``)."""
    effect = _effect(lambda array: _in_stitch_frame(array, camera))
    return next(orientation for orientation in _STITCH_TRANSFORMS if _effect(orientation.apply) == effect)


def stitch_camera(orientation: ImageOrientation) -> DetectorCamera:
    """The camera whose raw images Stitch turns by ``orientation``: an image rotation saved before the
    camera existed is ``stitch_camera(ImageOrientation(rotation=R))`` (R = 0: ``STITCH_CAMERA``)."""
    effect = _effect(orientation.apply)
    for rows in CAMERA_AXES:
        for columns in CAMERA_AXES:
            if rows[0] != columns[0] and _effect(stitch_orientation(DetectorCamera(rows, columns)).apply) == effect:
                return DetectorCamera(rows, columns)
    raise ValueError('not an image transform')  # a transpose alone, say: no camera is one
