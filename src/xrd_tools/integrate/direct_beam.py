"""Streaming direct-beam distance calibration for uniform, flat detectors.

See ``docs/core/direct_beam.md`` for the finite geometry and coordinate contract.
No integration, plotting, GUI, file writes or calibration-asset construction.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import copy
import logging
import os
from pathlib import Path
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from xrd_tools.core.geometry.diffractometer import ImageOrientation
from xrd_tools.core.scan import FrameSource, ScanFrame
from xrd_tools.integrate.calibration import get_detector
from xrd_tools.io.image import read_image


@dataclass(frozen=True)
class DirectBeamResult:
    """One entry per original row; rejected centres/residuals are NaN.

    ``distance_std_m`` is an unweighted regression standard error conditional on
    the chosen model, known angles and pitch; it excludes systematic error.
    ``slope_deg_per_pixel`` is the signed local slope at angle zero for tangent,
    or the fitted notebook slope for linear. Arrays use the oriented frame.
    """

    distance_m: float
    distance_std_m: float
    angles_deg: np.ndarray
    frame_ids: tuple
    point_ids: tuple
    centres_px: np.ndarray
    centre_std_px: np.ndarray
    used: np.ndarray
    rejection_reasons: tuple[str | None, ...]
    predictions_px: np.ndarray
    residuals_px: np.ndarray
    model: str
    angle_zero_deg: float | None
    pixel_pitch_m: float
    image_shape: tuple[int, int]
    slope_deg_per_pixel: float
    intercept_px: float
    #: The (x, y) beam pixel the strips were placed by: as given, or found (``find_beam_spot``).
    beam_pixel: tuple[float, float] | None = None
    #: The direction the spot moves ('horizontal': along x, 'vertical': along y).
    motion: str | None = None

    @property
    def distance_mm(self) -> float:
        return self.distance_m * 1000.0

    @property
    def zero_beam_pixel(self) -> tuple[float, float] | None:
        """The (x, y) pixel the direct beam hits at the zero angle: the fitted centre along the
        motion (at ``angle_zero_deg``; the linear model's intercept is at angle 0) and the strips'
        position across it. ``beam_pixel`` only placed the strips; found on the frame nearest the
        zero, it is off the zero-angle beam along the motion whenever that frame is not at it."""
        if self.beam_pixel is None or self.motion not in ('horizontal', 'vertical'):
            return None
        centre = (self.intercept_px if self.model == 'tangent'
                  else self.intercept_px + (self.angle_zero_deg or 0.) / self.slope_deg_per_pixel)
        x, y = self.beam_pixel
        return (float(centre), float(y)) if self.motion == 'horizontal' else (float(x), float(centre))

    @property
    def motion_sign(self) -> int:
        return int(np.sign(self.slope_deg_per_pixel))


@dataclass(frozen=True)
class DirectBeamScanInput:
    """Source inputs valid inside :func:`open_direct_beam_scan`'s context.

    Images are lazy; the caller still supplies detector, beam pixel and fit
    settings. ``frame_ids=None`` lets the fitter retain explicit image references.
    """

    angles_deg: np.ndarray
    images: Iterable
    frame_ids: tuple | None
    point_ids: tuple


def _uniform_flat(detector):
    # Check BEFORE deepcopy: pyFAI's generic clone omits instance geometry flags.
    if (not detector.IS_FLAT or not detector.uniform_pixel
            or detector.spline is not None):
        raise ValueError('direct-beam calibration requires uniform, flat, undistorted pixels')


def _effective_detector(detector, shape):
    det = copy.deepcopy(detector)
    if det.shape is None:
        det.shape = det.max_shape
    if det.shape is None:
        raise ValueError('detector shape is required (generic: max_shape)')
    if tuple(det.shape) != shape:
        # pyFAI's guess_binning clears cached masks. Inspect the cache without
        # materializing a native mask; False/None mean no existing mask array.
        existing_mask = det._mask
        if isinstance(existing_mask, np.ndarray) and existing_mask.shape != shape:
            raise ValueError('detector mask shape differs from image shape; '
                             'configure binning before assigning the mask')
        if det.max_shape is None or any(m % n for m,n in zip(det.max_shape,shape)):
            raise ValueError(f'image shape {shape} is not an integer binning of {det.max_shape}')
        if det.force_pixel:
            if not det.guess_binning(shape):
                raise ValueError(f'unsupported detector binning for image shape {shape}')
        else:
            det.binning = tuple(m // n for m,n in zip(det.max_shape,shape))
        if tuple(det.shape) != shape:
            raise ValueError('effective detector shape differs from image shape')
        if isinstance(existing_mask, np.ndarray):
            det.mask = existing_mask
    _uniform_flat(det)
    pitches = np.array([det.pixel1,det.pixel2],dtype=float)
    if not np.all(np.isfinite(pitches) & (pitches > 0)):
        raise ValueError('detector pixel pitches must be finite positive metres')
    return det, pitches


def _gaussian_line(x, background, gradient, amplitude, centre, sigma):
    return background + gradient*x + amplitude*np.exp(-.5*((x-centre)/sigma)**2)


def _peak(image, invalid, beam, motion, half_width, roi, saturation):
    h,w = image.shape
    x0,y0,x1,y1 = (0,0,w,h) if roi is None else roi
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError('ROI must be nonempty, half-open and within the oriented image')
    if not (x0 <= beam[0] < x1 and y0 <= beam[1] < y1):
        raise ValueError('beam_pixel must be inside the oriented image/ROI')
    if motion == 'vertical':
        image, invalid = image.T, invalid.T
        x0,y0,x1,y1 = y0,x0,y1,x1
        beam = beam[::-1]
    row = int(np.floor(beam[1]+.5))
    lo,hi = max(y0,row-half_width), min(y1,row+half_width+1)
    strip = image[lo:hi,x0:x1]
    bad = invalid[lo:hi,x0:x1] | ~np.isfinite(strip) | (strip < 0)
    # Static invalid pixels are not measurements. Still reject unmasked saturation
    # rather than masking it and fitting the remaining wings as a valid beam.
    limit = saturation
    if np.issubdtype(image.dtype,np.integer):
        native_limit = np.iinfo(image.dtype).max
        limit = native_limit if limit is None else min(limit,native_limit)
    if limit is not None and np.any((strip >= limit) & ~bad):
        raise ValueError('saturated pixels in beam strip/ROI')
    valid = ~bad.any(axis=0)
    x = np.arange(x0,x1,dtype=float)
    profile = strip.sum(axis=0,dtype=float)  # new array, never a mutable image view
    if valid.sum() < 7:
        raise ValueError('too few valid profile pixels for Gaussian peak')
    xv,yv = x[valid],profile[valid]
    # A linear baseline from the ends keeps the Gaussian seed independent of slope.
    ends = np.r_[0:min(5,len(xv)//3), max(0,len(xv)-5):len(xv)]
    gradient,background = np.polyfit(xv[ends],yv[ends],1)
    signal = yv - (background+gradient*xv)
    amplitude = float(signal.max())
    if amplitude <= max(1e-10, 1e-8*float(np.max(np.abs(yv)))):
        raise ValueError('no positive beam peak above background')
    with warnings.catch_warnings():
        warnings.simplefilter('error',OptimizeWarning)
        fit,cov = curve_fit(_gaussian_line,xv,yv,
            p0=(background,gradient,amplitude,xv[np.argmax(signal)],2.),
            bounds=([-np.inf,-np.inf,0,x0-.5,.25],
                    [np.inf,np.inf,np.inf,x1-.5,(x1-x0)/2]),maxfev=5000)
    _,_,height,centre,sigma = fit
    noise = np.sqrt(np.mean((yv-_gaussian_line(xv,*fit))**2))
    if (not np.all(np.isfinite(cov)) or height <= 5*noise
            or sigma >= (x1-x0)/4 or not x0-1e-6 <= centre <= x1-1+1e-6):
        raise ValueError('poor or unconstrained Gaussian peak fit')
    core = np.abs(x-centre) <= 2*sigma
    if np.any(~valid & core):
        raise ValueError('masked/gap/invalid pixels intersect fitted beam core')
    error = float(np.sqrt(max(0.,cov[3,3])))
    if error > sigma:
        raise ValueError('beam centre uncertainty exceeds peak width')
    return float(centre),error


def find_beam_spot(image, invalid=None, *, box: int = 5) -> tuple[float, float]:
    """Where the direct beam hits an oriented frame, as (x, y) = (column, row).

    The brightest ``box`` × ``box`` sum (a lone hot pixel is diluted ``box``² times),
    refined as the background-subtracted intensity centroid of the 11 × 11 pixels around
    it (a narrower window biases it toward the brightest pixel). Invalid pixels (``invalid``, negative or nonfinite) count as zero. A frame with no
    spot clear of its background is refused.
    """
    from scipy.ndimage import uniform_filter
    values = np.asarray(image, dtype=float)
    bad = ~np.isfinite(values) | (values < 0)
    if invalid is not None:
        bad |= np.asarray(invalid, dtype=bool)
    if values.ndim != 2 or bad.all():
        raise ValueError('no valid pixels to find the direct beam on')
    clean = np.where(bad, 0.0, values)
    summed = uniform_filter(clean, size=box, mode='constant')
    row, col = np.unravel_index(int(np.argmax(summed)), summed.shape)
    level = float(np.median(summed[~bad]))
    spread = 1.4826 * float(np.median(np.abs(summed[~bad] - level)))
    if not summed[row, col] - level > max(10 * spread, .5 * abs(level), 1e-12):
        raise ValueError('no direct-beam spot stands clear of the background')
    rows, cols = slice(max(row - 5, 0), row + 6), slice(max(col - 5, 0), col + 6)
    weights = np.where(bad[rows, cols], 0., np.clip(clean[rows, cols] - float(np.median(clean[~bad])), 0., None))
    grid_rows, grid_cols = np.mgrid[rows, cols][:, :weights.shape[0], :weights.shape[1]]
    total = float(weights.sum())
    return float((weights * grid_cols).sum() / total), float((weights * grid_rows).sum() / total)


def _load_beam_image(item, options):
    if item is None:
        raise ValueError('missing frame')
    if isinstance(item,np.ndarray):
        return item
    if isinstance(item, ScanFrame):
        return item.load_image()
    opts = options.copy()
    path = item
    if isinstance(item,tuple):
        if len(item) not in (2,3):
            raise ValueError('frame reference must be (path,frame[,dataset])')
        path,opts['frame'] = item[:2]
        if len(item) == 3:
            opts['dataset_path'] = item[2]
    return read_image(path,preserve_dtype=True,exact_frame=True,**opts)


def _prepare_beam_image(image, det, orient, mask, state):
    if image.ndim != 2 or not all(image.shape):
        raise ValueError('image shape must be nonempty 2-D')
    if not state:
        effective,pitches = _effective_detector(det,image.shape)
        detector_mask = effective.mask
        if detector_mask is not None:
            if detector_mask.shape != image.shape:
                raise ValueError('detector mask shape differs from image shape')
            detector_mask = orient.apply(detector_mask).astype(bool)
        if orient.swaps_axes:
            pitches = pitches[::-1]
        state.update(raw_shape=image.shape, effective=effective, detector_mask=detector_mask,
                     pitches=pitches)
    if image.shape != state['raw_shape']:
        raise ValueError(f"image shape {image.shape} differs from {state['raw_shape']}")
    oriented = orient.apply(image)
    invalid = (np.zeros(oriented.shape,bool) if state['detector_mask'] is None
               else state['detector_mask'].copy())
    dummy, tolerance = state['effective'].get_dummies(image)
    if dummy is not None:
        invalid |= np.abs(oriented.astype(np.float32)-dummy) <= tolerance
    if mask is not None:
        if np.shape(mask) != oriented.shape:
            raise ValueError('mask shape must match the oriented image')
        invalid |= np.asarray(mask,dtype=bool)
    return oriented, invalid


def infer_direct_beam_motion(images: Sequence, *, detector, detector_config=None,
        orientation=None, mask=None, saturation=None, roi=None, reader_options=None,
        cancelled=None, return_details=False):
    """Measure the dominant displacement between the first and last usable frames.

    Items are lazy references, as in the fitter. Only endpoint candidates are decoded,
    one at a time. Beam extraction shares the fitter's detector mask, orientation,
    Gaussian peak checks and inclusive saturation rejection. A stationary or diagonal
    trajectory cannot select a strip axis safely and is refused.
    """
    orient = orientation or ImageOrientation()
    if detector_config is None:
        det = get_detector(detector)
    else:
        from pyFAI.detectors import detector_factory
        det = detector_factory(detector, config=dict(detector_config))
    _uniform_flat(det)
    state = {}

    def endpoint(indices, excluded=None):
        reason = 'no direct-beam spot'
        for index in indices:
            if index == excluded:
                continue
            if cancelled is not None and cancelled.is_set():
                raise ValueError('direct-beam detection cancelled')
            try:
                image, invalid = _prepare_beam_image(
                    _load_beam_image(images[index], dict(reader_options or {})), det, orient, mask, state)
                if roi is not None:
                    x0, y0, x1, y1 = roi
                    outside = np.ones(image.shape, dtype=bool)
                    outside[y0:y1, x0:x1] = False
                    invalid |= outside
                spot = find_beam_spot(image, invalid)
                x, _ = _peak(image, invalid, spot, 'horizontal', 2, roi, saturation)
                y, _ = _peak(image, invalid, spot, 'vertical', 2, roi, saturation)
                return index, np.array((x, y))
            except (ValueError, OSError, RuntimeError) as error:
                reason = str(error)
        raise ValueError('No usable direct-beam endpoint: ' + reason)

    first, start = endpoint(range(len(images)))
    last, stop = endpoint(range(len(images) - 1, -1, -1), first)
    delta = stop - start
    dx, dy = np.abs(delta)
    if os.environ.get('XDART_DEBUG_DIRECT_BEAM'):
        logging.getLogger(__name__).warning('direct_beam endpoints=%s,%s pixels=%s,%s '
            'displacement=%g,%g orientation=%s', first, last, start, stop, dx, dy, orient)
    if max(dx, dy) < 2 or max(dx, dy) < 3 * min(dx, dy):
        raise ValueError('Ambiguous or insignificant direct-beam endpoint motion')
    motion = 'horizontal' if dx > dy else 'vertical'
    return (motion, tuple(float(value) for value in delta), (first, last)) if return_details else motion


def calibrate_direct_beam(
    angles_deg: Sequence[float], images: Iterable, *, detector,
    beam_pixel: tuple[float, float] | None = None, detector_config: Mapping | None = None,
    motion: str = 'horizontal', strip_half_width: int = 2,
    model: str = 'tangent', angle_zero_deg: float | None = None,
    orientation: ImageOrientation | None = None, roi: tuple[int,int,int,int] | None = None,
    mask: np.ndarray | None = None, saturation: float | None = None,
    reader_options: Mapping | None = None, frame_ids: Sequence | None = None,
    point_ids: Sequence | None = None,
) -> DirectBeamResult:
    """Fit distance from explicit degree angles and ordered images/references.

    An item is a 2-D array, lazy ``ScanFrame``, path, ``(path, frame)`` or
    ``(path, frame, dataset)``;
    ``None`` denotes a missing frame. References use ``read_image`` with exact
    frame selection. ``reader_options`` accepts raw dtype/header/shape and HDF
    dataset/frame options, never image transforms or masking.

    ``detector`` is a registered pyFAI name or configured detector. A generic
    name ``'Detector'`` accepts ``detector_config`` with ``pixel1`` (row metres),
    ``pixel2`` (column metres), ``max_shape``. Integer full-panel binning is
    resolved against each scan's first readable image; crops are not binning.
    All frames must then have that shape. Detector objects/images are not mutated.

    ``orientation`` transforms raw arrays and detector masks once. The supplied
    ``beam_pixel``, ``mask`` and half-open ROI ``(x0,y0,x1,y1)`` ALREADY describe
    the oriented image. ``motion`` is also in that frame. Detector_config's
    pyFAI orientation does not apply another array transform.

    Tangent fits centre = c0 + b*tan(angle-angle_zero), requiring an explicit
    known zero; distance = abs(b)*pitch. Only c0 and b are free. The distance is
    perpendicular sample-to-plane separation for a sample-centred detector arm.
    ``model='linear'`` uses the notebook's inverse regression and local distance
    approximation, without needing a known zero. No tilt/wavelength/PONI is fit.
    The beam pixel selects the strip only, not the fitted intercept. Without one
    (``beam_pixel=None``) it is found (:func:`find_beam_spot`) on the frame nearest
    ``angle_zero_deg`` (the middle angle without a zero), a missing or unreadable
    frame passing to the next nearest; the result's ``beam_pixel`` says which.

    Negative/nonfinite pixels and masks exclude whole strip-profile bins; invalid
    beam cores and saturation reject a frame. Set ``saturation`` to the detector's
    acquisition limit (inclusive); integer dtype maximum is also rejected.
    Unknown clipping/count-rate effects cannot be inferred from intensity alone.
    At least three usable rows at distinct angles and significant motion are required.
    """
    angles = np.array(angles_deg,dtype=float,copy=True)
    if angles.ndim != 1 or not np.isfinite(angles).all():
        raise ValueError('angles_deg must be a finite one-dimensional array')
    if len(angles) < 3:
        raise ValueError('at least three angle/frame rows are required')
    if np.unique(angles).size < 3:
        raise ValueError('at least three distinct angles are required')
    if model not in {'tangent','linear'}:
        raise ValueError("model must be 'tangent' or 'linear'")
    if model == 'tangent':
        if angle_zero_deg is None or not np.isfinite(angle_zero_deg):
            raise ValueError('tangent model requires finite explicit angle_zero_deg')
        if np.any(np.abs(angles-angle_zero_deg) >= 89):
            raise ValueError('tangent angles must stay within 89 degrees of angle_zero_deg')
    if motion not in {'horizontal','vertical'}:
        raise ValueError("motion must be 'horizontal' or 'vertical'")
    if not isinstance(strip_half_width,(int,np.integer)) or strip_half_width < 0:
        raise ValueError('strip_half_width must be a nonnegative integer')
    beam = None if beam_pixel is None else np.asarray(beam_pixel,dtype=float)
    if beam is not None and (beam.shape != (2,) or not np.isfinite(beam).all()):
        raise ValueError('beam_pixel must be finite (x,y)')
    if roi is not None and (len(roi) != 4 or any(not isinstance(v,(int,np.integer)) for v in roi)):
        raise ValueError('ROI must contain four integer bounds')
    if saturation is not None and (not np.isfinite(saturation) or saturation <= 0):
        raise ValueError('saturation must be finite and positive')
    orient = ImageOrientation() if orientation is None else orientation
    if not isinstance(orient,ImageOrientation):
        raise TypeError('orientation must be an ImageOrientation')
    options = dict(reader_options or {})
    allowed = {'frame','detector_shape','raw_dtype','raw_header_skip','dataset_path'}
    if options.keys()-allowed:
        raise ValueError(f'unsupported reader_options: {sorted(options.keys()-allowed)}')
    if detector_config is None:
        det = get_detector(detector)
    else:
        if not isinstance(detector,str):
            raise ValueError('detector_config requires a registered detector name')
        from pyFAI.detectors import detector_factory
        det = detector_factory(detector,config=dict(detector_config))
    _uniform_flat(det)
    ids = None if frame_ids is None else tuple(frame_ids)
    points = tuple(range(len(angles))) if point_ids is None else tuple(point_ids)
    if (ids is not None and len(ids) != len(angles)) or len(points) != len(angles):
        raise ValueError('identifier and angle lengths differ')
    centres = np.full(len(angles),np.nan)
    errors = centres.copy()
    reasons, references = [],[]
    state = {}  # the first readable image's layout: raw shape, effective detector, its mask, the pitch

    def load(item):
        return _load_beam_image(item, options)

    def oriented_and_invalid(image):
        result = _prepare_beam_image(image, det, orient, mask, state)
        state['pitch'] = float(state['pitches'][1 if motion == 'horizontal' else 0])
        return result

    iterator = iter(images)  # streamed: no frame is kept past its row
    if beam is None:  # the zero frame first: the rows' items are gathered (lazy references stay undecoded)
        items = list(iterator)
        if len(items) != len(angles):
            raise ValueError('image and angle lengths differ')
        zero = np.median(angles) if angle_zero_deg is None else angle_zero_deg
        problems = []
        for index in np.argsort(np.abs(angles-zero), kind='stable'):
            try:
                beam = np.asarray(find_beam_spot(*oriented_and_invalid(load(items[index]))), dtype=float)
                break
            except (OSError,ValueError,IndexError,KeyError,RuntimeError) as exc:
                problems.append(f'row {int(index)}: {exc}')
        else:
            raise ValueError('no frame shows the direct beam; enter the beam pixel ('
                             + '; '.join(problems[:3]) + ')')

        def released():
            for index in range(len(items)):
                item, items[index] = items[index], None
                yield item
        iterator = released()
    for index in range(len(angles)):
        try:
            item = next(iterator)
        except StopIteration as exc:
            raise ValueError('image and angle lengths differ') from exc
        reference = ((item.source_identity, item.index) if isinstance(item, ScanFrame)
                     else index if isinstance(item, np.ndarray) or item is None else item)
        references.append(reference)
        try:
            oriented, invalid = oriented_and_invalid(load(item))
            centres[index],errors[index] = _peak(oriented,invalid,beam,motion,
                strip_half_width,roi,saturation)
        except (OSError,ValueError,IndexError,KeyError,RuntimeError,OptimizeWarning) as exc:
            reasons.append(f'{type(exc).__name__}: {exc}')
        else:
            reasons.append(None)
        # No scan stack or array-valued frame identifiers are retained.
        item = oriented = invalid = None
    sentinel = object()
    if next(iterator,sentinel) is not sentinel:
        raise ValueError('image and angle lengths differ')
    pitch = state.get('pitch')
    effective_shape = None if not state else tuple(state['raw_shape'][::-1] if orient.swaps_axes
                                                   else state['raw_shape'])
    used = np.isfinite(centres)
    if used.sum() < 3 or np.unique(angles[used]).size < 3:
        detail = '; '.join(f'row {i}: {r}' for i,r in enumerate(reasons) if r)
        raise ValueError(f'need at least three usable rows at distinct angles; {detail}')
    # Fit two parameters only, centred for conditioning. Equal weights avoid
    # treating exact synthetic/profile fits as infinitely precise observations.
    if model == 'tangent':
        coordinate = np.tan(np.deg2rad(angles-angle_zero_deg))
        x,y = coordinate[used],centres[used]
    else:
        x,y = centres[used],angles[used]
    dx = x-x.mean()
    ss = float(dx@dx)
    if ss <= np.finfo(float).eps*max(1.,float(x@x)):
        raise ValueError('degenerate calibration coordinate range')
    slope = float(dx@(y-y.mean())/ss)
    intercept = float(y.mean()-slope*x.mean())
    regression_residual = y-(intercept+slope*x)
    slope_std = float(np.sqrt((regression_residual@regression_residual)/(len(x)-2)/ss))
    if os.environ.get('XDART_DEBUG_DIRECT_BEAM'):
        logging.getLogger(__name__).warning('direct_beam motion=%s orientation=%s shape=%s '
            'used=%s centre_span_px=%g slope=%g slope_std=%g', motion, orient,
            state['raw_shape'], int(used.sum()), float(np.ptp(centres[used])), slope, slope_std)
    if abs(slope) <= max(1e-12,3*slope_std) or np.ptp(centres[used]) < 1e-6:
        raise ValueError('degenerate or insignificant beam motion')
    if model == 'tangent':
        distance = abs(slope)*pitch
        uncertainty = slope_std*pitch
        predictions = intercept+slope*coordinate
        slope_deg = float(np.rad2deg(1/slope))
        intercept_px = intercept
    else:
        radians = abs(np.deg2rad(slope))
        if radians >= np.pi/2:
            raise ValueError('linear slope is outside the local small-angle regime')
        distance = pitch/np.tan(radians)
        uncertainty = pitch/np.sin(radians)**2*np.deg2rad(slope_std)
        predictions = (angles-intercept)/slope
        slope_deg = slope
        intercept_px = -intercept/slope
    return DirectBeamResult(float(distance),float(uncertainty),angles,
        tuple(references) if ids is None else ids,points,centres,errors,used,
        tuple(reasons),predictions,centres-predictions,model,angle_zero_deg,pitch,
        effective_shape,slope_deg,float(intercept_px),(float(beam[0]),float(beam[1])),motion)


@contextmanager
def open_direct_beam_scan(
    scan_file: str | Path | FrameSource, scan: str | int | None = None, *,
    images: Iterable | None = None, image_file: str | Path | None = None,
    image_dir: str | Path | None = None, filename_template: str | None = None,
    point_index_origin: int = 0, motor: str = 'del',
    angles_deg: Sequence[float] | None = None,
    frame_indices: Sequence[int] | None = None, entry: str = 'entry',
):
    """Load paired angles and lazy images from SPEC, raw NeXus or FrameSource.

    SPEC requires an exact scan number/repetition and either ``image_file``
    (an Eiger/NeXus container), ordered ``images``, or ``image_dir`` plus a
    filename template. Raw NeXus needs no scan selector; the existing metadata
    reader supplies ``motor``, or ``angles_deg`` supplies explicit degree angles.
    Beam pixel, mask, detector and acquisition cutoff are never inferred here.

    ``frame_indices`` selects ordered original labels (zero-based for files),
    and their corresponding metadata rows. Automatic whole-container pairing
    requires equal counts; an explicit subset requires only its paired rows.
    For an arbitrary different angle/frame association use the numerical API.
    Explicit ``images`` already defines its own order and cannot be reselected.

    A caller-owned FrameSource supplies raw frames and either ``motors`` arrays,
    per-frame ``metadata_for`` values, or explicit ``angles_deg`` aligned with
    its full ``frame_indices`` order. Its name/path labels provenance; callers
    can override IDs in the fitter. It is never closed here. Container handles
    opened here close on context exit, including errors. Consume images inside
    the context. Units and detector-arm motor meaning remain caller choices.
    """
    from xrd_tools.io.spec import is_spec_file, read_spec_scan_table
    from xrd_tools.io.nexus import read_nexus
    from xrd_tools.sources.cursor import ContainerCursor

    if not isinstance(point_index_origin, int):
        raise ValueError('point_index_origin must be an integer')
    template_input = image_dir is not None or filename_template is not None
    if sum((images is not None, image_file is not None, template_input)) > 1:
        raise ValueError('supply only one of images, image_file, or image_dir/filename_template')
    with ExitStack() as resources:
        key = None
        references = False
        if isinstance(scan_file, FrameSource):
            if scan is not None or images is not None or image_file is not None or template_input:
                raise ValueError('FrameSource already supplies images; no scan or image options allowed')
            labels = tuple(scan_file.frame_indices)
            identity = getattr(scan_file, 'path', getattr(scan_file, 'name', 'source'))
            load = scan_file.load_frame
            if angles_deg is None:
                motors = getattr(scan_file, 'motors', {})
                if motor in motors:
                    angles_deg = motors[motor]
                else:
                    metadata = getattr(scan_file, 'metadata_for', None)
                    if metadata is None:
                        raise ValueError(f'source has no {motor!r} metadata; supply angles_deg')
                    try:
                        angles_deg = [metadata(label)[motor] for label in labels]
                    except KeyError as exc:
                        raise ValueError(f'source has no {motor!r} angle column') from exc
        else:
            path = Path(scan_file)
            if scan is not None or is_spec_file(path):
                if scan is None:
                    raise ValueError('SPEC input requires an explicit scan number/repetition')
                if angles_deg is not None:
                    raise ValueError('SPEC supplies angles; use the numerical API for explicit angles')
                key = str(scan)
                if '.' not in key:
                    key += '.1'
                columns, _, count = read_spec_scan_table(path, key)
                if motor not in columns:
                    raise ValueError(f'{motor!r} is not a scanned column in SPEC {key}')
                angles_deg = columns[motor]
                if images is not None:
                    if frame_indices is not None:
                        raise ValueError('explicit images already select their rows; omit frame_indices')
                    yield DirectBeamScanInput(np.array(angles_deg, dtype=float), images, None,
                        tuple((key, i + point_index_origin) for i in range(count)))
                    return
                if image_file is None:
                    if image_dir is None or filename_template is None or '{point' not in filename_template:
                        raise ValueError('supply image_file, images, or image_dir/template with {point}')
                    number, repetition = key.split('.')
                    labels = tuple(range(count))
                    identity = path
                    def load(index):
                        return Path(image_dir) / filename_template.format(
                            point=index + point_index_origin, scan=int(number), repetition=int(repetition))
                    references = True
                else:
                    path = Path(image_file)
            else:
                if images is not None or image_file is not None or template_input:
                    raise ValueError('NeXus input already supplies images; omit image options')
                if angles_deg is None:
                    metadata = read_nexus(path, entry, motor_names=[motor], counter_names=[])
                    if motor not in metadata.angles:
                        raise ValueError(f'NeXus has no {motor!r} angle column; supply angles_deg')
                    angles_deg = metadata.angles[motor]
            if not references:
                cursor = resources.enter_context(ContainerCursor(path, entry=entry))
                labels = tuple(range(cursor.frame_count))
                identity, load = path, cursor.read_frame

        angles = np.array(angles_deg, dtype=float, copy=True)
        if angles.ndim != 1 or not np.isfinite(angles).all():
            raise ValueError('angles_deg must be a finite one-dimensional array')
        if frame_indices is None and len(angles) != len(labels):
            raise ValueError(f'angle/image count mismatch: {len(angles)} angles, {len(labels)} frames')
        selected = labels if frame_indices is None else tuple(frame_indices)
        if (any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError('frame_indices must contain unique integer labels')
        positions = {label: i for i, label in enumerate(labels)}
        if any(label not in positions for label in selected):
            raise ValueError('selected frame label is absent from source')
        rows = [positions[label] for label in selected]
        if any(row >= len(angles) for row in rows):
            raise ValueError('selected frame has no corresponding angle row')
        points = (tuple((key, row + point_index_origin) for row in rows) if key is not None
                  else tuple((identity, label) for label in selected))
        if references:
            image_items = (load(label) for label in selected)
            ids = None
        else:
            image_items = (ScanFrame(index=label, source_identity=str(identity),
                loader=lambda frame: load(frame.index)) for label in selected)
            ids = tuple((identity, label) for label in selected)
        yield DirectBeamScanInput(angles[rows], image_items, ids, points)


def calibrate_direct_beam_scan(
    scan_file: str | Path | FrameSource, scan: str | int | None = None, *,
    images: Iterable | None = None, image_file: str | Path | None = None,
    image_dir: str | Path | None = None, filename_template: str | None = None,
    point_index_origin: int = 0, motor: str = 'del',
    angles_deg: Sequence[float] | None = None,
    frame_indices: Sequence[int] | None = None, entry: str = 'entry',
    **calibration_options,
) -> DirectBeamResult:
    """One-call wrapper around source loading and the numerical distance fit.

    See :func:`open_direct_beam_scan` for sources/selection and
    :func:`calibrate_direct_beam` for detector, explicit beam pixel and fit options.
    Existing SPEC calls retain their scan-number/repetition and image semantics.
    """
    with open_direct_beam_scan(scan_file, scan, images=images, image_file=image_file,
            image_dir=image_dir, filename_template=filename_template,
            point_index_origin=point_index_origin, motor=motor, angles_deg=angles_deg,
            frame_indices=frame_indices, entry=entry) as inputs:
        calibration_options.setdefault('frame_ids', inputs.frame_ids)
        calibration_options.setdefault('point_ids', inputs.point_ids)
        return calibrate_direct_beam(inputs.angles_deg, inputs.images, **calibration_options)


__all__ = ['DirectBeamResult', 'DirectBeamScanInput', 'open_direct_beam_scan',
           'calibrate_direct_beam', 'calibrate_direct_beam_scan']
