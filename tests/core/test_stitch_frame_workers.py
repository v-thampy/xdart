"""Stitch's frame workers (PERF-STITCH-2; maintainer, 2026-09-30: "look into implementing multiple workers like in
int"): the streaming MultiGeometry merge integrates several frames at once on threads, as Int's reduction does,
while one thread reads the frames and merges their sums in frame order. Any worker count gives the same result,
bit for bit; pyFAI's numexpr work holds the one process-wide lock Int's workers hold (numexpr 2.14 no longer
serializes its calls); a failure or a stopped read ends the run as one worker's would, leaving no thread."""
import dataclasses
import inspect
import math
import threading

import numexpr
import numpy as np
import pytest

from xrd_tools.integrate import multi, pyfai_numexpr
from xrd_tools.integrate.frame_screen import FrameScreen, FrameScreenPolicy
from xrd_tools.integrate.stitch_gi import StitchGI
from tests.core.test_xu_extended_arm import SHAPE as XU_SHAPE, _receipt, transferred  # noqa: F401

SHAPE = (48, 64)
ARM = np.deg2rad([2., 6., 10., 14., 18., 22., 26., 30., 34., 38.])  # each frame's rot2
GI = StitchGI(("q_ip", "q_oop"), incident_angle_deg=.4)
CONFIGS = ("1d", "1d-corrections", "1d-screen", "2d", "2d-iq", "2d-iq-2th-map", "1d-gi", "2d-gi")


def _ai(rot2):
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    return AzimuthalIntegrator(dist=.2, poni1=24 * 172e-6, poni2=32 * 172e-6, rot2=rot2,
                               detector=Detector(pixel1=172e-6, pixel2=172e-6, max_shape=SHAPE),
                               wavelength=.729e-10)


def _factory():
    return (_ai(rot2) for rot2 in ARM)


def _images():
    rng = np.random.default_rng(11)
    frames = [rng.poisson(40, SHAPE).astype(float) + 5. for _ in ARM]
    frames[4][:] = 0.  # a dropped frame: the screen leaves it out
    return frames


def _stitch(config, workers, images=None, factory=_factory):
    images = _images() if images is None else images
    count = len(ARM)
    common = dict(correct_solid_angle=True, polarization_factor=.99, method="full", workers=workers)
    if config == "1d":
        return multi.stitch_1d_streaming(images, count, factory, npt=64, **common)
    if config == "1d-corrections":
        rng = np.random.default_rng(3)
        return multi.stitch_1d_streaming(
            images, count, factory, npt=64, dark=5. + rng.uniform(-1, 1, SHAPE), flat=rng.uniform(.8, 1.2, SHAPE),
            error_model="poisson", normalization=np.linspace(1., 2., count), **common)
    if config == "1d-screen":
        screen = FrameScreen(FrameScreenPolicy(exclude="partial", window=2), list(range(count)), [0] * count)
        return multi.stitch_1d_streaming(images, count, factory, npt=64, frame_screen=screen, **common), \
            [verdict.excluded for verdict in screen.verdicts]
    if config == "2d":
        return multi.stitch_2d_streaming(images, count, factory, npt_rad=48, npt_azim=36, **common)
    if config == "2d-iq":
        return multi.stitch_2d_with_1d_streaming(images, count, factory, npt_rad=48, npt_azim=36, npt_1d=64, **common)
    if config == "2d-iq-2th-map":
        return multi.stitch_2d_with_1d_streaming(images, count, factory, npt_rad=48, npt_azim=36, npt_1d=64,
                                                 unit="2th_deg", unit_1d="q_A^-1", **common)
    if config == "1d-gi":
        return multi.stitch_1d_streaming(images, count, factory, npt=64, gi=GI, gi_angles=[(.4, 0.)] * count,
                                         **common)
    assert config == "2d-gi"
    return multi.stitch_2d_with_1d_streaming(images, count, factory, npt_rad=48, npt_azim=36, npt_1d=64, gi=GI,
                                             gi_angles=[(.4, 0.)] * count, **common)


def _arrays(value):
    """Every array (and number) a Stitch returns, in order."""
    if isinstance(value, np.ndarray):
        return [value]
    if isinstance(value, (bool, int, float)):
        return [np.asarray(value)]
    if isinstance(value, (tuple, list)):
        return [array for item in value for array in _arrays(item)]
    if dataclasses.is_dataclass(value):
        return [array for field in dataclasses.fields(value) for array in _arrays(getattr(value, field.name))]
    return []


@pytest.mark.parametrize("config", CONFIGS)
def test_any_worker_count_merges_the_same_sums_bit_for_bit(config):
    one = _arrays(_stitch(config, 1))
    assert len(one) >= 4 and any(array.dtype.kind == "f" and np.nansum(array) > 0 for array in one)
    for workers in (2, 4):
        many = _arrays(_stitch(config, workers))
        assert len(many) == len(one)
        for serial, parallel in zip(one, many):
            assert serial.dtype == parallel.dtype and serial.shape == parallel.shape
            assert np.array_equal(serial, parallel, equal_nan=serial.dtype.kind == "f"), (config, workers)
    if config == "1d-screen":
        assert _stitch(config, 4)[1] == [False] * 4 + [True] + [False] * 5  # the dropped frame, left out



@pytest.mark.parametrize("unit", ("q_A^-1", "2th_deg"))
def test_the_range_scouting_finds_array_from_units_extents_on_any_worker_count(unit):
    """PERF-STITCH-3's scouting pass: each frame's pixel positions once, outside the lock (pyFAI's Cython),
    only the unit formulas under it, ``workers`` frames at once; the extents are pyFAI's ``array_from_unit``'s,
    bit for bit. (The census below checks every numexpr call holds the lock.)"""
    from pyFAI import units
    expected = [math.inf, -math.inf, math.inf, -math.inf]
    for integrator in _factory():
        integrator.setChiDiscAtPi()
        for offset, scouted in ((0, units.to_unit(unit)), (2, units.CHI_DEG)):
            values = np.asarray(integrator.array_from_unit(unit=scouted), dtype=float)
            expected[offset] = min(expected[offset], float(values.min()))
            expected[offset + 1] = max(expected[offset + 1], float(values.max()))
    for workers in (1, 4):
        radial, azimuth = multi._resolve_streaming_ranges(_factory, len(ARM), unit=unit, radial_range=None,
                                                          azimuth_range=None, workers=workers)
        assert (*radial, *azimuth) == tuple(expected), workers


def test_a_parallax_geometry_scouts_through_pyfais_own_call():
    """Positions with a parallax correction evaluate numexpr: such a geometry keeps pyFAI's own call, under
    the lock, never the positions outside it."""
    from pyFAI import units
    calls = []

    class Parallax:
        _parallax, wavelength = object(), 1e-10

        def setChiDiscAtPi(self):
            calls.append("disc")

        def reset(self, collect_garbage):
            calls.append("reset")

        def position_array(self, *args, **kwargs):
            raise AssertionError("positions outside the lock")

        def array_from_unit(self, unit):
            calls.append(unit.name)
            return np.arange(4.)
    assert multi._scout_extents(Parallax(), units.to_unit("q_A^-1"), True, True) == ((0., 3.), (0., 3.))
    assert calls == ["disc", "q_A^-1", "chi_deg", "reset"]

class _OwnedLock:
    """The shared lock, knowing which thread holds it."""

    def __init__(self):
        self._lock, self.owner = threading.Lock(), None

    def __enter__(self):
        self._lock.acquire()
        self.owner = threading.get_ident()
        return self

    def __exit__(self, *_exc):
        self.owner = None
        self._lock.release()


class _Formula:
    """A compiled numexpr formula that records who evaluates it."""

    def __init__(self, compiled, check):
        self._compiled, self._check = compiled, check

    def __call__(self, *args, **kwargs):
        self._check("formula")
        return self._compiled(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._compiled, name)


@pytest.mark.parametrize("config", CONFIGS)
def test_every_numexpr_call_holds_the_pyfai_lock(config, monkeypatch):
    """pyFAI reaches numexpr in unit formulas, the polarization array, GI fiber units and ``union``: with four
    workers every such call (numexpr.evaluate, a compiled formula, a unit's equation) is made by the thread
    holding the lock Int's workers hold."""
    import pyFAI.units as units

    lock = _OwnedLock()
    monkeypatch.setattr(pyfai_numexpr, "PYFAI_NUMEXPR_LOCK", lock)
    calls, unlocked = [], []

    def check(site):
        calls.append(site)
        if lock.owner != threading.get_ident():
            unlocked.append((site, threading.current_thread().name))

    evaluate, compile_ = numexpr.evaluate, numexpr.NumExpr
    depth = inspect.signature(evaluate).parameters["_frame_depth"].default

    def checked_evaluate(*args, **kwargs):
        check("evaluate")
        kwargs["_frame_depth"] = kwargs.get("_frame_depth", depth) + 1  # numexpr reads its caller's locals
        return evaluate(*args, **kwargs)

    def checked_compile(*args, **kwargs):
        check("compile")
        return _Formula(compile_(*args, **kwargs), check)
    monkeypatch.setattr(numexpr, "evaluate", checked_evaluate)
    monkeypatch.setattr(numexpr, "NumExpr", checked_compile)
    for registry in (value for value in vars(units).values() if isinstance(value, dict)):
        for unit in registry.values():
            if isinstance(unit, units.Unit) and getattr(unit, "_equation_ne", None) is not None \
                    and unit.equation is unit._equation_ne:
                def equation(*args, _inner=unit.equation, **kwargs):
                    check("unit")
                    return _inner(*args, **kwargs)
                monkeypatch.setattr(unit, "equation", equation)
    _stitch(config, 4)
    assert "evaluate" in calls  # union's means, at least
    assert unlocked == [], unlocked[:5]


def test_at_most_two_frames_a_worker_are_read_and_not_merged():
    workers, read, merged, most = 3, [0], [0], [0]

    def images():
        for image in _images():
            read[0] += 1
            most[0] = max(most[0], read[0] - merged[0])
            yield image

    class Counting(FrameScreen):
        def push(self, total, item):
            merged[0] += 1
            return super().push(total, item)
    screen = Counting(FrameScreenPolicy(exclude="partial", window=2), list(range(len(ARM))), [0] * len(ARM))
    multi.stitch_1d_streaming(images(), len(ARM), _factory, npt=64, method="full", frame_screen=screen, workers=workers)
    assert read[0] == merged[0] == len(ARM) and most[0] <= 2 * workers


class _Stopped(Exception):
    pass


def _frame_threads():
    return [thread for thread in threading.enumerate() if thread.name.startswith("xdart-stitch-frame")]


@pytest.mark.parametrize("workers", (1, 4))
def test_a_failing_frame_raises_what_one_worker_raises_and_leaves_no_thread(workers):
    images = _images()
    images[6] = images[6][0]  # 1-D: this frame fails in its worker
    with pytest.raises(ValueError, match="image must be 2-D, got shape"):
        _stitch("2d-iq", workers, images=images)
    assert _frame_threads() == []


@pytest.mark.parametrize("workers", (1, 4))
def test_a_stopped_read_ends_the_run_and_keeps_no_geometry(workers):
    """A cancelled Stitch stops its read (``run_stitch`` raises StitchCancelled from the frame iterator): the
    frames not started are dropped, the running ones finish, and no integrator keeps its geometry arrays."""
    built = []

    def factory():
        for rot2 in ARM:
            built.append(_ai(rot2))
            yield built[-1]

    def images():
        for index, image in enumerate(_images()):
            if index == 7:
                raise _Stopped
            yield image
    with pytest.raises(_Stopped):
        _stitch("2d-iq", workers, images=images(), factory=factory)
    assert _frame_threads() == []
    assert len(built) == len(ARM) + 7  # the range scouting's, then the frames read before the stop
    assert all(not integrator._cached_array for integrator in built)


def test_workers_must_be_a_positive_int_and_several_take_full_splitting():
    for bad in (0, -1, 1.0, "2"):
        with pytest.raises(ValueError, match="positive int"):
            _stitch("1d", bad)
    # the operation's method; another method's geometry is not proven to keep numexpr under the lock
    with pytest.raises(ValueError, match="'full' splitting; 'BBox' takes one worker"):
        multi.stitch_1d_streaming(_images(), len(ARM), _factory, npt=64, method="BBox", workers=2)
    multi.stitch_1d_streaming(_images(), len(ARM), _factory, npt=64, method="BBox", workers=1)


# PERF-STITCH-3 (maintainer, 2026-09-30: "i'd try 2 or 3. start with the easiest one first"): xrayutilities'
# own threads (3) measured no gain (its conversion is ~2 % of a scan-14 Stitch); its frames' NumPy histograms run
# on the workers while every xrayutilities call stays on the thread its runtime session belongs to (2).
XU_ANGLES = {label: (12. + 3.5 * (label - 1), -7. + 3. * (label - 1)) for label in range(1, 9)}
XU_CONFIGS = ("xu-1d", "xu-corrections-screen", "xu-gi-2d")


def _xu_stitch(tmp_path, transferred, config, workers):  # noqa: F811
    from xrd_tools.integrate.xu_stitch import run_xu_hist_stitch_1d

    class Source:
        frame_indices = tuple(XU_ANGLES)

        def metadata_for(self, label):
            return {"del": XU_ANGLES[label][0], "nu": XU_ANGLES[label][1], "eta": .4, "chi": 90., "mu": 0.}

        def load_frame(self, label):
            image = np.random.default_rng(label).uniform(5., 50., XU_SHAPE).astype(np.float32)
            if label == 5:
                image[:] = 0.  # a dropped frame: the screen leaves it out
            return image
    common = dict(frame_indices=tuple(XU_ANGLES), npt=48, radial_range=(.5, 6.), monitor_key=None,
                  max_frame_bytes=8 * 1024 * 1024, workers=workers)
    if config == "xu-corrections-screen":
        rng = np.random.default_rng(7)
        count = len(XU_ANGLES)
        common.update(polarization_factor=.99, dark=np.full(XU_SHAPE, 1.5, np.float32),
                      flat=rng.uniform(.8, 1.2, XU_SHAPE).astype(np.float32),
                      frame_screen=FrameScreen(FrameScreenPolicy(exclude="partial", window=2), list(XU_ANGLES),
                                               [0] * count))
    elif config == "xu-gi-2d":
        common.update(gi=StitchGI(("q_ip", "q_oop"), incidence_motor="eta", tilt_motor="chi", ip_range=(-4., 4.),
                                  oop_range=(0., 1.5), sample_motors=(("mu", "mu"), ("phi", None))),
                      map_grid=(32, 24, (-180., 180.)))
    return run_xu_hist_stitch_1d(_receipt(tmp_path, transferred[0]), Source(), **common)


@pytest.mark.parametrize("config", XU_CONFIGS)
def test_xrayutilities_workers_bin_the_same_frames_bit_for_bit(tmp_path, transferred, config):  # noqa: F811
    one = _arrays(_xu_stitch(tmp_path / "one", transferred, config, 1))
    assert len(one) >= 4 and any(array.dtype.kind == "f" and np.nansum(array) > 0 for array in one)
    for workers in (2, 4):
        many = _arrays(_xu_stitch(tmp_path / f"w{workers}", transferred, config, workers))
        assert len(many) == len(one)
        for serial, parallel in zip(one, many):
            assert serial.dtype == parallel.dtype and serial.shape == parallel.shape
            assert np.array_equal(serial, parallel, equal_nan=serial.dtype.kind == "f"), (config, workers)
    assert _frame_threads() == []
