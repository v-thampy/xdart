"""CI-VIEWERS-HANG-128: no pyFAI integrator xdart builds runs ``gc.collect()`` on a geometry change.

pyFAI collects garbage whenever a geometry parameter changes unless ``auto_gc`` is off. On a Run worker that full
collection can finalize Qt objects off the GUI thread; CI hung that way. Frozen before the product change.
"""

from __future__ import annotations

import ast
import gc
import inspect
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src" / "xrd_tools"
_PYFAI_CONSTRUCTORS = {"AzimuthalIntegrator", "FiberIntegrator", "Geometry"}


@pytest.fixture
def collections(monkeypatch):
    calls: list[tuple] = []
    real = gc.collect

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(gc, "collect", counting)
    return calls


def _assert_quiet(integrator, collections) -> None:
    assert integrator.auto_gc is False
    before = len(collections)
    integrator.wavelength = (integrator.wavelength or 1.0e-10) * 1.01
    integrator.reset_engines()
    assert len(collections) == before


def _poni():
    from xrd_tools.core.containers import PONI

    return PONI(dist=0.2, poni1=0.0006, poni2=0.0007, wavelength=1.0e-10, detector="Detector")


def test_poni_integrators_and_their_worker_clones_never_collect_garbage(collections) -> None:
    from xrd_tools.integrate.calibration import (
        _clone_integrator_preserving_calibration,
        poni_to_integrator,
    )

    _assert_quiet(poni_to_integrator(_poni()), collections)
    _assert_quiet(_clone_integrator_preserving_calibration(poni_to_integrator(_poni())), collections)


def test_detector_calibration_integrators_never_collect_garbage(collections) -> None:
    from xrd_tools.core.geometry.diffractometer import DetectorCalibration
    from xrd_tools.integrate.calibration import detector_calibration_to_integrator

    calibration = DetectorCalibration(
        _poni(), {"pixel1": 1.0e-4, "pixel2": 1.0e-4, "max_shape": [12, 14]},
    )
    _assert_quiet(detector_calibration_to_integrator(calibration), collections)


def test_fiber_integrators_never_collect_garbage(collections) -> None:
    from xrd_tools.integrate.calibration import poni_to_fiber_integrator, poni_to_integrator
    from xrd_tools.integrate.stitch_hist import _as_fiber

    _assert_quiet(poni_to_fiber_integrator(_poni(), incident_angle=0.2), collections)
    _assert_quiet(_as_fiber(poni_to_integrator(_poni())), collections)


def test_calibration_seeding_keeps_pyfai_collection_off_by_default() -> None:
    from xrd_tools.integrate.calibration_seed import predict_frame_seed

    assert inspect.signature(predict_frame_seed).parameters["collect_garbage"].default is False


def _sets_auto_gc_off(function: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "auto_gc" for target in node.targets)
        and isinstance(node.value, ast.Constant)
        and node.value.value is False
        for node in ast.walk(function)
    )


def _builds_pyfai_geometry(function: ast.AST) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name in _PYFAI_CONSTRUCTORS or name == "promote":
            return True
    return False


def test_every_pyfai_geometry_xrd_tools_builds_turns_collection_off() -> None:
    offenders = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and _builds_pyfai_geometry(node)
                and not _sets_auto_gc_off(node)
            ):
                offenders.append(f"{path.relative_to(_SRC.parent)}:{node.lineno} {node.name}")
    assert offenders == []


def test_the_scan_ring_search_never_collects_garbage(tmp_path, collections) -> None:
    """NEXT-129-SMALL-FIXES finding H: pyFAI's SingleGeometry builds a refinement of its own from the integrator's
    config, which drops auto_gc, and that refinement's constructor ran two full gc.collect() per frame on the
    calibration worker (XDART_DEBUG_CALIBRATION_GC: 128 a Refine, about 12 s, the first freeing 10,949 objects);
    one such collection hung a GUI test run for 10.5 h. Frozen before the product change."""
    from tests.core.test_stitch_autocalibration import ROW_ANGLES, five_parameter_seed, ring_scan
    from xrd_tools.integrate.stitch_autocalibration import extract_scan_ring_points

    spec = ring_scan(tmp_path / "scan", [(delta, 0.0) for delta in (6.0, 20.0, 34.0)])
    points = extract_scan_ring_points(spec, five_parameter_seed(ROW_ANGLES), max_frames=3)
    assert points.controls, "the search found ring points"
    assert collections == [], f"{len(collections)} full collections ran during the ring search"


def test_the_ring_searchs_single_geometry_is_built_as_pyfais_own(tmp_path, monkeypatch) -> None:
    """Finding H's construction adapter builds what pyFAI's SingleGeometry builds for the same frame (the same
    attributes, refinement geometry and seeded ring points), with collection off in its refinement only."""
    import numpy as np
    from pyFAI.goniometer import SingleGeometry
    from tests.core.test_stitch_autocalibration import ROW_ANGLES, five_parameter_seed, ring_scan
    from xrd_tools.integrate import stitch_autocalibration as owner

    built, adapter = [], owner._single_geometry

    def recording(label, image, calibrant, integrator):  # a pass-through: each frame's inputs and result
        single = adapter(label, image, calibrant, integrator)
        built.append((single, label, image, calibrant, integrator))
        return single

    monkeypatch.setattr(owner, "_single_geometry", recording)
    spec = ring_scan(tmp_path / "scan", [(delta, 0.0) for delta in (6.0, 20.0, 34.0)])
    owner.extract_scan_ring_points(spec, five_parameter_seed(ROW_ANGLES), max_frames=3)
    assert len(built) == 3
    for single, label, image, calibrant, integrator in built:
        reference = SingleGeometry(label, image=image, calibrant=calibrant, geometry=integrator)
        assert set(vars(single)) == set(vars(reference))
        assert single.geometry_refinement.get_config() == reference.geometry_refinement.get_config()
        assert single.geometry_refinement.auto_gc is False and reference.geometry_refinement.auto_gc is True
        assert single.detector.get_config() == reference.detector.get_config()
        np.testing.assert_array_equal(owner._seeded_extract(single, label, 2.0),
                                      owner._seeded_extract(reference, label, 2.0))
