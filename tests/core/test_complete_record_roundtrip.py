"""6a acceptance: a PURELY HEADLESS run writes the complete v2 record.

run_reduction + NexusSink(source_base=...) on frames whose raw images live
in real source files inside a project root must produce a file where:

* ``get_raw_frame`` resolves the raw via the per-frame source pointer;
* ``read_frame_view`` returns the thumbnail + source ref;
* the record is N1-portable: MOVE the whole project directory and
  resolution still works (relative pointers + @source_base semantics).
"""
from __future__ import annotations

import shutil

import numpy as np
import pytest
from pyFAI.detectors import Detector
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from xrd_tools.reduction import NexusSink, run_reduction
from xrd_tools.reduction.core import (
    Frame,
    Integration1DPlan,
    Integration2DPlan,
    ReductionPlan,
    Scan,
)

_SHAPE = (64, 60)


@pytest.fixture()
def project(tmp_path):
    """A project root holding real single-frame TIFF sources."""
    fabio = pytest.importorskip("fabio")
    root = tmp_path / "proj"
    (root / "raw").mkdir(parents=True)
    rng = np.random.default_rng(11)
    frames = []
    for i in range(3):
        img = (rng.random(_SHAPE) * 1000).astype(np.float32)
        src = root / "raw" / f"img_{i:04d}.tif"
        fabio.tifimage.TifImage(data=img).write(str(src))
        frames.append(Frame(index=i, image=img, source_path=src,
                            source_frame_index=0,
                            metadata={"i0": 1.0 + i, "tth": 10.0 + i,
                                      "th": 0.5 + 0.1 * i}))
    return root, frames


@pytest.fixture()
def small_ai():
    det = Detector(pixel1=75e-6, pixel2=75e-6, max_shape=_SHAPE)
    return AzimuthalIntegrator(
        dist=0.2, poni1=0.0024, poni2=0.00225,
        wavelength=1.0e-10, detector=det,
    )


def test_headless_run_writes_complete_portable_record(project, tmp_path,
                                                      small_ai):
    root, frames = project
    out = root / "processed" / "scan.nxs"
    plan = ReductionPlan(
        integration_1d=Integration1DPlan(npt=50),
        integration_2d=Integration2DPlan(npt_rad=50, npt_azim=36),
    )
    sink = NexusSink(path=out, source_base=root, overwrite=True)
    from xrd_tools.core.geometry.diffractometer import DiffractometerGeometry
    result = run_reduction(
        plan,
        Scan("s", frames, integrator=small_ai,
             geometry=DiffractometerGeometry.two_circle()),
        sink=sink,
    )
    assert result.n_processed == 3

    import h5py
    with h5py.File(out, "r") as f:
        entry = f["entry"]
        assert entry.attrs["source_base"]
        for i in range(3):
            fg = entry[f"frames/frame_{i:04d}"]
            assert "thumbnail" in fg
            path = fg["source/path"][()]
            path = path.decode() if isinstance(path, bytes) else str(path)
            assert path == f"raw/img_{i:04d}.tif"      # RELATIVE = portable
        # finish-time geometry derived from scan_data via Scan.geometry
        pfg = entry["per_frame_geometry"]
        np.testing.assert_array_equal(pfg["frame_index"][()], [0, 1, 2])
        np.testing.assert_allclose(pfg["rot1"][()],
                                   np.deg2rad([10.0, 11.0, 12.0]), rtol=1e-6)
        np.testing.assert_allclose(pfg["incident_angle"][()],
                                   [0.5, 0.6, 0.7], rtol=1e-6)

    # Reader round-trip on the headless-written file
    from xrd_tools.io import get_raw_frame
    from xrd_tools.io.frame_view import read_frame_view
    raw = get_raw_frame(out, 1)
    np.testing.assert_allclose(np.asarray(raw), np.asarray(frames[1].image))
    fv = read_frame_view(out, 1)
    assert fv.thumbnail is not None
    assert fv.intensity_1d is not None and fv.intensity_1d.shape == (50,)
    assert fv.intensity_2d is not None
    # the reader resolves the stored relative pointer to an absolute path
    assert fv.source_path is not None
    assert fv.source_path.replace("\\", "/").endswith("raw/img_0001.tif")
    assert fv.source_frame_index == 0  # single-frame TIFF sources

    # N1 portability: MOVE the whole project (sources + output together)
    moved = tmp_path / "relocated"
    shutil.move(str(root), str(moved))
    moved_out = moved / "processed" / "scan.nxs"
    raw2 = get_raw_frame(moved_out, 2)
    np.testing.assert_allclose(np.asarray(raw2), np.asarray(frames[2].image))
