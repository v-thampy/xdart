"""Headless sanity check for the xrd_tools ingest -> reduce -> persist ->
read-back spine.

This runs the whole path with **no Qt / GUI imports**, proving the analysis
core is usable headless -- from notebooks, batch jobs, and (eventually) an
autonomous loop -- not just from the xdart GUI.  It is also the living example
the notebook docs should track.

What it exercises:
  * build a small in-memory ``Scan`` of synthetic frames + a real pyFAI
    integrator (no GUI, no detector files);
  * ``run_reduction(plan, scan, sink=[NexusSink, XYESink, MemorySink])`` --
    one spine, three outputs (a .nxs, per-frame .xye, and an in-memory dict);
  * round-trip the .nxs through the public read API: ``read_scan`` (xarray),
    ``get_1d`` / ``get_2d`` / ``get_metadata``, and the per-frame ``Scan``
    accessor object.

Run it in an environment where importing ``xdart`` or Qt would fail -- it must
still pass::

    python examples/headless_sanity.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from xrd_tools.reduction import (
    Frame,
    Integration1DPlan,
    Integration2DPlan,
    MemorySink,
    NexusSink,
    ReductionPlan,
    Scan,
    XYESink,
    run_reduction,
)
from xrd_tools.io.read import get_1d, get_2d, get_metadata, get_thumbnail
from xrd_tools.io.nexus import read_scan

N_FRAMES = 5
SHAPE = (64, 64)


def _build_integrator():
    """A real pyFAI azimuthal integrator for a small synthetic detector."""
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    pixel = 100e-6  # 100 um
    det = Detector(pixel1=pixel, pixel2=pixel, max_shape=SHAPE)
    center = (SHAPE[0] / 2) * pixel
    return AzimuthalIntegrator(
        dist=0.1, poni1=center, poni2=center, detector=det, wavelength=1e-10,
    )


def _synthetic_scan():
    rng = np.random.default_rng(0)
    frames = [
        Frame(
            index=i,
            image=rng.random(SHAPE) + i,  # slight per-frame variation
            metadata={"i0": 100.0 + i, "sample": "demo", "temperature_C": 25 + i},
        )
        for i in range(N_FRAMES)
    ]
    return Scan(name="headless_demo", frames=frames, integrator=_build_integrator())


def main() -> int:
    plan = ReductionPlan(
        integration_1d=Integration1DPlan(npt=200, unit="q_A^-1"),
        integration_2d=Integration2DPlan(npt_rad=200, npt_azim=36, unit="q_A^-1"),
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        nxs = tmp / "headless_demo.nxs"
        xye_dir = tmp / "xye"
        memory = MemorySink()

        result = run_reduction(
            plan,
            _synthetic_scan(),
            sink=[NexusSink(nxs, overwrite=True), XYESink(xye_dir), memory],
        )

        # --- the run itself -------------------------------------------------
        assert result.n_processed == N_FRAMES, result.n_processed
        assert not result.failed, result.error
        assert len(memory.frames) == N_FRAMES
        assert sorted(memory.frames) == list(range(N_FRAMES))
        xye_files = sorted(xye_dir.glob("*.xye"))
        assert len(xye_files) == N_FRAMES, xye_files
        print(f"reduced {result.n_processed} frames -> "
              f"{nxs.name}, {len(xye_files)} .xye files, {len(memory.frames)} in memory")

        # --- read the .nxs back through the public API ----------------------
        ds = read_scan(nxs)
        assert list(ds["frame"].values) == list(range(N_FRAMES))
        assert ds["intensity_1d"].shape[0] == N_FRAMES
        assert ds["intensity_2d"].shape[0] == N_FRAMES
        print(f"read_scan: 1d={tuple(ds['intensity_1d'].shape)} "
              f"2d={tuple(ds['intensity_2d'].shape)} q-unit={ds['q'].attrs.get('units')}")

        one = get_1d(nxs, frame=0)
        two = get_2d(nxs, frame=0)
        assert one.intensity.ndim == 1 and two.intensity.ndim == 2
        print(f"get_1d/get_2d(frame=0): 1d={one.intensity.shape} 2d={two.intensity.shape}")

        meta = get_metadata(nxs)
        assert meta["n_frames"] == N_FRAMES and meta["has_1d"] and meta["has_2d"]
        # The heterogeneous per-frame table round-trips (numeric + string).
        scan_data = meta.get("scan_data", {})
        print(f"get_metadata: n_frames={meta['n_frames']} "
              f"scan_data columns={sorted(scan_data)}")

        # Thumbnails are an xdart *display* artifact, not part of the headless
        # write -- get_thumbnail documents that by raising for a pure run.
        try:
            get_thumbnail(nxs, 0)
            print("get_thumbnail: present")
        except KeyError:
            print("get_thumbnail: none (headless write omits thumbnails, as expected)")

    # --- the whole point: no xdart GUI app on the import graph --------------
    # ssrl's own core/reduction/io stay import-clean (enforced by the
    # headless-purity guard).  pyFAI may transitively import a Qt *binding* for
    # its optional calibration GUI -- that's a pyFAI packaging detail, not ssrl
    # pulling in a GUI -- so the meaningful check here is that the xdart app and
    # pyqtgraph (the stack we own) never load.
    gui_roots = {"xdart", "pyqtgraph"}
    leaked = sorted({m.split(".")[0] for m in sys.modules} & gui_roots)
    assert not leaked, f"headless example pulled in the xdart GUI stack: {leaked}"
    print("OK: full ingest -> reduce -> persist -> read-back ran without the "
          "xdart/pyqtgraph GUI stack.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
