from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np

from xrd_tools.core.containers import IntegrationResult1D
from xrd_tools.core.provenance import read_provenance
from xrd_tools.reduction import (
    Frame,
    Integration1DPlan,
    NexusSink,
    ReductionPlan,
    Scan,
    run_reduction,
)
from xrd_tools.reduction.provenance_config import build_reduction_config
import xrd_tools.reduction.core as reduction_core


def _r1d(value: float = 1.0) -> IntegrationResult1D:
    return IntegrationResult1D(
        radial=np.array([0.0, 1.0]),
        intensity=np.array([value, value + 1.0]),
        sigma=np.array([0.1, 0.2]),
        unit="q_A^-1",
    )


def test_build_reduction_config_preserves_gui_scan_shape() -> None:
    class Geometry:
        preset = "psic"

        def to_json(self) -> str:
            return '{"rot1":{"source_motor":"eta"}}'

        def all_referenced_motors(self):
            return ["eta", "del"]

    scan = SimpleNamespace(
        bai_1d_args={"numpoints": 3000, "unit": "q_A^-1"},
        bai_2d_args={"npt_rad": 500, "npt_azim": 500},
        gi_config={"gi_mode_1d": "q_total", "gi_mode_2d": "qip_qoop"},
        gi_freeze_diagnostic="GI: output grid set from the first frames",
        geometry=Geometry(),
        raw_files=["raw_0000.tif"],
        meta_file="scan.spec",
    )

    config, inputs = build_reduction_config(scan)

    assert config == {
        "bai_1d_args": {"numpoints": 3000, "unit": "q_A^-1"},
        "bai_2d_args": {"npt_rad": 500, "npt_azim": 500},
        "gi_config": {"gi_mode_1d": "q_total", "gi_mode_2d": "qip_qoop"},
        "gi_freeze_diagnostic": "GI: output grid set from the first frames",
        "geometry": {
            "convention": "psic",
            "mapping_json": '{"rot1":{"source_motor":"eta"}}',
            "motor_sources": {"eta": "eta", "del": "del"},
        },
    }
    assert inputs == {"raw_files": ["raw_0000.tif"], "meta_file": "scan.spec"}


def test_headless_nexus_sink_writes_reduction_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        reduction_core,
        "integrate_1d",
        lambda image, ai, **kwargs: _r1d(float(np.sum(image))),
    )
    raw0 = tmp_path / "raw_0000.tif"
    raw1 = tmp_path / "raw_0001.tif"
    raw0.write_bytes(b"raw pointer target 0")
    raw1.write_bytes(b"raw pointer target 1")
    out = tmp_path / "headless.nxs"
    plan = ReductionPlan(
        integration_1d=Integration1DPlan(
            npt=2,
            unit="q_A^-1",
            method="csr",
            radial_range=(0.0, 1.0),
            monitor_key="i0",
        ),
        integration_2d=None,
    )
    scan = Scan(
        "headless",
        [
            Frame(
                0,
                image=np.ones((2, 2)),
                metadata={"i0": 2.0},
                source_path=raw0,
            ),
            Frame(
                1,
                image=np.full((2, 2), 2.0),
                metadata={"i0": 4.0},
                source_path=raw1,
            ),
        ],
        integrator=object(),
    )

    result = run_reduction(plan, scan, NexusSink(out, overwrite=True))

    assert result.n_processed == 2
    with h5py.File(out, "r") as h5:
        assert h5["entry/reduction"].attrs["NX_class"] in ("NXprocess", b"NXprocess")
        assert "entry/reduction/program" in h5
        assert "entry/reduction/config/bai_1d_args" in h5
        assert "entry/reduction/inputs/raw_files" in h5

    provenance = read_provenance(out)
    assert provenance["program"] == "xrd-tools"
    assert provenance["config"]["bai_1d_args"]["npt"] == 2
    assert provenance["config"]["bai_1d_args"]["radial_range"] == [0.0, 1.0]
    assert provenance["config"]["bai_1d_args"]["monitor"] == "i0"
    assert provenance["config"]["bai_2d_args"] == {}
    assert provenance["config"]["gi"] is False
    assert provenance["inputs"]["raw_files"] == [str(raw0), str(raw1)]


class _ExplodingFrames:
    """Frame series that raises if anything iterates it.

    Mirrors the GUI ``LiveFrameSeries`` failure mode: walking it triggers a
    per-frame disk read under ``file_lock``.  Used here as an observable
    sentinel to prove whether ``build_reduction_config`` touches the series.
    """

    def __iter__(self):
        raise RuntimeError(
            "frame series walked for raw-input enumeration on the display path"
        )


def test_display_snapshot_skips_frame_walk() -> None:
    # A scan with no cheap raw_files / metadata source: the ONLY way to build
    # ``inputs`` is to walk ``frames`` -- which is exactly the GUI-thread freeze
    # (frame_series.__getitem__ disk-read under file_lock, once per frame).
    scan = SimpleNamespace(
        bai_1d_args={"numpoints": 3000, "unit": "q_A^-1"},
        bai_2d_args={"npt_rad": 500},
        gi=False,
        frames=_ExplodingFrames(),
    )

    # The display-provenance path (include_inputs=False) must NOT touch frames.
    config, inputs = build_reduction_config(scan, include_inputs=False)
    assert inputs == {}
    assert config["bai_1d_args"] == {"numpoints": 3000, "unit": "q_A^-1"}
    assert config["bai_2d_args"] == {"npt_rad": 500}

    # Guard: the default (authoritative) path DOES walk the series, so the skip
    # above is a real avoidance of the disk-walk, not a vacuous no-op.
    import pytest

    with pytest.raises(RuntimeError):
        build_reduction_config(scan)


def test_build_reduction_config_import_is_qt_xdart_pure() -> None:
    root = os.fspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    src = os.path.join(root, "src")
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        src
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src, env["PYTHONPATH"]])
    )
    code = """
import importlib
import sys

importlib.import_module("xrd_tools.reduction.provenance_config")
forbidden = sorted(
    name for name in sys.modules
    if name == "xdart"
    or name.startswith("xdart.")
    or name.startswith(("PySide", "PyQt", "qtpy"))
)
if forbidden:
    raise SystemExit("\\n".join(forbidden))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
