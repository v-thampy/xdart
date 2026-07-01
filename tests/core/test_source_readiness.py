from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

from xrd_tools.core.scan import SourceCapabilities, SourceKind, SourceSpec
from xrd_tools.sources.readiness import (
    capabilities_for_processed,
    describe_source_readiness,
)


class _FakeSource:
    kind = SourceKind.MEMORY

    def __init__(
        self,
        *,
        frame_indices=(0,),
        image=None,
        metadata=None,
        capabilities=None,
        motors=None,
        fail_load=False,
        kind=None,
    ):
        self._frame_indices = list(frame_indices)
        self._image = np.ones((2, 3)) if image is None else image
        self._metadata = dict(metadata or {})
        self._motors = dict(motors or {})
        self._fail_load = fail_load
        self.capabilities = capabilities or SourceCapabilities()
        if kind is not None:
            self.kind = kind

    @property
    def frame_indices(self):
        return list(self._frame_indices)

    @property
    def motors(self):
        return dict(self._motors)

    def load_frame(self, index):
        if self._fail_load:
            raise RuntimeError("not yet available")
        return self._image

    def metadata_for(self, index):
        return dict(self._metadata)


def test_describe_source_readiness_projects_source_caps_and_probe():
    source = _FakeSource(
        metadata={"energy_keV": 12.0, "psi": 0.2},
        motors={"omega": np.array([1.0])},
        capabilities=SourceCapabilities(
            has_metadata=True,
            has_geometry=True,
            has_raw_references=True,
        ),
    )

    caps = describe_source_readiness(source)

    assert caps.has_frames is True
    assert caps.has_raw is True
    assert caps.raw_reachable is True
    assert caps.has_metadata is True
    assert caps.has_motors is True
    assert caps.has_energy is True
    assert caps.has_geometry is True
    assert caps.has_psi_metadata is True


def test_describe_source_readiness_preserves_true_live_escape_hatch():
    source = _FakeSource(
        frame_indices=(),
        fail_load=True,
        kind=SourceKind.LIVE,
        capabilities=SourceCapabilities(
            is_streaming=True,
            has_metadata=True,
            has_geometry=True,
        ),
    )

    caps = describe_source_readiness(source, probe=True)

    assert caps.has_frames is True
    assert caps.has_raw is True
    assert caps.raw_reachable is True


def test_describe_source_readiness_opens_source_spec():
    caps = describe_source_readiness(
        SourceSpec("live", SourceKind.LIVE),
        probe=True,
    )

    assert caps.has_frames is True
    assert caps.has_raw is True
    assert caps.raw_reachable is True


def test_capabilities_for_processed_consumes_metadata_capabilities():
    caps = capabilities_for_processed(
        {
            "has_1d": True,
            "capabilities": [
                "frames_record",
                "source_base",
                "two_d_kind",
                "rsm",
            ],
            "frames": np.arange(3),
        }
    )

    assert caps.has_1d is True
    assert caps.has_2d is True
    assert caps.has_raw is True
    assert caps.raw_reachable is True
    assert caps.has_scan_metadata is True
    assert caps.has_rsm is True


def test_sources_readiness_import_and_processed_caps_are_pure():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(root / "src") + (os.pathsep + existing if existing else "")
    code = textwrap.dedent(
        """
        import sys

        from xrd_tools.sources.readiness import capabilities_for_processed

        caps = capabilities_for_processed({
            "capabilities": ["frames_record", "rsm"],
            "has_1d": True,
        })
        assert caps.has_1d and caps.has_raw and caps.has_rsm

        forbidden = (
            "xdart",
            "PySide6",
            "PySide2",
            "PyQt5",
            "PyQt6",
            "pyqtgraph",
            "pyFAI",
            "h5py",
            "fabio",
        )
        bad = sorted(
            root
            for root in forbidden
            if root in sys.modules
            or any(name == root or name.startswith(root + ".") for name in sys.modules)
        )
        if bad:
            print(",".join(bad))
            raise SystemExit(1)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
