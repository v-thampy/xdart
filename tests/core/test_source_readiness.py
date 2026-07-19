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


def test_describe_source_readiness_nonexistent_path_has_no_frames(tmp_path):
    """H18 hazard-1 fix (H5 finding 1): ``open_source`` builds an
    ImageFileSource around a typo'd path without stat-ing it, which used to
    yield a phantom ``frame_indices == [0]`` (has_frames/has_raw=True) for a
    NONEXISTENT file.  The describe gate now stats non-live local URIs first,
    so a run gate consuming the core answer can never enable Run on nothing."""
    missing = tmp_path / "nope" / "gone_0001.tif"   # parent dir absent too

    caps = describe_source_readiness(str(missing))

    assert caps.has_frames is False
    assert caps.has_raw is False
    assert caps.raw_reachable is False

    # SourceSpec spelling of the same path is gated identically.
    caps = describe_source_readiness(SourceSpec(str(missing)))
    assert caps.has_frames is False


def test_describe_source_readiness_live_spec_missing_path_keeps_escape_hatch(tmp_path):
    """The stat gate must NOT demote a configured LIVE location: a live
    acquisition's file may legitimately not exist yet at gate time."""
    caps = describe_source_readiness(
        SourceSpec(str(tmp_path / "future_master.h5"), SourceKind.LIVE))

    assert caps.has_frames is True
    assert caps.has_raw is True
    assert caps.raw_reachable is True


def test_describe_source_readiness_empty_location_is_never_ready():
    """H18 label rule (from the H5 live-no-label row, adopted headlessly): an
    explicitly EMPTY location is never ready — even for SourceKind.LIVE, where
    the escape hatch used to trust the LIVE kind alone.  The escape hatch
    governs a *configured* live source; with no location there is nothing to
    run."""
    for value in ("", "   ", SourceSpec(""), SourceSpec("", SourceKind.LIVE)):
        caps = describe_source_readiness(value)
        assert caps.has_frames is False, value
        assert caps.has_raw is False, value
        assert caps.raw_reachable is False, value


def test_describe_source_readiness_remote_uri_passes_the_stat_gate(monkeypatch):
    """A ``scheme://`` URI (the Tiled shape) is not stat-able locally and must
    pass through the nonexistent-path gate untouched."""
    from xrd_tools.sources import readiness as R

    source = _FakeSource()
    monkeypatch.setattr(R, "_open_source", lambda _v: source)

    caps = R.describe_source_readiness("https://tiled.example/api/v1/scan")

    assert caps.has_frames is True
    assert caps.raw_reachable is True


def test_capabilities_for_processed_raw_reachable_probe_override():
    """H18 hazard-2 seam (H5 finding 2): the frames_record capability only
    proves the record EXISTS; for an orphaned record the default mirror
    overstates raw_reachable.  Callers inject the frame-0 probe answer
    (describe_source_readiness(path).raw_reachable); None keeps the pure
    no-reopen mirror."""
    metadata = {"capabilities": ["frames_record"], "has_1d": True}

    mirrored = capabilities_for_processed(metadata)
    assert mirrored.has_raw is True and mirrored.raw_reachable is True

    probed_bad = capabilities_for_processed(metadata, raw_reachable=False)
    assert probed_bad.has_raw is True          # the record still exists...
    assert probed_bad.raw_reachable is False   # ...but its raw master is gone

    probed_ok = capabilities_for_processed(metadata, raw_reachable=True)
    assert probed_ok.raw_reachable is True

    # No record at all: the probe cannot conjure raw out of nothing.
    empty = capabilities_for_processed({}, raw_reachable=True)
    assert empty.has_raw is False and empty.raw_reachable is False


def test_live_spec_keeps_escape_hatch_when_open_fails(monkeypatch):
    """Fallback path (open_source returns None): a LIVE spec must keep the
    true-live escape hatch (raw_reachable=True), not collapse to all-False — a
    live acquisition may legitimately have no frame 0 yet at gate time."""
    from types import SimpleNamespace
    from xrd_tools.sources import readiness as R

    monkeypatch.setattr(R, "_open_source", lambda _v: None)   # force the fallback
    caps = R.describe_source_readiness(SimpleNamespace(kind=R.SourceKind.LIVE))
    assert caps.raw_reachable is True
    assert caps.has_frames is True
    assert caps.has_raw is True
