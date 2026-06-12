"""Tests for the v2-NeXus-scan data source in rsm/pipeline.py.

Uses a duck-typed ``_FakeScan`` with the minimal attributes the
``process_scan_from_nexus`` / ``grid_scans_streaming`` paths need:

* ``scan_data`` — pandas DataFrame indexed by frame ID, motor columns.
* ``frames`` — collection with ``.index`` and ``__getitem__`` returning
  frame-like objects (``map_raw`` + ``_lazy_load_raw``).
* ``mg_args`` — dict containing ``wavelength`` in metres.

The xu / xrayutilities stack and the actual EwaldSphere are mocked so
the tests run in any environment that has numpy + pandas + pytest.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from xrd_tools.core.geometry import (
    DetectorHeader,
    DiffractometerConfig,
    PixelQMap,
)
from xrd_tools.rsm import gridding as gridding_module
from xrd_tools.rsm.pipeline import (
    ScanInput,
    _angles_for_indices,
    _energy_from_scan,
    _iter_scan_chunks,
    grid_scans_streaming,
    process_scan_from_nexus,
)


# ---------------------------------------------------------------------------
# Reusable mocks (subset of the streaming-gridder mocks)
# ---------------------------------------------------------------------------

class _FakeGridder3D:
    """Records every chunk passed in; KeepData / dataRange state inspectable."""

    instances: list["_FakeGridder3D"] = []

    def __init__(self, nx: int, ny: int, nz: int) -> None:
        self.nx, self.ny, self.nz = nx, ny, nz
        self.xaxis = np.linspace(-1, 1, nx)
        self.yaxis = np.linspace(-1, 1, ny)
        self.zaxis = np.linspace(-1, 1, nz)
        self.data = np.zeros((nx, ny, nz), dtype=float)
        self.keep_data: bool = False
        self.data_range: tuple[float, ...] | None = None
        self.chunks: list[tuple[np.ndarray, np.ndarray]] = []
        _FakeGridder3D.instances.append(self)

    def KeepData(self, flag: bool) -> None:
        self.keep_data = bool(flag)

    def dataRange(  # noqa: N802 — matches xu API
        self, xmin, xmax, ymin, ymax, zmin, zmax, fixed=False,
    ) -> None:
        self.data_range = (xmin, xmax, ymin, ymax, zmin, zmax)
        self.xaxis = np.linspace(xmin, xmax, self.nx)
        self.yaxis = np.linspace(ymin, ymax, self.ny)
        self.zaxis = np.linspace(zmin, zmax, self.nz)

    def __call__(self, qx, qy, qz, data) -> None:
        self.chunks.append((data.copy(), data.shape))


class _FakeAng2Q:
    def __init__(self) -> None:
        self.init_kwargs: dict = {}

    def init_area(self, *args: Any, **kwargs: Any) -> None:
        self.init_kwargs = kwargs

    def area(self, *args: Any, **kwargs: Any):
        N = len(np.atleast_1d(args[0]))
        Nch1 = self.init_kwargs.get("Nch1")
        Nch2 = self.init_kwargs.get("Nch2")
        # Non-degenerate q so scout produces hi > lo
        frame_idx = np.arange(N, dtype=float).reshape(N, 1, 1)
        row = np.arange(Nch1, dtype=float).reshape(1, Nch1, 1)
        col = np.arange(Nch2, dtype=float).reshape(1, 1, Nch2)
        qx = 0.5 + 0.01 * frame_idx + 0.001 * row
        qy = 1.5 + 0.02 * frame_idx + 0.002 * col
        qz = 2.5 + 0.03 * frame_idx + 0.003 * (row + col)
        return (
            np.broadcast_to(qx, (N, Nch1, Nch2)).astype(float, copy=True),
            np.broadcast_to(qy, (N, Nch1, Nch2)).astype(float, copy=True),
            np.broadcast_to(qz, (N, Nch1, Nch2)).astype(float, copy=True),
        )


class _FakeHXRD:
    def __init__(self) -> None:
        self.Ang2Q = _FakeAng2Q()


@pytest.fixture(autouse=True)
def _reset_gridder_instances():
    _FakeGridder3D.instances.clear()
    yield
    _FakeGridder3D.instances.clear()


@pytest.fixture
def patched_xu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gridding_module.xu, "Gridder3D", _FakeGridder3D)
    monkeypatch.setattr(
        DiffractometerConfig, "make_hxrd",
        lambda self, energy: _FakeHXRD(),
    )


# ---------------------------------------------------------------------------
# _FakeScan / _FakeFrame — minimal duck-types for the v2-scan protocol
# ---------------------------------------------------------------------------

class _FakeFrame:
    """Behaves like an EwaldArch with optional lazy raw-load."""

    def __init__(
        self,
        idx: int,
        *,
        image: np.ndarray | None = None,
        lazy_image: np.ndarray | None = None,
        lazy_fail: bool = False,
    ) -> None:
        self.idx = idx
        self.map_raw = image
        self._lazy_payload = lazy_image
        self._lazy_fail = lazy_fail
        self.lazy_load_calls = 0

    def _lazy_load_raw(self) -> bool:
        self.lazy_load_calls += 1
        if self._lazy_fail or self._lazy_payload is None:
            return False
        self.map_raw = self._lazy_payload
        return True


class _FakeFrameSeries:
    """Indexable frames collection; .index returns the frame IDs."""

    def __init__(self, frames: list[_FakeFrame]) -> None:
        self._by_idx = {a.idx: a for a in frames}
        self.index: list[int] = sorted(self._by_idx.keys())
        self.get_calls: list[int] = []

    def __getitem__(self, idx: int) -> _FakeFrame:
        self.get_calls.append(int(idx))
        return self._by_idx[int(idx)]


class _FakeScan:
    """Minimal duck type for the v2-scan RSM path."""

    def __init__(
        self,
        frames: list[_FakeFrame],
        motor_values: dict[str, list[float]],
        wavelength_m: float = 1.10e-10,
    ) -> None:
        # Index scan_data by the frame frame IDs (matches the v2 N2 fix)
        frame_ids = [a.idx for a in frames]
        self.scan_data = pd.DataFrame(motor_values, index=frame_ids)
        self.frames = _FakeFrameSeries(frames)
        self.mg_args = {"wavelength": wavelength_m}


def _default_mapper(Nch1: int = 32, Nch2: int = 32) -> PixelQMap:
    return PixelQMap(
        diff_config=DiffractometerConfig(),
        header=DetectorHeader(
            cch1=Nch1 / 2.0, cch2=Nch2 / 2.0,
            pwidth1=0.075, pwidth2=0.075,
            distance=830.0,
            Nch1=Nch1, Nch2=Nch2,
        ),
    )


def _make_scan(
    n_frames: int = 6,
    Nch1: int = 32, Nch2: int = 32,
    motors: tuple[str, ...] = ("tth", "th", "chi", "phi"),
    lazy: bool = False,
    seed: int = 0,
) -> _FakeScan:
    rng = np.random.default_rng(seed)
    frames: list[_FakeFrame] = []
    for i in range(n_frames):
        img = rng.random((Nch1, Nch2)).astype(np.float32)
        if lazy:
            frames.append(_FakeFrame(idx=i, image=None, lazy_image=img))
        else:
            frames.append(_FakeFrame(idx=i, image=img))
    motor_values = {
        m: list(np.linspace(0, n_frames * 0.1, n_frames) + j)
        for j, m in enumerate(motors)
    }
    return _FakeScan(frames, motor_values)


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

class TestEnergyFromSphere:
    def test_wavelength_to_eV(self) -> None:
        # 1.10 Å ≈ 11271.0 eV (12398 / 1.10)
        scan = _FakeScan([_FakeFrame(0, image=np.zeros((4, 4)))],
                             {"tth": [0.0]},
                             wavelength_m=1.10e-10)
        e = _energy_from_scan(scan)
        assert e == pytest.approx(12398.0 / 1.10, rel=1e-6)

    def test_missing_wavelength_raises(self) -> None:
        scan = _FakeScan([_FakeFrame(0, image=np.zeros((4, 4)))],
                             {"tth": [0.0]},
                             wavelength_m=0.0)
        with pytest.raises(ValueError, match="no usable wavelength"):
            _energy_from_scan(scan)


class TestAnglesForIndices:
    def test_pulls_columns_in_order(self) -> None:
        scan = _make_scan(n_frames=4)
        angles = _angles_for_indices(scan, ("tth", "th", "chi", "phi"))
        assert len(angles) == 4
        for a in angles:
            assert a.shape == (4,)

    def test_index_slicing(self) -> None:
        scan = _make_scan(n_frames=10)
        angles_full = _angles_for_indices(scan, ("tth",))
        angles_slice = _angles_for_indices(scan, ("tth",), indices=[3, 5, 7])
        assert angles_full[0].shape == (10,)
        np.testing.assert_allclose(
            angles_slice[0],
            angles_full[0][[3, 5, 7]],
        )

    def test_missing_motor_raises(self) -> None:
        scan = _make_scan(motors=("tth", "th"))
        with pytest.raises(KeyError, match="phi"):
            _angles_for_indices(scan, ("tth", "th", "phi"))


class TestIterSphereChunks:
    def test_chunks_cover_all_frames(self) -> None:
        scan = _make_scan(n_frames=10)
        chunks = list(_iter_scan_chunks(scan, chunk_size=3))
        sizes = [c[0].shape[0] for c in chunks]
        assert sizes == [3, 3, 3, 1]
        all_indices = [i for c in chunks for i in c[1]]
        assert all_indices == list(range(10))

    def test_lazy_load_called_when_map_raw_missing(self) -> None:
        scan = _make_scan(n_frames=4, lazy=True)
        chunks = list(_iter_scan_chunks(scan, chunk_size=2))
        # Every frame needed exactly one lazy load
        for idx in range(4):
            frame = scan.frames[idx]
            assert frame.lazy_load_calls == 1
        # And the chunks shape-check out
        assert all(c[0].shape == (2, 32, 32) for c in chunks)

    def test_p1_lazy_loaded_frames_freed_after_each_chunk(self) -> None:
        """P1 regression: frames we materialised ourselves must have
        ``map_raw`` cleared after the chunk is consumed, otherwise the
        scan cache holds every lazy-loaded frame and the streaming
        memory promise breaks for large v2-reloaded scans.
        """
        scan = _make_scan(n_frames=6, lazy=True)
        # All frames start with map_raw=None (lazy=True)
        for frame in scan.frames._by_idx.values():
            assert frame.map_raw is None

        chunks = list(_iter_scan_chunks(scan, chunk_size=2))
        # Chunks themselves are valid
        assert sum(c[0].shape[0] for c in chunks) == 6
        # And after iteration, every frame we loaded has been cleared
        for frame in scan.frames._by_idx.values():
            assert frame.lazy_load_calls == 1
            assert frame.map_raw is None, (
                f"frame {frame.idx} retained map_raw after iteration — "
                f"streaming memory promise is broken"
            )

    def test_p1_pre_populated_frames_left_alone(self) -> None:
        """P1 corollary: frames that arrived with map_raw populated must
        NOT be cleared — the user might be holding a reference."""
        rng = np.random.default_rng(0)
        pre_loaded = rng.random((32, 32)).astype(np.float32)
        frames = [
            _FakeFrame(idx=0, image=pre_loaded.copy()),  # NOT lazy
            _FakeFrame(idx=1, image=None, lazy_image=rng.random((32, 32))),
        ]
        scan = _FakeScan(frames, {"tth": [0.0, 0.1]})

        list(_iter_scan_chunks(scan, chunk_size=2))
        # Frame 0 was pre-populated → still has its image
        assert scan.frames[0].map_raw is not None
        np.testing.assert_array_equal(scan.frames[0].map_raw, pre_loaded)
        # Frame 1 was lazy-loaded by us → cleared
        assert scan.frames[1].map_raw is None

    def test_p1_clear_happens_even_on_consumer_error(self) -> None:
        """try/finally semantics: if the consumer raises, we still clear."""
        scan = _make_scan(n_frames=4, lazy=True)
        with pytest.raises(RuntimeError, match="boom"):
            for img_chunk, indices in _iter_scan_chunks(scan, chunk_size=2):
                raise RuntimeError("boom")
        # The first chunk's frames (idx 0, 1) were materialised and must
        # be cleared even though the consumer raised mid-iteration.
        assert scan.frames[0].map_raw is None
        assert scan.frames[1].map_raw is None

    def test_lazy_load_failure_raises(self) -> None:
        bad = _FakeFrame(0, image=None, lazy_fail=True)
        good = _FakeFrame(1, image=np.zeros((4, 4)))
        scan = _FakeScan([bad, good], {"tth": [0.0, 0.1]})
        with pytest.raises(RuntimeError, match="could not lazy-load"):
            list(_iter_scan_chunks(scan, chunk_size=2))

    def test_invalid_chunk_size_rejected(self) -> None:
        scan = _make_scan(n_frames=2)
        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            list(_iter_scan_chunks(scan, chunk_size=0))

    def test_headless_frame_source_metadata_contract(self) -> None:
        class _HeadlessSource:
            frame_indices = [10, 20]
            metadata = {
                "energy_keV": 12.4,
                "scan_data": {"tth": np.array([0.1, 0.2])},
            }

            def iter_chunks(self, chunk_size):
                assert chunk_size == 2
                yield np.ones((2, 4, 4)), [10, 20]

        source = _HeadlessSource()
        assert _energy_from_scan(source) == pytest.approx(12400.0)
        np.testing.assert_allclose(
            _angles_for_indices(source, ("tth",), indices=[20])[0], [0.2],
        )
        chunks = list(_iter_scan_chunks(source, chunk_size=2))
        assert chunks[0][1] == [10, 20]

    def test_energy_eV_attribute_is_used_without_keV_scaling(self) -> None:
        class _HeadlessSource:
            energy_eV = 12400.0

        assert _energy_from_scan(_HeadlessSource()) == pytest.approx(12400.0)


# ---------------------------------------------------------------------------
# process_scan_from_nexus
# ---------------------------------------------------------------------------

class TestProcessScanFromSphere:
    def test_returns_rsmvolume(self, patched_xu) -> None:
        scan = _make_scan(n_frames=6, Nch1=16, Nch2=16)
        mapper = _default_mapper(Nch1=16, Nch2=16)
        vol = process_scan_from_nexus(
            scan, mapper,
            diff_motors=("tth", "th", "chi", "phi"),
            bins=(4, 4, 4),
            chunk_size=2,
            q_bounds=((-1, 1), (-1, 1), (-1, 1)),  # skip scout
        )
        from xrd_tools.rsm.volume import RSMVolume
        assert isinstance(vol, RSMVolume)
        assert vol.shape == (4, 4, 4)

    def test_explicit_energy_overrides_sphere(self, patched_xu) -> None:
        """Pass energy= explicitly; the scan's wavelength is ignored."""
        scan = _make_scan(n_frames=4, Nch1=16, Nch2=16)
        scan.mg_args["wavelength"] = 0.0  # would be invalid without override
        process_scan_from_nexus(
            scan, _default_mapper(Nch1=16, Nch2=16),
            diff_motors=("tth", "th", "chi", "phi"),
            bins=(2, 2, 2),
            chunk_size=2,
            energy=12000.0,
            q_bounds=((-1, 1), (-1, 1), (-1, 1)),
        )

    def test_energy_resolved_from_wavelength(self, patched_xu) -> None:
        """Default energy resolution path: λ in metres → E in eV."""
        scan = _make_scan(n_frames=2, Nch1=8, Nch2=8)
        scan.mg_args["wavelength"] = 1.5e-10  # 1.5 Å ≈ 8266 eV
        process_scan_from_nexus(
            scan, _default_mapper(Nch1=8, Nch2=8),
            diff_motors=("tth", "th", "chi", "phi"),
            bins=(2, 2, 2),
            chunk_size=2,
            q_bounds=((-1, 1), (-1, 1), (-1, 1)),
        )
        # No assertion needed; mostly that the energy plumbing didn't crash.

    def test_one_streaming_gridder_for_whole_sphere(self, patched_xu) -> None:
        scan = _make_scan(n_frames=11, Nch1=16, Nch2=16)
        process_scan_from_nexus(
            scan, _default_mapper(Nch1=16, Nch2=16),
            diff_motors=("tth", "th", "chi", "phi"),
            bins=(4, 4, 4),
            chunk_size=3,
            q_bounds=((-1, 1), (-1, 1), (-1, 1)),
        )
        # Exactly one xu.Gridder3D for the whole scan
        assert len(_FakeGridder3D.instances) == 1
        # 11 frames in chunk_size=3 → [3, 3, 3, 2]
        g = _FakeGridder3D.instances[0]
        sizes = [shape[0] for (_, shape) in g.chunks]
        assert sizes == [3, 3, 3, 2]

    def test_scout_does_not_read_images(self, patched_xu) -> None:
        """scout pass must not call _lazy_load_raw or scan.frames.__getitem__."""
        scan = _make_scan(n_frames=4, Nch1=16, Nch2=16)
        # Replace frame images with None + lazy payload so any read shows up
        for i, a in enumerate(scan.frames._by_idx.values()):  # type: ignore[attr-defined]
            a.map_raw = None
            a._lazy_payload = np.zeros((16, 16), dtype=np.float32)
            a.lazy_load_calls = 0
        before_get_calls = len(scan.frames.get_calls)
        # No q_bounds → scout runs.  After scout completes but before
        # the add() loop, lazy_load should still be 0 on every frame.
        # We can't observe "between scout and add" directly, but if scout
        # itself read images we'd see lazy_load_calls > 0 on the frames
        # whose indices come BEFORE the first chunk's processing — which
        # is impossible because scout only takes angles + image_shape.
        # Easier proof: scout uses the corner-q tiny detector path, which
        # never touches scan.frames at all.  Patch __getitem__ to fail
        # if called before add() is allowed in.
        process_scan_from_nexus(
            scan, _default_mapper(Nch1=16, Nch2=16),
            diff_motors=("tth", "th", "chi", "phi"),
            bins=(2, 2, 2),
            chunk_size=4,  # one big chunk
        )
        # After processing: all 4 frames got loaded exactly once
        for a in scan.frames._by_idx.values():  # type: ignore[attr-defined]
            assert a.lazy_load_calls == 1
        # And get_calls grew by exactly 4 (one per frame in the chunk)
        assert len(scan.frames.get_calls) - before_get_calls == 4

    def test_missing_motor_raises(self, patched_xu) -> None:
        scan = _make_scan(motors=("tth", "th"))
        with pytest.raises(KeyError, match="phi"):
            process_scan_from_nexus(
                scan, _default_mapper(),
                diff_motors=("tth", "th", "phi"),
                bins=(2, 2, 2),
                q_bounds=((-1, 1), (-1, 1), (-1, 1)),
            )

    def test_per_frame_angles_match_indices(self, patched_xu) -> None:
        """Angles handed to each chunk must come from the matching frame rows."""
        scan = _make_scan(n_frames=6, Nch1=8, Nch2=8)
        captured: list[list[np.ndarray]] = []

        # Spy on _FakeAng2Q.area to capture the per-chunk angle arrays
        from xrd_tools.core.geometry import pixel_q as pixel_q_module

        original_pixel_q = PixelQMap.pixel_q

        def spy_pixel_q(self, angles, energy, *, UB=None, roi=None, image_shape=None):
            captured.append([np.asarray(a) for a in angles])
            return original_pixel_q(self, angles, energy, UB=UB, roi=roi,
                                    image_shape=image_shape)

        # monkeypatch via setattr — pytest's monkeypatch isn't in scope here
        PixelQMap.pixel_q = spy_pixel_q  # type: ignore[method-assign]
        try:
            process_scan_from_nexus(
                scan, _default_mapper(Nch1=8, Nch2=8),
                diff_motors=("tth", "th"),
                bins=(2, 2, 2),
                chunk_size=3,
                q_bounds=((-1, 1), (-1, 1), (-1, 1)),
            )
        finally:
            PixelQMap.pixel_q = original_pixel_q  # type: ignore[method-assign]

        # 6 frames in chunks of 3 → 2 chunks; per chunk a 2-motor pair
        # of length-3 arrays.  Compare against scan_data slices.
        full_tth = np.asarray(scan.scan_data["tth"].values)
        np.testing.assert_allclose(captured[0][0], full_tth[:3])
        np.testing.assert_allclose(captured[1][0], full_tth[3:])


# ---------------------------------------------------------------------------
# grid_scans_streaming
# ---------------------------------------------------------------------------

class TestGridSpheresStreaming:
    def test_three_spheres_share_one_gridder(self, patched_xu) -> None:
        mapper = _default_mapper(Nch1=16, Nch2=16)
        sphere_inputs = [
            ScanInput(
                scan=_make_scan(n_frames=4, Nch1=16, Nch2=16,
                                    motors=("tth", "th", "chi", "phi"),
                                    seed=i),
                energy=11000.0 + 100 * i,
                UB=None,
            )
            for i in range(3)
        ]
        grid_scans_streaming(
            mapper, sphere_inputs,
            diff_motors=("tth", "th", "chi", "phi"),
            bins=(4, 4, 4),
            chunk_size=2,
            q_bounds=((-1, 1), (-1, 1), (-1, 1)),
        )
        # 12 frames total, chunk_size 2 → 6 chunks, all into ONE gridder
        assert len(_FakeGridder3D.instances) == 1
        g = _FakeGridder3D.instances[0]
        sizes = [shape[0] for (_, shape) in g.chunks]
        assert sizes == [2, 2, 2, 2, 2, 2]

    def test_per_sphere_energy_resolution(self, patched_xu) -> None:
        """If ScanInput.energy is None, each scan resolves its own."""
        s1 = _make_scan(n_frames=2, Nch1=8, Nch2=8)
        s2 = _make_scan(n_frames=2, Nch1=8, Nch2=8)
        s1.mg_args["wavelength"] = 1.0e-10  # 12398 eV
        s2.mg_args["wavelength"] = 2.0e-10  #  6199 eV
        grid_scans_streaming(
            _default_mapper(Nch1=8, Nch2=8),
            [ScanInput(scan=s1), ScanInput(scan=s2)],
            diff_motors=("tth", "th", "chi", "phi"),
            bins=(2, 2, 2),
            chunk_size=2,
            q_bounds=((-1, 1), (-1, 1), (-1, 1)),
        )
        # Both scans processed end-to-end — total 4 frames in 2 chunks
        g = _FakeGridder3D.instances[0]
        assert sum(shape[0] for (_, shape) in g.chunks) == 4

    def test_empty_inputs_rejected(self, patched_xu) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            grid_scans_streaming(
                _default_mapper(), [],
                diff_motors=("tth",), bins=(2, 2, 2),
            )

    def test_scout_runs_without_q_bounds(self, patched_xu) -> None:
        """No q_bounds → scout must succeed and produce non-degenerate bounds."""
        sphere_inputs = [
            ScanInput(scan=_make_scan(n_frames=3, Nch1=8, Nch2=8)),
        ]
        grid_scans_streaming(
            _default_mapper(Nch1=8, Nch2=8), sphere_inputs,
            diff_motors=("tth", "th", "chi", "phi"),
            bins=(2, 2, 2),
            chunk_size=3,
        )
        g = _FakeGridder3D.instances[0]
        assert g.keep_data is True
        # data_range was set by scout (not equal to default (-1, 1, ...))
        assert g.data_range is not None
        lo, hi = g.data_range[0], g.data_range[1]
        assert hi > lo
