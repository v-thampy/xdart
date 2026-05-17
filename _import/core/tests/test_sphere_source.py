"""Tests for the v2-NeXus-sphere data source in rsm/pipeline.py.

Uses a duck-typed ``_FakeSphere`` with the minimal attributes the
``process_scan_from_sphere`` / ``grid_spheres_streaming`` paths need:

* ``scan_data`` — pandas DataFrame indexed by frame ID, motor columns.
* ``arches`` — collection with ``.index`` and ``__getitem__`` returning
  arch-like objects (``map_raw`` + ``_lazy_load_raw``).
* ``mg_args`` — dict containing ``wavelength`` in metres.

The xu / xrayutilities stack and the actual EwaldSphere are mocked so
the tests run in any environment that has numpy + pandas + pytest.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from ssrl_xrd_tools.core.geometry import (
    DetectorHeader,
    DiffractometerConfig,
    PixelQMap,
)
from ssrl_xrd_tools.rsm import gridding as gridding_module
from ssrl_xrd_tools.rsm.pipeline import (
    SphereInput,
    _angles_for_indices,
    _energy_from_sphere,
    _iter_sphere_chunks,
    grid_spheres_streaming,
    process_scan_from_sphere,
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
# _FakeSphere / _FakeArch — minimal duck-types for the v2-sphere protocol
# ---------------------------------------------------------------------------

class _FakeArch:
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


class _FakeArchSeries:
    """Indexable arches collection; .index returns the frame IDs."""

    def __init__(self, arches: list[_FakeArch]) -> None:
        self._by_idx = {a.idx: a for a in arches}
        self.index: list[int] = sorted(self._by_idx.keys())
        self.get_calls: list[int] = []

    def __getitem__(self, idx: int) -> _FakeArch:
        self.get_calls.append(int(idx))
        return self._by_idx[int(idx)]


class _FakeSphere:
    """Minimal duck type for the v2-sphere RSM path."""

    def __init__(
        self,
        arches: list[_FakeArch],
        motor_values: dict[str, list[float]],
        wavelength_m: float = 1.10e-10,
    ) -> None:
        # Index scan_data by the arch frame IDs (matches the v2 N2 fix)
        frame_ids = [a.idx for a in arches]
        self.scan_data = pd.DataFrame(motor_values, index=frame_ids)
        self.arches = _FakeArchSeries(arches)
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


def _make_sphere(
    n_frames: int = 6,
    Nch1: int = 32, Nch2: int = 32,
    motors: tuple[str, ...] = ("tth", "th", "chi", "phi"),
    lazy: bool = False,
    seed: int = 0,
) -> _FakeSphere:
    rng = np.random.default_rng(seed)
    arches: list[_FakeArch] = []
    for i in range(n_frames):
        img = rng.random((Nch1, Nch2)).astype(np.float32)
        if lazy:
            arches.append(_FakeArch(idx=i, image=None, lazy_image=img))
        else:
            arches.append(_FakeArch(idx=i, image=img))
    motor_values = {
        m: list(np.linspace(0, n_frames * 0.1, n_frames) + j)
        for j, m in enumerate(motors)
    }
    return _FakeSphere(arches, motor_values)


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

class TestEnergyFromSphere:
    def test_wavelength_to_eV(self) -> None:
        # 1.10 Å ≈ 11271.0 eV (12398 / 1.10)
        sphere = _FakeSphere([_FakeArch(0, image=np.zeros((4, 4)))],
                             {"tth": [0.0]},
                             wavelength_m=1.10e-10)
        e = _energy_from_sphere(sphere)
        assert e == pytest.approx(12398.0 / 1.10, rel=1e-6)

    def test_missing_wavelength_raises(self) -> None:
        sphere = _FakeSphere([_FakeArch(0, image=np.zeros((4, 4)))],
                             {"tth": [0.0]},
                             wavelength_m=0.0)
        with pytest.raises(ValueError, match="no usable wavelength"):
            _energy_from_sphere(sphere)


class TestAnglesForIndices:
    def test_pulls_columns_in_order(self) -> None:
        sphere = _make_sphere(n_frames=4)
        angles = _angles_for_indices(sphere, ("tth", "th", "chi", "phi"))
        assert len(angles) == 4
        for a in angles:
            assert a.shape == (4,)

    def test_index_slicing(self) -> None:
        sphere = _make_sphere(n_frames=10)
        angles_full = _angles_for_indices(sphere, ("tth",))
        angles_slice = _angles_for_indices(sphere, ("tth",), indices=[3, 5, 7])
        assert angles_full[0].shape == (10,)
        np.testing.assert_allclose(
            angles_slice[0],
            angles_full[0][[3, 5, 7]],
        )

    def test_missing_motor_raises(self) -> None:
        sphere = _make_sphere(motors=("tth", "th"))
        with pytest.raises(KeyError, match="phi"):
            _angles_for_indices(sphere, ("tth", "th", "phi"))


class TestIterSphereChunks:
    def test_chunks_cover_all_frames(self) -> None:
        sphere = _make_sphere(n_frames=10)
        chunks = list(_iter_sphere_chunks(sphere, chunk_size=3))
        sizes = [c[0].shape[0] for c in chunks]
        assert sizes == [3, 3, 3, 1]
        all_indices = [i for c in chunks for i in c[1]]
        assert all_indices == list(range(10))

    def test_lazy_load_called_when_map_raw_missing(self) -> None:
        sphere = _make_sphere(n_frames=4, lazy=True)
        chunks = list(_iter_sphere_chunks(sphere, chunk_size=2))
        # Every arch needed exactly one lazy load
        for idx in range(4):
            arch = sphere.arches[idx]
            assert arch.lazy_load_calls == 1
        # And the chunks shape-check out
        assert all(c[0].shape == (2, 32, 32) for c in chunks)

    def test_p1_lazy_loaded_frames_freed_after_each_chunk(self) -> None:
        """P1 regression: arches we materialised ourselves must have
        ``map_raw`` cleared after the chunk is consumed, otherwise the
        sphere cache holds every lazy-loaded frame and the streaming
        memory promise breaks for large v2-reloaded scans.
        """
        sphere = _make_sphere(n_frames=6, lazy=True)
        # All arches start with map_raw=None (lazy=True)
        for arch in sphere.arches._by_idx.values():
            assert arch.map_raw is None

        chunks = list(_iter_sphere_chunks(sphere, chunk_size=2))
        # Chunks themselves are valid
        assert sum(c[0].shape[0] for c in chunks) == 6
        # And after iteration, every arch we loaded has been cleared
        for arch in sphere.arches._by_idx.values():
            assert arch.lazy_load_calls == 1
            assert arch.map_raw is None, (
                f"arch {arch.idx} retained map_raw after iteration — "
                f"streaming memory promise is broken"
            )

    def test_p1_pre_populated_frames_left_alone(self) -> None:
        """P1 corollary: arches that arrived with map_raw populated must
        NOT be cleared — the user might be holding a reference."""
        rng = np.random.default_rng(0)
        pre_loaded = rng.random((32, 32)).astype(np.float32)
        arches = [
            _FakeArch(idx=0, image=pre_loaded.copy()),  # NOT lazy
            _FakeArch(idx=1, image=None, lazy_image=rng.random((32, 32))),
        ]
        sphere = _FakeSphere(arches, {"tth": [0.0, 0.1]})

        list(_iter_sphere_chunks(sphere, chunk_size=2))
        # Arch 0 was pre-populated → still has its image
        assert sphere.arches[0].map_raw is not None
        np.testing.assert_array_equal(sphere.arches[0].map_raw, pre_loaded)
        # Arch 1 was lazy-loaded by us → cleared
        assert sphere.arches[1].map_raw is None

    def test_p1_clear_happens_even_on_consumer_error(self) -> None:
        """try/finally semantics: if the consumer raises, we still clear."""
        sphere = _make_sphere(n_frames=4, lazy=True)
        with pytest.raises(RuntimeError, match="boom"):
            for img_chunk, indices in _iter_sphere_chunks(sphere, chunk_size=2):
                raise RuntimeError("boom")
        # The first chunk's arches (idx 0, 1) were materialised and must
        # be cleared even though the consumer raised mid-iteration.
        assert sphere.arches[0].map_raw is None
        assert sphere.arches[1].map_raw is None

    def test_lazy_load_failure_raises(self) -> None:
        bad = _FakeArch(0, image=None, lazy_fail=True)
        good = _FakeArch(1, image=np.zeros((4, 4)))
        sphere = _FakeSphere([bad, good], {"tth": [0.0, 0.1]})
        with pytest.raises(RuntimeError, match="could not lazy-load"):
            list(_iter_sphere_chunks(sphere, chunk_size=2))

    def test_invalid_chunk_size_rejected(self) -> None:
        sphere = _make_sphere(n_frames=2)
        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            list(_iter_sphere_chunks(sphere, chunk_size=0))


# ---------------------------------------------------------------------------
# process_scan_from_sphere
# ---------------------------------------------------------------------------

class TestProcessScanFromSphere:
    def test_returns_rsmvolume(self, patched_xu) -> None:
        sphere = _make_sphere(n_frames=6, Nch1=16, Nch2=16)
        mapper = _default_mapper(Nch1=16, Nch2=16)
        vol = process_scan_from_sphere(
            sphere, mapper,
            diff_motors=("tth", "th", "chi", "phi"),
            bins=(4, 4, 4),
            chunk_size=2,
            q_bounds=((-1, 1), (-1, 1), (-1, 1)),  # skip scout
        )
        from ssrl_xrd_tools.rsm.volume import RSMVolume
        assert isinstance(vol, RSMVolume)
        assert vol.shape == (4, 4, 4)

    def test_explicit_energy_overrides_sphere(self, patched_xu) -> None:
        """Pass energy= explicitly; the sphere's wavelength is ignored."""
        sphere = _make_sphere(n_frames=4, Nch1=16, Nch2=16)
        sphere.mg_args["wavelength"] = 0.0  # would be invalid without override
        process_scan_from_sphere(
            sphere, _default_mapper(Nch1=16, Nch2=16),
            diff_motors=("tth", "th", "chi", "phi"),
            bins=(2, 2, 2),
            chunk_size=2,
            energy=12000.0,
            q_bounds=((-1, 1), (-1, 1), (-1, 1)),
        )

    def test_energy_resolved_from_wavelength(self, patched_xu) -> None:
        """Default energy resolution path: λ in metres → E in eV."""
        sphere = _make_sphere(n_frames=2, Nch1=8, Nch2=8)
        sphere.mg_args["wavelength"] = 1.5e-10  # 1.5 Å ≈ 8266 eV
        process_scan_from_sphere(
            sphere, _default_mapper(Nch1=8, Nch2=8),
            diff_motors=("tth", "th", "chi", "phi"),
            bins=(2, 2, 2),
            chunk_size=2,
            q_bounds=((-1, 1), (-1, 1), (-1, 1)),
        )
        # No assertion needed; mostly that the energy plumbing didn't crash.

    def test_one_streaming_gridder_for_whole_sphere(self, patched_xu) -> None:
        sphere = _make_sphere(n_frames=11, Nch1=16, Nch2=16)
        process_scan_from_sphere(
            sphere, _default_mapper(Nch1=16, Nch2=16),
            diff_motors=("tth", "th", "chi", "phi"),
            bins=(4, 4, 4),
            chunk_size=3,
            q_bounds=((-1, 1), (-1, 1), (-1, 1)),
        )
        # Exactly one xu.Gridder3D for the whole sphere
        assert len(_FakeGridder3D.instances) == 1
        # 11 frames in chunk_size=3 → [3, 3, 3, 2]
        g = _FakeGridder3D.instances[0]
        sizes = [shape[0] for (_, shape) in g.chunks]
        assert sizes == [3, 3, 3, 2]

    def test_scout_does_not_read_images(self, patched_xu) -> None:
        """scout pass must not call _lazy_load_raw or sphere.arches.__getitem__."""
        sphere = _make_sphere(n_frames=4, Nch1=16, Nch2=16)
        # Replace arch images with None + lazy payload so any read shows up
        for i, a in enumerate(sphere.arches._by_idx.values()):  # type: ignore[attr-defined]
            a.map_raw = None
            a._lazy_payload = np.zeros((16, 16), dtype=np.float32)
            a.lazy_load_calls = 0
        before_get_calls = len(sphere.arches.get_calls)
        # No q_bounds → scout runs.  After scout completes but before
        # the add() loop, lazy_load should still be 0 on every arch.
        # We can't observe "between scout and add" directly, but if scout
        # itself read images we'd see lazy_load_calls > 0 on the arches
        # whose indices come BEFORE the first chunk's processing — which
        # is impossible because scout only takes angles + image_shape.
        # Easier proof: scout uses the corner-q tiny detector path, which
        # never touches sphere.arches at all.  Patch __getitem__ to fail
        # if called before add() is allowed in.
        process_scan_from_sphere(
            sphere, _default_mapper(Nch1=16, Nch2=16),
            diff_motors=("tth", "th", "chi", "phi"),
            bins=(2, 2, 2),
            chunk_size=4,  # one big chunk
        )
        # After processing: all 4 arches got loaded exactly once
        for a in sphere.arches._by_idx.values():  # type: ignore[attr-defined]
            assert a.lazy_load_calls == 1
        # And get_calls grew by exactly 4 (one per frame in the chunk)
        assert len(sphere.arches.get_calls) - before_get_calls == 4

    def test_missing_motor_raises(self, patched_xu) -> None:
        sphere = _make_sphere(motors=("tth", "th"))
        with pytest.raises(KeyError, match="phi"):
            process_scan_from_sphere(
                sphere, _default_mapper(),
                diff_motors=("tth", "th", "phi"),
                bins=(2, 2, 2),
                q_bounds=((-1, 1), (-1, 1), (-1, 1)),
            )

    def test_per_frame_angles_match_indices(self, patched_xu) -> None:
        """Angles handed to each chunk must come from the matching frame rows."""
        sphere = _make_sphere(n_frames=6, Nch1=8, Nch2=8)
        captured: list[list[np.ndarray]] = []

        # Spy on _FakeAng2Q.area to capture the per-chunk angle arrays
        from ssrl_xrd_tools.core.geometry import pixel_q as pixel_q_module

        original_pixel_q = PixelQMap.pixel_q

        def spy_pixel_q(self, angles, energy, *, UB=None, roi=None, image_shape=None):
            captured.append([np.asarray(a) for a in angles])
            return original_pixel_q(self, angles, energy, UB=UB, roi=roi,
                                    image_shape=image_shape)

        # monkeypatch via setattr — pytest's monkeypatch isn't in scope here
        PixelQMap.pixel_q = spy_pixel_q  # type: ignore[method-assign]
        try:
            process_scan_from_sphere(
                sphere, _default_mapper(Nch1=8, Nch2=8),
                diff_motors=("tth", "th"),
                bins=(2, 2, 2),
                chunk_size=3,
                q_bounds=((-1, 1), (-1, 1), (-1, 1)),
            )
        finally:
            PixelQMap.pixel_q = original_pixel_q  # type: ignore[method-assign]

        # 6 frames in chunks of 3 → 2 chunks; per chunk a 2-motor pair
        # of length-3 arrays.  Compare against scan_data slices.
        full_tth = np.asarray(sphere.scan_data["tth"].values)
        np.testing.assert_allclose(captured[0][0], full_tth[:3])
        np.testing.assert_allclose(captured[1][0], full_tth[3:])


# ---------------------------------------------------------------------------
# grid_spheres_streaming
# ---------------------------------------------------------------------------

class TestGridSpheresStreaming:
    def test_three_spheres_share_one_gridder(self, patched_xu) -> None:
        mapper = _default_mapper(Nch1=16, Nch2=16)
        sphere_inputs = [
            SphereInput(
                sphere=_make_sphere(n_frames=4, Nch1=16, Nch2=16,
                                    motors=("tth", "th", "chi", "phi"),
                                    seed=i),
                energy=11000.0 + 100 * i,
                UB=None,
            )
            for i in range(3)
        ]
        grid_spheres_streaming(
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
        """If SphereInput.energy is None, each sphere resolves its own."""
        s1 = _make_sphere(n_frames=2, Nch1=8, Nch2=8)
        s2 = _make_sphere(n_frames=2, Nch1=8, Nch2=8)
        s1.mg_args["wavelength"] = 1.0e-10  # 12398 eV
        s2.mg_args["wavelength"] = 2.0e-10  #  6199 eV
        grid_spheres_streaming(
            _default_mapper(Nch1=8, Nch2=8),
            [SphereInput(sphere=s1), SphereInput(sphere=s2)],
            diff_motors=("tth", "th", "chi", "phi"),
            bins=(2, 2, 2),
            chunk_size=2,
            q_bounds=((-1, 1), (-1, 1), (-1, 1)),
        )
        # Both spheres processed end-to-end — total 4 frames in 2 chunks
        g = _FakeGridder3D.instances[0]
        assert sum(shape[0] for (_, shape) in g.chunks) == 4

    def test_empty_inputs_rejected(self, patched_xu) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            grid_spheres_streaming(
                _default_mapper(), [],
                diff_motors=("tth",), bins=(2, 2, 2),
            )

    def test_scout_runs_without_q_bounds(self, patched_xu) -> None:
        """No q_bounds → scout must succeed and produce non-degenerate bounds."""
        sphere_inputs = [
            SphereInput(sphere=_make_sphere(n_frames=3, Nch1=8, Nch2=8)),
        ]
        grid_spheres_streaming(
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
