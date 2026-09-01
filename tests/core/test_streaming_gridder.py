"""Tests for the streaming RSM gridder (xrd_tools.rsm.gridding).

Covers:

* :class:`StreamingGridder` lifecycle (set_bounds / scout / add / to_volume)
  with a mocked ``xu.Gridder3D`` so we can verify chunk handoff without
  installing xrayutilities.
* :func:`grid_img_data_streaming` chunking behaviour — equivalence to
  the single-shot path within binning tolerance, scout-pass bounds.
* :func:`grid_scans_streaming` multi-scan accumulation.
* :func:`_corner_pixel_q` tiny-detector trick: cch'/pwidth' produce a
  3×3 grid that lands on the original detector's corners and edge midpoints.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import pickle
from threading import Thread
from typing import Any
import weakref

import numpy as np
import pytest

from xrd_tools.core.geometry import (
    DetectorHeader,
    DiffractometerConfig,
    PixelQMap,
)
# Import directly from submodules to avoid the rsm/__init__.py chain pulling
# in pipeline → io.spec → silx (silx is not in the test sandbox).
from xrd_tools.rsm import gridding as gridding_module
from xrd_tools.rsm.gridding import (
    RSMGridChunkLease,
    RSMGridChunkReleaseError,
    StreamingGridder,
    StreamingScan,
    _corner_pixel_q,
    grid_img_data_streaming,
    grid_scans_streaming,
)
from xrd_tools.rsm.volume import RSMVolume


# ---------------------------------------------------------------------------
# Mocks for xu.Gridder3D + DiffractometerConfig.make_hxrd
# ---------------------------------------------------------------------------

@dataclass
class _ChunkRecord:
    qx: np.ndarray
    qy: np.ndarray
    qz: np.ndarray
    data: np.ndarray


class _FakeGridder3D:
    """Records every chunk passed in; exposes the accumulated data as a sum."""

    instances: list["_FakeGridder3D"] = []

    def __init__(self, nx: int, ny: int, nz: int) -> None:
        self.nx, self.ny, self.nz = nx, ny, nz
        self.xaxis = np.linspace(-1, 1, nx)
        self.yaxis = np.linspace(-1, 1, ny)
        self.zaxis = np.linspace(-1, 1, nz)
        self.data = np.zeros((nx, ny, nz), dtype=float)
        self.keep_data: bool = False
        self.normalize: bool = True
        self.data_range: tuple[float, ...] | None = None
        self.data_range_fixed: bool = False
        self.chunks: list[_ChunkRecord] = []
        _FakeGridder3D.instances.append(self)

    def KeepData(self, flag: bool) -> None:
        self.keep_data = bool(flag)

    def Normalize(self, flag: bool) -> None:  # noqa: N802 — matches xu API
        # P6: the real gridder runs Normalize(False) so .data is the bare SUM
        # (Σ over the bin), which the two-gridder Σraw/Σnorm form relies on.
        self.normalize = bool(flag)

    def dataRange(  # noqa: N802 — matches xu API
        self,
        xmin: float, xmax: float,
        ymin: float, ymax: float,
        zmin: float, zmax: float,
        fixed: bool = False,
    ) -> None:
        self.data_range = (xmin, xmax, ymin, ymax, zmin, zmax)
        self.data_range_fixed = fixed
        # Re-grid axes to match the requested range (close enough for tests)
        self.xaxis = np.linspace(xmin, xmax, self.nx)
        self.yaxis = np.linspace(ymin, ymax, self.ny)
        self.zaxis = np.linspace(zmin, zmax, self.nz)

    def __call__(
        self,
        qx: np.ndarray,
        qy: np.ndarray,
        qz: np.ndarray,
        data: np.ndarray,
    ) -> None:
        self.chunks.append(_ChunkRecord(qx.copy(), qy.copy(), qz.copy(), data.copy()))
        # Mirror Normalize(False): .data accumulates the bare SUM over the bin
        # (nansum, so an all-masked chunk contributes 0, never NaN-poisons).
        self.data += float(np.nansum(data))


class _FakeAng2Q:
    """Returns shape-consistent zero/one/two-valued q arrays so shape checks pass."""

    def __init__(self) -> None:
        self.init_kwargs: dict = {}

    def init_area(self, *args: Any, **kwargs: Any) -> None:
        self.init_kwargs = kwargs

    def area(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Determine N from the first positional (sample-axis) argument
        N = len(np.atleast_1d(args[0]))
        Nch1 = self.init_kwargs.get("Nch1")
        Nch2 = self.init_kwargs.get("Nch2")
        # Return q arrays that vary with frame + pixel so scouting produces
        # non-degenerate (lo, hi) bounds.
        frame_idx = np.arange(N, dtype=float).reshape(N, 1, 1)
        row_idx = np.arange(Nch1, dtype=float).reshape(1, Nch1, 1)
        col_idx = np.arange(Nch2, dtype=float).reshape(1, 1, Nch2)
        qx = 0.5 + 0.01 * frame_idx + 0.001 * row_idx
        qy = 1.5 + 0.02 * frame_idx + 0.002 * col_idx
        qz = 2.5 + 0.03 * frame_idx + 0.003 * (row_idx + col_idx)
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
    """Patch the xu.Gridder3D used by gridding.py + DiffractometerConfig.make_hxrd."""
    monkeypatch.setattr(
        gridding_module, "_GRIDDER3D_OVERRIDE", _FakeGridder3D
    )
    monkeypatch.setattr(
        DiffractometerConfig, "make_hxrd",
        lambda self, energy: _FakeHXRD(),
    )


def _default_mapper(Nch1: int = 64, Nch2: int = 64) -> PixelQMap:
    return PixelQMap(
        diff_config=DiffractometerConfig(),
        header=DetectorHeader(
            cch1=Nch1 / 2.0, cch2=Nch2 / 2.0,
            pwidth1=0.075, pwidth2=0.075,
            distance=830.0,
            Nch1=Nch1, Nch2=Nch2,
        ),
    )


def test_streaming_gridder_explicit_runtime_owns_every_xu_access(monkeypatch):
    from xrayutilities import config

    from xrd_tools.core.geometry.xu_runtime import (
        XuRuntimeUnsupported,
        xu_runtime_session,
    )

    before = config.NTHREADS
    runtime = xu_runtime_session()

    class GuardedGridder:
        def __init__(self, nx, ny, nz):
            assert runtime.active
            self._xaxis = np.linspace(-1.0, 1.0, nx)
            self._yaxis = np.linspace(-1.0, 1.0, ny)
            self._zaxis = np.linspace(-1.0, 1.0, nz)
            self._data = np.zeros((nx, ny, nz), dtype=float)

        @property
        def xaxis(self):
            assert runtime.active
            return self._xaxis

        @property
        def yaxis(self):
            assert runtime.active
            return self._yaxis

        @property
        def zaxis(self):
            assert runtime.active
            return self._zaxis

        @property
        def data(self):
            assert runtime.active
            return self._data

        def KeepData(self, _flag):  # noqa: N802
            assert runtime.active

        def Normalize(self, _flag):  # noqa: N802
            assert runtime.active

        def dataRange(self, *bounds, **_kwargs):  # noqa: N802
            assert runtime.active
            self._xaxis = np.linspace(bounds[0], bounds[1], len(self._xaxis))
            self._yaxis = np.linspace(bounds[2], bounds[3], len(self._yaxis))
            self._zaxis = np.linspace(bounds[4], bounds[5], len(self._zaxis))

        def __call__(self, _qx, _qy, _qz, data):
            assert runtime.active
            self._data += float(np.nansum(data))

    monkeypatch.setattr(
        DiffractometerConfig,
        "make_hxrd",
        lambda self, energy: _FakeHXRD(),
    )
    with runtime:
        monkeypatch.setattr(runtime.xu, "Gridder3D", GuardedGridder)
        gridder = StreamingGridder(
            _default_mapper(4, 5),
            (3, 3, 3),
            runtime_session=runtime,
        )
        gridder.set_bounds((-1, 1), (-1, 1), (-1, 1))
        image = np.ones((2, 4, 5), dtype=float)
        angles = [np.zeros(2, dtype=float)]
        gridder.add(image, angles, 12_000.0)
        volume = gridder.to_volume()
        assert volume.shape == (3, 3, 3)

        failures = []

        def foreign_thread():
            try:
                StreamingGridder(
                    _default_mapper(4, 5),
                    (3, 3, 3),
                    runtime_session=runtime,
                )
            except BaseException as error:
                failures.append(error)

        thread = Thread(target=foreign_thread)
        thread.start()
        thread.join(5)
        assert not thread.is_alive()
        assert len(failures) == 1
        assert type(failures[0]) is XuRuntimeUnsupported
        assert failures[0].code == "XU_RUNTIME_SESSION_INACTIVE"
    assert config.NTHREADS == before
    with pytest.raises(XuRuntimeUnsupported) as raised:
        StreamingGridder(
            _default_mapper(4, 5),
            (3, 3, 3),
            runtime_session=runtime,
        )
    assert raised.value.code == "XU_RUNTIME_SESSION_INACTIVE"


def _lease_with_root_refs(
    *,
    retain_raw: bool = False,
    retain_conditioned: bool = False,
):
    if retain_raw and retain_conditioned:
        raise ValueError("retain only one caller root per proof")
    raw_root = np.arange(40, dtype=np.uint16).reshape(2, 4, 5)
    conditioned_root = np.arange(40, dtype=np.float64).reshape(2, 4, 5)
    raw = raw_root.view()
    conditioned = conditioned_root.view()
    raw_ref = weakref.ref(raw_root)
    conditioned_ref = weakref.ref(conditioned_root)
    retained = (
        raw_root
        if retain_raw
        else conditioned_root
        if retain_conditioned
        else None
    )
    lease = RSMGridChunkLease.from_arrays(raw, conditioned, 2)
    raw = conditioned = raw_root = conditioned_root = None
    return lease, raw_ref, conditioned_ref, retained


def _install_shared_q_backend(monkeypatch, q_refs, retained_q=None):
    class SharedAng2Q(_FakeAng2Q):
        def area(self, *args, **kwargs):
            count = len(np.atleast_1d(args[0]))
            rows = self.init_kwargs["Nch1"]
            columns = self.init_kwargs["Nch2"]
            root = np.empty((count, rows, columns, 3), dtype=np.float64)
            root[..., 0] = 0.25
            root[..., 1] = 0.50
            root[..., 2] = 0.75
            q_refs.append(weakref.ref(root))
            if retained_q is not None:
                retained_q.append(root)
            return root[..., 0], root[..., 1], root[..., 2]

    class SharedHxrd:
        Ang2Q = SharedAng2Q()

    monkeypatch.setattr(
        DiffractometerConfig,
        "make_hxrd",
        lambda self, energy: SharedHxrd(),
    )


def _retaining_gridder_type(channel, retained, *, fail=False):
    created = []

    class RetainingGridder(_FakeGridder3D):
        def __init__(self, *args):
            self.channel = len(created)
            created.append(self)
            super().__init__(*args)

        def __call__(self, qx, qy, qz, data):
            if self.channel == channel:
                retained.append(data)
            super().__call__(qx, qy, qz, data)
            if fail and self.channel == channel:
                raise RuntimeError("private retained backend failure")

    return RetainingGridder


def _exercise_leased_release_failure(
    monkeypatch,
    *,
    lease,
    gridder_type=_FakeGridder3D,
    retained_q=None,
):
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session

    q_refs = []
    _install_shared_q_backend(monkeypatch, q_refs, retained_q)
    runtime = xu_runtime_session()
    with runtime:
        monkeypatch.setattr(runtime.xu, "Gridder3D", gridder_type)
        gridder = StreamingGridder(
            _default_mapper(4, 5),
            (3, 3, 3),
            runtime_session=runtime,
        )
        gridder.set_bounds((-1, 1), (-1, 1), (-1, 1))
        with pytest.raises(RSMGridChunkReleaseError) as refused:
            gridder.add_leased(
                lease,
                tuple(np.zeros(2, dtype=np.float64) for _ in range(6)),
                12_000.0,
                UB=np.eye(3, dtype=np.float64),
            )
        assert refused.value.code == "RSM_CHUNK_RELEASE_FAILED"
        assert lease.consumed is True
        assert gridder.n_frames_processed == 0
        with pytest.raises(RSMGridChunkReleaseError):
            gridder.to_volume()
        second, *_rest = _lease_with_root_refs()
        with pytest.raises(RSMGridChunkReleaseError):
            gridder.add_leased(
                second,
                tuple(np.zeros(2, dtype=np.float64) for _ in range(6)),
                12_000.0,
                UB=np.eye(3, dtype=np.float64),
            )
        assert second.consumed is False
    return q_refs


def test_rsm_chunk_lease_public_release_seam_and_factory_guards(monkeypatch):
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session

    with pytest.raises(TypeError):
        RSMGridChunkLease(
            np.zeros((1, 4, 5), dtype=np.uint16),
            np.zeros((1, 4, 5), dtype=np.float64),
            1,
        )
    lease, raw_ref, conditioned_ref, retained = _lease_with_root_refs()
    assert retained is None
    for operation in (copy.copy, copy.deepcopy, pickle.dumps, replace):
        with pytest.raises(TypeError):
            operation(lease)
    if hasattr(copy, "replace"):
        with pytest.raises(TypeError):
            copy.replace(lease)
    for name, value in (
        ("_raw", np.zeros((2, 4, 5), dtype=np.uint16)),
        ("_conditioned", np.zeros((2, 4, 5), dtype=np.float64)),
        ("frame_count", 1),
        ("_consumed", True),
    ):
        with pytest.raises(TypeError, match="immutable"):
            setattr(lease, name, value)

    q_refs = []
    _install_shared_q_backend(monkeypatch, q_refs)
    runtime = xu_runtime_session()
    with runtime:
        monkeypatch.setattr(runtime.xu, "Gridder3D", _FakeGridder3D)
        gridder = StreamingGridder(
            _default_mapper(4, 5),
            (3, 3, 3),
            runtime_session=runtime,
        )
        gridder.set_bounds((-1, 1), (-1, 1), (-1, 1))
        receipt = gridder.add_leased(
            lease,
            tuple(np.zeros(2, dtype=np.float64) for _ in range(6)),
            12_000.0,
            UB=np.eye(3, dtype=np.float64),
        )
        assert receipt.frame_count == 2
        assert receipt.q_root_count == 1
        assert receipt.release_passed is True
        assert gridder.n_frames_processed == 2
        assert gridder.to_volume().shape == (3, 3, 3)

    assert raw_ref() is None
    assert conditioned_ref() is None
    assert len(q_refs) == 1 and q_refs[0]() is None
    for operation in (copy.copy, copy.deepcopy, pickle.dumps, replace):
        with pytest.raises(TypeError):
            operation(receipt)
    if hasattr(copy, "replace"):
        with pytest.raises(TypeError):
            copy.replace(receipt)
    with pytest.raises((TypeError, RuntimeError)):
        gridder.add_leased(
            lease,
            tuple(np.zeros(2, dtype=np.float64) for _ in range(6)),
            12_000.0,
            UB=np.eye(3, dtype=np.float64),
        )


def test_rsm_chunk_root_drift_is_refused_before_consumption(monkeypatch):
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session

    lease, _raw_ref, _conditioned_ref, retained = _lease_with_root_refs()
    assert retained is None
    replacement_raw = np.ones((2, 4, 5), dtype=np.uint16)
    replacement_conditioned = np.ones((2, 4, 5), dtype=np.float64)
    object.__setattr__(lease, "_raw", replacement_raw)
    object.__setattr__(lease, "_conditioned", replacement_conditioned)
    q_refs = []
    _install_shared_q_backend(monkeypatch, q_refs)
    runtime = xu_runtime_session()
    with runtime:
        monkeypatch.setattr(runtime.xu, "Gridder3D", _FakeGridder3D)
        gridder = StreamingGridder(
            _default_mapper(4, 5),
            (3, 3, 3),
            runtime_session=runtime,
        )
        gridder.set_bounds((-1, 1), (-1, 1), (-1, 1))
        with pytest.raises(TypeError, match="roots changed"):
            gridder.add_leased(
                lease,
                tuple(np.zeros(2, dtype=np.float64) for _ in range(6)),
                12_000.0,
                UB=np.eye(3, dtype=np.float64),
            )
        assert lease.consumed is False
        assert gridder.n_frames_processed == 0
        assert q_refs == []
        gridder.add(
            np.ones((1, 4, 5), dtype=np.float64),
            [np.zeros(1, dtype=np.float64)],
            12_000.0,
        )
        assert gridder.n_frames_processed == 1
    assert len(q_refs) == 1 and q_refs[0]() is None


def test_rsm_chunk_conditioned_root_retention_poisons_grid(monkeypatch):
    lease, raw_ref, conditioned_ref, retained_conditioned = (
        _lease_with_root_refs(retain_conditioned=True)
    )
    q_refs = _exercise_leased_release_failure(
        monkeypatch,
        lease=lease,
    )
    assert raw_ref() is None
    assert conditioned_ref() is retained_conditioned
    assert len(q_refs) == 1 and q_refs[0]() is None
    retained_conditioned = None


def test_rsm_chunk_q_root_retention_poisons_grid(monkeypatch):
    lease, raw_ref, conditioned_ref, retained = _lease_with_root_refs()
    assert retained is None
    retained_q = []
    q_refs = _exercise_leased_release_failure(
        monkeypatch,
        lease=lease,
        retained_q=retained_q,
    )
    assert raw_ref() is None
    assert conditioned_ref() is None
    assert len(retained_q) == 1
    assert len(q_refs) == 1 and q_refs[0]() is retained_q[0]
    retained_q.clear()


@pytest.mark.parametrize("channel", [0, 1], ids=["raw-payload", "norm-payload"])
def test_rsm_chunk_feed_payload_retention_poisons_grid(monkeypatch, channel):
    lease, raw_ref, conditioned_ref, retained = _lease_with_root_refs()
    assert retained is None
    retained_payload = []
    gridder_type = _retaining_gridder_type(channel, retained_payload)
    q_refs = _exercise_leased_release_failure(
        monkeypatch,
        lease=lease,
        gridder_type=gridder_type,
    )
    assert raw_ref() is None
    assert conditioned_ref() is None
    assert len(q_refs) == 1 and q_refs[0]() is None
    assert len(retained_payload) == 1
    retained_payload.clear()


def test_rsm_chunk_release_failure_precedes_private_backend_error(monkeypatch):
    lease, raw_ref, conditioned_ref, retained = _lease_with_root_refs()
    assert retained is None
    retained_payload = []
    gridder_type = _retaining_gridder_type(
        0,
        retained_payload,
        fail=True,
    )
    q_refs = _exercise_leased_release_failure(
        monkeypatch,
        lease=lease,
        gridder_type=gridder_type,
    )
    assert raw_ref() is None
    assert conditioned_ref() is None
    assert len(q_refs) == 1 and q_refs[0]() is None
    assert len(retained_payload) == 1
    retained_payload.clear()


def test_rsm_chunk_release_failure_poisons_partial_grid(monkeypatch):
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session

    lease, raw_ref, conditioned_ref, retained_raw = _lease_with_root_refs(
        retain_raw=True
    )
    q_refs = []
    _install_shared_q_backend(monkeypatch, q_refs)
    runtime = xu_runtime_session()
    with runtime:
        monkeypatch.setattr(runtime.xu, "Gridder3D", _FakeGridder3D)
        gridder = StreamingGridder(
            _default_mapper(4, 5),
            (3, 3, 3),
            runtime_session=runtime,
        )
        gridder.set_bounds((-1, 1), (-1, 1), (-1, 1))
        with pytest.raises(RSMGridChunkReleaseError) as refused:
            gridder.add_leased(
                lease,
                tuple(np.zeros(2, dtype=np.float64) for _ in range(6)),
                12_000.0,
                UB=np.eye(3, dtype=np.float64),
            )
        assert refused.value.code == "RSM_CHUNK_RELEASE_FAILED"
        assert gridder.n_frames_processed == 0
        with pytest.raises(RSMGridChunkReleaseError):
            gridder.to_volume()
        with pytest.raises(RSMGridChunkReleaseError):
            gridder.add(
                np.ones((1, 4, 5), dtype=np.float64),
                [np.zeros(1)],
                12_000.0,
            )
    assert raw_ref() is retained_raw
    assert conditioned_ref() is None
    assert len(q_refs) == 1 and q_refs[0]() is None
    retained_raw = None


def test_rsm_chunk_backend_failure_without_retention_still_poisons(monkeypatch):
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session

    created = []

    class FailingGridder(_FakeGridder3D):
        def __init__(self, *args):
            self.channel = len(created)
            created.append(self)
            super().__init__(*args)

        def __call__(self, qx, qy, qz, data):
            super().__call__(qx, qy, qz, data)
            if self.channel == 0:
                raise RuntimeError("backend failure without retention")

    lease, raw_ref, conditioned_ref, retained = _lease_with_root_refs()
    assert retained is None
    q_refs = []
    _install_shared_q_backend(monkeypatch, q_refs)
    runtime = xu_runtime_session()
    with runtime:
        monkeypatch.setattr(runtime.xu, "Gridder3D", FailingGridder)
        gridder = StreamingGridder(
            _default_mapper(4, 5),
            (3, 3, 3),
            runtime_session=runtime,
        )
        gridder.set_bounds((-1, 1), (-1, 1), (-1, 1))
        with pytest.raises(RuntimeError, match="without retention"):
            gridder.add_leased(
                lease,
                tuple(np.zeros(2, dtype=np.float64) for _ in range(6)),
                12_000.0,
                UB=np.eye(3, dtype=np.float64),
            )
        assert lease.consumed is True
        assert gridder.n_frames_processed == 0
        second, *_rest = _lease_with_root_refs()
        with pytest.raises(RSMGridChunkReleaseError):
            gridder.add_leased(
                second,
                tuple(np.zeros(2, dtype=np.float64) for _ in range(6)),
                12_000.0,
                UB=np.eye(3, dtype=np.float64),
            )
        assert second.consumed is False
        with pytest.raises(RSMGridChunkReleaseError):
            gridder.to_volume()
    assert raw_ref() is None
    assert conditioned_ref() is None
    assert len(q_refs) == 1 and q_refs[0]() is None


def test_rsm_chunk_collection_failure_is_stable_and_scout_fails_fast(
    monkeypatch,
):
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session

    lease, raw_ref, conditioned_ref, retained = _lease_with_root_refs()
    assert retained is None
    q_refs = []
    _install_shared_q_backend(monkeypatch, q_refs)
    runtime = xu_runtime_session()
    with runtime:
        monkeypatch.setattr(runtime.xu, "Gridder3D", _FakeGridder3D)
        gridder = StreamingGridder(
            _default_mapper(4, 5),
            (3, 3, 3),
            runtime_session=runtime,
        )
        gridder.set_bounds((-1, 1), (-1, 1), (-1, 1))
        monkeypatch.setattr(
            gridding_module,
            "_bounded_release_collect",
            lambda: (_ for _ in ()).throw(RuntimeError("collector failed")),
        )
        with pytest.raises(RSMGridChunkReleaseError) as refused:
            gridder.add_leased(
                lease,
                tuple(np.zeros(2, dtype=np.float64) for _ in range(6)),
                12_000.0,
                UB=np.eye(3, dtype=np.float64),
            )
        assert refused.value.code == "RSM_CHUNK_RELEASE_FAILED"
        assert gridder.n_frames_processed == 0
        monkeypatch.setattr(
            gridding_module,
            "_corner_pixel_q",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("poisoned scout must not map q")
            ),
        )
        with pytest.raises(RSMGridChunkReleaseError):
            gridder.scout(
                [
                    (
                        [np.zeros(1, dtype=np.float64)],
                        12_000.0,
                        np.eye(3, dtype=np.float64),
                        (1, 4, 5),
                    )
                ]
            )
    assert raw_ref() is None
    assert conditioned_ref() is None
    assert len(q_refs) == 1 and q_refs[0]() is None


# ---------------------------------------------------------------------------
# _corner_pixel_q
# ---------------------------------------------------------------------------

class TestCornerPixelQ:
    """The tiny-detector trick must reduce calls to a tiny edge scout."""

    def test_uses_3x3_virtual_detector(self, patched_xu) -> None:
        mapper = _default_mapper(Nch1=514, Nch2=1030)
        angles = [np.array([0.0, 0.1, 0.2])]
        qx, qy, qz = _corner_pixel_q(mapper, angles, energy=12000.0)
        assert qx.shape == (3, 3, 3)
        assert qy.shape == (3, 3, 3)
        assert qz.shape == (3, 3, 3)

    def test_cch_and_pwidth_scale(self, patched_xu) -> None:
        """Verify the tiny-detector header math directly via init_area kwargs."""
        # Construct a mapper whose make_hxrd we can inspect
        captured: dict = {}

        class _Tracker(_FakeAng2Q):
            def init_area(self, *args: Any, **kwargs: Any) -> None:
                super().init_area(*args, **kwargs)
                captured.update(kwargs)

        class _TrackerHXRD:
            Ang2Q = _Tracker()

        from xrd_tools.core.geometry import diffractometer as diff_module

        original = DiffractometerConfig.make_hxrd
        try:
            DiffractometerConfig.make_hxrd = lambda self, energy: _TrackerHXRD()  # type: ignore[method-assign]
            mapper = _default_mapper(Nch1=514, Nch2=1030)
            _corner_pixel_q(mapper, [np.array([0.0])], energy=12000.0)
            # cch' = cch / ((N - 1) / 2), pwidth' = pwidth * ((N - 1) / 2)
            assert captured["Nch1"] == 3
            assert captured["Nch2"] == 3
            assert captured["cch1"] == pytest.approx(257.0 / (513 / 2))
            assert captured["cch2"] == pytest.approx(515.0 / (1029 / 2))
            assert captured["pwidth1"] == pytest.approx(0.075 * (513 / 2))
            assert captured["pwidth2"] == pytest.approx(0.075 * (1029 / 2))
        finally:
            DiffractometerConfig.make_hxrd = original  # type: ignore[method-assign]

    def test_scout_includes_edge_midpoints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _MidpointPeak(_FakeAng2Q):
            def area(self, *args: Any, **kwargs: Any):
                n = len(np.atleast_1d(args[0]))
                n1 = self.init_kwargs["Nch1"]
                n2 = self.init_kwargs["Nch2"]
                row = np.arange(n1, dtype=float).reshape(1, n1, 1)
                col = np.arange(n2, dtype=float).reshape(1, 1, n2)
                # Corners are 0, edge midpoints/center include the peak at 10.
                qx2 = 10.0 - 10.0 * np.abs(row - 1.0)
                qy2 = 20.0 - 20.0 * np.abs(col - 1.0)
                qz2 = qx2 + qy2
                return (
                    np.broadcast_to(qx2, (n, n1, n2)).astype(float, copy=True),
                    np.broadcast_to(qy2, (n, n1, n2)).astype(float, copy=True),
                    np.broadcast_to(qz2, (n, n1, n2)).astype(float, copy=True),
                )

        class _MidpointPeakHXRD:
            Ang2Q = _MidpointPeak()

        monkeypatch.setattr(
            DiffractometerConfig, "make_hxrd",
            lambda self, energy: _MidpointPeakHXRD(),
        )
        mapper = _default_mapper(Nch1=9, Nch2=9)
        qx, qy, qz = _corner_pixel_q(mapper, [np.array([0.0])], energy=12000.0)

        assert qx.shape == (1, 3, 3)
        assert float(np.nanmax(qx)) == pytest.approx(10.0)
        assert float(np.nanmax(qy)) == pytest.approx(20.0)
        assert float(np.nanmax(qz)) == pytest.approx(30.0)

    def test_rejects_tiny_detector(self, patched_xu) -> None:
        mapper = _default_mapper(Nch1=1, Nch2=64)
        with pytest.raises(ValueError, match="corner scout requires"):
            _corner_pixel_q(mapper, [np.array([0.0])], energy=12000.0)


# ---------------------------------------------------------------------------
# StreamingGridder lifecycle
# ---------------------------------------------------------------------------

class TestStreamingGridder:
    def test_add_before_bounds_raises(self, patched_xu) -> None:
        sg = StreamingGridder(_default_mapper(), (10, 10, 10))
        with pytest.raises(RuntimeError, match="bounds not set"):
            sg.add(np.zeros((1, 64, 64)), [np.array([0.0])], energy=12000.0)

    def test_to_volume_before_add_raises(self, patched_xu) -> None:
        sg = StreamingGridder(_default_mapper(), (10, 10, 10))
        sg.set_bounds((-1, 1), (-1, 1), (-1, 1))
        with pytest.raises(RuntimeError, match="no chunks processed"):
            sg.to_volume()

    def test_set_bounds_twice_raises(self, patched_xu) -> None:
        sg = StreamingGridder(_default_mapper(), (10, 10, 10))
        sg.set_bounds((-1, 1), (-1, 1), (-1, 1))
        with pytest.raises(RuntimeError, match="bounds already set"):
            sg.set_bounds((-2, 2), (-2, 2), (-2, 2))

    def test_inverted_bounds_rejected(self, patched_xu) -> None:
        sg = StreamingGridder(_default_mapper(), (10, 10, 10))
        with pytest.raises(ValueError, match="qx range"):
            sg.set_bounds((1, -1), (-1, 1), (-1, 1))
        with pytest.raises(ValueError, match="qy range"):
            sg.set_bounds((-1, 1), (1, -1), (-1, 1))
        with pytest.raises(ValueError, match="qz range"):
            sg.set_bounds((-1, 1), (-1, 1), (1, -1))

    def test_gridder_configured_for_streaming(self, patched_xu) -> None:
        """KeepData(True) and fixed dataRange are the heart of the streaming path."""
        sg = StreamingGridder(_default_mapper(), (8, 9, 10))
        sg.set_bounds((-1, 1), (-2, 2), (-3, 3))
        sg.add(
            np.zeros((1, 64, 64)),
            [np.array([0.0])],
            energy=12000.0,
        )
        g = _FakeGridder3D.instances[-1]
        assert g.keep_data is True
        assert g.data_range_fixed is True
        assert g.data_range == (-1, 1, -2, 2, -3, 3)

    def test_chunk_handoff_preserves_per_frame_count(self, patched_xu) -> None:
        sg = StreamingGridder(_default_mapper(), (8, 9, 10))
        sg.set_bounds((-1, 1), (-2, 2), (-3, 3))

        # Three chunks of varying sizes
        for chunk_n in (3, 4, 2):
            sg.add(
                np.random.default_rng(chunk_n).random((chunk_n, 64, 64)),
                [np.linspace(0, chunk_n * 0.1, chunk_n)],
                energy=12000.0,
            )
        assert sg.n_frames_processed == 9
        # _FakeGridder3D records every chunk
        g = _FakeGridder3D.instances[-1]
        assert [c.data.shape[0] for c in g.chunks] == [3, 4, 2]
        # One StreamingGridder uses TWO xu.Gridder3D instances (P6): the
        # Σraw and Σnorm accumulators of the Σraw/Σnorm grid.
        assert len(_FakeGridder3D.instances) == 2

    def test_scout_sets_bounds_from_corner_q(self, patched_xu) -> None:
        sg = StreamingGridder(_default_mapper(), (8, 9, 10))
        ((qx_lo, qx_hi), (qy_lo, qy_hi), (qz_lo, qz_hi)) = sg.scout(
            [([np.array([0.0, 0.1])], 12000.0, None, (64, 64))],
        )
        # _FakeAng2Q.area now varies with frame + pixel, so bounds are
        # non-degenerate (lo < hi on every axis).
        assert qx_lo < qx_hi
        assert qy_lo < qy_hi
        assert qz_lo < qz_hi
        # Bounds were pushed into the gridder via set_bounds → dataRange.
        g = _FakeGridder3D.instances[-1]
        assert g.data_range == (qx_lo, qx_hi, qy_lo, qy_hi, qz_lo, qz_hi)
        assert g.data_range_fixed is True

    def test_scout_pad_widens_bounds(self, patched_xu) -> None:
        sg_no_pad = StreamingGridder(_default_mapper(), (4, 4, 4))
        no_pad = sg_no_pad.scout(
            [([np.array([0.0, 0.1, 0.2])], 12000.0, None, (32, 32))],
        )
        sg_pad = StreamingGridder(_default_mapper(), (4, 4, 4))
        with_pad = sg_pad.scout(
            [([np.array([0.0, 0.1, 0.2])], 12000.0, None, (32, 32))],
            pad=0.1,
        )
        for ax in range(3):
            assert with_pad[ax][0] < no_pad[ax][0]
            assert with_pad[ax][1] > no_pad[ax][1]

    def test_2d_chunk_promoted_to_3d(self, patched_xu) -> None:
        sg = StreamingGridder(_default_mapper(), (4, 4, 4))
        sg.set_bounds((-1, 1), (-1, 1), (-1, 1))
        sg.add(
            np.zeros((64, 64)),  # 2D — single frame
            [np.array([0.0])],
            energy=12000.0,
        )
        assert sg.n_frames_processed == 1
        g = _FakeGridder3D.instances[-1]
        assert g.chunks[-1].data.shape == (1, 64, 64)

    def test_p2_chunk_size_independence(self, patched_xu) -> None:
        """P2 regression: streaming results must NOT depend on chunk_size.

        Pre-fix, the per-chunk std==0 masking made a pixel that was
        constant within a small chunk get masked, while the same pixel
        in a single-chunk run wouldn't be — so results drifted with
        chunk_size.  The fix removes the chunk-local std heuristic in
        favour of an explicit caller-supplied ``static_mask``.
        """
        mapper = _default_mapper(Nch1=16, Nch2=16)
        # Image stack with at least one column whose intra-chunk-std is 0
        # for chunk_size=2 but >0 for chunk_size=4 (deliberately
        # constructed to expose the bug if it ever resurfaces).
        rng = np.random.default_rng(7)
        img = rng.random((8, 16, 16)).astype(float)
        # Column 5 is constant within every adjacent pair (chunk_size=2)
        for start in range(0, 8, 2):
            img[start:start + 2, :, 5] = 42.0
        angles = [np.linspace(0, 0.7, 8)]

        out_chunk2 = grid_img_data_streaming(
            mapper, img, angles, energy=12000.0,
            bins=(4, 4, 4), chunk_size=2,
            q_bounds=((-1, 1), (-1, 1), (-1, 1)),
        )
        out_chunk8 = grid_img_data_streaming(
            mapper, img, angles, energy=12000.0,
            bins=(4, 4, 4), chunk_size=8,
            q_bounds=((-1, 1), (-1, 1), (-1, 1)),
        )
        out_chunk1 = grid_img_data_streaming(
            mapper, img, angles, energy=12000.0,
            bins=(4, 4, 4), chunk_size=1,
            q_bounds=((-1, 1), (-1, 1), (-1, 1)),
        )
        # All three runs see the SAME pixel data because no chunk-local
        # mask is applied.  Verify by comparing the per-chunk data
        # arrays the mocked gridder recorded.  (Each run now builds the
        # Σraw/Σnorm PAIR, so there are 3 × 2 = 6 instances-with-chunks.)
        instances = [
            inst for inst in _FakeGridder3D.instances
            if inst.chunks  # ignore the ones that never received data
        ]
        flats = [
            np.concatenate([c.data.ravel() for c in inst.chunks])
            for inst in instances
        ]
        # Every gridder saw the SAME total pixel count — chunk_size only
        # changes how that data is sliced, not what's fed.
        assert len({f.size for f in flats}) == 1
        # And no NaN anywhere (no chunk-local static masking introduced any).
        assert all(int(np.sum(np.isnan(f))) == 0 for f in flats)
        # Output volumes also have the same shape (sanity).
        assert out_chunk2.shape == out_chunk8.shape == out_chunk1.shape

    def test_p2_static_mask_applied_per_chunk(self, patched_xu) -> None:
        """``static_mask`` is applied independently per chunk — the same
        mask hits every frame regardless of chunk_size.
        """
        mapper = _default_mapper(Nch1=8, Nch2=8)
        img = np.ones((6, 8, 8), dtype=float)
        angles = [np.linspace(0, 0.5, 6)]
        mask = np.zeros((8, 8), dtype=bool)
        mask[0, 0] = True  # mask one pixel

        grid_img_data_streaming(
            mapper, img, angles, energy=12000.0,
            bins=(2, 2, 2), chunk_size=2,
            q_bounds=((-1, 1), (-1, 1), (-1, 1)),
            static_mask=mask,
        )
        # Every chunk should have pixel (0, 0) set to NaN in every frame
        g = _FakeGridder3D.instances[-1]
        for chunk in g.chunks:
            assert np.all(np.isnan(chunk.data[:, 0, 0]))
            # Other pixels intact
            assert not np.any(np.isnan(chunk.data[:, 1, 1]))

    def test_p2_static_mask_shape_validation(self, patched_xu) -> None:
        mapper = _default_mapper(Nch1=8, Nch2=8)
        sg = StreamingGridder(mapper, (2, 2, 2))
        sg.set_bounds((-1, 1), (-1, 1), (-1, 1))
        with pytest.raises(ValueError, match="static_mask shape"):
            sg.add(
                np.zeros((2, 8, 8)),
                [np.array([0.0, 0.1])],
                energy=12000.0,
                static_mask=np.zeros((4, 4), dtype=bool),  # wrong shape
            )


# ---------------------------------------------------------------------------
# grid_img_data_streaming
# ---------------------------------------------------------------------------

class TestGridImgDataStreaming:
    def test_chunks_cover_all_frames(self, patched_xu) -> None:
        mapper = _default_mapper(Nch1=32, Nch2=32)
        img = np.random.default_rng(0).random((10, 32, 32))
        angles = [np.linspace(0, 1, 10)]
        out = grid_img_data_streaming(
            mapper, img, angles, energy=12000.0,
            bins=(4, 4, 4),
            chunk_size=3,
        )
        assert isinstance(out, RSMVolume)
        assert out.shape == (4, 4, 4)
        # 10 frames in chunks of 3 → 4 chunks (3, 3, 3, 1)
        g = _FakeGridder3D.instances[-1]
        assert [c.data.shape[0] for c in g.chunks] == [3, 3, 3, 1]

    def test_explicit_q_bounds_skips_scout(self, patched_xu) -> None:
        """When q_bounds is provided, no scout should happen → only one xu.Gridder3D."""
        mapper = _default_mapper(Nch1=32, Nch2=32)
        img = np.random.default_rng(0).random((6, 32, 32))
        angles = [np.linspace(0, 1, 6)]
        grid_img_data_streaming(
            mapper, img, angles, energy=12000.0,
            bins=(4, 4, 4),
            chunk_size=2,
            q_bounds=((-1, 1), (-2, 2), (-3, 3)),
        )
        # No scout (q_bounds given) → only the Σraw/Σnorm accumulator pair
        # is constructed (the scout path uses mapper.pixel_q, not a Gridder3D).
        assert len(_FakeGridder3D.instances) == 2
        assert _FakeGridder3D.instances[0].data_range == (-1, 1, -2, 2, -3, 3)

    def test_mismatched_angles_rejected(self, patched_xu) -> None:
        mapper = _default_mapper()
        img = np.zeros((10, 64, 64))
        with pytest.raises(ValueError, match="length N matching"):
            grid_img_data_streaming(
                mapper, img,
                [np.array([0.0, 0.1, 0.2])],  # only 3 entries for 10 frames
                energy=12000.0,
            )

    def test_2d_img_rejected(self, patched_xu) -> None:
        mapper = _default_mapper()
        with pytest.raises(ValueError, match=r"img must be \(N, H, W\)"):
            grid_img_data_streaming(
                mapper, np.zeros((64, 64)),
                [np.array([0.0])], energy=12000.0,
            )


# ---------------------------------------------------------------------------
# grid_scans_streaming
# ---------------------------------------------------------------------------

class TestGridScansStreaming:
    def test_multiple_scans_share_one_gridder(self, patched_xu) -> None:
        mapper = _default_mapper(Nch1=32, Nch2=32)
        scans = [
            StreamingScan(
                img=np.random.default_rng(i).random((4, 32, 32)),
                angles=[np.linspace(0, 0.4, 4) + i],
                energy=11000.0 + 100 * i,
                UB=None,
                roi=None,
            )
            for i in range(3)
        ]
        out = grid_scans_streaming(
            mapper, scans,
            bins=(4, 4, 4),
            chunk_size=2,
            q_bounds=((-5, 5), (-5, 5), (-5, 5)),
        )
        assert isinstance(out, RSMVolume)
        # ONE accumulator PAIR (Σraw, Σnorm) for the entire multi-scan run
        assert len(_FakeGridder3D.instances) == 2
        # And it sees all 12 frames in 2-frame chunks → 6 chunks
        g = _FakeGridder3D.instances[0]
        assert [c.data.shape[0] for c in g.chunks] == [2, 2, 2, 2, 2, 2]

    def test_per_scan_roi_applied(self, patched_xu) -> None:
        mapper = _default_mapper(Nch1=32, Nch2=32)
        scans = [
            StreamingScan(
                img=np.random.default_rng(0).random((2, 32, 32)),
                angles=[np.array([0.0, 0.1])],
                energy=12000.0,
                roi=(4, 28, 8, 24),  # 24 x 16 ROI
            ),
        ]
        grid_scans_streaming(
            mapper, scans,
            bins=(4, 4, 4),
            chunk_size=2,
            q_bounds=((-1, 1), (-1, 1), (-1, 1)),
        )
        g = _FakeGridder3D.instances[0]
        # Image chunk should be the ROI-cropped size
        assert g.chunks[-1].data.shape == (2, 24, 16)

    def test_empty_scans_rejected(self, patched_xu) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            grid_scans_streaming(_default_mapper(), [], bins=(4, 4, 4))


# ---------------------------------------------------------------------------
# The Σraw / Σnorm accumulator core (P6) — real xrayutilities
# ---------------------------------------------------------------------------

class TestTwoGridderAccumulator:
    """The shared Σraw/Σnorm accumulator the RSM grid is built on.

    Uses the REAL ``xu.Gridder3D`` (not the fake) — these pin the actual
    binning arithmetic the corrections ride on.
    """

    def test_sum_ratio_else_count_mean(self) -> None:
        pytest.importorskip("xrayutilities")
        from xrd_tools.rsm.gridding import (
            _feed_pair, _new_gridder, _pair_intensity,
        )
        from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
        bins, bounds = (4, 4, 4), (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        # two pixels in the SAME bin: raw 10, 20 with per-pixel norm 1, 3
        q = np.array([0.5, 0.5])
        img = np.array([10.0, 20.0])

        with xu_runtime_session() as runtime:
            gr = _new_gridder(bins, bounds, runtime_session=runtime)
            gn = _new_gridder(bins, bounds, runtime_session=runtime)
            _feed_pair(gr, gn, q, q, q, img, np.array([1.0, 3.0]))
            out = _pair_intensity(gr, gn)
            # Σraw/Σnorm = (10 + 20)/(1 + 3) = 7.5.
            assert np.nanmax(out) == pytest.approx(7.5)
            assert np.isnan(out).any()      # empty bins are NaN, not 0

            gr2 = _new_gridder(bins, bounds, runtime_session=runtime)
            gn2 = _new_gridder(bins, bounds, runtime_session=runtime)
            _feed_pair(gr2, gn2, q, q, q, img, 1.0)
            # unit norm ⇒ the count-mean (10+20)/2 = 15
            assert np.nanmax(_pair_intensity(gr2, gn2)) == pytest.approx(15.0)

    def test_masked_pixel_does_not_bias_denominator(self) -> None:
        """A NaN-data (masked) pixel drops from BOTH sums — not just the
        numerator — so it can't drag the Σraw/Σnorm mean down."""
        pytest.importorskip("xrayutilities")
        from xrd_tools.rsm.gridding import (
            _feed_pair, _new_gridder, _pair_intensity,
        )
        from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
        bins, bounds = (4, 4, 4), (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        q = np.array([0.5, 0.5])
        img = np.array([10.0, np.nan])      # 2nd pixel masked
        with xu_runtime_session() as runtime:
            gr = _new_gridder(bins, bounds, runtime_session=runtime)
            gn = _new_gridder(bins, bounds, runtime_session=runtime)
            _feed_pair(gr, gn, q, q, q, img, np.array([1.0, 1.0]))
            # only the finite pixel survives ⇒ mean 10, NOT 10/2 = 5
            assert np.nanmax(_pair_intensity(gr, gn)) == pytest.approx(10.0)

    def test_nan_coord_pixel_dropped_not_clamped(self) -> None:
        """A non-finite q coordinate is dropped (xu would otherwise clamp it
        onto an edge bin and corrupt it)."""
        pytest.importorskip("xrayutilities")
        from xrd_tools.rsm.gridding import (
            _feed_pair, _new_gridder, _pair_intensity,
        )
        from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
        bins, bounds = (4, 4, 4), (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        qx = np.array([0.5, np.nan])        # 2nd pixel has a NaN coordinate
        q = np.array([0.5, 0.5])
        img = np.array([10.0, 99.0])
        with xu_runtime_session() as runtime:
            gr = _new_gridder(bins, bounds, runtime_session=runtime)
            gn = _new_gridder(bins, bounds, runtime_session=runtime)
            _feed_pair(gr, gn, qx, q, q, img, 1.0)
            # the 99 at the NaN coord is gone; only the clean 10 remains
            assert np.nanmax(_pair_intensity(gr, gn)) == pytest.approx(10.0)
