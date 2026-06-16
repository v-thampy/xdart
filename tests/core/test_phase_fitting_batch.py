"""Tests for batch phase-fitting storage and NeXus streaming helpers."""

from __future__ import annotations

import h5py
import numpy as np

import xrd_tools.analysis.fitting.batch as batch_mod
from xrd_tools.analysis.fitting.batch import FitConfig, FitResultStore, fit_nexus


class _Param:
    def __init__(self, value):
        self.value = value


class _Phase:
    def __init__(self, name):
        self.name = name


class _Fitter:
    def __init__(self):
        self.phases = [_Phase("alpha"), _Phase("beta")]


class _Result:
    success = True
    redchi = 1.25
    q_shift = 0.02

    def __init__(self):
        self.fitter = _Fitter()
        self.params = {
            "p0_scale": _Param(2.0),
            "p1_scale": _Param(1.0),
            "p0_a": _Param(4.0),
            "p1_a": _Param(3.0),
        }

    def phase_fractions(self):
        return {"alpha": 2.0 / 3.0, "beta": 1.0 / 3.0}

    def lattice_params(self, idx):
        return {"a": 4.0 - idx}

    def width_params(self, idx):
        return {"sigma": 0.01 + idx}


def test_fit_result_store_can_drop_heavy_results_but_keep_summaries():
    store = FitResultStore(keep_results=False)
    store.append(_Result(), index=7, label="scan-7", elapsed=0.5)

    assert len(store) == 1
    assert store[0]["result"] is None
    assert store.results == []
    assert store[0]["phase_fractions"] == {
        "alpha": 2.0 / 3.0,
        "beta": 1.0 / 3.0,
    }
    assert store[0]["lattice_params"]["alpha"]["a"] == 4.0
    assert store[0]["width_params"]["beta"]["sigma"] == 1.01
    assert store[0]["params_values"]["p0_scale"] == 2.0

    df = store.to_dataframe()
    assert list(df["label"]) == ["scan-7"]
    assert df.loc[0, "frac_alpha"] == 2.0 / 3.0
    assert df.loc[0, "alpha_a"] == 4.0


def test_fit_nexus_streams_rows_without_full_read(monkeypatch, tmp_path):
    path = tmp_path / "scan.nxs"
    q = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    intensity = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    sigma = np.sqrt(intensity).astype(np.float32)
    with h5py.File(path, "w") as h5:
        g = h5.create_group("entry/integrated_1d")
        g.create_dataset("q", data=q)
        g.create_dataset("intensity", data=intensity)
        g.create_dataset("sigma", data=sigma)
        g.create_dataset("frame_index", data=np.array([10, 11], dtype=np.int64))

    def fail_read_scan(*args, **kwargs):
        raise AssertionError("fit_nexus must not materialize read_scan output")

    def fake_fit_patterns(patterns, n_patterns, phases, config, **kwargs):
        rows = list(patterns)
        assert n_patterns == 2
        assert kwargs["labels"] == ["10", "11"]
        assert kwargs["keep_results"] is False
        np.testing.assert_allclose(rows[0][0], q)
        np.testing.assert_allclose(rows[0][1], intensity[0])
        np.testing.assert_allclose(rows[0][2], sigma[0])
        np.testing.assert_allclose(rows[1][1], intensity[1])
        store = FitResultStore(keep_results=False)
        store.append(_Result(), index=0, label="10")
        return store

    import xrd_tools.io.nexus as nexus_mod

    monkeypatch.setattr(nexus_mod, "read_scan", fail_read_scan)
    monkeypatch.setattr(batch_mod, "_fit_patterns", fake_fit_patterns)

    store = fit_nexus(path, phases=[], config=FitConfig())

    assert len(store) == 1
    assert store[0]["result"] is None


def test_resolve_keep_results_auto_threshold_boundary():
    assert batch_mod._resolve_keep_results("auto", 10, 10) is True
    assert batch_mod._resolve_keep_results("auto", 11, 10) is False
    assert batch_mod._resolve_keep_results(True, 999, 0) is True
    assert batch_mod._resolve_keep_results(False, 0, 999) is False
