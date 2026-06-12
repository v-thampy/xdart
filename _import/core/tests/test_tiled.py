"""Tests for ssrl_xrd_tools.io.tiled (mocked — tiled is optional)."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

import ssrl_xrd_tools.io.tiled as tiled_mod
from ssrl_xrd_tools.core.metadata import ScanMetadata

try:
    from ssrl_xrd_tools.io.tiled import (
        connect_tiled,
        list_scans,
        read_tiled_run,
        _HAS_TILED,
    )
except ImportError:
    # Module-level guard: only happens if ssrl_xrd_tools itself can't be imported.
    _HAS_TILED = False
    pytestmark = pytest.mark.skip(reason="tiled module not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_run():
    """Create a mock Tiled run object with an xarray-like data stream."""
    run = MagicMock()
    run.metadata = {
        "start": {
            "scan_id": 42,
            "energy": 12.0,
            "wavelength": 1.033,
            "sample_name": "test_sample",
            "motors": ["th", "tth"],
            "detectors": ["i0", "i1"],
            "plan_name": "scan",
        }
    }

    # Dataset mock: __contains__ always returns True; __getitem__ returns
    # an object whose .values is a 1-D float array.
    var = MagicMock()
    var.values = np.linspace(0, 1, 10)

    ds = MagicMock()
    ds.__contains__ = MagicMock(return_value=True)
    ds.__getitem__ = MagicMock(return_value=var)

    stream = MagicMock()
    stream.read.return_value = ds

    run.__getitem__ = MagicMock(return_value=stream)
    return run


@pytest.fixture
def mock_client(mock_run):
    """Create a mock Tiled catalog client."""
    client = MagicMock()
    client.__getitem__ = MagicMock(return_value=mock_run)
    client.items.return_value = [("uid_abc", mock_run)]
    return client


# ---------------------------------------------------------------------------
# TestReadTiledRun
# ---------------------------------------------------------------------------

class TestReadTiledRun:
    @pytest.fixture(autouse=True)
    def _enable_tiled(self, monkeypatch):
        monkeypatch.setattr(tiled_mod, "_HAS_TILED", True)

    def test_returns_scan_metadata(self, mock_client):
        meta = read_tiled_run(mock_client, "uid_abc")
        assert isinstance(meta, ScanMetadata)

    def test_energy(self, mock_client):
        meta = read_tiled_run(mock_client, "uid_abc")
        np.testing.assert_allclose(meta.energy, 12.0)

    def test_wavelength(self, mock_client):
        meta = read_tiled_run(mock_client, "uid_abc")
        np.testing.assert_allclose(meta.wavelength, 1.033, rtol=1e-4)

    def test_source(self, mock_client):
        meta = read_tiled_run(mock_client, "uid_abc")
        assert meta.source == "tiled"

    def test_h5_path_is_none(self, mock_client):
        meta = read_tiled_run(mock_client, "uid_abc")
        assert meta.h5_path is None

    def test_scan_id_stringified(self, mock_client):
        meta = read_tiled_run(mock_client, 42)
        assert meta.scan_id == "42"

    def test_sample_name(self, mock_client):
        meta = read_tiled_run(mock_client, "uid_abc")
        assert meta.sample_name == "test_sample"

    def test_scan_type(self, mock_client):
        meta = read_tiled_run(mock_client, "uid_abc")
        assert meta.scan_type == "scan"

    def test_default_motors_from_start_doc(self, mock_client):
        meta = read_tiled_run(mock_client, "uid_abc")
        assert "th" in meta.angles
        assert "tth" in meta.angles

    def test_default_counters_from_start_doc(self, mock_client):
        meta = read_tiled_run(mock_client, "uid_abc")
        assert "i0" in meta.counters
        assert "i1" in meta.counters

    def test_custom_motor_names(self, mock_client):
        meta = read_tiled_run(mock_client, "uid_abc", motor_names=["th"], counter_names=[])
        assert set(meta.angles.keys()) == {"th"}
        assert meta.counters == {}

    def test_keyerror_missing_scan(self, mock_client):
        mock_client.__getitem__.side_effect = KeyError("not_found")
        with pytest.raises(KeyError):
            read_tiled_run(mock_client, "not_found")

    def test_angles_are_float_arrays(self, mock_client):
        meta = read_tiled_run(mock_client, "uid_abc")
        for arr in meta.angles.values():
            assert arr.dtype == np.float64

    def test_counters_are_float_arrays(self, mock_client):
        meta = read_tiled_run(mock_client, "uid_abc")
        for arr in meta.counters.values():
            assert arr.dtype == np.float64


# ---------------------------------------------------------------------------
# TestConnectTiled
# ---------------------------------------------------------------------------

class TestConnectTiled:
    def test_raises_without_tiled(self, monkeypatch):
        monkeypatch.setattr(tiled_mod, "_HAS_TILED", False)
        with pytest.raises(ImportError, match="tiled"):
            connect_tiled("http://example.com")

    def test_calls_from_uri(self, monkeypatch):
        monkeypatch.setattr(tiled_mod, "_HAS_TILED", True)
        fake_client = object()
        fake_from_uri = MagicMock(return_value=fake_client)
        # raising=False: create the attribute even when tiled is not installed
        monkeypatch.setattr(tiled_mod, "_tiled_from_uri", fake_from_uri, raising=False)

        result = connect_tiled("http://example.com", api_key="mykey")
        assert result is fake_client
        fake_from_uri.assert_called_once_with("http://example.com", api_key="mykey")

    def test_calls_from_uri_no_key(self, monkeypatch):
        monkeypatch.setattr(tiled_mod, "_HAS_TILED", True)
        fake_from_uri = MagicMock(return_value=None)
        monkeypatch.setattr(tiled_mod, "_tiled_from_uri", fake_from_uri, raising=False)

        connect_tiled("http://example.com")
        # api_key kwarg must NOT be passed when it is None
        call_kwargs = fake_from_uri.call_args[1]
        assert "api_key" not in call_kwargs


# ---------------------------------------------------------------------------
# TestListScans
# ---------------------------------------------------------------------------

class TestListScans:
    @pytest.fixture(autouse=True)
    def _enable_tiled(self, monkeypatch):
        monkeypatch.setattr(tiled_mod, "_HAS_TILED", True)

    def test_raises_without_tiled(self, mock_client, monkeypatch):
        monkeypatch.setattr(tiled_mod, "_HAS_TILED", False)
        with pytest.raises(ImportError, match="tiled"):
            list_scans(mock_client)

    def test_returns_list(self, mock_client):
        result = list_scans(mock_client)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_dict_keys(self, mock_client):
        result = list_scans(mock_client)
        expected_keys = {"scan_id", "uid", "plan_name", "sample_name", "num_points"}
        for item in result:
            assert set(item.keys()) == expected_keys

    def test_limit(self, mock_run, monkeypatch):
        monkeypatch.setattr(tiled_mod, "_HAS_TILED", True)
        many_runs = [(str(i), mock_run) for i in range(20)]
        client = MagicMock()
        client.items.return_value = many_runs
        result = list_scans(client, limit=5)
        assert len(result) == 5
