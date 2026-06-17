"""Round-trip and safety tests for the shared HDF5 codec.

Covers the post-0.35.2 safety refactor: ``ast.literal_eval`` replaces the
old ``eval`` paths, and YAML loading defaults to ``SafeLoader``. The tests
guard both correctness (every persisted literal round-trips) and the
security property (a payload crafted to exploit the old ``eval`` path must
not execute).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

import h5py

from xrd_tools.core import hdf5 as codec


@pytest.fixture()
def tmp_h5(tmp_path):
    return tmp_path / "roundtrip.h5"


class TestRoundTrip:
    """Persisted literals and containers survive a write/read round trip."""

    def _roundtrip(self, tmp_h5, key, value):
        with h5py.File(tmp_h5, "w") as f:
            codec.data_to_h5(value, f, key)
        with h5py.File(tmp_h5, "r") as f:
            return codec.h5_to_data(f[key])

    def test_none(self, tmp_h5):
        assert self._roundtrip(tmp_h5, "x", None) is None

    @pytest.mark.parametrize("value", [0, 1, -1, 1_000_000, True, False])
    def test_scalar_int_bool(self, tmp_h5, value):
        assert self._roundtrip(tmp_h5, "x", value) == value

    @pytest.mark.parametrize("value", [0.0, 1.5, -2.25, 1e-9])
    def test_scalar_float(self, tmp_h5, value):
        result = self._roundtrip(tmp_h5, "x", value)
        assert float(result) == pytest.approx(value)

    def test_str(self, tmp_h5):
        assert self._roundtrip(tmp_h5, "x", "hello") == "hello"

    def test_array(self, tmp_h5):
        arr = np.arange(24).reshape(2, 3, 4)
        out = self._roundtrip(tmp_h5, "x", arr)
        np.testing.assert_array_equal(out, arr)

    def test_dict_with_int_keys(self, tmp_h5):
        value = {0: "a", 1: "b", 42: "c"}
        out = self._roundtrip(tmp_h5, "x", value)
        assert out == value

    def test_dict_with_tuple_keys(self, tmp_h5):
        """Tuple keys are the main reason the codec needs a literal parser.

        `ast.literal_eval` handles them natively without any eval.
        """
        value = {(0, 1): "x", (2, 3): "y"}
        out = self._roundtrip(tmp_h5, "x", value)
        assert out == value

    def test_nested_dict(self, tmp_h5):
        """Lists round-trip as numpy arrays under the current codec, so
        compare values structurally rather than with ``==``."""
        value = {"a": {"b": {"c": 1, "d": [1, 2, 3]}}}
        out = self._roundtrip(tmp_h5, "x", value)
        assert set(out) == {"a"}
        assert set(out["a"]) == {"b"}
        assert set(out["a"]["b"]) == {"c", "d"}
        assert out["a"]["b"]["c"] == 1
        np.testing.assert_array_equal(out["a"]["b"]["d"], [1, 2, 3])

    def test_series(self, tmp_h5):
        s = pd.Series([1.0, 2.0, 3.0], name="demo")
        out = self._roundtrip(tmp_h5, "x", s)
        pd.testing.assert_series_equal(out, s)

    def test_dataframe(self, tmp_h5):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        out = self._roundtrip(tmp_h5, "x", df)
        # HDF5 round-trip loses column dtype bookkeeping but not values.
        np.testing.assert_array_equal(out.values, df.values)


class TestSafety:
    """The codec must not execute code when reading malformed payloads."""

    def test_soft_list_eval_rejects_arbitrary_code(self):
        """A payload that would execute under ``eval`` must *not* execute
        under the new literal parser; it should come back as a string."""
        malicious = b"__import__('os').system('touch /tmp/xrd_pwn_probe')"
        out = codec.soft_list_eval([malicious])
        # We should get either the decoded string back or the raw bytes,
        # never a side effect.
        assert out[0] == malicious.decode() or out[0] == malicious
        # Sanity: the file shouldn't exist.
        import os
        assert not os.path.exists("/tmp/xrd_pwn_probe")

    def test_soft_list_eval_still_parses_legitimate_literals(self):
        """Tuples, ints, floats, strings all still round-trip."""
        items = [b"(0, 1)", b"42", b"1.5", b"'hello'", b"[1, 2, 3]"]
        out = codec.soft_list_eval(items)
        assert out == [(0, 1), 42, 1.5, "hello", [1, 2, 3]]

    def test_h5_to_dict_safe_on_malicious_keys(self, tmp_h5):
        """A crafted key that would execute under the old ``eval`` must
        be treated as a plain string now."""
        with h5py.File(tmp_h5, "w") as f:
            grp = f.create_group("d")
            grp.attrs["encoded"] = "dict"
            grp.create_dataset(
                "__import__('os').system('touch /tmp/xrd_pwn_key')",
                data=1,
            )
        with h5py.File(tmp_h5, "r") as f:
            out = codec.h5_to_dict(f["d"])
        # Key should come back as a literal string since it's not a
        # valid Python literal.
        assert any(isinstance(k, str) and "os" in k for k in out)
        import os
        assert not os.path.exists("/tmp/xrd_pwn_key")

    def test_default_yaml_loader_is_safe(self, tmp_h5):
        """Reading an old ``encoded='yaml'`` payload must use SafeLoader by
        default — no arbitrary Python object construction."""
        with h5py.File(tmp_h5, "w") as f:
            # Stash a YAML payload that would otherwise execute via the
            # legacy ``!!python/object/apply`` tag. SafeLoader rejects it.
            payload = b"!!python/object/apply:os.system ['touch /tmp/xrd_pwn_yaml']"
            f.create_dataset("x", data=payload, dtype=h5py.string_dtype())
            f["x"].attrs["encoded"] = "yaml"
        with h5py.File(tmp_h5, "r") as f:
            with pytest.raises(yaml.YAMLError):
                codec.h5_to_data(f["x"])
        import os
        assert not os.path.exists("/tmp/xrd_pwn_yaml")


class TestPortableCompression:
    """The pandas-table writers never emit lzf (ARM64-macOS bus-error filter):
    an ``lzf`` request is normalized to gzip on every platform, matching the
    converged ``_comp_kwargs`` / containers policy."""

    def test_series_lzf_request_is_written_as_gzip(self, tmp_h5):
        s = pd.Series(np.arange(32, dtype=np.float64), name="vals")
        with h5py.File(tmp_h5, "w") as f:
            codec.series_to_h5(s, f, "s", compression="lzf")
        with h5py.File(tmp_h5, "r") as f:
            assert f["s/data"].compression == "gzip"        # never lzf
            assert f["s/index"].compression == "gzip"
        # value still round-trips
        with h5py.File(tmp_h5, "r") as f:
            out = codec.h5_to_data(f["s"])
        np.testing.assert_allclose(np.asarray(out), np.arange(32))

    def test_dataframe_lzf_request_is_written_as_gzip(self, tmp_h5):
        df = pd.DataFrame(np.arange(12, dtype=np.float64).reshape(4, 3))
        with h5py.File(tmp_h5, "w") as f:
            codec.dataframe_to_h5(df, f, "d", compression="lzf")
        with h5py.File(tmp_h5, "r") as f:
            assert f["d/data"].compression == "gzip"

    def test_data_to_h5_default_is_gzip(self, tmp_h5):
        s = pd.Series(np.arange(16, dtype=np.float64), name="v")
        with h5py.File(tmp_h5, "w") as f:
            codec.data_to_h5(s, f, "s")                      # default compression
        with h5py.File(tmp_h5, "r") as f:
            assert f["s/data"].compression == "gzip"
