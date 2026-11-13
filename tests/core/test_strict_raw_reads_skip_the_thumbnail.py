"""C-2 of NEXT-1210-RESPONSIVENESS: a raw read decodes the thumbnail only when it substitutes it.

The plan (handoffs/review_fixes_1210_2026-11-11.md, C-2, from the 1.2.9 review's A4): ``io/read.py`` decoded a
frame's thumbnail before a strict read that never uses it. The source pointer is read first; the thumbnail is fetched
only where substitution is allowed and the raw source cannot be read. Rows: a broken thumbnail with a valid raw source
reads; ``allow_thumbnail=False`` never opens the thumbnail.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from xrd_tools.io import read
from xrd_tools.io.read import get_raw_frame
from xrd_tools.io.schema import THUMBNAIL_LUT_ATTRS
from tests.core.test_image_source import processed_with_master  # noqa: F401 (fixture)


def _decodes(monkeypatch):
    calls = []
    original = read._dequantize_thumbnail
    monkeypatch.setattr(read, "_dequantize_thumbnail", lambda ds: calls.append(ds.name) or original(ds))
    return calls


@pytest.mark.parametrize("allow_thumbnail", (True, False))
def test_a_broken_thumbnail_never_stops_a_valid_raw_read(processed_with_master, allow_thumbnail):  # noqa: F811
    nxs, raw, _thumb = processed_with_master
    with h5py.File(nxs, "r+") as handle:
        handle["entry/frames/frame_0000/thumbnail"].attrs[THUMBNAIL_LUT_ATTRS[0]] = "not a number"
    np.testing.assert_array_equal(get_raw_frame(nxs, 0, allow_thumbnail=allow_thumbnail), raw[1])


def test_a_raw_read_decodes_the_thumbnail_only_to_substitute_it(processed_with_master, monkeypatch):  # noqa: F811
    nxs, raw, thumb = processed_with_master
    calls = _decodes(monkeypatch)
    np.testing.assert_array_equal(get_raw_frame(nxs, 0, allow_thumbnail=False), raw[1])
    assert calls == []  # strict: never opened
    np.testing.assert_array_equal(get_raw_frame(nxs, 0), raw[1])
    assert calls == []  # the raw source read: nothing to substitute
    (nxs.parent / "scan_master.h5").unlink()
    with pytest.raises(KeyError, match="thumbnail fallback disabled"):
        get_raw_frame(nxs, 0, allow_thumbnail=False)
    assert calls == []
    np.testing.assert_allclose(get_raw_frame(nxs, 0), thumb, atol=(thumb.max() - thumb.min()) / 255)
    assert calls == ["/entry/frames/frame_0000/thumbnail"]  # substituted, so decoded once
