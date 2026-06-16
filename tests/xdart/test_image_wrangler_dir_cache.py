from __future__ import annotations

import os
from types import SimpleNamespace


def test_image_directory_seed_probe_is_burst_cached(monkeypatch, tmp_path):
    from xdart.gui.tabs.static_scan.wranglers import image_wrangler

    (tmp_path / "scan_0001.tif").write_bytes(b"not-a-real-tiff")
    calls: list[str] = []
    real_walk = os.walk

    def counted_walk(path):
        calls.append(os.fspath(path))
        yield from real_walk(path)

    monkeypatch.setattr(image_wrangler.os, "walk", counted_walk)
    monkeypatch.setattr(image_wrangler, "match_img_detector", lambda *_: True)

    holder = SimpleNamespace(
        img_dir=str(tmp_path),
        include_subdir=False,
        file_filter="*",
        meta_ext=None,
        meta_dir="",
        poni_file="",
        poni=None,
        _img_dir_probe_cache=(None, None, 0.0),
        exists_meta_file=lambda _fname: True,
    )

    match = lambda _stem: True
    first = image_wrangler.imageWrangler._find_image_directory_seed(
        holder, match, ".tif",
    )
    second = image_wrangler.imageWrangler._find_image_directory_seed(
        holder, match, ".tif",
    )

    assert first == second
    assert calls == [str(tmp_path)]

    # A real later setup/start is outside the burst window and rescans.
    key, value, _cached_at = holder._img_dir_probe_cache
    holder._img_dir_probe_cache = (key, value, -10.0)
    image_wrangler.imageWrangler._find_image_directory_seed(
        holder, match, ".tif",
    )
    assert calls == [str(tmp_path), str(tmp_path)]

