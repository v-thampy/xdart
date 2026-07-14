from __future__ import annotations

import os
from types import SimpleNamespace


def _holder(tmp_path, **over):
    h = SimpleNamespace(
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
    for key, val in over.items():
        setattr(h, key, val)
    return h


def _counted_walk(calls):
    real_walk = os.walk

    def counted_walk(path):
        calls.append(os.fspath(path))
        yield from real_walk(path)

    return counted_walk


def test_found_seed_is_cached_indefinitely(monkeypatch, tmp_path):
    # A FOUND seed for a given key does not change, so it is cached INDEFINITELY:
    # setup() runs on every parameter edit and must not re-walk a large image
    # directory on each one (the dominant source-config GUI stall).
    from xdart.gui.tabs.static_scan.wranglers import image_wrangler

    (tmp_path / "scan_0001.tif").write_bytes(b"not-a-real-tiff")
    calls: list[str] = []
    monkeypatch.setattr(image_wrangler.os, "walk", _counted_walk(calls))
    monkeypatch.setattr(image_wrangler, "match_img_detector", lambda *_: True)

    holder = _holder(tmp_path)
    match = lambda _stem: True  # noqa: E731

    first = image_wrangler.imageWrangler._find_image_directory_seed(holder, match, ".tif")
    second = image_wrangler.imageWrangler._find_image_directory_seed(holder, match, ".tif")
    assert first == second
    assert calls == [str(tmp_path)]                      # exactly one walk

    # Even long past the 0.5s burst window, a found seed is NOT re-walked.
    key, value, _ = holder._img_dir_probe_cache
    holder._img_dir_probe_cache = (key, value, -10.0)
    image_wrangler.imageWrangler._find_image_directory_seed(holder, match, ".tif")
    assert calls == [str(tmp_path)]                      # still one walk (cached)


def test_source_key_change_rescans(monkeypatch, tmp_path):
    # A genuine source change (dir/ext/filter/subdir/meta/poni) changes the cache
    # key, which must invalidate the cache and rescan.
    from xdart.gui.tabs.static_scan.wranglers import image_wrangler

    (tmp_path / "scan_0001.tif").write_bytes(b"not-a-real-tiff")
    calls: list[str] = []
    monkeypatch.setattr(image_wrangler.os, "walk", _counted_walk(calls))
    monkeypatch.setattr(image_wrangler, "match_img_detector", lambda *_: True)

    holder = _holder(tmp_path)
    match = lambda _stem: True  # noqa: E731
    image_wrangler.imageWrangler._find_image_directory_seed(holder, match, ".tif")
    assert calls == [str(tmp_path)]

    holder.file_filter = "other"                         # key change -> rescan
    image_wrangler.imageWrangler._find_image_directory_seed(holder, match, ".tif")
    assert calls == [str(tmp_path), str(tmp_path)]


def test_negative_result_is_not_poison_cached(monkeypatch, tmp_path):
    # No matching frame yet (empty/filling dir): a None result must NOT be cached
    # indefinitely — the directory may still be filling — so it re-walks after the
    # burst TTL and picks up a newly-arrived frame (no negative-cache poisoning).
    from xdart.gui.tabs.static_scan.wranglers import image_wrangler

    calls: list[str] = []
    monkeypatch.setattr(image_wrangler.os, "walk", _counted_walk(calls))
    monkeypatch.setattr(image_wrangler, "match_img_detector", lambda *_: True)

    holder = _holder(tmp_path)                           # empty dir -> no seed
    match = lambda _stem: True  # noqa: E731
    assert image_wrangler.imageWrangler._find_image_directory_seed(holder, match, ".tif") is None
    assert calls == [str(tmp_path)]

    # Within the burst window: coalesced (no re-walk).
    image_wrangler.imageWrangler._find_image_directory_seed(holder, match, ".tif")
    assert calls == [str(tmp_path)]

    # After the TTL, with a frame now present: re-walks and finds it.
    key, value, _ = holder._img_dir_probe_cache
    holder._img_dir_probe_cache = (key, value, -10.0)
    (tmp_path / "scan_0001.tif").write_bytes(b"not-a-real-tiff")
    found = image_wrangler.imageWrangler._find_image_directory_seed(holder, match, ".tif")
    assert calls == [str(tmp_path), str(tmp_path)]
    assert found is not None


def test_nexus_seed_discovery_does_not_require_sidecar_metadata(
        monkeypatch, tmp_path):
    """Meta Type enriches a raw container; it must not gate its discovery.

    NeXus files carry detector data and may carry embedded motor/counter
    metadata without any sidecar.  With the GUI defaulting to ``auto``, the old
    seed probe rejected every such file before the directory worker could see
    it.
    """
    from xdart.gui.tabs.static_scan.wranglers import image_wrangler

    source = tmp_path / "raw_00001.nxs"
    source.write_bytes(b"not-a-real-nexus")
    monkeypatch.setattr(image_wrangler, "match_img_detector", lambda *_: True)

    holder = _holder(
        tmp_path,
        meta_ext="auto",
        exists_meta_file=lambda _fname: False,
    )
    match = lambda _stem: True  # noqa: E731

    found = image_wrangler.imageWrangler._find_image_directory_seed(
        holder, match, ".nxs")

    assert found == str(source)
