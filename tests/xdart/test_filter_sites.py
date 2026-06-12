"""F1 GUI seam: the wrangler-side filter plumbing around compile_filter.

The grammar itself is pinned headlessly in tests/core/test_filters.py;
these tests pin the worker-side wrapper (_name_filter: malformed ->
warn-once -> match NOTHING) and the suffix-stripping at the Eiger
master-queue site.
"""
from pathlib import Path
from types import MethodType, SimpleNamespace

from xdart.gui.tabs.static_scan.wranglers import image_wrangler_thread as iwt


def test_name_filter_compiles_and_matches():
    match = iwt._name_filter("scan001 -bg")
    assert match("Eiger_scan001_0001")
    assert not match("Eiger_scan001_bg_0001")


def test_name_filter_malformed_matches_nothing_and_warns_once(caplog):
    iwt._warned_bad_filters.clear()
    with caplog.at_level("WARNING"):
        match = iwt._name_filter("abc |")        # trailing OR -> ValueError
        assert not match("abc_anything")          # conservative: NO names
        assert not match("zzz")
        iwt._name_filter("abc |")                 # same expr again
    warnings = [r for r in caplog.records if "Invalid Filter" in r.message]
    assert len(warnings) == 1                     # once per expression


def test_eiger_master_queue_applies_filter_to_stem(tmp_path):
    """The queue globs *_master.h5 and filters on the name MINUS the
    _master.h5 suffix — '(scan001 | scan003)' selects exactly those."""
    for stem in ("Eiger_scan001", "Eiger_scan002", "Eiger_scan003"):
        (tmp_path / f"{stem}_master.h5").touch()
    (tmp_path / "Eiger_scan001_data_000001.h5").touch()   # not a master

    holder = SimpleNamespace(
        img_dir=str(tmp_path),
        include_subdir=False,
        file_filter="scan001 | scan003",
        _eiger_master_queue=[],
        _eiger_done_masters=set(),
    )
    holder._eiger_refill_master_queue = MethodType(
        iwt.imageThread._eiger_refill_master_queue, holder)
    holder._eiger_refill_master_queue()
    names = sorted(Path(p).name for p in holder._eiger_master_queue)
    assert names == ["Eiger_scan001_master.h5", "Eiger_scan003_master.h5"]
