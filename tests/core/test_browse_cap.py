"""browse_publication_max_items — the 1 GiB byte-budget browse 1D cap.

Sibling of live_record_store_max_items with a more conservative 1 GiB ceiling
(vs the live 2 GiB), so a large Show-All (up to a few thousand frames) keeps
every cheap 1D-light publication resident with no eviction.
"""

from xrd_tools.core import browse_publication_max_items, live_record_store_max_items
from xrd_tools.core.staging import (
    BROWSE_PUBLICATION_MAX_BYTES,
    LIVE_RECORD_MAX_BYTES,
    LIVE_RECORD_MIN_ITEMS,
    live_record_trace_bytes,
)

_GIB = 1024 ** 3


def test_ceiling_is_1_gib_vs_live_2_gib():
    assert BROWSE_PUBLICATION_MAX_BYTES == 1 * _GIB
    assert LIVE_RECORD_MAX_BYTES == 2 * _GIB


def test_realistic_scan_fits_with_no_eviction():
    # A large scan (3621 frames) must fit under the cap for both a 2000-pt scan
    # and the 3000-pt npt=None fallback -> Show All hydrates once, no thrash.
    for npt in (2000, None):
        cap = browse_publication_max_items(npt=npt)
        assert cap >= 3621, (npt, cap)


def test_budget_never_exceeds_1_gib_at_the_ceiling():
    # With ample RAM the ceiling binds: cap * trace_bytes <= 1 GiB.
    cap = browse_publication_max_items(npt=2000, total_ram_bytes=1024 * _GIB)
    assert cap * live_record_trace_bytes(2000) <= 1 * _GIB
    # exactly the ceiling / trace_bytes when RAM is not the limit
    assert cap == int((1 * _GIB) / live_record_trace_bytes(2000))


def test_ram_detection_failure_falls_back_to_floor():
    assert browse_publication_max_items(npt=2000, total_ram_bytes=0) == LIVE_RECORD_MIN_ITEMS
    assert LIVE_RECORD_MIN_ITEMS >= 512


def test_browse_ceiling_is_below_live_for_the_same_ram():
    ram = 1024 * _GIB           # ceiling-bound for both
    browse = browse_publication_max_items(npt=2000, total_ram_bytes=ram)
    live = live_record_store_max_items(npt=2000, total_ram_bytes=ram)
    assert browse < live
    # ~half of live (1 GiB vs 2 GiB ceiling); allow int-truncation slack of 1
    assert abs(browse * 2 - live) <= 1


def test_live_helper_unchanged_at_2_gib():
    # the shared refactor must not move the live default off 2 GiB
    ram = 1024 * _GIB
    assert (live_record_store_max_items(npt=2000, total_ram_bytes=ram)
            == int((2 * _GIB) / live_record_trace_bytes(2000)))
