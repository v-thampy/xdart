"""MEM-2 — RAM-aware heavy-staging window sizing.

``heavy_window(frame_bytes)`` = clamp(int(0.25 * total_RAM / frame_bytes), 16, 64),
with an ``XDART_HEAVY_WINDOW`` override (clamped [8, 128]), RAM tiers when the
frame shape is unknown, and a detection-failure fallback to today's 64.
"""

import xrd_tools.core.staging as hw
from xrd_tools.core import heavy_window, heavy_window_log_line
from xrd_tools.core import live_record_store_max_items, live_record_trace_bytes

GiB = 1024 ** 3
MiB = 1024 ** 2


# ── budget math: 25% of total RAM / frame_bytes, clamped [16, 64] ────────────
def test_large_ram_clamps_to_max():
    # 64 GiB, 64 MiB frames: 0.25*64/0.0625 = 256 -> clamp 64
    assert heavy_window(64 * MiB, total_ram_bytes=64 * GiB, env={}) == 64


def test_four_gib_hits_min_boundary():
    # 4 GiB, 64 MiB frames: 0.25*4GiB / 64MiB = 16 exactly (the min)
    assert heavy_window(64 * MiB, total_ram_bytes=4 * GiB, env={}) == 16


def test_eight_gib_midrange():
    # 8 GiB, 64 MiB frames: 0.25*8GiB / 64MiB = 32
    assert heavy_window(64 * MiB, total_ram_bytes=8 * GiB, env={}) == 32


def test_two_gib_clamps_up_to_min():
    # 2 GiB -> 0.25*2GiB / 64MiB = 8 -> clamp UP to 16
    assert heavy_window(64 * MiB, total_ram_bytes=2 * GiB, env={}) == 16


def test_tiny_detector_never_shrinks_below_today():
    # a small (2 MiB) frame -> budget/frame huge -> clamp to 64
    assert heavy_window(2 * MiB, total_ram_bytes=16 * GiB, env={}) == 64


def test_big_ram_big_frame_still_64():
    # a 137 GiB workstation w/ 36 MiB float64 Eiger frames -> 64 (today's behavior)
    assert heavy_window(36 * MiB, total_ram_bytes=137 * GiB, env={}) == 64


# NOTE: the MEM-2 handoff's example "16 GB + 64 MB -> 16" does not match the
# stated formula: 0.25*16GiB / 64MiB = 64 (the min clamp only bites near 4 GiB).
# Implemented per the stated formula; flagged for the maintainer to retune the
# fraction if a tighter budget on 16 GB boxes is wanted.
def test_sixteen_gib_matches_formula_not_the_slip():
    assert heavy_window(64 * MiB, total_ram_bytes=16 * GiB, env={}) == 64


# ── env override: pins the value, clamped to [8, 128] ────────────────────────
def test_override_pins_value():
    got = heavy_window(64 * MiB, total_ram_bytes=4 * GiB,
                       env={"XDART_HEAVY_WINDOW": "48"})
    assert got == 48   # overrides the computed 16


def test_override_clamps_high_and_low():
    big = {"XDART_HEAVY_WINDOW": "999"}
    small = {"XDART_HEAVY_WINDOW": "1"}
    assert heavy_window(64 * MiB, total_ram_bytes=137 * GiB, env=big) == 128
    assert heavy_window(64 * MiB, total_ram_bytes=137 * GiB, env=small) == 8


def test_override_16_for_small_ram_simulation():
    # the H3 plateau gate drives a 16-frame window via the override
    got = heavy_window(64 * MiB, total_ram_bytes=137 * GiB,
                       env={"XDART_HEAVY_WINDOW": "16"})
    assert got == 16


def test_malformed_override_falls_through_to_formula():
    got = heavy_window(64 * MiB, total_ram_bytes=64 * GiB,
                       env={"XDART_HEAVY_WINDOW": "not-an-int"})
    assert got == 64


# ── detection failure + shape-unknown RAM tiers ─────────────────────────────
def test_detection_failure_defaults_to_64(monkeypatch):
    monkeypatch.setattr(hw, "total_physical_ram_bytes", lambda: None)
    assert hw.heavy_window(64 * MiB, env={}) == 64


def test_shape_unknown_uses_ram_tiers():
    assert heavy_window(None, total_ram_bytes=8 * GiB, env={}) == 16     # < 16 GiB
    assert heavy_window(None, total_ram_bytes=20 * GiB, env={}) == 32    # < 32 GiB
    assert heavy_window(None, total_ram_bytes=64 * GiB, env={}) == 64    # else


def test_zero_or_negative_frame_bytes_falls_to_tiers():
    assert heavy_window(0, total_ram_bytes=8 * GiB, env={}) == 16


# ── BW-A6: byte-budgeted live 1D record-store count ─────────────────────────
def test_live_record_store_count_uses_two_gib_cap():
    trace = live_record_trace_bytes(3000)
    assert trace == 3000 * 8 * 9
    assert live_record_store_max_items(
        3000, total_ram_bytes=128 * GiB
    ) == max(4096, int((2 * GiB) / trace))


def test_live_record_store_count_uses_five_percent_ram_budget():
    trace = live_record_trace_bytes(3000)
    assert live_record_store_max_items(
        3000, total_ram_bytes=24 * GiB
    ) == max(4096, int((0.05 * 24 * GiB) / trace))


def test_live_record_store_count_keeps_4096_floor():
    assert live_record_store_max_items(
        30000, total_ram_bytes=4 * GiB
    ) == 4096
    assert live_record_store_max_items(
        3000, total_ram_bytes=None
    ) >= 4096


def test_live_record_trace_bytes_falls_back_to_3000_points():
    assert live_record_trace_bytes(None) == live_record_trace_bytes(3000)
    assert live_record_trace_bytes(-1) == live_record_trace_bytes(3000)


# ── the run-start log line ──────────────────────────────────────────────────
def test_log_line_shape():
    line = heavy_window_log_line(32, 64 * MiB, total_ram_bytes=32 * GiB)
    assert line.startswith("heavy window: 32 frames")
    assert "GB RAM" in line and "MB/frame" in line


def test_log_line_marks_override():
    line = heavy_window_log_line(16, 64 * MiB, total_ram_bytes=137 * GiB,
                                 overridden=True)
    assert "override" in line
