"""MEM-3 — RAM-aware reduction worker-pool cap.

Throughput saturates at ~4 workers (the measured knee); each worker deep-copies
the pyFAI integrator (~1 GB for a large detector), so the pool is capped at the
knee, memory-aware, and NEVER maps to ``None`` (the old ``Cores=1 -> None``
silently built a ~20-worker default pool).
"""

import xrd_tools.core.staging as st
from xrd_tools.core import reduction_worker_cap

GiB = 1024 ** 3
BIG = 137 * GiB      # roomy workstation
SMALL = 8 * GiB      # small-RAM box (< 16 GiB floor)


def _cpu(monkeypatch, n):
    monkeypatch.setattr(st.os, "cpu_count", lambda: n)


# ── default knee + user-wins ────────────────────────────────────────────────
def test_default_is_the_knee_of_4(monkeypatch):
    _cpu(monkeypatch, 16)
    assert reduction_worker_cap(None, total_ram_bytes=BIG, env={}) == 4


def test_user_cores_wins_over_the_knee(monkeypatch):
    _cpu(monkeypatch, 16)
    assert reduction_worker_cap(8, total_ram_bytes=BIG, env={}) == 8


def test_cores_1_is_honest_not_none(monkeypatch):
    _cpu(monkeypatch, 16)
    got = reduction_worker_cap(1, total_ram_bytes=BIG, env={})
    assert got == 1 and got is not None       # the fixed latent 1->20 bug


def test_requested_clamped_to_cpu(monkeypatch):
    _cpu(monkeypatch, 8)
    assert reduction_worker_cap(100, total_ram_bytes=BIG, env={}) == 8


def test_requested_hard_capped_at_16(monkeypatch):
    _cpu(monkeypatch, 64)
    assert reduction_worker_cap(100, total_ram_bytes=BIG, env={}) == 16


# ── small-RAM floor ─────────────────────────────────────────────────────────
def test_small_ram_floors_requested_to_2(monkeypatch):
    _cpu(monkeypatch, 16)
    assert reduction_worker_cap(8, total_ram_bytes=SMALL, env={}) == 2


def test_small_ram_floors_default_to_2(monkeypatch):
    _cpu(monkeypatch, 16)
    assert reduction_worker_cap(None, total_ram_bytes=SMALL, env={}) == 2


# ── env override ────────────────────────────────────────────────────────────
def test_env_override_pins_and_wins(monkeypatch):
    _cpu(monkeypatch, 16)
    env = {"XDART_REDUCTION_WORKERS": "6"}
    assert reduction_worker_cap(1, total_ram_bytes=BIG, env=env) == 6


def test_env_override_clamped(monkeypatch):
    _cpu(monkeypatch, 16)
    assert reduction_worker_cap(1, total_ram_bytes=BIG,
                                env={"XDART_REDUCTION_WORKERS": "999"}) == 16
    assert reduction_worker_cap(1, total_ram_bytes=BIG,
                                env={"XDART_REDUCTION_WORKERS": "0"}) == 1


def test_env_override_beats_small_ram_floor(monkeypatch):
    # an explicit env value is a deliberate choice; not floored
    _cpu(monkeypatch, 16)
    assert reduction_worker_cap(1, total_ram_bytes=SMALL,
                                env={"XDART_REDUCTION_WORKERS": "8"}) == 8


def test_malformed_env_falls_through_to_knee(monkeypatch):
    _cpu(monkeypatch, 16)
    assert reduction_worker_cap(None, total_ram_bytes=BIG,
                                env={"XDART_REDUCTION_WORKERS": "nope"}) == 4


# ── invariants ──────────────────────────────────────────────────────────────
def test_always_returns_a_positive_int_on_real_machine():
    got = reduction_worker_cap()
    assert isinstance(got, int) and got >= 1


def test_ram_detection_failure_still_returns_knee(monkeypatch):
    _cpu(monkeypatch, 16)
    monkeypatch.setattr(st, "total_physical_ram_bytes", lambda: None)
    # total unknown -> no small-RAM floor -> the knee
    assert reduction_worker_cap(None, env={}) == 4
