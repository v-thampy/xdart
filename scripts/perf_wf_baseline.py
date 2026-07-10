# -*- coding: utf-8 -*-
"""Overlay/Waterfall accumulator perf baseline (V1 canonical-grid, Stage 0).

Scripted offscreen publish loop through the OV acceptance harness
(``tests.xdart.ov_harness.OVHarness``) — the REAL controller → adapter →
``accumulate_waterfall`` path — measuring the two seams the V1 stages touch:

* per-append ``accumulate_waterfall`` time (the accumulate leg, timed at the
  display_publication call site), and
* ``_history_to_payload`` build time at full accumulator depth (the per-render
  payload leg — the one per-tick regression vector once transforms move to
  draw time).

Each frame publishes with ``select="only"`` so every render resolves ONLY the
new frame — the production live O(new) tick shape (PERF-3's
``_live_overlay_render_labels`` contract) — while the accumulator grows
append-only to ``--frames`` rows.  The harness re-checks the OV invariants
after every event, so the total wall time includes that instrumentation
overhead; compare totals only against runs of this same script.

Run (Stage 0 baseline, re-run after Stages 3 and 4 for the ≤5% bar):

    XDART_PERF=1 QT_QPA_PLATFORM=offscreen pixi run python \
        scripts/perf_wf_baseline.py

The harness never reaches the pyqtgraph draw, so the XDART_PERF ``update_wf``
render_ms log lines do not fire here; the payload-build leg above is the
headless stand-in for that per-render cost.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", _REPO_ROOT, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _stats(seconds):
    ms = np.asarray(seconds, dtype=float) * 1e3
    return (float(ms.mean()), float(np.percentile(ms, 95)), float(ms.max()),
            int(ms.size))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=3600,
                        help="frames to publish (default 3600)")
    parser.add_argument("--npt", type=int, default=2000,
                        help="1D grid points per frame (default 2000)")
    parser.add_argument("--heavy-window", type=int, default=64,
                        help="store heavy residency cap (default 64 — the "
                             "production heavy_window() ceiling, pinned for "
                             "reproducibility)")
    parser.add_argument("--payload-calls", type=int, default=20,
                        help="_history_to_payload timing calls at full depth")
    args = parser.parse_args()

    from tests.xdart.ov_harness import OVHarness
    import xdart.gui.tabs.static_scan.display_publication as dp

    harness = OVHarness(max_heavy_items=args.heavy_window)

    # Time every accumulate_waterfall call at the adapter's call site.  With
    # select="only" (non-slice Overlay) exactly one accumulate fires per
    # publish, so this list is the per-append series.
    accumulate_times = []
    real_accumulate = dp.accumulate_waterfall

    def timed_accumulate(*a, **kw):
        t0 = time.perf_counter()
        out = real_accumulate(*a, **kw)
        accumulate_times.append(time.perf_counter() - t0)
        return out

    publish_times = []
    dp.accumulate_waterfall = timed_accumulate
    try:
        t_start = time.perf_counter()
        for i in range(args.frames):
            t0 = time.perf_counter()
            harness.publish(i, npt=args.npt, select="only")
            publish_times.append(time.perf_counter() - t0)
        total_wall = time.perf_counter() - t_start
    finally:
        dp.accumulate_waterfall = real_accumulate

    count = harness.persistent_count
    assert count == args.frames, (
        f"accumulator holds {count} rows, expected {args.frames}")

    # Payload-build leg at full depth (the plan's _history_to_payload probe).
    history = harness.widget._waterfall_history
    adapter = dp.PublicationDisplayAdapter(
        harness.store, widget=harness.widget, labels=())
    payload_times = []
    for _ in range(args.payload_calls):
        t0 = time.perf_counter()
        adapter._history_to_payload(history)
        payload_times.append(time.perf_counter() - t0)

    print("V1 canonical-grid — Overlay/Waterfall perf baseline")
    print(f"date={time.strftime('%Y-%m-%d %H:%M:%S')}  commit={_git_sha()}  "
          f"python={platform.python_version()}  numpy={np.__version__}  "
          f"platform={platform.platform()}")
    print(f"frames={args.frames}  npt={args.npt}  "
          f"heavy_window={args.heavy_window}  "
          f"loop=OVHarness.publish(select='only')  [O(new) live tick shape]")
    print()
    header = (f"{'metric':44s} {'mean_ms':>9s} {'p95_ms':>9s} "
              f"{'max_ms':>9s} {'n':>6s}")
    print(header)
    print("-" * len(header))
    for name, series in (
        ("accumulate_waterfall (per append)", accumulate_times),
        ("publish tick (harness: render+invariants)", publish_times),
        (f"_history_to_payload @N={args.frames}", payload_times),
    ):
        mean, p95, peak, n = _stats(series)
        print(f"{name:44s} {mean:9.3f} {p95:9.3f} {peak:9.3f} {n:6d}")
    print("-" * len(header))
    print(f"total wall: {total_wall:.1f} s for {args.frames} frames "
          f"({args.frames / total_wall:.0f} frames/s); "
          f"accumulator rows={count}, grid={np.asarray(history.x).size} pts")


if __name__ == "__main__":
    main()
