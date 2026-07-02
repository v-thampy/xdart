# Live checkpoint — Session 1 (pre-v1.0.0 tag)

**Runner:** maintainer · **Setup:** `conda activate xrd_test`, `PYTHONPATH=$PWD/src XDART_PERF=1 xdart`
from `~/repos/xrd-tools-integrate` at the final merge candidate. Record pass/fail + HEAD sha per
item; this doc is the single artifact that discharges every pending live gate (consolidates the
gates scattered across panel-v2 §14, steps7_8 A-Step-C, the fix handoffs, roadmap serial-XYE, and
the 2026-06 visual passes). Data: the 651-frame Eiger baseline + the Eiger_TiN 800+-frame scan.

## A. Freeze/display fixes (this week's commits: 084f3410, efa0b396, 8d35a37c, 03492440, 4d2f3957)
- [ ] A1 Shift+arrow sweep, Single AND Overlay, ~800-frame scan, IDLE: no beachball, no floating
      plot window, final selection fully rendered.
- [ ] A2 Same sweep DURING a live run: no beachball, no run abort (hydration↔writer lock), live
      display keeps updating.
- [ ] A3 Overlay across the eviction boundary: select frames older than the 64-frame window —
      existing traces preserved, selected traces appear as hydration completes (converges to the
      full selection; no fresh-plot, no one-trace end state).
- [ ] A4 Raw image after a fast multi-select: no bouncing between frames.
- [ ] A5 Overlay/Waterfall during a live run with Auto-Last: smooth catch-up after
      pause→resume→Auto Last mid-scan (~2 Hz full re-select is expected, not per-flush); final
      stack complete at run end (the throttle's run-end re-select).
- [ ] A6 `[PERF] flush:` render/drain legs bounded (not ramping with frame count); note RSS at end
      of the sweep test (expect low GB, not >8 GB).

## B. Panel-v2 §14 flip re-pass (the four post-flip fixes now in tree)
- [ ] B1 Start/Stop/Append/Live cycle — real QThread teardown clean.
- [ ] B2 Average Scan (streaming): exactly ONE averaged frame/result; mean image + metadata sane.
      (Also covers the frame_view freeze-in-place change — a crash on frame 1 of a SECOND
      Average run would implicate the running-mean buffer; if seen, report immediately.)
- [ ] B3 Threshold Max entry (e.g. 1000) excludes pixels WITHOUT clicking the enable toggle;
      0/0 does not auto-enable.
- [ ] B4 Reintegrate 1D + 2D match a fresh run (standard AND GI); controls re-enable after
      Reintegrate completes.
- [ ] B5 GI mode switch live; reload; session restore with native run plan default-on.
- [ ] B6 XYE viewer + Image viewer transitions; browse during idle.
- [ ] B7 Fresh `xdart -f`: Project/Save blank until chosen; loaded processed `.nxs` does not
      enable a fresh Run (readiness points at Reintegrate).

## C. Phase-5 A-Step-C store items (steps7_8:84 — the PENDING checkpoint)
- [ ] C1 Scroll-back to evicted frames during a PAUSE: hydrates and displays (writer idle).
- [ ] C2 >64-frame Overall/Sum/Average: correct over ALL frames, built off-thread (no freeze).
- [ ] C3 Overlay preserve on evicted current frame (the 59d1f8f1 case) still holds.
- [ ] C4 New-scan boundary in directory mode: frames panel rescopes cleanly, no name flicker,
      no restart-at-N numbering.

## D. Odds and ends promised to this session
- [ ] D1 Serial (non-streaming) XYE flush writes complete files (roadmap item, never live-verified).
- [ ] D2 N2 batch submit-per-read cadence sanity (followup §0.1).
- [ ] D3 Share-Axis link visual pass; ROI stats dialog §10 visual eyeball.
- [ ] D4 Timer floor: `XDART_FLUSH_MS=80` now clamps (warning logged, render keeps working).

**PASS ⇒ proceed to RC-8** (merge → tag v1.0.0 → publish; recipe in
`handoff_chunks_jul2026.md`). Any FAIL: stop, report the item + `kill -USR1 <pid>` stack if a
freeze, fix-forward on `feature/remediation`, re-run only the failed section.
