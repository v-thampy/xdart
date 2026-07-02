# Live findings ledger — maintainer-reported display bugs

**Rule:** every display-fix agent updates the Status column in the SAME commit as its fix; the
orchestrator checks this ledger at every review; the maintainer's live session verifies every
`fixed-unverified` row before anything is called done. Do NOT close a row on tests alone —
`live-verified (date)` requires the maintainer's repro.

| ID | Symptom (maintainer's words) | Repro | Status |
|----|------------------------------|-------|--------|
| OV-1 | **When Overlay is first selected, the selected trace gets ERASED when the next trace is selected.** After that it accumulates normally for a while. NOT hydration-dependent — happens with recent, adjacent frames. | Mode → Overlay with frame A displayed → click frame B → A vanishes, only B; C/D then accumulate | fixed-unverified (round-3: seed payload-owned overlay history on Overlay/Waterfall entry; verify Session-1 A3) |
| OV-2 | Selecting a frame much older than current → whole plot redraws WITHOUT the old traces (one trace) | Overlay w/ many traces, paused live run, click frame outside the 64-window (e.g. 331 @ current ~576) | fixed-unverified (round-3: unchanged normUpdate no longer wipes accumulator; monotonic evicted-selection test; verify Session-1 A3) |
| OV-3 | Hydrated frames fresh-plot instead of overplotting | Overlay, select evicted frames, wait for hydration | fixed-unverified (`8d35a37c` scheduler; verify at Session-1 item A3) |
| OV-4 | Overlay clears when the CURRENT frame is evicted during live | live run past 64 frames, Overlay + Auto-Last | fixed-unverified (`59d1f8f1`/`30ecf58a`; verify at Session-1 item C3) |
| FS-1 | Beachball on fast shift+arrow sweep (Single + Overlay) | ~800-frame scan, hold shift+arrow | **live-verified fixed** (2026-07-02, maintainer: "beach ball problem appears to be solved"; commits `84393c00`+`8d35a37c`) |
| FS-2 | Floating empty 1D-plot window pops up during sweeps | sweep across the 15-curve auto-waterfall boundary | fixed-unverified (`84393c00` detach/attach helpers; verify Session-1 A1) |
| FS-3 | Raw image bounces between frames after fast multi-select | fast multi-select, then wait | fixed-unverified (`8d35a37c` generation scheduler; verify Session-1 A4) |
| RN-1 | Run aborted mid-scan: writer `r+` open failed while hydration read held the file | scroll to evicted frames during live run | fixed-unverified (`084f3410` file_lock serialization; verify Session-1 A2) |
| MS-1 | Frame-count reconciliation: dispatch=288 / processed=287 / post-live indexed=283 (2026-07-02 00:51 log) | pause mid-run, read the post-live index line | **WATCH** — check the counts reconcile at run END; if not, persist-before-evict question |

**Acceptance test that covers the OV family (round-3 handoff):** accumulator count is MONOTONIC
through every step of: Overlay-mode entry (seeded with the displayed trace) → resident click →
evicted click → hydration completion → repaint. Only `Clear`, a new scan/source (`reset_key`),
or a REAL norm-channel change may reset it.
