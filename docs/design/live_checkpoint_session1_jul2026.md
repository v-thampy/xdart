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
- [ ] D5 MS-1: pause mid-run — dispatch/processed/indexed counts reconcile in the post-live
      index line (no silent drop).

## E. H8/8a store-flip live checks
- [ ] E1 Store-only display sanity: reloaded scan draws Single, Overlay, Waterfall, Sum, and
      Average without scan-display `data_1d`/`data_2d` mirror reads.
- [ ] E2 Scroll-back to evicted frames: first paint from local disk feels responsive
      (budget: <1 s) and converges to the requested frame/trace set.
- [ ] E3 GI sub-mode switching: live mode switches and reload switches both draw the durable
      W-backed mode data without recomputation or stale-mode drift.
- [ ] E4 Explicit subset spanning evicted frames: Sum/Average hydrates every requested frame or
      refuses/blank-awaits; it never draws a resident-only partial subset.
- [ ] E5 Retired mirror telemetry stays absent: during the whole session there are no
      `display read ignored legacy mirror` lines (any hit means old code resurfaced).
- [ ] E6 GI sub-mode switching with mirrors gone: live and reload switching both draw the
      persisted W-backed per-mode data; same `(frame, mode)` view before/after reload.
- [ ] E7 Live/Append/Reintegrate sanity post-deletion: live run, Append, Reintegrate 1D,
      and Reintegrate 2D still publish through the store and repaint without scan-display
      row tables.
- [ ] E8 Viewer modes unchanged: XYE multi-select, Image Viewer, and NeXus preview keep
      their viewer-row behavior after the Role-A mirror retirement.

## F. Late-wave additions (2026-07-02, post-checklist fixes)
- [ ] F1 BR-1: click between processed .nxs files in the data browser with plots up — nothing
      blanks until a FRAME of the new scan is clicked; run-start still clears properly.
- [ ] F2 GI-1 (was P0): χGI axis, Auto range vs explicit −180..180 on the APS SiO2 file —
      identical 1D profile live (the offscreen gate says byte-equal; eyeball it once).
- [ ] F3 APS Zoe end-to-end with Meta Type = auto (fresh session default): .tif.metadata
      discovered, frame processes, metadata viewer POPULATED.
- [ ] F4 Append on the fully-processed 651-frame Eiger scan: ~instant, INFO (not WARNING)
      "already processed" line, post-live reconcile runs, viewer loads with last frame selected.
- [ ] F5 Shortcuts smoke: Cmd+R run/pause (with a focused editor: commits then runs),
      Cmd+Shift+C stops mid-run, Cmd+O / Cmd+S, Cmd+Shift+A blocked mid-run.
- [ ] F6 Wrong-file guard (LD-1): sweep onto evicted frames, immediately click a different
      scan — no frames from the previous file appear.
- [ ] F7 OV-6 live cross-scan overlay: Overlay/Waterfall scan A, then run/load scan B with the
      same 1D axis kind + npt — A traces stay and B traces append; frame-number collisions show
      both scan-qualified traces. Repeat with a different npt or axis kind — accumulator resets.
- [ ] F8 OV-6 browser judgment: with an overlay up, click a compatible processed `.nxs` in the
      data browser, then click a frame — prior traces stay and the clicked frame appends. Note
      whether this browsing behavior is desired long-term.

## G. Final fixed-unverified additions (2026-07-03)
- [ ] G1 CF-1/CF-2 Append config-mismatch BOTH ways: load Standard, flip to Grazing, Run in
      Append -> Run-click modal appears; No does not run; Yes flips to Replace and
      re-integrates. Repeat Grazing->Standard. Eiger `_master` target resolves to the same file;
      loaded data keeps stored units until Yes/Replace. CF-3 cold path: fresh xdart launch with NO
      viewer-loaded scan, partial Standard target already on disk (e.g. 210/651), panel = Grazing,
      Run in Append -> same modal appears before any frame processes; No leaves the file
      untouched and produces no duplicate writer traceback. Also run the worker-backstop variant
      with the source path not yet known/live-empty: it must abort before frame 1, preserving the
      existing target.
- [ ] G2 UI-2/UI-3 fresh-launch χ (c/w) slice toggle no-crash with no data + slice c/w + Pin
      re-enable after append/browser-loaded Int 2D cake arrives and disable again on clear.
- [ ] G3 OV-7 texture-cut workflow: Pin two χ/q centers on one frame, move live c/w — pinned cuts
      survive norm/BG/unit rebuilds; live current cut is styled gray/lightweight at ~75% opacity
      and does NOT consume palette colors; the live current sits one y-offset slot above the last
      pin, and Pin freezes it exactly there. In Single mode, Pin stays disabled.
- [ ] G4 MEM-1a long fast live run — peak RSS bounded, dropped labels re-hydrate.
- [ ] G5 MEM-1b GI scan RSS bounded, reload shows 1D present / dropped 2D ABSENT.
- [ ] G6 MEM-1c series-average Append onto existing output is BLOCKED with actionable message
      (not silent no-op).
- [ ] G7 MEM-1d Image Viewer Cmd+A select-all on a large stack stays bounded (LRU cap wins).
- [ ] G8 BW-A2 Append degraded-load drill: simulate/force an Append target read failure on an
      already-integrated `.nxs`; Run aborts before processing/writing, the target file remains
      byte-identical, and a normal Append load does not full-rewrite prior rows on first flush.
- [ ] G9 MEM-2 heavy-window sanity (`XDART_HEAVY_WINDOW` override respected; small-RAM tier).
- [ ] G10 MEM-3 651-frame baseline stays ~25-26 s at the new default worker cap.
- [ ] G11a RN-2 live-scale bulk overlay: 3600-frame overlay hydration does not hold `.nxs`
      read-only across writer flush retries.
- [ ] G11b BL-3 metadata auto-sidecar: with a junk companion present (a per-frame `.poni` /
      `.tif.log` / oversize / binary alongside the real `.tif.metadata`), the run locks onto the
      REAL sidecar — log shows `metadata: auto locked onto ...` for the right file, and motors/
      counters in the `.nxs` are correct (not junk). A 1-2 field explicit sidecar loads.
- [ ] G12 BL-6/S-17 overlay x-grid (F7/F8): overlay scan A, then run/load scan B with the SAME
      axis+npt but a DIFFERENT radial_range (recalibrate or edit the range) — scan B's peaks
      render at their CORRECT q/2θ (aligned to scan A's grid), not shifted; a frame with an empty
      grid does not blank the overlay.
- [ ] G13 OV-7b Pin-absorbs-current: in Overlay+slice, pin a couple of χ/q cuts on one frame,
      then Pin the CURRENT c/w — it becomes exactly ONE new pinned trace (no duplicate, no extra
      offset). Repro: pin χ=18±10 ⇒ TWO traces, not three. Spin the c/w to a new value ⇒ the
      live "current" cut REAPPEARS; re-dial back to a pinned value ⇒ it disappears again. The
      live current renders as a gray DASHED line only (no markers), ~75% opacity, "· current".
      OV-7c: with two pins, the current previews slot 2; Pin turns it solid/color in that same
      slot; the next moved current appears at slot 3.
- [ ] G14 S-14/S-18 overlay identity: (S-14) RE-RUN the same scan (same name) while overlaid —
      the new run's curves REPLACE the old (not dropped as stale under the old labels); a boundary
      to a DIFFERENT compatible scan still APPENDS (OV-6). (S-18) Pin a slice cut, then load a
      different scan — the pin does NOT rematerialize on the new scan's frame N under the old
      legend.
- [ ] G15 S-5/S-6 config re-key: (S-5, GI) switch the 1D GI mode (e.g. Qoop → χGI) with a
      frozen range set — the new mode is NOT clipped to a stale wedge (the range re-derives);
      switching q↔2θ in standard mode re-keys the radial range. (S-6, headless) a scan with a
      MIXED-CASE monitor key writes NORMALIZED data (map_norm matches the values).
- [ ] G16 S-16 norm-channel reset: in Overlay, change the Norm Channel mid-overlay — the
      accumulator RESETS (the previously-accumulated curves clear) so normalized and
      un-normalized traces are never mixed on one plot.
- [ ] G17 S-3 Append signature: Append onto an existing scan after changing chi_offset (or
      monitor / polarization / error model / GI incidence) but NOT the axis/npt/range — the
      Run-click modal now BLOCKS it (was silently mixed). Verify NO false modal when Appending onto
      a PRE-UPGRADE .nxs (written before this field existed) with an unchanged grid.
- [ ] G18 **S-4 SHIP GATE (written-data): real-data χ validation.** In STANDARD mode with a
      Mode-A (I-vs-χ) reduction + a non-zero chi_offset (default 90°), the written 1D χ axis must
      match the 2D cake χ AND the team χ reference. Run the `scripts/` validator against real
      del-only data before shipping S-4 — a written-data frame fix must not land on user files
      blind (mirror the GI-convention validation discipline). Headless 1D↔2D consistency is
      already green; this gate is the ABSOLUTE frame.
      **AMENDED (round-3): the harness MUST use an EXPLICIT PARTIAL χ range** (e.g. 0–90°, not
      Auto / full −180..180) — auto & full-domain masked the input-shift bug, so an auto-only
      validation would pass a broken build.  Verify the 1D peak at a known χ falls at the SAME χ
      in the 1D profile and the 2D cake (both offset-applied) for that partial range.
      **AMENDED (round-4): include a q/2θ-axis 1D explicit wedge using the same panel χ range**
      (for example q-axis 1D with χ 0–90° and chi_offset=90°) and compare it to the 2D cake's
      matching wedge.  This catches the q/2θ branch's input-shift regression; no 1D χ output
      relabel is involved.  Harness: `scripts/g18_s4_chi_validation.py` (use
      `--reference-chi` when running against the team absolute χ reference).
- [ ] G19 OV-7c waterfall slot: in Overlay+slice with 2 pins, the live "current" cut renders one
      step ABOVE the last pin (slot 2 = n_pins), not buried under pin #1's baseline. Pin it ⇒ the
      new pinned trace stays at the SAME y (no jump); the next current appears one slot higher.
      Zero pins ⇒ current at slot 0.

**PASS ⇒ proceed to RC-8** (merge → tag v1.0.0 → publish; recipe in
`handoff_chunks_jul2026.md`). Any FAIL: stop, report the item + `kill -USR1 <pid>` stack if a
freeze, fix-forward on `feature/remediation`, re-run only the failed section.
