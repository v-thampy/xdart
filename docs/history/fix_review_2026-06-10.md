# Fresh-Eyes Review of the Fix Round — ssrl b111e5b..75b4765, xdart fc080b4..f185341

**Date:** 2026-06-10 · Follow-up to `deep_review_2026-06-09.md`
**Method:** two independent adversarial review agents (one per repo) reading the full diffs plus surrounding
current-tree code, with the headline cross-repo finding re-verified by hand afterward. ssrl's suite was run in
the review sandbox: **819 passed, 7 skipped** (delta vs the claimed 824 is environmental — pyFAI version /
optional-dep skips, zero real failures). xdart's suite could not be run in the sandbox (no Qt/pyFAI stack);
run `conda run -n xrd_edit python -m pytest tests/ -q` on the Mac before tagging.

---

## Bottom line

The fix round is high quality: **every commit does what its message claims**, all the prior release-gate items
(G1 wavelength, G2 @source_base) and streaming-hardening items (S1/S2/S6/S8, C1, T0-5/6/7) are genuinely fixed,
and the new tests are revert-sensitive (they'd fail if the fixes were backed out). No semaphore double-release,
no lost `task_done`, no new deadlock, no lock-ordering inversion from `command_lock` (it's a leaf lock).

**Two items remain before I'd call the pair release-ready:**

1. **[V, medium] Unprotected `submit()` raise tears down the wrangler QThread and bypasses the fail-loud
   surfacing chain.** `ReductionSession.submit()` re-raises a recorded `_failure` at its precheck
   (`reduction/core.py`, `if self._failure is not None: raise self._failure`) — and its *own* timed-acquire
   comment explains why raising from submit is dangerous: "raising here escapes through `run()` which has no
   except, tearing down the QThread — the same trap the GIFreezeError fix addressed." But the `_failure`
   re-raise (and the new dead-writer detection that routes through it) does exactly that: the streaming
   dispatch loop calls `session.submit()` bare (`image_wrangler_thread.py:1600`), `_dispatch_batch` call sites
   (~:670/:739/:752) are unwrapped, and `imageThread.run()` (:519) has no outer except. So a mid-run sink
   write failure — including the **new G2 `@source_base` RuntimeError** — is recorded by the writer thread,
   then the *next* submit raises it into `run()`: unhandled-exception teardown, no "Save FAILED" label, no
   `_close_reduction_session`, session never finished, output left in tmp/partial state without the loud
   chain that BLOCKER-2 built. The existing fail-loud test exercises the `finish()`-time path only.
   *Fix (small):* wrap the submit (or the dispatch loop) in try/except that records the error, sets
   `command='stop'` under `command_lock`, and routes through `_close_reduction_session` — i.e., the same
   treatment the GIFreezeError and timed-acquire paths already got. Add a mid-run-failure test.
2. **[V, medium-low] P2 "GI provenance disclosure" doesn't persist on the real GUI path.** The
   `gi_freeze_diagnostic` stamp is gated by `_write_reduction`'s `is_replace or finalize or "reduction" not in
   entry` condition (`nexus_writer.py:281`), but `initialize_scan()` writes `/entry/reduction` *before* the
   prepass sets the diagnostic, subsequent sink flushes pass `finalize=False`, and the image wrangler never
   passes `finalize=True`. So on the actual streaming GI batch path the diagnostic never reaches the file —
   the unit test passes vacuously because its first save already carries the stamp. Log + status label still
   fire, so this is disclosure-persistence only, not data corruption. *Fix:* include
   "`gi_freeze_diagnostic` newer than what's on disk" in the `_write_reduction` gate, or have the stamp
   trigger a one-shot reduction rewrite — then make the test exercise initialize-then-stamp ordering.

---

## Per-fix verdicts — ssrl_xrd_tools (→ 0.41.0)

| Fix | Verdict | Notes |
|---|---|---|
| #2 cancel-aware `submit()` | **Correct [V]** | Semaphore accounting sound (timed-acquire returns slotless; post-acquire cancel releases). Test pins it. |
| #4 bounded `finish()` join | **Correct [V]** | Timeout trips cancel token, records `TimeoutError` (preserves earlier failure via `or`), idempotent re-call. Cosmetic: double RuntimeWarning on timeout. |
| T0-5 no sink teardown while writer alive | **Correct [V]** | finish/abort skipped on `_writer_timed_out`; spy-sink test pins both. |
| T0-5b timed-out finish names data location | **Correct [V]** | `_tmp_path → _active_path → path` resolves in both modes. Degrades to no location for `CompositeSink` — `_output_path` would be a better last fallback (minor). |
| T0-6 abort preserves `.partial` | **Correct [V]** | Unlink → `tmp.replace(<output>.partial)`; finish-failure path verified to route through abort with `_tmp_path` still set. Nits: successive failures overwrite the same `.partial`; zero-frame failure still leaves one. |
| T0-7 writer survival | **Correct [V]** | Post-write else-clause wrapped; `int(frame.index)` moved inside try. Dead writer with free slots can silently swallow up to `inflight_max` frames before submit trips — bounded, eventually loud, acceptable. Feeds remaining-item #1 above. |
| S2 retention + `release_products` | **Correct [V]** | Replace detection fully migrated to `_seen_idxs` (both executors, grep-verified); release keeps `_seen_idxs` so re-feed idempotency holds, test-pinned. **API semantic change:** streaming+durable-sink `run_reduction` now returns `result.frames == {}` — documented + tested, but deserves a release-note line. Edge: `sink=[]` coerces to an unreachable internal MemorySink with retain=False → products unrecoverable (obscure, low). |
| C1 schema-version check | **Correct-with-caveats [V]** | `warn_if_newer_schema` warns only on `ver > 2`; absent/legacy stamps silent (parametrized test incl. None). Covers `read_scan`, `read_scan_metadata` (→ `get_metadata`, `open_scan().metadata`), `FrameViewReader`. **Gap:** `get_1d/get_2d/get_thumbnail/get_frames` open the file directly and never check — the notebook-facing readers are exactly where a v3 user would hit the opaque KeyError. Extend (cheap). |
| S8 monitor warning | **Correct-with-caveats [V]** | Non-castable values now also warn. **Caveat:** `_warned_monitor_keys` is module-level → once per key per *process*: in a long-lived GUI session, scan #1's dead `i0` permanently silences the warning for all later scans using `i0`. Key by `(scan_name, key)` or per-session set. The new tests dodge this with unique key names. |
| S6 SWMR | **Neutralized, not implemented [V]** | Right call: `open_nexus_writer(swmr=True)` raises `NotImplementedError` before file creation (test asserts no file left); docstring no longer advertises it. Residue: `NexusSink.swmr` field survives with a misleading error message and a dead term in the atomic-mode decision — deprecate/remove. |
| G4 pyFAI pin | **Correct [V]** | Both repos now `>=2025.3,<2025.12`. Side-finding [S]: no cp310 wheels for pyFAI ≥2025.3 → `requires-python >= 3.10` implies source builds on 3.10; consider `>=3.11`. |

## Per-fix verdicts — xdart (→ 0.40.0)

| Fix | Verdict | Notes |
|---|---|---|
| G1 wavelength sentinel | **Correct [V]** | New `modules/wavelength.py` is clean; tolerance matches the old writer guard exactly. Writer (`instrument/source/wavelength_A`) and v2 reader read/write the same path through the already-open handle. All three former trust points fixed (display chain, `sources.py`, `reduction.py`); only remaining raw consumer is the instrument-change fingerprint (harmless). Lifecycle complete: cleared on reset, v2-load entry, both `set_datafile` repoints, and synchronous `new_scan`. Genuine 1.0 Å beams are trusted from every authoritative source; rejected only in the mg_args-only legacy path — correct trade, tested. Minor [S]: the stamp rides the instrument-signature rewrite gate; an exotic frames-but-no-integrator path could defer it to finalize (not reachable from the GUI today). |
| G2 @source_base stamp | **Correct [V]** | Now `raise RuntimeError`, consistent with the append-conflict rejection; streaming path surfaces via writer-failure → finish → "Save FAILED" + stop. Caveats: swallowed (loudly logged) by `_enter_pause`'s catch-all during a pause flush; serial-path raise escapes `run()` — which is remaining-item #1, not new exposure. Test would fail on revert. |
| Runtime version guard | **Correct-with-caveats [V]** | Hard `SystemExit` in `main()`; warns instead of failing when the capability probe passes despite a stale editable version stamp — right for the editable workflow. Floor-sync test pins `MIN_SSRL_VERSION ==` pyproject floor (0.41.0, matches ssrl). **Gap:** probe checks `relative_source_path`/`drain`/`finish(join_timeout)` but not `retain_products`, which xdart passes as a hard kwarg — a mid-range editable ssrl passes the probe then TypeErrors at session open. Add it to the probe. |
| T0-8 sink registry lifecycle | **Correct [V]** | finish/abort clear `_registry`; both run only after writer join, no race with `write()`. On writer-join timeout the registry isn't cleared but the session close drops the sink reference — nothing leaks past close. |
| P1 fail-loud sink `write()` | **Correct [V]** | Raise recorded by writer loop, surfaces at finish → stop. `register()` strictly precedes `submit()`; cancelled submits leave registry entries that T0-8 now clears. `replace()` stays silent-miss with fallback — acceptable for reintegration. |
| T0-9/RS-1 pause drain | **Correct [V]** | Drain-timeout skips save/flush AND the `_frames_since_save` reset (persist-before-evict preserved); `sigPaused` still fires. Residual [S, low]: after a timeout the browse guard lifts while the writer is provably non-idle — every write path still holds `file_lock` + h5pool pause so reads coordinate; document or re-engage the guard on timeout. Test nails the exact contract. |
| RS-2 `command_lock` TOCTOU | **Correct [V]** | Leaf lock, never nested with `file_lock`/`data_lock`/h5pool. Both worker self-stop sites write under it; `_on_resume` emits outside the lock and re-checks. Cosmetic: a stop landing between emit and re-check leaves the button on "Resume" until `finished` — transient. |
| T0-4 GI abort → warn-and-proceed | **Implemented as recorded policy [V]** | The fail-open is deliberate and bounded: T0-3's `_gi_ranges_fully_pinned` preflight (correctly mirrors both freeze self-skip conditions) removes the false aborts; freeze *errors* stay fail-closed; disclosure = log + elide-guarded status label + provenance stamp. But see remaining-item #2: the provenance stamp doesn't actually persist on the real path. Stale "TEST-ONLY" comment block at `image_wrangler_thread.py:135-144` should go — the `_freeze_gi_*` wrappers are production again. |
| P2 GI provenance disclosure | **NOT fixed on the real path [V]** | Remaining-item #2 above. |
| UI-1 checkbox toggles + blow-out guard | **Correct-with-caveats [V]** | No signal loops (pyqtgraph `setOpts` filters unchanged values; reassert/sync helpers no-op on equal state); chevron peek doesn't enable; width guard (Ignored policy + elide + tooltip) routed through both wranglers' `showLabel`. Cosmetics [S]: `setFlags` emits `itemChanged`, so a group `setOpts` can re-expand a manually collapsed enabled group; status elide is set-time only (no re-elide on resize). |
| S2 scoping + serial release | **Correct [V]** | `retain_products=False` only for streaming+sink; chunked keeps retention; no xdart caller reads streaming `session.frames` (grep-verified); harvest happens before `release_products`; retention-matrix test present. |

---

## Consolidated remaining items (priority order)

1. **[V, med]** Wrap the streaming dispatch `submit()` raise path (xdart) — restores the fail-loud chain for
   mid-run write failures; small fix + one test. *The* item to do before merge to dev.
2. **[V, med-low]** Persist `gi_freeze_diagnostic` on the real path (gate ordering in `_write_reduction`).
3. **[V, med-low]** S8 warning granularity: module-level once-per-process set silences later scans in a
   long-lived GUI; key per scan/session.
4. **[V, low]** Version-guard probe: add `retain_products`. C1 check: extend to `get_1d/get_2d/get_thumbnail`.
5. **[V, low]** Deprecate/remove `NexusSink.swmr` field + dead atomic-mode term.
6. **[S, low]** Pause drain-timeout browse-guard note; py3.10-vs-pyFAI-wheels floor decision; CompositeSink
   timed-out-finish location fallback; `.partial` overwrite/zero-frame nits; UI-1 re-expand + resize-elide
   cosmetics; stale TEST-ONLY comment.
7. **Process:** two new ssrl tests would *hang* (not fail) on revert (`test_finish_join_timeout_loud_on_stuck_worker`,
   `test_submit_detects_dead_writer`) — add `pytest-timeout` markers. Add a release-note line for the
   `result.frames == {}` semantic change.

## Release readiness

**ssrl 0.41.0: ready.** No blockers; items 3–5 are accept-or-fix-cheaply.
**xdart 0.40.0: ready after item 1** (and ideally 2). Run the full xdart suite on the Mac (not runnable in the
review sandbox), then proceed with the coordinated release, ssrl first.
