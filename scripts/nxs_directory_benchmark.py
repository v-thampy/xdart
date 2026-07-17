#!/usr/bin/env python
"""Repeatable headless NXS-directory reduction benchmark (v1.1.2 Phase M0).

Establishes a before/after BASELINE for Image-Directory NXS ingestion by driving
the HEADLESS discover -> probe -> open -> reduce pipeline (``xrd_tools.sources`` +
``xrd_tools.reduction`` — the same streaming engine the GUI uses by default),
emitting the :mod:`xrd_tools.perf.metrics` schema as JSON.

M0 is behavior-neutral: it changes no runtime code and does not touch the GUI
wrangler.  It measures only facts observable at real, harness-owned seams:

* exactly ``--cores N`` reduction workers (a harness-owned ``ThreadPoolExecutor``
  the harness also shuts down) — never the engine's adaptive default;
* frames *written* from real ``sink.write()`` events (a harness sink wrapper);
* frames *reduced* from ``ReductionResult.n_processed``; frames *durable* from an
  on-disk re-count after ``sink.finish()``; frame *submit/accept* is left ``None``
  (that engine seam is not exposed without runtime edits);
* first-frame latency = run clock (started BEFORE enumeration) -> the first real
  sink write (no warm pre-decode);
* Python-level ``h5py.File`` constructions categorized by path
  (source / output / verification / other / failed) — a wrapper-visible proxy.

Destination safety (Tranche A ``check_output_not_source``): every generated NeXus
output and ``--json-out`` is validated against every source path, the PONI, and
the source directory before any write; outputs go to a fresh per-repeat
subdirectory; canonical scan-stem collisions are rejected before reduction.

Usage::

    pixi run python scripts/nxs_directory_benchmark.py \\
      --source-dir <dir of raw .nxs> --poni <file.poni> \\
      --output-dir <TEMP dir> --mode 1d --cores 4 --repeat 3 --json-out out.json

Skips cleanly (exit 0) when the source/PONI are absent so it is CI-safe.
"""
from __future__ import annotations

import argparse
import contextlib
import logging
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger("nxs_directory_benchmark")

_MASTER_EXTS = ("h5", "hdf5")


# ---------------------------------------------------------------------------
# small path helpers
# ---------------------------------------------------------------------------

def _norm(path) -> str:
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def _scan_name(path: Path) -> str:
    """Canonical container -> scan name (full stem; strip an Eiger ``_master``)."""
    stem = path.stem
    return stem[:-7] if stem.lower().endswith("_master") else stem


def _git_sha(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# categorized, scoped h5py.File construction counter (M0-R5)
# ---------------------------------------------------------------------------

class H5FileCounter:
    """Count Python-level ``h5py.File`` constructions inside the block, by path.

    A measurement technique (no library change): temporarily wraps
    ``h5py.File.__init__`` and classifies each attempt:

    * ``source`` — the container MASTER itself (a path in *master_paths*); the
      re-open count R2's single-open cursor must collapse;
    * ``source_external`` — the master's external-linked detector data files
      (source-side reads that are not the master);
    * ``output`` — under the generated *output_root*;
    * ``verification`` — durability re-reads (gated by :attr:`verifying`);
    * ``other`` — an unresolvable path;
    * ``failed`` — the constructor raised.

    Restored on exit, including on error.  Counts Python-level ``h5py.File(...)``
    attempts — a wrapper-visible PROXY: low-level data-file/external-link opens
    performed inside libhdf5 may not pass through this Python constructor.
    """

    def __init__(self, master_paths, output_root) -> None:
        self._masters = {_norm(p) for p in master_paths}
        self._output = _norm(output_root)
        from xrd_tools.perf.metrics import new_open_counts
        self.counts = new_open_counts()
        self.verifying = False
        self._orig = None

    def _classify(self, path) -> str:
        if self.verifying:
            return "verification"
        if path is None:
            return "other"
        try:
            rp = _norm(path)
        except Exception:
            return "other"
        if rp in self._masters:
            return "source"
        if rp == self._output or rp.startswith(self._output + os.sep):
            return "output"
        return "source_external"

    def __enter__(self) -> "H5FileCounter":
        import h5py
        self._orig = h5py.File.__init__
        counter = self

        def _wrapped(fself, *a, **k):
            path = a[0] if a else k.get("name")
            try:
                result = counter._orig(fself, *a, **k)
            except BaseException:
                counter.counts["failed"] += 1
                raise
            counter.counts[counter._classify(path)] += 1
            return result

        h5py.File.__init__ = _wrapped  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: object) -> None:
        import h5py
        if self._orig is not None:
            h5py.File.__init__ = self._orig  # type: ignore[assignment]

    @contextlib.contextmanager
    def verification(self):
        """Mark opens inside the block as durability *verification*."""
        prev = self.verifying
        self.verifying = True
        try:
            yield
        finally:
            self.verifying = prev


# ---------------------------------------------------------------------------
# harness-owned measuring sink wrapper (M0-R2 / M0-R3)
# ---------------------------------------------------------------------------

class MeasuringSink:
    """Wrap a real reduction sink to observe honest lifecycle events.

    Records the perf-counter timestamp of each real ``write()`` (the first is the
    run's first-write event for first-frame latency), counts writes
    (frames_written), and times the real ``finish()`` call (finish_s).
    All other sink hooks the engine probes (``worker_process``, ``abort``,
    ``replace``, ``flush``, …) delegate straight to the wrapped sink via
    ``__getattr__`` so its behavior is unchanged.
    """

    def __init__(self, real) -> None:
        self._real = real
        self.write_times: list[float] = []
        self.finish_s: float | None = None
        self.first_write_ts: float | None = None

    def begin(self, scan, plan) -> None:
        return self._real.begin(scan, plan)

    def write(self, frame, reduction) -> None:
        # Only a SUCCESSFUL real write counts: call the wrapped sink FIRST; a
        # sink that raises records neither a write nor a first-write timestamp
        # (the exception propagates unchanged).
        result = self._real.write(frame, reduction)
        ts = time.perf_counter()
        if self.first_write_ts is None:
            self.first_write_ts = ts
        self.write_times.append(ts)
        return result

    def finish(self, result) -> None:
        t0 = time.perf_counter()
        try:
            return self._real.finish(result)
        finally:
            self.finish_s = time.perf_counter() - t0

    def __getattr__(self, name):
        # delegate optional hooks (worker_process/abort/replace/flush/...) so the
        # wrapped sink's behavior is preserved exactly.
        return getattr(self._real, name)


# ---------------------------------------------------------------------------
# discovery + probing (headless)
# ---------------------------------------------------------------------------

def _enumerate(source_dir: Path, ext: str, recursive: bool) -> list[Path]:
    """Name-only enumeration matching the wrangler's discovery suffix rule."""
    ext = ext.lower().lstrip(".")
    suffix = f"_master.{ext}" if ext in _MASTER_EXTS else f".{ext}"
    it = source_dir.rglob("*") if recursive else source_dir.glob("*")
    files = [p for p in it if p.is_file() and p.name.lower().endswith(suffix)]
    return sorted(files, key=lambda p: p.name)


def _durable_frame_count(output_path: Path) -> int:
    """Frames actually on disk after the sink finished (durability, not submit)."""
    import h5py
    import numpy as np

    if not output_path.exists():
        return 0
    labels: set[int] = set()
    try:
        with h5py.File(output_path, "r") as h5:
            e = h5.get("entry")
            if e is None:
                return 0
            for g in ("integrated_1d", "integrated_2d"):
                grp = e.get(g)
                if grp is not None and "frame_index" in grp:
                    labels.update(
                        int(v) for v in np.asarray(grp["frame_index"][()]).ravel())
    except OSError:
        return 0
    return len(labels)


# ---------------------------------------------------------------------------
# destination safety (M0-R4)
# ---------------------------------------------------------------------------

def _validate_destination(dest, *, sources, poni, source_dir, recursive):
    """Raise ``OutputCollisionError`` if *dest* collides with a source/PONI/tree.

    Uses the released Tranche A guard (never a weaker local comparison): *dest*
    must not be any source file, the PONI, or lie in the watched source tree.
    """
    from xrd_tools.io.output_safety import check_output_not_source
    inputs = [str(poni)] + [str(s) for s in sources]
    check_output_not_source(
        str(dest),
        input_files=inputs,
        watched_dirs=[str(source_dir)],
        recursive=recursive,
        container_directory_mode=True,
    )


# ---------------------------------------------------------------------------
# per-container measurement
# ---------------------------------------------------------------------------

def _measured_nexus_stack_source(path, *, entry, metrics, cursor=None):
    """Build an instrumented public source without retaining detector arrays.

    The harness observes values at the source's real ``metadata_for`` and
    ``iter_chunks`` seams.  It deliberately wraps no reduction implementation:
    production still owns cursor lifetime, planning, decode, submit, and sink
    ordering.
    """

    from xrd_tools.sources.nexus import NexusStackSource

    class MeasuredNexusStackSource(NexusStackSource):
        def __init__(self, path, *, entry, metrics):
            self._metrics = metrics
            self.cursor_consumptions = 0
            super().__init__(path, entry=entry, cursor=cursor)

        def metadata_for(self, index):
            self._metrics.metadata_reads = (
                int(self._metrics.metadata_reads or 0) + 1)
            return super().metadata_for(index)

        def iter_chunks(self, chunk_size):
            self.cursor_consumptions += 1
            for images, labels in super().iter_chunks(chunk_size):
                nbytes = int(getattr(images, "nbytes", 0))
                self._metrics.block_reads = (
                    int(self._metrics.block_reads or 0) + 1)
                self._metrics.source_logical_bytes = (
                    int(self._metrics.source_logical_bytes or 0) + nbytes)
                self._metrics.retained_owner_bytes_peak = max(
                    int(self._metrics.retained_owner_bytes_peak or 0), nbytes)
                yield images, labels

    return MeasuredNexusStackSource(path, entry=entry, metrics=metrics)

def _bench_container(path, poni, poni_path, repeat_dir, output_root, plan, cores,
                     entry, frame_limit, source_dir, source_paths, recursive):
    """Measure one container. Returns the ContainerMetrics (first-write ts is on
    ``cm.reserved['first_write_ts']`` when a frame was written).

    ``poni`` is the loaded calibration (for the integrator); ``poni_path`` is the
    real filesystem path threaded into destination safety so the released guard
    actually checks the PONI file (M0-R4 correction)."""
    from xrd_tools.perf.metrics import ContainerMetrics, Timer
    from xrd_tools.io.output_safety import OutputCollisionError
    from xrd_tools.core.staging import source_block_budget_bytes
    from xrd_tools.sources.cursor import ContainerCursor
    from xrd_tools.sources.descriptor import describe_container
    from xrd_tools.sources.read_plan import plan_reads
    from xrd_tools.reduction import run_reduction, NexusSink
    from xrd_tools.integrate.calibration import poni_to_integrator

    scan_name = _scan_name(path)
    cm = ContainerMetrics(path=str(path), scan_name=scan_name)
    counter = H5FileCounter([path], output_root)

    with counter:
        # EXPLICIT PROBE: one master open builds the descriptor (readiness +
        # layout facts: frame count, shape, native dtype, chunks) with no pixel
        # decode.  This one open supersedes the old probe(2 opens) + dataset
        # resolve(1) + layout(1) — the readiness verdict stays separately
        # observable in ``probe_s`` and ``state``.
        with Timer() as t:
            descriptor = describe_container(path, entry=entry)
        cm.probe_s = t.elapsed
        cm.state = descriptor.state.value
        cm.nframes = descriptor.frame_count
        cm.frame_shape = descriptor.frame_shape
        cm.dtype = None if descriptor.dtype is None else str(descriptor.dtype)
        cm.chunks = descriptor.chunks
        if cm.state != "ready":
            cm.skip_reason = descriptor.reason
            cm.open_counts = dict(counter.counts)
            return cm
        cm.frames_discovered = int(descriptor.frame_count or 0)

        try:
            if descriptor.dataset_path is None:
                cm.state, cm.skip_reason = "imageless", "no image dataset"
                cm.open_counts = dict(counter.counts)
                return cm

            # destination safety BEFORE any write (M0-R4); the real PONI PATH is
            # checked here, not the loaded calibration object.
            out_path = repeat_dir / f"{scan_name}.nxs"
            try:
                _validate_destination(out_path, sources=source_paths,
                                      poni=poni_path, source_dir=source_dir,
                                      recursive=recursive)
            except OutputCollisionError as exc:
                cm.state, cm.skip_reason = "invalid", f"unsafe output: {exc}"
                cm.open_counts = dict(counter.counts)
                return cm
            if out_path.exists():  # preflighted in main(); belt-and-suspenders
                cm.state, cm.skip_reason = "invalid", "output already exists"
                cm.open_counts = dict(counter.counts)
                return cm

            limit = (descriptor.frame_count if frame_limit is None
                     else min(frame_limit, descriptor.frame_count))
            # The source computes this same layout-aware plan at its actual
            # consumption seam.  Recording the expected decision here provides
            # per-container observability without opening or decoding pixels.
            read_plan = plan_reads(
                descriptor.frame_count, descriptor.frame_shape, descriptor.dtype,
                descriptor.chunks, source_block_budget_bytes(),
                frame_interval=(0, limit), two_d=descriptor.is_2d)
            cm.read_plan_block_frames = read_plan.block_frames
            cm.read_plan_chunk_aligned = read_plan.chunk_aligned
            cm.read_plan_fallback_reason = read_plan.fallback_reason or None

            integrator = poni_to_integrator(poni)
            # Construct the real public source.  It only gathers source facts;
            # the streamed reduction below owns the live cursor and reads one
            # bounded owner block at a time.  No ``ScanFrame.image`` list is
            # assembled here, so RSS cannot grow with detector frame count.
            source_cursor = None
            with Timer() as t:
                source_cursor = ContainerCursor(path, entry=entry).open()
                source = _measured_nexus_stack_source(
                    path, entry=entry, metrics=cm, cursor=source_cursor)
                source.name = scan_name
                source.integrator = integrator
                # ``NexusStackSource`` has contiguous 0-based labels.  Limit
                # its source manifest before run_reduction materializes frames;
                # its read plan uses the same [0, limit) interval.
                source._frame_indices = source._frame_indices[:limit]
            cm.open_s = t.elapsed
            # Keep the v2 timing boundary honest: materialize metadata before
            # reduction, but leave pixels to the public streaming producer.
            # The source caches this provider, so the later canonical Scan
            # construction incurs no per-frame master reopen.
            with Timer() as t:
                if limit:
                    source.metadata_for(0)
            cm.metadata_s = t.elapsed
            cm.frames_input = int(limit)
        except Exception as exc:
            if "source_cursor" in locals() and source_cursor is not None:
                source_cursor.close()
            # a SETUP failure (before a sink exists): nothing was written.
            cm.state = "invalid"
            cm.error = f"{type(exc).__name__}: {exc}"[:200]
            cm.skip_reason = cm.skip_reason or "setup error"
            logger.debug("container setup failed: %s", path, exc_info=True)
            cm.open_counts = dict(counter.counts)
            return cm

        # reduction with EXACTLY ``cores`` harness-owned workers (M0-R1).  A
        # later failure must NOT discard writes/timings already observed: harvest
        # the wrapped sink in a finally, and leave frames_reduced=None unless an
        # honest ReductionResult is returned (never inferred).
        sink = MeasuringSink(NexusSink(path=out_path, overwrite=True,
                                      source_base=repeat_dir))
        executor = ThreadPoolExecutor(max_workers=cores)
        reduce_timer = Timer()
        result = None
        try:
            with reduce_timer:
                result = run_reduction(
                    plan, source, sink=sink, execution="streaming",
                    chunk_size=read_plan.block_frames, executor=executor)
        except Exception as exc:
            cm.state = "invalid"
            cm.error = f"{type(exc).__name__}: {exc}"[:200]
            cm.skip_reason = cm.skip_reason or "reduction error"
            logger.debug("reduction failed: %s", path, exc_info=True)
        finally:
            executor.shutdown(wait=True)  # shutdown belongs to the harness
            source.close()
            cm.reduce_s = reduce_timer.elapsed
            cm.frames_written = len(sink.write_times)   # observed SUCCESSFUL writes
            cm.finish_s = sink.finish_s
            if sink.first_write_ts is not None:
                cm.reserved["first_write_ts"] = sink.first_write_ts
            cm.cursor_opens = int(source.cursor_consumptions)
        if result is not None:
            cm.frames_reduced = int(getattr(result, "n_processed", 0))
            if getattr(result, "failed", False):
                cm.error = cm.error or getattr(result, "error", None)
        # else: no honest ReductionResult -> frames_reduced stays None.

        # durability: an INDEPENDENT post-finish on-disk fact (verification opens).
        with counter.verification():
            cm.frames_durable = _durable_frame_count(out_path)

    cm.open_counts = dict(counter.counts)
    return cm


# ---------------------------------------------------------------------------
# one full pass
# ---------------------------------------------------------------------------

def run_once(args, poni, plan, repeat_index) -> "object":
    from xrd_tools.perf.metrics import EnvProvenance, RunMetrics

    source_dir = Path(args.source_dir)
    output_root = Path(args.output_dir)
    poni_path = Path(args.poni)
    run_start = time.perf_counter()  # BEFORE enumeration (M0-R3)

    rm = RunMetrics(
        mode=args.mode, cores=args.cores,
        source_dir=str(source_dir), output_root=str(output_root),
        recursive=args.recursive,
        env=EnvProvenance.capture(git_sha=_git_sha(Path(__file__).resolve().parents[1])),
    )

    from xrd_tools.perf.metrics import Timer
    with Timer() as t:
        candidates = _enumerate(source_dir, args.ext, args.recursive)
    rm.enumerate_s = t.elapsed
    rm.n_candidates = len(candidates)
    source_paths = [str(p) for p in candidates]

    # fresh, unique per-repeat destination (M0-R4).
    repeat_dir = output_root / f"repeat_{repeat_index:02d}"
    repeat_dir.mkdir(parents=True, exist_ok=False)

    # canonical scan-stem collision detection BEFORE reduction (M0-R4).
    by_stem: dict[str, list[Path]] = {}
    for p in candidates:
        by_stem.setdefault(_scan_name(p), []).append(p)
    colliding = {name for name, ps in by_stem.items() if len(ps) > 1}

    first_write_latency: float | None = None
    for path in candidates:
        from xrd_tools.perf.metrics import ContainerMetrics
        name = _scan_name(path)
        if name in colliding:
            rm.add(ContainerMetrics(
                path=str(path), scan_name=name, state="invalid",
                skip_reason="duplicate canonical scan stem"))
            continue
        cm = _bench_container(
            path, poni, poni_path, repeat_dir, output_root, plan, args.cores,
            args.entry, args.frame_limit, source_dir, source_paths, args.recursive)
        rm.add(cm)
        fw = cm.reserved.pop("first_write_ts", None)
        if fw is not None:
            rel = fw - run_start
            if first_write_latency is None or rel < first_write_latency:
                first_write_latency = rel
        out_path = repeat_dir / f"{name}.nxs"
        if cm.state == "ready" and out_path.exists():
            rm.outputs.append(str(out_path))

    rm.first_frame_latency_s = first_write_latency
    rm.total_s = time.perf_counter() - run_start
    return rm


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def _build_plan(mode: str):
    from xrd_tools.reduction import Integration1DPlan, Integration2DPlan, ReductionPlan

    mode = mode.lower()
    if mode == "1d":
        return ReductionPlan(integration_1d=Integration1DPlan(), integration_2d=None)
    if mode == "2d":
        return ReductionPlan(integration_1d=None, integration_2d=Integration2DPlan())
    if mode == "both":
        return ReductionPlan(
            integration_1d=Integration1DPlan(), integration_2d=Integration2DPlan())
    raise SystemExit(f"--mode must be 1d|2d|both, got {mode!r}")


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-dir", required=True, help="directory of raw NXS containers")
    p.add_argument("--poni", required=True, help="PONI calibration file")
    p.add_argument("--output-dir", default=None,
                   help="output ROOT (must be OUTSIDE source-dir); a fresh temp "
                        "root is used if omitted. Each repeat writes to a unique "
                        "repeat_NN/ subdirectory.")
    p.add_argument("--ext", default="nxs", help="container extension (nxs|h5|hdf5)")
    p.add_argument("--mode", default="1d", help="1d | 2d | both")
    p.add_argument("--cores", type=int, default=1,
                   help="EXACT reduction worker count (>=1)")
    p.add_argument("--repeat", type=int, default=1, help="repeat passes for median/range")
    p.add_argument("--recursive", action="store_true", help="descend into subdirectories")
    p.add_argument("--frame-limit", type=int, default=None,
                   help="cap frames per container (quick smoke runs)")
    p.add_argument("--entry", default="entry", help="NXentry group name")
    p.add_argument("--json-out", default=None, help="write the full JSON result here")
    return p.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    # reject non-positive core counts BEFORE opening data or outputs (M0-R1).
    if args.cores < 1:
        logger.error("--cores must be >= 1 (got %d)", args.cores)
        return 2
    if args.repeat < 1:
        logger.error("--repeat must be >= 1 (got %d)", args.repeat)
        return 2

    source_dir = Path(args.source_dir)
    poni_path = Path(args.poni)
    if not source_dir.is_dir():
        logger.info("SKIP: source dir not found: %s", source_dir)
        return 0
    if not poni_path.exists():
        logger.info("SKIP: PONI not found: %s", poni_path)
        return 0

    tmp_ctx = None
    if args.output_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="nxs_bench_out_")
        args.output_dir = tmp_ctx.name

    from xrd_tools.io.output_safety import (
        OutputCollisionError,
        check_output_not_source,
        paths_same_file,
    )
    from xrd_tools.integrate.calibration import load_poni
    from xrd_tools.perf.metrics import summarize_runs, write_json

    output_root = Path(args.output_dir)
    candidates = _enumerate(source_dir, args.ext, args.recursive)
    source_paths = [str(p) for p in candidates]
    scan_names = sorted({_scan_name(p) for p in candidates})

    # Plan EVERY destination for EVERY repeat before the first writer opens
    # (M0-R4): the output root, each repeat directory, and each generated .nxs.
    repeat_dirs = [output_root / f"repeat_{i:02d}" for i in range(args.repeat)]
    generated = [rd / f"{n}.nxs" for rd in repeat_dirs for n in scan_names]

    # 1. output root + every generated output must be OUTSIDE the source tree
    #    (recursive=True, so a source subdirectory is refused too) and must not
    #    clobber a source or the PONI path.
    try:
        for dest in [output_root / "probe.nxs", *generated]:
            check_output_not_source(
                str(dest), input_files=[str(poni_path), *source_paths],
                watched_dirs=[str(source_dir)], recursive=True,
                container_directory_mode=True)
    except OutputCollisionError as exc:
        logger.error("REFUSED output: %s", exc)
        return 2

    # 2. --json-out must not equal the output root, a planned repeat directory,
    #    any planned generated .nxs, a source, or the PONI — nor sit in the
    #    source tree.  A clean refusal writes nothing.
    if args.json_out:
        jo = Path(args.json_out)
        forbidden = [output_root, *repeat_dirs, *generated, poni_path,
                     *(Path(s) for s in source_paths)]
        if any(paths_same_file(jo, f) for f in forbidden):
            logger.error("REFUSED --json-out: collides with a planned output / "
                         "source / PONI: %s", jo)
            return 2
        try:
            check_output_not_source(
                str(jo), input_files=[str(poni_path), *source_paths],
                watched_dirs=[str(source_dir)], recursive=True,
                container_directory_mode=True)
        except OutputCollisionError as exc:
            logger.error("REFUSED --json-out: %s", exc)
            return 2

    # 3. refuse a pre-existing repeat directory or generated output — no clobber,
    #    no partial outputs; every repeat uses a fresh unique destination (this is
    #    the existing-output rejection driven through the real CLI path).
    for existing in [*repeat_dirs, *generated]:
        if existing.exists():
            logger.error("REFUSED: benchmark output already exists: %s", existing)
            return 2

    poni = load_poni(poni_path)
    plan = _build_plan(args.mode)

    runs = []
    try:
        for i in range(args.repeat):
            rm = run_once(args, poni, plan, i)
            logger.info("%s (repeat %d/%d)", rm.one_line_summary(), i + 1, args.repeat)
            runs.append(rm)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    payload = {"summary": summarize_runs(runs), "runs": [r.to_dict() for r in runs]}
    if args.json_out:
        logger.info("wrote %s", write_json(args.json_out, payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
