# xrd_tools/perf/metrics.py
"""Headless metrics vocabulary for the NXS directory workstream (M0 baseline).

A *small* schema of event timings + counters plus environment provenance, JSON
round-trippable, so a before/after benchmark comparison is reproducible on one
machine.  Import-light and Qt-free: only stdlib at import time; heavy version
probes (pyFAI/h5py/psutil) are lazy and best-effort.

**Honesty over defaults (correction 2026-07-14).** Every lifecycle count and
timing records ONLY a fact the harness observes at a real, harness-owned seam.
A phase the harness cannot observe without editing production runtime code is
left ``None`` (not inferred from an input count, not aliased from another phase).
:data:`OBSERVATION_SOURCES` documents where each field comes from.

Field ownership across phases (unchanged): M0 populates the seams reachable from
the headless path; prefetch decode/block/bytes (R2), resource-build counts
(C1/C2), and finer submit/integrate split (S) are added by their owning phase,
never speculatively here.
"""

from __future__ import annotations

import json
import os
import platform as _platform
import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generator, Mapping

#: Bump when a field's *meaning* changes.  v2 = the honest-observation correction
#: (renamed/redefined lifecycle counts + categorized opens + first-write latency).
SCHEMA_VERSION = 2

#: Human-readable provenance of every observed field, surfaced in JSON so a
#: reader never mistakes an unobserved/None field for zero.
OBSERVATION_SOURCES: dict[str, str] = {
    "frames_discovered": "container image-dataset shape[0], read from HDF5 "
                         "metadata without decoding a frame",
    "frames_input": "frames the harness offered to run_reduction "
                    "(len(scan.frames) after any --frame-limit)",
    "frames_reduced": "ReductionResult.n_processed (engine-reported completions)",
    "frames_written": "count of real sink.write() invocations observed by the "
                      "harness-owned measuring sink wrapper",
    "frames_durable": "distinct integrated frame_index on disk, counted after "
                      "sink.finish() returns (durability verification)",
    "frames_submitted": "UNOBSERVED: the streaming engine's frame-accept seam is "
                        "not exposed without production runtime edits; left None",
    "finish_s": "per-container wall time measured around the wrapped "
                "sink.finish() call",
    "session_finish_total_s": "run-level sum of per-container finish_s",
    "first_frame_latency_s": "run clock started BEFORE enumeration -> timestamp "
                             "of the first real sink.write() across the run "
                             "(skipped candidates before it are included)",
    "open_counts": "Python-level h5py.File construction attempts, categorized by "
                   "resolved path, plus failed constructions; a WRAPPER-VISIBLE "
                   "PROXY — low-level libhdf5 external-link/data-file opens may "
                   "bypass the Python wrapper",
    "open_counts.source": "opens of the container MASTER file itself — the "
                          "wrapper-visible re-open count R2's single-open cursor "
                          "must collapse (the R2 before/after target)",
    "open_counts.source_external": "opens of the master's external-linked "
                                   "detector data files (source-side reads, "
                                   "distinct from master re-opens)",
    "open_counts.verification": "durability re-reads of the generated output "
                                "(EXCLUDED from the source R2 target)",
    "open_counts.other": "wrapper-visible h5py.File constructions with no "
                         "resolvable path (a FileID/low-level arg) — in this "
                         "pipeline these are predominantly libhdf5 external-link "
                         "data-file resolutions of the source surfaced as Python "
                         "File objects; counted but not path-attributable",
}

#: Categories of :attr:`ContainerMetrics.open_counts` / run aggregate.
#: ``source`` = container-master opens (the R2 single-open target); it is kept
#: separate from ``source_external`` (external-linked data files), ``output``
#: (generated output read/write), ``verification`` (durability re-reads),
#: ``other`` (unresolvable), and ``failed`` (constructor raised).
OPEN_CATEGORIES = (
    "source", "source_external", "output", "verification", "other", "failed")


# ---------------------------------------------------------------------------
# timing primitives
# ---------------------------------------------------------------------------

class Timer:
    """A restartable wall-clock stopwatch (``time.perf_counter``)."""

    __slots__ = ("_t0", "elapsed")

    def __init__(self) -> None:
        self._t0 = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._t0


@contextmanager
def timed(sink: dict[str, float], key: str) -> Generator[None, None, None]:
    """Accumulate the elapsed seconds of the block into ``sink[key]`` (+=)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        sink[key] = sink.get(key, 0.0) + (time.perf_counter() - t0)


def new_open_counts() -> dict[str, int]:
    """A zeroed categorized-open-count dict."""
    return {c: 0 for c in OPEN_CATEGORIES}


# ---------------------------------------------------------------------------
# environment provenance
# ---------------------------------------------------------------------------

@dataclass
class EnvProvenance:
    """Machine + library context a benchmark result must carry to be comparable."""

    git_sha: str | None = None
    python: str = ""
    platform: str = ""
    numpy: str | None = None
    pyfai: str | None = None
    h5py: str | None = None
    hdf5: str | None = None
    hdf5plugin: str | None = None
    cpu_count: int | None = None
    ram_gb: float | None = None

    @classmethod
    def capture(cls, *, git_sha: str | None = None) -> "EnvProvenance":
        """Best-effort snapshot; a missing optional library records ``None``."""
        def _ver(mod: str) -> str | None:
            # pyFAI exposes ``version`` (a str), most libs ``__version__``;
            # try both and only accept a string.
            try:
                m = __import__(mod)
                for attr in ("__version__", "version"):
                    v = getattr(m, attr, None)
                    if isinstance(v, str) and v:
                        return v
                return None
            except Exception:
                return None

        h5_hdf5 = None
        try:
            import h5py  # noqa: WPS433
            h5_hdf5 = ".".join(str(x) for x in h5py.h5.get_libversion())
        except Exception:
            h5_hdf5 = None

        ram_gb = None
        try:
            import psutil  # noqa: WPS433
            ram_gb = round(psutil.virtual_memory().total / 1e9, 1)
        except Exception:
            try:
                ram_gb = round(
                    os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
            except (ValueError, OSError, AttributeError):
                ram_gb = None

        return cls(
            git_sha=git_sha,
            python=_platform.python_version(),
            platform=_platform.platform(),
            numpy=_ver("numpy"),
            pyfai=_ver("pyFAI"),
            h5py=_ver("h5py"),
            hdf5=h5_hdf5,
            hdf5plugin=_ver("hdf5plugin"),
            cpu_count=os.cpu_count(),
            ram_gb=ram_gb,
        )


# ---------------------------------------------------------------------------
# per-container + per-run metrics
# ---------------------------------------------------------------------------

@dataclass
class ContainerMetrics:
    """One discovered container's probe/open/reduce facts for a single run.

    See :data:`OBSERVATION_SOURCES` for exactly what each count/timing observes.
    """

    path: str
    scan_name: str = ""
    #: ready | processed_output | imageless | in_progress | invalid
    state: str = "ready"
    skip_reason: str | None = None
    error: str | None = None

    # layout, read from HDF5 metadata WITHOUT decoding a frame.
    nframes: int | None = None
    frame_shape: tuple[int, ...] | None = None
    dtype: str | None = None
    chunks: tuple[int, ...] | None = None

    # timings (seconds); None = not measured on this path.
    probe_s: float | None = None
    dataset_resolve_s: float | None = None
    open_s: float | None = None
    metadata_s: float | None = None
    reduce_s: float | None = None
    #: wall time around the wrapped sink.finish() for THIS container.
    finish_s: float | None = None

    # lifecycle counts — each an observed fact (or None if unobserved).
    frames_discovered: int = 0
    frames_input: int = 0
    frames_reduced: int | None = None
    frames_written: int = 0
    frames_durable: int = 0
    #: engine frame-accept is not observable without runtime edits -> None.
    frames_submitted: int | None = None

    #: categorized Python-level h5py.File constructions for this container.
    open_counts: dict[str, int] = field(default_factory=new_open_counts)

    #: later-phase per-container counters attach here (R2 read plan, C build).
    reserved: dict[str, Any] = field(default_factory=dict)

    @property
    def source_opens(self) -> int:
        """Wrapper-visible source h5py.File constructions (the R2 target)."""
        return int(self.open_counts.get("source", 0))


@dataclass
class RunMetrics:
    """Aggregate of one benchmark run (one pass over the source directory)."""

    schema_version: int = SCHEMA_VERSION
    mode: str = "1d"
    #: EXACT reduction worker count used (a fresh ThreadPoolExecutor(max_workers)).
    cores: int = 1
    source_dir: str = ""
    output_root: str = ""
    recursive: bool = False

    n_candidates: int = 0
    enumerate_s: float | None = None

    containers: list[ContainerMetrics] = field(default_factory=list)

    # aggregate lifecycle counts (each summed only over containers that observed it).
    frames_discovered: int = 0
    frames_input: int = 0
    frames_reduced: int = 0
    frames_written: int = 0
    frames_durable: int = 0
    #: None while unobserved (never inferred); stays None (F-NXS-9 honesty).
    frames_submitted: int | None = None
    skipped_by_reason: dict[str, int] = field(default_factory=dict)

    #: categorized run-total h5py.File constructions.
    open_counts: dict[str, int] = field(default_factory=new_open_counts)

    # aggregate timings.
    total_s: float | None = None
    reduce_total_s: float | None = None
    session_finish_total_s: float | None = None
    first_frame_latency_s: float | None = None

    env: EnvProvenance | None = None
    observation_sources: dict[str, str] = field(
        default_factory=lambda: dict(OBSERVATION_SOURCES))
    #: generated output paths (per container) for inspectability.
    outputs: list[str] = field(default_factory=list)
    reserved: dict[str, Any] = field(default_factory=dict)

    def add(self, container: ContainerMetrics) -> None:
        """Append a container and fold its OBSERVED counts into the aggregates."""
        self.containers.append(container)
        self.frames_discovered += container.frames_discovered
        self.frames_input += container.frames_input
        if container.frames_reduced is not None:
            self.frames_reduced += container.frames_reduced
        self.frames_written += container.frames_written
        self.frames_durable += container.frames_durable
        for cat in OPEN_CATEGORIES:
            self.open_counts[cat] += int(container.open_counts.get(cat, 0))
        if container.finish_s is not None:
            self.session_finish_total_s = (
                (self.session_finish_total_s or 0.0) + container.finish_s)
        if container.reduce_s is not None:
            self.reduce_total_s = (self.reduce_total_s or 0.0) + container.reduce_s
        if container.state != "ready" and container.skip_reason:
            self.skipped_by_reason[container.skip_reason] = (
                self.skipped_by_reason.get(container.skip_reason, 0) + 1)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def one_line_summary(self) -> str:
        """The single concise INFO line per run (no per-frame flood)."""
        skips = ", ".join(f"{k}={v}" for k, v in sorted(self.skipped_by_reason.items()))
        oc = self.open_counts
        return (
            f"[NXS-BENCH] mode={self.mode} cores={self.cores} "
            f"candidates={self.n_candidates} "
            f"frames disc={self.frames_discovered} in={self.frames_input} "
            f"red={self.frames_reduced} wr={self.frames_written} "
            f"durable={self.frames_durable} "
            f"| h5open master={oc['source']} ext={oc['source_external']} "
            f"other={oc['other']} out={oc['output']} verif={oc['verification']} "
            f"fail={oc['failed']} "
            f"| enum={_fmt(self.enumerate_s)} reduce={_fmt(self.reduce_total_s)} "
            f"finish={_fmt(self.session_finish_total_s)} "
            f"first_write={_fmt(self.first_frame_latency_s)} total={_fmt(self.total_s)}"
            + (f" | skipped: {skips}" if skips else "")
        )


# ---------------------------------------------------------------------------
# multi-repeat summary
# ---------------------------------------------------------------------------

def summarize_runs(runs: list[RunMetrics]) -> dict[str, Any]:
    """Median + range over a repeat set, for the fields performance-compared.

    CI asserts *structure*; wall-clock acceptance is a separate same-host
    comparison, so this only reduces the repeated timings — no verdict.
    """
    def _stats(values: list[float | None]) -> dict[str, float] | None:
        xs = [float(v) for v in values if v is not None]
        if not xs:
            return None
        return {"median": statistics.median(xs), "min": min(xs),
                "max": max(xs), "n": len(xs)}

    keyed = {
        "total_s": [r.total_s for r in runs],
        "enumerate_s": [r.enumerate_s for r in runs],
        "reduce_total_s": [r.reduce_total_s for r in runs],
        "session_finish_total_s": [r.session_finish_total_s for r in runs],
        "first_frame_latency_s": [r.first_frame_latency_s for r in runs],
    }
    return {
        "repeats": len(runs),
        "cores": (runs[0].cores if runs else None),
        "timings": {k: _stats(v) for k, v in keyed.items()},
        "source_opens_median": _median_int(
            [r.open_counts.get("source", 0) for r in runs]),
        "frames_written": (runs[0].frames_written if runs else 0),
        "frames_durable": (runs[0].frames_durable if runs else 0),
    }


def _median_int(values: list[int]) -> int | None:
    return int(statistics.median(values)) if values else None


# ---------------------------------------------------------------------------
# json helpers
# ---------------------------------------------------------------------------

def _fmt(seconds: float | None) -> str:
    return "-" if seconds is None else f"{seconds:.3f}s"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path | str, payload: Any) -> Path:
    """Write *payload* (a dict/list, or object with ``to_dict``) as pretty JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = payload.to_dict() if hasattr(payload, "to_dict") else _jsonable(payload)
    p.write_text(json.dumps(data, indent=2, sort_keys=False))
    return p
