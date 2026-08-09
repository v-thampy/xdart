# -*- coding: utf-8 -*-
"""
imageThread — worker thread for image_wrangler.

Handles all image processing, integration, background subtraction,
and file I/O in a separate QThread.

@author: thampy, walroth
"""

# Standard library imports
import logging
import os
import queue
import re
import threading
import time
import numpy as np
from pathlib import Path
from collections import Counter, deque
from typing import NamedTuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# F1: shared boolean Filter grammar (unordered AND, '|'/OR, -term/NOT) —
# one compiled predicate replaces the old filter-encoded globs at all
# three sites (Image Directory, Eiger _master.h5 queue, BG Match).
from xrd_tools.core.filters import compile_filter as _compile_name_filter

_warned_bad_filters: set[str] = set()


def _name_filter(expr):
    """Compiled Filter predicate; a malformed expression warns once per
    expression and falls back to matching NOTHING.  Conservative on
    purpose: the old filter-encoded glob also matched nothing on garbage
    input, and a match-all fallback would process every file in the
    directory (or pick an arbitrary background at the BG Match site)."""
    try:
        return _compile_name_filter(expr)
    except ValueError as exc:
        key = str(expr)
        if key not in _warned_bad_filters:
            _warned_bad_filters.add(key)
            logger.warning("Invalid Filter expression %r (%s); matching "
                           "NO names until it is corrected", expr, exc)
        return lambda name: False


def _nexus_integrated_frame_labels(path, *, entry="entry"):
    """Return the unique integrated frame labels already on disk.

    This intentionally reads only the two small ``frame_index`` datasets.  The
    append-resume path needs no axes, provenance, detector calibration, or
    frame payloads, so constructing a ``LiveScan`` here turns a cheap cursor
    lookup into a full processed-scan hydration.
    """
    with h5py.File(path, "r") as h5:
        if entry not in h5:
            raise ValueError(f"{path} has no {entry!r} group")
        labels_1d, labels_2d = _nexus_integrated_frame_sets(h5, entry=entry)
    return labels_1d | labels_2d


def _nexus_integrated_frame_sets(h5, *, entry="entry"):
    """Return separate 1D and 2D frame-label sets from an open handle."""
    if entry not in h5:
        raise ValueError(f"HDF5 file has no {entry!r} group")
    groups = h5[entry]
    result = []
    for group_name in ("integrated_1d", "integrated_2d"):
        group = groups.get(group_name)
        if group is None or "frame_index" not in group:
            result.append(set())
            continue
        result.append({
            int(v) for v in np.asarray(group["frame_index"][()]).ravel()
        })
    return tuple(result)


_APPEND_SOURCE_SNAPSHOT_KEY = "_xdart_append_source_snapshot"


def _nexus_processed_source_snapshot(h5, completed, *, entry="entry"):
    """Read the source stamp/extent beside the latest completed frame."""
    if not completed or entry not in h5:
        return None
    frame_group = h5.get(
        f"{entry}/frames/frame_{max(int(v) for v in completed):04d}/source")
    if frame_group is None or "path" not in frame_group:
        return None
    try:
        path = frame_group["path"][()]
        if isinstance(path, bytes):
            path = path.decode("utf-8", errors="replace")
        path = str(path)
        if not os.path.isabs(path):
            source_base = h5[entry].attrs.get("source_base")
            if isinstance(source_base, bytes):
                source_base = source_base.decode("utf-8", errors="replace")
            base = (
                str(source_base)
                if source_base
                else os.path.dirname(os.path.abspath(str(h5.filename)))
            )
            path = os.path.abspath(os.path.join(base, path))
        attrs = frame_group.attrs
        required = ("file_size", "file_mtime_ns", "frame_count")
        if any(name not in attrs for name in required):
            return None
        return {
            "path": os.path.abspath(path),
            "size": int(attrs["file_size"]),
            "mtime_ns": int(attrs["file_mtime_ns"]),
            "frame_count": int(attrs["frame_count"]),
            "dataset_path": str(attrs.get("dataset_path", "") or ""),
            "self_contained": bool(attrs.get("self_contained", False)),
        }
    except (KeyError, TypeError, ValueError, OSError):
        return None


def _nexus_append_cursor(path, *, require_2d, entry="entry"):
    """Read mode-aware completion labels and provenance in one file open."""
    with h5py.File(path, "r") as h5:
        labels_1d, labels_2d = _nexus_integrated_frame_sets(h5, entry=entry)
        completed = labels_1d & labels_2d if require_2d else labels_1d
        provenance = dict(read_provenance_from_handle(h5, entry=entry))
        provenance[_APPEND_SOURCE_SNAPSHOT_KEY] = (
            _nexus_processed_source_snapshot(
                h5, completed, entry=entry))
        committed_prefix = decode_committed_append_prefix(h5, entry=entry)
    return completed, provenance, committed_prefix


def _nexus_integrated_frame_count(path, *, entry="entry"):
    """Return the number of unique integrated frame labels already on disk."""
    return len(_nexus_integrated_frame_labels(path, entry=entry))

# pyFAI / fabio / h5py
import fabio
import h5py

# Qt imports
from pyqtgraph import Qt

# Project imports
from xdart.modules.live import LiveFrame, LiveScan
from xrd_tools.io import AppendRefused, decode_committed_append_prefix, resolve_output_target
from xrd_tools.core.provenance import read_provenance_from_handle
from xrd_tools.integrate.gid import gi_1d_output_axis_key
from xrd_tools.integrate.calibration import poni_to_integrator, get_detector
from xrd_tools.reduction import (
    GIMode,
    Integration1DPlan,
    ReductionPlan,
    prepare_gi_freeze,
)
from xrd_tools.session.run_configuration import (
    RunConfigurationRefused,
    require_run_configuration,
)
from xrd_tools.session.readiness import (
    AppendConfigMismatchError,
    append_config_difference_lines,
    append_config_mismatch_check,
    processing_config_from_mapping,
)
from xrd_tools.sources.image import ImageFileSource, TiffSeriesSource
from xrd_tools.sources.probe import ProbeState
from xrd_tools.io.image import read_image, count_frames
from xrd_tools.io.output_safety import OutputCollisionError, check_output_not_source
from xrd_tools.io.processed_scan_id import ProcessedXdartInputError
from xrd_tools.io.metadata import read_image_metadata
from xdart.utils import get_series_avg
from xdart.modules.reduction import (
    apply_frozen_run_configuration as _apply_frozen_run_configuration,
    freeze_live_scan_gi_ranges,
    StandardPlanCache,
    sync_live_scan_gi_settings,
)
from .wrangler_widget import (
    wranglerThread,
)

# Live/network-share partial-write tolerance (get_next_image).  When the glob
# picks up a detector image the instrument is still flushing to disk (esp. an
# SMB/NFS beamline share), fabio hits a truncated/empty file and raises "Could
# not interpret magic string" -- which, unhandled, escaped run() and KILLED the
# live thread (looked like a silent timeout).  Tolerance has two layers so a
# slow write is NEVER dropped yet a corrupt file can't wedge live:
#   * per-call retry budget -- a single read attempt retries briefly to absorb
#     a file that finishes flushing within ~a second (no re-poll needed);
#   * cross-sweep deadline -- a file still unreadable after the budget is left
#     in the queue (NOT committed to self.processed) and re-polled on later
#     LIVE watch sweeps, so an eventually-completing write is read, not lost;
#     only once it has been unreadable this long since first seen is it treated
#     as genuinely corrupt and skipped, so a bad file can't wedge the watch.
# (This replaces an earlier size-stability heuristic that could false-drop a
# frame whose write plateaued >grace on a stalling share -- see WS-X reviews.)
# Env overrides for a laggy share; clamped to sane ranges.
def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return min(max(v, lo), hi)

_FRAME_READ_RETRY_BUDGET = _env_float("XDART_FRAME_READ_RETRY", 1.0, 0.0, 30.0)
_FRAME_READ_DEADLINE = _env_float("XDART_FRAME_READ_DEADLINE", 30.0, 1.0, 600.0)

# A container path can become visible before its detector dataset and even its
# NXWriter markers are committed. A zero-frame open is provisional while the
# file is young. Retry it without blocking later ready candidates; only a
# stable old file is retired as genuinely imageless.
_CONTAINER_READY_RETRY = _env_float(
    "XDART_CONTAINER_READY_RETRY", 0.5, 0.05, 5.0)
_CONTAINER_READY_DEADLINE = _env_float(
    "XDART_CONTAINER_READY_DEADLINE", 30.0, 1.0, 600.0)
#: Self-contained container inputs: ONE run reads many frames out of a single
#: file (Eiger master, NeXus stack, processed .nxs) instead of one file per
#: frame.  Used by the frozen container-versus-series gate (R4B-14).
_CONTAINER_SUFFIXES = frozenset({"h5", "hdf5", "nxs"})
_CONTAINER_SOURCE_KINDS = frozenset(
    {"eiger_master", "nexus_stack", "processed_nexus"})


def _suffix_format_token(suffix):
    """Return the FORMAT token of one match suffix or path extension.

    Frozen directory sources carry the freeze owner's match suffixes
    (``"_master.h5"``, ``".nxs"``); paths carry extensions (``".h5"``).  Both
    answer the same question — which format is this — so both reduce to the text
    after the final dot: ``"_master.hdf5" -> "hdf5"``, ``".nxs" -> "nxs"``,
    ``"nxs" -> "nxs"``, ``".tif" -> "tif"``.
    """
    return str(suffix).rsplit(".", 1)[-1].strip().lower()

_APPEND_CURSOR_MEMO_LIMIT = 1024

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _is_eiger_master(path):
    """Return True if path looks like an Eiger HDF5 master file (*_master.h5 / *_master.hdf5)."""
    return Path(path).stem.lower().endswith('_master')


def _paths_with_suffix(root, suffix, *, recursive=False):
    """Yield files below *root* whose name ends with *suffix*, ignoring case."""
    suffix = str(suffix or '').lower()
    if not suffix:
        return []
    root = Path(root)
    iterator = root.rglob('*') if recursive else root.glob('*')
    return (
        p for p in iterator
        if p.is_file() and p.name.lower().endswith(suffix)
    )


def _source_observation(path):
    import xrd_tools.sources.registry  # noqa: F401
    from xrd_tools.sources.adapters import candidate_owner
    from xrd_tools.sources.discover import Candidate
    path = Path(path)
    stat = path.stat()
    return Candidate(path, getattr(candidate_owner(path), "id", ""), int(stat.st_size), int(stat.st_mtime_ns))


class _SourceMetadata(dict):
    """Legacy metadata mapping carrying immutable producer facts out-of-band."""
    __slots__ = ("facts",)

    def __init__(self, values, facts):
        super().__init__(values)
        self.facts = dict(facts)


def _raw_lives_in_source(path):
    """Return True if the raw image for ``path`` is already embedded in the
    source file (Eiger master or any HDF5/NeXus container).

    When this is the case, writing ``map_raw`` into the output scan HDF5
    is pure duplication and can dominate the per-frame write cost.
    """
    if not path:
        return False
    ext = Path(path).suffix.lower()
    return ext in _RAW_EMBEDDED_EXTS or _is_eiger_master(path)


def _get_scan_info(fname):
    """Return (scan_name, img_number) for a file path.

    Strips trailing _<digits> or -<digits> suffix from the stem to get scan_name.
    Falls back to (stem, None) when no numeric suffix is found.
    """
    return _split_scan_suffix(Path(fname).stem)


def scan_name_from_source(path):
    """THE canonical source-file → scan-name rule (Codex F2).

    Container files (`.nxs` / `.h5` / `.hdf5`) keep the **FULL stem** — the numeric
    suffix is part of the scan identity: `LaB6_0710_1025pm_00005.nxs` stays
    `LaB6_0710_1025pm_00005`.  That is what the processed `.nxs` filename AND the
    plot titles / legend labels must show (dropping the `00005` was the bug).  An
    Eiger `_master` HDF5 strips only the `_master` tag; a per-file image series
    (`.tif`, …) strips the trailing `_<digits>` frame index.

    Shared by the worker (`_eiger_scan_name`), the GUI frame-boundary parser
    (`_scan_key_from_source`), the append-target resolver
    (`_append_scan_name_for_source`) and the overlay helper so they can never
    disagree again — the three-way divergence F2 fixed.  Pure string parse, no I/O.
    """
    p = Path(str(path))
    stem = p.stem
    if stem.lower().endswith("_master"):
        return stem[:-7]
    if p.suffix.lower() in (".h5", ".hdf5", ".nxs"):
        return stem
    return _get_scan_info(path)[0]


def _series_frame_sort_key(path):
    scan_name, img_number = _get_scan_info(path)
    number_key = -1 if img_number is None else img_number
    return str(scan_name), number_key, natural_keys_int(str(path))


def _split_scan_suffix(stem):
    match = _FRAME_SUFFIX_PATTERN.match(stem)
    if not match:
        return stem, None
    return match.group(1), int(match.group(2))


def _gi_2d_range_keys(args):
    """Return the GI 2D range keys for the selected output mode."""
    mode = args.get('gi_mode_2d', 'qip_qoop')
    if mode == 'qip_qoop':
        return 'x_range', 'y_range'
    return 'radial_range', 'azimuth_range'


def _padded_axis_range(axis, pad_fraction=0.02):
    """Return a small coverage margin around an integrated axis' finite extent.

    The 2% margin is load-bearing: a fresh per-frame integration can land a few
    bins beyond a scout's auto-range extent (binning discretization), and
    without it the frozen range would CLIP that real data.  The empty bins the
    margin creates beyond the real data should be NaN-filled (not a spurious
    dummy) so they are not plotted — see the NaN-empty follow-up.

    Returns ``None`` when the axis is missing, has no finite samples, or is
    *collapsed* (span <= 0 — every finite value identical).  A collapsed
    axis means the scout integration was degenerate (e.g. GI at a 0°
    incidence); freezing a tiny range from it would clamp every subsequent
    frame onto that collapsed grid and blank the whole scan.  Returning
    ``None`` leaves the range unfrozen so the caller can surface the problem
    instead of silently squashing the output.
    """
    if axis is None:
        return None
    arr = np.asarray(axis, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    span = hi - lo
    if span <= 0:
        return None
    pad = max(span * pad_fraction, 1e-9)
    return lo - pad, hi + pad


def _result_intensity_all_dummy(result, dummy=-1.0):
    """True if a 2D integration result has no real signal.

    A grazing-incidence map integrated at a degenerate incidence (e.g. a
    defaulted 0°) comes back as an all-dummy (``<= -1``) / empty grid — a
    blank cake.  The scout uses this to refuse freezing a representative
    grid off a blank scout frame, and tests use it to guard against the
    eiger "all -1.0" regression.
    """
    intensity = getattr(result, 'intensity', None)
    if intensity is None:
        return False
    arr = np.asarray(intensity, dtype=float)
    if arr.size == 0:
        return True
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return True
    return bool(np.all(finite <= dummy))


def _freeze_gi_2d_ranges_from_result(args, result):
    """Freeze missing GI 2D auto-range args from one scout result."""
    x_key, y_key = _gi_2d_range_keys(args)
    missing = [key for key in (x_key, y_key) if args.get(key) is None]
    if not missing:
        return False
    ranges = {
        x_key: _padded_axis_range(getattr(result, 'radial', None)),
        y_key: _padded_axis_range(getattr(result, 'azimuthal', None)),
    }
    changed = False
    for key in missing:
        if ranges[key] is not None:
            args[key] = ranges[key]
            changed = True
    return changed


def _gi_1d_output_range_key(gi_mode_1d):
    """Which integration range param controls the 1D *output* axis for a GI mode.

    Thin delegate to the canonical ssrl mapping
    (:func:`xrd_tools.integrate.gid.gi_1d_output_axis_key`): ``azimuth_range``
    for q_oop/exit_angle (out-of-plane output), else ``radial_range``.  Freezing
    the wrong key leaves the output axis auto-ranging per incidence → a
    non-uniform stack the writer rejects."""
    return gi_1d_output_axis_key(gi_mode_1d)


def _freeze_gi_1d_range_from_result(args, result, gi_mode_1d=None):
    """Freeze the missing GI 1D *output-axis* range from one scout result so all
    frames share one axis.  Picks radial_range vs azimuth_range by mode (see
    :func:`_gi_1d_output_range_key`)."""
    mode = gi_mode_1d if gi_mode_1d is not None else args.get('gi_mode_1d')
    key = _gi_1d_output_range_key(mode)
    if args.get(key) is not None:
        return False
    rng = _padded_axis_range(getattr(result, 'radial', None))
    if rng is None:
        return False
    if mode in ('exit_angle', 'chi_gi'):
        rng = (max(float(rng[0]), -180.0), min(float(rng[1]), 180.0))
    args[key] = rng
    return True


# ---------------------------------------------------------------------------
# Natural sort helpers
# ---------------------------------------------------------------------------

# Pre-compiled regex patterns (avoids recompilation on every sort key call)
_INT_PATTERN = re.compile(r'(\d+)')
_FLOAT_PATTERN = re.compile(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)')
_FRAME_SUFFIX_PATTERN = re.compile(r'^(.*?)[_-](\d+)$')

# Number of frames the background prefetch worker may read ahead of the main
# collect loop (the prefetch queue's maxsize).  This is the read‖reduce OVERLAP
# budget: the collect loop blocks on dispatch (reduce+write backpressure) for
# seconds per chunk, and the prefetcher can only run ahead this many frames
# before its queue fills and it stalls — so a tiny value makes collect_read and
# dispatch ADDITIVE instead of overlapped ([PERF-SUMMARY], 2026-06-15: Eiger 1D
# read is decompression-bound ~24 ms/frame, single-thread prefetch floor ~16 s,
# but additive gives ~25 s).  Trade-off: 18 MB/frame, so a large queue costs RAM
# and can contend with the writer.  Env-tunable for perf experiments; default 4
# preserves the prior behaviour.  (The bulk-read path is a dead end for per-frame
# -compressed Eiger data — it does NOT amortize decompression.)
try:
    _PREFETCH_QUEUE_SIZE = max(1, int(os.environ.get("XDART_PREFETCH_QUEUE_SIZE", "4")))
except (TypeError, ValueError):
    _PREFETCH_QUEUE_SIZE = 4

# R2: the prefetcher's HDF5 bulk-read block size is no longer a fixed 16 frames.
# It is now the layout-aware, byte-bounded ``ReadPlan.block_frames`` computed per
# master from the detector's native chunk cadence + dtype + a source-block byte
# budget (``xrd_tools.sources.read_plan.plan_reads`` /
# ``xrd_tools.core.staging.source_block_budget_bytes``): a native two-frame
# Eiger cadence instead of an arbitrary 16-frame block, and the 32-frame Bluesky
# decode floor bounded to the byte budget — bounding the prefetch's retained
# memory that the 651-frame Int 2D case is sensitive to.

# File extensions whose raw image data already lives in the source file
# — no need to duplicate `map_raw` into the output scan HDF5.  This is
# the single biggest write-time win for multi-frame NeXus / HDF5 inputs.
_RAW_EMBEDDED_EXTS = frozenset({'.h5', '.hdf5', '.nxs'})


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys_int(text):
    """Sort key for human-order sorting of strings with integers.

    See: http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [atoi(c) for c in _INT_PATTERN.split(text)]


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys_float(text):
    """Sort key for human-order sorting of strings with floats.

    See: https://stackoverflow.com/a/12643073/190597
    """
    return [atof(c) for c in _FLOAT_PATTERN.split(text)]


def natural_sort_ints(list_to_sort):
    return sorted(list_to_sort, key=natural_keys_int)


def natural_sort_float(list_to_sort):
    return sorted(list_to_sort, key=natural_keys_float)


class GISourceMotorDiscovery(NamedTuple):
    """What the worker learned about ONE just-classified source, as a value.

    O-1b R4A-1(iii) (review §49.3 item 5, §49.8).  When the pre-Run preview
    found nothing usable, the run itself is the first thing that classifies a
    container -- so the FIRST such container hands its motor/counter names to
    the GUI here.

    This is a VALUE PROJECTION, not a second run/source authority:

    * ``run_configuration`` is the EXACT accepted ``FrozenRunConfiguration``
      object this worker was admitted with, so the wrapper qualifies delivery by
      ``is`` identity.  A foreign or merely equal-VALUED configuration, a stale
      source generation, a later run and a post-close arrival are all inert,
      because none of them can produce that object;
    * ``motors``/``counters`` are DETACHED name tuples -- no live handle, no
      provider, no mutable panel list crosses the signal; and
    * it retains no reference to the worker, so it adds no entry to the
      ``f512c705`` transitive-consumer inventory.

    Nothing here may mutate, replace or re-freeze the accepted configuration, or
    change the motor the worker actually integrates with.
    """

    run_configuration: object
    source_path: str
    motors: tuple
    counters: tuple


# ---------------------------------------------------------------------------
# imageThread
# ---------------------------------------------------------------------------

class imageThread(wranglerThread):
    """Thread for controlling image processing.  Receives and manages a
    command and signal queue to pass commands from the main thread and
    communicate back relevant signals.

    attributes:
        command_q: mp.Queue, queue to send commands to process
        file_lock: mp.Condition, process safe lock for file access
        scan_name: str, name of current scan
        fname: str, full path to data file.
        img_file: str, path to image file
        poni_dict: str, Poni File name
        detector: str, Detector name
        input_q: mp.Queue, queue for commands sent from parent
        signal_q: mp.Queue, queue for commands sent from process
        command: command passed to start, stop etc.
        run_configuration: the ONE accepted FrozenRunConfiguration, published
            by the wrapper at admission.  Acquisition shape, source family,
            traversal, reader selection, output policy, GI geometry, live mode
            and the worker cap are read from it and from nowhere else -- the
            constructor parameters that once mirrored them were retired at
            R4-G (see __init__).

    signals:
        showLabel: str, sends out text to be used in specLabel

    methods:
        run: Main method, called by start
    """
    showLabel = Qt.QtCore.Signal(str)
    # Pause: emitted (on the worker thread) AFTER _enter_pause has drained the
    # in-flight window + flushed the sink to .nxs at a frame boundary -- the GUI
    # slot then lifts the freeze guard, race-safely (writer is provably quiet).
    sigPaused = Qt.QtCore.Signal()
    # Mid-run Append config mismatch (guard held, target preserved, run
    # stopping cleanly): carries the user-facing message for the GUI's
    # one-modal warning + status text — see _handle_append_config_mismatch.
    sigAppendMismatch = Qt.QtCore.Signal(str)
    # DIR-2 lazy convergence: (path, nframes, (size, mtime_ns), authoritative)
    # as each container is opened and again when it is retired.  The GUI uses
    # every stamped value for display convergence, but only a finalized,
    # self-contained retirement value may optimize a later Append Run.
    sigContainerCount = Qt.QtCore.Signal(str, int, object, bool)
    # O-1b R4A-1(iii): the FIRST JIT-classified container's motor + counter
    # names, published ONCE per accepted run as an immutable
    # :class:`GISourceMotorDiscovery`.  Display-only -- the wrapper qualifies it
    # by the exact accepted configuration OBJECT before translating it onto the
    # existing GI hydration projection, and nothing here can alter the run.
    sigGISourceMotors = Qt.QtCore.Signal(object)
    sigRetainedCustody = Qt.QtCore.Signal(int, str)

    def __init__(
            self,
            command_queue,
            file_lock,
            fname,
            scan_name,
            poni,
            img_file,
            bg_type,
            bg_file,
            bg_dir,
            bg_matching_par,
            bg_match_fname,
            bg_file_filter,
            bg_scale,
            bg_norm_channel,
            gi_mode_1d,
            gi_mode_2d,
            command,
            scan,
            parent=None):
        """R4-G retired eighteen parameters here (W-1R-D had already deleted
        every worker slot they seeded, review §44.2).  Acquisition shape,
        source family, traversal, reader selection, output policy, GI
        geometry, live mode and the worker cap are read from the accepted
        ``FrozenRunConfiguration`` and nowhere else."""

        super().__init__(command_queue, fname, file_lock, parent)

        # Pause: the LiveScan currently being processed (a local in
        # process_scan), stashed so _enter_pause's serial branch can flush it.
        self._active_scan = None
        # N1: project root for portable @source_base; set from the wrangler in
        # setup() (None -> absolute raw paths, back-compat).
        self.source_base = None
        self.scan_name = scan_name
        self.poni = poni
        # GENERIC-DETECTOR FIX: when a Run adopts a loaded processed scan's
        # geometry (image_wrangler._adopt_loaded_scan_run_inputs), the wrangler
        # also hands over the restored PIXEL-BEARING integrator keyed on the
        # adopted poni.  The poni-identity rebuild block below REUSES it instead
        # of rebuilding a pixel-less integrator from the (name-only) PONI.  Both
        # default None (a normal poni-file Run rebuilds as before).
        self._adopted_poni = None
        self._adopted_integrator = None
        self._adopted_fiber_integrator = None
        self.img_file = img_file
        # Optional explicit SPEC search dir.  Set by the wrangler
        # widget (set_meta_dir / setup) — None / '' falls back to
        # the xrd_tools default heuristic.  Threaded through
        # to every read_image_metadata call below.
        self.meta_dir = None
        self.bg_type = bg_type
        self.bg_file = bg_file
        self.bg_dir = bg_dir
        self.bg_matching_par = bg_matching_par
        self.bg_match_fname = bg_match_fname
        self.bg_file_filter = bg_file_filter
        self.bg_scale = bg_scale
        self.bg_norm_channel = bg_norm_channel
        self.gi_mode_1d = gi_mode_1d
        self.gi_mode_2d = gi_mode_2d
        # Shared cadence and publication state are initialized by the base.
        self.command = command
        # The mutable DISPLAY scan.  Retained ONLY as the target of the backward
        # GI-mode acquisition projection (``_project_gi_modes_onto_display_scan``);
        # it is never a run-configuration source (W-1.2 case 12).
        self.scan = scan
        # The ONE accepted frozen run configuration, published by the wrapper at
        # admission, plus the accepted-generation watermark that makes a
        # superseded object a typed refusal instead of a silent stale run.
        self.run_configuration = None
        self.run_configuration_floor = 0
        self._admitted_run_configuration = None
        # H19: supplied by the mounted Source card immediately before Run.
        # Both are value/headless boundaries; the mutable DirectoryIndex stays
        # owned by DirectoryIndexSession's serialized executor.
        self.source_run_plan = None
        self.source_index_session = None
        # Value-only copy of the Source card's stamp-qualified lazy frame-count
        # memo.  It lets Append retire a fully complete container without
        # opening that raw master again; it is never a discovery authority.
        self.source_frame_count_snapshot = {}
        self.source_pending_count = 0
        self._source_plan_reported = set()
        self._h19_observed_master_paths = ()
        self._h19_ready_master_candidates = {}
        self._h19_seed_pending = False
        self._h19_queue_already_current = False
        self._eiger_master_candidate = None

        self.user = None
        self.mask = None
        self.detector_shape = None   # full-res detector (raw) shape (H, W)
        self.detector = None
        self.img_fnames = []
        self.processed = []
        # fname -> monotonic time first seen unreadable, for the live
        # cross-sweep re-poll deadline (see _read_frame_tolerant / get_next_image).
        self._frame_read_clocks = {}
        self.processed_scans = []
        # Eiger HDF5 lazy frame state
        self._eiger_master_path = None
        self._eiger_frame_idx = 0
        self._eiger_nframes = 0
        self._eiger_master_queue = deque()
        self._directory_walk_iter = None
        self._eiger_done_masters = set()
        self._eiger_retry_after = {}
        self._eiger_zero_frame_seen = {}
        self._eiger_open_state = None
        self._eiger_single_file_done = False  # finalized-consumed fixed point
        self._eiger_single_file_provisional = False  # watch-gate fact (SF-2)
        # R2: one sustained ContainerCursor per h5py-backed master (created,
        # read, and closed entirely on the prefetch owner thread) replaces the
        # raw persistent h5py.File/dataset for NeXus/Bluesky reads.  fabio stays
        # PRIMARY for real external-link Eiger masters.
        self._eiger_cursor = None        # xrd_tools ContainerCursor (h5py path)
        self._eiger_descriptor = None    # descriptor paired with the cursor
        self._eiger_read_plan = None     # layout-aware, byte-bounded block plan
        self._eiger_provider = None      # materialized per-frame metadata provider
        self._eiger_fabio_handle = None  # persistent fabio.EigerImage (primary Eiger)
        self._eiger_metadata_cache = {}  # master metadata is stable across frames
        self._source_snapshot_by_path = {}
        # Bluesky/NXWriter embedded per-frame table + wavelength, per master
        # (see _bluesky_source_for) — caches None for non-Bluesky masters too.
        self._bluesky_source_cache = {}
        # Background prefetch state (populated on demand)
        self._prefetch_queue = None      # queue.Queue of frame tuples or None sentinel
        self._prefetch_thread = None     # threading.Thread running the reader
        self._prefetch_stop_evt = None   # threading.Event — set to cancel worker
        # O8: distinguish clean end-of-stream from worker failure.
        # ``_prefetch_worker`` sets this to the str(exception) when
        # it dies on an unexpected error; the consumer surfaces it
        # via ``showLabel`` so the user sees "Eiger read failed: ..."
        # instead of a silent end-of-scan.  None means "no error".
        self._prefetch_error = None
        self._plan_cache = StandardPlanCache()
        self._append_skip_frames_by_scan = {}
        self._append_source_snapshot_by_scan = {}
        self._append_committed_prefix_by_scan = {}
        # Stamp-qualified processed-output cursors survive Stop -> Run on this
        # wrangler thread.  Unchanged outputs can then be validated with one
        # stat instead of reopening hundreds of small NeXus products.  The
        # per-run map above remains the immutable snapshot used while a Run is
        # active; this memo is only an optimization source for building it.
        self._append_cursor_memo = {}
        self._append_skip_without_reading = 0
        self._append_config_mismatch = False
        self._discovered_frame_count = 0
        self._skip_reason_counts = Counter()
        self.files_processed = self._last_files_processed = 0
        self.files_processed_by_output = {}
        self._last_files_processed_by_output = {}

    # ── The accepted run configuration (O-1a-W1A) ────────────────────────

    def _require_run_configuration(self, stage):
        """Return the EXACT admitted frozen configuration, or refuse (typed).

        The WORKER-ENTRY identity gate (review §39.5 Phase 1 item 5 / Phase 2
        item 1).  After admission the frozen configuration is the sole
        run-configuration authority, so this refuses -- before any source read,
        writer open, session mutation or reduction -- when the carrier is
        missing, not frozen, stale, or a genuine but DIFFERENT
        ``FrozenRunConfiguration`` than the one the wrapper admitted
        (same-generation, future-generation, or an equal-valued reconstruction).

        O-1a-W1R (review §39.2 W1R-P1-1): the comparison is exact object
        identity against the admission ledger, not ``generation >= floor``.  A
        bare floor never authorizes execution.
        """

        frozen = require_run_configuration(
            getattr(self, "run_configuration", None),
            stage=stage,
            floor=int(getattr(self, "run_configuration_floor", 0) or 0),
            expected=getattr(self, "_admitted_run_configuration", None),
        )
        return frozen

    def _project_gi_modes_onto_display_scan(self, frozen):
        """Backward GI-mode write onto the mutable DISPLAY scan (retained).

        This is an acquisition/display projection ONLY: the browser's scan shows
        the GI axis modes this run is producing.  It is deliberately NOT a run
        input -- the worker builds its per-run scan from the frozen
        configuration, so nothing written here can reach the run, the plan, or
        the written provenance (W-1.2 case 11).
        """

        if not frozen.gi.enabled or self.scan is None:
            return
        self.scan.bai_1d_args['gi_mode_1d'] = self.gi_mode_1d
        self.scan.bai_2d_args['gi_mode_2d'] = self.gi_mode_2d

    # ── Main entry point ─────────────────────────────────────────────────

    def run(self):
        """Initializes specProcess and watches for new commands from
        parent or signals from the process.
        """
        t0 = time.time()
        # W-1.2 case 5: refuse BEFORE any stateful operation of this run.  A
        # worker without the accepted configuration performs no source read, no
        # output open, no cache clear and no reduction -- it returns, and the
        # ordinary finish delivery unwinds the run state.
        try:
            frozen = imageThread._require_run_configuration(
                self, "image-worker-run")
        except RunConfigurationRefused as exc:
            logger.error("run refused: %s", exc)
            self.command = 'stop'
            try:
                self.showLabel.emit(f"Run refused: {exc}")
            except Exception:
                logger.debug("showLabel emit failed for refusal", exc_info=True)
            return
        # This must be the FIRST stateful operation of a new Run.  A prior
        # generation can outlive the bounded Stop join while blocked in HDF5;
        # clearing any cursor/cache before deciding to refuse the restart would
        # corrupt that old reader while it unwinds.
        if not self._prefetch_stop_prior():
            message = "Previous detector reader is still stopping; retry Run shortly"
            logger.warning(message)
            self.command = 'stop'
            try:
                self.showLabel.emit(message)
            except Exception:
                logger.debug("showLabel emit failed for lingering reader",
                             exc_info=True)
            return
        if (self._scan_session_adapter is not None
                and not self._close_reduction_session()):
            return
        if not self.release_retained_custody():
            return
        self._reduction_write_error = None
        # A timed-out prior cleanup intentionally leaves its HDF5 handle open.
        # Once the reader is confirmed dead, close it before clearing/reusing
        # any detector state for this generation.
        self._eiger_close_master()
        if (self.poni is None or (self.img_file == ''
                and not self._directory_source_armed(frozen))):
            return
        self.img_fnames.clear()
        self.processed.clear()
        self._frame_read_clocks.clear()
        self.processed_scans.clear()
        # Pause: start each run with a fresh serial-flush handle + save counter
        # so an early pause of THIS run can never flush a prior (stopped) run's
        # scan via stale state.
        self._active_scan = None
        self._frames_since_save = 0
        self._eiger_master_path = None
        self._eiger_frame_idx = 0
        self._eiger_nframes = 0
        self._eiger_master_queue.clear()
        self._directory_walk_iter = None
        self._eiger_done_masters.clear()
        # R4A-5 per-run provisional bookkeeping.  Cleared here, which is exactly
        # what makes "left for the next Run" true: a container this run gave up
        # on is visible again the moment a new Run starts.
        self._eiger_provisional_masters = set()
        self._eiger_provisional_exhausted = set()
        self._eiger_provisional_recheck_done = False
        self._source_plan_reported.clear()
        seed_candidates = tuple(
            getattr(getattr(self, "source_run_plan", None), "candidates", ())
            or ())
        self._h19_observed_master_paths = tuple(
            str(candidate.path) for candidate in seed_candidates)
        self._h19_ready_master_candidates = {
            str(candidate.path): candidate for candidate in seed_candidates}
        # The plan was frozen from the Source card's exact READY observation.
        # Consume that value snapshot once before asking the shared session for
        # another full recursive observation.
        self._h19_seed_pending = bool(seed_candidates)
        self._h19_pending_count = max(
            0, int(getattr(self, "source_pending_count", 0) or 0))
        self._h19_unprobed_count = None
        self._h19_queue_already_current = False
        self._eiger_master_candidate = None
        self._eiger_retry_after.clear()
        self._eiger_zero_frame_seen.clear()
        self._eiger_open_state = None
        self._eiger_single_file_done = False
        self._eiger_single_file_provisional = False
        self._eiger_metadata_cache.clear()
        self._bluesky_source_cache.clear()
        # Start each run with an empty producer->consumer display slot so a frame
        # the GUI never popped (a missed/raced consume, or a frame published after
        # the final flush) can't leak a whole LiveFrame (~18 MB raw) into the next
        # run.  Safe to clear here: the prior run's consumer is done and this run
        # hasn't published yet.
        self._published_frames.clear()
        self._prefetch_queue = None
        self._prefetch_thread = None
        self._prefetch_stop_evt = None
        self._prefetch_error = None
        self._append_skip_frames_by_scan = {}
        self._append_source_snapshot_by_scan = {}
        self._append_committed_prefix_by_scan = {}
        self._source_snapshot_by_path.clear()
        self._append_skip_without_reading = 0
        self._append_config_mismatch = False
        self._discovered_frame_count = 0
        self._skip_reason_counts = Counter()
        self.files_processed = self._last_files_processed = 0
        self.files_processed_by_output = {}
        self._last_files_processed_by_output = {}
        # Per-run perf accumulators -> [PERF-SUMMARY] at end of run.  Breaks the
        # worker time into consumer read(queue-wait)/integrate/write + the actual
        # background prefetch I/O, so ONE run shows the bottleneck (read vs
        # integrate vs write) without bisecting.  Written by _process_one (main
        # thread) and _prefetch_worker (bg thread) on disjoint keys.
        self._perf = {
            # collect/flush level (runs for BOTH the streaming and serial paths):
            'collect_read': 0.0,      # main-thread queue-wait on the prefetcher
            'dispatch': 0.0,          # _dispatch_batch: reduce + write for the chunk
            'dispatch_frames': 0,
            'prefetch_io': 0.0,       # actual background HDF5 block I/O
            'prefetch_frames': 0,
            # serial-path per-frame breakdown (only populated by _process_one):
            'frame': 0.0, '1d': 0.0, '2d': 0.0, 'h5': 0.0, 'csv': 0.0, 'n': 0,
        }
        self.detector = get_detector(self.poni.detector) if self.poni.detector else None
        self.sub_label = ''
        det_mask = self.detector.mask if self.detector is not None else None  # pyFAI .mask property
        if frozen.mask_file and os.path.exists(frozen.mask_file):
            try:
                custom_mask = np.asarray(read_image(frozen.mask_file), dtype=bool)
                # Validate the Mask File against the detector FRAME shape: the
                # built-in mask shape, or (for detectors with no built-in mask,
                # e.g. RayonixMx225) the geometry shape.  A shape mismatch can't
                # index the frame, so warn + ignore -- now consistent for BOTH
                # cases (previously only checked when a built-in mask existed),
                # which also keeps the persisted detector_shape trustworthy
                # (adversarial review: a mismatched mask must not be stored as
                # the detector shape).
                if det_mask is not None:
                    _ref_shape = tuple(det_mask.shape)
                elif self.detector is not None:
                    _gs = (getattr(self.detector, "shape", None)
                           or getattr(self.detector, "max_shape", None))
                    _ref_shape = tuple(_gs) if _gs is not None else None
                else:
                    _ref_shape = None
                if _ref_shape is not None and tuple(custom_mask.shape) != _ref_shape:
                    logger.warning('Mask file shape %s does not match detector shape %s — ignoring custom mask',
                                  custom_mask.shape, _ref_shape)
                else:
                    det_mask = det_mask | custom_mask if det_mask is not None else custom_mask
            except Exception as e:
                logger.warning('Could not load mask file %s: %s', frozen.mask_file, e)
        self.mask = np.flatnonzero(det_mask) if det_mask is not None else None
        # The full-res raw frame shape (H, W) the flat mask indices index into —
        # carried onto the scan + persisted so a reloaded thumbnail-only scan can
        # map the detector gap mask into thumbnail coordinates.  Prefer det_mask's
        # shape (a mismatched Mask File was rejected above, so it's the true frame
        # shape); otherwise capture the detector geometry shape even with NO mask
        # (future per-frame masks / cold reloads), per review.
        if det_mask is not None:
            self.detector_shape = tuple(det_mask.shape)
        elif self.detector is not None:
            _gs = (getattr(self.detector, "shape", None)
                   or getattr(self.detector, "max_shape", None))
            self.detector_shape = tuple(_gs) if _gs is not None else None
        else:
            self.detector_shape = None
        self._cached_gi_incident_angle = None

        imageThread._project_gi_modes_onto_display_scan(self, frozen)

        try:
            self.process_scan(frozen)
        except AppendRefused as exc:
            self._retain_dynamic_failure(exc, "Append refused")
        except AppendConfigMismatchError as exc:
            # Backstop: process_scan handles this at its initialize_scan call
            # sites; nothing may escape run() as an unhandled QThread
            # exception (the v1.0.1 mid-run mode-change beamline crash).
            self._handle_append_config_mismatch(exc)
        except Exception as exc:
            # GENERIC backstop for the same invariant (DIR-3: a Windows
            # replace-save PermissionError escaped run() and killed the
            # whole directory run through the GUI excepthook).  Fail loud
            # per RUN, not per process: log with traceback, tell the user,
            # stop cleanly — the finally below still releases the reduction
            # session, prefetcher and Eiger handle.
            logger.exception("run stopped by unhandled error")
            self.command = 'stop'
            try:
                self.showLabel.emit(f"Run stopped: {exc} (see log)")
            except Exception:
                logger.debug("showLabel emit failed", exc_info=True)
        finally:
            if not self._close_reduction_session():
                self.command = "stop"
            # Stop background prefetcher from the main thread BEFORE closing the
            # master handle.  _eiger_close_master() is called from inside the
            # prefetch worker when switching masters, so triggering a stop from
            # there would self-join — only the main thread should tear down the
            # prefetcher.
            prefetch_stopped = self._prefetch_stop_prior()
            if prefetch_stopped:
                self._eiger_close_master()  # ensure Eiger handle is released
            else:
                logger.warning(
                    "detector reader is still unwinding after Stop; its HDF5 "
                    "handle will be closed before the next Run")
        logger.info('Total Time: %.2fs', time.time() - t0)
        _perf = getattr(self, '_perf', None)
        if _perf and (_perf['dispatch_frames'] or _perf['prefetch_frames'] or _perf['n']):
            _df = max(_perf['dispatch_frames'], 1)
            _pf = max(_perf['prefetch_frames'], 1)
            # collect_read = main-thread queue-wait on the prefetcher; dispatch =
            # reduce+write per chunk; prefetch_io = real background HDF5 block I/O.
            # If collect_read ~= prefetch_io the bottleneck is raw read; if dispatch
            # dominates it's reduce/write.
            logger.info(
                '[PERF-SUMMARY] dispatch_frames=%d | collect_read(queue-wait)=%.2fs '
                'dispatch(reduce+write)=%.2fs | prefetch_io=%.2fs (%d frames, %.1f ms/frame)'
                ' | per-frame ms: read=%.1f dispatch=%.1f',
                _perf['dispatch_frames'], _perf['collect_read'], _perf['dispatch'],
                _perf['prefetch_io'], _perf['prefetch_frames'],
                _perf['prefetch_io'] / _pf * 1e3,
                _perf['collect_read'] / _df * 1e3, _perf['dispatch'] / _df * 1e3,
            )
            if _perf['n']:   # serial path only: per-frame integrate/write split
                _n = _perf['n']
                logger.info(
                    '[PERF-SUMMARY serial] %d frames: frame_init=%.2fs int_1d=%.2fs '
                    'int_2d=%.2fs h5_write=%.2fs csv=%.2fs',
                    _n, _perf['frame'], _perf['1d'], _perf['2d'], _perf['h5'],
                    _perf['csv'],
                )
        # Final echo so the user can copy the output path straight from
        # the terminal without scrolling back to the per-scan banner.
        # Trailing newline gives a visual gap before the next scan's
        # 'New Scan' banner (or before the next prompt if this was the
        # last scan in the session).
        if not frozen.run_options.get("xye_only", False):
            # THIS run's own output path (``initialize_scan`` set it).  The old
            # display-scan fallback is deleted: the browser's scan may point at a
            # completely different file by the time a run ends.
            output_path = getattr(self, 'fname', None)
            if output_path:
                logger.info('Output file: %s\n', output_path)

    def _install_run_integrator(self, scan):
        """Build (or reuse) the scan's cached AzimuthalIntegrator for this run.

        Called when ``self.poni`` differs from the scan's currently-cached poni
        (a fresh scan, or a user-loaded calibration).  The default path rebuilds
        from ``self.poni`` via ``poni_to_integrator``.

        GENERIC-DETECTOR FIX: a PONI dataclass carries only a detector *name*, so
        rebuilding for an unnamed/generic detector yields a pyFAI integrator with
        ``_pixel1``/``_pixel2`` = None — and the reduction then crashes in
        ``calc_cartesian_positions`` (``NoneType * float``).  When this run ADOPTED
        a loaded processed scan's geometry (image_wrangler._adopt_loaded_scan_run_inputs),
        the wrangler also handed over that scan's restored PIXEL-BEARING integrator,
        keyed on the adopted poni object.  Reuse it here whenever ``self.poni`` IS
        that adopted poni (identity check) so the pixel size survives.  A
        genuinely-new user-loaded .poni is a different object, so it still rebuilds.
        """
        adopted_ai = getattr(self, "_adopted_integrator", None)
        if (adopted_ai is not None
                and self.poni is getattr(self, "_adopted_poni", None)):
            scan._cached_integrator = adopted_ai
            scan._cached_fiber_integrator = getattr(
                self, "_adopted_fiber_integrator", None)
            logger.info(
                "[RUN-CAL] reusing restored pixel-bearing integrator for %s "
                "(generic detector — poni rebuild would drop pixel size)",
                getattr(scan, "name", "scan"))
        else:
            scan._cached_integrator = poni_to_integrator(self.poni)
            scan._cached_fiber_integrator = None
        scan._cached_poni = self.poni

    def _append_skip_enabled(self, frozen):
        return (frozen.output_mode == 'Append'
                and not frozen.run_options.get("xye_only", False))

    def _append_output_number(self, frozen, img_number):
        if (
            frozen.run_options.get("series_average", False)
            and frozen.source.family != "directory"
        ):
            return 1
        return 1 if img_number is None else img_number

    def _record_discovered_frame(self, count=1):
        self._discovered_frame_count = (
            getattr(self, "_discovered_frame_count", 0) + count)

    def _record_skip_reason(self, reason, count=1):
        reasons = getattr(self, "_skip_reason_counts", None)
        if reasons is None:
            reasons = Counter()
            self._skip_reason_counts = reasons
        reasons[str(reason)] += count

    @contextmanager
    def _optional_lock(self, lock):
        if lock is None:
            yield
        else:
            with lock:
                yield

    def _scan_frame_index_snapshot(self, scan):
        frames = getattr(scan, "frames", None)
        lock = getattr(scan, "scan_lock", None)
        with self._optional_lock(lock):
            index = getattr(frames, "index", ())
            return set(index if index is not None else ())

    def _warn_append_snapshot_failed(self, scan_name, out_path, exc):
        warned = getattr(self, "_append_skip_snapshot_warnings", None)
        if warned is None:
            warned = set()
            self._append_skip_snapshot_warnings = warned
        key = str(scan_name)
        if key in warned:
            return
        warned.add(key)
        logger.warning(
            "append skip snapshot unavailable for %s; proceeding without "
            "pre-read skips: %s",
            out_path, exc,
        )

    def _remember_append_skip_snapshot(self, frozen, scan_name, frame_index=None, *, scan=None):
        if not self._append_skip_enabled(frozen) or scan_name is None:
            return
        cache = getattr(self, "_append_skip_frames_by_scan", None)
        if cache is None:
            cache = {}
            self._append_skip_frames_by_scan = cache
        key = str(scan_name)
        # The lightweight cursor owns Append completion.  A loaded LiveScan's
        # frame index is the union of its 1D and 2D labels, which is not enough
        # to prove an Int 2D frame is complete.  Do not replace a mode-aware
        # cursor that was already established before the raw read.
        if key in cache:
            return
        try:
            existing = (self._scan_frame_index_snapshot(scan)
                        if scan is not None
                        else set(frame_index or ()))
        except Exception as exc:
            out_path = self._append_output_path(frozen, scan_name)
            self._warn_append_snapshot_failed(scan_name, out_path, exc)
            existing = set()
        cache[key] = existing

    def _append_output_path(self, frozen, scan_name):
        return os.fspath(resolve_output_target(
            frozen.save_path, str(scan_name), mode=frozen.output_mode))

    def _append_run_start_scan_names(self, frozen):
        if not self._append_skip_enabled(frozen):
            return []

        names = []
        seen = set()

        def add(name):
            if name is None:
                return
            key = str(name)
            if key and key not in seen:
                seen.add(key)
                names.append(key)

        img_file = getattr(self, "img_file", None)

        if frozen.source.family == "directory":
            plan = getattr(self, "source_run_plan", None)
            if plan is not None:
                for candidate in plan.candidates:
                    add(self._eiger_scan_name(candidate.path))
                return names
            img_dir = frozen.source.filesystem_root
            if not img_dir:
                return names
            match = _name_filter((frozen.source.name_filter or ""))
            root = Path(img_dir)
            include_subdir = bool(frozen.source.recursive)
            # Self-contained masters get one output scan per file; plain images
            # get the series name.  The frozen source carries the DISCOVERY
            # spelling itself ('_master.h5' for an Eiger File Type, '.tif' for a
            # plain one), so this no longer reconstructs the read path's suffix
            # rule from an extension token -- and every alternative the freeze
            # owner recorded is honored, not just the first.
            container = imageThread._frozen_source_is_container(self, frozen)
            for suffix in frozen.source.suffixes:
                candidates = _paths_with_suffix(
                    root, suffix, recursive=include_subdir)
                for path in natural_sort_ints([str(p) for p in candidates]):
                    p = Path(path)
                    if not match(p.name[:-len(suffix)]):
                        continue
                    add(self._eiger_scan_name(p) if container
                        else _get_scan_info(p)[0])
            return names

        if img_file:
            ext = Path(img_file).suffix.lower().lstrip(".")
            if _is_eiger_master(img_file) or ext in ('h5', 'hdf5', 'nxs'):
                add(self._eiger_scan_name(img_file))
            else:
                add(_get_scan_info(img_file)[0])
        if not names:
            add(getattr(self, "scan_name", None))
        return names

    def _load_append_skip_snapshot(self, frozen, scan_name):
        """Load one scan's append cursor, once, without hydrating a LiveScan.

        Normal directory runs call this lazily when the reader reaches a scan.
        That keeps Run and Stop responsive in a directory containing hundreds
        of processed outputs while preserving the skip-before-raw-read contract.
        A stamp-qualified value memo survives Stop -> Run on the same wrangler,
        so unchanged outputs need only one ``stat`` on subsequent runs.
        """
        if not self._append_skip_enabled(frozen) or scan_name is None:
            return set()
        cache = getattr(self, "_append_skip_frames_by_scan", None)
        if cache is None:
            cache = {}
            self._append_skip_frames_by_scan = cache
        source_cache = getattr(
            self, "_append_source_snapshot_by_scan", None)
        if source_cache is None:
            source_cache = {}
            self._append_source_snapshot_by_scan = source_cache
        prefix_cache = getattr(
            self, "_append_committed_prefix_by_scan", None)
        if prefix_cache is None:
            prefix_cache = {}
            self._append_committed_prefix_by_scan = prefix_cache
        key = str(scan_name)
        if key in cache and key in prefix_cache:
            return cache[key]
        if getattr(self, "command", None) == "stop":
            return set()

        out_path = self._append_output_path(frozen, key)
        if not os.path.exists(out_path):
            cache[key] = set()
            source_cache[key] = None
            prefix_cache[key] = None
            return cache[key]

        require_2d = not bool(frozen.skip_2d)
        memo = getattr(self, "_append_cursor_memo", None)
        if memo is None:
            memo = self._append_cursor_memo = {}
        memo_key = (os.path.abspath(out_path), require_2d)
        try:
            stat = os.stat(out_path)
            before_stamp = (
                int(getattr(stat, "st_dev", 0)),
                int(getattr(stat, "st_ino", 0)),
                int(stat.st_size),
                int(stat.st_mtime_ns),
            )
        except OSError as exc:
            memo.pop(memo_key, None)
            self._warn_append_snapshot_failed(key, out_path, exc)
            raise

        hit = memo.get(memo_key)
        try:
            if hit is not None and hit[0] == before_stamp:
                existing = set(hit[1])
                processed_config = hit[2]
                source_snapshot = hit[3]
                committed_prefix = hit[4]
            else:
                memo.pop(memo_key, None)
                with self._optional_lock(getattr(self, "file_lock", None)):
                    existing, provenance, committed_prefix = _nexus_append_cursor(
                        out_path,
                        require_2d=require_2d,
                    )
                source_snapshot = provenance.get(
                    _APPEND_SOURCE_SNAPSHOT_KEY)
                processed_config = processing_config_from_mapping(provenance)
                # Cache only a transactionally stable read.  A product replaced
                # while its cursor was being read remains valid for this run's
                # historical snapshot, but it must be reopened next Run.
                try:
                    stat = os.stat(out_path)
                    after_stamp = (
                        int(getattr(stat, "st_dev", 0)),
                        int(getattr(stat, "st_ino", 0)),
                        int(stat.st_size),
                        int(stat.st_mtime_ns),
                    )
                except OSError:
                    after_stamp = None
                if after_stamp == before_stamp:
                    memo[memo_key] = (
                        before_stamp,
                        frozenset(existing),
                        processed_config,
                        source_snapshot,
                        committed_prefix,
                    )
                    while len(memo) > _APPEND_CURSOR_MEMO_LIMIT:
                        memo.pop(next(iter(memo)))
            current_config = processing_config_from_mapping(
                frozen.processing_mapping())
            append_check = append_config_mismatch_check(
                frozen.output_mode, processed_config, current_config)
            if not append_check.ok:
                raise self._append_config_mismatch_error(
                    out_path, append_check, processed_config, current_config)
        except AppendConfigMismatchError:
            raise
        except Exception as exc:
            self._warn_append_snapshot_failed(key, out_path, exc)
            memo.pop(memo_key, None)
            raise
        cache[key] = existing
        source_cache[key] = source_snapshot
        prefix_cache[key] = committed_prefix
        return cache[key]

    def _prime_append_skip_snapshots_for_run(self, frozen):
        """Eagerly prime cursors for the series-average no-op safeguard only.

        Normal runs load one cursor lazily in ``_append_skip_snapshot``.  Keep
        this all-scan variant for series averaging, where an existing frame 1
        must refuse the run before any source frame is consumed.
        """
        if not self._append_skip_enabled(frozen):
            return
        for scan_name in self._append_run_start_scan_names(frozen):
            if getattr(self, "command", None) == "stop":
                break
            self._load_append_skip_snapshot(frozen, scan_name)

    def _series_average_append_blocker(self, frozen):
        """MEM-1c: refuse a silently-empty series-average Append run.

        ``_append_output_number`` collapses EVERY source frame of a series
        average to output frame 1.  In Append mode, if that averaged output
        already exists on disk, ``_should_skip_before_read`` would skip every
        source frame and the run would produce NOTHING — a silent no-op that
        only logs a benign INFO line and looks like success.  Detect it up
        front (the append-skip snapshots are already primed) and return an
        actionable reason so the run is refused loudly instead.  Returns the
        user-facing message, or ``None`` when the run may proceed.
        """
        if not (frozen.run_options.get("series_average", False)
                and self._append_skip_enabled(frozen)):
            return None
        collapsed = self._append_output_number(frozen, None)   # series-average => 1
        for scan_name in self._append_run_start_scan_names(frozen):
            if collapsed in self._append_skip_snapshot(frozen, scan_name):
                return (
                    f"Averaged output already exists for '{scan_name}' — the "
                    "whole series would be skipped and nothing written.  "
                    "Switch write mode to Replace, or clear the target, before "
                    "running a series average.")
        return None

    def _append_skip_snapshot(self, frozen, scan_name):
        """Return this run's append-skip frame snapshot for *scan_name*.

        The first lookup for a scan reads only its on-disk ``frame_index``
        datasets; subsequent per-frame lookups are pure in-memory operations.
        """
        if not self._append_skip_enabled(frozen) or scan_name is None:
            return set()
        key = str(scan_name)
        cache = getattr(self, "_append_skip_frames_by_scan", None)
        if cache is None:
            cache = {}
            self._append_skip_frames_by_scan = cache
        if key in cache:
            return cache[key]
        return self._load_append_skip_snapshot(frozen, key)

    def _should_skip_before_read(self, frozen, scan_name, img_number):
        output_img_number = self._append_output_number(frozen, img_number)
        if output_img_number in self._append_skip_snapshot(frozen, scan_name):
            self._append_skip_without_reading = (
                getattr(self, "_append_skip_without_reading", 0) + 1)
            self._record_skip_reason("already processed")
            return True
        return False

    def _append_frame_complete(self, frozen, scan_name, img_number, scan):
        """Use the same mode-aware cursor at raw-read and dispatch boundaries."""
        if self._append_skip_enabled(frozen):
            return img_number in self._append_skip_snapshot(frozen, scan_name)
        return img_number in getattr(getattr(scan, "frames", None), "index", ())

    @staticmethod
    def _append_config_mismatch_error(
            out_path, check, processed_config, current_config):
        details = append_config_difference_lines(
            processed_config,
            current_config,
            getattr(check, "mismatched_fields", ()),
        )
        detail_text = "\n".join(f"- {line}" for line in details)
        if not detail_text:
            detail_text = "- Stored and current integration settings differ."
        name = os.path.basename(os.fspath(out_path))
        message = (
            "Integration settings changed mid-run.\n\n"
            f"Append stopped before modifying {name} because the existing "
            "processed scan was created with different settings.\n\n"
            f"Changed settings:\n{detail_text}\n\n"
            f"The append target {name} was preserved.\n\n"
            "Switch output mode to Replace and run again, or restore the "
            "previous integration settings."
        )
        return AppendConfigMismatchError(message, check)

    def _format_skip_reasons(self):
        reasons = getattr(self, "_skip_reason_counts", Counter())
        if not reasons:
            return ""
        parts = []
        for reason, count in reasons.most_common():
            parts.append(f"{reason} ({count})" if count != 1 else reason)
        return "; ".join(parts)

    def _format_no_frames_discovered_message(self, frozen):
        directory = frozen.source.filesystem_root or (
            str(Path(getattr(self, "img_file", "")).parent)
            if getattr(self, "img_file", None) else "")
        ext = "/".join(frozen.source.format_tokens) or "<any>"
        pattern = (frozen.source.name_filter or "") or "<none>"
        return (
            "No frames discovered"
            f" (directory: {directory or '<unknown>'}, "
            f"ext: .{ext}, pattern: {pattern})"
        )

    def _report_run_skip_summary(self, frozen, files_processed):
        skipped = getattr(self, "_append_skip_without_reading", 0)
        if skipped:
            logger.info(
                "append: skipping %d already-processed frame(s) without reading",
                skipped,
            )

        discovered = getattr(self, "_discovered_frame_count", 0)
        if getattr(self, "command", None) == 'stop':
            return
        if discovered <= 0:
            if files_processed <= 0:
                msg = self._format_no_frames_discovered_message(frozen)
                logger.warning(msg)
                try:
                    self.showLabel.emit(msg)
                except Exception:
                    logger.debug("showLabel emit failed for no-frame warning",
                                 exc_info=True)
            return
        if files_processed >= discovered:
            return
        if frozen.run_options.get("series_average", False) and files_processed > 0:
            return
        msg = (
            f"{files_processed} of {discovered} discovered frame(s) processed"
        )
        reasons = self._format_skip_reasons()
        if reasons:
            msg = f"{msg}: {reasons}"
        only_already_processed = (
            files_processed == 0
            and discovered > 0
            and getattr(self, "_skip_reason_counts", Counter())
            == Counter({"already processed": discovered})
        )
        log = logger.info if only_already_processed else logger.warning
        log(msg)
        try:
            self.showLabel.emit(msg)
        except Exception:
            logger.debug("showLabel emit failed for zero-frame warning",
                         exc_info=True)

    def process_scan(self, frozen):
        if frozen.run_options.get("xye_only", False):
            raise TypeError("dynamic XYE output is not mounted")
        if frozen.run_options.get("series_average", False):
            raise TypeError("dynamic series average is not mounted")
        try:
            check_output_not_source(self._append_output_path(frozen, self.scan_name), **self._output_safety_args(frozen))
        except OutputCollisionError as exc:
            self._handle_output_collision(exc); return
        scan = None
        files_processed = 0
        files_processed_by_output = Counter()

        def record_processed(scan_obj, count):
            nonlocal files_processed
            count = int(count or 0)
            files_processed += count
            output = getattr(scan_obj, "data_file", None)
            if count > 0 and output:
                try:
                    key = os.path.normcase(os.path.abspath(os.fspath(output)))
                except (TypeError, ValueError):
                    logger.debug("invalid processed output path: %r", output)
                else:
                    files_processed_by_output[key] += count
            self.files_processed = self._last_files_processed = files_processed
            self.files_processed_by_output = dict(files_processed_by_output)
            self._last_files_processed_by_output = dict(files_processed_by_output)

        pending, poll_s = [], 0.1
        container = imageThread._frozen_source_is_container(self, frozen)
        while self.command != "stop":
            self._wait_if_paused(frozen)
            if self.command == "stop":
                break
            if not pending:
                started = time.time()
                read = self.get_next_image(frozen)
                img_file, scan_name, img_number, img_data, img_meta = read
                facts, img_meta = getattr(img_meta, "facts", {}), dict(img_meta or {})
                if img_data is None:
                    if not frozen.live_mode:
                        break
                    self.showLabel.emit("Watching for new files...")
                    time.sleep(poll_s)
                    poll_s = min(2.0, poll_s * 2.0)
                    continue
                poll_s = 0.1
                img_number = 1 if img_number is None else int(img_number)
                self.scan_name = scan_name
                if scan is None or scan.name != scan_name:
                    if scan is not None and not self._close_reduction_session():
                        raise RuntimeError("outgoing dynamic scan did not settle")
                    if self.command == "stop":
                        break
                    try:
                        scan = self.initialize_scan()
                    except OutputCollisionError as exc:
                        self._handle_output_collision(exc)
                        break
                    except AppendConfigMismatchError as exc:
                        self._handle_append_config_mismatch(exc)
                        break
                    except OSError as exc:
                        self._handle_initialize_scan_write_error(exc)
                        break
                    self._active_scan = scan
                    self._install_run_integrator(scan)
                pending.append((img_file, img_number, img_data, img_meta,
                    self.get_background(frozen, img_file, img_number, img_meta),
                    time.time() - started, facts))
            accepted = self._dispatch_batch(frozen, scan, pending)
            if not accepted:
                continue
            facts = pending[0][6]
            if facts.get("commit_path") is not None:
                self._commit_frame(
                    facts["commit_path"],
                    count_discovery=not facts.get("discovery_counted", False),
                )
            record_processed(scan, accepted)
            pending.clear()
            if frozen.source.source_kind == "image_file" and not container:
                break

        self.files_processed = files_processed
        self._last_files_processed = files_processed
        self.files_processed_by_output = dict(files_processed_by_output)
        self._last_files_processed_by_output = dict(files_processed_by_output)
        if frozen.batch_mode and files_processed > 0:
            self.sigUpdate.emit(-1)
        report_skip_summary = getattr(self, "_report_run_skip_summary", None)
        if callable(report_skip_summary):
            report_skip_summary(frozen, files_processed)
        logger.info('Total Files Processed: %d', files_processed)

    def _dynamic_append_intent(self, frozen, scan, label, facts):
        cached = facts.get("intent")
        if cached is not None:
            return cached
        from dataclasses import replace
        import xrd_tools.io as io_api
        from xrd_tools.session import required_result_modes
        observation = facts["observation"]
        prior = getattr(scan, "_same_run_intent", None)
        if prior is None:
            prefix = getattr(scan, "_committed_append_prefix", None)
            prior = prefix.intent if prefix is not None else None
        generation = int(prior.source.generation) + 1 if prior else int(frozen.generation)
        label = int(label)
        modes = tuple(f"{mode.kind}:{mode.key}"
                      for mode in required_result_modes(facts["plan"]))
        path = os.path.abspath(os.fspath(observation.path))
        if facts["container"]:
            accepted_extent = int(facts["source_frame_idx"]) + 1
            if label != accepted_extent:
                raise ValueError("container output labels must match their source prefix")
            if prior is None and accepted_extent != 1:
                raise ValueError("container output cannot fabricate a missing prefix")
            if prior is not None and tuple(prior.labels) != tuple(range(1, accepted_extent)):
                raise ValueError("container output requires an exact accepted prefix")
            external = tuple(io_api.AppendExternalMember(
                os.fspath(member.path), dataset_path, int(member.size), int(member.mtime_ns),
                int(start), int(stop), int(ordinal))
                for member, dataset_path, start, stop, ordinal in facts["external_members"])
            source = io_api.AppendSource(
                path=path, adapter_id=str(observation.adapter_id),
                size=int(observation.size), mtime_ns=int(observation.mtime_ns),
                extent=int(facts["extent"]), dataset_paths=facts["dataset_paths"], external_members=external,
                generation=generation)
            source = io_api.truncate_append_source(source, accepted_extent)
        else:
            members = () if prior is None else prior.source.image_members
            existing = next((member for member in members
                             if os.path.abspath(member.path) == path), None)
            if existing is not None:
                if ((existing.size, existing.mtime_ns) == observation.version_stamp
                        and label in prior.labels):
                    facts["intent"] = prior
                    return prior
                raise ValueError("an accepted TIFF member changed in place")
            ordinal = len(members)
            members = (*members, io_api.AppendImageMember(
                path=path, size=int(observation.size), mtime_ns=int(observation.mtime_ns),
                source_start=ordinal, source_stop=ordinal + 1, ordinal=ordinal))
            source = (io_api.AppendSource(
                path=path, adapter_id=str(observation.adapter_id),
                size=int(observation.size), mtime_ns=int(observation.mtime_ns),
                extent=1, image_members=members, generation=generation) if prior is None
                else replace(prior.source, extent=len(members), image_members=members,
                             generation=generation))
        labels = (*prior.labels, label) if prior else (label,)
        intent = io_api.AppendIntent(
            "entry", str(getattr(scan, "source_base", None) or ""),
            (str(prior.source_identity) if prior is not None else path),
            io_api.science_fingerprint(frozen.processing_mapping()), modes, source, labels)
        facts["intent"] = intent
        return intent

    def _dispatch_batch(self, frozen, scan, pending, *, force_save=False):
        if not pending:
            return 0
        img_file, label, data, meta, bg_raw, _read, facts = pending[0]
        live = facts.get("live")
        if live is None:
            live = self._build_scout(frozen, scan, pending[0][:6])
            live.source_file = os.path.abspath(os.fspath(img_file))
            live.source_frame_idx = int(facts["source_frame_idx"])
            live.skip_map_raw = scan.skip_2d or _raw_lives_in_source(img_file)
            if (frozen.gi.enabled and self._scan_session_adapter is None
                    and not self._gi_ranges_fully_pinned(frozen, scan)):
                sync_live_scan_gi_settings(
                    scan, incidence_motor=frozen.gi.scan_incidence_motor,
                    sample_orientation=frozen.gi.sample_orientation, tilt_angle=frozen.gi.tilt_angle)
                freeze_live_scan_gi_ranges(
                    scan, (live,), scan_name=str(scan.name),
                    global_mask=self.mask, integrator=scan._cached_integrator,
                    poni=self.poni, integrate_2d=not scan.skip_2d, gi_freeze_mode="first_frame")
            facts["live"] = live
        plan = facts.get("plan")
        if plan is None:
            plan = facts["plan"] = self._plan_cache.get(
                scan, integrate_2d=not scan.skip_2d)
        intent = self._dynamic_append_intent(frozen, scan, label, facts)
        adapter = self._scan_session_adapter
        if adapter is None:
            scan._same_run_intent = intent
            path = os.path.abspath(os.fspath(scan.data_file))
            try:
                adapter = self._mount_dynamic_reduction_session(
                    (int(frozen.generation), path), frozen=frozen, scan=scan,
                    plan=plan, pending_frame=live, output_path=path, gui_thread_id=self.gui_thread_id)
            except AppendRefused:
                raise
            except RuntimeError:
                if self._scan_session_adapter is not None:
                    raise
                retries = int(facts.get("dispatch_retries", 0)) + 1
                facts["dispatch_retries"] = retries
                if retries >= 32:
                    raise
                return 0
            self.sigUpdateFile.emit(
                str(scan.name), path, bool(frozen.gi.enabled),
                str(frozen.gi.scan_incidence_motor), frozen.source.source_kind == "image_file", False)
        elif intent != getattr(scan, "_same_run_intent", None):
            try:
                adapter.extend_live(intent)
            except RuntimeError as exc:
                if str(exc) != "same-run continuation requires a drained writer":
                    raise
                started = facts.setdefault("extend_retry_started", time.monotonic())
                if time.monotonic() - started >= self.PAUSE_DRAIN_TIMEOUT:
                    raise
                time.sleep(0.01)
                return 0
            facts.pop("extend_retry_started", None)
            scan._same_run_intent = intent
        from xrd_tools.session import DynamicFrameIdentity
        key = facts.setdefault("key", DynamicFrameIdentity(
            os.path.abspath(os.fspath(img_file)), int(facts["source_frame_idx"])))
        if not facts.get("discovered", False):
            adapter.discover(key, group=os.path.abspath(scan.data_file),
                             ordinal=int(label), output_label=int(label))
            facts["discovered"] = True
        token = facts["token"] = adapter.begin_attempt(key, source_revision=int(facts["source_revision"]))
        adapter.record_enqueued(token)
        if not adapter.submit(live, attempt_token=token):
            adapter.record_failed(token, retryable=True)
            return 0
        self._frames_since_save += 1
        if adapter.should_flush(self._frames_since_save,
                unsaved_in_memory=None, force=bool(force_save)):
            adapter.commit_epoch()
            self._frames_since_save = 0
        return 1

    def _maybe_warn_live_gi_clip(self, frozen) -> None:
        """One-time advisory for live GI runs (#75).

        In a live run the common GI output grid is frozen from the FIRST frame
        only — there's no lookahead to later incidence angles — so a scan that
        sweeps incidence can write clipped later frames (uniform axis, no crash,
        truncated tail).  Batch reprocessing uses the union of the incidence
        extremes (#70) and recovers the full range.  Phrased conditionally
        ("if this scan sweeps …") because live can't yet know whether incidence
        varies, and source-agnostic (no assumption about the frame source).
        Batch runs bracket all frames, so this never fires there.
        """
        if (not frozen.gi.enabled or frozen.batch_mode
                or getattr(self, '_warned_live_gi_clip', False)):
            return
        self._warned_live_gi_clip = True
        msg = ('Live GI: output range frozen from the first frame — if this '
               'scan sweeps a range of incidence angles, later frames may be '
               'clipped. Reprocess in batch for the full range.')
        logger.warning(msg)
        try:
            self.showLabel.emit(msg)
        except Exception:
            logger.debug("showLabel emit failed for live-GI clip warning",
                         exc_info=True)

    def _scout_pending_frames(self, frozen, pending):
        """Return bounded representative pending entries for the GI freeze.

        Selection helper for :meth:`_freeze_gi_1d_auto_range` /
        :meth:`_freeze_gi_2d_auto_ranges` (the streaming batch prepass —
        live production code; see the cluster note near
        ``_padded_axis_range``).  ssrl's ReductionSession has its own
        first-chunk freeze for the serial/chunked paths.  Real-data tests
        also inspect this selection directly.
        """
        if len(pending) <= 1:
            return list(pending)
        motor = frozen.gi.scan_incidence_motor
        try:
            float(motor)
            return [pending[0]]
        except (TypeError, ValueError):
            pass
        resolved = []
        any_unresolved = False
        for i, entry in enumerate(pending):
            meta = entry[3] or {}
            try:
                resolved.append((i, float(meta.get(motor))))
            except (TypeError, ValueError):
                any_unresolved = True
        idxs = set()
        if resolved:
            idxs.add(min(resolved, key=lambda item: item[1])[0])
            idxs.add(max(resolved, key=lambda item: item[1])[0])
        if any_unresolved or not resolved:
            idxs.update((0, len(pending) - 1))
        return [pending[i] for i in sorted(idxs)]

    def _build_scout(self, frozen, scan, entry):
        """Build a temporary ``LiveFrame`` for the headless freeze adapter."""
        img_file, img_number, img_data, img_meta, bg_raw, _ = entry[:6]
        img_data = self._apply_threshold_inline(frozen, img_data)
        frame_mask = self._resolve_frame_mask(frozen, scan, img_data)
        scratch = LiveFrame(
            img_number, img_data, poni=self.poni,
            scan_info=img_meta, static=True, gi=frozen.gi.enabled,
            th_mtr=frozen.gi.scan_incidence_motor, bg_raw=bg_raw,
            sample_orientation=frozen.gi.sample_orientation,
            tilt_angle=frozen.gi.tilt_angle,
            series_average=frozen.run_options.get("series_average", False),
            integrator=scan._cached_integrator,
            mask=frame_mask,
        )
        if img_file:
            scratch.source_file = os.path.abspath(str(img_file))
        if _raw_lives_in_source(img_file):
            scratch.source_frame_idx = int(img_number) - 1
        else:
            scratch.source_frame_idx = 0
        if frozen.gi.enabled:
            scratch._get_incident_angle()
        return scratch

    def _freeze_gi_1d_auto_range(self, frozen, scan, pending) -> None:
        """Compatibility wrapper for tests; delegates freeze to ssrl."""
        if not frozen.gi.enabled or not pending:
            return
        args = getattr(scan, 'bai_1d_args', None)
        if not isinstance(args, dict):
            return
        key = gi_1d_output_axis_key(args.get('gi_mode_1d', 'q_total'))
        if args.get(key) is not None:
            return
        sync_live_scan_gi_settings(
            scan,
            incidence_motor=frozen.gi.scan_incidence_motor,
            sample_orientation=frozen.gi.sample_orientation,
            tilt_angle=frozen.gi.tilt_angle,
        )
        scouts = [self._build_scout(frozen, scan, entry)
                  for entry in self._scout_pending_frames(frozen, pending)]
        freeze_live_scan_gi_ranges(
            scan,
            scouts,
            scan_name=str(getattr(scan, "name", "scan")),
            global_mask=self.mask,
            integrator=getattr(scan, "_cached_integrator", None),
            poni=self.poni,
            integrate_1d=True,
            integrate_2d=False,
            gi_freeze_mode="scout_union" if frozen.batch_mode else "first_frame",
        )

    def _freeze_gi_2d_auto_ranges(self, frozen, scan, pending) -> None:
        """Compatibility wrapper for tests; delegates freeze to ssrl."""
        if (not frozen.gi.enabled or frozen.run_options.get("xye_only", False) or getattr(scan, 'skip_2d', False)
                or not pending):
            return
        args = getattr(scan, 'bai_2d_args', None)
        if not isinstance(args, dict):
            return
        keys = _gi_2d_range_keys(args)
        if all(args.get(key) is not None for key in keys):
            return
        sync_live_scan_gi_settings(
            scan,
            incidence_motor=frozen.gi.scan_incidence_motor,
            sample_orientation=frozen.gi.sample_orientation,
            tilt_angle=frozen.gi.tilt_angle,
        )
        scouts = [self._build_scout(frozen, scan, entry)
                  for entry in self._scout_pending_frames(frozen, pending)]
        freeze_live_scan_gi_ranges(
            scan,
            scouts,
            scan_name=str(getattr(scan, "name", "scan")),
            global_mask=self.mask,
            integrator=getattr(scan, "_cached_integrator", None),
            poni=self.poni,
            integrate_1d=False,
            integrate_2d=True,
            gi_freeze_mode="scout_union" if frozen.batch_mode else "first_frame",
        )

    def _gi_ranges_fully_pinned(self, frozen, scan) -> bool:
        """True when every GI output range this run would auto-freeze is
        already explicitly set (T0-3).

        Mirrors the self-skip conditions of :meth:`_freeze_gi_1d_auto_range`
        (the active 1D mode's output-axis range key is set) and
        :meth:`_freeze_gi_2d_auto_ranges` (both of the active 2D mode's range
        keys are set; 2D is irrelevant for skip_2d runs).  When all
        relevant ranges are pinned there is no auto grid to freeze, so the
        whole-scan scout — and its fail-closed abort on unverifiable sources —
        is unnecessary.  Conservative: any non-dict args → False (let the
        normal prepass decide)."""
        args_1d = getattr(scan, 'bai_1d_args', None)
        if not isinstance(args_1d, dict):
            return False
        key = gi_1d_output_axis_key(args_1d.get('gi_mode_1d', 'q_total'))
        if args_1d.get(key) is None:
            return False
        if frozen.run_options.get("xye_only", False) or getattr(scan, 'skip_2d', False):
            return True
        args_2d = getattr(scan, 'bai_2d_args', None)
        if not isinstance(args_2d, dict):
            return False
        return all(args_2d.get(k) is not None
                   for k in _gi_2d_range_keys(args_2d))

    # ── BLOCKER 1: whole-scan GI grid freeze (streaming batch) ──────────────
    def _gi_freeze_whole_scan_prepass(self, frozen, scan) -> bool:
        """Freeze the GI common q/χ grid from the WHOLE scan's incidence range
        BEFORE the streaming session opens.

        The streaming batch session is built from the FIRST chunk only, so its
        ``scout_union`` would bracket chunk 1's incidence range and clip later,
        higher-incidence frames (BLOCKER 1).  Here a CHEAP metadata-only sweep
        finds the global lowest+highest-incidence frames, loads ONLY those two
        images, and runs the existing whole-scan freeze
        (:meth:`_freeze_gi_1d_auto_range` / :meth:`_freeze_gi_2d_auto_ranges`,
        which delegate to ``freeze_live_scan_gi_ranges``) to write the UNION
        ranges into ``scan.bai_*_args``.  The streaming session then rebuilds its
        plan with those ranges (``StandardPlanCache`` keys on ``bai_*_args``) and
        its own per-session freeze early-returns -> the whole batch shares the
        union grid.  The two scouts are integrated only inside the throwaway
        freeze session, never submitted to the streaming session -> no double
        processing.

        Batch + streaming + GI only.  True-live keeps ``first_frame`` serial (it
        has no whole scan ahead of time); the chunked batch path already builds
        the full pending up front so its first/last ARE the scan extremes.
        Runs once per scan.  Eiger single-master (one incidence per master) and
        fixed/manual incidence collapse to the existing behaviour (no whole-scan
        scout) -- the grid was never chunk-clipped there.

        **Policy (T0-4, Vivek 2026-06-09): warn-and-proceed.**  When the
        whole-scan incidence range cannot be established (unverifiable source,
        unreadable metadata, scout failure), the run PROCEEDS on the session's
        own first-chunk freeze with a one-time user-visible advisory, instead
        of aborting.  Rationale: frames are binned natively onto the common
        grid (values inside the range are exact — no interpolation), and at
        the beamline incidence varies too little for the cropped extreme tails
        to matter; the union sweep is kept where it is free (Image-Series with
        readable metadata).  The ONLY fail-closed exit left is a freeze that
        actually errors (e.g. a degenerate scout cake) — proceeding there
        would integrate onto a broken grid, not a slightly narrow one.

        Returns ``True`` when the caller may open the session, ``False`` when
        the run was aborted (freeze error only; caller must not proceed).
        """
        if not (frozen.gi.enabled and frozen.batch_mode):
            return True
        if getattr(self, "_gi_prepass_scan_id", None) == id(scan):
            return True
        if self._gi_ranges_fully_pinned(frozen, scan):
            # T0-3: every GI output range this run would auto-freeze is already
            # explicitly set — the freeze functions self-skip and the session's
            # own per-session freeze early-returns, so there is no auto grid to
            # chunk-clip and no scout sweep is needed.  Without this pre-flight
            # the "unverifiable" abort below fired even for fully-pinned runs
            # (and its "set fixed/manual GI ranges" remedy didn't work).
            self._gi_prepass_scan_id = id(scan)   # latch: decided for this scan
            return True
        try:
            status, scouts = self._gi_whole_scan_scout_entries(frozen, scan)
        except Exception as exc:
            # T0-4 warn-and-proceed: a scout failure means we lose the union
            # sweep, not correctness — the first-chunk freeze is acceptable.
            self._warn_gi_first_chunk_freeze(
                f"scout pre-pass raised ({exc})", scan)
            self._gi_prepass_scan_id = id(scan)
            return True
        if status == "abort":
            # Sweep ran but couldn't read enough incidences from metadata.
            self._warn_gi_first_chunk_freeze(
                "could not establish the whole-scan incidence range from "
                "frame metadata", scan)
            self._gi_prepass_scan_id = id(scan)
            return True
        if status == "unverifiable":
            # Image-Directory or named-motor Eiger: per-frame incidence can't
            # be cheaply swept up front (for Eiger it lives in the SPEC
            # sidecar, per frame).  T0-4 policy: proceed on the first-chunk
            # freeze with an advisory — values inside the grid are exact; only
            # extreme-incidence tails beyond it are cropped.
            # An operator LABEL, built here at the presentation edge: the frozen
            # source owns execution values, never GUI mode names (§43.1).
            source = ("Image Directory" if frozen.source.family == "directory"
                      else ("Eiger master"
                            if str(frozen.source.source_kind) == "eiger_master"
                            else "this source"))
            self._warn_gi_first_chunk_freeze(
                f"the whole-scan incidence range cannot be swept up front for "
                f"'{source}'", scan)
            self._gi_prepass_scan_id = id(scan)
            return True
        if status == "freeze":
            # These self-skip when the relevant ranges are already set, and
            # freeze the UNION over the two extreme scouts we hand them (NOT a
            # chunk).  A degenerate scout cake raises GIFreezeError here --
            # that stays FAIL-CLOSED via _abort_gi_prepass (the freeze ran and
            # produced a broken grid, not a narrow one) and must not escape
            # the worker thread (run() has no except).
            try:
                self._freeze_gi_1d_auto_range(frozen, scan, scouts)
                self._freeze_gi_2d_auto_ranges(frozen, scan, scouts)
            except Exception as exc:
                self._abort_gi_prepass(f"whole-scan grid freeze failed ({exc})")
                return False
        # status == "skip": fixed/manual/single-incidence/Eiger -- the session's
        # own freeze is correct (the grid was never chunk-clipped there).
        self._gi_prepass_scan_id = id(scan)   # latch only after success
        return True

    def _warn_gi_first_chunk_freeze(self, reason: str, scan=None) -> None:
        """T0-4 warn-and-proceed advisory: the GI output grid will be frozen
        from the first frames instead of the whole-scan incidence union.
        Values inside the grid are exact (frames bin natively onto it); only
        extreme-incidence tails beyond it are cropped — accepted at the
        beamline (incidence varies too little to matter).  One per scan: the
        prepass latches the scan id right after this fires.

        When ``scan`` is given the advisory is also stamped onto it
        (``gi_freeze_diagnostic``) so the writer persists it in
        ``/entry/reduction/config`` — the disclosure survives in the output
        file, not just as a transient GUI label."""
        msg = (
            'GI: ' + reason + ' — output grid set from the first frames; '
            're-integrate with explicit 1D/2D ranges if you need the full '
            'incidence extent.')
        logger.info(msg)
        if scan is not None:
            try:
                scan.gi_freeze_diagnostic = msg
            except Exception:
                logger.debug("could not stamp gi_freeze_diagnostic",
                             exc_info=True)
        try:
            self.showLabel.emit(msg)
        except Exception:
            logger.debug("showLabel emit failed for GI first-chunk-freeze "
                         "advisory", exc_info=True)

    def _abort_gi_prepass(self, reason: str) -> None:
        """Fail-closed exit for a GI freeze that actually ERRORED (degenerate
        scout cake etc.): surface a user-visible error and STOP the streaming
        GI run rather than integrate onto a broken grid.  (Unverifiable /
        unreadable-metadata cases warn-and-proceed instead — see
        _warn_gi_first_chunk_freeze.)"""
        msg = (
            'GI batch aborted: cannot freeze a whole-scan q/χ grid (' + reason +
            '). Refusing to integrate onto a partial grid that would clip later '
            'frames. Set Theta Motor to Manual and enter the incident angle, or '
            'check the per-frame metadata, then restart.')
        logger.error(msg)
        try:
            self.showLabel.emit(msg)
        except Exception:
            logger.debug("showLabel emit failed for GI prepass abort",
                         exc_info=True)
        # Stop the collection loop loudly (mirrors a write-failure stop); the
        # close path then drains/closes any partially-built session.  Under
        # command_lock so a concurrent GUI pause() can't overwrite it (RS-2).
        # getattr: tests drive this on duck holders without the lock.
        _lock = getattr(self, "command_lock", None)
        if _lock is not None:
            with _lock:
                self.command = 'stop'
        else:
            self.command = 'stop'

    def _frame_source_for(self, frozen, scan):
        r"""Build a core :class:`FrameSource` over this scan's per-file image
        series for the headless GI freeze prepass (ADR-0006 STEP 2).

        Returns:
          - :class:`ImageFileSource` for a single detector file (``single_img``);
          - :class:`TiffSeriesSource` over the STRICT ``^{scan}_\d+\.{ext}$`` file
            list from :meth:`_enumerate_scan_files` (NOT ``from_directory`` — a
            directory sweep would scoop up neighbour scans / background frames
            that the anchored regex deliberately excludes);
          - ``None`` for Eiger/HDF5 masters and Image-Directory sources that
            cannot be cheaply per-frame swept (the caller maps ``None`` to the
            old empty-``_enumerate_scan_files`` skip-vs-unverifiable split).

        The manifest metadata format is ``self.meta_ext`` so the headless sweep
        reads the EXACT sidecars the GUI does; a falsy meta_ext ⇒ no manifest ⇒
        the source's ``has_scan_manifest`` capability is False ⇒
        ``prepare_gi_freeze`` reports 'unverifiable' (warn-and-proceed)."""
        img_file = getattr(self, "img_file", None)
        if not img_file or _is_eiger_master(img_file):
            return None
        if imageThread._frozen_source_is_container(self, frozen):
            return None
        if frozen.source.family == "directory":
            return None
        meta_fmt = frozen.run_options.get("meta_ext") or None
        meta_dir = getattr(self, "meta_dir", None)
        if (frozen.source.source_kind == "image_file"):
            return ImageFileSource(
                img_file, metadata_format=meta_fmt, meta_dir=meta_dir)
        source_spec = frozen.thaw_source_spec()
        frozen_files = tuple(
            getattr(source_spec, "options", {}).get("files", ())
            if source_spec is not None else ()
        )
        files = (
            [(str(path), _get_scan_info(path)[1]) for path in frozen_files]
            if frozen_files
            else self._enumerate_scan_files(frozen)
        )
        if not files:
            return None
        return TiffSeriesSource(
            [fname for fname, _num in files],
            name=(
                getattr(source_spec, "options", {}).get("scan_name")
                if source_spec is not None else None
            ),
            metadata_format=meta_fmt, meta_dir=meta_dir)

    def _gi_whole_scan_scout_entries(self, frozen, scan):
        """Decide how to freeze the whole-scan GI grid and, when needed, gather
        the scout images.  Returns ``(status, entries)``:

        - ``("skip", [])`` — fixed/manual angle, Eiger/master, a single-frame
          scan, or a swept scan with one incidence: the existing chunk-local /
          session freeze is correct (the grid was never clipped).
        - ``("freeze", [lo_entry, hi_entry])`` — a varying-incidence per-file
          scan whose global incidence extremes were discovered by core; the
          ``entries`` are ``(img_file, img_number, img_data, img_meta, bg_raw,
          0.0)`` for the lowest+highest incidence frames, images loaded.
        - ``("unverifiable", [])`` — a source that can't be cheaply swept up
          front (Image-Directory, or a possibly-multi-master Eiger): the caller
          warns and proceeds on the first-chunk freeze (T0-4 policy).
        - ``("abort", [])`` — a multi-file scan whose incidence we genuinely
          cannot establish (fewer than two readable incidences across the whole
          series).  The caller also warns and proceeds (T0-4; name predates it).

        ADR-0006 STEP 2: the whole-scan incidence DISCOVERY now lives in
        ``xrd_tools.reduction.prepare_gi_freeze`` (image-free, never raises) over
        a core :class:`FrameSource`; xdart keeps only the scout LOAD + the freeze
        invocation (which can't move to core: chunk 1 can't see the last frame).
        """
        motor = frozen.gi.scan_incidence_motor
        try:
            float(motor)        # fixed/manual: one angle for the whole scan
            return "skip", []
        except (TypeError, ValueError):
            pass
        source = self._frame_source_for(frozen, scan)
        if source is None:
            # _frame_source_for() returns None for sources that can't be cheaply
            # swept per-frame.  Preserve the old skip-vs-unverifiable split:
            #   RISKY ("unverifiable", warn-and-proceed): Image-Directory per-file
            #     GI sweep, or a (possibly multi-master) Eiger master with a
            #     non-fixed motor — the session WOULD clip later frames.
            #   SAFE ("skip"): missing source attrs / no img_file — not an angle
            #     dependence sweep at all (so the grid was never chunk-clipped).
            img_file = getattr(self, "img_file", None) or ""
            if frozen.source.family == "directory":
                return "unverifiable", []
            if img_file and _is_eiger_master(img_file):
                return "unverifiable", []
            return "skip", []
        # Headless whole-scan incidence DISCOVERY (ADR-0006): core sweeps the
        # source's metadata-only manifest and pins the global incidence extremes
        # into plan.extra["gi_freeze_scout_indices"].  The minimal GI plan is a
        # vehicle: a non-None gi with an unpinned 1D output range so discovery
        # isn't short-circuited (whether THIS run needs a freeze at all is the
        # caller's _gi_ranges_fully_pinned decision, not this method's).  The
        # real incidence motor is passed explicitly.
        plan = ReductionPlan(
            integration_1d=Integration1DPlan(),  # radial_range None => freeze
            gi=GIMode(incidence_motor=str(motor)),
        )
        plan2, diag = prepare_gi_freeze(source, plan, incidence_motor=motor)
        if diag.status == "skip":
            # fixed/single incidence, <2 frames, or one distinct incidence.
            return "skip", []
        if diag.status == "unverifiable":
            # >=2 frames but <2 readable incidences (or no manifest): we cannot
            # establish the global range -> warn-and-proceed (old "abort").
            return "abort", []
        # diag.status == "frozen": load the extreme scouts BY INDEX against the
        # SAME source prepare_gi_freeze swept.  A TiffSeriesSource labels frames
        # by POSITION (1..N) in the strict file list, so the scout indices are
        # positional -- mapping each back through the source recovers the real
        # on-disk file + its scan img_number (non-contiguous / non-1-based
        # filenames stay correct; the contiguous Combi4 fixture would not catch
        # a positional/number confusion on its own).
        indices = plan2.extra.get("gi_freeze_scout_indices") or []
        entries = []
        for idx in indices:
            frame = source.frame_for(int(idx))
            fname = str(frame.source_path) if frame.source_path else ""
            meta = dict(frame.metadata)
            data = np.asarray(source.load_frame(int(idx)), dtype=float)
            _sname, img_number = _get_scan_info(fname)
            # bg is irrelevant to the frozen AXIS extent (geometry+incidence
            # driven), so a failing/missing background must NOT abort an
            # otherwise-valid GI run -- degrade to 0 for the axis-only scout.
            try:
                bg = self.get_background(frozen, fname, img_number, meta)
            except Exception:
                logger.debug("GI scout background failed for %s; using 0 for the "
                             "axis-only scout", fname, exc_info=True)
                bg = 0.0
            entries.append((fname, img_number, data, meta, bg, 0.0))
        return "freeze", entries

    def _enumerate_scan_files(self, frozen):
        """Sorted ``(fname, img_number)`` for a per-file scan series, metadata
        only (NO image read).  Returns ``[]`` for Eiger/HDF5 masters and
        Image-Directory sources that can't be cheaply per-frame swept (and,
        defensively, for any host missing the source attrs)."""
        img_file = getattr(self, "img_file", None)
        if not img_file or _is_eiger_master(img_file):
            return []
        if imageThread._frozen_source_is_container(self, frozen):
            return []
        if frozen.source.family == "directory":
            return []
        # A per-file series froze exactly one format token (its member's own
        # suffix); the container and directory shapes returned above.
        img_ext = (frozen.source.format_tokens or ("",))[0]
        scan_name = getattr(self, "scan_name", None)
        img_dir = frozen.source.filesystem_root
        if not scan_name or not img_dir:
            return []
        _series_re = re.compile(
            rf'^{re.escape(scan_name)}[_-]\d+\.{re.escape(img_ext)}$',
            re.IGNORECASE,
        )
        paths = [str(p) for p in _paths_with_suffix(Path(img_dir), f'.{img_ext}')
                 if _series_re.match(p.name)]
        out = []
        for fname in natural_sort_ints(paths):
            _sname, snumber = _get_scan_info(fname)
            out.append((fname, snumber))
        return out

    # ── Pause / Resume ──────────────────────────────────────────────────────
    def _wait_if_paused(self, frozen) -> None:
        """Block while a Pause is in effect, WITHOUT tearing down the scan or
        session.  Pause is a THIRD command state between the run state
        (``'start'``) and ``'stop'`` (the loops treat everything != 'stop' as go,
        so 'pause' must be special-cased here).

        On the first entry to pause, :meth:`_enter_pause` brings the file to a
        frame-boundary idle state and emits ``sigPaused`` so the GUI lifts the
        freeze guard for browsing; then we spin until ``command`` leaves
        ``'pause'`` — either ``'start'`` (resume: the caller continues its loop on
        the same open session) or ``'stop'`` (the caller's existing stop-check
        breaks next).  Call at the TOP of every processing loop, ABOVE the
        stop-check, so nothing new is read/submitted while paused.  The
        ``while`` exits on stop, so this is shutdown-safe (close() sets 'stop').
        """
        if self.command != 'pause':
            return
        self._enter_pause(frozen)
        while self.command == 'pause':
            time.sleep(0.05)
        # Resume (or stop): clear the session's paused flag (4a) so the next
        # adapter.submit() isn't rejected.  No-op without an adapter / when
        # not paused / finished; harmless on the stop path.
        adapter = getattr(self, '_scan_session_adapter', None)
        if self.command == "start" and adapter is not None:
            adapter.resume()

    #: Bound the pause drain so a hung pool worker (stalled IO / runaway pyFAI)
    #: can't deadlock the pause; the wait also bails early on Stop/close.
    PAUSE_DRAIN_TIMEOUT = 30.0

    def _enter_pause(self, frozen) -> None:
        """Publish Paused only after the dynamic epoch is durably quiescent."""
        try:
            adapter = self._scan_session_adapter
            if adapter is not None:
                if not adapter.quiesce(timeout=self.PAUSE_DRAIN_TIMEOUT):
                    raise RuntimeError("pause drain timed out before durable commit")
                if adapter.should_flush(
                    self._frames_since_save, unsaved_in_memory=None, force=True,
                ):
                    adapter.commit_epoch()
                    self._frames_since_save = 0
        except BaseException as exc:
            self._retain_dynamic_failure(exc, "Pause failed; run stopped")
            logger.error("dynamic pause failed; run stopped", exc_info=True)
            return
        self.sigPaused.emit()

    # ── Eiger HDF5 helpers ────────────────────────────────────────────────

    def _get_nframes(self, master_path):
        """Return frame count for a master file, 0 on failure."""
        return count_frames(master_path)

    def _eiger_source_facts(self, frame_idx):
        master = getattr(self, "_eiger_master_candidate", None)
        current = _source_observation(self._eiger_master_path)
        if (master is None or Path(master.path) != Path(
                self._eiger_master_path) or current != master):
            raise RuntimeError("Eiger master changed outside its retained cursor")
        descriptor = self._eiger_descriptor
        paths = tuple(getattr(descriptor, "segment_paths", ()) or ())
        if not paths and getattr(descriptor, "dataset_path", None):
            paths = (descriptor.dataset_path,)
        members, start = [], 0
        handle = (getattr(self._eiger_cursor, "_h5", None) or
                  getattr(self._eiger_fabio_handle, "h5", None))
        if handle is None:
            raise OSError("Eiger source has no retained detector owner")
        if not paths:
            data = handle.get("entry/data")
            if isinstance(data, h5py.Dataset):
                paths = ("/entry/data",)
            elif isinstance(data, h5py.Group):
                paths = tuple(f"{data.name}/{name}" for name in sorted(data)
                              if isinstance(data.get(name), h5py.Dataset))
        if not paths:
            entry = handle.get("entry")
            if isinstance(entry, h5py.Group):
                paths = tuple(f"{entry.name}/{name}" for name in sorted(entry)
                              if name.startswith("data") and isinstance(entry.get(name), h5py.Dataset))
        if not paths:
            raise ValueError("unsupported Eiger detector topology: no detector dataset")
        if handle is not None:
            kinds = set()
            for ordinal, selector in enumerate(paths):
                owner = handle
                parts = [part for part in str(selector).split("/") if part]
                if not parts:
                    raise ValueError("unsupported Eiger detector topology: empty path")
                for part in parts[:-1]:
                    link = owner.get(part, getlink=True)
                    if not isinstance(link, h5py.HardLink):
                        raise ValueError("unsupported Eiger detector topology: indirect ancestor")
                    owner = owner.get(part)
                    if not isinstance(owner, h5py.Group):
                        raise ValueError("unsupported Eiger detector topology: non-group ancestor")
                link = owner.get(parts[-1], getlink=True)
                if not isinstance(link, (h5py.HardLink, h5py.ExternalLink)):
                    raise ValueError("unsupported Eiger detector topology: indirect detector leaf")
                dataset = handle.get(selector)
                if (not isinstance(dataset, h5py.Dataset) or bool(dataset.is_virtual)
                        or dataset.id.get_create_plist().get_external_count()):
                    raise ValueError("unsupported Eiger detector topology: non-owned storage")
                kind = "external" if isinstance(link, h5py.ExternalLink) else "hard"
                kinds.add(kind)
                shape = dataset.shape
                stop = start + (1 if len(shape) == 2 else int(shape[0]))
                if kind == "external":
                    target = Path(link.filename)
                    if not target.is_absolute():
                        target = Path(self._eiger_master_path).parent / target
                    observation = _source_observation(target)
                    members.append((observation, str(link.path), start, stop, ordinal))
                start = stop
            if len(kinds) > 1:
                raise ValueError("unsupported Eiger detector topology: mixed storage owners")
        return dict(observation=master, source_revision=max(1, int(master.mtime_ns)),
                    container=True, extent=int(self._eiger_nframes), dataset_paths=paths,
                    external_members=tuple(members), source_frame_idx=int(frame_idx),
                    commit_path=None)

    @staticmethod
    def _h19_cursor_is_self_contained(cursor, descriptor):
        """True only when every resolved detector dataset is a hard link.

        ``segment_paths`` is empty for both a self-contained single dataset and
        a single external link, so its length alone cannot prove ownership.
        Inspect link metadata through the cursor's already-open handle; no new
        HDF5 open or pixel read is performed.
        """
        handle = getattr(cursor, "_h5", None)
        if handle is None or descriptor is None:
            return False
        paths = tuple(getattr(descriptor, "segment_paths", ()) or ())
        if not paths:
            dataset_path = getattr(descriptor, "dataset_path", None)
            paths = (dataset_path,) if dataset_path else ()
        if not paths:
            return False
        try:
            return all(
                handle.get(path, getlink=True).__class__.__name__ == "HardLink"
                for path in paths
            )
        except Exception:
            return False

    def _emit_container_count(self, path, nframes, *, authoritative=False):
        """DIR-2 lazy convergence: hand the GUI a container's frame
        count the moment it is known (open / retire).

        The identity stamp is captured here, beside the count, rather than by
        the later GUI-thread slot.  Otherwise a file can grow between emission
        and delivery and pair an old count with a new stamp.  ``authoritative``
        is additionally constrained to an exact-current H19 Candidate backed
        by a finalized, self-contained ``.nxs`` cursor; fabio/external-link
        counts describe only data landed so far and must never bulk-skip a
        later Append run.
        """
        sig = getattr(self, 'sigContainerCount', None)
        if not nframes:
            return
        imageThread._remember_source_snapshot(self, path, nframes)
        if sig is None:
            return
        try:
            stat = os.stat(path)
            stamp = (int(stat.st_size), int(stat.st_mtime_ns))
        except OSError:
            return
        if authoritative:
            authoritative = (
                imageThread._eiger_open_count_is_authoritative(self, path)
                and tuple(self._eiger_master_candidate.version_stamp) == stamp
            )
        try:
            sig.emit(str(path), int(nframes), stamp, bool(authoritative))
        except Exception:
            logger.debug('sigContainerCount emit failed', exc_info=True)

    def _remember_source_snapshot(self, path, nframes):
        """Retain source stamp + authoritative extent for frames we write."""
        if not path or int(nframes or 0) <= 0:
            return
        candidate = getattr(self, "_eiger_master_candidate", None)
        descriptor = getattr(self, "_eiger_descriptor", None)
        try:
            if (
                candidate is not None
                and Path(candidate.path) == Path(path)
            ):
                size, mtime_ns = candidate.version_stamp
            else:
                stat = os.stat(path)
                size, mtime_ns = int(stat.st_size), int(stat.st_mtime_ns)
        except OSError:
            return
        snapshot = {
            "size": int(size),
            "mtime_ns": int(mtime_ns),
            "frame_count": int(nframes),
            "dataset_path": (
                str(getattr(descriptor, "dataset_path", "") or "")
                if descriptor is not None else ""
            ),
            "self_contained": bool(
                getattr(descriptor, "self_contained", False)
                if descriptor is not None else False
            ),
        }
        cache = getattr(self, "_source_snapshot_by_path", None)
        if cache is None:
            cache = {}
            self._source_snapshot_by_path = cache
        cache[os.path.abspath(str(path))] = snapshot

    def _source_snapshot_for_frame(self, path):
        if not path:
            return {}
        return dict(getattr(
            self, "_source_snapshot_by_path", {}
        ).get(os.path.abspath(str(path)), {}))

    def _eiger_open_count_is_authoritative(self, path):
        """Whether the active raw cursor proves a final whole-file count."""
        candidate = getattr(self, "_eiger_master_candidate", None)
        descriptor = getattr(self, "_eiger_descriptor", None)
        cursor = getattr(self, "_eiger_cursor", None)
        return bool(
            candidate is not None
            and Path(candidate.path) == Path(path)
            and Path(path).suffix.lower() == ".nxs"
            and cursor is not None
            and descriptor is not None
            and bool(getattr(descriptor, "finalized", False))
            and imageThread._h19_cursor_is_self_contained(cursor, descriptor)
            and imageThread._h19_candidate_is_current(candidate)
        )

    def _eiger_open_count_can_bulk_compare(self, path):
        """Whether this open supplies a count usable for this Append pass.

        Self-contained finalized NeXus counts may also be persisted.  An
        external-link Eiger/fabio count is intentionally run-local: the master
        is reopened on every Append run because linked data files can grow
        without changing the master stamp.
        """
        if self._eiger_open_count_is_authoritative(path):
            return True
        if int(getattr(self, "_eiger_nframes", 0) or 0) <= 0:
            return False
        if (
            Path(path).suffix.lower() in {".h5", ".hdf5"}
            and getattr(self, "_eiger_fabio_handle", None) is not None
        ):
            return True
        # A sustained cursor may represent an external-link .nxs master or a
        # Bluesky-shaped *_master.h5 that fabio could not open.  Its current
        # descriptor extent is safe for this one Append pass even though it is
        # deliberately not persisted as authoritative across runs.
        return bool(
            getattr(self, "_eiger_cursor", None) is not None
            and getattr(self, "_eiger_descriptor", None) is not None
        )

    def _eiger_open_master(self, frozen, master_path):
        """Open (or switch to) an Eiger / NeXus HDF5 file, keeping the handle.

        Strategy
        --------
        - ``.nxs`` files → skip fabio and open the R2 ``ContainerCursor`` (one
          sustained h5py handle that resolves the detector dataset + frame count
          + wavelength + the metadata provider once).  fabio's EigerImage is
          tuned for Eiger master layouts and does not reliably find image arrays
          in Bluesky-style NeXus files.
        - Otherwise (``_master.h5`` etc.):
            1. **Persistent fabio handle (primary)** — ``fabio.EigerImage``
               is purpose-built for Eiger master files and handles all
               firmware variants, external-link layouts (``_data_*.h5``),
               and frame indexing natively.
            2. **ContainerCursor fallback** — if fabio fails (e.g. a
               Bluesky ``*_master.h5``), the cursor resolves and reads the
               stack through the headless h5py path.

        The cursor is created here, read by the prefetch worker, and closed by
        :meth:`_eiger_close_master` — all on the prefetch owner thread (the same
        ownership the raw h5py handle used); its ``ReadPlan`` sizes the bulk
        block and its provider serves per-frame metadata.
        """
        queued_candidate = getattr(self, "_eiger_master_candidate", None)
        if (queued_candidate is not None
                and Path(queued_candidate.path) != Path(master_path)):
            queued_candidate = None
        if queued_candidate is None:
            queued_candidate = _source_observation(master_path)
        self._eiger_close_master()
        self._eiger_open_state = "opening"
        # Provisional-until-proven: recorded per open/outcome so the Phase-3
        # watch gate has a fact that survives the provisional close (which
        # clears the descriptor).
        self._eiger_single_file_provisional = False

        ext = Path(master_path).suffix.lower()
        is_nexus = ext == '.nxs'

        if not is_nexus:
            try:
                # Primary: fabio (handles all Eiger layouts)
                self._eiger_fabio_handle = fabio.open(master_path)
                if _source_observation(master_path) != queued_candidate:
                    self._eiger_fabio_handle.close()
                    self._eiger_fabio_handle = None
                    raise OSError("Eiger master changed while fabio opened it")
                self._eiger_master_candidate = queued_candidate
                self._eiger_nframes = self._eiger_fabio_handle.nframes
                self._eiger_open_state = "ready"
                # fabio counts only LANDED data files; zero frames on a live
                # master is a nascent state, not a definitive empty.
                self._eiger_single_file_provisional = (self._eiger_nframes == 0)
                self._emit_container_count(master_path,
                                           self._eiger_nframes)
                return
            except Exception as e:
                # DIR-2b: fabio's EigerImage rejects non-Eiger-shaped HDF5
                # (Bluesky NXWriter files named *_master.h5, or masters
                # with unreachable external links) with NotGoodReader — a
                # RuntimeError, NOT an IOError — which used to escape
                # here, crash the prefetch worker and END the whole
                # directory run ('No frames discovered'). Fall through
                # to the same h5py reader the .nxs branch uses; a young
                # zero-frame fallback is retried by the directory reader.
                logger.warning(
                    "fabio could not open %s (%s); falling back to h5py",
                    master_path, e)
                self._eiger_fabio_handle = None

        # h5py path (primary for .nxs, fallback for .h5/.hdf5 master files) via
        # the R2 ContainerCursor — one open handle, created HERE on the prefetch
        # owner thread, supplies descriptor + frame count + wavelength + the
        # per-frame metadata provider + all sustained reads (no repeated
        # per-container traversal).  fabio stays primary for real Eiger above.
        try:
            from xrd_tools.sources.cursor import (
                ContainerCursor,
                ContainerNotReadyError,
            )
            from xrd_tools.sources.probe import ProbeState

            cursor = ContainerCursor(
                master_path,
                entry='entry',
                candidate=queued_candidate,
            ).open()
            desc = cursor.descriptor
            if desc.state is ProbeState.PROCESSED_OUTPUT:
                # F-NXS-2: a processed xdart output was swept into the raw source
                # tree.  It carries only integrated results, never a detector
                # frame, so skip it per file (nframes=0 -> the caller's
                # retire-and-advance path) instead of re-ingesting an integrated
                # pattern or ending the stream.
                cursor.close()
                logger.info('Skipping processed xdart output (not a raw '
                            'acquisition): %s', master_path)
                self._record_skip_reason('processed xdart output')
                self._eiger_open_state = "processed xdart output"
                self._eiger_close_master()
                self._eiger_nframes = 0
                return
            if (
                frozen.source.family == "directory"
                and (
                    desc.state is ProbeState.IN_PROGRESS
                    or not bool(getattr(desc, "finalized", True))
                )
            ):
                # Directory mode consumes a container only after finality.  The
                # descriptor came from this same cursor open, replacing the old
                # is_unfinalized_nxwriter preflight (and its duplicate open).
                cursor.close()
                logger.info(
                    "Deferring provisional container until a later Run poll: %s",
                    master_path,
                )
                self._eiger_open_state = "not ready"
                self._eiger_close_master()
                self._eiger_nframes = 0
                self._eiger_single_file_provisional = True
                return
            if desc.dataset_path is None:
                cursor.close()
                log = (logger.debug if frozen.live_mode
                       and frozen.source.family == "directory"
                       else logger.warning)
                log('Could not find image dataset in %s', master_path)
                # ``finalized`` defaults true for formats without a lifecycle
                # marker.  Only a positively identified, finalized NXWriter
                # container is therefore terminally detectorless.  A plain or
                # still-unclassified HDF5 shell may be visible before its
                # detector tree lands and must retain the live retry path.
                self._eiger_open_state = (
                    "finalized no detector dataset"
                    if (
                        bool(getattr(desc, "is_bluesky", False))
                        and bool(getattr(desc, "finalized", False))
                    )
                    else "no detector dataset"
                )
                self._eiger_close_master()
                self._eiger_nframes = 0
                return
            self._eiger_bind_cursor(cursor)
            self._eiger_master_candidate = queued_candidate
            self._eiger_open_state = "ready"
            # R4A-1(iii): this container is now JIT-CLASSIFIED and has a live
            # metadata provider, so it is the earliest honest moment to hand the
            # GUI its motor/counter names.  Display-only and once per run.
            imageThread._publish_gi_source_motors(self, frozen, master_path)
            # An unfinalized container (NXWriter without end_time — e.g. a
            # created-but-still-empty detector dataset) is provisional even
            # though the open succeeded (NXS-SF-2).
            self._eiger_single_file_provisional = not bool(
                getattr(desc, 'finalized', True))
            self._emit_container_count(master_path, self._eiger_nframes)
        except ProcessedXdartInputError:
            # F-NXS-2: a processed xdart output was swept into the raw source
            # tree.  It carries only integrated results, never a detector frame,
            # so the finder refuses it BEFORE the largest-3D fallback could hand
            # back /entry/integrated_2d/intensity (a cake) as a detector stack.
            # Skip it per file (nframes=0 -> the caller's retire-and-advance
            # path) so the directory worker continues to the next valid raw
            # acquisition instead of re-ingesting an integrated pattern or
            # ending the stream.
            logger.info('Skipping processed xdart output (not a raw '
                        'acquisition): %s', master_path)
            self._record_skip_reason('processed xdart output')
            self._eiger_open_state = "processed xdart output"
            self._eiger_close_master()
            self._eiger_nframes = 0
        except ContainerNotReadyError as e:
            # NXS-SF-2: a nascent container (an NXWriter shell before its
            # detector tree, or a detector link whose target has not landed)
            # is a typed provisional state, not an open error — keep it
            # retryable by the caller's defer (directory) or watch
            # (single-file) machinery.
            logger.info('Container not ready yet (still being written), '
                        'will retry: %s (%s)', master_path, e)
            self._eiger_open_state = "not ready"
            self._eiger_close_master()
            self._eiger_nframes = 0
            self._eiger_single_file_provisional = True
        except OSError as e:
            # A live single-file source can be visible before the writer
            # releases its sharing lock.  Treat that first-open denial exactly
            # like a transient growth-reopen denial so Phase 3 remains armed;
            # batch/non-live callers retain the terminal open-error behavior.
            if (frozen.live_mode
                    and not frozen.batch_mode):
                logger.info('Container temporarily unavailable, will retry: '
                            '%s (%s)', master_path, e)
                self._eiger_open_state = "not ready"
                self._eiger_close_master()
                self._eiger_nframes = 0
                self._eiger_single_file_provisional = True
            else:
                logger.error('Error opening HDF5/NeXus file %s: %s',
                             master_path, e)
                self._eiger_open_state = "open error"
                self._eiger_close_master()
                self._eiger_nframes = 0
        except Exception as e:
            logger.error('Error opening HDF5/NeXus file %s: %s', master_path, e)
            self._eiger_open_state = "open error"
            self._eiger_close_master()
            self._eiger_nframes = 0

    @staticmethod
    def _bluesky_source_column_names(master_path):
        """``(motors, counters)`` NAMES for a Bluesky container, or ``None``.

        Deliberately the SAME io authority the pre-Run preview uses
        (``bluesky_all_motor_names`` / ``bluesky_counters``), so a dropdown
        filled mid-run by :meth:`_publish_gi_source_motors` is identical to one
        the preview would have filled.  Names only -- no per-frame table, no
        pixel, and the caller invokes it at most once per run.
        """
        try:
            import h5py
            from xrd_tools.io.bluesky_nexus import (
                bluesky_all_motor_names,
                bluesky_counters,
                is_bluesky_nxwriter,
                resolve_nxentry,
            )

            with h5py.File(master_path, 'r') as h5:
                entry = resolve_nxentry(h5)
                if entry is None or not is_bluesky_nxwriter(entry):
                    return None
                return (tuple(str(name)
                              for name in bluesky_all_motor_names(entry)),
                        tuple(str(name) for name in bluesky_counters(entry)))
        except Exception:
            logger.debug("Bluesky source column names unavailable for %s",
                         master_path, exc_info=True)
            return None

    def _publish_gi_source_motors(self, frozen, master_path):
        """R4A-1(iii): publish the FIRST classified container's motor names.

        Exactly one immutable :class:`GISourceMotorDiscovery` per accepted run.
        The latch is keyed on the accepted configuration OBJECT, so a second
        container in the same run, a re-opened master, and a replayed open are
        all silent, while a genuinely new run re-arms with no reset step that
        could be forgotten.

        Publication carries ``frozen`` -- the exact configuration this routed
        call was given -- and never reads it back into the run: the worker keeps
        integrating with the motor frozen at Run.
        """
        if getattr(self, "_gi_source_motors_published", None) is frozen:
            return
        if frozen is None:
            return
        provider = getattr(self, "_eiger_provider", None)
        descriptor = getattr(self, "_eiger_descriptor", None)
        # "JIT-classified WITH a metadata provider": a fabio-primary Eiger
        # master has no provider and no embedded columns, and is not a source of
        # motor names at all.
        if provider is None or not bool(getattr(descriptor, "is_bluesky", False)):
            return
        sig = getattr(self, "sigGISourceMotors", None)
        if sig is None:
            return
        columns = imageThread._bluesky_source_column_names(master_path)
        if columns is None:
            return
        motors, counters = columns
        # Latch BEFORE emitting so even a re-entrant/duplicate delivery attempt
        # cannot produce a second publication for this run.
        self._gi_source_motors_published = frozen
        if not motors and not counters:
            return
        try:
            sig.emit(GISourceMotorDiscovery(
                run_configuration=frozen,
                source_path=str(master_path),
                motors=tuple(motors),
                counters=tuple(counters),
            ))
        except Exception:
            logger.debug("GI source motor publication failed for %s",
                         master_path, exc_info=True)

    def _eiger_cursor_binding(self, cursor):
        """Build, but do not install, a coherent open-cursor binding.

        Refresh prepares this complete tuple before replacing anything, so a
        transient read/open failure keeps the old cursor and its matching
        descriptor, read plan, and metadata provider available for retry.
        """
        from xrd_tools.core.staging import source_block_budget_bytes
        from xrd_tools.sources.read_plan import plan_reads

        desc = cursor.descriptor
        read_plan = plan_reads(
            desc.frame_count, desc.frame_shape, desc.dtype, desc.chunks,
            source_block_budget_bytes(), two_d=desc.is_2d)
        provider = cursor.metadata_provider()
        # Keep the provider lazy.  Complete Append containers are retired from
        # descriptor.frame_count before any per-frame metadata table or pixel
        # is materialized.
        return desc, int(cursor.frame_count), read_plan, provider

    def _eiger_install_cursor_binding(self, cursor, binding):
        """Publish one fully prepared cursor binding on the owner thread."""
        desc, nframes, read_plan, provider = binding
        self._eiger_cursor = cursor
        self._eiger_descriptor = desc
        self._eiger_nframes = nframes
        self._eiger_read_plan = read_plan
        self._eiger_provider = provider

    def _eiger_bind_cursor(self, cursor):
        """Adopt an OPEN cursor after its plan and provider are ready."""
        self._eiger_install_cursor_binding(
            cursor, self._eiger_cursor_binding(cursor))

    def _eiger_reopen_cursor(self):
        """Close and reopen the h5py-backed cursor to pick up frames written
        since it was opened — the growth-refresh analogue of the fabio reopen
        (same reopen-to-refresh mechanism, NOT a new SWMR/recovery policy; the
        pop still defers unfinalized masters).  Best-effort; recomputes the
        frame count + read plan + provider."""
        from xrd_tools.sources.cursor import ContainerCursor

        path = self._eiger_master_path
        if path is None:
            return False
        old_cursor = self._eiger_cursor
        replacement = None
        try:
            candidate = _source_observation(path)
            replacement = ContainerCursor(
                path, entry='entry', candidate=candidate).open()
            if replacement.descriptor.dataset_path is None:
                replacement.close()
                return False
            binding = self._eiger_cursor_binding(replacement)
        except Exception:
            if replacement is not None:
                replacement.close()
            raise

        # Swap the coherent binding first.  Only the successful replacement can
        # release the old source owner; this keeps transient failures retryable.
        self._eiger_install_cursor_binding(replacement, binding)
        self._eiger_master_candidate = candidate
        if old_cursor is not None and old_cursor is not replacement:
            try:
                old_cursor.close()
            except Exception:
                logger.debug('Failed to close replaced Eiger cursor', exc_info=True)
        return True

    def _eiger_single_file_growth_outcome(self, frozen):
        """Classify a single-file source at exhaustion: 'grown'|'wait'|'end'.

        NXS-SF-1/SF-2 policy (handoff §6): a live run observes growth through
        ONE bounded transactional cursor reopen (build-then-swap — a cached
        ``Dataset.shape`` never grows without SWMR, so close-and-reopen is the
        only honest refresh) and continues from the prior logical index.  A
        nascent/not-ready container stays provisional ('wait'); a finalized
        fully-consumed container is a fixed point ('end').  Batch/non-live
        and Stop never wait.  This method never sleeps — the caller's watch
        loop paces the retries with its own backoff.
        """
        from xrd_tools.sources.cursor import ContainerNotReadyError

        stop_evt = getattr(self, '_prefetch_stop_evt', None)
        if (not frozen.live_mode
                or frozen.batch_mode
                or getattr(self, 'command', None) == 'stop'
                or (stop_evt is not None and stop_evt.is_set())):
            return 'end'
        if self._eiger_fabio_handle is not None:
            # fabio-primary Eiger: fabio's nframes counts only LANDED data
            # files, so a live catch-up (reader faster than the detector's
            # file writer) looks exactly like exhaustion.  That is never a
            # terminal fact for a LIVE run — wait and let the watch loop
            # re-poll (pre-latch behavior); batch/stop already returned
            # 'end' above.
            self._eiger_single_file_provisional = True
            return 'wait'
        # Close BEFORE reopening: HDF5 shares one same-process file image, so
        # a "fresh" open while the exhausted cursor is still alive would see
        # the STALE cached shape and growth could never be observed at the
        # first exhaustion poll.  Nothing is lost on a reopen failure — every
        # known frame is already consumed and the 'wait'/'end' paths close
        # the source regardless.
        self._eiger_close_master()
        try:
            if not self._eiger_refresh_master_handle(raise_errors=True):
                # Reopen succeeded but resolved no detector dataset: a
                # definitive imageless container (a still-writing shell
                # raises the typed not-ready error instead).
                self._eiger_single_file_provisional = False
                return 'end'
        except ContainerNotReadyError:
            self._eiger_single_file_provisional = True
            return 'wait'          # nascent shell / unlanded link: retry later
        except OSError:
            # Transient I/O — including a Windows-style sharing denial
            # (PermissionError) while the writer holds the file — must stay
            # provisional, never latch the run done (no platform check: the
            # exception type IS the contract).
            logger.debug('single-file growth reopen transiently failed for '
                         '%s', self._eiger_master_path, exc_info=True)
            self._eiger_single_file_provisional = True
            return 'wait'
        except Exception:
            logger.debug('single-file growth reopen failed terminally for %s',
                         self._eiger_master_path, exc_info=True)
            self._eiger_single_file_provisional = False
            return 'end'           # structural (unsupported rank, ...)
        self._emit_container_count(self._eiger_master_path, self._eiger_nframes)
        if self._eiger_frame_idx < self._eiger_nframes:
            return 'grown'
        desc = self._eiger_descriptor
        if desc is not None and getattr(desc, 'finalized', False):
            self._eiger_single_file_provisional = False
            return 'end'
        # Unfinalized and not grown: provisional.  Record the fact as a plain
        # flag — the close that accompanies the 'wait' sentinel clears
        # _eiger_descriptor, so the Phase-3 watch gate cannot read finality
        # off the (gone) descriptor.
        self._eiger_single_file_provisional = True
        return 'wait'

    def _eiger_single_file_watchable(self, frozen):
        """NXS-SF-2: whether a live single-file run should keep watching even
        though the initial collect produced NO scan — i.e. the container is
        still provisional (a nascent NXWriter shell, an unfinalized container
        whose detector dataset is still empty, or links whose targets have
        not landed).  Decided from the PROVISIONAL FLAG the open/outcome
        classifications record — never from ``_eiger_descriptor``, which the
        provisional close has already cleared by the time the Phase-3 gate
        runs.  Directory runs keep their own defer/queue machinery;
        definitive outcomes (processed output, finalized imageless, open
        error, finalized-and-consumed) never watch."""
        if frozen.source.family == "directory":
            return False
        if getattr(self, '_eiger_single_file_done', False):
            return False
        if getattr(self, '_eiger_master_path', None) is None:
            return False
        if getattr(self, '_eiger_open_state', None) == 'not ready':
            return True
        return bool(getattr(self, '_eiger_single_file_provisional', False))

    @staticmethod
    def _h5_stack_nframes(dset):
        """Logical frame count of an h5py image dataset.

        A 2-D dataset is a SINGLE-exposure detector frame — one logical frame
        (the same (1, H, W) normalization NexusImageStack applies, F6) — never
        ``shape[0]`` rows.  Every ``_eiger_nframes`` assignment from a live
        h5py dataset must come through here (the growth re-checks included), or
        a one-frame ``.nxs`` would be re-counted as H "frames" of rows."""
        return 1 if dset.ndim == 2 else int(dset.shape[0])

    def _eiger_close_master(self):
        """Close the active master's handles (fabio and/or the R2 cursor).

        Called on master-switch and imageless/processed skip (prefetch thread),
        and once more at run-start/run-end after ``_prefetch_stop_prior`` has
        confirmed the prefetch worker is dead — the same proven ownership the
        raw h5py handle used, so no live handle is ever touched across threads.
        ``ContainerCursor.close()`` is idempotent, so the post-death safety call
        is a no-op when the worker already closed on switch."""
        self._eiger_read_plan = None
        self._eiger_provider = None
        self._eiger_descriptor = None
        self._eiger_master_candidate = None
        if self._eiger_fabio_handle is not None:
            try:
                self._eiger_fabio_handle.close()
            except (IOError, OSError) as e:
                logger.debug("Failed to close fabio handle: %s", e)
            self._eiger_fabio_handle = None
        if self._eiger_cursor is not None:
            try:
                self._eiger_cursor.close()
            except Exception as e:
                logger.debug("Failed to close Eiger cursor: %s", e)
            self._eiger_cursor = None

    def _read_eiger_metadata(self, frozen, master_path):
        """Read per-master metadata once, returning a per-frame copy."""
        meta_ext = frozen.run_options.get("meta_ext")
        if not meta_ext:
            return {}
        key = (
            os.path.abspath(str(master_path)),
            str(meta_ext),
            os.path.abspath(str(self.meta_dir)) if self.meta_dir else '',
        )
        if key not in self._eiger_metadata_cache:
            self._eiger_metadata_cache[key] = read_image_metadata(
                master_path, meta_format=meta_ext, meta_dir=self.meta_dir,
            )
        return dict(self._eiger_metadata_cache[key])

    # ── Bluesky/NXWriter embedded per-frame metadata ─────────────────────
    #
    # A Bluesky (apstools ``NXWriter``) ``.nxs`` image stack carries its scan
    # motors + counters PER FRAME inside the HDF5 tree (``entry/data/<name>``)
    # rather than in a sidecar.  These helpers surface that table so each frame's
    # ``scan_info`` gets the real motor/counter values — the GI incidence angle
    # then resolves from the file's motor, the counters normalize, and the
    # written output carries them.  Every path is guarded so a non-Bluesky source
    # (Eiger master, plain NeXus, TIFF series, SPEC sidecar) is untouched.

    def _bluesky_source_for(self, master_path):
        """Read + cache a Bluesky/NXWriter source's per-frame table + wavelength.

        Returns ``{'table': {col: np.ndarray}, 'constants': {name: float},
        'wavelength_A': float | None}`` for a Bluesky ``.nxs`` master, else
        ``None`` (``constants`` holds per-scan values — fixed motors + the eiger
        counting time — broadcast to every frame).
        Read once per master and cached (including the ``None`` for a non-Bluesky
        source), so a plain Eiger master or a non-``.nxs`` source keeps the
        sidecar path unchanged and never re-opens the file per frame."""
        if not master_path:
            return None
        cache = getattr(self, '_bluesky_source_cache', None)
        if cache is None:
            cache = self._bluesky_source_cache = {}
        key = os.path.abspath(str(master_path))
        if key in cache:
            return cache[key]
        info = None
        try:
            if Path(master_path).suffix.lower() in ('.nxs', '.h5', '.hdf5'):
                import h5py
                from xrd_tools.io.bluesky_nexus import (
                    bluesky_constant_metadata,
                    bluesky_per_frame_table,
                    bluesky_wavelength,
                    is_bluesky_nxwriter,
                    resolve_nxentry,
                )
                with h5py.File(master_path, 'r') as h5:
                    entry = resolve_nxentry(h5)
                    if entry is not None and is_bluesky_nxwriter(entry):
                        table = {
                            k: np.asarray(v)
                            for k, v in bluesky_per_frame_table(entry).items()
                        }
                        # Per-scan CONSTANTS broadcast to every frame: the FIXED
                        # (non-scanned) motors — incl. a held-constant GI incidence
                        # motor, so incidence resolves whether scanned OR fixed —
                        # plus the eiger counting time.  Their values live in the
                        # baseline stream / detector config, not per-frame data.
                        constants = bluesky_constant_metadata(
                            entry, exclude=table.keys())
                        wl = bluesky_wavelength(entry)
                        info = {
                            'table': table,
                            'constants': constants,
                            'wavelength_A': (
                                float(wl) if np.isfinite(wl) else None),
                        }
        except Exception:
            logger.debug("Bluesky source read failed for %s",
                         master_path, exc_info=True)
            info = None
        cache[key] = info
        return info

    def _bluesky_frame_row(self, master_path, frame_idx):
        """Per-frame motor + counter values for a Bluesky ``.nxs`` frame.

        Returns ``{col: float}`` sliced from the embedded per-frame table
        (motors + counters + ``EPOCH``) at the 0-based *frame_idx*, or ``{}``
        when the master is not a Bluesky file / the index is out of range."""
        info = self._bluesky_source_for(master_path)
        if not info:
            return {}
        row = {}
        for col, arr in info['table'].items():
            try:
                if 0 <= frame_idx < len(arr):
                    row[col] = float(arr[frame_idx])
            except (TypeError, ValueError, IndexError):
                continue
        # Fixed motors (e.g. a constant GI incidence angle) broadcast the same
        # value to every frame; setdefault so a scanned column always wins.
        for col, val in info.get('constants', {}).items():
            row.setdefault(col, float(val))
        return row

    def _frame_scan_info(self, frozen, master_path, frame_idx):
        """The frame's ``scan_info``: the sidecar/scalar metadata (unchanged)
        overlaid with the Bluesky per-frame motor + counter row when the master
        is a Bluesky ``.nxs``.  For any other source this is exactly
        :meth:`_read_eiger_metadata` (frame_idx ignored)."""
        base = self._read_eiger_metadata(frozen, master_path)
        row = self._bluesky_row(master_path, frame_idx)
        if row:
            base.update(row)
        return base

    def _bluesky_row(self, master_path, frame_idx):
        """Per-frame Bluesky motor+counter row for *master_path*.

        Prefers the SUSTAINED cursor's MATERIALIZED metadata provider when
        *master_path* is the master the cursor is open on — the R2 collapse of
        the repeated per-container traversal, served from the one open handle
        with no per-frame reopen.  Falls back to the legacy per-master cached
        reader for the fabio-primary Eiger path (no cursor) or any other master
        (and for duck-typed test hosts with no cursor)."""
        provider = getattr(self, '_eiger_provider', None)
        current = getattr(self, '_eiger_master_path', None)
        if (provider is not None and current is not None
                and os.path.abspath(str(master_path))
                == os.path.abspath(str(current))):
            try:
                row = dict(provider.metadata_for(frame_idx))
                # ``metadata_for`` deliberately omits scanned motor columns.
                # A wrangler frame row is the complete per-frame record, so
                # append indexed motors after constants; scanned values win.
                for name, values in provider.motors().items():
                    try:
                        value = values[int(frame_idx)]
                    except (IndexError, KeyError, TypeError):
                        continue
                    row[str(name)] = (
                        value.item() if isinstance(value, np.generic) else value)
                return row
            except Exception:
                logger.debug('cursor provider metadata_for failed for %s',
                             master_path, exc_info=True)
                return {}
        return self._bluesky_frame_row(master_path, frame_idx)

    def _stamp_bluesky_wavelength(self, scan):
        """Record a Bluesky/NXWriter source's embedded wavelength on *scan* as a
        LOW-precedence provenance fallback (metres), so the processed output —
        and the run-end reload — carries a real wavelength instead of NaN.

        The PONI-derived integrator wavelength still wins for geometry (see
        nexus_writer._write_instrument's precedence order); this only fills
        ``mg_args`` when it still holds the constructor sentinel / no real value,
        so a PONI-supplied wavelength is never overridden.  Guarded: a
        non-Bluesky source or any read failure is a silent no-op."""
        try:
            info = self._bluesky_source_for(getattr(self, 'img_file', None))
            if not info:
                return
            wl_A = info.get('wavelength_A')
            if wl_A is None or not np.isfinite(wl_A) or wl_A <= 0:
                return
            mg = getattr(scan, 'mg_args', None)
            if not isinstance(mg, dict):
                return
            from xdart.modules.wavelength import is_default_wavelength_sentinel_m
            cur = mg.get('wavelength')
            if cur is None or is_default_wavelength_sentinel_m(cur):
                mg['wavelength'] = float(wl_A) * 1e-10
        except Exception:
            logger.debug("Bluesky wavelength stamp skipped", exc_info=True)

    @staticmethod
    def _eiger_scan_name(master_path):
        # Canonical rule (Codex F2): container .nxs/.h5 keeps the FULL stem,
        # '_master' is stripped — one shared source-of-truth with the GUI.
        return scan_name_from_source(master_path)

    def _eiger_retire_master(self, path):
        """Retire a completed or stably imageless path for this run."""
        if not path:
            return
        done = getattr(self, "_eiger_done_masters", None)
        if done is None:
            done = self._eiger_done_masters = set()
        retries = getattr(self, "_eiger_retry_after", None)
        if retries is not None:
            retries.pop(path, None)
        zero_seen = getattr(self, "_eiger_zero_frame_seen", None)
        if zero_seen is not None:
            zero_seen.pop(path, None)
        done.add(path)

    def _eiger_defer_zero_frame_master(self, frozen, path):
        """Retry a young zero-frame container without blocking later scans."""
        if (not frozen.live_mode
                or frozen.source.family != "directory"
                or getattr(self, "_eiger_open_state", None)
                in {
                    "processed xdart output",
                    "finalized no detector dataset",
                }):
            return False
        try:
            stat = os.stat(path)
            age = max(0.0, time.time() - stat.st_mtime)
            stamp = (
                int(getattr(stat, "st_dev", 0)),
                int(getattr(stat, "st_ino", 0)),
                int(stat.st_size),
                int(stat.st_mtime_ns),
            )
        except OSError:
            age = 0.0
            stamp = None
        deadline = float(getattr(
            self, "CONTAINER_READY_DEADLINE", _CONTAINER_READY_DEADLINE))
        now = time.monotonic()
        zero_seen = getattr(self, "_eiger_zero_frame_seen", None)
        if zero_seen is None:
            zero_seen = self._eiger_zero_frame_seen = {}
        observed = zero_seen.get(path)
        first_observation = observed is None or observed[0] != stamp
        if first_observation:
            observed = (stamp, now)
            zero_seen[path] = observed
        stable_for = now - observed[1]
        # A newly visible path always earns one readiness retry.  Beamline
        # shares may preserve an old writer mtime or have server/client clock
        # skew; using wall-clock age on the first observation permanently
        # retired a nascent container before its detector tree became visible.
        if (deadline <= 0.0
                or (not first_observation
                    and max(age, stable_for) >= deadline)):
            zero_seen.pop(path, None)
            return False
        retries = getattr(self, "_eiger_retry_after", None)
        if retries is None:
            retries = self._eiger_retry_after = {}
        retry_s = max(0.01, float(getattr(
            self, "CONTAINER_READY_RETRY", _CONTAINER_READY_RETRY)))
        retries[path] = now + retry_s
        logger.debug(
            "Deferring young zero-frame container for retry in %.2fs "
            "(age %.2fs, stable %.2fs, state=%s): %s",
            retry_s, age, stable_for,
            getattr(self, "_eiger_open_state", None), path,
        )
        return True

    def _eiger_retry_is_pending(self, path, *, now=None):
        """Return whether *path* must still wait for its live retry deadline.

        A retry delay paces repeated opens of an unchanged provisional file.
        It must not hide a writer transition, however: if the same path's cheap
        identity stamp changes, clear the delay immediately so a finalized or
        newly populated container can flow on the next poll.
        """
        retries = getattr(self, "_eiger_retry_after", None)
        if not retries:
            return False
        deadline = float(retries.get(path, 0.0) or 0.0)
        if now is None:
            now = time.monotonic()
        if deadline <= now:
            retries.pop(path, None)
            return False

        zero_seen = getattr(self, "_eiger_zero_frame_seen", None) or {}
        observed = zero_seen.get(path)
        if observed is None:
            return True
        try:
            stat = os.stat(path)
            current_stamp = (
                int(getattr(stat, "st_dev", 0)),
                int(getattr(stat, "st_ino", 0)),
                int(stat.st_size),
                int(stat.st_mtime_ns),
            )
        except OSError:
            return True
        if current_stamp == observed[0]:
            return True

        retries.pop(path, None)
        zero_seen.pop(path, None)
        return False

    def _eiger_close_or_defer_zero_frame_master(self, frozen):
        """Close a zero-frame open and clear its identity when deferred."""
        path = self._eiger_master_path
        self._eiger_close_master()
        if not self._eiger_defer_zero_frame_master(frozen, path):
            return False
        self._eiger_master_path = None
        self._eiger_frame_idx = 0
        self._eiger_nframes = 0
        return True

    def _directory_source_armed(self, frozen):
        """Whether Run owns a valid directory intent without a seed file."""
        spec = frozen.thaw_source_spec()
        try:
            from xrd_tools.sources import DirectorySourceSpec

            return bool(
                frozen.source.family == "directory"
                and isinstance(spec, DirectorySourceSpec)
                and spec.root.is_dir()
            )
        except Exception:
            return False

    def _h19_live_directory_armed(self, frozen):
        """Compatibility alias for the retired eager-plan arm."""
        return bool(
            frozen.live_mode
            and (
                imageThread._directory_source_armed(self, frozen)
                or (
                    frozen.source.family == "directory"
                    and getattr(self, "source_run_plan", None) is not None
                    and getattr(self, "source_index_session", None) is not None
                )
            )
        )

    @staticmethod
    def _h19_candidate_is_current(candidate):
        """Point-check one queued Candidate without re-walking its directory.

        READY results in ``DirectoryIndexSession`` are sticky only for the
        Candidate's exact ``(path, adapter, size, mtime)`` identity.  A single
        stat plus the shared adapter-owner lookup therefore preserves the
        fail-closed consumption check without an O(directory) observation for
        every queued master.
        """
        try:
            path = Path(candidate.path)
            stat = path.stat()
            # Register built-ins before consulting the one precedence owner.
            import xrd_tools.sources.registry  # noqa: F401
            from xrd_tools.sources.adapters import candidate_owner

            owner = candidate_owner(path)
            owner_id = owner.id if owner is not None else None
            return (
                owner_id == candidate.adapter_id
                and (int(stat.st_size), int(stat.st_mtime_ns))
                == tuple(candidate.version_stamp)
            )
        except Exception:
            # Candidate ownership is a fail-closed consumption boundary.  A
            # registry/adapter failure is no reason to open a path whose exact
            # READY identity can no longer be proved; the next authoritative
            # directory observation may adopt it again.
            return False

    def _eiger_complete_append_count(self, frozen, path, candidate):
        """Known frame count iff Append can retire this whole raw container."""
        if (not self._append_skip_enabled(frozen)
                or frozen.run_options.get("series_average", False)):
            return 0

        scan_name = self._eiger_scan_name(path)
        completed = self._append_skip_snapshot(frozen, scan_name)
        source_snapshot = (
            getattr(self, "_append_source_snapshot_by_scan", {}) or {}
        ).get(str(scan_name))
        nframes = 0
        if (
            Path(path).suffix.lower() == ".nxs"
            and source_snapshot
            and bool(source_snapshot.get("self_contained", False))
            and os.path.normcase(os.path.abspath(str(
                source_snapshot.get("path", ""))))
            == os.path.normcase(os.path.abspath(str(path)))
        ):
            stamp = None
            try:
                stamp = (
                    int(source_snapshot["size"]),
                    int(source_snapshot["mtime_ns"]),
                )
                nframes = int(source_snapshot["frame_count"])
            except (KeyError, TypeError, ValueError):
                nframes = 0
            if stamp is None or tuple(stamp) != tuple(candidate.version_stamp):
                nframes = 0

        # Same-process compatibility memo for products written before the
        # persisted source snapshot existed.  It remains stamp-qualified and
        # applies only to finalized self-contained counts.
        if nframes <= 0:
            snapshot = getattr(
                self, "source_frame_count_snapshot", None) or {}
            hit = snapshot.get(str(path))
            if hit is not None:
                try:
                    stamp, nframes = hit
                    nframes = int(nframes)
                except (TypeError, ValueError):
                    nframes = 0
                if tuple(stamp) != tuple(candidate.version_stamp):
                    nframes = 0
        if nframes <= 0:
            return 0
        if all(frame in completed for frame in range(1, nframes + 1)):
            return nframes
        return 0

    def _eiger_skip_complete_append_master(self, frozen, path, candidate):
        """Bulk-account and retire one fully durable Append input.

        The count is a stamp-qualified value copied at Run start and the output
        cursor still performs the normal mode/config validation.  A miss,
        stale count, partial output, or mismatch takes the unchanged raw-open
        path.
        """
        count = self._eiger_complete_append_count(frozen, path, candidate)
        if count <= 0:
            return False
        self._append_skip_without_reading = (
            getattr(self, "_append_skip_without_reading", 0) + count)
        self._record_discovered_frame(count)
        self._record_skip_reason("already processed", count)
        self._eiger_retire_master(str(path))
        return True

    def _eiger_skip_open_complete_append_master(self, frozen):
        """Cold-start Append fast path after one raw-container open.

        A fresh GUI has no finalized frame-count memo yet.  Opening a
        self-contained finalized NeXus container supplies that count without
        reading pixels; when the processed output covers the complete range,
        retire and account the container as one unit instead of advancing all
        frame indices merely to rediscover that every frame is complete.
        """
        path = getattr(self, "_eiger_master_path", None)
        if (
            frozen.source.family == "directory"
            and path
            and self._append_skip_enabled(frozen)
            and frozen.run_options.get("series_average", False)
        ):
            scan_name = self._eiger_scan_name(path)
            if 1 in self._append_skip_snapshot(frozen, scan_name):
                message = (
                    f"Averaged output already exists for '{scan_name}' — the "
                    "whole series would be skipped and nothing written. "
                    "Switch write mode to Replace, or clear the target."
                )
                logger.error("run refused: %s", message)
                try:
                    self.showLabel.emit(message)
                except Exception:
                    logger.debug("showLabel emit failed", exc_info=True)
                self.command = "stop"
                self._eiger_close_master()
                self._eiger_master_path = None
                self._eiger_frame_idx = 0
                self._eiger_nframes = 0
                return True
        if (frozen.source.family != "directory"
                or not path
                or not self._eiger_open_count_can_bulk_compare(path)
                or not self._append_skip_enabled(frozen)
                or frozen.run_options.get("series_average", False)):
            return False
        count = int(getattr(self, "_eiger_nframes", 0) or 0)
        if count <= 0:
            return False
        scan_name = self._eiger_scan_name(path)
        completed = self._append_skip_snapshot(frozen, scan_name)
        first_missing = next(
            (frame for frame in range(1, count + 1)
             if frame not in completed),
            None,
        )
        if first_missing is not None:
            # Skip the already-complete prefix without advancing through it.
            # Later holes still flow through the ordinary per-frame cursor.
            prefix = int(first_missing) - 1
            if prefix > 0:
                self._append_skip_without_reading = (
                    getattr(self, "_append_skip_without_reading", 0)
                    + prefix
                )
                self._record_discovered_frame(prefix)
                self._record_skip_reason("already processed", prefix)
            self._eiger_frame_idx = prefix
            return False

        self._append_skip_without_reading = (
            getattr(self, "_append_skip_without_reading", 0) + count)
        self._record_discovered_frame(count)
        self._record_skip_reason("already processed", count)
        self._eiger_retire_master(str(path))
        self._emit_container_count(path, count, authoritative=True)
        self._eiger_close_master()
        self._eiger_master_path = None
        self._eiger_frame_idx = 0
        self._eiger_nframes = 0
        # The remaining frozen queue was not changed by opening/retiring this
        # master.  Let the next loop pop it directly; an empty queue with
        # unresolved work is reconciled by _eiger_pop_next_master itself.
        self._h19_queue_already_current = True
        return True

    def _eiger_refill_master_queue(self, frozen):
        """Queue matching HDF5 master / NeXus files not yet processed."""
        if (getattr(self, "source_run_plan", None) is not None
                and getattr(self, "source_index_session", None) is not None):
            queued = set(self._eiger_master_queue)
            retries = getattr(self, "_eiger_retry_after", None)
            if retries is None:
                retries = self._eiger_retry_after = {}
            zero_seen = getattr(self, "_eiger_zero_frame_seen", None)
            if zero_seen is None:
                zero_seen = self._eiger_zero_frame_seen = {}
            if getattr(self, "_h19_seed_pending", False):
                # The frozen plan is already the exact READY Source-card
                # snapshot accepted by Run.  Use it once so Run does not queue
                # behind another recursive GUI poll before its first frame.
                self._h19_seed_pending = False
                ready_paths = tuple(
                    candidate.path
                    for candidate in getattr(
                        self.source_run_plan, "candidates", ()))
                self._h19_ready_master_candidates = {
                    str(candidate.path): candidate
                    for candidate in getattr(
                        self.source_run_plan, "candidates", ())}
                self._h19_observed_master_paths = tuple(
                    str(path) for path in ready_paths)
            else:
                ready_paths = self._h19_ready_master_paths()
            observed = set(getattr(self, "_h19_observed_master_paths", ()))
            for stale_path in set(retries) - observed:
                retries.pop(stale_path, None)
                zero_seen.pop(stale_path, None)
            now = time.monotonic()
            for path in ready_paths:
                value = str(path)
                if value in self._eiger_done_masters or value in queued:
                    continue
                # R4A-5: parked provisional containers rejoin the queue only
                # through the one end-of-sweep recheck, never by re-discovery.
                if imageThread._eiger_provisional_hold(self, value):
                    continue
                if imageThread._eiger_retry_is_pending(
                    self, value, now=now
                ):
                    continue
                retries.pop(value, None)
                self._eiger_master_queue.append(value)
            return

        match = _name_filter((frozen.source.name_filter or ""))
        suffixes = frozen.source.suffixes
        if not suffixes:
            # Never skip discovery SILENTLY — no match suffix here reads as
            # "found nothing" with no clue why (bl17-2, 2026-07-13).
            logger.warning(
                'directory discovery skipped: the accepted source froze no '
                'File Type suffix (directory: %s)',
                frozen.source.filesystem_root)
            return
        root = Path(frozen.source.filesystem_root)
        recursive = bool(frozen.source.recursive)
        walk = getattr(self, "_directory_walk_iter", None)
        if walk is None:
            walk = self._directory_walk_items(
                root, recursive=recursive, suffixes=suffixes, match=match)
            self._directory_walk_iter = walk
        queued = set(self._eiger_master_queue)
        retries = getattr(self, "_eiger_retry_after", None)
        if retries is None:
            retries = self._eiger_retry_after = {}
        zero_seen = getattr(self, "_eiger_zero_frame_seen", None)
        if zero_seen is None:
            zero_seen = self._eiger_zero_frame_seen = {}
        now = time.monotonic()
        # Bound each refill by files/directories visited, so Run and Stop do
        # not wait for a complete accidental high-level recursive traversal.
        visit_budget = 64
        for _ in range(visit_budget):
            if getattr(self, "command", None) == "stop":
                return
            try:
                mf = next(walk)
            except StopIteration:
                self._directory_walk_iter = None
                break
            if mf is None:
                continue  # one directory boundary; still charges the budget
            mf_str = str(mf)
            if mf_str in self._eiger_done_masters:
                continue
            # R4A-5: see the authoritative branch above.
            if imageThread._eiger_provisional_hold(self, mf_str):
                continue
            if imageThread._eiger_retry_is_pending(
                self, mf_str, now=now
            ):
                continue
            retries.pop(mf_str, None)
            if mf_str in queued:
                continue
            # DIR-2: refill is NAME-ONLY.  The F5 finalized-at-close check
            # used to run here — one HDF5 open per not-yet-queued file, i.e.
            # hundreds of opens over a beamline share before the FIRST frame
            # could be read (the bl17-2 24.5 s first-dispatch stall).  It
            # now runs once per file at POP time (_eiger_pop_next_master),
            # which preserves the loss-free guarantee: the check still
            # happens before a container is consumed and retired.
            self._eiger_master_queue.append(mf_str)

    def _directory_walk_items(
        self, root, *, recursive, suffixes, match,
    ):
        """Yield a bounded token for every visited directory entry.

        ``os.walk`` materializes every name in a directory before its first
        yield.  A very wide accidental root could therefore defeat the outer
        64-token budget even when no source matched.  This scanner reads at
        most 64 entries into a naturally-sorted local batch, yielding either a
        matching path or ``None`` for every entry before reading the next
        batch.  Recursive child directories are retained as value-only paths
        and visited after the current directory is incrementally exhausted.
        """
        root = Path(root)
        if not root.is_dir():
            return
        pending_dirs = [root]
        while pending_dirs:
            current = pending_dirs.pop()
            child_dirs = []
            # O-1b R4A-4: complete this directory's NAME-only listing and sort it
            # GLOBALLY before any content work.  Sorting each 64-entry scandir
            # batch in isolation lost `scan_2`-before-`scan_10` in every
            # directory wider than one batch, so a wide directory processed
            # `scan_100` first.  Name listing is cheap and stays inside the lazy
            # contract (the plan's R4A-4 clause); consumption below is still one
            # token per entry, so the caller's discovery budget is unchanged.
            try:
                with os.scandir(current) as entries:
                    by_name = {entry.name: entry for entry in entries}
            except OSError:
                continue
            for name in natural_sort_ints(list(by_name)):
                entry = by_name[name]
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                    is_file = entry.is_file(follow_symlinks=False)
                except OSError:
                    yield None
                    continue
                if is_dir:
                    if recursive:
                        child_dirs.append(Path(entry.path))
                    yield None
                    continue
                if not is_file:
                    yield None
                    continue
                low = str(name).lower()
                suffix = next(
                    (value for value in suffixes if low.endswith(value)),
                    "",
                )
                if suffix and match(str(name)[:-len(suffix)]):
                    yield Path(entry.path)
                else:
                    yield None
            if recursive:
                pending_dirs.extend(reversed(child_dirs))

    def _h19_ready_master_paths(self):
        """Reconcile one worker poll against the Source-card run baseline."""
        plan = getattr(self, "source_run_plan", None)
        session = getattr(self, "source_index_session", None)
        if plan is None or session is None:
            return ()
        try:
            known_unprobed = getattr(self, "_h19_unprobed_count", None)
            drain_known_catalog = (
                int(known_unprobed or 0) > 0
                or (
                    known_unprobed is None
                    and int(getattr(self, "_h19_pending_count", 0) or 0) > 0
                )
            )
            observation = session.observe(refresh=not drain_known_catalog)
            self._h19_unprobed_count = int(observation.unprobed_count)
            self._h19_pending_count = (
                int(observation.pending_count)
                + int(bool(observation.stale_drops)))
            self._h19_observed_master_paths = tuple(
                str(candidate.path)
                for candidate in observation.discovered_snapshot.candidates
            )
            reconciliation = plan.reconcile(observation.discovered_snapshot)
            for candidate in reconciliation.changed:
                result = observation.result_for(candidate)
                if result is not None and result.state is ProbeState.READY:
                    plan = plan.adopt(candidate, result)
            reconciliation = plan.reconcile(observation.discovered_snapshot)
            self.source_run_plan = plan
        except Exception as exc:
            message = f"Source directory temporarily unavailable: {exc}"
            logger.warning("authoritative directory reconciliation failed: %s", exc)
            try:
                self.showLabel.emit(message)
            except Exception:
                logger.debug("showLabel emit failed for source observation",
                             exc_info=True)
            raise RuntimeError(message) from exc

        for path in reconciliation.removed:
            key = ("removed", str(path))
            if key not in self._source_plan_reported:
                self._source_plan_reported.add(key)
                message = f"Skipping source removed after Run started: {path}"
                logger.warning(message)
                try:
                    self.showLabel.emit(message)
                except Exception:
                    logger.debug("showLabel emit failed", exc_info=True)
        for candidate in reconciliation.owner_flipped:
            key = ("owner", str(candidate.path))
            if key not in self._source_plan_reported:
                self._source_plan_reported.add(key)
                message = (
                    "Skipping source whose format owner changed after Run "
                    f"started: {candidate.path}"
                )
                logger.warning(message)
                try:
                    self.showLabel.emit(message)
                except Exception:
                    logger.debug("showLabel emit failed", exc_info=True)

        ready = {
            item.candidate.path
            for item in observation.candidates
            if item.result.state is ProbeState.READY
        }
        ordered = (*reconciliation.run_order, *reconciliation.appended)
        ready_candidates = tuple(
            candidate for candidate in ordered if candidate.path in ready)
        self._h19_ready_master_candidates = {
            str(candidate.path): candidate for candidate in ready_candidates}
        return tuple(candidate.path for candidate in ready_candidates)

    def _eiger_provisional_open_outcome(self):
        """Whether the LAST open produced a provisional (still-writing) file."""
        return bool(
            getattr(self, "_eiger_single_file_provisional", False)
            or getattr(self, "_eiger_open_state", None) == "not ready")

    def _eiger_provisional_hold(self, path):
        """Whether *path* is held out of THIS run's discovery by R4A-5.

        Either parked awaiting the one end-of-sweep recheck, or already
        rechecked and still provisional.  Neither state is retirement: the
        container was never consumed, and both sets are cleared at Run start.
        """
        return (
            path in (getattr(self, "_eiger_provisional_masters", None) or ())
            or path in (getattr(self, "_eiger_provisional_exhausted", None) or ())
        )

    def _eiger_park_provisional_master(self, frozen, path):
        """R4A-5: hold a provisional container instead of retiring it.

        The deferral itself was already correct -- ``_eiger_open_master``
        defers an ``IN_PROGRESS``/unfinalized directory container, which is what
        F5/DIR-2 fixed for the live watch.  What was missing is that BATCH (and
        any other non-live directory run) has no watch loop to re-poll it, so
        the deferred container fell through to the exhaustion branch, was
        RETIRED into ``_eiger_done_masters`` at first sight, and every frame it
        flushed later in the SAME run was lost.

        Returns ``True`` when the caller must neither retire nor count the
        container.  It is parked for exactly one end-of-sweep recheck; if it is
        still provisional then, it is held out of this run's discovery so the
        sweep cannot spin on an unchanged file -- but it is never recorded as
        consumed, so the next Run picks it up.
        """
        if (not path
                or frozen.live_mode
                or frozen.source.family != "directory"
                or not imageThread._eiger_provisional_open_outcome(self)):
            return False
        value = str(path)
        if getattr(self, "_eiger_provisional_recheck_done", False):
            exhausted = getattr(self, "_eiger_provisional_exhausted", None)
            if exhausted is None:
                exhausted = self._eiger_provisional_exhausted = set()
            if value not in exhausted:
                exhausted.add(value)
                logger.info(
                    "Container still being written after its end-of-sweep "
                    "recheck; leaving it for the next Run: %s", value)
            return True
        parked = getattr(self, "_eiger_provisional_masters", None)
        if parked is None:
            parked = self._eiger_provisional_masters = set()
        parked.add(value)
        return True

    def _eiger_recheck_provisional_masters(self, frozen):
        """R4A-5: ONE bounded end-of-sweep recheck; ``True`` if anything requeued.

        Requeued paths take the normal pop -> :meth:`_eiger_open_master` ->
        adapter-binding route again, so a container that finalized mid-run is
        read through a freshly built NXWriter cursor/provider rather than the
        stale binding that saw it provisional.  A cheap frame-count re-read
        would look cheaper and produce unreadable frames.

        At most once per Run, so a directory of permanently provisional
        containers finishes truthfully instead of spinning.
        """
        if frozen.live_mode or frozen.source.family != "directory":
            return False
        if getattr(self, "_eiger_provisional_recheck_done", False):
            return False
        self._eiger_provisional_recheck_done = True
        parked = sorted(getattr(self, "_eiger_provisional_masters", None) or ())
        self._eiger_provisional_masters = set()
        requeued = False
        for value in parked:
            if value in self._eiger_done_masters:
                continue
            if not os.path.exists(value):
                continue
            self._eiger_master_queue.append(value)
            requeued = True
        if requeued:
            logger.info(
                "Rechecking %d provisional container(s) before end of run",
                len(parked))
        return requeued

    def _eiger_pop_master_with_recheck(self, frozen):
        """The pop the directory reader uses: drained sweep -> one recheck."""
        next_master = self._eiger_pop_next_master(frozen)
        if next_master is not None:
            return next_master
        if not imageThread._eiger_recheck_provisional_masters(self, frozen):
            return None
        return self._eiger_pop_next_master(frozen)

    def _eiger_pop_next_master(self, frozen):
        """Pop the next master, deferring unfinalized NXWriter containers.

        F5 moved here from refill time (DIR-2): one finalized-at-close
        check per file at consumption, instead of one HDF5 open per file
        per refill poll.  A deferred (in-progress or unreadable/torn)
        container is dropped from the queue — it is neither consumed nor
        retired, so the next refill poll re-discovers it and it flows the
        moment it finalizes.  Applies to every container kind (post
        DIR-2b, *_master.h5 NXWriter files are readable too); plain
        non-NXWriter files are never deferred
        (``is_unfinalized_nxwriter`` answers False for them).
        """
        if (getattr(self, "source_run_plan", None) is not None
                and getattr(self, "source_index_session", None) is not None):
            stale_followups = 0
            while True:
                retries = getattr(self, "_eiger_retry_after", {})
                now = time.monotonic()
                candidates = getattr(
                    self, "_h19_ready_master_candidates", None) or {}
                stale_this_pass = False
                for _ in range(len(self._eiger_master_queue)):
                    path = self._eiger_master_queue.popleft()
                    if imageThread._eiger_retry_is_pending(
                        self, path, now=now
                    ):
                        self._eiger_master_queue.append(path)
                        continue
                    candidate = candidates.get(path)
                    if (candidate is None
                            or not imageThread._h19_candidate_is_current(
                                candidate)):
                        stale_this_pass = True
                        # The queued identity drifted after the frozen
                        # observation. Force one authoritative follow-up after
                        # the remaining queue drains so its replacement can be
                        # reprobed without a per-master directory walk.
                        self._h19_pending_count = max(
                            1, int(getattr(
                                self, "_h19_pending_count", 0) or 0))
                        key = ("stale-at-pop", str(path))
                        if key not in self._source_plan_reported:
                            self._source_plan_reported.add(key)
                            message = (
                                "Skipping source changed after readiness; it "
                                f"will be re-observed: {path}")
                            logger.warning(message)
                            try:
                                self.showLabel.emit(message)
                            except Exception:
                                logger.debug(
                                    "showLabel emit failed", exc_info=True)
                        continue
                    retries.pop(path, None)
                    skip_complete = getattr(
                        self, "_eiger_skip_complete_append_master", None)
                    if (skip_complete is not None
                            and skip_complete(frozen, path, candidate)):
                        continue
                    self._eiger_master_candidate = candidate
                    return path

                # The frozen READY snapshot can coexist with candidates still
                # awaiting the session's bounded probe budget. Once the seed
                # drains, observe only while each poll makes progress. A truly
                # in-progress writer stabilizes and returns to the Live cadence;
                # a converged directory performs zero extra walks.
                if (int(getattr(self, "_h19_pending_count", 0) or 0) <= 0
                        or getattr(self, "command", None) == "stop"):
                    return None
                # A directory session may briefly hand back the same sticky
                # READY identity even though the point stat already disproved
                # it.  Give that stale identity one authoritative follow-up,
                # then yield instead of requeueing it forever at 100% CPU.
                if stale_this_pass and stale_followups >= 1:
                    return None
                before = int(self._h19_pending_count)
                self._eiger_refill_master_queue(frozen)
                if stale_this_pass:
                    stale_followups += 1
                if self._eiger_master_queue:
                    continue
                after = int(
                    getattr(self, "_h19_pending_count", 0) or 0)
                if after <= 0 or after >= before:
                    return None

        while True:
            for _ in range(len(self._eiger_master_queue)):
                cand = self._eiger_master_queue.popleft()
                if imageThread._eiger_retry_is_pending(self, cand):
                    self._eiger_master_queue.append(cand)
                    continue
                try:
                    from xrd_tools.sources.adapters import candidate_owner
                    from xrd_tools.sources.discover import Candidate
                    import xrd_tools.sources.registry  # noqa: F401

                    path = Path(cand)
                    stat = path.stat()
                    owner = candidate_owner(path)
                    candidate = Candidate(
                        path=path,
                        adapter_id=owner.id,
                        size=int(stat.st_size),
                        mtime_ns=int(stat.st_mtime_ns),
                    )
                except Exception:
                    # A vanished/unstatable path is rediscovered on a later live
                    # pass. No content open is attempted for an unowned identity.
                    continue
                skip_complete = getattr(
                    self, "_eiger_skip_complete_append_master", None)
                if (
                    owner is not None
                    and skip_complete is not None
                    and skip_complete(frozen, cand, candidate)
                ):
                    continue
                self._eiger_master_candidate = (
                    candidate if owner is not None else None)
                return cand
            if (
                getattr(self, "command", None) == "stop"
                or getattr(self, "_directory_walk_iter", None) is None
            ):
                return None
            if frozen.live_mode:
                # One live reader request owns at most one bounded discovery
                # batch.  Returning an idle sentinel here keeps Stop/UI cadence
                # responsive on an accidentally broad recursive root; the
                # persistent iterator resumes on the next watch request.
                return None
            self._eiger_refill_master_queue(frozen)

    def _get_next_eiger_frame(self, frozen):
        """Return the next frame from Eiger / NeXus HDF5 file(s).

        Wraps :meth:`_get_next_eiger_frame_sync` with a background
        prefetcher so the next frame's disk read overlaps with the
        current frame's integration.  The synchronous reader is still
        available for the worker itself and for paths that don't want
        prefetching.

        Uses a short polling timeout so the main thread can break out
        of a blocked ``.get()`` if the user hits Stop and the prefetch
        worker exited without pushing a sentinel.
        """
        if self._prefetch_queue is None:
            self._start_prefetcher(frozen)
        # Poll the queue so we can cooperate with user Stop even if the
        # prefetcher died or hasn't pushed an end-of-stream sentinel yet.
        while True:
            if self.command == 'stop':
                return None, None, 1, None, {}
            try:
                item = self._prefetch_queue.get(timeout=0.25)
            except queue.Empty:
                # Worker drained and gone.  In a LIVE run this may just be an
                # empty/partial master so far (the split second before the
                # detector writes nimages, or between scans): reset the
                # prefetcher so the NEXT poll starts a fresh one and re-checks
                # (re-opening a growing master / picking up new masters), and
                # hand back an end-of-stream sentinel now so the watch loop
                # backs off between polls -- never a tight restart spin.  In
                # batch, a drained worker is the genuine end of the finite scan.
                if (self._prefetch_thread is None
                        or not self._prefetch_thread.is_alive()):
                    if (not frozen.batch_mode) and self.command != 'stop':
                        self._prefetch_queue = None
                        self._prefetch_thread = None
                    return None, None, 1, None, {}
                continue
            # O8: surface a worker-failure sentinel as a user-visible
            # status before propagating end-of-stream.  A clean end
            # has ``_prefetch_error is None``; a worker crash has the
            # error string set in the worker's except branch.
            if item[3] is None and self._prefetch_error:
                self.showLabel.emit(
                    f'Eiger read failed: {self._prefetch_error}'
                )
                # Clear so the next start (if any) doesn't re-emit.
                self._prefetch_error = None
            return item

    def _start_prefetcher(self, frozen):
        """Spin up the background prefetch thread (idempotent)."""
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            return
        prefetch_queue = queue.Queue(maxsize=_PREFETCH_QUEUE_SIZE)
        stop_evt = threading.Event()
        self._prefetch_queue = prefetch_queue
        self._prefetch_stop_evt = stop_evt
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            # The prefetch thread executes THIS run, so it receives the exact
            # accepted configuration at spawn rather than re-reading a carrier
            # from another thread (review §43.4).
            args=(frozen, prefetch_queue, stop_evt),
            name='eiger-prefetch',
            daemon=True,
        )
        self._prefetch_thread.start()

    def _push_frame_to_queue(self, item, prefetch_queue=None, stop_evt=None):
        """Put *item* onto the prefetch queue, cooperating with stop.

        Returns True if the item was queued, False if the worker was
        cancelled while blocking on a full queue.
        """
        # Snapshot-compatible defaults keep direct unit-test calls working.
        # A real worker receives generation-owned objects from
        # _start_prefetcher: cleanup may replace the public attributes after a
        # bounded join, but it must never invalidate a still-unwinding worker.
        prefetch_queue = (self._prefetch_queue
                          if prefetch_queue is None else prefetch_queue)
        stop_evt = (self._prefetch_stop_evt if stop_evt is None else stop_evt)
        if prefetch_queue is None or stop_evt is None:
            return False
        while not stop_evt.is_set() and self.command != 'stop':
            try:
                prefetch_queue.put(item, timeout=0.25)
                return True
            except queue.Full:
                continue
        return False

    def _prefetch_worker(self, frozen, prefetch_queue=None, stop_evt=None):
        """Read frames sequentially and push them onto the bounded queue.

        Uses bulk HDF5 slices (`dset[i:i+N]`) where possible so each chunk
        is decompressed once per N frames instead of N times.  Falls back
        to single-frame reads via :meth:`_get_next_eiger_frame_sync` for
        the initial master-file setup, fabio-backed sources, and
        live-growing files (where bulk reads past the known end are
        unsafe).

        The worker exits when:
          - a frame with ``img_data is None`` is produced (end of stream);
          - the stop event is set (cooperative cancellation);
          - ``self.command == 'stop'`` (user pressed Stop).
        Exceptions are logged and a sentinel tuple is pushed so the
        consumer can terminate cleanly.
        """
        prefetch_queue = (self._prefetch_queue
                          if prefetch_queue is None else prefetch_queue)
        stop_evt = (self._prefetch_stop_evt if stop_evt is None else stop_evt)
        if prefetch_queue is None or stop_evt is None:
            return
        sentinel_pushed = False
        try:
            while not stop_evt.is_set() and self.command != 'stop':
                # Always fetch the first frame of (the next) master through
                # the sync reader — it handles master-queue advancement,
                # handle opening, and frame-count refresh.
                item = self._get_next_eiger_frame_sync(frozen)
                if not self._push_frame_to_queue(
                        item, prefetch_queue=prefetch_queue, stop_evt=stop_evt):
                    return
                if item[3] is None:
                    # End of stream; worker is done — sentinel already queued.
                    sentinel_pushed = True
                    return

                # Fast path: bulk-read the remainder of this master through
                # the h5py dataset.  Skipped when fabio is primary (fabio
                # handles its own per-frame decoding) or when we're near
                # the tail of a live-growing file.
                while (not stop_evt.is_set()
                       and self.command != 'stop'
                       and self._eiger_fabio_handle is None
                       and self._eiger_cursor is not None
                       and self._eiger_frame_idx < self._eiger_nframes):

                    start = self._eiger_frame_idx
                    # R2: the layout-aware ReadPlan sizes the bulk window
                    # (byte-bounded, native-chunk-aligned) instead of the fixed
                    # 16-frame block — bounding the prefetch's retained memory.
                    plan = self._eiger_read_plan
                    block_frames = plan.block_frames if plan is not None else 1
                    end = min(start + max(1, block_frames), self._eiger_nframes)
                    scan_name = self._eiger_scan_name(self._eiger_master_path)

                    # Advance the shared frame cursor *before* dispatching so
                    # that any concurrent sync read (e.g. in fallback) does
                    # not re-serve these frames.
                    self._eiger_frame_idx = end
                    to_read = []
                    for frame_idx in range(start, end):
                        self._record_discovered_frame()
                        if self._should_skip_before_read(frozen, scan_name, frame_idx + 1):
                            continue
                        to_read.append(frame_idx)

                    if not to_read:
                        continue

                    groups = []
                    for frame_idx in to_read:
                        if not groups or frame_idx != groups[-1][1]:
                            groups.append([frame_idx, frame_idx + 1])
                        else:
                            groups[-1][1] = frame_idx + 1

                    bulk_failed = False
                    for group_start, group_end in groups:
                        _t_blk = time.time()
                        try:
                            source_before = self._eiger_source_facts(group_start)
                            # Native-dtype owner block via the sustained cursor.
                            block = np.asarray(
                                self._eiger_cursor.read_block(
                                    group_start, group_end).array)
                            source_after = self._eiger_source_facts(group_start)
                            if source_after != source_before:
                                raise RuntimeError("source changed during bulk read")
                        except Exception as e:
                            logger.warning(
                                'Bulk read failed (start=%d end=%d): %s; '
                                'falling back to per-frame read',
                                group_start, group_end, e,
                            )
                            self._eiger_frame_idx = group_start
                            self._eiger_refresh_master_handle()
                            bulk_failed = True
                            break  # outer worker loop resumes with sync reader
                        _t_blk = time.time() - _t_blk
                        _perf = getattr(self, '_perf', None)
                        if _perf is not None:
                            _perf['prefetch_io'] += _t_blk
                            _perf['prefetch_frames'] += (group_end - group_start)
                        # If a bulk read takes >50ms it can fight with the
                        # consumer for memory bandwidth — log so we can
                        # correlate against [TIMING] spikes on the consumer.
                        if _t_blk > 0.05:
                            logger.info(
                                '[PREFETCH] block frames %d-%d read in %.3fs',
                                group_start, group_end - 1, _t_blk,
                            )

                        for offset, frame_idx in enumerate(
                                range(group_start, group_end)):
                            if (stop_evt.is_set()
                                    or self.command == 'stop'):
                                return
                            item = (
                                self._eiger_master_path,
                                scan_name,
                                frame_idx + 1,   # 1-based img_number
                                # R2-R2: COPY the frame out of the owner block so
                                # the block is released as soon as its frames are
                                # dispatched.  Queuing a VIEW would pin the whole
                                # native block while the NEXT block is read — two
                                # live owner blocks at once, exceeding the
                                # single-block byte budget.  Only ONE owner block
                                # is ever live; the queued per-frame copies are a
                                # separate, bounded (queue-depth x frame) cost.
                                np.array(block[offset]),
                                # Per-frame scan_info: the sidecar/scalar
                                # metadata overlaid with the Bluesky per-frame
                                # motor + counter row (so each frame carries its
                                # own hy/i0.. values, not a shared master dict).
                                _SourceMetadata(self._frame_scan_info(
                                    frozen, self._eiger_master_path, frame_idx),
                                    dict(source_after, source_frame_idx=frame_idx)),
                            )
                            if not self._push_frame_to_queue(
                                    item, prefetch_queue=prefetch_queue,
                                    stop_evt=stop_evt):
                                return
                        # R2-R2 (final): EXPLICITLY release this group's owner
                        # block (and the last dispatched copy) BEFORE the next
                        # ``read_block`` — Python evaluates that call while a
                        # still-bound ``block`` owns the PREVIOUS array, so
                        # without this two owner blocks are live at every read
                        # boundary, doubling the single-block byte budget.  The
                        # same release keeps the final block from surviving the
                        # blocking sync-read wait between masters and the
                        # read-failure fallback below; the early Stop returns
                        # free the frame's locals outright.
                        block = None
                        item = None
                    if bulk_failed:
                        break
        except Exception as e:
            logger.exception('Eiger prefetch worker failed: %s', e)
            # O8: stamp the failure so the consumer of the sentinel
            # can tell "scan ended cleanly" from "worker crashed".
            # Surfaced via showLabel in _get_next_eiger_frame; without
            # this the user sees a silent end-of-scan and assumes
            # acquisition completed normally.
            self._prefetch_error = str(e)
        finally:
            # Guarantee the consumer unblocks no matter how we exit
            # (stop event, command=='stop', exception, or normal return).
            if not sentinel_pushed:
                try:
                    prefetch_queue.put(
                        (None, None, 1, None, {}), timeout=1.0,
                    )
                except queue.Full:
                    # Drain one slot so the sentinel can fit — the main
                    # thread has already seen stop, so dropping a queued
                    # frame is fine.
                    try:
                        prefetch_queue.get_nowait()
                        prefetch_queue.put_nowait(
                            (None, None, 1, None, {}),
                        )
                    except (queue.Empty, queue.Full):
                        pass

    def _prefetch_stop_prior(self):
        """Cancel a prefetch generation; return True once it has stopped.

        A slow HDF5/network read may outlive the bounded join.  Callers must not
        clear the generation's shared detector state or start its successor
        until this returns True.
        """
        stop_evt = self._prefetch_stop_evt
        prefetch_thread = self._prefetch_thread
        prefetch_queue = self._prefetch_queue
        if stop_evt is not None:
            stop_evt.set()
        if prefetch_thread is not None and prefetch_thread.is_alive():
            # Drain to unblock the worker on a full queue
            if prefetch_queue is not None:
                try:
                    while True:
                        prefetch_queue.get_nowait()
                except queue.Empty:
                    pass
            prefetch_thread.join(timeout=2.0)
        return prefetch_thread is None or not prefetch_thread.is_alive()

    def _read_eiger_frame_tolerant(self, frozen, frame_idx):
        """Read one Eiger frame, tolerating a data file that lags the master.

        An Eiger master declares ``nimages`` up front (written at scan start),
        but the per-frame data files stream in as the detector writes them;
        reading a frame whose data has not landed yet raises.  Unhandled, that
        set ``img_data=None``, which the prefetch worker took as END-OF-STREAM
        and exited -> live Eiger stalled the moment a data file lagged the
        master.  Here we retry (blocking only the background prefetch thread,
        never the GUI), refreshing the handle so newly written data files
        become visible, until the frame reads or a bounded deadline passes; then
        we return ``None`` so the caller skips just that one frame (not the
        whole stream).  Batch reads once (files already complete); Stop returns
        at once."""
        deadline = getattr(self, "FRAME_READ_DEADLINE", _FRAME_READ_DEADLINE)
        stop_evt = getattr(self, '_prefetch_stop_evt', None)
        waited = 0.0
        delay = 0.1
        while True:
            if (getattr(self, 'command', None) == 'stop'
                    or (stop_evt is not None and stop_evt.is_set())):
                return None
            try:
                if self._eiger_fabio_handle is not None:
                    _raw = (self._eiger_fabio_handle.data if frame_idx == 0
                            else self._eiger_fabio_handle.get_frame(frame_idx).data)
                    return np.asarray(_raw)
                if self._eiger_cursor is not None:
                    # F6: the cursor maps a lone 2-D dataset to frame 0 (never
                    # indexes a row out of a single-exposure frame).
                    idx = 0 if self._eiger_cursor.is_2d else frame_idx
                    return np.asarray(self._eiger_cursor.read_frame(idx))
                raise OSError("Eiger frame read requires its retained source owner")
            except Exception as e:
                if frozen.batch_mode or waited >= deadline:
                    logger.error('Error reading frame %d from %s: %s',
                                 frame_idx, self._eiger_master_path, e)
                    return None
                time.sleep(delay)
                waited += delay
                delay = min(delay * 1.5, 0.5)
                # Pick up data files written since the handle was opened.
                self._eiger_refresh_master_handle()

    def _eiger_refresh_master_handle(self, *, raise_errors=False):
        """Reopen/refresh the open Eiger handle so a subsequent read sees
        per-frame data files written since it was opened (the master is written
        up front; the data streams in after).  Best-effort -- a refresh failure
        must never crash the read."""
        try:
            if self._eiger_fabio_handle is not None:
                self._eiger_fabio_handle.close()
                self._eiger_fabio_handle = None
                candidate = _source_observation(self._eiger_master_path)
                replacement = fabio.open(self._eiger_master_path)
                if _source_observation(self._eiger_master_path) != candidate:
                    replacement.close()
                    raise OSError("Eiger master changed while fabio reopened it")
                self._eiger_fabio_handle = replacement
                self._eiger_master_candidate = candidate
                self._eiger_nframes = self._eiger_fabio_handle.nframes
                return True
            elif self._eiger_cursor is not None:
                # Growth-refresh analogue of the fabio reopen: reopen the cursor
                # so a subsequent read sees frames written since it was opened
                # (reopen-to-refresh, NOT a new SWMR policy).
                return self._eiger_reopen_cursor()
            elif getattr(self, '_eiger_master_path', None) is not None:
                # No handle at all — a single-file source whose detector tree
                # landed after a provisional (not-ready) open.  Build the real
                # cursor binding now so the tolerant read proceeds on the h5py
                # path instead of burning the deadline on the fabio fallback
                # (NXS-SF-2).
                return self._eiger_reopen_cursor()
        except Exception as e:
            logger.debug("Failed to refresh Eiger handle for %s: %s",
                         getattr(self, "_eiger_master_path", None), e)
            if raise_errors:
                raise
        return False

    def _get_next_eiger_frame_sync(self, frozen):
        """Return the next frame from Eiger HDF5 master file(s), one at a time.

        Keeps the h5py file handle open across frames to avoid the
        expensive open/close cycle per frame.  Tracks position with
        (_eiger_master_path, _eiger_frame_idx, _eiger_nframes).
        """
        while True:
            if getattr(self, "command", None) == "stop":
                return None, None, 1, None, {}
            # A finalized-and-consumed single file is a FIXED POINT: idle
            # watch polls return end-of-stream with zero source opens
            # (NXS-SF-1; handoff §8 "no spin-open while idle").
            if (getattr(self, '_eiger_single_file_done', False)
                    and frozen.source.family != "directory"):
                return None, None, 1, None, {}

            # ── Initialise on the very first call ────────────────────────────
            if self._eiger_master_path is None:
                if frozen.source.family == "directory":
                    queue_current = bool(getattr(
                        self, "_h19_queue_already_current", False))
                    self._h19_queue_already_current = False
                    if not queue_current:
                        self._eiger_refill_master_queue(frozen)
                    next_master = imageThread._eiger_pop_master_with_recheck(
                        self, frozen)
                    if next_master is None:
                        return None, None, 1, None, {}
                    self._eiger_master_path = next_master
                else:
                    self._eiger_master_path = self.img_file
                self._eiger_frame_idx = 0
                self._eiger_open_master(frozen, self._eiger_master_path)
                if self._eiger_skip_open_complete_append_master(frozen):
                    continue
                if self._eiger_nframes == 0:
                    parked = imageThread._eiger_park_provisional_master(
                        self, frozen, self._eiger_master_path)
                    self._eiger_close_or_defer_zero_frame_master(frozen)
                    if frozen.source.family == "directory":
                        # An imageless container (a diode/alignment scan in a
                        # mixed beamline directory) must not END the stream —
                        # loop back so the exhaustion branch retires it and
                        # advances to the next master (bl17-2 2026-07-12: one
                        # alignment file sorting FIRST killed the whole batch
                        # run with 'Total Files Processed: 0').
                        if parked:
                            # R4A-5: a PROVISIONAL container is different.  Drop
                            # its identity here so the exhaustion branch below
                            # cannot retire it (losing the frames it flushes
                            # later in this same run) nor re-count it with no
                            # cursor bound; it rejoins through the one
                            # end-of-sweep recheck.
                            self._eiger_master_path = None
                            self._eiger_frame_idx = 0
                            self._eiger_nframes = 0
                        continue
                    return None, None, 1, None, {}

            # ── Current master exhausted?  Try to advance ────────────────────
            if (self._eiger_frame_idx >= self._eiger_nframes
                    and frozen.live_mode and not frozen.batch_mode
                    and frozen.source.family == "directory"):
                path, candidate = self._eiger_master_path, self._eiger_master_candidate
                if (not imageThread._eiger_open_count_is_authoritative(self, path) and not self._eiger_refresh_master_handle()):
                    self._eiger_close_master()
                    retry_s = max(0.01, float(getattr(
                        self, "CONTAINER_READY_RETRY", _CONTAINER_READY_RETRY)))
                    self._eiger_retry_after[path] = time.monotonic() + retry_s
                    if not imageThread._h19_candidate_is_current(candidate): self._eiger_master_path = None; self._eiger_frame_idx = self._eiger_nframes = 0
                    return None, None, 1, None, {}

            if self._eiger_frame_idx >= self._eiger_nframes:
                if frozen.source.family == "directory":
                    if imageThread._eiger_park_provisional_master(
                            self, frozen, self._eiger_master_path):
                        # R4A-5: never retired, never counted.  Retiring it here
                        # at first sight is what lost a container that changed
                        # from IN_PROGRESS to complete during the same run, and
                        # a zero count for a still-writing file is not a fact.
                        self._eiger_close_master()
                    else:
                        self._eiger_retire_master(self._eiger_master_path)
                        self._emit_container_count(self._eiger_master_path,
                                                   self._eiger_nframes,
                                                   authoritative=True)
                        self._eiger_close_master()
                    if not self._eiger_master_queue:
                        self._eiger_refill_master_queue(frozen)
                    next_master = imageThread._eiger_pop_master_with_recheck(
                        self, frozen)
                    if next_master is None:
                        self._eiger_master_path = None
                        self._eiger_frame_idx = 0
                        self._eiger_nframes = 0
                        return None, None, 1, None, {}
                    self._eiger_master_path = next_master
                    self._eiger_frame_idx = 0
                    self._eiger_open_master(frozen, self._eiger_master_path)
                    if self._eiger_skip_open_complete_append_master(frozen):
                        continue
                    if self._eiger_nframes == 0:
                        # Imageless container mid-queue: retire-and-advance via
                        # the exhaustion branch, never end the stream (see the
                        # init-branch note).  A PROVISIONAL one is parked there
                        # instead, so it is not retired and not re-read.
                        if imageThread._eiger_park_provisional_master(
                                self, frozen, self._eiger_master_path):
                            self._eiger_close_master()
                            self._eiger_master_path = None
                            self._eiger_frame_idx = 0
                            self._eiger_nframes = 0
                            continue
                        self._eiger_close_or_defer_zero_frame_master(frozen)
                        continue
                else:
                    # NXS-SF-1/SF-2: a live single-file source may still be
                    # growing (or nascent).  One bounded transactional reopen
                    # observes growth; 'grown' continues from the PRIOR index
                    # (never restarts at zero), 'wait' hands the watch loop a
                    # sentinel to back off on, 'end' is the finalized fixed
                    # point (or batch/stop) and closes the source.
                    outcome = self._eiger_single_file_growth_outcome(frozen)
                    if outcome == 'grown':
                        continue
                    if outcome == 'end':
                        self._eiger_single_file_done = True
                    # Close on 'end' AND 'wait': an idle provisional wait must
                    # never pin a file another program is actively writing
                    # (Windows writers fail loudly on pinned files — the DIR-3
                    # class); the next watch poll reopens transactionally.
                    self._eiger_close_master()
                    return None, None, 1, None, {}

            # ── Read one frame ────────────────────────────────────────────────
            frame_idx = self._eiger_frame_idx
            self._eiger_frame_idx += 1
            scan_name = self._eiger_scan_name(self._eiger_master_path)
            img_number = frame_idx + 1  # 1-based
            self._record_discovered_frame()
            if self._should_skip_before_read(frozen, scan_name, img_number):
                continue

            try:
                source_before = self._eiger_source_facts(frame_idx)
            except (OSError, RuntimeError):
                self._eiger_frame_idx = frame_idx
                if not self._eiger_refresh_master_handle():
                    if frozen.live_mode and not frozen.batch_mode:
                        return None, None, 1, None, {}
                    raise
                continue
            img_data = self._read_eiger_frame_tolerant(frozen, frame_idx)
            if img_data is None:
                stop_evt = getattr(self, '_prefetch_stop_evt', None)
                if (getattr(self, 'command', None) == 'stop'
                        or (stop_evt is not None and stop_evt.is_set())):
                    return None, None, 1, None, {}  # genuine stop
                # A frame the master DECLARES (nimages) but whose data file
                # never became readable within the deadline (an aborted/dropped
                # frame): skip just this one and read on.  Do NOT emit an
                # end-of-stream sentinel for a mid-scan frame -- that made the
                # live worker exit on the first data file that lagged the master
                # and the watch stall forever.
                self._record_skip_reason("unreadable or empty image data")
                continue

            try:
                source_after = self._eiger_source_facts(frame_idx)
            except (OSError, RuntimeError):
                source_after = None
            if source_after != source_before:
                self._eiger_frame_idx = frame_idx
                self._eiger_refresh_master_handle()
                continue

            meta = self._frame_scan_info(frozen, self._eiger_master_path, frame_idx)

            return (self._eiger_master_path, scan_name, img_number, img_data,
                    _SourceMetadata(meta, source_after))

    # ── Image iteration ──────────────────────────────────────────────────

    def _read_frame_tolerant(self, fname):
        """Read one detector frame WITHOUT letting a partial-file read escape
        ``run()`` (an unhandled fabio error on a still-being-written file used
        to KILL the live QThread -- live processing stopped dead, looking like a
        timeout).

        Short bounded retry (``_FRAME_READ_RETRY_BUDGET`` s) absorbs a file that
        finishes flushing within ~a second.  If it is still unreadable, returns
        ``None`` and lets the caller decide: ``get_next_image`` re-polls it on a
        later LIVE sweep (never committing/dropping it until it reads or the
        cross-sweep deadline passes) or skips it in batch.  A good file reads on
        the first try with no added latency; a Stop between retries returns at
        once."""
        budget = getattr(self, "FRAME_READ_RETRY_BUDGET", _FRAME_READ_RETRY_BUDGET)
        waited = 0.0
        delay = 0.1
        while True:
            if self.command == 'stop':
                return None
            try:
                return np.asarray(read_image(fname), dtype=float)
            except Exception:
                if waited >= budget:
                    return None
                time.sleep(delay)
                waited += delay
                delay = min(delay * 1.5, 0.5)

    def _commit_frame(self, fname, *, count_discovery=True):
        """Mark a frame consumed for this run: record it processed, drop it from
        the pending queue, count it discovered.  Deferred until a DECISIVE read
        outcome (read succeeded, or gave up past the deadline) so a still-writing
        frame we re-poll isn't prematurely excluded by the next re-glob."""
        self.processed.append(fname)
        if self.img_fnames and self.img_fnames[0] == fname:
            self.img_fnames.popleft()
        self._frame_read_clock_map().pop(fname, None)
        if count_discovery:
            self._record_discovered_frame()

    def _frame_read_clock_map(self):
        """The fname -> first-unreadable-time map, lazily created (some test
        doubles build the thread via __new__ and skip __init__)."""
        clocks = getattr(self, "_frame_read_clocks", None)
        if clocks is None:
            clocks = self._frame_read_clocks = {}
        return clocks

    def _frame_read_deadline_reached(self, fname):
        """True once a present-but-unreadable LIVE frame has been unreadable for
        ``_FRAME_READ_DEADLINE`` s since first seen, so the caller stops
        re-polling and skips it -- a genuinely corrupt file can't wedge the live
        watch.  The first unreadable sighting starts the clock (returns False)."""
        clocks = self._frame_read_clock_map()
        first = clocks.get(fname)
        now = time.monotonic()
        if first is None:
            clocks[fname] = now
            return False
        return (now - first) >= getattr(
            self, "FRAME_READ_DEADLINE", _FRAME_READ_DEADLINE)

    def _frozen_source_is_container(self, frozen):
        """Container-versus-series, decided by the ACCEPTED frozen source (R4B-14).

        The old gate keyed on ``self.img_ext`` — a mutable panel mirror, i.e. a
        SECOND format authority that a mid-run panel edit could flip.  The
        decision now comes from the frozen source selection:

        * a frozen directory source is a container run iff any of its frozen
          suffixes names a container FORMAT;
        * a frozen file source is a container run iff its kind (or its own URI's
          suffix) is a container kind.

        W-1C: a frozen DIRECTORY source carries the freeze owner's MATCH
        SUFFIXES, not bare extensions — ``_controls_v2_container_index_config``
        emits ``("_master.h5",)`` for File Type h5 and
        ``("_master.hdf5", "_master.h5")`` for hdf5, because a container
        directory is matched by filename TAIL.  Comparing the whole suffix
        string therefore refused every real Eiger-master directory.  The
        container question is about the format, so compare the extension TOKENS
        the frozen source normalizes for exactly this purpose and let the emitter
        go on spelling its own match rule — the alternative, teaching this
        consumer the ``_master.*`` literals, would re-create exactly the emitter
        coupling R4B-14 removed.
        """

        source = frozen.source
        if str(source.source_kind) in _CONTAINER_SOURCE_KINDS:
            return True
        return any(token in _CONTAINER_SUFFIXES
                   for token in source.format_tokens)

    def get_next_image(self, frozen):
        """Gets next image in image series or in directory to process."""
        series_average = frozen.run_options.get("series_average", False)
        meta_ext = frozen.run_options.get("meta_ext")
        # O-1a-W1R (review §39.2 W1R-P1-5 items 1-2): the reader FAMILY is the
        # frozen source's decision.  The parent derived ``is_master`` from the
        # mutable ``img_file`` cursor and evaluated ``single_img`` before the
        # frozen container decision, so a frozen TIFF series plus a
        # master-shaped mutable path routed to the Eiger reader and a frozen
        # container directory plus ``single_img=True`` routed to plain-image
        # handling.  ``img_file`` stays the runtime cursor; it no longer decides
        # the family.
        is_container = imageThread._frozen_source_is_container(self, frozen)

        if (frozen.source.source_kind == "image_file") and not is_container:
            scan_name, img_number = _get_scan_info(self.img_file)
            self._record_discovered_frame()
            if self._should_skip_before_read(frozen, scan_name, img_number):
                self.sigUpdate.emit(self._append_output_number(frozen, img_number))
                return None, scan_name, img_number, None, {}
            before = _source_observation(self.img_file)
            img_data = self._read_frame_tolerant(self.img_file)
            if img_data is None:
                return None, scan_name, img_number, None, {}
            observation = _source_observation(self.img_file)
            if observation.version_stamp != before.version_stamp:
                return None, scan_name, img_number, None, {}
            meta = read_image_metadata(self.img_file, meta_format=meta_ext, meta_dir=self.meta_dir) if meta_ext else {}
            facts = dict(observation=observation, source_revision=max(1, observation.mtime_ns), container=False,
                         source_frame_idx=0, commit_path=None)
            return (self.img_file, scan_name, img_number, img_data,
                    _SourceMetadata(meta, facts))

        if is_container:
            return self._get_next_eiger_frame(frozen)

        if len(self.img_fnames) == 0:
            if frozen.source.family != "directory":
                source_spec = frozen.thaw_source_spec()
                frozen_files = tuple(
                    getattr(source_spec, "options", {}).get("files", ())
                    if source_spec is not None else ()
                )
                if frozen_files:
                    # Controls freezes the COMPLETE strict series at Run start.
                    # The picked member is selection context, never a lower
                    # frame bound (selecting frame 4 still means frames 1..N).
                    self.img_fnames = [Path(path) for path in frozen_files]
                else:
                    # Legacy fallback keeps the same full-series semantics.
                    token = (frozen.source.format_tokens or ("",))[0]
                    _series_re = re.compile(
                        rf'^{re.escape(self.scan_name)}[_-]\d+\.'
                        rf'{re.escape(token)}$',
                        re.IGNORECASE,
                    )
                    self.img_fnames = [
                        p for p in _paths_with_suffix(
                            Path(frozen.source.filesystem_root), f'.{token}')
                        if _series_re.match(p.name)
                    ]
            else:
                match = _name_filter((frozen.source.name_filter or ""))
                # The freeze owner's own match suffixes, all alternatives.
                candidates = [
                    path
                    for suffix in frozen.source.suffixes
                    for path in _paths_with_suffix(
                        Path(frozen.source.filesystem_root), suffix,
                        recursive=frozen.source.recursive)
                    if match(path.name[:-len(suffix)])
                ]
                self.img_fnames = candidates

            processed = set(str(f) for f in self.processed)
            self.img_fnames = [
                str(f) for f in self.img_fnames
                if str(f) not in processed
            ]

            self.img_fnames = deque(sorted(
                self.img_fnames, key=_series_frame_sort_key))

        img_file, scan_name, img_number, img_data, img_meta = None, None, 1, None, {}
        n = 0
        while len(self.img_fnames) > 0:
            fname = self.img_fnames[0]
            sname, snumber = _get_scan_info(fname)

            if (n > 0) and (scan_name != sname):
                break

            if self._should_skip_before_read(frozen, sname, snumber):
                self._commit_frame(fname)
                continue

            before = _source_observation(fname)
            data = self._read_frame_tolerant(fname)
            if data is None:
                # File is present but not readable yet.  In LIVE mode it may
                # still be flushing to disk: leave it at the head of the queue
                # and re-poll on the next sweep (never dropped) until the
                # cross-sweep deadline; only then -- or in batch, where every
                # file is already complete -- give up and skip it.
                if (not frozen.batch_mode) and not self._frame_read_deadline_reached(fname):
                    return None, sname, snumber, None, {}
                self._commit_frame(fname)
                self._record_skip_reason("unreadable image (gave up after deadline)")
                continue

            observation = _source_observation(fname)
            if observation.version_stamp != before.version_stamp:
                return None, sname, snumber, None, {}
            if not np.isfinite(data).any():
                self._commit_frame(fname)
                self._record_skip_reason("unreadable or empty image data")
                continue

            meta = read_image_metadata(fname, meta_format=meta_ext, meta_dir=self.meta_dir) if meta_ext else {}
            n += 1

            if (not series_average) or (snumber is None):
                self._record_discovered_frame()
                facts = dict(observation=observation, source_revision=max(1, observation.mtime_ns), container=False,
                             source_frame_idx=0, commit_path=fname, discovery_counted=True)
                return fname, sname, snumber, data, _SourceMetadata(meta, facts)
            else:
                if n == 1:
                    img_data = data
                    img_meta = meta
                else:
                    img_data += data
                    for (k, v) in meta.items():
                        try:
                            img_meta[k] = float(img_meta[k]) + float(meta[k])
                        except TypeError:
                            pass

                scan_name, img_file = sname, fname

        if n > 1:
            img_data /= n
            for (k, v) in img_meta.items():
                try:
                    img_meta[k] /= n
                except TypeError:
                    pass

        return img_file, scan_name, img_number, img_data, img_meta

    # ── Metadata / Background ────────────────────────────────────────────

    def get_meta_data(self, frozen, img_file):
        meta_ext = frozen.run_options.get("meta_ext")
        if not meta_ext:
            # GUI "none"/blank is an off switch.  The reusable headless reader
            # maps None to auto, so the GUI worker must guard before calling it.
            return {}
        return read_image_metadata(img_file, meta_format=meta_ext, meta_dir=self.meta_dir)

    def subtract_bg(self, frozen, img_data, img_file, img_number, img_meta):
        bg = self.get_background(frozen, img_file, img_number, img_meta)
        try:
            img_data -= bg
        except ValueError:
            pass

    def _handle_append_config_mismatch(self, exc):
        """Stop the run CLEANLY on a mid-run Append config mismatch.

        The guard in ``initialize_scan`` is CORRECT — 1D and 2D configs must
        not mix into one append file, and the target was preserved.  The
        DELIVERY used to be the bug: the bare RuntimeError escaped ``run()``
        (try/…/finally, no except) as an unhandled QThread exception and
        killed the run through the GUI excepthook.  Instead: the same clean
        stop as a user Stop (``command='stop'`` → the shared end-of-run tail
        still force-flushes the previous scan; ``run()``'s finally releases
        the reduction session, prefetcher and Eiger handle) plus
        ``sigAppendMismatch`` so the GUI surfaces the one-modal warning.
        """
        logger.error("run stopped: %s", exc)
        self._append_config_mismatch = True
        self.command = 'stop'
        try:
            self.sigAppendMismatch.emit(str(exc))
        except Exception:
            logger.debug("sigAppendMismatch emit failed", exc_info=True)

    def _handle_initialize_scan_write_error(self, exc):
        """Stop the run CLEANLY when the output file cannot be (re)written.

        DIR-3 (bl17-2, Windows/SMB): the replace-save's ``os.replace`` can
        fail with WinError 5 when an external program (antivirus, indexer,
        preview pane, another share client) holds the destination open — the
        writer retried and preserved the existing file; the DELIVERY must be
        the same clean stop as :meth:`_handle_append_config_mismatch`, never
        an unhandled QThread exception that kills the whole directory run.
        """
        logger.exception("run stopped: output file not writable: %s", exc)
        self.command = 'stop'
        try:
            self.showLabel.emit(
                "Run stopped: output file is locked by another program; "
                "the existing file was preserved (see log)")
        except Exception:
            logger.debug("showLabel emit failed", exc_info=True)

    def _output_safety_args(self, frozen):
        """Assemble the source/output collision-guard inputs from run config.

        Shared by the run-start preflight (fail loud before any read) and the
        per-scan guard in :meth:`initialize_scan` (the last checkpoint before a
        writer opens), so both consult ONE headless owner
        (:func:`xrd_tools.io.output_safety.check_output_not_source`).  Pure
        assembly — no I/O.
        """
        img_file = getattr(self, "img_file", "") or ""
        is_dir = frozen.source.family == "directory"
        # Self-contained container input (kept independent of the output suffix,
        # per the safety contract): a directory File Type of nxs/h5/hdf5, or a
        # single container/Eiger-master input file.
        container = (
            any(token in _CONTAINER_SUFFIXES
                for token in frozen.source.format_tokens)
            or (bool(img_file) and _is_eiger_master(img_file))
            or (bool(img_file)
                and Path(img_file).suffix.lower() in ('.h5', '.hdf5', '.nxs'))
        )
        watched = ([frozen.source.filesystem_root]
                   if is_dir and frozen.source.filesystem_root else [])
        # Known source file(s): the explicit input, and (best effort) the
        # container currently open.  Directory-mode same-file overwrite is caught
        # structurally by the watched-dir check, so a raced/prefetched master
        # here can only ever ADD a true collision, never mask one.
        inputs = []
        if img_file:
            inputs.append(img_file)
        cur = getattr(self, "_eiger_master_path", None)
        if cur:
            inputs.append(cur)
        return dict(
            input_files=inputs,
            watched_dirs=watched,
            recursive=bool(frozen.source.recursive),
            container_directory_mode=bool(container),
        )

    def _handle_output_collision(self, exc):
        """Stop the run CLEANLY when the output path would destroy/re-ingest its
        own input (F-NXS-1).

        The guard fires BEFORE any writer opens, so the raw source and any
        existing destination are untouched.  Delivery mirrors
        :meth:`_handle_initialize_scan_write_error`: the same clean stop as a
        user Stop plus an actionable operator message, never an unhandled QThread
        exception that would kill the whole directory run.
        """
        logger.error("run stopped: unsafe output path: %s", exc)
        self.command = 'stop'
        try:
            self.showLabel.emit(f"Run stopped: {exc}")
        except Exception:
            logger.debug("showLabel emit failed", exc_info=True)

    def initialize_scan(self):
        """If scan changes, initialize new LiveScan object.
        If mode is overwrite, replace existing HDF5 file, else append to it.

        O-1a-W1A: the per-run ``LiveScan`` is built ENTIRELY from the accepted
        frozen configuration -- ``frozen.scan_kwargs()`` for every integration /
        GI / threshold value, ``frozen.processing_mapping()`` for the Append
        comparison -- and carries the exact accepted object plus its detached
        JSON-native provenance projection BEFORE any writer opens.  The mutable
        display scan and the panel-derived thread mirrors are not read here.
        """
        # This is the last checkpoint before a writer can open, so the typed
        # refusal happens BEFORE the output-safety probe, the mkdir, and any
        # source read (W-1.2 case 5).
        frozen = imageThread._require_run_configuration(
            self, "initialize_scan")
        scan_kwargs = frozen.scan_kwargs()
        series_average = frozen.run_options.get("series_average", False)
        xye_only = frozen.run_options.get("xye_only", False)
        fname = os.fspath(resolve_output_target(
            frozen.save_path, self.scan_name, mode=frozen.output_mode))
        # F-NXS-1: refuse to (over)write when the derived output collides with a
        # raw source — Save Path == the watched raw directory makes source ==
        # output for a same-stem container, which overwrote a 1.8 MB raw
        # acquisition with a 26 KB empty processed container before any frame was
        # reduced.  This is the LAST checkpoint before any writer opens; raised
        # BEFORE mkdir/replace-save so both the raw source and any existing
        # destination keep their bytes.  Caught at the initialize_scan call sites
        # for a clean run stop (like the DIR-3 locked-destination handling).
        # The dormant XYE selection has no .nxs target and is refused earlier.
        if not xye_only:
            check_output_not_source(fname, **self._output_safety_args(frozen))
        # Eiger master files are pre-processed with the trailing
        # ``_master`` suffix stripped from scan_name (see
        # _get_next_eiger_frame). Without this sync, the wrangler
        # widget's self.fname (set from the original master filename
        # in image_wrangler.setup()) diverges from the actual scan
        # output path, and static_scan_widget.wrangler_finished
        # cannot find the generated file to reload at end of batch.
        self.fname = fname
        scan = LiveScan(self.scan_name,
                          data_file=fname,
                          static=True,
                          gi=bool(scan_kwargs["gi"]),
                          incidence_motor=scan_kwargs["incidence_motor"],
                          # acquisition shape, not run configuration: how frames
                          # arrive from THIS source, never a Controls value.
                          series_average=series_average,
                          single_img=(frozen.source.source_kind == "image_file"),
                          global_mask=self.mask,
                          detector_shape=self.detector_shape,
                          # J2: share lock with wrangler save path
                          file_lock=self.file_lock,
                          bai_1d_args=scan_kwargs["bai_1d_args"],
                          bai_2d_args=scan_kwargs["bai_2d_args"])
        _apply_frozen_run_configuration(scan, frozen)
        # N1: the project root -> entry/@source_base + relative raw source paths
        # in the writer (portable .nxs).  None -> absolute paths (back-compat).
        scan.source_base = getattr(self, "source_base", None)
        scan._committed_append_prefix = getattr(
            self, "_append_committed_prefix_by_scan", {}).get(str(scan.name))
        # v2 NeXus writer needs a Diffractometer to derive per-frame
        # rot1/rot2/rot3 + incidence-angle arrays from scan_data.  default_geometry()
        # picks the preset from what scan_data recorded: psic when nu/del are
        # present (RSM/6-circle), else the two-circle convention (rot1←tth,
        # incidence ← the resolved sample-tilt motor).  Override later from the
        # geometry UI panel when the user picks a different convention.
        scan.default_geometry()
        # Bluesky/NXWriter source: stamp the embedded wavelength as a low-
        # precedence provenance fallback so the processed output (and its
        # run-end reload) has a real wavelength instead of NaN.  No-op for any
        # other source; PONI geometry still wins (see _stamp_bluesky_wavelength).
        self._stamp_bluesky_wavelength(scan)

        logger.info('***** New Scan *****')
        if not xye_only:
            logger.info('Output file: %s', fname)

        return scan

    def get_mask(self, frozen):
        """Get mask array from mask file."""
        self.mask = self.detector.calc_mask()
        if frozen.mask_file and os.path.exists(frozen.mask_file):
            if self.mask is not None:
                try:
                    self.mask += fabio.open(frozen.mask_file).data
                except ValueError:
                    logger.warning('Mask file not valid for Detector (shape mismatch)')
                    pass
            else:
                self.mask = fabio.open(frozen.mask_file).data

        if self.mask is None:
            return None

        if self.mask.shape != self.detector.shape:
            logger.warning('Mask file not valid for Detector (shape %s != %s)', self.mask.shape, self.detector.shape)
            return None

        self.mask = np.flatnonzero(self.mask)

    def threshold(self, frozen, img_data):
        """Return flat indices of pixels outside [threshold_min, threshold_max]."""
        mask = (img_data < frozen.threshold.threshold_min) | (img_data > frozen.threshold.threshold_max)
        return np.flatnonzero(mask)

    def get_background(self, frozen, img_file, img_number, img_meta):
        """Subtract background image if bg_file or bg_dir specified."""
        meta_ext = frozen.run_options.get("meta_ext")
        if self.bg_type == 'None':
            return 0

        bg, bg_file, bg_meta, norm_factor = 0, None, None, 1
        self.sub_label, norm_label, bg_scale_label = '', '', ''

        if self.bg_type == 'Single BG File':
            if self.bg_file:
                bg_file = self.bg_file
                bg_meta = self.get_meta_data(frozen, bg_file)
        elif self.bg_type == 'Series Average':
            if self.bg_file:
                sname, fnames, bg, bg_meta = get_series_avg(self.bg_file, self.detector, meta_ext)
                if sname is None:
                    return 0
        else:
            if self.bg_dir and (self.bg_match_fname or self.bg_matching_par):
                bg_file_filter = 'bg' if not self.bg_file_filter else self.bg_file_filter
                match = _name_filter(bg_file_filter)
                if self.bg_match_fname:
                    # scan_name is DATA, not filter grammar: a name starting
                    # with '-' or containing '|'/'OR' must stay a literal
                    # substring requirement, so conjoin it outside the
                    # compiled expression.
                    scan_term = str(self.scan_name).lower()
                    match = (lambda name, _m=match, _t=scan_term:
                             _t in str(name).lower() and _m(name))
                if not meta_ext:
                    meta_files = []
                    suffix = ''
                else:
                    suffix = f'.{meta_ext}'
                    meta_files = sorted(
                        str(f) for f in _paths_with_suffix(frozen.source.filesystem_root, suffix)
                        if match(f.name[:-len(suffix)])
                    )

                for meta_file in meta_files:
                    bg_file = (f'{os.path.splitext(meta_file)[0]}'
                               f'.{(frozen.source.format_tokens or ("",))[0]}')
                    if bg_file == img_file:
                        bg_file = None
                        continue

                    bg_meta = self.get_meta_data(frozen, bg_file)
                    if self.bg_match_fname:
                        _, meta_img_num = _get_scan_info(meta_file)
                        if img_number == meta_img_num:
                            break
                    else:
                        try:
                            if bg_meta[self.bg_matching_par] == img_meta[self.bg_matching_par]:
                                break
                        except KeyError:
                            bg_file = None
                            continue

        if self.bg_type != 'Series Average':
            if bg_file is None:
                return 0.

            bg = np.asarray(read_image(bg_file), dtype=float)
            if bg is None or not np.isfinite(bg).any():
                return 0.

        if self.bg_scale != 1:
            bg *= self.bg_scale
            bg_scale_label = f'{self.bg_scale:0.2f} [Scale] x '
        if (self.bg_norm_channel != 'None') and (img_meta is not None) and (bg_meta is not None):
            try:
                if ((self.bg_norm_channel in img_meta.keys()) and
                        (self.bg_norm_channel in bg_meta.keys()) and
                        (bg_meta[self.bg_norm_channel] != 0)):
                    norm_factor = (img_meta[self.bg_norm_channel] / bg_meta[self.bg_norm_channel])
                    bg *= norm_factor
                    norm_label = f'{norm_factor:0.2f} [Normalized to Channel - {self.bg_norm_channel}] x '
            except (KeyError, TypeError):
                pass

        if self.bg_type != 'Series Average':
            self.sub_label = f'[Subtracted {bg_scale_label}{norm_label}{os.path.basename(bg_file)}]'
        else:
            self.sub_label = f'[Subtracted {bg_scale_label}{norm_label}{sname}]'

        return bg
