# ssrl_xrd_tools/integrate/batch.py
"""
Batch processing of detector scans and live directory watching for beamline
data reduction.

``process_scan`` / ``process_series`` replace the per-scan loops in the
experimental notebooks.  ``DirectoryWatcher`` replaces the common beamline
pattern::

    while True:
        for scan in find_new_scans(base_path):
            process_scan(scan, ...)
        time.sleep(30)

Optional dependency: ``watchdog`` (filesystem event backend).  When not
installed, ``DirectoryWatcher`` falls back to polling.  Do not add ``watchdog``
to pyproject.toml; it will be declared as an optional dependency later.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from ssrl_xrd_tools.io.export import write_h5
from ssrl_xrd_tools.io.image import (
    SUPPORTED_EXTS,
    find_image_files,
    get_detector_mask,
    read_image,
    read_image_stack,
)
from ssrl_xrd_tools.integrate.single import integrate_1d, integrate_2d

if TYPE_CHECKING:
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_path(p: Path | str) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _frame_done(h5_path: Path, frame_idx: int) -> bool:
    """Return True if frame group already exists in the output HDF5."""
    if not h5_path.exists():
        return False
    try:
        with h5py.File(h5_path, "r") as f:
            return str(frame_idx) in f
    except Exception:
        return False


def _nframes_hdf5(path: Path) -> int | None:
    """Return number of frames in an HDF5 master file, or None on error."""
    try:
        with h5py.File(path, "r") as f:
            for candidate in (
                "/entry/data/data",
                "/entry/instrument/detector/data",
                "/data",
            ):
                if candidate in f:
                    ds = f[candidate]
                    return 1 if ds.ndim == 2 else ds.shape[0]
            # fallback: largest dataset
            found: dict[str, h5py.Dataset] = {}
            f.visititems(
                lambda name, obj: found.update({name: obj})
                if isinstance(obj, h5py.Dataset)
                else None
            )
            if not found:
                return None
            ds = max(found.values(), key=lambda d: d.size)
            return 1 if ds.ndim == 2 else ds.shape[0]
    except Exception as exc:
        logger.warning("Could not determine frame count for %s: %s", path, exc)
        return None


def _collect_frames(
    scan_path: Path,
    detector: str,
    mask: np.ndarray | None,
    threshold: float,
    rotation: int,
) -> tuple[list[tuple[np.ndarray, int]], np.ndarray | None]:
    """
    Return ``[(image_array, frame_idx), ...]`` and the effective mask.

    Handles both HDF5 master files (all frames from the stack) and image
    directories (one file per frame).
    """
    det_mask = get_detector_mask(detector) if detector else None
    combined_mask: np.ndarray | None
    if mask is not None and det_mask is not None:
        combined_mask = mask | det_mask
    elif det_mask is not None:
        combined_mask = det_mask
    else:
        combined_mask = mask

    ext = scan_path.suffix.lower()
    if ext in {".h5", ".hdf5"} and scan_path.is_file():
        stack = read_image_stack(
            scan_path, mask=combined_mask, threshold=threshold, rotation=rotation
        )
        if stack.ndim == 2:
            stack = stack[np.newaxis]
        return [(stack[i], i) for i in range(stack.shape[0])], combined_mask

    if scan_path.is_dir():
        img_files = find_image_files(scan_path)
        frames = []
        for idx, p in enumerate(img_files):
            img = read_image(p, mask=combined_mask, threshold=threshold, rotation=rotation)
            frames.append((img, idx))
        return frames, combined_mask

    # single image file
    img = read_image(scan_path, mask=combined_mask, threshold=threshold, rotation=rotation)
    return [(img, 0)], combined_mask


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def process_scan(
    scan_dir: Path | str,
    ai: AzimuthalIntegrator,
    output_path: Path | str,
    npt: int = 4000,
    npt_rad: int = 1000,
    npt_azim: int = 1000,
    unit: str = "q_A^-1",
    method: str = "csr",
    mask: np.ndarray | None = None,
    azimuth_range: tuple[float, float] | None = None,
    radial_range: tuple[float, float] | None = None,
    threshold: float = 1e9,
    detector: str = "",
    rotation: int = 0,
    reprocess: bool = False,
    **kwargs: Any,
) -> Path:
    """
    Process all frames in a scan directory or HDF5 file and write results to HDF5.

    For each frame the function:

    1. Reads the detector image (applies threshold, mask, rotation).
    2. Integrates 1-D (via :func:`integrate_1d`) and 2-D cake (via
       :func:`integrate_2d`).
    3. Saves to *output_path* with :func:`write_h5` (groups named by frame
       index: ``"0"``, ``"1"``, …).
    4. Skips already-processed frames unless ``reprocess=True``.

    Parameters
    ----------
    scan_dir : Path or str
        Path to an HDF5 master file (*``_master.h5``*) or a directory of
        single-frame image files.
    ai : AzimuthalIntegrator
        Configured pyFAI integrator.
    output_path : Path or str
        Destination HDF5 file.  Created (or appended to) automatically.
    npt : int, optional
        Radial bins for the 1-D integration.
    npt_rad, npt_azim : int, optional
        Bins for the 2-D cake integration.
    unit : str, optional
        Radial unit, e.g. ``"q_A^-1"``.
    method : str, optional
        pyFAI integration method.
    mask : ndarray or None, optional
        Additional bad-pixel mask merged with the detector mask.
    azimuth_range : tuple of float or None, optional
        Azimuthal range in degrees for 2-D integration.
    radial_range : tuple of float or None, optional
        Radial range for both 1-D and 2-D integrations.
    threshold : float, optional
        Pixels above this value are replaced with NaN.
    detector : str, optional
        pyFAI detector name used to look up the built-in bad-pixel mask.
    rotation : int, optional
        Image rotation in degrees (multiple of 90).
    reprocess : bool, optional
        If ``True``, overwrite existing frame groups in the output HDF5.
    **kwargs
        Forwarded to both :func:`integrate_1d` and :func:`integrate_2d`.

    Returns
    -------
    Path
        Absolute path to the output HDF5 file.
    """
    scan_path = _as_path(scan_dir)
    out_path = _as_path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames, effective_mask = _collect_frames(
        scan_path, detector, mask, threshold, rotation
    )
    if not frames:
        logger.warning("No frames found in %s", scan_path)
        return out_path.resolve()

    n_total = len(frames)
    n_skipped = n_done = 0

    logger.info("Processing %s: %d frames → %s", scan_path.name, n_total, out_path)

    for img, idx in frames:
        if not reprocess and _frame_done(out_path, idx):
            n_skipped += 1
            continue

        try:
            r1d = integrate_1d(
                img, ai,
                npt=npt,
                unit=unit,
                method=method,
                mask=effective_mask,
                radial_range=radial_range,
                azimuth_range=azimuth_range,
                **kwargs,
            )
            r2d = integrate_2d(
                img, ai,
                npt_rad=npt_rad,
                npt_azim=npt_azim,
                unit=unit,
                method=method,
                mask=effective_mask,
                radial_range=radial_range,
                azimuth_range=azimuth_range,
                **kwargs,
            )
        except Exception as exc:
            logger.error(
                "Integration failed for %s frame %d: %s", scan_path.name, idx, exc
            )
            continue

        # IntegrationResult2D has shape (npt_rad, npt_azim); write_h5 expects
        # (npt_rad, npt_azim) as IQChi.
        write_h5(
            out_path,
            frame=idx,
            q=r1d.radial,
            intensity=r1d.intensity,
            iqchi=r2d.intensity,
            q_2d=r2d.radial,
            chi=r2d.azimuthal,
        )
        n_done += 1

    logger.info(
        "%s: %d processed, %d skipped (already done), %d frames total",
        scan_path.name,
        n_done,
        n_skipped,
        n_total,
    )
    return out_path.resolve()


def process_series(
    scan_paths: Sequence[Path | str],
    ai: AzimuthalIntegrator,
    output_dir: Path | str,
    reprocess: bool = False,
    **kwargs: Any,
) -> list[Path]:
    """
    Process a sequence of scans in order.

    Each scan is written to ``<output_dir>/<scan_stem>_processed.h5``.

    Parameters
    ----------
    scan_paths : sequence of Path or str
        Paths to HDF5 master files or scan directories.
    ai : AzimuthalIntegrator
        Configured pyFAI integrator.
    output_dir : Path or str
        Directory where per-scan HDF5 files are written.
    reprocess : bool, optional
        Passed through to :func:`process_scan`.
    **kwargs
        Forwarded to :func:`process_scan`.

    Returns
    -------
    list of Path
        Output HDF5 paths, one per input scan.
    """
    out_dir = _as_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[Path] = []
    n = len(scan_paths)  # type: ignore[arg-type]
    for i, raw_path in enumerate(scan_paths, start=1):
        scan_path = _as_path(raw_path)
        out_h5 = out_dir / f"{scan_path.stem}_processed.h5"
        logger.info("[%d/%d] Processing %s", i, n, scan_path.name)
        try:
            out = process_scan(
                scan_path,
                ai,
                out_h5,
                reprocess=reprocess,
                **kwargs,
            )
            results.append(out)
        except Exception as exc:
            logger.error("process_series: failed on %s: %s", scan_path.name, exc)
    return results


# ---------------------------------------------------------------------------
# Directory watcher
# ---------------------------------------------------------------------------

class DirectoryWatcher:
    """
    Watch a directory tree for new data files and process them automatically.

    Uses ``watchdog`` for filesystem-event-based watching when available,
    otherwise falls back to polling with ``time.sleep(poll_interval)``.

    Parameters
    ----------
    watch_dir : Path or str
        Root directory to watch.
    ai : AzimuthalIntegrator
        Configured pyFAI integrator shared across all processed scans.
    output_dir : Path or str
        Directory where processed HDF5 files are written.
    patterns : sequence of str, optional
        Glob patterns for matching new data files.
    recursive : bool, optional
        If ``True`` (default), watch subdirectories as well.
    poll_interval : float, optional
        Seconds between directory scans when watchdog is unavailable.
    **process_kwargs
        Forwarded to :func:`process_scan` for every new file.
    """

    def __init__(
        self,
        watch_dir: Path | str,
        ai: AzimuthalIntegrator,
        output_dir: Path | str,
        patterns: Sequence[str] = ("*_master.h5", "*.edf", "*.raw"),
        recursive: bool = True,
        poll_interval: float = 10.0,
        **process_kwargs: Any,
    ) -> None:
        self._watch_dir = _as_path(watch_dir)
        self._ai = ai
        self._output_dir = _as_path(output_dir)
        self._patterns = list(patterns)
        self._recursive = recursive
        self._poll_interval = float(poll_interval)
        self._process_kwargs = process_kwargs

        self._processed: set[Path] = set()
        self._stop_event = threading.Event()

        # Probe for watchdog
        try:
            import watchdog.observers  # noqa: F401
            self._has_watchdog = True
        except ImportError:
            self._has_watchdog = False
            logger.debug(
                "watchdog not installed; DirectoryWatcher will use polling "
                "(install watchdog for filesystem-event-based watching)"
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Start watching (blocking until stopped or interrupted).

        Uses watchdog when available; falls back to polling.
        Handles ``KeyboardInterrupt`` gracefully.
        """
        self._stop_event.clear()
        logger.info(
            "DirectoryWatcher started: watch_dir=%s output_dir=%s patterns=%s",
            self._watch_dir,
            self._output_dir,
            self._patterns,
        )
        try:
            if self._has_watchdog:
                self._run_watchdog()
            else:
                self._run_polling()
        except KeyboardInterrupt:
            logger.info("DirectoryWatcher: stopping (KeyboardInterrupt).")
        finally:
            self._stop_event.set()
            logger.info("DirectoryWatcher: stopped.")

    def start_background(self) -> threading.Thread:
        """
        Start watching in a daemon thread (non-blocking).

        Returns
        -------
        threading.Thread
            The background thread.  Call :meth:`stop` to request termination,
            then ``thread.join()`` to wait for clean exit.
        """
        t = threading.Thread(target=self.start, daemon=True, name="DirectoryWatcher")
        t.start()
        return t

    def stop(self) -> None:
        """Signal the watcher loop to stop after the current poll cycle."""
        self._stop_event.set()

    @property
    def processed_files(self) -> set[Path]:
        """Return a copy of the set of files that have been processed."""
        return set(self._processed)

    # ------------------------------------------------------------------
    # Internal: scan & process
    # ------------------------------------------------------------------

    def _find_matching_files(self) -> list[Path]:
        """Glob all patterns in the watch directory."""
        found: list[Path] = []
        glob_fn = self._watch_dir.rglob if self._recursive else self._watch_dir.glob
        for pattern in self._patterns:
            for p in glob_fn(pattern):
                if p.is_file():
                    found.append(p)
        return found

    def _process_new_file(self, path: Path) -> None:
        """Process a single newly detected file, skipping known files."""
        if path in self._processed:
            return
        self._processed.add(path)
        out_h5 = self._output_dir / f"{path.stem}_processed.h5"
        logger.info("DirectoryWatcher: new file detected: %s", path)
        try:
            process_scan(path, self._ai, out_h5, **self._process_kwargs)
        except Exception as exc:
            logger.error(
                "DirectoryWatcher: error processing %s: %s", path, exc
            )

    # ------------------------------------------------------------------
    # Internal: polling backend
    # ------------------------------------------------------------------

    def _run_polling(self) -> None:
        logger.debug("DirectoryWatcher: using polling backend (interval=%.1fs)", self._poll_interval)
        while not self._stop_event.is_set():
            for p in self._find_matching_files():
                if not self._stop_event.is_set():
                    self._process_new_file(p)
            self._stop_event.wait(timeout=self._poll_interval)

    # ------------------------------------------------------------------
    # Internal: watchdog backend
    # ------------------------------------------------------------------

    def _run_watchdog(self) -> None:
        from watchdog.events import FileSystemEventHandler, FileCreatedEvent
        from watchdog.observers import Observer

        watcher = self

        class _Handler(FileSystemEventHandler):
            def on_created(self, event: FileCreatedEvent) -> None:
                if event.is_directory:
                    return
                path = Path(event.src_path)
                if any(path.match(pat) for pat in watcher._patterns):
                    watcher._process_new_file(path)

        # Process any files that already exist before starting the observer
        for p in self._find_matching_files():
            self._process_new_file(p)

        observer = Observer()
        observer.schedule(_Handler(), str(self._watch_dir), recursive=self._recursive)
        observer.start()
        logger.debug("DirectoryWatcher: watchdog observer started.")
        try:
            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=1.0)
        finally:
            observer.stop()
            observer.join()
            logger.debug("DirectoryWatcher: watchdog observer stopped.")


__all__ = [
    "DirectoryWatcher",
    "process_scan",
    "process_series",
]
