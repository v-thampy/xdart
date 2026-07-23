# -*- coding: utf-8 -*-
"""Shared scan-source widget (the §3 design of design_shared_source_panel).

One reusable, kind-general source picker — used by the ROI Scan Plotter now and
the stitch/RSM wrangler later.  It assembles a :class:`SourceSpec`, opens it in
its probe worker only to read the raw-reachable flag and an immutable byte
preview of the first frame, closes it there, and emits a VALUE-ONLY
:class:`ScanSelection` via :data:`sigSourceChanged` (H19 §3: no live
``FrameSource``, file handle, or mutable ndarray crosses the worker boundary — a
consumer opens its own source from the spec inside its own I/O scope).  All
parsing/IO is headless (`xrd_tools.sources` / `io`); this is the thin Qt layer.

Two entry modes (§2.2): a single master **File**, or a **Directory** + a
Scan-kind dropdown (`discover_scans`).  The Scan selector then lists the
candidate scans (SPEC scan numbers / NeXus entries / discovered scans).  An
optional **Images** folder (SPEC) pairs raw frames; the **raw-reachable dot**
(metadata-independent) gates ROI/stitch/RSM downstream.  Grouping (combine scans
via `CompositeFrameSource`) shows only for stitch/RSM.
"""

import logging
import re
from dataclasses import dataclass
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from xrd_tools.sources.probe import probe_first_frame, raw_is_reachable  # noqa: F401

logger = logging.getLogger(__name__)

#: (label, SourceKind value) for the Directory-mode scan-kind dropdown.
_DIR_KINDS = [("SPEC", "spec"), ("TIFF / RAW series", "tiff_series"),
              ("NeXus (raw stack)", "nexus_stack"),
              ("Processed NeXus", "processed_nexus"), ("Eiger", "eiger_master")]
_SERIES_SUFFIXES = {".tif", ".tiff", ".raw"}
_RAW_DTYPES = ["int32", "uint32", "int16", "uint16", "float32", "float64"]


@dataclass(frozen=True, slots=True)
class ImagePreview:
    """Immutable transport form for one decoded detector image.

    Qt queued signals may cross threads.  Carrying a mutable ndarray there makes
    a frozen outer dataclass only superficially immutable, so the probe worker
    serializes the contiguous numeric payload to ``bytes`` and the GUI rebuilds
    a read-only NumPy view only when it needs to display the ROI picker.
    """

    shape: tuple[int, ...]
    dtype: str
    data: bytes

    def __post_init__(self) -> None:
        dtype = np.dtype(self.dtype)
        if dtype.hasobject:
            raise TypeError("image previews cannot carry object-dtype data")
        if not isinstance(self.data, bytes):
            raise TypeError("image preview data must be immutable bytes")
        if any(int(n) < 0 for n in self.shape):
            raise ValueError("image preview shape cannot contain negative sizes")
        expected = int(np.prod(self.shape, dtype=np.int64)) * dtype.itemsize
        if len(self.data) != expected:
            raise ValueError(
                f"preview byte count {len(self.data)} does not match "
                f"shape {self.shape} and dtype {dtype.str!r}")

    @classmethod
    def from_array(cls, image) -> "ImagePreview":
        array = np.ascontiguousarray(np.asarray(image))
        return cls(tuple(int(n) for n in array.shape), array.dtype.str,
                   array.tobytes(order="C"))

    def to_array(self) -> np.ndarray:
        array = np.frombuffer(self.data, dtype=np.dtype(self.dtype))
        return array.reshape(self.shape)


@dataclass(frozen=True, slots=True)
class ScanSelection:
    """The widget's output: a VALUE-ONLY classified scan observation.

    H19 §3: no live ``FrameSource`` / file handle crosses the async worker
    boundary or survives its scope.  The worker opens the source, probes the
    first frame, then CLOSES the source inside its own scope; only these value
    fields cross to the Qt thread.  A consumer that needs to iterate frames
    opens its own source from :attr:`spec` on demand (see the ROI Scan
    Plotter).
    """

    spec: object               # SourceSpec | None (a value)
    label: str
    reachable: bool            # raw frames loadable (probe) — independent of metadata
    first_image: ImagePreview | None

    def __post_init__(self) -> None:
        if self.first_image is not None and not isinstance(
                self.first_image, ImagePreview):
            object.__setattr__(
                self, "first_image", ImagePreview.from_array(self.first_image))


class ScanSourceWidget(QtWidgets.QWidget):
    """Pick a scan (any source kind) → emit a :class:`ScanSelection`."""

    #: emitted with a :class:`ScanSelection` (or ``None`` when cleared/invalid).
    sigSourceChanged = QtCore.Signal(object)
    #: internal thread-safe completion channel for async source probes.
    sigProbeDone = QtCore.Signal(object)
    #: internal completion channel for persistent directory observations.
    sigDirectoryDone = QtCore.Signal(object)
    #: emitted with the latest accepted, immutable DirectoryObservation.
    sigDirectoryChanged = QtCore.Signal(object)

    def __init__(self, mode="roi", parent=None, *, async_probe=False):
        super().__init__(parent)
        self._mode = mode
        self._controls_source = mode == "controls_source"
        self._allow_grouping = mode in ("stitch", "rsm")
        self._async_probe = bool(async_probe)
        self._probe_generation = 0
        self._probe_executor = None
        self._pending_sig = None
        self._candidates = []          # list[SourceSpec] for the Scan selector
        self._last_sig = None          # signature of the last-opened spec (dedupe)
        self._last_selection = None
        self.sigProbeDone.connect(self._on_probe_done)
        self.sigDirectoryDone.connect(self._on_directory_done)
        self._directory_session = None
        self._directory_observation = None
        self._directory_signature = None
        self._directory_future = None
        self._directory_request_token = 0
        self._directory_subdirs_lazy = False
        self._directory_timer = None
        self._build_ui()

    # ---- UI -------------------------------------------------------------
    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)

        if self._controls_source:
            from xrd_tools.sources import DirectoryIndexSession

            # Controls only needs a cheap direct-child filename count.  It must
            # not open HDF5 containers or recursively inspect a selected tree;
            # the worker owns those operations after Run starts.
            self._directory_session = DirectoryIndexSession(
                probe_candidates=False)
            self.directory_status = QtWidgets.QLabel("No directory index")
            self.directory_status.setObjectName("controlsV2DirectoryStatus")
            self.directory_status.setWordWrap(True)
            self.directory_status.setToolTip(
                "Counts matching files directly in the selected folder. "
                "Container metadata is opened only after Run starts; when "
                "Subdirs is enabled, child folders are processed lazily.")
            lay.addWidget(self.directory_status)
            self._directory_timer = QtCore.QTimer(self)
            self._directory_timer.setInterval(1000)
            self._directory_timer.timeout.connect(self.request_directory_poll)
            self._directory_timer.start()
            return

        # Row 1: entry mode + path + (kind label | scan-kind combo) + Choose.
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(6)
        self.dir_check = QtWidgets.QCheckBox("Folder")
        # Reserve room for the full label (it was clipping to "Folde" on macOS).
        self.dir_check.setMinimumWidth(
            self.dir_check.fontMetrics().horizontalAdvance("Folder") + 30)
        self.dir_check.setToolTip(
            "Directory mode: pick a folder + a scan kind; the folder is walked "
            "for matching scans")
        row1.addWidget(self.dir_check)
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("(no scan chosen)")
        row1.addWidget(self.path_edit, 1)
        self.dir_kind_combo = QtWidgets.QComboBox()
        for label, _ in _DIR_KINDS:
            self.dir_kind_combo.addItem(label)
        self.dir_kind_combo.setVisible(False)
        self.dir_kind_combo.setToolTip("Scan kind to look for in the folder")
        row1.addWidget(self.dir_kind_combo)
        self.kind_label = QtWidgets.QLabel("")
        self.kind_label.setMinimumWidth(56)
        row1.addWidget(self.kind_label)
        self.choose_btn = QtWidgets.QPushButton("Choose…")
        row1.addWidget(self.choose_btn)
        lay.addLayout(row1)

        # Row 2: scan/entry selector (shown when >1 candidate).
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(6)
        self.scan_label = QtWidgets.QLabel("Scan")
        self.scan_combo = QtWidgets.QComboBox()
        self.scan_combo.setMinimumWidth(90)
        self.scan_combo.setToolTip("Which scan / entry to use")
        row2.addWidget(self.scan_label)
        row2.addWidget(self.scan_combo)
        row2.addStretch(1)
        self.scan_label.setVisible(False)
        self.scan_combo.setVisible(False)
        lay.addLayout(row2)

        # Row 3: images folder + filename root + raw-reachable dot.
        self.images_row = QtWidgets.QWidget()
        row3 = QtWidgets.QHBoxLayout(self.images_row)
        row3.setContentsMargins(0, 0, 0, 0)
        row3.setSpacing(6)
        self.images_label = QtWidgets.QLabel("Images")
        row3.addWidget(self.images_label)
        self.image_dir_edit = QtWidgets.QLineEdit()
        self.image_dir_edit.setPlaceholderText("(image folder — blank = metadata only)")
        row3.addWidget(self.image_dir_edit, 1)
        self.image_dir_btn = QtWidgets.QPushButton("Folder…")
        row3.addWidget(self.image_dir_btn)
        self.image_stem_label = QtWidgets.QLabel("Filename contains")
        row3.addWidget(self.image_stem_label)
        self.image_stem_edit = QtWidgets.QLineEdit()
        self.image_stem_edit.setMaximumWidth(140)
        self.image_stem_edit.setPlaceholderText("auto")
        self.image_stem_edit.setToolTip(
            "Optional filename substring. Leave blank for automatic matching "
            "from the SPEC file name and scan number.")
        row3.addWidget(self.image_stem_edit)
        self.raw_dot = QtWidgets.QLabel("○ raw unavailable")
        self._raw_status_tooltip = (
            "Whether the selected scan's first detector frame can be read. "
            "ROI plotting is enabled only when this says raw ready.")
        self.raw_dot.setToolTip(self._raw_status_tooltip)
        row3.addWidget(self.raw_dot)
        # Raw-params toggle shares the images row (its params expand below).
        self.adv_btn = QtWidgets.QPushButton("Raw params ▾")
        self.adv_btn.setCheckable(True)
        row3.addWidget(self.adv_btn)
        lay.addWidget(self.images_row)

        # Collapsible advanced raw-read params (expand below the row).
        self.adv_box = QtWidgets.QWidget()
        self.adv_box.setVisible(False)
        adv = QtWidgets.QHBoxLayout(self.adv_box)
        adv.setContentsMargins(2, 0, 2, 0)
        adv.setSpacing(5)
        self.det_rows = QtWidgets.QLineEdit()
        self.det_cols = QtWidgets.QLineEdit()
        for e in (self.det_rows, self.det_cols):
            e.setMaximumWidth(60)
            e.setPlaceholderText("auto")
        self.dtype_combo = QtWidgets.QComboBox()
        self.dtype_combo.addItems(_RAW_DTYPES)
        self.header_skip = QtWidgets.QLineEdit()
        self.header_skip.setMaximumWidth(60)
        self.header_skip.setPlaceholderText("0")
        adv.addWidget(QtWidgets.QLabel("shape"))
        adv.addWidget(self.det_rows)
        adv.addWidget(QtWidgets.QLabel("×"))
        adv.addWidget(self.det_cols)
        adv.addWidget(QtWidgets.QLabel("dtype"))
        adv.addWidget(self.dtype_combo)
        adv.addWidget(QtWidgets.QLabel("header"))
        adv.addWidget(self.header_skip)
        adv.addStretch(1)
        lay.addWidget(self.adv_box)

        # Row 5: grouping (stitch/RSM only).
        self.group_row = QtWidgets.QWidget()
        grp = QtWidgets.QHBoxLayout(self.group_row)
        grp.setContentsMargins(0, 0, 0, 0)
        grp.setSpacing(6)
        grp.addWidget(QtWidgets.QLabel("Group"))
        self.group_edit = QtWidgets.QLineEdit()
        self.group_edit.setPlaceholderText("e.g. 1-3, 5, 7-9  (combine into one output)")
        grp.addWidget(self.group_edit, 1)
        lay.addWidget(self.group_row)
        self.group_row.setVisible(self._allow_grouping)

        self.choose_btn.clicked.connect(self._choose)
        self.dir_check.toggled.connect(self._on_mode_toggled)
        self.dir_kind_combo.currentIndexChanged.connect(self._refresh_candidates)
        self.scan_combo.currentIndexChanged.connect(self._emit_selection)
        self.image_dir_btn.clicked.connect(self._choose_image_dir)
        self.image_dir_edit.editingFinished.connect(self._emit_selection)
        self.image_stem_edit.editingFinished.connect(self._emit_selection)
        for w in (self.det_rows, self.det_cols, self.header_skip):
            w.editingFinished.connect(self._emit_selection)
        self.dtype_combo.currentIndexChanged.connect(self._emit_selection)
        self.adv_btn.toggled.connect(self._on_adv_toggled)

    def _on_adv_toggled(self, on):
        self.adv_box.setVisible(on)
        self.adv_btn.setText("Raw params ▴" if on else "Raw params ▾")

    def _on_mode_toggled(self, _on):
        self.dir_kind_combo.setVisible(self.dir_check.isChecked())
        self.kind_label.setVisible(not self.dir_check.isChecked())
        self.kind_label.setText("")            # stale until a file is chosen
        self.path_edit.clear()
        self._set_candidates([])

    # ---- picking --------------------------------------------------------
    def _choose(self):
        from xdart.utils.browse import browse_start_dir, remember_browse_path
        start = browse_start_dir(self.path_edit.text())
        if self.dir_check.isChecked():
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Choose a folder", start)
        else:
            # All files first/default — SPEC scan files are extensionless.
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Choose a scan", start,
                "All files (*);;Scans (*.nxs *.h5 *.hdf5 *.cxi *.tif *.tiff *.raw)")
        if path:
            remember_browse_path(path)
            self.path_edit.setText(path)
            self._refresh_candidates()

    def _choose_image_dir(self):
        from xdart.utils.browse import browse_start_dir, remember_browse_path
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose image folder",
            browse_start_dir(self.image_dir_edit.text()))
        if path:
            remember_browse_path(path)
            self.image_dir_edit.setText(path)
            self._emit_selection()

    # ---- candidate scans ------------------------------------------------
    def _refresh_candidates(self):
        """(Re)compute the candidate scan specs for the current path/mode and
        repopulate the Scan selector."""
        path = self.path_edit.text().strip()
        if not path:
            self._set_candidates([])
            return
        from xrd_tools.core.scan import SourceKind, SourceSpec
        from xrd_tools.sources import discover_scans, guess_source_kind
        try:
            if self.dir_check.isChecked():
                kind = _DIR_KINDS[self.dir_kind_combo.currentIndex()][1]
                specs = discover_scans(path, kind)
            else:
                kind = guess_source_kind(path)
                specs = self._file_candidates(path, kind, SourceKind, SourceSpec)
                # Show the RESOLVED kind (a picked TIFF resolves to a series), so
                # it's clear when one file pulls in the whole scan's frames.
                shown = specs[0].kind if specs else kind
                self.kind_label.setText(shown.value if hasattr(shown, "value") else str(shown))
        except Exception:
            logger.exception("scan-source: could not enumerate scans for %s", path)
            specs = []
        self._set_candidates(specs)

    @staticmethod
    def _file_candidates(path, kind, SourceKind, SourceSpec):
        """File mode → one spec per scan (SPEC) / per entry (NeXus) / else one."""
        if kind is SourceKind.SPEC:
            from xrd_tools.io.spec import list_spec_scans
            scans = list_spec_scans(path)
            return [SourceSpec(path, SourceKind.SPEC, options={"scan": s})
                    for s in scans] or [SourceSpec(path, SourceKind.SPEC)]
        if (kind is SourceKind.IMAGE_FILE
                and Path(path).suffix.lower() in _SERIES_SUFFIXES):
            # A picked TIFF/RAW means THIS scan's series (sidecars → metadata), not
            # the whole folder — which may hold several scans.  Filter the folder
            # glob to the picked file's scan stem = its name minus a trailing
            # _<frame number>; with no such suffix, fall back to the whole folder.
            p = Path(path)
            m = re.match(r"(.*?_)\d+$", p.stem)
            options = {"pattern": f"{m.group(1)}*"} if m else {}
            if p.suffix.lower() == ".raw":
                options["metadata_format"] = "auto"
            return [SourceSpec(p.parent, SourceKind.TIFF_SERIES, options=options)]
        if kind in (SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER,
                    SourceKind.PROCESSED_NEXUS):
            from xrd_tools.io.nexus import list_entries
            try:
                entries = list(list_entries(path))
            except Exception:
                entries = []
            if len(entries) > 1:                 # multi-entry NeXus → entry selector
                return [SourceSpec(path, kind, entry=e) for e in entries]
            return [SourceSpec(path, kind, entry=(entries[0] if entries else None))]
        return [SourceSpec(path, kind)]

    def _set_candidates(self, specs):
        self._cancel_pending_probe()
        self._candidates = list(specs)
        self.scan_combo.blockSignals(True)
        self.scan_combo.clear()
        for spec in self._candidates:
            self.scan_combo.addItem(self._candidate_label(spec))
        self.scan_combo.blockSignals(False)
        multi = len(self._candidates) > 1
        self.scan_label.setVisible(multi)
        self.scan_combo.setVisible(multi)
        # SPEC pairs external images; other kinds carry images inline → hide the
        # images folder field (the dot still reflects reachability).
        self._update_images_visibility()
        self._emit_selection()

    @staticmethod
    def _candidate_label(spec):
        opts = dict(getattr(spec, "options", {}) or {})
        tag = opts.get("scan") or getattr(spec, "entry", None)
        base = Path(str(spec.uri)).name
        return f"{base} [{tag}]" if tag else base

    def _update_images_visibility(self):
        """Adapt the images row to the candidate kind: SPEC pairs an EXTERNAL
        image folder; a processed NeXus reuses the field as a 'Repoint raw'
        (``source_root``) for a moved tree; other kinds carry images inline (the
        dot still reflects reachability)."""
        from xrd_tools.core.scan import SourceKind
        spec = self._current_candidate()
        kind = spec.kind if spec is not None else None
        is_spec = kind is SourceKind.SPEC
        is_proc = kind is SourceKind.PROCESSED_NEXUS
        show_pairing = is_spec or is_proc
        self.images_label.setText(
            "Raw data root" if is_proc else "Raw image folder")
        self.image_dir_edit.setPlaceholderText(
            "(raw tree root, if the data moved)" if is_proc
            else "(blank = automatic matching next to the SPEC file)")
        for widget in (self.images_label, self.image_dir_edit, self.image_dir_btn):
            widget.setVisible(show_pairing)
        self.image_stem_label.setVisible(is_spec)
        self.image_stem_edit.setVisible(is_spec)
        self.image_dir_edit.setEnabled(show_pairing)
        self.image_dir_btn.setEnabled(show_pairing)
        self.image_stem_edit.setEnabled(is_spec)
        if self._selected_is_binary_raw() and not self.adv_btn.isChecked():
            self.adv_btn.setChecked(True)

    def _selected_is_binary_raw(self):
        """Whether the current direct image selection is a headerless RAW file."""
        if Path(self.path_edit.text().strip()).suffix.lower() == ".raw":
            return True
        spec = self._current_candidate()
        options = dict(getattr(spec, "options", {}) or {}) if spec else {}
        return str(options.get("pattern", "")).lower().endswith(".raw")

    def _raw_unavailable_hint(self):
        """Return an operator-facing reason why the raw probe cannot run."""
        from xrd_tools.core.scan import SourceKind

        if (self._selected_is_binary_raw()
                and "detector_shape" not in self._read_image_kwargs()
                and not self._raw_shape_is_inferable(
                    [Path(self.path_edit.text().strip())])):
            return (
                "○ enter RAW shape",
                "Open Raw params and enter detector rows, columns, and dtype. "
                "These values are required to decode headerless RAW frames.",
            )

        spec = self._current_candidate()
        if spec is None or spec.kind is not SourceKind.SPEC:
            return None
        options = dict(getattr(spec, "options", {}) or {})
        scan = options.get("scan")
        if not scan:
            return None
        directory_text = self.image_dir_edit.text().strip()
        directory = (Path(directory_text) if directory_text
                     else Path(str(spec.uri)).parent)
        stem = self.image_stem_edit.text().strip()
        if not stem:
            scan_number = str(scan).split(".")[0]
            stem = f"{Path(str(spec.uri)).stem}_scan{scan_number}_"
        if not directory.is_dir():
            return (
                "○ image folder missing",
                f"Raw image folder does not exist: {directory}",
            )
        try:
            from xrd_tools.io.image import find_image_files
            files = find_image_files(directory, stem=stem)
        except Exception:
            return None
        if not files:
            return (
                "○ no matching images",
                f"No detector files containing {stem!r} were found in "
                f"{directory} for the selected SPEC scan.",
            )
        raw_files = [path for path in files if path.suffix.lower() == ".raw"]
        if (raw_files and "detector_shape" not in self._read_image_kwargs()
                and not self._raw_shape_is_inferable(raw_files)):
            return (
                "○ enter RAW shape",
                "Matching headerless RAW frames were found. Open Raw params "
                "and enter detector rows, columns, and dtype to decode them.",
            )
        return None

    def _raw_shape_is_inferable(self, files):
        """Whether the first matching RAW frame has one exact known layout."""
        if not files:
            return False
        kwargs = self._read_image_kwargs()
        try:
            from xrd_tools.io.image import infer_raw_detector_shape
            return infer_raw_detector_shape(
                files[0],
                raw_dtype=kwargs.get("raw_dtype", "int32"),
                raw_header_skip=kwargs.get("raw_header_skip", 0),
            ) is not None
        except (OSError, TypeError, ValueError):
            return False

    def _current_candidate(self):
        i = self.scan_combo.currentIndex()
        if 0 <= i < len(self._candidates):
            return self._candidates[i]
        return self._candidates[0] if self._candidates else None

    # ---- read params + emit --------------------------------------------
    def _read_image_kwargs(self):
        out = {"raw_dtype": self.dtype_combo.currentText()}
        try:
            r = int(self.det_rows.text()) if self.det_rows.text().strip() else None
            c = int(self.det_cols.text()) if self.det_cols.text().strip() else None
        except ValueError:
            r = c = None
        if r and c:
            out["detector_shape"] = (r, c)
        try:
            skip = int(self.header_skip.text()) if self.header_skip.text().strip() else 0
        except ValueError:
            skip = 0
        if skip:
            out["raw_header_skip"] = skip
        return out

    def _build_spec(self):
        """The current candidate spec augmented with the images + raw-param
        fields, or None."""
        from xrd_tools.core.scan import SourceKind, SourceSpec
        spec = self._current_candidate()
        if spec is None:
            return None
        options = dict(getattr(spec, "options", {}) or {})
        if spec.kind is SourceKind.SPEC:
            image_dir = self.image_dir_edit.text().strip()
            stem = self.image_stem_edit.text().strip()
            if not image_dir:                       # auto: images next to the spec
                auto = self._auto_image_dir(spec, options, stem)
                if auto:
                    image_dir, stem = auto[0], (stem or auto[1])
            if image_dir:
                options["image_dir"] = image_dir
                if stem:
                    options["image_stem"] = stem
                rk = self._read_image_kwargs()
                if rk:
                    options["read_image_kwargs"] = rk
        elif spec.kind is SourceKind.PROCESSED_NEXUS:
            root = self.image_dir_edit.text().strip()
            if root:
                options["source_root"] = root       # repoint a moved raw tree
        elif spec.kind in (SourceKind.IMAGE_FILE, SourceKind.TIFF_SERIES):
            options.update(self._read_image_kwargs())
        return SourceSpec(spec.uri, spec.kind, entry=getattr(spec, "entry", None),
                          options=options)

    @staticmethod
    def _auto_image_dir(spec, options, stem):
        """SPEC auto-image-folder (design §3): when no folder is typed, try the
        spec file's own directory with the default stem; returns ``(dir, stem)``
        if matching images exist, else None."""
        scan = options.get("scan")
        if not scan:
            return None
        from xrd_tools.io.image import find_image_files
        parent = Path(str(spec.uri)).parent
        auto_stem = stem or f"{Path(str(spec.uri)).stem}_scan{str(scan).split('.')[0]}_"
        try:
            if find_image_files(parent, stem=auto_stem):
                return str(parent), auto_stem
        except Exception:
            pass
        return None

    @staticmethod
    def _spec_signature(spec):
        opts = dict(getattr(spec, "options", {}) or {})
        return (str(spec.uri), str(spec.kind), str(getattr(spec, "entry", None)),
                repr(sorted((k, repr(v)) for k, v in opts.items())))

    def _emit_selection(self):
        spec = self._build_spec()
        if spec is None:
            self._cancel_pending_probe()
            self._last_sig = self._last_selection = None
            self._set_dot(False)
            self.sigSourceChanged.emit(None)
            return
        sig = self._spec_signature(spec)
        if sig == self._last_sig and self._last_selection is not None:
            # unchanged spec — re-emit the cached selection without re-opening or
            # re-decoding a (possibly multi-MB Eiger) frame.
            self.sigSourceChanged.emit(self._last_selection)
            return
        if self._async_probe:
            self._start_async_probe(spec, sig)
            return
        self._finish_sync_probe(spec, sig)

    def _finish_sync_probe(self, spec, sig):
        try:
            reachable, first_image = self._probe_source(spec)
        except Exception:
            logger.exception("scan-source: open_source failed for %s", spec.uri)
            self._last_sig = self._last_selection = None
            self._set_dot(False)
            self.sigSourceChanged.emit(None)
            return
        self._set_dot(reachable)
        selection = ScanSelection(
            spec=spec, label=self._candidate_label(spec),
            reachable=reachable, first_image=first_image)
        self._last_sig, self._last_selection = sig, selection
        self.sigSourceChanged.emit(selection)

    @staticmethod
    def _probe_source(spec):
        """VALUE-ONLY probe (H19 §3): open the source, probe the first frame,
        then CLOSE the source inside this scope so no ``FrameSource`` / file
        handle escapes to the Qt thread or survives the worker.  Returns
        ``(reachable, first_image)`` where ``first_image`` is an immutable
        :class:`ImagePreview`, so no mutable ndarray crosses the worker boundary
        and the payload stays valid after the source is closed."""
        from xrd_tools.sources import open_source

        source = open_source(spec)
        try:
            reachable, first_image = probe_first_frame(source)
            if first_image is not None:
                first_image = ImagePreview.from_array(first_image)
        finally:
            closer = getattr(source, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    logger.debug(
                        "scan-source: source close failed", exc_info=True)
        return reachable, first_image

    def _start_async_probe(self, spec, sig):
        self._probe_generation += 1
        gen = self._probe_generation
        self._pending_sig = sig
        self._set_dot(False, text="probing…")
        if self._probe_executor is None:
            self._probe_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="xdart-source-probe",
            )

        def _work():
            try:
                reachable, first_image = self._probe_source(spec)
                return gen, sig, spec, reachable, first_image, None
            except Exception as exc:  # pragma: no cover - logged in GUI path
                return gen, sig, spec, False, None, exc

        future = self._probe_executor.submit(_work)

        def _emit_done(fut):
            try:
                result = fut.result()
            except CancelledError:
                return
            except Exception as exc:  # pragma: no cover - defensive callback guard
                result = (gen, sig, spec, False, None, exc)
            try:
                self.sigProbeDone.emit(result)
            except RuntimeError:
                # The widget may have been deleted while an off-thread probe was
                # unwinding.  The generation token already invalidates the work;
                # this guard prevents a teardown-only warning/crash.
                return

        future.add_done_callback(_emit_done)

    def _on_probe_done(self, result):
        gen, sig, spec, reachable, first_image, exc = result
        if gen != self._probe_generation or sig != self._pending_sig:
            return
        self._pending_sig = None
        if exc is not None:
            logger.warning(
                "scan-source: open_source failed for %s",
                spec.uri,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            self._last_sig = self._last_selection = None
            self._set_dot(False)
            self.sigSourceChanged.emit(None)
            return
        self._set_dot(reachable)
        selection = ScanSelection(
            spec=spec, label=self._candidate_label(spec),
            reachable=reachable, first_image=first_image)
        self._last_sig, self._last_selection = sig, selection
        self.sigSourceChanged.emit(selection)

    def _cancel_pending_probe(self):
        self._probe_generation += 1
        self._pending_sig = None

    def _set_dot(self, reachable, *, text=None):
        hint = self._raw_unavailable_hint() if text is None and not reachable else None
        if hint is not None:
            text, tooltip = hint
            self.raw_dot.setToolTip(tooltip)
        else:
            self.raw_dot.setToolTip(self._raw_status_tooltip)
        self.raw_dot.setText(
            text or ("● raw ready" if reachable else "○ raw unavailable"))
        self.raw_dot.setStyleSheet(
            "color: #50fa7b;" if reachable else "color: #888;")

    def shutdown_probe_worker(self):
        executor = self._probe_executor
        self._probe_executor = None
        self._cancel_pending_probe()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        timer = self._directory_timer
        if timer is not None:
            timer.stop()
        session = self._directory_session
        self._directory_session = None
        if session is not None:
            session.close()

    def closeEvent(self, event):
        self.shutdown_probe_worker()
        super().closeEvent(event)

    # ---- persistent Controls-V2 directory index -----------------------
    @property
    def directory_session(self):
        return self._directory_session

    @property
    def directory_observation(self):
        return self._directory_observation

    @property
    def directory_subdirs_lazy(self):
        return self._directory_subdirs_lazy

    def configure_directory(
        self, root, *, recursive=False, name_filter=None, suffixes=(),
        subdirs_lazy=False,
    ):
        if not self._controls_source or self._directory_session is None:
            raise RuntimeError("widget is not in controls_source mode")
        previous_generation = self._directory_session.request_generation
        # Recursive intent is presentation-only here.  Even when the operator
        # selected Subdirs, Source status looks at direct children only.
        self._directory_subdirs_lazy = bool(subdirs_lazy or recursive)
        generation = self._directory_session.configure(
            root, recursive=False, name_filter=name_filter,
            suffixes=suffixes)
        if generation != previous_generation:
            self._directory_observation = None
            self._directory_signature = None
            self.directory_status.setText("Checking directory…")
        self.request_directory_poll()
        return generation

    def clear_directory(self):
        if self._directory_session is None:
            return
        self._directory_session.clear()
        self._directory_observation = None
        self._directory_signature = None
        self._directory_subdirs_lazy = False
        self.directory_status.setText("No container-directory source")
        self.sigDirectoryChanged.emit(None)

    def request_directory_poll(self, *, refresh=True):
        session = self._directory_session
        if session is None or session.configured is None:
            return
        future = self._directory_future
        if future is not None and not future.done():
            return
        future = session.observe_async(refresh=bool(refresh))
        self._directory_future = future
        self._directory_request_token += 1
        request_token = self._directory_request_token

        def _emit_done(done):
            try:
                result = done.result()
            except CancelledError:
                return
            except Exception as exc:
                result = exc
            try:
                self.sigDirectoryDone.emit((request_token, result))
            except RuntimeError:
                return

        future.add_done_callback(_emit_done)

    def _on_directory_done(self, payload):
        if (
            isinstance(payload, tuple)
            and len(payload) == 2
            and isinstance(payload[0], int)
        ):
            request_token, result = payload
            if request_token != self._directory_request_token:
                # A done future can queue its Qt signal, then a newer request
                # can start before that queued delivery runs.  The old
                # completion must not clear or overwrite the newer owner.
                return
        else:
            # Compatibility for direct unit-test calls and older signal users.
            result = payload
        self._directory_future = None
        if isinstance(result, Exception):
            logger.warning("source-card directory observation failed: %s", result)
            self.directory_status.setText("Directory temporarily unavailable")
            return
        session = self._directory_session
        if session is None:
            return
        if result.request_generation != session.request_generation:
            # Configuration changed while the serialized owner was polling.
            # Do not wait for the periodic timer; immediately observe the
            # latest requested root/filter generation.
            self.request_directory_poll()
            return
        signature = (
            result.request_generation,
            result.discovered_snapshot.generation,
            tuple(
                (str(item.candidate.path), item.candidate.version_stamp,
                 item.candidate.adapter_id, item.result.state.value)
                for item in result.candidates
            ),
            int(result.unprobed_count),
            int(result.stale_drops),
        )
        changed = signature != self._directory_signature
        self._directory_signature = signature
        self._directory_observation = result
        count = len(result.discovered_snapshot.candidates)
        noun = "file" if count == 1 else "files"
        parts = [f"{count} matching {noun} in this folder"]
        if self._directory_subdirs_lazy:
            parts.append("subfolders processed during Run")
        self.directory_status.setText(" · ".join(parts))
        if changed:
            self.sigDirectoryChanged.emit(result)

    # ---- public API for consumers --------------------------------------
    def set_uri(self, uri):
        """Programmatically load a file path (e.g. the dialog's default scan)."""
        if not uri:
            return
        if self._controls_source:
            self.configure_directory(uri)
            return
        self.dir_check.setChecked(False)
        self.path_edit.setText(str(uri))
        self._refresh_candidates()

    def scan_groups(self):
        """The grouping field parsed into groups (stitch/RSM), or None."""
        text = self.group_edit.text().strip()
        if not (self._allow_grouping and text):
            return None
        from xrd_tools.sources import parse_scan_groups
        try:
            return parse_scan_groups(text)
        except ValueError:
            return None
