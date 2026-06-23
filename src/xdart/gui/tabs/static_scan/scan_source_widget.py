# -*- coding: utf-8 -*-
"""Shared scan-source widget (the §3 design of design_shared_source_panel).

One reusable, kind-general source picker — used by the ROI Scan Plotter now and
the stitch/RSM wrangler later.  It assembles a :class:`SourceSpec`, opens it, and
emits a :class:`ScanSelection` (spec + opened FrameSource + raw-reachable flag +
the first frame) via :data:`sigSourceChanged`.  All parsing/IO is headless
(`xrd_tools.sources` / `io`); this is the thin Qt layer.

Two entry modes (§2.2): a single master **File**, or a **Directory** + a
Scan-kind dropdown (`discover_scans`).  The Scan selector then lists the
candidate scans (SPEC scan numbers / NeXus entries / discovered scans).  An
optional **Images** folder (SPEC) pairs raw frames; the **raw-reachable dot**
(metadata-independent) gates ROI/stitch/RSM downstream.  Grouping (combine scans
via `CompositeFrameSource`) shows only for stitch/RSM.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

logger = logging.getLogger(__name__)

#: (label, SourceKind value) for the Directory-mode scan-kind dropdown.
_DIR_KINDS = [("SPEC", "spec"), ("TIFF / RAW series", "tiff_series"),
              ("NeXus (raw stack)", "nexus_stack"),
              ("Processed NeXus", "processed_nexus"), ("Eiger", "eiger_master")]
_TIFF_SUFFIXES = {".tif", ".tiff"}
_RAW_DTYPES = ["int32", "uint32", "int16", "uint16", "float32", "float64"]


@dataclass
class ScanSelection:
    """The widget's output: an opened, classified scan source."""

    spec: object               # SourceSpec | None
    source: object             # FrameSource | None
    label: str
    reachable: bool            # raw frames loadable (probe) — independent of metadata
    first_image: object        # ndarray | None (the probed frame; reused by ROI)


class ScanSourceWidget(QtWidgets.QWidget):
    """Pick a scan (any source kind) → emit a :class:`ScanSelection`."""

    #: emitted with a :class:`ScanSelection` (or ``None`` when cleared/invalid).
    sigSourceChanged = QtCore.Signal(object)

    def __init__(self, mode="roi", parent=None):
        super().__init__(parent)
        self._mode = mode
        self._allow_grouping = mode in ("stitch", "rsm")
        self._candidates = []          # list[SourceSpec] for the Scan selector
        self._last_sig = None          # signature of the last-opened spec (dedupe)
        self._last_selection = None
        self._build_ui()

    # ---- UI -------------------------------------------------------------
    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)

        # Row 1: entry mode + path + (kind label | scan-kind combo) + Choose.
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(6)
        self.dir_check = QtWidgets.QCheckBox("Folder")
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
        row3.addWidget(QtWidgets.QLabel("Root"))
        self.image_stem_edit = QtWidgets.QLineEdit()
        self.image_stem_edit.setMaximumWidth(140)
        self.image_stem_edit.setPlaceholderText("auto")
        self.image_stem_edit.setToolTip(
            "Filename substring selecting this scan's images (default "
            "{spec}_scan{N}_)")
        row3.addWidget(self.image_stem_edit)
        self.raw_dot = QtWidgets.QLabel("○ no raw")
        self.raw_dot.setToolTip("Whether raw frames are reachable (gates ROI)")
        row3.addWidget(self.raw_dot)
        lay.addWidget(self.images_row)

        # Row 4: advanced raw-read params (collapsible).
        self.adv_btn = QtWidgets.QPushButton("Raw params ▾")
        self.adv_btn.setCheckable(True)
        adv_row = QtWidgets.QHBoxLayout()
        adv_row.addWidget(self.adv_btn)
        adv_row.addStretch(1)
        lay.addLayout(adv_row)
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
        if self.dir_check.isChecked():
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose a folder")
        else:
            # All files first/default — SPEC scan files are extensionless.
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Choose a scan", "",
                "All files (*);;Scans (*.nxs *.h5 *.hdf5 *.cxi *.tif *.tiff)")
        if path:
            self.path_edit.setText(path)
            self._refresh_candidates()

    def _choose_image_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose image folder")
        if path:
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
                self.kind_label.setText(kind.value if hasattr(kind, "value") else str(kind))
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
                and Path(path).suffix.lower() in _TIFF_SUFFIXES):
            # a picked TIFF is the first of a folder series (sidecars → metadata)
            return [SourceSpec(Path(path).parent, SourceKind.TIFF_SERIES)]
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
        self.images_label.setText("Repoint raw" if is_proc else "Images")
        self.image_dir_edit.setPlaceholderText(
            "(raw tree root, if the data moved)" if is_proc
            else "(image folder — blank = auto / metadata only)")
        self.image_dir_edit.setEnabled(is_spec or is_proc)
        self.image_dir_btn.setEnabled(is_spec or is_proc)
        self.image_stem_edit.setEnabled(is_spec)        # filename root: SPEC only

    def _current_candidate(self):
        i = self.scan_combo.currentIndex()
        if 0 <= i < len(self._candidates):
            return self._candidates[i]
        return self._candidates[0] if self._candidates else None

    # ---- read params + emit --------------------------------------------
    def _read_image_kwargs(self):
        out = {}
        try:
            r = int(self.det_rows.text()) if self.det_rows.text().strip() else None
            c = int(self.det_cols.text()) if self.det_cols.text().strip() else None
        except ValueError:
            r = c = None
        if r and c:
            out["detector_shape"] = (r, c)
            out["raw_dtype"] = self.dtype_combo.currentText()   # only with a shape
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
        from xrd_tools.sources import open_source
        from .scan_plot_dialog import probe_first_frame
        try:
            source = open_source(spec)
        except Exception:
            logger.exception("scan-source: open_source failed for %s", spec.uri)
            self._last_sig = self._last_selection = None
            self._set_dot(False)
            self.sigSourceChanged.emit(None)
            return
        reachable, first_image = probe_first_frame(source)
        self._set_dot(reachable)
        selection = ScanSelection(
            spec=spec, source=source, label=self._candidate_label(spec),
            reachable=reachable, first_image=first_image)
        self._last_sig, self._last_selection = sig, selection
        self.sigSourceChanged.emit(selection)

    def _set_dot(self, reachable):
        self.raw_dot.setText("● raw" if reachable else "○ no raw")
        self.raw_dot.setStyleSheet(
            "color: #50fa7b;" if reachable else "color: #888;")

    # ---- public API for consumers --------------------------------------
    def set_uri(self, uri):
        """Programmatically load a file path (e.g. the dialog's default scan)."""
        if not uri:
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
