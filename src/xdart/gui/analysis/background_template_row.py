# -*- coding: utf-8 -*-
"""The background-template row: one shared owner for the Peak Fitter's and Phase Analysis's template
(NEXT-128-PEAK-FIT).

The maintainer, 2026-10-30: "The background option can be used in peak fitter as well" and "the background template
is read from a file ... This could be a Nexus file with 1d data or perhaps a xye file". The row holds one template:

- "Use displayed": the host captures the displayed trace and, when it can, names the file, frame and 1-D mode behind
  it (a candidate). The first fit checks that the file reproduces the trace: then the row becomes that named source;
  otherwise the capture stays usable but unnamed (not saved, not exported), and the row says why.
- "Choose file…": a NeXus result (its frame and 1-D mode, refused at fit time when missing) or an .xye file (its x
  unit, which a file name only hints at). The first completed fit records the file's content identity (``adopt``);
  export waits for it, and a file changed since then is refused by name until it is chosen again.
- "Clear" forgets the template.

The headless owner reads the files (``xrd_tools.analysis.background_template``), on the host's fit worker; this row
never reads one (a restored setting is checked with a stat). It builds no editor of its own (hard rule 14): its rows
are the shared ``FormRow`` and ``ActionButton``, its status a label.
"""
from __future__ import annotations

from pathlib import Path

from pyqtgraph.Qt import QtCore, QtWidgets

from xrd_tools.analysis.background_template import (
    NEXUS_MODES, XYE_UNITS, BackgroundTemplateRefused, BackgroundTemplateSource, is_template_identity,
    template_file_problem, xye_unit_hint,
)
from xrd_tools.session.readiness import ControlAction, ControlActionSpec, SectionId

from xdart.gui.widgets.controls_panel import ActionButton, FormRow

#: the file suffixes read as NeXus results; any other file is read as .xye columns
_NEXUS_SUFFIXES = frozenset({".nxs", ".nexus", ".h5", ".hdf5"})
_EXPORT = "BACKGROUND_TEMPLATE_NOTEBOOK_UNAVAILABLE"


def _path(name):
    return ("background_template", name)


class BackgroundTemplateRow(QtWidgets.QWidget):
    """The template row; ``what`` names the template in its texts ("template", Phase Analysis's "substrate").

    ``changed`` is emitted after every change of the template, its source, identity or reason. The host's "Use
    displayed" is the dialog's (``capture_btn``); a host that can name the displayed trace's file installs
    ``describe_displayed`` (returning ``(path, frame, mode)`` or None), which pre-fills a chosen file's frame and mode
    when it is that file."""

    changed = QtCore.Signal()

    def __init__(self, parent=None, *, what="template", capture_label="Use displayed", clear_label="Clear"):
        super().__init__(parent)
        self.setObjectName("backgroundTemplateRow")
        self._what = what
        self._template = None   # a captured DisplayedTraceInput
        self._source = None     # a BackgroundTemplateSource: chosen, restored, bound, or a capture's candidate
        self._identity = None   # the content identity a completed fit recorded
        self._reason = ""       # why the template cannot be named, read or exported now
        self._file = None       # (kind, path) of a chosen or bound file, whose rows rebuild its source
        self.describe_displayed = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.status = QtWidgets.QLabel("", self)
        self.status.setObjectName("backgroundTemplateStatus")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self._buttons = QtWidgets.QHBoxLayout()
        self._buttons.setContentsMargins(0, 0, 0, 0)
        self.capture_btn = ActionButton(ControlActionSpec(ControlAction.PREVIEW, capture_label, SectionId.PROCESSING),
                                        self)
        self.capture_btn.set_purpose_tooltip(
            f"Capture the displayed trace as the {what}: show the bare {what}'s pattern first. Its file, frame and "
            "mode name it once the first fit confirms that file holds this curve; until then it is not saved or "
            "exported.")
        self.choose_btn = ActionButton(ControlActionSpec(ControlAction.PREVIEW, "Choose file…", SectionId.PROCESSING),
                                       self)
        self.choose_btn.set_purpose_tooltip(
            f"Read the {what} from a file: a processed NeXus result (its frame and 1-D mode below) or an .xye file "
            "(its x unit below). The first fit records the file's content; it is then saved with the settings and "
            "exported to notebooks.")
        self.clear_btn = ActionButton(ControlActionSpec(ControlAction.PREVIEW, clear_label, SectionId.PROCESSING),
                                      self)
        self.clear_btn.set_purpose_tooltip(f"Forget the {what}.")
        for button in (self.capture_btn, self.choose_btn, self.clear_btn):
            button.setAutoDefault(False)
            self._buttons.addWidget(button)
        self._buttons.addStretch(1)
        layout.addLayout(self._buttons)
        self.unit_row = FormRow(label="x unit", path=_path("unit"), value="", kind="combo",
                                choices=("", *(unit for unit, _label in XYE_UNITS)),
                                choice_labels={"": "Choose the unit…", **dict(XYE_UNITS)})
        self.unit_row.set_purpose_tooltip(
            "The .xye file's x unit. A file name's xdart prefix (iq_, itth_, iqip_, iqoop_, iexit_) only suggests "
            "it: confirm it. It must be the fit's unit, or convert to it without a wavelength (Å⁻¹ and nm⁻¹; "
            "degrees and radians).")
        self.frame_row = FormRow(label="Frame", path=_path("frame"), value="", kind="int", minimum=0,
                                 maximum=1_000_000_000, allow_blank=True)
        self.frame_row.set_purpose_tooltip(
            "The NeXus result's frame label to read; a frame the file does not hold is refused at the fit.")
        self.mode_row = FormRow(label="1-D mode", path=_path("mode"), value=NEXUS_MODES[0], kind="combo",
                                choices=NEXUS_MODES)
        self.mode_row.set_purpose_tooltip(
            "The NeXus result's 1-D curve: default for a single result; q_total, q_ip, q_oop, exit_angle or chi_gi "
            "for grazing incidence. A mode the file does not hold is refused at the fit, naming the file's modes.")
        for row in (self.unit_row, self.frame_row, self.mode_row):
            layout.addWidget(row)
        self.capture_btn.setObjectName("backgroundTemplateCapture")
        self.choose_btn.clicked.connect(self._choose)
        self.clear_btn.clicked.connect(self.clear)
        self.unit_row.editor.currentIndexChanged.connect(lambda _index: self._edited())
        self.mode_row.editor.currentIndexChanged.connect(lambda _index: self._edited())
        self.frame_row.valueChanged.connect(lambda _path, _value: self._edited())
        self._sync()

    # ---- the row's state -------------------------------------------------
    @property
    def template(self):
        """The captured displayed trace (a ``DisplayedTraceInput``) while it is not named by its file, or None."""
        return self._template

    @property
    def source(self):
        """The ``BackgroundTemplateSource`` (chosen, restored, bound, or a capture's candidate), or None."""
        return self._source

    @property
    def identity(self):
        """The content identity the first completed fit recorded for ``source``, or None."""
        return self._identity

    @property
    def reason(self):
        """Why the template cannot be named, read or exported now ("" when nothing is wrong)."""
        return self._reason

    def has_template(self) -> bool:
        return self._template is not None or self._source is not None

    def values(self) -> dict:
        """The fit plan's template fields."""
        return {"background_template": self._template,
                "background_template_source": None if self._source is None else self._source.to_pairs(),
                "background_template_identity": self._identity}

    def fit_refusal(self, missing: str = "") -> str:
        """Why a template background cannot fit now ("" when it can): a chosen file that still needs its unit or
        frame, or no template (``missing``, else a default message)."""
        if self.has_template():
            return ""
        if self._file is not None and self._reason:
            return self._reason
        return missing or (f"The template backgrounds need a {self._what}: use the displayed trace or choose a NeXus "
                           "or .xye file.")

    def export_reason(self) -> str:
        """Why a notebook cannot carry the template ("" when it can: a file-named source whose content a fit
        recorded)."""
        if self._template is None and self._source is None and self._file is None:
            return ""
        why = self._reason or (
            f"the displayed {self._what} is not named by its file yet; fit once so that file is checked"
            if self._template is not None else
            f"fit once so the {self._what} file's content is recorded" if self._identity is None else "")
        return f"{_EXPORT}: {why}" if why else ""

    def section(self):
        """The settings section: the named file and its identity (None when none is named: a captured trace is
        not saved)."""
        if self._source is None or self._template is not None:
            return None
        return {"source": dict(self._source.to_pairs()), "identity": self._identity}

    @staticmethod
    def parse_section(section):
        """A settings section as ``(source, identity)``, or None for none; ValueError says why it is malformed, before
        anything applies."""
        if section is None:
            return None
        if type(section) is not dict or "source" not in section:
            raise ValueError("the background template settings are malformed")
        source = BackgroundTemplateSource.from_pairs(section["source"])
        identity = section.get("identity")
        if identity is not None and not is_template_identity(identity):
            raise ValueError("the background template's recorded identity is malformed")
        return source, identity

    def restore(self, parsed) -> None:
        """Restore a parsed section (``parse_section``; None clears the row), reading no file: a missing one is
        named."""
        if parsed is None:
            self.clear()
            return
        source, identity = parsed
        self._template, self._source, self._identity = None, source, identity
        self._file = (source.kind, source.path)
        self._reason = template_file_problem(source)
        self._sync()

    def state(self):
        """Everything the row holds, for a settings load that must roll back."""
        return (self._template, self._source, self._identity, self._reason, self._file)

    def set_state(self, state) -> None:
        self._template, self._source, self._identity, self._reason, self._file = state
        self._sync()

    # ---- what the host and the fits change ---------------------------------
    def set_capture(self, trace, candidate=None, reason="") -> None:
        """The host's capture of the displayed trace, and the file, frame and mode it names (``candidate``, a
        ``BackgroundTemplateSource``) or why it names none (``reason``)."""
        self._template, self._source, self._identity, self._file = trace, candidate, None, None
        self._reason = "" if candidate is not None else (
            reason or f"the displayed {self._what} has no NeXus file and frame behind it")
        self._sync()

    def clear(self) -> None:
        self._template = self._source = self._identity = self._file = None
        self._reason = ""
        self._sync()

    def adopt(self, result) -> bool:
        """A completed fit's template echo: it records the identity the fit read for the row's current source, so
        export can carry it; a capture its file reproduced becomes that named source, one it did not stays unnamed.
        A result for another source changes nothing."""
        echo = getattr(result, "background_template_source", None)
        if self._source is None or echo is None or tuple(echo) != self._source.to_pairs():
            return False
        identity = getattr(result, "background_template_identity", "")
        if identity:
            self._identity, self._reason = identity, ""
            if self._template is not None:
                self._template = None
                self._file = (self._source.kind, self._source.path)
        elif self._template is not None:
            self._source = None
            self._reason = next((line for line in getattr(result, "diagnostics", ())
                                 if "BACKGROUND_TEMPLATE_UNREPRODUCIBLE" in line),
                                "BACKGROUND_TEMPLATE_UNREPRODUCIBLE: its file does not reproduce the captured trace")
        else:
            return False
        self._sync()
        return True

    def refuse(self, result) -> bool:
        """A fit refused for its template (a ``BACKGROUND_TEMPLATE_`` code) of the row's current template: the row
        says why."""
        code = str(getattr(result, "code", ""))
        echo = getattr(result, "background_template_source", None)
        current = None if self._source is None else self._source.to_pairs()
        if not code.startswith("BACKGROUND_TEMPLATE_") or (None if echo is None else tuple(echo)) != current:
            return False
        detail = next(iter(getattr(result, "diagnostics", ()) or ()), "")
        self._reason = detail if detail.startswith(code) else f"{code}: {detail}" if detail else code
        self._sync()
        return True

    # ---- choosing a file ---------------------------------------------------
    def _file_chooser(self):
        """The chosen template file ("" when cancelled), through the shared ``file_dialogs`` (tests replace it)."""
        from xdart.gui.widgets import file_dialogs
        picks = file_dialogs.browse(self, f"Choose the {self._what} file",
                                    file_filter=file_dialogs.BACKGROUND_TEMPLATE_FILTER,
                                    current="" if self._source is None else self._source.path)
        return picks[0] if picks else ""

    def _choose(self) -> None:
        path = self._file_chooser()
        if not path:
            return
        kind = "nexus" if Path(path).suffix.casefold() in _NEXUS_SUFFIXES else "xye"
        self._template, self._identity, self._file = None, None, (kind, str(path))
        with QtCore.QSignalBlocker(self.unit_row.editor), QtCore.QSignalBlocker(self.mode_row.editor):
            if kind == "xye":
                # a name never fixes the scale (Codex REVIEW-3 F1): xdart writes iq_ for Å⁻¹ and nm⁻¹ alike, so the
                # unit waits for the user's choice; the name's suggestion is in the reason
                self.unit_row.editor.setCurrentIndex(0)
            else:
                frame, mode = None, NEXUS_MODES[0]
                displayed = self.describe_displayed() if callable(self.describe_displayed) else None
                if displayed is not None and _same_file(displayed[0], path):
                    frame, mode = displayed[1], displayed[2]
                self.frame_row.editor.setValue(frame)
                index = self.mode_row.editor.findText(mode)
                self.mode_row.editor.setCurrentIndex(index if index >= 0 else 0)
        self._rebuild()

    def _edited(self) -> None:
        """A user's edit of the unit, frame or mode: a chosen file's source is rebuilt and must be fitted again."""
        if self._file is None or self._template is not None:
            return
        self._identity = None
        self._rebuild()

    def _rebuild(self) -> None:
        kind, path = self._file
        self._source, self._reason = None, ""
        try:
            if kind == "xye":
                unit = self.unit_row.current_value()
                if unit:
                    self._source = BackgroundTemplateSource("xye", path, unit=unit)
                else:
                    hint = xye_unit_hint(path)
                    suggestion = (f"its name suggests {dict(XYE_UNITS).get(hint, hint)}, but a name never fixes the "
                                  "scale (Å⁻¹ or nm⁻¹, degrees or radians)" if hint else
                                  "its name does not say it")
                    self._reason = f"BACKGROUND_TEMPLATE_UNIT_REQUIRED: choose the .xye file's x unit; {suggestion}"
            else:
                frame = self.frame_row.editor.value() if not self.frame_row.refusal() else None
                if frame is None:
                    self._reason = "BACKGROUND_TEMPLATE_INVALID_SOURCE: enter the frame to read from this NeXus file"
                else:
                    self._source = BackgroundTemplateSource("nexus", path, frame=frame,
                                                            mode=self.mode_row.current_value())
        except BackgroundTemplateRefused as refused:
            self._reason = str(refused)
        self._sync()

    # ---- the projection ----------------------------------------------------
    def describe(self) -> str:
        """The template as the row's status names it."""
        template, source = self._template, self._source
        if template is not None:
            name = f"{template.title} ({template.label})" if template.title else template.label
            if source is not None:
                return f"Displayed: {name}; its file is checked at the first fit: {source.describe()}"
            return f"Displayed: {name}; not named by a file, so it is not saved or exported"
        if source is not None:
            state = "recorded" if self._identity is not None else "read at the first fit"
            return f"{source.describe()} ({state})" + (f"; {self._reason}" if self._reason else "")
        if self._file is not None:
            return f"{Path(self._file[1]).name}: {self._reason}"
        return f"No {self._what}"

    def _sync(self) -> None:
        text = self.describe()
        self.status.setText(text)
        detail = self._reason or (self._source.path if self._source is not None else "")
        self.status.setToolTip(
            f"The {self._what} the template backgrounds scale. " + (f"{text}. {detail}" if detail else text))
        kind = self._file[0] if self._file is not None else (None if self._source is None else self._source.kind)
        pending = self._template is not None
        self.unit_row.setVisible(kind == "xye")
        self.frame_row.setVisible(kind == "nexus")
        self.mode_row.setVisible(kind == "nexus")
        source = self._source
        with QtCore.QSignalBlocker(self.unit_row.editor), QtCore.QSignalBlocker(self.mode_row.editor):
            if source is not None and source.kind == "xye":
                self.unit_row.editor.setCurrentIndex(max(0, self.unit_row.editor.findData(source.unit)))
            elif source is not None:
                self.frame_row.editor.setValue(source.frame)
                index = self.mode_row.editor.findText(source.mode)
                self.mode_row.editor.setCurrentIndex(index if index >= 0 else 0)
        for row in (self.unit_row, self.frame_row, self.mode_row):
            row.editor.setEnabled(not pending)
        present = self._template is not None or self._source is not None or self._file is not None
        spec = self.clear_btn.spec
        self.clear_btn.apply_spec(ControlActionSpec(spec.action, spec.label, spec.section, enabled=present,
                                                    reason="" if present else f"No {self._what} is set."))
        self.changed.emit()

    def set_help_topic(self, topic_id: str) -> None:
        """Bind F1 and a "?" to ``topic_id``, as the shared rows do."""
        from xdart.gui.widgets.context_help import bind_topic
        bind_topic(self, topic_id)
        if not hasattr(self, "help_button"):
            self.help_button = ActionButton.for_help(self, topic_id)
            self._buttons.addWidget(self.help_button)


def _same_file(first, second) -> bool:
    try:
        return Path(first).expanduser().resolve() == Path(second).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return str(first) == str(second)
