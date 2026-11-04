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

NEXT-128-WINDOW-POLISH (the maintainer, 2026-11-03: "the Use displayed/choose file etc options are for the template,
right? they should only be enabled if template is selected. if there is a message about the file selected it should
show up in a popup window rather than squeezing it in that space"; his "Same in both" and "Ask in the popup"): the host
says when its background uses the template (``set_available``) and when a batch holds its fit worker (``set_busy``);
an .xye file's unit, and a NeXus file's frame when the display does not name it, are asked in a popup (the shared
``AdvancedDialog``; the name's unit is only mentioned, never preselected); a message about the file opens once in a
popup and the status stays one short line, the whole message in its tooltip.

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


class CurveFileRow(QtWidgets.QWidget):
    """The file half shared by the template row and the fit windows' Source row (NEXT-128-WINDOW-POLISH item 11): a
    status line, a row of buttons ending in the "?", a NeXus file's Frame and 1-D mode rows and an .xye file's x unit
    row (the shared ``FormRow``), the file chooser (the shared ``file_dialogs``) and the popups that ask an .xye file's
    unit and a NeXus file's frame (the shared ``AdvancedDialog``) or show a message about a file. ``what`` names the
    curve in every text ("template", "substrate", "source"); ``filter_name`` is the chooser's ``file_dialogs`` filter."""

    def __init__(self, parent=None, *, what, filter_name="BACKGROUND_TEMPLATE_FILTER"):
        super().__init__(parent)
        self._what = what
        self._filter_name = filter_name
        self._announced = ""    # the last message shown in a popup
        self.message_box = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.status = QtWidgets.QLabel("", self)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self._buttons = QtWidgets.QHBoxLayout()
        self._buttons.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._buttons)

    def _curve_rows(self, root, *, unit_tip, frame_tip, mode_tip):
        """The x unit, Frame and 1-D mode rows under ``root`` (a settings path's first part), added to the layout."""
        unit_row = FormRow(label="x unit", path=(root, "unit"), value="", kind="combo",
                           choices=("", *(unit for unit, _label in XYE_UNITS)),
                           choice_labels={"": "Choose the unit…", **dict(XYE_UNITS)})
        unit_row.set_purpose_tooltip(unit_tip)
        frame_row = FormRow(label="Frame", path=(root, "frame"), value="", kind="int", minimum=0,
                            maximum=1_000_000_000, allow_blank=True)
        frame_row.set_purpose_tooltip(frame_tip)
        mode_row = FormRow(label="1-D mode", path=(root, "mode"), value=NEXUS_MODES[0], kind="combo",
                           choices=NEXUS_MODES)
        mode_row.set_purpose_tooltip(mode_tip)
        for row in (unit_row, frame_row, mode_row):
            self.layout().addWidget(row)
        return unit_row, frame_row, mode_row

    # ---- the file half's popups and chooser (shared) -----------------------------
    def _notify(self, headline: str, detail: str) -> None:
        """The popup: Qt's message box, as Int's page shows its notices, not modal; a newer message replaces it."""
        if self.message_box is not None:
            self.message_box.close()
            self.message_box.deleteLater()
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.NoIcon, f"The {self._what}", headline,
                                    QtWidgets.QMessageBox.StandardButton.Ok, self)
        box.setInformativeText(detail)
        box.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        box.show()
        self.message_box = box

    # ---- the popups that ask -----------------------------------------------------
    def _ask_unit(self, path: str, hint) -> str:
        """An .xye file's x unit, asked in a popup ("" when cancelled; tests replace it)."""
        dialog, unit_row = self._unit_dialog(path, hint)
        try:
            accepted = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
            return unit_row.current_value() if accepted else ""
        finally:
            dialog.deleteLater()

    def _unit_dialog(self, path: str, hint):
        """The unit popup: the shared Advanced dialog with OK and Cancel, a line naming the file and what its name
        suggests, and the unit row, chosen by the user (OK waits for it)."""
        name = Path(path).name
        suggestion = (f"Its name suggests {dict(XYE_UNITS).get(hint, hint)}, but xdart writes the same prefix for Å⁻¹ "
                      "and nm⁻¹ (and for degrees and radians), so the name cannot decide." if hint else
                      "Its name does not say.")
        note = QtWidgets.QLabel(f"Which unit is the x column of {name} in?\n{suggestion}")
        note.setWordWrap(True)
        note.setToolTip("An .xye file records no unit: it must be the fit's, or convert to it without a wavelength.")
        unit_row = FormRow(label="x unit", path=_path("unit"), value="", kind="combo",
                           choices=("", *(unit for unit, _label in XYE_UNITS)),
                           choice_labels={"": "Choose the unit…", **dict(XYE_UNITS)})
        unit_row.set_purpose_tooltip("The unit of the file's x column.")
        return self._asking_dialog(f"The {self._what}'s x unit", (note, unit_row),
                                   lambda: bool(unit_row.current_value())), unit_row

    def _ask_frame(self, path: str, frame, mode):
        """A NeXus file's frame and 1-D mode, asked in a popup: ``(frame, mode)``, or None when cancelled."""
        dialog, frame_row, mode_row = self._frame_dialog(path, frame, mode)
        try:
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return None
            return frame_row.editor.value(), mode_row.current_value()
        finally:
            dialog.deleteLater()

    def _frame_dialog(self, path: str, frame, mode):
        note = QtWidgets.QLabel(f"Which frame and 1-D curve of {Path(path).name} is the {self._what}?")
        note.setWordWrap(True)
        note.setToolTip("A processed NeXus result holds many frames; the display names one only when it shows this "
                        "file.")
        frame_row = FormRow(label="Frame", path=_path("frame"), value="" if frame is None else str(frame), kind="int",
                            minimum=0, maximum=1_000_000_000)
        frame_row.set_purpose_tooltip("The frame label to read.")
        mode_row = FormRow(label="1-D mode", path=_path("mode"), value=mode, kind="combo", choices=NEXUS_MODES)
        mode_row.set_purpose_tooltip("The 1-D curve to read: default for a single result, or a grazing-incidence "
                                     "mode.")
        dialog = self._asking_dialog(f"The {self._what}'s frame", (note, frame_row, mode_row),
                                     lambda: not frame_row.refusal())
        frame_row.editor.textChanged.connect(lambda _text: self._ok_follows(dialog, lambda: not frame_row.refusal()))
        return dialog, frame_row, mode_row

    def _asking_dialog(self, title, rows, ready):
        from xdart.gui.tabs.scattering.advanced_editor import AdvancedDialog
        dialog = AdvancedDialog(self, title=title, object_name="backgroundTemplateAskDialog", accept_buttons=True)
        dialog.add_group("", rows)
        for row in rows:
            if isinstance(row, FormRow) and row._kind == "combo":
                row.editor.currentIndexChanged.connect(lambda _index: self._ok_follows(dialog, ready))
        self._ok_follows(dialog, ready)
        return dialog

    @staticmethod
    def _ok_follows(dialog, ready) -> None:
        dialog.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(bool(ready()))

    def _file_chooser(self):
        """The chosen file ("" when cancelled), through the shared ``file_dialogs`` (tests replace it)."""
        from xdart.gui.widgets import file_dialogs
        source = getattr(self, "_source", None)
        picks = file_dialogs.browse(self, f"Choose the {self._what} file",
                                    file_filter=getattr(file_dialogs, self._filter_name),
                                    current="" if source is None else source.path)
        return picks[0] if picks else ""

    def set_help_topic(self, topic_id: str) -> None:
        """Bind F1 and a "?" to ``topic_id``, as the shared rows do."""
        from xdart.gui.widgets.context_help import bind_topic
        bind_topic(self, topic_id)
        if not hasattr(self, "help_button"):
            self.help_button = ActionButton.for_help(self, topic_id)
            self._buttons.addWidget(self.help_button)


class BackgroundTemplateRow(CurveFileRow):
    """The template row; ``what`` names the template in its texts ("template", Phase Analysis's "substrate").

    ``changed`` is emitted after every change of the template, its source, identity or reason. The host's "Use
    displayed" is the dialog's (``capture_btn``); a host that can name the displayed trace's file installs
    ``describe_displayed`` (returning ``(path, frame, mode)`` or None), which pre-fills a chosen file's frame and mode
    when it is that file."""

    changed = QtCore.Signal()

    def __init__(self, parent=None, *, what="template", capture_label="Use displayed", clear_label="Clear"):
        super().__init__(parent, what=what)
        self.setObjectName("backgroundTemplateRow")
        self._template = None   # a captured DisplayedTraceInput
        self._source = None     # a BackgroundTemplateSource: chosen, restored, bound, or a capture's candidate
        self._identity = None   # the content identity a completed fit recorded
        self._reason = ""       # why the template cannot be named, read or exported now
        self._file = None       # (kind, path) of a chosen or bound file, whose rows rebuild its source
        self._available, self._waits, self._busy = True, "", ""  # the host's background and batch state
        self.describe_displayed = None
        self.status.setObjectName("backgroundTemplateStatus")
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
        self.unit_row, self.frame_row, self.mode_row = self._curve_rows(
            "background_template",
            unit_tip="The .xye file's x unit. A file name's xdart prefix (iq_, itth_, iqip_, iqoop_, iexit_) only "
                     "suggests it: confirm it. It must be the fit's unit, or convert to it without a wavelength (Å⁻¹ "
                     "and nm⁻¹; degrees and radians).",
            frame_tip="The NeXus result's frame label to read; a frame the file does not hold is refused at the fit.",
            mode_tip="The NeXus result's 1-D curve: default for a single result; q_total, q_ip, q_oop, exit_angle or "
                     "chi_gi for grazing incidence. A mode the file does not hold is refused at the fit, naming the "
                     "file's modes.")
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
        self._announce()

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
        self._announced = ""
        self._template, self._source, self._identity, self._file = trace, candidate, None, None
        self._reason = "" if candidate is not None else (
            reason or f"the displayed {self._what} has no NeXus file and frame behind it")
        self._sync()

    def clear(self) -> None:
        self._announced = ""
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
        self._announce()
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
        self._announce()
        return True

    # ---- the host's state ----------------------------------------------------
    def set_available(self, available: bool, reason: str = "") -> None:
        """Whether the host's background uses the template: off, every control waits, saying ``reason``; the template
        held is kept for when it is chosen again."""
        if (bool(available), str(reason)) != (self._available, self._waits):
            self._available, self._waits = bool(available), str(reason)
            self._apply_controls()

    def set_busy(self, reason: str = "") -> None:
        """A host's batch holds its fit worker (``reason``; "" when idle): Use displayed waits, saying why."""
        if str(reason) != self._busy:
            self._busy = str(reason)
            self._apply_controls()

    def _apply_controls(self) -> None:
        """Each control as the host's state and the row's own allow: the captured trace's file rows lock until its fit
        names it; Clear needs a template."""
        waits = "" if self._available else self._waits
        present = self._template is not None or self._source is not None or self._file is not None
        for button, enabled, reason in (
                (self.capture_btn, self._available and not self._busy, waits or self._busy),
                (self.choose_btn, self._available, waits),
                (self.clear_btn, self._available and present,
                 waits or ("" if present else f"No {self._what} is set."))):
            spec = button.spec
            button.apply_spec(ControlActionSpec(spec.action, spec.label, spec.section, enabled=enabled,
                                                reason=reason if not enabled else ""))
        for row in (self.unit_row, self.frame_row, self.mode_row):
            row.editor.setEnabled(self._available and self._template is None)

    # ---- the messages ----------------------------------------------------------
    def _short(self) -> str:
        """The row's message in a few plain words (its whole text is in the tooltip and the popup)."""
        reason = self._reason
        for code, words in (("UNIT_REQUIRED", "choose its x unit"), ("INVALID_SOURCE", "choose its frame"),
                            ("CHANGED", "changed since its fit: choose it again"),
                            ("UNREPRODUCIBLE", "its file does not hold this trace")):
            if code in reason:
                return words
        if "missing" in reason.casefold():
            return "file missing"
        return "cannot be used: see its message" if reason else ""

    def _announce(self) -> None:
        """A message about this source, once, in a popup (``_notify``); another file gets its own message."""
        reason = self._reason
        if not reason:
            self._announced = ""
            return
        notice = (self._file, None if self._source is None else self._source.to_pairs(), reason)
        if notice == self._announced:
            return
        self._announced = notice
        name = (Path(self._file[1]).name if self._file is not None else
                Path(self._source.path).name if self._source is not None else f"The displayed {self._what}")
        self._notify(f"{name}: {self._short()}", reason)

    # ---- choosing a file ---------------------------------------------------
    def _choose(self) -> None:
        path = self._file_chooser()
        if not path:
            return
        self._announced = ""  # a deliberate new selection can report the same refusal again
        kind = "nexus" if Path(path).suffix.casefold() in _NEXUS_SUFFIXES else "xye"
        if kind == "xye":
            # a name never fixes the scale (Codex REVIEW-3 F1): xdart writes iq_ for Å⁻¹ and nm⁻¹ alike, so the popup
            # asks and preselects nothing; cancelled, the unit waits for the row's choice
            unit = self._ask_unit(str(path), xye_unit_hint(path))
        else:
            frame, mode = None, NEXUS_MODES[0]
            displayed = self.describe_displayed() if callable(self.describe_displayed) else None
            if displayed is not None and _same_file(displayed[0], path):
                frame, mode = displayed[1], displayed[2]
            else:
                asked = self._ask_frame(str(path), frame, mode)
                if asked is not None:
                    frame, mode = asked
        self._template, self._identity, self._file = None, None, (kind, str(path))
        with QtCore.QSignalBlocker(self.unit_row.editor), QtCore.QSignalBlocker(self.mode_row.editor):
            if kind == "xye":
                self.unit_row.editor.setCurrentIndex(max(0, self.unit_row.editor.findData(unit)) if unit else 0)
            else:
                self.frame_row.editor.setValue(frame)
                index = self.mode_row.editor.findText(mode)
                self.mode_row.editor.setCurrentIndex(index if index >= 0 else 0)
        self._rebuild()
        if not any(code in self._reason for code in ("UNIT_REQUIRED", "INVALID_SOURCE")):
            self._announce()  # a waiting choice was just asked; anything else about the file opens a popup

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
        """The template as the row's status names it, in one short line (a message's whole text is in the tooltip and
        its popup)."""
        template, source = self._template, self._source
        if template is not None:  # a captured trace is usable whether or not a file names it
            name = f"{template.title} ({template.label})" if template.title else template.label  # with its axis
            if "UNREPRODUCIBLE" in self._reason:
                return f"Displayed: {name}: its file does not hold this trace; not saved or exported"
            return f"Displayed: {name}" + ("; its file is checked at the first fit" if source is not None else
                                           "; not named by a file: not saved or exported")
        if source is not None:
            state = self._short() if self._reason else (
                "recorded" if self._identity is not None else "read at the first fit")
            return f"{source.describe()}: {state}"
        if self._file is not None:
            return f"{Path(self._file[1]).name}: {self._short()}"
        return f"No {self._what}: use the displayed trace or choose a file"

    def _sync(self) -> None:
        text = self.describe()
        self.status.setText(text)
        detail = self._reason or (self._source.path if self._source is not None else "")
        self.status.setToolTip(
            f"The {self._what} the template backgrounds scale. " + (f"{text}. {detail}" if detail else text))
        kind = self._file[0] if self._file is not None else (None if self._source is None else self._source.kind)
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
        self._apply_controls()
        self.changed.emit()


def _same_file(first, second) -> bool:
    try:
        return Path(first).expanduser().resolve() == Path(second).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return str(first) == str(second)


#: the Source row's texts (NEXT-128-WINDOW-POLISH item 11)
_DISPLAY_TEXT = "Int's display (the trace Int shows)"
_XYE_REFUSED = ("An .xye file records no wavelength, which phase fitting needs to place each reflection: choose a "
                "processed NeXus result.")


def _as_source(text: str) -> str:
    """A template message reworded for the fit's source (its codes and nouns)."""
    return str(text).replace("BACKGROUND_TEMPLATE_", "FIT_SOURCE_").replace("template", "source")


class FitSourceRow(CurveFileRow):
    """The fit windows' Source (NEXT-128-WINDOW-POLISH item 11; the maintainer, 2026-11-03: "an option to choose a
    separate source than the nexus file selected in the main window in the fit windows", "Yes, in 1.2.8 with Batch"):
    Int's display (the default) or a file curve, as a ``BackgroundTemplateSource`` (a NeXus result's frame and 1-D
    mode, or an .xye file and its x unit), the value type the template row already names a file curve with.

    "Int's display" follows Int: the host fits the trace Int shows. "Choose file…" asks the file, then its frame and
    1-D mode or its x unit in the template row's popups; ``xye=False`` (Phase Analysis) refuses an .xye file in a popup,
    since it records no wavelength. Editing the 1-D mode or x unit row changes the source. ``changed`` is emitted after
    every change. The row reads no file (a chosen one is checked with a stat) and the source is not saved with the
    settings: it is the input, not the recipe.

    The Frame row (NEXT-128-WINDOW-POLISH item 14: the maintainer, 2026-11-03, "we also want a frame selection widget
    like in the main window to be able to go to different frames"; his "Mirror Int's frames (Recommended)") is Int's
    shared ``FrameNavigator`` on the shared ``FormRow``: its host fills it with Int's frames while the source is Int's
    display (``mirror_frames``) or a NeXus file's labels (``set_frame_labels``), and hears each user's step or pick as
    ``frameChosen`` (a frame key, or a label), answering a file's with ``step_frame``. It replaces the typed frame field
    and is hidden for an .xye file, which holds one curve; ``set_busy`` holds it while a batch runs."""

    changed = QtCore.Signal()
    #: a user's step or pick in the Frame row: Int's frame key (Int's display) or the file's frame label
    frameChosen = QtCore.Signal(object)

    def __init__(self, parent=None, *, xye=True):
        super().__init__(parent, what="source", filter_name="BACKGROUND_TEMPLATE_FILTER" if xye else "NEXUS_FILTER")
        self.setObjectName("fitSourceRow")
        self.status.setObjectName("fitSourceStatus")
        self._xye = bool(xye)
        self._source = None   # the chosen BackgroundTemplateSource, or None: Int's display
        self._reason = ""     # why the chosen file cannot be read now ("" when it can)
        self.describe_displayed = None
        self.follow_btn = ActionButton(ControlActionSpec(ControlAction.PREVIEW, "Int's display", SectionId.PROCESSING),
                                       self)
        self.follow_btn.set_purpose_tooltip(
            "Fit what Int shows: Reload and Fit take Int's displayed trace (its frame, 1-D mode, axis, χ slice, "
            "normalization and Set BG); Batch fits every frame of the shown NeXus result from its stored 1-D curves.")
        self.choose_btn = ActionButton(ControlActionSpec(ControlAction.PREVIEW, "Choose file…", SectionId.PROCESSING),
                                       self)
        self.choose_btn.set_purpose_tooltip(
            "Fit a file instead of Int's display: a processed NeXus result's frame and 1-D mode"
            + (" or an .xye file and its x unit" if xye else "")
            + ". Reload and Fit read that frame's stored 1-D curve; Batch fits every frame of the NeXus result. The "
              "source is not saved with the settings.")
        for button in (self.follow_btn, self.choose_btn):
            button.setAutoDefault(False)
            self._buttons.addWidget(button)
        self._buttons.addStretch(1)
        self.frames_row = FormRow(label="Frame", path=("fit_source", "frames"), value="", kind="frames")
        self._frames_tip = ("The frame Reload and Fit read, stepped with < and > or picked from the list: following "
                            "Int's display, Int's frames, and a step moves the main window to that frame, which this "
                            "window then shows; with a chosen NeXus file, its frames (Int stays where it is).")
        self.frames_row.set_purpose_tooltip(self._frames_tip)
        self.frames_row.valueChanged.connect(lambda _path, frame: self.frameChosen.emit(frame))
        self.layout().addWidget(self.frames_row)
        self.unit_row, self.frame_row, self.mode_row = self._curve_rows(
            "fit_source",
            unit_tip="The .xye file's x unit, chosen when the file was: a file name's xdart prefix only suggests it.",
            frame_tip="The NeXus result's frame that Reload and Fit read; Batch fits every frame. A frame the file "
                      "does not hold is refused by name.",
            mode_tip="The NeXus result's 1-D curve: default for a single result; q_total, q_ip, q_oop, exit_angle or "
                     "chi_gi for grazing incidence. A mode the file does not hold is refused, naming its modes.")
        self.follow_btn.clicked.connect(self.use_display)
        self.choose_btn.clicked.connect(self._choose)
        self.unit_row.editor.currentIndexChanged.connect(lambda _index: self._edited())
        self.mode_row.editor.currentIndexChanged.connect(lambda _index: self._edited())
        self.frame_row.valueChanged.connect(lambda _path, _value: self._edited())
        self._sync()

    @property
    def source(self):
        """The chosen ``BackgroundTemplateSource``, or None: Int's display."""
        return self._source

    @property
    def reason(self):
        """Why the chosen file cannot be read now ("" when it can, or with Int's display)."""
        return self._reason

    def input_refusal(self) -> str:
        """A refused editor must not silently reuse the previously committed frame or unit."""
        if self._source is not None:
            if self._source.kind == "nexus":
                reason = self.frame_row.refusal() or (
                    "choose a source frame" if self.frame_row.editor.value() is None else "")
                if reason:
                    return f"FIT_SOURCE_INVALID: Frame: {reason}"
            if self._source.kind == "xye" and not self.unit_row.current_value():
                return "FIT_SOURCE_UNIT_REQUIRED: choose the source's x unit"
        return ""

    def report_problem(self, detail: str) -> None:
        """A worker's read refusal, once for this source; reporting it changes no source or routing state."""
        if self._source is None:
            return
        notice = (self._source.to_pairs(), str(detail))
        if notice != self._announced:
            self._announced = notice
            self._notify(f"{Path(self._source.path).name}: cannot be read", str(detail))

    def set_help_topic(self, topic_id: str) -> None:
        """Bind F1 and the "?"s (the buttons' and the Frame row's) to ``topic_id``."""
        super().set_help_topic(topic_id)
        self.frames_row.set_help_topic(topic_id, button=True)

    def mirror_frames(self, selector) -> None:
        """Int's display: the Frame row lists ``selector``'s frames (Int's footer) and its current one; emits nothing."""
        self.frames_row.editor.mirror(selector)

    def set_frame_labels(self, labels) -> None:
        """A NeXus source: the Frame row lists its frame ``labels``, the source's selected; emits nothing."""
        source = self._source
        name = "" if source is None else Path(source.path).name
        self.frames_row.editor.set_labels(tuple(labels), current=None if source is None else source.frame,
                                          tip="Frame {label} of " + name.replace("{", "{{").replace("}", "}}"))

    def select_frame(self, frame) -> None:
        """Show ``frame`` selected in the Frame row (a frame key or a label); emits nothing."""
        self.frames_row.editor.select(frame)

    def step_frame(self, frame) -> None:
        """A NeXus source's frame becomes ``frame`` (its 1-D mode and entry kept): the host's answer to a step."""
        source = self._source
        if source is None or source.kind != "nexus":
            return
        self._announced = ""
        self._set(BackgroundTemplateSource("nexus", source.path, frame=int(frame), mode=source.mode,
                                           entry=source.entry))
        self.select_frame(int(frame))

    def set_busy(self, reason: str = "") -> None:
        """Hold the Frame row while ``reason`` says why (a running fit or batch); "" releases it."""
        self.frames_row.editor.setEnabled(not reason)
        self.frames_row.set_reason(reason)

    def use_display(self) -> None:
        """Follow Int's display again."""
        if self._source is None:
            return
        self._source, self._reason, self._announced = None, "", ""
        self._sync()

    def _choose(self) -> None:
        path = self._file_chooser()
        if not path:
            return
        name = Path(path).name
        if Path(path).suffix.casefold() not in _NEXUS_SUFFIXES:
            if not self._xye:
                self._notify(f"{name}: not a NeXus result", _XYE_REFUSED)
                return
            # a name never fixes the scale (Codex REVIEW-3 F1): the popup asks and preselects nothing
            unit = self._ask_unit(str(path), xye_unit_hint(path))
            if not unit:
                return  # cancelled: the source stays as it was
            source = BackgroundTemplateSource("xye", str(path), unit=unit)
        else:
            current = self._source if self._source is not None and self._source.kind == "nexus" else None
            frame, mode = (None, NEXUS_MODES[0]) if current is None else (current.frame, current.mode)
            displayed = self.describe_displayed() if callable(self.describe_displayed) else None
            if displayed is not None and _same_file(displayed[0], path):
                frame, mode = displayed[1], displayed[2]  # the frame Int shows, offered first
            asked = self._ask_frame(str(path), frame, mode)
            if asked is None:
                return
            source = BackgroundTemplateSource("nexus", str(path), frame=int(asked[0]), mode=asked[1])
        self._announced = ""  # a deliberate new choice can report the same problem again
        self._set(source)

    def _set(self, source) -> None:
        if source == self._source:
            return
        self._source = source
        self._reason = _as_source(template_file_problem(source))
        self._sync()
        if self._reason:
            notice = (source.to_pairs(), self._reason)
            if notice != self._announced:
                self._announced = notice
                self._notify(f"{Path(source.path).name}: cannot be read", self._reason)

    def _edited(self) -> None:
        """A user's edit of the Frame, 1-D mode or x unit row: the source becomes that frame, mode or unit."""
        source = self._source
        if source is None:
            return
        try:
            if source.kind == "xye":
                unit = self.unit_row.current_value()
                edited = BackgroundTemplateSource("xye", source.path, unit=unit) if unit else source
            else:
                frame = None if self.frame_row.refusal() else self.frame_row.editor.value()
                edited = source if frame is None else BackgroundTemplateSource(
                    "nexus", source.path, frame=int(frame), mode=self.mode_row.current_value(), entry=source.entry)
        except BackgroundTemplateRefused:
            edited = source
        if edited != source:
            self._set(edited)

    def describe(self) -> str:
        """The source in one short line: Int's display, or the file and what is read from it."""
        if self._source is None:
            return _DISPLAY_TEXT
        text = self._source.describe()
        return f"{text}: cannot be read" if self._reason else text

    def _sync(self) -> None:
        source = self._source
        text = self.describe()
        self.status.setText(text)
        detail = self._reason or ("" if source is None else source.path)
        self.status.setToolTip("What Fit, Reload and Batch read: Int's display or a chosen file. "
                               + (f"{text}. {detail}" if detail else f"{text}."))
        kind = None if source is None else source.kind
        self.unit_row.setVisible(kind == "xye")
        # the Frame row's navigator steps the frame (item 14); the typed field keeps the source's frame for its checks
        self.frame_row.setVisible(False)
        self.frames_row.setVisible(kind != "xye")
        self.mode_row.setVisible(kind == "nexus")
        # the frame editor is not blocked: setValue emits no row edit, and its validity display must follow
        with QtCore.QSignalBlocker(self.unit_row.editor), QtCore.QSignalBlocker(self.mode_row.editor):
            if kind == "xye":
                self.unit_row.editor.setCurrentIndex(max(0, self.unit_row.editor.findData(source.unit)))
            elif kind == "nexus":
                self.frame_row.editor.setValue(source.frame)
                index = self.mode_row.editor.findText(source.mode)
                self.mode_row.editor.setCurrentIndex(index if index >= 0 else 0)
        spec = self.follow_btn.spec
        self.follow_btn.apply_spec(ControlActionSpec(spec.action, spec.label, spec.section, enabled=source is not None,
                                                     reason="" if source is not None else
                                                     "Int's display is the source."))
        self.changed.emit()
