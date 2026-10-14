"""A page's saved settings: File ▸ Save/Load Settings and the last session, one owner for every page that keeps
them (Stitch, RSM).

The maintainer, 2026-09-24: "File ▸ Save/Load Settings and last-session restore covering every input, the Advanced
options and any calibration panel, never per-use confirmations, and a fresh start neither reading nor writing it".

A page's settings are one versioned JSON document, ``{"kind": "xdart-<page>-settings", "version": 1, ...}``. The
page supplies its own part: the fields it saves, its own sections, the order its fields apply in, and its
calibration window's state (that window's ``settings()``, which never hold a per-use confirmation). This owner
keeps the rest:

- the document (``SettingsKind``): its kind and version checked (a foreign file, or one from a newer xdart, is
  refused with the reason), the shared sections' shapes, and its text (indented, keys sorted);
- the shared sections, read and restored alike on every page: the fields as plain values, the source files by
  source kind, Int's threshold policy as one state, and the run bar's Append ⇄ Replace; and the check that each
  saved field fits its row (``check_fields``: a choice among its choices, a toggle on or off), RSM's so far;
- File ▸ Save and Load (``PageSettings``, behind the page's ``settings_io``): the chooser through
  ``file_dialogs``, the remembered folder, ``.json`` added to a bare name, an atomic write, a load refused while the
  page is busy, and a document that cannot apply rolled back to the inputs before it. Never a confirmation;
- the last session: restored when the page is built, remembered when it closes cleanly. A fresh start
  (``XDART_SESSION_FRESH``) neither reads nor writes it (``xdart.utils.session``).

Unknown keys are ignored and a missing field keeps the page's value, so an older or partial document restores what
it can.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable

from xdart.gui.pages.values import ActionCompleted, ActionOutcome, ActionRefused, CloseReceipt, PageCleanup

_LOG = logging.getLogger(__name__)


def plain(value: Any) -> Any:
    """A value as the document keeps it: None, a bool, an int or a str as is, a float if finite (None if not),
    anything else as its text."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


@dataclass(frozen=True, slots=True)
class SettingsKind:
    """One page's settings document: its kind and version, its key in the session file, the page's name in every
    message ("Stitch"), and the page's own sections that must be objects (Stitch's ``gi``)."""

    kind: str
    version: int
    session_key: str
    label: str
    sections: tuple[str, ...] = ()

    @property
    def outcome(self) -> str:
        """The outcomes' prefix: ``<page>-settings-saved``, ``-loaded``, ``-save-cancelled``, ``-load-cancelled``."""
        return f"{self.label.casefold()}-settings"

    def document(self, page, fields, **sections) -> dict[str, Any]:
        """The page's document now: the shared sections, its ``fields`` the keys it saves, then the page's own
        ``sections`` in the order given (the session file keeps the order)."""
        state = page._threshold_state
        return {
            "kind": self.kind,
            "version": self.version,
            "fields": {key: plain(page._value(key)) for key in fields},
            "sources": {str(kind): [str(path) for path in paths]
                        for kind, paths in page._paths_by_kind.items() if paths},
            "threshold": {"apply_threshold": bool(state.apply_threshold),
                          "threshold_min": plain(state.threshold_min),
                          "threshold_max": plain(state.threshold_max),
                          "mask_saturation": bool(state.mask_saturation)},
            "calibration": page._calibration_settings_now(),
            **sections,
        }

    def validate(self, document: Any) -> dict[str, Any]:
        """Return the document if it is this page's settings and this version can read it; ValueError says why
        not."""
        if (type(document) is not dict or document.get("kind") != self.kind
                or type(document.get("version")) is not int):
            raise ValueError(f"not an xdart {self.label} settings file")
        if document["version"] > self.version:
            raise ValueError(f"these {self.label} settings come from a newer xdart")
        for key in ("fields", "sources", "threshold", *self.sections):
            if type(document.get(key, {})) is not dict:
                raise ValueError(f"{self.label} settings {key} are malformed")
        calibration = document.get("calibration")
        if calibration is not None and (type(calibration) is not dict or any(
                type(calibration.get(key, {})) is not dict for key in ("fields", "choices"))):
            raise ValueError(f"{self.label} settings calibration is malformed")
        return document

    def dump(self, document: dict[str, Any]) -> str:
        return json.dumps(self.validate(document), indent=2, sort_keys=True)

    def load(self, text: str) -> dict[str, Any]:
        try:
            document = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"not an xdart {self.label} settings file") from error
        return self.validate(document)


def document_data_format(document: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """The page's Data format and Files/Folder input a document names, applied before its other fields, and the
    conflict an older document leaves to the user ("" for none). A current document names them itself; an older
    one (LT-SMALL-FIXES-20261014, D2) through its Source kind and its calibration window's saved calibrant and
    direct-beam sources (``legacy_data_format``); a document naming neither keeps the page's."""
    from xrd_tools.sources.selection import legacy_data_format

    fields = document.get("fields", {})
    current = {key: fields[key] for key in ("data_format", "input_mode") if key in fields}
    if current or "source_kind" not in fields:
        return current, ""
    calibration = document.get("calibration") if isinstance(document.get("calibration"), dict) else {}
    inputs, choices = calibration.get("fields") or {}, calibration.get("choices") or {}

    def chosen(name):
        choice = choices.get(name)
        return choice.get("data") if isinstance(choice, dict) else None
    beam_used = chosen("rough_mode") in (None, "scan")  # a distance typed or skipped measures no beam scan
    roles = (("the calibrant scan", chosen("source_kind"), bool(str(inputs.get("source") or "").strip())),
             ("the direct-beam scan", chosen("beam_source_kind"),
              beam_used and bool(str(inputs.get("beam_source") or "").strip())))
    data_format, mode, conflict = legacy_data_format(str(fields["source_kind"]), roles)
    return {"data_format": data_format, "input_mode": mode}, conflict


def saved_fields(page, document: dict[str, Any], skip=frozenset()) -> dict[str, Any]:
    """The document's fields the page has, but those in ``skip`` (one shown from the others, or restored as a
    policy)."""
    return {key: value for key, value in document.get("fields", {}).items()
            if key in page._fields and key not in skip}


def check_fields(page, fields: dict[str, Any], open_choices=frozenset()) -> None:
    """Refuse a saved field its row cannot hold, naming the row (ValueError), before anything applies: a choice
    outside its row's choices (unless they are a source's own names, ``open_choices``), a toggle that is not true
    or false, or a list or an object in a text row. A newer or hand-edited document is so refused, never half
    read: a Coordinates the page cannot grid would otherwise wait for Preview to fail."""
    from xrd_tools.session.readiness import ControlFieldKind

    for key, value in fields.items():
        field = page._fields[key]
        if field.kind == ControlFieldKind.COMBO and key not in open_choices:
            if value not in field.choices:
                raise ValueError(f"{field.label}: {value!r} is not one of its choices")
        elif field.kind == ControlFieldKind.BOOL:
            if type(value) is not bool:
                raise ValueError(f"{field.label}: {value!r} is not on or off")
        elif value is not None and (type(value) is bool or not isinstance(value, (str, int, float))):
            raise ValueError(f"{field.label}: {value!r} is not text or a number")


def restore_sources(page, document: dict[str, Any]) -> None:
    """The document's source files by source kind; the current kind's become the selection."""
    page._paths_by_kind = {str(kind): tuple(str(path) for path in paths)
                           for kind, paths in document.get("sources", {}).items()
                           if type(paths) is list}
    page.set_sources(page._paths_by_kind.get(page._value("source_kind"), ()))


def saved_threshold(document: dict[str, Any]):
    """The document's threshold policy, Int's ``ThresholdState``; None when it holds none."""
    from xrd_tools.session.threshold_editing import ThresholdState

    threshold = document.get("threshold")
    if not threshold:
        return None
    return ThresholdState(bool(threshold.get("apply_threshold")),
                          threshold.get("threshold_min"), threshold.get("threshold_max"),
                          bool(threshold.get("mask_saturation")))


def restore_write_mode(page, document: dict[str, Any]) -> None:
    """The run bar's Append ⇄ Replace ("Overwrite"), when the document names one."""
    if document.get("write_mode") in ("Append", "Overwrite"):
        page.run_bar.set_write_mode(document["write_mode"])
        page._write_mode_changed()


class PageSettings:
    """One page's File ▸ Save/Load Settings and last session (the module docstring).

    ``document(page)`` is the page's document now and ``apply(page, document)`` checks one and restores it in the
    page's order; ``busy()`` refuses a load while the page works (True, or why as text: Stitch's calibration window
    working), and ``close()`` is the page's own close. The page keeps the surface both pages have: ``_status(text,
    ready=)``, ``_readiness``, ``_value("project_root")`` (where the chooser starts without a remembered folder),
    ``_closing``, and ``_settings_path_chooser(action, start)``, which returns the path chosen or None (tests
    replace it)."""

    def __init__(self, page, kind: SettingsKind, *, document: Callable, apply: Callable,
                 busy: Callable[[], bool | str], close: Callable[[], CloseReceipt]):
        self._page = page
        self.kind = kind
        self._document, self._apply, self._busy, self._close = document, apply, busy, close
        self._closing_settings = None  # captured at the first close, saved once the page is clean
        self._remembered = False

    def settings(self) -> dict[str, Any]:
        return self._document(self._page)

    def apply(self, document) -> None:
        """Restore a document; one that cannot apply leaves the inputs as they were."""
        before = self.settings()
        try:
            self._apply(self._page, document)
        except Exception:
            self._apply(self._page, before)
            raise

    def choose_path(self, action: str, start: str):
        """The page's chooser: ``file_dialogs`` with the settings filter, saving or opening."""
        from xdart.gui.widgets import file_dialogs

        title = f"{'Save' if action == 'save' else 'Load'} {self.kind.label} settings"
        picks = file_dialogs.browse(self._page, title, save=action == "save",
                                    file_filter=file_dialogs.SETTINGS_FILTER, current=start)
        return picks[0] if picks else None

    def _path(self, action: str):
        from xdart.utils.browse import browse_start_dir, remember_browse_path

        page = self._page
        selected = page._settings_path_chooser(action, browse_start_dir("", fallback=page._value("project_root")))
        if not selected:
            return None
        path = Path(selected).expanduser()
        if action == "save" and not path.suffix:
            path = path.with_suffix(".json")
        remember_browse_path(path)
        return path

    def save(self) -> ActionOutcome:
        """File ▸ Save Settings."""
        label = self.kind.label
        path = self._path("save")
        if path is None:
            return ActionCompleted(f"{self.kind.outcome}-save-cancelled")
        try:
            text = self.kind.dump(self.settings())
            temporary = path.with_name(f".{path.name}.tmp")
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, path)
        except (OSError, ValueError) as error:
            self._page._status(f"Could not save the {label} settings: {error}", ready=False)
            return ActionRefused(str(error))
        self._page._status(f"{label} settings saved: {path.name}")
        return ActionCompleted(f"{self.kind.outcome}-saved")

    def load(self) -> ActionOutcome:
        """File ▸ Load Settings."""
        label = self.kind.label
        busy = self._busy()
        if busy:
            return ActionRefused(busy if type(busy) is str else f"{label} is busy; load settings when it is idle")
        path = self._path("load")
        if path is None:
            return ActionCompleted(f"{self.kind.outcome}-load-cancelled")
        try:
            self.apply(self.kind.load(path.read_text(encoding="utf-8")))
        except Exception as error:  # noqa: BLE001 - a settings file is user input
            reason = str(error) if isinstance(error, (OSError, ValueError)) else (
                f"these {label} settings cannot be applied here ({type(error).__name__}: {error})")
            self._page._status(f"Could not load the {label} settings: {reason}", ready=False)
            return ActionRefused(reason)
        self._page._status(f"{label} settings loaded: {path.name}", ready=self._page._readiness[1])
        return ActionCompleted(f"{self.kind.outcome}-loaded")

    def restore_last(self) -> None:
        """Restore the settings the last session left (nothing in a fresh session)."""
        from xdart.utils.session import load_session

        document = load_session().get(self.kind.session_key)
        if document is None:
            return
        try:
            self.apply(document)
        except Exception:  # noqa: BLE001 - saved settings are a convenience
            _LOG.warning("Could not restore the last %s settings", self.kind.label, exc_info=True)
            self._page._status(f"Could not restore the last {self.kind.label} settings", ready=False)

    def close_and_remember(self) -> CloseReceipt:
        """Close the page and keep its settings for the next session once it is clean."""
        if self._closing_settings is None and not self._page._closing:
            try:
                self._closing_settings = self.settings()
            except Exception:  # noqa: BLE001
                _LOG.warning("Could not capture the %s settings", self.kind.label, exc_info=True)
        receipt = self._close()
        if (receipt.status is PageCleanup.CLEAN and not self._remembered
                and self._closing_settings is not None):
            from xdart.utils.session import save_session

            save_session({self.kind.session_key: self._closing_settings})
            self._remembered = True
        return receipt


__all__ = [
    "PageSettings", "SettingsKind", "check_fields", "plain", "restore_sources", "restore_write_mode",
    "saved_fields", "saved_threshold",
]
