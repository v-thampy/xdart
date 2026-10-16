"""Shared, rarely changed NeXus/Eiger options beside a page's Data format."""
from dataclasses import replace

from xdart.gui.tabs.scattering.advanced_editor import AdvancedDialog
from xdart.gui.widgets.controls_panel import ActionButton
from xrd_tools.session.readiness import ControlAction, ControlActionSpec, SectionId


class DataFormatOptions(AdvancedDialog):
    """Mount Int's Entry FormRow in its Advanced shell; values stay with the page's settings owner."""

    def __init__(self, parent, entry):
        super().__init__(parent, title="NeXus / Eiger options", object_name="dataFormatOptions")
        self.entry = entry
        entry.editor.setPlaceholderText("Auto")
        self._entry_purpose = (
            "NeXus entry shared by this page's sample, direct-beam and calibrant files. Blank (Auto) keeps "
            "automatic sample discovery and existing calibration entries. An explicit entry must exist; "
            "another entry is never substituted.")
        entry.set_purpose_tooltip(self._entry_purpose)
        self.add_group("NeXus / Eiger", (entry,))
        self.button = ActionButton(ControlActionSpec(ControlAction.SOURCE_OPTIONS, "…", SectionId.PROJECT), parent)
        self.button.setAccessibleName("NeXus / Eiger options")
        self.button.setMaximumWidth(40)
        self.button.set_purpose_tooltip("Advanced options for NeXus / Eiger files, including the shared Entry.")
        self.button.clicked.connect(self._open)

    def _open(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def reconcile(self, data_format, *, enabled=True, reason=""):
        visible = data_format in {"spec_container", "nexus"}
        self.button.setVisible(visible)
        self.button.apply_spec(replace(self.button.spec, enabled=enabled, reason=reason))
        self.entry.set_purpose_tooltip(self._entry_purpose if enabled else reason or "Input settings are locked.")
        self.set_editors_enabled(enabled)
        if not visible:
            self.hide()
