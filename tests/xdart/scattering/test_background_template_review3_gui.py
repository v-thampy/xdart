"""NEXT-128-PEAK-FIT, Codex REVIEW-3 F1 (REWORK on 0d830cf8): an .xye file's name never fixes its unit (REVIEW-1 R3:
"A filename prefix is only a hint; an ambiguous one requires a choice"). xdart writes ``iq_`` for q in Å⁻¹ and in
nm⁻¹, ``itth_`` for 2θ in degrees and radians, so every prefix leaves the scale open: the shared row suggests the
unit and waits for the user's choice. Frozen before the correction (red on 0d830cf8); Codex's discriminator, with the
suggestion and the explicit choice added.
"""
import numpy as np
from pyqtgraph.Qt import QtWidgets


def test_f1_an_iq_prefix_waits_for_an_explicit_scale_choice(tmp_path, monkeypatch):
    from xdart.gui.analysis.background_template_row import BackgroundTemplateRow
    from xrd_tools.session.display_logic import xye_prefix_for_unit

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert xye_prefix_for_unit("q_nm^-1") == xye_prefix_for_unit("q_A^-1") == "iq"
    path = tmp_path / "iq_substrate.xye"
    np.savetxt(path, np.column_stack((np.linspace(10., 20., 6), np.arange(6.) + 1)))
    row = BackgroundTemplateRow()
    try:
        monkeypatch.setattr(row, "_file_chooser", lambda: str(path))
        row.choose_btn.click()
        app.processEvents()
        assert row.source is None, f"ambiguous prefix silently resolved to {row.source.unit}"
        assert "BACKGROUND_TEMPLATE_UNIT_REQUIRED" in row.fit_refusal()
        assert "Å⁻¹" in row.reason and "nm⁻¹" in row.reason  # the name's suggestion, and the scale left open
        editor = row.unit_row.editor
        editor.setCurrentIndex(editor.findData("q_nm^-1"))  # the user's choice
        app.processEvents()
        assert row.source is not None and row.source.unit == "q_nm^-1" and row.reason == ""
    finally:
        row.close()
