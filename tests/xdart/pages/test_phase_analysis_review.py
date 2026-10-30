"""Review regressions on the real Int window and Stitch host, with isolated state."""
import json

import numpy as np
import pytest

from tests.xdart.pages.test_phase_analysis_stitch import isolated_library, qapp
from tests.xdart.pages.test_stitch_fit_host import _finish
from tests.xdart.scattering.test_phase_analysis_window import empty_page, _open
from tests.xdart.scattering.test_phase_reference_overlays import _stitch
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind


def test_stitch_range_survives_axis_change_and_settings_reload(qapp, tmp_path):
    from xrd_tools.core.containers import convert_radial_axis
    with _stitch(qapp, tmp_path / "stitch") as (page, _view, _payload):
        page._view_command(ShellCommand(ShellCommandKind.SET_PLOT_AXIS, "2theta"))
        page.tools.page_tool("phase_analysis").click()
        host = page._fit_hosts["phase"]
        _finish(host, qapp)
        dialog = host.dialog
        old = tuple(host.snapshot.axes[0].values[[2, -3]])
        for editor, value in zip(dialog.fit_range_row.editors, old):
            editor.setText(f"{value:.12g}")
        expected = convert_radial_axis(np.asarray(old), "2th_deg", "q_A^-1", host.snapshot.wavelength_angstrom)
        path = tmp_path / "phase-settings.json"
        dialog._settings_path_chooser = lambda _action, _start: str(path)
        assert type(dialog.settings_io.save()).__name__ == "ActionCompleted"
        page._view_command(ShellCommand(ShellCommandKind.SET_PLOT_AXIS, "Q"))
        host.reload()
        _finish(host, qapp)
        np.testing.assert_allclose(dialog.vnext_fit_values()["fit_range"], expected, rtol=1e-9)
        assert dialog.vnext_fit_values()["fit_range_unit"] == "q_A^-1"
        dialog.full_btn.click()
        assert type(dialog.settings_io.load()).__name__ == "ActionCompleted"
        np.testing.assert_allclose(dialog.vnext_fit_values()["fit_range"], expected, rtol=1e-9)
        document = json.loads(path.read_text())
        document["fit_range_unit"] = "unknown"
        path.write_text(json.dumps(document))
        before = dialog.vnext_fit_values()
        assert type(dialog.settings_io.load()).__name__ == "ActionRefused"
        assert dialog.vnext_fit_values() == before


def test_saved_zero_march_axis_refuses_the_whole_document(qapp):
    with empty_page(qapp) as page:
        dialog = _open(qapp, page)
        document = dialog._settings_document(dialog)
        document["fields"].update(texture="march_dollase", march_h=0, march_k=0, march_l=0,
                                  profile="gaussian")
        before = dialog._settings_document(dialog)
        with pytest.raises(ValueError, match="March"):
            dialog._apply_settings(dialog, document)
        assert dialog._settings_document(dialog) == before


def test_stitch_fit_projects_cif_receipts_and_calibrant_notice(qapp, tmp_path):
    from tests.core.test_reference_phase_sources import _fit_inputs
    paths, _trace = _fit_inputs(tmp_path)
    with _stitch(qapp, tmp_path / "stitch") as (page, _view, _payload):
        page.tools.page_tool("phase_analysis").click()
        host = page._fit_hosts["phase"]
        _finish(host, qapp)
        dialog = host.dialog
        dialog.phase_source.add_cif_paths((str(paths[0]),))
        dialog.phase_source.add_calibrant("LaB6")
        dialog._maxfev_row.set_number(8)
        qapp.processEvents()
        host._request("fit", (page._fit_current("phase"),))
        _finish(host, qapp)
        result = host.outcomes[-1].result
        assert result is not None and result.code == "OK"
        for receipt in result.cif_receipts:
            assert receipt.sha256 in dialog._receipts[receipt.expected_sha256 or receipt.sha256]
        assert "LaB6" in dialog.status.text() and "not fitted" in dialog.status.text()
