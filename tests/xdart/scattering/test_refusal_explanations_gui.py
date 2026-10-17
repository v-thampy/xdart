"""Frozen P1 E1, the GUI half: actual Stitch and RSM refusals projected onto their pages. Moved verbatim
from ``tests/core/test_refusal_explanations.py``: the core job installs no Qt display stack."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.analysis.module_transaction import ModuleDisposition
from tests.core.test_native_stitch_operation import _native


def _page_projection(result, family):
    from pyqtgraph.Qt import QtWidgets
    from xdart.gui.pages.operation_owner import OperationIdentity, OperationTerminalStatus
    from xdart.gui.pages.services import empty_host_services
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    class Status:
        def show(self, text, timeout_ms=0):
            pass
    identity = OperationIdentity(1, result.request)
    if family == 'stitch':
        from xdart.gui.tabs.stitch.page import StitchPage
        from xdart.gui.tools.stitch_owner import (
            StitchOwnerAction, StitchOwnerOutcome, StitchOwnerOutcomeKind, StitchOwnerUpdate)
        page = StitchPage(empty_host_services(Status()))
        update = StitchOwnerUpdate(identity, StitchOwnerAction.RUN,
            terminal_status=OperationTerminalStatus.RETURNED,
            outcome=StitchOwnerOutcome(StitchOwnerOutcomeKind.RESULT, result=result))
        accept = page._accept_outcome
    else:
        from xdart.gui.tabs.rsm.page import RSMPage
        from xdart.gui.tools.rsm_owner import (
            RSMOwnerAction, RSMOwnerOutcome, RSMOwnerOutcomeKind, RSMOwnerUpdate)
        page = RSMPage(empty_host_services(Status()))
        update = RSMOwnerUpdate(identity, RSMOwnerAction.RUN,
            terminal_status=OperationTerminalStatus.RETURNED,
            outcome=RSMOwnerOutcome(RSMOwnerOutcomeKind.RESULT, result=result))
        accept = page._accept
    try:
        accept(update)
        text = page.run_bar.readinessLabel.text()
        assert result.terminal.code in text and result.terminal.diagnostic in text
    finally:
        (page.shutdown if family == 'stitch' else page.close_page)()
        page.deleteLater(); app.processEvents()


def test_native_stitch_refusal_retains_reason_and_page_code(tmp_path):
    from xrd_tools.analysis.stitch_operation import run_stitch_operation
    request = _native(tmp_path, 'multigeometry', 'q_A^-1', grouped=False)
    np.save(request.plan.mask_path, np.ones((195, 487), dtype=np.uint8))
    result = run_stitch_operation(request)
    assert result.terminal.diagnostic == 'mask no longer matches Preview'
    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == 'MASK_IDENTITY_MISMATCH'
    assert not Path(request.module.output.target).exists()
    _page_projection(result, 'stitch')


def test_rsm_v2_early_output_folder_refusal_retains_reason_and_notice(tmp_path, monkeypatch):
    from tests.core.test_rsm_operation_r2 import _form, _write_member
    from xrd_tools.analysis.rsm_values import prepare_rsm_tool_v2
    from xrd_tools.analysis.rsm_operation import (
        RSMOperationRefused, _create_rsm_v2_output_folder, run_rsm_operation_v2)
    from xrd_tools.sources.spec import SpecSource
    form = _form(tmp_path, (_write_member(tmp_path, 0),))
    form = replace(form, output_path=tmp_path / 'output' / 'late' / 'rsm-v2.nexus')
    request = prepare_rsm_tool_v2(form).request
    (tmp_path / 'output' / 'late').write_text('blocking file')
    with pytest.raises(RSMOperationRefused) as original:
        _create_rsm_v2_output_folder(request)
    decoded = []
    real_load = SpecSource.load_frame
    def counted(*args, **kwargs):
        decoded.append(True)
        return real_load(*args, **kwargs)
    monkeypatch.setattr(SpecSource, 'load_frame', counted)
    result = run_rsm_operation_v2(request)
    assert result.terminal.diagnostic == 'RSM output path authority changed after Preview'
    assert result.terminal.diagnostic == str(original.value)
    assert result.terminal.code == 'OUTPUT_IDENTITY_MISMATCH'
    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert not decoded and not Path(form.output_path).exists()
    _page_projection(result, 'rsm')
