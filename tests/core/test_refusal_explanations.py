"""Frozen P1 E1: actual refusals and publication rollback. The two rows that also project a refusal
onto the Stitch and RSM pages need Qt, so they run with the offscreen GUI tests
(``tests/xdart/scattering/test_refusal_explanations_gui.py``)."""
from dataclasses import replace
from pathlib import Path

import numpy as np

from xrd_tools.analysis.module_transaction import ModuleDisposition
from tests.core.test_native_stitch_operation import _native


def test_module_prepublication_bridge_retains_stitch_reason(tmp_path, monkeypatch):
    import xrd_tools.analysis.stitch_operation as owner
    from xrd_tools.io.analysis_artifact import AnalysisArtifactOverwrite
    request = _native(tmp_path, 'multigeometry', 'q_A^-1', grouped=False)
    owner.run_stitch_operation(request)  # the prior output must be an actual qualified artifact
    output = replace(request.module.output, overwrite=AnalysisArtifactOverwrite.REPLACE)
    request = owner.prepare_stitch_operation(request.module.source, output, request.plan, project_root=tmp_path)
    target = Path(output.target)
    before = target.read_bytes()
    real_write = owner._write_record
    written = []
    def write_then_drift(*args, **kwargs):
        real_write(*args, **kwargs)
        written.append(True)
        np.save(request.plan.mask_path, np.ones((195, 487), dtype=np.uint8))
    monkeypatch.setattr(owner, '_write_record', write_then_drift)
    result = owner.run_stitch_operation(request)
    assert result.terminal.diagnostic == 'mask no longer matches Preview'
    assert written == [True]
    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == 'MASK_IDENTITY_MISMATCH'
    assert target.read_bytes() == before
    assert not list(target.parent.glob('*.pending*'))
    # Restoring the actual mask permits a new owner to admit the same output slot.
    mask = np.zeros((195, 487), dtype=np.uint8); mask[20:35] = 1
    np.save(request.plan.mask_path, mask)
    monkeypatch.setattr(owner, '_write_record', real_write)
    fresh = owner.prepare_stitch_operation(request.module.source, output, replace(request.plan), project_root=tmp_path)
    retry = owner.run_stitch_operation(fresh)
    assert retry.terminal.disposition is ModuleDisposition.COMMITTED, retry.terminal
