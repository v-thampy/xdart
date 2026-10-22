"""AUD-9 stage 2b C5: a UB given for an HKL map, typed or from a named SPEC scan, replaces the sources' own; a
source with no UB refuses HKL (``RSM_UB_MISSING``) and maps in Q (the maintainer, 2026-10-01: "For files with no
ub matrix, a warning with a chance to enter the ub matrix manually or through a spec file can be offered"). The
page's warning and rows are G8's."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.analysis.rsm_values import RSMToolPreflightRefused, prepare_rsm_tool_v2
from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.rsm_operation import (
    RSMFrameCorrections,
    RSMOperationRefused,
    RSMUBOverride,
    _bind_rsm_ub,
    _revalidate_rsm_ub,
    run_rsm_operation_v2,
)
from xrd_tools.io.spec import get_energy_and_UB
from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame
from xrd_tools.sources.spec import SpecSource
from tests.core.test_rsm_operation_r2 import _form, _write_member
from tests.core.test_rsm_operation_r2_real_data import _form as _real_form, rsm_r2_project  # noqa: F401

_IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
REFERENCE = "8a88826f85eda95bf9fb4eb9cd043b7b822e0827e4ec3d285203f9256f43b324"


def _without_ub(member) -> None:
    spec = Path(member.spec_path)
    spec.write_text("".join(line for line in spec.read_text().splitlines(keepends=True)
                            if not line.startswith("#G3")), encoding="utf-8")


def _synthetic_run(monkeypatch) -> None:
    """The synthetic rows' frames and q bounds, as the r2 science rows set them."""
    import xrd_tools.analysis.rsm_operation as rsm_operation
    monkeypatch.setattr(rsm_operation, "_resolve_exact_rsm_q_bounds_active",
                        lambda *_args, **_kwargs: ((-8.0, 8.0), (-8.0, 8.0), (-8.0, 8.0)))

    def full_detector_frame(self, index):
        ordinal = int(str(self.name).split(" [", 1)[0].rsplit("_", 1)[-1])
        return np.full((195, 487), 10 + ordinal + int(index), dtype=np.float64)
    monkeypatch.setattr(SpecSource, "load_frame", full_detector_frame)


def test_a_given_ub_is_bound_checked_and_refused_when_unusable(tmp_path):
    typed = _bind_rsm_ub(RSMUBOverride(matrix=_IDENTITY), RSMCoordinateFrame.HKL)
    assert typed.matrix == _IDENTITY and typed.spec_receipt is None
    assert typed.to_provenance() == {"kind": "manual", "matrix": list(_IDENTITY)}
    assert _bind_rsm_ub(None, RSMCoordinateFrame.HKL) is None
    for matrix in ((1.0, 0, 0, 0, 0, 0, 0, 0, 1.0), (float("nan"),) + _IDENTITY[1:]):  # singular; not finite
        with pytest.raises(RSMOperationRefused) as refused:
            _bind_rsm_ub(RSMUBOverride(matrix=matrix), RSMCoordinateFrame.HKL)
        assert refused.value.code == "RSM_UB_INVALID"
    with pytest.raises(RSMOperationRefused) as refused:
        _bind_rsm_ub(RSMUBOverride(matrix=_IDENTITY), RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU)
    assert refused.value.code == "RSM_UB_OVERRIDE_UNUSED"
    with pytest.raises(TypeError):
        RSMUBOverride(matrix=_IDENTITY, spec_path="x", scan="1")
    member = _write_member(tmp_path, 0, ub=(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0))
    from_spec = _bind_rsm_ub(RSMUBOverride(spec_path=str(member.spec_path), scan="1.1"), RSMCoordinateFrame.HKL)
    assert from_spec.matrix == (2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0)
    provenance = from_spec.to_provenance()
    assert (provenance["kind"], provenance["name"], provenance["scan"]) == ("spec", "RSM_synth_0", "1.1")
    # a bare scan number is its first occurrence, as Stitch's sources key it; a scan the file lacks is refused
    bare_number = _bind_rsm_ub(RSMUBOverride(spec_path=str(member.spec_path), scan=" 1 "), RSMCoordinateFrame.HKL)
    assert bare_number.to_provenance()["scan"] == "1.1" and bare_number.identity() == from_spec.identity()
    with pytest.raises(RSMOperationRefused) as refused:
        _bind_rsm_ub(RSMUBOverride(spec_path=str(member.spec_path), scan="7"), RSMCoordinateFrame.HKL)
    assert refused.value.code == "RSM_UB_INVALID"
    _revalidate_rsm_ub(from_spec)  # unchanged: no refusal
    Path(member.spec_path).write_text(Path(member.spec_path).read_text() + "\n", encoding="utf-8")
    with pytest.raises(RSMOperationRefused) as refused:
        _revalidate_rsm_ub(from_spec)
    assert refused.value.code == "SOURCE_IDENTITY_MISMATCH"
    bare = _write_member(tmp_path, 1)
    _without_ub(bare)
    with pytest.raises(RSMOperationRefused) as refused:
        _bind_rsm_ub(RSMUBOverride(spec_path=str(bare.spec_path), scan="1.1"), RSMCoordinateFrame.HKL)
    assert refused.value.code == "RSM_UB_MISSING"


def test_a_source_without_a_ub_refuses_hkl_maps_in_q_and_takes_a_given_ub(tmp_path, monkeypatch):
    _synthetic_run(monkeypatch)
    member = _write_member(tmp_path, 0)
    _without_ub(member)
    form = _form(tmp_path, (member,))
    with pytest.raises(RSMToolPreflightRefused) as refused:
        prepare_rsm_tool_v2(form)
    assert refused.value.code == "RSM_UB_MISSING"
    q_form = replace(form, coordinate_frame=RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU,
                     ub=RSMUBOverride(matrix=_IDENTITY))  # a Q form neither passes nor fingerprints a UB
    assert q_form.fingerprint == replace(form, coordinate_frame=RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU).fingerprint
    assert prepare_rsm_tool_v2(q_form).request.plan.ub_override is None
    given = replace(form, ub=RSMUBOverride(matrix=(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0)))
    assert given.fingerprint != form.fingerprint
    request = prepare_rsm_tool_v2(given).request
    assert request.plan.ub_override.matrix == (2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0)
    assert np.array_equal(np.asarray(request.preflight.members[0].ub), np.diag([2.0, 2.0, 2.0]))
    assert request.plan.to_provenance()["ub_source"]["kind"] == "manual"


def test_a_given_ub_equal_to_the_sources_reproduces_its_volume(tmp_path, monkeypatch):
    """A typed UB and a SPEC-scan UB with the source's own numbers grid the same volume (the result fingerprint
    covers the stored arrays); only the plan's identity and provenance name the override. A form without one
    keeps its identity and records no ``ub_source``."""
    _synthetic_run(monkeypatch)
    members = (_write_member(tmp_path, 0), _write_member(tmp_path, 1))
    form = _form(tmp_path, members)
    base = run_rsm_operation_v2(prepare_rsm_tool_v2(form).request)
    assert base.terminal.disposition is ModuleDisposition.COMMITTED, base.terminal
    assert "ub_source" not in json.loads(base.payload.provenance_json)["plan"]
    results = {}
    for name, ub in (("typed", RSMUBOverride(matrix=_IDENTITY)),
                     ("spec", RSMUBOverride(spec_path=str(members[1].spec_path), scan="1.1"))):
        run = replace(form, ub=ub, output_path=Path(tmp_path) / "output" / f"rsm-{name}.nexus")
        result = run_rsm_operation_v2(prepare_rsm_tool_v2(run).request)
        assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
        assert result.payload.result_fingerprint == base.payload.result_fingerprint, name
        assert json.loads(result.payload.provenance_json)["plan"]["ub_source"]["kind"] == ("manual" if name == "typed"
                                                                                            else "spec")
        results[name] = result


@pytest.mark.slow
def test_scan_43_with_its_own_ub_given_is_the_reference(rsm_r2_project):
    """Scan 43 at 40³, every correction off, with its own #G3 given typed and then from the SPEC scan: the
    reference result ``8a88826f…b324`` both times."""
    form = replace(_real_form(rsm_r2_project, (43,), 40), corrections=RSMFrameCorrections())
    member = form.members[0]
    _energy, ub = get_energy_and_UB(str(member.spec_path), member.scan)
    for name, given in (("typed", RSMUBOverride(matrix=tuple(float(value) for value in np.asarray(ub).ravel()))),
                        ("spec", RSMUBOverride(spec_path=str(member.spec_path), scan=member.scan))):
        run = replace(form, ub=given, output_path=Path(rsm_r2_project) / f"rsm-c5-{name}.nexus")
        result = run_rsm_operation_v2(prepare_rsm_tool_v2(run).request)
        assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
        assert result.payload.result_fingerprint == REFERENCE, name
