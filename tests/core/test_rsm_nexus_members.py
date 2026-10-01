"""AUD-9 stage 2b C5c: raw NeXus stacks as RSM members (the maintainer, 2026-10-01: "there should be the
capability of using nexus files as source as well. And directory mode just like stitch").

A synthetic Bluesky NXWriter entry, written once (the identity-fixture technique): the six psic positioners
near scan 43's pose, five Pilatus-shaped uint32 frames, the energy in the Eiger configuration (eV), and in one
file scan 43's own #G3 as ``sample/ub_matrix``. No real-file HKL claim: the 17-2 NeXus test files record no UB.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.rsm_operation import (
    RSMCoordinateFrame, RSMNormalizationMode, RSMNormalizationPolicy, RSMUBOverride, run_rsm_operation_v2,
)
from xrd_tools.core.scan import SourceKind
from xdart.gui.tools.rsm_values import (
    RSMFrameSelector, RSMNexusMemberForm, RSMToolFormV2, RSMToolPreflightRefused, prepare_rsm_tool_v2,
    rsm_tool_preset,
)
from xrd_tools.analysis.rsm_geometry_asset import CANONICAL_RSM_GEOMETRY_LOCATOR, rsm_geometry_asset_input
from tests.core.test_bluesky_nexus import _write_bluesky_nxwriter
from tests.core.test_rsm_operation_r2_real_data import rsm_r2_project  # noqa: F401 (the fixture)

#: scan 43's own UB (STO_align, ``#G3``), row by row in SPEC's convention (2π included)
_SCAN43_G3 = (0.004218049485, 2.17211105e-08, -1.886835968, 1.611067634, -2.593377176e-16, 0.004940057955,
              5.686937195e-11, -1.611073156, -2.543905112e-08)
_ENERGY_EV = 15000.0
_FRAMES = 5


def _write_rsm_nexus(path: Path, *, ub: bool) -> Path:
    _write_bluesky_nxwriter(path, n=_FRAMES)
    angles = {"mu": np.full(_FRAMES, -6.883), "eta": np.linspace(42.34, 42.44, _FRAMES),
              "chi": np.full(_FRAMES, 200.29), "phi": np.zeros(_FRAMES), "nu": np.zeros(_FRAMES),
              "del": np.full(_FRAMES, 47.767)}
    frames = np.random.default_rng(43).poisson(20.0, size=(_FRAMES, 195, 487)).astype(np.uint32)
    with h5py.File(path, "r+") as stream:
        config = stream["entry/instrument/bluesky/metadata/configuration/eiger/data"]
        config["eiger_cam_photon_energy"][()] = _ENERGY_EV
        config["eiger_cam_wavelength"][()] = 12398.41984 / _ENERGY_EV
        for name, values in angles.items():
            stream["entry/data"].create_dataset(name, data=values)
            positioner = stream["entry/instrument/positioners"].create_group(name)
            positioner.attrs["NX_class"] = "NXpositioner"
            positioner.create_dataset("value", data=values)
        del stream["entry/data/eiger_image"]
        del stream["entry/instrument/detectors/eiger/data"]
        image = stream["entry/data"].create_dataset("eiger_image", data=frames)
        image.attrs["signal_type"] = "detector"
        stream["entry/instrument/detectors/eiger/data"] = image
        if ub:
            sample = stream["entry"].require_group("sample")
            sample.attrs["NX_class"] = "NXsample"
            sample.create_dataset("ub_matrix", data=np.asarray(_SCAN43_G3).reshape(3, 3))
    return path


@pytest.fixture(scope="module")
def nexus_project(rsm_r2_project):
    for name, ub in (("member_ub.nxs", True), ("member_noub.nxs", False)):
        if not (rsm_r2_project / name).exists():  # written once
            _write_rsm_nexus(rsm_r2_project / name, ub=ub)
    return rsm_r2_project


def _form(project: Path, name: str, *, frame=RSMCoordinateFrame.HKL, ub=None, motors=None) -> RSMToolFormV2:
    preset = rsm_tool_preset()
    member = RSMNexusMemberForm(project / name, SourceKind.NEXUS_STACK, "entry",
                                RSMFrameSelector(0, _FRAMES - 1, 1), preset.detector_shape,
                                motors or preset.motor_selectors)
    identity = RSMNormalizationPolicy(RSMNormalizationMode.IDENTITY, None, None, (0.0, 0.0, 0.0, 0.0))
    stem = Path(name).stem + ("-q" if frame is RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU else "")
    return RSMToolFormV2(
        project, rsm_geometry_asset_input(CANONICAL_RSM_GEOMETRY_LOCATOR), (member,), preset.conditioning,
        identity, (20, 20, 20), preset.chunk_size, preset.max_frame_bytes, None,
        project / f"rsm-c5c-{stem}{'-given' if ub is not None else ''}.nexus",
        coordinate_frame=frame, ub=ub)


@pytest.mark.slow
def test_a_nexus_member_with_a_ub_grids_in_hkl_and_the_same_ub_given_matches_it(nexus_project):
    """The NeXus entry's own UB (read in SPEC's #G3 convention) and the same nine numbers given at Preview to
    an entry without one grid to the same result fingerprint; the member records its entry, kind and the UB
    convention in its source options, no scan, and the energy in eV (the reader's keV times 1000: a missing
    factor would read 15 eV)."""
    request = prepare_rsm_tool_v2(_form(nexus_project, "member_ub.nxs")).request
    member, = request.preflight.members
    assert member.source_scan == ""
    assert json.loads(member.source_options_json) == {"entry": "entry", "kind": "nexus_stack",
                                                      "ub_convention": "spec-g3"}
    assert member.energy_eV == _ENERGY_EV
    assert np.array_equal(np.asarray(member.ub), np.asarray(_SCAN43_G3).reshape(3, 3))
    own = run_rsm_operation_v2(request)
    assert own.terminal.disposition is ModuleDisposition.COMMITTED, own.terminal
    given = run_rsm_operation_v2(prepare_rsm_tool_v2(
        _form(nexus_project, "member_noub.nxs", ub=RSMUBOverride(matrix=_SCAN43_G3))).request)
    assert given.terminal.disposition is ModuleDisposition.COMMITTED, given.terminal
    assert own.payload.result_fingerprint == given.payload.result_fingerprint


@pytest.mark.slow
def test_a_nexus_member_without_a_ub_is_refused_in_hkl_and_maps_in_q(nexus_project):
    with pytest.raises(RSMToolPreflightRefused) as refused:
        prepare_rsm_tool_v2(_form(nexus_project, "member_noub.nxs"))
    assert refused.value.code == "RSM_UB_MISSING" and "NeXus entry records no UB" in str(refused.value)
    result = run_rsm_operation_v2(prepare_rsm_tool_v2(
        _form(nexus_project, "member_noub.nxs", frame=RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU)).request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal


@pytest.mark.slow
def test_a_psic_role_mapped_to_a_column_the_entry_lacks_is_named(nexus_project):
    from xrd_tools.analysis.module_transaction import MetadataColumnSelector
    preset = rsm_tool_preset()
    motors = tuple((role, MetadataColumnSelector("omega") if role == "mu" else selector)
                   for role, selector in preset.motor_selectors)
    with pytest.raises(RSMToolPreflightRefused) as refused:
        prepare_rsm_tool_v2(_form(nexus_project, "member_ub.nxs", motors=motors))
    assert refused.value.code == "RSM_MOTOR_MISSING"
    assert "entry entry of member_ub.nxs: the psic role mu maps to column 'omega'" in str(refused.value)


@pytest.mark.slow
def test_the_preview_names_a_nexus_members_entry_and_its_ub_convention(nexus_project):
    from xdart.gui.tools.rsm_values import rsm_preflight_summary_lines
    lines = rsm_preflight_summary_lines(prepare_rsm_tool_v2(_form(nexus_project, "member_ub.nxs")).summary)
    member = next(line for line in lines if line.startswith("Member 1:"))
    assert "member_ub.nxs · NeXus entry entry · 5 frames" in member
    assert any("read in SPEC's #G3 convention" in line for line in lines)
