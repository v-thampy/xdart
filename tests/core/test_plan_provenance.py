"""StitchPlan / RSMPlan provenance() ↔ from_provenance() round-trip.

The reload half of the GUI section-3 contract: a plan's processing options
serialize to a provenance dict (persisted in the .nxs) and rebuild from it. The
geometry/mask/UB are NOT in provenance (they persist separately) — from_provenance
reattaches whatever is passed.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from xrd_tools.analysis.plans import RSMPlan, StitchPlan
from xrd_tools.core.containers import PONI
from xrd_tools.corrections.grazing import (
    GICorrectionStack,
    GI_EXIT_ANGLE_CONVENTION,
    GISettings,
    LEGACY_GI_EXIT_ANGLE_CONVENTION,
)
from xrd_tools.corrections.stack import CorrectionStack


def test_stitchplan_provenance_roundtrip():
    plan = StitchPlan(
        base_poni=PONI(0.2, 0.01, 0.02, wavelength=1.0e-10),
        backend="pyfai_hist", mode="2d", unit="q_A^-1",
        npt_1d=1234, npt_rad_2d=800, npt_azim_2d=360,
        radial_range=(0.1, 5.0), azimuth_range=(-180.0, 180.0),
        monitor_key="i0",
        corrections=CorrectionStack(solid_angle=True, polarization_factor=0.97),
        gi=GISettings(corrections=GICorrectionStack(material="Si", energy_eV=10000.0,
                                                    footprint=True),
                      incident_angle_deg=0.3, sample_orientation=3, tilt_deg=1.0),
    )
    back = StitchPlan.from_provenance(plan.provenance())
    assert back.backend == "pyfai_hist" and back.mode == "2d"
    assert back.npt_1d == 1234 and back.npt_rad_2d == 800 and back.npt_azim_2d == 360
    assert back.radial_range == (0.1, 5.0) and back.azimuth_range == (-180.0, 180.0)
    assert back.monitor_key == "i0"
    assert back.corrections.solid_angle is True
    assert back.corrections.polarization_factor == 0.97
    assert back.gi.incident_angle_deg == 0.3 and back.gi.sample_orientation == 3
    assert back.gi.corrections.material == "Si" and back.gi.corrections.footprint is True
    assert back.gi.gi_exit_angle_convention == GI_EXIT_ANGLE_CONVENTION
    assert back.gi.integrator_wavelength_m == 1.0e-10
    assert back.gi.corrections.energy_eV == 10000.0
    # geometry/mask are NOT in provenance — None unless reattached
    assert back.diffractometer is None and back.mask is None


def test_stitchplan_from_provenance_reattaches_geometry():
    prov = StitchPlan(backend="pyfai_hist", npt_1d=500).provenance()
    sentinel = object()
    mask = np.zeros((4, 4), dtype=bool)
    back = StitchPlan.from_provenance(prov, diffractometer=sentinel, mask=mask)
    assert back.diffractometer is sentinel
    assert back.mask is mask and back.npt_1d == 500


def test_stitchplan_provenance_roundtrip_no_corrections():
    plan = StitchPlan(backend="multigeometry", mode="1d", npt_1d=2000)
    back = StitchPlan.from_provenance(plan.provenance())
    assert back.corrections is None and back.gi is None
    assert back.backend == "multigeometry" and back.npt_1d == 2000


def test_rsmplan_provenance_roundtrip():
    plan = RSMPlan(
        mapper=None, bins=(64, 65, 66), diff_motors=("mu", "eta", "nu", "del"),
        energy=10500.0, q_bounds=((-1.0, 1.0), (0.0, 2.0), (-0.5, 0.5)),
        roi=(2, 30, 4, 40), chunk_size=4,
        corrections=CorrectionStack(solid_angle=True),
        gi=GISettings(corrections=GICorrectionStack(material="Si", energy_eV=10500.0),
                      incident_angle_deg=0.25),
    )
    back = RSMPlan.from_provenance(plan.provenance())
    assert back.bins == (64, 65, 66)
    assert back.diff_motors == ("mu", "eta", "nu", "del")
    assert back.energy == 10500.0
    assert back.q_bounds == ((-1.0, 1.0), (0.0, 2.0), (-0.5, 0.5))
    assert back.roi == (2, 30, 4, 40) and back.chunk_size == 4
    assert back.corrections.solid_angle is True
    assert back.gi.incident_angle_deg == 0.25
    from xrd_tools.core.energy import energy_eV_to_wavelength_m
    assert back.gi.gi_exit_angle_convention == GI_EXIT_ANGLE_CONVENTION
    assert back.gi.integrator_wavelength_m == energy_eV_to_wavelength_m(10500.0)
    assert back.gi.corrections.energy_eV == 10500.0
    # mapper/UB/mask reattached, not from provenance
    assert back.mapper is None


def test_rsmplan_from_provenance_reattaches_mapper():
    prov = RSMPlan(mapper=None, bins=(8, 8, 8)).provenance()
    sentinel = object()
    back = RSMPlan.from_provenance(prov, mapper=sentinel, UB=np.eye(3))
    assert back.mapper is sentinel
    assert back.UB is not None and back.bins == (8, 8, 8)


def test_frozen_gi_exact_current_and_legacy_keysets_and_identity():
    from xrd_tools.session.run_configuration import (
        FrozenGIConfiguration,
        GIIntent,
        RunIntent,
    )

    current_run = RunIntent(
        gi=GIIntent(enabled=True, incidence_motor="Manual", th_val=0.3),
    ).freeze()
    current_map = current_run.gi.as_dict()
    assert len(current_map) == 9
    assert current_map["gi_exit_angle_convention"] == GI_EXIT_ANGLE_CONVENTION
    assert len(current_run._content_fingerprint_value()[8]) == 9
    snapshot = current_run.native_int_snapshot()
    assert snapshot["gi_exit_angle_convention"] == GI_EXIT_ANGLE_CONVENTION
    assert snapshot["gi_config"]["gi_exit_angle_convention"] == (
        GI_EXIT_ANGLE_CONVENTION
    )

    historical_map = dict(current_map)
    historical_map.pop("gi_exit_angle_convention")
    legacy_gi = FrozenGIConfiguration.from_dict(historical_map)
    assert legacy_gi.gi_exit_angle_convention == LEGACY_GI_EXIT_ANGLE_CONVENTION
    assert legacy_gi.as_dict() == historical_map
    legacy_run = replace(current_run, gi=legacy_gi)
    assert len(legacy_run._content_fingerprint_value()[8]) == 8
    assert legacy_run.fingerprint != current_run.fingerprint
    legacy_snapshot = legacy_run.native_int_snapshot()
    assert legacy_snapshot["gi_exit_angle_convention"] == (
        LEGACY_GI_EXIT_ANGLE_CONVENTION
    )
    assert "gi_exit_angle_convention" not in legacy_snapshot["gi_config"]

    with pytest.raises(ValueError, match="unsupported keyset"):
        FrozenGIConfiguration.from_dict({**current_map, "third_schema": True})
    with pytest.raises(ValueError, match="unsupported GI exit-angle"):
        FrozenGIConfiguration.from_dict({
            **current_map,
            "gi_exit_angle_convention": "upstream_exit_map",
        })


def test_non_gi_fingerprint_and_eight_key_mapping_ignore_convention_marker():
    from xrd_tools.session.run_configuration import GIIntent, RunIntent

    current = RunIntent(gi=GIIntent(enabled=False)).freeze()
    legacy = RunIntent(gi=GIIntent(
        enabled=False,
        gi_exit_angle_convention=LEGACY_GI_EXIT_ANGLE_CONVENTION,
    )).freeze()
    assert current.gi.as_dict() == legacy.gi.as_dict()
    assert len(current.gi.as_dict()) == 8
    assert current.fingerprint == legacy.fingerprint
