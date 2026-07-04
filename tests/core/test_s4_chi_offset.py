"""S-4 — chi_offset flows to Integration1DPlan.azimuth_offset (mirror the 2D),
NOT a shift of the input azimuth range.

GI-1 "auto == explicit -180..180" held in GI mode but FAILED in STANDARD mode
because chi_offset (default 90 deg) shifted the 1D INPUT range and left the 1D
chi OUTPUT axis in the raw pyFAI frame, 90 deg out of frame with the 2D cake chi
(which re-adds the offset to its output).  These drive the real plan builder.
"""

from xrd_tools.session.readiness import (
    build_native_int_reduction_plan_from_args as build,
)


def test_standard_chi_offset_becomes_1d_azimuth_offset_range_unshifted():
    a1 = {"unit": "chi_deg", "chi_offset": 90.0,
          "azimuth_range": (-180.0, 180.0), "npt": 180}
    plan = build(a1, {"unit": "q_A^-1"}, gi_enabled=False)
    p1 = plan.integration_1d
    assert p1.azimuth_offset == 90.0             # carried onto the plan...
    assert p1.azimuth_range == (-180.0, 180.0)   # ...NOT popped-and-shifted


def test_standard_1d_and_2d_offsets_agree():
    plan = build({"unit": "chi_deg", "chi_offset": 90.0},
                 {"unit": "q_A^-1", "azimuth_offset": 90.0}, gi_enabled=False)
    assert (plan.integration_1d.azimuth_offset
            == plan.integration_2d.azimuth_offset == 90.0)


def test_zero_chi_offset_is_a_noop():
    plan = build({"unit": "chi_deg", "azimuth_range": (-180.0, 180.0)},
                 {"unit": "q_A^-1"}, gi_enabled=False)
    assert plan.integration_1d.azimuth_offset == 0.0
    assert plan.integration_1d.azimuth_range == (-180.0, 180.0)
