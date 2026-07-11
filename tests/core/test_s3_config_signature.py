"""S-3 — the Append config signature includes value-affecting, grid-preserving
params, compared BACKWARD-TOLERANTLY.

mask/PONI/chi_offset/monitor/polarization/error_model/GI-incidence change the
WRITTEN NUMBERS while leaving the axis/npt/range grid identical, so they pass
both the Run-click modal and the axis backstop -> mixed provenance under a
/entry/reduction that claims the first run's config.  A field ABSENT from a
pre-upgrade stored config compares as unknown (no false-positive modal on every
existing .nxs).  Production-wired: the real append_config_mismatch_check on real
config mappings, no monkeypatch.
"""

from xrd_tools.session.readiness import (
    AppendConfigMismatchError,
    append_config_mismatch_check,
)


def _cfg(a1=None, a2=None):
    base = {"unit": "q_A^-1"}
    return {"bai_1d_args": {**base, **(a1 or {})},
            "bai_2d_args": {**base, **(a2 or {})}}


def _blocks(stored, current):
    return not append_config_mismatch_check("Append", stored, current).ok


# ── value-affecting params now block a mixed Append (both sides present) ──────
def test_chi_offset_change_blocks_append():
    assert _blocks(_cfg({"chi_offset": 0.0}), _cfg({"chi_offset": 90.0}))


def test_monitor_change_blocks_append():
    assert _blocks(_cfg({"monitor": "i0"}), _cfg({"monitor": "i1"}))


def test_polarization_change_blocks_append():
    assert _blocks(_cfg({"polarization_factor": 0.95}),
                   _cfg({"polarization_factor": 0.99}))


def test_error_model_change_blocks_append():
    assert _blocks(_cfg({"error_model": "poisson"}), _cfg({"error_model": "azimuthal"}))


def test_gi_incidence_change_blocks_append():
    stored = {"bai_1d_args": {"unit": "q_A^-1", "gi_mode_1d": "chi_gi"},
              "bai_2d_args": {}, "gi": True, "gi_config": {"th_val": 0.2}}
    current = {"bai_1d_args": {"unit": "q_A^-1", "gi_mode_1d": "chi_gi"},
               "bai_2d_args": {}, "gi": True, "gi_config": {"th_val": 0.5}}
    assert _blocks(stored, current)


# ── backward tolerance: absent-on-either-side is skipped ─────────────────────
def test_stored_missing_field_is_backward_tolerant():
    # a pre-upgrade .nxs never recorded chi_offset -> no false modal
    assert not _blocks(_cfg(), _cfg({"chi_offset": 90.0}))


def test_current_missing_field_is_backward_tolerant():
    assert not _blocks(_cfg({"monitor": "i0"}), _cfg())


def test_matching_value_affecting_params_pass():
    assert not _blocks(_cfg({"chi_offset": 90.0, "monitor": "i0"}),
                       _cfg({"chi_offset": 90.0, "monitor": "i0"}))


# ── no regression on the existing grid fields ────────────────────────────────
def test_npt_change_still_blocks():
    assert _blocks(_cfg({"npt": 1000}), _cfg({"npt": 2000}))


def test_non_append_mode_never_blocks():
    assert append_config_mismatch_check(
        "Replace", _cfg({"chi_offset": 0.0}), _cfg({"chi_offset": 90.0})).ok


# ── mid-run delivery: the typed error + a reason that names WHAT changed ─────
def test_mismatch_reason_names_the_differing_fields():
    # The beamline case: Int 1D -> Int 2D mid-run, BOTH sides Standard mode.
    # display_mode alone read "processed: Standard · current: Standard" — true
    # but useless; the reason must name the actual differing field(s).
    check = append_config_mismatch_check(
        "Append", _cfg(a2={"npt_rad": 500}), _cfg(a2={"npt_rad": 1000}))

    assert check.ok is False
    assert check.mismatched_fields == ("2D radial points",)
    assert (check.processed_label, check.current_label) == (
        "Standard", "Standard")
    for label in check.mismatched_fields:
        assert label in check.reason
    assert "switch write mode to Replace" in check.reason


def test_append_config_mismatch_error_carries_check():
    # The typed error the run loop catches to stop cleanly (instead of a bare
    # RuntimeError escaping the worker thread): still a RuntimeError subclass
    # for pre-existing broad handlers, and it carries the full check so the
    # GUI can name the differing fields without re-deriving the comparison.
    check = append_config_mismatch_check(
        "Append", _cfg(a2={"npt_rad": 500}), _cfg(a2={"npt_rad": 1000}))
    err = AppendConfigMismatchError(check.reason, check)

    assert isinstance(err, RuntimeError)
    assert err.check is check
    assert err.check.mismatched_fields == ("2D radial points",)
    assert "2D radial points" in str(err)


def test_gi_chi_offset_change_does_not_block_append():
    # Review follow-up: chi_offset is INERT for GI (S-4 zeroes the GI
    # azimuth_offset; GI chi uses FiberIntegrator's own convention), so changing
    # it on a GI scan must NOT trip the Append modal -- it does not alter written
    # GI data.  Standard mode still blocks (test_chi_offset_change_blocks_append).
    stored = {"bai_1d_args": {"unit": "q_A^-1", "chi_offset": 0.0},
              "bai_2d_args": {"unit": "q_A^-1"}, "gi": True,
              "gi_config": {"th_val": 0.2}}
    current = {"bai_1d_args": {"unit": "q_A^-1", "chi_offset": 90.0},
               "bai_2d_args": {"unit": "q_A^-1"}, "gi": True,
               "gi_config": {"th_val": 0.2}}
    assert append_config_mismatch_check("Append", stored, current).ok
