# -*- coding: utf-8 -*-
"""VNX-FIXED-OPERATION-SLOTS-20260903 — ordinary Run adopts `_int1d`/`_int2d`.

Pins the Run half of the stable slot vocabulary: which slot each processing mode
publishes into, that an unrecognised mode is refused rather than defaulted, and
that the ROOT FAMILY recorded on the planned output agrees exactly with the stem
the target was named from.

That last one is the load-bearing row. The family is what every LATER operation
consumes; if it ever disagreed with the stem used for the filename, a
Reintegrate would resolve a slot in a family the Run never wrote.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from xdart.gui.tabs.scattering.output_preflight import (
    _RUN_MODE_SLOTS,
    _resolved_generated_target,
    _run_artifact_family,
    _run_output_slot,
)


def _configuration(mode: str, save_path: str) -> SimpleNamespace:
    """A stand-in carrying only what target selection reads.

    `_run_output_slot` takes the FrozenRunConfiguration branch for anything that
    is not an exact OutputCandidate, so a namespace exercises the real branch.
    """
    return SimpleNamespace(processing_mode=mode, save_path=save_path)


@pytest.mark.parametrize(
    ("mode", "slot"),
    (("Int 1D", "_int1d"), ("Int 2D", "_int2d"), ("Int 1D (XYE)", "_int1d")),
)
def test_each_run_mode_publishes_into_its_own_slot(tmp_path, mode, slot):
    configuration = _configuration(mode, str(tmp_path))
    assert _run_output_slot(configuration) == slot
    resolved = _resolved_generated_target(str(tmp_path), "scan12", slot)
    assert resolved == tmp_path / f"scan12{slot}.nexus"


def test_the_run_slot_table_is_exactly_the_native_run_modes():
    """No mode may be added to the table without a deliberate decision.

    `Int 1D (XYE)` writes NO `.nexus` at all -- the sink construction is guarded
    behind `if not xye_only:` -- but a target is still resolved for it as a
    grouping key, so it maps rather than raising.
    """
    assert set(_RUN_MODE_SLOTS) == {"Int 1D", "Int 2D", "Int 1D (XYE)"}
    assert set(_RUN_MODE_SLOTS.values()) == {"_int1d", "_int2d"}


@pytest.mark.parametrize(
    "mode", ("1D Viewer", "2D Viewer", "int 1d", "Int 3D"),
)
def test_a_declared_but_unknown_mode_is_refused_not_defaulted(tmp_path, mode):
    """A wrong slot is a run writing to the wrong file; fail loudly instead.

    Note the case sensitivity: `"int 1d"` is refused, not accepted. The mode
    strings are exact vocabulary, and a near-miss is more likely a bug than a
    spelling the policy should tolerate.
    """
    with pytest.raises(ValueError, match="no stable output slot"):
        _run_output_slot(_configuration(mode, str(tmp_path)))


def test_an_absent_run_mode_is_refused_like_an_unknown_one():
    """Fable F2 on `4fe073e8`: an empty mode used to take NO slot.

    The old row enshrined that, on the argument that `from_start_capture` signs
    every candidate.  That named the wrong guard -- the branch fires on the
    FROZEN configuration's `processing_mode`, which `RunIntent.__post_init__`
    and `freeze()` both accept as `""`.  The reviewer drove a real admission
    with it and got an `AdmissionReceipt` planning the un-slotted
    `scan_0001.nexus`; only the disabled Start button stood in the way.

    An absent mode is an unrecognised mode.  Publishing under no slot at all is
    the same class of wrong as publishing under another mode's slot.
    """
    for absent in ("", "   "):
        with pytest.raises(ValueError, match="no stable output slot"):
            _run_output_slot(_configuration(absent, "/tmp/whatever"))


def test_1d_and_2d_runs_of_one_scan_do_not_share_a_file(tmp_path):
    """The two modes own separate slots, so neither can overwrite the other."""
    one = _resolved_generated_target(str(tmp_path), "scan12", "_int1d")
    two = _resolved_generated_target(str(tmp_path), "scan12", "_int2d")
    assert one != two
    assert {one.name, two.name} == {"scan12_int1d.nexus", "scan12_int2d.nexus"}
    # Neither is the pre-slot name, which is what makes this a rename at all.
    assert (tmp_path / "scan12.nexus") not in {one, two}


def test_the_recorded_family_matches_the_stem_the_target_was_named_from(tmp_path):
    """Family and filename must be derived from the SAME stem.

    Checked for both shapes of `save_path`: a directory request (the family is
    the scan name) and an explicit file request (the family is its stem). If
    these ever diverge, the persisted family names a family whose slot file does
    not exist.
    """
    for save_path, scan_name, expected_family in (
        (str(tmp_path), "scan12", "scan12"),
        (str(tmp_path / "chosen.nexus"), "scan12", "chosen"),
        (str(tmp_path / "chosen.h5"), "scan12", "chosen"),
    ):
        configuration = _configuration("Int 2D", save_path)
        family = _run_artifact_family(configuration, scan_name)
        assert family == expected_family
        target = _resolved_generated_target(
            save_path, scan_name, _run_output_slot(configuration),
        )
        # THE INVARIANT: the filename is exactly `<recorded family><slot>.nexus`.
        assert target.name == f"{family}_int2d.nexus"


@pytest.mark.parametrize(
    "sub_folder", ("2026.09.04", "run.001", "sample1.5V", "plain"),
)
def test_a_dotted_source_sub_folder_cannot_split_family_from_filename(
    tmp_path, sub_folder,
):
    """Fable F1 on `4fe073e8`, the row the blind invariant above could not see.

    A recursive directory run places each source beneath its own parent, so the
    per-candidate output directory used to be handed down as a REQUEST STRING
    and re-inspected for a suffix.  A sub-folder named `2026.09.04` reads as a
    suffix, so it was treated as an explicit FILE request: the run published
    `processed/2026.09_int2d.nexus` -- collapsed into the parent, stem truncated
    -- while recording the family `scan_0001`.  A later Reintegrate then
    resolved `processed/scan_0001_reintegrate1d.nexus`, a slot in a family whose
    Run file does not exist.

    The row above cannot catch this: it passes `save_path` itself as the
    request, so its two inputs never differ.  This one drives the naming owner
    the way the directory planner does, with a separate per-candidate directory.
    """
    from xdart.gui.tabs.scattering.output_preflight import (
        _generated_target_in,
        _run_output_naming,
    )

    configuration = _configuration("Int 2D", str(tmp_path / "processed"))
    output_directory = tmp_path / "processed" / sub_folder
    directory, family = _run_output_naming(
        configuration, "scan_0001", output_directory,
    )
    target = _generated_target_in(
        directory, family, _run_output_slot(configuration),
    )

    # The directory is used AS a directory: the file lands inside the dotted
    # sub-folder, not collapsed into its parent with a truncated stem.
    assert directory == output_directory
    assert target.parent == output_directory
    # And the invariant holds regardless of what the folder is called.
    assert family == "scan_0001"
    assert target.name == f"{family}_int2d.nexus"


def test_an_explicit_file_request_still_gets_a_slot(tmp_path):
    """A chosen filename supplies the FAMILY; the slot is still appended.

    Under ADR-0010 the public name is always `<family><slot>.nexus`, so honouring
    a caller's bare filename would be the one bypass the vocabulary cannot
    tolerate. A non-`.nexus` suffix is normalised as before.
    """
    for requested in ("chosen.nexus", "chosen.h5", "chosen.nxs"):
        resolved = _resolved_generated_target(
            str(tmp_path / requested), "scan12", "_int2d",
        )
        assert resolved == tmp_path / "chosen_int2d.nexus"


def test_a_name_that_cannot_be_a_family_is_refused_before_the_run(tmp_path):
    """RULED 2026-09-04: refuse at PLAN time, not after hours of integration.

    This row previously PINNED the opposite -- that a non-canonical stem was
    recorded verbatim -- and set out both candidate remedies for the maintainer.
    The ruling took the plan-time refusal.

    Why it matters: the family is written INTO the artifact for every later
    operation to consume, and a name that breaks the rule cannot resolve
    `<family><slot>.nexus` at all.  Recorded silently, it produced a Run that
    succeeded and an Average that then refused with "artifact family is not
    canonical" -- a message naming nothing the operator could act on, arriving
    after the work was already done.

    ACCEPTED COST, stated: a run whose own file would have been written fine now
    refuses.  `_draft.nexus` is the realistic trigger; over-80-character names
    are possible but unusual -- a 68-character beamline stem passes.
    """
    for stem in ("_draft", ".hidden", "-dash", "trailing "):
        configuration = _configuration("Int 1D", str(tmp_path / f"{stem}.nexus"))
        with pytest.raises(ValueError, match="as a processed-result name"):
            _run_artifact_family(configuration, "unused-scan-name")

    # ...and an ordinary beamline stem is untouched, including a long one.
    for stem in ("scan12", "Sample A 001", "2026_beamtime_sampleA_gi_0p15deg_0001"):
        configuration = _configuration("Int 1D", str(tmp_path / f"{stem}.nexus"))
        assert _run_artifact_family(configuration, "unused") == stem
