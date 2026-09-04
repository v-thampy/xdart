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


def test_no_declared_mode_takes_no_slot_rather_than_guessing(tmp_path):
    """An undeclared mode yields NO slot; it does not pick one.

    Distinct from the row above, and the distinction is the safety property.
    A DECLARED mode this policy does not know is dangerous -- continuing would
    publish under some other mode's slot -- so it raises. Nothing declared at
    all is a source-shaped double that names a target without ever saying how it
    would be reduced; inventing `_int2d` there would name a file for a mode
    nobody chose. Every GUI candidate is signed by `from_start_capture` and
    always carries a mode, so this branch is not reachable from a real run.
    """
    assert _run_output_slot(_configuration("", str(tmp_path))) == ""
    assert _resolved_generated_target(str(tmp_path), "scan12", "") == (
        tmp_path / "scan12.nexus"
    )


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
