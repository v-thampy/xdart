"""AUD-9 stage 2b C5: RSM's sources through Stitch's own owner (``resolve_stitch_sources``: discovery, the Selection
syntax, Directory mode), one RSM per group (design §2: ``[42-44]`` one RSM from three scans, ``42,43`` two), and a
psic role whose column the scan lacks named (``RSM_MOTOR_MISSING``). NeXus members come with C5c."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from xrd_tools.analysis.rsm_values import (
    RSMFrameSelector,
    RSMToolPreflightRefused,
    prepare_rsm_tool_v2,
    rsm_member_groups,
)
from xrd_tools.analysis.module_transaction import MetadataColumnSelector
from xrd_tools.analysis.stitch_sources import StitchSourceSelection
from tests.core.test_rsm_operation_r2 import _ROLES, _form

_MOTORS = tuple((role, MetadataColumnSelector(role, 0)) for role in _ROLES)


def _scan(number: int, eta: float) -> str:
    return f"""#S {number} ascan eta {eta} {eta + 1} 1 1
#D Mon Jan 15 10:30:00 2024
#P0 13000.007 {10 + number} 20 30 40 50
#G3 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0
#N 4
#L eta  Seconds  Seconds  foil status
{eta} 1 10 101
{eta + 1} 2 20 110

"""


def _pair(root: Path, *, with_images=(1, 2), scans=(1, 2, 3)) -> Path:
    """One SPEC file, ``RSM_pair``, with three scans; images (Stitch's default stem ``RSM_pair_scanN_``) for
    the scans in ``with_images`` only."""
    root.mkdir(parents=True, exist_ok=True)
    spec = root / "RSM_pair"
    spec.write_text("#F RSM_pair\n#E 1\n#D Mon Jan 15 10:30:00 2024\n#O0 energy  mu  chi  phi  nu  del\n\n"
                    + "".join(_scan(number, float(number)) for number in scans), encoding="utf-8")
    for number in with_images:
        for label in range(2):
            tifffile.imwrite(root / f"RSM_pair_scan{number}_{label:04d}.tif",
                             np.arange(20, dtype=np.uint16).reshape(4, 5) + number + label)
    return spec


def _groups(selection):
    return rsm_member_groups(selection, frame_selector=RSMFrameSelector(0, 1, 1), detector_shape=(195, 487),
                             raw_dtype="uint16", raw_header_skip=0, motor_selectors=_MOTORS)


def test_stitch_selection_syntax_makes_rsm_member_groups(tmp_path):
    spec = _pair(tmp_path / "data")
    one, skipped = _groups(StitchSourceSelection("Spec", (str(spec),), "[1-2]"))
    assert [[member.scan for member in group] for group in one] == [["1.1", "2.1"]] and skipped == ()
    member = one[0][0]
    assert (Path(member.spec_path), Path(member.image_dir), member.image_stem) == (
        spec, spec.parent, "RSM_pair_scan1_")
    assert member.motor_selectors == _MOTORS and member.detector_shape == (195, 487)
    two, _skipped = _groups(StitchSourceSelection("Spec", (str(spec),), "1,2"))
    assert [[member.scan for member in group] for group in two] == [["1.1"], ["2.1"]]
    # a scan with no images is skipped, as Stitch skips it
    with_skip, skipped = _groups(StitchSourceSelection("Spec", (str(spec),), "[1-3]"))
    assert [[member.scan for member in group] for group in with_skip] == [["1.1", "2.1"]]
    assert [(Path(path).name, key) for path, key, _reason in skipped] == [("RSM_pair", "3.1")]
    # Directory mode: every scan of every SPEC file, one group a scan
    folder, _skipped = _groups(StitchSourceSelection("Directory", (str(spec.parent),)))
    assert [[member.scan for member in group] for group in folder] == [["1.1"], ["2.1"]]


def test_member_groups_prepare_as_one_rsm_each(tmp_path, monkeypatch):
    import xrd_tools.analysis.rsm_operation as rsm_operation
    monkeypatch.setattr(rsm_operation, "_resolve_exact_rsm_q_bounds_active",
                        lambda *_args, **_kwargs: ((-8.0, 8.0), (-8.0, 8.0), (-8.0, 8.0)))
    spec = _pair(tmp_path / "data")
    (group,), _skipped = _groups(StitchSourceSelection("Spec", (str(spec),), "[1-2]"))
    request = prepare_rsm_tool_v2(_form(tmp_path, group)).request
    assert [member.source_scan for member in request.preflight.members] == ["1.1", "2.1"]


def test_a_psic_role_mapped_to_a_column_the_scan_lacks_is_named(tmp_path):
    spec = _pair(tmp_path / "data")
    (group,), _skipped = _groups(StitchSourceSelection("Spec", (str(spec),), "1"))
    motors = tuple((role, MetadataColumnSelector("chi_missing" if role == "chi" else selector.name, 0))
                   for role, selector in _MOTORS)
    with pytest.raises(RSMToolPreflightRefused) as refused:
        prepare_rsm_tool_v2(_form(tmp_path, (replace(group[0], motor_selectors=motors),)))
    assert refused.value.code == "RSM_MOTOR_MISSING"
    assert "psic role chi" in str(refused.value) and "'chi_missing'" in str(refused.value)


def test_an_explicit_nexus_selection_is_a_nexus_member_since_c5c():
    """C5c: a raw NeXus stack is an RSM member (its path, kind and entry), where C5b refused it."""
    from xrd_tools.analysis.rsm_values import RSMNexusMemberForm
    root = Path(os.environ.get("XDART_TEST_DATA", "")) / "nexus"
    masters = sorted(root.glob("*_master.h5")) if root.is_dir() else []
    if not masters:
        pytest.skip("no NeXus master file in XDART_TEST_DATA")
    groups, _skipped = _groups(StitchSourceSelection("NeXus", (str(masters[0]),)))
    members = [member for group in groups for member in group]
    assert members and all(type(member) is RSMNexusMemberForm for member in members)
    assert Path(members[0].nexus_path) == masters[0].resolve() and members[0].entry


def test_directory_mode_maps_a_folders_nexus_files_beside_its_spec_scans(tmp_path, monkeypatch):
    """C5c: in Directory mode a folder's raw NeXus stacks are members beside its SPEC scans (C5b skipped them).
    Stitch's resolver is given one NeXus group beside the SPEC ones (the mapping's branch; the resolver itself
    is Stitch's, tested there)."""
    from xrd_tools.analysis.rsm_values import RSMNexusMemberForm
    import xrd_tools.analysis.stitch_sources as stitch_sources
    from xrd_tools.core.scan import SourceKind, SourceSpec
    spec = _pair(tmp_path / "data")
    real = stitch_sources.resolve_stitch_sources

    def with_a_nexus_group(selection, read_image_kwargs):
        resolved = real(selection, read_image_kwargs)
        nexus = (SourceSpec(tmp_path / "data" / "burst_master.h5", SourceKind.EIGER_MASTER, entry="entry"),)
        return replace(resolved, groups=resolved.groups + (nexus,), names=resolved.names + (("burst_master",),))
    monkeypatch.setattr(stitch_sources, "resolve_stitch_sources", with_a_nexus_group)
    groups, skipped = _groups(StitchSourceSelection("Directory", (str(spec.parent),)))
    assert [[member.scan for member in group] for group in groups[:2]] == [["1.1"], ["2.1"]]
    nexus, = groups[2]
    assert type(nexus) is RSMNexusMemberForm and (Path(nexus.nexus_path).name, nexus.entry) == (
        "burst_master.h5", "entry")
    assert [(Path(path).name, entry) for path, entry, _reason in skipped] == [("RSM_pair", "3.1")]


def test_rsm_automatic_stem_retains_discovered_scan_padding(tmp_path):
    spec = _pair(tmp_path / "padded", with_images=(1,), scans=(1,))
    for path in spec.parent.glob("*_scan1_*.tif"):
        path.rename(path.with_name(path.name.replace("_scan1_", "_scan001_")))
    (group,), skipped = _groups(StitchSourceSelection("Spec", (str(spec),), "1"))
    assert skipped == ()
    assert group[0].image_stem == "RSM_pair_scan001_"
