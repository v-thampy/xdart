"""Historical content gate for the multi-result NeXus layout (ADR-0003).

The committed signature retains the preceding q/chi dataset names. Current
writes use v3 neutral axes and display labels; those deltas are asserted before
comparison. All other names, numerical values, storage properties and attrs
remain pinned to the unchanged historical fixture. This is a normalized content
comparison, not a claim to reproduce a byte-identical historical file.
"""
import copy
import json
from pathlib import Path

import h5py

FIXTURE = Path(__file__).parent / "fixtures" / "multimode_signature.json"


def test_multimode_record_matches_historical_content_with_v3_axes(tmp_path):
    from tests.core._multimode_fixture import write_reference_multimode_scan
    from tests.core.h5sig import h5_content_signature

    out = write_reference_multimode_scan(str(tmp_path / "mm.nexus"))
    now = h5_content_signature(out)
    ref = json.loads(FIXTURE.read_text())

    # Check each intentional v3 delta directly before mapping it to the frozen
    # signature. No numerical, storage, mode-selection or unit facts are elided.
    comparable = copy.deepcopy(now)
    with h5py.File(out, "r") as f:
        for group_path, axis_labels in (
            ("entry/integrated_1d", ("Q (Å⁻¹)",)),
            ("entry/integrated_1d/q_oop", ("Q_oop (Å⁻¹)",)),
            ("entry/integrated_2d", ("Q_ip (Å⁻¹)", "Q_oop (Å⁻¹)")),
            ("entry/integrated_2d/q_chi", ("Q (Å⁻¹)", "χ (°)")),
        ):
            group = f[group_path]
            expected_axes = (("frame_index", "axis_1") if len(axis_labels) == 1
                             else ("frame_index", "axis_2", "axis_1"))
            assert tuple(group.attrs["axes"]) == expected_axes
            assert "q" not in group and "chi" not in group
            comparable[group_path]["attrs"]["axes"] = ref[group_path]["attrs"]["axes"]
            for number, label in enumerate(axis_labels, start=1):
                neutral = f"axis_{number}"
                old = "q" if number == 1 else "chi"
                assert group[neutral].attrs["long_name"] == label
                value = comparable.pop(f"{group_path}/{neutral}")
                del value["attrs"]["long_name"]
                comparable[f"{group_path}/{old}"] = value

    missing = sorted(set(ref) - set(comparable))
    added = sorted(set(comparable) - set(ref))
    assert not missing and not added, (
        f"tree changed: missing={missing[:6]} added={added[:6]}"
    )
    diffs = [k for k in sorted(ref) if ref[k] != comparable[k]]
    assert diffs == [], (
        "content changed at: " + ", ".join(diffs[:8]) + "\n"
        + "\n".join(f"  {k}: ref={ref[k]} now={comparable[k]}" for k in diffs[:3])
    )
