from __future__ import annotations

from types import SimpleNamespace

import h5py
import pytest

from tests.core._v2_record_fixture import (
    _GI_1D_UNITS,
    _result_1d,
    write_gi_reference_scan,
)
from xrd_tools.core import DEFAULT_MODE_KEY
from xrd_tools.io import NexusRecordWriter, WriterIncomplete
from xrd_tools.io.append import (
    AppendIntent,
    AppendSource,
    begin_same_run_lineage,
    commit_append_lineage,
    decode_committed_append_prefix,
)
from xrd_tools.io.nexus import (
    read_stitched,
    validate_integrated_stack_write,
    write_integrated_stack,
)
from xrd_tools.io.nexus_record import frame_record_from_live_frame
from xrd_tools.io.processed_scan_id import require_current_processed_groups
from xrd_tools.io.schema import (
    REINTEGRATE_SHADOW_COMPLETE_ATTR,
    REINTEGRATE_SHADOW_SUFFIX,
)


def _gi_scan(tmp_path):
    target = tmp_path / "current-gi.nexus"
    source_root = tmp_path / "Project"
    write_gi_reference_scan(target, source_root, compression="gzip")
    return target, source_root


def test_current_admission_owns_the_exact_ordered_gi_inventory(tmp_path):
    target, _source_root = _gi_scan(tmp_path)
    with h5py.File(target, "r") as handle:
        processed = require_current_processed_groups(handle)
        assert processed.primary_mode_1d == "q_total"
        assert processed.primary_mode_2d == "qip_qoop"
        assert processed.modes_1d == (
            "q_total", "q_ip", "q_oop", "exit_angle", "chi_gi",
        )
        assert processed.modes_2d == (
            "qip_qoop", "q_chi", "exit_angles",
        )
        assert processed.mode_group("1d", "q_oop").name.endswith(
            "/integrated_1d/q_oop"
        )


@pytest.mark.parametrize("corruption", ["unknown", "malformed"])
def test_read_stitched_rejects_every_malformed_unrequested_gi_sibling(
    tmp_path,
    corruption,
):
    target, _source_root = _gi_scan(tmp_path)
    with h5py.File(target, "r+") as handle:
        parent = handle["entry/integrated_1d"]
        if corruption == "unknown":
            parent.create_group("rogue_result")
        else:
            del parent["q_oop/intensity"]
    with pytest.raises(ValueError, match="not a current xdart"):
        read_stitched(target)


def test_nested_soft_link_is_refused_by_admission_and_writer_preflight(tmp_path):
    target, _source_root = _gi_scan(tmp_path)
    with h5py.File(target, "r+") as handle:
        parent = handle["entry/integrated_1d"]
        del parent["q_oop"]
        parent["q_oop"] = h5py.SoftLink("/entry/integrated_1d/q_ip")
    with h5py.File(target, "r") as handle:
        with pytest.raises(ValueError, match="not a current xdart"):
            require_current_processed_groups(handle)
    with h5py.File(target, "r+") as handle:
        with pytest.raises(ValueError, match="not local hard storage"):
            validate_integrated_stack_write(
                handle["entry"],
                frame_indices=[0],
                results_1d=[
                    _result_1d(0, unit=_GI_1D_UNITS["q_oop"], offset=3)
                ],
                group_name_1d="integrated_1d/q_oop",
                allow_rebuild=False,
            )


def test_record_writer_refuses_linked_existing_result_before_mutation(tmp_path):
    target, source_root = _gi_scan(tmp_path)
    with h5py.File(target, "r+") as handle:
        parent = handle["entry/integrated_1d"]
        del parent["q_oop"]
        parent["q_oop"] = h5py.SoftLink("/entry/integrated_1d/q_ip")
    before = target.read_bytes()
    writer = NexusRecordWriter(
        target,
        atomic=False,
        overwrite=False,
        flush_every=None,
        source_base=source_root,
    )
    with pytest.raises(WriterIncomplete) as caught:
        writer.begin(primary_mode_1d="q_total", primary_mode_2d="qip_qoop")
    assert caught.value.outcome.pending_owner == "begin"
    writer.abort()
    assert target.read_bytes() == before


def test_append_decoder_uses_complete_admitted_mode_owners(tmp_path):
    target, source_root = _gi_scan(tmp_path)
    source_path = source_root / "frame_0000.tif"
    stat = source_path.stat()
    intent = AppendIntent(
        "entry",
        str(source_root),
        "current-gi-source",
        "current-gi-science",
        ("1d:q_total", "2d:qip_qoop"),
        AppendSource(
            str(source_path),
            "tiff",
            stat.st_size,
            stat.st_mtime_ns,
            3,
        ),
        (0, 1, 2),
    )
    decision = begin_same_run_lineage(intent)
    with h5py.File(target, "r+") as handle:
        commit_append_lineage(
            handle["entry"], decision, written_labels=(0, 1, 2),
        )
    with h5py.File(target, "r") as handle:
        assert decode_committed_append_prefix(handle).committed_labels == (0, 1, 2)
    with h5py.File(target, "r+") as handle:
        parent = handle["entry/integrated_1d"]
        del parent["q_oop"]
        parent["q_oop"] = h5py.SoftLink("/entry/integrated_1d/q_ip")
    with h5py.File(target, "r") as handle:
        with pytest.raises(ValueError, match="not a current xdart"):
            decode_committed_append_prefix(handle)


def test_component_wise_local_hard_nested_result_remains_writable(tmp_path):
    target, _source_root = _gi_scan(tmp_path)
    with h5py.File(target, "r+") as handle:
        labels = validate_integrated_stack_write(
            handle["entry"],
            frame_indices=[0],
            results_1d=[
                _result_1d(0, unit=_GI_1D_UNITS["q_oop"], offset=3)
            ],
            group_name_1d="integrated_1d/q_oop",
            allow_rebuild=False,
        )
    assert labels == [0]


def test_late_invalid_multimode_write_does_not_mutate_any_bytes(tmp_path):
    target, _source_root = _gi_scan(tmp_path)
    before = target.read_bytes()
    with h5py.File(target, "r+") as handle:
        with pytest.raises(ValueError, match="length must match"):
            write_integrated_stack(
                handle["entry"],
                frame_indices=[0],
                results_1d=[
                    _result_1d(0, unit=_GI_1D_UNITS["q_total"], offset=99)
                ],
                extra_modes_1d={"q_ip": ()},
                extra_mode_indices_1d={"q_ip": ()},
                primary_mode_1d="q_total",
            )
    assert target.read_bytes() == before


def test_text_false_shadow_marker_is_never_promoted(tmp_path):
    target, _source_root = _gi_scan(tmp_path)
    with h5py.File(target, "r+") as handle:
        entry = handle["entry"]
        entry.move("integrated_1d", f"integrated_1d{REINTEGRATE_SHADOW_SUFFIX}")
        shadow = entry[f"integrated_1d{REINTEGRATE_SHADOW_SUFFIX}"]
        shadow.attrs[REINTEGRATE_SHADOW_COMPLETE_ATTR] = "false"
    with h5py.File(target, "r") as handle:
        with pytest.raises(ValueError, match="not a current xdart"):
            require_current_processed_groups(handle)


def _live_frame(*, modes_1d):
    active = _result_1d(0, unit=_GI_1D_UNITS["q_total"], offset=1)
    return SimpleNamespace(
        idx=0,
        gi=True,
        scan_info={"th": 0.2},
        int_1d=active,
        int_2d=None,
        gi_1d=modes_1d(active),
        gi_2d={},
        thumbnail=None,
        source_file=None,
        source_frame_idx=0,
        map_raw=None,
        _get_incident_angle=lambda: 0.2,
    )


def test_historical_live_gi_key_is_not_translated():
    frame = _live_frame(modes_1d=lambda active: {"qtotal": active})
    with pytest.raises(ValueError, match="unknown canonical 1d mode key"):
        frame_record_from_live_frame(frame)


def test_empty_map_defaults_but_unowned_named_selector_fails():
    frame = _live_frame(modes_1d=lambda _active: {})
    record = frame_record_from_live_frame(frame)
    assert record.active_mode_1d == DEFAULT_MODE_KEY
    assert record.modes_1d == (DEFAULT_MODE_KEY,)
    with pytest.raises(ValueError, match="selector .* has no owned result"):
        frame_record_from_live_frame(frame, active_mode_1d="q_oop")
