from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from tests.core._v2_record_fixture import (
    _GI_1D_UNITS,
    _result_1d,
    _result_2d,
    write_gi_reference_scan,
)
from xrd_tools.core import DEFAULT_MODE_KEY
from xrd_tools.io import (
    NexusRecordWriter,
    RecordWrite,
    WriterFinalization,
    WriterIncomplete,
)
from xrd_tools.io.append import (
    AppendIntent,
    AppendSource,
    begin_same_run_lineage,
    commit_append_lineage,
    decode_committed_append_prefix,
)
from xrd_tools.io.nexus import (
    open_nexus_writer,
    read_stitched,
    validate_integrated_stack_write,
    write_nexus,
    write_nexus_frame,
    write_integrated_stack,
)
from xrd_tools.io.nexus_record import frame_record_from_live_frame
from xrd_tools.io.processed_scan_id import require_current_processed_groups
from xrd_tools.io.schema import (
    REINTEGRATE_SHADOW_COMPLETE_ATTR,
    REINTEGRATE_SHADOW_SUFFIX,
    SCHEMA_NAME_ATTR,
    SCHEMA_VERSION_ATTR,
)


def _gi_scan(tmp_path):
    target = tmp_path / "current-gi.nexus"
    source_root = tmp_path / "Project"
    write_gi_reference_scan(target, source_root, compression="gzip")
    return target, source_root


def _h5_structure(path):
    """Freeze every owned object, shape, dtype, and attribute for mutation proof."""
    structure = []
    with h5py.File(path, "r") as handle:
        def remember(name, node):
            attrs = tuple(
                (
                    key,
                    tuple(node.attrs.get_id(key).shape),
                    node.attrs.get_id(key).dtype.str,
                    repr(np.asarray(node.attrs[key]).tolist()),
                )
                for key in sorted(node.attrs)
            )
            if isinstance(node, h5py.Dataset):
                identity = ("dataset", tuple(node.shape), node.dtype.str)
            else:
                identity = ("group",)
            structure.append((name, identity, attrs))

        remember("/", handle)
        handle.visititems(remember)
    return tuple(structure)


def _replace_dataset(group, name, values):
    attrs = dict(group[name].attrs)
    del group[name]
    dataset = group.create_dataset(name, data=values)
    dataset.attrs.update(attrs)


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


def test_new_gi_record_persists_exact_current_identity_and_reloads(tmp_path):
    from xrd_tools.corrections.grazing import GI_EXIT_ANGLE_CONVENTION
    from xrd_tools.core.provenance import read_provenance
    from xrd_tools.session.run_configuration import (
        FrozenGIConfiguration,
        GIIntent,
        RunIntent,
    )

    frozen = RunIntent(
        gi=GIIntent(enabled=True, incidence_motor="Manual", th_val=0.3),
    ).freeze()
    target = tmp_path / "new-current-gi.nexus"
    writer = NexusRecordWriter(target, overwrite=True, flush_every=None)
    writer.begin(primary_mode_1d="q_total")
    writer.write(RecordWrite(
        label=0,
        result_1d=_result_1d(0, unit=_GI_1D_UNITS["q_total"]),
        mode_1d="q_total",
    ))
    writer.finish(WriterFinalization(
        frame_indices=(0,),
        provenance_config={
            "bai_1d_args": {"gi_mode_1d": "q_total"},
            "bai_2d_args": {"gi_mode_2d": "qip_qoop"},
            "gi": True,
            "gi_config": frozen.gi.scan_config(),
            "run_configuration": frozen.as_provenance(),
        },
    ))

    provenance = read_provenance(target)
    outer = provenance["config"]["run_configuration"]
    assert len(outer["gi"]) == 9
    assert outer["gi"]["gi_exit_angle_convention"] == GI_EXIT_ANGLE_CONVENTION
    assert FrozenGIConfiguration.from_dict(outer["gi"]) == frozen.gi
    assert provenance["config"]["gi_config"][
        "gi_exit_angle_convention"
    ] == GI_EXIT_ANGLE_CONVENTION

    with h5py.File(target, "r") as handle:
        raw = handle["entry/reduction/config/run_configuration"][()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        assert json.loads(raw)["gi"]["gi_exit_angle_convention"] == (
            GI_EXIT_ANGLE_CONVENTION
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


@pytest.mark.parametrize(
    "corruption",
    (
        "foreign_unstamped",
        "missing_declared_intensity",
        "wrong_declared_intensity_dtype",
        "wrong_declared_intensity_shape",
    ),
)
def test_record_writer_begin_requires_strict_current_target_before_mutation(
    tmp_path,
    corruption,
):
    target, source_root = _gi_scan(tmp_path)
    with h5py.File(target, "r+") as handle:
        entry = handle["entry"]
        if corruption == "foreign_unstamped":
            del entry.attrs[SCHEMA_NAME_ATTR]
            del entry.attrs[SCHEMA_VERSION_ATTR]
        else:
            leaf = entry["integrated_1d/q_oop"]
            if corruption == "missing_declared_intensity":
                del leaf["intensity"]
            elif corruption == "wrong_declared_intensity_dtype":
                _replace_dataset(
                    leaf,
                    "intensity",
                    np.asarray(leaf["intensity"][()], dtype=np.float64),
                )
            else:
                _replace_dataset(
                    leaf,
                    "intensity",
                    np.asarray(leaf["intensity"][:, :-1], dtype=np.float32),
                )

    before_bytes = target.read_bytes()
    before_structure = _h5_structure(target)
    writer = NexusRecordWriter(
        target,
        atomic=False,
        overwrite=False,
        flush_every=None,
        source_base=source_root,
    )
    refusal = None
    try:
        writer.begin(primary_mode_1d="q_total", primary_mode_2d="qip_qoop")
    except WriterIncomplete as error:
        refusal = error
    finally:
        writer.abort()

    assert refusal is not None
    assert refusal.outcome.pending_owner == "begin"
    assert target.read_bytes() == before_bytes
    assert _h5_structure(target) == before_structure


def test_record_writer_begin_still_creates_a_genuinely_new_target(tmp_path):
    target = tmp_path / "new-current.nexus"
    assert not target.exists()
    writer = NexusRecordWriter(
        target,
        atomic=False,
        overwrite=False,
        flush_every=None,
        complete_record=False,
    )
    writer.begin()
    writer.write(
        RecordWrite(
            label=0,
            result_1d=_result_1d(0, unit="q_A^-1", offset=1),
        )
    )
    writer.finish()

    with h5py.File(target, "r") as handle:
        processed = require_current_processed_groups(handle)
        assert processed.modes_1d == (DEFAULT_MODE_KEY,)


@pytest.mark.parametrize("layout", ("contiguous", "fixed-chunked"))
def test_record_writer_begin_refuses_non_appendable_frame_index_before_mutation(
    tmp_path,
    layout,
):
    target, source_root = _gi_scan(tmp_path)
    with h5py.File(target, "r+") as handle:
        group = handle["entry/integrated_1d/q_oop"]
        values = np.asarray(group["frame_index"][()])
        attrs = dict(group["frame_index"].attrs)
        del group["frame_index"]
        kwargs = (
            {}
            if layout == "contiguous"
            else {"chunks": True, "maxshape": values.shape}
        )
        fixed = group.create_dataset("frame_index", data=values, **kwargs)
        fixed.attrs.update(attrs)

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


@pytest.mark.parametrize(
    "alias", ("same-group", "cross-mode", "cross-dimension"),
)
def test_record_writer_begin_refuses_row_semantic_hard_alias_before_mutation(
    tmp_path,
    alias,
):
    target, source_root = _gi_scan(tmp_path)
    with h5py.File(target, "r+") as handle:
        parent = handle["entry/integrated_1d"]
        group = parent["q_oop"]
        if alias == "same-group":
            del group["sigma"]
            group["sigma"] = group["intensity"]
        elif alias == "cross-mode":
            del group["intensity"]
            group["intensity"] = parent["q_ip/intensity"]
        else:
            group_2d = handle["entry/integrated_2d"]
            del group_2d["frame_index"]
            group_2d["frame_index"] = group["frame_index"]

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


def test_complete_shadow_remains_readable_but_ordinary_writer_refuses(tmp_path):
    target, source_root = _gi_scan(tmp_path)
    shadow_name = f"integrated_1d{REINTEGRATE_SHADOW_SUFFIX}"
    with h5py.File(target, "r+") as handle:
        entry = handle["entry"]
        entry.move("integrated_1d", shadow_name)
        entry[shadow_name].attrs[REINTEGRATE_SHADOW_COMPLETE_ATTR] = np.bool_(True)
    with h5py.File(target, "r") as handle:
        recovered = require_current_processed_groups(handle)
        assert recovered.integrated_1d.name.endswith(f"/{shadow_name}")

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


@pytest.mark.parametrize("shape", ("sequential", "batch"))
def test_record_writer_refuses_decreasing_new_labels_before_mutation(
    tmp_path,
    shape,
):
    target = tmp_path / f"decreasing-{shape}.nexus"
    writer = NexusRecordWriter(
        target,
        atomic=False,
        overwrite=False,
        flush_every=None,
        complete_record=False,
    )
    writer.begin()
    if shape == "sequential":
        writer.write(RecordWrite(label=2, result_1d=_result_1d(2)))
        writer._h5.flush()
        before = target.read_bytes()
        with pytest.raises(ValueError, match="strictly increasing"):
            writer.write(RecordWrite(label=1, result_1d=_result_1d(1)))
        writer._h5.flush()
        assert target.read_bytes() == before
        writer.finish()
        with h5py.File(target, "r") as handle:
            processed = require_current_processed_groups(handle)
            assert tuple(processed.integrated_1d["frame_index"][()]) == (2,)
    else:
        writer._h5.flush()
        before = target.read_bytes()
        with pytest.raises(ValueError, match="strictly increasing"):
            writer.write_batch(
                (
                    RecordWrite(label=2, result_1d=_result_1d(2)),
                    RecordWrite(label=1, result_1d=_result_1d(1)),
                )
            )
        writer._h5.flush()
        assert target.read_bytes() == before
        writer.abort()


@pytest.mark.parametrize("writer_api", ("incremental", "complete"))
def test_public_nexus_writers_refuse_a_new_label_before_the_cursor_without_mutation(
    tmp_path,
    writer_api,
):
    target = tmp_path / f"decreasing-{writer_api}.nexus"
    write_nexus(target, results_1d={2: _result_1d(2)}, overwrite=True)

    if writer_api == "incremental":
        with open_nexus_writer(target) as handle:
            handle.flush()
            before = target.read_bytes()
            with pytest.raises(ValueError, match="strictly increasing"):
                write_nexus_frame(handle, 1, result_1d=_result_1d(1))
            handle.flush()
            assert target.read_bytes() == before
    else:
        before = target.read_bytes()
        with pytest.raises(ValueError, match="strictly increasing"):
            write_nexus(target, results_1d={1: _result_1d(1)})
        assert target.read_bytes() == before


def test_incremental_frame_preflights_both_dimensions_before_either_mutates(
    tmp_path,
):
    target = tmp_path / "mixed-cursors.nexus"
    write_nexus(target, results_1d={0: _result_1d(0)}, overwrite=True)
    write_nexus(target, results_2d={2: _result_2d(2)})

    with open_nexus_writer(target) as handle:
        handle.flush()
        before = target.read_bytes()
        with pytest.raises(ValueError, match="strictly increasing"):
            write_nexus_frame(
                handle,
                1,
                result_1d=_result_1d(1),
                result_2d=_result_2d(1),
            )
        handle.flush()
        assert target.read_bytes() == before


def test_incremental_writer_close_requires_a_current_processed_artifact(tmp_path):
    target = tmp_path / "empty-incremental.nexus"
    handle = open_nexus_writer(target, overwrite=True)
    with pytest.raises(ValueError, match="not a current xdart"):
        handle.close()
    assert not handle.id.valid
    with h5py.File(target, "r") as invalid:
        with pytest.raises(ValueError, match="not a current xdart"):
            require_current_processed_groups(invalid)


@pytest.mark.parametrize(
    "writer_api", ("record", "incremental", "complete", "sink"),
)
def test_public_writers_reject_non_nexus_target_before_creation(
    tmp_path,
    writer_api,
):
    parent = tmp_path / writer_api
    target = parent / "processed.nxs"
    with pytest.raises(ValueError, match=r"\.nexus"):
        if writer_api == "record":
            NexusRecordWriter(target, complete_record=False)
        elif writer_api == "incremental":
            open_nexus_writer(target, overwrite=True)
        elif writer_api == "sink":
            from xrd_tools.reduction import NexusSink

            NexusSink(target, overwrite=True)
        else:
            write_nexus(
                target,
                results_1d={0: _result_1d(0)},
                overwrite=True,
            )
    assert not parent.exists()


def test_nexus_sink_refuses_malformed_existing_target_before_transaction(
    tmp_path,
    monkeypatch,
):
    from xrd_tools.reduction import NexusSink, ReductionPlan, Scan
    from xrd_tools.reduction import core as reduction_core

    target, _source_root = _gi_scan(tmp_path)
    with h5py.File(target, "r+") as handle:
        group = handle["entry/integrated_1d/q_oop"]
        values = np.asarray(group["frame_index"][()])
        del group["frame_index"]
        group.create_dataset("frame_index", data=values)
    before = target.read_bytes()
    transaction_effects = []
    monkeypatch.setattr(
        reduction_core,
        "get_output_transaction_coordinator",
        lambda: SimpleNamespace(
            admit=lambda *_args, **_kwargs: transaction_effects.append("admitted"),
        ),
    )

    sink = NexusSink(target, overwrite=False)
    with pytest.raises(ValueError, match="not a current xdart"):
        sink.begin(Scan("empty", []), ReductionPlan())
    assert transaction_effects == []
    assert target.read_bytes() == before


def test_existing_append_shadow_refuses_before_preflight_reservation(
    tmp_path,
    monkeypatch,
):
    from xrd_tools.io import output_transaction as transaction_module
    from xrd_tools.reduction import NexusSink, ReductionPlan, Scan

    target, _source_root = _gi_scan(tmp_path)
    shadow_name = f"integrated_1d{REINTEGRATE_SHADOW_SUFFIX}"
    with h5py.File(target, "r+") as handle:
        entry = handle["entry"]
        entry.move("integrated_1d", shadow_name)
        entry[shadow_name].attrs[REINTEGRATE_SHADOW_COMPLETE_ATTR] = np.bool_(True)

    before = target.read_bytes()
    before_paths = tuple(sorted(path.name for path in tmp_path.iterdir()))
    transaction_effects = []
    monkeypatch.setattr(
        transaction_module,
        "get_output_transaction_coordinator",
        lambda: SimpleNamespace(
            admit=lambda *_args, **_kwargs: transaction_effects.append("admitted"),
        ),
    )
    intent = AppendIntent(
        entry="entry",
        source_base=str(tmp_path),
        source_identity="raw/scan",
        science_fingerprint="science-v1",
        modes=("1d:q_total",),
        source=AppendSource(
            path=str(tmp_path / "raw.h5"),
            adapter_id="nexus_hdf5",
            size=0,
            mtime_ns=0,
            extent=1,
        ),
        labels=(0,),
    )

    sink = NexusSink.for_existing_append(target, intent)
    with pytest.raises(ValueError, match="read-only recovery"):
        sink.begin(Scan("existing", []), ReductionPlan(integration_2d=None))
    assert transaction_effects == []
    assert sink.append_preflight is None
    assert sink.abort(None) is None
    assert target.read_bytes() == before
    assert tuple(sorted(path.name for path in tmp_path.iterdir())) == before_paths


def test_record_writer_finish_requires_strict_current_artifact(tmp_path):
    target = tmp_path / "empty-finish.nexus"
    writer = NexusRecordWriter(
        target,
        atomic=False,
        flush_every=None,
        complete_record=False,
    )
    writer.begin()
    with pytest.raises(WriterIncomplete) as caught:
        writer.finish()
    assert caught.value.outcome.pending_owner == "admission"
    writer.abort()


def test_repeated_finish_re_admits_the_published_target(tmp_path):
    target = tmp_path / "repeat-finish.nexus"
    writer = NexusRecordWriter(
        target,
        atomic=False,
        flush_every=None,
        complete_record=False,
    )
    writer.begin()
    writer.write(RecordWrite(label=0, result_1d=_result_1d(0)))
    writer.finish()
    with h5py.File(target, "r+") as handle:
        group = handle["entry/integrated_1d"]
        values = np.asarray(group["frame_index"][()])
        del group["frame_index"]
        group.create_dataset("frame_index", data=values)

    with pytest.raises(ValueError, match="not a current xdart"):
        writer.finish()


@pytest.mark.parametrize("canonical_kind", ("soft", "external", "dataset"))
def test_complete_shadow_is_used_only_when_canonical_slot_is_absent(
    tmp_path,
    canonical_kind,
):
    target, _source_root = _gi_scan(tmp_path)
    shadow_name = f"integrated_1d{REINTEGRATE_SHADOW_SUFFIX}"
    foreign = tmp_path / "foreign-result.h5"
    if canonical_kind == "external":
        with h5py.File(foreign, "w") as handle:
            handle.create_group("foreign_result")

    with h5py.File(target, "r+") as handle:
        entry = handle["entry"]
        entry.copy("integrated_1d", shadow_name)
        shadow = entry[shadow_name]
        shadow.attrs[REINTEGRATE_SHADOW_COMPLETE_ATTR] = np.bool_(True)
        del entry["integrated_1d"]

    with h5py.File(target, "r") as handle:
        recovered = require_current_processed_groups(handle)
        assert recovered.integrated_1d.name.endswith(f"/{shadow_name}")

    with h5py.File(target, "r+") as handle:
        entry = handle["entry"]
        if canonical_kind == "soft":
            entry["integrated_1d"] = h5py.SoftLink(f"/entry/{shadow_name}")
        elif canonical_kind == "external":
            entry["integrated_1d"] = h5py.ExternalLink(
                foreign.name,
                "/foreign_result",
            )
        else:
            entry.create_dataset(
                "integrated_1d",
                data=np.asarray([1], dtype=np.int8),
            )

    with h5py.File(target, "r") as handle:
        assert handle["entry"].get("integrated_1d", getlink=True) is not None
        with pytest.raises(ValueError, match="not a current xdart"):
            require_current_processed_groups(handle)


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
