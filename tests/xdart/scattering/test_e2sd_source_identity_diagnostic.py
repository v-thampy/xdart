from dataclasses import replace
from pathlib import Path

import fabio
import h5py
import numpy as np
import pytest

from tests.xdart.scattering._e2sd_support import write_poni
from xdart.gui.tabs.scattering.adapters import run_executor as _bootstrap_sources
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.contracts import (
    PlannedOutput,
    SourceCapture,
    StartCapture,
)
from xdart.gui.tabs.scattering.events import RequestId
from xdart.gui.tabs.scattering.output_preflight import (
    OutputCandidate,
    _directory_items,
    _validate_exact_tiff_gi_motor,
    execution_plan_values,
    prepare_output,
    source_snapshots,
    validate_admitted_receipt,
)
from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.core.metadata import resolve_incident_angle
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.reduction import run_reduction
import xrd_tools.reduction.core as reduction_core
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.readiness import build_native_int_reduction_plan_from_args
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xrd_tools.sources.selection import image_series_spec
from xrd_tools.sources.adapters import candidate_owner
from xrd_tools.sources.discover import Candidate
from xrd_tools.sources.registry import open_source
from xrd_tools.sources.run_plan import RunCandidatePlan
from xrd_tools.sources.selection import DirectorySourceSpec


def _write_tiff(path: Path, value: int) -> None:
    fabio.tifimage.TifImage(
        data=np.full((4, 4), value, dtype=np.uint16)
    ).write(str(path))


def test_directory_tiff_auto_metadata_reaches_planned_source(
    tmp_path: Path,
) -> None:
    first = tmp_path / "scan_0001.tif"
    second = tmp_path / "scan_0002.tif"
    _write_tiff(first, 1)
    _write_tiff(second, 2)
    first.with_suffix(".txt").write_text(
        "th=0.15\nexposure=1.0\nsequence=1\n",
        encoding="utf-8",
    )
    second.with_suffix(".txt").write_text(
        "th=0.25\nexposure=2.0\nsequence=2\n",
        encoding="utf-8",
    )
    owner = candidate_owner(first)
    assert owner is not None
    states = tuple(path.stat() for path in (first, second))
    plan = RunCandidatePlan(
        generation=0,
        candidates=tuple(
            Candidate(path, owner.id, state.st_size, state.st_mtime_ns)
            for path, state in zip((first, second), states)
        ),
        root=tmp_path,
        recursive=False,
        name_filter=None,
    )
    candidate = OutputCandidate(
        DirectorySourceSpec(
            tmp_path,
            suffixes=(".tif",),
            metadata_format="auto",
        ),
        "",
        "",
        str(tmp_path / "processed"),
        "{}",
        "test-fingerprint",
    )

    items = _directory_items(candidate, plan)

    assert len(items) == 1
    assert items[0].source_spec.options["metadata_format"] == "auto"
    source = open_source(items[0].source_spec)
    assert source.frame_for(1).metadata["th"] == pytest.approx(0.15)
    assert source.frame_for(2).metadata["th"] == pytest.approx(0.25)


def test_admitted_tiff_gi_run_resolves_each_frame_th_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    first = raw / "scan_0001.tif"
    second = raw / "scan_0002.tif"
    _write_tiff(first, 1)
    _write_tiff(second, 2)
    first.with_suffix(".txt").write_text(
        "th=0.15\nexposure=1.0\nsequence=1\n",
        encoding="utf-8",
    )
    second.with_suffix(".txt").write_text(
        "th=0.25\nexposure=2.0\nsequence=2\n",
        encoding="utf-8",
    )
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source_spec = DirectorySourceSpec(
        raw,
        suffixes=(".tif",),
        metadata_format="auto",
    )
    intent = RunIntent(
        source_spec=source_spec,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
        bai_1d_args={"npt": 2},
        bai_2d_args={"npt_rad": 2, "npt_azim": 2},
        gi=GIIntent(
            enabled=True,
            incidence_motor="th",
            th_val=9.9,
            mode_1d="q_ip",
        ),
    )
    snapshot = RunIntentStore(intent).snapshot()
    request = RequestId(905)
    start = StartCapture(
        request,
        1,
        snapshot,
        SourceCapture(request, 1, source_spec),
    )
    sessions: list[object] = []
    receipt = prepare_output(
        start,
        cancelled=lambda: False,
        session_owner=sessions.append,
    )
    assert len(sessions) == 1
    opened = None
    try:
        assert receipt.gi_motor_choices == ("th", "exposure", "sequence")
        assert len(receipt.outputs) == 1
        assert receipt.outputs[0].labels == (1, 2)
        item = receipt.outputs[0].item
        admitted = item.source_stamp.admitted_motor_values
        assert tuple(
            (Path(value.source_path), value.motor, value.value)
            for value in admitted
        ) == (
            (first, "th", 0.15),
            (second, "th", 0.25),
        )
        assert item.source_stamp.as_dict()["admitted_motor_values"] == [
            {"source_path": str(first), "motor": "th", "value": 0.15},
            {"source_path": str(second), "motor": "th", "value": 0.25},
        ]
        forged_options = dict(item.source_spec.options)
        forged_options["admitted_motor_values"] = (
            (str(first), "th", 0.15),
            (str(second), "th", 7.25),
        )
        with pytest.raises(
            ValueError,
            match="do not match source stamp",
        ):
            PlannedOutput(
                SourceSpec(
                    item.source_spec.uri,
                    item.source_spec.kind,
                    options=forged_options,
                ),
                item.source_path,
                item.target,
                item.source_stamp,
                item.candidate,
                item.descriptor,
                item.motor_names,
            )

        empty_stamp = replace(
            item.source_stamp,
            admitted_motor_values=(),
        )
        empty_options = dict(item.source_spec.options)
        empty_options["admitted_motor_values"] = ()
        incomplete = PlannedOutput(
            SourceSpec(
                item.source_spec.uri,
                item.source_spec.kind,
                options=empty_options,
            ),
            item.source_path,
            item.target,
            empty_stamp,
            item.candidate,
            item.descriptor,
            item.motor_names,
        )
        with pytest.raises(
            ValueError,
            match="finite value in every admitted TIFF",
        ):
            _validate_exact_tiff_gi_motor(intent, (incomplete,))

        validate_admitted_receipt(receipt, sessions[0])
        # A rewrite after the final validation cannot change the accepted GI
        # angles consumed by this exact run.
        second.with_suffix(".txt").write_text(
            "th=7.25\nexposure=2.0\nsequence=2\n",
            encoding="utf-8",
        )
        opened = open_source(item.source_spec)
        scan = opened.to_scan(poni=object())
        assert tuple(frame.source_path for frame in scan.frames) == (first, second)
        assert all(frame.geometry is None for frame in scan.frames)

        configuration = snapshot.thaw().freeze(
            gi_motor_choices=receipt.gi_motor_choices
        )
        args_1d, args_2d, values = execution_plan_values(configuration)
        plan = build_native_int_reduction_plan_from_args(
            args_1d,
            args_2d,
            **values,
        )
        assert plan.gi is not None
        assert plan.gi.incident_angle is None
        assert plan.gi.incidence_motor == "th"

        angles_1d: list[float] = []
        angles_2d: list[float] = []

        def integrate_1d(_image, _integrator, **kwargs):
            angles_1d.append(float(kwargs["incident_angle"]))
            return IntegrationResult1D(
                radial=np.array([0.0, 1.0]),
                intensity=np.ones(2),
                unit="qip_A^-1",
            )

        def integrate_2d(_image, _integrator, **kwargs):
            angles_2d.append(float(kwargs["incident_angle"]))
            return IntegrationResult2D(
                radial=np.array([0.0, 1.0]),
                azimuthal=np.array([0.0, 1.0]),
                intensity=np.ones((2, 2)),
                unit="qip_A^-1",
                azimuthal_unit="qoop_A^-1",
            )

        monkeypatch.setattr(
            reduction_core,
            "poni_to_fiber_integrator",
            lambda *_args, **_kwargs: object(),
        )
        monkeypatch.setattr(reduction_core, "integrate_gi_1d", integrate_1d)
        monkeypatch.setattr(reduction_core, "integrate_gi_2d", integrate_2d)

        result = run_reduction(plan, scan)

        assert result.n_processed == 2
        assert angles_1d == pytest.approx([0.15, 0.25])
        assert angles_2d == pytest.approx([0.15, 0.25])
    finally:
        if opened is not None:
            close = getattr(opened, "close", None)
            if callable(close):
                close()
        sessions[0].close()


def test_admitted_tiff_motor_snapshot_removes_casefold_aliases(
    tmp_path: Path,
) -> None:
    image = tmp_path / "scan_0001.tif"
    _write_tiff(image, 1)
    image.with_suffix(".txt").write_text(
        "TH=9.0\nth=0.15\nexposure=1\n",
        encoding="utf-8",
    )
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(
        tmp_path,
        suffixes=(".tif",),
        metadata_format="auto",
    )
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
        gi=GIIntent(enabled=True, incidence_motor="th"),
    )).snapshot()
    request = RequestId(908)
    sessions: list[object] = []
    receipt = prepare_output(
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, source),
        ),
        cancelled=lambda: False,
        session_owner=sessions.append,
    )
    opened = None
    try:
        item = receipt.outputs[0].item
        opened = open_source(item.source_spec)
        metadata = dict(opened.frame_for(1).metadata)

        assert "TH" not in metadata
        assert metadata["th"] == pytest.approx(0.15)
        assert resolve_incident_angle(metadata, "th") == pytest.approx(0.15)
    finally:
        if opened is not None:
            close = getattr(opened, "close", None)
            if callable(close):
                close()
        sessions[0].close()


def test_direct_tiff_series_admission_owns_and_revalidates_sidecar_motors(
    tmp_path: Path,
) -> None:
    first = tmp_path / "scan_0001.tif"
    second = tmp_path / "scan_0002.tif"
    _write_tiff(first, 1)
    _write_tiff(second, 2)
    first.with_suffix(".txt").write_text(
        "th=0.15\nexposure=1.0\nsequence=1\nlabel=before\n",
        encoding="utf-8",
    )
    second_sidecar = second.with_suffix(".txt")
    second_sidecar.write_text(
        "th=0.25\nexposure=2.0\nsequence=2\nlabel=before\n",
        encoding="utf-8",
    )
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    selected = image_series_spec(first)
    legacy_options = dict(selected.options)
    legacy_options.pop("metadata_format")
    source = SourceSpec(
        selected.uri,
        selected.kind,
        options=legacy_options,
    )
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
        gi=GIIntent(enabled=True, incidence_motor="th"),
    )).snapshot()
    typed_source = snapshot.thaw().source_spec
    assert type(typed_source) is SourceSpec
    assert typed_source.options["metadata_format"] == "auto"
    request = RequestId(906)
    receipt = prepare_output(
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, typed_source),
        ),
        cancelled=lambda: False,
        session_owner=lambda _session: None,
    )

    assert receipt.gi_motor_choices == ("th", "exposure", "sequence")
    assert receipt.outputs[0].item.group.motor_names == receipt.gi_motor_choices
    assert receipt.candidate.source == typed_source
    assert (
        receipt.candidate.processing_mapping()["source"]["options"]
        ["metadata_format"]
        == "auto"
    )
    item = receipt.outputs[0].item
    assert tuple(
        (
            Path(value.source_path),
            None
            if value.metadata_file is None
            else Path(value.metadata_file.path),
        )
        for value in item.source_stamp.metadata_sources
    ) == (
        (first, first.with_suffix(".txt")),
        (second, second_sidecar),
    )
    persisted = item.source_stamp.as_dict()["metadata_sources"]
    assert tuple(value["source_path"] for value in persisted) == (
        str(first),
        str(second),
    )
    assert all(value["metadata_file"] is not None for value in persisted)
    snapshots = source_snapshots(item)
    assert snapshots[str(first.with_suffix(".txt"))]["source_role"] == (
        "image_metadata"
    )
    assert snapshots[str(second_sidecar)]["source_role"] == "image_metadata"

    second_sidecar.write_text(
        "th=0.25\nexposure=2.0\nsequence=2\nlabel=after\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="authoritative motor knowledge changed after admission",
    ):
        validate_admitted_receipt(receipt, None)


def test_direct_tiff_no_sidecar_state_invalidates_when_one_appears(
    tmp_path: Path,
) -> None:
    image = tmp_path / "scan_0001.tif"
    _write_tiff(image, 1)
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = image_series_spec(image)
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
        gi=GIIntent(enabled=True, incidence_motor="Manual"),
    )).snapshot()
    request = RequestId(909)
    receipt = prepare_output(
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, source),
        ),
        cancelled=lambda: False,
        session_owner=lambda _session: None,
    )

    metadata_sources = receipt.outputs[0].item.source_stamp.metadata_sources
    assert len(metadata_sources) == 1
    assert metadata_sources[0].source_path == str(image)
    assert metadata_sources[0].metadata_file is None
    assert receipt.outputs[0].item.source_stamp.as_dict()[
        "metadata_sources"
    ] == [{"source_path": str(image), "metadata_file": None}]

    image.with_suffix(".txt").write_text(
        "th=0.15\nexposure=1\nsequence=1\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="authoritative motor knowledge changed after admission",
    ):
        validate_admitted_receipt(receipt, None)


@pytest.mark.parametrize("hardlink_alias", (False, True))
def test_selected_tiff_output_cannot_overwrite_consumed_metadata_sidecar(
    tmp_path: Path,
    hardlink_alias: bool,
) -> None:
    image = tmp_path / "scan_0001.tif"
    _write_tiff(image, 1)
    sidecar = image.with_suffix(".txt")
    sidecar.write_text(
        "th=0.15\nexposure=1\nsequence=1\n",
        encoding="utf-8",
    )
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    target = sidecar
    if hardlink_alias:
        target = tmp_path / "overwrite_alias.nxs"
        target.hardlink_to(sidecar)
    source = image_series_spec(image)
    request = RequestId(910)
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(target),
        output_mode="Overwrite",
    )).snapshot()

    with pytest.raises(ValueError, match="same file as input"):
        prepare_output(
            StartCapture(
                request,
                1,
                snapshot,
                SourceCapture(request, 1, source),
            ),
            cancelled=lambda: False,
            session_owner=lambda _session: None,
        )


def test_selected_series_symlink_retarget_is_source_change(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.tif"
    second = tmp_path / "second.tif"
    selected = tmp_path / "selected.tif"
    _write_tiff(first, 1)
    _write_tiff(second, 2)
    selected.symlink_to(first)
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = SourceSpec(
        tmp_path,
        SourceKind.TIFF_SERIES,
        options={
            "files": (str(selected),),
            "selected_file": str(selected),
            "scan_name": "selected",
        },
    )
    store = RunIntentStore(
        RunIntent(
            source_spec=source,
            poni_file=str(poni),
            save_path=str(tmp_path / "processed"),
            output_mode="Overwrite",
        )
    )
    snapshot = store.snapshot()
    request = RequestId(902)
    start = StartCapture(
        request,
        1,
        snapshot,
        SourceCapture(request, 1, source),
    )
    receipt = prepare_output(
        start,
        cancelled=lambda: False,
        session_owner=lambda _session: None,
        targets_owner=lambda _paths: None,
    )

    selected.unlink()
    selected.symlink_to(second)

    with pytest.raises(ValueError, match="source candidate changed") as error:
        validate_admitted_receipt(receipt, None)
    print(error.value)


def test_selected_series_is_still_valid_without_retarget(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.tif"
    selected = tmp_path / "selected.tif"
    _write_tiff(first, 1)
    selected.symlink_to(first)
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = SourceSpec(
        tmp_path,
        SourceKind.TIFF_SERIES,
        options={
            "files": (str(selected),),
            "selected_file": str(selected),
            "scan_name": "selected",
        },
    )
    store = RunIntentStore(
        RunIntent(
            source_spec=source,
            poni_file=str(poni),
            save_path=str(tmp_path / "processed"),
            output_mode="Overwrite",
        )
    )
    snapshot = store.snapshot()
    request = RequestId(903)
    start = StartCapture(
        request,
        1,
        snapshot,
        SourceCapture(request, 1, source),
    )
    receipt = prepare_output(
        start,
        cancelled=lambda: False,
        session_owner=lambda _session: None,
        targets_owner=lambda _paths: None,
    )
    validate_admitted_receipt(receipt, None)


def test_selected_eiger_master_keeps_container_owner_and_frame_count(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "scan_data_000001.h5"
    master = tmp_path / "scan_master.h5"
    with h5py.File(sidecar, "w") as handle:
        handle.create_group("entry").create_group("data").create_dataset(
            "data",
            data=np.ones((2, 4, 5), dtype=np.uint16),
        )
    with h5py.File(master, "w") as handle:
        data = handle.create_group("entry").create_group("data")
        data["data_000001"] = h5py.ExternalLink(
            sidecar.name,
            "/entry/data/data",
        )
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = image_series_spec(master)
    assert source.kind is SourceKind.NEXUS_STACK
    request = RequestId(904)
    capture = FilesystemSourceAdapter().capture(source, request)
    snapshot = RunIntentStore(
        RunIntent(
            source_spec=source,
            poni_file=str(poni),
            save_path=str(tmp_path / "processed"),
            output_mode="Overwrite",
        )
    ).snapshot()

    receipt = prepare_output(
        StartCapture(request, 1, snapshot, capture),
        cancelled=lambda: False,
        session_owner=lambda _session: None,
        targets_owner=lambda _paths: None,
    )

    assert len(receipt.outputs) == 1
    item = receipt.outputs[0].item
    assert item.source_spec.kind is SourceKind.EIGER_MASTER
    assert item.source_stamp.adapter_id == "nexus_hdf5"
    assert item.source_stamp.frame_count == 2
    assert tuple(
        external.stop - external.first
        for external in item.source_stamp.external_members
    ) == (2,)
    validate_admitted_receipt(receipt, None)


@pytest.mark.parametrize("hardlink_alias", (False, True))
def test_selected_eiger_output_cannot_overwrite_external_data_member(
    tmp_path: Path,
    hardlink_alias: bool,
) -> None:
    sidecar = tmp_path / "scan_data_000001.h5"
    master = tmp_path / "scan_master.h5"
    with h5py.File(sidecar, "w") as handle:
        handle.create_group("entry").create_group("data").create_dataset(
            "data",
            data=np.ones((2, 4, 5), dtype=np.uint16),
        )
    with h5py.File(master, "w") as handle:
        data = handle.create_group("entry").create_group("data")
        data["data_000001"] = h5py.ExternalLink(
            sidecar.name,
            "/entry/data/data",
        )
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = image_series_spec(master)
    target = sidecar
    if hardlink_alias:
        target = tmp_path / "overwrite_alias.h5"
        target.hardlink_to(sidecar)
    request = RequestId(907)
    capture = FilesystemSourceAdapter().capture(source, request)
    snapshot = RunIntentStore(
        RunIntent(
            source_spec=source,
            poni_file=str(poni),
            save_path=str(target),
            output_mode="Overwrite",
        )
    ).snapshot()

    with pytest.raises(ValueError, match="same file as input"):
        prepare_output(
            StartCapture(request, 1, snapshot, capture),
            cancelled=lambda: False,
            session_owner=lambda _session: None,
            targets_owner=lambda _paths: None,
        )
