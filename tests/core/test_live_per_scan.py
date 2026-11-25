"""Production-wired H1–H9 for NEXT-1219-LIVE-PER-SCAN.

Observe real selected media with an injected monotonic time. The sole
baseline-only discriminator is an explicit absent-capability assertion.
"""
from __future__ import annotations

import errno
import importlib
import importlib.util
import inspect
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from tests.core.test_bluesky_nexus import _write_bluesky_nxwriter, plain_nexus_file
from tests.core.test_source_edge_contract import _dangling_canonical, _land_canonical
from tests.core.test_spec_occurrence import _append, _counting, _rewrite, _symlink
from tests.core.test_stitch_operation import _native_form, _ring
from xrd_tools.analysis import stitch_sources
from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.scan_operations import AnalysisDisposition, MetadataTablePlan, run_metadata_table
from xrd_tools.analysis.stitch_operation import run_stitch_operation, stitch_output_current
from xrd_tools.analysis.stitch_sources import StitchSourceSelection, resolve_stitch_sources
from xrd_tools.analysis.stitch_values import StitchToolPreflightRefused, prepare_stitch_tool
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.io.bluesky_nexus import is_bluesky_nxwriter
from xrd_tools.io.metadata import SpecOccurrenceVerification
from xrd_tools.io.spec import UnusableSpecScan, spec_capture_scope, spec_occurrence
from xrd_tools.sources.point_images import PointImagePolicy
from xrd_tools.sources.registry import open_source

DECODE = {"detector_shape": (2, 3), "raw_dtype": "uint16", "raw_header_skip": 3, "rotation": 90}
HEADER = "#F sample\n#E 1\n#D today\n#O0 del  nu\n\n"
SCAN = "ascan del 14 15 1 1"
_LIVE_MODULE = "xrd_tools.analysis.live_scans"


class MissingLiveCapability(AssertionError):
    pass


pytestmark = pytest.mark.xfail(
    importlib.util.find_spec(_LIVE_MODULE) is None,
    strict=True, raises=MissingLiveCapability,
    reason="baseline has no per-scan Live observer or native handoff",
)


def _live(row):
    if importlib.util.find_spec(_LIVE_MODULE) is None:
        raise MissingLiveCapability(f"{row}: missing per-scan Live observation/admission")
    return importlib.import_module(_LIVE_MODULE)


def _spec_rows(number, command, rows):
    return (f"#S {number} {command}\n#D today\n#P0 14 -10\n#N 2\n#L del  nu\n"
            + "".join(f"{14 + i} -10\n" for i in range(rows)))


def _spec(root, number=2, command=SCAN, rows=2):
    path = root / "sample"
    path.write_text(HEADER + _spec_rows(number, command, rows), encoding="utf-8", newline="\n")
    return path


def _raw(path, scan, *indices, stem=None):
    for index in indices:
        value = np.arange(6, dtype=np.uint16).reshape(2, 3) + index
        (path.parent / f"{stem or path.stem}_scan{scan}_{index:04d}.raw").write_bytes(
            b"HDR" + value.tobytes())


def _selection(path, scans="2", **kwargs):
    return StitchSourceSelection("Spec", (str(path),), scans, **kwargs)


def _decode(form):
    return dict(detector_shape=form.detector_shape, raw_dtype=form.raw_dtype,
                raw_header_skip=form.raw_header_skip, rotation=0)


def _native_keys(resolution):
    return tuple(tuple((Path(member.uri), member.options.get("scan") or member.entry)
                       for member in group) for group in resolution.groups)


def _planned_keys(groups):
    return tuple(tuple((Path(member.path), member.key) for member in group.members) for group in groups)


def _only(observer, now, status, **kwargs):
    row, = observer.poll(now=now, **kwargs).groups
    assert row.status == status, (now, row)
    if status != "ready":
        assert row.reason and row.evidence is None, row
    return row


def _handoff(live, row, attempt="attempt-1"):
    return live.LiveScanHandoff.from_ready(row, session_id="headless-live", attempt_id=attempt)


def _observe_form(row, form):
    live = _live(row)
    assert "live_handoff" in inspect.signature(prepare_stitch_tool).parameters, (
        f"{row}: finite Stitch preflight lacks exact Live handoff")
    observer = live.LiveScanObserver(form.source_selection, _decode(form),
                                     select_labels=form.frame_selector.select)
    return live, observer


def _ready_form(row, form):
    live, observer = _observe_form(row, form)
    _only(observer, 0, "waiting")
    return live, observer, _handoff(live, _only(observer, 2, "ready"))


def _assert_committed(result):
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    return result


def _live_preflight(form, handoff):
    preflight = prepare_stitch_tool(form, live_handoff=handoff)
    request, = preflight.requests
    return request


def _assert_preflight_refused(form, handoff):
    with pytest.raises(StitchToolPreflightRefused) as refusal:
        prepare_stitch_tool(form, live_handoff=handoff)
    assert refusal.value.code == "LIVE_HANDOFF_CHANGED", refusal.value


def _assert_before_science_refusal(request):
    accepted = []
    terminal = run_stitch_operation(request, on_accepted=accepted.append).terminal
    assert terminal.disposition is ModuleDisposition.REFUSED and not accepted, terminal
    assert terminal.code == "LIVE_HANDOFF_CHANGED", terminal


@pytest.mark.parametrize("change", ("rewrite", "replace"))
def test_h1_spec_stillness_tracks_the_occurrence_and_raw(tmp_path, change):
    path = _spec(tmp_path)
    _raw(path, 2, 0, 1)
    selection = _selection(path)
    assert _native_keys(resolve_stitch_sources(selection, DECODE)) == (((path, "2.1"),),)
    with spec_capture_scope():
        proof = spec_occurrence(path, "2.1")
    assert proof.rows == 2 and SpecOccurrenceVerification((proof,)).holds(proof)

    live = _live("H1")
    observer = live.LiveScanObserver(selection, DECODE)
    _only(observer, 0, "waiting")
    _only(observer, 1.99, "waiting")
    ready = _only(observer, 2, "ready")
    _append(path, _spec_rows(3, SCAN, 1))
    assert SpecOccurrenceVerification((proof,)).holds(proof)
    assert _only(observer, 4, "ready").evidence == ready.evidence
    if change == "rewrite":
        _rewrite(path, "15 -10\n", "15 -11\n")
    else:
        replacement = path.with_name("copy")
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    assert not SpecOccurrenceVerification((proof,)).holds(proof)
    _only(observer, 6, "waiting")
    _only(observer, 8, "ready")


def test_h1_an_unterminated_selected_row_cannot_seal(tmp_path):
    path = _spec(tmp_path)
    _raw(path, 2, 0, 1)
    selection = _selection(path)
    assert resolve_stitch_sources(selection, DECODE).groups
    _append(path, "16 -10")
    live = _live("H1")
    observer = live.LiveScanObserver(selection, DECODE)
    _only(observer, 0, "waiting")
    _only(observer, 60, "waiting")


@pytest.mark.parametrize(("command", "points"), (
    ("ascan del 0 2 2 0.1", 3),
    ("dscan del -1 1 2 0.1", 3),
    ("a2scan del 0 2 nu 0 2 2 0.1", 3),
    ("d3scan del -1 1 nu -1 1 phi -1 1 2 0.1", 3),
    ("mesh del 0 1 1 nu 0 2 2 0.1", 6),
    ("dmesh del -1 1 1 nu -1 1 2 0.1", 6),
    ("hklscan 0 1 0 1 0 1 2 0.1", 3),
    ("loopscan 3 0.1", 3),
))
def test_h2_supported_counts_do_not_complete_one_point_early(tmp_path, command, points):
    path = _spec(tmp_path, command=command, rows=points - 1)
    _raw(path, 2, *range(points - 1))
    selection = _selection(path)
    # Finite processing can intentionally run the underfull scan.
    assert resolve_stitch_sources(selection, DECODE).groups
    observer = _live("H2").LiveScanObserver(selection, DECODE)
    _only(observer, 0, "waiting")
    _only(observer, 2, "waiting")
    _append(path, f"{14 + points - 1} -10\n")
    _raw(path, 2, points - 1)
    _only(observer, 4, "waiting")
    _only(observer, 6, "ready")


@pytest.mark.parametrize("command", (
    "ascan del 0 2 7 0.1", "blast del 0 2 1 0.1", "ascan del 0 2 1.5 0.1",
    "ascan del 0 2 1 0.1 trailing", "mesh del 0 1 1 nu 0 2 0.1",
    "loopscan 1.2 0.1",
))
def test_h2_valid_successor_closes_only_a_complete_numeric_prefix(tmp_path, command):
    path = _spec(tmp_path, command=command)
    selection = _selection(path)
    _raw(path, 2, 0, 1)
    with spec_capture_scope():
        assert spec_occurrence(path, "2.1").rows == 2
    assert resolve_stitch_sources(selection, DECODE).groups
    observer = _live("H2").LiveScanObserver(selection, DECODE)
    prefix = path.read_bytes()
    for index, suffix in enumerate((b"", b"#F other\n", b"#S 3", b"#S 3   \n", b"#S invalid command\n")):
        path.write_bytes(prefix + suffix)
        _only(observer, 4 * index, "waiting")
        _only(observer, 4 * index + 2, "waiting")
    path.write_bytes(prefix + _spec_rows(3, SCAN, 0).encode())
    _only(observer, 20, "waiting")
    _only(observer, 22, "ready")


@pytest.mark.parametrize("successor", ("removed", "partial"))
def test_h2_successor_loss_refuses_both_run_and_append_current(tmp_path, successor):
    form = _native_form(tmp_path, "multigeometry", scans="1")
    path = Path(form.spec_path)
    text = path.read_text(encoding="utf-8")
    selected, rest = text.split("\n#S 2 ", 1)
    selected = selected.replace("ascan del 14 15 1 1", "ascan del 14 15 7 1")
    path.write_text(selected + "\n#S 2 " + rest, encoding="utf-8", newline="\n")
    finite = prepare_stitch_tool(form).request
    _assert_committed(run_stitch_operation(finite))
    assert stitch_output_current(finite)
    target = Path(finite.module.output.target)
    before = target.read_bytes()
    with spec_capture_scope():
        proof = spec_occurrence(path, "1.1")

    _live_api, observer, handoff = _ready_form("H2", form)
    live_request = _live_preflight(form, handoff)
    assert stitch_output_current(live_request)
    ending = "\n#S 2" if successor == "partial" else "\n"
    path.write_text(selected + ending, encoding="utf-8", newline="\n")
    assert SpecOccurrenceVerification((proof,)).holds(proof)
    _assert_preflight_refused(form, handoff)
    _assert_before_science_refusal(live_request)
    assert not stitch_output_current(live_request)
    _only(observer, 6, "waiting")
    assert target.read_bytes() == before


def test_h3_raw_coverage_decode_and_access_recovery(tmp_path, monkeypatch):
    path = _spec(tmp_path, command="ascan del 14 16 2 1", rows=3)
    _append(path, _spec_rows(3, SCAN, 2))
    selection = _selection(path, image_stem="frozen_{occurrence}_scan{scan}_")
    _raw(path, 2, 0, 2, stem="frozen_2.1")
    _raw(path, 3, 0, 1, stem="frozen_3.1")
    with pytest.raises(UnusableSpecScan, match="3 metadata points but 2 image frames"):
        resolve_stitch_sources(selection, DECODE)
    live = _live("H3")
    observer = live.LiveScanObserver(selection, DECODE)
    _only(observer, 0, "waiting")
    _only(observer, 8, "waiting")
    _raw(path, 2, 1, stem="frozen_2.1")
    source = open_source(resolve_stitch_sources(selection, DECODE).groups[0][0])
    np.testing.assert_array_equal(source.load_native_frame(1),
                                  np.rot90(np.arange(1, 7, dtype=np.uint16).reshape(2, 3)))

    original_open = os.open
    blocked = True
    probes = 0

    def unavailable(name, *args, **kwargs):
        nonlocal probes
        if not isinstance(name, int) and Path(name) == path:
            probes += 1
            if blocked:
                raise PermissionError(errno.EACCES, "selected SPEC temporarily inaccessible", name)
        return original_open(name, *args, **kwargs)

    monkeypatch.setattr(os, "open", unavailable)
    revision = path.stat()
    _only(observer, 10, "waiting")
    assert probes == 1
    blocked = False
    _only(observer, 12, "waiting")
    _only(observer, 14, "ready")
    assert probes >= 2 and path.stat().st_mtime_ns == revision.st_mtime_ns


def test_h3_spec_eiger_waits_for_the_real_shard_and_rejects_bare_master(tmp_path):
    path = _spec(tmp_path, command="ascan del 14 16 2 1", rows=3)
    original, shard = _dangling_canonical(tmp_path)
    master = original.with_name("sample_scan2_master.h5")
    original.replace(master)
    selection = _selection(path, nexus_entry="entry")
    with spec_capture_scope():
        assert spec_occurrence(path, "2.1").rows == 3
    live = _live("H3")
    observer = live.LiveScanObserver(selection, {})
    _only(observer, 0, "waiting")
    _only(observer, 2, "waiting")
    _land_canonical(shard)
    lowered = resolve_stitch_sources(selection, {}).groups[0][0]
    source = open_source(lowered)
    np.testing.assert_array_equal(source.load_native_frame(2), np.arange(128, 192).reshape(8, 8))
    _only(observer, 4, "waiting")
    _only(observer, 6, "ready")
    standalone = live.LiveScanObserver(StitchSourceSelection("NeXus", (str(master),)), {})
    skipped = _only(standalone, 0, "skipped")
    assert "detector" in skipped.reason.lower() and master.name in skipped.reason


def test_h4_missing_explicit_occurrence_is_a_planned_wait(tmp_path):
    path = _spec(tmp_path)
    _raw(path, 2, 0, 1)
    selection = _selection(path, "5")
    with pytest.raises(ValueError, match=r"scan 5 is not present"):
        resolve_stitch_sources(selection, DECODE)
    observer = _live("H4").LiveScanObserver(selection, DECODE)
    pending = _only(observer, 0, "waiting")
    assert _planned_keys((pending.group,)) == (((path, "5.1"),),)
    _append(path, _spec_rows(5, SCAN, 2))
    _raw(path, 5, 0, 1)
    _only(observer, 2, "waiting")
    assert _only(observer, 4, "ready").group.identity == pending.group.identity


def test_h5_each_output_waits_for_its_exact_ordered_members(tmp_path):
    path = _spec(tmp_path)
    _append(path, _spec_rows(3, SCAN, 2))
    _raw(path, 2, 0, 1)
    _raw(path, 3, 0, 1)
    selection = _selection(path, "2,[2-3],[2,2.1]")
    expected = (((path, "2.1"),), ((path, "2.1"), (path, "3.1")), ((path, "2.1"),))
    assert _native_keys(resolve_stitch_sources(selection, DECODE)) == expected
    live = _live("H5")
    assert "planned_group" in inspect.signature(resolve_stitch_sources).parameters, (
        "H5: native resolver lacks exact planned-group lowering")
    missing = tmp_path / "sample_scan3_0001.raw"
    pixels = missing.read_bytes()
    missing.unlink()
    observer = live.LiveScanObserver(selection, DECODE)
    observer.poll(now=0)
    a, b, c = observer.poll(now=2).groups
    assert [row.status for row in (a, b, c)] == ["ready", "waiting", "ready"]
    assert _planned_keys((a.group, b.group, c.group)) == expected
    assert len({row.group.identity for row in (a, b, c)}) == 3
    assert _native_keys(resolve_stitch_sources(selection, DECODE, planned_group=a.group)) == expected[:1]
    missing.write_bytes(pixels)
    assert observer.poll(now=4).groups[1].status == "waiting"
    b = observer.poll(now=6).groups[1]
    assert b.status == "ready"
    assert _native_keys(resolve_stitch_sources(selection, DECODE, planned_group=b.group)) == expected[1:2]


def _root_with_sibling(path, *, native=False):
    _write_bluesky_nxwriter(path, n=2)
    with h5py.File(path, "r+") as stream:
        entry = stream["entry"]
        if native:
            config = entry["instrument/bluesky/metadata/configuration/eiger/data"]
            config["eiger_cam_wavelength"][()] = 1.0
            config["eiger_cam_photon_energy"][()] = 12398.41984
            for motor, values in (("del", (14., 15.)), ("nu", (-10., -10.))):
                entry["data"].create_dataset(motor, data=values)
                entry["instrument/positioners"].create_group(motor).create_dataset("value", data=values)
            del entry["data/eiger_image"]
            del entry["instrument/detectors/eiger/data"]
            images = entry["data"].create_dataset("eiger_image",
                                                  data=np.stack([_ring(0), _ring(1)]).astype(np.uint32))
            images.attrs["signal_type"] = "detector"
            entry["instrument/detectors/eiger/data"] = images
        stream.copy("entry", "sibling")
    return path


def _link_detector(path, entry, shard):
    with h5py.File(path, "r+") as stream:
        group = stream[entry]
        pixels = group["data/eiger_image"][()]
        del group["data/eiger_image"]
        del group["instrument/detectors/eiger/data"]
        group["data/eiger_image"] = h5py.ExternalLink(shard.name, "/detector")
    return pixels


def _write_shard(path, pixels):
    with h5py.File(path, "w") as stream:
        stream.create_dataset("detector", data=pixels).attrs["signal_type"] = "detector"


def _entry_selection(path, key=""):
    return StitchSourceSelection("NeXus", (str(path),), nexus_entry=key)


def _native_nexus_form(root):
    form = _native_form(root, "multigeometry", scans="1")
    path = _root_with_sibling(root / "finished.nxs", native=True)
    return replace(form, source_selection=_entry_selection(path, "entry")), path


def test_h6_all_present_entries_gate_an_exact_override(tmp_path, plain_nexus_file):
    path = _root_with_sibling(tmp_path / "mixed.nxs")
    shard = tmp_path / "sibling-raw.h5"
    selection = _entry_selection(path, "entry")
    override = _entry_selection(path, "missing")
    assert _native_keys(resolve_stitch_sources(selection, {})) == (((path, "entry"),),)
    with pytest.raises(ValueError, match=r"NeXus entry 'missing' is not present"):
        resolve_stitch_sources(override, {})
    pixels = _link_detector(path, "sibling", shard)
    with h5py.File(path, "r+") as stream:
        assert is_bluesky_nxwriter(stream) and "end_time" in stream["entry"]
        del stream["sibling/end_time"]
    with h5py.File(plain_nexus_file, "r+") as generic:
        generic["entry"].create_dataset("end_time", data=b"2026-07-10T00:01:00")
        assert not is_bluesky_nxwriter(generic)

    live = _live("H6")
    observer = live.LiveScanObserver(selection, {})
    pending = _only(observer, 0, "waiting")
    _only(observer, 2, "waiting")
    with h5py.File(path, "r+") as stream:
        stream["sibling"].create_dataset("end_time", data=b"2026-07-10T00:01:00")
    _only(observer, 4, "waiting")
    _only(observer, 6, "waiting")
    _write_shard(shard, pixels)
    table = run_metadata_table(MetadataTablePlan(SourceSpec(path, SourceKind.NEXUS_STACK, entry="sibling")))
    assert table.disposition is AnalysisDisposition.COMPLETED and table.labels == (0, 1), table
    _only(observer, 8, "waiting")
    ready = _only(observer, 10, "ready")
    assert ready.group.identity == pending.group.identity
    assert _planned_keys((ready.group,)) == (((path, "entry"),),)
    assert _native_keys(resolve_stitch_sources(selection, {}, planned_group=ready.group)) == (((path, "entry"),),)

    for invalid in (override, _entry_selection(plain_nexus_file)):
        invalid_observer = live.LiveScanObserver(invalid, {})
        _only(invalid_observer, 0, "waiting")
        _only(invalid_observer, 2, "waiting")

    with h5py.File(path, "r+") as stream:
        stream.copy("entry", "later")
        del stream["later/data/eiger_image"]
        del stream["later/instrument/detectors/eiger/data"]
    _only(observer, 12, "waiting")
    _only(observer, 14, "waiting")  # sibling fallback may not qualify this exact entry
    with h5py.File(path, "r+") as stream:
        stream["later/data/eiger_image"] = stream["entry/data/eiger_image"]
    _only(observer, 16, "waiting")
    revised = _only(observer, 18, "ready")
    assert revised.group.identity == ready.group.identity and revised.evidence != ready.evidence
    handoff = _handoff(live, revised)
    with h5py.File(path, "r+") as stream:
        stream.copy("entry", "new-acquisition")
    result = observer.poll(now=20, consumed=(handoff,))
    assert not any(row.status == "ready" for row in result.groups)
    assert tuple((Path(name), key) for name, key in result.late_entries) == ((path, "new-acquisition"),)
    assert _native_keys(resolve_stitch_sources(selection, {})) == (((path, "entry"),),)
    assert "new-acquisition" in {key for _, key in _native_keys(
        resolve_stitch_sources(_entry_selection(path), {}))[0]}


def test_h6_unselected_shard_replacement_revokes_run_and_append(tmp_path):
    form, path = _native_nexus_form(tmp_path)
    shard = tmp_path / "sibling-raw.h5"
    pixels = _link_detector(path, "sibling", shard)
    _write_shard(shard, pixels)
    finite = prepare_stitch_tool(form).request
    _assert_committed(run_stitch_operation(finite))
    assert stitch_output_current(finite)
    output = Path(finite.module.output.target)
    before = output.read_bytes()
    _live_api, observer, handoff = _ready_form("H6", form)
    live_request = _live_preflight(form, handoff)
    assert stitch_output_current(live_request)
    root_stat = path.stat()
    replacement = tmp_path / "replace.h5"
    _write_shard(replacement, pixels + 1)
    replacement.replace(shard)
    assert path.stat().st_mtime_ns == root_stat.st_mtime_ns and path.stat().st_size == root_stat.st_size
    assert stitch_output_current(finite)
    _assert_preflight_refused(form, handoff)
    _assert_before_science_refusal(live_request)
    assert not stitch_output_current(live_request)
    _only(observer, 4, "waiting")
    _only(observer, 6, "ready")
    assert output.read_bytes() == before


def test_h6_accepted_run_keeps_exact_root_custody(tmp_path, monkeypatch):
    from xrd_tools.analysis import stitch_operation

    form, path = _native_nexus_form(tmp_path)
    finite = prepare_stitch_tool(form).request
    assert finite.module.source.selected_labels == (0, 1)
    _live_api, _observer, handoff = _ready_form("H6", form)
    request = _live_preflight(form, handoff)
    accepted = []
    changed = []
    admit = stitch_operation.admit_module_artifact

    def late_entry(*args, **kwargs):
        assert len(accepted) == 1
        with h5py.File(path, "r+") as stream:
            stream.copy("entry", "too-late")
        changed.append(True)
        return admit(*args, **kwargs)

    # Change the actual file after science, before the existing publication admission.
    monkeypatch.setattr(stitch_operation, "admit_module_artifact", late_entry)
    result = run_stitch_operation(request, on_accepted=accepted.append)
    assert changed == [True] and len(accepted) == 1
    assert result.terminal.disposition is ModuleDisposition.REFUSED, result.terminal
    assert not Path(request.module.output.target).exists()


@pytest.mark.parametrize("at", ("after-observe", "after-preflight"))
@pytest.mark.parametrize("change", ("remove-member", "replace-detector", "change-raw-mapping", "linked-shard"))
def test_h7_mutations_cannot_cross_native_admission(tmp_path, at, change):
    form = _native_form(tmp_path, "multigeometry", scans="[1,2]")
    path = Path(form.spec_path)
    images = Path(form.image_dir)
    if change == "linked-shard":
        master = images / "sample_scan2_master.h5"
        shard = images / "sample_scan2_data_000001.h5"
        _write_bluesky_nxwriter(master, n=2)
        with h5py.File(master, "r+") as stream:
            del stream["entry/data/eiger_image"]
            del stream["entry/instrument/detectors/eiger/data"]
            data = stream["entry/data"].create_dataset("eiger_image", data=np.stack([_ring(0), _ring(1)]))
            data.attrs["signal_type"] = "detector"
            stream["entry/instrument/detectors/eiger/data"] = data
        pixels = _link_detector(master, "entry", shard)
        _write_shard(shard, pixels)
        for raw in images.glob("*scan2_*.raw"):
            raw.unlink()
    if change == "change-raw-mapping":
        for index in (2, 3):
            raw = images / f"sample_scan2_{index:04d}.raw"
            raw.write_bytes((images / f"sample_scan2_{index - 2:04d}.raw").read_bytes())
        selection = replace(form.source_selection, point_image_choices=(
            (str(path), "2.1", PointImagePolicy(2, "select", image_index=1)),))
        form = replace(form, source_selection=selection)

    finite = prepare_stitch_tool(form).request
    assert finite.manifest.member_frames == ((0, 0), (0, 1), (1, 0), (1, 1))
    _live_api, _observer, handoff = _ready_form("H7", form)
    request = _live_preflight(form, handoff) if at == "after-preflight" else None
    if change == "remove-member":
        path.write_text(path.read_text(encoding="utf-8").split("\n#S 2 ", 1)[0] + "\n",
                        encoding="utf-8", newline="\n")
    elif change == "replace-detector":
        raw = images / "sample_scan2_0001.raw"
        replacement = images / "fresh.raw"
        replacement.write_bytes(raw.read_bytes())
        os.utime(replacement, ns=(raw.stat().st_atime_ns, raw.stat().st_mtime_ns))
        replacement.replace(raw)
    elif change == "change-raw-mapping":
        # Same selected point labels, count and policy. Only the physical ordering changes.
        first = images / "sample_scan2_0000.raw"
        second = images / "sample_scan2_0001.raw"
        temporary = images / "moving.raw"
        first.replace(temporary)
        second.replace(first)
        temporary.replace(second)
    else:
        replacement = images / "fresh.h5"
        _write_shard(replacement, pixels + 1)
        replacement.replace(shard)
    if request is None:
        _assert_preflight_refused(form, handoff)
    else:
        _assert_before_science_refusal(request)
    assert not Path(finite.module.output.target).exists()


def test_h7_exact_native_request_matches_finite_and_accepts_unrelated_growth(tmp_path, monkeypatch):
    from xrd_tools.analysis import stitch_native_values

    form = replace(_native_form(tmp_path, "multigeometry", scans="3,[1-2]"), output_family=None)
    finite = prepare_stitch_tool(replace(
        form, source_selection=replace(form.source_selection, scans="[1-2]"))).request
    ordinary = _assert_committed(run_stitch_operation(finite))
    images = Path(form.image_dir)
    for index in (0, 1):
        (images / f"sample_scan3_{index:04d}.raw").write_bytes(
            (images / f"sample_scan1_{index:04d}.raw").read_bytes())
    live, observer = _observe_form("H7", form)
    observer.poll(now=0)
    rows = observer.poll(now=2).groups
    assert [row.status for row in rows] == ["ready", "ready"]
    handoff = _handoff(live, rows[1])
    seen = []
    metadata = stitch_native_values.run_metadata_table

    def trace(plan, *args, **kwargs):
        seen.append(plan.source.options["scan"])
        return metadata(plan, *args, **kwargs)

    monkeypatch.setattr(stitch_native_values, "run_metadata_table", trace)
    request = _live_preflight(form, handoff)
    assert seen == ["1.1", "2.1"]
    assert request.module.fingerprint == finite.module.fingerprint
    assert request.manifest.to_provenance() == finite.manifest.to_provenance()
    assert request.provenance_json == finite.provenance_json
    assert stitch_output_current(request)
    _append(form.spec_path, _spec_rows(4, SCAN, 1))
    accepted = []
    watched = _assert_committed(run_stitch_operation(request, on_accepted=accepted.append))
    assert len(accepted) == 1
    assert watched.payload.inspection.result_fingerprint == ordinary.payload.inspection.result_fingerprint
    np.testing.assert_array_equal(watched.payload.intensity, ordinary.payload.intensity)
    assert watched.payload.provenance_json == ordinary.payload.provenance_json


def test_h8_one_ordered_listing_drives_finite_and_live(tmp_path, monkeypatch):
    root = tmp_path / "discover"
    root.mkdir()
    sub = root / "sub"
    sub.mkdir()
    spec = _spec(root)
    _append(spec, _spec_rows(3, SCAN, 2))
    _raw(spec, 2, 0, 1)
    _raw(spec, 3, 0, 1)
    early = _write_bluesky_nxwriter(root / "a_take.nxs", n=2)
    late = _write_bluesky_nxwriter(sub / "b_take.nxs", n=2)
    _write_bluesky_nxwriter(sub / "c_exclude.nxs", n=2)
    _symlink(sub / "z_take_alias.nxs", early)
    # Even a filename that looks like a scan candidate is still a detector raw file.
    (sub / "take_fake.raw").write_bytes(b"#F detector-only\n")
    selections = (
        _selection(spec, "3,[2-3]"),
        StitchSourceSelection("NeXus", (str(late), str(early))),
        StitchSourceSelection("Directory", (str(root),), recursive=True, name_filter="sample | take -exclude"),
    )
    expected = (
        (((spec, "3.1"),), ((spec, "2.1"), (spec, "3.1"))),
        (((late, "entry"), (early, "entry")),),
        (((spec, "2.1"),), ((spec, "3.1"),), ((early, "entry"),), ((late, "entry"),)),
    )
    for selection, keys in zip(selections, expected):
        assert _native_keys(resolve_stitch_sources(selection, DECODE)) == keys

    _append(spec, _spec_rows(4, SCAN, 1))
    progress = []
    result = resolve_stitch_sources(replace(selections[0], stitch_all=True), DECODE,
                                    progress=lambda done, total: progress.append((done, total)))
    assert _native_keys(result) == (((spec, "2.1"),), ((spec, "3.1"),))
    assert result.skipped[0][:2] == (str(spec), "4.1")
    assert progress == [(1, 3), (2, 3), (3, 3)]
    _raw(spec, 3, 2, 3)
    result = resolve_stitch_sources(selections[0], DECODE)
    assert not result.groups
    assert len(result.pending) == 1 and result.pending[0].images_per_point == 2
    for index in (2, 3):
        (root / f"sample_scan3_{index:04d}.raw").unlink()
    spec.write_text(spec.read_text(encoding="utf-8").split("#S 4 ", 1)[0],
                    encoding="utf-8", newline="\n")

    live = _live("H8")
    assert hasattr(stitch_sources, "list_planned_stitch_groups"), "H8: shared planned-group listing is missing"
    listing = stitch_sources.list_planned_stitch_groups
    calls = []

    def count(selection, *args, **kwargs):
        calls.append(selection)
        return listing(selection, *args, **kwargs)

    monkeypatch.setattr(stitch_sources, "list_planned_stitch_groups", count)
    for selection, keys in zip(selections, expected):
        calls.clear()
        assert _native_keys(resolve_stitch_sources(selection, DECODE)) == keys
        assert calls == [selection]
        calls.clear()
        observer = live.LiveScanObserver(selection, DECODE)
        assert _planned_keys(tuple(row.group for row in observer.poll(now=0).groups)) == keys
        assert calls == [selection]


@pytest.mark.parametrize("fallback", (False, True))
def test_h9_spec_poll_shares_capture_and_listing_and_handles_unchanged_time(
        tmp_path, monkeypatch, ctime_seam, win32_pathname_ctime, fallback):
    from xrd_tools.sources import spec as spec_source

    path = _spec(tmp_path)
    _append(path, _spec_rows(3, SCAN, 2))
    for scan in (2, 3):
        _raw(path, scan, 0, 1)
    selection = _selection(path, stitch_all=True)
    if fallback:
        monkeypatch.setattr(ctime_seam, "IDENTITY_CARRIES_CTIME", False)
        win32_pathname_ctime(path)
    parsers, opens = _counting(monkeypatch)
    assert len(resolve_stitch_sources(selection, DECODE).groups) == 2
    assert sum(parsers.values()) == 1 and opens[str(path.resolve())] == 1

    live = _live("H9")
    scandir = os.scandir
    read_layout = spec_source._read_spec_image_layout
    listings = []
    layouts = []

    def count_listing(folder):
        if not isinstance(folder, int) and Path(folder) == tmp_path:
            listings.append(Path(folder))
        return scandir(folder)

    def count_layout(path, *args, **kwargs):
        layouts.append(Path(path))
        return read_layout(path, *args, **kwargs)

    def read_plane(*_args, **_kwargs):
        pytest.fail("the metadata-only Live observer loaded a detector plane")

    monkeypatch.setattr(os, "scandir", count_listing)
    monkeypatch.setattr(spec_source, "_read_spec_image_layout", count_layout)
    monkeypatch.setattr(spec_source.SpecSource, "load_frame", read_plane)
    monkeypatch.setattr(spec_source.SpecSource, "load_native_frame", read_plane)
    observer = live.LiveScanObserver(selection, DECODE)

    def poll(now):
        parsers.clear()
        opens.clear()
        listings.clear()
        layouts.clear()
        rows = observer.poll(now=now).groups
        assert sum(parsers.values()) <= 1 and opens.get(str(path.resolve()), 0) <= 1, (parsers, opens)
        assert len(listings) <= 1, listings
        assert len(layouts) == len(set(layouts)), layouts
        return rows

    assert [row.status for row in poll(0)] == ["waiting", "waiting"]
    assert [row.status for row in poll(2)] == ["ready", "ready"]
    assert not layouts
    if fallback:
        assert opens[str(path.resolve())] == 1
        revision = path.stat()
        _rewrite(path, "15 -10\n#S 3", "15 -11\n#S 3")
        os.utime(path, ns=(revision.st_atime_ns, revision.st_mtime_ns))
        assert [row.status for row in poll(4)] == ["waiting", "ready"]
        assert [row.status for row in poll(6)] == ["ready", "ready"]
    else:
        assert not parsers and not opens
    _append(path, _spec_rows(4, SCAN, 1))
    assert [row.status for row in poll(8)] == ["ready", "ready", "waiting"]
    _append(path, "15 -10\n")
    _raw(path, 4, 0, 1)
    assert [row.status for row in poll(10)] == ["ready", "ready", "waiting"]
    assert [row.status for row in poll(12)] == ["ready", "ready", "ready"]


def test_h9_recursive_poll_does_not_reopen_consumed_container_graph(tmp_path, monkeypatch):
    from xrd_tools.sources import descriptor, execution_graph

    root = tmp_path / "directory"
    root.mkdir()
    path = _root_with_sibling(root / "acquisition.nxs")
    selection = StitchSourceSelection("Directory", (str(root),), recursive=True)
    assert _native_keys(resolve_stitch_sources(selection, DECODE)) == (((path, "entry"), (path, "sibling")),)
    live = _live("H9")

    opens = Counter()
    real_scandir = os.scandir
    real_layout = descriptor._stack_facts
    real_hdf5 = h5py.File.__init__
    real_graph = execution_graph.qualify_source_execution_graph

    def scandir(name):
        if not isinstance(name, int) and Path(name) == root:
            opens["traversal"] += 1
        return real_scandir(name)

    def layout(stream, *args, **kwargs):
        if Path(stream.filename) == path:
            opens["layout"] += 1
        return real_layout(stream, *args, **kwargs)

    def hdf5(self, name, *args, **kwargs):
        if isinstance(name, (str, Path)) and Path(name) == path:
            opens["container_opens"] += 1
        return real_hdf5(self, name, *args, **kwargs)

    def graph(source, *args, **kwargs):
        if Path(source.uri) == path:
            opens["graph"] += 1
        return real_graph(source, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", scandir)
    monkeypatch.setattr(descriptor, "_stack_facts", layout)
    monkeypatch.setattr(h5py.File, "__init__", hdf5)
    monkeypatch.setattr(execution_graph, "qualify_source_execution_graph", graph)
    observer = live.LiveScanObserver(selection, DECODE)
    _only(observer, 0, "waiting")
    ready = _only(observer, 2, "ready")
    assert 0 < opens["layout"] and 0 < opens["container_opens"], opens
    handoff = _handoff(live, ready)
    assert _native_keys(resolve_stitch_sources(selection, DECODE, planned_group=handoff.group)) == (
        ((path, "entry"), (path, "sibling")),)
    opens.clear()
    for now in (4, 6):
        assert not any(row.status == "ready" for row in observer.poll(now=now, consumed=(handoff,)).groups)
    assert opens["container_opens"] == opens["layout"] == opens["graph"] == 0, opens
    assert opens["traversal"] <= 2, opens
    with h5py.File(path, "r+") as stream:
        stream.copy("entry", "late")
    sub = root / "incoming"
    sub.mkdir()
    spec = _spec(sub)
    _raw(spec, 2, 0, 1)
    opens.clear()
    waiting = observer.poll(now=8, consumed=(handoff,))
    assert not any(row.status == "ready" for row in waiting.groups)
    assert tuple((Path(name), key) for name, key in waiting.late_entries) == ((path, "late"),)
    ready_groups = tuple(row.group for row in observer.poll(now=10, consumed=(handoff,)).groups
                         if row.status == "ready")
    assert _planned_keys(ready_groups) == (((spec, "2.1"),),)
    assert opens["layout"] == opens["graph"] == 0 and opens["traversal"] <= 2, opens
