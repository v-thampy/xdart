"""Focused thin-GUI oracle for the headless Average Scan operation."""
from __future__ import annotations
import copy
from dataclasses import fields, replace
import json
from pathlib import Path
import threading
import h5py
import numpy as np
import pytest
import tifffile
from xdart.gui.tabs.scattering.adapters import external_operation as adapter
from xdart.gui.tabs.scattering.adapters.external_operation import OperationSlot
from xdart.gui.tabs.scattering.operation_values import OperationContextStamp, OperationTerminalStatus
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.reduction import (
    AverageFiniteCountsEvidence, AverageScanRecipe, AverageScanResult,
    Integration1DPlan, Integration2DPlan, ReductionPlan,
)


def _result(disposition: str, target: str = "/detached/average.nxs") -> AverageScanResult:
    from xrd_tools.io.output_transaction import StreamTerminal
    committed = disposition == "COMMITTED"
    evidence = (AverageFiniteCountsEvidence(
        "average_scan_v1", 2, (1, 1), "<u4", "c" * 64, 1, 2, 0,
        (1, 1), "gzip", 1, True, False) if committed else None)
    diagnostics = {
        "COMMITTED": ("", ""),
        "REFUSED": ("AVERAGE_TEST_REFUSED", "detached refusal"),
        "CANCELLED": ("AVERAGE_TEST_CANCELLED", "detached cancellation"),
        "ABORTED": ("AVERAGE_TEST_ABORT", "detached diagnostic"),
    }
    code, diagnostic = diagnostics[disposition]
    return AverageScanResult(
        disposition=disposition, target=target, entry="entry",
        operation_identity="a" * 64, science_identity="b" * 64,
        contributor_extent=2, logical_labels=(1,),
        committed_labels=(1,) if committed else (),
        metadata_denominators=(("I0", 2),) if committed else (), finite_counts=evidence,
        diagnostic_code=code, diagnostic=diagnostic,
        h23_phase="committed" if committed else None,
        commit_identity=StreamTerminal(target, 1, "d" * 64, 1) if committed else None,
    )


def _source(tmp_path: Path, arrays=None) -> SourceSpec:
    tmp_path.mkdir(exist_ok=True)
    arrays = arrays or (np.arange(4, dtype="u2").reshape(2, 2),
                        np.arange(4, dtype="u2").reshape(2, 2) + 2)
    files = []
    for index, array in enumerate(arrays, 1):
        path = tmp_path / f"scan_{index:04d}.tif"
        tifffile.imwrite(path, array); files.append(str(path))
    selected = Path(files[0])
    return SourceSpec(tmp_path, SourceKind.TIFF_SERIES, options={
        "selected_file": str(selected), "files": tuple(files),
        "pattern": "scan_*.tif", "scan_name": "scan", "metadata_format": None,
    })


def _join(slot: OperationSlot, identity):
    worker = slot._worker; assert worker is not None
    worker.join(20); assert not worker.is_alive()
    update = slot.poll(identity); assert update is not None and update.terminal is not None
    return update


def _stub_integrators(monkeypatch):
    from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
    from xrd_tools.reduction import core
    calls = []
    def one(image, _ai, *, npt, normalization_factor=None, **_kwargs):
        value = float(np.nanmean(image)) / (normalization_factor or 1.0)
        calls.append(("1d", value))
        return IntegrationResult1D(np.arange(npt), np.full(npt, value), None, "q_A^-1")
    def two(image, _ai, *, npt_rad, npt_azim, normalization_factor=None, **_kwargs):
        value = float(np.nanmean(image)) / (normalization_factor or 1.0)
        calls.append(("2d", value))
        return IntegrationResult2D(
            np.arange(npt_rad), np.arange(npt_azim),
            np.full((npt_rad, npt_azim), value), None, "q_A^-1", "chi_deg",
        )
    monkeypatch.setattr(core, "integrate_1d", one)
    monkeypatch.setattr(core, "integrate_2d", two)
    return calls


@pytest.fixture
def qapp():
    from pyqtgraph.Qt import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_average_private_request_builds_recipe_and_enumerates_only_on_worker(
    tmp_path, monkeypatch
) -> None:
    from xdart.gui.tabs.scattering.contracts import AcceptedScientificAssets
    from xdart.gui.tabs.scattering import output_preflight
    from xrd_tools.reduction.background import FrameBackgroundPlan
    from xrd_tools.session.experiment_state import (
        CalibrationState, FactStatus, MaskState, PoniValues,
    )
    source = _source(tmp_path)
    class AdvisoryFiles(list):
        armed = False
        def __iter__(self):
            if self.armed and threading.current_thread() is threading.main_thread():
                raise AssertionError("advisory files iterated on the caller thread")
            return super().__iter__()
    mutable_options = copy.deepcopy(dict(source.options))
    mutable_options["files"] = AdvisoryFiles(mutable_options["files"])
    source = SourceSpec(source.uri, source.kind, options=mutable_options)
    reduction_extra = {"nested": [1, {"value": 2}], "enabled_modes_1d": ["q"]}
    reduction = ReductionPlan(
        integration_1d=Integration1DPlan(npt=7, extra=reduction_extra),
        integration_2d=Integration2DPlan(npt_rad=4, npt_azim=3),
    )
    poni_path = str(tmp_path / "accepted.poni"); mask_path = str(tmp_path / "accepted.npy")
    config = {"max_shape": [5, 7], "orientation": 3}
    assets = AcceptedScientificAssets(
        (0.2, 0.0002, 0.0003, 0.0, 0.0, 0.0, 1e-10, "Pilatus300kw"),
        "|b1", (2, 2), bytes((0, 1, 0, 0)), "p" * 64, "m" * 64,
        json.dumps(config, sort_keys=True, separators=(",", ":")),
    )
    calibration = CalibrationState(
        PoniValues(*assets.poni_values[:7]), assets.poni_values[7], config,
        "", assets.poni_sha256, poni_path,
        MaskState(mask_path, assets.mask_sha256, assets.mask_dtype,
                  assets.mask_shape, FactStatus.PRESENT), FactStatus.PRESENT,
    )
    from xrd_tools.core.geometry.diffractometer import DetectorCalibration
    from xrd_tools.integrate import calibration as calibration_module
    real_reconstruct = calibration_module.detector_calibration_to_integrator
    accepted_calibration = assets.detector_calibration
    accepted_detector = real_reconstruct(accepted_calibration).detector
    def detector_truth(value, detector):
        return (
            type(detector), tuple(detector.shape), tuple(detector.max_shape),
            detector.pixel1, detector.pixel2, int(detector.orientation),
            DetectorCalibration(value.poni, detector.get_config()).to_json(),
        )
    accepted_truth = detector_truth(accepted_calibration, accepted_detector)
    background = FrameBackgroundPlan(
        mode="Single BG File", locator=str(source.options["selected_file"]),
    )
    requests = {"workers": 1}; env = {"OMP_NUM_THREADS": "1"}
    target = tmp_path / "average.nxs"; source_base = tmp_path / "source-base"
    source_copy = SourceSpec(source.uri, source.kind, source.metadata_uri, source.entry,
                             copy.deepcopy(dict(source.options)))
    reduction_copy = copy.deepcopy(reduction)
    expected = AverageScanRecipe(
        source_copy, target, reduction_copy, entry="entry", source_base=source_base,
        output_mode="Overwrite", live_mode=True, save_xye=True, batch_mode=True,
        calibration=calibration, background=background,
        numeric_metadata_keys=("I0",), invariant_metadata_keys=("temperature",),
        envelope_bytes=4 << 30, resource_requests=dict(requests), resource_env=dict(env),
    )
    from xrd_tools.reduction import average as average_module
    from xrd_tools.sources import selection
    from xrd_tools.sources.image import TiffSeriesSource
    entered, release = threading.Event(), threading.Event(); calls = []; effects = []
    def observed(name, function):
        def call(*args, **kwargs):
            effects.append((name, threading.current_thread().name))
            return function(*args, **kwargs)
        return call
    for owner in (selection, average_module):
        for name in ("single_image_spec", "image_series_spec"):
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, observed(name, getattr(owner, name)))
    real_stat, real_open, real_iterdir = Path.stat, Path.open, Path.iterdir
    monkeypatch.setattr(Path, "stat", observed("stat", real_stat))
    monkeypatch.setattr(Path, "open", observed("open", real_open))
    monkeypatch.setattr(Path, "iterdir", observed("iterdir", real_iterdir))
    monkeypatch.setattr(average_module, "qualify_source_execution_graph",
                        lambda *_a, **_k: pytest.fail("qualification preceded public runner"))
    monkeypatch.setattr(average_module, "resolve_session_policy",
                        lambda *_a, **_k: pytest.fail("allocation preceded public runner"))
    monkeypatch.setattr(TiffSeriesSource, "metadata_for",
                        lambda *_a, **_k: pytest.fail("metadata preceded public runner"))
    real_recipe = AverageScanRecipe
    def load(intent):
        calls.append(("assets", threading.current_thread().name,
                      intent.poni_file, intent.mask_file))
        entered.set(); assert release.wait(5)
        return assets
    def reconstruct(value):
        result = real_reconstruct(value)
        calls.append(("reconstruct", threading.current_thread().name,
                      id(value), value.to_json(), detector_truth(value, result.detector)))
        return result
    def Recipe(*args, **kwargs):
        value = real_recipe(*args, **kwargs)
        calls.append(("recipe", threading.current_thread().name, value))
        return value
    def run(recipe, **kwargs):
        calls.append(("run", threading.current_thread().name, recipe, kwargs))
        return _result("REFUSED", recipe.target)
    monkeypatch.setattr(output_preflight, "_load_scientific_assets", load)
    monkeypatch.setattr(calibration_module, "detector_calibration_to_integrator", reconstruct)
    monkeypatch.setattr(adapter, "detector_calibration_to_integrator", reconstruct,
                        raising=False)
    monkeypatch.setattr(adapter, "AverageScanRecipe", Recipe)
    monkeypatch.setattr(adapter, "run_average_scan", run)
    slot = OperationSlot()
    mutable_options["files"].armed = True
    identity = slot.begin_average(
        source, target, reduction, entry="entry", source_base=source_base,
        output_mode="Overwrite", live_mode=True, save_xye=True, batch_mode=True,
        poni_file=poni_path, mask_file=mask_path, background=background,
        numeric_metadata_keys=("I0",), invariant_metadata_keys=("temperature",),
        envelope_bytes=4 << 30, resource_requests=requests, resource_env=env,
        stamp=OperationContextStamp(0),
    )
    assert identity is not None and entered.wait(5)
    assert type(slot._frozen).__name__ == "_AverageRequest"
    assert tuple(item.name for item in fields(type(slot._frozen))) == (
        "source_syntax", "target", "target_was_path", "entry", "source_base",
        "source_base_was_path", "output_mode", "live_mode", "save_xye",
        "batch_mode", "reduction_syntax", "poni_file", "mask_file",
        "background_syntax", "numeric_metadata_keys", "invariant_metadata_keys",
        "envelope_bytes", "resource_requests", "resource_env",
    )
    assert slot._frozen.source_syntax == (
        str(source.uri), True, source.kind.value, None, False, source.entry,
        str(source.options["selected_file"]), "scan_*.tif", "scan", None,
        None, None, None, None, None, None,
    )
    assert len(slot._frozen.source_syntax) == 16
    assert calls == [("assets", calls[0][1], poni_path, mask_path)]
    assert calls[0][1].startswith("scattering-operation-")
    assert effects == []
    mutable_options["files"].append(str(tmp_path / "late.tif"))
    reduction_extra["nested"][1]["value"] = 99
    reduction_extra["enabled_modes_1d"].append("chi")
    reduction.integration_1d.npt = 99
    requests["workers"] = 8; env["OMP_NUM_THREADS"] = "8"
    cancel_event = slot._cancel_event; release.set()
    update = _join(slot, identity)
    assert update.terminal.status is OperationTerminalStatus.RETURNED
    assert [row[0] for row in calls] == ["assets", "reconstruct", "reconstruct", "recipe", "run"]
    assert all(row[1].startswith("scattering-operation-") for row in calls)
    assert calls[1][2] != calls[2][2]
    assert calls[1][3:] == calls[2][3:] == (accepted_calibration.to_json(), accepted_truth)
    assert calls[3][2] == expected and calls[3][2].calibration == calibration
    assert calls[3][2].calibration.detector_config["max_shape"] == (5, 7)
    assert calls[3][2].calibration.detector_config["orientation"] == 3
    assert calls[3][2].calibration.value_fingerprint == ""
    assert not hasattr(calls[3][2].calibration.mask, "values")
    assert calls[4][2] is calls[3][2]
    assert calls[4][3]["cancel_token"] is cancel_event
    assert callable(calls[4][3]["publication_gate"])
    assert effects and all(thread.startswith("scattering-operation-")
                           for _name, thread in effects)

    bad_values = list(assets.poni_values); bad_values[6] = 0.0
    detector_config = copy.deepcopy(accepted_detector.get_config())
    mutants = []
    foreign = type("ForeignDetector", (), {})()
    for name in ("shape", "max_shape", "pixel1", "pixel2", "orientation"):
        setattr(foreign, name, getattr(accepted_detector, name))
    foreign.get_config = lambda: copy.deepcopy(detector_config); mutants.append(foreign)
    for name, value in (("shape", (4, 7)), ("max_shape", (6, 7)),
                        ("_pixel1", accepted_detector.pixel1 * 2),
                        ("_pixel2", accepted_detector.pixel2 * 2), ("_orientation", 2)):
        mutant = copy.copy(accepted_detector); setattr(mutant, name, value)
        mutant.get_config = lambda config=detector_config: copy.deepcopy(config)
        mutants.append(mutant)
    mutant = copy.copy(accepted_detector)
    mutant.get_config = lambda: {**copy.deepcopy(detector_config), "orientation": 2}; mutants.append(mutant)
    reconstructed = tuple(type("Reconstructed", (), {"detector": value})()
                          for value in mutants)
    def divergent(result):
        seen = []
        def rebuild(value):
            seen.append(1)
            if len(seen) == 1: return real_reconstruct(value)
            if len(seen) == 2: return result
            return pytest.fail("unexpected third reconstruction")
        return rebuild
    huge = replace(assets, poni_detector_config_json=json.dumps(
        {"orientation": 3, "payload": "x" * 65_536}, sort_keys=True, separators=(",", ":")))
    missing_poni = replace(assets, poni_values=None, poni_sha256=None,
                           poni_detector_config_json=None)
    missing_mask = replace(assets, mask_dtype=None, mask_shape=None,
                           mask_bytes=None, mask_sha256=None)
    def target_bytes():
        with real_open(target, "rb") as stream: return stream.read()
    target.write_bytes(b"prior-average-target"); target_before = target_bytes()
    rows = (
        (AcceptedScientificAssets(tuple(bad_values), assets.mask_dtype, assets.mask_shape,
         assets.mask_bytes, assets.poni_sha256, assets.mask_sha256,
         assets.poni_detector_config_json), real_reconstruct,
         "AVERAGE_CALIBRATION_UNREPRESENTABLE", poni_path, mask_path, 0),
        (replace(assets, mask_shape=(0, 2), mask_bytes=b""), real_reconstruct,
         "AVERAGE_MASK_UNREPRESENTABLE", poni_path, mask_path, 0),
        *((assets, divergent(result),
           "AVERAGE_DETECTOR_CONFIG_UNREPRESENTABLE", poni_path, mask_path, 2)
          for result in reconstructed),
        (huge, None, "AVERAGE_DETECTOR_CONFIG_UNREPRESENTABLE", poni_path, mask_path, 0),
        (missing_poni, None, "AVERAGE_CALIBRATION_UNAVAILABLE", poni_path, mask_path, 0),
        (missing_mask, None, "AVERAGE_MASK_UNAVAILABLE", poni_path, mask_path, 0),
        (OSError("AVERAGE_ASSET_UNSTABLE"), None, "AVERAGE_ASSET_UNSTABLE",
         poni_path, mask_path, 0),
    )
    for loaded, rebuild, diagnostic, requested_poni, requested_mask, expected_rebuilds in rows:
        with monkeypatch.context() as patch:
            effects = []
            def row_load(_intent):
                effects.append("assets")
                if isinstance(loaded, BaseException): raise loaded
                return loaded
            def row_reconstruct(value):
                effects.append("reconstruct")
                return pytest.fail("reconstruction reached after refusal") if rebuild is None else rebuild(value)
            patch.setattr(output_preflight, "_load_scientific_assets", row_load)
            patch.setattr(calibration_module, "detector_calibration_to_integrator", row_reconstruct)
            patch.setattr(adapter, "detector_calibration_to_integrator", row_reconstruct, raising=False)
            patch.setattr(adapter, "AverageScanRecipe", lambda *_a, **_k: pytest.fail("recipe constructed after refusal"))
            patch.setattr(adapter, "run_average_scan", lambda *_a, **_k: pytest.fail("runner reached after refusal"))
            refused = OperationSlot(); refused_identity = refused.begin_average(
                source_copy, target, reduction_copy, poni_file=requested_poni,
                mask_file=requested_mask, stamp=OperationContextStamp(0))
            assert refused_identity is not None
            refused_update = _join(refused, refused_identity)
        assert refused_update.terminal.status is OperationTerminalStatus.FAILED
        assert refused_update.terminal.diagnostic.endswith(f": {diagnostic}")
        assert effects == ["assets"] + ["reconstruct"] * expected_rebuilds
        assert target_bytes() == target_before
    from xrd_tools.sources import DirectorySourceSpec
    directory = OperationSlot(); before = (tuple(calls), tuple(effects), target_bytes())
    assert directory.begin_average(
        DirectorySourceSpec(tmp_path), target, reduction_copy,
        poni_file=poni_path, mask_file=mask_path, stamp=OperationContextStamp(0),
    ) is None
    assert (directory._identity, directory._worker, directory._frozen) == (None, None, None)
    assert (tuple(calls), tuple(effects), target_bytes()) == before


def test_average_publication_gate_linearizes_cancel_wins_and_seal_wins(
    tmp_path, monkeypatch
) -> None:
    from xrd_tools.io import get_average_finite_counts
    from xrd_tools.io.output_transaction import OutputTransaction
    from xrd_tools.reduction import average as average_module
    _stub_integrators(monkeypatch)
    real_run = average_module.run_average_scan

    def exercise(name, *, cancel_first=False, throw=False):
        root = tmp_path / name; source = _source(root)
        target = root / "average.nxs"
        with h5py.File(target, "w") as handle:
            handle.create_dataset("prior", data=np.arange(5, dtype="<i4"))
        before = target.read_bytes()
        before_gate = threading.Event(); after_gate = threading.Event()
        enter_gate = threading.Event(); leave_gate = threading.Event()
        progress_waiting = threading.Event(); progress_release = threading.Event()
        terminal_waiting = threading.Event(); terminal_release = threading.Event()
        gate_results = []; commits = []; progress_seen = []
        real_commit = OutputTransaction.commit_stream
        real_body = OperationSlot._run_average_request
        def commit(owner, *args, **kwargs):
            commits.append(id(owner)); return real_commit(owner, *args, **kwargs)
        def body(owner, *args, **kwargs):
            terminal = real_body(owner, *args, **kwargs)
            terminal_waiting.set(); assert terminal_release.wait(5)
            return terminal
        def run(recipe, **kwargs):
            slot_gate = kwargs["publication_gate"]
            progress_cb = kwargs["progress_cb"]
            def held_progress(value):
                if not progress_seen:
                    progress_seen.append(value); progress_waiting.set()
                    assert progress_release.wait(5)
                return progress_cb(value)
            def held_gate():
                before_gate.set(); assert enter_gate.wait(5)
                if throw:
                    gate_results.append("throw"); after_gate.set()
                    raise OSError("publication gate exploded")
                accepted = slot_gate(); gate_results.append(accepted)
                after_gate.set(); assert leave_gate.wait(5)
                return accepted
            return real_run(recipe, **{**kwargs, "progress_cb": held_progress,
                                       "publication_gate": held_gate})
        with monkeypatch.context() as patch:
            patch.setattr(adapter, "run_average_scan", run)
            patch.setattr(OutputTransaction, "commit_stream", commit)
            patch.setattr(OperationSlot, "_run_average_request", body)
            slot = OperationSlot()
            identity = slot.begin_average(
                source, target, ReductionPlan(
                    integration_1d=Integration1DPlan(npt=3),
                ), stamp=OperationContextStamp(0),
            )
            assert identity is not None and progress_waiting.wait(5)
            assert not slot._cancel_sealed; progress_release.set()
            assert before_gate.wait(5)
            if cancel_first:
                assert slot.cancel(identity); enter_gate.set()
            else:
                enter_gate.set(); assert after_gate.wait(5)
                if not throw:
                    assert not slot.cancel(identity)
            leave_gate.set()
            assert terminal_waiting.wait(5)
            assert len(commits) == (1 if not cancel_first and not throw else 0)
            held = slot.poll(identity)
            assert held is None or held.terminal is None
            terminal_release.set()
            update = _join(slot, identity)
        assert len(gate_results) == 1
        if cancel_first:
            assert gate_results == [False]
            assert update.terminal.status is OperationTerminalStatus.CANCELLED
            assert target.read_bytes() == before
            with pytest.raises((KeyError, ValueError)):
                get_average_finite_counts(target)
        elif throw:
            assert gate_results == ["throw"]
            assert update.terminal.status is OperationTerminalStatus.FAILED
            assert "publication gate exploded" in update.terminal.diagnostic
            assert target.read_bytes() == before
        else:
            assert gate_results == [True]
            assert update.terminal.status is OperationTerminalStatus.RETURNED
            assert update.terminal.payload.disposition == "COMMITTED"
            assert get_average_finite_counts(target).evidence.contributor_extent == 2

    exercise("cancel-wins", cancel_first=True)
    exercise("seal-wins")
    exercise("gate-throws", throw=True)
    notebook = tmp_path / "notebook"
    result = real_run(AverageScanRecipe(
        _source(notebook), notebook / "average.nxs",
        ReductionPlan(integration_1d=Integration1DPlan(npt=3)),
    ), publication_gate=None)
    assert result.disposition == "COMMITTED"


def test_direct_and_gui_average_match_after_fresh_reopen(tmp_path, monkeypatch) -> None:
    from xrd_tools.core.provenance import read_provenance
    from xrd_tools.io import get_1d, get_2d, get_average_finite_counts, get_metadata
    from xrd_tools.reduction import AverageScanRecipe, run_average_scan
    from xrd_tools.reduction import average as average_module
    calls = _stub_integrators(monkeypatch)
    root = tmp_path / "parity"; source = _source(root)
    for index, path in enumerate(source.options["files"], 1):
        Path(path).with_suffix(".txt").write_text(
            f"# Counters\nI0 = {index}.0\n# Motors\n\n"
            "User: p36, time: Mon Jan 15 10:30:00 2024  # Temp\n"
        )
    source = SourceSpec(source.uri, source.kind, options={
        **dict(source.options), "metadata_format": "txt",
    })
    reduction = ReductionPlan(
        integration_1d=Integration1DPlan(npt=4, monitor_key="I0"),
        integration_2d=Integration2DPlan(
            npt_rad=3, npt_azim=2, monitor_key="I0",
        ),
    )
    direct_target = root / "direct.nxs"; gui_target = root / "gui.nxs"
    direct = run_average_scan(AverageScanRecipe(
        source, direct_target, reduction, numeric_metadata_keys=("I0",),
    ))
    assert direct.disposition == "COMMITTED"
    slot = OperationSlot()
    identity = slot.begin_average(
        source, gui_target, reduction, numeric_metadata_keys=("I0",),
        stamp=OperationContextStamp(0),
    )
    assert identity is not None
    update = _join(slot, identity)
    gui = update.terminal.payload
    assert update.terminal.status is OperationTerminalStatus.RETURNED
    assert gui.disposition == "COMMITTED"
    assert direct.science_identity == gui.science_identity
    assert direct.operation_identity != gui.operation_identity
    assert direct.metadata_denominators == gui.metadata_denominators == (("I0", 2),)
    for getter, fields_to_compare in (
        (get_1d, ("q", "intensity", "sigma", "q_unit", "frames")),
        (get_2d, ("q", "chi", "intensity", "q_unit", "chi_unit", "frames")),
    ):
        direct_value = getter(direct_target, frame=1)
        gui_value = getter(gui_target, frame=1)
        for name in fields_to_compare:
            left, right = getattr(direct_value, name), getattr(gui_value, name)
            if isinstance(left, np.ndarray):
                np.testing.assert_allclose(left, right, equal_nan=True)
            else:
                assert left == right
    np.testing.assert_array_equal(get_average_finite_counts(direct_target).values,
                                  get_average_finite_counts(gui_target).values)
    assert tuple(average_module.iter_average_contributors(direct_target)) == tuple(average_module.iter_average_contributors(gui_target))
    direct_provenance = read_provenance(direct_target)["config"]["average_scan_v1"]
    gui_provenance = read_provenance(gui_target)["config"]["average_scan_v1"]
    operation_identities = tuple(value.pop("operation_identity") for value in (direct_provenance, gui_provenance))
    assert direct_provenance == gui_provenance
    assert operation_identities == (direct.operation_identity, gui.operation_identity)
    assert operation_identities[0] != operation_identities[1]
    direct_scan = get_metadata(direct_target)["scan_data"]
    gui_scan = get_metadata(gui_target)["scan_data"]
    assert set(direct_scan) == set(gui_scan) and "I0" in direct_scan
    for name in direct_scan:
        np.testing.assert_allclose(direct_scan[name], gui_scan[name], equal_nan=True)
    np.testing.assert_allclose(direct_scan["I0"], [1.5])
    import hashlib
    for result, target in ((direct, direct_target), (gui, gui_target)):
        commit = result.commit_identity
        assert (commit is not None and commit.target == str(target.resolve())
                and type(commit.ordinal) is int and commit.ordinal > 0)
        assert commit.size == target.stat().st_size and commit.digest == hashlib.sha256(target.read_bytes()).hexdigest()
    assert direct.commit_identity.digest != gui.commit_identity.digest
    assert len(calls) == 4


def test_average_stop_stale_close_and_single_slot_truth(tmp_path, monkeypatch) -> None:
    entered, release = threading.Event(), threading.Event()

    def run(_recipe, *, cancel_token, **_kwargs):
        entered.set(); release.wait(2)
        return _result("CANCELLED" if cancel_token.is_set() else "REFUSED")

    monkeypatch.setattr(adapter, "run_average_scan", run)
    slot = OperationSlot()
    source = _source(tmp_path)
    identity = slot.begin_average(
        source, tmp_path / "one.nxs", ReductionPlan(),
        stamp=OperationContextStamp(1, "context", 2),
    )
    assert identity is not None and entered.wait(2)
    assert slot.begin_average(
        source, tmp_path / "two.nxs", ReductionPlan(),
        stamp=OperationContextStamp(1, "context", 2),
    ) is None
    slot.observe_stamp(OperationContextStamp(2, "context", 2))
    assert slot.cancel(identity)
    pending = slot.close()
    assert pending.cleanup_status.value == "cleanup_pending"
    assert pending.identity is identity and not pending.cancel_accepted
    release.set(); worker = slot._worker; assert worker is not None
    worker.join(5); assert not worker.is_alive()
    cleaned = slot.close()
    assert cleaned.cleanup_status.value == "cleaned"
    assert cleaned.terminal.status is OperationTerminalStatus.CANCELLED
    assert cleaned.stale and not slot.owned
    assert slot.close() is cleaned


def test_average_terminal_projection_preserves_typed_truth_and_reload_boundary(
    tmp_path, monkeypatch, qapp
) -> None:
    from tests.xdart.scattering.test_p3_experiment_operation_composition import _page
    from xdart.gui.tabs.scattering.operation_values import OperationUpdate
    from xrd_tools.reduction import run_average_scan
    source = _source(tmp_path / "source")
    target = tmp_path / "committed.nxs"; reduction = ReductionPlan(integration_1d=Integration1DPlan(npt=3))
    _stub_integrators(monkeypatch)
    committed = run_average_scan(AverageScanRecipe(source, target, reduction))
    assert committed.disposition == "COMMITTED"
    base = _result("REFUSED", str(target.resolve()))
    source_pending = replace(base, disposition="SETTLEMENT_PENDING",
        diagnostic_code="AVERAGE_SOURCE_CLEANUP_PENDING",
        diagnostic="AVERAGE_SOURCE_CLEANUP_PENDING")
    h23_pending = replace(base, disposition="SETTLEMENT_PENDING",
        diagnostic_code="AVERAGE_H23_SETTLEMENT_PENDING",
        diagnostic="AVERAGE_H23_SETTLEMENT_PENDING", h23_phase="ready-to-retry")
    assert source_pending.h23_phase is None and h23_pending.h23_phase == "ready-to-retry"
    malformed_rows = (
        (committed, {"committed_labels": ()}), (committed, {"finite_counts": None}),
        (committed, {"h23_phase": None}), (committed, {"commit_identity": None}),
        (committed, {"contributor_extent": 1}), (base, {"finite_counts": committed.finite_counts}),
        (source_pending, {"h23_phase": "ready-to-retry"}),
        (h23_pending, {"diagnostic_code": "WRONG", "diagnostic": "WRONG"}),
        (base, {"logical_labels": ()}), (base, {"logical_labels": (True,)}), (committed, {"committed_labels": (True,)}),
        (base, {"metadata_denominators": (("", 0),)}),
    )
    for owner, changes in malformed_rows:
        with pytest.raises(ValueError, match="result contract"): replace(owner, **changes)

    def scheduled(result, *, verification=None):
        with monkeypatch.context() as patch:
            patch.setattr(adapter, "run_average_scan", lambda *_a, **_k: result)
            slot = OperationSlot()
            identity = slot.begin_average(source, target, reduction, stamp=OperationContextStamp(0))
            assert identity is not None
            update = _join(slot, identity)
        expected_status = {"COMMITTED": OperationTerminalStatus.RETURNED,
            "REFUSED": OperationTerminalStatus.RETURNED, "CANCELLED": OperationTerminalStatus.CANCELLED,
            "ABORTED": OperationTerminalStatus.FAILED}[result.disposition]
        expected_diagnostic = f"{result.diagnostic_code}: {result.diagnostic}" if result.disposition == "ABORTED" else ""
        if verification is not None:
            expected_status = OperationTerminalStatus.FAILED
            expected_diagnostic = f"AVERAGE_COMMIT_VERIFICATION_FAILED: {verification}"
        assert update.terminal.status is expected_status
        assert update.terminal.payload is result
        assert update.terminal.diagnostic == expected_diagnostic
        return identity, update

    page, store = _page(tmp_path, monkeypatch)
    reloads = []; reintegrate_reloads = []; catalog = []; notices = []
    monkeypatch.setattr(page._context_controller, "begin_browse", lambda value:
        reloads.append(value))
    monkeypatch.setattr(
        page._context_controller, "reload_reintegrate_browse",
        lambda *args: reintegrate_reloads.append(args),
    )
    monkeypatch.setattr(page, "_request_browser_catalog", lambda: catalog.append(1))
    monkeypatch.setattr(page, "_notice", lambda value: notices.append(value))

    def arm(identity):
        page._average_identity = identity; page._average_revision = store.revision
        page._average_target = str(target.resolve()); page._average_entry = "entry"

    identity, update = scheduled(committed)
    arm(identity)
    from xrd_tools.core import provenance as provenance_module
    def forbidden(*_args, **_kwargs):
        pytest.fail("the page performed committed-artifact I/O")
    with monkeypatch.context() as patch:
        patch.setattr(provenance_module, "read_provenance", forbidden)
        patch.setattr(Path, "stat", forbidden); patch.setattr(Path, "open", forbidden)
        assert page._consume_average_update(update)
    assert reloads == [str(target.resolve())] and catalog == [1] and reintegrate_reloads == []
    assert all(getattr(page, name) is None for name in (
        "_average_identity", "_average_revision", "_average_target", "_average_entry",
    ))

    expected_notice = {
        "REFUSED": "AVERAGE_TEST_REFUSED: detached refusal",
        "CANCELLED": "Average cancelled.",
        "ABORTED": "AVERAGE_TEST_ABORT: detached diagnostic",
    }
    for disposition in ("REFUSED", "CANCELLED", "ABORTED"):
        result = _result(disposition, str(target.resolve()))
        current, terminal = scheduled(result)
        arm(current); before_notices = len(notices)
        assert page._consume_average_update(terminal)
        assert len(reloads) == 1
        assert len(notices) == before_notices + 1
        assert notices[-1] == expected_notice[disposition]

    wrong_commit = replace(committed.commit_identity, digest="e" * 64)
    for token, malformed in (
        ("target", _result("COMMITTED", str(tmp_path / "wrong.nxs"))),
        ("entry", replace(committed, entry="wrong")),
        ("operation", replace(committed, operation_identity="e" * 64)),
        ("science", replace(committed, science_identity="f" * 64)),
        ("commit", replace(committed, commit_identity=wrong_commit)),
    ):
        bad, update = scheduled(malformed, verification=token)
        arm(bad); before_notices = len(notices)
        assert page._consume_average_update(update)
        assert len(reloads) == 1 and len(notices) == before_notices + 1
        assert notices[-1] == (
            f"Average failed: AVERAGE_COMMIT_VERIFICATION_FAILED: {token}"
        )

    stale, stale_update = scheduled(committed)
    arm(stale); before_notices = len(notices)
    assert page._consume_average_update(OperationUpdate(
        stale, terminal=stale_update.terminal, stale=True,
    ))
    assert len(reloads) == 1
    assert len(notices) == before_notices

    from xrd_tools.sources import execution_graph
    entered, release = threading.Event(), threading.Event()
    real_close = execution_graph._AverageSourceReadWindow.close; closes = []
    def held_close(window):
        closes.append(id(window))
        if len(closes) == 1: return real_close(window)
        entered.set()
        if not release.is_set(): raise OSError("retained source cleanup")
        return real_close(window)
    monkeypatch.setattr(execution_graph._AverageSourceReadWindow, "close", held_close)
    pending_source = _source(tmp_path / "pending-source")
    slot = OperationSlot(); pending_identity = slot.begin_average(
        pending_source, tmp_path / "pending.nxs", reduction,
        stamp=OperationContextStamp(0),
    )
    assert pending_identity is not None and entered.wait(5)
    pending = slot.poll(pending_identity)
    assert slot._worker is not None and slot._worker.is_alive()
    assert pending is None or pending.terminal is None
    release.set(); terminal = _join(slot, pending_identity)
    assert terminal.terminal.payload.disposition == "COMMITTED"
    page.close_workspace(); page.deleteLater(); qapp.processEvents()
