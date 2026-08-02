from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from xrd_tools.core import energy as core_energy
from xrd_tools.session.experiment_state import (
    CalibrationState,
    CasStatus,
    EnergySource,
    EnergyState,
    ExperimentEditor,
    ExperimentEditorPort,
    ExperimentProvenance,
    ExperimentState,
    FactStatus,
    GeometryState,
    IncidenceKind,
    IncidenceState,
    MaskState,
    OrientationState,
    PoniValues,
    SampleState,
)


def _calibration(
    *,
    wavelength_m: float = 1.0e-10,
    mask: MaskState | None = None,
    detector_config: dict[str, float] | None = None,
) -> CalibrationState:
    return CalibrationState(
        values=PoniValues(
            dist=0.2,
            poni1=0.01,
            poni2=0.02,
            rot1=0.0,
            rot2=0.1,
            rot3=-0.1,
            wavelength_m=wavelength_m,
        ),
        detector_id="Eiger4M",
        detector_config=detector_config or {"pixel1": 75e-6, "pixel2": 75e-6},
        value_fingerprint="v" * 64,
        source_sha256="s" * 64,
        source_uri="calibration.poni",
        mask=mask or MaskState.absent(),
        status=FactStatus.PRESENT,
    )


def _state(
    *,
    experiment_id: str = "experiment-a",
    revision: int = 3,
    wavelength_m: float = 1.1e-10,
    evidence: dict[str, float] | None = None,
    detector_config: dict[str, float] | None = None,
    notes: tuple[str, ...] | list[str] = ("imported", "reviewed"),
) -> ExperimentState:
    return ExperimentState(
        experiment_id=experiment_id,
        revision=revision,
        calibration=_calibration(detector_config=detector_config),
        geometry=GeometryState(
            gi_enabled=True,
            incidence=IncidenceState(
                kind=IncidenceKind.MOTOR,
                motor_name="th",
                motor_resolved=True,
            ),
            tilt_deg=0.2,
            convention_id="pyfai-fiber-v1",
        ),
        energy=EnergyState(
            wavelength_m=wavelength_m,
            source=EnergySource.OPERATOR,
            evidence=evidence or {
                "operator": wavelength_m,
                "calibration": 1.0e-10,
            },
            status=FactStatus.CONFLICT,
        ),
        sample=SampleState(
            sample_id="sample-17",
            display_name="LaB6 reference",
            orientation=OrientationState(
                code=4,
                convention_id="pyfai-fiber-sample-orientation-v1",
            ),
            tags=("powder", "standard"),
        ),
        provenance=ExperimentProvenance(
            source_kind="operator",
            source_uri="beamtime.json",
            digest="d" * 64,
            imported_at="2026-08-02T12:00:00Z",
            operator="maintainer",
            notes=notes,
        ),
    )


def test_scientific_projection_and_fingerprint_are_deterministic() -> None:
    first = _state(
        evidence={"operator": 1.1e-10, "calibration": 1.0e-10},
        detector_config={"pixel2": 75e-6, "pixel1": 75e-6},
        notes=["imported", "reviewed"],
    )
    second = _state(
        evidence={"calibration": 1.0e-10, "operator": 1.1e-10},
        detector_config={"pixel1": 75e-6, "pixel2": 75e-6},
        notes=("imported", "reviewed"),
    )

    assert first.scientific_content() == second.scientific_content()
    assert first.content_fingerprint == second.content_fingerprint
    encoded = json.dumps(
        first.scientific_content(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert "experiment-a" not in encoded
    assert first.to_provenance()["content_fingerprint"] == first.content_fingerprint


def test_identity_and_revision_are_outside_hash_but_remain_coordinates() -> None:
    original = _state(experiment_id="a", revision=3)
    foreign = _state(experiment_id="b", revision=99)

    assert original.content_fingerprint == foreign.content_fingerprint
    assert original.to_provenance()["experiment_id"] == "a"
    assert foreign.to_provenance()["revision"] == 99
    changed = dataclasses.replace(
        original,
        energy=EnergyState(
            wavelength_m=1.2e-10,
            source=EnergySource.OPERATOR,
            evidence={"operator": 1.2e-10, "calibration": 1.0e-10},
            status=FactStatus.CONFLICT,
        ),
    )
    assert changed.content_fingerprint != original.content_fingerprint


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PoniValues(0.2, 0.0, 0.0, 0.0, 0.0, float("nan"), 1e-10),
        lambda: GeometryState(tilt_deg=float("inf")),
        lambda: EnergyState(
            wavelength_m=0.0,
            source=EnergySource.OPERATOR,
            evidence={"operator": 1e-10},
            status=FactStatus.PRESENT,
        ),
        lambda: EnergyState(
            wavelength_m=1e-10,
            source=EnergySource.OPERATOR,
            evidence={"operator": float("nan")},
            status=FactStatus.PRESENT,
        ),
        lambda: SampleState(
            sample_id="s",
            display_name="sample",
            orientation=OrientationState(0, "orientation-v1"),
        ),
        lambda: ExperimentProvenance(notes=(object(),)),
    ],
)
def test_closed_fields_reject_nonfinite_nonpositive_and_non_json(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_components_are_deeply_immutable() -> None:
    state = _state(detector_config={"pixel1": 75e-6, "offset": 0.0})
    assert state.calibration.detector_config["offset"] == 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.revision = 4  # type: ignore[misc]
    with pytest.raises(TypeError):
        state.energy.evidence["late"] = 1.5e-10  # type: ignore[index]
    with pytest.raises(TypeError):
        state.calibration.detector_config["pixel1"] = 1.0  # type: ignore[index]


def test_energy_status_cannot_hide_evidence_or_an_unqualified_selection() -> None:
    with pytest.raises(ValueError):
        EnergyState(evidence={"operator": 1.0e-10})
    with pytest.raises(ValueError):
        EnergyState(
            wavelength_m=1.0e-10,
            evidence={"operator": 1.1e-10},
            status=FactStatus.CONFLICT,
        )
    with pytest.raises(ValueError):
        EnergyState(
            wavelength_m=1.0e-10,
            source=EnergySource.OPERATOR,
            evidence={"operator": 1.1e-10},
            status=FactStatus.PRESENT,
        )


def test_schema_version_rejects_bool_alias_for_one() -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(_state(), schema_version=True)


def test_geometry_has_no_poni_or_wavelength_authority() -> None:
    forbidden = {
        "dist",
        "poni1",
        "poni2",
        "rot1",
        "rot2",
        "rot3",
        "wavelength",
        "wavelength_m",
        "detector",
        "detector_id",
    }
    assert forbidden.isdisjoint(GeometryState.__dataclass_fields__)
    state = _state(wavelength_m=1.1e-10)
    assert state.calibration.values is not None
    assert state.calibration.values.wavelength_m == pytest.approx(1.0e-10)
    assert state.energy.wavelength_m == pytest.approx(1.1e-10)


def test_sample_identity_label_and_orientation_are_independent() -> None:
    sample = _state().sample
    assert sample.sample_id == "sample-17"
    assert sample.display_name == "LaB6 reference"
    assert sample.orientation.code == 4
    assert len({sample.sample_id, sample.display_name, str(sample.orientation.code)}) == 3


def test_energy_conversion_delegates_to_existing_core_functions(monkeypatch) -> None:
    seen: list[tuple[str, float]] = []

    def to_wavelength(value: float) -> float:
        seen.append(("to-wavelength", value))
        return 2.5e-10

    def to_energy(value: float) -> float:
        seen.append(("to-energy", value))
        return 43210.0

    monkeypatch.setattr(core_energy, "energy_eV_to_wavelength_m", to_wavelength)
    monkeypatch.setattr(core_energy, "wavelength_m_to_energy_eV", to_energy)

    energy = EnergyState.from_energy_eV(9000.0, source=EnergySource.OPERATOR)
    assert energy.wavelength_m == pytest.approx(2.5e-10)
    assert energy.energy_eV == pytest.approx(43210.0)
    assert seen == [("to-wavelength", 9000.0), ("to-energy", 2.5e-10)]


def test_editor_cas_is_identity_and_revision_qualified() -> None:
    initial = _state(experiment_id="experiment-a", revision=3)
    editor = ExperimentEditor(initial)
    assert isinstance(editor, ExperimentEditorPort)

    candidate = _calibration(wavelength_m=1.2e-10)
    foreign = editor.propose_calibration(
        candidate,
        experiment_id="experiment-b",
        expected_revision=3,
    )
    assert foreign.status is CasStatus.FOREIGN_IDENTITY
    assert editor.current() is initial

    stale = editor.propose_calibration(
        candidate,
        experiment_id="experiment-a",
        expected_revision=2,
    )
    assert stale.status is CasStatus.STALE_REVISION
    assert editor.current() is initial

    accepted = editor.propose_calibration(
        candidate,
        experiment_id="experiment-a",
        expected_revision=3,
    )
    assert accepted.status is CasStatus.ACCEPTED
    assert accepted.state.revision == 4
    assert editor.current() is accepted.state
    assert accepted.state.energy.wavelength_m == initial.energy.wavelength_m
    assert accepted.state.energy.evidence["calibration"] == pytest.approx(1.2e-10)

    no_change = editor.propose_calibration(
        candidate,
        experiment_id="experiment-a",
        expected_revision=4,
    )
    assert no_change.status is CasStatus.NO_CHANGE
    assert no_change.state is accepted.state


def test_editor_mask_proposal_changes_only_mask_and_one_revision() -> None:
    initial = _state()
    editor = ExperimentEditor(initial)
    mask = MaskState(
        source_uri="mask.edf",
        sha256="m" * 64,
        dtype="bool",
        shape=(2167, 2070),
        status=FactStatus.PRESENT,
    )
    outcome = editor.propose_mask(
        mask,
        experiment_id=initial.experiment_id,
        expected_revision=initial.revision,
    )
    assert outcome.status is CasStatus.ACCEPTED
    assert outcome.state.revision == initial.revision + 1
    assert outcome.state.calibration.mask is mask
    assert outcome.state.energy is initial.energy
    assert outcome.state.geometry is initial.geometry
    assert outcome.state.sample is initial.sample


def test_q3_modules_keep_the_frozen_import_and_owner_boundary() -> None:
    root = Path(__file__).resolve().parents[2]
    state_path = root / "src/xrd_tools/session/experiment_state.py"
    reload_path = root / "src/xrd_tools/session/experiment_reload.py"
    combined = state_path.read_text() + reload_path.read_text()
    forbidden = (
        "xrd_tools.session.run_configuration",
        "from .run_configuration",
        "import run_configuration",
        "ScanNormAggregate",
        "WavelengthUnit",
        "canonical_wavelength_m",
        "StartPipeline",
        "PageKey",
        "xdart.gui",
        "PySide",
        "PyQt",
        "qtpy",
    )
    assert {needle for needle in forbidden if needle in combined} == set()
    # The imported PONI wavelength is read only for energy arbitration and its
    # explicit provenance projection; no writer/integrator/display consumer exists.
    assert combined.count("values.wavelength_m") == 2


def test_q3_module_import_is_headless_and_does_not_eagerly_load_h5py() -> None:
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    code = """
import sys
before = set(sys.modules)
import xrd_tools.session.experiment_state
import xrd_tools.session.experiment_reload
loaded = set(sys.modules) - before
forbidden = ("h5py", "PySide", "PyQt", "qtpy", "pyFAI", "fabio", "xdart")
bad = sorted(name for name in loaded if any(
    name == root or name.startswith(root + ".") for root in forbidden
))
raise SystemExit("\\n".join(bad) if bad else 0)
"""
    result = subprocess.run([sys.executable, "-c", code], env=env, text=True,
                            capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
