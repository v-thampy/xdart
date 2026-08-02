"""J1 frozen oracle: the selected-page experiment editor provider seam.

Frozen RED on exact parent ``715baf54`` for every ``HostServices`` row (the
``experiments`` field and ``experiment_for`` method are absent there), while
the accepted Q3 editor row and the type-only import row are green. The
backing provider is a spy around real Q3 editors, per the J1 kickoff.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from xdart.gui.pages.services import (
    DiagnosticIdentity,
    ExecutionProfile,
    HostServices,
    empty_host_services,
)
from xdart.gui.pages.values import PageKey
from xrd_tools.session.experiment_state import (
    CalibrationState,
    CasStatus,
    EnergySource,
    EnergyState,
    ExperimentEditor,
    ExperimentEditorPort,
    ExperimentState,
    FactStatus,
    GeometryState,
    MaskState,
    SampleState,
)

SELECTED = PageKey("scattering-workspace")
FOREIGN = PageKey("static-scan")


def _editor() -> ExperimentEditor:
    return ExperimentEditor(ExperimentState(
        experiment_id="exp-j1",
        revision=0,
        calibration=CalibrationState(),
        geometry=GeometryState(),
        energy=EnergyState.from_energy_eV(
            12_398.0, source=EnergySource.OPERATOR),
        sample=SampleState(sample_id="sample-j1"),
    ))


class _Status:
    def show(self, text: str, timeout_ms: int = 0) -> None:
        return None


class _NoneIntents:
    def store_for(self, _key):
        return None


class _NoneExecution:
    def executor_for(self, _key):
        return None


class _NoneSources:
    def source_port_for(self, _key):
        return None


class _EditorSpy:
    """Backing provider spy over real Q3 editors; records every consult."""

    def __init__(self, editors):
        self.editors = dict(editors)
        self.calls = []

    def experiment_for(self, key):
        self.calls.append(key)
        return self.editors.get(key)


def _host(experiments) -> HostServices:
    return HostServices(
        status=_Status(),
        run_intents=_NoneIntents(),
        execution=_NoneExecution(),
        sources=_NoneSources(),
        experiments=experiments,
        execution_profile=ExecutionProfile.TEST,
        diagnostics=DiagnosticIdentity("tests.j1"),
    )


def test_q3_editor_protocol_and_owner_are_present_and_functional():
    editor = _editor()
    assert isinstance(editor, ExperimentEditorPort)
    state = editor.current()
    outcome = editor.propose_mask(
        MaskState(source_uri="mask://j1", sha256="digest", dtype="uint8",
                  shape=(2, 2), status=FactStatus.PRESENT),
        experiment_id=state.experiment_id,
        expected_revision=state.revision,
    )
    assert outcome.status is CasStatus.ACCEPTED
    assert editor.current().revision == state.revision + 1


def test_page_bound_services_deliver_the_exact_selected_editor():
    editor = _editor()
    spy = _EditorSpy({SELECTED: editor})
    bound = _host(spy).for_page(SELECTED)
    assert bound.experiment_for(SELECTED) is editor
    assert spy.calls == [SELECTED]


def test_foreign_key_returns_none_before_consulting_the_provider():
    spy = _EditorSpy({SELECTED: _editor()})
    bound = _host(spy).for_page(SELECTED)
    assert bound.experiment_for(FOREIGN) is None
    assert spy.calls == []


def test_two_page_bound_views_cannot_cross_deliver_editors():
    editor_a, editor_b = _editor(), _editor()
    spy = _EditorSpy({SELECTED: editor_a, FOREIGN: editor_b})
    host = _host(spy)
    view_a, view_b = host.for_page(SELECTED), host.for_page(FOREIGN)
    assert view_a.experiment_for(SELECTED) is editor_a
    assert view_b.experiment_for(FOREIGN) is editor_b
    assert view_a.experiment_for(FOREIGN) is None
    assert view_b.experiment_for(SELECTED) is None
    assert spy.calls == [SELECTED, FOREIGN]


def test_empty_host_services_return_none_for_every_key():
    services = empty_host_services(_Status())
    bound = services.for_page(SELECTED)
    for key in (SELECTED, FOREIGN):
        assert services.experiment_for(key) is None
        assert bound.experiment_for(key) is None


def test_existing_provider_isolation_is_unchanged_beside_the_new_seam():
    calls = []

    class _Intents:
        def store_for(self, key):
            calls.append(("intent", key))
            return ("intent", key)

    class _Execution:
        def executor_for(self, key):
            calls.append(("executor", key))
            return ("executor", key)

    class _Sources:
        def source_port_for(self, key):
            calls.append(("source", key))
            return ("source", key)

    bound = HostServices(
        status=_Status(),
        run_intents=_Intents(),
        execution=_Execution(),
        sources=_Sources(),
        experiments=_EditorSpy({}),
        execution_profile=ExecutionProfile.TEST,
        diagnostics=DiagnosticIdentity("tests.j1"),
    ).for_page(SELECTED)
    assert bound.run_intents.store_for(SELECTED) == ("intent", SELECTED)
    assert bound.execution.executor_for(SELECTED) == ("executor", SELECTED)
    assert bound.sources.source_port_for(SELECTED) == ("source", SELECTED)
    assert bound.run_intents.store_for(FOREIGN) is None
    assert bound.execution.executor_for(FOREIGN) is None
    assert bound.sources.source_port_for(FOREIGN) is None
    assert calls == [
        ("intent", SELECTED), ("executor", SELECTED), ("source", SELECTED)]


def test_importing_services_does_not_eagerly_load_the_q3_module():
    root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    code = (
        "import sys\n"
        "import xdart.gui.pages.services\n"
        "assert 'xrd_tools.session.experiment_state' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
