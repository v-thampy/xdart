"""Science-free host services, bound to one selected page key at mount."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, TYPE_CHECKING

from .values import PageKey

if TYPE_CHECKING:
    from xrd_tools.session.experiment_state import ExperimentEditorPort


class StatusPresenter(Protocol):
    def show(self, text: str, timeout_ms: int = 0) -> None: ...


class RunIntentProvider(Protocol):
    def store_for(self, key: PageKey) -> object | None: ...


class ExecutionProvider(Protocol):
    def executor_for(self, key: PageKey) -> object | None: ...


class SourceProvider(Protocol):
    def source_port_for(self, key: PageKey) -> object | None: ...


class _ExperimentProvider(Protocol):
    def experiment_for(self, key: PageKey) -> ExperimentEditorPort | None: ...


class ExecutionProfile(str, Enum):
    LIVE = "live"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class DiagnosticIdentity:
    logger_namespace: str

    def __post_init__(self) -> None:
        if not self.logger_namespace.strip():
            raise ValueError("diagnostic logger namespace must be nonempty")


@dataclass(frozen=True, slots=True)
class _SelectedRunIntents:
    selected: PageKey
    provider: RunIntentProvider

    def store_for(self, key: PageKey) -> object | None:
        if key != self.selected:
            return None
        return self.provider.store_for(self.selected)


@dataclass(frozen=True, slots=True)
class _SelectedExecution:
    selected: PageKey
    provider: ExecutionProvider

    def executor_for(self, key: PageKey) -> object | None:
        if key != self.selected:
            return None
        return self.provider.executor_for(self.selected)


@dataclass(frozen=True, slots=True)
class _SelectedSources:
    selected: PageKey
    provider: SourceProvider

    def source_port_for(self, key: PageKey) -> object | None:
        if key != self.selected:
            return None
        return self.provider.source_port_for(self.selected)


@dataclass(frozen=True, slots=True)
class _SelectedExperiments:
    selected: PageKey
    provider: _ExperimentProvider

    def experiment_for(self, key: PageKey) -> ExperimentEditorPort | None:
        if key != self.selected:
            return None
        return self.provider.experiment_for(self.selected)


class _NullExperiments:
    def experiment_for(self, _key: PageKey) -> None:
        return None


@dataclass(frozen=True, slots=True)
class HostServices:
    status: StatusPresenter
    run_intents: RunIntentProvider
    execution: ExecutionProvider
    sources: SourceProvider
    execution_profile: ExecutionProfile
    diagnostics: DiagnosticIdentity
    _experiments: _ExperimentProvider = field(
        default_factory=_NullExperiments,
        repr=False,
    )

    def experiment_for(self, key: PageKey) -> ExperimentEditorPort | None:
        """Return the borrowed experiment editor for that exact key, or None."""
        return self._experiments.experiment_for(key)

    def for_page(self, key: PageKey) -> "HostServices":
        """Return providers that refuse every key except the selected one."""
        return HostServices(
            status=self.status,
            run_intents=_SelectedRunIntents(key, self.run_intents),
            execution=_SelectedExecution(key, self.execution),
            sources=_SelectedSources(key, self.sources),
            execution_profile=self.execution_profile,
            diagnostics=self.diagnostics,
            _experiments=_SelectedExperiments(key, self._experiments),
        )


class _NullRunIntents:
    def store_for(self, _key: PageKey) -> None:
        return None


class _NullExecution:
    def executor_for(self, _key: PageKey) -> None:
        return None


class _NullSources:
    def source_port_for(self, _key: PageKey) -> None:
        return None


def empty_host_services(
    status: StatusPresenter,
    *,
    execution_profile: ExecutionProfile = ExecutionProfile.LIVE,
    logger_namespace: str = "xdart.gui.pages",
) -> HostServices:
    return HostServices(
        status=status,
        run_intents=_NullRunIntents(),
        execution=_NullExecution(),
        sources=_NullSources(),
        execution_profile=execution_profile,
        diagnostics=DiagnosticIdentity(logger_namespace),
    )
