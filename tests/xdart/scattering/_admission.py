from pathlib import Path
import time

from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import FrozenRunConfiguration, RunIntent
from xdart.gui.tabs.scattering.adapters.run_executor import (
    StandardRunExecutor,
    _AdmissionOperation,
)
from xdart.gui.tabs.scattering.contracts import (
    AdmittedOutput,
    AdmissionFailure,
    AdmissionReceipt,
    AdmissionReleased,
    AdmissionToken,
    OutputFact,
    SourceCapture,
    StartCapture,
)
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.output_preflight import (
    AcceptedScientificAssets,
    OutputCandidate,
    OutputDisposition,
    _series_item,
)


def admission_for(capture: StartCapture) -> AdmissionReceipt:
    candidate = OutputCandidate(
        capture.source_capture.source,
        "",
        "",
        "",
        "{}",
        "",
    )
    return AdmissionReceipt(
        capture.request_id,
        capture.intent_snapshot.revision,
        capture.source_capture,
        candidate,
        (),
        AcceptedScientificAssets(None, None, None, None, None, None),
        capture.source_capture.gi_motor_choices,
    )


def install_admission(
    executor: StandardRunExecutor,
    configuration: FrozenRunConfiguration,
    capture: SourceCapture,
) -> AdmissionReceipt:
    source = configuration.thaw_source_spec()
    path = Path(source.options.get("selected_file") or source.uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    snapshot = RunIntentStore(RunIntent.from_frozen(configuration)).snapshot()
    start_capture = StartCapture(capture.request_id, 1, snapshot, capture)
    candidate = OutputCandidate.from_start_capture(
        start_capture,
        AcceptedScientificAssets(None, None, None, None, None, None),
        (),
    )
    item = _series_item(candidate, source)
    receipt = AdmissionReceipt(
        capture.request_id,
        0,
        capture,
        candidate,
        (
            AdmittedOutput(
                item,
                OutputDisposition.WRITE,
                (1,),
                OutputFact(False),
            ),
        ),
        AcceptedScientificAssets(None, None, None, None, None, None),
        (),
    )
    token = AdmissionToken(capture.request_id, 0)
    operation = _AdmissionOperation(token, start_capture)
    operation.finish_worker(receipt)
    executor._admission = operation
    return receipt


class ImmediateAdmission:
    def begin_admission(self, capture: StartCapture) -> AdmissionToken:
        token = AdmissionToken(
            capture.request_id,
            capture.intent_snapshot.revision,
        )
        self._test_admission = (token, admission_for(capture))
        return token

    def poll_admission(self, token: AdmissionToken):
        owned, receipt = self._test_admission
        return receipt if token is owned else None

    def cancel_admission(self, token: AdmissionToken) -> AdmissionReleased:
        if getattr(self, "_test_admission", (None,))[0] is token:
            self._test_admission = (None, None)
        return AdmissionReleased(token, CleanupStatus.CLEANED)

    def release_admission(self, token: AdmissionToken) -> AdmissionReleased:
        return self.cancel_admission(token)


def await_admission(
    executor: StandardRunExecutor,
    capture: StartCapture,
    timeout: float = 30.0,
) -> AdmissionReceipt:
    token = executor.begin_admission(capture)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = executor.poll_admission(token)
        if type(result) is AdmissionReceipt:
            return result
        if type(result) is AdmissionFailure:
            raise AssertionError(result.reason)
        time.sleep(0.01)
    raise AssertionError("output admission did not finish")
