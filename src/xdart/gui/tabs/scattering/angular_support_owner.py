"""The one owner of a view's angular coverage reconstructions (COVERAGE-SUPPORT-20261013, D5).

An older Int output saved no angular coverage; Browse alone may rebuild a frame's from the output's own record
(``xrd_tools.analysis.angular_support.reconstruct_saved_support``). The strip's view mounts this owner and asks it, once
per draw, for the shown frames of that output: one request for the whole shown set, its basis built once. The owner
keeps exactly the shown set's completed profiles, so a redraw, a reorder or a smaller selection asks for nothing, and
a changed selection keeps every frame still shown and asks one job for the rest (the shared ``LatestRequestOwner``:
one computation running, the latest asked for winning). Nothing else is reconstructed: saved, run, disabled (“Record
angular coverage” off) and Stitch results carry their own profile or reason, and the view never asks for them.
"""
from __future__ import annotations

from pyqtgraph.Qt import QtCore

from xdart.gui.pages.operation_owner import OperationTerminalStatus
from xdart.gui.widgets.latest_request import LatestRequestOwner
from xrd_tools.core.angular_support import AngularSupportProfile, SupportOrigin, SupportReason

#: A profile's facts beside its values (window, edges, provenance), counted in its retained bytes.
_PROFILE_FACT_BYTES = 4096


def _reconstruct(request, cancel, _progress):
    """The worker's job: the requested frames rebuilt from their output's record (bound late, so tests can hold it)."""
    from xrd_tools.analysis.angular_support import reconstruct_saved_support
    return reconstruct_saved_support(request, cancel=cancel.is_set)


def _profile_bytes(profile: AngularSupportProfile) -> int:
    return _PROFILE_FACT_BYTES + (0 if profile.supported_deg is None else int(profile.supported_deg.nbytes))


class AngularSupportOwner(QtCore.QObject):
    """Older Int frames' reconstructions for one view's strip: the shown set's completed profiles, bounded by what
    the view shows (Browse's resident rows), and one job for the shown frames still missing. ``clear`` (the strip off,
    the view cleared) and ``shutdown`` (the page closing) retire it: CLEAN, or PENDING while a computation returns."""

    #: A reconstruction completed: the view draws it.
    reconstructed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # the owner asks only for frames it does not keep, so a request is never "unchanged" from the shown one
        self._requests = LatestRequestOwner(lambda request, cancel, progress: _reconstruct(request, cancel, progress),
                                            self._done, same=lambda a, b: a.key == b.key,
                                            unchanged=lambda _request, _shown: False, parent=self)
        self._source: tuple | None = None
        self._shown: frozenset[int] = frozenset()
        self._completed: dict[int, AngularSupportProfile] = {}
        self._completions = 0

    @property
    def worker(self):
        """The ``SingleWorkerOwner`` computing, for hosts that report or retire it."""
        return self._requests.worker

    @property
    def computing(self) -> bool:
        """Whether a reconstruction asked for has not completed."""
        return self._requests.computing

    def profiles(self, source, labels) -> dict[int, AngularSupportProfile]:
        """The completed profiles of ``source``'s shown ``labels`` (any order); the missing ones are being rebuilt
        in one job. A frame no longer shown is let go; another output or load starts afresh."""
        identity = (source.path, source.entry, source.snapshot, source.labels)
        shown = frozenset(int(label) for label in labels)
        if identity != self._source:
            self._source, self._completed = identity, {}
        if shown != self._shown:
            self._shown = shown
            self._completed = {label: profile for label, profile in self._completed.items() if label in shown}
        missing = tuple(sorted(shown - self._completed.keys()))
        if missing:
            self._requests.ask(source.request(missing))
        else:
            self._requests.ask(None)  # a wholly retained selection retires an obsolete expansion
        return dict(self._completed)

    def completed_requests(self) -> int:
        """How many reconstruction jobs have completed (a set shown again from its kept profiles adds none)."""
        return self._completions

    def retained_keys(self) -> int:
        return len(self._completed)

    def retained_bytes(self) -> int:
        return sum(_profile_bytes(profile) for profile in self._completed.values())

    def clear(self) -> None:
        """Let every kept profile go, and drop a computation still running when it returns."""
        self._source, self._shown, self._completed = None, frozenset(), {}
        self._requests.release()

    def shutdown(self):
        """Close with the view's page: CLEAN, or PENDING while a computation still runs (the host asks again)."""
        self._source, self._shown, self._completed = None, frozenset(), {}
        return self._requests.shutdown()

    def _done(self, request, terminal) -> bool:
        self._completions += 1
        if (request.path, request.entry, request.snapshot, request.labels) == self._source:
            if terminal.status is OperationTerminalStatus.RETURNED and type(terminal.payload) is dict:
                rebuilt = terminal.payload
            else:
                failed = AngularSupportProfile.unavailable(
                    SupportReason.LEGACY_INSUFFICIENT, origin=SupportOrigin.RECONSTRUCTED,
                    detail=str(terminal.failure_message or "the reconstruction failed")[:200])
                rebuilt = {label: failed for label in request.selected}
            self._completed.update({label: profile for label, profile in rebuilt.items()
                                    if label in self._shown and type(profile) is AngularSupportProfile})
        self.reconstructed.emit()
        return True


__all__ = ["AngularSupportOwner"]
