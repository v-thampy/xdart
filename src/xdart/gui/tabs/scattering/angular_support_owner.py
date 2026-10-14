"""The one owner of a view's angular coverage reconstructions (COVERAGE-SUPPORT-20261013, D5).

An older Int output saved no angular coverage; Browse alone may rebuild a frame's from the output's own record
(``xrd_tools.analysis.angular_support.reconstruct_saved_support``). The strip's view mounts this owner and asks it for
the frames it shows; it composes the shared ``LatestRequestOwner`` (one computation running, the latest asked for
winning) and keeps a few completed profiles, so a frame shown again is not computed again. Nothing else is
reconstructed: saved, run, disabled ("Record angular coverage" off) and Stitch results carry their own profile or
reason, and the view never asks for them.
"""
from __future__ import annotations

from collections import OrderedDict

from pyqtgraph.Qt import QtCore

from xdart.gui.pages.operation_owner import OperationTerminalStatus
from xdart.gui.widgets.latest_request import LatestRequestOwner
from xrd_tools.core.angular_support import AngularSupportProfile, SupportOrigin, SupportReason

#: Completed reconstructions kept: the last frames shown, within a byte bound (the request's facts are scalars).
MAX_RETAINED_KEYS = 8
MAX_RETAINED_BYTES = 64 << 20
#: A profile's facts beside its values (window, edges, provenance), counted against the byte bound.
_PROFILE_FACT_BYTES = 4096


def _reconstruct(request, cancel, _progress):
    """The worker's job: one frame rebuilt from its output's record (bound late, so tests can hold it)."""
    from xrd_tools.analysis.angular_support import reconstruct_saved_support
    return reconstruct_saved_support(request, cancel=cancel.is_set)


def _profile_bytes(profile: AngularSupportProfile) -> int:
    return _PROFILE_FACT_BYTES + (0 if profile.supported_deg is None else int(profile.supported_deg.nbytes))


class AngularSupportOwner(QtCore.QObject):
    """Older Int frames' reconstructions for one view's strip: one running and one latest asked for, at most
    ``MAX_RETAINED_KEYS`` completed within ``MAX_RETAINED_BYTES``. ``clear`` (the strip off, the view cleared) and
    ``shutdown`` (the page closing) retire it: CLEAN, or PENDING while a computation still returns."""

    #: A reconstruction completed: the view draws it.
    reconstructed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._requests = LatestRequestOwner(lambda request, cancel, progress: _reconstruct(request, cancel, progress),
                                            self._done, same=lambda a, b: a.key == b.key, parent=self)
        self._completed: OrderedDict[tuple, AngularSupportProfile] = OrderedDict()
        self._completions = 0

    @property
    def worker(self):
        """The ``SingleWorkerOwner`` computing, for hosts that report or retire it."""
        return self._requests.worker

    @property
    def computing(self) -> bool:
        """Whether a reconstruction asked for has not completed."""
        return self._requests.computing

    def profile(self, request) -> AngularSupportProfile | None:
        """``request``'s completed profile; else None, and it is the one asked for (the latest wins)."""
        profile = self._completed.get(request.key)
        if profile is not None:
            self._completed.move_to_end(request.key)
            return profile
        self._requests.ask(request)
        return None

    def completed_requests(self) -> int:
        """How many reconstructions have completed (a frame shown again from the kept ones adds none)."""
        return self._completions

    def retained_keys(self) -> int:
        return len(self._completed)

    def retained_bytes(self) -> int:
        return sum(_profile_bytes(profile) for profile in self._completed.values())

    def clear(self) -> None:
        """Let every kept profile go, and drop a computation still running when it returns."""
        self._completed.clear()
        self._requests.release()

    def shutdown(self):
        """Close with the view's page: CLEAN, or PENDING while a computation still runs (the host asks again)."""
        self._completed.clear()
        return self._requests.shutdown()

    def _done(self, request, terminal) -> bool:
        if terminal.status is OperationTerminalStatus.RETURNED and type(terminal.payload) is AngularSupportProfile:
            profile = terminal.payload
        else:
            profile = AngularSupportProfile.unavailable(
                SupportReason.LEGACY_INSUFFICIENT, origin=SupportOrigin.RECONSTRUCTED,
                detail=str(terminal.failure_message or "the reconstruction failed")[:200])
        self._completions += 1
        self._completed[request.key] = profile
        while len(self._completed) > MAX_RETAINED_KEYS or (
                len(self._completed) > 1 and self.retained_bytes() > MAX_RETAINED_BYTES):
            self._completed.popitem(last=False)
        self.reconstructed.emit()
        return True


__all__ = ["AngularSupportOwner", "MAX_RETAINED_BYTES", "MAX_RETAINED_KEYS"]
