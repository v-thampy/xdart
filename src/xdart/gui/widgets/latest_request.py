# -*- coding: utf-8 -*-
"""One tool's computations off the GUI thread, the latest request winning: a ``SingleWorkerOwner`` (the pages'
worker owner) driven from the GUI thread by a poll timer.

AUD-9 stage 2b item 6 wrote this loop twice, in the 3D Viewer's ``VolumeView`` and RSM's Custom Slice window; it is
now this one owner, which each mounts (the maintainer cares about reuse: hard rule 14). The tool supplies only its
job and its callbacks:

- ``job(request, cancel, progress)``: the computation, on the worker (the ``SingleWorkerOwner`` runner; a tool binds
  its module-level job late, so its tests can hold a result);
- ``done(request, terminal)``: on the GUI thread, the terminal of a request still asked for, RETURNED with its
  payload or FAILED; the tool draws the result or names the refusal, and returns whether the request is now shown;
- ``same(a, b)``: two requests are the same computation (a running one is kept, a finished one still asked for);
- ``unchanged(request, shown)``, ``same`` by default: the request would show what is shown, so asking it computes
  nothing (``VolumeView`` decides that from the index it keeps, not from the request shown).

``ask(request)`` makes ``request`` (None: none) the one asked for and returns whether a computation is asked for. A
different request on the worker is cancelled (its job sees its cancel event) and dropped when it returns, its
result let go; at most one runs, and later asks coalesce to the latest; a request that would show what is shown
computes nothing. While the worker holds a computation the timer (50 ms, as the pages poll their owners) polls it:
a finished request is handed to ``done`` only if it is still the one asked for and was not cancelled (a cancelled
one asked for again is computed again); any other is dropped. ``forget()``: what the tool shows is no longer a
request's (a refusal, a cleared view). ``release()`` (the tool closing): forget, and drop whatever is computing when
it returns. ``shutdown()``: the owner's close, as the page's owners close: CLEAN, or PENDING while a job runs (the
host asks again); nothing is computed or handed over after it. Each of the three, called while ``done`` draws a
result (the window closing as it draws), wins over the "shown" that ``done`` then returns: nothing is shown, so the
same request asked for again is computed again (FIT-TOOLS D1a).

It lives here, with the shared GUI building blocks, because ``pages/operation_owner.py`` is standard-library only
by contract (``test_owner_is_stdlib_only_and_page_architecture_safe``) and this needs a Qt timer; it composes that
owner and adds no lifecycle of its own. The RSM page's reader (``RSMPage.reopen``) keeps its own loop for now: a
later adopter.
"""
from __future__ import annotations

from pyqtgraph.Qt import QtCore

from xdart.gui.pages.operation_owner import OperationTerminalStatus, SingleWorkerOwner
from xdart.gui.pages.values import PageCleanup


class LatestRequestOwner(QtCore.QObject):
    """A tool's computations on one ``SingleWorkerOwner``, the latest request winning (the module docstring)."""

    def __init__(self, job, done, *, same, unchanged=None, parent=None, interval_ms: int = 50):
        super().__init__(parent)
        self._owner = SingleWorkerOwner(job)
        self._done, self._same = done, same
        self._unchanged = same if unchanged is None else unchanged
        self._wanted = self._running = self._shown = None
        #: counts forget, release and shutdown: one inside ``done`` voids the "shown" that ``done`` returns
        self._epoch = 0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(int(interval_ms))
        self._timer.timeout.connect(self._poll)

    @property
    def worker(self) -> SingleWorkerOwner:
        """The ``SingleWorkerOwner`` this owner drives (its ``busy`` and ``close``), for hosts that report or retire
        it; requests still go through this owner."""
        return self._owner

    @property
    def computing(self) -> bool:
        """Whether a request asked for is not yet handed over (what is shown is the one before it)."""
        return self._wanted is not None

    @property
    def wanted(self):
        """The request asked for and not yet handed over, or None."""
        return self._wanted

    @property
    def running(self):
        """``(identity, request)`` of the computation on the worker (asked for, or being dropped), or None."""
        return self._running

    @property
    def shown(self):
        """The request whose result the tool shows (``done`` returned True), or None."""
        return self._shown

    def ask(self, request) -> bool:
        """``request`` is the one asked for (None: none); True when a computation is asked for."""
        if request is not None and self._shown is not None and self._unchanged(request, self._shown):
            request = None  # it would show what is shown: nothing to compute, and nothing else is wanted
        self._wanted = request
        if self._running is not None and not (request is not None and self._same(request, self._running[1])):
            self._owner.cancel(self._running[0])
        if request is None:
            return False
        self._start()
        return True

    def forget(self) -> None:
        """What the tool shows is no longer a request's result."""
        self._shown = None
        self._epoch += 1

    def release(self) -> None:
        """The tool closing: nothing is shown, and a computation still running is dropped (its buffers let go)
        when it returns."""
        self.forget()
        self.ask(None)

    def shutdown(self):
        """Close the owner with the tool's host: CLEAN, or PENDING while a job still runs (the host asks again)."""
        self._epoch += 1
        self._wanted = None
        receipt = self._owner.close()
        self._timer.stop()  # a closing owner reports nothing more; the host's next close retires it
        if receipt.status is PageCleanup.CLEAN:
            self._running = None
        return receipt

    def _start(self) -> None:
        if self._running is not None or self._wanted is None:
            return
        request = self._wanted
        identity = self._owner.begin(request)
        if identity is None:  # the owner is closed (the host closing): nothing more is computed
            return
        self._running = (identity, request)
        self._timer.start()

    def _poll(self) -> None:
        """A computation returned: handed to ``done`` if it is still the one asked for, else dropped; then the latest
        asked for."""
        if self._running is None:
            self._timer.stop()
            return
        identity, request = self._running
        update = self._owner.poll(identity)
        if update is None or update.terminal is None:
            return
        self._running = None
        terminal = update.terminal
        if (terminal.status is not OperationTerminalStatus.CANCELLED and self._wanted is not None
                and self._same(request, self._wanted)):  # a cancelled request asked for again is computed again
            self._wanted = None
            epoch = self._epoch
            shown = self._done(request, terminal)
            self._shown = request if shown and epoch == self._epoch else None  # forgotten inside done: not shown
        del update, terminal  # a dropped computation's buffers go with it
        if self._wanted is not None:
            self._start()
        elif self._running is None:
            self._timer.stop()


__all__ = ["LatestRequestOwner"]
