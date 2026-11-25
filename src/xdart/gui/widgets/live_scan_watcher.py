"""Shared observer cadence and one native handoff for RSM and Stitch Live."""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import count
from time import monotonic

from pyqtgraph.Qt import QtCore

from xdart.gui.pages.operation_owner import OperationTerminalStatus
from xdart.gui.widgets.latest_request import LatestRequestOwner
from xrd_tools.analysis.live_scans import LiveScanHandoff


@dataclass(frozen=True, slots=True)
class LiveRetryGate:
    """Check the exact external fact that blocked native admission, on the poll worker."""

    expected: object
    probe: Callable[[], object]


@dataclass(frozen=True, slots=True)
class LiveOutputSlot:
    """Native target facts, captured off the GUI thread by the output owner."""

    target: str
    keys: frozenset  # coordinator alias keys and, when present, the published filesystem object
    family: str = ""  # Stitch's session-wide casefold family, empty for RSM

    def conflicts(self, other):
        return (self.target == other.target or bool(self.keys & other.keys)
                or bool(self.family and self.family == other.family))


@dataclass(frozen=True, slots=True)
class _Refusal:
    evidence: object
    reason: str
    blocked: bool
    retry_at: float | None
    retry_gate: LiveRetryGate | None

    def permits(self, evidence, now):
        return not self.blocked and (evidence != self.evidence
                                     or (self.retry_at is not None and now >= self.retry_at))


@dataclass(eq=False, frozen=True, slots=True)
class _Poll:
    session: object
    observer: object
    now: float
    consumed: tuple
    retries: tuple
    invalidated: tuple


class LiveScanWatcher(QtCore.QObject):
    """The page dispatches native work and reports its accepted and settled boundaries."""

    def __init__(self, dispatch, status, state_changed, *, parent=None, clock=monotonic,
                 cadence=2.0):
        super().__init__(parent)
        self.clock = clock
        self._dispatch = dispatch
        self._status = status
        self._state_changed = state_changed
        self._cadence = cadence
        self._requests = LatestRequestOwner(
            self._observe, self._observed, same=lambda left, right: left is right, parent=self,
        )
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._attempts = count(1)
        self._session = self._observer = self._handoff = self._observation = None
        self._closing = self._dispatching = self._due = False
        self._next_poll = 0.0
        self._consumed, self._refusals, self._slots = {}, {}, {}
        self._invalidated = set()
        self._skipped = set()
        self._notice = ""
        self.preview = None

    @property
    def watching(self):
        return self._session is not None

    @property
    def polling(self):
        return self._requests.running is not None

    @property
    def busy(self):
        return self.watching or self.polling or self._handoff is not None

    @property
    def handoff(self):
        return self._handoff

    def _clear_accounting(self):
        self._consumed.clear()
        self._refusals.clear()
        self._slots.clear()
        self._invalidated.clear()
        self._skipped.clear()
        self._notice = ""

    def start(self, observer):
        if self._closing or self.busy:
            return False
        self._session, self._observer = object(), observer
        self._clear_accounting()
        self._due = False
        self._next_poll = self.clock()
        self._timer.start(round(self._cadence * 1000))
        self.poll_due()
        self._publish_state()
        return True

    def stop(self):
        if not self.watching:
            return
        self._session = self._observer = self._observation = None
        if self.preview is not None:
            self.preview.stop(closing=self._closing)
        self._due = False
        self._requests.release()
        if self._handoff is None:
            self._clear_accounting()
        if not self._closing and self.polling:
            # Released observations are dropped by LatestRequestOwner without its done callback.
            self._timer.start(50)
        else:
            self._timer.stop()
        self._publish_state()

    def shutdown(self):
        self._closing = True
        self.stop()
        self._timer.stop()
        return self._requests.shutdown()

    def _tick(self):
        if self.watching:
            self.poll_due()
            self.pump()
        elif not self.polling:
            self._timer.stop()
            self._publish_state()

    def poll_due(self):
        if not self.watching or self._closing:
            return
        now = self.clock()
        if now >= self._next_poll:
            self._due = True
        if not self._due or self.polling:
            return
        self._due = False
        self._next_poll = now + self._cadence
        self._requests.ask(_Poll(
            self._session, self._observer, now, tuple(self._consumed.values()),
            tuple((key, refused) for key, refused in self._refusals.items()
                  if refused.retry_gate is not None and not refused.blocked),
            tuple(self._invalidated),
        ))

    @staticmethod
    def _observe(request, cancel, _progress):
        for key in request.invalidated:
            if cancel.is_set():
                return None
            request.observer.invalidate(key)
        if request.observer.preview_observe is None:
            from xrd_tools.io.spec import spec_capture_scope as capture_scope
        else:
            from xrd_tools.sources.live_preview import preview_source_scope as capture_scope
        with capture_scope():
            observation = request.observer.poll(
                now=request.now, consumed=request.consumed, cancelled=cancel.is_set,
            )
            changed = []
            for key, refused in request.retries:
                if cancel.is_set():
                    return None
                gate = refused.retry_gate
                try:
                    current = gate.probe()
                except OSError:
                    continue  # a retry time, if supplied, is independent of a successful probe
                if current != gate.expected:
                    changed.append((key, refused))
            return observation, changed

    def _observed(self, request, terminal):
        if request.session is not self._session or self._closing:
            return False
        if terminal.status is OperationTerminalStatus.RETURNED and terminal.payload is not None:
            self._observation, changed = terminal.payload
            for key in request.invalidated:
                self._invalidated.discard(key)
            for key, refused in changed:
                if self._refusals.get(key) is refused:
                    del self._refusals[key]
        elif terminal.status is OperationTerminalStatus.FAILED:
            self._observation = None
            self._notice = terminal.failure_message or terminal.failure_type
        self._publish_state()
        self.pump()
        if self.preview is not None and self._observation is not None:
            self.preview.observe(self._observation.preview)
        self.poll_due()
        return False

    def pump(self):
        if (not self.watching or self._closing or self._handoff is not None
                or self._dispatching or self._observation is None):
            return
        self._dispatching = True
        try:
            for row in self._observation.groups:
                if not self.watching or self._closing:
                    break
                key = row.group.identity
                if row.status != "ready" or key in self._consumed or key in self._invalidated:
                    continue
                refusal = self._refusals.get(key)
                if refusal and not refusal.permits(row.evidence, self.clock()):
                    continue
                handoff = LiveScanHandoff.from_ready(
                    row, session_id=self._session, attempt_id=next(self._attempts),
                )
                self._handoff = handoff
                if not self._dispatch(handoff):
                    if self._handoff is handoff:
                        self._handoff = None
                    break  # native/page gate remains busy, leaving the ordered ready set intact
                if self._handoff is not None:
                    break
        finally:
            self._dispatching = False
        self._publish_state()

    def reserve_request(self, handoff, request, *, stitch=False):
        output = request.module.output
        return self.reserve_target(handoff, request.live_target_revision, output.artifact_family if stitch else None)

    def reserve_target(self, handoff, target, family=None):
        if target is None:
            return "The native output has no current revision"
        return self.reserve(handoff, LiveOutputSlot(target.name, target.keys, (family or "").casefold()))

    def committed_slot(self, handoff, terminal):
        slot = self._slots.get(handoff.group.identity)
        if slot is None or terminal.commit is None:
            return slot
        file = terminal.commit.terminal
        if file.device is None or not file.inode:
            return slot
        keys = frozenset(key for key in slot.keys if key[0] != "object")
        return LiveOutputSlot(slot.target, keys | {("object", file.device, file.inode)}, slot.family)

    def refuse(self, handoff, reason, *, code="", target=None, assets=None):
        from xrd_tools.analysis.live_scans import live_asset_revisions, observe_live_target

        gate = None if target is None else LiveRetryGate(target, lambda: observe_live_target(target.name))
        if gate is None and assets is not None and code != "LIVE_HANDOFF_CHANGED":
            paths, revisions = assets
            gate = LiveRetryGate(revisions, lambda: live_asset_revisions(paths))
        timed = any(word in code for word in ("BUSY", "LOCKED", "UNAVAILABLE", "LEASE"))
        if target is not None:
            collision = self.reserve_target(handoff, target)
            if collision:
                self.settle(handoff, consumed=False, blocked=True, reason=collision)
                return
        self.settle(handoff, consumed=False, reason=reason, retry_gate=gate,
                    retry_at=self.clock() + 2 * self._cadence if timed else None,
                    invalidate=code == "LIVE_HANDOFF_CHANGED")

    def reserve(self, handoff, slot):
        """Store a provisional slot, or return a stable collision reason."""
        if self._handoff is not handoff or not self.watching or self._closing:
            return "Live stopped before the output was reserved"
        key = handoff.group.identity
        for previous, claimed in self._slots.items():
            if previous != key and slot.conflicts(claimed):
                named = slot.family or slot.target
                return f"Output collision for {named} at {slot.target}, already claimed by {previous}"
        self._slots[key] = slot
        return ""

    def accept(self, handoff, *, skipped=False, reason="", slot=None):
        """The native worker crossed its acceptance barrier, or Append-current won."""
        if self._handoff is not handoff:
            return False
        key = handoff.group.identity
        self._consumed[key] = handoff.consumed()
        self._refusals.pop(key, None)
        if slot is not None:
            self._slots[key] = slot
        if skipped:
            self._skipped.add(key)
            reason = reason or "Append-current skipped a saved result"
        self._notice = reason or self._notice
        if self.preview is not None:
            self.preview.consume(handoff.group, skipped=skipped)
        return True

    def settle(self, handoff, *, consumed, reason="", retry_at=None,
               retry_gate=None, blocked=False, invalidate=False, slot=None):
        """Drain the exact native owner, including a terminal delivered after Stop."""
        if self._handoff is not handoff:
            return False
        key = handoff.group.identity
        if consumed or key in self._consumed:
            self.accept(handoff, reason=reason, slot=slot)
            if self.preview is not None:
                self.preview.settle(handoff.group)
        else:
            self._slots.pop(key, None)
            if self.watching:
                self._refusals[key] = _Refusal(
                    handoff.evidence, reason, blocked, retry_at, retry_gate,
                )
                if invalidate:
                    self._invalidated.add(key)
        self._handoff = None
        if not self.watching:
            self._clear_accounting()
        if not self._closing:
            self._publish_state()
            self.pump()
        return True

    def _publish_state(self):
        if self._closing:
            return
        if self.watching:
            groups = () if self._observation is None else self._observation.groups
            ready = sum(row.status == "ready" and row.group.identity not in self._consumed
                        for row in groups)
            waiting = sum(row.status == "waiting" for row in groups)
            skipped = sum(row.status == "skipped" for row in groups) + len(self._skipped)
            phase = "processing a scan" if self._handoff is not None else "watching"
            title = f"Live {phase} · {ready} ready · {waiting} waiting · {len(self._consumed)} consumed"
            if skipped:
                title += f" · {skipped} skipped"
            details = [title]
            if self._notice:
                details.append(self._notice)
            details.extend(refusal.reason for refusal in self._refusals.values() if refusal.reason)
            details.extend(row.reason for row in groups if row.reason)
            if self._observation is not None:
                details.extend(f"New entry for manual Run: {path} · {entry}"
                               for path, entry in self._observation.late_entries)
            self._status(title, "\n".join(dict.fromkeys(details)))
        elif self.busy:
            self._status("Stopping Live", "Waiting for the observation and native operation to settle.")
        self._state_changed()


__all__ = ["LiveOutputSlot", "LiveRetryGate", "LiveScanWatcher"]
