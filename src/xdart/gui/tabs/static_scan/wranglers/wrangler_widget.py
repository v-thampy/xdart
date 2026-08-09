# -*- coding: utf-8 -*-
"""
@author: walroth
"""

# Standard library imports
import logging
import os
from collections import deque
from queue import Queue
import threading
import traceback
from typing import NamedTuple

# Other imports
import numpy as np
from pathlib import Path

# Qt imports
from pyqtgraph import Qt
from pyqtgraph.parametertree import Parameter

# This module imports
from xrd_tools.session.run_configuration import (
    FrozenRunConfiguration,
    FrozenSourceSpec,
    RunConfigurationRefused,
    admit_run_configuration,
    require_run_configuration,
)
from ..run_config_debug import run_config_debug_log

logger = logging.getLogger(__name__)


def _frozen_source_is_admissible(source):
    """Whether a TYPED frozen source is also a REAL one (review §41.3.D).

    Presence and type are not sufficient: the plain-image Directory freeze builds
    a ``DirectorySourceSpec`` from whatever the card holds, so a nonexistent
    directory could reach Start as a well-typed but invalid source.  Validate the
    family's own invariants -- a directory needs a real root and a format token; a
    file-shaped source needs a uri.  An EMPTY but existing directory stays valid:
    the worker discovers candidates after Run.
    """
    if source.family == "directory":
        root = str(source.uri or "").strip()
        return (
            bool(root)
            and bool(source.suffixes)
            and Path(root).expanduser().is_dir()
        )
    return bool(str(source.uri or "").strip())


#: Operator-facing refusal text per POSITIVELY OBSERVED active run owner
#: (§9.10 step 2).  Any other label — notably the T-3.1 ``…-probe-error``
#: variants, where an owner's activity could NOT be determined — falls through to
#: :func:`_run_owner_refusal_status`'s honest "could not confirm" wording.  A
#: module-level function on purpose: the Start sentinels drive duck-typed
#: SimpleNamespace/MethodType holders that bind unbound methods individually, so a
#: shared helper must not be a new method they would each have to bind.
_RUN_OWNER_REFUSAL_STATUS = {
    "run": 'Previous run is still stopping — try again in a moment.',
    "wrangler": 'Previous run is still stopping — try again in a moment.',
    "reintegration": 'A reintegration is still finishing — try again in a moment.',
    "stitch": 'A stitch is still finishing — try again in a moment.',
}


def _run_owner_refusal_status(owner):
    """Refusal text for one run-owner admission decision.

    A KNOWN-active owner gets its specific wording.  An UNOBSERVABLE owner gets
    text that says so rather than claiming a run is stopping: the operator needs
    to know the Run was refused because the GUI could not confirm the previous
    run finished, which is a different (and log-worthy) situation."""
    text = _RUN_OWNER_REFUSAL_STATUS.get(str(owner))
    if text is not None:
        return text
    return (
        'Could not confirm that the previous run finished (' + str(owner)
        + ') — Run refused rather than risk starting over a live run. '
        'See the log, then try again.'
    )


# MEM-1a: the wrangler→GUI live display hand-off (``_published_frames``) stashes
# one fully-hydrated LiveFrame per frame for the GUI's ``update_data`` to pop.
# In live mode each retained frame still holds its ~18 MB raw (upcast ~64 MB
# float64) until it leaves the write-side staging window, so if the GUI thread
# falls behind the producer an *unbounded* dict grows to tens of GB → OOM.
# Bounding it with DROP-OLDEST is safe post-8a: the frame is already durable via
# the sink write, and the display re-hydrates any evicted label on demand (the
# store-first path).  The dropped entry is always the OLDEST undrained frame —
# never the freshly-signalled idx the GUI is about to consume next.
_PUBLISHED_FRAMES_CAP = 128


class _BoundedFrameHandoff(dict):
    """Insertion-ordered dict capped at ``cap`` entries (drop-oldest on insert).

    A plain ``dict`` with a size guard: every ``__setitem__`` that pushes past
    ``cap`` evicts the oldest key(s).  ``pop``/``get``/``clear`` are inherited
    unchanged, so the existing consumer (``update_data``: ``pop(idx, None)``,
    ``get(idx)``) keeps working — a dropped idx simply reads back as ``None``,
    which the consumer already tolerates.
    """

    def __init__(self, *args, cap: int = _PUBLISHED_FRAMES_CAP, **kwargs):
        super().__init__(*args, **kwargs)
        self._cap = max(1, int(cap))
        self._drop_warned = False

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        over = len(self) - self._cap
        if over > 0:
            for stale in list(self.keys())[:over]:
                super().pop(stale, None)
            if not self._drop_warned:
                self._drop_warned = True
                logger.warning(
                    "GUI behind producer: dropping oldest display hand-offs "
                    "(cap=%d); frames remain on disk and re-hydrate on demand",
                    self._cap)

    def clear(self):
        super().clear()
        self._drop_warned = False


# Sentinel used by ``_apply_threshold_inline`` to mark out-of-band
# pixels.  pyFAI's CSR integrator auto-skips NaN at integrate time
# without invalidating the mask CRC, so the per-frame threshold
# filter survives without forcing a per-frame LUT rebuild.
_THRESHOLD_NAN = np.float32(np.nan)

class _CommandCancelToken:
    """Duck-typed ssrl CancelToken bound to a wrangler thread command."""

    def __init__(self, owner):
        self._owner = owner

    @property
    def cancelled(self):
        return getattr(self._owner, 'command', None) == 'stop'

    def cancel(self):
        self._owner.command = 'stop'


class GIMotorHydration(NamedTuple):
    """Immutable, source-qualified GI theta-motor observation carried by
    :attr:`wranglerWidget.sigGIMotorOptions` (§13.6 / §13.11 hydration 1-6).

    Replaces the bare motor ``list`` payload so a late result cannot silently
    overwrite the current source's motor knowledge:

    * ``state`` is one of ``UNKNOWN`` / ``KNOWN_EMPTY`` / ``KNOWN_NONEMPTY``.
      ``UNKNOWN`` means the source was NOT inspected (e.g. a lazy recursive
      directory whose direct children hold no preview file); ``KNOWN_EMPTY``
      means a targeted inspection PROVED there are no eligible motors (§13.7).
    * ``source_fingerprint`` + ``generation`` are the source identity and the
      request/hydration epoch captured when the metadata request STARTED, so
      the static-widget owner can reject a delayed result from a superseded
      source (fingerprint mismatch) or an older same-root request (lower
      generation) BEFORE it updates stored knowledge or the visible dropdown.
    """

    state: str
    motors: tuple
    source_fingerprint: object
    generation: int

    # Knowledge states (string-equal to GIMotorObservation's, so the static
    # widget owner can compare payload.state directly).
    UNKNOWN = "UNKNOWN"
    KNOWN_EMPTY = "KNOWN_EMPTY"
    KNOWN_NONEMPTY = "KNOWN_NONEMPTY"


class GIHydrationRequestToken(NamedTuple):
    """Immutable request-local GI-hydration token (§19.4 req 1).

    Captured when a metadata request STARTS and carried back by the completing
    emit, so ownership is decided by the request identity — NOT by completion
    order.  Field order ``(generation, source_fingerprint)`` matches the
    ``(hydration.generation, hydration.source_fingerprint)`` tuple the owner
    compares against, and a :class:`GIHydrationRequestToken` compares equal to the
    plain 2-tuple with the same values."""

    generation: int
    source_fingerprint: object


class GIHydrationOutcome(NamedTuple):
    """Typed result of a hydration completion / announcement (§21.4 req 2).

    ``accepted`` is False for every inert outcome — a token that was already
    consumed, cancelled, superseded or invalidated on close, or a host without
    the Qt signal.  Nothing is emitted in that case, so a replayed completion
    cannot overwrite a result the owner already applied."""

    accepted: bool
    reason: str
    hydration: object


def _gi_outstanding_registry(obj) -> dict:
    """THE outstanding-request authority for *obj* (§21.4 req 3).

    An identity-keyed registry: a completion is authorized iff its exact token is
    still a key here, so correctness never depends on a capacity bound.  A MODULE
    FUNCTION (not a method) so duck-typed test hosts that bind only a subset of
    wrangler methods onto a ``SimpleNamespace`` never trip on a missing bound
    helper (the SimpleNamespace-double trap)."""
    registry = getattr(obj, "_gi_hydration_outstanding", None)
    if not isinstance(registry, dict):
        registry = obj._gi_hydration_outstanding = {}
    return registry


def _read_gi_source_fingerprint(obj):
    """The object's CURRENT source fingerprint, read defensively.

    A module function (not a method) so duck-typed test hosts that bind only a
    subset of wrangler methods onto a ``SimpleNamespace`` never trip on a missing
    bound helper (the SimpleNamespace-double trap)."""
    fp_getter = getattr(obj, "_gi_source_fingerprint", None)
    if callable(fp_getter):
        try:
            return fp_getter()
        except Exception:
            return None
    return None


class wranglerWidget(Qt.QtWidgets.QWidget):
    """Base class for wranglers. Extending this ensures all methods,
    signals, and attributes expected by ttheta_widget are present.
    Threads should be started by use of sigStart.emit, which ensures
    tthetaWidget handles initiation.
    
    attributes:
        command_queue: Queue, used to send commands to thread
        file_lock, mp.Condition, process safe lock for file access
        fname: str, path to data file
        parameters: pyqtgraph Parameter, stores parameters from user
        scan_name: str, current scan name, used to handle syncing data
        scan_args: dict, used as **kwargs in scan initialization.
            see LiveScan.
        thread: wranglerThread or subclass, QThread for controlling
            processes
    
    methods:
        enabled: Enables or disables interactivity
        set_fname: Method to safely change file name
        setup: Syncs thread parameters prior to starting
    
    signals:
        finished: Should be connected to thread.finished signal
        sigStart: Tells tthetaWidget to start the thread and prepare
            for new data.
        sigUpdateData: int, signals a new frame has been added.
        sigUpdateFile: (str, str, bool, str, bool, bool), sends new scan_name, file name
            GI flag (grazing incidence), theta motor for GI, single_image and
            series_average flag to static_scan_Widget.
    """
    sigStart = Qt.QtCore.Signal()
    sigUpdateData = Qt.QtCore.Signal(int)
    # sigUpdateFrame = Qt.QtCore.Signal(dict)
    sigUpdateFile = Qt.QtCore.Signal(str, str, bool, str, bool, bool)
    sigUpdateGI = Qt.QtCore.Signal(bool)
    # Reserved for the future XYE viewer handoff; C3 dynamic runs refuse XYE.
    sigXyeOutputReady = Qt.QtCore.Signal(str)
    # GI move (Stage B): hands the available SPEC incidence-motor columns to the
    # integrator panel's GI motor dropdown (the integrator owns the selection).
    # §13.6: the payload is now an immutable, source-qualified GIMotorHydration
    # (fingerprint + request epoch + knowledge state + motors), not a bare list,
    # so a late result under a newer source/request is rejected by the owner.
    sigGIMotorOptions = Qt.QtCore.Signal(object)
    finished = Qt.QtCore.Signal()
    started = Qt.QtCore.Signal()
    # Pause/Resume (Phase B): sigPaused fires once the run is frozen at a frame
    # boundary (the host then LIFTS the freeze guard for browsing); sigResuming
    # fires just before resuming (the host RE-ENGAGES the guard FIRST).  Emitted
    # only by wranglers that support pause (image wrangler); harmless elsewhere.
    sigPaused = Qt.QtCore.Signal()
    sigResuming = Qt.QtCore.Signal()

    # ------------------------------------------------------------------
    # GI motor hydration identity (§13.6).  The fingerprint + generation are
    # stamped onto every GIMotorHydration at emit; the static-widget owner
    # verifies them before it trusts the result.
    # ------------------------------------------------------------------
    def _gi_source_fingerprint(self):
        """Hashable identity of the wrangler's CURRENT source selection.

        Image-style default (directory / series / single-image inputs);
        :class:`nexusWrangler` overrides it with its own file/entry identity.
        Read defensively so duck-typed test hosts (SimpleNamespace) never
        raise here."""
        return (
            "image",
            str(getattr(self, "inp_type", "") or ""),
            str(getattr(self, "img_dir", "") or ""),
            str(getattr(self, "img_file", "") or ""),
            bool(getattr(self, "include_subdir", False)),
            str(getattr(self, "file_filter", "") or ""),
            str(getattr(self, "img_ext", "") or ""),
            str(getattr(self, "meta_ext", "") or ""),
        )

    def _gi_hydration_registry(self) -> dict:
        """THE outstanding-request authority: an identity-keyed registry (§21.4 req 3).

        Keyed by the immutable token itself, so correctness never depends on a
        capacity bound.  ``_gi_hydration_pending`` is a BOUNDED DIAGNOSTIC HISTORY
        beside it and authorizes nothing.  Delegates to the module-level accessor
        so no internal caller depends on this method being bound."""
        return _gi_outstanding_registry(self)

    def _next_gi_hydration_generation(self) -> int:
        """Open a GI-motor hydration request; bump + return the hydration epoch.

        §19.4 / §21.4: an immutable :class:`GIHydrationRequestToken` (this
        request's generation + the source fingerprint captured AT REQUEST START)
        is registered as OUTSTANDING in the identity-keyed registry.  The
        completing :meth:`_emit_gi_hydration` must carry that exact token; it is
        a SINGLE-COMPLETION CAPABILITY, retired atomically before the emit, so a
        replayed/duplicate completion cannot overwrite the first result.  A
        direct re-announcement of the current source is a different operation
        with a different method (:meth:`_announce_gi_hydration`) and never
        consumes a request."""
        gen = int(getattr(self, "_gi_hydration_generation", 0) or 0) + 1
        self._gi_hydration_generation = gen
        token = GIHydrationRequestToken(
            gen, _read_gi_source_fingerprint(self))
        _gi_outstanding_registry(self)[token] = True
        pending = getattr(self, "_gi_hydration_pending", None)
        if pending is None:
            pending = self._gi_hydration_pending = deque(maxlen=16)
        pending.append(token)          # bounded DIAGNOSTIC history only
        # The request a synchronous GUI path will complete (§21.4 req 5): the
        # discovery that opened it calls back into a different method, so the
        # token rides here instead of being re-derived from completion order.
        self._gi_hydration_open_token = token
        # Kept only for external readers / diagnostics.
        self._gi_hydration_request_token = token
        return gen

    def _begin_gi_hydration_request(self) -> "GIHydrationRequestToken":
        """Open a hydration request and RETURN its immutable request-local token.

        §19.4 req 2: an asynchronous discovery owner RETAINS this exact token in
        its completion closure and passes it back as
        ``_emit_gi_hydration(..., token=)``."""
        self._next_gi_hydration_generation()
        return self._gi_hydration_request_token

    def _gi_hydration_token_outstanding(self, token) -> bool:
        """Whether *token* is still an unconsumed outstanding request (§21.4)."""
        return token in _gi_outstanding_registry(self)

    def _retire_gi_hydration_token(self, token) -> bool:
        """Atomically PROVE *token* is outstanding and RETIRE it (§21.4 req 1).

        Returns ``True`` exactly once per token.  A token that was already
        consumed, cancelled, superseded, or invalidated on close returns
        ``False`` and must not produce an owner-applicable hydration (req 2)."""
        registry = _gi_outstanding_registry(self)
        if registry.pop(token, None) is None:
            return False
        if getattr(self, "_gi_hydration_open_token", None) == token:
            self._gi_hydration_open_token = None
        return True

    def _cancel_gi_hydration_request(self, token=None) -> bool:
        """Cancel one outstanding request so it can NEVER complete (§19.4 req 6).

        A request whose discovery failed (or was abandoned) is retired from the
        authority registry, and when it is the request that defines the current
        epoch the epoch advances so its identity is permanently superseded.
        ``token=None`` cancels the most-recently-opened outstanding request.
        Returns whether an outstanding token was retired."""
        registry = _gi_outstanding_registry(self)
        if token is None:
            if not registry:
                return False
            token = next(reversed(registry))
        removed = self._retire_gi_hydration_token(token)
        generation = int(getattr(token, "generation", 0) or 0)
        if generation and generation == int(
                getattr(self, "_gi_hydration_generation", 0) or 0):
            self._gi_hydration_generation = generation + 1
        if getattr(self, "_gi_hydration_request_token", None) == token:
            self._gi_hydration_request_token = None
        return removed

    def _invalidate_gi_hydration_requests(self) -> int:
        """Cancel EVERY outstanding request and advance the epoch (§19.4 req 6).

        The teardown/close primitive: after this, no previously issued token is
        outstanding, so no late completion can produce an owner-applicable
        result whatever order the callbacks arrive in.  IDEMPOTENT (§21.3 req 3)
        — calling it again on an already-invalidated wrangler retires nothing,
        opens nothing, and raises nothing.  Returns the new epoch."""
        _gi_outstanding_registry(self).clear()
        pending = getattr(self, "_gi_hydration_pending", None)
        if pending:
            pending.clear()
        self._gi_hydration_request_token = None
        self._gi_hydration_open_token = None
        # O-1b R4A-1(iii): this is the ONLY teardown primitive (the child
        # closeEvent plus the host's `_invalidate_all_gi_hydration`), so it is
        # also where "this wrangler is closed" becomes knowable to a value
        # delivery that carries its own identity and therefore never had a
        # token to invalidate.  A worker discovery arriving after teardown must
        # be inert, not merely late.
        self._gi_hydration_closed = True
        gen = int(getattr(self, "_gi_hydration_generation", 0) or 0) + 1
        self._gi_hydration_generation = gen
        return gen

    def closeEvent(self, event):
        """Secondary direct-close guard (§21.3 req 2).

        The PRIMARY teardown owner is ``staticWidget._invalidate_all_gi_hydration``,
        invoked at the start of the host's ``close()`` — Qt does NOT deliver a
        child's ``closeEvent`` when the parent closes, so this alone never covered
        the real tab/application lifecycle."""
        try:
            self._invalidate_gi_hydration_requests()
        except Exception:
            logger.debug(
                "[GI] hydration invalidation on close failed", exc_info=True)
        super().closeEvent(event)

    def _adopt_retained_custody(self, generation, path):
        self._viewer_preserve_token = (int(generation), os.path.normcase(os.path.abspath(path)), None)

    def dynamic_session_running(self, **facts):
        probe = getattr(getattr(self, "thread", None), "dynamic_session_running", None)
        return bool(callable(probe) and probe(**facts))

    def dynamic_cleanup_pending(self):
        probe = getattr(getattr(self, "thread", None), "dynamic_cleanup_pending", None)
        return bool(callable(probe) and probe())

    def _viewer_transition(self, transition, **facts):
        token = getattr(self, "_viewer_preserve_token", None)
        operation = facts.get("operation")
        if token is not None:
            generation, path, accepted = token
            incoming = facts.get("path")
            exact = (facts.get("generation") == generation and incoming is not None
                     and os.path.normcase(os.path.abspath(incoming)) == path)
            if (transition == "set_file" and exact and facts.get("internal")
                    and accepted is None and facts.get("accepted")):
                self._viewer_preserve_token = (generation, path, operation)
                return "preserve"
            if transition == "run_end_browse" and exact and accepted is None:
                return "preserve"
            if transition == "data_reset" and exact and accepted is not None:
                return "preserve"
            if (transition == "set_file_failed" and accepted is not None
                    and operation is accepted):
                self._viewer_preserve_token = (generation, path, None)
                return "preserve"
            if (transition == "thread_finished" and accepted is not None
                    and operation is accepted):
                self._viewer_preserve_token = None
                return "preserve"
        release = getattr(getattr(self, "thread", None), "release_retained_custody", None)
        if release is None or release():
            self._viewer_preserve_token = None
            return "released"
        return "cleanup-pending"

    @staticmethod
    def _gi_hydration_state(motors, proved: bool):
        """``(state, real_motors)`` for a hydration payload.

        ``proved`` records whether a TARGETED inspection ran: an empty motor list
        from a proved inspection is ``KNOWN_EMPTY``, but an empty list with
        nothing inspected is ``UNKNOWN`` (§13.7 — never classify a lazy recursive
        directory as known-empty).  A non-empty list is always
        ``KNOWN_NONEMPTY``."""
        real = tuple(
            str(m) for m in (motors or ())
            if str(m) and not any(x in str(m).lower() for x in ("roi", "pd"))
        )
        if real:
            return GIMotorHydration.KNOWN_NONEMPTY, real
        if proved:
            return GIMotorHydration.KNOWN_EMPTY, real
        return GIMotorHydration.UNKNOWN, real

    def _emit_gi_hydration(self, motors, *, proved: bool,
                           token) -> "GIHydrationOutcome":
        """COMPLETE the request identified by *token* (§21.4).

        ``token`` is REQUIRED and is a SINGLE-COMPLETION CAPABILITY: the request
        is atomically proved outstanding and retired BEFORE anything is emitted,
        so a duplicate/replayed completion — even one carrying the still-current
        token — produces NO owner-applicable emission and cannot overwrite the
        first result (§21.4 req 1-2).  A token that was consumed, cancelled,
        superseded, evicted, or invalidated on close yields the inert
        :data:`GIHydrationOutcome` ``NOT_OUTSTANDING``.

        A direct re-announcement of the CURRENT source is a different operation:
        use :meth:`_announce_gi_hydration`, which never consumes or infers a
        request (§21.4 req 4).  Everything is read through ``getattr``/module
        functions so duck-typed hosts without the Qt signal are a harmless no-op.
        """
        if token is None:
            return GIHydrationOutcome(False, "no token", None)
        # PROVE-AND-RETIRE FIRST: an unauthorized completion must not emit, and
        # an authorized one must not be replayable.
        # Inlined prove-and-retire — never a ``self.<helper>`` call — so a host
        # that binds only this method still enforces single completion.
        if _gi_outstanding_registry(self).pop(token, None) is None:
            return GIHydrationOutcome(False, "token not outstanding", None)
        if getattr(self, "_gi_hydration_open_token", None) == token:
            self._gi_hydration_open_token = None
        sig = getattr(self, "sigGIMotorOptions", None)
        if sig is None:
            return GIHydrationOutcome(False, "no signal", None)
        state, real = wranglerWidget._gi_hydration_state(motors, proved)
        hydration = GIMotorHydration(
            state, real,
            getattr(token, "source_fingerprint", None),
            int(getattr(token, "generation", 0) or 0))
        sig.emit(hydration)
        return GIHydrationOutcome(True, "completed", hydration)

    def _announce_gi_hydration(self, motors, *,
                               proved: bool) -> "GIHydrationOutcome":
        """Re-announce the CURRENT source's motor knowledge (§21.4 req 4).

        The structurally separate no-request path: a session restore or a direct
        establishment of the motor list that is NOT completing an outstanding
        asynchronous request.  It stamps the LIVE source fingerprint and the
        CURRENT epoch, and it never consumes, retires, or infers a request token
        — so it can never be used to counterfeit a completion, and an
        outstanding request stays outstanding for its own owner."""
        sig = getattr(self, "sigGIMotorOptions", None)
        if sig is None:
            return GIHydrationOutcome(False, "no signal", None)
        state, real = wranglerWidget._gi_hydration_state(motors, proved)
        hydration = GIMotorHydration(
            state, real,
            _read_gi_source_fingerprint(self),
            int(getattr(self, "_gi_hydration_generation", 0) or 0))
        sig.emit(hydration)
        return GIHydrationOutcome(True, "announced", hydration)

    def gi_hydration_is_current(self, hydration) -> bool:
        """Whether *hydration* still describes the CURRENT source + epoch.

        The static-widget owner calls this BEFORE updating stored motor
        knowledge or the visible theta options, so a delayed result from a
        superseded source (different fingerprint) or an older same-root request
        (lower generation) is ignored (§13.6 / §13.11 hydration 3-4).  A bare
        legacy list payload has no identity and is treated as current."""
        if not isinstance(hydration, GIMotorHydration):
            return True
        if hydration.source_fingerprint != self._gi_source_fingerprint():
            return False
        return int(hydration.generation) == int(
            getattr(self, "_gi_hydration_generation", 0) or 0)

    def __init__(self, fname, file_lock, parent=None):
        """fname: str, file path
        file_lock: mp.Condition, process safe lock
        """
        super().__init__(parent)
        self.file_lock = file_lock
        self.fname = fname
        self.scan_name = 'null_thread'
        # §13.11 hydration 5 / §13.7 / §15.12-C.4: GI-motor knowledge state.  Proof
        # initializes FALSE — a fresh wrangler that has NOT run a targeted metadata
        # inspection is UNKNOWN, never known-empty, so a bare empty emit preserves an
        # explicit / session-restored motor instead of resolving it to Manual.  The
        # discovery paths set this True/False EXPLICITLY once they have inspected the
        # source (a proven empty inspection — e.g. Eiger — is KNOWN_EMPTY), and it is
        # a persistent instance bit so a direct re-announce carries the prior proof.
        self._gi_motor_knowledge_proved = False
        # Hydration epoch + outstanding request tokens (§19.4 / §13.11 hydration 2 /
        # §15.12-C.3).  ``_gi_hydration_pending`` holds the immutable
        # :class:`GIHydrationRequestToken` captured when each metadata request
        # STARTED.  A completion carries its OWN token — passed explicitly by an
        # async owner (removed by identity), or taken as the CURRENT outstanding
        # request by a synchronous emit — so ownership is the request identity, NOT
        # completion order (the old OLDEST-first pop mis-stamped A with B's identity
        # under A-starts → B-starts → B-completes; §15.5 / §19.4).  Bounded so a
        # request-without-emit leak sheds its stalest token rather than growing.
        self._gi_hydration_generation = 0
        # §21.4 req 3: the identity-keyed registry is THE authority — a
        # completion is authorized iff its exact token is still a key here, and
        # retiring it is what makes an explicit token single-use.  Correctness
        # does not depend on any capacity bound.
        self._gi_hydration_outstanding = {}
        # Bounded DIAGNOSTIC history of opened requests.  It authorizes nothing:
        # eviction from here never invalidates a token, and membership here never
        # licenses a completion.
        self._gi_hydration_pending = deque(maxlen=16)
        self._gi_hydration_request_token = None
        self._gi_hydration_open_token = None
        # Set by `_invalidate_gi_hydration_requests` at teardown; read by the
        # identity-carrying value deliveries that have no token to invalidate.
        self._gi_hydration_closed = False
        self.parameters = Parameter.create(
            name='wrangler_widget', type='int', value=0
        )
        self.scan_args = {}
        # O-1a-W1: the accepted frozen run configuration for the CURRENT run, and
        # the accepted-generation watermark that makes a superseded object a
        # typed refusal.  Every wrangler admits through the same owner.
        self.run_configuration = None
        self.run_configuration_floor = 0
        # O-1a-W1R: the admission LEDGER -- the exact object this owner bound at
        # admission.  Consumption compares against it by ``is``; a generation
        # floor is not execution authorization (review §39.2 W1R-P1-1).
        self._admitted_run_configuration = None

        self.command_queue = Queue()
        self.thread = wranglerThread(self.command_queue, self.fname, self.file_lock, self)
        self.thread.finished.connect(self.finished.emit)
        self.thread.started.connect(self.started.emit)
        self.thread.sigUpdate.connect(self.sigUpdateData.emit)
        # self.thread.sigUpdateFrame.connect(self.sigUpdateFrame.emit)
        self.thread.sigUpdateGI.connect(self.sigUpdateGI.emit)
        self.thread.sigXyeOutputReady.connect(self.sigXyeOutputReady.emit)

        # Shared run-controls (CONTROLS section): the staticWidget owns one
        # StaticControls widget and ATTACHES it to the active wrangler, which
        # aliases its own control refs onto the shared widgets so all existing
        # run-lifecycle logic drives them.  None until attached.
        self._controls = None
        self._control_conns = []

    def enabled(self, enable):
        """Use this function to control what is enabled and disabled
        during integration.
        """
        pass

    def _active_run_owner(self):
        """The host's ONE active/stopping predicate (§9.10 step 2 / §31.3 item 3).

        Returns an owner label (``run`` / ``wrangler`` / ``reintegration`` /
        ``stitch``, or a ``…-probe-error`` variant) or ``None`` when every present
        owner was positively observed idle.

        T-3.1 (§32.3 item 3) — this wrapper must not erase a failure either:

        * the host predicate EXISTS but raises -> refuse.  Previously this fell
          through to the wrangler-only probe, so a broken admission owner
          silently narrowed the decision to one of its four owners;
        * the wrangler-only fallback runs ONLY when the host does not expose the
          shared predicate at all;
        * a fallback probe that RAISES also refuses.

        A fallback holder whose thread exposes no ``isRunning`` at all stays idle:
        that is absence, not failure (the §32.3 item-2 partial-construction
        carve-out), and it is the shape of the headless/duck-typed holders the
        Start sentinels drive."""
        host = getattr(self, "_h19_host", None)
        predicate = getattr(host, "_controls_v2_active_run_owner", None)
        if callable(predicate):
            try:
                return predicate()
            except Exception:
                logger.exception(
                    "run-owner admission predicate failed; refusing Start "
                    "rather than assuming every owner is idle")
                return "admission-check-probe-error"
        probe = getattr(getattr(self, "thread", None), "isRunning", None)
        if not callable(probe):
            return None
        try:
            return "wrangler" if bool(probe()) else None
        except Exception:
            logger.exception(
                "wrangler isRunning probe failed; refusing Start rather than "
                "assuming the worker is idle")
            return "wrangler-probe-error"

    # ── Run-configuration admission (O-1a-W1, shared by every wrangler) ──
    #
    # Static/duck-safe by design: the Start sentinels drive SimpleNamespace
    # holders that bind unbound methods individually, so these are called as
    # ``wranglerWidget._admit_run_configuration(self, ...)``.

    @staticmethod
    def _safe_status_text(obj, text):
        setter = getattr(obj, '_set_status_text', None)
        if callable(setter):
            setter(text)
            return
        label = getattr(getattr(obj, 'ui', None), 'specLabel', None)
        set_text = getattr(label, 'setText', None)
        if callable(set_text):
            set_text(text)

    @staticmethod
    def _admit_run_configuration(obj, stage):
        """Bind THE run configuration for this click exactly once, or refuse.

        One owner for both wranglers.  The Controls host produces exactly one
        frozen object; this BINDS that same object (identity, never a copy) onto
        the wrangler and its worker, records it as the consumption expectation,
        and advances the accepted-generation watermark.

        O-1a-W1R (review §39.2 W1R-P1-1, §39.5 Phase 1 items 1 and 3): first
        admission and later consumption have different rules and different
        owners.  This is the ADMISSION half, so it goes through
        :func:`admit_run_configuration`:

        * while an object is bound at the current generation, only THAT exact
          object may be re-admitted (an idempotent re-publication).  A different
          genuine ``FrozenRunConfiguration`` at the same generation is a REBIND
          and refuses ``foreign`` -- which is what the parent accepted, because
          its shared gate only knew the generation floor;
        * a strictly newer generation is a genuinely new accepted click;
        * the refusal is raised BEFORE any carrier is written, so a caller that
          has not yet mutated run state returns untouched (zero delta).
        """

        host = getattr(obj, "_h19_host", None)
        prepare = getattr(host, "_prepare_controls_v2_run_configuration", None)
        bound = getattr(obj, "run_configuration", None)
        # O-1a-W1R-D1 (review §40.1 P1-D, §40.3 D1 items 5-6): STAGE, then
        # validate identity and required frozen values, and only then PUBLISH --
        # once.  Every refusal below happens before the first carrier write, and
        # any tentative handoff the staging owner parked is consumed on the way
        # out, so a refused Start leaves wrapper, source, thread, generation and
        # pending state exactly as it found them.
        try:
            offered = prepare() if callable(prepare) else None
            frozen = admit_run_configuration(
                offered,
                stage=stage,
                floor=int(getattr(obj, "run_configuration_floor", 0) or 0),
                bound=(
                    bound if isinstance(bound, FrozenRunConfiguration) else None
                ),
            )
            wranglerWidget._require_admissible_frozen_values(
                obj, frozen, stage=stage)
            # O-1a-W1R-D1 (review §41.3.E): compute EVERY fallible projection
            # before the first carrier write.  The publication used to interleave
            # the wrapper binding, this thawed projection and the thread binding,
            # so a failure part-way through left a partially accepted run.  With
            # the only throwing step hoisted above the writes, the writes below are
            # bounded attribute stores on objects that already exist.
            source_projection = frozen.thaw_source_spec()
        except Exception:
            wranglerWidget._discard_staged_run_configuration(obj)
            raise
        thread = getattr(obj, "thread", None)
        wranglerWidget._bind_admitted_run_configuration(obj, frozen)
        # The wrapper's ``source_spec`` mirror moves here from the staging owner
        # (§40.3 D1 items 5-6).  The WORKER deliberately gets no thawed mirror:
        # a thawed ``SourceSpec`` carries a ``mappingproxy`` in ``options``, which
        # is unpicklable, and the worker already holds the exact frozen object.
        obj.source_spec = source_projection
        if thread is not None:
            wranglerWidget._bind_admitted_run_configuration(thread, frozen)
        return frozen

    @staticmethod
    def _require_admissible_frozen_values(obj, frozen, *, stage):
        """Refuse an accepted-shaped object that cannot answer for a run.

        O-1a-W1R-D1 (review §40.3 D1 item 4).  A value the worker would otherwise
        have to infer must be present in the frozen object, so admission refuses
        rather than letting execution fall back to a mutable mirror.
        ``imageWrangler`` requires ``save_path`` -- an accepted run with no output
        target -- and ``source``: §40.1 P1-A showed two real GUI paths admitting
        ``source is None`` and handing 58 worker reads back to writable state.

        ``source`` is a typed value object, not a string, so it is validated for
        presence and type; the remaining required names are text fields.
        """
        for required in getattr(obj, "_admission_required_frozen_values", ()):
            value = getattr(frozen, required, None)
            if required == "source":
                ok = (isinstance(value, FrozenSourceSpec)
                      and _frozen_source_is_admissible(value))
            else:
                ok = bool(str(value or ""))
            if not ok:
                raise RunConfigurationRefused(
                    "absent",
                    stage=stage,
                    detail=(
                        f"the frozen run configuration carries no {required}; "
                        "the run would have to infer it from mutable state"),
                    generation=int(frozen.generation),
                )
        # O-3N.R (§15.5 R1 item 1): a wrangler whose source has values the
        # generic presence check cannot judge validates its own SHAPE here --
        # still before the first carrier write, so the refusal is zero-delta.
        # Filesystem/HDF5 facts that need real I/O belong to the worker
        # preflight; this stays cheap.
        hook = getattr(obj, "_validate_admissible_source", None)
        if callable(hook):
            hook(frozen, stage=stage)

    @staticmethod
    def _discard_staged_run_configuration(obj):
        """Consume a tentative handoff a refused Start must not leave behind."""
        host = getattr(obj, "_h19_host", None)
        if host is not None and getattr(
                host, "_pending_controls_v2_run_configuration", None) is not None:
            host._pending_controls_v2_run_configuration = None

    @staticmethod
    def _bind_admitted_run_configuration(target, frozen):
        """Record ONE accepted object as both carrier and consumption expectation.

        ``run_configuration`` is the published carrier every consumer reads;
        ``_admitted_run_configuration`` is the admission LEDGER the worker-entry
        identity gate compares against by ``is``.  The ledger carries only that
        exact object -- no scalar mirrors, no second revision/fingerprint
        authority, no mutable owner (§39.5 Phase 1, final paragraph).
        """

        target.run_configuration = frozen
        target.run_configuration_floor = int(frozen.generation)
        target._admitted_run_configuration = frozen
        # O-1a-W1R-D2 (review §43.4): there is no stored "active" reference to
        # invalidate any more.  A run consumes the object its entry gate
        # qualified, passed down as an argument, so no previous run's policy can
        # remain observable once a new object is admitted -- and no pre-entry
        # read can resolve through the previous run at all.
        return frozen

    @staticmethod
    def _publish_run_configuration_to_thread(obj, thread=None):
        """Re-publish the accepted object onto a worker built AFTER admission.

        ``nexusWrangler.setup()`` replaces its thread on every run, so the newly
        constructed worker must receive the exact accepted identity -- carrier
        AND admission ledger -- rather than starting with none.
        """

        thread = thread if thread is not None else getattr(obj, "thread", None)
        frozen = getattr(obj, "run_configuration", None)
        if thread is None or not isinstance(frozen, FrozenRunConfiguration):
            return None
        expected = getattr(obj, "_admitted_run_configuration", None)
        if expected is not None and frozen is not expected:
            # The wrapper's carrier no longer matches what it admitted; refuse
            # rather than propagate the substitution to a fresh worker.
            raise RunConfigurationRefused(
                "foreign",
                stage="publish-to-worker",
                detail=(
                    "the wrapper carrier is not the object it admitted; a "
                    "substituted configuration may not reach a new worker"),
                generation=int(frozen.generation),
                floor=int(expected.generation),
            )
        return wranglerWidget._bind_admitted_run_configuration(thread, frozen)

    @staticmethod
    def _report_run_configuration_refusal(obj, refusal, *, origin):
        """Render one typed run-configuration refusal: visibly + structurally."""

        wranglerWidget._safe_status_text(
            obj,
            f"Run refused: no accepted run configuration ({refusal.reason}) "
            "— see the log.",
        )
        logger.warning("%s", refusal)
        try:
            run_config_debug_log(
                logger,
                "run_configuration_refused",
                widget=getattr(obj, "_h19_host", None),
                wrangler=obj,
                origin=origin,
                level="warning",
                **refusal.as_event_fields(),
            )
        except Exception:
            logger.debug("refusal event emit failed", exc_info=True)

    # ── Shared run-controls (CONTROLS section) ──────────────────────────
    def controls_profile(self):
        """Per-wrangler capability descriptor for the shared StaticControls:
        the mode items to show + whether Live / Batch / cores apply.  Base
        default = no Live/Batch (subclasses override)."""
        return {'modes': None, 'live': False, 'batch': False, 'cores': True}

    def attach_controls(self, controls):
        """Adopt the shared StaticControls widget.  Base just stores the ref;
        subclasses override to alias their own control attributes onto the shared
        widgets and wire the shared signals to their handlers (tracking
        connections via _connect_control for detach_controls)."""
        self._controls = controls

    def _connect_control(self, signal, slot):
        """Connect a shared-control signal to one of THIS wrangler's handlers and
        record it, so detach_controls (on wrangler swap) disconnects exactly
        these — preventing a stale wrangler from double-dispatching a click."""
        signal.connect(slot)
        self._control_conns.append((signal, slot))

    def detach_controls(self):
        """Disconnect this wrangler's shared-control connections (on swap, before
        the next wrangler attaches)."""
        for signal, slot in self._control_conns:
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._control_conns = []

    # ── Group-header toggles (UI-1, #81) ────────────────────────────────
    # Maps a toggle-group's name (e.g. 'GI') to its hidden enabling bool
    # child ('Grazing').  _install_group_toggles puts a REAL checkbox on
    # the group's header row: the checkbox is the on/off control, driving
    # the hidden bool that stays the source of truth the wrangler reads
    # (hidden so it can't repaint-uncheck while the tree is disabled
    # mid-run, #56).  Checking expands the group, unchecking collapses it;
    # a manual chevron expand just peeks at the options — it does NOT
    # enable the feature.
    _GROUP_TOGGLES = {}

    @staticmethod
    def _toggle_check_state(on):
        return (Qt.QtCore.Qt.CheckState.Checked if on
                else Qt.QtCore.Qt.CheckState.Unchecked)

    def _install_group_toggles(self, tree):
        """Add a checkbox to each _GROUP_TOGGLES group's header item and wire
        it both ways to the group's hidden enabling bool.  Call once, after
        ``tree.setParameters`` (the header items must exist)."""
        self._group_toggle_items = []
        for grp_name, bool_name in self._GROUP_TOGGLES.items():
            try:
                grp = self.parameters.child(grp_name)
                bool_param = grp.child(bool_name)
                item = next(iter(grp.items))
            except Exception:
                logger.debug("group-toggle install skipped for %s", grp_name,
                             exc_info=True)
                continue
            item.setFlags(item.flags()
                          | Qt.QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, self._toggle_check_state(bool_param.value()))
            self._group_toggle_items.append((item, grp, bool_param))
            bool_param.sigValueChanged.connect(self._sync_group_toggle_from_bool)
            # pyqtgraph's ParameterItem.optsChanged ends with updateFlags(),
            # which rebuilds the header's flags from the param opts and drops
            # ItemIsUserCheckable (any setOpts — expanded, visible — strips
            # the checkbox).  Re-assert it after every opts change; connected
            # AFTER the item's own optsChanged so it runs post-updateFlags.
            grp.sigOptionsChanged.connect(self._reassert_group_toggle_flags)
        if self._group_toggle_items:
            tree.itemChanged.connect(self._on_group_toggle_item_changed)

    def _reassert_group_toggle_flags(self, _param=None, _opts=None):
        checkable = Qt.QtCore.Qt.ItemFlag.ItemIsUserCheckable
        for item, _grp, _bool_param in getattr(self, '_group_toggle_items', ()):
            if not (item.flags() & checkable):
                item.setFlags(item.flags() | checkable)

    def _on_group_toggle_item_changed(self, item, column):
        """User (un)checked a toggle-group header: drive the hidden bool and
        open/fold the group to match."""
        if column != 0:
            return
        for it, grp, bool_param in getattr(self, '_group_toggle_items', ()):
            if it is item:
                on = (item.checkState(0)
                      == Qt.QtCore.Qt.CheckState.Checked)
                if bool(bool_param.value()) != on:
                    bool_param.setValue(on)
                    save = getattr(self, '_save_to_session', None)
                    if save is not None:
                        save()
                grp.setOpts(expanded=on)
                return

    def _sync_group_toggle_from_bool(self, param, value):
        """Programmatic bool change (session restore etc.): reflect it into
        the header checkbox."""
        for item, grp, bool_param in getattr(self, '_group_toggle_items', ()):
            if bool_param is param:
                state = self._toggle_check_state(bool(value))
                if item.checkState(0) != state:
                    item.setCheckState(0, state)
                return

    # ── Status label (specLabel / statusLabel) ──────────────────────────
    # A plain QLabel's minimum width is its full text width, so a long
    # status message (e.g. the live-GI clip advisory, ~180 chars) forces
    # the WHOLE window to expand horizontally.  Subclasses must route
    # status text through _set_status_text and call _guard_status_label
    # once after building their UI.

    def _status_label(self):
        """The status QLabel that messages route to.  Prefers the shared
        control-layer ``statusLabel`` (StaticControls) when controls are
        attached, so the message bar lives in ONE place (the control stack)
        instead of the wrangler's own orphaned label; falls back to the
        subclass's own label / specUI specLabel when standalone."""
        controls = getattr(self, '_controls', None)
        cl = getattr(controls, 'statusLabel', None) if controls is not None else None
        if cl is not None:
            return cl
        label = getattr(self, 'statusLabel', None)
        if label is not None:
            return label
        ui = getattr(self, 'ui', None)
        return getattr(ui, 'specLabel', None) if ui is not None else None

    def _guard_status_label(self):
        """Stop the status label from driving the window's minimum width:
        with an Ignored horizontal policy the label takes whatever width the
        layout gives it and overlong text clips instead of growing the window."""
        label = self._status_label()
        if label is None:
            return
        policy = label.sizePolicy()
        policy.setHorizontalPolicy(Qt.QtWidgets.QSizePolicy.Policy.Ignored)
        label.setSizePolicy(policy)

    def _status_bar(self):
        """The main window's BOTTOM QStatusBar, when this widget is hosted in a
        QMainWindow (the normal app).  None when standalone (tests / popped-out
        wranglers) — status then falls back to the elide-safe label."""
        try:
            win = self.window()
            fn = getattr(win, 'statusBar', None)
            return fn() if callable(fn) else None
        except Exception:
            return None

    def _set_status_text(self, text):
        """Route run/browse status to the main window's bottom status bar (the
        message bar).  Falls back to the elide-safe status label when there is no
        status bar (standalone): elides to the label's width, full text in the
        tooltip — see _guard_status_label."""
        text = text or ''
        bar = self._status_bar()
        if bar is not None:
            bar.showMessage(text)
            return
        label = self._status_label()
        if label is None:
            return
        label.setToolTip(text)
        if label.isVisible() and label.width() > 0:
            metrics = Qt.QtGui.QFontMetrics(label.font())
            text = metrics.elidedText(
                text, Qt.QtCore.Qt.TextElideMode.ElideRight, label.width() - 4)
        label.setText(text)

    def setup(self):
        """Sets the thread child object. Called by tthetaWidget prior
        to starting thread.
        """
        # Disconnect old thread signals to avoid duplicate emissions
        try:
            self.thread.finished.disconnect(self.finished.emit)
            self.thread.started.disconnect(self.started.emit)
            self.thread.sigUpdate.disconnect(self.sigUpdateData.emit)
            self.thread.sigUpdateGI.disconnect(self.sigUpdateGI.emit)
        except (TypeError, RuntimeError):
            pass  # Signals were never connected or already disconnected
        self.thread = wranglerThread(self.command_queue, self.fname, self.file_lock, self)
        self.thread.finished.connect(self.finished.emit)
        self.thread.started.connect(self.started.emit)
        self.thread.sigUpdate.connect(self.sigUpdateData.emit)
        self.thread.sigUpdateGI.connect(self.sigUpdateGI.emit)

    def set_fname(self, fname):
        """Changes fname attribute of self and thread.
        args:
            fname: str, path for new file.
        """
        with self.file_lock:
            if not self.thread.isRunning():
                self.fname = fname
                self.thread.fname = fname


class wranglerThread(Qt.QtCore.QThread):
    """Base class for wranglerThreads. Used to manage processes
    including data and command queues. Subclasses should override the
    run method.
    
    attributes:
        command_q: mp.Queue, queue to send commands to process
        file_lock: mp.Condition, process safe lock for file access
        fname: str, path to data file.
        input_q: mp.Queue, queue for commands sent from parent
        signal_q: mp.Queue, queue for commands sent from process
        run_configuration: the ONE accepted FrozenRunConfiguration for this
            run, published by the wrapper at admission (with its generation
            floor and the exact admitted object).

    methods:
        run: Called by start, main thread task.
    
    signals:
        sigUpdate: int, signals a new frame has been added.
        sigUpdateFile: (str, str, bool, str, bool, bool), sends new scan_name, file name
            GI flag (grazing incidence), theta motor for GI, single_image and
            series_average flag to static_scan_Widget.
        sigUpdateGI: bool, signals the grazing incidence condition has changed.
    """
    sigUpdate = Qt.QtCore.Signal(int)
    # sigUpdateFrame = Qt.QtCore.Signal(dict)
    sigUpdateFile = Qt.QtCore.Signal(str, str, bool, str, bool, bool)
    sigUpdateGI = Qt.QtCore.Signal(bool)
    sigXyeOutputReady = Qt.QtCore.Signal(str)

    def __init__(self, command_queue, fname, file_lock, parent=None):
        """command_queue: mp.Queue, queue for commands sent from parent
        fname: str, path to data file.
        file_lock: mp.Condition, process safe lock for file access

        R4-G retired ``scan_args`` here: it arrived EMPTY (the wrangler
        widget's live slot is filled later, at admission) and no worker
        stored it."""
        super().__init__(parent)
        self.input_q = command_queue # thread queue
        self.fname = fname
        self.file_lock = file_lock
        self.signal_q = Queue()
        self.command_q = Queue()
        # RS-2: serializes command TRANSITIONS between the GUI (pause/resume
        # check-then-set) and the worker's self-stop writes (write-failure
        # stop, GI freeze abort) — without it a self-stop landing between the
        # GUI's check and its 'pause' write was silently revived.
        self.command_lock = threading.Lock()
        # O-1a-W1: every worker carries the ONE accepted frozen run configuration
        # published by its wrapper at admission, plus the accepted-generation
        # watermark.  Declared on the base so image and NeXus workers admit
        # through the same owner.
        self.run_configuration = None
        self.run_configuration_floor = 0
        # O-1a-W1R: the exact object the wrapper admitted for this run.  The
        # worker-entry identity gate compares the carrier against it by ``is``.
        self._admitted_run_configuration = None

        # ── Shared batch-engine state ────────────────────────────────
        # Dynamic epoch cadence counter; reset after each committed flush.
        self._frames_since_save = 0

        # In-memory hand-off of just-integrated frames to the main
        # thread so it doesn't have to round-trip through disk.  The
        # main thread's update_data consumes this dict.
        self._published_frames: dict = _BoundedFrameHandoff()

        # Threshold filtering — subclass sets these from its UI; the
        # base default is "no threshold" so nexus / other wranglers
        # that don't expose a threshold UI pay nothing.
        # Auto-mask the uint16 ceiling (65535) as a saturated/dead sentinel.
        # ON by default = the long-standing behaviour; wranglers that expose
        # the Intensity-Threshold UI override this from the param tree.

        # Sub-label appended to log lines (e.g. "[Subtracted bg.tif]"
        # for SPEC bg-subtraction mode).  Empty string = no append.
        self.sub_label = ''

        # Mode flags read by the dispatch loops + the GUI's
        # wrangler_finished handler.
        self._scan_session_adapter = None
        self._retained_scan_session_adapter = None
        # BLOCKER 1: id of the scan whose whole-scan GI grid pre-pass has run, so
        # the freeze happens once per scan (not per chunk).  Reset on scan close.
        self._gi_prepass_scan_id = None
        # Set by _close_reduction_session when a streaming write/sink failure
        # surfaces from finish() — so the run can't report a false "success".
        self._reduction_write_error = None

    def run(self):
        """Main task. Subclasses (e.g. imageThread) override this."""
        pass

    # ── Shared batch helpers ────────────────────────────────────────

    def _cancel_token(self):
        return _CommandCancelToken(self)

    def release_retained_custody(self):
        if self._scan_session_adapter is not None:
            return False
        retained = self._retained_scan_session_adapter
        if retained is not None and not retained.release_retained_custody():
            return False
        self._retained_scan_session_adapter = None
        return True

    def _retain_dynamic_failure(self, error, status=None):
        self._reduction_write_error = error
        lock = getattr(self, "command_lock", None)
        if lock is None:
            self.command = "stop"
        else:
            with lock:
                self.command = "stop"
        if status:
            try:
                self.showLabel.emit(f"{status}: {error}")
            except Exception:
                logger.debug("showLabel emit failed for dynamic output", exc_info=True)

    @staticmethod
    def _resolve_dynamic_session_policy(
            *, frozen, plan, frame_shape, dtype, background_bytes=0,
            policy=None):
        from dataclasses import replace
        from types import SimpleNamespace
        import xrd_tools.session as session_api
        from xrd_tools.session.policy import requirements_from
        supplied = policy is not None
        requested = max(1, int(frozen.max_cores))
        requirements = requirements_from(
            SimpleNamespace(frame_shape=tuple(frame_shape), dtype=np.dtype(dtype)),
            plan, background_bytes=int(background_bytes))
        interval = 8 if plan.integration_2d is not None else 1000
        if policy is None:
            policy = session_api.resolve_session_policy(
                requirements, requested_workers=requested)
            allocation = policy.allocation
            policy = replace(policy, flush=session_api.FlushPolicy(
                interval=interval, cap=int(allocation.staging_items), margin=8))
        def invalid(detail, kind=ValueError):
            if supplied:
                raise RunConfigurationRefused(
                    "foreign", stage="dynamic-policy", detail=detail,
                    generation=int(frozen.generation))
            raise kind(detail)
        if type(policy) is not session_api.SessionPolicy:
            invalid("dynamic mount requires an exact SessionPolicy", TypeError)
        allocation = policy.allocation
        if type(allocation) is not session_api.SessionResourceAllocation:
            invalid("dynamic mount requires one exact allocation", TypeError)
        if allocation.requirements.fingerprint != requirements.fingerprint:
            invalid("dynamic policy requirement fingerprint changed")
        if not 1 <= int(allocation.workers) <= requested:
            invalid("dynamic policy worker grant is invalid")
        if type(policy.flush) is not session_api.FlushPolicy:
            invalid("dynamic mount requires an exact FlushPolicy", TypeError)
        expected = (interval, int(allocation.staging_items), 8)
        actual = (policy.flush.interval, policy.flush.cap, policy.flush.margin)
        if actual != expected:
            invalid("dynamic policy flush tuple changed")
        return policy

    def _mount_dynamic_reduction_session(
            self, key, *, frozen, scan, plan, pending_frame, output_path,
            gui_thread_id, policy=None):
        from xdart.modules.frame_publication import PublicationStore
        from xdart.modules.reduction import open_live_scan_session
        from xrd_tools.core import (DEFAULT_MODE_KEY, browse_publication_max_items,
                                    total_physical_ram_bytes)
        from xrd_tools.reduction import NexusSink
        import xrd_tools.session as session_api
        from .qt_nexus_sink import QtFrameObserver
        from .scan_session import ScanSessionAdapter
        normalized = os.path.abspath(os.fspath(output_path))
        lineage = (int(frozen.generation), normalized)
        if key != lineage:
            raise ValueError("dynamic mount key does not match its exact lineage")
        if frozen is not getattr(self, "_admitted_run_configuration", None):
            raise ValueError("dynamic mount requires the admitted run configuration")
        one_d = getattr(plan, "integration_1d", None)
        npt = int(getattr(one_d, "npt", 0) or 0)
        if one_d is None or npt <= 0:
            raise TypeError("dynamic GUI mount requires a positive 1-D plan")
        image = np.asarray(getattr(pending_frame, "map_raw", None))
        if image.ndim != 2:
            raise TypeError("dynamic GUI mount requires a 2-D detector frame")
        background = int(getattr(getattr(pending_frame, "bg_raw", None), "nbytes", 0))
        policy = self._resolve_dynamic_session_policy(
            frozen=frozen, plan=plan, frame_shape=image.shape,
            dtype=image.dtype, background_bytes=background, policy=policy)
        if self._scan_session_adapter is not None and not self._close_reduction_session():
            raise RuntimeError("prior dynamic session cleanup remains pending")
        if not wranglerThread.release_retained_custody(self):
            raise RuntimeError("prior light-1D custody cleanup remains pending")
        store = getattr(self, "publication_store", None)
        if type(store) is not PublicationStore:
            raise TypeError("dynamic GUI mount requires the artifact PublicationStore")
        intent = getattr(scan, "_same_run_intent", None)
        if intent is None:
            raise ValueError("dynamic GUI mount requires a frozen source lineage")
        committed_prefix = getattr(scan, "_committed_append_prefix", None)
        allocation = policy.allocation
        mode = (str(plan.gi.mode_1d.value) if getattr(plan, "gi", None)
                is not None else DEFAULT_MODE_KEY)
        dtype = np.dtype(np.float64)
        def buffer(role):
            return session_api.Light1DBufferLayout(npt, dtype.itemsize,
                f"{mode}:{role}", dtype.str, shared=False)
        layout = session_api.Light1DLayout((session_api.Light1DModeLayout(
            mode, buffer("coordinate"), buffer("intensity"),
            buffer("uncertainty") if one_d.error_model else None),), mode)
        total = max(0, int(total_physical_ram_bytes() or 0))
        rows = browse_publication_max_items(npt, total_ram_bytes=total)
        ceiling = (min(1024 ** 3, int(0.05 * total)) if total else
                   layout.shared_bytes + rows * layout.per_row_unique_ndarray_bytes)
        overwrite = frozen.output_mode != "Append"
        fresh_append = (not overwrite and not os.path.exists(normalized)
                        and committed_prefix is None)
        modes = session_api.required_result_modes(plan)
        target = f"nexus:{normalized}"
        adapter = None
        try:
            authority = session_api.SessionResourceAuthority.from_allocation(allocation)
            ledger = session_api.StageLedger(
                required_modes=modes, targets_by_mode={item: (target,) for item in modes})
            accounting = session_api.DynamicRunAccounting(
                ledger, run_generation=lineage[0],
                limits=session_api.DynamicAccountingLimits(1, 32, 128))
            lease = session_api.acquire_light_1d_retention(
                authority, owner=f"dynamic-gui-light-1d:{lineage[0]}:{lineage[1]}",
                generation=lineage[0], layout=layout,
                requested_rows=rows, compatibility_byte_ceiling=ceiling,
                gui_thread_id=gui_thread_id,
                funding_mode=session_api.Light1DFundingMode.REPLACE_PUBLICATION_A1,
                current_lineage_rows=None)
            adapter = ScanSessionAdapter(
                session=None, accounting=accounting, sink_graph=None,
                observer=None, publication_store=store, policy=policy,
                light_authority=authority, light_lease=lease,
                light_hooks=None, light_slot=None, generation=lineage[0])
            sink_values = dict(
                overwrite=overwrite, flush_every=None, atomic=False,
                source_base=getattr(scan, "source_base", None),
                file_lock=self.file_lock, incremental_finalization=True,
                run_configuration_provenance=frozen.as_provenance(),
                source_execution_provenance=getattr(scan, "source_execution_provenance", None),
                source_snapshots_provenance=getattr(scan, "source_snapshots_provenance", None))
            if frozen.output_mode == "Append" and not fresh_append:
                sink = NexusSink.for_existing_append(
                    normalized, intent, committed_prefix=committed_prefix,
                    **sink_values)
            else:
                sink = NexusSink(
                    normalized, same_run_intent=intent, **sink_values)
            adapter._sink_graph = sink
            session = open_live_scan_session(
                (pending_frame,), plan,
                scan_name=str(getattr(scan, "name", "scan")),
                global_mask=getattr(self, "mask", None),
                integrator=getattr(scan, "_cached_integrator", None),
                poni=getattr(self, "poni", None), executor=allocation.workers,
                cancel_token=self._cancel_token(), gi_freeze_mode=None,
                sink=sink, inflight_max=allocation.reduction_inflight,
                record_store=None, record_store_persisted_on_write=False,
                nexus_target=target, policy=policy, accounting=accounting)
            adapter._session = session
            session.set_generation(lineage[0])
            store.bind_allocation(allocation)
            store.bind_light_1d(lease)
            hooks = store.light_1d_cleanup_hooks(lease)
            adapter._light_hooks = hooks
            slot = session_api.Light1DCustodySlot(
                grant_id=lease.grant_id, owner=lease.owner,
                generation=lease.generation, cleanup_hooks=hooks)
            observer = QtFrameObserver(
                self, scan, store, lease, generation=lineage[0], active_mode_1d=mode,
                mask=getattr(self, "mask", None), publish_display=not bool(frozen.batch_mode))
            adapter._light_slot, adapter._observer = slot, observer
            accounting.bind_light_1d(lease, cleanup_hooks=hooks, custody_slot=slot)
            adapter._subscriptions = (session.on_frame_completed(observer.on_frame_completed),)
            self._scan_session_adapter = adapter
            return adapter
        except BaseException as primary:
            if adapter is None:
                raise
            try:
                command = self.command; adapter.stop()
                result = adapter.finish(join_timeout=60.0); self.command = command
                if bool(getattr(result, "failed", False)):
                    raise RuntimeError(str(getattr(
                        result, "error", None) or "dynamic mount cleanup failed"))
            except BaseException as cleanup:
                self._scan_session_adapter = adapter
                self._retain_dynamic_failure(cleanup)
                raise primary from cleanup
            raise

    def _close_reduction_session(self):
        self._gi_prepass_scan_id = None
        adapter = self._scan_session_adapter
        if adapter is not None:
            try:
                if getattr(self, "command", None) == "stop":
                    adapter.stop()
                result = adapter.finish(join_timeout=60.0)
            except BaseException as exc:
                wranglerThread._retain_dynamic_failure(self, exc, "Output cleanup remains pending")
                logger.error("dynamic reduction cleanup remains pending: %s", exc, exc_info=True)
                return False
            failed = bool(getattr(result, "failed", False))
            if failed:
                error = RuntimeError(str(getattr(result, "error", None) or "dynamic write failed"))
                wranglerThread._retain_dynamic_failure(self, error, "Dynamic output failed")
            self._scan_session_adapter = None
            slot = adapter._light_slot
            if slot is None:
                self._frames_since_save = 0
                return not failed
            slot_state = getattr(slot.state, "value", None)
            if slot_state == "retained":
                self._retained_scan_session_adapter = adapter
                signal = getattr(self, "sigRetainedCustody", None)
                if signal is not None:
                    signal.emit(int(adapter._publication_store.generation),
                                os.path.normcase(os.path.abspath(adapter._sink_graph.path)))
            elif slot_state not in {"cancelled", "released"}:
                self._scan_session_adapter = adapter
                error = RuntimeError(f"dynamic light-1D custody is {slot_state}")
                wranglerThread._retain_dynamic_failure(self, error, "Output cleanup remains pending")
                return False
            self._frames_since_save = 0
            return not failed

        return True

    def dynamic_session_running(self, *, path=None, generation=None):
        """Scalar healthy-run probe; the adapter never leaves the worker."""
        adapter = self._scan_session_adapter
        if (adapter is None or self._reduction_write_error is not None
                or adapter._session is None or not adapter._session.is_running):
            return False
        if (generation is not None and
                int(adapter._publication_store.generation) != int(generation)):
            return False
        if (path is not None and os.path.normcase(os.path.abspath(adapter._sink_graph.path))
                != os.path.normcase(os.path.abspath(os.fspath(path)))):
            return False
        return True

    def dynamic_cleanup_pending(self):
        """Scalar retained-owner probe, distinct from a healthy active run."""
        retained = self._retained_scan_session_adapter
        state = getattr(getattr(getattr(retained, "_light_slot", None), "state", None), "value", None)
        return self._scan_session_adapter is not None or state == "cleanup-pending"

    def _resolve_frame_mask(self, frozen, scan, img_data):
        """Return a stable per-scan "bad pixel" mask cached on the scan.

        Computed once from ``img_data < 0`` of the first frame seen
        by this scan; reused for every subsequent frame.  Keeping
        the mask stable across frames is what lets pyFAI's CSR
        engine cache stay valid — a single pixel changing in the
        mask invalidates the cache and forces a ~250 ms LUT rebuild
        (observed on Eiger scans where saturation flicker shifts the
        mask CRC frame-to-frame).

        Per-frame threshold filtering is NOT routed through this
        mask — see :meth:`_apply_threshold_inline` for that path
        (NaN-sentinel in the data, mask CRC unchanged).

        F3: callers in the parallel section should pre-warm the
        cache via :meth:`_prewarm_frame_mask` on the main thread
        BEFORE submitting work, so the cache is fully populated
        when N workers read it.  Without the prewarm, the first N
        workers all see ``None`` and race to write the same value —
        currently safe because every worker computes the SAME mask,
        but the invariant isn't enforced by the code and a future
        change (e.g. per-worker thresholding) could break it.
        """
        cached = getattr(scan, '_cached_data_mask', None)
        if cached is None:
            sat_fired = 0           # R3-A: how many saturation pixels were cut
            frame_size = 0
            try:
                from xdart.modules.reduction import compute_bad_pixel_mask
                from ..display_logic import integer_saturation_ceiling
                from xrd_tools.core.invalid import saturation_pixels
                arr0 = np.asarray(img_data)
                frame_size = int(arr0.size)
                mask_sat = bool(frozen.threshold.mask_saturation)
                # ONE masking implementation (xdart.modules.reduction) shared
                # with the reintegrate path so live ≡ reintegrate on the same
                # frame.  Pass the DISPLAY policy's ceiling (its legacy 65535
                # float fallback) so a float-typed raw masks identically on both
                # paths — keeping the live≡batch≡reload equivalence spine safe.
                # "Mask Saturated" (mask_sentinel) is the AUTHORITATIVE on/off:
                # OFF -> compute_bad_pixel_mask returns None -> nothing masked
                # (strong Bragg peaks that saturate are KEPT); ON -> negatives +
                # uint32 sentinel + fraction-guarded ceiling.  Computed-once +
                # cached, so pyFAI's CSR mask-CRC stays stable frame-to-frame.
                ceil = integer_saturation_ceiling(arr0)
                idx = compute_bad_pixel_mask(
                    arr0, mask_saturation=mask_sat, saturation_ceiling=ceil)
                # Preserve the legacy contract: an empty index array (not None)
                # when nothing is bad, so callers can pass it straight to pyFAI.
                cached = idx if idx is not None else np.array([], dtype=int)
                if mask_sat:
                    sat_fired = int(saturation_pixels(
                        arr0.astype(float).flatten(), ceiling=ceil).sum())
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug("frame-mask compute failed: %s", e)
                cached = None
            scan._cached_data_mask = cached
            # R3-A: the saturation mask is a default-ON behaviour change to the
            # INTEGRATION — surface it once per scan (this branch runs once;
            # later frames hit the cache).  OUTSIDE the try + guarded so the
            # advisory can never null the computed mask, and bare test holders
            # without the helper just skip it.
            if sat_fired:
                warn = getattr(self, '_warn_saturation_masked', None)
                if warn is not None:
                    warn(sat_fired, frame_size)
        return cached

    def _warn_saturation_masked(self, n_pixels: int, frame_size: int) -> None:
        """R3-A: one-time advisory that the 'Mask Saturated' policy actually
        excluded a detector-ceiling block from the INTEGRATION this run.

        The saturation mask is default-ON and changes integrated intensities,
        so a silent fire hides a real data effect.  Logged loud + surfaced in
        the GUI status line when a ``showLabel`` signal is present (absent on
        the bare test holders, so this no-ops there).  Called from the
        once-per-scan mask compute, so it fires once per run, not per frame.
        """
        pct = (100.0 * n_pixels / frame_size) if frame_size else 0.0
        msg = (f"Mask Saturated: {n_pixels} detector-ceiling pixel(s) "
               f"({pct:.2f}% of the frame) excluded from integration — a "
               f"dead/overflowed module. Untick 'Mask Saturated' to keep them.")
        logger.warning(msg)
        emit = getattr(getattr(self, 'showLabel', None), 'emit', None)
        if emit is not None:
            try:
                emit(msg)
            except Exception:
                logger.debug("showLabel emit failed for saturation advisory",
                             exc_info=True)

    def _prewarm_frame_mask(self, frozen, scan, img_data) -> None:
        """Populate ``scan._cached_data_mask`` on the main thread.

        F3 — prevents the racy initialization that happens when N
        parallel workers all simultaneously see a ``None`` cache and
        each compute + write the same mask.  Computing on the main
        thread before submitting any worker means every worker only
        ever does a cache *read* against a stable value.

        Idempotent: a no-op when the cache is already set.  Called
        from each wrangler's run loop with the first frame's
        ``img_data`` before the parallel section.
        """
        if getattr(scan, '_cached_data_mask', None) is not None:
            return
        self._resolve_frame_mask(frozen, scan, img_data)

    def _apply_threshold_inline(self, frozen, img_data):
        """Pre-clamp pixels outside the threshold band to NaN.

        Returns a fresh float32 array with out-of-band pixels
        replaced by NaN.  pyFAI's CSR integrator skips NaN pixels
        automatically (no ``dummy``/``delta_dummy`` kwargs needed),
        and NaN propagates cleanly through bg subtraction and monitor
        normalization arithmetic inside ``frame.integrate_1d/2d``.

        No-op when ``self.apply_threshold`` is False — subclasses
        that don't expose a threshold UI inherit a free pass-through.
        """
        if not frozen.threshold.apply_threshold:
            return img_data
        img = np.asarray(img_data, dtype=np.float32, copy=True)
        bad = (img < frozen.threshold.threshold_min) | (img > frozen.threshold.threshold_max)
        img[bad] = _THRESHOLD_NAN
        return img
