# -*- coding: utf-8 -*-
from __future__ import annotations

import logging

from xdart.modules.reduction import frame_from_live_frame
from xrd_tools.session import DynamicRunState, Light1DCleanupPending

logger = logging.getLogger(__name__)


class ScanSessionAdapter:
    def __init__(
            self, *, session, accounting, sink_graph, observer,
            publication_store, policy, light_authority, light_lease,
            light_hooks, light_slot, generation, subscriptions=()) -> None:
        self._session, self._accounting = session, accounting
        self._sink_graph, self._observer = sink_graph, observer
        self._publication_store, self._policy = publication_store, policy
        self._light_authority, self._light_lease = light_authority, light_lease
        self._light_hooks, self._light_slot = light_hooks, light_slot
        self._generation, self._subscriptions = int(generation), tuple(subscriptions)
        self._cleanup_retry_token = None

    def discover(self, key, *, group, ordinal, output_label) -> None:
        self._accounting.discover(key, group=group, ordinal=int(ordinal), output_label=int(output_label))

    def begin_attempt(self, key, *, source_revision):
        return self._accounting.begin_attempt(key, source_revision=int(source_revision))

    def record_enqueued(self, token) -> None:
        self._accounting.record_enqueued(token)

    def record_failed(self, token, *, error="submit refused", retryable=True) -> None:
        self._accounting.record_failed(token, error=str(error), retryable=bool(retryable))

    def record_cancelled(self, token, *, reason="cancelled") -> None:
        self._accounting.record_cancelled(token, reason=str(reason))

    def submit(self, live, *, attempt_token) -> bool:
        if self._observer is None or self._session is None:
            raise RuntimeError("dynamic session is not completely mounted")
        try:
            if not self._observer.register(live, attempt_token):
                return False
            accepted = self._session.submit(frame_from_live_frame(live), attempt_token=attempt_token)
        except BaseException as primary:
            try:
                self._observer.unregister(live, attempt_token)
                state = self._accounting.snapshot().attempt_states.get(attempt_token)
                if getattr(state, "value", None) in {"provisional", "enqueued"}:
                    self.record_failed(attempt_token, error=str(primary), retryable=True)
            except BaseException as cleanup:
                raise primary from cleanup
            raise
        if not accepted:
            self._observer.unregister(live, attempt_token)
        return bool(accepted)

    def quiesce(self, timeout=None) -> bool:
        return bool(self._session.pause(timeout=timeout))

    def resume(self) -> None:
        self._session.resume()

    def should_flush(self, frames_since_flush, *, unsaved_in_memory=None, force=False) -> bool:
        return bool(self._session.policy.should_flush(
            frames_since_flush=int(frames_since_flush),
            unsaved_in_memory=unsaved_in_memory, force=bool(force)))

    def stop(self) -> None:
        if self._session is not None:
            self._session.stop()

    def commit_epoch(self):
        return self._session.commit_epoch()

    def extend_live(self, intent):
        return self._session.extend_live(intent)

    def finish(self, *, join_timeout=60.0):
        partial_mount, result = self._subscriptions == (), None
        if self._session is not None:
            result = self._session.finish(raise_on_failure=False, join_timeout=join_timeout)
        elif self._sink_graph is not None:
            result = self._sink_graph.abort(None)
        if self._observer is not None:
            self._observer.close()
        self._unsubscribe()
        if partial_mount:
            self._release_partial_light()
        return result

    def _release_partial_light(self, *, reason="dynamic mount failed") -> None:
        lease, store = self._light_lease, self._publication_store
        bound = lease is not None and getattr(store, "_light_1d", None) is lease
        slot_state = getattr(getattr(self._light_slot, "state", None), "value", None)
        try:
            if slot_state == "retained":
                self._light_slot.release(reason=reason)
            elif slot_state == "cleanup-pending":
                self._light_slot.retry_cleanup(self._cleanup_retry_token)
            elif slot_state in {None, "pending"} and lease is not None:
                if bound and self._light_hooks is None:
                    self._light_hooks = store.light_1d_cleanup_hooks(lease)
                state = getattr(lease.state, "value", None)
                if state == "active":
                    lease.release(reason=reason, hooks=self._light_hooks if bound else None)
                elif state == "cleanup-pending":
                    lease.retry_cleanup(self._cleanup_retry_token, hooks=self._light_hooks if bound else None)
                elif state != "released":
                    raise RuntimeError(f"partial light-1D lease is unsettled: {state}")
                if slot_state == "pending":
                    self._light_slot.cancel(terminal=DynamicRunState.ABORTED)
            elif slot_state not in {"cancelled", "released"}:
                raise RuntimeError(f"partial light-1D custody is unsettled: {slot_state}")
        except Light1DCleanupPending as exc:
            self._cleanup_retry_token = exc.token
            raise
        self._cleanup_retry_token = None
        allocation = getattr(self._policy, "allocation", None)
        if (getattr(store, "allocation", None) is allocation
                and getattr(store, "_light_1d", None) is None):
            store.clear()

    def _unsubscribe(self) -> None:
        if self._subscriptions is None or not self._subscriptions:
            return
        failures = []
        for unsubscribe in self._subscriptions:
            try:
                unsubscribe()
            except BaseException as exc:
                failures.append((unsubscribe, exc))
        if failures:
            self._subscriptions = tuple(callback for callback, _ in failures)
            raise failures[0][1]
        self._subscriptions = None

    def release_retained_custody(self) -> bool:
        state = getattr(getattr(self._light_slot, "state", None), "value", None)
        if state not in {"pending", "cleanup-pending", "retained"}:
            return state in {"released", "cancelled"}
        if (state == "pending" and getattr(
                self._accounting, "_light_lease", None) is self._light_lease):
            raise RuntimeError("accounting still owns pending light-1D custody")
        try:
            self._release_partial_light(
                reason="dynamic mount failed" if state == "pending" else "display replacement")
        except Light1DCleanupPending as exc:
            logger.error("retained light-1D cleanup remains pending: %s", exc)
            return False
        return getattr(self._light_slot.state, "value", None) in {"released", "cancelled"}
