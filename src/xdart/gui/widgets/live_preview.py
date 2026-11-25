"""Shared Live preview lifecycle, foreground arbitration and view ownership."""

from dataclasses import dataclass, replace
from weakref import WeakSet

from pyqtgraph.Qt import QtCore

from xdart.gui.pages.operation_owner import OperationTerminalStatus
from xdart.gui.pages.values import PageCleanup
from xdart.gui.widgets.latest_request import LatestRequestOwner
from xrd_tools.analysis.live_preview import PreviewGroupObserver, PreviewProgress, RSMPreviewWork


_mounts = WeakSet()
_DISPLAY_RESERVE_BYTES = 64 * 1024 * 1024
_STAGING_RESERVE_BYTES = 512 * 1024 * 1024
_STATUS_TIP = ("Provisional preview from complete per-point detector files. The finished scan is processed and "
               "saved separately. A growing grid or changing hot-pixel mask can replay earlier points.")


@dataclass(frozen=True, slots=True)
class _Slice:
    epoch: object
    work: object
    target: object


@dataclass(frozen=True, slots=True)
class _Verify:
    request: _Slice
    snapshot: object
    verify_assets: object


@dataclass(frozen=True, slots=True)
class _RetiredGroup:
    identity: tuple
    members: tuple[str, ...]


class LivePreviewOwner(QtCore.QObject):
    """Drive one bounded science worker and accept snapshots through a fresh custody check."""

    def __init__(self, page, watcher, *, foreground, present, fallback):
        super().__init__(page)
        self._watcher = watcher
        self._foreground = foreground
        self._present = present
        self._fallback = fallback
        self._bar = page.run_bar
        self.preference = page.advanced.add_live_preview_setting()
        def open_advanced():
            if page.advanced_button.isEnabled():
                page.advanced_button.click()
                page.advanced.raise_()
                page.advanced.activateWindow()
                self.preference.setFocus()

        self._bar.mount_live_preview_option(open_advanced, self.preference.isChecked())
        self.preference.toggled.connect(self._bar.set_live_preview_enabled)
        self._requests = LatestRequestOwner(
            self._compute, self._computed, same=lambda left, right: left is right,
            parent=self, progress=self._slice_progress,
        )
        self._verification = LatestRequestOwner(
            self._verify, self._verified, same=lambda left, right: left is right, parent=self,
        )
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self.poll_due)
        self._epoch = self._work = self._candidate = self._observation = None
        self._snapshot = self._saved = self._done_target = self._staged = None
        self._display_witness = None
        self._recheck_display = False
        self._staging_bytes = 0
        self._published_at = float("-inf")
        self._retired = {}
        self._activity = PreviewProgress()
        self._ownership_cancelled = False
        self._worker = self._requests.worker
        _mounts.add(self)

    @property
    def clock(self):
        return self._watcher.clock

    def settings(self):
        return {"enabled": self.preference.isChecked()}

    @staticmethod
    def settings_enabled(document):
        value = document.get("live_preview", {})
        if type(value) is not dict:
            raise ValueError("Live preview settings are malformed")
        enabled = value.get("enabled", True)
        if type(enabled) is not bool:
            raise ValueError("Live preview enabled must be a boolean")
        return enabled

    def apply_settings(self, enabled):
        self.preference.setChecked(enabled)

    @property
    def watching(self):
        return self._epoch is not None

    @property
    def working(self):
        return self._requests.running is not None or self._verification.running is not None

    @property
    def draining(self):
        return self._ownership_cancelled and self._work is not None

    @property
    def snapshot(self):
        return self._snapshot

    @property
    def retired_groups(self):
        return tuple(self._retired.values())

    @property
    def progress(self):
        progress = self._work.progress if self._work is not None else self._activity
        if self._observation is not None:
            observation = self._observation
            progress = replace(progress, captures=observation.captures, parser_opens=observation.parser_opens,
                               observation_sweeps=observation.observation_sweeps)
        if progress.replay_target is not None and self._candidate is not None:
            target = progress.replay_target
            watermark = len(self._candidate.points)
            if watermark > target:
                progress = replace(progress, pending_watermark=watermark)
        return progress

    def start(self, observer, science, *, rsm=False, image_axis="q_chi"):
        self._activity = PreviewProgress()
        self._observation = None
        self._retired.clear()
        if not self.preference.isChecked():
            return
        if rsm:
            self._work = RSMPreviewWork(science)
        else:
            from xrd_tools.analysis.live_preview_stitch import StitchPreviewWork
            self._work = StitchPreviewWork(science, image_axis=image_axis)
        self._epoch = object()
        self._candidate = self._done_target = self._snapshot = self._staged = self._saved = None
        self._display_witness = None
        self._recheck_display = False
        self._published_at = float("-inf")
        self._ownership_cancelled = False
        observer.preview_observe = PreviewGroupObserver(observer.selection, observer.read_options,
                                                         observer.select_labels)
        self._timer.start()

    def observe(self, observation):
        if not self.watching or observation is None:
            return
        self._observation = observation
        old = self._candidate
        candidate = observation.candidate
        if candidate is not None and candidate.identity not in self._retired:
            self._candidate = candidate
        else:
            self._candidate = None
        if self._snapshot is not None and self._candidate is not None:
            shown = self._snapshot
            if self._candidate.identity == shown.group_identity:
                seen = self._candidate.points
                covered = shown.input_prefix.points
                if len(seen) < len(covered) or any(
                        left.identity != right.identity for (_, left), (_, right) in zip(seen, covered)):
                    self._clear_snapshot(shown.group_identity)
                elif self._done_target is not None:
                    # Asset drift can occur while the detector has paused.
                    self._recheck_display = True
        if old is not None and (self._candidate is None or not self._candidate.points) and self._snapshot is not None:
            if self._snapshot.group_identity not in self._retired:
                self._clear_snapshot()
        self.poll_due()

    def _yield(self):
        return self._ownership_cancelled or any(mount._foreground() for mount in tuple(_mounts))

    def _display_room(self, snapshot):
        # The old and replacement projections can coexist while the view swaps.
        visible = sum(mount._snapshot.display_bytes for mount in tuple(_mounts)
                      if mount._snapshot is not None and mount._snapshot is not snapshot)
        return visible + snapshot.display_bytes <= _DISPLAY_RESERVE_BYTES

    def _retire_staging(self):
        self._staged = self._done_target = None
        if self.working:
            self._recheck_display |= self._display_witness is not None
            self._verification.release()
        else:
            self._recheck_display |= bool(self._staging_bytes and self._display_witness is not None)
            self._requests.forget()
            self._verification.forget()
            self._work.retire_staging()
            self._staging_bytes = 0

    def _admit_staging(self, target):
        charge = self._work.staging_bytes(target)
        other = sum(mount._staging_bytes for mount in tuple(_mounts) if mount is not self)
        if charge + other > _STAGING_RESERVE_BYTES:
            return False
        self._staging_bytes = charge
        return True

    def poll_due(self):
        if not self.watching:
            if not self.working:
                self._timer.stop()
                was_active = self._work is not None
                self._work = None
                self._staging_bytes = 0
                self._bar.set_live_preview_working(False)
                if was_active:
                    self._watcher._state_changed()
            return
        if self._yield():
            # A running slice owns its state until it yields. The detached
            # displayed projection keeps its separate reservation.
            self._retire_staging()
            return
        if self._recheck_display and not self.working:
            self._recheck_display = False
            if self._display_witness is not None:
                self._verification.forget()
                self._verification.ask(self._display_witness)
                return
        current = self._candidate
        if (self._staged is not None and self.clock() >= self._published_at + 2 and not self.working
                and self._display_room(self._staged.snapshot)):
            staged, self._staged = self._staged, None
            if staged.request.target.identity not in self._retired:
                self._verification.ask(staged)
                return
        if self.working or current is None or not current.points:
            if self.progress.replay_target is not None:
                self._status("Rebuilding preview")
            elif current is not None and not current.points:
                self._work.progress = replace(self._work.progress, waiting=True)
            return
        if self._done_target is not None:
            prior, count, ended, waiting_observation = self._done_target
            if (prior.identity == current.identity and ended == current.ended and count == len(current.points)
                    and (waiting_observation is None or waiting_observation is self._observation)
                    and all(a.identity == b.identity for (_, a), (_, b) in zip(prior.points, current.points))):
                return
        if self._admit_staging(current):
            self._requests.ask(_Slice(self._epoch, self._work, current))
        else:
            self._status("Preview waiting for science memory")

    def _compute(self, request, cancel, progress):
        return request.work.run(request.target, cancel.is_set, self._yield, progress)

    def _slice_progress(self, request, update):
        if request.epoch is self._epoch:
            self._status(update.stage)

    def _computed(self, request, terminal):
        if request.epoch is not self._epoch or request.target.identity in self._retired:
            return False
        if terminal.status is not OperationTerminalStatus.RETURNED:
            request.work.invalidate(request.target.identity)
            self._clear_snapshot(request.target.identity)
            self._status("Preview waiting for verified data")
            return False
        snapshot = terminal.payload
        progress = request.work.progress
        if self._yield():
            self._retire_staging()
            return False
        if progress.waiting:
            self._done_target = (request.target, len(request.target.points), request.target.ended, self._observation)
        elif progress.replay_target is None and progress.included + progress.held + progress.excluded >= len(
                request.target.points):
            self._done_target = request.target, len(request.target.points), request.target.ended, None
        if snapshot is None:
            if progress.invalidated:
                self._clear_snapshot(request.target.identity)
            self._status("Preview waiting for complete points" if progress.waiting else "Preview screening points")
            return False
        if snapshot is not self._snapshot:
            verification = _Verify(request, snapshot, request.work.kernel.asset_witness())
            if self.clock() >= self._published_at + 2 and self._display_room(snapshot):
                self._staged = None
                self._verification.ask(verification)
            else:
                self._staged = verification
                if not self._display_room(snapshot):
                    self._status("Preview waiting for display memory")
        return True

    @staticmethod
    def _verify(verification, cancel, _progress):
        try:
            verification.verify_assets()
            snapshot = verification.snapshot
            return snapshot.input_prefix.verifies(metadata=snapshot.metadata_prefix, cancelled=cancel.is_set)
        except (OSError, ValueError):
            return False

    def _verified(self, verification, terminal):
        request = verification.request
        rechecking_display = verification is self._display_witness
        if (request.epoch is not self._epoch
                or request.target.identity in self._retired and not rechecking_display):
            return False
        if terminal.status is not OperationTerminalStatus.RETURNED or terminal.payload is not True:
            request.work.invalidate(request.target.identity)
            self._done_target = None
            self._clear_snapshot(request.target.identity)
            self._status("Preview waiting after an input change")
            return False
        snapshot = verification.snapshot
        if rechecking_display:
            return False
        if snapshot is self._snapshot:
            return True
        if self._yield():
            self._retire_staging()
            return False
        if not self._display_room(snapshot):
            self._staged = verification
            self._status("Preview waiting for display memory")
            return False
        visible = self._snapshot
        if (visible is not None and visible.group_identity == snapshot.group_identity
                and len(snapshot.points) < len(visible.points)
                and snapshot.points == visible.points[:len(snapshot.points)]):
            # Reconstructing retired staging must catch up to the reserved view.
            return False
        if self._snapshot is None or snapshot.rank >= self._snapshot.rank:
            self._publish(verification)
        return True

    def _publish(self, verification):
        snapshot = verification.snapshot
        if snapshot.group_identity in self._retired:
            return
        if self._saved is not None and snapshot.rank <= self._saved[0]:
            return
        try:
            self._present(snapshot)
        except (ValueError, TypeError) as error:
            self._status(f"Preview display refused: {error}")
            return
        self._snapshot = snapshot
        self._display_witness = verification
        self._published_at = self.clock()
        point_count = len(snapshot.points)
        text = f"Provisional · scan {snapshot.points[-1][0]} · {point_count} points shown"
        if self.progress.held:
            text += f" · {self.progress.held} held"
        if snapshot.mask_pending:
            text += " · Hot-pixel mask pending"
        elif snapshot.mask_update_queued:
            text += " · Hot-pixel mask update queued"
        elif snapshot.member_masks:
            text += " · Provisional hot-pixel mask"
        self._status(text)

    def _status(self, text):
        self._bar.set_readiness_summary(text, ready=True, tooltip=f"{text}\n{_STATUS_TIP}", live=True)

    def consume(self, group, *, skipped=False):
        if not self.watching:
            return
        info = _RetiredGroup(group.identity, tuple(member.key for member in group.members))
        self._retired.setdefault(group.identity, info)
        if skipped:
            self._clear_snapshot(group.identity)
        if self._candidate is not None and self._candidate.identity == group.identity:
            self._candidate = None

    def native_saved(self, group, path):
        """Register the saved fallback and answer whether native may select and paint it."""
        if not self.watching or self._observation is None:
            return True
        rank = self._observation.ranks.get(group.identity)
        if rank is None:
            rank = self._snapshot.rank if (self._snapshot is not None
                                           and self._snapshot.group_identity == group.identity) else (0, 0, 0)
        if self._saved is None or rank > self._saved[0]:
            self._saved = rank, path
        visible = self._snapshot
        if visible is None:
            return rank >= self._saved[0]
        if visible.group_identity == group.identity or rank > visible.rank:
            self._snapshot = None
            self._display_witness = None
            self._staged = None
            self._verification.forget()
            return True
        return False

    def _clear_snapshot(self, group=None):
        if self._staged is not None and (group is None or self._staged.snapshot.group_identity == group):
            self._staged = None
        if self._snapshot is not None and (group is None or self._snapshot.group_identity == group):
            self._snapshot = None
            self._display_witness = None
            self._verification.forget()
            self._fallback(None if self._saved is None else self._saved[1])

    def stop(self, *, closing=False):
        self._epoch = self._candidate = self._observation = self._done_target = None
        self._ownership_cancelled = True
        self._requests.release()
        self._verification.release()
        if closing:
            self._snapshot = self._staged = None
            self._display_witness = None
        else:
            self._clear_snapshot()
        if self.working:
            self._bar.set_live_preview_working(True)
            self._timer.start()

    def shutdown(self):
        self.stop(closing=True)
        self._timer.stop()
        results = self._requests.shutdown(), self._verification.shutdown()
        receipt = next((result for result in results if result.status is not PageCleanup.CLEAN), results[0])
        if receipt.status is PageCleanup.CLEAN:
            self._work = None
            self._staging_bytes = 0
        return receipt
