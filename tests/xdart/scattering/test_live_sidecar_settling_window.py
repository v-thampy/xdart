"""A sidecar settles within the limit from its first sighting (beamline 2026-10-25: four Live scans were skipped "the
metadata of <frame> did not settle within 10 s" a moment after their sidecars first appeared, about 10 s after the
image, because the settling window ran from the image's first wait). The executor's own wait owner, on a fake clock."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from xdart.gui.tabs.scattering.adapters import run_executor
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor

IMAGE = "scan_0002.tif"


def _watcher(monkeypatch):
    clock, skipped = [100.0], []
    monkeypatch.setattr(run_executor, "monotonic", lambda: clock[0])
    executor = StandardRunExecutor(join_timeout=1.)
    monkeypatch.setattr(executor, "_skip_live_append", lambda *args, **kwargs: skipped.append(kwargs["reason"]))
    run = SimpleNamespace(live_waiting={}, sidecars_seen=True, metadata_given_up=[])
    waits, given_up = {}, {}

    def look(at, waiting, **kwargs):
        clock[0] = at
        return executor._watch_metadata_wait(run, Path("/out/scan_int1d.nexus"), "key", waiting, waits, given_up,
                                             **kwargs)
    return look, skipped, waits


def test_a_sidecar_first_seen_near_the_limit_gets_its_own_settling_window(monkeypatch) -> None:
    look, skipped, waits = _watcher(monkeypatch)
    assert look(100.0, (IMAGE, "sidecar")) is False      # the image is listed, its sidecar not yet
    assert look(109.9, (IMAGE, "sidecar")) is False      # still none, 9.9 s on
    assert look(109.95, (IMAGE, "settling")) is False    # first seen: one look, not yet two alike
    assert look(110.05, (IMAGE, "settling")) is False    # changed again, past the image's 10 s: looked at again
    assert look(110.2, None, admitted=frozenset({IMAGE})) is False  # held still: admitted
    assert skipped == [] and "key" not in waits


def test_a_sidecar_that_never_holds_still_is_skipped_at_the_limit_from_its_first_sighting(monkeypatch) -> None:
    look, skipped, _waits = _watcher(monkeypatch)
    look(100.0, (IMAGE, "sidecar"))
    look(105.0, (IMAGE, "settling"))  # first seen
    look(106.0, (IMAGE, "sidecar"))   # gone for a look, then back: the first sighting still bounds the wait
    look(107.0, (IMAGE, "settling"))
    assert look(114.9, (IMAGE, "settling")) is False
    assert look(115.01, (IMAGE, "settling")) is True
    assert skipped == [f"the metadata of {IMAGE} did not settle within 10 s"]


def test_an_admit_poll_that_revives_the_wait_does_not_skip_a_sidecar_first_seen_near_the_limit(monkeypatch) -> None:
    # The beamline sequence (traced by xdart-2a): the look admits, then the attempt ends unfinished (``retried``) or
    # admits without the frame; the wait revives with its held kind, "settling", within its own window.
    for revive in (dict(retried=True), dict(admitted=frozenset({"scan_0001.tif"}))):
        look, skipped, waits = _watcher(monkeypatch)
        look(100.0, (IMAGE, "sidecar"))
        look(109.0, (IMAGE, "settling"))           # its sidecar first seen 9 s after the image
        assert look(110.05, None, **revive) is False, revive
        assert skipped == [] and waits["key"][3] == "settling", (revive, skipped, waits)


def test_a_sidecar_first_seen_past_the_images_limit_is_still_looked_at_twice(monkeypatch) -> None:
    look, skipped, waits = _watcher(monkeypatch)
    look(100.0, (IMAGE, "sidecar"))
    assert look(110.05, (IMAGE, "settling")) is False  # first seen at the first look past 10 s
    assert look(110.15, None, admitted=frozenset({IMAGE})) is False
    assert skipped == [] and "key" not in waits
