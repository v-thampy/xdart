"""A preview's adoption owns the bytes it actually used."""

from threading import Event

from tests.xdart.pages.live_per_scan_support import wait_until
from tests.xdart.pages.live_preview_support import advance_preview, start_preview
from tests.xdart.pages.test_live_per_scan_native import make_page, qapp  # noqa: F401
from tests.xdart.pages.test_live_preview_science import _prefix, _raw, _rsm_acquisition


def test_unprocessed_point_replacement_keeps_the_verified_prefix(qapp, make_page, tmp_path, monkeypatch):
    page = make_page("rsm")
    spec, images = _rsm_acquisition(page, tmp_path, values=(0,) * 16)
    for index in range(16):
        _raw(images / f"{spec.stem}_scan1_{index:04d}.raw", 32)
    owner = start_preview(page, qapp)
    pending, release, checked = Event(), Event(), Event()
    poll, adopt = owner._worker.poll, owner._verification._done

    def hold_compute(identity):
        if owner._worker._terminal is not None and (not release.is_set() or checked.is_set()):
            pending.set()
            return None
        return poll(identity)

    def observe_adoption(request, terminal):
        result = adopt(request, terminal)
        checked.set()
        return result

    monkeypatch.setattr(owner._worker, "poll", hold_compute)
    monkeypatch.setattr(owner._verification, "_done", observe_adoption)
    advance_preview(page, owner, qapp, drain=False)
    wait_until(qapp, pending.is_set)
    replacement = images / "replacement.raw"
    _raw(replacement, 91)
    replacement.replace(images / f"{spec.stem}_scan1_0015.raw")
    release.set()
    wait_until(qapp, checked.is_set)
    assert _prefix(owner, *(("1.1", index) for index in range(8)))
    assert owner.progress.invalidated == 0
