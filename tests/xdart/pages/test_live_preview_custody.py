"""A preview's adoption owns the bytes it actually used."""

from threading import Event

import pytest

from tests.xdart.pages.live_per_scan_support import wait_until
from tests.xdart.pages.live_preview_support import advance_preview, start_preview
from tests.xdart.pages.test_live_per_scan_native import _write_spec, make_page, qapp  # noqa: F401
from tests.xdart.pages.test_live_preview_science import (
    _next, _prefix, _raw, _rsm_acquisition, _stitch_acquisition, _stitch_raw, _stitch_scan,
)


def _hold_first_adoption(owner, monkeypatch):
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
    return pending, release, checked


def test_unprocessed_point_replacement_keeps_the_verified_prefix(qapp, make_page, tmp_path, monkeypatch):
    page = make_page("rsm")
    spec, images = _rsm_acquisition(page, tmp_path, values=(0,) * 16)
    owner = start_preview(page, qapp)
    pending, release, checked = _hold_first_adoption(owner, monkeypatch)
    for index in range(16):
        _raw(images / f"{spec.stem}_scan1_{index:04d}.raw", 32)
    advance_preview(page, owner, qapp, drain=False)
    wait_until(qapp, pending.is_set)
    replacement = images / "replacement.raw"
    _raw(replacement, 91)
    replacement.replace(images / f"{spec.stem}_scan1_0015.raw")
    release.set()
    wait_until(qapp, checked.is_set)
    assert _prefix(owner, *(("1.1", index) for index in range(8)))
    assert owner.progress.invalidated == 0


def test_unprocessed_stitch_energy_used_for_geometry_is_rechecked(qapp, make_page, tmp_path, monkeypatch):
    page = make_page("stitch")
    form, spec, _ = _stitch_acquisition(page, tmp_path, backend="xu_hist")
    header = spec.read_text(encoding="utf-8").split("#S 1", 1)[0]
    first = _stitch_scan(spec, 1, (14.,) * 16)
    earlier, last = first.rsplit("17000.018", 1)
    _write_spec(spec, header + earlier + "16995.018" + last)
    owner = start_preview(page, qapp)
    pending, release, checked = _hold_first_adoption(owner, monkeypatch)
    for index in range(16):
        _stitch_raw(form, 1, index, 30)
    advance_preview(page, owner, qapp, drain=False)
    wait_until(qapp, pending.is_set)
    assert owner._work.kernel.plan.metadata_energy_eV == 16995.018
    _write_spec(spec, header + first)
    release.set()
    wait_until(qapp, checked.is_set)
    assert owner.snapshot is None and owner.progress.invalidated > 0


@pytest.mark.parametrize("removed", ("covered-raw", "calibration"))
def test_removing_a_covered_input_revokes_the_visible_prefix(qapp, make_page, tmp_path, removed):
    page = make_page("rsm")
    spec, images = _rsm_acquisition(page, tmp_path, values=(0., 0.))
    owner = start_preview(page, qapp)
    first, last = (images / f"{spec.stem}_scan1_{index:04d}.raw" for index in range(2))
    _raw(first, 32)
    _raw(last, 44)
    _next(page, owner, qapp, lambda: _prefix(owner, ("1.1", 0), ("1.1", 1)))
    revoked = owner.snapshot
    if removed == "covered-raw":
        last.unlink()
    else:
        (tmp_path / "calibration/rsm_canonical.xu.json").unlink()
    advance_preview(page, owner, qapp)
    assert owner.snapshot is not revoked
