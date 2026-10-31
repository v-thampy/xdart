"""Terminal support: real available coverage and revision-bound cache ownership."""
from dataclasses import replace
import hashlib
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from tests.core.test_terminal_support_seal import _full_seal
from tests.xdart.scattering.test_angular_support_recording import _int_recording
from tests.xdart.scattering.test_angular_support_strip import _analytic, _masked_series, TOL
from tests.xdart.scattering.test_browse_reference_rings_run import _page, _run, _browsing
from tests.xdart.test_latest_request import qapp, _settled  # noqa: F401


@pytest.fixture
def source(qapp, tmp_path):
    from xdart.gui.tabs.scattering.context_projection import attach_saved_support_source
    from xdart.gui.tabs.scattering.shell_values import ScientificProjection
    with _page(qapp, _masked_series(tmp_path / "run")) as (page, _view, observed):
        _int_recording(qapp, page, toggle=True)
        _run(qapp, page, observed)
        context = _browsing(qapp, page)
        result = attach_saved_support_source(ScientificProjection(processing_mode="Int 1D"), context)
        result = result.saved_support_source
        assert result is not None and result.terminal is not None
        assert result.terminal.digest != hashlib.sha256(Path(result.path).read_bytes()).hexdigest(), \
            "the real writer must supply a held evidence seal, not a full-file digest"
    return result


def _change_axis_preserving_size_and_mtime(source):
    before = os.stat(source.path)
    with h5py.File(source.path, "r+") as handle:
        axis = handle[f"{source.entry}/integrated_1d/axis_1"]
        axis[...] = axis[()] * 1.1
    os.utime(source.path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with open(source.path, "rb") as handle:
        after = os.fstat(handle.fileno())
    assert after.st_size == before.st_size and after.st_mtime_ns == before.st_mtime_ns
    assert after.st_ctime_ns != source.terminal.ctime_ns
    return after


@pytest.mark.parametrize("domain", ["full", "held"])
def test_real_terminal_reconstruction_is_available_and_changed_revision_refuses(source, domain, monkeypatch):
    from xrd_tools.analysis.angular_support import reconstruct_saved_support
    from xrd_tools.io import output_transaction as transaction
    if domain == "full":
        terminal = _full_seal(Path(source.path))
        source = replace(source, terminal=terminal,
                         snapshot=transaction.revalidate_stream_terminal(source.path, terminal))
    plain = replace(source, terminal=None, snapshot=transaction.capture_target_snapshot(source.path))
    control = reconstruct_saved_support(plain.request(source.labels))
    assert control and all(profile.available for profile in control.values())
    for profile in control.values():
        np.testing.assert_allclose(profile.supported_deg, _analytic(profile), atol=TOL * 360)

    def no_hash(_handle):
        pytest.fail("a modern terminal seal must not hash the output")
    monkeypatch.setattr(transaction, "_sha256_handle", no_hash)
    actual = reconstruct_saved_support(source.request(source.labels))
    for label, profile in actual.items():
        assert profile.available
        np.testing.assert_array_equal(profile.supported_deg, control[label].supported_deg)
    _change_axis_preserving_size_and_mtime(source)
    refused = reconstruct_saved_support(source.request(source.labels))
    assert all(not profile.available and "could not be read" in profile.detail for profile in refused.values())


def test_terminal_revision_replaces_cached_profiles_even_when_snapshots_compare_equal(source, qapp):
    from xdart.gui.tabs.scattering.angular_support_owner import AngularSupportOwner
    from xrd_tools.io.output_transaction import revalidate_stream_terminal
    owner = AngularSupportOwner()
    try:
        owner.profiles(source, source.labels)
        _settled(owner._requests)
        old = owner.profiles(source, source.labels)
        assert old and all(profile.available for profile in old.values())
        changed = _change_axis_preserving_size_and_mtime(source)
        # A newly admitted object revision can retain a held stream's evidence digest.
        terminal = replace(source.terminal, ctime_ns=changed.st_ctime_ns)
        snapshot = revalidate_stream_terminal(source.path, terminal)
        assert snapshot == source.snapshot, "TargetSnapshot equality omits the terminal's ctime"
        updated = replace(source, snapshot=snapshot, terminal=terminal)
        assert owner.profiles(updated, updated.labels) == {}, "a new seal cannot reuse the previous coverage"
        assert updated.request(updated.labels).key != source.request(source.labels).key
        _settled(owner._requests)
        current = owner.profiles(updated, updated.labels)
        assert current and all(not profile.available for profile in current.values())
        assert owner.completed_requests() == 2
    finally:
        _settled(owner._requests)
        assert owner.shutdown().status.name == "CLEAN"
        owner.deleteLater()
