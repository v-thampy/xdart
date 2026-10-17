"""Round-2 review: reconstruction and cancelled reads release noncurrent heavy arrays."""
import gc
import weakref

import numpy as np
import pytest


def test_legacy_energy_batch_releases_previous_integrators(tmp_path, monkeypatch):
    from tests.core.test_browse_reference_rings import _saved_configuration, _selection
    from xrd_tools.analysis import angular_support as support
    from xrd_tools.integrate import calibration as calibration_module

    _frozen, saved = _saved_configuration(tmp_path, mode="Int 1D", one={
        "npt": 40, "method": "csr", "unit": "2th_deg", "radial_range": [2., 18.]})
    labels = (1, 2, 3, 4)
    basis = support.support_basis_from_saved(
        saved, _selection(labels, [16., 17., 18., 18.]), labels=labels, epochs=(labels,),
        mask=support.SavedSupportMask(signed=True), shape=(195, 487))
    axis = basis.integrator_for(1).integrate1d(
        np.ones((195, 487), np.float32), 40, unit="2th_deg", radial_range=(2., 18.), method="csr").radial
    first = weakref.ref(basis.integrator_for(1))
    created = []
    convert = calibration_module.detector_calibration_to_integrator

    def observed(*args, **kwargs):
        integrator = convert(*args, **kwargs)
        created.append(weakref.ref(integrator))
        return integrator

    monkeypatch.setattr(calibration_module, "detector_calibration_to_integrator", observed)
    monkeypatch.setattr(support, "_saved_support_basis", lambda request: (basis, axis))
    request = support.SavedSupportRequest("unused", "entry", None, labels, labels)
    profiles = support.reconstruct_saved_support(request)
    assert all(profile.available for profile in profiles.values())
    gc.collect()
    assert first() is None, "a completed wavelength must not retain its detector-sized integration caches"
    assert len(created) == 2 and created[0]() is None and created[1]() is not None
    np.testing.assert_array_equal(profiles[3].supported_deg, profiles[4].supported_deg)


def test_cancel_after_support_allocation_releases_the_bundle(tmp_path, monkeypatch):
    from tests.core.test_angular_support import _saved_int
    from xrd_tools.io.frame_view import FrameViewReader

    target, _ = _saved_int(tmp_path, frames=2)
    allocated = []
    read = FrameViewReader._read_support_row

    def observed(table, row, bundle):
        values = read(table, row, bundle)
        allocated.append(weakref.ref(values))
        return values

    monkeypatch.setattr(FrameViewReader, "_read_support_row", staticmethod(observed))
    with FrameViewReader(target) as reader:
        baseline = reader.retained_bytes
        with pytest.raises(InterruptedError):
            reader.read_1d_rows((1, 2), cancelled=lambda: bool(allocated))
        assert allocated, "cancel only after a real support allocation/read"
        gc.collect()
        assert all(ref() is None for ref in allocated)
        assert reader.retained_bytes == baseline
        assert reader.read(1).angular_support.available
