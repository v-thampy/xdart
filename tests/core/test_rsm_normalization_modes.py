"""RSM's four normalization choices (AUD-9 stage 2b G8b; decision 4, the maintainer 2026-10-01: "There are
monitors both before and after the foil. So add the 4th choice"): None, Monitor (a monitor after the foils,
which already counts their attenuation), Foil transmission × exposure, and Monitor × foil transmission (a
monitor before the foils). Every divisor is the shared beam divisor's (``frame_divisors``); the two original
modes keep their canonical value, so their fingerprints do not move."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from xrd_tools.analysis.module_transaction import MetadataColumnSelector
from xrd_tools.analysis.rsm_operation import (
    RSMNormalizationMode,
    RSMNormalizationPolicy,
    RSMOperationRefused,
    required_rsm_selectors_v2,
    rsm_normalization_divisors,
)
from xrd_tools.corrections.normalization import foil_transmission

_LENGTHS = (1.06, 3.04, 4.65, 9.5)
_FOIL = MetadataColumnSelector("foil status", 0)
_SECONDS = MetadataColumnSelector("Seconds", 0)
_MONITOR = MetadataColumnSelector("Seconds", 1)


def _policy(mode: RSMNormalizationMode) -> RSMNormalizationPolicy:
    if mode is RSMNormalizationMode.IDENTITY:
        return RSMNormalizationPolicy.identity()
    if mode is RSMNormalizationMode.FOIL_TRANSMISSION_EXPOSURE:
        return RSMNormalizationPolicy(mode, _FOIL, _SECONDS, _LENGTHS)
    if mode is RSMNormalizationMode.MONITOR:
        return RSMNormalizationPolicy(mode, None, None, (0.0, 0.0, 0.0, 0.0), monitor_selector=_MONITOR)
    return RSMNormalizationPolicy(mode, _FOIL, None, _LENGTHS, monitor_selector=_MONITOR)


def test_each_rsm_normalization_mode_divides_by_its_known_divisor():
    foil = np.array([0.0, 1101.0, 9999.0, 37.0])
    exposure = np.array([1.0, 2.0, 0.25, 3.0])
    monitor = np.array([1.0e5, 2.5e5, 3.0e4, 7.0e5])
    transmission = foil_transmission(foil, _LENGTHS)
    expected = {
        RSMNormalizationMode.IDENTITY: np.ones(4),
        RSMNormalizationMode.FOIL_TRANSMISSION_EXPOSURE: transmission * exposure,
        RSMNormalizationMode.MONITOR: monitor,
        RSMNormalizationMode.MONITOR_FOIL_TRANSMISSION: monitor * transmission,
    }
    arrays = {
        RSMNormalizationMode.IDENTITY: {},
        RSMNormalizationMode.FOIL_TRANSMISSION_EXPOSURE: {"foil_status": foil, "exposure_seconds": exposure},
        RSMNormalizationMode.MONITOR: {"monitor_counts": monitor},
        RSMNormalizationMode.MONITOR_FOIL_TRANSMISSION: {"foil_status": foil, "monitor_counts": monitor},
    }
    for mode, want in expected.items():
        got = rsm_normalization_divisors(_policy(mode), 4, **arrays[mode])
        assert got.tobytes() == np.asarray(want, dtype=np.float64).tobytes(), mode
        assert not got.flags.writeable
    with pytest.raises(RSMOperationRefused) as refused:  # a monitor mode without its counts
        rsm_normalization_divisors(_policy(RSMNormalizationMode.MONITOR), 4)
    assert refused.value.code == "NORMALIZATION_METADATA_MISSING" and "monitor" in str(refused.value)
    with pytest.raises(RSMOperationRefused) as refused:
        rsm_normalization_divisors(_policy(RSMNormalizationMode.MONITOR_FOIL_TRANSMISSION), 4,
                                   monitor_counts=monitor)
    assert refused.value.code == "NORMALIZATION_METADATA_MISSING" and "foil" in str(refused.value)
    with pytest.raises(ValueError, match="takes only its own"):  # an exposure is never combined with a monitor
        rsm_normalization_divisors(_policy(RSMNormalizationMode.MONITOR), 4, monitor_counts=monitor,
                                   exposure_seconds=exposure)
    with pytest.raises(RSMOperationRefused) as refused:  # the shared refusals, unchanged
        rsm_normalization_divisors(_policy(RSMNormalizationMode.MONITOR), 4,
                                   monitor_counts=np.array([1.0, 0.0, 1.0, 1.0]))
    assert refused.value.code == "INVALID_MONITOR_VALUE"


def test_rsm_normalization_policy_admits_each_mode_only_with_its_columns():
    zero = (0.0, 0.0, 0.0, 0.0)
    for mode in RSMNormalizationMode:
        policy = _policy(mode)
        assert policy.selectors == tuple(item for item in (policy.foil_selector, policy.exposure_selector,
                                                           policy.monitor_selector) if item is not None)
    # the original modes' canonical value is unchanged (four entries); a monitor mode adds its selector
    assert len(_policy(RSMNormalizationMode.IDENTITY)._canonical_value()) == 4
    assert len(_policy(RSMNormalizationMode.FOIL_TRANSMISSION_EXPOSURE)._canonical_value()) == 4
    assert len(_policy(RSMNormalizationMode.MONITOR)._canonical_value()) == 5
    assert len(_policy(RSMNormalizationMode.MONITOR_FOIL_TRANSMISSION)._canonical_value()) == 5
    refusals = (
        (RSMNormalizationMode.MONITOR, None, None, zero, None, TypeError),          # no monitor
        (RSMNormalizationMode.MONITOR, _FOIL, None, zero, _MONITOR, ValueError),     # a foil column
        (RSMNormalizationMode.MONITOR, None, _SECONDS, zero, _MONITOR, ValueError),  # an exposure
        (RSMNormalizationMode.MONITOR, None, None, _LENGTHS, _MONITOR, ValueError),  # lengths
        (RSMNormalizationMode.MONITOR_FOIL_TRANSMISSION, None, None, _LENGTHS, _MONITOR, TypeError),
        (RSMNormalizationMode.MONITOR_FOIL_TRANSMISSION, _FOIL, _SECONDS, _LENGTHS, _MONITOR, ValueError),
        (RSMNormalizationMode.MONITOR_FOIL_TRANSMISSION, _FOIL, None, zero, _MONITOR, ValueError),
        (RSMNormalizationMode.MONITOR_FOIL_TRANSMISSION, _FOIL, None, _LENGTHS, _FOIL, ValueError),  # same column
        (RSMNormalizationMode.FOIL_TRANSMISSION_EXPOSURE, _FOIL, _SECONDS, _LENGTHS, _MONITOR, ValueError),
        (RSMNormalizationMode.IDENTITY, None, None, zero, _MONITOR, ValueError),
    )
    for mode, foil, exposure, lengths, monitor, error in refusals:
        with pytest.raises(error):
            RSMNormalizationPolicy(mode, foil, exposure, lengths, monitor_selector=monitor)
    motors = (("eta", MetadataColumnSelector("eta", 0)),)
    assert _MONITOR in required_rsm_selectors_v2(motors, _policy(RSMNormalizationMode.MONITOR))


def test_a_monitor_before_the_foils_grids_as_the_foils_with_its_counts_as_the_exposure(tmp_path, monkeypatch):
    """Run-level known answer on the synthetic fixture (``Seconds`` twice, ``foil status``): Monitor × foil
    transmission with the second ``Seconds`` as the monitor divides each frame by exactly what Foil transmission ×
    exposure does with that column as the exposure, so the two volumes are byte-identical; Monitor alone divides
    by the counts (the Preview's divisor range is theirs) and is its own identity."""
    import xrd_tools.analysis.rsm_operation as rsm_operation
    from xrd_tools.analysis.rsm_values import prepare_rsm_tool_v2
    from xrd_tools.analysis.rsm_operation import run_rsm_operation_v2
    from xrd_tools.sources.spec import SpecSource
    from tests.core.test_rsm_operation_r2 import _form, _write_member

    monkeypatch.setattr(rsm_operation, "_resolve_exact_rsm_q_bounds_active",
                        lambda *_args, **_kwargs: ((-8.0, 8.0), (-8.0, 8.0), (-8.0, 8.0)))
    monkeypatch.setattr(SpecSource, "load_frame",
                        lambda self, index: np.full((195, 487), 10.0 + int(index), dtype=np.float64))
    members = (_write_member(tmp_path, 0),)
    base = _form(tmp_path, members)
    seconds_1 = MetadataColumnSelector("Seconds", 1)
    foil_exposure = RSMNormalizationPolicy(RSMNormalizationMode.FOIL_TRANSMISSION_EXPOSURE, _FOIL, seconds_1,
                                           _LENGTHS)
    monitor_foil = RSMNormalizationPolicy(RSMNormalizationMode.MONITOR_FOIL_TRANSMISSION, _FOIL, None, _LENGTHS,
                                          monitor_selector=seconds_1)
    monitor = RSMNormalizationPolicy(RSMNormalizationMode.MONITOR, None, None, (0.0, 0.0, 0.0, 0.0),
                                     monitor_selector=seconds_1)
    results, fingerprints = {}, {}
    for name, policy in (("foil_exposure", foil_exposure), ("monitor_foil", monitor_foil), ("monitor", monitor)):
        preflight = prepare_rsm_tool_v2(replace(base, normalization=policy,
                                                output_path=tmp_path / "output" / f"{name}.nexus"))
        if name == "monitor":
            assert preflight.summary.normalization_divisor_range == (10.0, 20.0)
        fingerprints[name] = preflight.request.module.fingerprint
        results[name] = run_rsm_operation_v2(preflight.request).payload
    assert len(set(fingerprints.values())) == 3
    np.testing.assert_array_equal(results["monitor_foil"].intensity, results["foil_exposure"].intensity)
    assert results["monitor_foil"].intensity.tobytes() == results["foil_exposure"].intensity.tobytes()
    assert not np.array_equal(results["monitor"].intensity, results["foil_exposure"].intensity,
                              equal_nan=True)
