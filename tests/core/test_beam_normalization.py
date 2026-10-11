"""The shared beam divisor (AUD-9 stage 2b C1; maintainer, 2026-10-01: "The one feature that needs to be added
to both the stitch and rsm pages is applying the filter/absorber/foil intensity correction. This is there in the
example RSM.ipynb in the example_notebooks.").

``xrd_tools.corrections.normalization.frame_divisors`` is the one owner of each frame's monitor, foil
transmission and exposure divisor; RSM's foil/exposure divisors are its, byte for byte. The oracle is the
notebook's own ``foil_to_factor`` (``example_notebooks/RSM/RSM_process.ipynb``, cell 4), copied verbatim.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrd_tools.corrections.normalization import (
    BeamNormalizationRefused,
    foil_transmission,
    frame_divisors,
)

#: The notebook's two absorption-length sets (cells 9 and 12): they depend on the energy.
_LENGTH_SETS = ((1.06, 3.04, 4.64, 13.22), (1.06, 3.04, 4.65, 9.5))


# --- copied verbatim from example_notebooks/RSM/RSM_process.ipynb, cell 4 ---
def foil_to_factor(foil_status, abs_lengths):
    factor = np.array(abs_lengths)

    results = []
    for number in foil_status:
        # Check if the number is a four-digit number
        if not number <= 9999:
            raise ValueError(f"{number} is not a four-digit number")

        # Create a NumPy array from the string
        digit_array = np.array([int(digit) for digit in str(int(number)).zfill(4)])
        # print(number, digit_array)
        # Perform dot product
        dot_product = np.exp(-np.dot(factor, digit_array))
        results.append(dot_product)
    return np.array(results)
# --- end of the notebook's code ---


@pytest.mark.parametrize("lengths", _LENGTH_SETS)
def test_the_foil_transmission_is_the_notebooks_for_every_status_code(lengths) -> None:
    """Every code from 0 to 9999 (each digit's every multiplicity, 0, 1234 and 9999 among them), both length
    sets: the shared transmission equals the notebook's ``foil_to_factor`` bit for bit."""
    codes = np.arange(10000, dtype=np.float64)
    ours = foil_transmission(codes, lengths)
    np.testing.assert_array_equal(ours, foil_to_factor(codes, lengths))
    assert ours[0] == 1.0  # no foil
    assert ours[1234] == np.exp(-(1 * lengths[0] + 2 * lengths[1] + 3 * lengths[2] + 4 * lengths[3]))
    # each foil alone at nine, thousands first: exact on every platform, unlike a sum of all four, whose rounding
    # follows the platform's dot product; 9999 itself is the notebook's above, bit for bit (maintainer, 2026-10-11)
    assert [ours[9000], ours[900], ours[90], ours[9]] == [np.exp(-9 * length) for length in lengths]


@pytest.mark.parametrize("code, notebook_refuses", [
    (10000.0, True),   # above four digits: both refuse
    (-1.0, True),      # negative: the notebook fails on the minus sign
    (1.5, False),      # non-integral: the notebook truncates it to 1; the shared owner refuses
    (np.nan, True),    # non-finite: the notebook fails converting it
])
def test_a_status_code_outside_four_whole_digits_is_refused(code, notebook_refuses) -> None:
    lengths = _LENGTH_SETS[1]
    with pytest.raises(BeamNormalizationRefused) as refused:
        foil_transmission(np.array([code]), lengths)
    assert refused.value.code == "FOIL_STATUS_INVALID"
    if notebook_refuses:
        with pytest.raises(ValueError):
            foil_to_factor(np.array([code]), lengths)
    else:
        assert foil_to_factor(np.array([code]), lengths)[0] == foil_to_factor(np.array([1.0]), lengths)[0]


def test_each_choice_divides_by_the_product_of_its_factors() -> None:
    """Decision 4's four choices (maintainer, 2026-10-01: "There are monitors both before and after the foil. So
    add the 4th choice"): None, Monitor (one after the foils), Foil transmission × exposure, and Monitor × foil
    transmission (one before the foils)."""
    lengths = _LENGTH_SETS[1]
    monitor = np.array([2.0, 5.0, 11.0])
    foil = np.array([0.0, 1.0, 1203.0])
    exposure = np.array([1.0, 0.5, 2.0])
    transmission = foil_to_factor(foil, lengths)
    np.testing.assert_array_equal(frame_divisors(3), np.ones(3))
    np.testing.assert_array_equal(frame_divisors(3, monitor=monitor), monitor)
    np.testing.assert_array_equal(
        frame_divisors(3, foil=foil, exposure=exposure, absorption_lengths=lengths), transmission * exposure)
    np.testing.assert_array_equal(
        frame_divisors(3, monitor=monitor, foil=foil, absorption_lengths=lengths), monitor * transmission)
    result = frame_divisors(3, foil=foil, exposure=exposure, absorption_lengths=lengths)
    assert not result.flags.writeable and result.dtype == np.float64


def test_an_exposure_is_never_combined_with_a_monitor_and_lengths_go_with_the_foil() -> None:
    with pytest.raises(ValueError, match="never combined with a monitor"):
        frame_divisors(1, monitor=[1.0], exposure=[1.0])
    with pytest.raises(ValueError, match="together"):
        frame_divisors(1, foil=[0.0], exposure=[1.0])
    with pytest.raises(ValueError, match="together"):
        frame_divisors(1, exposure=[1.0], absorption_lengths=_LENGTH_SETS[1])


@pytest.mark.parametrize("kwargs, code, message", [
    (dict(foil=[0.0, 0.0], exposure=[1.0]), "NORMALIZATION_METADATA_SHAPE_INVALID",
     "normalization metadata does not match frame membership"),
    (dict(foil=[0.5, 0.0], exposure=[0.0, 1.0]), "FOIL_STATUS_INVALID",
     "foil status must be finite integral values"),
    (dict(foil=[12345.0, 0.0], exposure=[1.0, 1.0]), "FOIL_STATUS_INVALID",
     "foil status must be between 0 and 9999"),
    (dict(foil=[0.0, 0.0], exposure=[1.0, 0.0]), "EXPOSURE_INVALID",
     "exposure seconds must be finite and strictly positive"),
    (dict(foil=[0.0, 0.0], monitor=[1.0, -1.0]), "INVALID_MONITOR_VALUE",
     "monitor values must be finite and strictly positive"),
])
def test_the_refusals_keep_rsms_codes_messages_and_order(kwargs, code, message) -> None:
    """Shape, then foil status, then exposure, then monitor: RSM's order and messages (rows 2 and 4 also carry a
    later fault, which must not be the one reported)."""
    with pytest.raises(BeamNormalizationRefused) as refused:
        frame_divisors(2, absorption_lengths=_LENGTH_SETS[1], **kwargs)
    assert (refused.value.code, str(refused.value)) == (code, message)


def test_a_vanishing_transmission_or_divisor_is_refused_with_its_factors_named() -> None:
    """A transmission that underflows to zero is TRANSMISSION_INVALID; a product that underflows is
    NORMALIZATION_DIVISOR_INVALID, naming its factors (RSM's "foil/exposure" message unchanged)."""
    with pytest.raises(BeamNormalizationRefused) as refused:
        frame_divisors(1, foil=[9000.0], exposure=[1.0], absorption_lengths=(1000.0, 0.0, 0.0, 0.0))
    assert (refused.value.code, str(refused.value)) == (
        "TRANSMISSION_INVALID", "foil transmission is non-finite or non-positive")
    for kwargs, names in ((dict(exposure=[1e-300]), "foil/exposure"), (dict(monitor=[1e-300]), "monitor/foil")):
        with pytest.raises(BeamNormalizationRefused) as refused:
            frame_divisors(1, foil=[1000.0], absorption_lengths=(300.0, 0.0, 0.0, 0.0), **kwargs)
        assert (refused.value.code, str(refused.value)) == (
            "NORMALIZATION_DIVISOR_INVALID", f"{names} divisor is non-finite or non-positive")


def test_rsm_delegates_its_foil_exposure_divisors_byte_for_byte_with_its_own_refusals() -> None:
    """RSM's ``rsm_normalization_divisors`` is the shared function's: the same bytes as the notebook's
    transmission times the exposure (its former expression, ``exp(-(digits @ lengths)) * exposure``, adds the
    terms in another order on x86 Linux), its refusals as ``RSMOperationRefused`` with the same codes and
    messages, and its policy-specific missing-metadata refusal kept."""
    from xrd_tools.analysis.module_transaction import MetadataColumnSelector
    from xrd_tools.analysis.rsm_operation import (
        RSMNormalizationMode, RSMNormalizationPolicy, RSMOperationRefused, rsm_normalization_divisors,
    )
    lengths = _LENGTH_SETS[1]
    policy = RSMNormalizationPolicy(RSMNormalizationMode.FOIL_TRANSMISSION_EXPOSURE,
                                    MetadataColumnSelector("foil status", 0), MetadataColumnSelector("Seconds", 0),
                                    lengths)
    foil = np.array([0.0, 1101.0, 9999.0, 37.0])
    exposure = np.array([1.0, 2.0, 0.25, 3.0])
    notebook = foil_to_factor(foil, lengths) * exposure
    result = rsm_normalization_divisors(policy, 4, foil_status=foil, exposure_seconds=exposure)
    assert result.tobytes() == notebook.tobytes() and not result.flags.writeable
    with pytest.raises(RSMOperationRefused) as refused:
        rsm_normalization_divisors(policy, 4, foil_status=np.array([0.0, 0.0, 0.0, 10000.0]),
                                   exposure_seconds=exposure)
    assert (refused.value.code, str(refused.value)) == ("FOIL_STATUS_INVALID",
                                                        "foil status must be between 0 and 9999")
    with pytest.raises(RSMOperationRefused) as refused:
        rsm_normalization_divisors(policy, 4, foil_status=foil, exposure_seconds=None)
    assert refused.value.code == "NORMALIZATION_METADATA_MISSING"
    np.testing.assert_array_equal(
        rsm_normalization_divisors(RSMNormalizationPolicy.identity(), 3), np.ones(3))
