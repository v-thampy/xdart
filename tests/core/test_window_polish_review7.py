"""R7 regressions frozen against the pushed item-14 object."""
import pytest


@pytest.mark.parametrize("phase", ("bg_film", "bgx_film"))
def test_declared_phase_names_win_over_background_prefixes(phase):
    from xdart.gui.analysis.peak_fit_util import group_families
    grouped = group_families(
        [f"frac_{phase}", f"{phase}_a", f"{phase}_gamma", "bg_c0", "bgx_c1"],
        phase=True, phase_names=(phase,))
    assert grouped["a"] == [(f"{phase}_a", phase)]
    assert grouped["gamma"] == [(f"{phase}_gamma", phase)]
    assert grouped["frac"] == [(f"frac_{phase}", phase)]
    assert grouped["bg_c0"] == [("bg_c0", None)]
    assert grouped["bgx_c1"] == [("bgx_c1", None)]
