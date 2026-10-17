"""The one Corrections row group (G8c unit 4c), Qt-free: its rows in order, as Stitch's and RSM's Advanced lay them
out, and a page's own sentence on a tip."""
import pytest

from xdart.gui.tools import correction_values as corrections


def test_the_group_declares_its_five_rows_once_in_order():
    rows = corrections.correction_rows()
    assert [(row.key, row.label, row.value, row.choices, row.browse) for row in rows] == [
        ("solid_angle", "Solid angle", "On", ("On", "Off"), False),
        ("polarization", "Polarization", "0.99", (), False),
        ("dark_path", "Dark", "", (), True),
        ("flat_path", "Flat field", "", (), True),
        ("flat_cutoff", "Flat cutoff (% of median)", "10", (), False)]
    assert tuple(row.tip for row in rows) == tuple(corrections.TIPS[key] for key in corrections.CORRECTION_KEYS)
    noted = corrections.correction_rows({"solid_angle": "Keep on."})
    assert noted[0].tip == corrections.TIPS["solid_angle"] + " Keep on." and noted[1:] == rows[1:]
    with pytest.raises(KeyError, match="no Corrections row error_model"):
        corrections.correction_rows({"error_model": "Stitch's own"})
