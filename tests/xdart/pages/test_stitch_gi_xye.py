"""Saved Stitch GI q_ip/q_oop XYE companions (RC-GI-EXPORT-INSTALL-20261016; LT-003's pre-release half).

Frozen before the product change. A GI Stitch result's saved cuts (``payload.gi``) are exported beside
its NeXus result as ``iqip_``/``iqoop_`` companions, named by the existing unit prefix. I(q)'s companion
keeps its name and bytes. At this RC a saved GI cut carries no sigma (``stitch_record`` refuses one), so
a GI companion's third column is Int's sqrt(abs(I)) compatibility estimate, labelled in its header.
Every companion target is checked before any is written; then I(q), q_ip and q_oop are written in that
order, each atomically, and a later failure may leave earlier companions updated. The Stitch owner
reports a failure as XYE_EXPORT_FAILED without revoking the committed result. Real saved results only;
the multigeometry backend is enough, because the export reads the saved projection whatever made it.
"""
from dataclasses import replace
import os
from pathlib import Path

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.stitch.page import StitchPage
from xdart.gui.tools.stitch_values import prepare_stitch_tool
from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.stitch_operation import run_stitch_operation
from xrd_tools.integrate.stitch_gi import GI_UNITS, StitchGI
from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, read_analysis_artifact
from xrd_tools.io.export import read_xye, write_stitch_xye, write_xye
from xrd_tools.session.display_logic import xye_unit_from_filename
from tests.xdart.pages.test_stitch_page import _Status, _wait
from tests.xdart.tools.test_stitch_native_gi import _SAMPLE, _gi_form
from tests.xdart.tools.test_stitch_native_values import _native_form

_PREFIX = {"q_ip": "iqip", "q_oop": "iqoop"}
_HEADER = (
    "# xdart Stitch GI cut {mode} saved in {nexus}",
    "# columns: {mode} ({unit}), intensity, sigma",
    "# sigma: estimated as sqrt(abs(intensity)) for compatibility, not a propagated uncertainty",
)


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _form(tmp_path, cuts, mode="1d"):
    form = (_native_form(tmp_path, "multigeometry", scans="1") if cuts is None else _gi_form(
        tmp_path, "multigeometry", StitchGI(cuts, incidence_motor="eta", tilt_motor="chi", sample_motors=_SAMPLE)))
    return form if mode == "1d" else replace(form, mode="2d", npt_rad_2d=24, npt_azim_2d=16)


def _committed(tmp_path, cuts, mode="1d"):
    request = prepare_stitch_tool(_form(tmp_path, cuts, mode)).request
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    target = Path(request.module.output.target)
    return target, read_analysis_artifact(target)


def _companion(target, prefix):
    return target.with_name(f"{prefix}_{target.stem}.xye")


def _assert_iq_unchanged(target, payload, tmp_path):
    """I(q)'s companion is byte-for-byte what the unchanged writer makes of the saved profile."""
    profile = payload if payload.kind is AnalysisArtifactKind.STITCH_1D else payload.profile
    reference = tmp_path / "reference" / "iq.xye"
    write_xye(reference, profile.axes[0][1], profile.intensity, profile.sigma, atomic=True)
    assert _companion(target, "iq").read_bytes() == reference.read_bytes()


def _assert_gi_companions(target, payload, modes=None):
    """Each saved cut's companion: its own axis, unit, intensity and the labelled estimate, exactly."""
    cuts = [(mode, cut) for mode, cut in payload.gi if modes is None or mode in modes]
    assert cuts
    for mode, cut in cuts:
        path = _companion(target, _PREFIX[mode])
        ((_axis, unit),) = cut.axis_units
        assert unit == GI_UNITS[mode] == xye_unit_from_filename(path.name)
        expected = [line.format(mode=mode, nexus=target.name, unit=unit) for line in _HEADER]
        lines = path.read_text().splitlines()
        assert lines[:3] == expected and [line for line in lines if line.startswith("#")] == expected
        x, y, sigma = read_xye(path)
        np.testing.assert_array_equal(x.astype("<f4"), cut.axes[0][1])
        np.testing.assert_array_equal(y.astype("<f4"), cut.intensity)  # NaN where saved, nowhere else
        assert cut.sigma is None  # a saved GI cut carries no sigma at this RC
        np.testing.assert_allclose(sigma, np.sqrt(np.abs(np.asarray(cut.intensity, dtype=float))),
                                   rtol=5e-9, atol=0, equal_nan=True)


def _xye_names(target):
    return {path.name for path in target.parent.glob("*.xye")}


@pytest.mark.parametrize("mode", ("1d", "2d"))
def test_g1_a_gi_stitch_exports_both_saved_cuts_beside_its_unchanged_i_of_q(tmp_path, mode):
    target, payload = _committed(tmp_path, ("q_ip", "q_oop"), mode)
    assert target.name == f"sample_scan1_gi_stitch{mode}.nexus"
    assert [cut_mode for cut_mode, _cut in payload.gi] == ["q_ip", "q_oop"]
    exported = write_stitch_xye(payload)
    assert exported == _companion(target, "iq")
    assert _xye_names(target) == {f"{prefix}_{target.stem}.xye" for prefix in ("iq", "iqip", "iqoop")}
    _assert_iq_unchanged(target, payload, tmp_path)
    _assert_gi_companions(target, payload)
    if mode == "1d":  # Int's 1D Viewer reads a labelled companion as read_xye does
        from xrd_tools.session.hydration import HydrationOutcome
        from tests.core.test_viewer_1d import _load, _mode_arrays
        path = _companion(target, "iqip")
        _reader, transport, port, completion = _load([path])
        assert completion.outcome is HydrationOutcome.HYDRATED
        ((coordinate, intensity, uncertainty),) = _mode_arrays(port)
        for actual, expected in zip((coordinate, intensity, uncertainty), read_xye(path), strict=True):
            assert np.array_equal(actual, expected, equal_nan=True)
        del coordinate, intensity, uncertainty, actual, expected
        assert port.holder.release("done") and transport.retire(join_timeout=1)


@pytest.mark.parametrize("cuts", (("q_ip",), ("q_oop",), None), ids=("q_ip-only", "q_oop-only", "standard"))
def test_g2_only_saved_cuts_get_companions_and_other_families_stay(tmp_path, cuts):
    target, payload = _committed(tmp_path, cuts)
    assert target.name == ("sample_scan1_stitch1d.nexus" if cuts is None else "sample_scan1_gi_stitch1d.nexus")
    assert [mode for mode, _cut in payload.gi] == list(cuts or ())
    # the other measurement family's Stitch companion, and an Int GI XYE family file
    others = (target.with_name("iqip_sample_scan1_gi_stitch1d.xye" if cuts is None else "iq_sample_scan1_stitch1d.xye"),
              target.parent / "sample_gi" / "iqip_sample_gi_0001.xye")
    for path in others:
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"1\t2\t3\n")
    exported = write_stitch_xye(payload)
    assert exported == _companion(target, "iq")
    assert _xye_names(target) == {exported.name, others[0].name} | {
        f"{_PREFIX[mode]}_{target.stem}.xye" for mode in cuts or ()}
    assert all(path.read_bytes() == b"1\t2\t3\n" for path in others)
    _assert_iq_unchanged(target, payload, tmp_path)
    if cuts:
        _assert_gi_companions(target, payload)


@pytest.mark.parametrize("atomic", (False, True))
def test_g3_a_header_keeps_supplied_sigma_and_the_default_bytes(tmp_path, atomic):
    x, y, sigma = [1., 2., 3.], [4., np.nan, -6.], [0., np.nan, .25]
    plain, labelled = tmp_path / "plain.xye", tmp_path / "labelled.xye"
    write_xye(plain, x, y, sigma, atomic=atomic)
    write_xye(labelled, x, y, sigma, atomic=atomic, header="first line\nsecond line")
    assert labelled.read_text() == "# first line\n# second line\n" + plain.read_text()
    for actual, expected in zip(read_xye(labelled), (x, y, sigma), strict=True):
        np.testing.assert_array_equal(actual, expected)  # supplied sigma written unchanged, NaN kept


@pytest.mark.parametrize("kind", ("symlink", "nexus-link", "raw-link"))
def test_g4_an_unsafe_gi_target_refuses_the_family_before_any_write(tmp_path, kind):
    target, payload = _committed(tmp_path, ("q_ip", "q_oop"))
    nexus_before = target.read_bytes()
    if kind == "symlink":
        guarded = tmp_path / "decoy.xye"
        guarded.write_bytes(b"decoy\n")
        planted, match = _companion(target, "iqip"), "symbolic link"
        planted.symlink_to(guarded)
    elif kind == "nexus-link":
        guarded, planted, match = target, _companion(target, "iqoop"), "same file as input"
        os.link(target, planted)
    else:
        from xrd_tools.io.read import resolve_source_master
        record = payload.contributing_frames[0]
        source = record.get("source_constituents", (record,))[0]
        guarded = resolve_source_master(source["source_path"], scan_file=target, source_base=str(tmp_path))
        assert guarded is not None
        planted, match = _companion(target, "iqip"), "same file as input"
        os.link(guarded, planted)
    before = guarded.read_bytes()
    with pytest.raises(ValueError, match=match):
        write_stitch_xye(payload)
    assert _xye_names(target) == {planted.name}  # nothing written: every target is checked first
    assert guarded.read_bytes() == before and target.read_bytes() == nexus_before


def test_g5_a_later_companion_failure_raises_after_earlier_ones_publish(tmp_path):
    target, payload = _committed(tmp_path, ("q_ip", "q_oop"))
    nexus_before = target.read_bytes()
    blocked = _companion(target, "iqoop")
    blocked.mkdir()
    with pytest.raises(OSError):
        write_stitch_xye(payload)
    assert blocked.is_dir() and not any(blocked.iterdir())
    _assert_iq_unchanged(target, payload, tmp_path)
    _assert_gi_companions(target, payload, modes={"q_ip"})
    assert [path.name for path in target.parent.iterdir() if path.name.endswith(".tmp")] == []
    assert target.read_bytes() == nexus_before


def test_g6_the_stitch_page_run_and_append_export_saved_cuts_and_report_a_failure(qapp, tmp_path):
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.resize(1400, 850)
        page.show()
        page.load_form(_form(tmp_path, ("q_ip", "q_oop")))
        _wait(page, qapp)
        page.run_bar.startButton.click()
        _wait(page, qapp, timeout=60)
        payload = page._current_payload()
        assert payload is not None, page.run_bar.readinessLabel.full_text()
        target = Path(payload.inspection.path)
        assert target.name == "sample_scan1_gi_stitch1d.nexus"
        iq, ip, oop = (_companion(target, prefix) for prefix in ("iq", "iqip", "iqoop"))
        assert iq.is_file() and ip.is_file() and oop.is_file(), page.run_bar.readinessLabel.full_text()
        _assert_iq_unchanged(target, payload, tmp_path)
        _assert_gi_companions(target, payload)
        nexus_before = target.stat().st_ino, target.stat().st_mtime_ns, target.read_bytes()
        ip.unlink()  # Append backfills a missing GI companion without repeating scientific publication
        page.run_bar.startButton.click()
        _wait(page, qapp, timeout=60)
        assert ip.is_file() and page.run_bar.readinessLabel.full_text().startswith("Up to date")
        assert (target.stat().st_ino, target.stat().st_mtime_ns, target.read_bytes()) == nexus_before
        _assert_gi_companions(target, payload)
        oop.unlink()
        oop.mkdir()  # a real later failure is reported; the committed science and display stay
        page.run_bar.startButton.click()
        _wait(page, qapp)
        assert "XYE_EXPORT_FAILED" in page.run_bar.readinessLabel.full_text()
        assert page._current_payload() is not None
        assert (target.stat().st_ino, target.stat().st_mtime_ns, target.read_bytes()) == nexus_before
        assert iq.is_file() and ip.is_file() and oop.is_dir()
        oop.rmdir()
    finally:
        _wait(page, qapp)
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()
