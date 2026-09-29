"""Stitched results are Int records (maintainer, 2026-09-29): Int's reader reads I(q) and the GI cuts;
``entry/stitch`` keeps coverage, normalization and the contributing frames."""
import json

import h5py
import numpy as np
import pytest

from xrd_tools.io.finite_artifact import _FinitePublicationSession
from xrd_tools.io.frame_view import read_frame_view
from xrd_tools.io.nexus import open_nexus_writer
from xrd_tools.io.processed_scan_id import is_current_processed_xdart_path
from xrd_tools.io.read import ProcessedScan
from xrd_tools.io.schema import CAPABILITIES, detect_capabilities
from xrd_tools.io.stitch_record import STITCH_GROUP, StitchedPattern, read_stitch_record, write_stitch_record

WAVELENGTH = .729


def _pattern(n, unit, *, seed, sigma=False):
    rng = np.random.default_rng(seed)
    coverage = rng.integers(0, 40, n).astype(float)
    normalization = coverage * rng.uniform(.5, 2., n)
    intensity = np.where(coverage > 0, rng.uniform(1., 9., n), np.nan)
    return StitchedPattern(np.linspace(-2. if unit.startswith("qip") else .5, 5., n), unit, intensity,
                           rng.uniform(.1, .3, n) if sigma else None, coverage, normalization)


def _write(tmp_path, pattern, **kwargs):
    """The stitched record's candidate, written as the Stitch output owner writes it."""
    target = tmp_path / "stitch.nexus"
    session = _FinitePublicationSession(target)
    document = session.start(lambda path: open_nexus_writer(path, entry="entry", compression="gzip",
                                                            overwrite=True))
    try:
        write_stitch_record(document["entry"], target, pattern, **kwargs)
        assert session.close_document()
    except BaseException:
        session.abort()
        raise
    return session, session.candidate


def _cast(values):
    return np.asarray(values, dtype="<f4").astype(float)


@pytest.mark.parametrize("unit", ["q_A^-1", "2th_deg"])
def test_a_stitched_pattern_is_an_int_record_row_with_its_diagnostics_and_frames(tmp_path, unit):
    pattern = _pattern(50, unit, seed=1, sigma=True)
    records = [{"frame_index": i, "source_path": f"/raw/f{i}.raw", "source_frame_index": 0} for i in range(3)]
    manifest = json.dumps({"records": [{"frame_index": i} for i in range(3)]})
    session, path = _write(tmp_path, pattern, wavelength_A=WAVELENGTH, frame_records=records,
                           frames_json=manifest, source_base="/raw")
    try:
        assert is_current_processed_xdart_path(path)
        assert ProcessedScan(path).frame_indices == [1]
        view = read_frame_view(path, 1)
        assert view.axis_1d.unit == unit
        np.testing.assert_array_equal(view.axis_1d.values, _cast(pattern.axis))
        np.testing.assert_array_equal(view.intensity_1d, _cast(pattern.intensity))
        np.testing.assert_array_equal(view.sigma_1d, _cast(pattern.sigma))
        stored = read_stitch_record(path)
        assert stored.gi == () and stored.wavelength_A == WAVELENGTH and stored.frames_json == manifest
        for name in ("axis", "intensity", "sigma", "coverage", "normalization"):
            np.testing.assert_array_equal(getattr(stored.pattern, name), _cast(getattr(pattern, name)))
        with h5py.File(path, "r") as handle:
            entry = handle["entry"]
            assert list(entry["frames"]) == ["frame_0001"]  # Int's frames: the result row only
            assert sorted(entry["stitch/frames"]) == ["frame_0000", "frame_0001", "frame_0002"]
            assert "stitched_1d" not in entry
    finally:
        session.abort()


@pytest.mark.parametrize("cuts", [("q_ip",), ("q_oop",), ("q_oop", "q_ip")])
def test_a_gi_stitch_writes_its_cuts_as_the_rows_extra_modes(tmp_path, cuts):
    """Qip, Qoop or Qip + Qoop (maintainer, 2026-09-29): each cut is a mode of the row beside I(q), read
    back in one order whatever the order given."""
    units = {"q_ip": "qip_A^-1", "q_oop": "qoop_A^-1"}
    pattern = _pattern(50, "q_A^-1", seed=2)
    gi = {cut: _pattern(30 + 5 * index, units[cut], seed=3 + index) for index, cut in enumerate(cuts)}
    session, path = _write(tmp_path, pattern, gi=tuple(gi.items()))  # no wavelength: the record has no header
    try:
        primary = read_frame_view(path, 1, mode_1d="q_total")
        np.testing.assert_array_equal(primary.intensity_1d, _cast(pattern.intensity))
        stored = read_stitch_record(path)
        assert [mode for mode, _value in stored.gi] == [cut for cut in ("q_ip", "q_oop") if cut in gi]
        assert stored.wavelength_A is None
        with h5py.File(path, "r") as handle:
            assert {"stitch", "multi_result_1d"} <= detect_capabilities(handle["entry"])
            assert CAPABILITIES["stitch"].marker == STITCH_GROUP
        for name in ("axis", "intensity", "coverage", "normalization"):
            np.testing.assert_array_equal(getattr(stored.pattern, name), _cast(getattr(pattern, name)))
        for cut, value in stored.gi:
            view = read_frame_view(path, 1, mode_1d=cut)
            assert view.axis_1d.unit == units[cut]
            np.testing.assert_array_equal(view.axis_1d.values, _cast(gi[cut].axis))
            np.testing.assert_array_equal(view.intensity_1d, _cast(gi[cut].intensity))
            for name in ("axis", "intensity", "coverage", "normalization"):
                np.testing.assert_array_equal(getattr(value, name), _cast(getattr(gi[cut], name)))
    finally:
        session.abort()


@pytest.mark.parametrize("gi", [(("exit_angle", "exit_angle_deg"),), (("q_ip", "qip_A^-1"), ("q_ip", "qip_A^-1"))])
def test_a_stitched_record_takes_only_distinct_q_ip_and_q_oop_cuts(tmp_path, gi):
    with pytest.raises(ValueError, match="q_ip"):
        _write(tmp_path, _pattern(10, "q_A^-1", seed=4),
               gi=tuple((mode, _pattern(10, unit, seed=5)) for mode, unit in gi))


def test_the_gi_cuts_fold_into_the_result_fingerprint_in_one_order_only_when_present():
    from xrd_tools.io.analysis_artifact import (
        AnalysisArtifactKind, AnalysisArtifactProjectionInvalid, project_analysis_artifact_result,
    )

    pattern, ip, oop = _pattern(20, "q_A^-1", seed=6), _pattern(12, "qip_A^-1", seed=7), _pattern(9, "qoop_A^-1", seed=8)

    def fingerprint(*gi_cuts, **plain):
        return project_analysis_artifact_result(
            kind=AnalysisArtifactKind.STITCH_1D, axes=(("q", pattern.axis),), axis_units=(("q", pattern.unit),),
            intensity=pattern.intensity, sigma=None, coverage=pattern.coverage,
            normalization=pattern.normalization, **({"gi_cuts": gi_cuts} if gi_cuts else {}), **plain,
        ).result_fingerprint

    def cut(mode, value, intensity=None):
        return (mode, value.unit, value.axis, value.intensity if intensity is None else intensity,
                value.coverage, value.normalization)

    changed = oop.intensity.copy()
    changed[np.flatnonzero(np.isfinite(changed))[0]] += 1.
    fingerprints = {fingerprint(), fingerprint(cut("q_ip", ip)), fingerprint(cut("q_oop", oop)),
                    fingerprint(cut("q_oop", oop, changed)), fingerprint(cut("q_ip", ip), cut("q_oop", oop)),
                    fingerprint(cut("q_ip", ip), cut("q_oop", oop, changed))}
    assert len(fingerprints) == 6
    assert fingerprint() == fingerprint(gi_cuts=())  # no cut: the plain result's own fingerprint
    with pytest.raises(AnalysisArtifactProjectionInvalid, match="order"):
        fingerprint(cut("q_oop", oop), cut("q_ip", ip))
