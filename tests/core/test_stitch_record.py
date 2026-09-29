"""Stitched results are Int records (maintainer, 2026-09-29): Int's reader reads I(q) and the GI cut;
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
        assert stored.gi is None and stored.wavelength_A == WAVELENGTH and stored.frames_json == manifest
        for name in ("axis", "intensity", "sigma", "coverage", "normalization"):
            np.testing.assert_array_equal(getattr(stored.pattern, name), _cast(getattr(pattern, name)))
        with h5py.File(path, "r") as handle:
            entry = handle["entry"]
            assert list(entry["frames"]) == ["frame_0001"]  # Int's frames: the result row only
            assert sorted(entry["stitch/frames"]) == ["frame_0000", "frame_0001", "frame_0002"]
            assert "stitched_1d" not in entry
    finally:
        session.abort()


@pytest.mark.parametrize("cut, unit", [("q_ip", "qip_A^-1"), ("q_oop", "qoop_A^-1")])
def test_a_gi_stitch_writes_its_cut_as_the_rows_extra_mode(tmp_path, cut, unit):
    pattern, gi = _pattern(50, "q_A^-1", seed=2), _pattern(30, unit, seed=3)
    session, path = _write(tmp_path, pattern, gi=(cut, gi))  # no wavelength: the record has no header
    try:
        primary = read_frame_view(path, 1, mode_1d="q_total")
        np.testing.assert_array_equal(primary.intensity_1d, _cast(pattern.intensity))
        view = read_frame_view(path, 1, mode_1d=cut)
        assert view.axis_1d.unit == unit
        np.testing.assert_array_equal(view.axis_1d.values, _cast(gi.axis))
        np.testing.assert_array_equal(view.intensity_1d, _cast(gi.intensity))
        stored = read_stitch_record(path)
        assert stored.gi[0] == cut and stored.wavelength_A is None
        with h5py.File(path, "r") as handle:
            assert {"stitch", "multi_result_1d"} <= detect_capabilities(handle["entry"])
            assert CAPABILITIES["stitch"].marker == STITCH_GROUP
        for name in ("axis", "intensity", "coverage", "normalization"):
            np.testing.assert_array_equal(getattr(stored.gi[1], name), _cast(getattr(gi, name)))
            np.testing.assert_array_equal(getattr(stored.pattern, name), _cast(getattr(pattern, name)))
    finally:
        session.abort()


def test_a_stitched_record_takes_only_the_q_ip_and_q_oop_cuts(tmp_path):
    with pytest.raises(ValueError, match="q_ip"):
        _write(tmp_path, _pattern(10, "q_A^-1", seed=4), gi=("exit_angle", _pattern(10, "exit_angle_deg", seed=5)))


def test_the_gi_cut_folds_into_the_result_fingerprint_only_when_present():
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, project_analysis_artifact_result

    pattern, cut = _pattern(20, "q_A^-1", seed=6), _pattern(12, "qip_A^-1", seed=7)

    def fingerprint(gi_cut=None, **plain):
        return project_analysis_artifact_result(
            kind=AnalysisArtifactKind.STITCH_1D, axes=(("q", pattern.axis),), axis_units=(("q", pattern.unit),),
            intensity=pattern.intensity, sigma=None, coverage=pattern.coverage,
            normalization=pattern.normalization, **({} if gi_cut is None else {"gi_cut": gi_cut}), **plain,
        ).result_fingerprint

    changed = cut.intensity.copy()
    changed[np.flatnonzero(np.isfinite(changed))[0]] += 1.
    cuts = [("q_ip", cut.unit, cut.axis, cut.intensity, cut.coverage, cut.normalization),
            ("q_ip", cut.unit, cut.axis, changed, cut.coverage, cut.normalization),
            ("q_oop", cut.unit, cut.axis, cut.intensity, cut.coverage, cut.normalization)]
    assert len({fingerprint(), *(fingerprint(value) for value in cuts)}) == 4
    assert fingerprint() == fingerprint(gi_cut=None)  # no cut: the plain result's own fingerprint
