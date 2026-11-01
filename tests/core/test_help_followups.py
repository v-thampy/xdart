"""NEXT-128-HELP-FOLLOWUPS (headless rows F3, F8): a NeXus template reads only its own 1-D mode, and every Phase
Analysis profile's amplitude is its area. Frozen before the product change (DESIGN-HELP-FOLLOWUPS-128.md; red on
72050876).

- F3, Codex's optional hardening from NEXT-128-PEAK-FIT REVIEW-4: "read_1d_rows currently reads other 1-D modes for
  the same frame before the loader selects one". The maintainer picked "Template read hardening".
- F8, the maintainer's "Fix both here (Recommended)": the phase engine says "All profiles are area-normalised ... makes
  phase fractions directly interpretable", but Pearson VII's area is amplitude/sqrt(2^(1/m) - 1), and Moffat's
  depends on its width.
"""
import numpy as np
import pytest
from scipy.integrate import quad

GI_MODES = ("q_total", "q_ip", "q_oop", "exit_angle", "chi_gi")


@pytest.fixture
def gi_record(tmp_path):
    from tests.core._v2_record_fixture import write_gi_reference_scan
    path = tmp_path / "gi.nexus"
    write_gi_reference_scan(path, tmp_path / "raw")
    return path


def _reader(path):
    from xrd_tools.io.frame_view import FrameViewReader
    return FrameViewReader(path, include_thumbnail=False, resolve_source=False)


# ---- F3 a template reads only its own 1-D mode --------------------------------------------------------------------

def test_f3_read_1d_rows_reads_only_the_requested_modes(gi_record):
    with _reader(gi_record) as reader:
        everything = reader.read_1d_rows((0,))
        only = reader.read_1d_rows((0,), modes=("q_ip",))
    assert tuple(mode.mode for mode in everything.modes) == GI_MODES
    assert tuple(mode.mode for mode in only.modes) == ("q_ip",)
    assert only.primary_mode is None  # q_total, the primary mode, was not asked for
    q_ip = next(mode for mode in everything.modes if mode.mode == "q_ip")
    np.testing.assert_array_equal(only.modes[0].intensity_rows[0], q_ip.intensity_rows[0])
    np.testing.assert_array_equal(only.modes[0].axis.values, q_ip.axis.values)


@pytest.mark.parametrize("modes, error", ((["q_ip"], TypeError), ((), TypeError), (("q_ip", 3), TypeError),
                                          (("q_total", "nope"), ValueError)))
def test_f3_a_bad_mode_selection_is_refused(gi_record, modes, error):
    with _reader(gi_record) as reader:
        assert tuple(mode.mode for mode in reader.read_1d_rows((0,), modes=("q_oop",)).modes) == ("q_oop",)
        with pytest.raises(error):
            reader.read_1d_rows((0,), modes=modes)


def test_f3_a_nexus_template_asks_only_for_its_own_mode(gi_record, monkeypatch):
    from xrd_tools.io.frame_view import FrameViewReader
    from xrd_tools.analysis.background_template import BackgroundTemplateSource, prepare_background_template
    with _reader(gi_record) as reader:
        expected = next(mode for mode in reader.read_1d_rows((0,)).modes if mode.mode == "q_ip")
    calls = []
    real = FrameViewReader.read_1d_rows

    def recording(self, labels, **kwargs):  # the real reader, its keyword arguments noted
        calls.append(kwargs.get("modes"))
        return real(self, labels, **kwargs)

    monkeypatch.setattr(FrameViewReader, "read_1d_rows", recording)
    prepared = prepare_background_template(BackgroundTemplateSource("nexus", str(gi_record), frame=0, mode="q_ip"))
    assert calls == [("q_ip",)]
    order = np.argsort(np.asarray(expected.axis.values, dtype=float), kind="stable")
    np.testing.assert_array_equal(prepared.y, np.asarray(expected.intensity_rows[0], dtype=float)[order])


# ---- F8 every phase profile's amplitude is its area ---------------------------------------------------------------

#: shapes beyond each profile's defaults, for the two profiles whose area depended on them
_EXTRA_SHAPES = {"pearson7": ({"expo": 3.0}, {"expo": 10.0}), "moffat": ({"moffat_beta": 1.5}, {"moffat_beta": 4.0})}


@pytest.mark.parametrize("sigma", (0.02, 0.08))
@pytest.mark.parametrize("name", sorted(__import__("xrd_tools.analysis.fitting.phase_fitting",
                                                   fromlist=["_PHASE_PROFILE_SPEC"])._PHASE_PROFILE_SPEC))
def test_f8_every_phase_profile_integrates_to_its_amplitude(name, sigma):
    from xrd_tools.analysis.fitting.phase_fitting import _PHASE_PROFILE_SPEC
    spec = _PHASE_PROFILE_SPEC[name]
    kw_map = dict(spec.get("kw_map", {}))
    for shape in (dict(spec.get("defaults", {})), *_EXTRA_SHAPES.get(name, ())):
        kwargs = {kw_map.get(key, key): value for key, value in shape.items()}

        def profile(x):
            return float(spec["func"](np.array([x]), np.array([0.0]), np.array([2.5]), np.array([sigma]),
                                      **kwargs)[0])

        area = quad(profile, -np.inf, 0.0, limit=800)[0] + quad(profile, 0.0, np.inf, limit=800)[0]
        assert area == pytest.approx(2.5, rel=1e-6), (name, shape, sigma, area)
