"""P3-1A detector-configuration science oracle."""
from __future__ import annotations
import builtins
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
import pyFAI
import pyFAI.detectors as detectors_module
import pyFAI.io.ponifile as ponifile_module
import pytest
from xrd_tools.integrate.calibration import load_detector_calibration
from xrd_tools.io import science_fingerprint
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import image_series_spec
from xdart.gui.tabs.scattering import output_preflight as preflight_module
from xdart.gui.tabs.scattering.adapters import dynamic_output, run_executor as executor_module
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor, _StandardRun
from xdart.gui.tabs.scattering.contracts import SourceCapture, StartCapture
from xdart.gui.tabs.scattering.events import RequestId, RunIdentity
from xdart.gui.tabs.scattering.output_preflight import OutputCandidate, _load_scientific_assets
_LIMIT = 1 << 20
_BASE = """poni_version: 2.1
Detector: Pilatus300kw
Detector_config: {config}
Distance: 0.1234
Poni1: 0.05
Poni2: 0.06
Rot1: 0.01
Rot2: 0.02
Rot3: 0.03
Wavelength: 1e-10
"""
class _ProbeComplete(Exception):
    pass
class _Bounded(BytesIO):
    def read(self, size: int = -1) -> bytes:
        assert 0 <= size <= _LIMIT + 1
        return super().read(size)
def _poni(config: str = '{"orientation":1}', **replacements: str) -> str:
    text = _BASE.format(config=config)
    for old, new in replacements.items():
        text = text.replace(old.replace("_", ": "), new)
    return text
def _intent(tmp_path: Path, text: str, name: str = "cal") -> tuple[RunIntent, Path]:
    path = tmp_path / f"{name}.poni"
    path.write_text(text, encoding="utf-8")
    raw = tmp_path / "raw_0001.tif"
    raw.write_bytes(b"raw")
    return RunIntent(source_spec=image_series_spec(raw), poni_file=str(path),
        save_path=str(tmp_path / f"{name}.nxs"), output_mode="Overwrite"), path
def _execute(intent: RunIntent, assets, monkeypatch, request: int = 1):
    configuration, observed = intent.freeze(), []
    class ProbeSource:
        def to_scan(self, *, poni, integrator, output_path):
            observed.append((poni, integrator)); raise _ProbeComplete
        def close(self): return None
    monkeypatch.setattr(executor_module, "open_source", lambda _source: ProbeSource())
    resources = None if assets is None else SimpleNamespace(
        admission=SimpleNamespace(scientific_assets=assets))
    capture = SourceCapture(RequestId(request), 1, configuration.thaw_source_spec())
    run = _StandardRun(configuration, RunIdentity.from_configuration(configuration),
        None, None, None, None, Path(configuration.save_path), capture=capture,
        resources=resources)
    with pytest.raises(_ProbeComplete):
        StandardRunExecutor()._construct(run)
    return observed[0]
def _signed(intent: RunIntent, assets):
    request = RequestId(20)
    capture = SourceCapture(request, 1, intent.source_spec)
    start = StartCapture(request, 1, RunIntentStore(intent).snapshot(), capture)
    candidate = OutputCandidate.from_start_capture(start, assets, ())
    return candidate._configuration, candidate.processing_mapping()
def test_p3_1a_admitted_orientation_reaches_standard_execution(tmp_path, monkeypatch):
    intent, _ = _intent(tmp_path, _poni())
    accepted = _load_scientific_assets(intent)
    observed = [int(_execute(intent, assets, monkeypatch, index)[1].detector.orientation)
                for index, assets in enumerate((accepted, None), 1)]
    assert observed == [1, 1]
def test_p3_1a_real_pyfai_2025_3_preserves_canonical_detector_config(tmp_path, monkeypatch):
    assert pyFAI.version == "2025.3.0"
    for index, config in enumerate(({"orientation": 1},
                                    {"max_shape": [194, 1475], "orientation": 3})):
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
        intent, path = _intent(tmp_path, _poni(canonical), f"real-{index}")
        direct, accepted = load_detector_calibration(path), _load_scientific_assets(intent)
        configuration, signed = _signed(intent, accepted)
        poni, integrator = _execute(intent, accepted, monkeypatch, index + 30)
        detector = integrator.detector
        assets = signed["accepted_scientific_assets"]
        assert set(assets) == {"poni_values", "poni_detector_config_json",
                               "poni_sha256", "mask_sha256"}
        assert accepted.poni_detector_config_json == canonical == assets["poni_detector_config_json"]
        assert direct.detector_config == accepted.detector_calibration.detector_config == config
        assert (poni.detector, type(detector).__name__, int(detector.orientation)) == (
            "Pilatus300kw", "Pilatus300kw", config["orientation"])
        assert tuple(detector.shape) == tuple(detector.max_shape) == tuple(
            config.get("max_shape", [195, 1475]))
        assert (detector.pixel1, detector.pixel2) == pytest.approx((172e-6, 172e-6))
        assert tuple(poni.to_dict().values())[:7] == pytest.approx(
            (0.1234, 0.05, 0.06, 0.01, 0.02, 0.03, 1e-10))
        assert dynamic_output._science_projection(configuration, signed)
def test_p3_1a_rejection_order_and_postconstruction_validation(tmp_path, monkeypatch):
    path = tmp_path / "invalid.poni"
    base = _poni()
    early = (base.replace("Pilatus300kw", "../Pilatus300kw"),
        base + "DISTANCE: 1\n", _poni('{"orientation":1,"orientation":1}'),
        base.replace("poni_version: 2.1\n", ""), base.replace("2.1", "1", 1),
        base.replace("2.1", "3", 1), base.replace("Detector_config: {\"orientation\":1}\n", ""),
        _poni("[]"), _poni('{"orientation":1,"unknown":2}'),
        _poni('{"orientation":1,"splineFile":null}'),
        _poni('{"orientation":1,"max_shape":"shape.edf"}'), "BROKEN\n" + base)
    def forbidden(self=None, pixel1=None, pixel2=None, max_shape=None, module_size=None, x_offset_file=None, y_offset_file=None, orientation=0): raise AssertionError("construction preceded raw refusal")
    with monkeypatch.context() as guard:
        guard.setattr(ponifile_module, "PoniFile", forbidden)
        guard.setattr(detectors_module, "detector_factory", forbidden)
        guard.setattr(detectors_module.Pilatus300kw, "__init__", forbidden)
        for text in early:
            path.write_text(text, encoding="utf-8")
            with pytest.raises(ValueError): load_detector_calibration(path)
        path.write_bytes(b"\xff")
        with pytest.raises(ValueError): load_detector_calibration(path)
    real, calls = ponifile_module.PoniFile, []
    def counted(*args, **kwargs):
        calls.append(1); return real(*args, **kwargs)
    monkeypatch.setattr(ponifile_module, "PoniFile", counted)
    post = (_poni('{"orientation":1,"pixel1":0.000172}'),
            base.replace("Distance: 0.1234", "Distance: 0"),
            base.replace("Rot1: 0.01", "Rot1: nan"),
            base.replace("Wavelength: 1e-10", "Wavelength: -1"))
    for index, text in enumerate(post, 1):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError): load_detector_calibration(path)
        assert len(calls) == index
def test_p3_1a_poni_qualification_is_bounded_and_stable(tmp_path, monkeypatch):
    oversized = _poni().encode() + b"#" * (_LIMIT + 1)
    with monkeypatch.context() as guard:
        guard.setattr(builtins, "open", lambda *_a, **_k: _Bounded(oversized))
        guard.setattr(ponifile_module, "PoniFile", lambda *_a, **_k: pytest.fail("parsed oversize"))
        with pytest.raises(ValueError): load_detector_calibration(tmp_path / "direct.poni")
    intent, path = _intent(tmp_path, _poni(), "outer")
    path.write_bytes(oversized)
    real_open = Path.open
    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", lambda self, *a, **k:
            _Bounded(oversized) if self == path else real_open(self, *a, **k))
        guard.setattr(preflight_module, "load_poni", lambda _path: pytest.fail("loaded oversize"))
        with pytest.raises(ValueError): _load_scientific_assets(intent)
    path.write_text(_poni(), encoding="utf-8")
    def changing(selected):
        value = load_detector_calibration(selected)
        selected.write_text(_poni('{"orientation":2}'), encoding="utf-8")
        return value
    monkeypatch.setattr(preflight_module, "load_poni", changing)
    with pytest.raises(ValueError, match="changed while admitted"):
        _load_scientific_assets(intent)
def test_p3_1a_signed_science_requires_exact_config_identity(tmp_path):
    intent, _ = _intent(tmp_path, _poni())
    accepted = _load_scientific_assets(intent)
    configuration, signed = _signed(intent, accepted)
    assets = signed["accepted_scientific_assets"]
    assert set(assets) == {"poni_values", "poni_detector_config_json",
                           "poni_sha256", "mask_sha256"}
    for changes in ({"poni_values": None}, {"poni_detector_config_json": None}):
        with pytest.raises(TypeError): replace(accepted, **changes)
    with pytest.raises(TypeError): replace(accepted, poni_detector_config_json='{"orientation": 1}')
    projection = dynamic_output._science_projection(configuration, signed)
    for key in ("poni_detector_config_json", "extra"):
        altered = json.loads(json.dumps(signed))
        if key == "extra": altered["accepted_scientific_assets"][key] = None
        else: altered["accepted_scientific_assets"].pop(key)
        with pytest.raises(TypeError): dynamic_output._science_projection(configuration, altered)
    altered = json.loads(json.dumps(signed))
    altered["accepted_scientific_assets"]["poni_detector_config_json"] = '{"orientation":2}'
    changed = dynamic_output._science_projection(configuration, altered)
    assert science_fingerprint(projection) != science_fingerprint(changed)
