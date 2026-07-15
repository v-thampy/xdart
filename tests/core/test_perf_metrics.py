"""Structure + harness-contract tests for the M0 benchmark (corrected 2026-07-14).

Two layers:

* schema/vocabulary structure tests (no wall-clock, no real data);
* **harness-contract tests** that drive the REAL ``run_once``/CLI seams on tiny
  synthetic HDF5 containers + a real PONI, pinning the honest-observation
  corrections (exact worker count, real write/finish/first-write events,
  categorized opens, destination safety).  Small spies wrap the harness-owned
  executor/sink/clock; the production reduction seam is never faked.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

from xrd_tools.perf.metrics import (
    OPEN_CATEGORIES,
    SCHEMA_VERSION,
    ContainerMetrics,
    EnvProvenance,
    RunMetrics,
    Timer,
    new_open_counts,
    summarize_runs,
    timed,
    write_json,
)

_HARNESS_PATH = (Path(__file__).resolve().parents[2]
                 / "scripts" / "nxs_directory_benchmark.py")

_DETECTOR = "pilatus100k"
_SHAPE = (195, 487)  # pyFAI pilatus100k


def _load_harness():
    spec = importlib.util.spec_from_file_location("nxs_directory_benchmark", _HARNESS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_raw_container(path: Path, nframes: int = 3, shape=_SHAPE):
    """A self-contained raw detector container (3-D stack, no external links)."""
    rng = np.random.default_rng(abs(hash(path.name)) % (2**32))
    data = (rng.random((nframes,) + tuple(shape)) * 100).astype(np.uint32)
    with h5py.File(path, "w") as f:
        e = f.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        det = e.create_group("instrument").create_group("detector")
        det.create_dataset("data", data=data)
    return path


def _write_poni(path: Path):
    from xrd_tools.core.containers import PONI
    from xrd_tools.integrate.calibration import save_poni
    save_poni(PONI(dist=0.15, poni1=0.017, poni2=0.017, wavelength=1e-10,
                   detector=_DETECTOR), path)
    return path


def _run_cli(harness, src_dir, poni, out_dir, *, cores=2, mode="1d", repeat=1,
             frame_limit=None, json_out=None, recursive=False, extra=()):
    argv = ["--source-dir", str(src_dir), "--poni", str(poni),
            "--output-dir", str(out_dir), "--mode", mode, "--cores", str(cores),
            "--repeat", str(repeat)]
    if frame_limit is not None:
        argv += ["--frame-limit", str(frame_limit)]
    if json_out is not None:
        argv += ["--json-out", str(json_out)]
    if recursive:
        argv += ["--recursive"]
    argv += list(extra)
    rc = harness.main(argv)
    data = json.loads(Path(json_out).read_text()) if (json_out and Path(json_out).exists()) else None
    return rc, data


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ===========================================================================
# schema / vocabulary structure (no wall-clock)
# ===========================================================================

def test_timer_and_timed_accumulate():
    with Timer() as t:
        time.sleep(0.001)
    assert isinstance(t.elapsed, float) and t.elapsed >= 0.0
    sink: dict[str, float] = {}
    with timed(sink, "p"):
        pass
    with timed(sink, "p"):
        pass
    assert sink["p"] >= 0.0


def test_env_provenance_reports_pyfai_h5py_hdf5():
    env = EnvProvenance.capture(git_sha="deadbeef")
    assert env.git_sha == "deadbeef"
    assert env.python and env.platform
    assert env.h5py and env.hdf5  # required provenance present in tests env


def test_open_categories_include_master_external_verification():
    for cat in ("source", "source_external", "output", "verification", "other", "failed"):
        assert cat in OPEN_CATEGORIES
    assert new_open_counts() == {c: 0 for c in OPEN_CATEGORIES}


def test_frames_submitted_default_is_none_not_zero():
    # F-NXS-9 honesty: an unobserved phase is None, never a misleading 0.
    cm = ContainerMetrics(path="/x.nxs")
    assert cm.frames_submitted is None
    rm = RunMetrics()
    assert rm.frames_submitted is None


def test_run_metrics_add_folds_observed_counts_and_opens():
    rm = RunMetrics(cores=4)
    a = ContainerMetrics(path="a.nxs", state="ready", frames_discovered=3,
                         frames_input=3, frames_reduced=3, frames_written=3,
                         frames_durable=3)
    a.open_counts.update(source=7, other=100, output=1, verification=2)
    rm.add(a)
    rm.add(ContainerMetrics(path="b.nxs", state="processed_output",
                            skip_reason="processed xdart output"))
    assert rm.frames_reduced == 3 and rm.frames_written == 3 and rm.frames_durable == 3
    assert rm.open_counts["source"] == 7 and rm.open_counts["other"] == 100
    assert rm.skipped_by_reason == {"processed xdart output": 1}


def test_run_metrics_to_dict_json_safe_and_documents_observation():
    rm = RunMetrics(mode="2d", cores=2, env=EnvProvenance.capture())
    rm.add(ContainerMetrics(path="a.nxs", frame_shape=(195, 487),
                            chunks=(1, 64, 64), dtype="uint32", frames_written=3))
    back = json.loads(json.dumps(rm.to_dict()))
    assert back["schema_version"] == SCHEMA_VERSION
    assert back["containers"][0]["frame_shape"] == [195, 487]
    # honesty is self-documenting in the JSON:
    assert "frames_submitted" in back["observation_sources"]
    assert "UNOBSERVED" in back["observation_sources"]["frames_submitted"]


def test_summarize_runs_median_and_source_opens():
    runs = []
    for total, opens in ((1.0, 10), (3.0, 12), (2.0, 11)):
        r = RunMetrics(cores=4)
        r.total_s = total
        r.open_counts["source"] = opens
        runs.append(r)
    s = summarize_runs(runs)
    assert s["timings"]["total_s"]["median"] == 2.0
    assert s["cores"] == 4
    assert s["source_opens_median"] == 11


def test_write_json_round_trips(tmp_path):
    out = write_json(tmp_path / "a" / "run.json",
                     {"runs": [RunMetrics(mode="1d").to_dict()]})
    assert json.loads(out.read_text())["runs"][0]["mode"] == "1d"


# ===========================================================================
# harness pure helpers
# ===========================================================================

def test_enumerate_matches_wrangler_suffix_rule(tmp_path):
    h = _load_harness()
    (tmp_path / "a_00001.nxs").write_bytes(b"")
    (tmp_path / "note.txt").write_bytes(b"")
    (tmp_path / "s_master.h5").write_bytes(b"")
    (tmp_path / "s_data.h5").write_bytes(b"")
    assert [p.name for p in h._enumerate(tmp_path, "nxs", False)] == ["a_00001.nxs"]
    assert [p.name for p in h._enumerate(tmp_path, "h5", False)] == ["s_master.h5"]


def test_build_plan_modes():
    h = _load_harness()
    assert h._build_plan("1d").integration_2d is None
    assert h._build_plan("2d").integration_1d is None
    both = h._build_plan("both")
    assert both.integration_1d is not None and both.integration_2d is not None
    with pytest.raises(SystemExit):
        h._build_plan("bogus")


def test_scan_name_strips_master(tmp_path):
    h = _load_harness()
    assert h._scan_name(Path("a/foo_master.h5")) == "foo"
    assert h._scan_name(Path("a/bar_00007.nxs")) == "bar_00007"


# ===========================================================================
# M0-R6 harness-contract tests (drive real seams)
# ===========================================================================

# --- 1. exact worker count; invalid cores fail before I/O -------------------

def test_cores_construct_exact_executor_sizes(tmp_path, monkeypatch):
    h = _load_harness()
    src = tmp_path / "src"; src.mkdir()
    _write_raw_container(src / "c_00001.nxs")
    poni = _write_poni(tmp_path / "cal.poni")

    seen: list[int] = []
    real_pool = h.ThreadPoolExecutor

    def _spy(*a, **k):
        seen.append(int(k.get("max_workers")))
        return real_pool(*a, **k)

    monkeypatch.setattr(h, "ThreadPoolExecutor", _spy)
    for cores in (1, 4):
        seen.clear()
        rc, _ = _run_cli(h, src, poni, tmp_path / f"out{cores}", cores=cores,
                         frame_limit=1, json_out=tmp_path / f"o{cores}.json")
        assert rc == 0
        assert seen == [cores], f"cores={cores} must build exactly {cores} workers, saw {seen}"


def test_invalid_cores_fail_before_io(tmp_path):
    h = _load_harness()
    src = tmp_path / "src"; src.mkdir()
    raw = _write_raw_container(src / "c_00001.nxs")
    poni = _write_poni(tmp_path / "cal.poni")
    before = _sha(raw)
    out = tmp_path / "out"
    rc = h.main(["--source-dir", str(src), "--poni", str(poni),
                 "--output-dir", str(out), "--cores", "0"])
    assert rc == 2
    assert not out.exists(), "no output dir may be created when cores is invalid"
    assert _sha(raw) == before


# --- 2. real reduction -> honest ordered observations -----------------------

def test_real_reduction_honest_lifecycle(tmp_path):
    h = _load_harness()
    src = tmp_path / "src"; src.mkdir()
    _write_raw_container(src / "scan_00001.nxs", nframes=3)
    poni = _write_poni(tmp_path / "cal.poni")
    rc, d = _run_cli(h, src, poni, tmp_path / "out", cores=2,
                     json_out=tmp_path / "o.json")
    assert rc == 0
    c = d["runs"][0]["containers"][0]
    assert c["frames_discovered"] == 3
    assert c["frames_input"] == 3
    assert c["frames_reduced"] == 3          # ReductionResult.n_processed
    assert c["frames_written"] == 3          # real sink.write() events
    assert c["frames_durable"] == 3          # on-disk after finish
    assert c["frames_submitted"] is None     # UNOBSERVED (not inferred/aliased)
    assert c["finish_s"] is not None and c["finish_s"] >= 0.0


# --- 3. sink failure / invalid container not reported written/durable -------

def test_invalid_container_not_reported_written_or_durable(tmp_path):
    h = _load_harness()
    src = tmp_path / "src"; src.mkdir()
    # wrong frame shape vs the PONI detector -> integration raises.
    _write_raw_container(src / "bad_00001.nxs", nframes=2, shape=(64, 64))
    poni = _write_poni(tmp_path / "cal.poni")
    rc, d = _run_cli(h, src, poni, tmp_path / "out", cores=1,
                     json_out=tmp_path / "o.json")
    assert rc == 0
    c = d["runs"][0]["containers"][0]
    assert c["state"] == "invalid"
    assert c["error"]
    assert c["frames_written"] == 0
    assert c["frames_durable"] == 0          # no inferred durability


# --- 4. session_finish_s measures the real finish interval ------------------

def test_measuring_sink_times_real_finish():
    h = _load_harness()

    class SlowSink:
        def begin(self, s, p): pass
        def write(self, f, r): pass
        def finish(self, result): time.sleep(0.05)

    ms = h.MeasuringSink(SlowSink())
    ms.begin(None, None)
    ms.write(type("F", (), {"index": 0})(), None)
    assert ms.first_write_ts is not None
    ms.finish(None)
    assert ms.finish_s is not None and ms.finish_s >= 0.05


def test_run_populates_session_finish_time(tmp_path):
    h = _load_harness()
    src = tmp_path / "src"; src.mkdir()
    _write_raw_container(src / "c_00001.nxs", nframes=2)
    poni = _write_poni(tmp_path / "cal.poni")
    _, d = _run_cli(h, src, poni, tmp_path / "out", cores=1, json_out=tmp_path / "o.json")
    assert d["runs"][0]["session_finish_total_s"] is not None
    assert d["runs"][0]["session_finish_total_s"] >= 0.0


# --- 5. first-frame timing: before enumeration, ends on first write ---------

def test_container_metrics_has_no_prewarm_first_frame_field():
    # the warm pre-read metric was removed; only run-level first-write latency.
    assert not hasattr(ContainerMetrics(path="x"), "first_frame_s")


def test_first_frame_latency_spans_enumeration_to_first_write(tmp_path):
    h = _load_harness()
    src = tmp_path / "src"; src.mkdir()
    _write_raw_container(src / "c_00001.nxs", nframes=2)
    poni = _write_poni(tmp_path / "cal.poni")
    _, d = _run_cli(h, src, poni, tmp_path / "out", cores=1, json_out=tmp_path / "o.json")
    r = d["runs"][0]
    c = r["containers"][0]
    ffl = r["first_frame_latency_s"]
    assert ffl is not None and ffl > 0.0
    # it must INCLUDE the pre-write phases (enumerate + probe + resolve + open +
    # metadata), i.e. it is not a warm decode measured after reduction started.
    pre = ((r["enumerate_s"] or 0) + (c["probe_s"] or 0)
           + (c["dataset_resolve_s"] or 0) + (c["open_s"] or 0)
           + (c["metadata_s"] or 0))
    assert ffl >= pre


# --- 6. open categorization + restore after exception -----------------------

def test_open_counter_categorizes_and_restores_on_exception(tmp_path):
    h = _load_harness()
    import h5py as _h5
    src = tmp_path / "src"; src.mkdir()
    master = _write_raw_container(src / "m_00001.nxs", nframes=1)
    out_root = tmp_path / "out"; out_root.mkdir()
    out_file = out_root / "gen.nxs"
    _write_raw_container(out_file, nframes=1)          # stand-in generated output
    bad = tmp_path / "bad.nxs"; bad.write_bytes(b"not hdf5")

    orig_init = _h5.File.__init__
    counter = h.H5FileCounter([master], out_root)
    with pytest.raises(OSError):
        with counter:
            with _h5.File(master, "r"):        # -> source (master)
                pass
            with _h5.File(out_file, "r"):      # -> output
                pass
            with counter.verification():
                with _h5.File(out_file, "r"):  # -> verification
                    pass
            _h5.File(bad, "r")                 # raises -> failed, then propagates
    assert counter.counts["source"] == 1
    assert counter.counts["output"] == 1
    assert counter.counts["verification"] == 1
    assert counter.counts["failed"] == 1
    # h5py.File.__init__ restored even though the block raised.
    assert _h5.File.__init__ is orig_init


# --- 7. --json-out destination safety (no byte changes) ---------------------

@pytest.mark.parametrize("target", ["source", "poni", "inside_source"])
def test_json_out_unsafe_targets_refused_without_byte_change(tmp_path, target):
    h = _load_harness()
    src = tmp_path / "src"; src.mkdir()
    raw = _write_raw_container(src / "c_00001.nxs")
    poni = _write_poni(tmp_path / "cal.poni")
    targets = {"source": raw, "poni": poni, "inside_source": src / "result.json"}
    json_out = targets[target]
    before = {p: _sha(p) for p in (raw, poni) if p.exists()}
    rc = h.main(["--source-dir", str(src), "--poni", str(poni),
                 "--output-dir", str(tmp_path / "out"),
                 "--json-out", str(json_out)])
    assert rc == 2, f"--json-out={target} must be refused"
    for p, sha in before.items():
        assert _sha(p) == sha, f"{p} bytes changed on refusal"


def test_output_dir_inside_source_refused(tmp_path):
    h = _load_harness()
    src = tmp_path / "src"; src.mkdir()
    _write_raw_container(src / "c_00001.nxs")
    poni = _write_poni(tmp_path / "cal.poni")
    rc = h.main(["--source-dir", str(src), "--poni", str(poni),
                 "--output-dir", str(src / "out")])  # inside source tree
    assert rc == 2


# --- 8. existing output + duplicate scan stems fail before reduction --------

def test_duplicate_scan_stems_rejected_before_reduction(tmp_path):
    h = _load_harness()
    src = tmp_path / "src"; src.mkdir()
    (src / "a").mkdir(); (src / "b").mkdir()
    _write_raw_container(src / "a" / "dup_00001.nxs")
    _write_raw_container(src / "b" / "dup_00001.nxs")  # same canonical stem
    poni = _write_poni(tmp_path / "cal.poni")
    rc, d = _run_cli(h, src, poni, tmp_path / "out", cores=1, recursive=True,
                     json_out=tmp_path / "o.json")
    assert rc == 0
    states = {c["scan_name"]: c["state"] for c in d["runs"][0]["containers"]}
    reasons = [c["skip_reason"] for c in d["runs"][0]["containers"]]
    assert all(s == "invalid" for s in states.values())
    assert all(r == "duplicate canonical scan stem" for r in reasons)
    assert d["runs"][0]["frames_written"] == 0


def test_existing_output_rejected(tmp_path):
    h = _load_harness()
    src = tmp_path / "src"; src.mkdir()
    master = _write_raw_container(src / "scan_00001.nxs")
    poni = _write_poni(tmp_path / "cal.poni")
    out_root = tmp_path / "out"
    repeat_dir = out_root / "repeat_00"
    repeat_dir.mkdir(parents=True)
    (repeat_dir / "scan_00001.nxs").write_bytes(b"pre-existing")  # would be clobbered
    plan = h._build_plan("1d")
    from xrd_tools.integrate.calibration import load_poni
    cm = h._bench_container(master, load_poni(poni), repeat_dir, out_root, plan,
                            1, "entry", None, src, [str(master)], False)
    assert cm.state == "invalid"
    assert "already exists" in (cm.skip_reason or "")


# --- 9. repeated runs use unique destinations -------------------------------

def test_repeats_use_unique_destinations(tmp_path):
    h = _load_harness()
    src = tmp_path / "src"; src.mkdir()
    _write_raw_container(src / "scan_00001.nxs", nframes=1)
    poni = _write_poni(tmp_path / "cal.poni")
    out_root = tmp_path / "out"
    rc, d = _run_cli(h, src, poni, out_root, cores=1, repeat=2,
                     json_out=tmp_path / "o.json")
    assert rc == 0
    assert (out_root / "repeat_00" / "scan_00001.nxs").exists()
    assert (out_root / "repeat_01" / "scan_00001.nxs").exists()
    # both repeats produced durable output; neither overwrote the other.
    assert all(run["frames_durable"] == 1 for run in d["runs"])


# --- 10. partial invalid run: valid prior output + truthful invalid report --

def test_partial_run_writes_valid_and_reports_invalid_truthfully(tmp_path):
    h = _load_harness()
    src = tmp_path / "src"; src.mkdir()
    _write_raw_container(src / "good_00001.nxs", nframes=2)          # valid
    _write_raw_container(src / "zz_bad_00001.nxs", nframes=2, shape=(64, 64))  # invalid, sorts last
    poni = _write_poni(tmp_path / "cal.poni")
    out_root = tmp_path / "out"
    rc, d = _run_cli(h, src, poni, out_root, cores=1, json_out=tmp_path / "o.json")
    assert rc == 0
    conts = {c["scan_name"]: c for c in d["runs"][0]["containers"]}
    good, bad = conts["good_00001"], conts["zz_bad_00001"]
    assert good["state"] == "ready" and good["frames_durable"] == 2
    assert (out_root / "repeat_00" / "good_00001.nxs").exists()
    assert bad["state"] == "invalid"
    assert bad["frames_written"] == 0 and bad["frames_durable"] == 0
    assert d["runs"][0]["frames_durable"] == 2   # only the valid container's
