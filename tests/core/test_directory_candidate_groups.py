"""A Folder's scans are grouped in one pass, with the same groups as before (NEXT-1215-SHARE-LOOKUPS).

``directory_candidate_groups`` rescanned the whole plan for every scan, running the adapter's scan-name parse on every
candidate each time; admission groups the plan twice, so a Folder of 327 saved scans and 930 images spent about 6 s
there on the beamline PC (maintainer, 2026-11-19: "the initial start is still pretty slow"; his pick "Dedupe per-output
checks (Recommended)"). The reference below is the 1.2.15 algorithm, run on the same real plan."""
from pathlib import Path

import numpy as np
import fabio

from xrd_tools.core.scan import SourceKind
from xrd_tools.sources import adapters as adapter_registry
from xrd_tools.sources import run_plan
from xrd_tools.sources.directory_session import DirectoryIndexSession
from xrd_tools.sources.run_plan import RunCandidatePlan, directory_candidate_groups


def _reference(plan):
    """1.2.15's ``directory_candidate_groups``, verbatim."""
    from xrd_tools.sources.adapters import get_adapter
    groups = []
    consumed = set()
    for candidate in plan.candidates:
        if candidate.path in consumed:
            continue
        adapter = get_adapter(candidate.adapter_id)
        if adapter is None:
            raise ValueError(f"directory candidate lost adapter {candidate.adapter_id!r}")
        if SourceKind.IMAGE_FILE in adapter.kinds:
            name = adapter.scan_name(candidate.path)
            members = tuple(value for value in plan.candidates
                            if (value.adapter_id == candidate.adapter_id and value.path.parent == candidate.path.parent
                                and adapter.scan_name(value.path) == name))
        else:
            members = (candidate,)
        consumed.update(value.path for value in members)
        groups.append(members)
    return tuple(groups)


def _image(path: Path, value: int = 1) -> None:
    fabio.tifimage.TifImage(data=np.full((8, 8), value, dtype=np.uint16)).write(str(path))


def _plan(root: Path, suffixes) -> RunCandidatePlan:
    session = DirectoryIndexSession(probe_candidates=False)
    try:
        session.configure(root, recursive=True, suffixes=suffixes)
        return RunCandidatePlan.from_snapshot(session.observe(refresh=True).discovered_snapshot)
    finally:
        session.close()


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "samples"
    for folder in ("", "calibration", "sampleA", "sampleB"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    names = ("Lab6_0001", "Lab6_0002", "lab6_0001", "LAB6_0003", "scan-0001", "scan-0002", "scan_0001", "odd", "a_b_0007",
             "a_b_0008", "x10s_11191151_0001", "x10s_11191151_0002")
    for index, name in enumerate(names):
        for folder in ("", "calibration", "sampleA"):
            _image(root / folder / f"{name}.tif", index + 1)
            (root / folder / f"{name}.txt").write_text("i0 1\n")
    _image(root / "sampleB" / "late_0001.tif")
    (root / "notes.txt").write_text("not a frame")
    return root


def test_the_groups_and_their_order_are_unchanged(tmp_path):
    root = _tree(tmp_path)
    for suffixes in ((".tif",), (".tif", ".txt")):
        plan = _plan(root, suffixes)
        assert len(plan.candidates) > 30
        assert directory_candidate_groups(plan) == _reference(plan)


def test_each_candidate_is_named_once(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    plan = _plan(root, (".tif",))
    calls = []
    real_get = adapter_registry.get_adapter

    def counting_get(adapter_id):
        adapter = real_get(adapter_id)
        if adapter is None or SourceKind.IMAGE_FILE not in adapter.kinds:
            return adapter
        real_name = adapter.scan_name

        class Counted:
            def __getattr__(self, name):
                return getattr(adapter, name)

            def scan_name(self, path):
                calls.append(path)
                return real_name(path)
        return Counted()

    monkeypatch.setattr(adapter_registry, "get_adapter", counting_get)
    groups = directory_candidate_groups(plan)
    assert sum(len(group) for group in groups) == len(plan.candidates)
    # The 1.2.15 loop parsed every candidate once per scan: hundreds of times here.
    assert len(calls) == len(plan.candidates), (len(calls), len(plan.candidates))
    assert run_plan.directory_candidate_groups is directory_candidate_groups
