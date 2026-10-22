"""LIVE-QUALIFY-REFUSAL-20261012 frozen headless oracle (Claude, October 12): C1-C5.

A metadata file that changes while a source qualification is reading it is a typed source-revision change, so the
live owner can retry it at its next poll; everything else the reader refuses stays what it was.

C1 and C2 fail at 1cfccd74: the change escapes `qualify_source_execution_graph` as a bare OSError. C3 and C4 are
discriminators that pass there and must keep passing: an unrelated OSError and an oversized bounded read stay
terminal. C5 fails there: Average's graph requalification reports the in-read change as AVERAGE_PREPARATION_FAILED.
"""
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import xrd_tools.io.metadata as metadata
from tests.core.test_p36_average import _series, _txt, module
from xrd_tools.core.scan import SourceSpec
from xrd_tools.io.metadata import MetadataInputTooLarge
from xrd_tools.reduction import AverageScanRecipe, Integration1DPlan, ReductionPlan
from xrd_tools.sources import execution_graph
from xrd_tools.sources.execution_graph import SourceRevisionChanged, qualify_source_execution_graph


def _one_image(tmp_path):
    """One TIFF with a .txt sidecar (I0 = 2.0), as a series source that reads that sidecar."""
    source = _series(tmp_path, (np.ones((2, 2), dtype=np.uint16),))
    image = Path(source.options["files"][0])
    _txt(image, counters=(("I0", 2.0),))
    return SourceSpec(source.uri, source.kind, options={**dict(source.options), "metadata_format": "txt"}), image


def test_c1_a_sidecar_rewritten_inside_a_qualification_read_is_a_source_revision_change(tmp_path, monkeypatch):
    source, image = _one_image(tmp_path)
    sidecar, real, written = image.with_suffix(".txt"), metadata.read_txt_metadata, []

    def rewrite_while_reading(path, *args, **kwargs):
        if not written:  # a non-atomic writer's second piece lands while the parser reads
            written.append(Path(path))
            with open(sidecar, "a") as stream:
                stream.write("late = 1\n")
        return real(path, *args, **kwargs)
    monkeypatch.setattr(metadata, "read_txt_metadata", rewrite_while_reading)
    with pytest.raises(SourceRevisionChanged, match="metadata source changed during read"):
        qualify_source_execution_graph(source)
    assert written == [sidecar]


def test_c2_directory_churn_through_every_read_is_a_source_revision_change(tmp_path, monkeypatch):
    source, image = _one_image(tmp_path)
    real, arrivals = metadata._find_sidecar, []

    def churn(path, ext):
        time.sleep(.02)  # one clock tick after the reader recorded the directory
        arrivals.append(image.parent / f"other_{len(arrivals):04d}.dat")
        arrivals[-1].write_bytes(b"x")
        return real(path, ext)
    monkeypatch.setattr(metadata, "_find_sidecar", churn)
    with pytest.raises(SourceRevisionChanged, match="metadata discovery changed during read"):
        qualify_source_execution_graph(source)
    assert len(arrivals) == 5, "M1 reads five times before it gives up"


def test_c3_an_unrelated_os_error_inside_a_read_stays_terminal(tmp_path, monkeypatch):
    source, _image = _one_image(tmp_path)

    def refuse(path, *_args, **_kwargs):
        raise PermissionError(f"permission denied (injected): {path}")
    monkeypatch.setattr(metadata, "read_txt_metadata", refuse)
    with pytest.raises(PermissionError, match="permission denied") as error:
        qualify_source_execution_graph(source)
    assert not isinstance(error.value, SourceRevisionChanged)


def test_c4_an_oversized_bounded_read_stays_terminal(tmp_path):
    source, image = _one_image(tmp_path)
    sidecar = image.with_suffix(".txt")
    sidecar.write_text(sidecar.read_text() + "# padding\n" + "x" * (1 << 16))
    with pytest.raises(MetadataInputTooLarge) as error:
        qualify_source_execution_graph(source, reader_binding="average_closed_v1")
    assert not isinstance(error.value, SourceRevisionChanged)


def test_c5_average_requalification_refuses_an_in_read_change_as_source_drift(tmp_path, monkeypatch):
    source, image = _one_image(tmp_path)
    sidecar = image.with_suffix(".txt")
    original = sidecar.read_bytes()
    armed, written, adopted = threading.local(), [], []
    real_requalify, real_read, real_motors = (
        module.requalify_source_execution_graph, metadata.read_txt_metadata, execution_graph._finite_motors)

    def requalify(*args, **kwargs):
        armed.on = True
        try:
            return real_requalify(*args, **kwargs)
        finally:
            armed.on = False

    def read(path, *args, **kwargs):
        if getattr(armed, "on", False) and not written:
            written.append(Path(path))  # lands after the bounded snapshot, before the reader's check
            sidecar.write_bytes(original.replace(b"I0 = 2.0", b"I0 = 9.0"))
        return real_read(path, *args, **kwargs)
    monkeypatch.setattr(module, "requalify_source_execution_graph", requalify)
    monkeypatch.setattr(metadata, "read_txt_metadata", read)
    monkeypatch.setattr(execution_graph, "_finite_motors",
                        lambda values: adopted.append(dict(values)) or real_motors(values))
    before = set(tmp_path.iterdir())
    runner = module.AverageScanRunner(AverageScanRecipe(
        source, tmp_path / "average.nxs", ReductionPlan(integration_1d=Integration1DPlan(npt=2)),
        numeric_metadata_keys=("I0",)))
    runner._execute_graph = lambda _graph: runner.plan
    refusal = runner.start()
    assert written == [sidecar], "the change landed inside a graph requalification read"
    assert (refusal.disposition, refusal.diagnostic_code) == ("REFUSED", "AVERAGE_SOURCE_DRIFT"), refusal.diagnostic
    assert runner.close() is refusal
    assert all(values.get("I0") != 9.0 for values in adopted)
    assert set(tmp_path.iterdir()) == before, "no output was written"
