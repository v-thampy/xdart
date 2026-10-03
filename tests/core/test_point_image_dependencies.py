"""Analysis receipts cover every used exposure and its native dependencies."""
from dataclasses import replace

import h5py
import pytest

from xrd_tools.analysis.scan_operations import (
    AnalysisDisposition, AnalysisSourceLeaseRefused, MetadataTablePlan,
    requalified_analysis_source, run_metadata_table,
)
from xrd_tools.sources.point_images import PointImagePolicy
from xrd_tools.core.scan import SourceKind
from tests.core.test_point_images import _confirmed, _eiger, _spec_images


@pytest.mark.parametrize("operation,indices", [("average", (0, 1, 2, 3)), ("select", (1, 3))])
def test_receipt_freezes_all_used_point_images(tmp_path, operation, indices):
    spec = _confirmed(_spec_images(tmp_path), PointImagePolicy(2, operation, image_index=2))
    result = run_metadata_table(MetadataTablePlan(spec))
    assert result.disposition is AnalysisDisposition.COMPLETED, result.diagnostics
    dependencies = {path.name for path, _resolved, _revision in result.receipt.dependency_revisions}
    assert {f"calibrant_scan001_{index:04}.tif" for index in indices} <= dependencies
    changed = tmp_path / f"calibrant_scan001_{indices[-1]:04}.tif"
    with changed.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(AnalysisSourceLeaseRefused):
        with requalified_analysis_source(result.receipt):
            pass


@pytest.mark.parametrize("with_spec", [False, True])
def test_receipt_freezes_external_eiger_shard(tmp_path, with_spec):
    spec, _images = _eiger(tmp_path)
    spec = replace(spec, kind=SourceKind.EIGER_MASTER)
    if with_spec:
        spec = replace(_spec_images(tmp_path, ()), options={
            "scan": "1.1", "image_dir": str(tmp_path), "image_stem": "two_per_point_",
            "require_complete_images": True,
        })
    spec = _confirmed(spec, PointImagePolicy(2, "average"))
    result = run_metadata_table(MetadataTablePlan(spec))
    assert result.disposition is AnalysisDisposition.COMPLETED, result.diagnostics
    dependencies = {path.name for path, _resolved, _revision in result.receipt.dependency_revisions}
    assert "two_per_point_data_000001.h5" in dependencies
    with h5py.File(tmp_path / "two_per_point_data_000001.h5", "r+") as stream:
        stream["entry/data/data"][1, 0, 0] = 99
    with pytest.raises(AnalysisSourceLeaseRefused):
        with requalified_analysis_source(result.receipt):
            pass
