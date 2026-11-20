"""NEXT-1217-EIGER-SOURCES: a raw NeXus stack request on a DECTRIS Eiger master reads it as the Eiger master its
content says it is (maintainer, 2026-11-20: "I'm trying to open an Eiger h5 file the to plot ROI stats for that and
it doesn't work"). The kind a name suggests, the Folder's "NeXus (raw stack)" pick and an Int profile's stored
``nexus_stack`` all refused it with SOURCE_IDENTITY_MISMATCH; the refinement is one way only."""
import pytest

from xrd_tools.analysis.plans import RoiSignal
from xrd_tools.analysis.scan_operations import (
    AnalysisDisposition, MetadataTablePlan, RoiPreviewPlan, RoiScanPlan, run_metadata_table, run_roi_preview,
    run_roi_scan,
)
from xrd_tools.core.roi import RoiSpec
from xrd_tools.core.scan import SourceKind, SourceSpec
from tests.core.test_container_descriptor import _chunked_stack, _eiger_master, _vendor_eiger_master

_PLANS = {
    "path": lambda path: MetadataTablePlan(str(path)),  # the window's File pick: the kind comes from the name
    "kind": lambda path: MetadataTablePlan(str(path), kind="nexus_stack"),
    "profile": lambda path: MetadataTablePlan(SourceSpec(path, SourceKind.NEXUS_STACK,
                                                         options={"metadata_format": "auto"})),
}


@pytest.mark.parametrize("request_shape", sorted(_PLANS))
def test_a_raw_stack_request_on_an_eiger_master_reads_the_eiger_master(tmp_path, request_shape):
    master = _vendor_eiger_master(tmp_path / "scan_master.h5", [tmp_path / "scan_data_000001.h5"])
    table = run_metadata_table(_PLANS[request_shape](master))
    assert (table.disposition, table.code) == (AnalysisDisposition.COMPLETED, "OK")
    assert table.receipt.resolved_kind is SourceKind.EIGER_MASTER
    assert SourceKind(table.receipt.source_spec.kind) is SourceKind.EIGER_MASTER
    assert len(table.labels) == 5
    # the ROI preview and scan reopen the receipt's own source and agree with it
    preview = run_roi_preview(RoiPreviewPlan.from_table(table, label=table.labels[0]))
    assert (preview.disposition, preview.image.shape) == (AnalysisDisposition.COMPLETED, (8, 8))
    roi = RoiSignal(RoiSpec(center_x=4, center_y=4, width_x=4, width_y=4), reducer="sum", name="roi")
    scan = run_roi_scan(RoiScanPlan.from_table(table, signals=(roi,)))
    assert (scan.disposition, scan.completed_labels) == (AnalysisDisposition.COMPLETED, table.labels)


def test_the_refinement_is_one_way(tmp_path):
    plain = _chunked_stack(tmp_path / "plain.nxs")
    bare = _eiger_master(tmp_path / "bare_master.h5", [tmp_path / "bare_data_000001.h5"])  # no DECTRIS block
    for stack in (plain, bare):
        table = run_metadata_table(MetadataTablePlan(str(stack)))
        assert (table.disposition, table.receipt.resolved_kind) == (AnalysisDisposition.COMPLETED,
                                                                    SourceKind.NEXUS_STACK)
        claimed = run_metadata_table(MetadataTablePlan(str(stack), kind="eiger_master"))
        assert (claimed.disposition, claimed.code) == (AnalysisDisposition.REFUSED, "SOURCE_IDENTITY_MISMATCH")
