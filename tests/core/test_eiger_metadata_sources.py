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


def test_a_receipt_answers_only_its_own_request(tmp_path):
    """The window's result check (``metadata_receipt_answers``): only ``_select``'s two container resolutions may
    differ, the entry it chose when none was asked for and the Eiger refinement; a Folder's candidates rely on both."""
    from xrd_tools.analysis.scan_operations import metadata_receipt_answers
    master = _vendor_eiger_master(tmp_path / "scan_master.h5", [tmp_path / "scan_data_000001.h5"])
    other = _vendor_eiger_master(tmp_path / "other_master.h5", [tmp_path / "other_data_000001.h5"])
    receipt = run_metadata_table(MetadataTablePlan(SourceSpec(master, SourceKind.NEXUS_STACK))).receipt
    for asked in (SourceSpec(master, SourceKind.NEXUS_STACK), SourceSpec(master, SourceKind.EIGER_MASTER),
                  SourceSpec(master, SourceKind.EIGER_MASTER, entry="entry"), receipt.source_spec):
        assert metadata_receipt_answers(receipt, asked), asked
    for asked in (SourceSpec(other, SourceKind.NEXUS_STACK), SourceSpec(master, SourceKind.NEXUS_STACK, entry="other"),
                  SourceSpec(master, SourceKind.PROCESSED_NEXUS), SourceSpec(master, SourceKind.SPEC),
                  SourceSpec(master, SourceKind.NEXUS_STACK, options={"scan": "1"}),
                  SourceSpec(master, SourceKind.NEXUS_STACK, metadata_uri=tmp_path / "meta"), str(master)):
        assert not metadata_receipt_answers(receipt, asked), asked
    plain = _chunked_stack(tmp_path / "plain.nxs")
    plain_receipt = run_metadata_table(MetadataTablePlan(SourceSpec(plain, SourceKind.NEXUS_STACK))).receipt
    assert metadata_receipt_answers(plain_receipt, SourceSpec(plain, SourceKind.NEXUS_STACK))
    assert not metadata_receipt_answers(plain_receipt, SourceSpec(plain, SourceKind.EIGER_MASTER))
