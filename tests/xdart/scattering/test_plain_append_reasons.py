"""An Append refusal is said plainly in the log's warning and the end-of-Run list, its technical text a debug line
(maintainer, 2026-10-26, after "Details: foreign Append identity [APPEND-SKIPPED]" for every scan he had processed
with another χ offset: "terms like foreign append identity are not very helpful for most users. these are good for
debugging but user facing warnings and info should be simpler and more direct"; "you can just say integration settings
different from earlier ones or something like that. it doesn't have to be very specific. if it is obvious where the
mismatch is that can be specified")."""
from __future__ import annotations

import pytest

from xdart.gui.tabs.scattering.adapters.run_executor import _plain_append_reason


@pytest.mark.parametrize("technical, plain", [
    ("foreign Append identity: integration settings", "it was made with different integration settings"),
    ("foreign Append identity: results", "it holds different results (1-D, 2-D or GI maps) than this Run makes"),
    ("foreign Append identity: Project folder", "it was made under a different Project folder"),
    ("foreign Append identity: source",
     "it was made from a different source (another folder, path spelling or file type)"),
    ("source does not extend exact image-member history: the output holds 5 images, the source shows 4; "
     "scan_0005.tif (image 5) is not listed",
     "the images it was made from changed (the output holds 5 images, the source shows 4; scan_0005.tif (image 5) is "
     "not listed)"),
    # the first image that differs by name; the fields that changed are the debug line's
    ("source does not extend exact image-member history: image 2 (scan_0001.tif): mtime_ns 1793054071866776687 → "
     "1793054071975305268 (+0.109 s); size 708 → 1220", "the images it was made from changed (image 2, scan_0001.tif)"),
    ("source does not extend exact image-member history: no single image differs",
     "the images it was made from changed"),
    ("source extent regressed", "the scan now has fewer images than its output holds"),  # an image removed
    ("source size regressed", "the images it was made from changed"),  # the series' first image rewritten
    ("source mtime_ns regressed", "the images it was made from changed"),
    # a Live Append's own refusals (a shrunken NeXus stack said "live source extension remapped requested labels")
    ("live source extension remapped requested labels", "the images it was made from changed"),
    ("live source extension remapped committed prefix", "the images it was made from changed"),
    ("live source extension changed Append identity", "it was made with different settings"),
    ("source does not extend exact external-member history", "the images it was made from changed"),
    ("source does not extend exact path",
     "it was made from a different source (another folder, path spelling or file type)"),
    ("an earlier Append of this output did not finish; use Replace to rebuild it (committed Append prefix lineage is "
     "not committed)", "an earlier Append of it did not finish"),
    # a remedy wrapped around another refusal is what is said: it names what to do
    ("an earlier Append of this output did not finish; use Replace to rebuild it (source does not extend exact "
     "image-member history: image 2 (scan_0001.tif): size 708 → 1220)", "an earlier Append of it did not finish"),
    ("another program has the output open; close it, then run again (foreign Append identity: integration settings)",
     "another program has it open; close that program, then run again"),
    ("DIMENSION_REPLACEMENT_APPEND_UNSUPPORTED", "it was reintegrated, and Append cannot add to a reintegrated output"),
    ("something new and unforeseen", "something new and unforeseen"),
    # a path is never read as a refusal's words
    ("[Errno 13] Permission denied: '/data/regressed_remapped/scan_int1d.nexus'",
     "[Errno 13] Permission denied: '/data/regressed_remapped/scan_int1d.nexus'"),
    ("[Errno 13] Permission denied: '/data/source size regressed/scan.nexus'",
     "[Errno 13] Permission denied: '/data/source size regressed/scan.nexus'"),
    ("[Errno 13] Permission denied: '/data/foreign Append identity: integration settings/scan.nexus'",
     "[Errno 13] Permission denied: '/data/foreign Append identity: integration settings/scan.nexus'"),
])
def test_an_append_refusal_reads_plainly(technical, plain):
    assert _plain_append_reason(technical) == plain


def test_a_live_append_onto_an_output_made_with_other_settings_says_so(tmp_path, caplog):
    """His Run: outputs made with one χ offset, then a Live Append with another; here the 1-D points differ."""
    import logging

    pytest.importorskip("pyqtgraph")
    from tests.xdart.scattering.test_browse_reference_rings_run import _poni
    from tests.xdart.scattering.test_live_qualify_refusal_run import _frame, _settle, _stop, _wait
    from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _start
    from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
    from xrd_tools.session.run_configuration import RunIntent
    from xrd_tools.sources.selection import DirectorySourceSpec

    raw = tmp_path / "raw"
    raw.mkdir()
    _poni(tmp_path)
    _frame(raw, "scan", 1, "txt", 1.0)

    def intent(mode, npt):
        return RunIntent(source_spec=DirectorySourceSpec(raw, suffixes=(".tif",), metadata_format="txt"),
                         poni_file=str(tmp_path / "cal.poni"), project_root=str(raw),
                         save_path=str(tmp_path / "processed"), output_mode=mode, processing_mode="Int 1D",
                         live_mode=True, max_cores=1, bai_1d_args={"npt": npt, "method": "numpy"})

    for mode, npt, request, grow in (("Overwrite", 8, 7761, False), ("Append", 16, 7762, True)):
        executor = StandardRunExecutor(join_timeout=2.)
        identity = _start(executor, intent(mode, npt), request_value=request)
        events = []
        try:
            caplog.set_level(logging.DEBUG)
            if grow:
                _settle(executor, events, 1.)
                _frame(raw, "scan", 2, "txt", 2.0)
            _wait(executor, events, lambda values: any("[APPEND-SKIPPED]" in r.getMessage() for r in caplog.records)
                  or (not grow and any(e.kind.name == "FRAME_READY" for e in values))
                  or any(e.kind in _TERMINAL for e in values), 30.)
            _settle(executor, events, .5)
        finally:
            _stop(executor, identity, events)
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "[APPEND-SKIPPED]" in
              r.getMessage()]
    assert warned and warned[0] == (
        "scan_int1d.nexus stops growing: it was made with different integration settings; its frames so far are kept "
        "and its later frames are skipped in this Run (Replace reprocesses it). [APPEND-SKIPPED]"), warned
    assert any(r.levelno == logging.DEBUG and "foreign Append identity: integration settings" in r.getMessage()
               for r in caplog.records)
