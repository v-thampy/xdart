"""Frozen P1 E2: exact finite F1 text, originals and exclusions."""
from xrd_tools.help_topics import diagnostic_text

EXPECTED = {
    'SOURCE_CANDIDATE_LIMIT_EXCEEDED': 'At most 256 source candidates and 1 MiB of canonical candidate storage.\nNext action: select an exact source or narrow the discovery request.',
    'INVALID_SOURCE_SELECTION': 'The source-selection options do not form a supported, consistent request.\nNext action: check the source, selection mode and accompanying options, then request metadata again.',
    'SOURCE_NOT_FOUND': 'Required source content could not be found or accessed.\nNext action: check the source and member paths, file availability and access permissions, then reopen the source.',
    'SOURCE_SELECTION_REQUIRED': 'One exact source, scan or entry must be selected.\nNext action: choose an available source, scan or entry, then request metadata again.',
    'SOURCE_KIND_UNSUPPORTED': 'This analysis path does not accept the selected source kind.\nNext action: choose a supported file-based source.',
    'SOURCE_IDENTITY_MISMATCH': 'The requested or retained source or frame identity could not be confirmed.\nNext action: reload the intended source and metadata, then recompute the current analysis.',
    'INVALID_CANONICAL_VALUE': 'The input or metadata does not satisfy the operation’s supported value format.\nNext action: check the source-selection values and metadata for malformed or unsupported values, then reload.',
    'METADATA_COLUMN_PROJECTION_UNSUPPORTED': 'This source cannot provide the requested occurrence-aware metadata-column projection.\nNext action: remove the explicit projection or choose a source that supports it.',
    'METADATA_COLUMN_PROJECTION_INVALID': 'A requested column name or occurrence is absent from the available metadata columns.\nNext action: select existing column names and occurrences, then reload metadata.',
    'METADATA_COLUMN_LIMIT_EXCEEDED': 'At most 256 table columns, including the generated frame-index column.\nNext action: request fewer metadata columns or choose a source with a smaller table.',
    'METADATA_TABLE_LIMIT_EXCEEDED': 'The canonical metadata-table payload must fit within 64 MiB.\nNext action: reduce the requested table or source payload, or choose a smaller source.',
    'SOURCE_REVISION_CHANGED': 'A source or dependency no longer matches the file state recorded for this operation.\nNext action: let file changes finish, then reload metadata and repeat the analysis.',
    'METADATA_ROW_LIMIT_EXCEEDED': 'At most 100,000 metadata rows.\nNext action: choose a source with fewer frames or rows.',
    'CANCELLED': 'The operation ended with cancellation rather than a completed result.\nNext action: start the analysis again when ready.',
    'SCAN_PLOT_NONNUMERIC_COLUMN': 'A requested plot column is absent, ambiguous, or lacks numeric data.\nNext action: choose a uniquely available numeric column from the current metadata table.',
    'METADATA_TABLE_REQUIRED': 'Plotting requires a completed metadata table.\nNext action: load metadata successfully, then request the plot again.',
    'SCAN_PLOT_Y_REQUIRED': 'No Y series could be selected for the plot.\nNext action: select an available numeric Y column or compute an ROI series.',
    'SCAN_PLOT_TRACE_LIMIT_EXCEEDED': 'At most 16 requested or emitted plot traces.\nNext action: select fewer Y series and account for names that produce both metadata and ROI traces.',
    'SCAN_PLOT_COLUMN_NOT_FOUND': 'A requested Y column is available in neither the metadata table nor the supplied ROI result.\nNext action: select a current available column or recompute the intended ROI series.',
    'SCAN_PLOT_LIMIT_EXCEEDED': 'The canonical plot payload must fit within 64 MiB.\nNext action: plot fewer series or choose a source with fewer rows.',
    'P3_7_LIVE_FITTING_UNAVAILABLE': 'Live fitting is unavailable in this analysis path.\nNext action: use an available fitting action; this live action remains disabled.',
    'P3_7_BATCH_DISPLAY_PROJECTION_UNAVAILABLE': 'Batch fitting through this display-projection path is unavailable.\nNext action: use an available fitting action; this batch action remains disabled.',
    'P3_7_PHASE_TEXTURE_UNAVAILABLE': 'Texture selection is unavailable in this phase-fitting path.\nNext action: continue with the available phase-fitting settings.',
    'P3_7_EXPORT_UNAVAILABLE': 'This export action is unavailable in the current analysis path.\nNext action: continue inspecting the current result; this export action remains disabled.',
    'P3_7_ANALYSIS_TERMINAL_INVALID': 'No valid terminal update is available for adoption.\nNext action: wait for the operation to finish or request it again; report the literal code if it repeats.',
    'P3_7_ANALYSIS_STALE': 'The result no longer belongs to the current analysis request.\nNext action: request analysis again for the current source and settings.',
    'P3_7_ANALYSIS_REFUSED': 'The outcome is not eligible for adoption as a completed result.\nNext action: review the original code and details, then retry a supported request.',
    'P3_7_GUI_RETENTION_LIMIT': 'Admission requires a valid nonnegative integer byte charge and no more than 128 MiB of combined retained analysis data.\nNext action: reduce requested data and retry; report this code if a small valid request still refuses.',
    'P3_7_ANALYSIS_RESULT_IDENTITY_MISMATCH': 'The returned result or its qualification does not match the current request.\nNext action: reload the current source and metadata, then request analysis again.',
    'ROI_RAW_UNAVAILABLE': 'The selected full raw preview could not load.\nNext action: check source/raw accessibility and request preview again. No discarded exception detail is inferred.',
    'ROI_RAW_SHAPE_INVALID': 'The preview is not one 2D detector image.\nNext action: select a source providing a 2D detector frame and request preview again.',
    'ROI_PREVIEW_LIMIT_EXCEEDED': 'The whole-frame preview exceeds the existing 64 MiB canonical admission cap.\nNext action: select a smaller source frame; a smaller rectangle cannot lower the whole-frame charge.',
    'ROI_IDENTITY_MISMATCH': 'ROI/table/source/labels or plotted series do not agree.\nNext action: reload the current table and recompute current ROI statistics; the failed comparison is not inferred.',
    'ROI_SIGNAL_COUNT_INVALID': 'The existing reducer accepts 1–5 signal ROIs.\nNext action: keep one to five signal definitions, then Compute again.',
    'ROI_RAW_UNAVAILABLE_ALL': 'Every completed raw frame was unavailable in the completed result.\nNext action: check raw access before recomputing; absent points are not covered zeros.',
    'ROI_INVALID_MASK_IGNORED': 'Static-mask resolution was attempted and failed; that mask was not applied.\nNext action: supply a mask resolving to detector shape and recompute through the path that accepts it.',
    'ROI_RESULT_LIMIT_EXCEEDED': 'The result exceeds the existing 64 MiB canonical admission cap.\nNext action: reduce requested frames or signal count; the cap is not a pixel-dimension limit.',
}


def test_f1_guidance_preserves_every_literal_and_exact_key():
    details = ('literal β\nline 2', 'SOURCE_NOT_FOUND')
    assert 'Next action: check the source and member paths' in diagnostic_text('SOURCE_NOT_FOUND', details)
    from xrd_tools.help_topics import _GUIDANCE
    assert _GUIDANCE == EXPECTED and len(EXPECTED) == 37
    for code, text in EXPECTED.items():
        assert diagnostic_text(code, ('detail', code)) == f'{code}\ndetail\n{code}\n\nGuidance\n{code}\n{text}'
        assert diagnostic_text('unknown', (code,)) == f'unknown\n{code}\n\nGuidance\n{code}\n{text}'
    for literal in ('unknown', '', 'OK', 'INVALID_CANCEL_TOKEN', 'INVALID_REQUALIFICATION_PLAN',
                    'prefix SOURCE_NOT_FOUND', 'SOURCE_NOT_FOUND suffix'):
        assert diagnostic_text(literal, ('literal β\nline 2',)) == literal+'\nliteral β\nline 2'
