"""LIVE-METADATA-DISCOVERY-RACE-20261012 frozen headless oracle (Claude, October 12): HR1-HR3.

A live acquisition adds files to a watched directory at any time. The metadata reader records the image's directory
before a read and refuses the read if it changed. Each change below is a real file created inside the reader's own
discovery window (after its first record, before its final check), through the reader's sidecar lookup, one clock
tick after the record: probe p01 saw APFS detect it without the pause; coarser filesystem clocks may not.
"""
import time

import pytest

import xrd_tools.io.metadata as metadata
from tests.core.test_metadata import _TXT_CONTENT


def _arrival(directory, index):
    """One more frame lands in the directory: its sidecar, then its image renamed into place whole."""
    time.sleep(.02)
    image = directory / f'scan_{index:04d}.tif'
    image.with_suffix('.txt').write_text(_TXT_CONTENT, encoding='utf-8')
    partial = directory / f'scan_{index:04d}.part'
    partial.write_bytes(b'')
    partial.replace(image)


def _frame(directory):
    image = directory / 'scan_0001.tif'
    image.write_bytes(b'')
    sidecar = image.with_suffix('.txt')
    sidecar.write_text(_TXT_CONTENT, encoding='utf-8')
    return image, sidecar


def test_hr1_a_read_that_another_frame_lands_beside_returns_its_own_sidecar(tmp_path, monkeypatch):
    image, sidecar = _frame(tmp_path)
    real, lookups = metadata._find_sidecar, []

    def lookup(path, ext):
        lookups.append(path)
        if len(lookups) == 1:
            _arrival(tmp_path, 2)
        return real(path, ext)
    monkeypatch.setattr(metadata, '_find_sidecar', lookup)
    observed = metadata.read_image_metadata_observed(image, 'txt')
    assert (observed.source_path, dict(observed.values)) == (sidecar, metadata.read_txt_metadata(sidecar))
    assert dict(observed.discovery_revisions) == {tmp_path: metadata._metadata_path_revision(tmp_path)}, \
        'the result records the directory as it stood for the read it returns'


def test_hr2_a_recorded_candidate_rewritten_during_an_auto_read_is_still_refused_at_once(tmp_path, monkeypatch):
    """Oracle correction (October 12, before any production commit): Auto records only the candidate sidecars that
    exist when a read begins (probe p02), so a candidate created mid-read is a directory change and is read again.
    What stays refused at once is a recorded candidate that changes."""
    monkeypatch.setattr(metadata, '_AUTO_SIDECAR_CACHE', {})
    monkeypatch.setattr(metadata, '_DIR_LISTING_CACHE', {})
    image, candidate = _frame(tmp_path)
    real, reads = metadata._read_auto_metadata_observed, []

    def read(*args, **kwargs):
        reads.append(True)
        if len(reads) == 1:
            time.sleep(.02)
            candidate.write_text(_TXT_CONTENT.replace('i0 = 1000.0', 'i0 = 12000.0'), encoding='utf-8')
        return real(*args, **kwargs)
    monkeypatch.setattr(metadata, '_read_auto_metadata_observed', read)
    with pytest.raises(OSError, match='changed during read'):
        metadata.read_image_metadata_observed(image, meta_format='auto')
    assert len(reads) == 1, 'a recorded candidate that changed is refused at once, never read again'


def test_hr3_a_directory_that_changes_during_every_read_is_still_refused(tmp_path, monkeypatch):
    image, _sidecar = _frame(tmp_path)
    real, lookups = metadata._find_sidecar, []

    def lookup(path, ext):
        lookups.append(path)
        if len(lookups) <= 20:
            _arrival(tmp_path, len(lookups) + 1)
        return real(path, ext)
    monkeypatch.setattr(metadata, '_find_sidecar', lookup)
    with pytest.raises(OSError, match='metadata discovery changed during read'):
        metadata.read_image_metadata_observed(image, 'txt')
    assert len(lookups) <= 5, f'{len(lookups)} reads: repeating a read is bounded'
