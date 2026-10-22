"""Deterministic review coverage for a SPEC append inside the reduction read."""
import threading
from pathlib import Path

import xrd_tools.io.metadata as metadata
import xrd_tools.sources.image as image_source
from tests.xdart.scattering.test_live_spec_metadata_custody_run import _kinds
from tests.xdart.scattering.test_live_spec_metadata_run import (
    STEM, _acquisition, _image, _output, _reduced, _row, _run,
)
from tests.xdart.scattering.test_live_qualify_refusal_run import (
    _said, _settle, _stop, _wait, _worker,
)


def test_metadata_growth_inside_the_reduction_read_retries_before_publication(tmp_path, monkeypatch):
    raw, spec = _acquisition(tmp_path)
    grew, refused = threading.Event(), threading.Event()
    state = threading.local()
    real_observed = image_source.read_image_metadata_observed
    real_spec = metadata._read_spec_metadata_observed

    def read_spec(*args, **kwargs):
        result = real_spec(*args, **kwargs)
        if getattr(state, 'reduction', False) and not grew.is_set():
            _row(spec, 1, 2)  # real growth after the read, before the real revision guard
            grew.set()
        return result

    def read_image(path, *args, **kwargs):
        state.reduction = _worker() and Path(path).name == f'{STEM.format(scan=1)}_0001.tif'
        try:
            return real_observed(path, *args, **kwargs)
        except metadata.MetadataSourceChanged:
            refused.set()
            raise
        finally:
            state.reduction = False

    monkeypatch.setattr(metadata, '_read_spec_metadata_observed', read_spec)
    monkeypatch.setattr(image_source, 'read_image_metadata_observed', read_image)
    executor, identity = _run(tmp_path, raw, 5301)
    events = []
    try:
        _wait(executor, events, _reduced(1), 20.)
        _row(spec, 1, 1)
        _image(raw, 1, 1)
        _wait(executor, events, _reduced(2), 20.)
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert grew.is_set() and refused.is_set(), 'the real reader refused growth inside the reduction read'
    assert _kinds(terminal) == ['STOPPED'] and not _said(events, 'stopped growing'), terminal
    assert _output(tmp_path) == ((1, 2), (100.0, 101.0))
