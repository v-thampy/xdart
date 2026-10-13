"""Review coverage for non-finite SPEC lines and published-row precedence."""
from pathlib import Path

import pytest

from tests.core.test_live_spec_metadata import _acquisition, _series
from xrd_tools.io.metadata import SpecRowInvalid
from xrd_tools.sources.execution_graph import published_metadata_drift, qualify_source_execution_graph


@pytest.mark.parametrize('token', ['nan', 'inf'])
def test_a_nonfinite_new_row_is_refused_naming_its_row(tmp_path, token):
    raw, spec, paths = _acquisition(tmp_path, images=2, rows=2)
    spec.write_text(spec.read_text().replace('0.1 101\n', f'0.1 {token}\n'))
    with pytest.raises(SpecRowInvalid, match='scan 1 row 2'):
        qualify_source_execution_graph(_series(raw, paths), infer_point_images=False)


@pytest.mark.parametrize('token', ['nan', 'inf'])
def test_a_nonfinite_published_row_reaches_the_metadata_fence(tmp_path, token):
    raw, spec, paths = _acquisition(tmp_path, images=2, rows=2)
    source = _series(raw, paths)
    published = qualify_source_execution_graph(source, infer_point_images=False)
    spec.write_text(spec.read_text().replace('0.0 100\n', f'0.0 {token}\n'))
    current = qualify_source_execution_graph(source, infer_point_images=False, published_members=2)
    assert [item.metadata_file for item in current.stamp.metadata_sources] == [None, None]
    assert {Path(path).name for path in published_metadata_drift(published, current)} == {'calibrant'}
