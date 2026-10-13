"""LIVE-QUALIFY-REFUSAL-20261012 unit rows for the two headless facts the live owner uses.

`published_metadata_drift` names the published members' metadata files that a newer qualification no longer admits
as they were, at the same extent or a larger one; `live_sidecar_observation` names what a live frame waits on.
"""
from pathlib import Path

import numpy as np
import tifffile

from tests.core.test_p36_average import _txt
from xrd_tools.io.metadata import live_sidecar_observation
from xrd_tools.sources.execution_graph import published_metadata_drift, qualify_source_execution_graph
from xrd_tools.sources.selection import image_series_spec


def _graph(first):
    return qualify_source_execution_graph(image_series_spec(first, metadata_format="auto"))


def _names(paths):
    return tuple(Path(path).name for path in paths)


def test_published_metadata_drift_compares_the_published_prefix_at_any_extent(tmp_path):
    first, second = tmp_path / "scan_0001.tif", tmp_path / "scan_0002.tif"
    tifffile.imwrite(first, np.ones((2, 2), dtype=np.uint16))
    _txt(first, counters=(("I0", 1.0),))
    one = _graph(first)
    assert published_metadata_drift(one, one) == ()
    tifffile.imwrite(second, np.ones((2, 2), dtype=np.uint16))  # a new member, without a sidecar yet
    two = _graph(first)
    assert published_metadata_drift(one, two) == (), "a larger extent with an unchanged prefix is not drift"
    assert published_metadata_drift(two, one) == (), "a shorter qualification is raw drift, not this fact's"
    _txt(second, counters=(("I0", 2.0),))
    assert published_metadata_drift(one, _graph(first)) == (), "metadata of an unpublished member is not drift"
    assert _names(published_metadata_drift(two, _graph(first))) == ("scan_0002.txt",), "appeared"
    _txt(first, counters=(("I0", 10.0),))
    assert _names(published_metadata_drift(one, _graph(first))) == ("scan_0001.txt",), "changed"


def test_live_sidecar_observation_names_the_selected_file_or_the_auto_candidates(tmp_path):
    image = tmp_path / "scan_0001.tif"
    tifffile.imwrite(image, np.ones((2, 2), dtype=np.uint16))
    assert live_sidecar_observation(image, "txt") == ()
    assert live_sidecar_observation(image, "auto") == ()
    _txt(image, counters=(("I0", 1.0),))
    (selected,) = live_sidecar_observation(image, "txt")
    assert Path(selected[0]).name == "scan_0001.txt" and selected[1] is not None
    assert live_sidecar_observation(image, "auto") == (selected,)
    assert live_sidecar_observation(image, "pdi") == ()
