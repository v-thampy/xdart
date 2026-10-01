"""Int's presentation background, its keyed path (AUD-9 stage 2b G8d): a keyed layout's background held by
Int's owner (one at a time, under its grant, released with the view's receipt) and subtracted panel by panel by
the shared projection step."""
from __future__ import annotations

import numpy as np

from xdart.gui.tabs.scattering.context_projection import apply_keyed_presentation_background
from xdart.gui.tabs.scattering.operation_values import OperationContextStamp
from xdart.gui.tabs.scattering.presentation_background import (
    DisplayBackgroundRendererReleaseReceipt, PresentationBackgroundOwner,
)
from xdart.gui.tabs.rsm.display import rsm_keyed_projection
from xrd_tools.session.rsm_viewer_model import RSMViewerModel
from tests.xdart.pages.test_rsm_page import _grid_values


def _projection(values):
    model = RSMViewerModel()
    return rsm_keyed_projection(values, model.snapshot(values))


def test_the_owner_holds_one_keyed_background_and_lets_go_only_with_the_views_receipt():
    owner = PresentationBackgroundOwner(capacity_bytes=536_870_912)
    stamp = OperationContextStamp(0)
    key = ("rsm-background", 1, "keyed", "/tmp/a", 1)
    assert not owner.activate_keyed("source", stamp=stamp, active_key=("rsm", 1, "raw"), nbytes=8)
    assert not owner.activate_keyed("source", stamp=stamp, active_key=key, nbytes=owner.capacity_bytes + 1)
    assert owner.activate_keyed("source", stamp=stamp, active_key=key, nbytes=8)
    assert owner.phase == "ACTIVE" and owner.active_key == key and owner.keyed_source() == "source"
    assert owner.active_bytes == 8 and owner.projection() is None  # no frame background drawn
    assert not owner.activate_keyed("other", stamp=stamp, active_key=key, nbytes=8)  # one at a time
    assert not owner.release(None) and owner.phase == "ACTIVE"  # the view must let go first
    assert owner.release(DisplayBackgroundRendererReleaseReceipt(key, True))
    assert owner.phase == "RELEASED" and owner.keyed_source() is None and owner.active_bytes == 0


def test_the_keyed_subtraction_is_panel_by_panel_and_refuses_another_grid():
    shown, background = _projection(_grid_values(3.0)), _projection(_grid_values(1.0))
    drawn = apply_keyed_presentation_background(shown, background, "the background a")
    assert drawn.background_set and drawn.background_note == "minus the background a"
    for got, mine, theirs in zip(drawn.panels, shown.panels, background.panels, strict=True):
        if hasattr(got, "image"):
            np.testing.assert_array_equal(got.image, mine.image - theirs.image)
        else:
            np.testing.assert_array_equal(got.intensity, mine.intensity - theirs.intensity)
    other = _projection(_grid_values(1.0, axis_offset=0.5))
    refused = apply_keyed_presentation_background(shown, other, "the background c")
    assert refused.background_set and "another grid" in refused.background_note
    assert all(np.array_equal(getattr(a, "image", None) if hasattr(a, "image") else a.intensity,
                              getattr(b, "image", None) if hasattr(b, "image") else b.intensity, equal_nan=True)
               for a, b in zip(refused.panels, shown.panels, strict=True))
    assert not apply_keyed_presentation_background(shown, None).background_set
