"""Real scientific rendering replaces thumbnails without a redundant blank."""

from dataclasses import replace

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.shell_values import HeavyProjection
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from tests.xdart.scattering.e3_shell_support import make_shell_projection


@pytest.mark.parametrize("prior_source", ("thumbnail", "full"))
def test_thumbnail_replacement_scrubs_only_outgoing_full_image(monkeypatch, prior_source):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shell = ScatteringWorkspaceShell()
    shell.resize(1100, 800)
    shell.show()
    state = make_shell_projection(frame_count=2, heavy_indices=(0, 1), plot_mode="Single")
    first, second = state.navigation.frames
    shape = (3, 4) if prior_source == "thumbnail" else (12, 16)
    first_pixels = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) + 1
    second_pixels = np.arange(12, dtype=np.uint16).reshape(3, 4) + 100
    first_pixels.flags.writeable = second_pixels.flags.writeable = False
    state = replace(state, scientific=replace(
        state.scientific, heavy=HeavyProjection(
            first, first_pixels, detector_shape=(12, 16), detector_source=prior_source,
        ),
    ))
    pane = shell.scientific.raw
    try:
        shell.apply_state(state)
        app.processEvents()
        image_item = pane.image
        np.testing.assert_array_equal(image_item.image, first_pixels.T[:, ::-1])
        clears, empty_before_replacement = [], []
        clear, set_image = pane.clear, pane.canvas.setImage

        def observed_clear():
            clears.append(True)
            return clear()

        def observed_set_image(*args, **kwargs):
            empty_before_replacement.append(
                pane.image.image is None and pane.canvas.raw_image.size == 0
                and pane.image.qimage is None
            )
            return set_image(*args, **kwargs)

        monkeypatch.setattr(pane, "clear", observed_clear)
        monkeypatch.setattr(pane.canvas, "setImage", observed_set_image)
        replacement = replace(
            state, revision=state.revision + 1,
            navigation=replace(state.navigation, current=second, selected=(second,)),
            scientific=replace(state.scientific, heavy=HeavyProjection(
                second, second_pixels, detector_shape=(12, 16), detector_source="thumbnail",
            )),
        )
        shell.apply_state(replacement)
        app.processEvents()
        assert pane.image is image_item
        np.testing.assert_array_equal(pane.image.image, second_pixels.T[:, ::-1])
        assert not np.shares_memory(pane.image.image, first_pixels)
        assert not np.shares_memory(pane.canvas.raw_image, first_pixels)
        assert not np.shares_memory(pane.canvas.displayed_image, first_pixels)
        assert clears == ([True] if prior_source == "full" else [])
        assert empty_before_replacement == [prior_source == "full"]
        shell.apply_state(replace(replacement, revision=replacement.revision + 1))
        assert empty_before_replacement == [prior_source == "full"]
        shell.apply_state(replace(
            replacement, revision=replacement.revision + 2,
            scientific=replace(replacement.scientific,
                heavy=HeavyProjection(second), retain_display=False),
        ))
        assert len(clears) == (2 if prior_source == "full" else 1)
        assert pane.image.image is None and pane.image.qimage is None
        assert pane.canvas.raw_image.size == pane.canvas.displayed_image.size == 0
    finally:
        shell.close()
        app.processEvents()
