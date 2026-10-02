"""Native coordinate box controls shared by notebook region-fit hosts."""
from __future__ import annotations

import numpy as np


class RegionRangeControls:
    """One reused range owner per native axis; three-dimensional slices are presentation only."""
    def __init__(self, snapshot, *, bounds=None, on_display=None):
        import ipywidgets as widgets
        from .peak_fit_viewer import FitRangeControls
        self.axes = snapshot.axes
        bounds = dict(bounds or ())
        self.ranges = {axis.name: FitRangeControls((float(axis.values.min()), float(axis.values.max())),
                                                  fit_bounds=bounds.get(axis.name),
                                                  label=f"{axis.name} ({axis.unit})") for axis in self.axes}
        self._callback = on_display
        self.slice_axis = widgets.Dropdown(options=[axis.name for axis in self.axes], value=self.axes[-1].name,
                                           description="Slice axis", disabled=len(self.axes) != 3)
        self.slice_coordinate = widgets.FloatText(description="Slice coordinate")
        self.slice_axis.observe(self._axis_changed, names="value")
        self.slice_coordinate.observe(self._display_changed, names="value")
        self._axis_changed(None)
        self.widget = widgets.VBox([controls.fit_bounds for controls in self.ranges.values()] +
                                    ([widgets.HBox([self.slice_axis, self.slice_coordinate])] if len(self.axes) == 3 else []))

    @property
    def bounds(self):
        return tuple((name, tuple(controls.fit_bounds.value)) for name, controls in self.ranges.items())

    def set_on_display(self, callback):
        """Install the host callback after its composition is ready."""
        self._callback = callback

    @property
    def plane(self):
        axis = next(axis for axis in self.axes if axis.name == self.slice_axis.value)
        return axis.name, int(np.argmin(np.abs(axis.values - self.slice_coordinate.value)))

    def _axis_changed(self, change):
        axis = next(axis for axis in self.axes if axis.name == self.slice_axis.value)
        self.slice_coordinate.description = f"{axis.name} ({axis.unit})"
        self.slice_coordinate.value = float(axis.values[axis.values.size // 2])

    def _display_changed(self, change):
        if self._callback is not None:
            self._callback()

    def dispose(self):
        from .fit_controls import _close_widgets
        self.slice_axis.unobserve(self._axis_changed, names="value")
        self.slice_coordinate.unobserve(self._display_changed, names="value")
        self._callback = None
        for controls in self.ranges.values():
            controls.dispose()
        _close_widgets(self.widget)
