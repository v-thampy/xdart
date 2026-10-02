"""Shared notebook result host over PatternViewer, selector/range controls and the public Plotly model."""
class WorkflowResultViewer:
    def __init__(self, outputs, mode):
        import ipywidgets as widgets
        from xrd_tools.gui.widgets import PatternViewer, PatternSelector, FitRangeControls
        from .workflow import workflow_result_references, read_workflow_result
        self.references = workflow_result_references(outputs, mode)
        if not self.references:
            raise ValueError("WORKFLOW_OUTPUT_REQUIRED: no committed result to display")
        self._read, self._disposed = read_workflow_result, False
        self.pattern = PatternViewer()
        self.selector = PatternSelector([ref.label for ref in self.references], on_select=self.select)
        self.log = widgets.Checkbox(value=False, description="Log intensity")
        self.log.observe(self._display, names="value")
        self.selector.overlays.observe(self._display, names="value")
        self.snapshot = self._read(self.references[0])
        axis = self.snapshot.axes[0].values
        self.ranges = FitRangeControls((float(axis.min()), float(axis.max())), on_display=self.render, fit_enabled=False)
        self.widget = widgets.VBox([self.selector.widget, self.log, self.ranges.widget, self.pattern.figure_widget])
        self.render()

    def select(self, index):
        from xrd_tools.gui.widgets import FitRangeControls
        self.snapshot = self._read(self.references[index])
        axis = self.snapshot.axes[0].values
        old = self.ranges
        self.ranges = FitRangeControls((float(axis.min()), float(axis.max())), on_display=self.render, fit_enabled=False)
        self.widget.children = (self.selector.widget, self.log, self.ranges.widget, self.pattern.figure_widget)
        old.dispose()
        self.render()

    def _display(self, change):
        self.render()

    def render(self):
        if self._disposed:
            return
        from xrd_tools.viz.plotly import plot_analysis_outcome
        comparisons = (self._read(self.references[i]) for i in self.selector.overlays.value[:4])
        figure = plot_analysis_outcome(self.snapshot, log_y=self.log.value,
                                      x_range=(self.ranges.display_min.value, self.ranges.display_max.value),
                                      comparisons=comparisons)
        if len(self.snapshot.axes) > 1:
            figure.update_xaxes(range=(self.ranges.display_min.value, self.ranges.display_max.value))
        with self.pattern.figure_widget.batch_update():
            self.pattern.figure_widget.data = ()
            for trace in figure.data:
                self.pattern.figure_widget.add_trace(trace)
            self.pattern.figure_widget.layout = figure.layout

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        self.log.unobserve(self._display, names="value")
        self.selector.overlays.unobserve(self._display, names="value")
        self.selector.dispose()
        self.ranges.dispose()
        self.pattern.dispose()
        from xrd_tools.gui.widgets.fit_controls import _close_widgets
        _close_widgets(self.widget)
