from __future__ import annotations

from dataclasses import replace

from matplotlib import colormaps as matplotlib_colormaps
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from xdart.gui.tabs.scattering.shell_widgets import ContentFitComboBox
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from xdart.gui.themes import apply_theme

from tests.xdart.scattering.e3_shell_support import make_shell_projection


def _items(combo: QtWidgets.QComboBox) -> tuple[str, ...]:
    return tuple(combo.itemText(index) for index in range(combo.count()))


def _required_closed_width(combo: QtWidgets.QComboBox) -> int:
    metrics = combo.fontMetrics()
    widest = max(
        (
            metrics.horizontalAdvance(combo.itemText(index))
            for index in range(combo.count())
        ),
        default=metrics.horizontalAdvance(""),
    )
    option = QtWidgets.QStyleOptionComboBox()
    combo.initStyleOption(option)
    return combo.style().sizeFromContents(
        QtWidgets.QStyle.ContentsType.CT_ComboBox,
        option,
        QtCore.QSize(widest, metrics.height()),
        combo,
    ).width()


def test_colormap_choices_are_complete_construction_owned_and_safe() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection()
    try:
        expected = ("Default", *tuple(matplotlib_colormaps))
        combo = shell.scientific.color_map
        model = combo.model()
        assert _items(combo) == expected
        shell.apply_state(state)
        assert combo.currentText() == "viridis"

        malformed = replace(
            state,
            revision=2,
            scientific=replace(
                state.scientific,
                color_maps=("Default", "not-a-real-map"),
                color_map="not-a-real-map",
            ),
        )
        shell.apply_state(malformed)

        assert combo.model() is model
        assert _items(combo) == expected
        assert combo.currentText() == "Default"
    finally:
        shell.close()


def test_scientific_combos_fit_closed_and_popup_text_after_font_change() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection()
    long_channel = "Incident monitor corrected by exposure time"
    try:
        shell.apply_state(
            replace(
                state,
                revision=2,
                scientific=replace(
                    state.scientific,
                    norm_channels=("Norm Channel", long_channel),
                    norm_channel=long_channel,
                ),
            )
        )
        combos = (
            shell.scientific.norm,
            shell.scientific.color_map,
            shell.scientific.image_axis,
            shell.scientific.plot_axis,
            shell.scientific.plot_mode,
        )
        for combo in combos:
            assert type(combo) is ContentFitComboBox
            assert type(combo.itemDelegate()) is QtWidgets.QStyledItemDelegate
            assert (
                combo.view().textElideMode()
                is QtCore.Qt.TextElideMode.ElideNone
            )
            assert combo.minimumWidth() >= _required_closed_width(combo)
            widest = max(
                combo.fontMetrics().horizontalAdvance(combo.itemText(index))
                for index in range(combo.count())
            )
            assert combo.view().minimumWidth() >= widest

        before = shell.scientific.norm.minimumWidth()
        font = QtGui.QFont(shell.scientific.norm.font())
        font.setPointSizeF(max(1.0, font.pointSizeF() * 1.5))
        shell.scientific.norm.setFont(font)
        app.processEvents()
        assert shell.scientific.norm.currentText() == long_channel
        assert shell.scientific.norm.minimumWidth() > before
        assert (
            shell.scientific.norm.minimumWidth()
            >= _required_closed_width(shell.scientific.norm)
        )
    finally:
        shell.close()


def test_vnext_scientific_toolbar_uses_one_square_surface_height() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    apply_theme(app, "dark", font_scale="default", spacing="normal")
    shell = ScatteringWorkspaceShell()
    try:
        shell.apply_state(make_shell_projection())
        shell.resize(1500, 1000)
        shell.show()
        app.processEvents()

        surfaces = (
            shell.scientific.norm,
            shell.scientific.background,
            shell.scientific.color_map,
            shell.scientific.log_scale,
            shell.scientific.plot_axis,
            shell.scientific.slice,
            shell.scientific.slice_center,
            shell.scientific.slice_width,
            shell.scientific.pin,
            shell.scientific.plot_mode,
            shell.scientific.options,
            shell.scientific.clear,
            shell.scientific.share_axis,
            shell.scientific.image_axis,
            shell.scientific.frame_selector,
        )
        assert {surface.height() for surface in surfaces} == {28}
        assert {
            (surface.minimumHeight(), surface.maximumHeight())
            for surface in surfaces
        } == {(28, 28)}

        qss = app.styleSheet()
        selector = "QWidget#scatteringWorkspaceShell QLineEdit,"
        assert selector in qss
        block = qss[qss.index(selector):]
        block = block[:block.index("}")]
        assert "border-radius: 0px;" in block
        # QSS height is the 26 px content box; the 1 px border on both sides
        # yields the measured 28 px outer widget geometry above.
        assert "min-height: 26px;" in block
        assert "max-height: 26px;" in block
    finally:
        shell.close()
