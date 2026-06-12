"""UI-1 (#81): Grazing / Intensity-Threshold group-header checkbox toggles.

The group HEADER carries a real checkbox -- that checkbox is the on/off
toggle.  The enabling bool stays hidden as the source of truth the wrangler
reads (a hidden bool can't repaint-uncheck when the parameter tree is
hard-disabled mid-run, #56, which a visible disabled checkbox suffers).
``wranglerWidget._install_group_toggles`` wires it both ways:

- user (un)checks the header  -> hidden bool flips, group opens/folds;
- programmatic bool change (session restore) -> checkbox reflects it;
- a manual chevron expand only PEEKS at the options -- it does NOT enable
  the feature (the old expand==toggle coupling silently enabled features
  when the user expanded a group just to look).

These run headless (offscreen Qt) and drive the REAL path: a real
``wranglerWidget`` instance, a real ``ParameterTree``, and the QTreeWidget
``itemChanged`` signal that a user click emits.
"""
import threading

import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import QtWidgets
from pyqtgraph.Qt import QtCore
from pyqtgraph.parametertree import Parameter, ParameterTree

from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import wranglerWidget
from xdart.gui.tabs.static_scan.wranglers.image_wrangler import (
    params as image_params, imageWrangler)
from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler import (
    params as nexus_params, nexusWrangler)

Checked = QtCore.Qt.CheckState.Checked
Unchecked = QtCore.Qt.CheckState.Unchecked


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _make(params_list, group_toggles):
    """A real wranglerWidget driving the real install path on the given
    wrangler's params list (skips the heavy subclass __init__)."""
    w = wranglerWidget('f', threading.Condition())
    w._GROUP_TOGGLES = group_toggles
    w.parameters = Parameter.create(name='p', type='group',
                                    children=params_list)
    tree = ParameterTree()
    tree.setParameters(w.parameters, showTop=False)
    w._install_group_toggles(tree)
    return w, tree


@pytest.mark.parametrize("wrangler_cls, params_list, titles", [
    (imageWrangler, image_params,
     {'GI': ('Grazing', 'Grazing Incidence'),
      'Mask': ('Threshold', 'Intensity Threshold')}),
    (nexusWrangler, nexus_params,
     {'GI': ('Grazing', 'Grazing Incidence')}),
])
def test_header_checkbox_drives_hidden_bool(qapp, wrangler_cls, params_list,
                                            titles):
    w, tree = _make(params_list, wrangler_cls._GROUP_TOGGLES)

    for grp_name, (bool_name, title) in titles.items():
        grp = w.parameters.child(grp_name)
        b = grp.child(bool_name)
        item = next(iter(grp.items))   # the header ParameterItem in the tree

        # Header carries the title + a checkable checkbox; bool hidden & off.
        assert grp.title() == title, f"{grp_name} header title is {grp.title()!r}"
        assert b.opts.get('visible') is False, f"{bool_name} must be hidden (#56)"
        assert b.value() is False
        assert item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable, \
            f"{grp_name} header has no checkbox"
        assert item.checkState(0) == Unchecked

        # User CHECKS the header (real QTreeWidget itemChanged path)
        # -> bool turns ON and the group opens.
        item.setCheckState(0, Checked)
        assert b.value() is True, f"checking {grp_name} didn't enable {bool_name}"
        assert grp.opts.get('expanded') is True

        # User UNCHECKS -> bool OFF, group folds.
        item.setCheckState(0, Unchecked)
        assert b.value() is False, f"unchecking {grp_name} didn't disable {bool_name}"
        assert grp.opts.get('expanded') is False

        # The checkbox must SURVIVE opts changes: pyqtgraph's
        # ParameterItem.optsChanged rebuilds the item flags (updateFlags)
        # and silently drops ItemIsUserCheckable — both our own
        # setOpts(expanded=...) above and the N1 disclosure's show()/hide()
        # hit that path.  (setCheckState in this test works regardless of
        # flags, so only this flag assertion catches the regression.)
        assert item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable, \
            f"{grp_name} checkbox lost after toggling"
        grp.hide()
        grp.show()
        assert item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable, \
            f"{grp_name} checkbox lost after disclosure show/hide"


@pytest.mark.parametrize("wrangler_cls, params_list", [
    (imageWrangler, image_params),
    (nexusWrangler, nexus_params),
])
def test_manual_expand_peeks_without_enabling(qapp, wrangler_cls, params_list):
    """Expanding a toggle group via the chevron must NOT switch it on --
    expand==toggle silently enabled features when the user only peeked."""
    w, tree = _make(params_list, wrangler_cls._GROUP_TOGGLES)

    for grp_name, bool_name in wrangler_cls._GROUP_TOGGLES.items():
        grp = w.parameters.child(grp_name)
        b = grp.child(bool_name)
        item = next(iter(grp.items))

        tree.itemExpandedEvent(item)       # the user opens the chevron
        assert b.value() is False, \
            f"peeking into {grp_name} must not enable {bool_name}"
        assert item.checkState(0) == Unchecked
        tree.itemCollapsedEvent(item)
        assert b.value() is False


@pytest.mark.parametrize("wrangler_cls, params_list", [
    (imageWrangler, image_params),
    (nexusWrangler, nexus_params),
])
def test_programmatic_bool_change_updates_checkbox(qapp, wrangler_cls,
                                                   params_list):
    """Session restore sets the hidden bool directly; the header checkbox
    (and the group fold state) must follow."""
    w, _tree = _make(params_list, wrangler_cls._GROUP_TOGGLES)

    for grp_name, bool_name in wrangler_cls._GROUP_TOGGLES.items():
        grp = w.parameters.child(grp_name)
        b = grp.child(bool_name)
        item = next(iter(grp.items))

        b.setValue(True)
        assert item.checkState(0) == Checked
        assert grp.opts.get('expanded') is True

        b.setValue(False)
        assert item.checkState(0) == Unchecked
        assert grp.opts.get('expanded') is False


def test_image_expand_active_groups_collapses_when_off(qapp):
    """UI-1: ``_expand_active_groups`` folds the toggle groups to match the
    hidden bool (restored OFF collapses, ON expands).  Drives the real method
    bound to a light holder."""
    import types
    root = Parameter.create(name='p', type='group', children=image_params)
    ParameterTree().setParameters(root, showTop=False)
    holder = types.SimpleNamespace(
        _GROUP_TOGGLES=imageWrangler._GROUP_TOGGLES,
        parameters=root,
    )
    run = imageWrangler._expand_active_groups.__get__(holder)

    # GI on, Threshold off -> GI expands, Mask collapses.
    root.child('GI').child('Grazing').setValue(True)
    root.child('Mask').child('Threshold').setValue(False)
    root.child('GI').setOpts(expanded=False)     # pretend folded
    root.child('Mask').setOpts(expanded=True)    # pretend open
    run()
    assert root.child('GI').opts.get('expanded') is True
    assert root.child('Mask').opts.get('expanded') is False
