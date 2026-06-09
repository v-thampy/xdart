"""UI-1 (#81): Grazing / Intensity-Threshold group-header click-to-expand toggles.

The group HEADER expand/collapse IS the on/off toggle; the enabling bool is
hidden -- it stays the source of truth the wrangler reads, but a hidden bool
can't repaint-uncheck when the parameter tree is hard-disabled mid-run (#56,
which a visible disabled checkbox suffers).  pyqtgraph's ``syncExpanded`` mirrors
the user's tree expand/collapse into the param opts, and the wrangler's
``_sync_group_toggle_from_expand`` slot maps that onto the hidden bool.

These run headless (offscreen Qt); they drive the REAL pyqtgraph expand path
(``ParameterTree.itemExpandedEvent`` -> ``expandedChangedEvent`` -> syncExpanded
-> ``sigOptionsChanged`` -> the wrangler slot) rather than poking the bool
directly, so a regression in any link of that chain fails here.
"""
import types

import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import QtWidgets
from pyqtgraph.parametertree import Parameter, ParameterTree

from xdart.gui.tabs.static_scan.wranglers.image_wrangler import (
    params as image_params, imageWrangler)
from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler import (
    params as nexus_params, nexusWrangler)


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


_KEEPALIVE = []


def _wire(wrangler_cls, root):
    """Connect the wrangler's real ``_sync_group_toggle_from_expand`` slot to the
    toggle groups, exactly as the wrangler does in its signal setup -- but bound
    to a light holder so we skip the heavy widget ``__init__``.  (A plain closure,
    not a SimpleNamespace-bound method, which PySide6's signal connect rejects;
    kept alive past the connect so it isn't GC'd.)"""
    holder = types.SimpleNamespace(
        _EXPAND_TOGGLE_GROUPS=wrangler_cls._EXPAND_TOGGLE_GROUPS,
        parameters=root,
        _save_to_session=lambda *a, **k: None,
    )

    def slot(param, opts):
        wrangler_cls._sync_group_toggle_from_expand(holder, param, opts)

    _KEEPALIVE.append(slot)
    for grp in wrangler_cls._EXPAND_TOGGLE_GROUPS:
        root.child(grp).sigOptionsChanged.connect(slot)
    return holder


@pytest.mark.parametrize("wrangler_cls, params_list, titles", [
    (imageWrangler, image_params,
     {'GI': ('Grazing', 'Grazing Incidence'),
      'Mask': ('Threshold', 'Intensity Threshold')}),
    (nexusWrangler, nexus_params,
     {'GI': ('Grazing', 'Grazing Incidence')}),
])
def test_group_header_toggle_drives_hidden_bool(qapp, wrangler_cls, params_list, titles):
    root = Parameter.create(name='p', type='group', children=params_list)
    tree = ParameterTree()
    tree.setParameters(root, showTop=False)
    _wire(wrangler_cls, root)

    for grp_name, (bool_name, title) in titles.items():
        grp = root.child(grp_name)
        b = grp.child(bool_name)

        # Header carries the title + the sync flag; the bool is hidden & off.
        assert grp.opts.get('syncExpanded') is True, f"{grp_name} not syncExpanded"
        assert grp.title() == title, f"{grp_name} header title is {grp.title()!r}"
        assert b.opts.get('visible') is False, f"{bool_name} must be hidden (#56)"
        assert b.value() is False

        item = next(iter(grp.items))   # the ParameterItem in the tree

        # User EXPANDS the header (real path) -> bool turns ON.
        tree.itemExpandedEvent(item)
        assert b.value() is True, f"expanding {grp_name} didn't enable {bool_name}"

        # User COLLAPSES the header -> bool turns OFF.
        tree.itemCollapsedEvent(item)
        assert b.value() is False, f"collapsing {grp_name} didn't disable {bool_name}"


def test_image_expand_active_groups_collapses_when_off(qapp):
    """UI-1: ``_expand_active_groups`` syncs expanded == the hidden bool BOTH
    ways for the header-toggle groups (so a restored OFF state collapses, ON
    expands).  Drives the real method bound to a light holder."""
    root = Parameter.create(name='p', type='group', children=image_params)
    ParameterTree().setParameters(root, showTop=False)
    holder = types.SimpleNamespace(
        _EXPAND_TOGGLE_GROUPS=imageWrangler._EXPAND_TOGGLE_GROUPS,
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
