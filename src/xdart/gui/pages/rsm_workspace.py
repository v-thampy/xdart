"""Lazy adapter for the native RSM workspace."""


def build_rsm_workspace(services, parent):
    from xdart.gui.tabs.rsm.page import build_rsm_page
    return build_rsm_page(services, parent)
