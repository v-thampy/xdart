"""Lazy adapter for the native Stitch workspace."""


def build_stitch_workspace(services, parent):
    from xdart.gui.tabs.stitch.page import build_stitch_page
    return build_stitch_page(services, parent)
