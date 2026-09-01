"""Explicit built-in page catalog; later pages append descriptors here."""

from __future__ import annotations

from .descriptors import PageDescriptor, ToolDescriptor
from .values import (
    PageCapability,
    PageLifecycle,
    RSM_TOOL_KEY,
    SCATTERING_PAGE_KEY,
    STITCH_TOOL_KEY,
)


DEFAULT_PAGE_KEY = SCATTERING_PAGE_KEY


def _build_scattering_workspace(services, parent):
    from .scattering_workspace import build_scattering_workspace
    return build_scattering_workspace(services, parent)


def _build_stitch_tool(services, parent):
    from xdart.gui.tools.stitch_tool import build_stitch_tool

    return build_stitch_tool(services, parent)


def _build_rsm_tool(services, parent):
    from xdart.gui.tools.rsm_tool import build_rsm_tool

    return build_rsm_tool(services, parent)


SCATTERING_WORKSPACE_PAGE = PageDescriptor(
    key=SCATTERING_PAGE_KEY,
    label="Scattering Workspace",
    description="Reduction workspace",
    icon_key="scattering-workspace",
    category="workspace",
    order=0,
    lifecycle=PageLifecycle.EXIT_ONLY,
    capabilities=frozenset({
        PageCapability.OPEN_FOLDER,
        PageCapability.SETTINGS_PERSISTENCE,
        PageCapability.RUN_CONTROL,
        PageCapability.WRITE_MODE_TOGGLE,
        PageCapability.SLICE_PIN,
        PageCapability.RUN_ACTIVITY,
        PageCapability.APP_MENU_HOSTS,
    }),
    build=_build_scattering_workspace,
)

BUILTIN_PAGES = (SCATTERING_WORKSPACE_PAGE,)

STITCH_TOOL = ToolDescriptor(
    key=STITCH_TOOL_KEY,
    label="Stitching",
    description="Combine an exact scan selection into one stitched pattern",
    icon_key="stitch",
    category="analysis",
    order=100,
    build=_build_stitch_tool,
    tool_kind="analysis",
)

RSM_TOOL = ToolDescriptor(
    key=RSM_TOOL_KEY,
    label="Reciprocal Space Map",
    description="Grid one exact psic SPEC scan in reciprocal space",
    icon_key="rsm",
    category="analysis",
    order=110,
    build=_build_rsm_tool,
    tool_kind="analysis",
)

BUILTIN_DESCRIPTORS = (*BUILTIN_PAGES, STITCH_TOOL, RSM_TOOL)
