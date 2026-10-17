"""Explicit built-in page catalog; later pages append descriptors here."""

from __future__ import annotations

from .descriptors import PageDescriptor
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


def _build_stitch_page(services, parent):
    from .stitch_workspace import build_stitch_workspace
    return build_stitch_workspace(services, parent)


def _build_rsm_page(services, parent):
    from .rsm_workspace import build_rsm_workspace
    return build_rsm_workspace(services, parent)


SCATTERING_WORKSPACE_PAGE = PageDescriptor(
    key=SCATTERING_PAGE_KEY,
    label="Scattering Workspace",
    description="Reduction workspace",
    icon_key="scattering-workspace",
    category="workspace",
    order=0,
    lifecycle=PageLifecycle.RETAINED,
    capabilities=frozenset({
        PageCapability.OPEN_FOLDER,
        PageCapability.SETTINGS_PERSISTENCE,
        PageCapability.RUN_CONTROL,
        PageCapability.WRITE_MODE_TOGGLE,
        PageCapability.SLICE_PIN,
        PageCapability.RUN_ACTIVITY,
        PageCapability.APP_MENU_HOSTS,
        PageCapability.PROCESSING_MODE,
    }),
    build=_build_scattering_workspace,
)

STITCH_PAGE = PageDescriptor(
    key=STITCH_TOOL_KEY,
    label="Stitch 1D",
    description="Native finite Stitch workspace",
    icon_key="stitch",
    category="workspace",
    order=100,
    lifecycle=PageLifecycle.RETAINED,
    capabilities=frozenset({
        PageCapability.OPEN_FOLDER, PageCapability.SETTINGS_PERSISTENCE,
        PageCapability.RUN_CONTROL, PageCapability.RUN_ACTIVITY, PageCapability.APP_MENU_HOSTS,
        PageCapability.PROCESSING_MODE, PageCapability.WRITE_MODE_TOGGLE,
    }),
    build=_build_stitch_page,
)

# The native RSM workspace (AUD-9) on the RSM tool's key: Int's and Stitch's mode boxes and the host's
# open_tool("rsm") reach it unchanged. The classic RSM dialog was retired once the page had its workflows
# (AUD-9 stage 2b G10), as the Stitching dialog was.
RSM_PAGE = PageDescriptor(
    key=RSM_TOOL_KEY,
    label="RSM",
    description="Native reciprocal-space map workspace",
    icon_key="rsm",
    category="workspace",
    order=200,
    lifecycle=PageLifecycle.RETAINED,
    capabilities=frozenset({
        PageCapability.OPEN_FOLDER, PageCapability.SETTINGS_PERSISTENCE, PageCapability.RUN_CONTROL,
        PageCapability.RUN_ACTIVITY, PageCapability.APP_MENU_HOSTS, PageCapability.PROCESSING_MODE,
        PageCapability.WRITE_MODE_TOGGLE,
    }),
    build=_build_rsm_page,
)

BUILTIN_PAGES = (SCATTERING_WORKSPACE_PAGE, STITCH_PAGE, RSM_PAGE)

BUILTIN_DESCRIPTORS = BUILTIN_PAGES
