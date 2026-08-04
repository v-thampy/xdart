"""Explicit built-in page catalog; later pages append descriptors here."""

from __future__ import annotations

from .descriptors import PageDescriptor
from .values import PageCapability, PageKey, PageLifecycle


DEFAULT_PAGE_KEY = PageKey("static-scan")


def _build_legacy_static(services, parent):
    from .legacy_static import build_legacy_static
    return build_legacy_static(services, parent)


LEGACY_STATIC_PAGE = PageDescriptor(
    key=DEFAULT_PAGE_KEY,
    label="Static Scan",
    description="Legacy static-scan reduction workspace",
    icon_key="static-scan",
    category="workspace",
    order=0,
    lifecycle=PageLifecycle.EXIT_ONLY,
    capabilities=frozenset({
        PageCapability.SETTINGS_PERSISTENCE,
        PageCapability.RUN_CONTROL,
        PageCapability.WRITE_MODE_TOGGLE,
        PageCapability.SLICE_PIN,
        PageCapability.RUN_ACTIVITY,
        PageCapability.LAYOUT_DIAGNOSTICS,
    }),
    build=_build_legacy_static,
)

def _build_scattering_workspace(services, parent):
    from .scattering_workspace import build_scattering_workspace
    return build_scattering_workspace(services, parent)


SCATTERING_WORKSPACE_PAGE = PageDescriptor(
    key=PageKey("scattering-workspace"),
    label="Scattering Workspace",
    description="vNext scattering reduction workspace (opt-in)",
    icon_key="scattering-workspace",
    category="workspace",
    order=1,
    lifecycle=PageLifecycle.EXIT_ONLY,
    capabilities=frozenset({
        PageCapability.OPEN_FOLDER,
        PageCapability.RUN_CONTROL,
        PageCapability.WRITE_MODE_TOGGLE,
        PageCapability.RUN_ACTIVITY,
        PageCapability.APP_MENU_HOSTS,
    }),
    build=_build_scattering_workspace,
)

BUILTIN_PAGES = (LEGACY_STATIC_PAGE, SCATTERING_WORKSPACE_PAGE)
