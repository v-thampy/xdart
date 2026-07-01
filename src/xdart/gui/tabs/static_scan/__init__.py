"""Static-scan GUI package.

Keep package import light so pure helper modules such as
``xdart.gui.tabs.static_scan.controls_logic`` remain importable in a Qt-free
test process.  The heavy widget is loaded only when requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .static_scan_widget import staticWidget as staticWidget


def __getattr__(name: str):
    if name == "staticWidget":
        from .static_scan_widget import staticWidget

        return staticWidget
    raise AttributeError(name)


__all__ = ["staticWidget"]
