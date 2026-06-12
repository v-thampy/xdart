"""Reserved entry point for a future standalone xrd_tools GUI.

The graphical surfaces that ship today live in two places:

* The Jupyter-widget viewers in this package
  (``powder_1d_viewer``, ``powder_2d_viewer``, ``rsm_viewer``,
  ``napari_viewer``), intended for use inside notebooks.
* `xdart <https://github.com/v-thampy/xdart>`_, a standalone PySide6
  desktop application that imports ``xrd_tools`` as a dependency.

This module exists only as a reserved entry point in case a
library-level standalone launcher is added in the future.
"""
from __future__ import annotations


def main() -> None:
    """Not implemented. Use the Jupyter widgets or ``xdart`` instead."""
    raise NotImplementedError(
        "xrd_tools has no standalone GUI launcher. "
        "Use the Jupyter widget viewers in xrd_tools.gui.* inside a "
        "notebook, or install and run xdart for a full desktop app: "
        "https://github.com/v-thampy/xdart"
    )


if __name__ == "__main__":
    main()
