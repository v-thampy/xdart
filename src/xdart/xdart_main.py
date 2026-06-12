# -*- coding: utf-8 -*-
"""Thin console entry for the ``xdart`` script.

The single xrd-tools distribution installs this entry point even without
the GUI extra, so the Qt stack import happens lazily here: a missing
PySide6/pyqtgraph yields a friendly install hint instead of a raw
ImportError traceback.  All real startup logic lives in
:mod:`xdart._gui_main`.
"""
import sys


def main():
    try:
        import PySide6   # noqa: F401  -- probe the GUI stack cheaply
        import pyqtgraph  # noqa: F401
    except ImportError:
        print(
            'xdart requires the GUI extra: pip install "xrd-tools[gui]"',
            file=sys.stderr,
        )
        return 1
    from xdart._gui_main import run
    return run()


if __name__ == "__main__":
    sys.exit(main())
