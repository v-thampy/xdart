# -*- coding: utf-8 -*-
# Manual GUI test — not run by pytest automatically.
# Run with: python tests/test_image_widget.py

# Standard Library imports
import sys

if __name__ == '__main__':
    # Qt imports only needed when running as a script
    from PySide6 import QtGui

    if __name__ == '__main__':
        from config import xdart_dir
    if xdart_dir not in sys.path:
        sys.path.append(xdart_dir)

    from xdart.gui.widgets.image_widget import XDImageWidget

    app = QtGui.QApplication(sys.argv)
    test = XDImageWidget()
    test.show()
    app.exec()
