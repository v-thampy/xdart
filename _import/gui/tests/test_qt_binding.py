def test_pyqtgraph_uses_pyside6():
    import pyqtgraph.Qt as pg_qt

    assert pg_qt.QT_LIB == "PySide6"
