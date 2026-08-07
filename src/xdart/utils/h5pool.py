"""Compatibility adapter for the headless process-wide HDF5 pool."""

from xrd_tools.session.io_coordination import H5FilePool, get_pool

__all__ = ["H5FilePool", "get_pool"]
