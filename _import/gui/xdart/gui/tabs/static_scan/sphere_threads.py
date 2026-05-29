"""Deprecated module path — use :mod:`.scan_threads`.

Thin shim kept after the sphere/arch → scan/frame rename so older
import sites referencing ``sphere_threads`` still resolve.
"""
from .scan_threads import fileHandlerThread, integratorThread

__all__ = ["fileHandlerThread", "integratorThread"]
