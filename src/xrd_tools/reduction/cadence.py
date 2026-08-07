# -*- coding: utf-8 -*-
"""Flush cadence — COMPATIBILITY RE-EXPORT (H10-C2-A).

The cadence decision now lives behind the one session run-policy owner,
:mod:`xrd_tools.session.policy`; this module keeps the historical import path
working and defines nothing.  Still pure: the session policy is stdlib-only,
so the purity guard that loads this file by path pulls no Qt/h5py/numpy.
"""
from __future__ import annotations

from xrd_tools.session.policy import FlushPolicy

__all__ = ["FlushPolicy"]
