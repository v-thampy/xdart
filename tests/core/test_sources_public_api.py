# -*- coding: utf-8 -*-
"""R1 step 7 — the new public names resolve through xrd_tools.sources's
lazy __getattr__/dir(), the same seam every existing export already uses."""

from __future__ import annotations

import xrd_tools.sources as sources

_NEW_NAMES = (
    "SourceFormatAdapter",
    "register_adapter",
    "get_adapter",
    "adapter_for_kind",
    "all_adapters",
    "Candidate",
    "enumerate_candidates",
    "DirectoryIndex",
    "Snapshot",
    "IndexDelta",
    "RetryState",
    "DEFAULT_RETRY_DEADLINE",
    "ProbeResult",
    "ProbeState",
    "nxwriter_finalization_policy",
)


def test_new_r1_names_are_lazily_resolvable_attributes():
    for name in _NEW_NAMES:
        assert hasattr(sources, name), name


def test_new_r1_names_are_in_dir_and_all():
    for name in _NEW_NAMES:
        assert name in dir(sources), name
        assert name in sources.__all__, name


def test_directory_index_is_the_same_class_via_package_and_submodule():
    from xrd_tools.sources.directory_index import DirectoryIndex as _Direct

    assert sources.DirectoryIndex is _Direct
