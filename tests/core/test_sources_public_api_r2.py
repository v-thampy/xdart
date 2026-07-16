# -*- coding: utf-8 -*-
"""R2 — the new descriptor / read-plan public names resolve through
xrd_tools.sources's lazy ``__getattr__``/``dir()`` seam, exactly like the R1
exports (parity with ``test_sources_public_api.py``)."""

from __future__ import annotations

import xrd_tools.sources as sources

_R2_NAMES = (
    "ContainerDescriptor",
    "describe_container",
    "describe_container_from_open",
    "ReadPlan",
    "plan_reads",
)


def test_r2_names_are_lazily_resolvable():
    for name in _R2_NAMES:
        assert hasattr(sources, name), name


def test_r2_names_are_in_dir_and_all():
    for name in _R2_NAMES:
        assert name in dir(sources), name
        assert name in sources.__all__, name


def test_container_descriptor_same_class_via_package_and_submodule():
    from xrd_tools.sources.descriptor import ContainerDescriptor as _Direct

    assert sources.ContainerDescriptor is _Direct


def test_plan_reads_same_function_via_package_and_submodule():
    from xrd_tools.sources.read_plan import plan_reads as _direct

    assert sources.plan_reads is _direct
