"""O-1a-W1R Phase 0/2 — the run-owned carrier census and its owner-graph guard.

Simple facts about the ACTUAL production tree and owner graph (CLAUDE.md rule 9
— explicitly NOT a general Python taint analyzer):

1. every run-owned execution carrier named by review §39.5 Phase 0 item 1 is
   classified, and the classification table in this module is the frozen Phase 0
   inventory;
2. every ``FROZEN_EXECUTION_POLICY`` carrier on the two worker classes is a
   read-only projection of the accepted frozen object, not a mutable instance
   attribute -- so no execution read of the corresponding mutable field remains;
3. a write to such a name during an admitted run lands in the zero-reader
   display/compatibility slot and changes nothing an execution read sees;
4. ``RUNTIME_SOURCE_CURSOR`` carriers stay writable (frame cursor, discovered
   name, reader state) and are NOT projections.

This file is Qt-light: it imports the worker classes but constructs no widget.
"""

from __future__ import annotations

import inspect

import pytest

from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread
from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import nexusThread
from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import (
    FrozenRunProjection,
)
from xrd_tools.session import RunIntent


# --------------------------------------------------------------------------- #
# The frozen Phase 0 inventory (review §39.5 Phase 0 items 1-2).
#
# Exactly one class per carrier.  No unclassified read proceeds.
# --------------------------------------------------------------------------- #

FROZEN_EXECUTION_POLICY = {
    # output
    "write_mode": "frozen.output_mode",
    "h5_dir": "frozen.save_path",
    "project_folder": "frozen.project_root",
    # scientific policy
    "apply_threshold": "frozen.threshold.apply_threshold",
    "threshold_min": "frozen.threshold.threshold_min",
    "threshold_max": "frozen.threshold.threshold_max",
    "mask_sentinel": "frozen.threshold.mask_saturation",
    "mask_file": "frozen.mask_file",
    "poni_file": "frozen.poni_file",
    # GI
    "gi": "frozen.gi.enabled",
    "incidence_motor": "frozen.gi.scan_incidence_motor",
    "sample_orientation": "frozen.gi.sample_orientation",
    "tilt_angle": "frozen.gi.tilt_angle",
    # mode / parallelism
    "live_mode": "frozen.live_mode",
    "batch_mode": "frozen.batch_mode",
    "max_cores": "frozen.max_cores",
    "xye_only": "frozen.run_options['xye_only']",
    "series_average": "frozen.run_options['series_average']",
    "meta_ext": "frozen.run_options['meta_ext']",
    # source family / traversal (never the frame cursor)
    "inp_type": "frozen.source family",
    "img_ext": "frozen.source format token",
    "img_dir": "frozen.source root",
    "single_img": "frozen.source kind",
    "include_subdir": "frozen.source.recursive",
    "file_filter": "frozen.source.name_filter",
    "source_spec": "frozen.thaw_source_spec()",
    "scan_args": "frozen.scan_args()",
}

RUNTIME_SOURCE_CURSOR = {
    "img_file": "the frame/container currently being read",
    "img_fnames": "the discovered pending queue",
    "scan_name": "the discovered scan name",
    "processed": "the processed-frame cursor",
    "poni": "the installed PONI/integrator object (qualified at worker entry)",
    "detector": "the pyFAI detector built from the installed PONI",
    "mask": "the resolved flat mask indices",
    "meta_dir": "the resolved metadata directory",
    "source_base": "the portable source root stamped by the writer",
}

DISPLAY_ONLY = {
    "bg_file": "background selection is display/setup state (no W-1 owner)",
    "bg_type": "background selection is display/setup state (no W-1 owner)",
}

DEFERRED_OUTSIDE_W1 = {
    "skip_2d": (
        "the broader NeXus display-scan/browse ownership split stays R4-C; only "
        "the execution alias in nexusThread._run_impl comes into W-1R and is "
        "guarded by test_w1r_frozen_authority.py"
    ),
}


def test_the_phase_0_inventory_partitions_every_named_carrier():
    """§39.5 Phase 0 item 2 — exactly one class per carrier, no overlap."""
    tables = (
        FROZEN_EXECUTION_POLICY,
        RUNTIME_SOURCE_CURSOR,
        DISPLAY_ONLY,
        DEFERRED_OUTSIDE_W1,
    )
    seen: dict[str, int] = {}
    for index, table in enumerate(tables):
        for name in table:
            assert name not in seen, (
                f"{name!r} is classified twice (tables {seen[name]} and {index})")
            seen[name] = index
    # Every carrier §39.5 Phase 0 item 1 enumerates must be present.
    required = {
        "inp_type", "img_ext", "img_dir", "img_file", "single_img",
        "include_subdir", "file_filter", "source_spec", "source_base",
        "poni", "poni_file", "h5_dir", "project_folder", "write_mode",
        "live_mode", "batch_mode", "xye_only", "series_average", "skip_2d",
        "max_cores", "mask_file", "apply_threshold", "threshold_min",
        "threshold_max", "mask_sentinel", "gi", "incidence_motor",
        "sample_orientation", "tilt_angle", "scan_args", "meta_ext",
    }
    missing = sorted(required - set(seen))
    assert missing == [], f"unclassified run-owned carriers: {missing}"


@pytest.mark.parametrize("worker", [imageThread, nexusThread])
def test_frozen_policy_carriers_are_projections_on_both_workers(worker):
    """§39.5 Phase 2 item 3 — the mutable execution field no longer exists.

    A ``FrozenRunProjection`` descriptor resolves the name through the exact
    accepted frozen object; the historical name is kept so no consumer needs an
    indirection, but it is not an execution INPUT any more.
    """
    declared = {
        name
        for name in FROZEN_EXECUTION_POLICY
        if isinstance(inspect.getattr_static(worker, name, None),
                      FrozenRunProjection)
    }
    # Both workers must project every policy carrier they actually consume.
    consumed = _carriers_read_by(worker)
    missing = sorted((consumed & set(FROZEN_EXECUTION_POLICY)) - declared)
    assert missing == [], (
        f"{worker.__name__} still reads mutable execution carriers: {missing}")


@pytest.mark.parametrize("worker", [imageThread, nexusThread])
def test_runtime_cursor_carriers_are_not_projections(worker):
    """§39.5 Phase 2 item 4 — runtime values stay real, writable runtime data."""
    for name in RUNTIME_SOURCE_CURSOR:
        static = inspect.getattr_static(worker, name, None)
        assert not isinstance(static, FrozenRunProjection), (
            f"{worker.__name__}.{name} is a runtime cursor, not frozen policy")


def _carriers_read_by(worker) -> set[str]:
    """Names in the carrier tables that ``worker``'s own module reads on self."""
    import ast

    module = inspect.getmodule(worker)
    tree = ast.parse(inspect.getsource(module))
    names = set(FROZEN_EXECUTION_POLICY) | set(RUNTIME_SOURCE_CURSOR)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in names \
                and isinstance(node.ctx, ast.Load) \
                and isinstance(node.value, ast.Name) \
                and node.value.id == "self":
            found.add(node.attr)
    return found


def test_a_display_write_during_an_admitted_run_has_zero_execution_readers():
    """§39.5 Phase 2 item 6 — the backward write is a one-way projection.

    Uses a bare object with the projection descriptors bound through the real
    worker class, so this asserts the production descriptor, not a copy.
    """
    frozen = RunIntent(
        processing_mode="Int 2D",
        output_mode="Append",
        save_path="/accepted/output",
        max_cores=3,
    ).freeze()

    class _Host(imageThread):  # real descriptors, no Qt construction
        def __init__(self):  # noqa: D107
            pass

    host = _Host()
    host.run_configuration = frozen

    host.write_mode = "Overwrite"
    host.h5_dir = "/late/output"
    host.max_cores = 99
    host.apply_threshold = True

    assert host.write_mode == "Append"
    assert host.h5_dir == "/accepted/output"
    assert host.max_cores == 3
    assert host.apply_threshold is False
