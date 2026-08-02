from __future__ import annotations

import json
from pathlib import Path


_INVENTORY = (
    Path(__file__).with_name("fixtures")
    / "e3_shell_inventory_ff5380a7.json"
)


def _inventory() -> dict[str, object]:
    return json.loads(_INVENTORY.read_text(encoding="utf-8"))


def test_e3_ui0_inventory_is_anchored_to_exact_canonical_commit() -> None:
    inventory = _inventory()

    assert inventory["canonical_commit"] == (
        "ff5380a7716df5927c19fe6f96c9cfce02cd12cd"
    )
    assert inventory["capture_checksum_sha256"] == (
        "ece35e271a3c921b718845f247a214459f5d0403c1a5fd9906de8e1d134d3335"
    )
    assert inventory["viewports"] == [
        [1440, 900],
        [1920, 1080],
        [1024, 900],
    ]


def test_e3_ui0_inventory_freezes_three_columns_and_operator_surface() -> None:
    inventory = _inventory()
    columns = inventory["columns"]
    assert [column["id"] for column in columns] == [
        "browser",
        "scientific",
        "controls",
    ]

    browser, scientific, controls = (
        set(column["items"]) for column in columns
    )
    assert {
        "File",
        "Config",
        "Help",
        "DATA BROWSER",
        "Date",
        "Refresh",
        "Scans",
        "Frames",
        "Show All",
        "Metadata",
        "Auto Last",
        "Peak Fitting",
        "Phase Fitting",
        "Plot Metadata",
    } <= browser
    assert {
        "Norm Channel",
        "Set BG",
        "raw",
        "cake",
        "Single",
        "Overlay",
        "Waterfall",
        "frame selector",
        "global progress",
    } <= scientific
    assert {
        "1 PROJECT",
        "2 EXPERIMENT",
        "3 SOURCE",
        "4 PROCESSING",
        "Standard",
        "Grazing",
        "Batch",
        "Cores",
        "Run",
        "Stop",
        "Append",
        "Overwrite",
    } <= controls


def test_e3_ui0_inventory_freezes_memory_and_authority_boundaries() -> None:
    inventory = _inventory()
    presentation = inventory["presentation"]
    ownership = inventory["ownership"]

    assert presentation["history_scope"] == "all retained cheap 1-D rows"
    assert presentation["heavy_selection"] == (
        "emit hydration and preserve last qualified heavy presentation"
    )
    assert ownership["shell"] == [
        "Qt widgets",
        "last scalar reconciliation key",
    ]
    assert {
        "scan",
        "store",
        "source",
        "file handle",
        "executor",
        "context",
        "worker",
        "writer",
        "ParameterTree",
        "legacy staticWidget",
        "legacy H5Viewer",
        "legacy displayFrameWidget",
    } <= set(ownership["forbidden"])
