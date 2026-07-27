"""O-1b frozen oracle — R4-A follow-ups under the lazy-discovery contract.

Red at `c4ff402a`, driven through the REAL ``imageWrangler`` selection path: a
real parameter tree, the real ``get_img_fname`` Directory leg, the real preview
discovery and adopt helpers, and real on-disk NXWriter containers.  No stub sits
on the seam under test (CLAUDE.md rule 2).

Coverage, one case per plan item:

* R4A-1 — a Subdirs-enabled root whose matching containers live only in
  subfolders must still populate the GI theta-motor dropdown, within a bounded
  descent budget (<=8 container opens, <=0.75 s);
* R4A-2 — an unusable natural-sort-first direct child (torn, empty, or a
  processed xdart output) must fall back to a usable sibling instead of clearing
  GI/BG options;
* R4A-1(iii) — when nothing is usable anywhere the dropdown stays ``['Manual']``
  with a status hint, and a mid-run arrival is display-only.
"""

from __future__ import annotations

import os
import time
import types
from pathlib import Path
from types import MethodType

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pyqtgraph")

from tests.core.test_bluesky_nexus import (  # noqa: E402
    _write_bluesky_nxwriter,
)


class _FakeSignal:
    def __init__(self):
        self.emissions = []

    def emit(self, *values):
        self.emissions.append(values)

    def connect(self, *_a, **_k):
        pass


def _holder(tmp_path, *, recursive=False, ext="nxs", meta_ext="auto"):
    """A real ``imageWrangler`` method host over a real parameter tree."""
    import xdart.gui.gui_utils  # noqa: F401  # registers 'str_browse'
    from pyqtgraph.parametertree import Parameter

    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import (
        imageWrangler,
        params,
    )

    root = Parameter.create(
        name="image_wrangler", type="group", children=params)
    signal = root.child("Signal")
    signal.child("inp_type").setValue("Image Directory")
    signal.child("img_dir").setValue(str(tmp_path))
    signal.child("img_ext").setValue(ext)
    signal.child("include_subdir").setValue(recursive)
    signal.child("File").setValue("")

    host = types.SimpleNamespace(
        parameters=root,
        img_file="", img_dir=str(tmp_path), img_ext=ext,
        inp_type="Image Directory", single_img=False,
        include_subdir=recursive, meta_ext=meta_ext, meta_dir="",
        file_filter="", scan_parameters=[], motors=[], counters=[],
        incidence_motor="th", poni=None, _bluesky_cols_cache=None,
        source_spec=None, _gi_motor_knowledge_proved=False,
        sigGIMotorOptions=_FakeSignal(),
        # Production-visible status seam.  A private ``status_hints`` list
        # would let a test-only host pass without an operator ever seeing it.
        showLabel=_FakeSignal(),
    )
    for name in (
        "_read_bluesky_source_columns", "get_scan_parameters",
        "set_pars_from_meta", "set_gi_motor_options", "set_gi_th_motor",
        "set_bg_norm_options", "set_bg_matching_options", "exists_meta_file",
        "_sync_meta_ext_to_img_ext", "_directory_metadata_preview_suffixes",
        "_directory_metadata_preview_file", "_adopt_directory_metadata_preview",
        "get_img_fname", "_gi_source_fingerprint",
        "_next_gi_hydration_generation", "_gi_hydration_registry",
        "_retire_gi_hydration_token", "_emit_gi_hydration",
        "_announce_gi_hydration",
    ):
        setattr(host, name, MethodType(getattr(imageWrangler, name), host))
    return host, root


def _motor_choices(root):
    return list(root.child("GI").child("th_motor").opts["limits"])


def _processed_xdart_nxs(path: Path) -> Path:
    """A processed xdart output: valid HDF5, no raw detector frames."""
    import h5py

    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        proc = entry.create_group("xdart_process")
        proc.attrs["program"] = "ssrl_xrd_tools"
        entry.create_group("integrated")
    return path


# --------------------------------------------------------------------------- #
# R4A-2 — fall back past an unusable natural-sort-first direct child.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("first_kind", ["empty", "processed"])
def test_unusable_first_child_falls_back_to_a_usable_sibling(
        tmp_path, first_kind):
    """R4A-2.  The preview inspected ONLY ``candidates[0]``, so a torn or
    processed file sorting first cleared the GI/BG option lists even though a
    usable sibling was right there."""
    first = tmp_path / "aaa_scan.nxs"
    if first_kind == "empty":
        first.write_bytes(b"")
    else:
        _processed_xdart_nxs(first)
    _write_bluesky_nxwriter(tmp_path / "bbb_scan.nxs")

    host, root = _holder(tmp_path)
    host.get_img_fname()

    choices = _motor_choices(root)
    assert "hy" in choices, (
        f"the usable sibling's motor was not adopted; dropdown={choices}")
    assert host.motors, "motor list cleared despite a usable sibling"


# --------------------------------------------------------------------------- #
# R4A-1 — nested-only Subdirs, within a bounded descent budget.
# --------------------------------------------------------------------------- #

def test_nested_only_subdirs_populates_motors_when_recursive(tmp_path):
    """R4A-1.  A Subdirs-enabled root whose matching containers live only in
    subfolders left the dropdown ``['Manual']`` and emitted an empty option
    list, where the pre-R4-A recursive seed populated real motors."""
    child = tmp_path / "day1" / "sample_a"
    child.mkdir(parents=True)
    _write_bluesky_nxwriter(child / "scan_0001.nxs")

    host, root = _holder(tmp_path, recursive=True)
    host.get_img_fname()

    choices = _motor_choices(root)
    assert "hy" in choices, (
        f"nested container's motor was not adopted; dropdown={choices}")


def test_nested_descent_is_bounded_in_opens_and_wall_time(tmp_path):
    """R4A-1's budget: at most 8 container opens AND at most 0.75 s.

    A wide nested root with no usable container anywhere must stop early rather
    than walk the tree.
    """
    for index in range(40):
        sub = tmp_path / f"dir_{index:03d}"
        sub.mkdir()
        (sub / f"torn_{index:03d}.nxs").write_bytes(b"")

    host, root = _holder(tmp_path, recursive=True)
    opens = []
    real_reader = host._read_bluesky_source_columns

    def counting_reader(path):
        opens.append(str(path))
        return real_reader(path)

    host._read_bluesky_source_columns = counting_reader

    started = time.monotonic()
    host.get_img_fname()
    elapsed = time.monotonic() - started

    # Discriminating in BOTH directions: a descent must actually happen (this
    # assertion is what makes the budget meaningful rather than vacuously true
    # while no descent exists at all), and it must stop inside the budget.
    assert opens, (
        "no nested container was opened, so the budget below proves nothing")
    assert len(opens) <= 8, f"descent opened {len(opens)} containers: {opens}"
    assert elapsed <= 0.75, f"descent took {elapsed:.3f} s"
    # nothing usable -> the dropdown stays Manual-only and says so
    assert _motor_choices(root) == ["Manual"], _motor_choices(root)


def test_non_recursive_root_does_not_descend(tmp_path):
    """The lazy contract: Subdirs OFF must not walk subfolders at all."""
    child = tmp_path / "nested"
    child.mkdir()
    _write_bluesky_nxwriter(child / "scan_0001.nxs")

    host, root = _holder(tmp_path, recursive=False)
    opens = []
    real_reader = host._read_bluesky_source_columns
    host._read_bluesky_source_columns = (
        lambda path: opens.append(str(path)) or real_reader(path))

    host.get_img_fname()

    assert opens == [], f"a non-recursive preview descended: {opens}"
    assert _motor_choices(root) == ["Manual"]


# --------------------------------------------------------------------------- #
# R4A-1(iii) — nothing usable: Manual plus a hint, and mid-run arrivals are
# display-only.
# --------------------------------------------------------------------------- #

def test_no_usable_container_leaves_manual_and_reports_a_hint(tmp_path):
    (tmp_path / "torn_0001.nxs").write_bytes(b"")

    host, root = _holder(tmp_path)
    host.get_img_fname()

    assert _motor_choices(root) == ["Manual"]
    assert host._gi_motor_knowledge_proved is False, (
        "an unusable preview must leave motor knowledge UNKNOWN, not "
        "known-empty, so an explicit GI motor is not resolved to Manual")
    hints = [values[0] for values in host.showLabel.emissions if values]
    assert any("motor" in str(hint).lower() for hint in hints), (
        "the operator got no hint that motors will fill during the run; "
        f"hints={hints}")


def test_first_jit_container_hydrates_display_without_changing_frozen_motor(
        tmp_path):
    """R4A-1(iii).  The real worker's first JIT-classified NXWriter source
    publishes its discovered motors through the wrapper's existing hydration
    signal, while the exact configuration admitted before discovery remains
    immutable.

    The provider assertions make the red discriminating: classification really
    found ``hy``; the missing behavior is delivery to the GUI, not an unreadable
    fixture or an empty metadata table.
    """
    import threading

    from pyqtgraph.Qt import QtWidgets

    from tests.xdart._accepted_run import (
        accepted_run,
        admitted_worker,
        directory_source,
        gi_intent,
    )
    import xdart.gui.gui_utils  # noqa: F401  # registers str_browse
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import (
        imageWrangler,
    )
    from xdart.modules.live import LiveScan

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    root = tmp_path / "raw"
    nested = root / "day1"
    nested.mkdir(parents=True)
    container = nested / "scan_0001.nxs"
    _write_bluesky_nxwriter(container)
    output = tmp_path / "processed"
    output.mkdir()
    scan = LiveScan(
        "preview", data_file=str(output / "preview.nxs"), static=True)
    wrapper = imageWrangler("", threading.RLock(), scan)
    worker = wrapper.thread
    try:
        signal = wrapper.parameters.child("Signal")
        signal.child("inp_type").setValue("Image Directory")
        signal.child("img_dir").setValue(str(root))
        signal.child("img_ext").setValue("nxs")
        signal.child("include_subdir").setValue(True)
        signal.child("File").setValue("")
        wrapper.get_img_fname()
        assert _motor_choices(wrapper.parameters) == ["Manual"]

        frozen = accepted_run(
            save_path=str(output),
            source_spec=directory_source(root, ext="nxs", recursive=True),
            gi=gi_intent(enabled=True, incidence_motor="halpha"),
        )
        admitted_worker(wrapper, frozen=frozen)
        admitted_worker(worker, frozen=frozen)
        effective_before = frozen.gi.effective_motor
        emitted = []
        wrapper.sigGIMotorOptions.connect(emitted.append)

        worker._eiger_master_path = str(container)
        worker._eiger_open_master(frozen, str(container))
        worker._frame_scan_info(frozen, str(container), 0)
        assert worker._eiger_provider is not None
        assert "hy" in worker._eiger_provider.motors()
        app.processEvents()

        assert emitted
        assert "hy" in tuple(emitted[-1].motors)
        assert "hy" in _motor_choices(wrapper.parameters)
        assert frozen.gi.effective_motor == effective_before == "halpha"
    finally:
        worker._eiger_close_master()
        wrapper.close()
        wrapper.deleteLater()
        app.processEvents()
