from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest

from tests.xdart.test_o3n_execution_owner import (
    _select_nexus,
    _start_recorder,
    _write_poni,
    qapp,
    widget,
)
from tests.xdart.test_o3n_nexus_freeze_identity import _zero_delta_snapshot
from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (
    nexusThread,
)
from xrd_tools.session.run_configuration import RunConfigurationRefused


def _arm_container_without_frames(wrangler, tmp_path):
    source = tmp_path / "empty.nxs"
    with h5py.File(source, "w") as handle:
        group = handle.create_group("entry")
        group.attrs["NX_class"] = "NXentry"
    output = tmp_path / "out"
    output.mkdir()
    wrangler.parameters.child("Calibration", "poni_file").setValue(
        _write_poni(tmp_path / "cal.poni")
    )
    wrangler.parameters.child("NeXus File", "nexus_file").setValue(str(source))
    wrangler.parameters.child("NeXus File", "entry").setValue("entry")
    wrangler.parameters.child("Output", "h5_dir").setValue(str(output))
    wrangler.parameters.child("Project", "project_folder").setValue(str(tmp_path))
    return source, output


def test_strict_preflight_rejects_an_existing_dataset_as_entry(
    widget, tmp_path, monkeypatch
):
    wrangler = _select_nexus(widget)
    source = tmp_path / "dataset-entry.nxs"
    with h5py.File(source, "w") as handle:
        actual = handle.create_group("actual")
        actual.attrs["NX_class"] = "NXentry"
        handle.create_dataset("selected", data=[1])
    output = tmp_path / "out"
    output.mkdir()
    wrangler.parameters.child("Calibration", "poni_file").setValue(
        _write_poni(tmp_path / "cal.poni")
    )
    wrangler.parameters.child("NeXus File", "nexus_file").setValue(str(source))
    wrangler.parameters.child("NeXus File", "entry").setValue("selected")
    wrangler.parameters.child("Output", "h5_dir").setValue(str(output))
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()
    thread = wrangler.thread
    target = nexusThread._frozen_source_target(thread.run_configuration)

    with pytest.raises(RunConfigurationRefused):
        nexusThread._preflight_execution_target(thread, target)


def test_direct_run_impl_performs_strict_preflight_before_content_read(
    widget, tmp_path, monkeypatch
):
    wrangler = _select_nexus(widget)
    source, output = _arm_container_without_frames(wrangler, tmp_path)
    wrangler.parameters.child("NeXus File", "entry").setValue("missing")
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()
    thread = wrangler.thread
    frozen = thread.run_configuration
    observed = []

    class StopProbe(BaseException):
        pass

    def content_read(*args, **kwargs):
        observed.append((args, kwargs))
        raise StopProbe

    monkeypatch.setattr(
        "xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread.read_nexus",
        content_read,
    )
    with pytest.raises(RunConfigurationRefused):
        nexusThread._run_impl(thread, frozen)
    assert observed == []
    assert list(output.iterdir()) == []


def test_overwrite_does_not_delete_prior_result_until_raw_stack_is_proved(
    widget, tmp_path, monkeypatch
):
    wrangler = _select_nexus(widget)
    source, output = _arm_container_without_frames(wrangler, tmp_path)
    widget.controls.set_write_mode("Overwrite")
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()
    thread = wrangler.thread
    target = nexusThread._frozen_source_target(thread.run_configuration)
    prior = Path(target.output_path)
    prior.write_bytes(b"prior-result")

    thread.run()

    assert prior.read_bytes() == b"prior-result"


def test_missing_frozen_poni_cannot_reuse_a_stale_worker_poni(
    widget, tmp_path, monkeypatch
):
    wrangler = _select_nexus(widget)
    source, output = _arm_container_without_frames(wrangler, tmp_path)
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()
    thread = wrangler.thread
    frozen = thread.run_configuration
    # Construct an accepted-shaped configuration with no calibration values,
    # while the worker still carries the prior PONI object.
    from dataclasses import replace

    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (
        PreparedNexusExecution,
    )

    no_poni = replace(frozen, poni_file="", _poni_values=("none",))
    # O-3N.R.2 §17.1: adoption consumes the prepared ENVELOPE, so there is one
    # derivation of the target and one construction of the calibration.
    target = nexusThread._frozen_source_target(no_poni)
    assert target.poni() is None
    prepared = PreparedNexusExecution(no_poni, target)
    nexusThread._adopt_frozen_source_target(thread, prepared)

    assert thread.poni is None


def test_public_start_without_calibration_refuses_zero_delta(
    widget, tmp_path, monkeypatch
):
    wrangler = _select_nexus(widget)
    source = tmp_path / "source.nxs"
    with h5py.File(source, "w") as handle:
        group = handle.create_group("entry")
        group.attrs["NX_class"] = "NXentry"
    output = tmp_path / "out"
    output.mkdir()
    wrangler.parameters.child("NeXus File", "nexus_file").setValue(str(source))
    wrangler.parameters.child("NeXus File", "entry").setValue("entry")
    wrangler.parameters.child("Output", "h5_dir").setValue(str(output))
    wrangler.parameters.child("Calibration", "poni_file").setValue("")
    wrangler.poni_file = ""
    wrangler.poni = None
    before = _zero_delta_snapshot(widget, wrangler)
    events = _start_recorder(wrangler, monkeypatch)

    wrangler.start()

    assert events == []
    assert _zero_delta_snapshot(widget, wrangler) == before


def test_incompatible_append_is_refused_before_existing_target_is_used(
    widget, tmp_path, monkeypatch
):
    wrangler = _select_nexus(widget)
    source, output = _arm_container_without_frames(wrangler, tmp_path)
    widget.controls.set_write_mode("Append")
    _start_recorder(wrangler, monkeypatch)
    wrangler.start()
    thread = wrangler.thread
    frozen = thread.run_configuration
    target = nexusThread._frozen_source_target(frozen)
    prior = Path(target.output_path)
    prior.write_bytes(b"prior incompatible run")
    scan = SimpleNamespace(data_file=str(prior))
    # O-3N.R.2 §17.4: the output transaction is owned by the envelope, so the
    # refusal cannot be bypassed by re-entering with the same frozen object.
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (
        PreparedNexusExecution,
    )

    prepared = PreparedNexusExecution(frozen, target)

    with pytest.raises(RunConfigurationRefused):
        nexusThread._prepare_output_for_run(thread, prepared, scan)
    assert prior.read_bytes() == b"prior incompatible run"
    assert prepared.output_committed is False
