"""Frozen E1b-a source-capture ownership cases."""

from pathlib import Path

import pytest

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.sources.selection import DirectorySourceSpec
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.events import RequestId


def _series() -> SourceSpec:
    return SourceSpec(
        Path("/data/frame_0001.tif"),
        SourceKind.TIFF_SERIES,
        options={"files": ("/data/frame_0001.tif",)},
    )


def test_capture_uses_adapter_owned_monotonic_epochs() -> None:
    adapter = FilesystemSourceAdapter()
    first_request = RequestId(1)
    second_request = RequestId(2)

    first = adapter.capture(_series(), first_request)
    second = adapter.capture(_series(), second_request)

    assert first.request_id is first_request
    assert second.request_id is second_request
    assert first.source_epoch == 1
    assert second.source_epoch == 2
    assert second.source is not first.source
    assert second.gi_motor_choices is None


def test_old_request_cancellation_cannot_cancel_newer_capture() -> None:
    adapter = FilesystemSourceAdapter()
    first = adapter.capture(_series(), RequestId(1))
    second = adapter.capture(_series(), RequestId(2))

    adapter.cancel(first.request_id)
    third = adapter.capture(_series(), RequestId(3))
    adapter.cancel(second.request_id)

    assert (first.source_epoch, second.source_epoch, third.source_epoch) == (1, 2, 3)
    assert third.request_id.value == 3


def test_capture_accepts_e2_directory_without_enumerating_contents() -> None:
    adapter = FilesystemSourceAdapter()
    source = DirectorySourceSpec(Path("/data"))

    capture = adapter.capture(source, RequestId(1))

    assert capture.source == source
    assert not hasattr(capture, "candidate_plan")
