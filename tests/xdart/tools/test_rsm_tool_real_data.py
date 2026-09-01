from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from xdart.gui.tools.rsm_values import prepare_rsm_tool, rsm_tool_preset
from xrd_tools.analysis.module_transaction import MetadataColumnSelector
from xrd_tools.sources.spec import SpecSource


_SPEC_SHA256 = "12e0d05219cb0309f7fb5f170c8fb12c507e7603dcd71b3954721d1b6e0c90b1"
_Q_BOUNDS = (
    (0.9991454998430791, 1.0507535379970891),
    (0.8498770429375965, 1.0673826805071531),
    (2.8230899655686454, 3.18407314269762),
)


def _rsm_root() -> Path:
    configured = os.environ.get("XDART_TEST_DATA")
    if not configured:
        pytest.skip("XDART_TEST_DATA is not configured")
    root = Path(configured) / "RSM"
    if not root.is_dir():
        pytest.skip("authenticated RSM test data is unavailable")
    return root


@pytest.mark.slow
def test_scan43_preview_authenticates_without_decoding_or_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _rsm_root()
    spec_path = root / "STO_align"
    assert hashlib.sha256(spec_path.read_bytes()).hexdigest() == _SPEC_SHA256

    output = root / f".xdart-rsm-preview-{os.getpid()}-{tmp_path.name}.nexus"
    assert not output.exists()
    form = rsm_tool_preset().form(root, output)

    def forbidden_decode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("RSM Preview must not decode detector images")

    monkeypatch.setattr(SpecSource, "load_frame", forbidden_decode)
    preflight = prepare_rsm_tool(form)
    summary = preflight.summary

    assert summary.source_relative_path == "STO_align"
    assert summary.source_scan == "43.1"
    assert summary.image_directory_relative_path == "images"
    assert summary.output_relative_path == output.name
    assert summary.selected_labels == tuple(range(61))
    assert len(summary.members) == 61
    assert MetadataColumnSelector("Seconds", 0) in summary.required_selectors
    assert summary.raw_header_skip == 0
    assert summary.energy_eV == 13000.007
    assert summary.q_bounds == _Q_BOUNDS
    assert all(
        member.relative_path
        == f"images/b_thampy_STO_align_scan43_{member.label:04d}.raw"
        for member in summary.members
    )
    assert all(
        any(
            name == "Seconds" and occurrence == 0
            for name, occurrence, _value in member.values
        )
        for member in summary.members
    )
    assert not output.exists()
