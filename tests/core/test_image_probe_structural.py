"""Structural image-probe guards for files that are still landing."""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from xrd_tools.sources.adapters import candidate_owner
from xrd_tools.sources.directory_session import DirectoryIndexSession
from xrd_tools.sources.probe import ProbeState
from xrd_tools.sources import registry as _registered_source_registry


def _probe(path: Path):
    # Importing registry installs the built-in adapters; keep that bootstrap
    # explicit in this narrowly focused adapter test.
    assert _registered_source_registry is not None
    owner = candidate_owner(path)
    assert owner is not None
    return owner.probe(path)


def _tiff_payload(tmp_path: Path) -> tuple[bytes, int]:
    """Return a complete TIFF and its first declared pixel-data offset."""
    tifffile = pytest.importorskip("tifffile")
    complete = tmp_path / "complete-fixture.tif"
    tifffile.imwrite(
        complete,
        np.arange(32 * 48, dtype=np.uint16).reshape(32, 48),
        compression=None,
        metadata=None,
    )
    payload = complete.read_bytes()
    with tifffile.TiffFile(complete) as handle:
        page = handle.pages[0]
        offsets = tuple(int(value) for value in page.dataoffsets)
        byte_counts = tuple(int(value) for value in page.databytecounts)
    complete.unlink()
    assert offsets and len(offsets) == len(byte_counts)
    assert max(offset + size for offset, size in zip(offsets, byte_counts)) <= len(payload)
    first_offset = min(offsets)
    assert 0 < first_offset < len(payload)
    return payload, first_offset


def test_partial_tiff_payload_stays_provisional_without_entering_fabio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A complete IFD with missing pixels is retryable without decoder noise."""
    import fabio

    payload, payload_offset = _tiff_payload(tmp_path)
    source = tmp_path / "landing_0001.tif"
    source.write_bytes(payload[:payload_offset])
    initial_inode = source.stat().st_ino

    decoder_calls: list[Path] = []
    real_open = fabio.open

    def guarded_open(filename, *args, **kwargs):
        candidate = Path(filename)
        decoder_calls.append(candidate)
        if candidate == source and candidate.stat().st_size < len(payload):
            raise AssertionError("Fabio entered an incomplete TIFF revision")
        return real_open(filename, *args, **kwargs)

    monkeypatch.setattr(fabio, "open", guarded_open)

    with caplog.at_level(logging.WARNING, logger="fabio.tifimage"):
        # Exercise the adapter directly and the persistent session route.  An
        # unchanged provisional stamp must remain retryable rather than being
        # cached as a terminal failure.
        assert _probe(source).state is ProbeState.IN_PROGRESS
        assert _probe(source).state is ProbeState.IN_PROGRESS
        session = DirectoryIndexSession()
        try:
            session.configure(tmp_path, suffixes=(".tif",))
            first = session.observe()
            second = session.observe()
            assert first.pending_count == second.pending_count == 1
            assert first.content_opens == second.content_opens == 1
            assert decoder_calls == []

            # Grow the exact same filesystem object.  The size stamp wakes the
            # session immediately and the first complete probe enters Fabio
            # exactly once.
            with source.open("ab") as stream:
                stream.write(payload[payload_offset:])
            assert source.stat().st_ino == initial_inode
            ready = session.observe()
            assert ready.result_for(source) is not None
            assert ready.result_for(source).state is ProbeState.READY
            assert ready.pending_count == 0
            assert ready.content_opens == 1
            assert decoder_calls == [source]
        finally:
            session.close()

    noisy = [
        record for record in caplog.records
        if record.name.startswith("fabio.tifimage")
        and record.levelno >= logging.WARNING
    ]
    assert noisy == []


@pytest.mark.parametrize("prefix", (b"", b"I", b"M"), ids=("empty", "I", "M"))
def test_short_possible_tiff_prefix_never_enters_fabio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prefix: bytes,
) -> None:
    import fabio

    source = tmp_path / "landing.tif"
    source.write_bytes(prefix)
    decoder_calls: list[Path] = []

    def forbidden_open(filename, *args, **kwargs):
        decoder_calls.append(Path(filename))
        raise AssertionError("Fabio entered a definitely incomplete TIFF header")

    monkeypatch.setattr(fabio, "open", forbidden_open)

    assert _probe(source).state is ProbeState.IN_PROGRESS
    assert _probe(source).state is ProbeState.IN_PROGRESS
    assert decoder_calls == []


def test_foreign_one_byte_tiff_prefix_remains_decoder_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fabio

    source = tmp_path / "foreign.tif"
    source.write_bytes(b"x")
    decoder_calls: list[Path] = []

    def rejecting_open(filename, *args, **kwargs):
        decoder_calls.append(Path(filename))
        raise OSError("decoder rejects foreign header")

    monkeypatch.setattr(fabio, "open", rejecting_open)

    assert _probe(source).state is ProbeState.IN_PROGRESS
    assert decoder_calls == [source]


@pytest.mark.parametrize(
    ("byteorder", "bigtiff"),
    ((">", False), ("<", True)),
    ids=("big-endian-classic", "little-endian-bigtiff"),
)
def test_tiff_header_variants_reject_truncated_payload_before_fabio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    byteorder: str,
    bigtiff: bool,
) -> None:
    import fabio

    tifffile = pytest.importorskip("tifffile")
    complete = tmp_path / "variant-complete.tif"
    tifffile.imwrite(
        complete,
        np.arange(32 * 48, dtype=np.uint16).reshape(32, 48),
        compression=None,
        metadata=None,
        byteorder=byteorder,
        bigtiff=bigtiff,
    )
    payload = complete.read_bytes()
    with tifffile.TiffFile(complete) as handle:
        offsets = tuple(int(value) for value in handle.pages[0].dataoffsets)
    complete.unlink()
    first_offset = min(offsets)
    assert 0 < first_offset < len(payload)

    source = tmp_path / "variant-landing.tif"
    source.write_bytes(payload[:first_offset])
    decoder_calls: list[Path] = []

    def forbidden_open(filename, *args, **kwargs):
        decoder_calls.append(Path(filename))
        raise AssertionError("Fabio entered an incomplete TIFF variant")

    monkeypatch.setattr(fabio, "open", forbidden_open)

    assert _probe(source).state is ProbeState.IN_PROGRESS
    assert decoder_calls == []


@pytest.mark.parametrize(
    "write_options",
    (
        {"rowsperstrip": 4},
        {"tile": (16, 16)},
    ),
    ids=("multi-strip", "tiled"),
)
def test_complete_strip_and_tile_tiffs_remain_ready(
    tmp_path: Path,
    write_options: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fabio

    tifffile = pytest.importorskip("tifffile")
    source = tmp_path / "complete.tif"
    tifffile.imwrite(
        source,
        np.arange(32 * 32, dtype=np.uint16).reshape(32, 32),
        compression=None,
        metadata=None,
        **write_options,
    )
    payload = source.read_bytes()
    with tifffile.TiffFile(source) as handle:
        page = handle.pages[0]
        offsets = tuple(int(value) for value in page.dataoffsets)
        byte_counts = tuple(int(value) for value in page.databytecounts)
    required_end = max(
        offset + size for offset, size in zip(offsets, byte_counts))
    assert required_end <= len(payload)

    decoder_calls: list[Path] = []
    real_open = fabio.open

    def guarded_open(filename, *args, **kwargs):
        decoder_calls.append(Path(filename))
        return real_open(filename, *args, **kwargs)

    monkeypatch.setattr(fabio, "open", guarded_open)

    source.write_bytes(payload[:required_end - 1])
    assert _probe(source).state is ProbeState.IN_PROGRESS
    assert decoder_calls == []

    source.write_bytes(payload)
    assert _probe(source).state is ProbeState.READY
    assert decoder_calls == [source]


def test_non_tiff_format_remains_decoder_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fabio

    source = tmp_path / "frame.edf"
    fabio.edfimage.EdfImage(
        data=np.arange(8 * 12, dtype=np.uint16).reshape(8, 12),
    ).write(str(source))
    decoder_calls: list[Path] = []
    real_open = fabio.open

    def counting_open(filename, *args, **kwargs):
        decoder_calls.append(Path(filename))
        return real_open(filename, *args, **kwargs)

    monkeypatch.setattr(fabio, "open", counting_open)

    assert _probe(source).state is ProbeState.READY
    assert decoder_calls == [source]


def test_unrecognized_tiff_header_falls_back_to_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must not bless or permanently reject an unknown header."""
    import fabio

    source = tmp_path / "unknown.tif"
    source.write_bytes(b"not a recognized TIFF header")
    decoder_calls: list[Path] = []

    def rejecting_open(filename, *args, **kwargs):
        decoder_calls.append(Path(filename))
        raise OSError("decoder rejects malformed TIFF")

    monkeypatch.setattr(fabio, "open", rejecting_open)

    result = _probe(source)
    assert result.state is ProbeState.IN_PROGRESS
    assert "decoder" in (result.reason or "").lower()
    assert decoder_calls == [source]


def test_tiff_cumulative_ifd_budget_falls_back_to_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Many individually legal IFDs cannot multiply the probe work bound."""
    import fabio

    entry_count = 40_000
    first_ifd = 8
    second_ifd = first_ifd + 2 + entry_count * 12 + 4
    offset_entry = struct.pack(
        "<HHI4s", 273, 4, 1, struct.pack("<I", first_ifd))
    count_entry = struct.pack(
        "<HHI4s", 279, 4, 1, struct.pack("<I", 1))
    first_directory = (
        struct.pack("<H", entry_count)
        + offset_entry
        + count_entry
        + bytes((entry_count - 2) * 12)
        + struct.pack("<I", second_ifd)
    )
    source = tmp_path / "bounded.tif"
    source.write_bytes(
        b"II" + struct.pack("<HI", 42, first_ifd)
        + first_directory
        + struct.pack("<H", entry_count)
    )

    decoder_calls: list[Path] = []

    def rejecting_open(filename, *args, **kwargs):
        decoder_calls.append(Path(filename))
        raise OSError("synthetic bounded fixture is not a complete image")

    monkeypatch.setattr(fabio, "open", rejecting_open)

    assert _probe(source).state is ProbeState.IN_PROGRESS
    assert decoder_calls == [source]


def test_tiff_change_during_header_probe_is_provisional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fabio

    payload, _payload_offset = _tiff_payload(tmp_path)
    source = tmp_path / "changing.tif"
    source.write_bytes(payload)
    owner = candidate_owner(source)
    assert owner is not None

    real_stat = Path.stat
    target_stats = 0

    def changing_stat(path, *args, **kwargs):
        nonlocal target_stats
        current = real_stat(path, *args, **kwargs)
        if path != source:
            return current
        target_stats += 1
        if target_stats < 2:
            return current
        return SimpleNamespace(
            st_dev=current.st_dev,
            st_ino=current.st_ino,
            st_size=current.st_size,
            st_mtime_ns=current.st_mtime_ns + 1,
        )

    decoder_calls: list[Path] = []

    def forbidden_open(filename, *args, **kwargs):
        decoder_calls.append(Path(filename))
        raise AssertionError("decoder entered a concurrently changing TIFF")

    monkeypatch.setattr(Path, "stat", changing_stat)
    monkeypatch.setattr(fabio, "open", forbidden_open)

    result = owner.probe(source)
    assert result.state is ProbeState.IN_PROGRESS
    assert "changed" in (result.reason or "").lower()
    assert target_stats >= 2
    assert decoder_calls == []
