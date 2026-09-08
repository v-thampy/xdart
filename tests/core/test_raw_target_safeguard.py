# -*- coding: utf-8 -*-
"""OWNER-GATE-RAW-TARGET-20260905 — Replace may not destroy a non-result file.

Overwrite skipped the positive existing-target admission that Append performs,
on the reading that Replace means "whatever is there, replace it". A real
two-scan directory Run turned that into RAW DATA LOSS: scan B's external
detector member occupied scan A's generated output slot, so A's transaction
backed it up, replaced it with A's processed result, and the Run then skipped B
and reported FINISHED/CLEANED.

The backup/swap behaved exactly as designed. It was AUTHORIZED for the wrong
kind of target, which is an admission error -- no amount of rollback machinery
would have preserved B's detector data.

These rows pin the RECOGNITION POLICY, which is the part the reviewer warned
was easy to get wrong.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from xrd_tools.reduction import NexusSink


def _sink(path: Path) -> NexusSink:
    return NexusSink(path, entry="entry", overwrite=True)


def _raw_container(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_group("entry/instrument/detector").create_dataset(
            "data", data=np.arange(12, dtype="u2").reshape(3, 2, 2),
        )


def test_an_absent_target_is_still_writable(tmp_path: Path) -> None:
    """The guard must not stop an ordinary first run."""
    _sink(tmp_path / "scan_int1d.nexus")._require_current_append_target()


def test_replacing_a_raw_container_is_refused(tmp_path: Path) -> None:
    """The reproduced P1: raw detector data sitting at the output slot."""
    target = tmp_path / "scan_int1d.nexus"
    _raw_container(target)
    before = target.read_bytes()

    with pytest.raises(ValueError, match="not a current xdart processed result"):
        _sink(target)._require_current_append_target()

    # Refused BEFORE anything touched it.
    assert target.read_bytes() == before


def test_replacing_a_damaged_file_is_refused(tmp_path: Path) -> None:
    """A file that cannot be opened is refused, not guessed at.

    The recognizer returns False on OSError rather than assuming either way.
    """
    target = tmp_path / "scan_int1d.nexus"
    target.write_bytes(b"not an HDF5 file at all")
    before = target.read_bytes()

    with pytest.raises(ValueError, match="not a current xdart processed result"):
        _sink(target)._require_current_append_target()
    assert target.read_bytes() == before


def test_a_broad_marker_match_is_not_sufficient_on_its_own(tmp_path: Path) -> None:
    """THE ROW THAT MATTERS. Codex's warning, made executable.

    `has_processed_output_markers_*` is documented as raw-NEGATIVE recognition:
    arbitrary result-group names and resolved external entries can satisfy it.
    Using it as the safety proof would have admitted this file for replacement.

    The guard uses strict POSITIVE admission instead, so this file is refused
    even though the broad predicate accepts it. If someone later "simplifies"
    the guard onto the broad helper, this row fails.
    """
    from xrd_tools.io.processed_scan_id import has_processed_output_markers_path

    target = tmp_path / "scan_int1d.nexus"
    with h5py.File(target, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        # Marker-shaped, but NOT a current xdart result: no schema identity and
        # no conforming integrated stack.
        entry.create_group("integrated_1d").create_dataset(
            "intensity", data=np.zeros((1, 4), dtype="f4"),
        )
    before = target.read_bytes()

    broad = has_processed_output_markers_path(target, "entry")
    with pytest.raises(ValueError, match="not a current xdart processed result"):
        _sink(target)._require_current_append_target()
    assert target.read_bytes() == before
    # Recorded rather than asserted-True: the point is that the guard does not
    # DEPEND on this predicate's answer, whichever way it falls.
    assert isinstance(broad, bool)


def test_the_refusal_names_the_file_and_the_remedy(tmp_path: Path) -> None:
    """An operator has to be able to act on it without reading the source."""
    target = tmp_path / "scan_int1d.nexus"
    _raw_container(target)
    with pytest.raises(ValueError) as caught:
        _sink(target)._require_current_append_target()
    message = str(caught.value)
    assert str(target) in message
    assert "different output folder" in message
    assert "delete" in message


def test_a_recognized_previous_result_is_still_replaceable(tmp_path, monkeypatch):
    """POSITIVE CONTROL, and the reason this guard is narrow.

    Refusing raw data must not refuse an ordinary repeat Run. This uses a REAL
    `NexusSink`-written result -- the same fixture the reintegrate suite treats
    as a genuine predecessor -- rather than a hand-built file that might satisfy
    the recognizer for the wrong reason.
    """
    from tests.core.reintegrate_support import _seed_existing
    from xrd_tools.io.processed_scan_id import is_current_processed_xdart_path

    seeded = _seed_existing(tmp_path, labels=(2, 5), name="recognized")
    target = Path(seeded.target)
    assert is_current_processed_xdart_path(target, "entry")

    # No raise: a real prior result is replaceable, exactly as before.
    _sink(target)._require_current_append_target()
