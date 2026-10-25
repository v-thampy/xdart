"""Writer-owned output timestamps may settle late; size/inode and Browse stay strict."""
import logging
import os

import pytest

from tests.core.test_h23_output_transaction import (
    _begin_in_place, _prepared_in_place, _reopen, _sealed_checkpoint,
)


def _touch(path):
    state = path.stat()
    os.utime(path, ns=(state.st_atime_ns, state.st_mtime_ns + 86_500_000))


def test_writer_timestamp_drift_survives_checkpoint_close_commit_and_reopen(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    prepared = _prepared_in_place(tmp_path)
    api, target, _coordinator, transaction, *_rest, lease = prepared
    attempt = _begin_in_place(prepared)
    transaction.authorize_stream_mutation(attempt, lease=lease)
    with target.open("ab") as stream:
        stream.write(b"-new")
    checkpoint = _sealed_checkpoint(api, transaction, attempt, lease, target)
    _touch(target)
    transaction.promote_stream_checkpoint(attempt, checkpoint, lease=lease)
    _touch(target)
    close = transaction.begin_stream_close(attempt, lease=lease)
    transaction.seal_stream_close(attempt, close, lease=lease)
    _touch(target)
    transaction.seal_stream_close(attempt, close, lease=lease)
    _touch(target)
    terminal = transaction.seal_stream_terminal(attempt, lease=lease)
    _touch(target)
    with pytest.raises(api.TargetChanged):
        api.revalidate_stream_terminal(target, terminal)
    transaction.commit_stream_epoch(attempt, lease=lease)
    _touch(target)
    reopened = _reopen(prepared, attempt)
    _touch(target)
    transaction.authorize_stream_mutation(reopened, lease=lease)
    transaction.abort_stream(reopened, lease=lease)
    assert target.read_bytes() == b"prior-new"
    assert any("writer output timestamps changed" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("change", ("size", "inode"))
def test_writer_timestamp_tolerance_still_refuses_changed_object(tmp_path, change):
    prepared = _prepared_in_place(tmp_path)
    api, target, _coordinator, transaction, *_rest, lease = prepared
    attempt = _begin_in_place(prepared)
    if change == "size":
        with target.open("ab") as stream:
            stream.write(b"foreign")
    else:
        replacement = tmp_path / "foreign"
        replacement.write_bytes(target.read_bytes())
        os.replace(replacement, target)
    with pytest.raises(api.TargetChanged):
        transaction.authorize_stream_mutation(attempt, lease=lease)
