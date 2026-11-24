"""A SPEC capture retires while its operation still owns the terminal."""
import os
from pathlib import Path

import pytest

from tests.core.test_rsm_operation_r2 import _form, _runnable, _write_member
from xrd_tools.analysis import module_transaction, rsm_operation
from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.rsm_values import prepare_rsm_tool_v2
from xrd_tools.io import spec as io_spec


@pytest.mark.parametrize("terminal", ("committed", "cleanup_pending", "verification_pending"))
def test_spec_spool_retires_after_real_rsm_publication(tmp_path, monkeypatch, terminal):
    _runnable(monkeypatch)
    request = prepare_rsm_tool_v2(_form(tmp_path, (_write_member(tmp_path, 0),))).request
    target = Path(request.module.output.target)
    execution = rsm_operation.RSMOperationExecutionV2(request)
    real_unlink = Path.unlink
    spool_attempts = []

    def unlink_once(path, *args, **kwargs):
        if path.name.startswith("xdart-spec-"):
            spool_attempts.append(path)
            assert target.is_file()
            if len(spool_attempts) == 1:
                raise PermissionError("transient private SPEC unlink fault")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink_once)
    try:
        if terminal == "cleanup_pending":
            with monkeypatch.context() as fault:
                def fail_readback(*args, **kwargs):
                    raise OSError("transient module readback fault")

                fault.setattr(module_transaction, "inspect_analysis_artifact", fail_readback)
                with pytest.raises(rsm_operation.RSMOperationCleanupPendingV2) as pending:
                    execution.run()
                assert pending.value.execution is execution
            result = pending.value.retry_cleanup()
        elif terminal == "verification_pending":
            with monkeypatch.context() as fault:
                def fail_verification(*args, **kwargs):
                    raise OSError("transient detached reload fault")

                fault.setattr(rsm_operation, "read_analysis_artifact", fail_verification)
                with pytest.raises(rsm_operation.RSMOperationVerificationErrorV2) as pending:
                    execution.run()
                assert pending.value.execution is execution
            result = execution.retry_verification()
        else:
            result = execution.run()
        assert result.terminal.disposition is ModuleDisposition.COMMITTED
        assert result.terminal.commit.terminal.target == str(target)
        assert len(spool_attempts) == 2 and spool_attempts[0] == spool_attempts[1]
        assert not spool_attempts[0].exists()
    finally:
        for path in set(spool_attempts):
            real_unlink(path, missing_ok=True)


def test_persistent_spool_fault_reports_locator_without_discarding_commit(
    tmp_path, monkeypatch, caplog,
):
    _runnable(monkeypatch)
    request = prepare_rsm_tool_v2(_form(tmp_path, (_write_member(tmp_path, 0),))).request
    real_unlink = Path.unlink
    spool_attempts = []

    def blocked_unlink(path, *args, **kwargs):
        if path.name.startswith("xdart-spec-"):
            spool_attempts.append(path)
            raise PermissionError("persistent private SPEC unlink fault")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", blocked_unlink)
    try:
        result = rsm_operation.run_rsm_operation_v2(request)
        assert result.terminal.disposition is ModuleDisposition.COMMITTED
        assert len(spool_attempts) == 2 and spool_attempts[0] == spool_attempts[1]
        assert spool_attempts[0].is_file()
        assert str(spool_attempts[0]) in caplog.text
        assert "persistent private SPEC unlink fault" in caplog.text
    finally:
        for path in set(spool_attempts):
            real_unlink(path, missing_ok=True)


@pytest.mark.parametrize("persistent", (False, True), ids=("transient", "persistent"))
def test_spec_spool_locator_survives_failed_parser_close(tmp_path, monkeypatch, caplog, persistent):
    member = _write_member(tmp_path, 0)
    real_parser = io_spec.SpecFile
    paths = []
    parsers = []
    close_attempts = []

    class CloseFault:
        def __init__(self, path):
            self.path = Path(path)
            paths.append(self.path)
            self.parser = real_parser(path)
            parsers.append(self.parser)

        def keys(self):
            return self.parser.keys()

        def close(self):
            assert self.path.is_file()
            close_attempts.append(self.path)
            if persistent or len(close_attempts) == 1:
                raise OSError("private SPEC parser close fault")
            self.parser.close()

    monkeypatch.setattr(io_spec, "SpecFile", CloseFault)
    try:
        with io_spec.spec_capture_scope():
            assert io_spec.list_spec_scans(member.spec_path) == ["1.1"]
        assert len(paths) == 1 and close_attempts == paths * 2
        assert paths[0].is_file() is persistent
        if persistent:
            assert str(paths[0]) in caplog.text
            assert "private SPEC parser close fault" in caplog.text
    finally:
        if persistent or len(close_attempts) < 2:
            for parser in parsers:
                parser.close()
        for path in paths:
            path.unlink(missing_ok=True)


@pytest.mark.parametrize("source_read_fault", (False, True), ids=("close", "read_and_close"))
def test_spec_capture_retires_spool_if_original_descriptor_close_fails(
    tmp_path, monkeypatch, source_read_fault,
):
    source = str(_write_member(tmp_path, 0).spec_path)
    spool = tmp_path / "spool"
    spool.mkdir()
    monkeypatch.setattr(io_spec.tempfile, "tempdir", str(spool))
    real_open, real_fstat, real_close = os.open, os.fstat, os.close
    source_descriptor = None
    close_faults = []

    def open_source(path, *args, **kwargs):
        nonlocal source_descriptor
        descriptor = real_open(path, *args, **kwargs)
        if path == source:
            source_descriptor = descriptor
        return descriptor

    def stat_source(descriptor):
        if source_read_fault and descriptor == source_descriptor and any(spool.iterdir()):
            raise OSError("source descriptor stat fault after copying")
        return real_fstat(descriptor)

    def close_source(descriptor):
        nonlocal source_descriptor
        real_close(descriptor)
        if descriptor == source_descriptor:
            source_descriptor = None
            close_faults.append(descriptor)
            raise OSError("source descriptor close fault")

    monkeypatch.setattr(os, "open", open_source)
    monkeypatch.setattr(os, "fstat", stat_source)
    monkeypatch.setattr(os, "close", close_source)
    expected = "stat fault after copying" if source_read_fault else "close fault"
    try:
        with pytest.raises(OSError, match=expected), io_spec.spec_capture_scope():
            io_spec.list_spec_scans(source)
        assert len(close_faults) == 1
        assert not list(spool.iterdir())
    finally:
        for path in spool.iterdir():
            path.unlink()
