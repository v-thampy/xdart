from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus
from xrd_tools.sources.image import TiffSeriesSource

from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_e1b1_terminal_cleanup import _prepared_run
from tests.xdart.scattering.test_p1b_output_graph import (
    _live_directory_intent, _write_tiff,
)


def test_construct_loop_adopts_two_artifacts_as_one_atomic_current_scope(
    monkeypatch, tmp_path,
):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("a", "b"):
        _write_tiff(raw / f"{name}_0001.tif", 1)
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    intent = _live_directory_intent(
        raw, tmp_path / "processed", poni, processing_mode="Int 1D",
    )
    intent.live_mode = False
    executor, run, _admission = _prepared_run(tmp_path, intent=intent)
    observed, closed_sources = [], []
    real_construct = executor._construct

    def construct(owned_run, **kwargs):
        result = real_construct(owned_run, **kwargs)
        context = owned_run.context_runtime.context
        observed.append((context, context.current_scope, owned_run.scan, kwargs["item"]))
        return result

    monkeypatch.setattr(executor, "_construct", construct)
    monkeypatch.setattr(
        TiffSeriesSource, "close", lambda source: closed_sources.append(source),
        raising=False,
    )
    try:
        executor._run(run)
        terminal = executor.drain_events()[-1]
        assert terminal.kind is StandardEventKind.FINISHED, terminal.detail
        assert len(observed) == 2
        context, scope_a, scan_a, item_a = observed[0]
        context_b, scope_b, scan_b, item_b = observed[1]
        assert context_b is context
        assert scope_a.display_scan is scan_a
        assert scope_b is not scope_a
        assert (
            scope_b.scan_key, scope_b.source, scope_b.display_scan,
            scope_b.commit_epoch,
        ) == (
            scan_b.name, str(item_b.source_spec.uri), scan_b,
            scope_a.commit_epoch + 1,
        )
        assert context.scan is scan_a
        assert context.commit_gate.epoch == scope_b.commit_epoch
        assert context.hydration_owner.scan_key == scope_b.scan_key
        assert context.hydration_owner.source == scope_b.source
        assert context.hydration_owner.epoch == scope_b.commit_epoch
        assert context.display_bindings().scan is scan_b
        assert tuple(run.display.artifacts) == (str(item_a.target), str(item_b.target))
        assert item_a.target.is_file() and item_b.target.is_file()
        assert len(closed_sources) == 2
    finally:
        assert executor.close(run.identity).cleanup_status is CleanupStatus.CLEANED
