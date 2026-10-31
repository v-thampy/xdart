"""TERMINAL-SUPPORT-128: a real Run handoff's Browse context hands its seal to the angular-coverage rebuild
(Claude, 2026-10-30). The real loaded context from a sealed Run handoff (test_p34's terminal-mode load of a seeded Run
output, whose writer seal is held) is attached to a projection exactly as Browse does; the rebuild then revalidates
against that seal, with no whole-file hash and no false "could not be read" refusal.
"""
from __future__ import annotations

from tests.xdart.scattering.test_p34_reintegrate_operation import _loaded_page, qapp  # noqa: F401


def test_a_real_run_handoff_context_gives_the_strip_its_seal(tmp_path, monkeypatch, qapp):  # noqa: F811
    from xdart.gui.tabs.scattering.context_projection import attach_saved_support_source
    from xdart.gui.tabs.scattering.shell_values import ScientificProjection
    from xrd_tools.analysis.angular_support import reconstruct_saved_support
    from xrd_tools.io import output_transaction as transaction

    page, _store, seeded, context = _loaded_page(tmp_path, monkeypatch, qapp, terminal_browse=True)
    try:
        seal = context.load_request.terminal_commit_identity
        assert seal is seeded.terminal.commit_identity
        source = attach_saved_support_source(ScientificProjection(processing_mode="Int 1D"), context)
        source = source.saved_support_source
        assert source is not None and source.terminal is seal
        real, hashed = transaction._sha256_handle, []

        def counting(handle):
            hashed.append(1)
            return real(handle)
        monkeypatch.setattr(transaction, "_sha256_handle", counting)
        profiles = reconstruct_saved_support(source.request(seeded.labels))
        assert sorted(profiles) == sorted(seeded.labels) and hashed == []
        assert not any("could not be read" in profile.detail for profile in profiles.values())
    finally:
        page.close_workspace()
