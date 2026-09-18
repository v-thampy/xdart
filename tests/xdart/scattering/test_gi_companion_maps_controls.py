"""GI-COMPANION-20260918 — the Processing 2D Axis choice, from edit to plan.

"Q-χ + Qip-Qoop" is a SELECTION of two existing GI modes: q_ip–q_oop stays the
primary and q–χ is declared alongside it.  It must round-trip through the saved
profile, leave the persisted nine-key GI block alone, reach only an ordinary
Run's plan, and be part of the Append identity.
"""

from __future__ import annotations

import pytest

from xdart.gui.tabs.scattering.adapters.dynamic_output import _mode_tokens
from xdart.gui.tabs.scattering.controls_editing import reduce_control_edit
from xdart.gui.tabs.scattering.controls_inventory import (
    GI_2D_AXES,
    INT_2D_AXIS,
    integration_values,
)
from xdart.gui.tabs.scattering.output_preflight import native_int_reduction_plan
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xrd_tools.io.schema import GI_MODE_KEYS_2D
from xrd_tools.reduction.core import companion_modes_2d
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xrd_tools.session.run_intent_profile import (
    dump_run_intent_profile,
    load_run_intent_profile,
)
from xrd_tools.sources.selection import image_series_spec

COMBINED = "Q-χ + Qip-Qoop"


def _intent() -> RunIntent:
    return RunIntent(
        source_spec=image_series_spec("/tmp/gi-companion-source.tif"),
        poni_file="/tmp/gi-companion.poni", save_path="/tmp/gi-companion-out",
        processing_mode="Int 2D",
        gi=GIIntent(enabled=True, incidence_motor="Manual", th_val=0.2),
    )


def _choose(intent: RunIntent, label: str) -> RunIntent:
    result = reduce_control_edit(RunIntentStore(intent).snapshot(), INT_2D_AXIS, label)
    assert type(result) is RunIntent, result
    return result


@pytest.mark.parametrize(("label", "primary", "run_modes", "tokens"), (
    ("Qip-Qoop", "qip_qoop", ("qip_qoop",), ("1d:q_total", "2d:qip_qoop")),
    ("Q-χ", "q_chi", ("q_chi",), ("1d:q_total", "2d:q_chi")),
    (COMBINED, "qip_qoop", ("qip_qoop", "q_chi"),
     ("1d:q_total", "2d:qip_qoop", "2d:q_chi")),
    ("Exit", "exit_angles", ("exit_angles",), ("1d:q_total", "2d:exit_angles")),
))
def test_each_choice_declares_exactly_its_direct_maps(label, primary, run_modes, tokens):
    # Reach every choice FROM the combined one, so a stale selection would show.
    intent = _choose(_choose(_intent(), COMBINED), label) if label != COMBINED else _choose(_intent(), COMBINED)
    frozen = intent.freeze()

    assert intent.gi.mode_2d == primary
    assert integration_values(intent)[INT_2D_AXIS] == label
    run_plan = native_int_reduction_plan(frozen, companion_modes_2d=True)
    assert run_plan.extra["enabled_modes_2d"] == run_modes
    assert companion_modes_2d(run_plan) == run_modes[1:]
    # Average and the XYE lookup translate the same configuration one-mode.
    assert native_int_reduction_plan(frozen).extra["enabled_modes_2d"] == run_modes[:1]
    # The selection never reaches pyFAI as a keyword.
    assert "gi_companion_modes_2d" not in run_plan.integration_2d.extra
    assert _mode_tokens(frozen) == tokens
    # The persisted GI block keeps its exact current keyset.
    assert len(frozen.as_provenance()["gi"]) == 9

    restored = load_run_intent_profile(dump_run_intent_profile(intent))
    assert integration_values(restored)[INT_2D_AXIS] == label
    assert restored.freeze().fingerprint == frozen.fingerprint


def test_the_combined_label_is_not_a_mode_key():
    assert COMBINED not in GI_2D_AXES
    assert set(GI_2D_AXES.values()) == set(GI_MODE_KEYS_2D)
    intent = _choose(_intent(), COMBINED)
    assert intent.gi.mode_2d in GI_MODE_KEYS_2D


def test_a_saved_single_mode_profile_loads_unchanged():
    """A profile written before this choice existed has no companion entry."""
    text = dump_run_intent_profile(_choose(_intent(), "Q-χ"))
    assert "gi_companion_modes_2d" not in text
    restored = load_run_intent_profile(text)
    assert restored.gi.mode_2d == "q_chi"
    assert integration_values(restored)[INT_2D_AXIS] == "Q-χ"
    assert integration_values(_intent())[INT_2D_AXIS] == "Qip-Qoop"   # the default


def test_reintegration_stays_single_mode():
    preparation = ScatteringWorkspace._reintegrate_preparation(
        _choose(_intent(), COMBINED), "2d",
    )
    selected = preparation["selected_plan"]
    assert selected["gi_mode"] == "qip_qoop"
    assert "gi_companion_modes_2d" not in selected["bai_args"]


def test_a_stale_selection_naming_the_primary_is_calculated_once():
    """Q-χ alone is stored once, whatever a hand-written selection says."""
    intent = _choose(_intent(), "Q-χ")
    intent.bai_2d_args["gi_companion_modes_2d"] = ["q_chi"]
    frozen = intent.freeze()
    plan = native_int_reduction_plan(frozen, companion_modes_2d=True)
    assert plan.extra["enabled_modes_2d"] == ("q_chi",)
    assert _mode_tokens(frozen) == ("1d:q_total", "2d:q_chi")
    assert integration_values(intent)[INT_2D_AXIS] == "Q-χ"
