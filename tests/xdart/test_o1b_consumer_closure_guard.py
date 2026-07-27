"""O-1b — the derived execution-consumer closure guard and its inventory.

Replaces the W-1R-D3/D4 guard rows that keyed on receiver NAMES (`thread`,
`worker`) and on an enumerated file list.  Both evasions were real: `QtNexusSink`
hid behind `self._host` (review §45.1), and `staticWidget.update_data` hid behind
a local named `published`.  This guard instead DERIVES what holds a worker
reference and compares the derivation against the frozen inventory, so a new
downstream consumer fails until it is declared and classified.

Bounded by construction (CLAUDE.md rule 9): it tracks a worker passed as a call
argument or assigned to an attribute, plus simple locals bound to either. It is
not a general Python dataflow analyzer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.xdart._o1b_consumer_inventory import (
    DECLARED_CALLBACKS,
    DECLARED_CONSUMERS,
    FORBIDDEN_POLICY_FIELDS,
    OPERATIONAL_ALLOWLIST,
    WORKER_CLASSES,
    derive_callback_handoffs,
    derive_consumer_closure,
    derive_worker_alias_attributes,
    forbidden_reads_through_aliases,
)
from xdart.gui.tabs.static_scan.wranglers import wrangler_widget

_SRC = Path(wrangler_widget.__file__).parents[5]

VALID_CLASSIFICATIONS = {"OPERATIONAL", "POLICY_BY_ARGUMENT", "VALUE_ONLY"}


@pytest.fixture(scope="module")
def closure():
    return derive_consumer_closure(_SRC)


def test_the_derived_closure_matches_the_frozen_inventory(closure):
    """A new downstream consumer is a BLOCKER until it is inventoried.

    The inventory is the acceptance artifact: an unclassified constructor,
    callback, sink or adapter reachable from the run path fails here.
    """
    undeclared = sorted(set(closure) - set(DECLARED_CONSUMERS))
    assert undeclared == [], (
        "these objects are handed a worker reference but are not in the O-1b "
        f"inventory: {undeclared}. Declare each with its alias, the worker site "
        "that hands it over, a classification, a required typed value boundary, "
        "an exact-identity or source-qualified test, a contradictory-carrier "
        "poison test, and a guard row.")
    stale = sorted(set(DECLARED_CONSUMERS) - set(closure))
    assert stale == [], (
        f"the inventory declares consumers that no longer exist: {stale}")


def test_every_declared_consumer_is_classified_and_matches_its_alias(closure):
    for name, entry in DECLARED_CONSUMERS.items():
        assert entry["classification"] in VALID_CLASSIFICATIONS, name
        assert entry["note"].strip(), f"{name} has no classification rationale"
        derived = closure[name]
        assert entry["aliases"] == frozenset(derived["aliases"]), (
            f"{name} alias drift: declared {sorted(entry['aliases'])}, "
            f"derived {sorted(derived['aliases'])}")
        assert set(entry["handed_by"]) == set(derived["handed_by"]), (
            f"{name} handoff drift: declared {sorted(entry['handed_by'])}, "
            f"derived {sorted(derived['handed_by'])}")


def test_no_run_policy_is_reached_through_any_worker_reference(closure):
    """The whole point: policy access through a worker reference is forbidden.

    Scoped by derivation -- every declared consumer alias, every attribute the
    tree stores a worker under, and simple locals bound to either -- never by
    receiver name and never by a file list.
    """
    offenders = forbidden_reads_through_aliases(_SRC, closure)
    assert offenders == [], (
        "run/source/GI policy reached through a worker reference; route it from "
        f"the accepted configuration instead: {offenders}")


def test_the_guard_scope_is_derived_not_assumed():
    """The alias attribute names come from the tree, not from this file."""
    derived = derive_worker_alias_attributes(_SRC)
    assert derived, "no attribute stores a worker construction any more"
    source = Path(__file__).read_text()
    for name in derived:
        assert f'"{name}"' not in source and f"'{name}'" not in source, (
            f"the guard hardcodes the alias name {name!r}; it must derive it")


def test_operational_access_stays_allowlisted_and_disjoint():
    """Signals, locks, cancellation and buffers are explicitly permitted."""
    overlap = FORBIDDEN_POLICY_FIELDS & OPERATIONAL_ALLOWLIST
    assert overlap == set(), (
        f"a name is both forbidden policy and operational: {sorted(overlap)}")
    for expected in ("showLabel", "command", "command_lock", "file_lock",
                     "_xye_buffer", "_published_frames"):
        assert expected in OPERATIONAL_ALLOWLIST


def test_worker_callbacks_receive_policy_at_handoff():
    """The accepted configuration must be PASSED at the handoff, not declared.

    O-1b-0.1: the first version of this row only re-read the inventory's own
    fields, so it would have passed even if the spawn dropped the argument
    entirely.  It now derives the real ``Thread(target=..., args=...)`` call and
    proves the first spawn argument is the routed policy parameter that the
    target method declares.
    """
    derived = derive_callback_handoffs(_SRC)
    assert derived, "no worker hands a bound method to a thread any more"

    undeclared = sorted(set(derived) - set(DECLARED_CALLBACKS))
    assert undeclared == [], (
        f"callback handoffs missing from the inventory: {undeclared}")

    for name, handoff in derived.items():
        entry = DECLARED_CALLBACKS[name]
        assert entry["classification"] in VALID_CLASSIFICATIONS, name
        cls, _, method = name.partition(".")
        assert cls in WORKER_CLASSES, name
        assert method, name
        params = handoff["target_params"]
        assert params[:2] == ["self", "frozen"], (
            f"{name} does not take the accepted configuration as its first "
            f"required argument; signature starts {params[:2]}")
        assert handoff["args"], (
            f"{name} is spawned with no arguments, so the concurrent reader "
            f"cannot have been handed a configuration ({handoff['site']})")
        assert handoff["args"][0] == "frozen", (
            f"{name} is spawned with {handoff['args'][0]!r} first, not the "
            f"accepted configuration ({handoff['site']})")


# --------------------------------------------------------------------------- #
# Bounded proofs that the derivation catches the evasions it claims to.
# Synthetic sources, one shape each -- deliberately NOT a taint analyzer.
# --------------------------------------------------------------------------- #

def _derive_from(tmp_path, source, name="probe.py"):
    (tmp_path / name).write_text(source)
    return tmp_path


def test_derivation_catches_annotated_worker_storage(tmp_path):
    """`self.thread: imageThread = ...` is an AnnAssign, not an Assign."""
    root = _derive_from(tmp_path, '''
class imageThread:
    pass


class Host:
    def __init__(self):
        self.annotated_slot: imageThread = imageThread()
''')
    assert "annotated_slot" in derive_worker_alias_attributes(root)


def test_derivation_catches_an_aliased_worker_constructor(tmp_path):
    """A local alias for the worker class still constructs a worker."""
    root = _derive_from(tmp_path, '''
class wranglerThread:
    pass


builder = wranglerThread


class Host:
    def __init__(self):
        self.aliased_slot = builder()
''')
    assert "aliased_slot" in derive_worker_alias_attributes(root)


def test_derivation_catches_a_worker_handed_via_a_local_self_alias(tmp_path):
    """`me = self; Sink(me)` must still make Sink a declared consumer."""
    root = _derive_from(tmp_path, '''
class Sink:
    def __init__(self, host):
        self._host = host


class imageThread:
    def go(self):
        me = self
        return Sink(me)
''')
    closure = derive_consumer_closure(root)
    assert "Sink" in closure, sorted(closure)
    assert closure["Sink"]["aliases"] == {"self._host"}
