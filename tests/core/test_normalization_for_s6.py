"""S-6 — _normalization_for delegates to the ONE canonical monitor resolver.

Before this, the reduction spine's _normalization_for did its own exact/upper/
lower key lookup and accepted negatives, while the GUI mirror used
resolve_monitor_norm (case-insensitive, rejects <=0).  A mixed-case monitor key
made the spine write UN-normalized data while map_norm claimed normalization.
These drive the real _normalization_for (no monkeypatch on the resolver).
"""

from types import SimpleNamespace

from xrd_tools.reduction.core import _normalization_for


def _frame(metadata):
    return SimpleNamespace(normalization_factor=None, index=0, metadata=metadata)


def test_resolves_mixed_case_monitor_key():
    # plan configures 'monitor'; metadata carries 'MonItor' -> the canonical
    # case-insensitive resolver finds it (the old exact/upper/lower missed it).
    got = _normalization_for(_frame({"MonItor": 2.0}),
                             SimpleNamespace(monitor_key="monitor"),
                             warned_keys=set())
    assert got == 2.0


def test_rejects_zero_and_negative_monitor():
    for bad in (0.0, -5.0):
        got = _normalization_for(_frame({"monitor": bad}),
                                 SimpleNamespace(monitor_key="monitor"),
                                 warned_keys=set())
        assert got is None, f"monitor {bad} must be unusable (canonical rejects <=0)"


def test_explicit_normalization_factor_still_wins():
    frame = SimpleNamespace(normalization_factor=3.0, index=0,
                            metadata={"monitor": 9.0})
    assert _normalization_for(frame, SimpleNamespace(monitor_key="monitor")) == 3.0


def test_no_monitor_key_is_none():
    assert _normalization_for(_frame({"monitor": 5.0}),
                              SimpleNamespace(monitor_key=None)) is None
