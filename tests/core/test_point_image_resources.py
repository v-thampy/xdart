"""Retained native statistics consume space in the existing source envelope."""
from dataclasses import replace

from xrd_tools.session.policy import SessionResourceRequirements, resolve_session_policy


def test_point_variance_is_funded_for_every_queued_and_inflight_image():
    baseline = SessionResourceRequirements(195, 487, 8, modes_1d=1, npt_1d=24)
    grouped = replace(baseline, source_auxiliary_bytes=8 * baseline.pixels)
    requests = {'workers': 2, 'reduction_inflight': 5, 'queue_depth': 3}
    before = resolve_session_policy(baseline, envelope_bytes=8 << 30, requests=requests, env={}).allocation
    after = resolve_session_policy(grouped, envelope_bytes=8 << 30, requests=requests, env={}).allocation
    assert dict(before.counts) == dict(after.counts)
    copies = after.queue_depth + 1 + after.reduction_inflight
    assert after.categories['source_native'] - before.categories['source_native'] == copies * grouped.source_auxiliary_bytes
    assert after.assigned_bytes <= after.envelope_bytes


def test_ordinary_reintegrate_allocation_keeps_legacy_recipe_and_fingerprint():
    from xrd_tools.reduction.reintegrate import _allocation_recipe, _policy
    requirements = SessionResourceRequirements(195, 487, 2, modes_1d=1, npt_1d=24)
    allocation = resolve_session_policy(requirements, envelope_bytes=8 << 30, env={}).allocation
    recipe = _allocation_recipe(allocation)
    assert 'source_auxiliary_bytes' not in recipe['requirements']
    restored = _policy(requirements, {'version': 1, 'kind': 'explicit', 'allocation': recipe}).allocation
    assert restored.requirements.fingerprint == tuple(recipe['requirements'].values())
    assert dict(restored.categories) == dict(allocation.categories)
