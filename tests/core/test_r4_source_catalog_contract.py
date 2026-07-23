"""R4-D descriptor retention and physical source-open count reproducer."""

from __future__ import annotations

from tests.core.test_container_cursor import _count_h5_opens, _stack
from xrd_tools.sources.cursor import ContainerCursor
from xrd_tools.sources.descriptor import ContainerDescriptor
from xrd_tools.sources.directory_session import DirectoryIndexSession
from xrd_tools.sources.run_plan import RunCandidatePlan


def _retained_descriptor(observed_candidate):
    """Accept either a record field or a descriptor-valued probe result."""
    descriptor = getattr(observed_candidate, "descriptor", None)
    if descriptor is None and isinstance(
        observed_candidate.result, ContainerDescriptor
    ):
        descriptor = observed_candidate.result
    return descriptor


def test_r4d_catalog_retains_one_open_descriptor_for_cursor_handoff(
    tmp_path, monkeypatch,
):
    """One catalog observation should retain facts needed by the run cursor."""
    path, _data = _stack(
        tmp_path / "scan.nxs",
        shape=(4, 3, 5),
        chunks=(2, 3, 5),
    )
    opens = _count_h5_opens(monkeypatch)

    session = DirectoryIndexSession(max_probes_per_observation=1)
    try:
        session.configure(tmp_path, suffixes=(".nxs",))
        observation = session.observe()
        item = observation.candidates[0]
        descriptor = _retained_descriptor(item)
        plan = RunCandidatePlan.from_observation(observation)
        catalog_opens = opens.count(str(path))

        # Model the next production owner. Reading pixels still requires one
        # cursor handle; readiness/layout/count discovery must not require
        # three earlier handles or lose the descriptor it already resolved.
        with ContainerCursor(path, candidate=item.candidate) as cursor:
            cursor_count = cursor.frame_count

        diagnostic = (
            observation.content_opens,
            catalog_opens,
            getattr(descriptor, "frame_count", None),
            cursor_count,
            opens.count(str(path)),
        )

        # Desired: one logical candidate probe, one physical catalog open with
        # a retained four-frame descriptor, then one cursor open for pixels.
        # Current captured result at 41292079 is (1, 3, None, 4, 4).
        assert diagnostic == (1, 1, 4, 4, 2), (
            "split source-fact authorities remain visible as "
            f"{diagnostic}; probe result type={type(item.result).__name__}, "
            f"slots={getattr(item.result, '__slots__', ())}"
        )
        assert plan.descriptor_for(item.candidate) is descriptor
        assert plan.frame_count_snapshot() == {
            str(path): (item.candidate.version_stamp, 4),
        }
    finally:
        session.close()
