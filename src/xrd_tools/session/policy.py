# -*- coding: utf-8 -*-
"""The ONE run-policy owner: each run owns one immutable SessionPolicy, so a
policy built inside the cadence decision is a second owner.  Stdlib only."""
from __future__ import annotations
import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

__all__ = ["FlushPolicy", "SessionPolicy", "SessionResourceRequirements",
           "SessionResourceAllocation", "SessionEnvelopeError", "floor_bytes",
           "resolve_session_policy", "requirements_from", "minimum_bytes",
           "default_envelope_bytes"]

INTEGRATOR_RESERVE_BYTES = _GIB = 1024 ** 3   #: one worker's integrator copy
MAX_MODES_1D, MAX_MODES_2D = 5, 3      #: the finite schema maxima
PREFETCH_QUEUE_ENV = "XDART_PREFETCH_QUEUE_SIZE"   #: the ONE clamped request
DEFAULT_PREFETCH_QUEUE_DEPTH, DEFAULT_ENVELOPE_FRACTION = 4, 0.25
CATEGORIES = ("source_native", "staging", "records", "publication", "worker")
#: the one deterministic order spare capacity is granted in
GRANT_ORDER = ("workers", "reduction_inflight", "queue_depth",
               "owner_block_bytes", "staging_items", "record_heavy_items",
               "publication_heavy_items", "thumbnail_items", "record_items",
               "publication_items")


@dataclass(frozen=True, slots=True)
class FlushPolicy:
    """``cap - margin`` is the hard persist-before-evict bound."""
    interval: int = 8
    cap: int = 64
    margin: int = 8

    def hard_threshold(self) -> int:
        return max(1, self.cap - self.margin)

    def should_flush(self, *, frames_since_flush: int,
                     unsaved_in_memory: int | None = None,
                     force: bool = False) -> bool:
        """force is force-IF-PENDING: the empty-buffer check precedes it."""
        if frames_since_flush <= 0:
            return False
        if force:
            return True
        if frames_since_flush >= self.interval:
            return True
        pressure = (frames_since_flush if unsaved_in_memory is None
                    else unsaved_in_memory)
        return pressure >= self.hard_threshold()


@dataclass(frozen=True, slots=True)
class SessionPolicy:
    """One cadence owner plus, for a coordinated run, one resource allocation."""
    flush: FlushPolicy = FlushPolicy()
    allocation: SessionResourceAllocation | None = None

    def should_flush(self, *, frames_since_flush: int,
                     unsaved_in_memory: int | None = None,
                     force: bool = False) -> bool:
        return self.flush.should_flush(frames_since_flush=frames_since_flush,
                                       unsaved_in_memory=unsaved_in_memory,
                                       force=force)


def _int(name: str, value, *, low: int = 0, high: int | None = None) -> int:
    """Reject booleans/non-integral values, then enforce the bound."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-boolean integer; got {value!r}")
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name}={value} is outside {low}..{high}")
    return value

@dataclass(frozen=True, slots=True)
class SessionResourceRequirements:
    """Descriptor + plan facts; its complete value IS the fingerprint."""
    height: int
    width: int
    native_itemsize: int
    background_bytes: int = 0
    modes_1d: int = 0
    modes_2d: int = 0
    npt_1d: int = 0
    npt_rad: int = 0
    npt_azim: int = 0
    sigma_1d: int = 0
    sigma_2d: int = 0
    worker_background_bytes: int = 0
    def __post_init__(self) -> None:
        for n in ("height", "width", "native_itemsize"):
            _int(n, getattr(self, n), low=1)
        for n in ("background_bytes", "worker_background_bytes",
                  "npt_1d", "npt_rad", "npt_azim"):
            _int(n, getattr(self, n))
        _int("modes_1d", self.modes_1d, high=MAX_MODES_1D)
        _int("modes_2d", self.modes_2d, high=MAX_MODES_2D)
        _int("sigma_1d", self.sigma_1d, high=1)
        _int("sigma_2d", self.sigma_2d, high=1)
        if (self.modes_1d and self.npt_1d <= 0) or (
                self.modes_2d and min(self.npt_rad, self.npt_azim) <= 0):
            raise ValueError("an enabled mode needs positive output-grid sizes")
    @property
    def pixels(self) -> int: return self.height * self.width
    @property
    def native_frame_bytes(self) -> int: return self.pixels * self.native_itemsize
    @property
    def thumbnail_bytes(self) -> int: return 4 * min(self.pixels, 256 * 256)
    @property
    def result_1d_bytes(self) -> int:
        return self.modes_1d * 8 * self.npt_1d * (2 + self.sigma_1d)
    @property
    def result_2d_bytes(self) -> int:
        return self.modes_2d * 8 * (
            (1 + self.sigma_2d) * self.npt_rad * self.npt_azim
            + self.npt_rad + self.npt_azim)
    @property
    def fingerprint(self) -> tuple:
        return tuple(getattr(self, name) for name in self.__slots__)
@dataclass(frozen=True, slots=True)
class SessionResourceAllocation:
    """One immutable grant: requirements, counts, categories, ``M``/``F``."""
    requirements: SessionResourceRequirements
    envelope_bytes: int
    counts: Mapping[str, int]
    categories: Mapping[str, int]
    minimum_bytes: int
    floor_bytes: int
    assigned_bytes: int
    origin: str
    oversize_excess_bytes: int = 0
    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))
        object.__setattr__(self, "categories",
                           MappingProxyType(dict(self.categories)))
    def __getattr__(self, name: str) -> int:
        try:
            return self.counts[name]
        except KeyError:
            raise AttributeError(name) from None
class SessionEnvelopeError(ValueError):
    """Typed refusal, before any pixel/read-plan/queue/sink/executor effect."""
    def __init__(self, *, required_bytes: int, available_bytes: int,
                 floor_bytes: int, categories, requirements) -> None:
        super().__init__(
            f"session envelope {available_bytes} B cannot hold the fixed floor "
            f"{floor_bytes} B (complete minimum {required_bytes} B)")
        self.required_bytes = required_bytes
        self.available_bytes = available_bytes
        self.floor_bytes = floor_bytes
        self.categories = MappingProxyType(dict(categories))
        self.requirements = requirements
def _declared_modes(plan, extra, key: str, enabled: bool, cap: int) -> int:
    """Absent -> schema max; declared -> exact unique count; else reject."""
    declared = getattr(plan, key, None)
    if declared is None:
        declared = extra.get(key)
    if not enabled:
        if declared is not None:
            raise ValueError(f"{key} declared for a disabled dimension")
        return 0
    if declared is None:
        return cap
    if isinstance(declared, (str, bytes)) or not hasattr(declared, "__iter__"):
        raise ValueError(f"{key} must be a sequence of mode names")
    count = len(set(declared))
    if not 1 <= count <= cap:
        raise ValueError(
            f"{key} declares {count} modes; the finite schema maximum is {cap}")
    return count
def _mode_counts(plan, one_d, two_d) -> tuple[int, int]:
    """Standard: one mode per dimension; GI charges its declared sets."""
    if getattr(plan, "gi", None) is None:
        return (1 if one_d is not None else 0, 1 if two_d is not None else 0)
    extra = getattr(plan, "extra", None) or {}
    return (_declared_modes(plan, extra, "enabled_modes_1d",
                            one_d is not None, MAX_MODES_1D),
            _declared_modes(plan, extra, "enabled_modes_2d",
                            two_d is not None, MAX_MODES_2D))
def requirements_from(descriptor, plan, *, background_bytes: int = 0,
                     worker_background_bytes: int = 0
                     ) -> SessionResourceRequirements:
    """Combine a pixel-free descriptor with the plan; duck-typed, stdlib-only."""
    shape = tuple(getattr(descriptor, "frame_shape", None) or ())
    dtype = getattr(descriptor, "dtype", None)
    if len(shape) < 2 or dtype is None:
        raise ValueError("an exact detector shape and native dtype are "
                         "required; a fallback envelope is refused")
    one_d = getattr(plan, "integration_1d", None)
    two_d = getattr(plan, "integration_2d", None)
    modes_1d, modes_2d = _mode_counts(plan, one_d, two_d)
    return SessionResourceRequirements(
        height=int(shape[-2]), width=int(shape[-1]),
        native_itemsize=int(dtype.itemsize),
        background_bytes=background_bytes,
        worker_background_bytes=worker_background_bytes,
        modes_1d=modes_1d, modes_2d=modes_2d,
        npt_1d=int(getattr(one_d, "npt", 0) or 0),
        npt_rad=int(getattr(two_d, "npt_rad", 0) or 0),
        npt_azim=int(getattr(two_d, "npt_azim", 0) or 0),
        sigma_1d=1 if getattr(one_d, "error_model", None) else 0,
        sigma_2d=1 if getattr(two_d, "error_model", None) else 0)
def _categories(req: SessionResourceRequirements, counts: dict, *,
                detector: bool = True) -> dict:
    """The five ratified totals; ``detector=False`` strips only detector terms
    (``F``).  Staging/records/publication are never a union."""
    P = req.native_frame_bytes if detector else 0
    G = req.background_bytes if detector else 0
    T = req.thumbnail_bytes if detector else 0
    per_worker = (8 * req.pixels + 4 * req.pixels + req.worker_background_bytes
                  if detector else 0)
    owner = counts["owner_block_bytes"] if detector else 0
    A1, A2 = req.result_1d_bytes, req.result_2d_bytes
    return {
        # +1 is the producer-held frame; in-flight views ALIAS it, charged once.
        "source_native": owner + (counts["queue_depth"] + 1
                                  + counts["reduction_inflight"]) * P,
        "staging": counts["staging_items"] * (P + G + A1 + A2 + T),
        "records": (counts["record_items"] * A1
                    + counts["record_heavy_items"] * A2),
        "publication": (counts["publication_items"] * A1
                        + counts["publication_heavy_items"] * (P + G + A2)
                        + counts["thumbnail_items"] * T),
        "worker": (counts["workers"] * (INTEGRATOR_RESERVE_BYTES + per_worker)
                   + counts["reduction_inflight"] * (A1 + A2)),
    }
def _minimum_counts(req: SessionResourceRequirements) -> dict:
    return {name: low for name, (low, _unit) in _tunables(req).items()}
def minimum_bytes(req: SessionResourceRequirements) -> int:
    """``M`` — the complete minimum with every tunable count at its minimum."""
    return sum(_categories(req, _minimum_counts(req)).values())
def floor_bytes(req: SessionResourceRequirements) -> int:
    """``M`` less only detector terms; the reserve and every ``A1``/``A2`` stay."""
    return sum(_categories(req, _minimum_counts(req), detector=False).values())
def _tunables(req: SessionResourceRequirements) -> dict:
    """``{name: (minimum, unit_bytes)}``; a worker also forces an in-flight slot."""
    P, G, T = req.native_frame_bytes, req.background_bytes, req.thumbnail_bytes
    A1, A2 = req.result_1d_bytes, req.result_2d_bytes
    return {
        "workers": (1, INTEGRATOR_RESERVE_BYTES + 12 * req.pixels
                    + req.worker_background_bytes + P + A1 + A2),
        "reduction_inflight": (1, P + A1 + A2),
        "queue_depth": (1, P),
        "owner_block_bytes": (P, P),
        "staging_items": (1, P + G + A1 + A2 + T),
        "record_heavy_items": (1, A2),
        "publication_heavy_items": (1, P + G + A2),
        "thumbnail_items": (1, T),
        "record_items": (1, A1),
        "publication_items": (1, A1),
    }
def _bounds(req: SessionResourceRequirements, requests) -> dict:
    """The ten request keys are the grant names; each a non-boolean integer
    at/above its minimum, an UPPER bound.  Runs on EVERY route."""
    table = _tunables(req)
    unknown = set(requests or ()) - set(table)
    if unknown:
        raise ValueError(f"unknown resource requests: {sorted(unknown)}")
    return {name: (_int(name, requests[name], low=low)
                   if requests and name in requests else low)
            for name, (low, _u) in table.items()}
def _grant(req: SessionResourceRequirements, bounds: dict,
           envelope: int, *, automatic_inflight: bool = False) -> dict:
    """Raise each count toward its bound in ``GRANT_ORDER``, recomputing the
    remainder each step so no category can overrun."""
    table = _tunables(req)
    counts = {name: low for name, (low, _u) in table.items()}
    wants = dict(bounds)
    if automatic_inflight:
        wants["reduction_inflight"] = 2 * wants["workers"]
    # A worker forces an in-flight slot, so the worker grant is capped by the
    # INFLIGHT request: clamp workers down, never raise inflight.
    wants["workers"] = min(wants["workers"], wants["reduction_inflight"])
    spent = sum(_categories(req, counts).values())
    for name in GRANT_ORDER:
        unit = table[name][1]
        if unit <= 0:
            continue
        step = req.native_frame_bytes if name == "owner_block_bytes" else 1
        take = max(0, min((wants[name] - counts[name]) // step,
                          (envelope - spent) // unit))
        counts[name] += take * step
        spent += take * unit
        if name == "workers":
            counts["reduction_inflight"] = max(counts["reduction_inflight"],
                                               counts["workers"])
            if automatic_inflight:
                wants["reduction_inflight"] = 2 * counts["workers"]
    return counts
def _prefetch_queue_request(env) -> int:
    """The ONE parse of the prefetch-depth request, off the frozen snapshot."""
    raw = env.get(PREFETCH_QUEUE_ENV)
    try:
        return max(1, int(str(raw).strip()))
    except (TypeError, ValueError):
        return DEFAULT_PREFETCH_QUEUE_DEPTH
def default_envelope_bytes(env=None) -> int:
    """A quarter of total RAM; Python/Qt/file handles are OUTSIDE it."""
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        total = 0
    return int(total * DEFAULT_ENVELOPE_FRACTION) if total > 0 else 8 * _GIB
def _default_requests(req: SessionResourceRequirements, requested_workers,
                      env) -> dict:
    """These requests feed this ONE resolution; consumers receive grants."""
    from xrd_tools.core.staging import (heavy_window, live_record_store_max_items,
                                        reduction_worker_cap,
                                        source_block_budget_bytes)
    window = heavy_window(8 * req.pixels, env=env)
    workers = reduction_worker_cap(requested_workers, env=env)
    light = live_record_store_max_items(req.npt_1d or None)
    return {"queue_depth": _prefetch_queue_request(env),
            "owner_block_bytes": source_block_budget_bytes(env=env),
            "staging_items": window, "record_heavy_items": window,
            "publication_heavy_items": window, "thumbnail_items": 512,
            "record_items": light, "publication_items": light,
            "workers": workers, "reduction_inflight": 2 * workers}
def _build(req, counts, envelope, floor, minimum, origin,
           oversize=0) -> SessionResourceAllocation:
    cats = _categories(req, counts)
    return SessionResourceAllocation(
        requirements=req, envelope_bytes=envelope, categories=cats,
        minimum_bytes=minimum, floor_bytes=floor,
        assigned_bytes=sum(cats.values()), origin=origin,
        oversize_excess_bytes=oversize, counts=dict(counts))
def _validate_explicit(req, allocation, envelope, floor, minimum):
    """Explicit allocations are EVIDENCE: recomputed, rejected before effects."""
    if allocation.requirements.fingerprint != req.fingerprint:
        raise ValueError("explicit allocation was built for other requirements")
    counts = dict(allocation.counts)
    low = _minimum_counts(req)
    if set(counts) != set(low):
        raise ValueError("explicit allocation must carry exactly the ten grants")
    for name, value in counts.items():
        _int(f"explicit {name}", value, low=low[name])
    if counts["owner_block_bytes"] % req.native_frame_bytes:
        raise ValueError("owner_block_bytes must be whole native frames")
    if set(allocation.categories) != set(CATEGORIES):
        raise ValueError("explicit allocation must carry exactly five categories")
    for name, value in allocation.categories.items():
        _int(f"category {name}", value)
    for name in ("envelope_bytes", "minimum_bytes", "floor_bytes",
                 "assigned_bytes", "oversize_excess_bytes"):
        _int(name, getattr(allocation, name))
    if not 1 <= counts["workers"] <= counts["reduction_inflight"]:
        raise ValueError(
            "explicit allocation violates 1 <= workers <= reduction_inflight")
    if envelope < minimum and counts != low:
        raise ValueError(
            "the detector-driven oversize exception admits only exact minimum "
            "counts; an above-minimum grant cannot ride it")
    expected = _build(req, counts, envelope, floor, minimum,
                      allocation.origin, oversize=max(0, minimum - envelope))
    if dict(allocation.categories) != dict(expected.categories):
        raise ValueError("explicit allocation category breakdown is not exact")
    for name in ("minimum_bytes", "floor_bytes", "assigned_bytes",
                 "envelope_bytes", "oversize_excess_bytes"):
        if getattr(allocation, name) != getattr(expected, name):
            raise ValueError(f"explicit allocation {name} is not exact")
    if not expected.oversize_excess_bytes and expected.assigned_bytes > envelope:
        raise ValueError("explicit allocation exceeds its envelope")
def resolve_session_policy(requirements: SessionResourceRequirements, *,
                           envelope_bytes: int | None = None,
                           flush: FlushPolicy | None = None,
                           requests=None, allocation=None,
                           requested_workers=None, env=None) -> SessionPolicy:
    """The ONE resolver.  Below ``F`` raises :class:`SessionEnvelopeError`;
    ``F <= envelope < M`` yields one oversize config at exact minima."""
    req = requirements
    # Validate EACH worker spelling alone before comparing: ``True == 1`` and
    # ``1.0 == 1``, so comparing first lets the merge erase a malformed value.
    if requested_workers is not None:
        _int("requested_workers", requested_workers, low=1)
    if requests and "workers" in requests:
        _int("workers", requests["workers"], low=1)
        if requested_workers is not None and requests["workers"] != requested_workers:
            raise ValueError("requested_workers and requests['workers'] disagree")
    if requested_workers is not None and allocation is not None \
            and not (requests and "workers" in requests):
        requests = dict(requests or {}, workers=requested_workers)
    frozen_env = dict(os.environ) if env is None else dict(env)
    if allocation is not None:
        declared = int(allocation.envelope_bytes)
        if envelope_bytes is not None and int(envelope_bytes) != declared:
            raise ValueError(
                f"envelope override {int(envelope_bytes)} does not match the "
                f"explicit allocation's declared {declared}")
        envelope = declared
    else:
        envelope = int(default_envelope_bytes(frozen_env)
                       if envelope_bytes is None else envelope_bytes)
    bounds = _bounds(req, requests)
    floor, minimum = floor_bytes(req), minimum_bytes(req)
    if envelope < floor:
        raise SessionEnvelopeError(
            required_bytes=minimum, available_bytes=envelope, floor_bytes=floor,
            requirements=req, categories=_categories(req, _minimum_counts(req)))
    if allocation is not None:
        _validate_explicit(req, allocation, envelope, floor, minimum)
        for name, bound in bounds.items():
            if requests and name in requests and allocation.counts[name] > bound:
                raise ValueError(
                    f"explicit {name}={allocation.counts[name]} exceeds its "
                    f"request bound {bound}")
        built = allocation          # the caller's EXACT object survives
    elif envelope < minimum:
        built = _build(req, _minimum_counts(req), envelope, floor, minimum,
                       "automatic", oversize=minimum - envelope)
    else:
        wanted = _bounds(req, {**_default_requests(req, requested_workers,
                                                   frozen_env),
                               **(requests or {})})
        built = _build(req, _grant(
            req, wanted, envelope,
            automatic_inflight=not (requests and "reduction_inflight" in requests),
        ), envelope, floor,
                       minimum, "automatic")
    return SessionPolicy(flush=flush if flush is not None else FlushPolicy(),
                         allocation=built)
