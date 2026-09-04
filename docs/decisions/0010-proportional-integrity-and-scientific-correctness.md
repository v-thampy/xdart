# ADR-0010: proportional integrity mechanisms and real-data scientific gates

**Status:** accepted · 2026-09-03

## Context

The vNext implementation accumulated recovery, rollback, ownership, and
authentication machinery around failures that are possible in principle but
rare in the finite, local workflows xdart actually runs.  Some of that machinery
became harder to reason about than the operation it protected, added closeout
latency, and required a large battery of tests for states that the simpler
immutable-source design can make unreachable.

At the same time, small synthetic arrays cannot establish that pyFAI
integration, stitching, reciprocal-space conversion, or persisted reloads are
scientifically correct on beamline data.  Synthetic fixtures and real data have
different jobs; treating either as a substitute for the other weakens the test
strategy.

## Decision

### Prefer a simple failure model

Finite transformations never mutate their raw or source artifact.  They publish
processed results into stable, operation-specific slots.  Ordinary streamed Run
retains its existing high-throughput output transaction and STOP behavior.
Derived finite operations either atomically replace their slot with a closed,
scientifically validated result or fail loudly and leave both the source and the
prior slot untouched.  An incomplete derived candidate may be discarded and the
operator may retry; it does not require a general rollback state machine.

The normal integrity toolbox for derived finite operations is deliberately
small:

- validate inputs and qualify any existing processed result slot;
- keep the source immutable;
- use one clearly owned writer for the candidate;
- close and scientifically validate the candidate before publication; and
- use a platform's dependable atomic-publication primitive when it is both
  available and materially useful.

### Use stable operation result slots

The public processed namespace is deliberately human-readable and stable.  For
an artifact family named `<family>`, the operation result slots are:

- `<family>_int1d.nexus`;
- `<family>_int2d.nexus`;
- `<family>_average.nexus`;
- `<family>_reintegrate1d.nexus`;
- `<family>_reintegrate2d.nexus`;
- `<family>_stitch1d.nexus`;
- `<family>_stitch2d.nexus`; and
- `<family>_rsm.nexus`.

Public filenames do not contain a version, content hash, operation identity,
timestamp, or random token.  Full science, source, operation, and publication
identities remain recorded inside provenance.  Repeating an operation for the
same family replaces that operation's processed result slot; it does not create
an accumulating public sequence of version-named files.

For ordinary streamed Run, this decision changes only target naming to the
`_int1d.nexus` and `_int2d.nexus` slots.  Run's hot integration, writer, and
output-transaction loop and its existing Overwrite and STOP semantics remain
unchanged.  Run is not converted to the derived-operation hidden-candidate
protocol.  A hard crash may therefore leave its public result or hidden backup
unresolved; automatic crash recovery is not guaranteed, while the raw/source
artifact remains safe.

Finite Average, Reintegration, Stitch, and RSM build a hidden candidate in the
target directory.  The candidate is closed and scientifically validated before
one atomic replacement publishes it at the stable slot.  The prior slot remains
visible and untouched until that publication step.  STOP during Average
discards the partial candidate and leaves the prior slot unchanged.  The same
pre-publication failure rule applies to the other finite operations.

Automatic recovery after a hard process or machine crash is explicitly out of
scope for these finite operations.  A pre-publication crash may leave a hidden
candidate for ordinary cleanup, while the raw/source artifact and prior public
slot remain safe.  A crash after atomic replacement leaves the new, already
closed and validated slot.  This bounded contract does not justify a journal,
backup graph, or cross-process recovery state machine.

Append and Live remain separate because they intentionally extend an existing
artifact.  Their durability policy does not change the finite-operation slot
contract.

The derived-operation output policy is implemented at admission and publication
boundaries.  It must not add work to or otherwise restructure Run's hot
integration, writer, or output-transaction loop, and it must preserve the
accepted performance floors used for promotion.

Additional journals, replay protocols, rollback graphs, cross-object
authentication, recovery epochs, or duplicated ownership layers are not added
merely because a corruption scenario can be imagined.  Such a mechanism needs
all of the following before it enters production:

1. a credible failure mode grounded in an observed incident, a reproducible
   fault, or a platform guarantee that demonstrably does not hold;
2. material likelihood or impact for an actual supported workflow;
3. evidence that source immutability, fail-loud validation, and operator retry
   are insufficient;
4. a measurable acceptance oracle, including fault-injection coverage when
   applicable; and
5. an identified owner and a removal condition so temporary safeguards do not
   become permanent ceremony.

Evidence may nominate a threat, but the project owner must explicitly approve
that threat as credible and material before it can authorize production design,
implementation, or a permanent test matrix.  Reviewer concern, a theoretical
possibility, or the mere existence of a prototype is not implicit approval.
Until that decision is recorded, the proposed safeguard remains unimplemented.

A nomination to the project owner must name the supported workflow, concrete
evidence, plausible likelihood and impact, why the normal immutable-source
model is insufficient, the smallest proposed safeguard, and its code, test,
performance, and maintenance cost.  The recorded owner disposition is one of
approve, reject, or defer; silence is not approval.

Existing mechanisms are not grandfathered.  During simplification, a mechanism
that cannot be justified as scientific correctness, ordinary bounded-resource
ownership, or a cheap local guard must be presented through the same owner gate.
Without an explicit approval record, it is a removal candidate together with
its mechanism-specific tests.

The burden of proof belongs to the extra mechanism.  A rare edge case does not
justify a large permanent state machine by default.  Cheap, local guards may
still be worthwhile; complexity must remain proportional to the demonstrated
risk.

Append and Live operations intentionally continue an existing artifact and may
need stronger durability guarantees.  Any such guarantees require a separate,
evidence-backed decision and must not leak recovery complexity into finite Run,
Average, Reintegration, Stitch, or RSM operations.

When an integrity mechanism is retired, its mechanism-specific tests and dead
states are retired with it.  Tests must not preserve obsolete architecture.

### Keep scientific correctness non-negotiable

Scientific correctness is a separate concern and is not relaxed by the
complexity rule.  Coordinate conventions, geometry, units, masks,
normalization, integration values, stitching, RSM transforms, and persisted
reload equivalence must remain explicit and testable.  A scientific mismatch is
not dismissed as an unlikely corruption edge case, and tolerances are not
widened to make a gate pass.

The evidence model has two layers:

- Small synthetic fixtures run frequently and pin contracts, boundary behavior,
  units, shapes, labels, and known numerical invariants.
- Authenticated real beamline data pins scientific fidelity, persisted-vs-direct
  equivalence, supported detector/file behavior, and representative performance.

Real-data gates are mandatory for promotion.  A promotion run must name and
authenticate its corpus, set `XDART_TEST_DATA` explicitly, record the exact test
selection and result, and fail rather than silently skip when required data is
missing.  The large private corpus need not be copied into ordinary hosted CI or
used for every unit test.

## Consequences

- Finite Run, Average, Reintegration, Stitch, and RSM share an understandable
  namespace: immutable source and stable processed result slots.
- Ordinary streamed Run keeps its existing high-throughput output transaction,
  Overwrite, and STOP semantics; only its target naming changes.
- Average, Reintegration, Stitch, and RSM use validated hidden candidates and
  atomic replacement, leaving the prior slot untouched until publication.
- Public names remain stable and readable while exact identities remain in
  provenance.
- Hard-crash auto-recovery is not part of the finite-operation contract.  A
  streamed Run may leave an unresolved public result or hidden backup, but its
  raw/source artifact remains safe.
- Code and tests devoted only to rollback or speculative recovery states can be
  removed as those routes are migrated.
- Fast deterministic tests remain useful during development, while promotion
  cannot be declared from synthetic evidence alone.
- New safety complexity requires concrete evidence and an explicit acceptance
  contract; scientific correctness remains a release blocker regardless of
  implementation convenience.
