# xrd_tools/perf/ — headless performance measurement vocabulary (v1.1.2 M0).
"""Headless, Qt-free performance vocabulary for the NXS directory workstream.

M0 (measurement gate) defines a *small, stable* event/counter schema
(:mod:`xrd_tools.perf.metrics`) and a repeatable benchmark harness
(``scripts/nxs_directory_benchmark.py``) that drives the HEADLESS
discover → probe → open → reduce pipeline and records a before/after baseline.

Deliberately minimal: M0 populates only the seams it can measure from the
headless path without changing any runtime behavior.  Later phases add their own
counters *at their owner* (R1/R2: source open/read + directory index; C1/C2:
resource-build counts; S: durable-write accounting), extending this same
vocabulary rather than a second one.  No Qt, no GUI state model.
"""

from xrd_tools.perf.metrics import (  # noqa: F401
    OBSERVATION_SOURCES,
    OPEN_CATEGORIES,
    SCHEMA_VERSION,
    ContainerMetrics,
    EnvProvenance,
    RunMetrics,
    Timer,
    new_open_counts,
    summarize_runs,
    timed,
    write_json,
)

__all__ = [
    "OBSERVATION_SOURCES",
    "OPEN_CATEGORIES",
    "SCHEMA_VERSION",
    "ContainerMetrics",
    "EnvProvenance",
    "RunMetrics",
    "Timer",
    "new_open_counts",
    "summarize_runs",
    "timed",
    "write_json",
]
