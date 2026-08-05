"""Project-owned pyFAI FiberIntegrator policy.

Imported lazily by :mod:`xrd_tools.integrate.gid` so importing the public
headless module does not select a Qt binding through pyFAI.
"""

from pyFAI.integrator.fiber import FiberIntegrator


class _XrdToolsFiberIntegrator(FiberIntegrator):
    """Preserve full geometry reset without forced process-wide GC."""

    def reset(self, collect_garbage: bool = True) -> None:
        # ``collect_garbage`` is intentionally ignored for this project-owned
        # integrator.  super().reset(False) still calls Geometry.reset() and
        # resets/pops every cached integration engine; ordinary Python GC owns
        # eventual cyclic-object collection without serializing every frame.
        super().reset(collect_garbage=False)


__all__: list[str] = []
