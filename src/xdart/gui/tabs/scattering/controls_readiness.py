"""Filesystem-backed Controls readiness facts outside passive Qt views."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SectionHeaderProjection:
    """One qualified Controls header fact, independent of Qt ownership."""

    text: str
    ready: bool = False
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ControlsReadinessProjection:
    """Immutable readiness values consumed by the passive Controls view."""

    project: SectionHeaderProjection = SectionHeaderProjection("")
    experiment: SectionHeaderProjection = SectionHeaderProjection("")
    processing: SectionHeaderProjection = SectionHeaderProjection("")


def project_directories_ready(
    project_root: object,
    save_path: object,
) -> tuple[bool, str]:
    """Qualify the existing project and existing-or-creatable save target."""

    project = _path(project_root)
    output = _path(save_path)
    if project is None or not project.is_dir():
        return False, "Choose an existing project folder."
    if output is None:
        return False, "Choose a save directory."
    if not _directory_target_is_ready(output):
        return False, "Choose a valid, creatable save directory target."
    return (
        True,
        "Project folder and save directory target are ready.",
    )


def _path(value: object) -> Path | None:
    if type(value) is not str or not value.strip():
        return None
    try:
        return Path(value).expanduser()
    except (OSError, RuntimeError, ValueError):
        return None


def _directory_target_is_ready(path: Path) -> bool:
    """Check existing-or-creatable directory truth without mutating disk."""

    try:
        if path.exists():
            return path.is_dir() and os.access(
                path,
                os.W_OK | os.X_OK,
            )
        # A not-yet-created directory may legitimately contain dots. Reject
        # only the explicit processed-artifact shape this directory field used
        # to receive, rather than treating every suffix as a filename.
        if path.suffix.casefold() == ".nxs":
            return False
        ancestor = path.parent
        while not ancestor.exists():
            parent = ancestor.parent
            if parent == ancestor:
                return False
            ancestor = parent
        return ancestor.is_dir() and os.access(
            ancestor,
            os.W_OK | os.X_OK,
        )
    except (OSError, RuntimeError, ValueError):
        return False

__all__ = [
    "ControlsReadinessProjection",
    "SectionHeaderProjection",
    "project_directories_ready",
]
