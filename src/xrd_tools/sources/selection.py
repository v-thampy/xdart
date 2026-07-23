"""Typed, value-only source selections frozen at Run boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from xrd_tools.core.scan import SourceKind, SourceSpec

_FRAME_SUFFIX = re.compile(r"^(.*?)[_-](\d+)$")


@dataclass(frozen=True, slots=True)
class DirectorySourceSpec:
    """Immutable Image-Directory intent with no discovered file contents."""

    root: Path
    recursive: bool = False
    suffixes: tuple[str, ...] = ()
    name_filter: str | None = None
    generation: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser())
        object.__setattr__(
            self,
            "suffixes",
            tuple(str(value).lower() for value in self.suffixes if value),
        )
        object.__setattr__(
            self,
            "name_filter",
            str(self.name_filter) if self.name_filter else None,
        )
        object.__setattr__(self, "generation", int(self.generation))


def image_series_spec(selected_file: str | Path) -> SourceSpec:
    """Freeze the complete numbered series containing *selected_file*.

    The selected member supplies its own suffix and parsed scan stem.  Hidden
    fields from another source mode therefore cannot affect membership, and
    selecting frame 4 does not silently redefine the series as frames 4 onward.
    Enumeration is name-only; no image payload is opened.
    """
    selected = Path(selected_file).expanduser()
    match = _FRAME_SUFFIX.match(selected.stem)
    suffix = selected.suffix
    if match is None or not suffix:
        files = (selected,)
        scan_name = selected.stem
        pattern = selected.name
    else:
        scan_name = match.group(1)
        candidate_re = re.compile(
            rf"^{re.escape(scan_name)}[_-](\d+){re.escape(suffix)}$",
            re.IGNORECASE,
        )
        members: list[tuple[int, str, Path]] = []
        try:
            siblings = selected.parent.iterdir()
        except OSError:
            siblings = ()
        for sibling in siblings:
            if not sibling.is_file():
                continue
            sibling_match = candidate_re.match(sibling.name)
            if sibling_match is not None:
                members.append((
                    int(sibling_match.group(1)),
                    sibling.name.casefold(),
                    sibling,
                ))
        files = tuple(
            member[2] for member in sorted(members, key=lambda item: item[:2]))
        if not files and selected.is_file():
            files = (selected,)
        pattern = f"{scan_name}_*{suffix}"

    return SourceSpec(
        selected.parent,
        SourceKind.TIFF_SERIES,
        options={
            "selected_file": str(selected),
            "files": tuple(str(path) for path in files),
            "pattern": pattern,
            "scan_name": scan_name,
        },
    )


__all__ = ["DirectorySourceSpec", "image_series_spec"]
