"""Shared fallback for a retained workspace without a saved Project."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True, slots=True)
class ProjectDefaults:
    read: Callable[[str], str]
    write: Callable[[str, str], None]
    output_file: bool = False

    def location(self) -> tuple[str, str]:
        project = str(self.read("project_root") or "").strip()
        output = str(self.read("output_path") or "").strip()
        if output:
            path = Path(output).expanduser()
            if not path.is_absolute():
                path = Path(project) / path
            output = str(path.parent if self.output_file else path)
        return project, output

    def inherit(self, location: tuple[str, str]) -> None:
        if str(self.read("project_root") or "").strip() or not location[0]:
            return
        self.write("project_root", location[0])
        if location[1]:
            output = location[1]
            if self.output_file:
                output = str(Path(output) / Path(self.read("output_path")).name)
            self.write("output_path", output)
