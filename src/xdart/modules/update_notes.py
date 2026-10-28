"""Bounded, plain-language release summaries for the shared update offer (Qt-free)."""
from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.request import Request, urlopen

from packaging.version import InvalidVersion, Version

RELEASES_URL = "https://github.com/v-thampy/xdart/releases"
_MAX_BYTES = 1 << 20


@dataclass(frozen=True)
class ReleaseNotes:
    text: str
    url: str
    available: bool = True


def summarize(markdown: str, installed: str, target: str) -> str:
    """Only newer, published-version sections through target; bold bullet lead-ins are the short summaries."""
    low, high = Version(str(installed)), Version(str(target))
    releases, current, group = {}, None, ""
    names = {"added": "New", "new": "New", "changed": "Changed", "fixed": "Fixed", "fixes": "Fixed",
             "security": "Security"}
    for line in markdown.splitlines():
        if match := re.match(r"^##\s+v?(\S+)", line):
            try:
                version = Version(match[1])
            except InvalidVersion:
                current = None
                continue
            current = releases.setdefault(version, {}) if low < version <= high else None
            group = ""
        elif match := re.match(r"^###\s+(.+)", line):
            group = names.get(match[1].strip().lower(), "Changes")
        elif current is not None and (match := re.match(r"^[-*]\s+(.+)", line)):
            value = match[1].strip()
            bold = re.match(r"\*\*(.+?)\*\*", value)
            value = bold[1] if bold else value
            value = value.rstrip(":").replace("`", "").strip()
            if value:
                current.setdefault(group or "Changes", []).append(value[:300])
    if high not in releases:
        raise ValueError("the available version has no published release notes")
    lines = []
    for version in sorted(releases, reverse=True)[:20]:
        lines.append(f"Version {version}")
        for section, bullets in releases[version].items():
            lines.append(section)
            lines.extend(f"• {bullet}" for bullet in bullets[:50])
        lines.append("")
    return "\n".join(lines).strip()[:40000] or "This release has no summary yet."


def fetch_release_notes(installed, target, *, opener=urlopen) -> ReleaseNotes:
    """Fetch one immutable tag off the GUI thread. Network/format failure never prevents an explicit update."""
    link = RELEASES_URL
    try:
        version = Version(str(target))
        link = f"{RELEASES_URL}/tag/v{version}"
        request = Request(f"https://raw.githubusercontent.com/v-thampy/xdart/v{version}/RELEASE_NOTES.md",
                          headers={"User-Agent": "xdart-update-notes"})
        with opener(request, timeout=5) as response:
            raw = response.read(_MAX_BYTES + 1)
        if len(raw) > _MAX_BYTES:
            raise ValueError("release notes exceed the size limit")
        return ReleaseNotes(summarize(raw.decode("utf-8"), installed, target), link)
    except Exception:
        return ReleaseNotes("Release notes could not be loaded. You can read them with Open release notes, "
                            "or choose Update to continue.", link, False)
