"""Bounded, plain-language release summaries for the shared update offer (Qt-free).

``XDART_RELEASE_NOTES_FILE`` (testing aid): read the notes from this local Markdown file instead of the target's
GitHub tag. A development build has no tag, so this is the only way to see a filled list before a real release.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import re
from urllib.request import Request, urlopen

from packaging.version import InvalidVersion, Version

RELEASES_URL = "https://github.com/v-thampy/xdart/releases"
_MAX_BYTES = 1 << 20
#: Longest summary line; a longer lead-in or sentence is cut at a word and ends with "…".
_MAX_CHARS = 160


@dataclass(frozen=True)
class ReleaseNotes:
    text: str
    url: str
    available: bool = True


def _short(text: str) -> str:
    """A bullet's summary: its bold lead-in (which may wrap onto the next lines), else its first sentence, as plain
    text of at most ``_MAX_CHARS`` characters."""
    bold = re.match(r"\*\*(.+?)\*\*", text)
    value = bold[1] if bold else re.split(r"(?<=[.;])\s", text, maxsplit=1)[0]
    value = value.replace("**", "").replace("`", "").strip().rstrip(":;.").strip()
    if len(value) > _MAX_CHARS:
        value = value[:_MAX_CHARS - 1].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return value


def summarize(markdown: str, installed: str, target: str) -> str:
    """Only newer, published-version sections through target, grouped by their headings; one short line per bullet."""
    low, high = Version(str(installed)), Version(str(target))
    releases, current, group, bullet = {}, None, "", None
    names = {"added": "New", "new": "New", "changed": "Changed", "fixed": "Fixed", "fixes": "Fixed",
             "security": "Security"}

    def finish():
        nonlocal bullet
        value = _short(" ".join(bullet)) if bullet else ""
        if value and current is not None:
            current.setdefault(group or "Changes", []).append(value)
        bullet = None

    for line in markdown.splitlines():
        if bullet is not None and line[:1].isspace() and line.strip() and not re.match(r"^\s*[-*]\s", line):
            bullet.append(line.strip())   # a wrapped line of the bullet above
            continue
        finish()
        if match := re.match(r"^##\s+v?(\S+)", line):
            try:
                version = Version(match[1])
            except InvalidVersion:
                current = None
                continue
            current = releases.setdefault(version, {}) if low < version <= high else None
            group = ""
        elif match := re.match(r"^###\s+(.+)", line):
            heading = match[1].strip()
            group = names.get(heading.lower(), heading)
        elif current is not None and (match := re.match(r"^[-*]\s+(.+)", line)):
            bullet = [match[1].strip()]
    finish()
    if high not in releases:
        raise ValueError("the available version has no published release notes")
    lines = []
    for version in sorted(releases, reverse=True)[:20]:
        lines.append(f"Version {version}")
        for section, bullets in releases[version].items():
            lines.append(section)
            lines.extend(f"• {bullet}" for bullet in bullets[:50])
        if not releases[version]:
            lines.append("• No summary for this version; Open release notes has the details.")
        lines.append("")
    return "\n".join(lines).strip()[:40000]


def fetch_release_notes(installed, target, *, opener=urlopen) -> ReleaseNotes:
    """Fetch one immutable tag off the GUI thread. Network/format failure never prevents an explicit update."""
    link = RELEASES_URL
    try:
        version = Version(str(target))
        link = f"{RELEASES_URL}/tag/v{version}"
        local = os.environ.get("XDART_RELEASE_NOTES_FILE")
        if local:
            with open(local, "rb") as fh:
                raw = fh.read(_MAX_BYTES + 1)
        else:
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
