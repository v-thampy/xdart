"""Release summaries stay bounded, plain and filtered by installed/available versions."""
from io import BytesIO

from xdart.modules import update_notes as notes


MARKDOWN = '''# Releases
## 1.3.0
### Added
- **Future feature:** excluded
## 1.2.8 (today)
### Added
- **Clearer update controls:** long technical body
  More implementation details.
### Fixes
- **Keeps your selection:** more details
## 1.2.7
### Changed
- **Faster browsing:** details
## 1.2.6
### Added
- **Already installed:** excluded
'''


def test_summary_selects_only_newer_versions_and_short_bullets():
    text = notes.summarize(MARKDOWN, '1.2.6', '1.2.8')
    assert text.index('Version 1.2.8') < text.index('Version 1.2.7')
    assert 'New\n• Clearer update controls' in text and 'Fixed\n• Keeps your selection' in text
    assert 'Already installed' not in text and 'Future feature' not in text and 'technical body' not in text


def test_notes_fetch_is_tag_pinned_and_bounded():
    seen = []
    def fetch(request, **kwargs):
        seen.append((request.full_url, kwargs['timeout']))
        return BytesIO(MARKDOWN.encode())
    result = notes.fetch_release_notes('1.2.6', '1.2.8', opener=fetch)
    assert result.available and '/v1.2.8/RELEASE_NOTES.md' in seen[0][0] and seen[0][1] == 5
    assert result.url.endswith('/tag/v1.2.8')
    oversized = notes.fetch_release_notes('1.2.6', '1.2.8', opener=lambda *a, **k: BytesIO(b'x' * ((1<<20)+1)))
    assert not oversized.available


def test_missing_target_and_offline_notes_leave_an_explicit_choice():
    def offline(*args, **kwargs):
        raise OSError('offline')
    assert not notes.fetch_release_notes('1.2.6', '1.2.8', opener=offline).available
    assert not notes.fetch_release_notes('1.2.6', '1.2.9', opener=lambda *a, **k: BytesIO(MARKDOWN.encode())).available
    assert 'choose Update' in notes.fetch_release_notes(None, None).text


def test_wrapped_lead_ins_and_plain_bullets_read_as_short_plain_text():
    markdown = '''## 2.0.0
### Added
- **A lead-in that wraps onto
  the next line:** and its body
- A plain bullet with **inner bold** and `code`; a second clause
  that wraps.
### Known limitations
- **Still slow on a network share:** details
## 1.9.0
'''
    text = notes.summarize(markdown, '1.9.0', '2.0.0')
    assert '• A lead-in that wraps onto the next line\n' in text
    assert '• A plain bullet with inner bold and code\n' in text
    assert 'Known limitations\n• Still slow on a network share' in text
    assert '**' not in text and '`' not in text and 'second clause' not in text


def test_a_version_without_bullets_says_so():
    assert 'Version 2.0.0\n• No summary for this version' in notes.summarize('## 2.0.0\nProse.\n## 1.9.0\n', '1.9.0', '2.0.0')


def test_the_shipped_release_notes_read_as_short_plain_points():
    import re
    from pathlib import Path
    from packaging.version import Version
    shipped = (Path(__file__).resolve().parents[2] / 'RELEASE_NOTES.md').read_text(encoding='utf-8')
    newest = max(Version(v) for v in re.findall(r'^##\s+v?(\d\S*)', shipped, re.M))
    text = notes.summarize(shipped, '0.0.1', str(newest))
    bullets = [line for line in text.splitlines() if line.startswith('• ')]
    assert len(bullets) > 50 and '**' not in text and '`' not in text
    assert max(len(line) for line in bullets) <= 2 + notes._MAX_CHARS


def test_a_local_notes_file_stands_in_for_the_tag(tmp_path, monkeypatch):
    local = tmp_path / 'notes.md'
    local.write_text(MARKDOWN, encoding='utf-8')
    monkeypatch.setenv('XDART_RELEASE_NOTES_FILE', str(local))
    def no_network(*args, **kwargs):
        raise AssertionError('the network must not be used')
    result = notes.fetch_release_notes('1.2.6', '1.2.8', opener=no_network)
    assert result.available and '• Clearer update controls' in result.text
