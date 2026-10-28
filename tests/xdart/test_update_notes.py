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
