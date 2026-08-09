"""PublicationStore RAM-aware and explicit-cap policy coverage."""


def test_publication_store_default_is_ram_aware(monkeypatch):
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "24")
    from xdart.modules.frame_publication import PublicationStore
    assert PublicationStore()._max_heavy_items == 24


def test_publication_store_explicit_cap_still_honored():
    from xdart.modules.frame_publication import PublicationStore
    assert PublicationStore(max_heavy_items=7)._max_heavy_items == 7
