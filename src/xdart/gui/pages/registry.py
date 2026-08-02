"""Explicit, deterministic descriptor registry with no construction state."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .descriptors import Descriptor, PageDescriptor, ToolDescriptor
from .values import PageKey


class DuplicatePageKeyError(ValueError):
    pass


class RegistryFrozenError(RuntimeError):
    pass


class PageRegistry:
    def __init__(self, descriptors: Iterable[Descriptor] = ()) -> None:
        self._descriptors: dict[PageKey, Descriptor] = {}
        self._frozen = False
        self._ordered: tuple[Descriptor, ...] = ()
        for descriptor in descriptors:
            self.register(descriptor)

    def register(self, descriptor: Descriptor) -> None:
        if self._frozen:
            raise RegistryFrozenError("page registry is frozen")
        if not isinstance(descriptor, (PageDescriptor, ToolDescriptor)):
            raise TypeError("registry accepts page or tool descriptors")
        if descriptor.key in self._descriptors:
            raise DuplicatePageKeyError(str(descriptor.key))
        self._descriptors[descriptor.key] = descriptor

    def freeze(self) -> "PageRegistry":
        if not self._frozen:
            self._ordered = tuple(sorted(
                self._descriptors.values(),
                key=lambda item: (item.category, item.order, str(item.key)),
            ))
            self._frozen = True
        return self

    @property
    def frozen(self) -> bool:
        return self._frozen

    def __iter__(self) -> Iterator[Descriptor]:
        if not self._frozen:
            raise RegistryFrozenError("freeze the registry before iteration")
        return iter(self._ordered)

    def get(self, key: PageKey | str) -> Descriptor | None:
        return self._descriptors.get(PageKey(str(key)))

    def select(
        self, persisted_key: PageKey | str | None, default_key: PageKey
    ) -> PageDescriptor:
        if not self._frozen:
            raise RegistryFrozenError("freeze the registry before selection")
        selected = self.get(persisted_key) if persisted_key is not None else None
        if isinstance(selected, PageDescriptor):
            return selected
        default = self.get(default_key)
        if not isinstance(default, PageDescriptor):
            raise KeyError(f"default page is not registered: {default_key}")
        return default
