# -*- coding: utf-8 -*-
"""Closed, recursively copied value grammars for session authorities."""
from __future__ import annotations

from types import MappingProxyType


_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_IDENTITY_ATOMS = {int, str, bytes}
_METADATA_ATOMS = {bool, int, float, str, bytes}


def freeze_identity(
    value,
    name: str,
    *,
    extension=None,
    description: str = "exact int/string/bytes/tuple values",
    active: set[int] | None = None,
):
    """Copy one stable dictionary identity from a closed value grammar."""
    if type(value) in _IDENTITY_ATOMS:
        return value
    if active is None:
        active = set()
    identity = id(value)
    if identity in active:
        raise ValueError(f"{name} cannot contain a reference cycle")
    active.add(identity)
    try:
        if type(value) is tuple:
            return tuple(
                freeze_identity(
                    item,
                    name,
                    extension=extension,
                    description=description,
                    active=active,
                )
                for item in value
            )
        if extension is not None:
            recurse = lambda item: freeze_identity(
                item,
                name,
                extension=extension,
                description=description,
                active=active,
            )
            frozen = extension(value, recurse)
            if frozen is not NotImplemented:
                return frozen
    finally:
        active.remove(identity)
    raise TypeError(f"{name} must contain only {description}")


def require_exact_string(value, name: str, *, nonempty: bool = True) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if nonempty and not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _freeze_metadata_value(value, name: str, active: set[int]):
    if getattr(value, "nbytes", None) is not None and getattr(
        value, "shape", None,
    ) is not None:
        raise ValueError(f"{name} cannot own an unaccounted array buffer")
    if value is None or type(value) in _METADATA_ATOMS:
        return value
    identity = id(value)
    if identity in active:
        raise ValueError(f"{name} cannot contain a reference cycle")
    active.add(identity)
    try:
        if type(value) in {dict, _MAPPING_PROXY_TYPE}:
            frozen = {}
            for key, item in value.items():
                require_exact_string(key, f"{name} key", nonempty=False)
                frozen[key] = _freeze_metadata_value(item, name, active)
            return MappingProxyType(frozen)
        if type(value) in {tuple, list}:
            return tuple(
                _freeze_metadata_value(item, name, active) for item in value
            )
        if type(value) in {set, frozenset}:
            try:
                return frozenset(
                    _freeze_metadata_value(item, name, active) for item in value
                )
            except TypeError as exc:
                raise TypeError(
                    f"{name} set members must freeze to hashable values"
                ) from exc
    finally:
        active.remove(identity)
    raise TypeError(
        f"{name} values must be exact built-in scalars or containers"
    )


def freeze_metadata_mapping(value, name: str):
    if type(value) not in {dict, _MAPPING_PROXY_TYPE}:
        raise TypeError(f"{name} must be an exact built-in mapping")
    frozen = _freeze_metadata_value(value, name, set())
    if type(frozen) is not _MAPPING_PROXY_TYPE:  # pragma: no cover
        raise TypeError(f"{name} must freeze to an immutable mapping")
    return frozen
