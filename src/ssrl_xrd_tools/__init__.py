"""Deprecated compatibility shim: ``ssrl_xrd_tools`` is now ``xrd_tools``.

The package was renamed in the xrd-tools monorepo (2026-06).  This shim
keeps old user code/notebooks importable with TRUE module identity:

    import ssrl_xrd_tools                  -> xrd_tools (same object)
    from ssrl_xrd_tools.io.read import X  -> xrd_tools.io.read.X

A meta-path finder aliases every ``ssrl_xrd_tools.*`` import to the real
``xrd_tools.*`` module object (``create_module`` returns the existing
module; ``_init_module_attrs`` with ``override=False`` leaves its
``__name__``/``__spec__`` untouched), so isinstance checks and
module-level singletons keep working across both spellings.

First-party code must NEVER import this module (guarded by a test).
"""
import importlib
import importlib.abc
import importlib.machinery
import sys
import warnings

_OLD = "ssrl_xrd_tools"
_NEW = "xrd_tools"

warnings.warn(
    "ssrl_xrd_tools has been renamed to xrd_tools (xrd-tools monorepo); "
    "update imports — this shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

_target = importlib.import_module(_NEW)


class _AliasLoader(importlib.abc.Loader):
    def create_module(self, spec):
        # Return the REAL module; the import system registers it under the
        # alias key and (override=False) leaves its attributes alone.
        return importlib.import_module(
            _NEW + spec.name[len(_OLD):]
        )

    def exec_module(self, module):  # already executed as xrd_tools.*
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == _OLD or fullname.startswith(_OLD + "."):
            return importlib.machinery.ModuleSpec(fullname, _AliasLoader())
        return None


sys.meta_path.insert(0, _AliasFinder())
sys.modules[__name__] = _target
