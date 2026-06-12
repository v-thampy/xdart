"""The ssrl_xrd_tools shim: old imports work, with a DeprecationWarning."""
import importlib
import sys
import warnings


def test_shim_aliases_package_and_submodules():
    for mod in list(sys.modules):
        if mod.startswith("ssrl_xrd_tools"):
            del sys.modules[mod]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import ssrl_xrd_tools  # noqa: F401
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    import xrd_tools
    assert sys.modules["ssrl_xrd_tools"] is xrd_tools

    # Submodule paths resolve through the alias.
    from ssrl_xrd_tools.io.read import relative_source_path  # noqa: F401
    mod = importlib.import_module("ssrl_xrd_tools.reduction")
    assert mod is importlib.import_module("xrd_tools.reduction")
