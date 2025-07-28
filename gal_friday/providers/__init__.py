"""Provider implementations for historical data."""

from typing import Any

# Lazy loading for providers to avoid dependency hell
_provider_modules = {
    "APIDataProvider": ("api_provider", "APIDataProvider"),
    "DatabaseDataProvider": ("database_provider", "DatabaseDataProvider"),
    "LocalFileDataProvider": ("local_file_provider", "LocalFileDataProvider"),
}

__all__ = [
    "APIDataProvider",
    "DatabaseDataProvider",
    "LocalFileDataProvider",
]


def __getattr__(name: str) -> Any:
    """Lazy import providers to avoid circular dependency issues."""
    if name in _provider_modules:
        module_name, class_name = _provider_modules[name]

        # Import the module lazily
        from importlib import (  # noqa: PLC0415 - architectural: dynamic provider loading to avoid circular dependencies
            import_module,
        )

        module = import_module(f".{module_name}", package=__name__)

        # Get the class from the module
        provider_class = getattr(module, class_name)

        # Cache it in globals for subsequent access
        globals()[name] = provider_class

        return provider_class

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
