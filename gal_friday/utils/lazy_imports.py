"""Lazy import utilities to handle optional dependencies without PLC0415 errors."""

from collections.abc import Callable
import contextlib
import importlib
import importlib.util
import logging
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

if TYPE_CHECKING:
    import types

T = TypeVar("T")


class LazyImport:
    """Lazy import that only imports when accessed.

    This allows optional dependencies to be declared at module level
    without forcing the import at startup time.

    Example:
        # At module level (no PLC0415 error)
        elasticsearch = LazyImport('elasticsearch', 'Elasticsearch')

        # In function/method
        if elasticsearch.available:
            client = elasticsearch()  # Import happens here
    """

    def __init__(self, module_name: str, attribute: str | None = None) -> None:
        """Initialize lazy import.

        Args:
            module_name: Full module path (e.g., 'elasticsearch')
            attribute: Optional attribute to import from module (e.g., 'Elasticsearch')
        """
        self.module_name = module_name
        self.attribute = attribute
        self._module: Any | None = None
        self._attr: Any | None = None
        self._error: ImportError | None = None
        self._loaded = False

    def _load(self) -> None:
        """Load the module and attribute if not already loaded."""
        if self._loaded:
            return

        try:
            self._module = importlib.import_module(self.module_name)
            if self.attribute:
                self._attr = getattr(self._module, self.attribute)
            self._loaded = True
        except ImportError as e:
            self._error = e
            self._loaded = True  # Don't retry
        except AttributeError:
            self._error = ImportError(
                f"Module '{self.module_name}' has no attribute '{self.attribute}'",
            )
            self._loaded = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the imported attribute (for classes/functions)."""
        self._load()
        if self._error:
            raise self._error

        if self.attribute and self._attr is not None:
            return self._attr(*args, **kwargs)
        raise TypeError(f"Module '{self.module_name}' is not callable")

    def __getattr__(self, name: str) -> Any:
        """Access attributes of the imported module/attribute."""
        self._load()
        if self._error:
            raise self._error

        if self.attribute and self._attr:
            return getattr(self._attr, name)
        if self._module:
            return getattr(self._module, name)
        raise AttributeError(f"No attribute '{name}'")

    @property
    def available(self) -> bool:
        """Check if the module is available without importing."""
        if self._loaded:
            return self._error is None

        # Check if importable without actually importing
        try:
            spec = importlib.util.find_spec(self.module_name)
        except (ImportError, ModuleNotFoundError, ValueError):
            return False
        else:
            return spec is not None

    def get(self, default: T | None = None) -> Any | T:
        """Get the imported object or a default if not available."""
        if not self.available:
            return default

        self._load()
        if self._error:
            return default

        return self._attr if self.attribute else self._module

    def get_module(self) -> Any:
        """Get the imported module (not the attribute)."""
        self._load()
        if self._error:
            raise self._error
        return self._module

    def require(self) -> Any:
        """Get the imported object, raising error if not available."""
        self._load()
        if self._error:
            raise self._error

        return self._attr if self.attribute else self._module


class OptionalDependency:
    """Decorator for functions that require optional dependencies.

    Example:
        talib = LazyImport('talib')

        @OptionalDependency(talib, "TA-Lib is required for technical analysis")
        def calculate_rsi(prices):
            return talib.RSI(prices)
    """

    def __init__(
        self,
        lazy_import: LazyImport,
        error_message: str | None = None,
        fallback: Callable[..., Any] | None = None,
    ) -> None:
        """Initialize optional dependency decorator.

        Args:
            lazy_import: The LazyImport instance to check
            error_message: Custom error message if dependency is missing
            fallback: Optional fallback function if dependency is missing
        """
        self.lazy_import = lazy_import
        self.error_message = error_message
        self.fallback = fallback

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Wrap the function to check for dependency."""

        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not self.lazy_import.available:
                if self.fallback:
                    return self.fallback(*args, **kwargs)  # type: ignore[no-any-return]

                error_msg = self.error_message or (
                    f"Optional dependency '{self.lazy_import.module_name}' is not installed. "
                    f"Install it to use {func.__name__}."
                )
                raise ImportError(error_msg)

            return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return cast("Callable[..., T]", wrapper)


class OptionalFeature:
    """Enhanced decorator for optional features with better error handling and logging.

    Example:
        @OptionalFeature(
            'elasticsearch',
            feature_name='Elasticsearch logging',
            install_instructions='pip install elasticsearch>=8.0'
        )
        def log_to_elasticsearch(data):
            # Function implementation
    """

    def __init__(
        self,
        dependency: str | LazyImport,
        feature_name: str | None = None,
        install_instructions: str | None = None,
        fallback: Callable[..., Any] | None = None,
        log_warning: bool = True,
    ) -> None:
        """Initialize optional feature decorator.

        Args:
            dependency: Module name or LazyImport instance
            feature_name: Human-readable feature name for error messages
            install_instructions: Specific installation instructions
            fallback: Fallback function if dependency is missing
            log_warning: Whether to log a warning when feature is unavailable
        """
        if isinstance(dependency, str):
            self.lazy_import = LazyImport(dependency)
        else:
            self.lazy_import = dependency

        self.feature_name = feature_name or f"{self.lazy_import.module_name} feature"
        self.install_instructions = install_instructions
        self.fallback = fallback
        self.log_warning = log_warning
        self._warning_logged = False

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Wrap the function with optional feature handling."""

        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not self.lazy_import.available:
                if self.log_warning and not self._warning_logged:
                    logger = logging.getLogger(func.__module__)

                    msg = f"{self.feature_name} is not available."
                    if self.install_instructions:
                        msg += f" To enable, {self.install_instructions}"
                    logger.warning(msg)
                    self._warning_logged = True

                if self.fallback:
                    return self.fallback(*args, **kwargs)  # type: ignore[no-any-return]

                error_msg = f"{self.feature_name} requires '{self.lazy_import.module_name}'."
                if self.install_instructions:
                    error_msg += f" {self.install_instructions}"
                raise ImportError(error_msg)

            return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        # Set __wrapped__ attribute for introspection (cast to Any to avoid type issues)
        cast("Any", wrapper).__wrapped__ = func
        return cast("Callable[..., T]", wrapper)


class ConditionalImport:
    """Context manager for conditional imports with graceful fallback.

    Example:
        with ConditionalImport('scipy.stats') as stats:
            if stats:
                return stats.norm.pdf(x)
            else:
                # Fallback implementation
                return manual_normal_pdf(x)
    """

    def __init__(self, module_name: str, attribute: str | None = None) -> None:
        """Initialize conditional import context manager.

        Args:
            module_name: Module to import
            attribute: Optional attribute to get from module
        """
        self.module_name = module_name
        self.attribute = attribute
        self.module: types.ModuleType | None = None
        self.error: ImportError | AttributeError | None = None

    def __enter__(self) -> Any | None:
        """Try to import the module."""
        try:
            self.module = importlib.import_module(self.module_name)
            if self.attribute:
                return getattr(self.module, self.attribute)
        except (ImportError, AttributeError) as e:
            self.error = e
            return None
        else:
            return self.module

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> Literal[False]:
        """Clean up."""
        self.module = None
        return False

    @property
    def available(self) -> bool:
        """Check if the import was successful."""
        return self.error is None


# Common optional dependencies for Gal Friday
elasticsearch = LazyImport("elasticsearch", "Elasticsearch")
async_elasticsearch = LazyImport("elasticsearch", "AsyncElasticsearch")
influxdb = LazyImport("influxdb_client", "InfluxDBClient")
influxdb_point = LazyImport("influxdb_client", "Point")
influxdb_error = LazyImport("influxdb_client.client.exceptions", "InfluxDBError")
influxdb_synchronous = LazyImport("influxdb_client.client.write_api", "SYNCHRONOUS")
influxdb_v1 = LazyImport("influxdb", "InfluxDBClient")  # For v1.x client
talib = LazyImport("talib")
daemon = LazyImport("daemon")
daemon_pidfile = LazyImport("daemon.pidfile", "TimeoutPIDLockFile")
arch = LazyImport("arch", "arch_model")
scipy_interpolate = LazyImport("scipy.interpolate", "CubicSpline")
redis = LazyImport("redis", "Redis")
aioredis = LazyImport("redis.asyncio")
psutil = LazyImport("psutil")
matplotlib = LazyImport("matplotlib")
pyplot = LazyImport("matplotlib.pyplot")
google_secretmanager = LazyImport("google.cloud", "secretmanager")


def check_optional_dependencies() -> dict[str, bool]:
    """Check which optional dependencies are available.

    Returns:
        Dictionary mapping dependency names to availability status
    """
    return {
        "elasticsearch": elasticsearch.available,
        "influxdb": influxdb.available,
        "talib": talib.available,
        "daemon": daemon.available,
        "arch": arch.available,
        "scipy": scipy_interpolate.available,
        "redis": redis.available,
        "aioredis": aioredis.available,
        "psutil": psutil.available,
        "matplotlib": matplotlib.available,
    }


def require_optional(dep_name: str) -> Any:
    """Require an optional dependency by name.

    Args:
        dep_name: Name of the dependency

    Returns:
        The imported module/class

    Raises:
        ImportError: If dependency is not available
        KeyError: If dependency name is unknown
    """
    deps_map = {
        "elasticsearch": elasticsearch,
        "influxdb": influxdb,
        "talib": talib,
        "daemon": daemon,
        "arch": arch,
        "scipy_interpolate": scipy_interpolate,
        "redis": redis,
        "aioredis": aioredis,
        "psutil": psutil,
        "matplotlib": matplotlib,
        "pyplot": pyplot,
    }

    if dep_name not in deps_map:
        raise KeyError(f"Unknown optional dependency: {dep_name}")

    return deps_map[dep_name].require()


# Example usage patterns:
if __name__ == "__main__":
    # Check available dependencies
    for _dep, _available in check_optional_dependencies().items():
        pass

    # Example 1: Using with conditional logic
    if elasticsearch.available:
        pass
        # Would create an ES client here - import happens lazily when called
        # Usage would be: client = elasticsearch(hosts=['localhost:9200'])
    else:
        pass

    # Example 2: Using get() with default
    es_class = elasticsearch.get(default=None)
    if es_class:
        pass

    # Example 3: Using require() (will raise if not available)
    with contextlib.suppress(ImportError):
        talib_module = talib.require()
