# Task: Provide proper generic typing for caching decorator to retain type information

### 1. Context
- **File:** `gal_friday/utils/performance_optimizer.py`
- **Line:** `728`
- **Keyword/Pattern:** `TODO`
- **Current State:** Caching decorator lacks proper generic typing, causing loss of type information

### 2. Problem Statement
The caching decorator in the performance optimizer does not preserve function type information, leading to poor type checking support and reduced IDE functionality. This makes it difficult to maintain type safety in cached functions and reduces developer productivity.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Implement Generic Type Support:** Add proper generic typing with TypeVar and Protocol support
2. **Create Type-Safe Decorator:** Preserve function signatures and return types
3. **Add Cache Type Annotations:** Type-safe cache key and value handling
4. **Implement Overload Support:** Handle multiple function overloads with proper typing
5. **Create Cache Statistics Typing:** Type-safe cache performance metrics
6. **Build Cache Configuration Typing:** Comprehensive configuration type definitions

#### b. Pseudocode or Implementation Sketch
```python
from typing import (
    TypeVar, Generic, Protocol, Callable, Any, Dict, Optional, Union, 
    Tuple, ParamSpec, Concatenate, overload, cast, Type, runtime_checkable
)
from typing_extensions import ParamSpec, Concatenate
from functools import wraps, update_wrapper
from dataclasses import dataclass, field
import time
import threading
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
import hashlib
import pickle

# Type variables for generic support
P = ParamSpec('P')
T = TypeVar('T')
K = TypeVar('K')  # Cache key type
V = TypeVar('V')  # Cache value type

@runtime_checkable
class Hashable(Protocol):
    """Protocol for hashable cache keys"""
    def __hash__(self) -> int: ...

@runtime_checkable
class CacheableFunction(Protocol[P, T]):
    """Protocol for cacheable functions"""
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T: ...

@dataclass
class CacheEntry(Generic[V]):
    """Type-safe cache entry"""
    value: V
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def touch(self) -> None:
        """Update access information"""
        self.access_count += 1
        self.last_access = time.time()

@dataclass
class CacheConfig:
    """Type-safe cache configuration"""
    max_size: Optional[int] = 128
    ttl: Optional[float] = None
    typed_keys: bool = True
    thread_safe: bool = True
    eviction_policy: str = "lru"
    key_serializer: Optional[Callable[[Any], str]] = None
    value_serializer: Optional[Callable[[Any], bytes]] = None
    value_deserializer: Optional[Callable[[bytes], Any]] = None

class CacheKeyGenerator(Generic[K]):
    """Type-safe cache key generator"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        
    def generate_key(self, func: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> K:
        """Generate type-safe cache key"""
        
        if self.config.key_serializer:
            return cast(K, self.config.key_serializer((func.__name__, args, kwargs)))
        
        # Default key generation with type information
        key_parts = [func.__module__, func.__qualname__]
        
        # Add arguments to key
        for arg in args:
            if self.config.typed_keys:
                key_parts.append(f"{type(arg).__name__}:{repr(arg)}")
            else:
                key_parts.append(repr(arg))
        
        # Add keyword arguments to key
        for k, v in sorted(kwargs.items()):
            if self.config.typed_keys:
                key_parts.append(f"{k}={type(v).__name__}:{repr(v)}")
            else:
                key_parts.append(f"{k}={repr(v)}")
        
        # Create hash of key parts
        key_string = "|".join(key_parts)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        
        return cast(K, key_hash)

class TypeSafeCache(Generic[K, V]):
    """Type-safe cache implementation"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: Dict[K, CacheEntry[V]] = OrderedDict()
        self._lock = threading.RLock() if config.thread_safe else None
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: K) -> Optional[V]:
        """Get value from cache with type safety"""
        
        with self._lock if self._lock else nullcontext():
            if key not in self._cache:
                self.misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self.misses += 1
                return None
            
            # Update access information
            entry.touch()
            
            # Move to end for LRU
            if self.config.eviction_policy == "lru":
                self._cache.move_to_end(key)
            
            self.hits += 1
            return entry.value
    
    def put(self, key: K, value: V) -> None:
        """Put value in cache with type safety"""
        
        with self._lock if self._lock else nullcontext():
            # Check if eviction is needed
            if (self.config.max_size is not None and 
                len(self._cache) >= self.config.max_size and 
                key not in self._cache):
                self._evict_entries()
            
            # Create cache entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=self.config.ttl
            )
            
            self._cache[key] = entry
    
    def _evict_entries(self) -> None:
        """Evict entries based on policy"""
        
        if not self._cache:
            return
        
        if self.config.eviction_policy == "lru":
            # Remove least recently used
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        elif self.config.eviction_policy == "lfu":
            # Remove least frequently used
            lfu_key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
            del self._cache[lfu_key]
        
        self.evictions += 1
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock if self._lock else nullcontext():
            self._cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_size": len(self._cache),
            "max_size": self.config.max_size
        }

class TypedCacheDecorator(Generic[P, T]):
    """Type-safe caching decorator that preserves function signatures"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.cache: TypeSafeCache[str, T] = TypeSafeCache(self.config)
        self.key_generator: CacheKeyGenerator[str] = CacheKeyGenerator(self.config)
    
    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        """
        Create type-safe caching decorator
        Replaces TODO with proper generic typing that retains type information
        """
        
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Generate cache key
            cache_key = self.key_generator.generate_key(func, args, kwargs)
            
            # Try to get from cache
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            self.cache.put(cache_key, result)
            
            return result
        
        # Preserve function metadata
        update_wrapper(wrapper, func)
        
        # Add cache management methods
        wrapper.cache_clear = self.cache.clear  # type: ignore
        wrapper.cache_stats = self.cache.stats  # type: ignore
        wrapper.cache_info = self.cache.stats   # type: ignore
        
        return wrapper

# Function overloads for different configuration scenarios
@overload
def cache() -> Callable[[Callable[P, T]], Callable[P, T]]: ...

@overload
def cache(*, max_size: Optional[int] = ...) -> Callable[[Callable[P, T]], Callable[P, T]]: ...

@overload
def cache(*, ttl: Optional[float] = ...) -> Callable[[Callable[P, T]], Callable[P, T]]: ...

@overload
def cache(*, max_size: Optional[int] = ..., ttl: Optional[float] = ...) -> Callable[[Callable[P, T]], Callable[P, T]]: ...

@overload
def cache(config: CacheConfig) -> Callable[[Callable[P, T]], Callable[P, T]]: ...

def cache(
    config: Optional[CacheConfig] = None,
    *,
    max_size: Optional[int] = None,
    ttl: Optional[float] = None,
    typed_keys: bool = True,
    thread_safe: bool = True,
    eviction_policy: str = "lru"
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Type-safe caching decorator with proper generic typing
    
    This replaces the TODO at line 728 with comprehensive generic type support
    that preserves function signatures and return types.
    """
    
    # Create configuration if not provided
    if config is None:
        config = CacheConfig(
            max_size=max_size,
            ttl=ttl,
            typed_keys=typed_keys,
            thread_safe=thread_safe,
            eviction_policy=eviction_policy
        )
    
    return TypedCacheDecorator[P, T](config)

# Advanced typing for method caching
class MethodCacheDescriptor(Generic[T]):
    """Type-safe method caching descriptor"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.caches: weakref.WeakKeyDictionary[Any, TypeSafeCache[str, T]] = weakref.WeakKeyDictionary()
        self.key_generator = CacheKeyGenerator[str](self.config)
    
    def __set_name__(self, owner: Type[Any], name: str) -> None:
        self.name = name
    
    def __get__(self, instance: Any, owner: Optional[Type[Any]] = None) -> Callable[..., T]:
        if instance is None:
            return self  # type: ignore
        
        # Get or create cache for this instance
        if instance not in self.caches:
            self.caches[instance] = TypeSafeCache[str, T](self.config)
        
        instance_cache = self.caches[instance]
        original_method = getattr(owner, f"_original_{self.name}")
        
        @wraps(original_method)
        def cached_method(*args: Any, **kwargs: Any) -> T:
            cache_key = self.key_generator.generate_key(original_method, args, kwargs)
            
            cached_result = instance_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            result = original_method(instance, *args, **kwargs)
            instance_cache.put(cache_key, result)
            
            return result
        
        return cached_method

def cached_method(config: Optional[CacheConfig] = None) -> Callable[[Callable[..., T]], MethodCacheDescriptor[T]]:
    """Decorator for creating cached methods with proper typing"""
    
    def decorator(method: Callable[..., T]) -> MethodCacheDescriptor[T]:
        # Store original method
        method_name = method.__name__
        
        def class_decorator(cls: Type[Any]) -> Type[Any]:
            setattr(cls, f"_original_{method_name}", method)
            setattr(cls, method_name, MethodCacheDescriptor[T](config))
            return cls
        
        return class_decorator(method)  # type: ignore
    
    return decorator

# Async function support
from typing import Awaitable, Coroutine

AsyncP = ParamSpec('AsyncP')
AsyncT = TypeVar('AsyncT')

class AsyncTypedCacheDecorator(Generic[AsyncP, AsyncT]):
    """Type-safe async caching decorator"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.cache: TypeSafeCache[str, AsyncT] = TypeSafeCache(self.config)
        self.key_generator: CacheKeyGenerator[str] = CacheKeyGenerator(self.config)
    
    def __call__(self, func: Callable[AsyncP, Awaitable[AsyncT]]) -> Callable[AsyncP, Awaitable[AsyncT]]:
        @wraps(func)
        async def async_wrapper(*args: AsyncP.args, **kwargs: AsyncP.kwargs) -> AsyncT:
            cache_key = self.key_generator.generate_key(func, args, kwargs)
            
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            result = await func(*args, **kwargs)
            self.cache.put(cache_key, result)
            
            return result
        
        return async_wrapper

def async_cache(config: Optional[CacheConfig] = None) -> Callable[[Callable[AsyncP, Awaitable[AsyncT]]], Callable[AsyncP, Awaitable[AsyncT]]]:
    """Type-safe async caching decorator"""
    return AsyncTypedCacheDecorator[AsyncP, AsyncT](config)

# Context manager for null context
from contextlib import nullcontext

# Example usage with preserved types
if __name__ == "__main__":
    # Function with preserved signature
    @cache(max_size=100, ttl=300)
    def expensive_calculation(x: int, y: float) -> str:
        return f"Result: {x + y}"
    
    # Type checker knows the return type is str
    result: str = expensive_calculation(1, 2.5)
    
    # Method caching example
    class DataProcessor:
        @cached_method(CacheConfig(max_size=50))
        def process_data(self, data: List[int]) -> Dict[str, int]:
            return {"sum": sum(data), "count": len(data)}
    
    # Async function example
    @async_cache(CacheConfig(max_size=200))
    async def async_operation(param: str) -> int:
        # Simulate async work
        await asyncio.sleep(0.1)
        return len(param)
```

#### c. Key Considerations & Dependencies
- **Type Safety:** Complete generic type support with TypeVar and ParamSpec; preserved function signatures
- **Performance:** Efficient key generation and cache operations; thread-safe implementations
- **Compatibility:** Support for both sync and async functions; method caching support
- **Configuration:** Flexible cache configuration with type-safe parameters

### 4. Acceptance Criteria
- [ ] Proper generic typing preserves function signatures and return types
- [ ] TypeVar and ParamSpec support for comprehensive type safety
- [ ] Overloaded decorator signatures for different configuration options
- [ ] Support for both sync and async function caching
- [ ] Method caching with instance-specific cache management
- [ ] Type-safe cache key generation and storage
- [ ] Comprehensive cache statistics with proper typing
- [ ] Thread-safe cache operations with proper type annotations
- [ ] Cache configuration with full type safety
- [ ] TODO placeholder is completely replaced with production-ready generic typing 