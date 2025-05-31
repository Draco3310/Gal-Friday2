"""Enterprise-grade event store with persistence and caching."""

from datetime import datetime, timedelta
from typing import Any, TypeVar, Generic
from collections import deque
import asyncio
import json
from uuid import UUID

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gal_friday.core.events import Event, EventType
from gal_friday.dal.models.event_log import EventLog
from gal_friday.logger_service import LoggerService

T = TypeVar('T', bound=Event)

class EventStore:
    """Enterprise-grade event store with persistence and caching.
    
    Features:
    - Async PostgreSQL persistence
    - In-memory LRU cache for recent events
    - Event replay capabilities
    - Type-safe event retrieval
    - Automatic event serialization/deserialization
    """
    
    def __init__(
        self,
        session_maker: async_sessionmaker[AsyncSession],
        logger: LoggerService,
        cache_size: int = 10000,
        cache_ttl_seconds: int = 3600
    ) -> None:
        """Initialize event store.
        
        Args:
            session_maker: SQLAlchemy async session maker
            logger: Logger service
            cache_size: Maximum number of events to cache
            cache_ttl_seconds: Cache entry TTL in seconds
        """
        self.session_maker = session_maker
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # In-memory cache with TTL
        self._cache: dict[UUID, tuple[Event, datetime]] = {}
        self._cache_order: deque[UUID] = deque(maxlen=cache_size) # Added type hint
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)
        
        # Event type registry for deserialization
        self._event_registry: dict[str, type[Event]] = {}
        self._register_event_types()
        
        # Background cache cleanup task
        self._cleanup_task: asyncio.Task | None = None
    
    async def start(self) -> None:
        """Start the event store and background tasks."""
        self._cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
        self.logger.info(
            "Event store started",
            source_module=self._source_module
        )
    
    async def stop(self) -> None:
        """Stop the event store and cleanup resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self._cache.clear()
        self._cache_order.clear()
        
        self.logger.info(
            "Event store stopped",
            source_module=self._source_module
        )
    
    async def store_event(self, event: Event) -> None:
        """Store an event in both cache and database.
        
        Args:
            event: Event to store
        """
        try:
            # Add to cache
            self._add_to_cache(event)
            
            # Persist to database
            async with self.session_maker() as session:
                event_type_attr = getattr(event, "event_type", None)
                if not isinstance(event_type_attr, EventType):
                    self.logger.error(
                        f"Event {event.event_id} (source: {event.source_module}, type: {type(event).__name__}) "
                        f"is missing a valid 'event_type' attribute of type EventType."
                    )
                    # Optionally, raise an error or handle as appropriate
                    # For now, let's attempt to get a default or raise
                    raise ValueError(f"Event {event.event_id} is missing a valid event_type.")

                event_log = EventLog(
                    event_id=event.event_id,
                    event_type=event_type_attr.value, # Use the validated attribute
                    source_module=event.source_module,
                    timestamp=event.timestamp,
                    data=self._serialize_event(event)
                )
                session.add(event_log)
                await session.commit()
            
            self.logger.debug(
                f"Stored event {event.event_id}",
                source_module=self._source_module,
                context={
                    "event_type": event.event_type.value,
                    "source": event.source_module
                }
            )
            
        except Exception:
            self.logger.exception(
                "Failed to store event",
                source_module=self._source_module,
                context={"event_id": str(event.event_id)}
            )
            raise
    
    async def get_event(
        self, 
        event_id: UUID, 
        event_type: type[T] | None = None
    ) -> T | None:
        """Retrieve a single event by ID.
        
        Args:
            event_id: Event ID to retrieve
            event_type: Expected event type for type safety
            
        Returns:
            Event if found, None otherwise
        """
        # Check cache first
        cached = self._get_from_cache(event_id)
        if cached:
            if event_type and not isinstance(cached, event_type):
                self.logger.warning(
                    f"Event type mismatch: expected {event_type.__name__}, "
                    f"got {type(cached).__name__}",
                    source_module=self._source_module
                )
                return None
            return cached  # type: ignore
        
        # Load from database
        try:
            async with self.session_maker() as session:
                result = await session.execute(
                    select(EventLog).where(EventLog.event_id == event_id)
                )
                event_log = result.scalar_one_or_none()
                
                if not event_log:
                    return None
                
                event = self._deserialize_event(event_log)
                
                # Validate type if specified
                if event_type and not isinstance(event, event_type):
                    self.logger.warning(
                        f"Event type mismatch: expected {event_type.__name__}, "
                        f"got {type(event).__name__}",
                        source_module=self._source_module
                    )
                    return None
                
                # Add to cache
                self._add_to_cache(event)
                
                return event  # type: ignore
                
        except Exception:
            self.logger.exception(
                "Failed to retrieve event",
                source_module=self._source_module,
                context={"event_id": str(event_id)}
            )
            return None
    
    async def get_events_by_correlation(
        self,
        correlation_id: UUID,
        event_types: list[type[Event]] | None = None,
        since: datetime | None = None,
        until: datetime | None = None
    ) -> list[Event]:
        """Retrieve all events with a specific correlation ID.
        
        Args:
            correlation_id: Correlation ID (e.g., signal_id)
            event_types: Filter by event types
            since: Start timestamp filter
            until: End timestamp filter
            
        Returns:
            List of matching events ordered by timestamp
        """
        try:
            async with self.session_maker() as session:
                query = select(EventLog).where(
                    EventLog.data["correlation_id"].astext == str(correlation_id)
                )
                
                if event_types:
                    type_names = [et.__name__ for et in event_types]
                    query = query.where(EventLog.event_type.in_(type_names))
                
                if since:
                    query = query.where(EventLog.timestamp >= since)
                
                if until:
                    query = query.where(EventLog.timestamp <= until)
                
                query = query.order_by(EventLog.timestamp)
                
                result = await session.execute(query)
                event_logs = result.scalars().all()
                
                events = []
                for log in event_logs:
                    event = self._deserialize_event(log)
                    if event:
                        events.append(event)
                        self._add_to_cache(event)
                
                return events
                
        except Exception:
            self.logger.exception(
                "Failed to retrieve events by correlation",
                source_module=self._source_module,
                context={"correlation_id": str(correlation_id)}
            )
            return []
    
    async def get_events_by_type(
        self,
        event_type: type[T],
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100
    ) -> list[T]:
        """Retrieve events of a specific type.
        
        Args:
            event_type: Event type to retrieve
            since: Start timestamp filter
            until: End timestamp filter
            limit: Maximum number of events to return
            
        Returns:
            List of matching events ordered by timestamp desc
        """
        try:
            async with self.session_maker() as session:
                query = select(EventLog).where(
                    EventLog.event_type == event_type.__name__
                )
                
                if since:
                    query = query.where(EventLog.timestamp >= since)
                
                if until:
                    query = query.where(EventLog.timestamp <= until)
                
                query = query.order_by(EventLog.timestamp.desc()).limit(limit)
                
                result = await session.execute(query)
                event_logs = result.scalars().all()
                
                events: list[T] = []
                for log in event_logs:
                    event = self._deserialize_event(log)
                    if event and isinstance(event, event_type):
                        events.append(event)
                        self._add_to_cache(event)
                
                return events
                
        except Exception:
            self.logger.exception(
                "Failed to retrieve events by type",
                source_module=self._source_module,
                context={"event_type": event_type.__name__}
            )
            return []
    
    def _add_to_cache(self, event: Event) -> None:
        """Add event to cache with TTL."""
        from datetime import UTC
        
        event_id = event.event_id
        
        # Remove oldest if at capacity
        if len(self._cache) >= self._cache_order.maxlen:
            oldest_id = self._cache_order[0]
            self._cache.pop(oldest_id, None)
        
        # Add to cache
        self._cache[event_id] = (event, datetime.now(UTC))
        self._cache_order.append(event_id)
    
    def _get_from_cache(self, event_id: UUID) -> Event | None:
        """Get event from cache if not expired."""
        from datetime import UTC
        
        cached = self._cache.get(event_id)
        if not cached:
            return None
        
        event, cached_at = cached
        if datetime.now(UTC) - cached_at > self._cache_ttl:
            # Expired
            self._cache.pop(event_id, None)
            return None
        
        return event
    
    async def _cache_cleanup_loop(self) -> None:
        """Background task to clean expired cache entries."""
        from datetime import UTC
        
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                now = datetime.now(UTC)
                expired_ids = []
                
                for event_id, (_, cached_at) in self._cache.items():
                    if now - cached_at > self._cache_ttl:
                        expired_ids.append(event_id)
                
                for event_id in expired_ids:
                    self._cache.pop(event_id, None)
                
                if expired_ids:
                    self.logger.debug(
                        f"Cleaned {len(expired_ids)} expired cache entries",
                        source_module=self._source_module
                    )
                    
            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception(
                    "Error in cache cleanup",
                    source_module=self._source_module
                )
    
    def _serialize_event(self, event: Event) -> dict[str, Any]:
        """Serialize event to JSON-compatible dict."""
        data = event.to_dict()
        
        # Add correlation IDs for common event types
        if hasattr(event, 'signal_id'):
            data['correlation_id'] = str(event.signal_id)
        elif hasattr(event, 'order_id'):
            data['correlation_id'] = str(event.order_id)
        
        return data
    
    def _deserialize_event(self, event_log: EventLog) -> Event | None:
        """Deserialize event from database."""
        try:
            event_class = self._event_registry.get(event_log.event_type)
            if not event_class:
                self.logger.warning(
                    f"Unknown event type: {event_log.event_type}",
                    source_module=self._source_module
                )
                return None
            
            return event_class.from_dict(event_log.data)
            
        except Exception:
            self.logger.exception(
                "Failed to deserialize event",
                source_module=self._source_module,
                context={"event_id": str(event_log.event_id)}
            )
            return None
    
    def _register_event_types(self) -> None:
        """Register all event types for deserialization."""
        from gal_friday.core.events import (
            MarketDataOHLCVEvent,
            MarketDataL2Event,
            MarketDataTickerEvent,
            MarketDataTradeEvent,
            TradeSignalProposedEvent,
            TradeSignalApprovedEvent,
            TradeSignalRejectedEvent,
            ExecutionReportEvent,
            SystemStateEvent,
            PotentialHaltTriggerEvent,
            ClosePositionCommand
        )
        
        event_types = [
            MarketDataOHLCVEvent,
            MarketDataL2Event,
            MarketDataTickerEvent,
            MarketDataTradeEvent,
            TradeSignalProposedEvent,
            TradeSignalApprovedEvent,
            TradeSignalRejectedEvent,
            ExecutionReportEvent,
            SystemStateEvent,
            PotentialHaltTriggerEvent,
            ClosePositionCommand
        ]
        
        for event_type in event_types:
            self._event_registry[event_type.__name__] = event_type 