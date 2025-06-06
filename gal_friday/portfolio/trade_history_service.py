"""Enterprise-grade trade history service with comprehensive data retrieval and caching."""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gal_friday.dal.repositories.fill_repository import FillRepository
from gal_friday.exceptions import DataValidationError
from gal_friday.models.fill import Fill
from gal_friday.utils.performance_optimizer import LRUCache

if TYPE_CHECKING:
    from gal_friday.logger_service import LoggerService


class TradeType(str, Enum):
    """Trade type enumeration."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class TradeRecord:
    """Comprehensive trade record with all required fields."""
    trade_id: str
    symbol: str
    trade_type: TradeType
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    strategy_id: str | None
    commission: Decimal
    commission_asset: str
    realized_pnl: Decimal | None
    order_id: str | None
    liquidity_type: str | None
    exchange: str

    def to_dict(self) -> dict[str, Any]:
        """Convert trade record to dictionary for API response."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "trade_type": self.trade_type.value,
            "quantity": str(self.quantity),
            "price": str(self.price),
            "timestamp": self.timestamp.isoformat() + "Z",
            "strategy_id": self.strategy_id,
            "commission": str(self.commission),
            "commission_asset": self.commission_asset,
            "realized_pnl": str(self.realized_pnl) if self.realized_pnl is not None else None,
            "order_id": self.order_id,
            "liquidity_type": self.liquidity_type,
            "exchange": self.exchange,
        }


@dataclass
class TradeHistoryRequest:
    """Request parameters for trade history retrieval."""
    trading_pair: str
    start_date: datetime | None = None
    end_date: datetime | None = None
    strategy_id: str | None = None
    limit: int = 1000
    offset: int = 0

    def __post_init__(self) -> None:
        """Validate request parameters."""
        if self.limit <= 0 or self.limit > 10000:
            raise DataValidationError("Limit must be between 1 and 10000")
        
        if self.offset < 0:
            raise DataValidationError("Offset must be non-negative")
        
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise DataValidationError("Start date must be before end date")


@dataclass
class TradeHistoryResponse:
    """Response containing trade history data with metadata."""
    trades: list[TradeRecord]
    total_count: int
    page_info: dict[str, Any]
    cache_info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary for API."""
        return {
            "trades": [trade.to_dict() for trade in self.trades],
            "total_count": self.total_count,
            "page_info": self.page_info,
            "cache_info": self.cache_info,
        }


class TradeHistoryService:
    """Enterprise-grade trade history service with caching and comprehensive filtering.
    
    Features:
    - Database integration with optimized queries
    - LRU caching for performance optimization
    - Filtering by date range, symbol, and strategy
    - Pagination for large datasets
    - Data validation and error handling
    - Comprehensive analytics interface
    """

    def __init__(
        self,
        session_maker: async_sessionmaker[AsyncSession],
        logger: "LoggerService",
        cache_size: int = 500,
        cache_ttl_seconds: int = 300,
    ) -> None:
        """Initialize the trade history service.

        Args:
            session_maker: SQLAlchemy async session maker
            logger: Logger service for system messages
            cache_size: Maximum number of cached queries
            cache_ttl_seconds: Cache time-to-live in seconds
        """
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # Initialize repositories
        self.fill_repository = FillRepository(session_maker, logger)
        
        # Initialize caching
        self.cache = LRUCache[TradeHistoryResponse](maxsize=cache_size)
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._cache_timestamps: dict[str, datetime] = {}

        self.logger.info(
            f"TradeHistoryService initialized with cache_size={cache_size}, "
            f"cache_ttl={cache_ttl_seconds}s",
            source_module=self._source_module,
        )

    async def get_trade_history(
        self,
        request: TradeHistoryRequest,
    ) -> TradeHistoryResponse:
        """Retrieve trade history with caching and comprehensive filtering.

        Args:
            request: Trade history request parameters

        Returns:
            TradeHistoryResponse containing filtered and paginated trade data

        Raises:
            DataValidationError: If request parameters are invalid
        """
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Check cache first
        cached_response = await self._get_from_cache(cache_key)
        if cached_response:
            self.logger.debug(
                f"Returning cached trade history for {request.trading_pair}",
                source_module=self._source_module,
            )
            return cached_response

        # Query database
        start_time = datetime.now(UTC)
        
        try:
            # Get fills from database
            fills = await self._query_fills_from_database(request)
            
            # Get total count for pagination
            total_count = await self._get_total_count(request)
            
            # Convert fills to trade records
            trade_records = [self._convert_fill_to_trade_record(fill) for fill in fills]
            
            # Calculate query time
            query_time = (datetime.now(UTC) - start_time).total_seconds()
            
            # Create response
            response = TradeHistoryResponse(
                trades=trade_records,
                total_count=total_count,
                page_info={
                    "limit": request.limit,
                    "offset": request.offset,
                    "has_next": (request.offset + len(trade_records)) < total_count,
                    "has_previous": request.offset > 0,
                    "total_pages": (total_count + request.limit - 1) // request.limit,
                    "current_page": (request.offset // request.limit) + 1,
                },
                cache_info={
                    "cached": False,
                    "query_time_seconds": round(query_time, 3),
                    "cache_key": cache_key,
                },
            )
            
            # Cache the response
            await self._cache_response(cache_key, response)
            
            self.logger.info(
                f"Retrieved {len(trade_records)} trades for {request.trading_pair} "
                f"(total: {total_count}, query_time: {query_time:.3f}s)",
                source_module=self._source_module,
            )
            
            return response

        except Exception as e:
            self.logger.exception(
                f"Error retrieving trade history for {request.trading_pair}: {e}",
                source_module=self._source_module,
            )
            raise

    async def get_trade_history_for_pair(
        self,
        trading_pair: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Simplified interface for getting trade history for a trading pair.
        
        This method provides a simplified interface that returns the trade data
        in the format expected by the existing PortfolioManager API.

        Args:
            trading_pair: Trading pair to get history for
            start_date: Optional start date filter
            end_date: Optional end date filter  
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of trade dictionaries in the format expected by PortfolioManager
        """
        request = TradeHistoryRequest(
            trading_pair=trading_pair,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )
        
        response = await self.get_trade_history(request)
        
        # Convert to the expected format for PortfolioManager compatibility
        return [
            {
                "timestamp": trade.timestamp.isoformat() + "Z",
                "side": trade.trade_type.value,
                "quantity": str(trade.quantity),
                "price": str(trade.price),
                "commission": str(trade.commission),
                "commission_asset": trade.commission_asset,
            }
            for trade in response.trades
        ]

    async def get_analytics_summary(
        self,
        trading_pair: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Get analytics summary for trade history.

        Args:
            trading_pair: Trading pair to analyze
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Dictionary containing trade analytics
        """
        request = TradeHistoryRequest(
            trading_pair=trading_pair,
            start_date=start_date,
            end_date=end_date,
            limit=10000,  # Get all trades for analytics
        )
        
        response = await self.get_trade_history(request)
        trades = response.trades
        
        if not trades:
            return {
                "total_trades": 0,
                "total_volume": "0",
                "total_commission": "0",
                "avg_trade_size": "0",
                "buy_trades": 0,
                "sell_trades": 0,
            }

        total_volume = sum(trade.quantity * trade.price for trade in trades)
        total_commission = sum(trade.commission for trade in trades)
        buy_trades = sum(1 for trade in trades if trade.trade_type == TradeType.BUY)
        sell_trades = sum(1 for trade in trades if trade.trade_type == TradeType.SELL)
        avg_trade_size = total_volume / len(trades) if trades else Decimal(0)

        return {
            "total_trades": len(trades),
            "total_volume": str(total_volume),
            "total_commission": str(total_commission),
            "avg_trade_size": str(avg_trade_size),
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "date_range": {
                "start": trades[-1].timestamp.isoformat() + "Z" if trades else None,
                "end": trades[0].timestamp.isoformat() + "Z" if trades else None,
            },
        }

    def _generate_cache_key(self, request: TradeHistoryRequest) -> str:
        """Generate a unique cache key for the request."""
        key_data = {
            "trading_pair": request.trading_pair,
            "start_date": request.start_date.isoformat() if request.start_date else None,
            "end_date": request.end_date.isoformat() if request.end_date else None,
            "strategy_id": request.strategy_id,
            "limit": request.limit,
            "offset": request.offset,
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    async def _get_from_cache(self, cache_key: str) -> TradeHistoryResponse | None:
        """Get response from cache if not expired."""
        cached_response = await self.cache.get(cache_key)
        if not cached_response:
            return None

        # Check if cache entry is expired
        cache_time = self._cache_timestamps.get(cache_key)
        if not cache_time or (datetime.now(UTC) - cache_time) > self.cache_ttl:
            # Remove expired entry
            await self.cache.set(cache_key, None)  # This will effectively remove it
            self._cache_timestamps.pop(cache_key, None)
            return None

        # Update cache info to indicate it was cached
        cached_response.cache_info["cached"] = True
        cached_response.cache_info["cache_age_seconds"] = (
            datetime.now(UTC) - cache_time
        ).total_seconds()

        return cached_response

    async def _cache_response(self, cache_key: str, response: TradeHistoryResponse) -> None:
        """Cache the response with timestamp."""
        await self.cache.set(cache_key, response)
        self._cache_timestamps[cache_key] = datetime.now(UTC)

    async def _query_fills_from_database(self, request: TradeHistoryRequest) -> list[Fill]:
        """Query fills from database based on request parameters."""
        if request.strategy_id:
            return await self.fill_repository.get_fills_by_strategy(
                strategy_id=request.strategy_id,
                start_date=request.start_date,
                end_date=request.end_date,
                limit=request.limit,
                offset=request.offset,
            )
        else:
            return await self.fill_repository.get_fills_by_trading_pair(
                trading_pair=request.trading_pair,
                start_date=request.start_date,
                end_date=request.end_date,
                limit=request.limit,
                offset=request.offset,
            )

    async def _get_total_count(self, request: TradeHistoryRequest) -> int:
        """Get total count of records for pagination."""
        return await self.fill_repository.get_fills_count_by_trading_pair(
            trading_pair=request.trading_pair,
            start_date=request.start_date,
            end_date=request.end_date,
        )

    def _convert_fill_to_trade_record(self, fill: Fill) -> TradeRecord:
        """Convert a Fill model to a TradeRecord."""
        return TradeRecord(
            trade_id=fill.fill_id or f"fill_{fill.fill_pk}",
            symbol=fill.trading_pair,
            trade_type=TradeType(fill.side.upper()),
            quantity=fill.quantity_filled,
            price=fill.fill_price,
            timestamp=fill.filled_at,
            strategy_id=None,  # Would need to join with Order/TradeSignal to get this
            commission=fill.commission,
            commission_asset=fill.commission_asset,
            realized_pnl=None,  # This would be calculated at the position level
            order_id=fill.exchange_order_id,
            liquidity_type=fill.liquidity_type,
            exchange=fill.exchange,
        )

    async def clear_cache(self) -> None:
        """Clear all cached data."""
        await self.cache.clear()
        self._cache_timestamps.clear()
        self.logger.info("Trade history cache cleared", source_module=self._source_module)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        cache_stats = self.cache.get_stats()
        return {
            **cache_stats,
            "ttl_seconds": self.cache_ttl.total_seconds(),
            "active_timestamps": len(self._cache_timestamps),
        } 