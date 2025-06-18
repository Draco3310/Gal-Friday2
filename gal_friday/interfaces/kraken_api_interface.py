"""Comprehensive Kraken API interface for robust trading operations.

This module provides a complete abstraction for Kraken API interactions,
including proper rate limiting, error handling, and comprehensive endpoint coverage
as required by the SRS specifications.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol


class KrakenOrderStatus(Enum):
    """Kraken order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    EXPIRED = "expired"


class KrakenOrderType(Enum):
    """Kraken order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop-loss"
    TAKE_PROFIT = "take-profit"
    STOP_LOSS_LIMIT = "stop-loss-limit"
    TAKE_PROFIT_LIMIT = "take-profit-limit"


@dataclass
class KrakenOrderResult:
    """Result from placing an order with Kraken."""
    success: bool
    order_id: str | None = None
    description: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    txid: list[str] | None = None


@dataclass
class KrakenOrderInfo:
    """Comprehensive order information from Kraken."""
    order_id: str
    status: KrakenOrderStatus
    order_type: KrakenOrderType
    side: str  # "buy" or "sell"
    volume: Decimal
    price: Decimal | None
    cost: Decimal | None
    fee: Decimal | None
    avg_price: Decimal | None
    stop_price: Decimal | None
    limit_price: Decimal | None
    misc: str | None
    oflags: str | None
    trades: list[str] | None
    open_timestamp: datetime | None
    close_timestamp: datetime | None
    reason: str | None


@dataclass
class KrakenBalance:
    """Account balance information."""
    currency: str
    balance: Decimal
    hold: Decimal
    available: Decimal


@dataclass
class KrakenTickerData:
    """Ticker data from Kraken."""
    pair: str
    ask: Decimal
    bid: Decimal
    last: Decimal
    volume: Decimal
    volume_weighted_avg: Decimal
    num_trades: int
    low: Decimal
    high: Decimal
    opening_price: Decimal


class KrakenAPIRateLimit(Protocol):
    """Protocol for Kraken API rate limiting."""

    @abstractmethod
    async def wait_for_private_capacity(self) -> None:
        """Wait until there's capacity for a private API call."""

    @abstractmethod
    async def wait_for_public_capacity(self) -> None:
        """Wait until there's capacity for a public API call."""

    @abstractmethod
    def reset(self) -> None:
        """Reset rate limit tracking."""


class KrakenAPIInterface(ABC):
    """Abstract interface for Kraken API operations.

    This interface defines all required Kraken API operations for the trading system,
    ensuring comprehensive functionality and proper error handling.
    """

    @abstractmethod
    async def get_server_time(self) -> dict[str, Any]:
        """Get Kraken server time.

        Returns:
            Server time information

        Raises:
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def get_system_status(self) -> dict[str, Any]:
        """Get Kraken system status.

        Returns:
            System status information

        Raises:
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def get_asset_info(self, assets: list[str] | None = None) -> dict[str, Any]:
        """Get asset information.

        Args:
            assets: List of assets to get info for, None for all

        Returns:
            Asset information dictionary

        Raises:
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def get_tradable_asset_pairs(
        self,
        pairs: list[str] | None = None) -> dict[str, Any]:
        """Get tradable asset pairs information.

        Args:
            pairs: List of pairs to get info for, None for all

        Returns:
            Asset pairs information dictionary

        Raises:
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def get_ticker_data(self, pairs: list[str]) -> dict[str, KrakenTickerData]:
        """Get ticker data for specified pairs.

        Args:
            pairs: List of trading pairs

        Returns:
            Dictionary mapping pairs to ticker data

        Raises:
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def get_ohlc_data(
        self,
        pair: str,
        interval: int = 1,
        since: int | None = None) -> dict[str, Any]:
        """Get OHLC data for a trading pair.

        Args:
            pair: Trading pair
            interval: Time frame interval in minutes
            since: Unix timestamp to get data since

        Returns:
            OHLC data dictionary

        Raises:
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def get_order_book(
        self,
        pair: str,
        count: int = 100) -> dict[str, Any]:
        """Get order book for a trading pair.

        Args:
            pair: Trading pair
            count: Maximum number of asks/bids

        Returns:
            Order book data

        Raises:
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def get_recent_trades(
        self,
        pair: str,
        since: int | None = None) -> dict[str, Any]:
        """Get recent trades for a trading pair.

        Args:
            pair: Trading pair
            since: Unix timestamp to get trades since

        Returns:
            Recent trades data

        Raises:
            ExecutionHandlerNetworkError: If network request fails
        """

    # Private API methods

    @abstractmethod
    async def get_account_balance(self) -> dict[str, KrakenBalance]:
        """Get account balance.

        Returns:
            Dictionary mapping currencies to balance info

        Raises:
            ExecutionHandlerAuthenticationError: If authentication fails
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def get_trade_balance(self, asset: str = "ZUSD") -> dict[str, Any]:
        """Get trade balance information.

        Args:
            asset: Base asset for balance calculation

        Returns:
            Trade balance information

        Raises:
            ExecutionHandlerAuthenticationError: If authentication fails
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def get_open_orders(self, trades: bool = False) -> dict[str, KrakenOrderInfo]:
        """Get open orders.

        Args:
            trades: Whether to include trades in output

        Returns:
            Dictionary mapping order IDs to order info

        Raises:
            ExecutionHandlerAuthenticationError: If authentication fails
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def get_closed_orders(
        self,
        trades: bool = False,
        start: int | None = None,
        end: int | None = None,
        ofs: int | None = None,
        closetime: str = "both") -> dict[str, KrakenOrderInfo]:
        """Get closed orders.

        Args:
            trades: Whether to include trades in output
            start: Starting unix timestamp
            end: Ending unix timestamp
            ofs: Result offset
            closetime: Which time to use for filtering

        Returns:
            Dictionary mapping order IDs to order info

        Raises:
            ExecutionHandlerAuthenticationError: If authentication fails
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def query_orders_info(
        self,
        order_ids: list[str],
        trades: bool = False) -> dict[str, KrakenOrderInfo]:
        """Query information about specific orders.

        Args:
            order_ids: List of order IDs to query
            trades: Whether to include trades in output

        Returns:
            Dictionary mapping order IDs to order info

        Raises:
            ExecutionHandlerAuthenticationError: If authentication fails
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def get_trades_history(
        self,
        type_filter: str = "all",
        trades: bool = False,
        start: int | None = None,
        end: int | None = None,
        ofs: int | None = None) -> dict[str, Any]:
        """Get trades history.

        Args:
            type_filter: Type[Any] of trade to retrieve
            trades: Whether to include trades info in output
            start: Starting unix timestamp
            end: Ending unix timestamp
            ofs: Result offset

        Returns:
            Trades history data

        Raises:
            ExecutionHandlerAuthenticationError: If authentication fails
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def query_trades_info(
        self,
        trade_ids: list[str],
        trades: bool = False) -> dict[str, Any]:
        """Query information about specific trades.

        Args:
            trade_ids: List of trade IDs to query
            trades: Whether to include trades info in output

        Returns:
            Trades information data

        Raises:
            ExecutionHandlerAuthenticationError: If authentication fails
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def get_open_positions(self, do_calcs: bool = False) -> dict[str, Any]:
        """Get open positions.

        Args:
            do_calcs: Whether to include unrealized P&L calculations

        Returns:
            Open positions data

        Raises:
            ExecutionHandlerAuthenticationError: If authentication fails
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def get_ledgers_info(
        self,
        asset: str = "all",
        aclass: str = "currency",
        type_filter: str = "all",
        start: int | None = None,
        end: int | None = None,
        ofs: int | None = None) -> dict[str, Any]:
        """Get ledgers information.

        Args:
            asset: Asset to retrieve ledger info for
            aclass: Asset class to retrieve info for
            type_filter: Type[Any] of ledger to retrieve
            start: Starting unix timestamp
            end: Ending unix timestamp
            ofs: Result offset

        Returns:
            Ledgers information data

        Raises:
            ExecutionHandlerAuthenticationError: If authentication fails
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def query_ledgers(self, ledger_ids: list[str]) -> dict[str, Any]:
        """Query specific ledgers.

        Args:
            ledger_ids: List of ledger IDs to query

        Returns:
            Ledger information data

        Raises:
            ExecutionHandlerAuthenticationError: If authentication fails
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def get_trade_volume(self, pairs: list[str] | None = None) -> dict[str, Any]:
        """Get trade volume and fees information.

        Args:
            pairs: List of pairs to get fee info for

        Returns:
            Trade volume and fees data

        Raises:
            ExecutionHandlerAuthenticationError: If authentication fails
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def request_export_report(
        self,
        report: str,
        description: str,
        format_type: str = "CSV",
        fields: str | None = None,
        start: int | None = None,
        end: int | None = None) -> dict[str, Any]:
        """Request export report.

        Args:
            report: Type[Any] of report to request
            description: Description for the report
            format_type: Format of the report (CSV, TSV)
            fields: Fields to include in report
            start: Starting unix timestamp
            end: Ending unix timestamp

        Returns:
            Export report request data

        Raises:
            ExecutionHandlerAuthenticationError: If authentication fails
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def get_export_report_status(self, report: str) -> dict[str, Any]:
        """Get export report status.

        Args:
            report: Report type to check status for

        Returns:
            Export report status data

        Raises:
            ExecutionHandlerAuthenticationError: If authentication fails
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def retrieve_export_report(self, id: str) -> bytes:
        """Retrieve completed export report.

        Args:
            id: Report ID to retrieve

        Returns:
            Report data as bytes

        Raises:
            ExecutionHandlerAuthenticationError: If authentication fails
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def delete_export_report(self, id: str, type: str) -> dict[str, Any]:
        """Delete export report.

        Args:
            id: Report ID to delete
            type: Type[Any] of report to delete

        Returns:
            Deletion confirmation

        Raises:
            ExecutionHandlerAuthenticationError: If authentication fails
            ExecutionHandlerNetworkError: If network request fails
        """

    # Trading methods

    @abstractmethod
    async def add_order(
        self,
        pair: str,
        type_: str,
        ordertype: str,
        volume: str,
        price: str | None = None,
        price2: str | None = None,
        leverage: str | None = None,
        oflags: str | None = None,
        starttm: str | None = None,
        expiretm: str | None = None,
        close: dict[str, str] | None = None,
        trading_agreement: str = "agree",
        validate: bool = False) -> KrakenOrderResult:
        """Add a new order.

        Args:
            pair: Trading pair
            type_: Order direction (buy/sell)
            ordertype: Order type (market, limit, etc.)
            volume: Order volume
            price: Price (required for limit orders)
            price2: Secondary price (for stop orders)
            leverage: Leverage (if supported)
            oflags: Order flags
            starttm: Scheduled start time
            expiretm: Expiration time
            close: Close order details
            trading_agreement: Trading agreement acknowledgment
            validate: Whether to validate order only

        Returns:
            Order result information

        Raises:
            ExecutionHandlerAuthenticationError: If authentication fails
            ExecutionHandlerOrderRejected: If order is rejected
            ExecutionHandlerNetworkError: If network request fails
            ExecutionHandlerRateLimited: If rate limited
        """

    @abstractmethod
    async def edit_order(
        self,
        txid: str,
        pair: str,
        volume: str | None = None,
        price: str | None = None,
        price2: str | None = None,
        oflags: str | None = None,
        newuserref: str | None = None,
        validate: bool = False) -> dict[str, Any]:
        """Edit an existing order.

        Args:
            txid: Transaction ID of order to edit
            pair: Trading pair
            volume: New order volume
            price: New price
            price2: New secondary price
            oflags: New order flags
            newuserref: New user reference ID
            validate: Whether to validate edit only

        Returns:
            Edit result information

        Raises:
            ExecutionHandlerAuthenticationError: If authentication fails
            ExecutionHandlerOrderRejected: If edit is rejected
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def cancel_order(self, txid: str) -> dict[str, Any]:
        """Cancel an order.

        Args:
            txid: Transaction ID of order to cancel

        Returns:
            Cancellation confirmation

        Raises:
            ExecutionHandlerAuthenticationError: If authentication fails
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def cancel_all_orders(self) -> dict[str, Any]:
        """Cancel all open orders.

        Returns:
            Cancellation confirmation for all orders

        Raises:
            ExecutionHandlerAuthenticationError: If authentication fails
            ExecutionHandlerNetworkError: If network request fails
        """

    @abstractmethod
    async def cancel_all_orders_after(self, timeout: int) -> dict[str, Any]:
        """Cancel all orders after a timeout.

        Args:
            timeout: Timeout in seconds

        Returns:
            Cancellation timer confirmation

        Raises:
            ExecutionHandlerAuthenticationError: If authentication fails
            ExecutionHandlerNetworkError: If network request fails
        """

    # WebSocket methods

    @abstractmethod
    async def get_websockets_token(self) -> dict[str, Any]:
        """Get WebSocket authentication token.

        Returns:
            WebSocket token information

        Raises:
            ExecutionHandlerAuthenticationError: If authentication fails
            ExecutionHandlerNetworkError: If network request fails
        """
