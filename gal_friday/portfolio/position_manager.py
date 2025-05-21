"""Position management functionality for the portfolio system."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Optional, TypedDict, Unpack

from ..exceptions import DataValidationError
from ..logger_service import LoggerService


@dataclass
class TradeInfo:
    """Represents information about a single trade."""

    timestamp: datetime
    trade_id: str
    side: str  # "BUY" or "SELL"
    quantity: Decimal
    price: Decimal
    fee: Decimal = Decimal(0)
    fee_currency: Optional[str] = None
    commission: Decimal = Decimal(0)
    commission_asset: Optional[str] = None


@dataclass
class PositionInfo:
    """Maintains information about a trading position."""

    trading_pair: str
    base_asset: str
    quote_asset: str
    quantity: Decimal = Decimal(0)
    average_entry_price: Decimal = Decimal(0)
    realized_pnl: Decimal = Decimal(0)
    trade_history: list[TradeInfo] = field(default_factory=list)
    last_updated: Optional[datetime] = None


class PositionManager:
    """Manages trading positions, including updates from trades and PnL calculations."""

    def __init__(self, logger_service: LoggerService) -> None:
        """
        Initialize the position manager.

        Args
        ----
            logger_service: Service for logging.
        """
        self.logger = logger_service
        self._source_module = self.__class__.__name__
        self._positions: dict[str, PositionInfo] = {}
        self._lock = asyncio.Lock()

    @property
    def positions(self) -> dict[str, PositionInfo]:
        """Return all positions as a read-only copy."""
        return self._positions.copy()

    def get_open_positions(self) -> list[PositionInfo]:
        """Return a list of open positions (non-zero quantity)."""
        return [pos for pos in self._positions.values() if pos.quantity != 0]

    def get_position(self, trading_pair: str) -> Optional[PositionInfo]:
        """Get a specific position by trading pair."""
        return self._positions.get(trading_pair)

    async def initialize_positions(
        self,
        initial_positions: dict[str, dict[str, Any]],
        split_symbol_func: Callable[[str], tuple[str, str]],
    ) -> None:
        """
        Initialize positions from a dictionary (e.g., from config or database).

        Args
        ----
            initial_positions: Dict where keys are trading_pair and values are dicts
                               of position attributes (quantity, average_entry_price, etc.).
            split_symbol_func: A function to split trading_pair into base and quote.
        """
        async with self._lock:
            self._positions.clear()
            for pair, pos_data in initial_positions.items():
                try:
                    base, quote = split_symbol_func(pair)
                    position = PositionInfo(
                        trading_pair=pair,
                        base_asset=base,
                        quote_asset=quote,
                        quantity=Decimal(str(pos_data.get("quantity", 0))),
                        average_entry_price=Decimal(str(pos_data.get("average_entry_price", 0))),
                        realized_pnl=Decimal(str(pos_data.get("realized_pnl", 0))),
                        last_updated=datetime.now().astimezone()
                    )
                    self._positions[pair] = position
                except ValueError:
                    self.logger.exception(
                        "Invalid trading pair format in initial_positions: %s", pair,
                        source_module=self._source_module,
                    )
                except Exception as e:
                    self.logger.exception(
                        "Error loading initial position for %s", pair,
                        exc_info=e,
                        source_module=self._source_module,
                    )
            self.logger.info(
                "Initialized %d positions.",
                len(self._positions),
                source_module=self._source_module
            )

    @dataclass
    class _UpdatePositionParams:
        """Parameters for updating a position from a trade."""

        trading_pair: str
        side: str
        quantity: Decimal
        price: Decimal
        timestamp: datetime
        trade_id: str
        fee: Decimal = field(default_factory=Decimal)
        fee_currency: Optional[str] = None
        commission: Decimal = field(default_factory=Decimal)
        commission_asset: Optional[str] = None

    @dataclass
    class _UpdatePositionParams:
        """Parameters for updating a position from a trade."""

        trading_pair: str
        side: str
        quantity: Decimal
        price: Decimal
        timestamp: datetime
        trade_id: str
        fee: Decimal = field(default_factory=Decimal)
        fee_currency: Optional[str] = None
        commission: Decimal = field(default_factory=Decimal)
        commission_asset: Optional[str] = None

        # This method is used by the dataclass for field initialization
        def __post_init__(self) -> None:
            """Validate the parameters after initialization."""
            if not isinstance(self.quantity, Decimal):
                self.quantity = Decimal(self.quantity)
            if not isinstance(self.price, Decimal):
                self.price = Decimal(self.price)
            if not isinstance(self.fee, Decimal):
                self.fee = Decimal(self.fee)
            if not isinstance(self.commission, Decimal):
                self.commission = Decimal(self.commission)

    @dataclass
    class _UpdatePositionKwargs(TypedDict, total=False):
        """Type definition for update_position_for_trade kwargs."""

        trading_pair: str
        side: str
        quantity: Decimal | float | str
        price: Decimal | float | str
        timestamp: datetime
        trade_id: str
        fee: Decimal | float | str
        fee_currency: Optional[str]
        commission: Decimal | float | str
        commission_asset: Optional[str]

    async def update_position_for_trade(
        self,
        **kwargs: Unpack[_UpdatePositionKwargs],
    ) -> tuple[Decimal, Optional[PositionInfo]]:
        """Update position based on a new trade.

        This is a convenience wrapper around _update_position_for_trade_impl
        that maintains backward compatibility.

        Args:
            **kwargs: Parameters for the trade update. See _UpdatePositionKwargs for details.

        Returns
        -------
            A tuple containing:
                - realized_pnl: The realized PnL from this trade
                - position: The updated position, or None if there was an error
        """
        return await self._update_position_for_trade_impl(
            self._UpdatePositionParams(**kwargs)
        )

    async def _update_position_for_trade_impl(
        self,
        params: _UpdatePositionParams,
    ) -> tuple[Decimal, Optional[PositionInfo]]:
        """Update position based on trade execution.

        Args:
            params: _UpdatePositionParams containing all trade parameters

        Returns
        -------
            A tuple containing:
                - realized_pnl: The realized PnL from this trade
                - position: The updated position, or None if there was an error
        """
        try:
            self._validate_trade_params(
                params.trading_pair,
                params.side,
                params.quantity,
                params.price
            )
        except DataValidationError as e:
            self.logger.exception(
                "Trade validation failed for %s",
                params.trading_pair,
                exc_info=e,
                source_module=self._source_module
            )
            return Decimal(0), None

        async with self._lock:
            position = await self._get_or_create_position(params.trading_pair)
            trade_record = self._create_trade_record_from_params(params)
            position.trade_history.append(trade_record)

            if params.side == "BUY":
                new_total_cost = (
                    position.average_entry_price * position.quantity
                ) + (params.price * params.quantity)
                position.quantity += params.quantity
                if position.quantity != 0:
                    position.average_entry_price = new_total_cost / position.quantity
                else:
                    position.average_entry_price = Decimal(0)
                realized_pnl_trade = Decimal(0)
            elif params.side == "SELL":
                if position.quantity < params.quantity:
                    self.logger.warning(
                        "Selling %s of %s, but current position is %s. "
                        "Position will go negative or this is a short sale.",
                        params.quantity,
                        params.trading_pair,
                        position.quantity,
                        source_module=self._source_module
                    )

                realized_pnl_trade = (
                    (params.price - position.average_entry_price) * params.quantity
                )
                position.realized_pnl += realized_pnl_trade
                position.quantity -= params.quantity
                if position.quantity == 0:
                    position.average_entry_price = Decimal(0)

            position.last_updated = params.timestamp
            self._positions[params.trading_pair] = position
            self.logger.info(
                "Updated position - Pair: %s, Side: %s, Qty: %s, Price: %s, "
                "New Qty: %s, AEP: %s, PnL: %s",
                params.trading_pair,
                params.side,
                params.quantity,
                params.price,
                position.quantity,
                position.average_entry_price,
                realized_pnl_trade,
                source_module=self._source_module
            )
            return realized_pnl_trade, position

    # Error messages for trade validation
    _INVALID_TRADE_SIDE_MSG = "Invalid trade side"
    _INVALID_QUANTITY_MSG = "Invalid quantity"
    _INVALID_PRICE_MSG = "Invalid price"

    def _validate_trade_params(
        self,
        _trading_pair: str,  # Unused, keeping for backward compatibility
        side: str,
        quantity: Decimal,
        price: Decimal,
    ) -> None:
        """Validate trade parameters.

        Args:
            _trading_pair: Unused parameter (kept for backward compatibility)
            side: Trade side (must be 'BUY' or 'SELL')
            quantity: Trade quantity (must be > 0)
            price: Trade price (must be > 0)

        Raises
        ------
            DataValidationError: If any parameter is invalid
        """
        if side not in ("BUY", "SELL"):
            raise DataValidationError(self._INVALID_TRADE_SIDE_MSG)

        if quantity <= Decimal(0):
            raise DataValidationError(self._INVALID_QUANTITY_MSG)

        if price <= Decimal(0):
            raise DataValidationError(self._INVALID_PRICE_MSG)

    # Constants for position validation
    _EXPECTED_PARTS_COUNT = 2  # Expected number of parts in a trading pair (BASE/QUOTE)

    # Error messages for position creation
    _INVALID_PAIR_FORMAT_MSG = "Trading pair format must be BASE/QUOTE"
    _EMPTY_ASSET_MSG = "Base or quote asset cannot be empty"
    _PAIR_CREATION_ERROR_MSG = "Invalid trading pair format for position creation: {}"
    _POSITION_CREATED_MSG = "Created new position for %s"
    _PARSE_PAIR_ERROR_MSG = "Failed to parse trading pair"

    async def _get_or_create_position(self, trading_pair: str) -> PositionInfo:
        """Get existing position or create a new one if it doesn't exist.

        Args:
            trading_pair: Trading pair in format 'BASE/QUOTE'

        Returns
        -------
            PositionInfo: The existing or newly created position

        Raises
        ------
            DataValidationError: If trading pair format is invalid
        """
        if trading_pair in self._positions:
            return self._positions[trading_pair]

        def _validate_parts(parts: list[str]) -> None:
            if len(parts) != self._EXPECTED_PARTS_COUNT:
                raise ValueError(self._INVALID_PAIR_FORMAT_MSG)

        def _validate_assets(base: str, quote: str) -> None:
            if not base or not quote:
                raise ValueError(self._EMPTY_ASSET_MSG)

        try:
            parts = trading_pair.split("/")
            _validate_parts(parts)
            base_asset, quote_asset = parts[0].strip(), parts[1].strip()
            _validate_assets(base_asset, quote_asset)

            self._positions[trading_pair] = PositionInfo(
                trading_pair=trading_pair,
                base_asset=base_asset,
                quote_asset=quote_asset
            )
            self.logger.info(
                self._POSITION_CREATED_MSG,
                trading_pair,
                source_module=self._source_module
            )

        except ValueError as e:
            self.logger.exception(
                "%s: %s",
                self._PARSE_PAIR_ERROR_MSG,
                trading_pair,
                source_module=self._source_module
            )
            raise DataValidationError(
                self._PAIR_CREATION_ERROR_MSG.format(trading_pair)
            ) from e

        return self._positions[trading_pair]

    async def get_total_realized_pnl(self, trading_pair: Optional[str] = None) -> Decimal:
        """Get total realized PnL for a specific pair or all pairs."""
        async with self._lock:
            if trading_pair:
                pos = self._positions.get(trading_pair)
                return pos.realized_pnl if pos else Decimal(0)
            return sum(pos.realized_pnl for pos in self._positions.values())

    @dataclass
    class _TradeRecordParams:
        """Container for trade record parameters."""

        timestamp: datetime
        trade_id: str
        trading_pair: str
        side: str
        quantity: Decimal
        price: Decimal
        fee: Decimal = field(default_factory=Decimal)
        fee_currency: Optional[str] = None
        commission: Decimal = field(default_factory=Decimal)
        commission_asset: Optional[str] = None

    class _TradeRecordKwargs(TypedDict, total=False):
        """Type definition for _create_trade_record kwargs."""

        timestamp: datetime
        trade_id: str
        trading_pair: str
        side: str
        quantity: Decimal | float | str
        price: Decimal | float | str
        fee: Decimal | float | str
        fee_currency: Optional[str]
        commission: Decimal | float | str
        commission_asset: Optional[str]

    def _create_trade_record(self, **kwargs: Unpack[_TradeRecordKwargs]) -> TradeInfo:
        """Create a trade record (legacy interface).

        This is a backward compatibility wrapper around _create_trade_record_from_params.
        New code should use _create_trade_record_from_params directly.

        Args:
            **kwargs: Parameters for the trade record. See _TradeRecordParams for details.

        Returns
        -------
            TradeInfo: The created trade record
        """
        params = self._TradeRecordParams(**kwargs)
        return self._create_trade_record_from_params(params)

    def _create_trade_record_from_params(
        self,
        params: _TradeRecordParams,
    ) -> TradeInfo:
        """Create a TradeInfo object from TradeRecordParams.

        Args:
            params: TradeRecordParams containing all trade parameters

        Returns
        -------
            TradeInfo: A new TradeInfo object with the provided parameters
        """
        return TradeInfo(
            timestamp=params.timestamp,
            trade_id=params.trade_id,
            trading_pair=params.trading_pair,
            side=params.side,
            quantity=params.quantity,
            price=params.price,
            fee=params.fee,
            fee_currency=params.fee_currency,
            commission=params.commission,
            commission_asset=params.commission_asset,
        )
