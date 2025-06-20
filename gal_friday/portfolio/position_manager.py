"""Position management functionality for the portfolio system using SQLAlchemy."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime  # Added timezone
from decimal import Decimal
from typing import TypedDict, Unpack
import uuid

import asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gal_friday.config_manager import ConfigManager  # For asset splitting or other config
from gal_friday.dal.models.position import Position as PositionModel  # SQLAlchemy model
from gal_friday.dal.repositories.order_repository import OrderRepository
from gal_friday.dal.repositories.position_repository import PositionRepository
from gal_friday.exceptions import DataValidationError
from gal_friday.logger_service import LoggerService


class PositionCreationError(Exception):
    """Error raised when position creation fails."""


@dataclass
class TradeInfo:
    """Represents information about a single trade.

    This is a business logic representation of a trade that may include
    fields not directly stored in the Position model. Trade-specific
    details like fees and commissions should ideally be stored in a
    separate Trade/Order table in the database.
    """

    timestamp: datetime
    trade_id: str  # Could be linked to an Order ID or Exchange Trade ID
    side: str  # "BUY" or "SELL"
    quantity: Decimal
    price: Decimal
    fee: Decimal = Decimal(0)
    fee_currency: str | None = None
    # Note: commission and commission_asset are trade-specific details
    # that are typically stored in a Trade/Order table, not in Position.
    # They're included here for business logic calculations but won't
    # be persisted directly to the Position model.
    commission: Decimal = Decimal(0)
    commission_asset: str | None = None


# PositionInfo dataclass might be replaced by the Position SQLAlchemy model for direct DB interaction.
# However, it can still be useful for business logic or as a DTO if its structure differs
# from the DB model or if we want to decouple business logic from DB schema.
# For this refactor, we'll primarily use PositionModel from the DB.

# Note: The Position model in the database represents the current state
# of a position (quantity, average entry price, PnL, etc.) but does not
# store individual trade history. Trade history should be maintained in
# a separate table for audit and reconciliation purposes.

# ARCHITECTURAL DECISION:
# - Position table: Stores current state (quantity, avg price, PnL)
# - Trade/Order table: Stores individual trade records with fees/commissions
# - This separation allows for:
#   1. Efficient position queries without loading full trade history
#   2. Detailed audit trail of all trades
#   3. Proper handling of partial fills and trade corrections
#   4. Compliance with regulatory requirements for trade record keeping

class PositionManager:
    """Manages trading positions, including updates from trades and PnL calculations, using SQLAlchemy."""

    def __init__(
        self,
        logger_service: LoggerService,
        session_maker: async_sessionmaker[AsyncSession],
        config_manager: ConfigManager, # For asset splitting logic if needed
    ) -> None:
        """Initialize the position manager.

        Args:
        ----
            logger_service: Service for logging.
            session_maker: SQLAlchemy async_sessionmaker for database sessions.
            config_manager: Configuration manager.
        """
        self.logger = logger_service
        self._source_module = self.__class__.__name__
        self.session_maker = session_maker
        self.position_repository = PositionRepository(session_maker, logger_service)
        self.order_repository = OrderRepository(session_maker, logger_service)
        self.config_manager = config_manager
        self._lock = asyncio.Lock()  # Lock can still be useful for critical async
        # operations on a single position if needed

    # The `positions` property and `get_open_positions` will now fetch from DB.
    # `get_position` will also fetch from DB.

    async def get_open_positions(self) -> Sequence[PositionModel]:
        """Return a list[Any] of open positions (is_active=True) from the database."""
        self.logger.debug("Fetching all active positions.", source_module=self._source_module)
        return await self.position_repository.get_active_positions()
        # If PositionInfo DTO is still desired:
        # db_positions = await self.position_repository.get_active_positions()
        # return [self._map_model_to_position_info(p) for p in db_positions]


    async def get_position(self, trading_pair: str) -> PositionModel | None:
        """Get a specific active position by trading_pair from the database."""
        self.logger.debug(f"Fetching position for trading_pair: {trading_pair}", source_module=self._source_module)
        return await self.position_repository.get_position_by_pair(trading_pair)
        # If PositionInfo DTO is still desired:
        # db_position = await self.position_repository.get_position_by_pair(trading_pair)
        # return self._map_model_to_position_info(db_position) if db_position else None

    async def initialize_positions(
        self,
        # initial_positions: dict[str, dict[str, Any]], # This would now be pre-loading DB
        # split_symbol_func: Callable[..., tuple[str, str]], # Asset splitting can be internal or from config
    ) -> None:
        """Initializes positions by loading them from the database.
        The concept of initializing from a config dict[str, Any] is replaced by ensuring
        positions are correctly populated in the DB, possibly by a seeding process
        or by prior application state. This method could verify DB connection
        or log current position count.
        """
        # This method might no longer be needed if positions are purely DB driven.
        # Or, it could be used to load positions from DB into a cache if desired,
        # but the primary goal is to use DB as source of truth.
        # For now, let's assume it just logs the number of active positions.
        active_positions = await self.get_open_positions()
        self.logger.info(
            "PositionManager initialized. Found %d active positions in database.",
            len(active_positions),
            source_module=self._source_module)
        # If we were to sync from a config (e.g. for initial setup in a non-prod env):
        # async with self._lock:
        #     for pair, pos_data in initial_positions_from_config.items():
        #         existing_pos = await self.position_repository.get_position_by_pair(pair)
        #         if not existing_pos:
        #             # Create new position in DB
        #             # ... map pos_data to PositionModel fields ...
        #             # await self.position_repository.create(...)


    @dataclass
    class _UpdatePositionParams: # This can remain as an internal helper for parameter validation
        """Parameters for updating a position from a trade."""

        trading_pair: str
        side: str
        quantity: Decimal # Changed to Decimal
        price: Decimal    # Changed to Decimal
        timestamp: datetime
        trade_id: str
        order_id: str
        fee: Decimal = field(default_factory=Decimal) # Changed to Decimal
        fee_currency: str | None = None
        commission: Decimal = field(default_factory=Decimal) # Changed to Decimal
        commission_asset: str | None = None

        # __post_init__ for type conversion is no longer needed here,
        # as conversion will be done prior to instantiation.
        # It can be kept for other validations if any.
        # def __post_init__(self) -> None:
        #     """Validate the parameters after initialization."""
        #     pass


    @dataclass # This TypedDict defines the types for **kwargs, so it must allow broader types
    class _UpdatePositionKwargs(TypedDict, total=False):
        """Type[Any] definition for update_position_for_trade kwargs."""

        trading_pair: str
        side: str
        quantity: Decimal | float | str
        price: Decimal | float | str
        timestamp: datetime
        trade_id: str
        order_id: str
        fee: Decimal | float | str
        fee_currency: str | None
        commission: Decimal | float | str
        commission_asset: str | None

    async def update_position_for_trade(  # Keep public interface
        self,
        **kwargs: Unpack[_UpdatePositionKwargs],
    ) -> tuple[Decimal, PositionModel | None]:  # Returns SQLAlchemy PositionModel
        """Update position based on a new trade.

        Args:
            **kwargs: Parameters for the trade update. See _UpdatePositionKwargs for details.
                     Must include order_id for position-order relationship tracking.

        Returns:
        -------
            A tuple[Any, ...] containing:
                - realized_pnl_trade: The realized PnL from this specific trade.
                - position: The updated PositionModel instance, or None if an error occurred.
        """
        try:
            # Extract and convert arguments to Decimal before creating _UpdatePositionParams
            qty_arg = kwargs.get("quantity")
            price_arg = kwargs.get("price")
            order_id_arg = kwargs.get("order_id")  # NEW: Required for relationship tracking

            # Provide defaults for fee and commission if not in kwargs, matching dataclass defaults
            fee_arg = kwargs.get("fee", Decimal(0))
            commission_arg = kwargs.get("commission", Decimal(0))

            # Ensure required fields are present (those not in _UpdatePositionKwargs
            # or without defaults in _UpdatePositionParams)
            if qty_arg is None or price_arg is None:
                raise ValueError("Quantity and price must be provided.")

            if order_id_arg is None:
                raise ValueError("order_id is required for position-order relationship tracking.")

            params = self._UpdatePositionParams(
                trading_pair=kwargs["trading_pair"],
                side=kwargs["side"],
                quantity=Decimal(str(qty_arg)) if not isinstance(qty_arg, Decimal) else qty_arg,
                price=Decimal(str(price_arg)) if not isinstance(price_arg, Decimal) else price_arg,
                timestamp=kwargs["timestamp"],
                trade_id=kwargs["trade_id"],
                order_id=str(order_id_arg),  # NEW: Store order_id for linking
                fee=Decimal(str(fee_arg)) if not isinstance(fee_arg, Decimal) else fee_arg,
                fee_currency=kwargs.get("fee_currency"),
                commission=Decimal(str(commission_arg)) if not isinstance(commission_arg, Decimal) else commission_arg,
                commission_asset=kwargs.get("commission_asset"))
        except (TypeError, ValueError, KeyError):
            self.logger.exception(
                "Invalid parameters for update_position_for_trade: ",
                source_module=self._source_module, context=kwargs,
            )
            return Decimal(0), None

        return await self._update_position_for_trade_impl(params)

    async def _update_position_for_trade_impl( # Keep as main logic implementation
        self,
        params: _UpdatePositionParams, # params now has fields guaranteed to be Decimal
    ) -> tuple[Decimal, PositionModel | None]: # Returns SQLAlchemy PositionModel
        """Update position based on trade execution. Persists changes to the database.

        Args:
            params: _UpdatePositionParams containing all trade parameters including order_id.

        Returns:
        -------
            A tuple[Any, ...] containing:
                - realized_pnl_trade: The realized PnL from this specific trade.
                - position_model: The updated PositionModel instance from DB, or None if an error.
        """
        try:
            # Now params.quantity and params.price are Decimal, so this call should be fine.
            self._validate_trade_params(
                params.trading_pair,
                params.side,
                params.quantity,
                params.price)

        except DataValidationError:
            self.logger.exception(
                f"Trade validation failed for {params.trading_pair}: ",
                source_module=self._source_module,
                context=params.__dict__)
            return Decimal(0), None

        async with self._lock: # Lock to ensure atomic read-modify-write for a given trading_pair
            position_model = await self._get_or_create_db_position(params.trading_pair, params.side)

            current_quantity = position_model.quantity
            avg_entry_price = position_model.entry_price # PositionModel uses 'entry_price' for AEP
            realized_pnl_trade = Decimal(0)

            if params.side.upper() == "BUY":
                # For buys, new AEP = (old_value + new_value) / (old_qty + new_qty)
                new_total_cost = (avg_entry_price * current_quantity) + (params.price * params.quantity)
                new_quantity = current_quantity + params.quantity

                position_model.quantity = new_quantity
                if new_quantity != Decimal(0): # Avoid division by zero if position becomes zero
                    position_model.entry_price = new_total_cost / new_quantity
                else:
                    position_model.entry_price = Decimal(0) # Reset AEP if quantity is zero

            elif params.side.upper() == "SELL":
                if current_quantity < params.quantity:
                    self.logger.warning(
                        f"Selling {params.quantity} of {params.trading_pair}, but current position is "
                        f"{current_quantity}. Position will go negative or this represents a short sale "
                        f"opening/extension.",
                        source_module=self._source_module)

                # Realized PnL = (sell_price - avg_entry_price) * sell_quantity
                realized_pnl_trade = (params.price - avg_entry_price) * params.quantity
                position_model.realized_pnl = (position_model.realized_pnl or Decimal(0)) + realized_pnl_trade
                new_quantity = current_quantity - params.quantity
                position_model.quantity = new_quantity

                if new_quantity == Decimal(0):
                    position_model.entry_price = Decimal(0) # Reset AEP
                    position_model.is_active = False # Close position if quantity is zero
                    position_model.closed_at = params.timestamp # Record closing time

            else: # Should be caught by _validate_trade_params
                self.logger.error(f"Invalid trade side: {params.side}", source_module=self._source_module)
                return Decimal(0), None

            # Update position timestamp
            if hasattr(position_model, "updated_at"):
                position_model.updated_at = params.timestamp

            try:
                # First, update the position in the database
                update_data = {
                    "quantity": position_model.quantity,
                    "entry_price": position_model.entry_price,
                    "current_price": position_model.current_price,
                    "realized_pnl": position_model.realized_pnl,
                    "unrealized_pnl": position_model.unrealized_pnl,
                    "is_active": position_model.is_active,
                }
                updated_pos = await self.position_repository.update(str(position_model.id), update_data)
                if not updated_pos:
                    self.logger.error(
                        f"Failed to update position {position_model.id} in DB.",
                        source_module=self._source_module,
                    )
                    return realized_pnl_trade, None

                # NEW: Link the order to this position for audit trail
                await self._link_order_to_position(params.order_id, str(position_model.id))

                self.logger.info(
                    "Updated position in DB - Pair: %s, Side: %s, Trade Qty: %s, Trade Price: %s, "
                    "New Pos Qty: %s, New AEP: %s, Trade PnL: %s, Linked Order: %s",
                    params.trading_pair, params.side, params.quantity, params.price,
                    updated_pos.quantity, updated_pos.entry_price, realized_pnl_trade, params.order_id,
                    source_module=self._source_module)
            except Exception:
                self.logger.exception(
                    f"Error updating position in DB for {params.trading_pair}: ",
                    source_module=self._source_module,
                )
                return realized_pnl_trade, None
            else:
                return realized_pnl_trade, updated_pos

    async def _link_order_to_position(self, order_id: str, position_id: str) -> None:
        """Link an order to a position for audit trail purposes.

        Args:
            order_id: The UUID of the order as string
            position_id: The UUID of the position as string
        """
        try:
            # Update the order to reference this position
            updated_order = await self.order_repository.update(
                order_id,
                {"position_id": position_id},
            )

            if updated_order:
                self.logger.debug(
                    f"Successfully linked order {order_id} to position {position_id}",
                    source_module=self._source_module)
            else:
                self.logger.warning(
                    f"Order {order_id} not found for position linking - may be external order",
                    source_module=self._source_module)
        except Exception:
            # Don't fail position update if order linking fails
            self.logger.exception(
                f"Failed to link order {order_id} to position {position_id}: ",
                source_module=self._source_module)

    # Error messages for trade validation (can remain the same)
    _INVALID_TRADE_SIDE_MSG = "Invalid trade side"
    _INVALID_QUANTITY_MSG = "Invalid quantity" # Can remain
    _INVALID_PRICE_MSG = "Invalid price" # Can remain

    def _validate_trade_params( # This method can largely remain the same
        self,
        _trading_pair: str,
        side: str,
        quantity: Decimal, # Already Decimal from _UpdatePositionParams
        price: Decimal,    # Already Decimal from _UpdatePositionParams
    ) -> None:
        """Validate trade parameters."""
        if side.upper() not in ("BUY", "SELL"): # Normalize side for comparison
            raise DataValidationError(self._INVALID_TRADE_SIDE_MSG)
        if quantity <= Decimal(0):
            raise DataValidationError(self._INVALID_QUANTITY_MSG)
        if price <= Decimal(0):
            raise DataValidationError(self._INVALID_PRICE_MSG)

    # _get_or_create_position needs to interact with the database now.
    async def _get_or_create_db_position(self, trading_pair: str, side_if_creating: str) -> PositionModel:
        """Get existing active position from DB or create a new one if it doesn't exist."""
        position = await self.position_repository.get_position_by_pair(trading_pair)
        if position:
            if not position.is_active: # Reactivating a closed position? Or new one?
                 self.logger.warning(
                    f"Position for {trading_pair} exists but is inactive. "
                    "Treating as new for this context.",
                    source_module=self._source_module,
                )
                 # This logic might need refinement: should we reactivate or error?
                 # For now, assume we create a new one if the existing is inactive.
                 # This implies `get_position_by_pair` should only return active ones, or this logic handles it.
                 # If `get_position_by_pair` already filters by is_active=True, then this `if` is for a case
                 # where an inactive one was somehow fetched, or if we want to prevent re-opening.
                 # Let's assume get_position_by_pair returns only active.
                 # So if `position` is None, we create.
                 # Position is active and found.
            return position

        # Position not found or inactive, create a new one
        self.logger.info(
            f"No active position found for {trading_pair}. Creating new one.",
            source_module=self._source_module,
        )

        # Asset splitting logic - assuming it's from config or a utility
        # base_asset, quote_asset = self.config_manager.split_trading_pair(trading_pair) # Example

        # For now, ID will be auto-generated by DB if not provided.
        # Side of the position is determined by the first trade that opens it.
        # Or, if it's a short, it's 'SHORT', if long, it's 'LONG'. This is simplified here.
        # The PositionModel has `side` (LONG/SHORT). A trade has `side` (BUY/SELL).
        # This needs careful mapping. If first trade is BUY, position is LONG. If SELL, position is SHORT.
        # The `side_if_creating` argument would be the intended position side (LONG/SHORT).

        new_pos_data = {
            "id": uuid.uuid4(), # Generate new UUID for the position
            "trading_pair": trading_pair,
            "side": side_if_creating.upper(), # "LONG" or "SHORT"
            "quantity": Decimal(0),
            "entry_price": Decimal(0),
            "current_price": Decimal(0), # Will be updated by market data
            "realized_pnl": Decimal(0),
            "unrealized_pnl": Decimal(0),
            "opened_at": datetime.now(UTC),
            "is_active": True,
        }
        try:
            created_position = await self.position_repository.create(new_pos_data)
            self.logger.info(
                f"Created new position in DB for {trading_pair} with ID {created_position.id}",
                source_module=self._source_module,
            )
        except Exception as e:
            self.logger.exception(
                f"Error creating new position in DB for {trading_pair}: ",
                source_module=self._source_module,
            )
            # Re-raise with additional context for proper error handling upstream
            raise PositionCreationError(f"Failed to create position for {trading_pair}: {e!s}") from e
        else:
            return created_position

    async def get_total_realized_pnl(self, trading_pair: str | None = None) -> Decimal:
        """Get total realized PnL for a specific pair or all pairs from database."""
        # Implementation uses repository pattern for data access
        # For large datasets, consider implementing an aggregate query method in the repository
        if trading_pair:
            position = await self.position_repository.get_position_by_pair(trading_pair)
            return position.realized_pnl if position and position.realized_pnl else Decimal(0)

        # For all pairs:
        all_positions = await self.position_repository.find_all(filters={"is_active": True}) # Or all, including closed
        return sum((pos.realized_pnl for pos in all_positions if pos.realized_pnl is not None), Decimal(0))

    # Internal data structures for trade record management
    # These are used to construct TradeInfo objects for position updates
    @dataclass
    class _TradeRecordParams: # Can remain for internal use if TradeInfo objects are constructed
        """Container for trade record parameters."""

        timestamp: datetime
        trade_id: str
        trading_pair: str
        side: str
        quantity: Decimal | float | str
        price: Decimal | float | str
        fee: Decimal | float | str = field(default_factory=Decimal)
        fee_currency: str | None = None
        commission: Decimal | float | str = field(default_factory=Decimal)
        commission_asset: str | None = None

        def __post_init__(self) -> None:
            """Validate the parameters after initialization."""
            if not isinstance(self.quantity, Decimal):
                self.quantity = Decimal(str(self.quantity))
            if not isinstance(self.price, Decimal):
                self.price = Decimal(str(self.price))
            if not isinstance(self.fee, Decimal):
                self.fee = Decimal(str(self.fee))
            if not isinstance(self.commission, Decimal):
                self.commission = Decimal(str(self.commission))

    class _TradeRecordKwargs(TypedDict, total=False):
        """Type[Any] definition for _create_trade_record kwargs."""

        timestamp: datetime
        trade_id: str
        trading_pair: str
        side: str
        quantity: Decimal | float | str
        price: Decimal | float | str
        fee: Decimal | float | str
        fee_currency: str | None
        commission: Decimal | float | str
        commission_asset: str | None

    def _create_trade_record(self, **kwargs: Unpack[_TradeRecordKwargs]) -> TradeInfo:
        """Create a trade record (legacy interface).

        This is a backward compatibility wrapper around _create_trade_record_from_params.
        New code should use _create_trade_record_from_params directly.

        Args:
            **kwargs: Parameters for the trade record. See _TradeRecordParams for details.

        Returns:
        -------
            TradeInfo: The created trade record
        """
        params = self._TradeRecordParams(**kwargs)
        return self._create_trade_record_from_params(params)

    def _create_trade_record_from_params(
        self,
        params: _TradeRecordParams) -> TradeInfo:
        """Create a TradeInfo object from TradeRecordParams.

        Args:
            params: TradeRecordParams containing all trade parameters

        Returns:
        -------
            TradeInfo: A new TradeInfo object with the provided parameters
        """
        # Ensure all numeric parameters are Decimal
        quantity = (
            params.quantity
            if isinstance(params.quantity, Decimal)
            else Decimal(str(params.quantity))
        )
        price = params.price if isinstance(params.price, Decimal) else Decimal(str(params.price))
        fee = params.fee if isinstance(params.fee, Decimal) else Decimal(str(params.fee))
        commission = (
            params.commission
            if isinstance(params.commission, Decimal)
            else Decimal(str(params.commission))
        )

        return TradeInfo(
            timestamp=params.timestamp,
            trade_id=params.trade_id,
            side=params.side,
            quantity=quantity,
            price=price,
            fee=fee,
            fee_currency=params.fee_currency,
            commission=commission,
            commission_asset=params.commission_asset)
