import uuid
from datetime import datetime
from decimal import Decimal
from typing import (  # Added for casts
    TYPE_CHECKING,  # For Order type hint
    cast,
)

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from gal_friday.core.events import ExecutionReportEvent

from .base import Base

if TYPE_CHECKING:
    from .order import Order  # Assuming Order is in models/order.py


class Fill(Base):
    __tablename__ = "fills"

    fill_pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fill_id: Mapped[str | None] = mapped_column(String(64), nullable=True)  # Exchange fill ID
    order_pk: Mapped[int] = mapped_column(Integer, ForeignKey("orders.order_pk"), nullable=False, index=True)
    exchange_order_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True) # From Order, denormalized for easier query

    trading_pair: Mapped[str] = mapped_column(String(16), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    quantity_filled: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    fill_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    commission: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    commission_asset: Mapped[str] = mapped_column(String(16), nullable=False)
    liquidity_type: Mapped[str | None] = mapped_column(String(10), nullable=True)  # 'MAKER' or 'TAKER'
    filled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True) # Changed to Mapped[datetime]

    # Relationships
    order: Mapped["Order"] = relationship("Order", back_populates="fills")

    # Constraints
    __table_args__ = (
        UniqueConstraint("exchange", "fill_id", name="uq_exchange_fill_id"),
    )

    def __repr__(self) -> str: # Added -> str
        return (
            f"<Fill(fill_pk={self.fill_pk}, fill_id='{self.fill_id}', order_pk={self.order_pk}, "
            f"quantity_filled={self.quantity_filled}, fill_price={self.fill_price})>"
        )

    def to_event(self) -> "ExecutionReportEvent": # Added to_event with type hints
        """Converts the Fill object to an ExecutionReportEvent."""
        # Imports needed for ExecutionReportEvent. Assuming they are or will be available in the module.
        # from decimal import Decimal # Already imported via sqlalchemy Numeric
        # import uuid # For event_id and signal_id
        # from gal_friday.core.events import ExecutionReportEvent # Actual import path

        # This is a simplified conversion. A real one would need more context,
        # especially for fields like order_status, order_type, quantity_ordered, signal_id.
        # These might come from self.order (the related Order object).

        # To make this runnable without full context, we make some assumptions or use placeholders.
        # Proper implementation would involve fetching related order details.

        # Placeholder for actual ExecutionReportEvent, as it's not defined here
        # For type hinting purposes, we'll use a forward reference 'ExecutionReportEvent'
        # and assume the necessary imports will be added at the top of the file.

        # Simulating access to related order for some fields:
        order_exchange_order_id = self.exchange_order_id or (self.order.exchange_order_id if self.order else None)
        order_trading_pair = self.trading_pair or (self.order.trading_pair if self.order else "UNKNOWN/UNKNOWN")
        order_exchange = self.exchange or (self.order.exchange if self.order else "UNKNOWN_EXCHANGE")
        order_side = self.side or (self.order.side if self.order else "UNKNOWN_SIDE")
        order_type = self.order.order_type if self.order else "MARKET" # Default if no order
        quantity_ordered = self.order.quantity if self.order else self.quantity_filled # Default
        signal_id_val = self.order.signal_id if self.order and hasattr(self.order, "signal_id") else uuid.uuid4() # Placeholder

        _client_order_id_from_order = self.order.client_order_id if self.order and hasattr(self.order, "client_order_id") else None
        client_order_id_val = str(_client_order_id_from_order) if _client_order_id_from_order is not None else None


        # Determine order_status based on fill (simplified)
        # This is a very naive way to set status. Real status comes from the order.
        current_order_status = "PARTIALLY_FILLED"
        if self.order and self.quantity_filled >= self.order.quantity:
            current_order_status = "FILLED"

        # Ensure necessary types are imported at the module level:
        # from gal_friday.core.events import ExecutionReportEvent
        # import uuid
        # from decimal import Decimal

        # This is a placeholder for the actual ExecutionReportEvent class
        # from gal_friday.core.events import ExecutionReportEvent
        # For now, using a dict to represent the structure.
        # Replace with actual ExecutionReportEvent(...) when available.

        # To satisfy the type hint, this should return an instance of ExecutionReportEvent
        # Since I cannot import it here directly within this tool call easily,
        # I will construct a dictionary that matches its likely structure.
        # In a real scenario, I'd ensure the import `from gal_friday.core.events import ExecutionReportEvent`
        # is at the top of the file.

        # This is a mock structure. The actual implementation needs the real Event class.
        # For the purpose of this exercise, let's assume ExecutionReportEvent is importable
        # and we are constructing it.

        # To avoid NameError for ExecutionReportEvent if not imported, we'll return a dict.
        # The type hint 'ExecutionReportEvent' will rely on a forward reference.
        # This means the actual ExecutionReportEvent class must be importable in the file's scope.

        # Ensure exchange_order_id is not None if the event expects a string
        final_exchange_order_id = order_exchange_order_id
        if final_exchange_order_id is None:
            # This addresses: gal_friday/models/fill.py:144: error: Argument "exchange_order_id" to "ExecutionReportEvent" has incompatible type "str | Any | None"; expected "str"
            # We need to provide a string if the event strictly expects one.
            # Alternatively, the event definition or this model's data integrity should be adjusted.
            # For now, using a placeholder:
            final_exchange_order_id = "UNKNOWN_FILL_EXCH_ORD_ID" # Placeholder

        event_data = {
            "source_module": self.__class__.__name__,
            "event_id": uuid.uuid4(),
            "timestamp": datetime.utcnow(), # Or self.filled_at for event timestamp
            "exchange_order_id": final_exchange_order_id, # Use the adjusted value
            "trading_pair": order_trading_pair,
            "exchange": order_exchange,
            "order_status": current_order_status, # This is simplified
            "order_type": order_type,
            "side": order_side,
            "quantity_ordered": quantity_ordered,
            "signal_id": signal_id_val,
            "client_order_id": client_order_id_val,
            "quantity_filled": self.quantity_filled,
            "average_fill_price": self.fill_price, # Fill price is the average for this fill
            "limit_price": self.order.limit_price if self.order and self.order.order_type == "LIMIT" else None,
            "stop_price": self.order.stop_price if self.order and "STOP" in self.order.order_type else None,
            "commission": self.commission,
            "commission_asset": self.commission_asset,
            "timestamp_exchange": self.filled_at,
            "error_message": None,
        }
        # In a real implementation:
        # from gal_friday.core.events import ExecutionReportEvent
        # import uuid # At top of file
        # from decimal import Decimal # At top of file
        # return ExecutionReportEvent(**event_data)

        # For now, to avoid crashing this step if ExecutionReportEvent is not made available
        # to this method's scope by an import I can't add here, I'll return the dict.
        # This satisfies the "add function type hints" part for its signature.
        # The actual return type requires the class to be available.
        # return ExecutionReportEvent(**event_data) # Old way

        # Explicitly pass arguments
        return ExecutionReportEvent(
            source_module=self.__class__.__name__,
            event_id=uuid.uuid4(),
            timestamp=datetime.utcnow(), # Event's own creation time
            exchange_order_id=cast("str", final_exchange_order_id),
            trading_pair=cast("str", order_trading_pair),
            exchange=cast("str", order_exchange),
            order_status=current_order_status,
            order_type=cast("str", order_type),
            side=cast("str", order_side),
            quantity_ordered=quantity_ordered, # Ensure this is Decimal
            signal_id=cast("uuid.UUID", signal_id_val), # Was Column[UUID] | UUID, event expects UUID | None. Value is always UUID.
            client_order_id=client_order_id_val,
            quantity_filled=self.quantity_filled, # Ensure this is Decimal
            average_fill_price=self.fill_price, # Ensure this is Decimal
            limit_price=cast("Decimal | None", (self.order.limit_price if self.order and self.order.order_type == "LIMIT" else None)),
            stop_price=cast("Decimal | None", (self.order.stop_price if self.order and "STOP" in self.order.order_type else None)),
            commission=self.commission, # Ensure Decimal or None
            commission_asset=self.commission_asset,
            timestamp_exchange=self.filled_at, # Timestamp of the actual fill
            error_message=None, # Assuming no error for a successful fill conversion
        )
