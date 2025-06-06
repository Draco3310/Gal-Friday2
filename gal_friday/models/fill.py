import uuid
import logging
from datetime import datetime
from decimal import Decimal
from typing import (
    TYPE_CHECKING,
    Dict,
    Optional,
    Any,
    cast,
    ClassVar,
)

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from gal_friday.core.events import ExecutionReportEvent, ExecutionReportParams

from .base import Base

if TYPE_CHECKING:
    from .order import Order


class ExecutionEventBuilder:
    """Enterprise-grade builder for creating execution report events from fill data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_execution_event(self, fill: 'Fill') -> ExecutionReportEvent:
        """
        Create actual ExecutionReportEvent from fill data.
        Replaces placeholder logic with complete event construction.
        
        Args:
            fill: Fill instance containing execution data
            
        Returns:
            ExecutionReportEvent: Fully constructed and validated event
            
        Raises:
            ValueError: If fill data is invalid or incomplete
            AttributeError: If required fill attributes are missing
        """
        try:
            # Validate fill data before processing
            self._validate_fill_data(fill)
            
            # Extract core execution information with proper fallbacks
            params = ExecutionReportParams(
                source_module=fill.__class__.__name__,
                exchange_order_id=self._get_exchange_order_id(fill),
                trading_pair=fill.trading_pair,
                exchange=fill.exchange,
                order_status=self._determine_order_status(fill),
                order_type=self._get_order_type(fill),
                side=fill.side.upper(),
                quantity_ordered=self._get_quantity_ordered(fill),
                signal_id=self._get_signal_id(fill),
                client_order_id=self._get_client_order_id(fill),
                quantity_filled=fill.quantity_filled,
                average_fill_price=fill.fill_price,
                limit_price=self._get_limit_price(fill),
                stop_price=self._get_stop_price(fill),
                commission=fill.commission,
                commission_asset=fill.commission_asset,
                timestamp_exchange=fill.filled_at,
                error_message=None  # No error for successful fill
            )
            
            # Create and validate event using factory method
            event = ExecutionReportEvent.create(params)
            
            self.logger.info(
                f"Created execution event for fill {fill.fill_id}: "
                f"{event.quantity_filled} {event.trading_pair} @ {event.average_fill_price}"
            )
            
            return event
            
        except Exception as e:
            self.logger.error(f"Error creating execution event for fill {getattr(fill, 'fill_id', 'unknown')}: {e}")
            raise
    
    def _validate_fill_data(self, fill: 'Fill') -> None:
        """Validate fill data before event creation."""
        if not fill:
            raise ValueError("Fill data cannot be None")
        
        if not fill.trading_pair:
            raise ValueError("Fill must have a trading pair")
        
        if not fill.exchange:
            raise ValueError("Fill must have an exchange")
        
        if not fill.side:
            raise ValueError("Fill must have a side (BUY/SELL)")
        
        if fill.quantity_filled <= 0:
            raise ValueError(f"Fill quantity must be positive, got: {fill.quantity_filled}")
        
        if fill.fill_price <= 0:
            raise ValueError(f"Fill price must be positive, got: {fill.fill_price}")
        
        if not fill.commission_asset:
            raise ValueError("Fill must have a commission asset")
    
    def _get_exchange_order_id(self, fill: 'Fill') -> str:
        """Extract exchange order ID with proper fallback logic."""
        # First try the denormalized exchange_order_id on the fill
        if fill.exchange_order_id:
            return str(fill.exchange_order_id)
        
        # Then try the related order's exchange_order_id
        if fill.order and fill.order.exchange_order_id:
            return str(fill.order.exchange_order_id)
        
        # Fallback to client order ID if available
        if fill.order and fill.order.client_order_id:
            return f"client_{fill.order.client_order_id}"
        
        # Final fallback - this should rarely happen in production
        return f"fill_{fill.fill_pk}"
    
    def _determine_order_status(self, fill: 'Fill') -> str:
        """Determine order status based on fill and order data."""
        if not fill.order:
            # If we don't have order context, assume filled
            return "FILLED"
        
        # Use the order's current status if available
        if hasattr(fill.order, 'status') and fill.order.status:
            return fill.order.status.upper()
        
        # Calculate status based on fill vs order quantities
        if hasattr(fill.order, 'quantity_ordered'):
            # Get cumulative filled quantity for this order
            total_filled = self._calculate_cumulative_filled_quantity(fill)
            
            if total_filled >= fill.order.quantity_ordered:
                return "FILLED"
            elif total_filled > 0:
                return "PARTIALLY_FILLED"
        
        return "FILLED"  # Default for individual fill
    
    def _get_order_type(self, fill: 'Fill') -> str:
        """Get order type from related order data."""
        if fill.order and hasattr(fill.order, 'order_type') and fill.order.order_type:
            return fill.order.order_type.upper()
        
        # Infer from liquidity type if available
        if fill.liquidity_type:
            # MAKER orders are typically LIMIT, TAKER could be MARKET
            return "LIMIT" if fill.liquidity_type == "MAKER" else "MARKET"
        
        return "MARKET"  # Conservative default
    
    def _get_quantity_ordered(self, fill: 'Fill') -> Decimal:
        """Get the original ordered quantity."""
        if fill.order and hasattr(fill.order, 'quantity_ordered'):
            return fill.order.quantity_ordered
        
        # If no order context, use the fill quantity as a fallback
        return fill.quantity_filled
    
    def _get_signal_id(self, fill: 'Fill') -> Optional[uuid.UUID]:
        """Extract signal ID from related order."""
        if fill.order and hasattr(fill.order, 'signal_id') and fill.order.signal_id:
            if isinstance(fill.order.signal_id, uuid.UUID):
                return fill.order.signal_id
            # Try to convert string to UUID
            try:
                return uuid.UUID(str(fill.order.signal_id))
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid signal_id format: {fill.order.signal_id}")
        
        return None
    
    def _get_client_order_id(self, fill: 'Fill') -> Optional[str]:
        """Get client order ID from related order."""
        if fill.order and hasattr(fill.order, 'client_order_id') and fill.order.client_order_id:
            return str(fill.order.client_order_id)
        
        return None
    
    def _get_limit_price(self, fill: 'Fill') -> Optional[Decimal]:
        """Get limit price from related order if it's a limit order."""
        if (fill.order and 
            hasattr(fill.order, 'limit_price') and 
            fill.order.limit_price and
            self._get_order_type(fill) in ["LIMIT", "STOP_LIMIT"]):
            return fill.order.limit_price
        
        return None
    
    def _get_stop_price(self, fill: 'Fill') -> Optional[Decimal]:
        """Get stop price from related order if it's a stop order."""
        if (fill.order and 
            hasattr(fill.order, 'stop_price') and 
            fill.order.stop_price and
            "STOP" in self._get_order_type(fill)):
            return fill.order.stop_price
        
        return None
    
    def _calculate_cumulative_filled_quantity(self, fill: 'Fill') -> Decimal:
        """Calculate cumulative filled quantity for the order."""
        if not fill.order:
            return fill.quantity_filled
        
        # Sum all fills for this order
        total_filled = Decimal('0')
        if hasattr(fill.order, 'fills') and fill.order.fills:
            for order_fill in fill.order.fills:
                total_filled += order_fill.quantity_filled
        else:
            # If we can't access other fills, use this fill's quantity
            total_filled = fill.quantity_filled
        
        return total_filled


class ExecutionEventPublisher:
    """Publisher for execution report events with monitoring and error handling."""
    
    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        
        # Event tracking for monitoring
        self.published_events = 0
        self.failed_publications = 0
        self.last_publication_time: Optional[datetime] = None
    
    async def publish_execution_event(self, event: ExecutionReportEvent) -> bool:
        """
        Publish execution event to event bus with comprehensive error handling.
        
        Args:
            event: ExecutionReportEvent to publish
            
        Returns:
            bool: True if published successfully, False otherwise
        """
        try:
            if not self.event_bus:
                self.logger.warning("No event bus configured for execution events")
                return False
            
            # Publish to execution reports topic
            await self.event_bus.publish('execution_reports', event)
            
            # Update statistics
            self.published_events += 1
            self.last_publication_time = datetime.utcnow()
            
            self.logger.info(
                f"Published execution event {event.event_id} for order {event.exchange_order_id}"
            )
            
            return True
            
        except Exception as e:
            self.failed_publications += 1
            self.logger.error(
                f"Failed to publish execution event {getattr(event, 'event_id', 'unknown')}: {e}",
                exc_info=True
            )
            return False
    
    def get_publication_stats(self) -> Dict[str, Any]:
        """Get publication statistics for monitoring."""
        return {
            'published_events': self.published_events,
            'failed_publications': self.failed_publications,
            'success_rate': (
                self.published_events / (self.published_events + self.failed_publications)
                if (self.published_events + self.failed_publications) > 0 else 0.0
            ),
            'last_publication_time': self.last_publication_time.isoformat() if self.last_publication_time else None
        }


class Fill(Base):
    """Enhanced Fill model with enterprise-grade execution event creation."""
    
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
    filled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    # Relationships
    order: Mapped["Order"] = relationship("Order", back_populates="fills")

    # Constraints
    __table_args__ = (
        UniqueConstraint("exchange", "fill_id", name="uq_exchange_fill_id"),
    )

    # Class-level builders (can be dependency-injected in production)
    _event_builder: ClassVar[Optional[ExecutionEventBuilder]] = None
    _event_publisher: ClassVar[Optional[ExecutionEventPublisher]] = None

    def __repr__(self) -> str:
        return (
            f"<Fill(fill_pk={self.fill_pk}, fill_id='{self.fill_id}', order_pk={self.order_pk}, "
            f"quantity_filled={self.quantity_filled}, fill_price={self.fill_price})>"
        )

    @classmethod
    def set_event_builder(cls, builder: ExecutionEventBuilder) -> None:
        """Set the event builder (for dependency injection)."""
        cls._event_builder = builder

    @classmethod
    def set_event_publisher(cls, publisher: ExecutionEventPublisher) -> None:
        """Set the event publisher (for dependency injection)."""
        cls._event_publisher = publisher

    def to_event(self) -> ExecutionReportEvent:
        """
        Create ExecutionReportEvent from fill data using enterprise-grade event construction.
        
        This method replaces the previous placeholder execution event logic with
        comprehensive event creation, validation, and error handling.
        
        Returns:
            ExecutionReportEvent: Fully constructed and validated execution event
            
        Raises:
            ValueError: If fill data is invalid
            RuntimeError: If event creation fails
        """
        # Use class-level builder or create new one
        builder = self._event_builder or ExecutionEventBuilder()
        
        try:
            return builder.create_execution_event(self)
        except Exception as e:
            # Log the error but still raise it for proper error handling upstream
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to create execution event for fill {self.fill_id}: {e}")
            raise RuntimeError(f"Execution event creation failed: {e}") from e

    async def publish_execution_event(self) -> bool:
        """
        Create and publish execution event with comprehensive error handling.
        
        Returns:
            bool: True if event was created and published successfully
        """
        try:
            # Create the event
            event = self.to_event()
            
            # Use class-level publisher or create new one
            publisher = self._event_publisher or ExecutionEventPublisher()
            
            # Publish the event
            return await publisher.publish_execution_event(event)
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to publish execution event for fill {self.fill_id}: {e}")
            return False

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of this fill's execution details for monitoring and reporting.
        
        Returns:
            dict: Summary of execution details
        """
        return {
            'fill_id': self.fill_id,
            'order_pk': self.order_pk,
            'exchange_order_id': self.exchange_order_id,
            'trading_pair': self.trading_pair,
            'exchange': self.exchange,
            'side': self.side,
            'quantity_filled': float(self.quantity_filled),
            'fill_price': float(self.fill_price),
            'commission': float(self.commission),
            'commission_asset': self.commission_asset,
            'liquidity_type': self.liquidity_type,
            'filled_at': self.filled_at.isoformat(),
            'gross_value': float(self.quantity_filled * self.fill_price),
            'net_value': float(self.quantity_filled * self.fill_price - self.commission)
        }
