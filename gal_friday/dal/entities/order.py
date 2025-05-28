"""Order entity model."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from gal_friday.dal.base import BaseEntity


@dataclass
class OrderEntity(BaseEntity):
    """Database entity for orders."""

    order_id: str
    signal_id: str
    trading_pair: str
    exchange: str
    side: str  # BUY/SELL
    order_type: str  # MARKET/LIMIT
    quantity: Decimal
    limit_price: Decimal | None
    status: str
    exchange_order_id: str | None
    created_at: datetime
    updated_at: datetime | None
    filled_quantity: Decimal = Decimal("0")
    average_fill_price: Decimal | None = None
    commission: Decimal | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to database record."""
        return {
            "id": self.order_id,
            "signal_id": self.signal_id,
            "trading_pair": self.trading_pair,
            "exchange": self.exchange,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": float(self.quantity),
            "limit_price": float(self.limit_price) if self.limit_price else None,
            "status": self.status,
            "exchange_order_id": self.exchange_order_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "filled_quantity": float(self.filled_quantity),
            "average_fill_price": (
                float(self.average_fill_price)
                if self.average_fill_price else None
            ),
            "commission": (
                float(self.commission)
                if self.commission else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrderEntity":
        """Create from database record."""
        return cls(
            order_id=str(data["id"]),
            signal_id=str(data["signal_id"]),
            trading_pair=data["trading_pair"],
            exchange=data["exchange"],
            side=data["side"],
            order_type=data["order_type"],
            quantity=Decimal(str(data["quantity"])),
            limit_price=Decimal(str(data["limit_price"])) if data["limit_price"] else None,
            status=data["status"],
            exchange_order_id=data["exchange_order_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            filled_quantity=Decimal(str(data["filled_quantity"])),
            average_fill_price=(
                Decimal(str(data["average_fill_price"]))
                if data["average_fill_price"] else None
            ),
            commission=(
                Decimal(str(data["commission"]))
                if data["commission"] else None
            ),
        )
