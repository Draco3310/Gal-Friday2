"""Position entity model."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from gal_friday.dal.base import BaseEntity


@dataclass
class PositionEntity(BaseEntity):
    """Database entity for positions."""

    position_id: str
    trading_pair: str
    side: str  # LONG/SHORT
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    opened_at: datetime
    closed_at: datetime | None
    is_active: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to database record."""
        return {
            "id": self.position_id,
            "trading_pair": self.trading_pair,
            "side": self.side,
            "quantity": float(self.quantity),
            "entry_price": float(self.entry_price),
            "current_price": float(self.current_price),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(self.unrealized_pnl),
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PositionEntity":
        """Create from database record."""
        return cls(
            position_id=str(data["id"]),
            trading_pair=data["trading_pair"],
            side=data["side"],
            quantity=Decimal(str(data["quantity"])),
            entry_price=Decimal(str(data["entry_price"])),
            current_price=Decimal(str(data["current_price"])),
            realized_pnl=Decimal(str(data["realized_pnl"])),
            unrealized_pnl=Decimal(str(data["unrealized_pnl"])),
            opened_at=data["opened_at"],
            closed_at=data["closed_at"],
            is_active=data["is_active"],
        )
