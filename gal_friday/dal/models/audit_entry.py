"""SQLAlchemy model for audit entries."""

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, TIMESTAMP, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from gal_friday.dal.models import Base


class AuditEntry(Base):
    """Model for audit trail entries."""

    __tablename__ = "audit_entries"

    # Primary key
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)

    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, index=True)

    # Order information
    order_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)

    # Execution details
    filled_quantity: Mapped[Decimal | None] = mapped_column(nullable=True)
    average_price: Mapped[Decimal | None] = mapped_column(nullable=True)
    commission: Mapped[Decimal | None] = mapped_column(nullable=True)
    realized_pnl: Mapped[Decimal | None] = mapped_column(nullable=True)

    # Risk metrics
    consecutive_losses: Mapped[int] = mapped_column(nullable=False, default=0)
    risk_events: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    risk_metrics_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    # Metadata
    service: Mapped[str] = mapped_column(String(50), nullable=False, default="risk_manager")
    environment: Mapped[str] = mapped_column(String(50), nullable=False, default="production")
    risk_manager_version: Mapped[str] = mapped_column(String(20), nullable=False, default="1.0.0")
    audit_schema_version: Mapped[str] = mapped_column(String(20), nullable=False, default="1.0")
    instance_id: Mapped[str] = mapped_column(String(100), nullable=False, default="unknown")

    # Indexes are defined above on individual columns

    def __repr__(self) -> str:
        """String representation of the audit entry."""
        return (
            f"<AuditEntry("
            f"id={self.id}, "
            f"order_id={self.order_id}, "
            f"symbol={self.symbol}, "
            f"side={self.side}, "
            f"status={self.status}, "
            f"timestamp={self.timestamp}"
            f")>"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the audit entry to a dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "status": self.status,
            "filled_quantity": float(self.filled_quantity) if self.filled_quantity else None,
            "average_price": float(self.average_price) if self.average_price else None,
            "commission": float(self.commission) if self.commission else None,
            "realized_pnl": float(self.realized_pnl) if self.realized_pnl else None,
            "consecutive_losses": self.consecutive_losses,
            "risk_events": self.risk_events,
            "risk_metrics_snapshot": self.risk_metrics_snapshot,
            "service": self.service,
            "environment": self.environment,
            "risk_manager_version": self.risk_manager_version,
            "audit_schema_version": self.audit_schema_version,
            "instance_id": self.instance_id,
        }
