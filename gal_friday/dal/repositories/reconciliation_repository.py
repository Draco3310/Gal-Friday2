"""Repository for reconciliation data persistence using SQLAlchemy."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any
import uuid

from pydantic import (
    BaseModel,
    Field,
    ValidationError as PydanticValidationError,
    ValidationInfo,
    field_validator,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gal_friday.dal.base import BaseRepository
from gal_friday.dal.models.position_adjustment import PositionAdjustment
from gal_friday.dal.models.reconciliation_event import ReconciliationEvent

# ReconciliationReport and ReconciliationStatus would now likely be service-layer or domain models, not directly handled by repo.


if TYPE_CHECKING:
    from gal_friday.logger_service import LoggerService


class ReconciliationValidationError(Exception):
    """Custom exception for reconciliation data validation errors."""

    def __init__(self, message: str, field_path: str | None = None,
                 validation_errors: dict[str, Any] | None = None) -> None:
        """Initialize reconciliation validation error.

        Args:
            message: Error message
            field_path: Field that caused the validation error
            validation_errors: Detailed validation errors from Pydantic
        """
        super().__init__(message)
        self.field_path = field_path
        self.validation_errors = validation_errors or {}


class ReconciliationStatus(str, Enum):
    """Valid reconciliation event statuses."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class ReconciliationType(str, Enum):
    """Valid reconciliation types."""
    POSITION = "position"
    TRADE = "trade"
    BALANCE = "balance"
    SETTLEMENT = "settlement"
    CORPORATE_ACTIONS = "corporate_actions"
    FEES = "fees"


class AdjustmentType(str, Enum):
    """Valid position adjustment types."""
    CORRECTION = "correction"
    REBALANCE = "rebalance"
    SPLIT = "split"
    DIVIDEND = "dividend"
    FEES = "fees"
    MERGER = "merger"
    SPINOFF = "spinoff"
    RIGHTS = "rights"


class ReconciliationEventSchema(BaseModel):
    """Schema for reconciliation event data with comprehensive validation."""

    # Required fields
    reconciliation_id: uuid.UUID | None = Field(
        default=None,
        description="Unique identifier for the reconciliation event",
    )
    timestamp: datetime = Field(
        description="When the reconciliation event occurred",
    )
    reconciliation_type: ReconciliationType = Field(
        description="Type[Any] of reconciliation being performed",
    )
    status: ReconciliationStatus = Field(
        description="Current status of the reconciliation",
    )
    report: dict[str, Any] = Field(
        description="Detailed reconciliation report data",
    )

    # Optional fields with defaults
    discrepancies_found: int = Field(
        default=0,
        ge=0,
        description="Number of discrepancies found during reconciliation",
    )
    auto_corrected: int = Field(
        default=0,
        ge=0,
        description="Number of discrepancies automatically corrected",
    )
    manual_review_required: int = Field(
        default=0,
        ge=0,
        description="Number of items requiring manual review",
    )
    duration_seconds: Decimal | None = Field(
        default=None,
        ge=0,
        description="Duration of reconciliation process in seconds",
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        """Validate timestamp is not in future and has timezone info."""
        if v.tzinfo is None:
            v = v.replace(tzinfo=UTC)

        # Allow some tolerance for future timestamps (5 minutes)
        max_future = datetime.now(UTC) + timedelta(minutes=5)
        if v > max_future:
            raise ValueError(f"Timestamp cannot be more than 5 minutes in the future: {v}")

        return v

    @field_validator("report")
    @classmethod
    def validate_report(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate report contains required structure."""
        if not isinstance(v, dict):
            raise ValueError("Report must be a dictionary")

        # Ensure basic report structure
        required_keys = ["summary", "timestamp"]
        for key in required_keys:
            if key not in v:
                v[key] = {}

        return v

    @field_validator("auto_corrected")
    @classmethod
    def validate_auto_corrected(cls, v: int, info: ValidationInfo) -> int:
        """Validate auto_corrected count doesn't exceed discrepancies_found."""
        if info.data and "discrepancies_found" in info.data:
            discrepancies = info.data["discrepancies_found"]
            if v > discrepancies:
                raise ValueError(
                    f"Auto-corrected count ({v}) cannot exceed discrepancies found ({discrepancies})",
                )
        return v

    @field_validator("manual_review_required")
    @classmethod
    def validate_manual_review(cls, v: int, info: ValidationInfo) -> int:
        """Validate manual review count is consistent with other counts."""
        if info.data and "discrepancies_found" in info.data and "auto_corrected" in info.data:
            discrepancies = info.data["discrepancies_found"]
            auto_corrected = info.data["auto_corrected"]
            remaining = discrepancies - auto_corrected
            if v > remaining:
                raise ValueError(
                    f"Manual review required ({v}) cannot exceed remaining discrepancies ({remaining})",
                )
        return v


class PositionAdjustmentSchema(BaseModel):
    """Schema for position adjustment data with comprehensive validation."""

    # Required fields
    reconciliation_id: uuid.UUID = Field(
        description="Reference to the reconciliation event",
    )
    trading_pair: str = Field(
        min_length=3,
        max_length=20,
        description="Trading pair symbol (e.g., BTC/USD)",
    )
    adjustment_type: AdjustmentType = Field(
        description="Type[Any] of position adjustment",
    )
    reason: str = Field(
        min_length=10,
        max_length=1000,
        description="Detailed reason for the adjustment",
    )
    authorized_by: str = Field(
        min_length=1,
        max_length=100,
        description="User or system that authorized the adjustment",
    )

    # Value fields (at least one must be provided)
    old_value: Decimal | None = Field(
        default=None,
        description="Previous position value",
    )
    new_value: Decimal | None = Field(
        default=None,
        description="New position value after adjustment",
    )
    quantity_change: Decimal | None = Field(
        default=None,
        description="Net change in position quantity",
    )

    # Optional fields
    price_adjustment: Decimal | None = Field(
        default=None,
        description="Price adjustment if applicable",
    )
    reference_id: str | None = Field(
        default=None,
        max_length=50,
        description="External reference ID",
    )
    notes: str | None = Field(
        default=None,
        max_length=2000,
        description="Additional notes about the adjustment",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict[str, Any],
        description="Additional metadata for the adjustment",
    )

    @field_validator("trading_pair")
    @classmethod
    def validate_trading_pair(cls, v: str) -> str:
        """Validate trading pair format."""
        v = v.strip().upper()
        if not v:
            raise ValueError("Trading pair cannot be empty")

        # Basic format validation (can be enhanced with specific patterns)
        if "/" not in v and "-" not in v:
            raise ValueError("Trading pair must contain separator (/ or -)")

        return v

    @field_validator("quantity_change")
    @classmethod
    def validate_quantity_change(cls, v: Decimal | None) -> Decimal | None:
        """Validate quantity change is not zero."""
        if v is not None and v == 0:
            raise ValueError("Quantity change cannot be zero")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Post-init validation to check business rules."""
        # At least one value field must be provided
        value_fields = [self.old_value, self.new_value, self.quantity_change]
        if all(v is None for v in value_fields):
            raise ValueError("At least one of old_value, new_value, or quantity_change must be provided")

        # Business rule: certain adjustment types require specific fields
        if self.adjustment_type in [AdjustmentType.SPLIT, AdjustmentType.DIVIDEND]:
            if self.old_value is None or self.new_value is None:
                raise ValueError(f"{self.adjustment_type.value} adjustments require both old_value and new_value")


class ReconciliationRepository(BaseRepository[ReconciliationEvent]):
    """Repository for ReconciliationEvent data persistence using SQLAlchemy."""

    def __init__(
        self, session_maker: async_sessionmaker[AsyncSession], logger: "LoggerService") -> None:
        """Initialize the reconciliation repository.

        Args:
            session_maker: SQLAlchemy async_sessionmaker for creating sessions.
            logger: Logger service instance.
        """
        super().__init__(session_maker, ReconciliationEvent, logger)

    def _validate_reconciliation_data(self, event_data: dict[str, Any]) -> ReconciliationEventSchema:
        """Validate incoming reconciliation event data.

        Args:
            event_data: Raw reconciliation event data

        Returns:
            Validated reconciliation event schema

        Raises:
            ReconciliationValidationError: If validation fails
        """
        try:
            # Generate ID if not provided
            if "reconciliation_id" not in event_data or event_data["reconciliation_id"] is None:
                event_data["reconciliation_id"] = uuid.uuid4()

            validated_data = ReconciliationEventSchema(**event_data)

            self.logger.debug(
                f"Reconciliation event data validated successfully for ID {validated_data.reconciliation_id}",
                source_module=self._source_module,
            )

            return validated_data

        except PydanticValidationError as e:
            error_details = {}
            for error in e.errors():
                field_path = ".".join(str(loc) for loc in error["loc"])
                error_details[field_path] = {
                    "message": error["msg"],
                    "type": error["type"],
                    "input": error.get("input"),
                }

            self.logger.exception(
                f"Reconciliation event validation failed: {error_details}",
                source_module=self._source_module,
            )

            raise ReconciliationValidationError(
                f"Invalid reconciliation event data: {e!s}",
                validation_errors=error_details,
            ) from e

    def _validate_position_adjustment_data(self, adjustment_data: dict[str, Any]) -> PositionAdjustmentSchema:
        """Validate incoming position adjustment data.

        Args:
            adjustment_data: Raw position adjustment data

        Returns:
            Validated position adjustment schema

        Raises:
            ReconciliationValidationError: If validation fails
        """
        try:
            validated_data = PositionAdjustmentSchema(**adjustment_data)

            self.logger.debug(
                f"Position adjustment data validated successfully for reconciliation {validated_data.reconciliation_id}",
                source_module=self._source_module,
            )

            return validated_data

        except PydanticValidationError as e:
            error_details = {}
            for error in e.errors():
                field_path = ".".join(str(loc) for loc in error["loc"])
                error_details[field_path] = {
                    "message": error["msg"],
                    "type": error["type"],
                    "input": error.get("input"),
                }

            self.logger.exception(
                f"Position adjustment validation failed: {error_details}",
                source_module=self._source_module,
            )

            raise ReconciliationValidationError(
                f"Invalid position adjustment data: {e!s}",
                validation_errors=error_details,
            ) from e

    async def _create_audit_trail(self, operation: str, entity_type: str,
                                entity_id: str, details: dict[str, Any]) -> None:
        """Create audit trail entry for reconciliation operations.

        Args:
            operation: Operation performed (create, update, delete)
            entity_type: Type[Any] of entity (reconciliation_event, position_adjustment)
            entity_id: ID of the entity
            details: Additional details about the operation
        """
        audit_entry = {
            "timestamp": datetime.now(UTC),
            "operation": operation,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "details": details,
            "source_module": self._source_module,
        }

        self.logger.info(
            f"Audit trail: {operation} {entity_type} {entity_id} - {audit_entry}",
            source_module=self._source_module,
        )

    async def save_reconciliation_event(
        self, event_data: dict[str, Any]) -> ReconciliationEvent:
        """Saves a reconciliation event with comprehensive validation.

        Args:
            event_data: Dictionary containing reconciliation event data

        Returns:
            Created ReconciliationEvent instance

        Raises:
            ReconciliationValidationError: If validation fails
            ValueError: If database operation fails
        """
        # Validate incoming data using comprehensive schema
        validated_data = self._validate_reconciliation_data(event_data)

        try:
            # Convert Pydantic model to dict[str, Any] for database creation
            db_data = validated_data.model_dump(exclude_none=True)

            # Create the reconciliation event
            reconciliation_event = await self.create(db_data)

            # Create audit trail entry
            await self._create_audit_trail(
                operation="create",
                entity_type="reconciliation_event",
                entity_id=str(reconciliation_event.reconciliation_id),
                details={
                    "reconciliation_type": validated_data.reconciliation_type.value,
                    "status": validated_data.status.value,
                    "discrepancies_found": validated_data.discrepancies_found,
                },
            )

            self.logger.info(
                f"Successfully saved reconciliation event {reconciliation_event.reconciliation_id}",
                source_module=self._source_module,
            )

            return reconciliation_event

        except Exception as e:
            self.logger.exception(
                f"Failed to save reconciliation event: {e!s}",
                source_module=self._source_module,
            )
            raise ValueError(f"Database operation failed: {e!s}") from e

    async def get_reconciliation_event(
        self, reconciliation_id: uuid.UUID) -> ReconciliationEvent | None:
        """Get a specific reconciliation event by its ID."""
        return await self.get_by_id(reconciliation_id)

    async def get_recent_reconciliation_events(
        self, days: int = 7, status: str | None = None) -> Sequence[ReconciliationEvent]:
        """Get reconciliation events from the last N days, optionally filtered by status."""
        cutoff_date = datetime.now(UTC) - timedelta(days=days)

        # Validate status if provided
        if status:
            try:
                ReconciliationStatus(status)
            except ValueError:
                valid_statuses = [s.value for s in ReconciliationStatus]
                raise ValueError(f"Invalid status '{status}'. Valid options: {valid_statuses}")

        async with self.session_maker() as session:
            stmt = select(ReconciliationEvent).where(ReconciliationEvent.timestamp > cutoff_date)
            if status:
                stmt = stmt.where(ReconciliationEvent.status == status)
            stmt = stmt.order_by(ReconciliationEvent.timestamp.desc())

            result = await session.execute(stmt)
            events = result.scalars().all()
            self.logger.debug(f"Found {len(events)} reconciliation events from last {days} days.", source_module=self._source_module)
            return events

    async def save_position_adjustment(
        self, adjustment_data: dict[str, Any]) -> PositionAdjustment:
        """Saves a position adjustment with comprehensive validation and audit trail.

        Args:
            adjustment_data: Dictionary containing position adjustment data

        Returns:
            Created PositionAdjustment instance

        Raises:
            ReconciliationValidationError: If validation fails
            ValueError: If database operation fails or reconciliation_id doesn't exist
        """
        # Validate incoming data using comprehensive schema
        validated_data = self._validate_position_adjustment_data(adjustment_data)

        # Verify reconciliation event exists
        reconciliation_event = await self.get_reconciliation_event(validated_data.reconciliation_id)
        if reconciliation_event is None:
            raise ValueError(
                f"Reconciliation event {validated_data.reconciliation_id} does not exist",
            )

        try:
            # Convert Pydantic model to dict[str, Any] for database creation
            db_data = validated_data.model_dump(exclude_none=True)

            # Create PositionAdjustment instance
            async with self.session_maker() as session:
                instance = PositionAdjustment(**db_data)
                session.add(instance)
                await session.commit()
                await session.refresh(instance)

                # Create audit trail entry
                await self._create_audit_trail(
                    operation="create",
                    entity_type="position_adjustment",
                    entity_id=str(instance.adjustment_id),
                    details={
                        "reconciliation_id": str(validated_data.reconciliation_id),
                        "trading_pair": validated_data.trading_pair,
                        "adjustment_type": validated_data.adjustment_type.value,
                        "authorized_by": validated_data.authorized_by,
                        "reason": validated_data.reason[:100] + "..." if len(validated_data.reason) > 100 else validated_data.reason,
                    },
                )

                self.logger.info(
                    f"Successfully saved position adjustment {instance.adjustment_id} for reconciliation {validated_data.reconciliation_id}",
                    source_module=self._source_module,
                )

                return instance

        except Exception as e:
            self.logger.exception(
                f"Failed to save position adjustment: {e!s}",
                source_module=self._source_module,
            )
            raise ValueError(f"Database operation failed: {e!s}") from e

    async def get_adjustments_for_event(
        self, reconciliation_id: uuid.UUID) -> Sequence[PositionAdjustment]:
        """Get all position adjustments for a specific reconciliation event."""
        async with self.session_maker() as session:
            stmt = (
                select(PositionAdjustment)
                .where(PositionAdjustment.reconciliation_id == reconciliation_id)
                .order_by(PositionAdjustment.adjusted_at.desc())
            )
            result = await session.execute(stmt)
            adjustments = result.scalars().all()
            self.logger.debug(f"Found {len(adjustments)} adjustments for event {reconciliation_id}", source_module=self._source_module)
            return adjustments

    async def get_adjustment_history(
        self, trading_pair: str | None = None, days: int = 30) -> Sequence[PositionAdjustment]:
        """Get history of position adjustments, optionally filtered by trading_pair."""
        cutoff_date = datetime.now(UTC) - timedelta(days=days)

        # Validate trading_pair format if provided
        if trading_pair:
            try:
                # Use the same validation as in the schema
                trading_pair = trading_pair.strip().upper()
                if "/" not in trading_pair and "-" not in trading_pair:
                    raise ValueError("Trading pair must contain separator (/ or -)")
            except Exception as e:
                raise ValueError(f"Invalid trading pair format: {e!s}") from e

        async with self.session_maker() as session:
            stmt = select(PositionAdjustment).where(PositionAdjustment.adjusted_at > cutoff_date)
            if trading_pair:
                stmt = stmt.where(PositionAdjustment.trading_pair == trading_pair)
            stmt = stmt.order_by(PositionAdjustment.adjusted_at.desc())

            result = await session.execute(stmt)
            adjustments = result.scalars().all()
            self.logger.debug(f"Retrieved adjustment history for last {days} days.", source_module=self._source_module)
            return adjustments

# _parse_report is removed as the repository now deals with SQLAlchemy models directly.
# Transformation to domain/service layer objects (like ReconciliationReport) would happen in a service layer.
