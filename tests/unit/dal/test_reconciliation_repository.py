"""Unit tests for ReconciliationRepository validation and functionality."""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError as PydanticValidationError

from gal_friday.dal.repositories.reconciliation_repository import (
    AdjustmentType,
    PositionAdjustmentSchema,
    ReconciliationEventSchema,
    ReconciliationRepository,
    ReconciliationStatus,
    ReconciliationType,
    ReconciliationValidationError,
)


class TestReconciliationEventSchema:
    """Test ReconciliationEventSchema validation."""

    def test_valid_reconciliation_event_data(self):
        """Test validation with valid reconciliation event data."""
        valid_data = {
            "timestamp": datetime.now(UTC),
            "reconciliation_type": ReconciliationType.POSITION,
            "status": ReconciliationStatus.COMPLETED,
            "report": {"summary": "Test reconciliation", "details": {}},
            "discrepancies_found": 5,
            "auto_corrected": 3,
            "manual_review_required": 2,
            "duration_seconds": Decimal("15.5")
        }
        
        schema = ReconciliationEventSchema(**valid_data)
        assert schema.reconciliation_type == ReconciliationType.POSITION
        assert schema.status == ReconciliationStatus.COMPLETED
        assert schema.discrepancies_found == 5
        assert schema.auto_corrected == 3
        assert schema.manual_review_required == 2

    def test_auto_generated_reconciliation_id(self):
        """Test that reconciliation_id is auto-generated when not provided."""
        data = {
            "timestamp": datetime.now(UTC),
            "reconciliation_type": ReconciliationType.TRADE,
            "status": ReconciliationStatus.PENDING,
            "report": {"summary": "Test"}
        }
        
        schema = ReconciliationEventSchema(**data)
        assert schema.reconciliation_id is None  # Will be generated in repository

    def test_timestamp_timezone_handling(self):
        """Test timestamp timezone validation."""
        # Timestamp without timezone should be handled
        naive_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        data = {
            "timestamp": naive_timestamp,
            "reconciliation_type": ReconciliationType.BALANCE,
            "status": ReconciliationStatus.COMPLETED,
            "report": {"summary": "Test"}
        }
        
        schema = ReconciliationEventSchema(**data)
        assert schema.timestamp.tzinfo == UTC

    def test_future_timestamp_validation(self):
        """Test that future timestamps are rejected (with some tolerance)."""
        future_timestamp = datetime.now(UTC) + timedelta(hours=1)
        data = {
            "timestamp": future_timestamp,
            "reconciliation_type": ReconciliationType.SETTLEMENT,
            "status": ReconciliationStatus.FAILED,
            "report": {"summary": "Test"}
        }
        
        with pytest.raises(PydanticValidationError) as exc_info:
            ReconciliationEventSchema(**data)
        
        assert "cannot be more than 5 minutes in the future" in str(exc_info.value)

    def test_auto_corrected_exceeds_discrepancies_validation(self):
        """Test validation when auto_corrected exceeds discrepancies_found."""
        data = {
            "timestamp": datetime.now(UTC),
            "reconciliation_type": ReconciliationType.POSITION,
            "status": ReconciliationStatus.PARTIAL,
            "report": {"summary": "Test"},
            "discrepancies_found": 3,
            "auto_corrected": 5  # More than discrepancies_found
        }
        
        with pytest.raises(PydanticValidationError) as exc_info:
            ReconciliationEventSchema(**data)
        
        assert "cannot exceed discrepancies found" in str(exc_info.value)

    def test_manual_review_exceeds_remaining_validation(self):
        """Test validation when manual_review_required exceeds remaining discrepancies."""
        data = {
            "timestamp": datetime.now(UTC),
            "reconciliation_type": ReconciliationType.FEES,
            "status": ReconciliationStatus.COMPLETED,
            "report": {"summary": "Test"},
            "discrepancies_found": 5,
            "auto_corrected": 3,
            "manual_review_required": 4  # More than remaining (2)
        }
        
        with pytest.raises(PydanticValidationError) as exc_info:
            ReconciliationEventSchema(**data)
        
        assert "cannot exceed remaining discrepancies" in str(exc_info.value)

    def test_negative_values_validation(self):
        """Test that negative values are rejected for count fields."""
        data = {
            "timestamp": datetime.now(UTC),
            "reconciliation_type": ReconciliationType.CORPORATE_ACTIONS,
            "status": ReconciliationStatus.PENDING,
            "report": {"summary": "Test"},
            "discrepancies_found": -1  # Negative value
        }
        
        with pytest.raises(PydanticValidationError) as exc_info:
            ReconciliationEventSchema(**data)
        
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_report_structure_validation(self):
        """Test report structure validation and auto-completion."""
        data = {
            "timestamp": datetime.now(UTC),
            "reconciliation_type": ReconciliationType.TRADE,
            "status": ReconciliationStatus.COMPLETED,
            "report": {}  # Empty report
        }
        
        schema = ReconciliationEventSchema(**data)
        assert "summary" in schema.report
        assert "timestamp" in schema.report


class TestPositionAdjustmentSchema:
    """Test PositionAdjustmentSchema validation."""

    def test_valid_position_adjustment_data(self):
        """Test validation with valid position adjustment data."""
        reconciliation_id = uuid.uuid4()
        valid_data = {
            "reconciliation_id": reconciliation_id,
            "trading_pair": "BTC/USD",
            "adjustment_type": AdjustmentType.CORRECTION,
            "reason": "Position correction due to late trade settlement",
            "authorized_by": "system_admin",
            "old_value": Decimal("1.5"),
            "new_value": Decimal("1.45"),
            "quantity_change": Decimal("-0.05")
        }
        
        schema = PositionAdjustmentSchema(**valid_data)
        assert schema.reconciliation_id == reconciliation_id
        assert schema.trading_pair == "BTC/USD"
        assert schema.adjustment_type == AdjustmentType.CORRECTION
        assert schema.old_value == Decimal("1.5")

    def test_trading_pair_format_validation(self):
        """Test trading pair format validation."""
        base_data = {
            "reconciliation_id": uuid.uuid4(),
            "adjustment_type": AdjustmentType.REBALANCE,
            "reason": "Valid reason for adjustment",
            "authorized_by": "user123",
            "quantity_change": Decimal("0.1")
        }
        
        # Valid formats
        valid_pairs = ["BTC/USD", "ETH-EUR", "btc/usd", "eth-eur"]
        for pair in valid_pairs:
            data = {**base_data, "trading_pair": pair}
            schema = PositionAdjustmentSchema(**data)
            assert "/" in schema.trading_pair or "-" in schema.trading_pair
            assert schema.trading_pair.isupper()  # Should be converted to uppercase
        
        # Invalid formats
        invalid_pairs = ["BTCUSD", "", "BTC"]
        for pair in invalid_pairs:
            data = {**base_data, "trading_pair": pair}
            with pytest.raises(PydanticValidationError):
                PositionAdjustmentSchema(**data)

    def test_reason_validation(self):
        """Test reason field validation."""
        base_data = {
            "reconciliation_id": uuid.uuid4(),
            "trading_pair": "BTC/USD",
            "adjustment_type": AdjustmentType.FEES,
            "authorized_by": "admin",
            "old_value": Decimal("100"),
            "new_value": Decimal("95")
        }
        
        # Valid reason
        valid_data = {**base_data, "reason": "Fee adjustment for transaction processing costs"}
        schema = PositionAdjustmentSchema(**valid_data)
        assert len(schema.reason) >= 10
        
        # Too short reason
        with pytest.raises(PydanticValidationError) as exc_info:
            PositionAdjustmentSchema(**{**base_data, "reason": "Short"})
        assert "at least 10 characters" in str(exc_info.value)
        
        # Placeholder text
        placeholder_reasons = ["This is a placeholder reason", "TODO: add real reason", "Test adjustment"]
        for reason in placeholder_reasons:
            with pytest.raises(PydanticValidationError) as exc_info:
                PositionAdjustmentSchema(**{**base_data, "reason": reason})
            assert "cannot contain placeholder text" in str(exc_info.value)

    def test_quantity_change_zero_validation(self):
        """Test that zero quantity change is rejected."""
        data = {
            "reconciliation_id": uuid.uuid4(),
            "trading_pair": "ETH/USD",
            "adjustment_type": AdjustmentType.CORRECTION,
            "reason": "Valid reason for position adjustment",
            "authorized_by": "system",
            "quantity_change": Decimal("0")  # Zero change not allowed
        }
        
        with pytest.raises(PydanticValidationError) as exc_info:
            PositionAdjustmentSchema(**data)
        assert "cannot be zero" in str(exc_info.value)

    def test_at_least_one_value_field_required(self):
        """Test that at least one value field must be provided."""
        data = {
            "reconciliation_id": uuid.uuid4(),
            "trading_pair": "BTC/USD",
            "adjustment_type": AdjustmentType.SPLIT,
            "reason": "Stock split adjustment processing",
            "authorized_by": "admin"
            # No value fields provided
        }
        
        with pytest.raises(PydanticValidationError) as exc_info:
            PositionAdjustmentSchema(**data)
        assert "At least one of old_value, new_value, or quantity_change must be provided" in str(exc_info.value)

    def test_split_dividend_requires_old_new_values(self):
        """Test that SPLIT and DIVIDEND adjustments require old_value and new_value."""
        base_data = {
            "reconciliation_id": uuid.uuid4(),
            "trading_pair": "AAPL/USD",
            "reason": "Stock split 2:1 adjustment processing",
            "authorized_by": "corporate_actions"
        }
        
        # Test SPLIT
        data = {**base_data, "adjustment_type": AdjustmentType.SPLIT, "quantity_change": Decimal("100")}
        with pytest.raises(PydanticValidationError) as exc_info:
            PositionAdjustmentSchema(**data)
        assert "adjustments require both old_value and new_value" in str(exc_info.value)
        
        # Test DIVIDEND
        data = {**base_data, "adjustment_type": AdjustmentType.DIVIDEND, "quantity_change": Decimal("5")}
        with pytest.raises(PydanticValidationError) as exc_info:
            PositionAdjustmentSchema(**data)
        assert "adjustments require both old_value and new_value" in str(exc_info.value)
        
        # Valid SPLIT with both values
        valid_data = {
            **base_data, 
            "adjustment_type": AdjustmentType.SPLIT,
            "old_value": Decimal("100"),
            "new_value": Decimal("200")
        }
        schema = PositionAdjustmentSchema(**valid_data)
        assert schema.adjustment_type == AdjustmentType.SPLIT

    def test_field_length_validations(self):
        """Test field length validations."""
        base_data = {
            "reconciliation_id": uuid.uuid4(),
            "trading_pair": "BTC/USD",
            "adjustment_type": AdjustmentType.CORRECTION,
            "reason": "Valid reason for the adjustment",
            "quantity_change": Decimal("0.1")
        }
        
        # authorized_by too long
        long_authorized_by = "x" * 101  # Max is 100
        with pytest.raises(PydanticValidationError):
            PositionAdjustmentSchema(**{**base_data, "authorized_by": long_authorized_by})
        
        # reference_id too long
        long_reference_id = "x" * 51  # Max is 50
        with pytest.raises(PydanticValidationError):
            PositionAdjustmentSchema(**{**base_data, "authorized_by": "admin", "reference_id": long_reference_id})
        
        # notes too long
        long_notes = "x" * 2001  # Max is 2000
        with pytest.raises(PydanticValidationError):
            PositionAdjustmentSchema(**{**base_data, "authorized_by": "admin", "notes": long_notes})


class TestReconciliationRepository:
    """Test ReconciliationRepository validation and functionality."""

    @pytest.fixture
    def mock_logger(self):
        """Mock logger service."""
        logger = MagicMock()
        logger.debug = MagicMock()
        logger.info = MagicMock()
        logger.error = MagicMock()
        return logger

    @pytest.fixture
    def mock_session_maker(self):
        """Mock session maker."""
        session_maker = MagicMock()
        return session_maker

    @pytest.fixture
    def repository(self, mock_session_maker, mock_logger):
        """Create repository instance with mocked dependencies."""
        return ReconciliationRepository(mock_session_maker, mock_logger)

    def test_validate_reconciliation_data_success(self, repository):
        """Test successful reconciliation data validation."""
        valid_data = {
            "timestamp": datetime.now(UTC),
            "reconciliation_type": "position",
            "status": "completed",
            "report": {"summary": "Test reconciliation"},
            "discrepancies_found": 3,
            "auto_corrected": 2,
            "manual_review_required": 1
        }
        
        result = repository._validate_reconciliation_data(valid_data)
        assert isinstance(result, ReconciliationEventSchema)
        assert result.reconciliation_type == ReconciliationType.POSITION
        assert result.status == ReconciliationStatus.COMPLETED

    def test_validate_reconciliation_data_with_id_generation(self, repository):
        """Test reconciliation data validation with ID generation."""
        data = {
            "timestamp": datetime.now(UTC),
            "reconciliation_type": "trade",
            "status": "pending",
            "report": {"summary": "Test"}
        }
        
        result = repository._validate_reconciliation_data(data)
        # ID should be generated and added to the data
        assert "reconciliation_id" in data
        assert isinstance(data["reconciliation_id"], uuid.UUID)

    def test_validate_reconciliation_data_validation_error(self, repository, mock_logger):
        """Test reconciliation data validation with validation errors."""
        invalid_data = {
            "timestamp": datetime.now(UTC) + timedelta(hours=2),  # Too far in future
            "reconciliation_type": "invalid_type",
            "status": "completed",
            "report": {"summary": "Test"}
        }
        
        with pytest.raises(ReconciliationValidationError) as exc_info:
            repository._validate_reconciliation_data(invalid_data)
        
        assert "Invalid reconciliation event data" in str(exc_info.value)
        assert exc_info.value.validation_errors
        mock_logger.error.assert_called_once()

    def test_validate_position_adjustment_data_success(self, repository):
        """Test successful position adjustment data validation."""
        valid_data = {
            "reconciliation_id": uuid.uuid4(),
            "trading_pair": "BTC/USD",
            "adjustment_type": "correction",
            "reason": "Position correction due to late settlement",
            "authorized_by": "admin",
            "old_value": Decimal("1.5"),
            "new_value": Decimal("1.45")
        }
        
        result = repository._validate_position_adjustment_data(valid_data)
        assert isinstance(result, PositionAdjustmentSchema)
        assert result.adjustment_type == AdjustmentType.CORRECTION
        assert result.trading_pair == "BTC/USD"

    def test_validate_position_adjustment_data_validation_error(self, repository, mock_logger):
        """Test position adjustment data validation with validation errors."""
        invalid_data = {
            "reconciliation_id": uuid.uuid4(),
            "trading_pair": "INVALID",  # No separator
            "adjustment_type": "correction",
            "reason": "Short",  # Too short
            "authorized_by": "admin"
            # No value fields
        }
        
        with pytest.raises(ReconciliationValidationError) as exc_info:
            repository._validate_position_adjustment_data(invalid_data)
        
        assert "Invalid position adjustment data" in str(exc_info.value)
        assert exc_info.value.validation_errors
        mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_audit_trail(self, repository, mock_logger):
        """Test audit trail creation."""
        await repository._create_audit_trail(
            operation="create",
            entity_type="reconciliation_event",
            entity_id="test-id",
            details={"test": "data"}
        )
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "Audit trail: create reconciliation_event test-id" in call_args[0][0]
        assert "audit_data" in call_args[1]["extra"]

    @pytest.mark.asyncio
    async def test_save_reconciliation_event_success(self, repository, mock_logger):
        """Test successful reconciliation event saving."""
        # Mock the create method
        repository.create = AsyncMock()
        mock_event = MagicMock()
        mock_event.reconciliation_id = uuid.uuid4()
        repository.create.return_value = mock_event
        
        # Mock the audit trail method
        repository._create_audit_trail = AsyncMock()
        
        valid_data = {
            "timestamp": datetime.now(UTC),
            "reconciliation_type": "position",
            "status": "completed",
            "report": {"summary": "Test reconciliation"}
        }
        
        result = await repository.save_reconciliation_event(valid_data)
        
        assert result == mock_event
        repository.create.assert_called_once()
        repository._create_audit_trail.assert_called_once()
        mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_save_reconciliation_event_validation_failure(self, repository, mock_logger):
        """Test reconciliation event saving with validation failure."""
        invalid_data = {
            "timestamp": datetime.now(UTC),
            "reconciliation_type": "invalid_type",
            "status": "completed",
            "report": {"summary": "Test"}
        }
        
        with pytest.raises(ReconciliationValidationError):
            await repository.save_reconciliation_event(invalid_data)

    @pytest.mark.asyncio
    async def test_save_position_adjustment_success(self, repository, mock_logger):
        """Test successful position adjustment saving."""
        reconciliation_id = uuid.uuid4()
        
        # Mock the get_reconciliation_event method
        repository.get_reconciliation_event = AsyncMock()
        repository.get_reconciliation_event.return_value = MagicMock()  # Event exists
        
        # Mock session and instance
        mock_session = AsyncMock()
        mock_instance = MagicMock()
        mock_instance.adjustment_id = uuid.uuid4()
        repository.session_maker.return_value.__aenter__.return_value = mock_session
        repository.session_maker.return_value.__aexit__.return_value = None
        
        # Mock the audit trail method
        repository._create_audit_trail = AsyncMock()
        
        valid_data = {
            "reconciliation_id": reconciliation_id,
            "trading_pair": "BTC/USD",
            "adjustment_type": "correction",
            "reason": "Position correction due to settlement",
            "authorized_by": "admin",
            "quantity_change": Decimal("0.1")
        }
        
        # Mock PositionAdjustment constructor to return our mock instance
        with patch('gal_friday.dal.repositories.reconciliation_repository.PositionAdjustment', return_value=mock_instance):
            result = await repository.save_position_adjustment(valid_data)
        
        assert result == mock_instance
        repository._create_audit_trail.assert_called_once()
        mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_save_position_adjustment_nonexistent_reconciliation(self, repository):
        """Test position adjustment saving with nonexistent reconciliation event."""
        # Mock the get_reconciliation_event method to return None
        repository.get_reconciliation_event = AsyncMock(return_value=None)
        
        valid_data = {
            "reconciliation_id": uuid.uuid4(),
            "trading_pair": "BTC/USD",
            "adjustment_type": "correction",
            "reason": "Position correction for settlement",
            "authorized_by": "admin",
            "quantity_change": Decimal("0.1")
        }
        
        with pytest.raises(ValueError) as exc_info:
            await repository.save_position_adjustment(valid_data)
        
        assert "does not exist" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_recent_reconciliation_events_with_invalid_status(self, repository):
        """Test getting recent events with invalid status."""
        with pytest.raises(ValueError) as exc_info:
            await repository.get_recent_reconciliation_events(status="invalid_status")
        
        assert "Invalid status" in str(exc_info.value)
        assert "Valid options:" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_adjustment_history_with_invalid_trading_pair(self, repository):
        """Test getting adjustment history with invalid trading pair."""
        with pytest.raises(ValueError) as exc_info:
            await repository.get_adjustment_history(trading_pair="INVALID")
        
        assert "Invalid trading pair format" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_adjustment_history_with_valid_trading_pair(self, repository):
        """Test getting adjustment history with valid trading pair format validation."""
        # Mock session
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result
        repository.session_maker.return_value.__aenter__.return_value = mock_session
        repository.session_maker.return_value.__aexit__.return_value = None
        
        # Should not raise error with valid trading pair
        result = await repository.get_adjustment_history(trading_pair="btc/usd")
        assert result == []

    def test_reconciliation_validation_error_attributes(self):
        """Test ReconciliationValidationError custom attributes."""
        validation_errors = {"field1": {"message": "error", "type": "value_error"}}
        error = ReconciliationValidationError(
            message="Test error",
            field_path="test.field",
            validation_errors=validation_errors
        )
        
        assert str(error) == "Test error"
        assert error.field_path == "test.field"
        assert error.validation_errors == validation_errors

    def test_enum_values(self):
        """Test that enum values are correctly defined."""
        # Test ReconciliationStatus
        assert ReconciliationStatus.PENDING == "pending"
        assert ReconciliationStatus.COMPLETED == "completed"
        assert ReconciliationStatus.FAILED == "failed"
        
        # Test ReconciliationType
        assert ReconciliationType.POSITION == "position"
        assert ReconciliationType.TRADE == "trade"
        assert ReconciliationType.BALANCE == "balance"
        
        # Test AdjustmentType
        assert AdjustmentType.CORRECTION == "correction"
        assert AdjustmentType.SPLIT == "split"
        assert AdjustmentType.DIVIDEND == "dividend" 