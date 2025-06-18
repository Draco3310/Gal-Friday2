"""WebSocket message processing and sequencing."""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from gal_friday.logger_service import LoggerService
from typing import Any


@dataclass
class SequencedMessage:
    """Message with sequence tracking."""
    sequence: int
    channel: str
    data: dict[str, Any]
    timestamp: datetime
    processed: bool = False


class SequenceTracker:
    """Track message sequences for gap detection."""

    def __init__(self, logger: LoggerService) -> None:
        """Initialize the sequence tracker.

        Args:
            logger: Logger service instance for logging messages
        """
        self.logger = logger
        self._source_module = self.__class__.__name__

        # Track sequences per channel
        self.sequences: dict[str, int] = {}
        self.gaps: dict[str, list[tuple[int, int]]] = defaultdict(list)

    def check_sequence(self, channel: str, sequence: int) -> list[tuple[int, int]] | None:
        """Check for sequence gaps.

        Returns:
            List of gap ranges if gaps detected, None otherwise
        """
        expected = self.sequences.get(channel, 0) + 1

        if sequence == expected:
            # Normal sequence
            self.sequences[channel] = sequence
            return None

        if sequence > expected:
            # Gap detected
            gap = (expected, sequence - 1)
            self.gaps[channel].append(gap)
            self.sequences[channel] = sequence

            self.logger.warning(
                f"Sequence[Any] gap detected in {channel}: {gap}",
                source_module=self._source_module)
            return [gap]

        # Out of order or duplicate
        self.logger.warning(
            f"Out of order message in {channel}: expected {expected}, got {sequence}",
            source_module=self._source_module)
        return None

    def get_gaps(self, channel: str) -> list[tuple[int, int]]:
        """Get all gaps for a channel."""
        return self.gaps.get(channel, [])

    def clear_gap(self, channel: str, start: int, end: int) -> None:
        """Clear a gap after recovery."""
        if channel in self.gaps:
            self.gaps[channel] = [
                (s, e) for s, e in self.gaps[channel]
                if not (s == start and e == end)
            ]


class MessageCache:
    """Cache for deduplication and replay."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 10000) -> None:
        """Initialize the message cache.

        Args:
            ttl_seconds: Time to live for cached messages in seconds
            max_size: Maximum number of messages to cache
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache: deque[SequencedMessage] = deque(maxlen=max_size)
        self.message_ids: set[str] = set()

    def add(self, message: SequencedMessage) -> bool:
        """Add message to cache.

        Returns:
            True if new message, False if duplicate
        """
        # Create unique ID
        msg_id = f"{message.channel}:{message.sequence}"

        if msg_id in self.message_ids:
            return False

        # Add to cache
        self.cache.append(message)
        self.message_ids.add(msg_id)

        # Clean old messages
        self._cleanup()

        return True

    def get_unprocessed(self, channel: str,
                       start_seq: int,
                       end_seq: int) -> list[SequencedMessage]:
        """Get unprocessed messages in sequence range."""
        messages = []

        for msg in self.cache:
            if (msg.channel == channel and
                start_seq <= msg.sequence <= end_seq and
                not msg.processed):
                messages.append(msg)

        return sorted(messages, key=lambda m: m.sequence)

    def mark_processed(self, channel: str, sequence: int) -> None:
        """Mark message as processed."""
        for msg in self.cache:
            if msg.channel == channel and msg.sequence == sequence:
                msg.processed = True
                break

    def _cleanup(self) -> None:
        """Remove expired messages."""
        cutoff = datetime.now(UTC) - timedelta(seconds=self.ttl_seconds)

        while self.cache and self.cache[0].timestamp < cutoff:
            old_msg = self.cache.popleft()
            msg_id = f"{old_msg.channel}:{old_msg.sequence}"
            self.message_ids.discard(msg_id)


class WebSocketMessageProcessor:
    """Process and validate WebSocket messages."""

    def __init__(self, logger: LoggerService) -> None:
        """Initialize the WebSocket message processor.

        Args:
            logger: Logger service instance for logging messages
        """
        self.logger = logger
        self._source_module = self.__class__.__name__

        self.sequence_tracker = SequenceTracker(logger)
        self.message_cache = MessageCache()

        # Message validation rules
        self.required_fields = {
            "openOrders": ["orderid", "status", "descr"],
            "ownTrades": ["orderid", "pair", "vol", "price"],
            "book": ["pair"],
            "ticker": ["pair", "c", "v"],
        }

    async def process_message(self,
                            channel: str,
                            data: dict[str, Any],
                            sequence: int | None = None) -> SequencedMessage | None:
        """Process and validate WebSocket message."""
        try:
            # Validate message format
            if not self._validate_message(channel, data):
                return None

            # Create sequenced message
            message = SequencedMessage(
                sequence=sequence or 0,
                channel=channel,
                data=data,
                timestamp=datetime.now(UTC))

            # Check for duplicates
            if not self.message_cache.add(message):
                self.logger.debug(
                    f"Duplicate message ignored: {channel}:{sequence}",
                    source_module=self._source_module)
                return None

            # Check sequence if provided
            if sequence:
                gaps = self.sequence_tracker.check_sequence(channel, sequence)
                if gaps:
                    # Handle gaps asynchronously
                    task = asyncio.create_task(self._handle_gaps(channel, gaps))
                    # Store task reference to prevent garbage collection
                    task.add_done_callback(lambda t: t.result() if t.cancelled() else None)

            return message

        except Exception:
            self.logger.exception(
                "Error processing WebSocket message",
                source_module=self._source_module)
            return None

    def _validate_message(self, channel: str, data: dict[str, Any]) -> bool:
        """Validate message has required fields."""
        required = self.required_fields.get(channel, [])

        for field in required:
            if field not in data:
                self.logger.error(
                    f"Missing required field '{field}' in {channel} message",
                    source_module=self._source_module)
                return False

        return True

    async def _handle_gaps(self, channel: str, gaps: list[tuple[int, int]]) -> None:
        """Handle sequence gaps."""
        for start, end in gaps:
            self.logger.info(
                f"Attempting to recover gap {start}-{end} in {channel}",
                source_module=self._source_module)

            # Check cache for missing messages
            cached = self.message_cache.get_unprocessed(channel, start, end)

            if len(cached) == (end - start + 1):
                # All messages found in cache
                self.logger.info(
                    f"Recovered gap from cache: {channel} {start}-{end}",
                    source_module=self._source_module)

                # Mark gap as resolved
                self.sequence_tracker.clear_gap(channel, start, end)
            else:
                # Need to request missing messages
                # In production, this would trigger a recovery mechanism
                self.logger.warning(
                    f"Unable to recover gap from cache: {channel} {start}-{end}",
                    source_module=self._source_module)