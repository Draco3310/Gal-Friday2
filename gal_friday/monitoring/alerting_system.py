"""Alerting system for Gal-Friday monitoring.

This module handles sending alerts through various channels including
email (SendGrid), SMS (Twilio), and webhooks (Discord/Slack).
"""

import asyncio
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import aiohttp
from sendgrid import SendGridAPIClient  # type: ignore
from sendgrid.helpers.mail import Mail  # type: ignore
from twilio.rest import Client as TwilioClient  # type: ignore

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService
from gal_friday.utils.secrets_manager import SecretsManager


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""
    EMAIL = "email"
    SMS = "sms"
    DISCORD = "discord"
    SLACK = "slack"
    WEBHOOK = "webhook"


@dataclass
class Alert:
    """Represents an alert to be sent."""
    alert_id: str
    title: str
    message: str
    severity: AlertSeverity
    source: str
    tags: dict[str, Any]
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "source": self.source,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat() if self.timestamp else "",
        }

    def get_hash(self) -> str:
        """Get hash for deduplication using SHA-256 for secure hashing."""
        content = f"{self.title}:{self.message}:{self.source}"
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class AlertRecipient:
    """Alert recipient configuration."""
    name: str
    channels: list[AlertChannel]
    email: str | None = None
    phone: str | None = None
    webhook_url: str | None = None
    severity_filter: list[AlertSeverity] | None = None
    tag_filters: dict[str, Any] | None = None
    quiet_hours: dict[str, str] | None = None  # {"start": "22:00", "end": "08:00"}

    def should_receive(self, alert: Alert) -> bool:
        """Check if recipient should receive this alert."""
        # Check severity filter
        if self.severity_filter and alert.severity not in self.severity_filter:
            return False

        # Check tag filters
        if self.tag_filters:
            for key, value in self.tag_filters.items():
                if alert.tags.get(key) != value:
                    return False

        # Check quiet hours
        if self.quiet_hours:
            current_hour = datetime.now().strftime("%H:%M")
            start = self.quiet_hours["start"]
            end = self.quiet_hours["end"]

            # Handle overnight quiet hours
            if start > end:
                if current_hour >= start or current_hour < end:
                    # Only send critical alerts during quiet hours
                    return alert.severity == AlertSeverity.CRITICAL
            elif start <= current_hour < end:
                return alert.severity == AlertSeverity.CRITICAL

        return True


class AlertDeliveryChannel(ABC):
    """Base class for alert delivery channels."""

    @abstractmethod
    async def send(self, alert: Alert, recipient: AlertRecipient) -> bool:
        """Send alert through this channel.

        Args:
            alert: The alert to send
            recipient: Recipient configuration

        Returns:
            bool: True if alert was sent successfully, False otherwise
        """

    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if channel is properly configured.

        Returns:
            bool: True if test was successful, False otherwise
        """


class EmailChannel(AlertDeliveryChannel):
    """SendGrid email delivery channel."""

    def __init__(self, api_key: str, from_email: str, logger: LoggerService) -> None:
        """Initialize the email channel.

        Args:
            api_key: SendGrid API key
            from_email: Sender email address
            logger: Logger instance for logging
        """
        self.client = SendGridAPIClient(api_key)
        self.from_email = from_email
        self.logger = logger
        self._source_module = self.__class__.__name__

    async def send(self, alert: Alert, recipient: AlertRecipient) -> bool:
        """Send alert via email."""
        if not recipient.email:
            return False

        try:
            # Format email content
            html_content = self._format_html_alert(alert)

            message = Mail(
                from_email=self.from_email,
                to_emails=recipient.email,
                subject=f"[{alert.severity.value.upper()}] {alert.title}",
                html_content=html_content)

            # Send email
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.client.send, message)

            if response.status_code in [200, 201, 202]:
                self.logger.info(
                    f"Email alert sent to {recipient.email}",
                    source_module=self._source_module,
                    context={"alert_id": alert.alert_id})
                return True
            self.logger.error(
                f"Failed to send email: {response.status_code}",
                source_module=self._source_module,
                context={"response": response.body})
            return False

        except Exception:
            self.logger.exception(
                "Error sending email alert",
                source_module=self._source_module)
            return False

    async def test_connection(self) -> bool:
        """Test SendGrid connection."""
        try:
            # Send test email
            test_alert = Alert(
                alert_id="test",
                title="Test Alert",
                message="This is a test alert from Gal-Friday",
                severity=AlertSeverity.INFO,
                source="AlertingSystem",
                tags={"test": True})

            test_recipient = AlertRecipient(
                name="Test",
                channels=[AlertChannel.EMAIL],
                email=self.from_email)

            return await self.send(test_alert, test_recipient)

        except Exception:
            return False

    def _format_html_alert(self, alert: Alert) -> str:
        """Format alert as HTML email."""
        severity_colors = {
            AlertSeverity.INFO: "#17a2b8",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.ERROR: "#dc3545",
            AlertSeverity.CRITICAL: "#721c24",
        }

        color = severity_colors.get(alert.severity, "#000000")

        tag_style = (
            "background-color: #e9ecef; padding: 2px 6px; "
            "margin: 2px; border-radius: 3px;"
        )
        # Tags HTML is not currently used in the template
        _ = "".join(
            f'<span style="{tag_style}">{k}: {v}</span>'
            for k, v in alert.tags.items()
        )

        timestamp_str = (
            alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
            if alert.timestamp
            else "Unknown"
        )

        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background-color: {color}; color: white; "
                 "padding: 20px; border-radius: 5px 5px 0 0;">
                <h2 style="margin: 0;">{alert.title}</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">
                    Severity: {alert.severity.value.upper()}
                </p>
            </div>
            <div style="padding: 20px; background-color: #f8f9fa; "
                 "border: 1px solid #dee2e6; border-top: none;">
                <p style="font-size: 16px; line-height: 1.5;">{alert.message}</p>
                <hr style="border: none; border-top: 1px solid #dee2e6;">
                <p><strong>Source:</strong> {alert.source}</p>
                <p><strong>Time:</strong> {timestamp_str}</p>
                <p style="margin-top: 20px; font-size: 12px; color: #6c757d;">
                    This is an automated message. Please do not reply to this email.
                </p>
            </div>
        </body>
        </html>
        """


class SMSChannel(AlertDeliveryChannel):
    """Twilio SMS delivery channel."""

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        logger: LoggerService) -> None:
        """Initialize the SMS channel.

        Args:
            account_sid: Twilio account SID
            auth_token: Twilio auth token
            from_number: Sender phone number
            logger: Logger instance for logging
        """
        self.client = TwilioClient(account_sid, auth_token)
        self.from_number = from_number
        self.logger = logger
        self._source_module = self.__class__.__name__
        self.SMS_CHAR_LIMIT = 160  # Standard SMS character limit

    async def send(self, alert: Alert, recipient: AlertRecipient) -> bool:
        """Send alert via SMS."""
        if not recipient.phone:
            return False

        try:
            # Format SMS message with character limit
            severity = f"[{alert.severity.value.upper()}]"
            message = f"{severity} {alert.title}\n{alert.message}"
            if len(message) > self.SMS_CHAR_LIMIT:
                truncate_at = self.SMS_CHAR_LIMIT - 3
                message = f"{message[:truncate_at]}..."

            # Send SMS
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.messages.create,
                message,
                self.from_number,
                recipient.phone)

            self.logger.info(
                f"SMS alert sent to {recipient.phone}",
                source_module=self._source_module,
                context={"alert_id": alert.alert_id, "sid": result.sid})
            return True

        except Exception:
            self.logger.exception(
                "Error sending SMS alert",
                source_module=self._source_module)
            return False

    async def test_connection(self) -> bool:
        """Test Twilio connection."""
        try:
            # Check account status
            account = await asyncio.get_event_loop().run_in_executor(
                None, self.client.api.accounts(self.client.account_sid).fetch)
            return bool(account.status == "active")
        except Exception:
            return False


class DiscordChannel(AlertDeliveryChannel):
    """Discord webhook delivery channel."""

    def __init__(self, logger: LoggerService) -> None:
        """Initialize the Discord channel.

        Args:
            logger: Logger instance for logging
        """
        self.logger = logger
        self._source_module = self.__class__.__name__
        self.session: aiohttp.ClientSession | None = None
        self.DISCORD_WEBHOOK_URL_PATTERN = "discord.com/api/webhooks"

    async def send(self, alert: Alert, recipient: AlertRecipient) -> bool:
        """Send alert via Discord webhook.

        Args:
            alert: Alert to send
            recipient: Recipient configuration

        Returns:
            bool: True if alert was sent successfully, False otherwise
        """
        if (not recipient.webhook_url or
            self.DISCORD_WEBHOOK_URL_PATTERN not in recipient.webhook_url):
            return False

        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            # Format Discord embed
            embed = self._format_discord_embed(alert)

            success_status_codes = (200, 204)
            payload = {"embeds": [embed]}
            async with self.session.post(
                recipient.webhook_url, json=payload) as response:
                if response.status in success_status_codes:
                    self.logger.info(
                        "Discord alert sent",
                        source_module=self._source_module,
                        context={"alert_id": alert.alert_id})
                    return True
                self.logger.error(
                    f"Failed to send Discord alert: {response.status}",
                    source_module=self._source_module)
                return False

        except Exception:
            self.logger.exception(
                "Error sending Discord alert",
                source_module=self._source_module)
            return False

    async def test_connection(self) -> bool:
        """Test Discord webhook."""
        # Discord webhooks don't have a test endpoint
        return True

    def _format_discord_embed(self, alert: Alert) -> dict[str, Any]:
        """Format alert as Discord embed."""
        # Discord color codes for different severity levels
        discord_colors = {
            AlertSeverity.INFO: 0x17A2B8,     # Teal
            AlertSeverity.WARNING: 0xFFC107,  # Yellow
            AlertSeverity.ERROR: 0xDC3545,    # Red
            AlertSeverity.CRITICAL: 0x721C24, # Dark Red
        }
        fields = [
            {"name": "Source", "value": alert.source, "inline": True},
            {"name": "Severity", "value": alert.severity.value.upper(), "inline": True},
        ]

        for key, value in alert.tags.items():
            fields.append({"name": key, "value": str(value), "inline": True})

        timestamp_str = (
            alert.timestamp.isoformat()
            if alert.timestamp
            else datetime.now(UTC).isoformat()
        )

        return {
            "title": alert.title,
            "description": alert.message,
            "color": discord_colors.get(alert.severity, 0x000000),  # Default to black
            "fields": fields,
            "timestamp": timestamp_str,
            "footer": {"text": f"Alert ID: {alert.alert_id}"},
        }


class SlackChannel(AlertDeliveryChannel):
    """Slack webhook delivery channel."""

    def __init__(self, logger: LoggerService) -> None:
        """Initialize the Slack channel.

        Args:
            logger: Logger instance for logging
        """
        self.logger = logger
        self._source_module = self.__class__.__name__
        self.session: aiohttp.ClientSession | None = None
        self.success_status = 200

    async def send(self, alert: Alert, recipient: AlertRecipient) -> bool:
        """Send alert via Slack webhook.

        Args:
            alert: Alert to send
            recipient: Recipient configuration

        Returns:
            bool: True if alert was sent successfully, False otherwise
        """
        if not recipient.webhook_url or "hooks.slack.com" not in recipient.webhook_url:
            return False

        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            # Format Slack message
            payload = self._format_slack_message(alert)

            async with self.session.post(
                recipient.webhook_url, json=payload) as response:
                if response.status == self.success_status:
                    self.logger.info(
                        "Slack alert sent",
                        source_module=self._source_module,
                        context={"alert_id": alert.alert_id})
                    return True
                self.logger.error(
                    f"Failed to send Slack alert: {response.status}",
                    source_module=self._source_module)
                return False

        except Exception:
            self.logger.exception(
                "Error sending Slack alert",
                source_module=self._source_module)
            return False

    async def test_connection(self) -> bool:
        """Test Slack webhook."""
        return True

    def _format_slack_message(self, alert: Alert) -> dict[str, Any]:
        """Format alert as Slack message."""
        emoji = {
            AlertSeverity.INFO: ":information_source:",
            AlertSeverity.WARNING: ":warning:",
            AlertSeverity.ERROR: ":x:",
            AlertSeverity.CRITICAL: ":rotating_light:",
        }

        timestamp_str = (
            alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
            if alert.timestamp
            else "Unknown"
        )

        return {
            "text": f"{emoji.get(alert.severity, '')} *{alert.title}*",
            "attachments": [{
                "color": {
                    AlertSeverity.INFO: "#17a2b8",
                    AlertSeverity.WARNING: "#ffc107",
                    AlertSeverity.ERROR: "#dc3545",
                    AlertSeverity.CRITICAL: "#721c24",
                }.get(alert.severity, "#000000"),
                "fields": [
                    {"title": "Message", "value": alert.message},
                    {"title": "Source", "value": alert.source, "short": True},
                    {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                    {"title": "Time", "value": timestamp_str, "short": True},
                ] + [
                    {"title": k, "value": str(v), "short": True}
                    for k, v in alert.tags.items()
                ],
                "footer": f"Alert ID: {alert.alert_id}",
            }],
        }


class AlertingSystem:
    """Main alerting system coordinator."""

    def __init__(
        self,
        config: ConfigManager,
        secrets: SecretsManager,
        logger: LoggerService) -> None:
        """Initialize the alerting system.

        Args:
            config: Configuration manager instance
            secrets: Secrets manager instance
            logger: Logger instance for logging
        """
        self.config = config
        self.secrets = secrets
        self.logger = logger
        self._source_module = self.__class__.__name__

        self.channels: dict[AlertChannel, AlertDeliveryChannel] = {}
        self.recipients: list[AlertRecipient] = []
        self.alert_history: dict[str, datetime] = {}  # For deduplication
        self.dedup_window = timedelta(minutes=5)

        self._initialize_channels()
        self._load_recipients()

    def _initialize_channels(self) -> None:
        """Initialize configured alert channels."""
        # Email channel (SendGrid)
        if self.config.get("alerting.email.enabled", False):
            api_key = self.secrets.get_secret("SENDGRID_API_KEY")
            from_email = self.config.get("alerting.email.from_address")

            if api_key and from_email:
                self.channels[AlertChannel.EMAIL] = EmailChannel(
                    api_key, from_email, self.logger)
                self.logger.info(
                    "Email alerting channel initialized",
                    source_module=self._source_module)

        # SMS channel (Twilio)
        if self.config.get("alerting.sms.enabled", False):
            account_sid = self.secrets.get_secret("TWILIO_ACCOUNT_SID")
            auth_token = self.secrets.get_secret("TWILIO_AUTH_TOKEN")
            from_number = self.config.get("alerting.sms.from_number")

            if account_sid and auth_token and from_number: # Using individual checks for clarity
                # Assertions to help mypy confirm the types are strings
                assert isinstance(account_sid, str), "TWILIO_ACCOUNT_SID must be a non-empty string"
                assert isinstance(auth_token, str), "TWILIO_AUTH_TOKEN must be a non-empty string"
                assert isinstance(from_number, str), "alerting.sms.from_number must be a non-empty string"

                self.channels[AlertChannel.SMS] = SMSChannel(
                    account_sid, auth_token, from_number, self.logger)
                self.logger.info(
                    "SMS alerting channel initialized",
                    source_module=self._source_module)
            else:
                self.logger.warning(
                    "SMS alerting enabled but missing one or more required configurations "
                    "(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, or alerting.sms.from_number was empty/None). "
                    "SMS channel will not be initialized.",
                    source_module=self._source_module)

        # Discord channel
        if self.config.get("alerting.discord.enabled", False):
            self.channels[AlertChannel.DISCORD] = DiscordChannel(self.logger)
            self.logger.info(
                "Discord alerting channel initialized",
                source_module=self._source_module)

        # Slack channel
        if self.config.get("alerting.slack.enabled", False):
            self.channels[AlertChannel.SLACK] = SlackChannel(self.logger)
            self.logger.info(
                "Slack alerting channel initialized",
                source_module=self._source_module)

    def _load_recipients(self) -> None:
        """Load alert recipients from configuration."""
        recipients_config = self.config.get("alerting.recipients", [])

        for recipient_data in recipients_config:
            recipient = AlertRecipient(
                name=recipient_data.get("name"),
                channels=[AlertChannel(c) for c in recipient_data.get("channels", [])],
                email=recipient_data.get("email"),
                phone=recipient_data.get("phone"),
                webhook_url=recipient_data.get("webhook_url"),
                severity_filter=[
                    AlertSeverity(s) for s in recipient_data.get("severity_filter", [])
                ] if recipient_data.get("severity_filter") else None,
                tag_filters=recipient_data.get("tag_filters"),
                quiet_hours=recipient_data.get("quiet_hours"))
            self.recipients.append(recipient)

        self.logger.info(
            f"Loaded {len(self.recipients)} alert recipients",
            source_module=self._source_module)

    async def send_alert(self, alert: Alert) -> dict[str, list[str]]:
        """Send an alert to all configured recipients.

        Args:
            alert: The alert to send.

        Returns:
            A dictionary mapping channels to lists of successful recipient names.
        """
        # Check for duplicate alerts
        alert_hash = alert.get_hash()
        if alert_hash in self.alert_history:
            last_sent = self.alert_history[alert_hash]
            if datetime.now(UTC) - last_sent < self.dedup_window:
                self.logger.info(
                    f"Suppressing duplicate alert: {alert.title}",
                    source_module=self._source_module)
                return {}

        # Update history
        self.alert_history[alert_hash] = datetime.now(UTC)

        # Clean old history entries
        cutoff = datetime.now(UTC) - self.dedup_window
        self.alert_history = {
            h: t for h, t in self.alert_history.items()
            if t > cutoff
        }

        # Send to recipients
        results: dict[str, list[str]] = {}

        for recipient in self.recipients:
            if not recipient.should_receive(alert):
                continue

            for channel in recipient.channels:
                if channel not in self.channels:
                    continue

                try:
                    success = await self.channels[channel].send(alert, recipient)
                    if success:
                        if channel.value not in results:
                            results[channel.value] = []
                        results[channel.value].append(recipient.name)
                except Exception as e:
                    self.logger.exception(
                        f"Error sending alert via {channel.value} to {recipient.name}",
                        source_module=self._source_module,
                        context={"error": str(e)})

        return results

    async def test_all_channels(self) -> dict[str, bool]:
        """Test all configured channels.

        Returns:
            A dictionary mapping channel names to test results.
        """
        results = {}

        for channel_type, channel in self.channels.items():
            try:
                results[channel_type.value] = await channel.test_connection()
            except Exception:
                self.logger.exception(
                    f"Error testing {channel_type.value} channel",
                    source_module=self._source_module)
                results[channel_type.value] = False

        return results

    async def close(self) -> None:
        """Clean up resources."""
        # Close aiohttp sessions
        for channel in self.channels.values():
            if hasattr(channel, "session") and channel.session:
                await channel.session.close()