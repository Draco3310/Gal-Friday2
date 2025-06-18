from datetime import UTC, datetime

import pytest

from gal_friday.monitoring.alerting_system import (
    Alert as ExternalAlert,
    AlertingSystem,
)
from gal_friday.monitoring_service import (
    Alert,
    AlertRule,
    AlertSeverity,
    MetricsCollectionSystem,
)


class DummyAlertingSystem(AlertingSystem):
    def __init__(self):
        self.sent = []

    async def send_alert(self, alert: ExternalAlert) -> dict[str, list[str]]:  # type: ignore[override]
        self.sent.append(alert)
        return {"log": ["dummy"]}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_send_alert_notifications_dispatches(mock_logger):
    alerting = DummyAlertingSystem()
    mcs = MetricsCollectionSystem(config={}, logger=mock_logger, alerting_system=alerting)

    alert = Alert(
        alert_id="1",
        name="test",
        condition="greater_than",
        severity=AlertSeverity.WARNING,
        message="warn",
        threshold=1.0,
        current_value=2.0,
        triggered_at=datetime.now(UTC),
    )
    rule = AlertRule(
        name="test",
        metric_name="m",
        condition="greater_than",
        threshold=1.0,
        severity=AlertSeverity.WARNING,
        message_template="warn",
    )

    await mcs._send_alert_notifications(alert, rule)

    assert len(alerting.sent) == 1
    assert alerting.sent[0].title == "test"
