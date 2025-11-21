import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock
from src.core.database import DatabaseHandler
from src.core.execution import PositionManager
from src.strategies.donchian_volatility import DonchianStrategy

@pytest.mark.asyncio
async def test_kernel_flow(tmp_path):
    # 1. Database
    db_path = tmp_path / "test.db"
    db = DatabaseHandler(str(db_path))

    # 2. Exchange Mock
    exchange = MagicMock()
    exchange.create_order = AsyncMock(return_value={'id': '123', 'price': 50000.0, 'average': 50000.0})

    # 3. Position Manager
    pm = PositionManager(db, exchange)

    # 4. Strategy
    config = {'symbol': 'BTC/USD', 'lower_length': 5, 'upper_length': 5}
    strategy = DonchianStrategy(config)

    # 5. Data - Trigger BUY
    # Donchian needs enough data for window.
    # lower_length=5.
    # We provide 6 points.
    # Previous 5 max is High. Current close > Previous 5 Max.

    data = {
        'open': [100]*10,
        'high': [100, 101, 102, 103, 104, 104, 104, 104, 104, 105],
        'low': [90]*10,
        'close': [100, 101, 102, 103, 104, 104, 104, 104, 104, 110], # 110 > 104 (max of prev 5?)
        'volume': [1000]*10
    }
    df = pd.DataFrame(data)

    # Run analyze
    signal = await strategy.analyze(df)

    # Debug print if needed
    print(f"Signal: {signal}")

    assert signal.action == 'BUY'

    # 6. Execute BUY
    # PositionManager uses signal.quantity or default 0.001
    signal.quantity = 1.0 # Set explicit quantity

    await pm.process_signal(signal)

    # 7. Verify DB - Position Created
    positions = db.get_open_positions('BTC/USD')
    assert len(positions) == 1
    assert positions[0]['remaining_qty'] == 1.0
    assert positions[0]['status'] == 'OPEN'

    # Verify Execution Recorded (Check DB directly)
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM executions WHERE position_id = ?", (positions[0]['id'],))
        executions = cursor.fetchall()
        assert len(executions) == 1
        assert executions[0]['reason'] == 'ENTRY'

    # 8. Trigger SELL
    # Close < Min of prev 5
    data_sell = {
        'open': [100]*10,
        'high': [120]*10,
        'low': [110, 110, 110, 110, 110, 110, 110, 110, 110, 110],
        'close': [115, 115, 115, 115, 115, 115, 115, 115, 115, 90], # 90 < 110
        'volume': [1000]*10
    }
    df_sell = pd.DataFrame(data_sell)

    signal_sell = await strategy.analyze(df_sell)
    assert signal_sell.action == 'SELL'

    # Execute SELL
    # signal_sell.quantity is None (from strategy).
    # PositionManager should sell all remaining_qty (1.0)

    await pm.process_signal(signal_sell)

    # 9. Verify DB - Position Closed
    positions = db.get_open_positions('BTC/USD')
    assert len(positions) == 0 # Closed

    # Verify status is CLOSED in DB
    with db.get_connection() as conn:
        cursor = conn.cursor()
        # Get the position we created
        pos_id = executions[0]['position_id']
        cursor.execute("SELECT * FROM positions WHERE id = ?", (pos_id,))
        pos = cursor.fetchone()
        assert pos['status'] == 'CLOSED'
        assert pos['remaining_qty'] == 0.0
