import logging
from typing import Dict, Optional
from decimal import Decimal
from datetime import datetime

import pandas as pd  # Assuming pandas for historical data structure

# Set Decimal precision context if needed elsewhere, but usually handled globally or per-module
# from decimal import getcontext
# getcontext().prec = 28

log = logging.getLogger(__name__)


class SimulatedMarketPriceService:
    """Provides synchronous access to the latest market prices based on historical
    data during a backtest simulation.
    """

    def __init__(self, historical_data: Dict[str, pd.DataFrame]):
        """
        Initializes the service with historical market data.

        Args:
            historical_data: A dictionary where keys are trading pairs (e.g., "XRP/USD")
                             and values are pandas DataFrames containing OHLCV data
                             indexed by timestamp (UTC).
        """
        self.historical_data = historical_data
        self._current_timestamp: Optional[datetime] = None

        # Validate data format minimally
        for pair, df in historical_data.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                log.warning(f"Historical data for {pair} does not have a DatetimeIndex.")
            if "close" not in df.columns:  # Assuming we use 'close' price
                log.warning(f"Historical data for {pair} is missing 'close' column.")

        log.info("SimulatedMarketPriceService initialized.")

    def update_time(self, timestamp: datetime) -> None:
        """Updates the current simulation time."""
        log.debug(f"Updating simulated time to: {timestamp}")
        self._current_timestamp = timestamp

    def get_latest_price(self, trading_pair: str) -> Optional[Decimal]:
        """Synchronously gets the latest known price for a trading pair at the
        current simulation time.

        Args:
            trading_pair: The trading pair symbol (e.g., "XRP/USD").

        Returns:
            The 'close' price as a Decimal, or None if unavailable.
        """
        if self._current_timestamp is None:
            log.error("Cannot get latest price: Simulation time not set.")
            return None

        pair_data = self.historical_data.get(trading_pair)
        if pair_data is None:
            # Handle requests for conversion pairs like USD/USD
            if trading_pair.count("/") == 1:
                base, quote = trading_pair.split("/")
                if base == quote:
                    return Decimal("1.0")  # Price of a currency against itself is 1

            log.warning(f"No historical data found for trading pair: {trading_pair}")
            return None

        try:
            # Use loc to find the price at the current timestamp.
            # Handle cases where the exact timestamp might not be in the index.
            if self._current_timestamp in pair_data.index:
                # Fix for indexing issue - use get_loc to find the position in the index
                idx_pos = pair_data.index.get_loc(self._current_timestamp)
                price = pair_data.iloc[idx_pos]["close"]
                # Ensure the price is Decimal
                return Decimal(str(price))

            # Timestamp might fall between bars. Use the last known close price (asof).
            try:
                # Use pandas asof for efficient lookup of the latest value
                price = pair_data["close"].asof(self._current_timestamp)
                if pd.isna(price):
                    log.warning(
                        "Could not find price for {} at or before {} "
                        "(asof returned NaN).".format(
                            trading_pair, self._current_timestamp
                        )
                    )
                    return None
                return Decimal(str(price))
            except KeyError:
                # Handle cases where asof might fail if timestamp is before first index
                log.warning(
                    "Could not find price for {} at or before {} "
                    "(timestamp likely before data start).".format(
                        trading_pair, self._current_timestamp
                    )
                )
                return None

        except KeyError:
            log.error(
                f"'close' column not found for {trading_pair}, though DataFrame exists."
            )
            return None
        except Exception as e:
            log.exception(
                "Error retrieving latest price for {} at {}".format(
                    trading_pair, self._current_timestamp
                ),
                exc_info=e,
            )
            return None


# Example Usage
async def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create dummy historical data
    idx1 = pd.date_range(
        start="2023-01-01 00:00:00",
        periods=5,
        freq="1min",
        tz="UTC"
    )
    data1 = pd.DataFrame(
        {"open": [10, 11, 12, 11, 13], "close": [11, 12, 11, 13, 14]},
        index=idx1
    )
    idx2 = pd.date_range(
        start="2023-01-01 00:00:00",
        periods=5,
        freq="1min",
        tz="UTC"
    )
    data2 = pd.DataFrame(
        {"open": [1, 1, 2, 2, 1], "close": [1, 2, 2, 1, 1]},
        index=idx2
    )

    hist_data = {"BTC/USD": data1, "ETH/USD": data2}

    price_service = SimulatedMarketPriceService(hist_data)

    # Test 1: Get price at an exact timestamp
    import pytz  # Import needed for example
    ts1 = datetime(2023, 1, 1, 0, 1, 0, tzinfo=pytz.UTC)
    price_service.update_time(ts1)
    btc_price1 = price_service.get_latest_price("BTC/USD")
    eth_price1 = price_service.get_latest_price("ETH/USD")
    usd_price = price_service.get_latest_price("USD/USD")  # Test self-conversion
    print(f"Prices at {ts1}: BTC={btc_price1}, ETH={eth_price1}, USD/USD={usd_price}")

    # Test 2: Get price between timestamps (should get previous close)
    ts2 = datetime(2023, 1, 1, 0, 1, 30, tzinfo=pytz.UTC)  # Between 00:01 and 00:02
    price_service.update_time(ts2)
    btc_price2 = price_service.get_latest_price("BTC/USD")
    print(f"Prices at {ts2}: BTC={btc_price2} (Should be same as {ts1})")

    # Test 3: Get price before data starts
    ts3 = datetime(2022, 12, 31, 23, 59, 0, tzinfo=pytz.UTC)
    price_service.update_time(ts3)
    btc_price3 = price_service.get_latest_price("BTC/USD")
    print(f"Prices at {ts3}: BTC={btc_price3} (Should be None)")

    # Test 4: Unknown pair
    price_service.update_time(ts1)  # Reset time
    unknown_price = price_service.get_latest_price("LTC/USD")
    print(f"Prices at {ts1}: LTC={unknown_price} (Should be None)")


if __name__ == "__main__":
    # Note: Need pandas installed (`pip install pandas`)
    try:
        import pandas  # noqa: F811 # Needed for example
        import pytz  # noqa: F811 # Needed for example

        # Create dummy historical data within main scope
        idx1_main = pd.date_range(
            start="2023-01-01 00:00:00",
            periods=5,
            freq="1min",
            tz="UTC"
        )
        data1_main = pd.DataFrame(
            {
                "open": [10, 11, 12, 11, 13],
                "close": [
                    Decimal("11.0"),
                    Decimal("12.0"),
                    Decimal("11.5"),
                    Decimal("13.0"),
                    Decimal("14.0"),
                ],
            },
            index=idx1_main,
        )
        idx2_main = pd.date_range(
            start="2023-01-01 00:00:00",
            periods=5,
            freq="1min",
            tz="UTC"
        )
        data2_main = pd.DataFrame(
            {
                "open": [1, 1, 2, 2, 1],
                "close": [
                    Decimal("1.1"),
                    Decimal("2.2"),
                    Decimal("2.1"),
                    Decimal("1.5"),
                    Decimal("1.0"),
                ],
            },
            index=idx2_main,
        )
        hist_data_main = {"BTC/USD": data1_main, "ETH/USD": data2_main}
        price_service_main = SimulatedMarketPriceService(hist_data_main)
        ts1_main = datetime(2023, 1, 1, 0, 1, 0, tzinfo=pytz.UTC)
        price_service_main.update_time(ts1_main)
        print(f"BTC Price at {ts1_main}: {price_service_main.get_latest_price('BTC/USD')}")
        ts2_main = datetime(2023, 1, 1, 0, 1, 30, tzinfo=pytz.UTC)
        price_service_main.update_time(ts2_main)
        print(f"BTC Price at {ts2_main}: {price_service_main.get_latest_price('BTC/USD')}")
        print(
            f"USD/USD Price at {ts2_main}: {price_service_main.get_latest_price('USD/USD')}"
        )
    except ImportError:
        print("Could not run example: pandas and/or pytz not installed.")
