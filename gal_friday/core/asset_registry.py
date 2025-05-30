"""Asset registry and abstraction layer for multi-asset, multi-exchange support.

This module provides a unified interface for managing different asset types and their
exchange-specific characteristics, enabling seamless expansion to stocks, futures, options,
and additional cryptocurrency exchanges.
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Protocol


class AssetType(Enum):
    """Enumeration of supported asset types."""
    CRYPTO = auto()
    STOCK = auto()
    FUTURES = auto()
    OPTIONS = auto()
    FOREX = auto()
    COMMODITY = auto()


class ExchangeType(Enum):
    """Enumeration of supported exchange types."""
    CRYPTO_EXCHANGE = auto()      # Kraken, Binance, Coinbase
    STOCK_EXCHANGE = auto()       # NYSE, NASDAQ, LSE
    FUTURES_EXCHANGE = auto()     # CME, ICE, EUREX
    OPTIONS_EXCHANGE = auto()     # CBOE, ISE
    FOREX_EXCHANGE = auto()       # OANDA, FXCM


@dataclass(frozen=True)
class TradingSession:
    """Trading session information for different markets."""
    open_time: time
    close_time: time
    timezone: str
    days_active: set[int] = field(default_factory=lambda: {0, 1, 2, 3, 4})  # Mon-Fri


@dataclass(frozen=True)
class AssetSpecification:
    """Comprehensive asset specification with exchange-specific details."""
    symbol: str                          # Universal symbol (e.g., "XRP/USD", "AAPL", "ES_202412")
    asset_type: AssetType
    base_asset: str | None = None     # For pairs/derivatives
    quote_asset: str | None = None    # For pairs
    underlying: str | None = None     # For derivatives

    # Trading specifications
    min_order_size: Decimal = Decimal("0.001")
    max_order_size: Decimal | None = None
    tick_size: Decimal = Decimal("0.01")
    lot_size: Decimal = Decimal("1")

    # Market characteristics
    typical_spread_bps: Decimal | None = None
    avg_daily_volume: Decimal | None = None
    market_cap: Decimal | None = None

    # Contract specifications (for derivatives)
    contract_size: Decimal | None = None
    expiry_date: datetime | None = None
    strike_price: Decimal | None = None
    option_type: str | None = None  # "call", "put"

    # Exchange-specific metadata
    exchange_symbol: str = ""            # Exchange-native symbol format
    exchange_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExchangeSpecification:
    """Exchange-specific trading rules and capabilities."""
    exchange_id: str
    exchange_type: ExchangeType
    name: str

    # Trading capabilities
    supports_limit_orders: bool = True
    supports_market_orders: bool = True
    supports_stop_orders: bool = False
    supports_bracket_orders: bool = False
    supports_margin: bool = False

    # Fee structure
    maker_fee_bps: Decimal = Decimal("0")
    taker_fee_bps: Decimal = Decimal("0")
    fee_currency: str = "USD"

    # Market data capabilities
    provides_l2_data: bool = True
    provides_tick_data: bool = False
    provides_options_chain: bool = False
    max_market_data_depth: int = 20

    # Trading sessions
    trading_sessions: list[TradingSession] = field(default_factory=list)

    # Rate limits and constraints
    max_orders_per_second: int | None = None
    max_message_rate: int | None = None

    # API characteristics
    supports_websocket: bool = True
    supports_private_ws: bool = False
    typical_latency_ms: float | None = None


class AssetRegistryProtocol(Protocol):
    """Protocol for asset registry implementations."""

    def register_asset(self, asset: AssetSpecification, exchange_id: str) -> None:
        """Register an asset with exchange-specific details."""
        ...

    def get_asset(self, symbol: str, exchange_id: str) -> AssetSpecification | None:
        """Get asset specification for a specific exchange."""
        ...

    def get_supported_assets(self, exchange_id: str) -> list[AssetSpecification]:
        """Get all assets supported by an exchange."""
        ...

    def normalize_symbol(self, symbol: str, from_exchange: str, to_exchange: str) -> str:
        """Convert symbol format between exchanges."""
        ...


class AssetRegistry:
    """Central registry for asset and exchange specifications."""

    def __init__(self) -> None:
        self._assets: dict[str, dict[str, AssetSpecification]] = {}  # exchange_id -> symbol -> spec
        self._exchanges: dict[str, ExchangeSpecification] = {}
        self._symbol_mappings: dict[str, dict[str, dict[str, str]]] = {}  # from_exchange -> to_exchange -> symbol_map

    def register_exchange(self, exchange: ExchangeSpecification) -> None:
        """Register an exchange with its specifications."""
        self._exchanges[exchange.exchange_id] = exchange

    def register_asset(self, asset: AssetSpecification, exchange_id: str) -> None:
        """Register an asset with exchange-specific details."""
        if exchange_id not in self._exchanges:
            raise ValueError(f"Exchange {exchange_id} not registered")

        if exchange_id not in self._assets:
            self._assets[exchange_id] = {}

        self._assets[exchange_id][asset.symbol] = asset

    def get_asset(self, symbol: str, exchange_id: str) -> AssetSpecification | None:
        """Get asset specification for a specific exchange."""
        return self._assets.get(exchange_id, {}).get(symbol)

    def get_supported_assets(self, exchange_id: str) -> list[AssetSpecification]:
        """Get all assets supported by an exchange."""
        return list(self._assets.get(exchange_id, {}).values())

    def get_exchange(self, exchange_id: str) -> ExchangeSpecification | None:
        """Get exchange specification."""
        return self._exchanges.get(exchange_id)

    def get_all_exchanges(self) -> list[ExchangeSpecification]:
        """Get all registered exchanges."""
        return list(self._exchanges.values())

    def normalize_symbol(self, symbol: str, from_exchange: str, to_exchange: str) -> str:
        """Convert symbol format between exchanges."""
        # Check if the exchange exists, return original symbol if not
        if from_exchange not in self._symbol_mappings:
            return symbol

        # Get the to_exchange mapping, return original if not found
        exchange_mappings = self._symbol_mappings[from_exchange]
        if to_exchange not in exchange_mappings:
            return symbol

        # Get the symbol mapping dict
        symbol_mapping_dict = exchange_mappings[to_exchange]
        return symbol_mapping_dict.get(symbol, symbol)  # Return original if no mapping

    def add_symbol_mapping(self, from_exchange: str, to_exchange: str,
                          symbol_map: dict[str, str]) -> None:
        """Add symbol format mappings between exchanges."""
        if from_exchange not in self._symbol_mappings:
            self._symbol_mappings[from_exchange] = {}
        self._symbol_mappings[from_exchange][to_exchange] = symbol_map

    def get_assets_by_type(self, asset_type: AssetType,
                          exchange_id: str | None = None) -> list[AssetSpecification]:
        """Get all assets of a specific type, optionally filtered by exchange."""
        assets = []
        exchanges_to_check = [exchange_id] if exchange_id else self._assets.keys()

        for exch_id in exchanges_to_check:
            for asset in self._assets.get(exch_id, {}).values():
                if asset.asset_type == asset_type:
                    assets.append(asset)

        return assets

    def validate_trading_pair(self, symbol: str, exchange_id: str) -> bool:
        """Validate if a trading pair is supported on an exchange."""
        asset = self.get_asset(symbol, exchange_id)
        exchange = self.get_exchange(exchange_id)

        if not asset or not exchange:
            return False

        # Add more sophisticated validation based on asset type and exchange capabilities
        if asset.asset_type == AssetType.OPTIONS and not exchange.provides_options_chain:
            return False

        return True


# Global asset registry instance
asset_registry = AssetRegistry()


def initialize_default_assets() -> None:
    """Initialize the registry with default assets and exchanges."""
    # Register Kraken exchange
    kraken_sessions = [
        TradingSession(
            open_time=time(0, 0),
            close_time=time(23, 59),
            timezone="UTC",
            days_active={0, 1, 2, 3, 4, 5, 6},  # 24/7 for crypto
        ),
    ]

    kraken_exchange = ExchangeSpecification(
        exchange_id="kraken",
        exchange_type=ExchangeType.CRYPTO_EXCHANGE,
        name="Kraken",
        supports_stop_orders=True,
        maker_fee_bps=Decimal("16"),  # 0.16%
        taker_fee_bps=Decimal("26"),  # 0.26%
        trading_sessions=kraken_sessions,
        max_orders_per_second=10,
        typical_latency_ms=50.0,
    )

    asset_registry.register_exchange(kraken_exchange)

    # Register crypto assets
    xrp_usd = AssetSpecification(
        symbol="XRP/USD",
        asset_type=AssetType.CRYPTO,
        base_asset="XRP",
        quote_asset="USD",
        min_order_size=Decimal("1"),
        tick_size=Decimal("0.0001"),
        exchange_symbol="XRPUSD",
    )

    doge_usd = AssetSpecification(
        symbol="DOGE/USD",
        asset_type=AssetType.CRYPTO,
        base_asset="DOGE",
        quote_asset="USD",
        min_order_size=Decimal("1"),
        tick_size=Decimal("0.00001"),
        exchange_symbol="DOGEUSD",
    )

    asset_registry.register_asset(xrp_usd, "kraken")
    asset_registry.register_asset(doge_usd, "kraken")

    # Future expansion examples (commented for now)
    """
    # Stock exchange example
    nasdaq_exchange = ExchangeSpecification(
        exchange_id="nasdaq",
        exchange_type=ExchangeType.STOCK_EXCHANGE,
        name="NASDAQ",
        supports_bracket_orders=True,
        maker_fee_bps=Decimal("0.5"),
        taker_fee_bps=Decimal("0.5"),
        trading_sessions=[
            TradingSession(
                open_time=time(9, 30),
                close_time=time(16, 0),
                timezone="US/Eastern",
                days_active={0, 1, 2, 3, 4}
            )
        ]
    )
    
    # Options exchange example
    cboe_exchange = ExchangeSpecification(
        exchange_id="cboe",
        exchange_type=ExchangeType.OPTIONS_EXCHANGE,
        name="CBOE",
        provides_options_chain=True,
        supports_bracket_orders=True
    )
    """


# Initialize default assets when module is imported
initialize_default_assets()
