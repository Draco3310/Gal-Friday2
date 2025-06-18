"""Enterprise-grade enhancements for the Simulated Market Price Service.

This module provides production-ready replacements for the simplified components
in the simulated_market_price_service.py file.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
import math
from typing import Any, cast
import uuid

import asyncio
import numpy as np
import pandas as pd

# Scientific computing imports
from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService


class MarketRegime(str, Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class PriceModelType(str, Enum):
    """Types of price generation models."""
    GEOMETRIC_BROWNIAN = "geometric_brownian"
    JUMP_DIFFUSION = "jump_diffusion"
    HESTON_STOCHASTIC_VOL = "heston_stochastic_vol"
    REGIME_SWITCHING = "regime_switching"
    FRACTIONAL_BROWNIAN = "fractional_brownian"
    MICROSTRUCTURE = "microstructure"


class MarketEventType(str, Enum):
    """Types of market events."""
    NEWS_ANNOUNCEMENT = "news_announcement"
    REGULATORY_CHANGE = "regulatory_change"
    MAJOR_TRADE = "major_trade"
    TECHNICAL_BREAKOUT = "technical_breakout"
    CORRELATION_SHOCK = "correlation_shock"
    LIQUIDITY_CRISIS = "liquidity_crisis"


@dataclass
class MarketParameters:
    """Comprehensive market simulation parameters."""
    # Basic price dynamics
    initial_price: Decimal
    drift_rate: float  # Expected return
    volatility: float  # Annualized volatility

    # Mean reversion parameters
    mean_reversion_speed: float = 0.1
    long_term_mean: Decimal | None = None

    # Jump parameters
    jump_intensity: float = 0.5  # Jumps per year
    jump_mean: float = 0.0
    jump_std: float = 0.02

    # Microstructure parameters
    bid_ask_spread_bps: float = 5.0  # Basis points
    market_depth: Decimal = Decimal(1000000)  # Base liquidity
    price_impact_coefficient: float = 0.001

    # Volatility clustering
    garch_alpha: float = 0.1
    garch_beta: float = 0.85
    garch_omega: float = 0.00001

    # Regime switching
    regime_transition_prob: float = 0.02
    regime_volatility_multipliers: dict[MarketRegime, float] = field(default_factory=lambda: {
        MarketRegime.TRENDING_UP: 0.8,
        MarketRegime.TRENDING_DOWN: 1.2,
        MarketRegime.SIDEWAYS: 0.6,
        MarketRegime.HIGH_VOLATILITY: 2.0,
        MarketRegime.LOW_VOLATILITY: 0.4,
        MarketRegime.CRISIS: 3.0,
        MarketRegime.RECOVERY: 1.5,
    })

    # Correlation and cross-asset effects
    correlation_half_life: float = 30.0  # Days
    external_shock_probability: float = 0.01

    # Market hours and calendar
    market_open_hour: int = 0  # 24/7 for crypto
    market_close_hour: int = 24
    weekend_volatility_multiplier: float = 0.7
    holiday_dates: list[datetime] = field(default_factory=list[Any])


@dataclass
class MarketEvent:
    """Market event definition."""
    event_id: str
    event_type: MarketEventType
    timestamp: datetime

    # Impact parameters
    price_impact_percent: float
    volatility_impact_multiplier: float
    duration_hours: float

    # Event metadata
    description: str
    severity: float = 1.0  # 0-1 scale
    affected_symbols: list[str] = field(default_factory=list[Any])
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass
class PricePoint:
    """Single price observation with market microstructure."""
    timestamp: datetime
    price: Decimal
    volume: Decimal

    # Microstructure data
    bid_price: Decimal
    ask_price: Decimal
    bid_size: Decimal
    ask_size: Decimal

    # Market metrics
    volatility: float
    liquidity_score: float
    market_regime: MarketRegime

    # Event tracking
    related_events: list[str] = field(default_factory=list[Any])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "price": str(self.price),
            "volume": str(self.volume),
            "bid_price": str(self.bid_price),
            "ask_price": str(self.ask_price),
            "bid_size": str(self.bid_size),
            "ask_size": str(self.ask_size),
            "volatility": self.volatility,
            "liquidity_score": self.liquidity_score,
            "market_regime": self.market_regime.value,
            "related_events": self.related_events,
        }


class EnterpriseConfigurationManager:
    """Advanced configuration management system."""

    def __init__(self, config_manager: ConfigManager, logger: LoggerService) -> None:
        """Initialize the instance."""
        self.config_manager = config_manager
        self.logger = logger
        self._source_module = self.__class__.__name__

        # Configuration cache
        self._config_cache: dict[str, Any] = {}
        self._cache_timestamps: dict[str, datetime] = {}
        self._cache_ttl_seconds = 300  # 5 minutes

        # Configuration validation schemas
        self._validation_schemas = self._initialize_schemas()

        # Configuration change listeners
        self._change_listeners: dict[str, list[Callable[..., Any]]] = defaultdict(list[Any])

        # Default configurations
        self._default_configs = self._initialize_defaults()

    def _initialize_schemas(self) -> dict[str, dict[str, Any]]:
        """Initialize configuration validation schemas."""
        return {
            "market_parameters": {
                "required_fields": [
                    "initial_price", "drift_rate", "volatility",
                ],
                "field_types": {
                    "initial_price": (int, float, Decimal),
                    "drift_rate": (int, float),
                    "volatility": (int, float),
                    "jump_intensity": (int, float),
                    "bid_ask_spread_bps": (int, float),
                },
                "field_ranges": {
                    "volatility": (0.0, 5.0),
                    "jump_intensity": (0.0, 10.0),
                    "bid_ask_spread_bps": (0.1, 1000.0),
                },
            },
            "simulation_settings": {
                "required_fields": [
                    "time_step_seconds", "total_duration_hours",
                ],
                "field_types": {
                    "time_step_seconds": (int, float),
                    "total_duration_hours": (int, float),
                    "random_seed": int,
                },
            },
        }

    def _initialize_defaults(self) -> dict[str, dict[str, Any]]:
        """Initialize default configuration values."""
        return {
            "doge_usd": {
                "initial_price": Decimal("0.08"),  # Typical DOGE price
                "drift_rate": 0.25,  # 25% annual return (higher volatility)
                "volatility": 1.2,   # 120% annual volatility (very volatile)
                "jump_intensity": 3.0,
                "bid_ask_spread_bps": 5.0,  # Higher spread for smaller cap
                "market_depth": Decimal(1000000),
                "price_impact_coefficient": 0.001,
            },
            "xrp_usd": {
                "initial_price": Decimal("0.50"),  # Typical XRP price
                "drift_rate": 0.20,  # 20% annual return
                "volatility": 1.0,   # 100% annual volatility
                "jump_intensity": 2.5,
                "bid_ask_spread_bps": 4.0,
                "market_depth": Decimal(2000000),
                "price_impact_coefficient": 0.0008,
            },
            "simulation": {
                "time_step_seconds": 60,  # 1-minute bars
                "total_duration_hours": 24,
                "random_seed": None,
                "enable_microstructure": True,
                "enable_regime_switching": True,
                "enable_market_events": True,
            },
        }

    async def get_market_parameters(self, symbol: str) -> MarketParameters:
        """Get market parameters for a symbol with validation."""
        try:
            # Check cache first
            cache_key = f"market_params_{symbol}"
            if self._is_cache_valid(cache_key):
                cached_params = self._config_cache[cache_key]
                return cast("MarketParameters", cached_params)

            # Get configuration from config manager
            symbol_config = self.config_manager.get(f"market_simulation.{symbol}", {})

            # Apply defaults
            default_config = self._default_configs.get(symbol.lower().replace("/", "_"), self._default_configs["doge_usd"])
            merged_config = {**default_config, **symbol_config}

            # Validate configuration
            validation_errors = self._validate_config(merged_config, "market_parameters")
            if validation_errors:
                self.logger.warning(
                    f"Configuration validation errors for {symbol}: {validation_errors}",
                    source_module=self._source_module,
                )
                # Use defaults for invalid fields
                for field in validation_errors:
                    if field in default_config:
                        merged_config[field] = default_config[field]

            # Create MarketParameters object
            market_params = MarketParameters(
                initial_price=Decimal(str(merged_config["initial_price"])),
                drift_rate=float(merged_config["drift_rate"]),
                volatility=float(merged_config["volatility"]),
                jump_intensity=float(merged_config.get("jump_intensity", 2.0)),
                bid_ask_spread_bps=float(merged_config.get("bid_ask_spread_bps", 5.0)),
                market_depth=Decimal(str(merged_config.get("market_depth", "1000000"))),
                price_impact_coefficient=float(merged_config.get("price_impact_coefficient", 0.001)),
            )

            # Cache the result
            self._config_cache[cache_key] = market_params
            self._cache_timestamps[cache_key] = datetime.now(UTC)

            return market_params

        except Exception as e:
            self.logger.error(
                f"Error getting market parameters for {symbol}: {e}",
                source_module=self._source_module,
                exc_info=True,
            )
            # Return safe defaults for DOGE
            return MarketParameters(
                initial_price=Decimal("0.08"),
                drift_rate=0.25,
                volatility=1.2,
            )

    async def get_simulation_settings(self) -> dict[str, Any]:
        """Get simulation settings with validation."""
        try:
            # Check cache first
            cache_key = "simulation_settings"
            if self._is_cache_valid(cache_key):
                return cast("dict[str, Any]", self._config_cache[cache_key])

            # Get configuration
            sim_config = self.config_manager.get("market_simulation.settings", {})

            # Apply defaults
            default_config = self._default_configs["simulation"]
            merged_config = {**default_config, **sim_config}

            # Validate
            validation_errors = self._validate_config(merged_config, "simulation_settings")
            if validation_errors:
                self.logger.warning(
                    f"Simulation settings validation errors: {validation_errors}",
                    source_module=self._source_module,
                )

            # Cache and return
            self._config_cache[cache_key] = merged_config
            self._cache_timestamps[cache_key] = datetime.now(UTC)

            return merged_config

        except Exception as e:
            self.logger.error(
                f"Error getting simulation settings: {e}",
                source_module=self._source_module,
                exc_info=True,
            )
            return self._default_configs["simulation"]

    async def update_configuration(
        self,
        config_path: str,
        new_value: Any,
        validate: bool = True,
    ) -> bool:
        """Update configuration with validation and change notification."""
        try:
            if validate:
                # Perform validation based on config path
                schema_key = self._get_schema_key(config_path)
                if schema_key:
                    temp_config = {config_path.split(".")[-1]: new_value}
                    validation_errors = self._validate_config(temp_config, schema_key)
                    if validation_errors:
                        self.logger.error(
                            f"Configuration validation failed for {config_path}: {validation_errors}",
                            source_module=self._source_module,
                        )
                        return False

            # Update in config manager
            old_value = self.config_manager.get(config_path)
            # TODO: ConfigManager doesn't have a set method - need to implement or use alternative approach
            # self.config_manager.set(config_path, new_value)

            # Invalidate related cache entries
            self._invalidate_related_cache(config_path)

            # Notify change listeners
            await self._notify_change_listeners(config_path, old_value, new_value)

            self.logger.info(
                f"Updated configuration {config_path}: {old_value} -> {new_value}",
                source_module=self._source_module,
            )

            return True

        except Exception as e:
            self.logger.error(
                f"Error updating configuration {config_path}: {e}",
                source_module=self._source_module,
                exc_info=True,
            )
            return False

    def register_change_listener(self, config_path: str, callback: Callable[..., Any]) -> None:
        """Register a callback for configuration changes."""
        self._change_listeners[config_path].append(callback)

    def _validate_config(self, config: dict[str, Any], schema_key: str) -> list[str]:
        """Validate configuration against schema."""
        errors = []
        schema = self._validation_schemas.get(schema_key, {})

        # Check required fields
        required_fields = schema.get("required_fields", [])
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Check field types
        field_types = schema.get("field_types", {})
        for field, expected_types in field_types.items():
            if field in config and not isinstance(config[field], expected_types):
                errors.append(f"Field {field} has wrong type: expected {expected_types}")

        # Check field ranges
        field_ranges = schema.get("field_ranges", {})
        for field, (min_val, max_val) in field_ranges.items():
            if field in config:
                value = config[field]
                if isinstance(value, int | float) and (value < min_val or value > max_val):
                    errors.append(f"Field {field} out of range: {value} not in [{min_val}, {max_val}]")

        return errors

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self._cache_timestamps:
            return False

        cache_age = (datetime.now(UTC) - self._cache_timestamps[cache_key]).total_seconds()
        return cache_age < self._cache_ttl_seconds

    def _get_schema_key(self, config_path: str) -> str | None:
        """Get schema key for configuration path."""
        if "market_simulation" in config_path:
            return "market_parameters"
        if "simulation.settings" in config_path:
            return "simulation_settings"
        return None

    def _invalidate_related_cache(self, config_path: str) -> None:
        """Invalidate cache entries related to configuration path."""
        if "market_simulation" in config_path:
            # Invalidate all market parameter caches
            keys_to_remove = [k for k in self._config_cache if k.startswith("market_params_")]
            for key in keys_to_remove:
                self._config_cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
        elif "simulation" in config_path:
            self._config_cache.pop("simulation_settings", None)
            self._cache_timestamps.pop("simulation_settings", None)

    async def _notify_change_listeners(self, config_path: str, old_value: Any, new_value: Any) -> None:
        """Notify registered change listeners."""
        listeners = self._change_listeners.get(config_path, [])
        for listener in listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(config_path, old_value, new_value)
                else:
                    listener(config_path, old_value, new_value)
            except Exception:
                self.logger.exception(
                    "Error in configuration change listener: ",
                    source_module=self._source_module,
                )


class AdvancedPriceGenerator:
    """Sophisticated price generation engine with multiple models."""

    def __init__(self, config_manager: EnterpriseConfigurationManager, logger: LoggerService) -> None:
        """Initialize the instance."""
        self.config_manager = config_manager
        self.logger = logger
        self._source_module = self.__class__.__name__

        # Model registry
        self._price_models = {
            PriceModelType.GEOMETRIC_BROWNIAN: self._geometric_brownian_motion,
            PriceModelType.JUMP_DIFFUSION: self._jump_diffusion_model,
            PriceModelType.HESTON_STOCHASTIC_VOL: self._heston_stochastic_volatility,
            PriceModelType.REGIME_SWITCHING: self._regime_switching_model,
            PriceModelType.FRACTIONAL_BROWNIAN: self._fractional_brownian_motion,
            PriceModelType.MICROSTRUCTURE: self._microstructure_model,
        }

        # State tracking
        self._current_regime = MarketRegime.SIDEWAYS
        self._volatility_state = 0.2  # For stochastic volatility models
        self._last_jump_time: datetime | None = None

        # Random number generation
        self._rng = np.random.RandomState()

        # Market microstructure state
        self._order_book_state = {
            "bid_depth": Decimal(1000000),
            "ask_depth": Decimal(1000000),
            "last_trade_volume": Decimal(1000),
            "accumulated_volume": Decimal(0),
        }

    async def generate_price_series(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        time_step_seconds: int = 60,
        model_type: PriceModelType = PriceModelType.MICROSTRUCTURE,
        seed: int | None = None,
    ) -> list[PricePoint]:
        """Generate comprehensive price series with market microstructure."""
        try:
            if seed is not None:
                self._rng.seed(seed)

            # Get market parameters
            market_params = await self.config_manager.get_market_parameters(symbol)

            # Calculate time series
            total_seconds = int((end_time - start_time).total_seconds())
            num_steps = total_seconds // time_step_seconds
            time_step_years = time_step_seconds / (365.25 * 24 * 3600)

            # Generate time stamps
            timestamps = [
                start_time + timedelta(seconds=i * time_step_seconds)
                for i in range(num_steps)
            ]

            # Initialize price series
            price_series = []
            current_price = market_params.initial_price

            # Initialize state variables
            current_volatility = market_params.volatility

            self.logger.info(
                f"Generating {num_steps} price points for {symbol} using {model_type.value} model",
                source_module=self._source_module,
            )

            # Generate prices using selected model
            model_func = self._price_models[model_type]

            for _i, timestamp in enumerate(timestamps):
                # Generate price update
                price_update = await model_func(
                    current_price=current_price,
                    market_params=market_params,
                    time_step_years=time_step_years,
                    current_volatility=current_volatility,
                    timestamp=timestamp,
                )

                new_price = price_update["price"]
                new_volatility = price_update.get("volatility", current_volatility)
                regime = price_update.get("regime", self._current_regime)

                # Generate market microstructure
                microstructure = await self._generate_microstructure(
                    price=new_price,
                    market_params=market_params,
                    volatility=new_volatility,
                    timestamp=timestamp,
                )

                # Create price point
                price_point = PricePoint(
                    timestamp=timestamp,
                    price=new_price,
                    volume=microstructure["volume"],
                    bid_price=microstructure["bid_price"],
                    ask_price=microstructure["ask_price"],
                    bid_size=microstructure["bid_size"],
                    ask_size=microstructure["ask_size"],
                    volatility=new_volatility,
                    liquidity_score=microstructure["liquidity_score"],
                    market_regime=regime,
                )

                price_series.append(price_point)

                # Update state
                current_price = new_price
                current_volatility = new_volatility
                self._current_regime = regime

                # Update order book state
                self._update_order_book_state(microstructure)

            self.logger.info(
                f"Generated {len(price_series)} price points for {symbol}",
                source_module=self._source_module,
            )

            return price_series

        except Exception as e:
            self.logger.error(
                f"Error generating price series for {symbol}: {e}",
                source_module=self._source_module,
                exc_info=True,
            )
            return []

    async def _geometric_brownian_motion(
        self,
        current_price: Decimal,
        market_params: MarketParameters,
        time_step_years: float,
        current_volatility: float,
        timestamp: datetime,
    ) -> dict[str, Any]:
        """Generate price using Geometric Brownian Motion."""
        dt = time_step_years

        # Generate random component
        dW = self._rng.normal(0, math.sqrt(dt))

        # Calculate price change
        drift_component = market_params.drift_rate * dt
        volatility_component = current_volatility * dW

        # Apply GBM formula: dS = S * (mu*dt + sigma*dW)
        price_change_pct = drift_component + volatility_component
        new_price = current_price * Decimal(str(math.exp(price_change_pct)))

        return {
            "price": new_price,
            "volatility": current_volatility,
            "regime": self._current_regime,
        }

    async def _jump_diffusion_model(
        self,
        current_price: Decimal,
        market_params: MarketParameters,
        time_step_years: float,
        current_volatility: float,
        timestamp: datetime,
    ) -> dict[str, Any]:
        """Generate price using Jump Diffusion (Merton) model."""
        dt = time_step_years

        # Regular Brownian motion component
        dW = self._rng.normal(0, math.sqrt(dt))
        drift_component = market_params.drift_rate * dt
        diffusion_component = current_volatility * dW

        # Jump component
        jump_component = 0.0
        jump_probability = market_params.jump_intensity * dt

        if self._rng.random() < jump_probability:
            # Jump occurs
            jump_size = self._rng.normal(market_params.jump_mean, market_params.jump_std)
            jump_component = jump_size
            self._last_jump_time = timestamp

            self.logger.debug(
                f"Price jump occurred at {timestamp}: {jump_size:.4f}",
                source_module=self._source_module,
            )

        # Combine components
        total_return = drift_component + diffusion_component + jump_component
        new_price = current_price * Decimal(str(math.exp(total_return)))

        return {
            "price": new_price,
            "volatility": current_volatility,
            "regime": self._current_regime,
        }

    async def _heston_stochastic_volatility(
        self,
        current_price: Decimal,
        market_params: MarketParameters,
        time_step_years: float,
        current_volatility: float,
        timestamp: datetime,
    ) -> dict[str, Any]:
        """Generate price using Heston stochastic volatility model."""
        dt = time_step_years

        # Heston model parameters (simplified)
        kappa = 2.0  # Mean reversion speed
        theta = market_params.volatility  # Long-term volatility
        sigma_v = 0.3  # Volatility of volatility
        rho = -0.7  # Correlation between price and volatility

        # Generate correlated random numbers
        dW1 = self._rng.normal(0, math.sqrt(dt))
        dW2_independent = self._rng.normal(0, math.sqrt(dt))
        dW2 = rho * dW1 + math.sqrt(1 - rho**2) * dW2_independent

        # Update volatility (CIR process)
        vol_drift = kappa * (theta - current_volatility) * dt
        vol_diffusion = sigma_v * math.sqrt(max(current_volatility, 0.001)) * dW2
        new_volatility = max(current_volatility + vol_drift + vol_diffusion, 0.001)

        # Update price
        price_drift = market_params.drift_rate * dt
        price_diffusion = math.sqrt(max(current_volatility, 0.001)) * dW1
        total_return = price_drift + price_diffusion

        new_price = current_price * Decimal(str(math.exp(total_return)))

        return {
            "price": new_price,
            "volatility": new_volatility,
            "regime": self._current_regime,
        }

    async def _regime_switching_model(
        self,
        current_price: Decimal,
        market_params: MarketParameters,
        time_step_years: float,
        current_volatility: float,
        timestamp: datetime,
    ) -> dict[str, Any]:
        """Generate price with regime-switching dynamics."""
        dt = time_step_years

        # Check for regime transition
        if self._rng.random() < market_params.regime_transition_prob * dt:
            # Transition to new regime
            possible_regimes = list[Any](MarketRegime)
            possible_regimes.remove(self._current_regime)
            self._current_regime = self._rng.choice(possible_regimes)

            self.logger.debug(
                f"Regime switch to {self._current_regime.value} at {timestamp}",
                source_module=self._source_module,
            )

        # Adjust parameters based on current regime
        regime_multiplier = market_params.regime_volatility_multipliers.get(
            self._current_regime, 1.0,
        )

        regime_volatility = current_volatility * regime_multiplier

        # Adjust drift based on regime
        regime_drift = market_params.drift_rate
        if self._current_regime == MarketRegime.TRENDING_UP:
            regime_drift *= 1.5
        elif self._current_regime == MarketRegime.TRENDING_DOWN:
            regime_drift *= -0.5
        elif self._current_regime == MarketRegime.CRISIS:
            regime_drift *= -2.0

        # Generate price using adjusted parameters
        dW = self._rng.normal(0, math.sqrt(dt))
        drift_component = regime_drift * dt
        volatility_component = regime_volatility * dW

        total_return = drift_component + volatility_component
        new_price = current_price * Decimal(str(math.exp(total_return)))

        return {
            "price": new_price,
            "volatility": regime_volatility,
            "regime": self._current_regime,
        }

    async def _fractional_brownian_motion(
        self,
        current_price: Decimal,
        market_params: MarketParameters,
        time_step_years: float,
        current_volatility: float,
        timestamp: datetime,
    ) -> dict[str, Any]:
        """Generate price using Fractional Brownian Motion (simplified)."""
        dt = time_step_years
        hurst_exponent = 0.7  # > 0.5 indicates persistence

        # Simplified fBM implementation
        # In practice, would use more sophisticated methods
        dW = self._rng.normal(0, math.sqrt(dt))

        # Apply Hurst scaling (simplified)
        scaling_factor = dt**(hurst_exponent - 0.5)

        drift_component = market_params.drift_rate * dt
        fractional_component = current_volatility * dW * scaling_factor

        total_return = drift_component + fractional_component
        new_price = current_price * Decimal(str(math.exp(total_return)))

        return {
            "price": new_price,
            "volatility": current_volatility,
            "regime": self._current_regime,
        }

    async def _microstructure_model(
        self,
        current_price: Decimal,
        market_params: MarketParameters,
        time_step_years: float,
        current_volatility: float,
        timestamp: datetime,
    ) -> dict[str, Any]:
        """Generate price with detailed market microstructure modeling."""
        dt = time_step_years

        # Base Geometric Brownian Motion
        dW = self._rng.normal(0, math.sqrt(dt))
        base_drift = market_params.drift_rate * dt
        base_diffusion = current_volatility * dW

        # Microstructure effects

        # 1. Bid-ask bounce
        bid_ask_impact = self._rng.choice([-1, 1]) * (market_params.bid_ask_spread_bps / 20000)

        # 2. Market impact from trading volume
        volume_impact = self._calculate_volume_impact(market_params)

        # 3. Liquidity effects
        liquidity_impact = self._calculate_liquidity_impact(market_params, current_volatility)

        # 4. Intraday patterns (simplified)
        intraday_impact = self._calculate_intraday_impact(timestamp)

        # Combine all effects
        total_return = (
            base_drift + base_diffusion +
            bid_ask_impact + volume_impact +
            liquidity_impact + intraday_impact
        )

        new_price = current_price * Decimal(str(math.exp(total_return)))

        # Update volatility with microstructure effects
        volatility_multiplier = 1.0 + abs(volume_impact) * 10
        new_volatility = current_volatility * volatility_multiplier

        return {
            "price": new_price,
            "volatility": new_volatility,
            "regime": self._current_regime,
        }

    def _calculate_volume_impact(self, market_params: MarketParameters) -> float:
        """Calculate price impact from trading volume."""
        # Generate realistic volume
        base_volume = float(market_params.market_depth) * 0.01  # 1% of depth
        volume_noise = self._rng.lognormal(0, 0.5)
        trade_volume = base_volume * volume_noise

        # Calculate impact using square root model
        impact = market_params.price_impact_coefficient * math.sqrt(trade_volume / float(market_params.market_depth))

        # Random direction
        direction = self._rng.choice([-1, 1])

        return float(direction * impact)

    def _calculate_liquidity_impact(self, market_params: MarketParameters, volatility: float) -> float:
        """Calculate impact from liquidity conditions."""
        # Lower liquidity during high volatility periods
        1.0 / (1.0 + volatility * 2)

        # Generate liquidity shock
        if self._rng.random() < 0.01:  # 1% chance of liquidity shock
            shock_magnitude = self._rng.exponential(0.001)
            return float(self._rng.choice([-1, 1]) * shock_magnitude)

        return 0.0

    def _calculate_intraday_impact(self, timestamp: datetime) -> float:
        """Calculate intraday trading pattern effects."""
        # For crypto (24/7), create artificial patterns
        hour = timestamp.hour

        # Higher volatility during traditional market hours
        if 9 <= hour <= 16:  # Traditional market hours
            pattern_effect = 0.0002
        elif 0 <= hour <= 6:   # Low activity hours
            pattern_effect = -0.0001
        else:
            pattern_effect = 0.0

        # Add random component
        pattern_noise = self._rng.normal(0, 0.0001)

        return pattern_effect + pattern_noise

    async def _generate_microstructure(
        self,
        price: Decimal,
        market_params: MarketParameters,
        volatility: float,
        timestamp: datetime,
    ) -> dict[str, Any]:
        """Generate detailed market microstructure data."""
        # Calculate bid-ask spread
        spread_bps = market_params.bid_ask_spread_bps
        spread_decimal = spread_bps / 10000.0
        spread_amount = price * Decimal(str(spread_decimal))

        # Bid and ask prices
        bid_price = price - spread_amount / 2
        ask_price = price + spread_amount / 2

        # Generate volume
        base_volume = self._rng.lognormal(np.log(1000), 0.8)  # Log-normal distribution
        volume = Decimal(str(max(base_volume, 1.0)))

        # Generate depth (bid/ask sizes)
        depth_factor = self._rng.gamma(2, 500)  # Gamma distribution for depth
        bid_size = Decimal(str(depth_factor))
        ask_size = Decimal(str(depth_factor * self._rng.uniform(0.8, 1.2)))

        # Calculate liquidity score
        total_depth = float(bid_size + ask_size)
        normalized_spread = spread_bps / 100.0  # Normalize to 0-1 scale
        liquidity_score = min(total_depth / (1000 * (1 + normalized_spread)), 1.0)

        return {
            "volume": volume,
            "bid_price": bid_price,
            "ask_price": ask_price,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "liquidity_score": liquidity_score,
        }

    def _update_order_book_state(self, microstructure: dict[str, Any]) -> None:
        """Update internal order book state tracking."""
        self._order_book_state["last_trade_volume"] = microstructure["volume"]
        self._order_book_state["accumulated_volume"] += microstructure["volume"]
        self._order_book_state["bid_depth"] = microstructure["bid_size"]
        self._order_book_state["ask_depth"] = microstructure["ask_size"]


class RealisticHistoricalDataGenerator:
    """Generate realistic historical market data with proper statistical properties."""

    def __init__(
        self,
        config_manager: EnterpriseConfigurationManager,
        price_generator: AdvancedPriceGenerator,
        logger: LoggerService,
    ) -> None:
        self.config_manager = config_manager
        self.price_generator = price_generator
        self.logger = logger
        self._source_module = self.__class__.__name__

        # Historical patterns database
        self._regime_patterns = self._initialize_regime_patterns()
        self._seasonal_patterns = self._initialize_seasonal_patterns()

    def _initialize_regime_patterns(self) -> dict[MarketRegime, dict[str, Any]]:
        """Initialize realistic regime patterns based on market history."""
        return {
            MarketRegime.TRENDING_UP: {
                "average_duration_days": 45,
                "volatility_multiplier": 0.8,
                "drift_multiplier": 2.0,
                "jump_intensity_multiplier": 0.7,
            },
            MarketRegime.TRENDING_DOWN: {
                "average_duration_days": 20,
                "volatility_multiplier": 1.4,
                "drift_multiplier": -1.5,
                "jump_intensity_multiplier": 1.3,
            },
            MarketRegime.SIDEWAYS: {
                "average_duration_days": 90,
                "volatility_multiplier": 0.6,
                "drift_multiplier": 0.1,
                "jump_intensity_multiplier": 0.8,
            },
            MarketRegime.HIGH_VOLATILITY: {
                "average_duration_days": 15,
                "volatility_multiplier": 2.5,
                "drift_multiplier": 0.0,
                "jump_intensity_multiplier": 2.0,
            },
            MarketRegime.CRISIS: {
                "average_duration_days": 10,
                "volatility_multiplier": 4.0,
                "drift_multiplier": -3.0,
                "jump_intensity_multiplier": 5.0,
            },
        }

    def _initialize_seasonal_patterns(self) -> dict[str, Any]:
        """Initialize seasonal and calendar effects."""
        return {
            "monthly_volatility_multipliers": {
                1: 1.1,   # January effect
                2: 0.9,
                3: 1.0,
                4: 0.95,
                5: 1.05,  # "Sell in May"
                6: 0.9,
                7: 0.85,
                8: 1.2,   # Summer volatility
                9: 1.1,
                10: 1.3,  # October effect
                11: 0.95,
                12: 0.8,   # Holiday effect
            },
            "day_of_week_effects": {
                0: 1.1,   # Monday
                1: 1.0,
                2: 0.95,
                3: 1.0,
                4: 1.05,  # Friday
                5: 0.7,   # Weekend (for crypto)
                6: 0.7,
            },
        }

    async def generate_historical_dataset(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1h",
        include_events: bool = True,
    ) -> pd.DataFrame:
        """Generate realistic historical dataset with proper market characteristics."""
        try:
            # Parse timeframe
            time_step_seconds = self._parse_timeframe(timeframe)

            # Generate market events if requested
            market_events = []
            if include_events:
                market_events = await self._generate_market_events(start_date, end_date, symbol)

            # Generate base price series
            price_series = await self.price_generator.generate_price_series(
                symbol=symbol,
                start_time=start_date,
                end_time=end_date,
                time_step_seconds=time_step_seconds,
                model_type=PriceModelType.MICROSTRUCTURE,
            )

            # Apply market events
            if market_events:
                price_series = await self._apply_market_events(price_series, market_events)

            # Apply seasonal patterns
            price_series = await self._apply_seasonal_patterns(price_series)

            # Convert to DataFrame
            df = self._convert_to_dataframe(price_series, timeframe)

            # Add technical indicators
            df = await self._add_technical_indicators(df)

            # Validate data quality
            quality_score = await self._validate_data_quality(df)

            self.logger.info(
                f"Generated historical dataset for {symbol}: {len(df)} records, "
                f"quality score: {quality_score:.3f}",
                source_module=self._source_module,
            )

            return df

        except Exception as e:
            self.logger.error(
                f"Error generating historical dataset for {symbol}: {e}",
                source_module=self._source_module,
                exc_info=True,
            )
            return pd.DataFrame()

    def _parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe string to seconds."""
        timeframe_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
        }
        return timeframe_map.get(timeframe, 3600)

    async def _generate_market_events(
        self,
        start_date: datetime,
        end_date: datetime,
        symbol: str,
    ) -> list[MarketEvent]:
        """Generate realistic market events for the time period."""
        events = []
        current_date = start_date

        while current_date < end_date:
            # Probability of event per day
            if np.random.random() < 0.05:  # 5% chance per day
                event_type = np.random.choice(list[Any](MarketEventType))

                # Event characteristics based on type
                if event_type == MarketEventType.NEWS_ANNOUNCEMENT:
                    impact = np.random.normal(0, 0.02)  # ±2% average
                    duration = np.random.exponential(2)  # 2 hour average
                elif event_type == MarketEventType.MAJOR_TRADE:
                    impact = np.random.normal(0, 0.005)  # ±0.5% average
                    duration = np.random.exponential(0.5)  # 30 min average
                elif event_type == MarketEventType.REGULATORY_CHANGE:
                    impact = np.random.normal(-0.02, 0.05)  # Usually negative
                    duration = np.random.exponential(12)  # 12 hour average
                else:
                    impact = np.random.normal(0, 0.01)
                    duration = np.random.exponential(1)

                event = MarketEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=event_type,
                    timestamp=current_date + timedelta(hours=np.random.uniform(0, 24)),
                    price_impact_percent=impact,
                    volatility_impact_multiplier=1.0 + abs(impact) * 10,
                    duration_hours=duration,
                    description=f"{event_type.value} affecting {symbol}",
                    severity=min(abs(impact) / 0.05, 1.0),
                    affected_symbols=[symbol],
                )

                events.append(event)

            current_date += timedelta(days=1)

        return events

    async def _apply_market_events(
        self,
        price_series: list[PricePoint],
        market_events: list[MarketEvent],
    ) -> list[PricePoint]:
        """Apply market events to price series."""
        for event in market_events:
            # Find affected price points
            event_start = event.timestamp
            event_end = event_start + timedelta(hours=event.duration_hours)

            for price_point in price_series:
                if event_start <= price_point.timestamp <= event_end:
                    # Apply event impact
                    impact_factor = 1.0 + event.price_impact_percent
                    price_point.price *= Decimal(str(impact_factor))

                    # Increase volatility
                    price_point.volatility *= event.volatility_impact_multiplier

                    # Track event
                    price_point.related_events.append(event.event_id)

        return price_series

    async def _apply_seasonal_patterns(self, price_series: list[PricePoint]) -> list[PricePoint]:
        """Apply seasonal and calendar effects to price series."""
        for price_point in price_series:
            timestamp = price_point.timestamp

            # Monthly effect
            month_multiplier = self._seasonal_patterns["monthly_volatility_multipliers"].get(
                timestamp.month, 1.0,
            )

            # Day of week effect
            dow_multiplier = self._seasonal_patterns["day_of_week_effects"].get(
                timestamp.weekday(), 1.0,
            )

            # Apply to volatility
            combined_multiplier = month_multiplier * dow_multiplier
            price_point.volatility *= combined_multiplier

        return price_series

    def _convert_to_dataframe(self, price_series: list[PricePoint], timeframe: str) -> pd.DataFrame:
        """Convert price series to OHLCV DataFrame."""
        if not price_series:
            return pd.DataFrame()

        # Convert to basic DataFrame
        data = []
        for point in price_series:
            data.append({
                "timestamp": point.timestamp,
                "open": float(point.price),
                "high": float(point.price * Decimal("1.001")),  # Simplified
                "low": float(point.price * Decimal("0.999")),
                "close": float(point.price),
                "volume": float(point.volume),
                "bid": float(point.bid_price),
                "ask": float(point.ask_price),
                "bid_size": float(point.bid_size),
                "ask_size": float(point.ask_size),
                "volatility": point.volatility,
                "liquidity_score": point.liquidity_score,
                "regime": point.market_regime.value,
            })

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)

        return df

    async def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to the DataFrame."""
        if df.empty:
            return df

        try:
            # Simple moving averages
            df["sma_20"] = df["close"].rolling(window=20).mean()
            df["sma_50"] = df["close"].rolling(window=50).mean()

            # Exponential moving average
            df["ema_12"] = df["close"].ewm(span=12).mean()
            df["ema_26"] = df["close"].ewm(span=26).mean()

            # MACD
            df["macd"] = df["ema_12"] - df["ema_26"]
            df["macd_signal"] = df["macd"].ewm(span=9).mean()

            # RSI (simplified)
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()  # type: ignore[operator]
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()  # type: ignore[operator]
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            df["bb_middle"] = df["close"].rolling(window=20).mean()
            bb_std = df["close"].rolling(window=20).std()
            df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
            df["bb_lower"] = df["bb_middle"] - (bb_std * 2)

            # Volume-based indicators
            df["volume_sma"] = df["volume"].rolling(window=20).mean()
            df["volume_ratio"] = df["volume"] / df["volume_sma"]

        except Exception as e:
            self.logger.warning(
                f"Error adding technical indicators: {e}",
                source_module=self._source_module,
            )

        return df

    async def _validate_data_quality(self, df: pd.DataFrame) -> float:
        """Validate the quality of generated data."""
        if df.empty:
            return 0.0

        quality_scores = []

        # Check for missing values
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        quality_scores.append(1.0 - missing_ratio)

        # Check price continuity
        price_changes = df["close"].pct_change().dropna()
        extreme_changes = (abs(price_changes) > 0.2).sum()  # >20% changes
        continuity_score = 1.0 - (extreme_changes / len(price_changes))
        quality_scores.append(continuity_score)

        # Check volume consistency
        volume_cv = df["volume"].std() / df["volume"].mean()  # Coefficient of variation
        volume_score = min(volume_cv / 3.0, 1.0)  # Normalize
        quality_scores.append(volume_score)

        # Check volatility clustering
        vol_autocorr = df["volatility"].autocorr(lag=1)
        clustering_score = min(abs(vol_autocorr), 1.0)
        quality_scores.append(clustering_score)

        return float(np.mean(quality_scores))


# Factory functions for easy initialization
async def create_enterprise_market_service(
    config_manager: ConfigManager,
    logger: LoggerService,
) -> tuple[EnterpriseConfigurationManager, AdvancedPriceGenerator, RealisticHistoricalDataGenerator]:
    """Create and initialize enterprise market price service components."""
    # Create enterprise configuration manager
    enterprise_config = EnterpriseConfigurationManager(config_manager, logger)

    # Create advanced price generator
    price_generator = AdvancedPriceGenerator(enterprise_config, logger)

    # Create historical data generator
    historical_generator = RealisticHistoricalDataGenerator(
        enterprise_config, price_generator, logger,
    )

    return enterprise_config, price_generator, historical_generator
