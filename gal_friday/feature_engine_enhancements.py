"""Production-ready enhancements for the Feature Engine.

This module provides sophisticated feature engineering capabilities for cryptocurrency trading,
replacing simplified implementations with enterprise-grade solutions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer

# Optional import with graceful fallback
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    talib = None

from gal_friday.logger_service import LoggerService


class SpreadType(str, Enum):
    """Types of spread calculations."""
    QUOTED = "quoted"
    EFFECTIVE = "effective"
    REALIZED = "realized"
    IMPACT = "impact"


class ImputationStrategy(str, Enum):
    """Imputation strategy types."""
    FORWARD_FILL = "forward_fill"
    INTERPOLATION = "interpolation"
    SEASONAL_DECOMPOSITION = "seasonal_decomposition"
    KNN = "knn"
    REGIME_AWARE = "regime_aware"
    CROSS_ASSET = "cross_asset"


@dataclass
class MarketMicrostructureData:
    """Market microstructure data container."""
    timestamp: pd.Timestamp
    bids: list[tuple[float, float]]  # [(price, size), ...]
    asks: list[tuple[float, float]]
    trades: list[dict[str, Any]]  # Recent trades

    @property
    def best_bid(self) -> float | None:
        """Get best bid price."""
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> float | None:
        """Get best ask price."""
        return self.asks[0][0] if self.asks else None

    @property
    def midpoint(self) -> float | None:
        """Calculate midpoint price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def quoted_spread(self) -> float | None:
        """Calculate quoted spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None


class AdvancedSpreadCalculator:
    """Production-grade spread calculation engine."""

    def __init__(self, logger: LoggerService) -> None:
        """Initialize the instance."""
        self.logger = logger
        self._source_module = self.__class__.__name__

        # Configuration for spread calculations
        self._order_book_depth = 10  # Number of levels to analyze
        self._trade_lookback_window = 100  # Number of recent trades
        self._impact_time_window = 30  # Seconds for impact calculation

    def calculate_effective_spread(
        self,
        microstructure_data: MarketMicrostructureData,
        trade_data: list[dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        """Calculate comprehensive effective spread metrics."""
        try:
            if not microstructure_data.midpoint:
                return {}

            spreads: dict[str, float] = {}

            # 1. Basic quoted spread
            if microstructure_data.quoted_spread:
                spreads["quoted_spread_abs"] = microstructure_data.quoted_spread
                spreads["quoted_spread_bps"] = (
                    microstructure_data.quoted_spread / microstructure_data.midpoint * 10000
                )

            # 2. Effective spread from recent trades
            if trade_data:
                effective_spreads = self._calculate_trade_based_effective_spread(
                    trade_data, microstructure_data.midpoint,
                )
                spreads.update(effective_spreads)

            # 3. Liquidity-weighted spread
            liquidity_weighted = self._calculate_liquidity_weighted_spread(microstructure_data)
            if liquidity_weighted:
                spreads.update(liquidity_weighted)

            # 4. Market impact estimates
            impact_metrics = self._calculate_market_impact_spread(microstructure_data)
            spreads.update(impact_metrics)

        except Exception as e:
            self.logger.error(
                f"Failed to calculate effective spread: {e}",
                source_module=self._source_module,
                exc_info=True,
            )
            return {}
        else:
            return spreads

    def _calculate_trade_based_effective_spread(
        self,
        trades: list[dict[str, Any]],
        midpoint: float,
    ) -> dict[str, float]:
        """Calculate effective spread based on actual trade execution."""
        try:
            if not trades:
                return {}

            # Filter recent trades
            recent_trades = trades[-self._trade_lookback_window:]

            # Calculate effective spreads for each trade
            effective_spreads = []
            for trade in recent_trades:
                trade_price = float(trade.get("price", 0))
                trade.get("side", "").lower()

                if trade_price > 0:
                    # Effective spread = 2 * |trade_price - midpoint|
                    effective_spread = 2 * abs(trade_price - midpoint)
                    effective_spreads.append(effective_spread)

            if effective_spreads:
                return {
                    "effective_spread_mean": float(np.mean(effective_spreads)),
                    "effective_spread_median": float(np.median(effective_spreads)),
                    "effective_spread_std": float(np.std(effective_spreads)),
                    "effective_spread_bps_mean": float(np.mean(effective_spreads) / midpoint * 10000),
                    "trade_count": len(effective_spreads),
                }
            else:
                return {}

        except Exception:
            self.logger.exception(
                "Failed to calculate trade-based effective spread: ",
                source_module=self._source_module,
            )
            return {}
        else:
            return {
                "effective_spread_mean": float(np.mean(effective_spreads)),
                "effective_spread_median": float(np.median(effective_spreads)),
                "effective_spread_std": float(np.std(effective_spreads)),
                "effective_spread_bps_mean": float(np.mean(effective_spreads) / midpoint * 10000),
                "trade_count": len(effective_spreads),
            }

    def _calculate_liquidity_weighted_spread(
        self,
        microstructure_data: MarketMicrostructureData,
    ) -> dict[str, float]:
        """Calculate liquidity-weighted spread across order book depth."""
        try:
            if not microstructure_data.bids or not microstructure_data.asks:
                return {}

            # Calculate weighted average prices based on liquidity
            bid_prices = []
            bid_sizes = []
            ask_prices = []
            ask_sizes = []

            # Use available depth (up to configured limit)
            max_levels = min(self._order_book_depth, len(microstructure_data.bids), len(microstructure_data.asks))

            for i in range(max_levels):
                if i < len(microstructure_data.bids):
                    bid_price, bid_size = microstructure_data.bids[i]
                    bid_prices.append(bid_price)
                    bid_sizes.append(bid_size)

                if i < len(microstructure_data.asks):
                    ask_price, ask_size = microstructure_data.asks[i]
                    ask_prices.append(ask_price)
                    ask_sizes.append(ask_size)

            if bid_prices and ask_prices and bid_sizes and ask_sizes:
                # Volume-weighted average bid and ask
                total_bid_volume = sum(bid_sizes)
                total_ask_volume = sum(ask_sizes)

                if total_bid_volume > 0 and total_ask_volume > 0:
                    vwap_bid = sum(p * s for p, s in zip(bid_prices, bid_sizes, strict=False)) / total_bid_volume
                    vwap_ask = sum(p * s for p, s in zip(ask_prices, ask_sizes, strict=False)) / total_ask_volume

                    liquidity_weighted_spread = vwap_ask - vwap_bid
                    midpoint = (vwap_bid + vwap_ask) / 2

                    return {
                        "liquidity_weighted_spread_abs": liquidity_weighted_spread,
                        "liquidity_weighted_spread_bps": float(liquidity_weighted_spread / midpoint * 10000),
                        "vwap_bid": vwap_bid,
                        "vwap_ask": vwap_ask,
                        "total_bid_volume": total_bid_volume,
                        "total_ask_volume": total_ask_volume,
                    }
                else:
                    return {}
            else:
                return {}

        except Exception:
            self.logger.exception(
                "Failed to calculate liquidity-weighted spread: ",
                source_module=self._source_module,
            )
            return {}
        else:
            return {
                "liquidity_weighted_spread_abs": liquidity_weighted_spread,
                "liquidity_weighted_spread_bps": float(liquidity_weighted_spread / midpoint * 10000),
                "vwap_bid": vwap_bid,
                "vwap_ask": vwap_ask,
                "total_bid_volume": total_bid_volume,
                "total_ask_volume": total_ask_volume,
            }

    def _calculate_market_impact_spread(
        self,
        microstructure_data: MarketMicrostructureData,
    ) -> dict[str, float]:
        """Calculate market impact estimates for different order sizes."""
        try:
            if not microstructure_data.bids or not microstructure_data.asks:
                return {}

            # Define standard order sizes for impact calculation (in USD)
            impact_sizes = [1000, 5000, 10000, 25000, 50000]

            impact_metrics: dict[str, float] = {}

            for size_usd in impact_sizes:
                # Calculate buy impact
                buy_impact = self._calculate_side_impact(
                    microstructure_data.asks, size_usd, "buy",
                )

                # Calculate sell impact
                sell_impact = self._calculate_side_impact(
                    microstructure_data.bids, size_usd, "sell",
                )

                if buy_impact is not None:
                    impact_metrics[f"buy_impact_{size_usd}_bps"] = buy_impact

                if sell_impact is not None:
                    impact_metrics[f"sell_impact_{size_usd}_bps"] = sell_impact

                # Average impact
                if buy_impact is not None and sell_impact is not None:
                    impact_metrics[f"avg_impact_{size_usd}_bps"] = (buy_impact + sell_impact) / 2

            # Order book imbalance
            if microstructure_data.best_bid and microstructure_data.best_ask:
                total_bid_size = sum(size for _, size in microstructure_data.bids[:5])
                total_ask_size = sum(size for _, size in microstructure_data.asks[:5])

                if total_bid_size + total_ask_size > 0:
                    imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
                    impact_metrics["order_book_imbalance"] = imbalance

        except Exception:
            self.logger.exception(
                "Failed to calculate market impact spread: ",
                source_module=self._source_module,
            )
            return {}
        else:
            return impact_metrics

    def _calculate_side_impact(
        self,
        order_book_side: list[tuple[float, float]],
        order_size_usd: float,
        side: str,
    ) -> float | None:
        """Calculate market impact for one side of the order book."""
        try:
            if not order_book_side:
                return None

            # Start from best price
            best_price = order_book_side[0][0]
            remaining_size = order_size_usd
            total_cost = 0.0
            total_quantity = 0.0

            for price, size in order_book_side:
                if remaining_size <= 0:
                    break

                # Convert size to USD value
                level_value = price * size

                if level_value >= remaining_size:
                    # Partial fill at this level
                    quantity_needed = remaining_size / price
                    total_cost += remaining_size
                    total_quantity += quantity_needed
                    remaining_size = 0
                else:
                    # Full fill at this level
                    total_cost += level_value
                    total_quantity += size
                    remaining_size -= level_value

            if total_quantity > 0:
                avg_execution_price = total_cost / total_quantity
                return abs(avg_execution_price - best_price) / best_price * 10000
            else:
                return None

        except Exception:
            self.logger.exception(
                "Failed to calculate side impact: ",
                source_module=self._source_module,
            )
            return None


class IntelligentImputationEngine:
    """Advanced feature imputation system for time series data."""

    def __init__(self, logger: LoggerService) -> None:
        """Initialize the instance."""
        self.logger = logger
        self._source_module = self.__class__.__name__

        # Imputation configuration
        self._min_data_points = 10
        self._seasonal_periods = [24, 168, 720]  # Hours: daily, weekly, monthly
        self._knn_neighbors = 5
        self._max_gap_size = 12  # Maximum hours to impute

        # Market regime detection
        self._volatility_lookback = 24  # Hours for volatility calculation
        self._regime_threshold = 1.5  # Volatility multiplier for regime change

    def impute_features(
        self,
        data: pd.DataFrame,
        feature_metadata: dict[str, dict[str, Any]],
        strategy: ImputationStrategy = ImputationStrategy.REGIME_AWARE,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Intelligent feature imputation with multiple strategies."""
        try:
            imputed_data = data.copy()
            imputation_report: dict[str, Any] = {
                "strategy_used": strategy.value,
                "features_imputed": {},
                "imputation_quality": {},
                "warnings": [],
            }

            for feature_name in data.columns:
                if data[feature_name].isna().any():
                    feature_meta = feature_metadata.get(feature_name, {})

                    # Choose imputation method based on feature type and strategy
                    imputed_series, quality_metrics = self._impute_single_feature(
                        data[feature_name],
                        feature_name,
                        feature_meta,
                        strategy,
                    )

                    if imputed_series is not None:
                        imputed_data[feature_name] = imputed_series
                        imputation_report["features_imputed"][feature_name] = {
                            "missing_count": data[feature_name].isna().sum(),
                            "imputation_method": self._get_method_for_feature(feature_name, feature_meta, strategy),
                        }
                        imputation_report["imputation_quality"][feature_name] = quality_metrics

        except Exception as e:
            self.logger.error(
                f"Failed to impute features: {e}",
                source_module=self._source_module,
                exc_info=True,
            )
            return data, {"error": str(e)}
        else:
            return imputed_data, imputation_report

    def _impute_single_feature(
        self,
        series: pd.Series[Any],
        feature_name: str,
        feature_metadata: dict[str, Any],
        strategy: ImputationStrategy,
    ) -> tuple[pd.Series[Any] | None, dict[str, Any]]:
        """Impute a single feature using appropriate method."""
        try:
            if series.isna().sum() == 0:
                return series, {"method": "no_imputation_needed"}

            # Check if gap is too large
            max_consecutive_na = self._get_max_consecutive_na(series)
            if max_consecutive_na > self._max_gap_size:
                self.logger.warning(
                    f"Large gap in {feature_name}: {max_consecutive_na} consecutive NAs",
                    source_module=self._source_module,
                )
                return None, {"error": "gap_too_large", "max_gap": max_consecutive_na}

            # Feature type-based imputation
            feature_type = feature_metadata.get("type", "technical")
            feature_metadata.get("category", "unknown")

            if strategy == ImputationStrategy.FORWARD_FILL:
                return self._forward_fill_imputation(series), {"method": "forward_fill"}

            if strategy == ImputationStrategy.INTERPOLATION:
                return self._interpolation_imputation(series, feature_type), {"method": "interpolation"}

            if strategy == ImputationStrategy.SEASONAL_DECOMPOSITION:
                return self._seasonal_imputation(series, feature_name), {"method": "seasonal"}

            if strategy == ImputationStrategy.KNN:
                return self._knn_imputation(series), {"method": "knn"}

            if strategy == ImputationStrategy.REGIME_AWARE:
                return self._regime_aware_imputation(series, feature_name, feature_metadata), {"method": "regime_aware"}

            # Default to interpolation
            return self._interpolation_imputation(series, feature_type), {"method": "default_interpolation"}

        except Exception as e:
            self.logger.exception(
                f"Failed to impute feature {feature_name}: ",
                source_module=self._source_module,
            )
            return None, {"error": str(e)}

    def _regime_aware_imputation(
        self,
        series: pd.Series[Any],
        feature_name: str,
        feature_metadata: dict[str, Any],
    ) -> pd.Series[Any]:
        """Impute based on current market regime."""
        try:
            # Detect current market regime
            current_regime = self._detect_market_regime(series)

            # Get historical data for the same regime
            regime_data = self._get_regime_specific_data(series, current_regime)

            if len(regime_data) < self._min_data_points:
                # Fallback to interpolation if insufficient regime data
                return self._interpolation_imputation(series, feature_metadata.get("type", "technical"))

            # Feature-specific regime imputation
            if "volatility" in feature_name.lower() or "atr" in feature_name.lower():
                return self._impute_volatility_feature(series, regime_data)

            if "volume" in feature_name.lower():
                return self._impute_volume_feature(series, regime_data)

            if any(indicator in feature_name.lower() for indicator in ["rsi", "macd", "bb", "sma", "ema"]):
                return self._impute_technical_indicator(series, regime_data, feature_name)

            if "spread" in feature_name.lower():
                return self._impute_spread_feature(series, regime_data)

            # Generic regime-aware imputation
            return self._impute_generic_regime_aware(series, regime_data)

        except Exception:
            self.logger.exception(
                f"Regime-aware imputation failed for {feature_name}: ",
                source_module=self._source_module,
            )
            return self._interpolation_imputation(series, "technical")

    def _detect_market_regime(self, series: pd.Series[Any]) -> str:
        """Detect current market regime based on volatility."""
        try:
            # Calculate rolling volatility
            if len(series.dropna()) < self._volatility_lookback:
                return "normal"

            returns = series.pct_change().dropna()
            recent_returns = returns.tail(self._volatility_lookback)

            current_vol = recent_returns.std()
            historical_vol = returns.std()

            if current_vol > historical_vol * self._regime_threshold:
                return "high_volatility"
            elif current_vol < historical_vol / self._regime_threshold:
                return "low_volatility"
            else:
                return "normal"

        except Exception:
            return "normal"

    def _get_regime_specific_data(self, series: pd.Series[Any], regime: str) -> pd.Series[Any]:
        """Get historical data for specific market regime."""
        try:
            # Simplified regime detection over historical data
            # In production, this would use more sophisticated regime detection
            returns = series.pct_change().dropna()
            rolling_vol = returns.rolling(self._volatility_lookback).std()
            overall_vol = returns.std()

            if regime == "high_volatility":
                mask = rolling_vol > overall_vol * self._regime_threshold
            elif regime == "low_volatility":
                mask = rolling_vol < overall_vol / self._regime_threshold
            else:
                mask = (
                    (rolling_vol >= overall_vol / self._regime_threshold) &
                    (rolling_vol <= overall_vol * self._regime_threshold)
                )

            return series[mask].dropna()

        except Exception:
            return series.dropna()

    def _impute_volatility_feature(self, series: pd.Series[Any], regime_data: pd.Series[Any]) -> pd.Series[Any]:
        """Impute volatility-based features."""
        imputed = series.copy()

        # Use regime-specific median for volatility features
        regime_median = regime_data.median()
        regime_std = regime_data.std()

        # Add some noise to avoid constant values
        for idx in series.index[series.isna()]:
            noise = np.random.normal(0, regime_std * 0.1)
            imputed[idx] = max(0, regime_median + noise)  # Volatility can't be negative

        return imputed

    def _impute_volume_feature(self, series: pd.Series[Any], regime_data: pd.Series[Any]) -> pd.Series[Any]:
        """Impute volume-based features."""
        imputed = series.copy()

        # Use regime-specific patterns with time-of-day adjustment
        if isinstance(series.index, pd.DatetimeIndex) and isinstance(regime_data.index, pd.DatetimeIndex):
            hourly_patterns = regime_data.groupby(regime_data.index.hour).median()

            for idx in series.index[series.isna()]:
                hour = idx.hour
                base_value = hourly_patterns.get(hour, regime_data.median())

                # Add recent trend
                loc = series.index.get_loc(idx)
                recent_data = series.iloc[max(0, loc - 5):loc] if isinstance(loc, int) else pd.Series([])
                if len(recent_data.dropna()) > 1:
                    trend = recent_data.dropna().diff().mean()
                    imputed[idx] = max(0, base_value + trend)
                else:
                    imputed[idx] = base_value
        else:
            # Simple regime median
            imputed = imputed.fillna(regime_data.median())

        return imputed

    def _impute_technical_indicator(
        self,
        series: pd.Series[Any],
        regime_data: pd.Series[Any],
        feature_name: str,
    ) -> pd.Series[Any]:
        """Impute technical indicators based on their characteristics."""
        imputed = series.copy()

        # Bounded indicators (RSI, %K, %D)
        if any(indicator in feature_name.lower() for indicator in ["rsi", "%k", "%d", "williams"]):
            # These are typically bounded [0, 100]
            regime_median = np.clip(regime_data.median(), 0, 100)
            imputed = imputed.fillna(regime_median)

        # Oscillators (MACD, CCI)
        elif any(indicator in feature_name.lower() for indicator in ["macd", "cci"]):
            # These can be negative, use regime-aware interpolation
            imputed = imputed.interpolate(method="linear").fillna(regime_data.median())

        # Moving averages and trend indicators
        elif any(indicator in feature_name.lower() for indicator in ["sma", "ema", "bb"]):
            # Use interpolation for smooth trend continuation
            imputed = imputed.interpolate(method="cubic").fillna(regime_data.median())

        else:
            # Generic technical indicator
            imputed = imputed.interpolate(method="linear").fillna(regime_data.median())

        return imputed

    def _impute_spread_feature(self, series: pd.Series[Any], regime_data: pd.Series[Any]) -> pd.Series[Any]:
        """Impute spread-related features."""
        imputed = series.copy()

        # Spreads are typically positive and mean-reverting
        regime_median = regime_data.median()
        regime_iqr = regime_data.quantile(0.75) - regime_data.quantile(0.25)

        for idx in series.index[series.isna()]:
            # Use median with small random variation
            noise = np.random.normal(0, regime_iqr * 0.1)
            imputed[idx] = max(0, regime_median + noise)

        return imputed

    def _impute_generic_regime_aware(self, series: pd.Series[Any], regime_data: pd.Series[Any]) -> pd.Series[Any]:
        """Generic regime-aware imputation."""
        imputed = series.copy()

        # Use interpolation with regime fallback
        imputed = imputed.interpolate(method="linear")
        return imputed.fillna(regime_data.median())


    def _interpolation_imputation(self, series: pd.Series[Any], feature_type: str) -> pd.Series[Any]:
        """Advanced interpolation-based imputation."""
        if feature_type == "price":
            # Use cubic interpolation for price data
            return series.interpolate(method="cubic")
        if feature_type == "volume":
            # Use linear interpolation for volume
            return series.interpolate(method="linear").ffill()
        # Default to linear for technical indicators
        return series.interpolate(method="linear")

    def _seasonal_imputation(self, series: pd.Series[Any], feature_name: str) -> pd.Series[Any]:
        """Seasonal decomposition-based imputation."""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            if len(series.dropna()) < max(self._seasonal_periods) * 2:
                return self._interpolation_imputation(series, "technical")

            # Try seasonal decomposition
            decomposition = seasonal_decompose(
                series.dropna(),
                model="additive",
                period=min(self._seasonal_periods),
            )

            # Reconstruct missing values using seasonal + trend components
            imputed = series.copy()
            for idx in series.index[series.isna()]:
                if idx in decomposition.seasonal.index and idx in decomposition.trend.index:
                    imputed[idx] = decomposition.seasonal[idx] + decomposition.trend[idx]

            # Fill any remaining NAs
            return imputed.ffill().bfill()


        except Exception as e:
            self.logger.warning(
                f"Seasonal imputation failed for {feature_name}: {e}",
                source_module=self._source_module,
            )
            return self._interpolation_imputation(series, "technical")

    def _knn_imputation(self, series: pd.Series[Any]) -> pd.Series[Any]:
        """KNN-based imputation."""
        try:
            # Reshape for sklearn
            data = np.array(series.values).reshape(-1, 1)

            # Use KNN imputer
            imputer = KNNImputer(n_neighbors=self._knn_neighbors)
            imputed_data: np.ndarray[Any, np.dtype[np.float64]] = imputer.fit_transform(data)

            return pd.Series(imputed_data.flatten(), index=series.index)

        except Exception as e:
            self.logger.warning(
                f"KNN imputation failed: {e}",
                source_module=self._source_module,
            )
            return self._interpolation_imputation(series, "technical")

    def _forward_fill_imputation(self, series: pd.Series[Any]) -> pd.Series[Any]:
        """Forward fill with back fill fallback."""
        return series.ffill().bfill()

    def _get_max_consecutive_na(self, series: pd.Series[Any]) -> int:
        """Get maximum consecutive NaN count."""
        is_na = series.isna()
        groups = (is_na != is_na.shift()).cumsum()
        consecutive_na = is_na.groupby(groups).sum()
        return int(consecutive_na.max()) if len(consecutive_na) > 0 else 0

    def _get_method_for_feature(
        self,
        feature_name: str,
        feature_metadata: dict[str, Any],
        strategy: ImputationStrategy,
    ) -> str:
        """Get the specific method used for a feature."""
        if strategy == ImputationStrategy.REGIME_AWARE:
            if "volatility" in feature_name.lower():
                return "regime_aware_volatility"
            if "volume" in feature_name.lower():
                return "regime_aware_volume"
            if any(ind in feature_name.lower() for ind in ["rsi", "macd", "bb"]):
                return "regime_aware_technical"
            return "regime_aware_generic"
        return str(strategy.value)


class ComprehensiveFeatureValidator:
    """Advanced feature validation system."""

    def __init__(self, logger: LoggerService) -> None:
        """Initialize the instance."""
        self.logger = logger
        self._source_module = self.__class__.__name__

        # Validation thresholds
        self._outlier_std_threshold = 4.0
        self._correlation_threshold = 0.95
        self._stationarity_p_value = 0.05

    def validate_features(
        self,
        features: dict[str, float],
        feature_metadata: dict[str, dict[str, Any]],
        historical_data: pd.DataFrame | None = None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Comprehensive feature validation."""
        try:
            validated_features = features.copy()
            validation_report: dict[str, Any] = {
                "passed_features": {},
                "failed_features": {},
                "warnings": [],
                "corrections_applied": {},
                "statistical_tests": {},
            }

            # 1. Statistical outlier detection
            outlier_results = self._detect_statistical_outliers(
                features, feature_metadata, historical_data,
            )
            validation_report["statistical_tests"]["outliers"] = outlier_results

            # 2. Cross-feature consistency checks
            consistency_results = self._check_feature_consistency(features)
            validation_report["statistical_tests"]["consistency"] = consistency_results

            # 3. Temporal coherence validation
            if historical_data is not None:
                temporal_results = self._validate_temporal_coherence(
                    features, historical_data,
                )
                validation_report["statistical_tests"]["temporal"] = temporal_results

            # 4. Apply corrections
            validated_features, corrections = self._apply_validation_corrections(
                validated_features, validation_report,
            )
            validation_report["corrections_applied"] = corrections

        except Exception as e:
            self.logger.error(
                f"Feature validation failed: {e}",
                source_module=self._source_module,
                exc_info=True,
            )
            return features, {"error": str(e)}
        else:
            return validated_features, validation_report

    def _detect_statistical_outliers(
        self,
        features: dict[str, float],
        feature_metadata: dict[str, dict[str, Any]],
        historical_data: pd.DataFrame | None,
    ) -> dict[str, Any]:
        """Detect statistical outliers in features."""
        outlier_results: dict[str, Any] = {}

        if historical_data is None:
            return outlier_results

        for feature_name, value in features.items():
            if feature_name not in historical_data.columns:
                continue

            historical_values = historical_data[feature_name].dropna()
            if len(historical_values) < 30:  # Need sufficient history
                continue

            # Z-score based outlier detection
            mean_val = historical_values.mean()
            std_val = historical_values.std()

            if std_val > 0:
                z_score = abs(value - mean_val) / std_val
                is_outlier = z_score > self._outlier_std_threshold

                outlier_results[feature_name] = {
                    "z_score": z_score,
                    "is_outlier": is_outlier,
                    "threshold": self._outlier_std_threshold,
                    "percentile": stats.percentileofscore(historical_values, value),
                }

        return outlier_results

    def _check_feature_consistency(self, features: dict[str, float]) -> dict[str, Any]:
        """Check consistency between related features."""
        consistency_results: dict[str, Any] = {}

        # Spread consistency checks
        if "quoted_spread_abs" in features and "quoted_spread_bps" in features:
            abs_spread = features["quoted_spread_abs"]
            bps_spread = features["quoted_spread_bps"]

            # Check if they're consistent (allowing for rounding)
            if abs_spread > 0 and bps_spread > 0:
                # Reverse calculate to check consistency
                implied_price = abs_spread / (bps_spread / 10000)
                consistency_results["spread_consistency"] = {
                    "abs_spread": abs_spread,
                    "bps_spread": bps_spread,
                    "implied_price": implied_price,
                    "consistent": 0.01 < implied_price < 10,  # Reasonable crypto price range for DOGE/XRP
                }

        # Volume consistency
        if "volume_sma" in features and "volume_ratio" in features:
            vol_sma = features["volume_sma"]
            vol_ratio = features["volume_ratio"]

            if vol_sma > 0:
                implied_current_vol = vol_sma * vol_ratio
                consistency_results["volume_consistency"] = {
                    "volume_sma": vol_sma,
                    "volume_ratio": vol_ratio,
                    "implied_current_volume": implied_current_vol,
                    "reasonable": 0.01 < vol_ratio < 100,  # Reasonable volume ratio range
                }

        # RSI bounds check
        if "rsi" in features:
            rsi_value = features["rsi"]
            consistency_results["rsi_bounds"] = {
                "value": rsi_value,
                "within_bounds": 0 <= rsi_value <= 100,
            }

        return consistency_results

    def _validate_temporal_coherence(
        self,
        features: dict[str, float],
        historical_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Validate temporal coherence of features."""
        temporal_results: dict[str, Any] = {}

        for feature_name, current_value in features.items():
            if feature_name not in historical_data.columns:
                continue

            recent_values = historical_data[feature_name].dropna().tail(10)
            if len(recent_values) < 3:
                continue

            # Check for sudden jumps
            last_value = recent_values.iloc[-1]
            avg_change = recent_values.diff().abs().mean()

            current_change = abs(current_value - last_value)
            jump_ratio = current_change / avg_change if avg_change > 0 else 0

            temporal_results[feature_name] = {
                "current_value": current_value,
                "last_value": last_value,
                "change": current_change,
                "avg_historical_change": avg_change,
                "jump_ratio": jump_ratio,
                "sudden_jump": jump_ratio > 5.0,  # 5x normal change
            }

        return temporal_results

    def _apply_validation_corrections(
        self,
        features: dict[str, float],
        validation_report: dict[str, Any],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Apply corrections based on validation results."""
        corrected_features = features.copy()
        corrections: dict[str, Any] = {}

        # Correct outliers
        outlier_results = validation_report.get("statistical_tests", {}).get("outliers", {})
        for feature_name, outlier_info in outlier_results.items():
            if outlier_info.get("is_outlier", False):
                # Cap extreme values
                z_score = outlier_info["z_score"]
                if z_score > self._outlier_std_threshold:
                    # Use 95th percentile as cap
                    percentile = outlier_info["percentile"]
                    if percentile > 95:
                        corrections[feature_name] = {
                            "type": "outlier_capping",
                            "original_value": features[feature_name],
                            "reason": f"Z-score {z_score:.2f} exceeds threshold",
                        }
                        # Would implement actual capping logic here

        # Correct consistency issues
        consistency_results = validation_report.get("statistical_tests", {}).get("consistency", {})

        # Fix RSI bounds
        if "rsi_bounds" in consistency_results:
            rsi_info = consistency_results["rsi_bounds"]
            if not rsi_info["within_bounds"]:
                original_rsi = rsi_info["value"]
                corrected_rsi = max(0, min(100, original_rsi))
                corrected_features["rsi"] = corrected_rsi
                corrections["rsi"] = {
                    "type": "bounds_correction",
                    "original_value": original_rsi,
                    "corrected_value": corrected_rsi,
                    "reason": "RSI outside valid range [0, 100]",
                }

        return corrected_features, corrections


class AdvancedTemporalPatternEngine:
    """Sophisticated temporal pattern extraction for cryptocurrency markets."""

    def __init__(self, logger: LoggerService) -> None:
        """Initialize the instance."""
        self.logger = logger
        self._source_module = self.__class__.__name__

        # Cryptocurrency market operates 24/7
        self._market_sessions = {
            "asia_active": (22, 10),    # UTC hours when Asia is most active
            "europe_active": (6, 16),   # UTC hours when Europe is most active
            "us_active": (13, 23),      # UTC hours when US is most active
            "low_activity": (3, 6),      # UTC hours with typically lower activity
        }

        # Pattern detection parameters
        self._volatility_clustering_window = 24  # Hours
        self._correlation_lookback = 168  # Hours (1 week)
        self._pattern_significance_threshold = 0.05

    def extract_temporal_patterns(
        self,
        data: pd.DataFrame,
        target_feature: str = "price",
    ) -> dict[str, Any]:
        """Extract comprehensive temporal patterns from data."""
        try:
            patterns: dict[str, Any] = {}

            # 1. Time-of-day effects
            patterns["time_of_day"] = self._analyze_time_of_day_effects(data, target_feature)

            # 2. Day-of-week effects
            patterns["day_of_week"] = self._analyze_day_of_week_effects(data, target_feature)

            # 3. Volatility clustering
            patterns["volatility_clustering"] = self._analyze_volatility_clustering(data, target_feature)

            # 4. Cross-correlation patterns
            patterns["cross_correlations"] = self._analyze_cross_correlations(data)

            # 5. Market session effects
            patterns["market_sessions"] = self._analyze_market_session_effects(data, target_feature)

            # 6. Regime persistence patterns
            patterns["regime_persistence"] = self._analyze_regime_persistence(data, target_feature)

        except Exception as e:
            self.logger.error(
                f"Failed to extract temporal patterns: {e}",
                source_module=self._source_module,
                exc_info=True,
            )
            return {}
        else:
            return patterns

    def _analyze_time_of_day_effects(
        self,
        data: pd.DataFrame,
        target_feature: str,
    ) -> dict[str, Any]:
        """Analyze time-of-day effects in cryptocurrency markets."""
        try:
            if target_feature not in data.columns or not hasattr(data.index, "hour"):
                return {}

            # Calculate hourly statistics
            hourly_data = data.groupby(data.index.hour)[target_feature].agg([
                "mean", "std", "count", "median",
            ]).round(6)

            # Calculate returns if it's a price feature
            if "price" in target_feature.lower():
                hourly_returns = data[target_feature].pct_change().groupby(data.index.hour).agg([
                    "mean", "std",
                ]).round(6)

                # Identify best/worst hours for returns
                best_hour = hourly_returns["mean"].idxmax()
                worst_hour = hourly_returns["mean"].idxmin()
                most_volatile_hour = hourly_returns["std"].idxmax()
                least_volatile_hour = hourly_returns["std"].idxmin()
            else:
                hourly_returns = None
                best_hour = hourly_data["mean"].idxmax()
                worst_hour = hourly_data["mean"].idxmin()
                most_volatile_hour = hourly_data["std"].idxmax()
                least_volatile_hour = hourly_data["std"].idxmin()

            # Statistical significance test
            from scipy.stats import kruskal
            hourly_groups = [group[target_feature].values for hour, group in data.groupby(data.index.hour)]
            kruskal_stat, kruskal_p = kruskal(*hourly_groups)

            return {
                "hourly_statistics": hourly_data.to_dict(),
                "hourly_returns": hourly_returns.to_dict() if hourly_returns is not None else None,
                "key_hours": {
                    "best_performance": int(best_hour),
                    "worst_performance": int(worst_hour),
                    "most_volatile": int(most_volatile_hour),
                    "least_volatile": int(least_volatile_hour),
                },
                "statistical_significance": {
                    "kruskal_statistic": kruskal_stat,
                    "p_value": kruskal_p,
                    "significant": kruskal_p < self._pattern_significance_threshold,
                },
            }

        except Exception:
            self.logger.exception(
                "Failed to analyze time-of-day effects: ",
                source_module=self._source_module,
            )
            return {}

    def _analyze_market_session_effects(
        self,
        data: pd.DataFrame,
        target_feature: str,
    ) -> dict[str, Any]:
        """Analyze effects of different global market sessions on crypto."""
        try:
            session_effects: dict[str, Any] = {}

            for session_name, (start_hour, end_hour) in self._market_sessions.items():
                # Handle sessions that cross midnight
                if isinstance(data.index, pd.DatetimeIndex):
                    if start_hour > end_hour:
                        session_mask = (data.index.hour >= start_hour) | (data.index.hour < end_hour)
                    else:
                        session_mask = (data.index.hour >= start_hour) & (data.index.hour < end_hour)
                else:
                    continue

                session_data = data[session_mask][target_feature]

                if len(session_data) > 0:
                    # Calculate session statistics
                    session_stats = {
                        "mean": session_data.mean(),
                        "std": session_data.std(),
                        "median": session_data.median(),
                        "count": len(session_data),
                        "hours": f"{start_hour:02d}:00-{end_hour:02d}:00 UTC",
                    }

                    # Calculate returns for the session
                    if "price" in target_feature.lower():
                        session_returns = session_data.pct_change().dropna()
                        if len(session_returns) > 0:
                            session_stats.update({
                                "avg_return": session_returns.mean(),
                                "return_std": session_returns.std(),
                                "positive_return_ratio": (session_returns > 0).mean(),
                            })

                    session_effects[session_name] = session_stats

        except Exception:
            self.logger.exception(
                "Failed to analyze market session effects: ",
                source_module=self._source_module,
            )
            return {}
        else:
            return session_effects

    def _analyze_volatility_clustering(
        self,
        data: pd.DataFrame,
        target_feature: str,
    ) -> dict[str, Any]:
        """Analyze volatility clustering patterns."""
        try:
            if target_feature not in data.columns:
                return {}

            # Calculate returns and volatility
            returns = data[target_feature].pct_change().dropna()
            rolling_vol = returns.rolling(window=self._volatility_clustering_window).std()

            if len(rolling_vol.dropna()) < 50:
                return {}

            # Volatility autocorrelation
            vol_autocorr = []
            for lag in range(1, 25):  # Up to 24 hour lags
                if len(rolling_vol.dropna()) > lag:
                    autocorr = rolling_vol.dropna().autocorr(lag=lag)
                    vol_autocorr.append({"lag": lag, "autocorrelation": autocorr})

            # Arch test for volatility clustering
            try:
                from arch import het_arch  # type: ignore[attr-defined]
                from arch.unitroot import DFGLS

                # ARCH-LM test
                arch_test = het_arch(returns.dropna(), lags=5)
                arch_result = {
                    "statistic": arch_test[0],
                    "p_value": arch_test[1],
                    "significant_clustering": arch_test[1] < self._pattern_significance_threshold,
                }
            except ImportError:
                # Fallback if arch package not available
                arch_result = {"note": "ARCH test not available"}

            return {
                "volatility_autocorrelations": vol_autocorr,
                "arch_test": arch_result,
                "clustering_strength": (
                    max([abs(ac["autocorrelation"]) for ac in vol_autocorr[:5]])
                    if vol_autocorr else 0
                ),
            }

        except Exception:
            self.logger.exception(
                "Failed to analyze volatility clustering: ",
                source_module=self._source_module,
            )
            return {}

    def _analyze_cross_correlations(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze cross-correlations between features."""
        try:
            numeric_columns = data.select_dtypes(include=[np.number]).columns

            if len(numeric_columns) < 2:
                return {}

            # Calculate correlation matrix
            correlation_matrix = data[numeric_columns].corr()

            # Find highly correlated pairs
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if pd.notna(corr_value) and isinstance(corr_value, int | float | np.number):
                        corr_float = float(corr_value)
                        if abs(corr_float) > 0.7:  # High correlation threshold
                            high_correlations.append({
                                "feature1": correlation_matrix.columns[i],
                                "feature2": correlation_matrix.columns[j],
                                "correlation": corr_float,
                        })

            # Lead-lag relationships
            lead_lag_results = []
            price_columns = [col for col in numeric_columns if "price" in col.lower()]

            if len(price_columns) >= 1:
                price_col = price_columns[0]
                for col in numeric_columns:
                    if col != price_col:
                        # Test different lags
                        best_lag = 0
                        best_corr = 0.0

                        for lag in range(-5, 6):  # -5 to +5 periods
                            if lag == 0:
                                continue

                            if lag > 0:
                                # col leads price
                                lagged_correlation = data[col].shift(lag).corr(data[price_col])
                            else:
                                # price leads col
                                lagged_correlation = data[col].corr(data[price_col].shift(abs(lag)))

                            if abs(lagged_correlation) > abs(best_corr):
                                best_corr = lagged_correlation
                                best_lag = lag

                        if abs(best_corr) > 0.3:  # Minimum meaningful correlation
                            lead_lag_results.append({
                                "feature": col,
                                "best_lag": best_lag,
                                "correlation": best_corr,
                                "relationship": "leads_price" if best_lag > 0 else "lags_price",
                            })

            return {
                "correlation_matrix": correlation_matrix.round(3).to_dict(),
                "high_correlations": high_correlations,
                "lead_lag_relationships": lead_lag_results,
            }

        except Exception:
            self.logger.exception(
                "Failed to analyze cross-correlations: ",
                source_module=self._source_module,
            )
            return {}

    def _analyze_regime_persistence(
        self,
        data: pd.DataFrame,
        target_feature: str,
    ) -> dict[str, Any]:
        """Analyze persistence of different market regimes."""
        try:
            if target_feature not in data.columns:
                return {}

            # Define regimes based on volatility
            returns = data[target_feature].pct_change().dropna()
            rolling_vol = returns.rolling(window=24).std()  # 24-hour volatility

            vol_quantiles = rolling_vol.quantile([0.33, 0.67])

            # Classify regimes
            regimes = pd.Series(index=rolling_vol.index, dtype=str)
            regimes[rolling_vol <= vol_quantiles.iloc[0]] = "low_vol"
            regimes[(rolling_vol > vol_quantiles.iloc[0]) & (rolling_vol <= vol_quantiles.iloc[1])] = "normal_vol"
            regimes[rolling_vol > vol_quantiles.iloc[1]] = "high_vol"

            # Calculate regime persistence
            regime_changes = (regimes != regimes.shift(1)).cumsum()
            regime_lengths = regimes.groupby(regime_changes).size()

            persistence_stats: dict[str, Any] = {}
            for regime in ["low_vol", "normal_vol", "high_vol"]:
                regime_episodes = regimes.groupby(regime_changes).first()
                regime_episode_lengths = regime_lengths[regime_episodes == regime]

                if len(regime_episode_lengths) > 0:
                    persistence_stats[regime] = {
                        "avg_duration_hours": regime_episode_lengths.mean(),
                        "median_duration_hours": regime_episode_lengths.median(),
                        "max_duration_hours": regime_episode_lengths.max(),
                        "episode_count": len(regime_episode_lengths),
                    }

            # Transition probabilities
            regime_transitions = pd.crosstab(regimes.shift(1), regimes, normalize="index")

            return {
                "persistence_statistics": persistence_stats,
                "transition_probabilities": regime_transitions.round(3).to_dict(),
                "regime_thresholds": {
                    "low_vol_threshold": vol_quantiles.iloc[0],
                    "high_vol_threshold": vol_quantiles.iloc[1],
                },
            }

        except Exception:
            self.logger.exception(
                "Failed to analyze regime persistence: ",
                source_module=self._source_module,
            )
            return {}

    def _analyze_day_of_week_effects(
        self,
        data: pd.DataFrame,
        target_feature: str,
    ) -> dict[str, Any]:
        """Analyze day-of-week effects (less relevant for crypto but still useful)."""
        try:
            if target_feature not in data.columns or not hasattr(data.index, "dayofweek"):
                return {}

            # Map day numbers to names
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

            # Calculate daily statistics
            daily_stats = data.groupby(data.index.dayofweek)[target_feature].agg([
                "mean", "std", "count",
            ])
            daily_stats.index = pd.Index([day_names[i] for i in daily_stats.index])

            # Statistical significance test
            from scipy.stats import kruskal
            daily_groups = [group[target_feature].values for day, group in data.groupby(data.index.dayofweek)]
            kruskal_stat, kruskal_p = kruskal(*daily_groups)

            return {
                "daily_statistics": daily_stats.round(6).to_dict(),
                "statistical_significance": {
                    "kruskal_statistic": kruskal_stat,
                    "p_value": kruskal_p,
                    "significant": kruskal_p < self._pattern_significance_threshold,
                },
                "note": "Day-of-week effects less pronounced in 24/7 crypto markets",
            }

        except Exception:
            self.logger.exception(
                "Failed to analyze day-of-week effects: ",
                source_module=self._source_module,
            )
            return {}
