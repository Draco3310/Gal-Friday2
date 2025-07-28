"""Core interfaces for the Gal-Friday trading system.

This module contains abstract base classes that define the essential interfaces
that various components of the system must implement.
"""

from .core_interfaces import (
    ConfigManagerInterface,
    ExchangeInfoServiceInterface,
    LoggerServiceInterface,
    MonitoringServiceInterface,
    PortfolioManagerInterface,
    PubSubManagerInterface,
    RiskManagerInterface,
    StrategyArbitratorInterface,
)
from .execution_handler_interface import ExecutionHandlerInterface, ExecutionHandlerProtocol
from .feature_engine_interface import FeatureEngineInterface, FeatureEngineProtocol
from .historical_data_service_interface import HistoricalDataService
from .market_data_protocols import (
    MarketDataAggregatorProtocol,
    MarketDataCacheProtocol,
    MarketDataProviderProtocol,
    MarketDataStreamProtocol,
    MarketDataValidatorProtocol,
    MarketPriceServiceProtocol,
)
from .market_price_service_interface import MarketPriceService
from .ml_ai_protocols import (
    MLPipelineProtocol,
    ModelDeploymentProtocol,
    ModelEvaluationProtocol,
    ModelLifecycleProtocol,
    ModelMonitoringProtocol,
    ModelPredictionProtocol,
    ModelRegistryProtocol,
    ModelRetrainingProtocol,
    ModelTrainingProtocol,
)
from .predictor_interface import PredictorInterface, PredictorProtocol
from .service_protocol import ServiceProtocol
from .strategy_interface import (
    EnsembleStrategyInterface,
    EnsembleStrategyProtocol,
    MARLStrategyInterface,
    MARLStrategyProtocol,
    StrategyInterface,
    StrategyProtocol,
)
from .websocket_protocols import (
    WebSocketAuthenticationProtocol,
    WebSocketConnectionProtocol,
    WebSocketHeartbeatProtocol,
    WebSocketMessageHandlerProtocol,
    WebSocketMetricsProtocol,
    WebSocketOrderBookProtocol,
    WebSocketRateLimitProtocol,
    WebSocketReconnectProtocol,
    WebSocketStreamProtocol,
    WebSocketTradeStreamProtocol,
)

__all__ = [
    "ConfigManagerInterface",
    "EnsembleStrategyInterface",
    "EnsembleStrategyProtocol",
    "ExchangeInfoServiceInterface",
    "ExecutionHandlerInterface",
    "ExecutionHandlerProtocol",
    "FeatureEngineInterface",
    "FeatureEngineProtocol",
    "HistoricalDataService",
    "LoggerServiceInterface",
    "MARLStrategyInterface",
    "MARLStrategyProtocol",
    "MLPipelineProtocol",
    "MarketDataAggregatorProtocol",
    "MarketDataCacheProtocol",
    "MarketDataProviderProtocol",
    "MarketDataStreamProtocol",
    "MarketDataValidatorProtocol",
    "MarketPriceService",
    "MarketPriceServiceProtocol",
    "ModelDeploymentProtocol",
    "ModelEvaluationProtocol",
    "ModelLifecycleProtocol",
    "ModelMonitoringProtocol",
    "ModelPredictionProtocol",
    "ModelRegistryProtocol",
    "ModelRetrainingProtocol",
    "ModelTrainingProtocol",
    "MonitoringServiceInterface",
    "PortfolioManagerInterface",
    "PredictorInterface",
    "PredictorProtocol",
    "PubSubManagerInterface",
    "RiskManagerInterface",
    "ServiceProtocol",
    "StrategyArbitratorInterface",
    "StrategyInterface",
    "StrategyProtocol",
    "WebSocketAuthenticationProtocol",
    "WebSocketConnectionProtocol",
    "WebSocketHeartbeatProtocol",
    "WebSocketMessageHandlerProtocol",
    "WebSocketMetricsProtocol",
    "WebSocketOrderBookProtocol",
    "WebSocketRateLimitProtocol",
    "WebSocketReconnectProtocol",
    "WebSocketStreamProtocol",
    "WebSocketTradeStreamProtocol",
]
