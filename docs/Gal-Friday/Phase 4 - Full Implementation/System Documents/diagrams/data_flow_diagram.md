# Gal-Friday System Data Flow Diagram

## Introduction

This diagram illustrates the primary data flows and component interactions within the Gal-Friday trading system. It provides a visual representation of how data is ingested, processed, and acted upon, and how various services communicate with each other, primarily through an event-driven architecture facilitated by the PubSubManager, as well as direct interactions and database usage.

## Diagram

```mermaid
graph LR
    %% External World
    subgraph External
        KrakenAPI[Kraken Exchange API<br/>(REST & WebSocket)]
    end

    %% Core Data Flow & Trading Logic
    subgraph TradingSystem[Gal-Friday Trading System]

        %% Configuration
        ConfigManager["ConfigManager<br/>(config.yaml)"]

        %% Databases
        subgraph Databases
            PostgreSQL["PostgreSQL DB<br/>(SQLAlchemy ORM)"]
            InfluxDB["InfluxDB<br/>(Time-Series Data)"]
        end

        %% Core Services & Event Bus
        PubSubManager["PubSubManager<br/>(Event Bus)"]
        LoggerService["LoggerService"]
        MonitoringService["MonitoringService"]
        HaltCoordinator["HaltCoordinator"]
        CLIService["CLIService"]

        %% Data Ingestion & Feature Engineering
        subgraph IngestionAndFeatures["Data Ingestion & Features"]
            DataIngestor["DataIngestor"]
            FeatureEngine["FeatureEngine"]
            HistoricalDataService["HistoricalDataService<br/>(e.g., KrakenHistorical)"]
        end

        %% Prediction
        subgraph Prediction["Prediction"]
            PredictionService["PredictionService"]
            ProcessPool["ProcessPoolExecutor<br/>(for Model Inference)"]
            ModelRegistry["ModelLifecycle::Registry"]
            ModelArtifacts["Model Artifacts<br/>(Local/Cloud Storage)"]
        end

        %% Trading Strategy & Risk
        subgraph StrategyAndRisk["Strategy & Risk"]
            StrategyArbitrator["StrategyArbitrator"]
            RiskManager["RiskManager"]
        end

        %% Execution & Portfolio
        subgraph ExecutionAndPortfolio["Execution & Portfolio"]
            ExecutionHandler["ExecutionHandler<br/>(e.g., KrakenExecutionHandler)"]
            PortfolioManager["PortfolioManager"]
            subgraph PortfolioSubComponents["Portfolio Sub-Components"]
                FundsManager["FundsManager"]
                PositionManager["PositionManager"]
                ValuationService["ValuationService"]
                MarketPriceService["MarketPriceService<br/>(e.g., KrakenMarketPrice)"]
            end
        end

        %% MLOps - Model Lifecycle
        subgraph MLOps["MLOps - Model Lifecycle"]
            ModelRegistry --> ModelArtifacts
            ExperimentManager["ModelLifecycle::ExperimentManager"]
            RetrainingPipeline["ModelLifecycle::RetrainingPipeline"]
            DriftDetector["ModelLifecycle::DriftDetector"]
        end

        %% DAL - Data Access Layer (conceptual, repositories use DBs)
        subgraph DAL["Data Access Layer (Repositories)"]
            OrderRepository["OrderRepository"]
            PositionRepository["PositionRepository"]
            LogRepository["LogRepository"]
            EventLogRepository["EventLogRepository"]
            ModelMetadataRepository["ModelMetadataRepository"]
        end

    end

    %% == Interactions ==

    %% Configuration Loading
    ConfigManager --> DataIngestor;
    ConfigManager --> FeatureEngine;
    ConfigManager --> PredictionService;
    ConfigManager --> StrategyArbitrator;
    ConfigManager --> RiskManager;
    ConfigManager --> ExecutionHandler;
    ConfigManager --> PortfolioManager;
    ConfigManager --> LoggerService;
    ConfigManager --> MonitoringService;
    ConfigManager --> ModelRegistry;
    ConfigManager --> ExperimentManager;
    ConfigManager --> RetrainingPipeline;
    ConfigManager --> HistoricalDataService;
    ConfigManager --> MarketPriceService;
    ConfigManager --> Databases; %% For DB URLs

    %% Data Ingestion Flow
    KrakenAPI -- "Real-time Market Data (WebSocket)" --> DataIngestor;
    DataIngestor -- "MarketDataOHLCVEvent<br/>MarketDataL2Event<br/>MarketDataTradeEvent" --> PubSubManager;
    HistoricalDataService -- "Historical OHLCV" --> FeatureEngine; %% For bootstrapping/backtesting
    KrakenAPI -- "Historical Data (REST)" --> HistoricalDataService;

    %% Feature Engineering
    PubSubManager -- "MarketData Events" --> FeatureEngine;
    FeatureEngine -- "FeatureEvent" --> PubSubManager;
    FeatureEngine -- "OHLCV History, L2 Books, Trades" --> Databases; %% (Conceptual, might store processed data)

    %% Prediction Flow
    PubSubManager -- "FeatureEvent" --> PredictionService;
    PredictionService -- "Loads Model Details" --> ModelRegistry;
    ModelRegistry -- "Fetches Artifact Path" --> ModelArtifacts;
    PredictionService -- "Submits Inference Task (Features, Model Path)" --> ProcessPool;
    ProcessPool -- "Executes model.predict()" --> PredictionService; %% (Result via Future)
    PredictionService -- "PredictionEvent" --> PubSubManager;
    PredictionService -- "Active Experiments Info" --> ExperimentManager;
    ExperimentManager -- "Routes to Variants (Conceptual)" --> PredictionService;


    %% Strategy & Risk Flow
    PubSubManager -- "PredictionEvent" --> StrategyArbitrator;
    StrategyArbitrator -- "Queries Current Price" --> MarketPriceService;
    StrategyArbitrator -- "TradeSignalProposedEvent" --> PubSubManager;

    PubSubManager -- "TradeSignalProposedEvent" --> RiskManager;
    RiskManager -- "Queries Portfolio State (Equity, Positions, Drawdown)" --> PortfolioManager;
    RiskManager -- "Queries Current Price" --> MarketPriceService;
    RiskManager -- "TradeSignalApprovedEvent<br/>TradeSignalRejectedEvent" --> PubSubManager;

    %% Execution Flow
    PubSubManager -- "TradeSignalApprovedEvent" --> ExecutionHandler; %% (via OrderExecutionManager if exists)
    ExecutionHandler -- "Submit/Cancel Order (REST)" --> KrakenAPI;
    KrakenAPI -- "Order Ack/Fill (REST/WebSocket)" --> ExecutionHandler;
    ExecutionHandler -- "ExecutionReportEvent" --> PubSubManager;
    ExecutionHandler -- "Queries Account Balances/Positions (REST)" --> KrakenAPI; %% For reconciliation or direct queries

    %% Portfolio Management
    PubSubManager -- "ExecutionReportEvent" --> PortfolioManager;
    PortfolioManager -- "Updates" --> FundsManager;
    PortfolioManager -- "Updates" --> PositionManager;
    PortfolioManager -- "Triggers Valuation" --> ValuationService;
    ValuationService -- "Queries Current Prices" --> MarketPriceService;
    ValuationService -- "Gets Balances/Positions" --> FundsManager;
    ValuationService -- "Gets Balances/Positions" --> PositionManager;
    PortfolioManager -- "Persists Snapshots/Trades" --> PostgreSQL; %% Via Repositories
    FundsManager -- "Persists Balances" --> PostgreSQL; %% Via Repositories
    PositionManager -- "Persists Positions/Trades" --> PostgreSQL; %% Via Repositories

    %% Logging
    DataIngestor -- "Log Messages" --> LoggerService;
    FeatureEngine -- "Log Messages" --> LoggerService;
    PredictionService -- "Log Messages" --> LoggerService;
    StrategyArbitrator -- "Log Messages" --> LoggerService;
    RiskManager -- "Log Messages" --> LoggerService;
    ExecutionHandler -- "Log Messages" --> LoggerService;
    PortfolioManager -- "Log Messages" --> LoggerService;
    MonitoringService -- "Log Messages" --> LoggerService;
    PubSubManager -- "LogEvent (from other services)" --> LoggerService; %% Centralized logging
    LoggerService -- "Console Output" --> operator>"Operator/Console"];
    LoggerService -- "File Logs" --> file[("Log Files")];
    LoggerService -- "DB Logs (AsyncPostgresHandler)" --> LogRepository;
    LoggerService -- "Time-Series Metrics" --> InfluxDB;
    LogRepository -- "Writes Logs" --> PostgreSQL;


    %% Monitoring & Control
    MonitoringService -- "Queries Portfolio State" --> PortfolioManager;
    MonitoringService -- "SystemStateEvent (RUNNING/HALTED)" --> PubSubManager;
    PubSubManager -- "PotentialHaltTriggerEvent<br/>SystemErrorEvent" --> MonitoringService;
    MonitoringService -- "Interacts with" --> HaltCoordinator;
    HaltCoordinator -- "Manages Halt Conditions" --> MonitoringService;
    CLIService -- "User Commands (Halt/Resume/Status)" --> MonitoringService;
    CLIService -- "User Commands (Shutdown)" --> main_app[MainAppController];
    CLIService -- "Queries Status/Portfolio" --> PortfolioManager;
    CLIService -- "Interacts with" --> HaltCoordinator; %% For recovery checklist

    %% MLOps Interactions
    RetrainingPipeline -- "Fetches Training Data" --> HistoricalDataService;
    RetrainingPipeline -- "Triggers Model Training" --> model_training_scripts["External Model Training Scripts/Process"];
    model_training_scripts -- "Registers New Model" --> ModelRegistry;
    ModelRegistry -- "Persists Metadata" --> ModelMetadataRepository;
    ModelMetadataRepository -- "Writes Metadata" --> PostgreSQL;
    DriftDetector -- "Monitors Production Predictions/Features" --> PredictionService; %% (Conceptually, or via logged data)
    DriftDetector -- "Signals Drift" --> RetrainingPipeline;
    ExperimentManager -- "Updates Model Stages in" --> ModelRegistry; %% For promoting models
    ExperimentManager -- "Persists Experiment Data" --> PostgreSQL; %% Via ExperimentRepository

    %% General Database Interactions (via Repositories)
    EventLogRepository -- "Persists All System Events" --> PostgreSQL;
    PubSubManager -- "All Events" --> EventLogRepository; %% If EventStore is active

    %% Market Price Service Interaction with Exchange
    MarketPriceService -- "Fetches Ticker/OHLC (REST)" --> KrakenAPI;

    %% Style definitions (optional for better readability)
    style Databases fill:#f9d,stroke:#333,stroke-width:2px
    style TradingSystem fill:#lightgrey,stroke:#333,stroke-width:2px
    style External fill:#lightblue,stroke:#333,stroke-width:2px
    style MLOps fill:#e6ffe6,stroke:#333,stroke-width:1px
    style DAL fill:#whitesmoke,stroke:#333,stroke-width:1px
    classDef service fill:#dae8fc,stroke:#666,stroke-width:1px,color:#000
    classDef eventbus fill:#fffacd,stroke:#8B4513,stroke-width:2px,color:#000
    classDef database fill:#ffddc1,stroke:#B8860B,stroke-width:2px,color:#000
    classDef external fill:#c1ffc1,stroke:#2E8B57,stroke-width:2px,color:#000
    classDef executor fill:#thistle,stroke:#4B0082,stroke-width:1px,color:#000

    class PubSubManager eventbus;
    class KrakenAPI external;
    class PostgreSQL,InfluxDB database;
    class ConfigManager,LoggerService,MonitoringService,HaltCoordinator,CLIService,DataIngestor,FeatureEngine,HistoricalDataService,PredictionService,StrategyArbitrator,RiskManager,ExecutionHandler,PortfolioManager,FundsManager,PositionManager,ValuationService,MarketPriceService,ModelRegistry,ExperimentManager,RetrainingPipeline,DriftDetector,OrderRepository,PositionRepository,LogRepository,EventLogRepository,ModelMetadataRepository service;
    class ProcessPool executor;
    class ModelArtifacts database;

end
```

## Key Flows and Interactions

This diagram illustrates the interconnected nature of the Gal-Friday trading system. Here are some of the key data flows and interactions:

1.  **Main Real-Time Trading Data Pipeline:**
    *   **Kraken API (WebSocket)** sends real-time market data (L2 order books, trades, OHLCV) to the **Data Ingestor**.
    *   **Data Ingestor** processes this raw data and publishes standardized `MarketDataOHLCVEvent`, `MarketDataL2Event`, and `MarketDataTradeEvent` to the **PubSubManager**.
    *   **Feature Engine** subscribes to these market data events, calculates various technical indicators and features, and publishes a `FeatureEvent` to the **PubSubManager**.
    *   **Prediction Service** subscribes to `FeatureEvent`s. It loads the appropriate ML model (details from **ModelLifecycle::Registry**, artifacts from **Model Artifacts Storage**) and uses a **ProcessPoolExecutor** for inference. It then publishes a `PredictionEvent` to the **PubSubManager**.
    *   **Strategy Arbitrator** subscribes to `PredictionEvent`s. It applies strategy logic (potentially querying **MarketPriceService** for current prices) and publishes a `TradeSignalProposedEvent` to the **PubSubManager**.
    -   **Risk Manager** subscribes to `TradeSignalProposedEvent`s. It queries **PortfolioManager** for current state (equity, positions, drawdown) and **MarketPriceService** for current prices to perform risk checks. It then publishes either a `TradeSignalApprovedEvent` or `TradeSignalRejectedEvent` to the **PubSubManager**.
    -   **Execution Handler** (e.g., `KrakenExecutionHandler`) subscribes to `TradeSignalApprovedEvent`s. It translates these into specific API calls to the **Kraken API (REST)** to place or manage orders.
    -   The **Kraken API** sends back acknowledgments or fill information (via REST or WebSocket private channels) to the **Execution Handler**.
    -   **Execution Handler** processes these responses and publishes `ExecutionReportEvent`s to the **PubSubManager**.
    -   **Portfolio Manager** subscribes to `ExecutionReportEvent`s. It updates its internal state (via **FundsManager**, **PositionManager**) and triggers **ValuationService** (which uses **MarketPriceService**) to recalculate portfolio metrics.

2.  **Role of PubSubManager:**
    *   The **PubSubManager** acts as the central asynchronous message bus, decoupling services. Most inter-service communication for real-time data flow happens via events published to and subscribed from the PubSubManager. This allows services to react to information without direct dependencies on the producers of that information.

3.  **Key Synchronous Interactions:**
    *   **Risk Manager** directly queries `PortfolioManager.get_current_state()` and `MarketPriceService.get_latest_price()` during its validation process.
    *   **Strategy Arbitrator** queries `MarketPriceService.get_latest_price()`.
    *   **ValuationService** queries `MarketPriceService` and gets data from `FundsManager` and `PositionManager`.
    *   **PredictionService** interacts with `ModelLifecycle::Registry` to fetch model details.
    *   Services needing configuration will query `ConfigManager` upon initialization.
    *   **CLIService** makes direct calls to `MonitoringService`, `PortfolioManager`, and `HaltCoordinator` to fulfill user commands.

4.  **Interactions with Databases:**
    *   **PostgreSQL (SQLAlchemy ORM):**
        *   `LoggerService` (via `LogRepository`) stores structured logs.
        *   `PortfolioManager` (via repositories like `PositionRepository`, `TradeRepository`) persists trade history, position snapshots, and potentially order details.
        *   `ModelLifecycle::Registry` (via `ModelMetadataRepository`) stores metadata about ML models.
        *   `ModelLifecycle::ExperimentManager` (via `ExperimentRepository`) stores experiment configurations and results.
        *   `ModelLifecycle::RetrainingPipeline` (via `RetrainingRepository`) stores retraining job details.
        *   `EventLogRepository` may store a log of all system events from PubSubManager for auditing or replay.
    *   **InfluxDB (Time-Series Data):**
        *   `LoggerService` may store time-series metrics (e.g., performance data, queue lengths).
        *   Potentially, `FeatureEngine` or `DataIngestor` could store processed market data or features here for specialized time-series analysis, though the diagram primarily shows this for logging.

5.  **Role of Supporting Services:**
    *   **ConfigManager:** Provides configuration parameters to all services at startup.
    *   **LoggerService:** Centralizes logging from all components, outputting to console, files, PostgreSQL, and InfluxDB.
    *   **MonitoringService:** Monitors overall system health, interacts with `HaltCoordinator`, queries `PortfolioManager` for drawdown status, and can trigger system halts. It also listens for `PotentialHaltTriggerEvent`s.
    *   **CLIService:** Allows operator interaction for status checks, manual halt/resume, and shutdown.

6.  **Interactions with MLOps Components:**
    *   **ModelLifecycle::Registry:** Queried by `PredictionService` for production models. Updated by `RetrainingPipeline` (or external training processes) when new models are trained and by `ExperimentManager` when models are promoted.
    *   **ModelLifecycle::ExperimentManager:** May influence `PredictionService` routing during A/B tests.
    -   **ModelLifecycle::RetrainingPipeline:** Uses `HistoricalDataService` for training data, triggers external training processes, and registers new models with the `Registry`. Its `DriftDetector` monitors predictions/features (conceptually from `PredictionService` or its outputs).

This diagram provides a high-level view of how data moves through the Gal-Friday system and how its various components collaborate to achieve automated trading.
