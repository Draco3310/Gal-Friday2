# Post-MVP Feature Upgrade Plan

**Project: Gal-Friday**

**Version: 0.1**

**Date: 2025-04-27**

**Status: Draft**

---

**Table of Contents:**

1.  Introduction
2.  Upgrade Categories & Features
    2.1 Model Enhancements
    2.2 Strategy Enhancements
    2.3 Risk Management Enhancements
    2.4 Data Enhancements
    2.5 Operational & Monitoring Enhancements
    2.6 Scalability & Architecture Enhancements
3.  Prioritization & Approach

---

## 1. Introduction

This document outlines potential features and enhancements for the Gal-Friday trading system to be considered *after* the successful development, testing, and deployment of the Minimum Viable Product (MVP). The MVP focuses on establishing core functionality (XRP/USD trading, XGBoost model, basic risk management on Kraken Sandbox/Live). This plan provides a roadmap for iteratively adding capabilities to achieve the full project vision outlined in the initial requirements and SRS [R1].

Features are categorized, and initial thoughts on priority and complexity are provided for planning purposes. Priority and scheduling will be determined based on MVP performance, resource availability, and evolving project goals.

## 2. Upgrade Categories & Features

| Category | Feature ID | Feature Description | Potential Priority | Est. Complexity | Related SRS Refs | Notes |
|---|---|---|---|---|---|---|
| **Model Enhancements**
| MOD-01 | Integrate **LSTM Model** | High | Large | FR-302, FR-306 | Requires TensorFlow/PyTorch integration, sequence handling. |
| | MOD-02 | Integrate **RandomForest Model** | Medium | Medium | FR-302, FR-306 | Simpler than LSTM, leverages existing feature structure. |
| | MOD-03 | Implement **Model Ensembling/Arbitration** | High | Medium | FR-307 | Logic to combine predictions from multiple models (XGBoost, LSTM, RF). |
| | MOD-04 | Add **Online Learning / Adaptive Models** | Low | Large | - | Models that update more frequently based on recent performance/data streams. |
| | MOD-05 | **Advanced Hyperparameter Tuning** | Medium | Medium | FR-310 (Implied) | Implement more sophisticated tuning (e.g., Bayesian Optimization) in retraining pipeline. |
| | MOD-06 | **Explainable AI (XAI)** Features | Low | Medium | - | Integrate tools (e.g., SHAP) to better understand model predictions. |
| **Strategy Enhancements**
| STR-01 | Add **DOGE/USD Trading Pair** | High | Medium | FR-102, FR-104, etc. | Requires configuration, data handling, potentially model retraining specific to DOGE. |
| | STR-02 | Implement **Dynamic Strategy Arbitration** | Medium | Medium | FR-408 (Future) | Logic to adjust strategy parameters (thresholds, R:R) based on volatility, model confidence, or market regime. |
| | STR-03 | Develop **Alternative Strategy Logic** | Medium | Large | - | Implement different core trading strategies beyond simple threshold crossing (e.g., mean reversion, breakout). |
| | STR-04 | Add **Time-Based Exits** | Medium | Small | FR-408 | Implement max holding period exit condition. |
| | STR-05 | Add **Prediction Reversal Exits** | Medium | Small | FR-408 | Implement exit condition based on model prediction flipping. |
| | STR-06 | Optimize **Order Placement Logic** | Medium | Medium | FR-604, FR-605 | More sophisticated limit order placement (e.g., mid-price targeting, passive placement), adaptive timeouts. |
| | STR-07 | **ATR-Based SL/TP Calculation:** Calculate SL/TP within StrategyArbitrator using ATR values. | Medium | Medium | - | Requires StrategyArbitrator access to ATR feature (from FeatureEvent or cache). |
| | STR-08 | **Multi-Strategy Arbitration:** Implement logic to load/run multiple strategies concurrently. | Medium | Medium | FR-408 (Future) | Requires config schema update and routing logic based on prediction metadata or external factors. |
| | STR-09 | **Stateful Strategy Logic:** Add capability for strategies to maintain state between events. | Medium | Medium | - | E.g., For cooldown periods after signals, tracking recent trends within the strategy itself. |
| | STR-10 | Implement **Post-Entry SL/TP Placement** | High | Medium | FR-406, FR-604 | Logic within ExecutionHandler (or triggered by Strategy) to place SL/TP orders after entry confirmation/fill. |
| **Risk Management Enhancements**
| | RSK-01 | Implement **Volatility-Adjusted Position Sizing** | Medium | Medium | SRS 2.5 (Implied) | Adjust size based on current market volatility (e.g., reduce size in high vol). |
| | RSK-02 | Add **Correlation Risk Checks** | Low | Medium | - | If adding more assets, check correlation before increasing exposure. |
| | RSK-03 | Implement **Dynamic Stop-Loss Logic** | Medium | Medium | - | E.g., Trailing stops based on price movement or ATR. |
| | RSK-04 | Add **Advanced HALT Triggers** | Medium | Medium | FR-905 | More sophisticated triggers (e.g., based on statistical deviations, specific news events). |
| | RSK-05 | Implement **Automated Recovery Options** | Low | Large | FR-908 (Beyond MVP) | Define conditions and logic for automated system restart after certain HALT types (requires extreme caution). |
| | RSK-06 | Track **Non-Quote Currency Balances** | Medium | Medium | - | Monitor balances of assets other than the quote currency (e.g., base asset) for overall portfolio exposure management. |
| **Data Enhancements**
| | DAT-01 | Integrate **News Feed API** | Medium | Large | FR-109 (Future) | Ingest real-time news, potentially use NLP for sentiment/impact analysis as filter/feature. |
| | DAT-02 | Integrate **Social Media Sentiment API** | Low | Large | FR-109 (Future) | Ingest sentiment scores (e.g., Twitter) as filter/feature. |
| | DAT-03 | Integrate **Blockchain Analytics Data** | Low | Large | - | E.g., Whale alerts, on-chain transaction volumes. |
| | DAT-04 | Utilize **Tick-Level Data** | Medium | Large | FR-1001 (Implied) | Incorporate higher-resolution data for feature engineering and backtesting (increases storage/processing significantly). |
| | DAT-05 | Implement **Feature Store** | Low | Medium | - | Centralized system for storing and serving features for training and inference. |
| **Operational & Monitoring Enhancements**
| | OPS-01 | Develop **Web-Based Monitoring Dashboard (GUI)** | Medium | Large | NFR-203 (Beyond MVP) | Real-time view of performance, positions, logs, system status. |
| | OPS-02 | Implement **Advanced Alerting** (Email/SMS) | Medium | Medium | FR-907 (Implied) | Integrate external notification services for critical alerts (HALT, large losses, errors). |
| | OPS-03 | Enhance **Backtester Realism** | Medium | Medium | FR-1003 | More sophisticated slippage models, latency simulation, L2-based matching (if L2 data added). |
| | OPS-04 | Add **Scenario Replay / Debugging Tools** | Medium | Medium | FR-1008 | Tools to easily replay specific historical periods with detailed logging for debugging. |
| | OPS-05 | Implement **Configuration Hot-Reloading** | Low | Medium | - | Allow updating certain configuration parameters without restarting the bot. |
| | OPS-06 | Enhance **CLI Functionality** | Low | Small | FR-3.12 | Add more commands for detailed status, performance queries, manual order overrides (use with caution). |
| | OPS-07 | Implement **Order Cancellation Logic** | Medium | Medium | FR-606 | Add `cancel_order` method using CancelOrder/CancelOrderBatch API endpoints. |
| | OPS-08 | **Robust DB Logging Error Handling** | Low | Small | FR-806 | Implement fallback logging (e.g., to file) if DB handler fails repeatedly. |
| | OPS-09 | **Code Readability Improvements** | Medium | Medium | NFR-401 | Refactor DataIngestor: split into smaller modules, extract event classes, improve error handling patterns. |
| | OPS-10 | **Standardize Error Handling** | Medium | Small | NFR-401 | Create unified error handling utilities to reduce code duplication and improve consistency. |
| | OPS-11 | **Configuration Management Refactor** | Low | Medium | NFR-401 | Move constants and configuration to dedicated files, implement enums for status and message types. |
| **Scalability & Architecture Enhancements**
| | ARC-01 | Add **Support for Additional Exchanges** | Low | Large | SRS 1.4 (Out of Scope) | Abstract exchange interactions (e.g., via `ccxt`) further to support platforms beyond Kraken. |
| | ARC-02 | **Optimize Database Performance** | Medium | Medium | - | Review indexing, query optimization, potential partitioning as data grows. |
| | ARC-03 | **Refactor for Horizontal Scaling** (If Needed) | Low | Very Large | NFR-505 (Implied) | If performance bottlenecks necessitate, consider extracting modules (e.g., PredictionService) into separate microservices (major architectural change). |
| | ARC-04 | Implement **Redundancy / High Availability** | Medium | Medium | NFR-801 (Implied) | More robust multi-instance deployment, potentially with leader election or load balancing. |
| | ARC-05 | Implement **WebSocket Private Feed Integration** | High | Large | FR-702 (Implied) | Connect to Kraken WS for real-time order status/fills, reducing REST polling. |

## 3. Prioritization & Approach

* **Prioritization:** The priority assigned above is preliminary. Actual prioritization should occur after evaluating the MVP's performance and stability. Factors include:
    * Impact on profitability and risk reduction.
    * User feedback and operational needs.
    * Dependencies between features.
    * Estimated development effort vs. expected benefit.
* **Approach:** Enhancements should be implemented iteratively, following a similar SDLC process (design, implement, test, deploy) for each significant feature or group of features. Rigorous testing (including backtesting the impact of the change) is essential before deploying any upgrade to the live environment.

---
**End of Document**
