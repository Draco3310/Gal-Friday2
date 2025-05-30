# Software Development Project Plan

**Project: Gal-Friday**

**Version: 0.2**

**Date: 2025-04-27**

**Status: Phase 1 & 2 Complete**

---

**Table of Contents:**

1.  Project Overview
2.  Methodology
3.  Project Plan Details (Task Table)
    3.1 Phase 1: Requirements Analysis & Planning
    3.2 Phase 2: System Design
    3.3 Phase 3: Implementation
    3.4 Phase 4: Testing 
    3.5 Phase 5: Deployment (Sandbox/Controlled Live)
    3.6 Phase 6: Maintenance & Monitoring
4.  Assumptions & Notes

---

## 1. Project Overview

* **Project Name:** Gal-Friday
* **Objective:** Develop an automated AI trading bot for XRP/USD and DOGE/USD on the Kraken exchange, targeting $75k/year income. Validate via backtesting, paper trading, and controlled live trading. (Ref: SRS 1.1)
* **Scope:** Defined in SRS 1.4.
* **Key Stakeholders:** Project Lead (User), Development Team (AI), Kraken (Platform).

## 2. Methodology

This project plan follows a Waterfall methodology, proceeding sequentially through distinct phases: Requirements Analysis & Planning, System Design, Implementation, Testing, Deployment, and Maintenance. While iterative refinement within phases (especially Implementation and Testing) is expected, the overall flow is sequential.

## 3. Project Plan Details (Task Table)

*(Note: Durations are estimates in working days/weeks and subject to refinement. Start/End dates are relative and depend on the project start date and preceding task completion.)*

| Phase | Task ID | Task Name | Description | Est. Duration | Owner | Deliverable | Status | Predecessors |
|---|---|---|---|---|---|---|---|---|
| **Phase 1: Requirements Analysis & Planning** | | | | | | | | |
| 1 | 1.1 | Define Project Scope & Objectives | Initial goal setting and scope definition. | 1 day | User, AI | High-Level Scope | **Complete** | - |
| 1 | 1.2 | Gather Detailed Requirements | Q&A sessions to refine features and parameters. | 2 days | User, AI | Detailed Notes | **Complete** | 1.1 |
| 1 | 1.3 | Draft SRS Document | Formalize functional and non-functional requirements based on Q&A. | 3 days | AI | SRS Draft (`srs_gal_friday_v0.1`) | **Complete** | 1.2 |
| 1 | 1.4 | Review & Approve SRS Document | User reviews and approves the SRS. | 1 day | User | Approved SRS | **Complete** | 1.3 |
| 1 | 1.5 | Draft Project Plan | Create this project plan document. | 1 day | AI | Project Plan Draft | **Complete** | 1.3 |
| 1 | 1.6 | Review & Approve Project Plan | User reviews and approves the Project Plan. | 1 day | User | Approved Project Plan | **Complete** | 1.5 |
| 1 | 1.7 | Draft Risk Analysis Document | Identify project, technical, and financial risks and mitigations. | 1 day | AI | Risk Analysis Draft | **Complete** | 1.3 |
| 1 | 1.8 | Review & Approve Risk Analysis | User reviews and approves the Risk Analysis. | 0.5 days | User | Approved Risk Analysis | **Complete** | 1.7 |
| 1 | 1.9 | Draft Project Charter/PID | Outline project objectives, scope, resources, stakeholders. | 0.5 days | AI | PID Draft | **Complete** | 1.3 |
| 1 | 1.10 | Review & Approve Project Charter/PID | User reviews and approves the PID. | 0.5 days | User | Approved PID | **Complete** | 1.9 |

| **Phase 2: System Design** | | | | | | | | |
| 2 | 2.1 | Design High-Level Architecture | Define overall structure (event-driven, modules). | 2 days | AI | Architecture Concept | **Complete** | 1.4, 1.6 |
| 2 | 2.2 | Create Architecture Diagram | Visualize modules and data flow. | 1 day | AI | Architecture Diagram | **Complete** | 2.1 |
| 2 | 2.3 | Define Inter-Module Communication | Specify event payloads and API formats. | 2 days | AI | Event/API Definitions | **Complete** | 2.1 |
| 2 | 2.4 | Design Database Schema | Define tables/fields for PostgreSQL and InfluxDB measurements. | 2 days | AI | DB Schema Definition | **Complete** | 2.1 |
| 2 | 2.5 | Create DB Schema Diagrams/Definitions | Visualize/Document the database schema. | 1 day | AI | DB Schema Diagram | **Complete** | 2.4 |
| 2 | 2.6 | Design Core Module Interfaces | Define key class/function signatures. | 2 days | AI | Interface Definitions | **Complete** | 2.1, 2.3 |
| 2 | 2.7 | Design Backtesting Engine Logic | Specify simulation mechanics (fees, slippage, data handling). | 1 day | AI | Backtester Design Doc | **Complete** | 2.1 |
| 2 | 2.8 | Design Model Retraining Pipeline | Specify data flow, validation, scheduling approach. | 1 day | AI | Retraining Pipeline Design | **Complete** | 2.1 |
| 2 | 2.9 | Review & Approve Design Documents | User reviews architecture, DB, interface designs. | 2 days | User | Approved Design Docs | **Complete** | 2.2, 2.3, 2.5, 2.6, 2.7, 2.8 |

| **Phase 3: Implementation** | | | | | | | | |
| 3 | 3.1 | Setup Development Environment | Configure local/cloud dev environment (Python, libs, IDE). | 1 day | AI | Dev Environment | Completed | 2.9 |
| 3 | 3.2 | Setup Cloud Infrastructure | Provision VM, install PostgreSQL/InfluxDB, configure network. | 2 days | AI | Cloud Infra Ready | Completed | 2.9 |
| 3 | 3.3 | Implement Data Ingestor Module | Code WebSocket connection, data parsing, L2 book mgmt (FR-1xx). | 4 days | AI | Data Ingestor Module Code | Completed | 3.1, 3.2 |
| 3 | 3.4 | Implement Feature Engine Module | Code indicator/feature calculations (FR-2xx). | 3 days | AI | Feature Engine Module Code | Completed | 3.3 |
| 3 | 3.5 | Implement Predictive Modeling Module | Code model loading, preprocessing, prediction (XGBoost - FR-3xx). | 3 days | AI | Prediction Module Code | Completed | 3.4 |
| 3 | 3.6 | Implement Strategy/Signal Module | Code signal generation logic (FR-4xx). | 2 days | AI | Strategy Module Code | Completed | 3.5 |
| 3 | 3.7 | Implement Risk Management Module | Code pre-trade checks, sizing (FR-5xx). | 3 days | AI | Risk Module Code | Completed | 3.8 |
| 3 | 3.8 | Implement Portfolio Management Module | Code position/balance tracking (FR-7xx). | 2 days | AI | Portfolio Module Code | Completed | 3.1, 3.2 |
| 3 | 3.9 | Implement Execution Handler Module | Code Kraken Sandbox API interaction (FR-6xx). | 4 days | AI | Execution Module Code | Completed | 3.7 |
| 3 | 3.10 | Implement Logging Module | Code logging to files/Postgres (FR-8xx). | 2 days | AI | Logging Module Code | Completed | 3.1, 3.2 |
| 3 | 3.10a | Create DB Schema for Logging | Define and create the PostgreSQL table schema required by LoggerService. | 0.5 days | AI | DB Schema Script/Definition | Completed | 3.2 |
| 3 | 3.11 | Implement Monitoring/HALT Module | Code basic health checks, HALT triggers (FR-9xx). | 2 days | AI | Monitoring Module Code | Completed | 3.7, 3.8, 3.10a |
| 3 | 3.12 | Implement Basic CLI | Code command-line interface for basic control (NFR-201). | 1 day | AI | CLI Code | Completed | 3.11 |
| 3 | 3.13 | Implement Configuration Handling | Code loading/parsing of config files (NFR-804), including logging DB settings. | 1 day | AI | Config Handling Code | Completed | 3.1 |
| 3 | 3.14 | Implement Basic Backtesting Engine | Code historical data simulation core (FR-10xx). | 4 days | AI | Backtester Code | Completed | 3.4, 3.5, 3.6, 3.7, 3.8 |
| 3 | 3.15 | Implement Model Training/Loading Script | Script to train initial models and load them (FR-3xx). | 2 days | AI | Training/Loading Script | Completed | 3.5 |
| 3 | 3.16 | Create Main Application Entry Point | Code main script (`main.py`) to initialize modules, executor, event buses, and manage lifecycle. | 1 day | AI | Main App Entry | Completed | 3.3 - 3.15 |
| 3 | 3.17 | Integrate LoggerService | Pass LoggerService instance to other modules and replace print/basic logging calls. | 1 day | AI | Integrated Logging | Completed | 3.10a, 3.3 - 3.9, 3.11-3.16 |
| 3 | 3.18 | Code Review & Refinement | Internal review and cleanup of implemented code. | 3 days | AI | Internally Reviewed Codebase | In-progress | 3.17 |

| **Phase 4: Testing ** | | | | | | | | |
| 4 | 4.1 | Develop Unit Tests | Create tests for individual functions/classes in each module. | 5 days | AI | Unit Test Suite | Pending | 3.18 |
| 4 | 4.2 | Execute & Debug Unit Tests | Run tests, fix failures until pass rate is high. | 3 days | AI | Passing Unit Tests Report | Pending | 4.1 |
| 4 | 4.3 | Develop Integration Tests | Create tests for interactions between modules (e.g., Data -> Feature -> Predict -> Signal -> Risk -> Execute). | 4 days | AI | Integration Test Suite | Pending | 3.18 |
| 4 | 4.4 | Execute & Debug Integration Tests | Run integration tests, fix failures. | 3 days | AI | Passing Integration Tests Report | Pending | 4.3 |
| 4 | 4.5 | Perform System Testing (Sandbox) | Test end-to-end flow using Kraken Sandbox API. | 3 days | AI | System Test Report (Sandbox) | Pending | 4.4 |
| 4 | 4.6 | Perform Backtesting | Run backtester on historical data with full settings. | 2 days | AI | Backtesting Results Report | Pending | 3.14, 4.4 |
| 4 | 4.7 | Analyze Backtesting Results | Review performance metrics, identify issues/areas for tuning. | 1 day | AI, User | Backtesting Analysis Summary | Pending | 4.6 |
| 4 | 4.8 | Perform Paper Trading (Sandbox) | Run bot in paper trading mode against live data. | 5 days | AI | Paper Trading Logs & Summary | Pending | 4.5 |
| 4 | 4.9 | Analyze Paper Trading Results | Review performance in simulated live conditions. | 1 day | AI, User | Paper Trading Analysis | Pending | 4.8 |
| 4 | 4.10 | Perform Basic Performance Testing | Assess latency and resource usage under simulated load. | 1 day | AI | Performance Test Notes | Pending | 4.5 |
| 4 | 4.11 | Perform Security Checks | Verify API key handling, config security. | 1 day | AI | Security Check Report | Pending | 3.18 |
| 4 | 4.12 | User Acceptance Testing (UAT) | User reviews logs, backtest/paper trade results, confirm full functionality. | 2 days | User | UAT Sign-off | Pending | 4.7, 4.9, 4.5 |

| **Phase 5: Deployment (Sandbox/Controlled Live)** | | | | | | | | |
| 5 | 5.1 | Prepare Production Environment | Set up/configure the target live trading environment (VM, DBs). | 1 day | AI | Production Environment Ready | Pending | 4.12 |
| 5 | 5.2 | Secure Production Environment | Implement firewall rules, access controls, monitoring setup. | 1 day | AI | Secured Environment Checklist | Pending | 5.1 |
| 5 | 5.3 | Create Deployment Scripts/Procedures | Automate or document the deployment process. | 1 day | AI | Deployment Documentation | Pending | 5.1 |
| 5 | 5.4 | Deploy Application to Production | Execute deployment scripts/procedures. | 0.5 days | AI | Application Deployed | Pending | 5.3 |
| 5 | 5.5 | Configure Production Settings | Set live API keys (securely!), final parameters. | 0.5 days | User, AI | Production Configuration Set | Pending | 5.4 |
| 5 | 5.6 | Perform Smoke Tests in Production | Basic checks to ensure the system runs and connects. | 0.5 days | AI | Smoke Test Pass | Pending | 5.5 |
| 5 | 5.7 | Initiate Live Trading (Controlled) | Start bot with very small size or limited capital, monitor closely. | Ongoing | User, AI | Live Trading Active | Pending | 5.6 |

| **Phase 6: Maintenance & Monitoring** | | | | | | | | |
| 6 | 6.1 | Monitor System Health | Ongoing monitoring of logs, alerts, resources. | Ongoing | AI, User | Monitoring Reports/Alerts | Pending | 5.7 |
| 6 | 6.2 | Monitor Trading Performance | Ongoing tracking of P&L, drawdown vs targets. | Ongoing | User, AI | Performance Reports | Pending | 5.7 |
| 6 | 6.3 | Perform Regular Backups | Schedule and verify database backups. | Ongoing | AI | Backup Logs | Pending | 5.1 |
| 6 | 6.4 | Apply Security Patches | Update OS and libraries as needed. | Periodic | AI | Patch Logs | Pending | 5.1 |
| 6 | 6.5 | Periodically Retrain ML Models | Run retraining pipeline based on schedule/performance. | Periodic | AI | Model Retraining Logs | Pending | 3.15 |
| 6 | 6.6 | Bug Fixing & Enhancements | Address issues found in production, implement minor improvements. | As Needed | AI | Code Updates / Bug Fix Reports | Pending | 5.7 |
| 6 | 6.7 | Review & Adapt Strategy | Periodically review performance and adjust configuration/strategy based on results. | Periodic | User, AI | Configuration Updates | Pending | 6.2 |
| 6 | 6.8 | Plan Future Enhancements | Prioritize and plan development for future features (more models, assets, features). | Periodic | User, AI | Future Feature Roadmap | Pending | 6.2 |

## 4. Assumptions & Notes

* Durations are estimates and may change based on complexity encountered.
* User availability for reviews and approvals is factored into the timeline but depends on Project Lead's schedule.
* Successful completion of preceding tasks is required before starting dependent tasks.
* Cloud infrastructure costs are assumed to be manageable within the project budget.
* Kraken Sandbox environment is assumed to be available and suitable for testing. If not, paper trading simulation needs enhancement.

---
**End of Document**
