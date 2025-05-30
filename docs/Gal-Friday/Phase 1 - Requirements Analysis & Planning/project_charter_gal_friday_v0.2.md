# Project Charter

**Project: Gal-Friday**

**Version: 0.2**

**Date: 2025-04-27**

---

## Document Control

**Document Information**

| Information       | Details                                    |
| :---------------- | :----------------------------------------- |
| Document Id       | `project_charter_gal_friday_v0.1`          |
| Document Owner    | [Project Lead Name - TBD by User]          |
| Issue Date        | 2025-04-27                                 |
| Last Saved Date   | 2025-04-27                                 |
| File Name         | Project Charter - Gal-Friday v0.1          |

**Document History**

| Version | Issue Date | Changes                                |
| :------ | :--------- | :------------------------------------- |
| 0.1     | 2025-04-27 | Initial Draft                          |

**Document Approvals**

| Role            | Name                              | Signature | Date       |
| :-------------- | :-------------------------------- | :-------- | :--------- |
| Project Sponsor | [Project Lead Name - TBD by User] | *(Pending)* | *(Pending)* |
| Project Manager | AI (Gemini)                       | N/A       | N/A        |
| *(Other roles as needed)* |                                   |           |            |

---

## 1. Executive Summary, Project Purpose, Goals and Success Criteria

### 1.1 Executive Summary
Project Gal-Friday aims to develop and deploy a sophisticated automated cryptocurrency trading bot. The system will leverage AI/ML predictive models (XGBoost, RandomForest, LSTM) to execute high-frequency scalping and day trading strategies on the Kraken exchange, specifically targeting XRP/USD and DOGE/USD pairs. The primary objective is to generate a consistent revenue stream, targeting $75k/year, operating with an initial capital of $100k. The project includes modules for real-time data ingestion, feature engineering, prediction, strategy execution, rigorous risk management, order placement, logging, monitoring, and simulation capabilities. The project will follow a Waterfall SDLC methodology, starting with detailed planning and design, followed by implementation, testing (including backtesting and paper trading), controlled deployment, and ongoing maintenance. Key risks involve market volatility, model accuracy, API dependency, and security, which will be mitigated through robust design, testing, and operational controls. The project requires dedicated development effort (provided by AI) and active involvement from the Project Lead for direction, configuration, and monitoring.

### 1.2 Project Purpose & Vision

**Project Purpose**
To create a reliable, automated trading system (Gal-Friday) capable of generating a target income of $75k/year by algorithmically trading XRP/USD and DOGE/USD on the Kraken exchange. The system will solve the need for automated, data-driven trading decisions and execution, operating 24/7 within defined risk parameters. Key objectives include achieving consistent profitability, managing risk effectively according to predefined limits (15% max drawdown, 0.5-1% risk/trade), and ensuring operational stability. The scope encompasses the full lifecycle from data acquisition to trade execution and monitoring. Deliverables include the operational trading bot software, supporting documentation (SRS, Design Docs, Test Reports), and performance logs.

**Vision**
To establish Gal-Friday as a dependable, intelligent, and profitable automated trading system that serves as a primary income source through disciplined, AI-driven cryptocurrency trading.

### 1.3 Project Goals and Objectives
*(Based on SMART principles)*

**Business Objectives:**
* **BO1:** Achieve an average net annual profit of $75,000 from trading activities after accounting for fees and potential losses, measured over a 12-month period post-deployment stabilization. (Specific, Measurable, Achievable [Target], Realistic [High Risk], Time-bound [Post-Stabilization])
* **BO2:** Maintain overall portfolio drawdown below 15% of the initial ($100k) or highest recorded equity. (Specific, Measurable, Achievable, Realistic, Time-bound [Ongoing])
* **BO3:** Limit daily drawdown to a maximum of 2% and weekly drawdown to a maximum of 5%. (Specific, Measurable, Achievable, Realistic, Time-bound [Ongoing])
* **BO4:** Ensure risk per trade does not exceed 1% of current account equity. (Specific, Measurable, Achievable, Realistic, Time-bound [Ongoing])

**Technology Objectives:**
* **TO1:** Develop and deploy a functional full version implementation of the Gal-Friday trading bot capable of executing the core trading loop (Data -> Predict -> Risk Check -> Execute) on the Kraken Sandbox environment within the timeline defined in the Project Plan [R2]. (Specific, Measurable, Achievable, Realistic, Time-bound)
* **TO2:** Implement robust connectivity to Kraken WebSocket (for L2/OHLCV data) and REST APIs (for execution) with automated reconnection logic. (Specific, Measurable, Achievable, Realistic, Time-bound [Implementation Phase])
* **TO3:** Integrate 3 ML models (XGBoost, RandomForest, LSTM) capable of generating price movement probability predictions. (Specific, Measurable, Achievable, Realistic, Time-bound [Implementation Phase])
* **TO4:** Implement all specified pre-trade risk checks and HALT conditions defined in the SRS [R1]. (Specific, Measurable, Achievable, Realistic, Time-bound [Implementation Phase])
* **TO5:** Achieve operational stability with >99.5% uptime during active trading periods post-deployment. (Specific, Measurable, Achievable, Realistic, Time-bound [Post-Deployment])

### 1.4 Project Success Criteria
The success of Project Gal-Friday will be measured against the following criteria:
* **Profitability:** Meeting or exceeding the net profit target (BO1) consistently over time.
* **Risk Management:** Adherence to all defined drawdown and risk-per-trade limits (BO2, BO3, BO4).
* **Functionality:** Successful implementation and operation of all features defined in the SRS [R1] for the target phase (Full scope).
* **Reliability:** Achieving operational uptime targets (TO5) and demonstrating robust error handling and recovery.
* **Scope Completion:** Delivering the defined scope within the agreed-upon timeline and budget framework (as outlined in Project Plan [R2]).
* **Stakeholder Satisfaction:** Meeting the expectations of the Project Lead regarding system performance, reporting, and manageability.

---

## 2. Project Scope, Quality Management and Timeline

### 2.1 Project Scope
(Ref: SRS 1.4)
The scope includes the design, development, testing, deployment, and maintenance of the Gal-Friday automated trading system for XRP/USD and DOGE/USD on Kraken.

**In Scope:**
* Real-time market data ingestion (L2, OHLCV) via WebSocket.
* Feature engineering (technical indicators, order book features).
* Predictive modeling using ML (XGBoost, RF, LSTM).
* Strategy arbitration and signal generation logic.
* Comprehensive risk management (drawdown limits, risk per trade, position sizing, pre-trade checks, HALT conditions).
* Automated order execution via Kraken REST/WebSocket APIs (Limit/Market orders).
* Portfolio tracking (balance, positions, equity, P&L).
* Extensive logging and auditing (files, PostgreSQL, InfluxDB).
* System monitoring and alerting (basic initially).
* Backtesting engine with realistic simulation.
* Paper trading capability (Kraken Sandbox or internal simulation).
* Deployment on a Linux cloud VM.
* Basic CLI for control and configuration via files.

**Out of Scope (Initially):**
* Trading on exchanges other than Kraken.
* Trading assets other than XRP/USD and DOGE/USD.
* Graphical User Interface (GUI).
* Strategies not based on the specified ML models.
* Integration with external portfolio management tools.
* Advanced alerting mechanisms (e.g., SMS/email - potential future enhancement).
* Fully automated recovery from HALT conditions (requires manual intervention).

**Exclusions, Assumptions and Constraints:** (Ref: SRS 2.7, Risk Analysis 2.4)
* **Exclusions:** See "Out of Scope" above.
* **Assumptions:** Availability and reliability of Kraken APIs, accuracy of market data, stability of cloud infrastructure, representativeness of historical data, availability of $100k capital.
* **Constraints:** Dependency on Kraken API functionality and rate limits, Python 3 environment, specific library usage, secure handling of API keys, adherence to SDLC, budget for cloud infrastructure.

**High-Level Milestones:** (Ref: Project Plan [R2])

| Milestone                       | Description                                     | Target Phase Completion |
| :------------------------------ | :---------------------------------------------- | :---------------------- |
| Phase 1 Complete                | Requirements & Planning Docs Approved           | Q2 2025                 |
| Phase 2 Complete                | System Design Docs Approved                     | Q2 2025                 |
| Full Implementation Complete    | Full system modules coded                       | Q3 2025                 |
| Full Testing Complete           | Unit, Integration, System, Backtest, Paper Test | Q3 2025                 |
| Full UAT Complete               | User sign-off on full functionality             | Q3 2025                 |
| Full Deployment (Sandbox/Live)  | Initial controlled deployment                   | Q4 2025                 |
| Full System Testing & Deployment| Testing and deployment of full features         | Q4 2025                 |

*(Note: Specific dates depend on project start and task durations)*

### 2.2 Quality Management
* **Quality Standards:** Adherence to requirements specified in the SRS [R1], coding standards (PEP 8), successful completion of all test phases outlined in the Project Plan [R2], meeting performance NFRs (latency, reliability).
* **Quality Assurance Guidelines:** Code reviews, comprehensive unit and integration testing, rigorous backtesting and paper trading before live deployment, adherence to SDLC processes, documentation reviews.
* **Quality Control Procedures:** Execution of unit tests, integration tests, system tests, backtests, paper trading simulations; tracking test results; bug tracking and resolution; User Acceptance Testing (UAT); monitoring of live performance against benchmarks and risk limits.

### 2.3 Project Timeline
A high-level overview of the project schedule is provided by the Milestones table (Section 2.1) and the detailed task list in the Project Plan [R2]. The initial focus is on delivering the full system by Q4 2025.

---

## 3. Resources, Costs and Budget

### 3.1 Resource Management
* **Project Lead / Sponsor (User):** Provides direction, requirements, domain expertise, reviews/approvals, capital, monitors performance. (Effort: Part-time, variable)
* **Development Team (AI - Gemini):** Responsible for design, implementation, testing, documentation drafting, deployment support, maintenance. (Effort: Dedicated as per project plan tasks)
* **Infrastructure:** Cloud VM (Linux), PostgreSQL Database, InfluxDB Database, Network Bandwidth.
* **Software:** Python environment, required libraries (open source), potential future costs for premium data feeds (news/sentiment - out of scope for full system).
* **Trading Capital:** $100,000 USD allocated by Project Lead.

### 3.2 Estimated Costs and Project Budget
* **Development Labor:** Provided by AI (no direct cost). Project Lead time is allocated.
* **Cloud Infrastructure:** Estimated $50 - $150 / month (VM + Databases, depending on specs and usage). Scalability may increase costs later.
* **Data Fees:** $0 initially (using standard Kraken feeds). Potential future costs ($50-$200+/month) if premium news/sentiment APIs are added.
* **Trading Fees:** Transaction costs imposed by Kraken, variable based on volume and order types (accounted for in P&L calculations).
* **Contingency:** Recommended buffer (e.g., 10-15%) for unforeseen infrastructure needs or extended testing phases.
* **Total Estimated Operational Cost (Full System):** ~$100-200/month (excluding trading fees and capital).

### 3.3 Cost-Benefit Analysis
* **Costs:** Initial trading capital ($100k), ongoing operational infrastructure costs (~$1.2k - $2.4k / year initially), Project Lead time investment, potential trading losses within risk limits.
* **Tangible Benefits:** Target income generation of $75k/year (High Risk). Potential for higher returns if successful. Automated execution saves user time.
* **Intangible Benefits:** Development of a sophisticated AI trading asset, learning experience in algorithmic trading and AI application, potential for future expansion/scaling.
* **Analysis:** The project presents a high-risk, high-reward scenario. The potential financial return ($75k/year target) significantly outweighs the estimated operational costs if successful. However, the risk of capital loss up to the defined drawdown limits (15% or $15k initially) is substantial and must be acknowledged. The project is justified based on the potential reward relative to the operational costs, provided the capital risk is acceptable to the Project Lead/Sponsor.

---

## 4. Project Team and Stakeholders

### 4.1 Project Team

| Role            | Organisation | Resource Name                     | Assignment Status | Assignment Date |
| :-------------- | :----------- | :-------------------------------- | :---------------- | :-------------- |
| Project Sponsor | User         | [Project Lead Name - TBD by User] | Assigned          | 2025-04-27      |
| Project Manager | AI           | Gemini                            | Assigned          | 2025-04-27      |
| Developer       | AI           | Gemini                            | Assigned          | 2025-04-27      |
| Tester          | AI           | Gemini                            | Assigned          | 2025-04-27      |
| *(Business Analyst)* | User         | [Project Lead Name - TBD by User] | Assigned          | 2025-04-27      |

**Project Organisation Chart:**
*(Simple structure for this project)*

+---------------------+| Project Sponsor     || [Project Lead Name] |-------++---------------------+       ||+---------------------+       || Project Manager     |<------+----->+---------------------+| (AI - Gemini)       |               | Project Team        |+---------------------+               | (AI - Development,  ||      Testing)       |+---------------------+
### 4.2 Project Stakeholders

| Name                              | Role                       | Power  | Interest | Contact Information |
| :-------------------------------- | :------------------------- | :----- | :------- | :------------------ |
| [Project Lead Name - TBD by User] | Project Sponsor, User      | High   | High     | TBD                 |
| AI (Gemini)                       | Project Manager, Dev Team  | Medium | High     | N/A                 |
| Kraken Exchange                   | Platform Provider          | Low    | Medium   | Via API / Support   |

---

## 5. Change and Risk Management

### 5.1 Change Management
* **Change Control:** Changes to scope, requirements, or baseline plans must be submitted via a change request process (formal documentation TBD if needed, initially via discussion and agreement between Project Lead and AI).
* **Change Approval:** Significant changes require approval from the Project Sponsor (Project Lead). Minor adjustments may be handled by the Project Manager (AI) with notification.
* **Change Log:** Changes impacting baseline documents (SRS, Plan, Charter) will be tracked via version control in the documents themselves.

### 5.2 Risk Management
* **Risk Identification & Analysis:** Performed and documented in the Risk Analysis Document (`risk_analysis_gal_friday_v0.1`). Key risks include market volatility, model failure, API dependency, software bugs, and security threats.
* **Risk Mitigation:** Strategies are defined in the Risk Analysis Document and implemented via requirements in the SRS, robust testing, monitoring, and operational procedures.
* **Risk Monitoring:** Risks will be monitored throughout the project lifecycle, particularly during testing and live operation. The Risk Analysis document may be updated if new significant risks emerge.

---
**End of Document**
