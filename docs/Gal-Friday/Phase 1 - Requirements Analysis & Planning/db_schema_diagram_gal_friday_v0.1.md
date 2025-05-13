erDiagram
    signals {
        UUID signal_id PK "Signal ID (PK)"
        VARCHAR(16) trading_pair "Trading Pair"
        VARCHAR(32) exchange "Exchange"
        VARCHAR(64) strategy_id "Strategy ID"
        VARCHAR(4) side "Side (BUY/SELL)"
        VARCHAR(10) entry_type "Entry Type"
        NUMERIC proposed_entry_price "Proposed Entry Price"
        NUMERIC proposed_sl_price "Proposed SL Price"
        NUMERIC proposed_tp_price "Proposed TP Price"
        UUID prediction_event_id "Prediction Event ID (Optional FK)"
        REAL prediction_value "Prediction Value"
        VARCHAR(10) status "Status (Proposed/Approved/Rejected)"
        TEXT rejection_reason "Rejection Reason"
        JSONB risk_check_details "Risk Check Details"
        TIMESTAMPTZ created_at "Created Timestamp"
    }

    orders {
        INTEGER order_pk PK "Order PK (Internal)"
        UUID client_order_id UK "Client Order ID (UK)"
        VARCHAR(64) exchange_order_id UK "Exchange Order ID (UK)"
        UUID signal_id FK "Signal ID (FK)"
        VARCHAR(16) trading_pair "Trading Pair"
        VARCHAR(32) exchange "Exchange"
        VARCHAR(4) side "Side (BUY/SELL)"
        VARCHAR(16) order_type "Order Type"
        NUMERIC quantity_ordered "Quantity Ordered"
        NUMERIC limit_price "Limit Price"
        NUMERIC stop_price "Stop Price"
        VARCHAR(20) status "Order Status"
        TEXT error_message "Error Message"
        TIMESTAMPTZ created_at "Created Timestamp"
        TIMESTAMPTZ submitted_at "Submitted Timestamp"
        TIMESTAMPTZ last_updated_at "Last Updated Timestamp"
    }

    fills {
        INTEGER fill_pk PK "Fill PK (Internal)"
        VARCHAR(64) fill_id "Exchange Fill ID"
        INTEGER order_pk FK "Order PK (FK)"
        VARCHAR(64) exchange_order_id "Exchange Order ID"
        VARCHAR(16) trading_pair "Trading Pair"
        VARCHAR(32) exchange "Exchange"
        VARCHAR(4) side "Side (BUY/SELL)"
        NUMERIC quantity_filled "Quantity Filled"
        NUMERIC fill_price "Fill Price"
        NUMERIC commission "Commission Amount"
        VARCHAR(16) commission_asset "Commission Asset"
        VARCHAR(10) liquidity_type "Liquidity Type (Maker/Taker)"
        TIMESTAMPTZ filled_at "Filled Timestamp"
    }

    trades {
        INTEGER trade_pk PK "Trade PK (Internal)"
        UUID trade_id UK "Trade ID (UK)"
        UUID signal_id FK "Signal ID (FK)"
        VARCHAR(16) trading_pair "Trading Pair"
        VARCHAR(32) exchange "Exchange"
        VARCHAR(64) strategy_id "Strategy ID"
        VARCHAR(4) side "Entry Side (BUY/SELL)"
        INTEGER entry_order_pk FK "Entry Order PK (FK)"
        INTEGER exit_order_pk FK "Exit Order PK (FK)"
        TIMESTAMPTZ entry_timestamp "Entry Timestamp"
        TIMESTAMPTZ exit_timestamp "Exit Timestamp"
        NUMERIC quantity "Total Quantity"
        NUMERIC average_entry_price "Avg Entry Price"
        NUMERIC average_exit_price "Avg Exit Price"
        NUMERIC total_commission "Total Commission"
        NUMERIC realized_pnl "Realized PnL"
        REAL realized_pnl_pct "Realized PnL %"
        VARCHAR(32) exit_reason "Exit Reason"
    }

    system_logs {
        BIGINT log_pk PK "Log PK (Internal)"
        TIMESTAMPTZ log_timestamp "Log Timestamp"
        VARCHAR(64) source_module "Source Module"
        VARCHAR(10) log_level "Log Level"
        TEXT message "Message"
        VARCHAR(16) trading_pair "Context: Trading Pair"
        UUID signal_id "Context: Signal ID"
        INTEGER order_pk "Context: Order PK"
        TEXT exception_type "Exception Type"
        TEXT stack_trace "Stack Trace"
        JSONB context "Additional Context"
    }

    portfolio_snapshots {
        INTEGER snapshot_pk PK "Snapshot PK (Internal)"
        TIMESTAMPTZ snapshot_timestamp UK "Snapshot Timestamp (UK)"
        NUMERIC total_equity "Total Equity"
        NUMERIC available_balance "Available Balance"
        REAL total_exposure_pct "Total Exposure %"
        REAL daily_drawdown_pct "Daily Drawdown %"
        REAL weekly_drawdown_pct "Weekly Drawdown %"
        REAL total_drawdown_pct "Total Drawdown %"
        JSONB positions "Positions Details"
    }

    configurations {
        INTEGER config_pk PK "Config PK (Internal)"
        VARCHAR(64) config_hash UK "Config Hash (UK)"
        JSONB config_content "Configuration Content"
        TIMESTAMPTZ loaded_at "Loaded Timestamp"
        BOOLEAN is_active "Is Active Flag"
    }

    %% Relationships
    signals ||--o{ orders : triggers
    signals ||--o{ trades : originates
    orders ||--o{ fills : contains
    orders ||--|{ trades : entry_for
    orders ||--|{ trades : exit_for

    %% Optional relationships (dotted lines)
    system_logs }..o{ signals : references
    system_logs }..o{ orders : references
