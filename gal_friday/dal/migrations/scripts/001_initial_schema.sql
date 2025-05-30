-- Initial database schema

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id UUID NOT NULL,
    trading_pair VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    limit_price DECIMAL(20, 8),
    status VARCHAR(20) NOT NULL,
    exchange_order_id VARCHAR(100),
    filled_quantity DECIMAL(20, 8) DEFAULT 0,
    average_fill_price DECIMAL(20, 8),
    commission DECIMAL(20, 8),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE INDEX idx_orders_signal_id ON orders(signal_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_created_at ON orders(created_at);

-- Positions table
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trading_pair VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8) NOT NULL,
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
    opened_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

CREATE INDEX idx_positions_pair ON positions(trading_pair);
CREATE INDEX idx_positions_active ON positions(is_active);
CREATE INDEX idx_positions_opened_at ON positions(opened_at);

-- Trade signals table
CREATE TABLE IF NOT EXISTS trade_signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trading_pair VARCHAR(20) NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,
    entry_price DECIMAL(20, 8),
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    confidence DECIMAL(5, 4),
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    executed_at TIMESTAMP
);

CREATE INDEX idx_signals_status ON trade_signals(status);
CREATE INDEX idx_signals_created_at ON trade_signals(created_at);

-- Reconciliation events table
CREATE TABLE IF NOT EXISTS reconciliation_events (
    reconciliation_id UUID PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    reconciliation_type VARCHAR(50),
    status VARCHAR(50),
    discrepancies_found INTEGER DEFAULT 0,
    auto_corrected INTEGER DEFAULT 0,
    manual_review_required INTEGER DEFAULT 0,
    report JSONB,
    duration_seconds DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_reconciliation_timestamp ON reconciliation_events(timestamp);
CREATE INDEX idx_reconciliation_status ON reconciliation_events(status);

-- Position adjustments table
CREATE TABLE IF NOT EXISTS position_adjustments (
    adjustment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    reconciliation_id UUID REFERENCES reconciliation_events(reconciliation_id),
    trading_pair VARCHAR(20),
    adjustment_type VARCHAR(50),
    old_value DECIMAL(20, 8),
    new_value DECIMAL(20, 8),
    reason TEXT,
    adjusted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_adjustments_reconciliation ON position_adjustments(reconciliation_id);

-- Model versions table
CREATE TABLE IF NOT EXISTS model_versions (
    model_id UUID PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    training_completed_at TIMESTAMP,
    stage VARCHAR(50) DEFAULT 'development',
    metrics JSONB,
    hyperparameters JSONB,
    feature_importance JSONB,
    artifact_path TEXT,
    UNIQUE(model_name, version)
);

CREATE INDEX idx_models_name ON model_versions(model_name);
CREATE INDEX idx_models_stage ON model_versions(stage);

-- Model deployments table
CREATE TABLE IF NOT EXISTS model_deployments (
    deployment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES model_versions(model_id),
    deployed_at TIMESTAMP NOT NULL,
    deployed_by VARCHAR(255),
    deployment_config JSONB,
    is_active BOOLEAN DEFAULT true
);

CREATE INDEX idx_deployments_model ON model_deployments(model_id);
CREATE INDEX idx_deployments_active ON model_deployments(is_active); 