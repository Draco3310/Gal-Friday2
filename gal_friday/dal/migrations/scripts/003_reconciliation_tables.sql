-- Reconciliation tables

-- Reconciliation events table
CREATE TABLE IF NOT EXISTS reconciliation_events (
    reconciliation_id UUID PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    reconciliation_type VARCHAR(50) NOT NULL, -- 'full', 'positions_only', 'balances_only'
    status VARCHAR(50) NOT NULL, -- 'success', 'failed', 'partial', 'in_progress'
    discrepancies_found INTEGER DEFAULT 0,
    auto_corrected INTEGER DEFAULT 0,
    manual_review_required INTEGER DEFAULT 0,
    report JSONB NOT NULL, -- Full report data
    duration_seconds DECIMAL(10, 3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_reconciliation_timestamp ON reconciliation_events(timestamp);
CREATE INDEX idx_reconciliation_status ON reconciliation_events(status);

-- Position adjustments table
CREATE TABLE IF NOT EXISTS position_adjustments (
    adjustment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    reconciliation_id UUID REFERENCES reconciliation_events(reconciliation_id),
    trading_pair VARCHAR(20) NOT NULL,
    adjustment_type VARCHAR(50) NOT NULL, -- 'position_quantity', 'balance', 'add_position'
    old_value DECIMAL(20, 8),
    new_value DECIMAL(20, 8),
    reason TEXT,
    adjusted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_adjustments_reconciliation ON position_adjustments(reconciliation_id);
CREATE INDEX idx_adjustments_pair ON position_adjustments(trading_pair);
CREATE INDEX idx_adjustments_timestamp ON position_adjustments(adjusted_at); 