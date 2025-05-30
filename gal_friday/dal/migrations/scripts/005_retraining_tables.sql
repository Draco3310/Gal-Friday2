-- Model retraining and drift detection tables

-- Retraining jobs table
CREATE TABLE IF NOT EXISTS retraining_jobs (
    job_id UUID PRIMARY KEY,
    model_id UUID NOT NULL,
    model_name VARCHAR(200) NOT NULL,
    trigger VARCHAR(50) NOT NULL, -- 'scheduled', 'drift_detected', 'performance_degraded', 'manual'
    drift_metrics JSONB,
    status VARCHAR(50) NOT NULL DEFAULT 'pending', -- 'pending', 'running', 'completed', 'failed'
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    new_model_id UUID,
    performance_comparison JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_retraining_model ON retraining_jobs(model_id);
CREATE INDEX idx_retraining_status ON retraining_jobs(status);
CREATE INDEX idx_retraining_created ON retraining_jobs(created_at);

-- Drift detection events table
CREATE TABLE IF NOT EXISTS drift_detection_events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL,
    drift_type VARCHAR(50) NOT NULL, -- 'concept_drift', 'data_drift', 'prediction_drift', 'performance_drift'
    metric_name VARCHAR(100) NOT NULL,
    drift_score DECIMAL(10, 6) NOT NULL,
    is_significant BOOLEAN DEFAULT FALSE,
    details JSONB,
    detected_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_drift_model ON drift_detection_events(model_id);
CREATE INDEX idx_drift_type ON drift_detection_events(drift_type);
CREATE INDEX idx_drift_detected ON drift_detection_events(detected_at);
CREATE INDEX idx_drift_significant ON drift_detection_events(is_significant);

-- Model performance history for drift detection
CREATE TABLE IF NOT EXISTS model_performance_history (
    performance_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL,
    evaluation_date DATE NOT NULL,
    predictions_made INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    signals_generated INTEGER DEFAULT 0,
    profitable_signals INTEGER DEFAULT 0,
    accuracy DECIMAL(5, 4),
    precision DECIMAL(5, 4),
    recall DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    total_return DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, evaluation_date)
);

CREATE INDEX idx_perf_model_date ON model_performance_history(model_id, evaluation_date);

-- Feature distribution snapshots for data drift detection
CREATE TABLE IF NOT EXISTS feature_distribution_snapshots (
    snapshot_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    snapshot_date DATE NOT NULL,
    mean DECIMAL(20, 8),
    std_dev DECIMAL(20, 8),
    min_value DECIMAL(20, 8),
    max_value DECIMAL(20, 8),
    p25 DECIMAL(20, 8),
    p50 DECIMAL(20, 8),
    p75 DECIMAL(20, 8),
    distribution_data JSONB, -- Full histogram data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_feature_snapshot ON feature_distribution_snapshots(model_id, feature_name, snapshot_date); 