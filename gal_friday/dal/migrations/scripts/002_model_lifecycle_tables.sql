-- Model lifecycle management tables

-- A/B testing experiments table
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    control_model_id UUID REFERENCES model_versions(model_id),
    treatment_model_id UUID REFERENCES model_versions(model_id),
    allocation_strategy VARCHAR(50) DEFAULT 'random',
    traffic_split DECIMAL(3, 2) DEFAULT 0.5,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    min_samples_per_variant INTEGER DEFAULT 1000,
    primary_metric VARCHAR(100),
    secondary_metrics JSONB,
    confidence_level DECIMAL(3, 2) DEFAULT 0.95,
    minimum_detectable_effect DECIMAL(5, 4) DEFAULT 0.01,
    max_loss_threshold DECIMAL(5, 4),
    status VARCHAR(50) DEFAULT 'created',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_experiments_status ON experiments(status);
CREATE INDEX idx_experiments_models ON experiments(control_model_id, treatment_model_id);

-- Prediction outcomes for A/B testing
CREATE TABLE IF NOT EXISTS prediction_outcomes (
    prediction_id UUID PRIMARY KEY,
    experiment_id UUID REFERENCES experiments(experiment_id),
    model_variant VARCHAR(20) NOT NULL, -- 'control' or 'treatment'
    timestamp TIMESTAMP NOT NULL,
    features JSONB,
    prediction DECIMAL(20, 8),
    confidence DECIMAL(5, 4),
    actual_outcome DECIMAL(20, 8),
    outcome_timestamp TIMESTAMP,
    error DECIMAL(20, 8),
    squared_error DECIMAL(20, 8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_outcomes_experiment ON prediction_outcomes(experiment_id);
CREATE INDEX idx_outcomes_timestamp ON prediction_outcomes(timestamp);

-- Model retraining jobs
CREATE TABLE IF NOT EXISTS retraining_jobs (
    job_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    trigger_reason VARCHAR(100),
    drift_type VARCHAR(50),
    drift_score DECIMAL(10, 6),
    drift_details JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    new_model_id UUID REFERENCES model_versions(model_id),
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_retraining_status ON retraining_jobs(status);
CREATE INDEX idx_retraining_model ON retraining_jobs(model_name);

-- Model performance history
CREATE TABLE IF NOT EXISTS model_performance_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES model_versions(model_id),
    timestamp TIMESTAMP NOT NULL,
    metric_name VARCHAR(100),
    metric_value DECIMAL(10, 6),
    sample_size INTEGER,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_performance_model ON model_performance_history(model_id);
CREATE INDEX idx_performance_timestamp ON model_performance_history(timestamp);

-- Drift detection results
CREATE TABLE IF NOT EXISTS drift_detection_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    check_timestamp TIMESTAMP NOT NULL,
    drift_detected BOOLEAN DEFAULT false,
    drift_type VARCHAR(50),
    drift_score DECIMAL(10, 6),
    p_value DECIMAL(10, 8),
    details JSONB,
    action_taken VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_drift_model ON drift_detection_results(model_name);
CREATE INDEX idx_drift_timestamp ON drift_detection_results(check_timestamp);
CREATE INDEX idx_drift_detected ON drift_detection_results(drift_detected); 