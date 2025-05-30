-- A/B Testing Experiment tables

-- Main experiments table
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id UUID PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    control_model_id UUID NOT NULL,
    treatment_model_id UUID NOT NULL,
    allocation_strategy VARCHAR(50) NOT NULL, -- 'random', 'deterministic', 'weighted', 'epsilon_greedy'
    traffic_split DECIMAL(3, 2) NOT NULL CHECK (traffic_split > 0 AND traffic_split < 1),
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    min_samples_per_variant INTEGER DEFAULT 1000,
    primary_metric VARCHAR(100) NOT NULL,
    secondary_metrics JSONB,
    confidence_level DECIMAL(3, 2) DEFAULT 0.95,
    minimum_detectable_effect DECIMAL(5, 4) DEFAULT 0.01,
    max_loss_threshold DECIMAL(10, 2),
    status VARCHAR(50) NOT NULL DEFAULT 'created', -- 'created', 'running', 'paused', 'completed', 'failed'
    completion_reason TEXT,
    results JSONB,
    config_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX idx_experiments_status ON experiments(status);
CREATE INDEX idx_experiments_dates ON experiments(start_time, end_time);
CREATE INDEX idx_experiments_models ON experiments(control_model_id, treatment_model_id);

-- Variant assignments tracking
CREATE TABLE IF NOT EXISTS experiment_assignments (
    experiment_id UUID REFERENCES experiments(experiment_id),
    event_id UUID NOT NULL,
    variant VARCHAR(20) NOT NULL, -- 'control' or 'treatment'
    assigned_at TIMESTAMP NOT NULL,
    PRIMARY KEY (experiment_id, event_id)
);

CREATE INDEX idx_assignments_experiment ON experiment_assignments(experiment_id);
CREATE INDEX idx_assignments_timestamp ON experiment_assignments(assigned_at);

-- Experiment outcomes for analysis
CREATE TABLE IF NOT EXISTS experiment_outcomes (
    outcome_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID REFERENCES experiments(experiment_id),
    event_id UUID NOT NULL,
    variant VARCHAR(20) NOT NULL,
    outcome_data JSONB NOT NULL,
    correct_prediction BOOLEAN,
    signal_generated BOOLEAN,
    trade_return DECIMAL(10, 4),
    recorded_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_outcomes_experiment ON experiment_outcomes(experiment_id);
CREATE INDEX idx_outcomes_variant ON experiment_outcomes(experiment_id, variant);
CREATE INDEX idx_outcomes_timestamp ON experiment_outcomes(recorded_at);

-- Experiment metrics summary (materialized view for performance)
CREATE MATERIALIZED VIEW IF NOT EXISTS experiment_metrics_summary AS
SELECT 
    e.experiment_id,
    e.name,
    e.status,
    e.start_time,
    e.end_time,
    COUNT(DISTINCT ea.event_id) as total_assignments,
    COUNT(DISTINCT CASE WHEN ea.variant = 'control' THEN ea.event_id END) as control_assignments,
    COUNT(DISTINCT CASE WHEN ea.variant = 'treatment' THEN ea.event_id END) as treatment_assignments,
    AVG(CASE WHEN eo.variant = 'control' AND eo.correct_prediction THEN 1 ELSE 0 END) as control_accuracy,
    AVG(CASE WHEN eo.variant = 'treatment' AND eo.correct_prediction THEN 1 ELSE 0 END) as treatment_accuracy,
    SUM(CASE WHEN eo.variant = 'control' THEN eo.trade_return ELSE 0 END) as control_total_return,
    SUM(CASE WHEN eo.variant = 'treatment' THEN eo.trade_return ELSE 0 END) as treatment_total_return
FROM experiments e
LEFT JOIN experiment_assignments ea ON e.experiment_id = ea.experiment_id
LEFT JOIN experiment_outcomes eo ON e.experiment_id = eo.experiment_id AND ea.event_id = eo.event_id
GROUP BY e.experiment_id, e.name, e.status, e.start_time, e.end_time;

CREATE INDEX idx_metrics_summary_experiment ON experiment_metrics_summary(experiment_id); 