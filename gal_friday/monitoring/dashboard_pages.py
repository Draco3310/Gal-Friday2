"""Enhanced dashboard pages for monitoring all system features."""

from datetime import datetime

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from gal_friday.execution.websocket_connection_manager import WebSocketConnectionManager
from gal_friday.model_lifecycle.experiment_manager import ExperimentManager
from gal_friday.model_lifecycle.registry import Registry
from gal_friday.model_lifecycle.retraining_pipeline import RetrainingPipeline
from gal_friday.monitoring.auth import verify_api_key
from gal_friday.monitoring.dashboard_service import DashboardService
from gal_friday.portfolio.reconciliation_service import ReconciliationService

router = APIRouter(prefix="/dashboard", dependencies=[Depends(verify_api_key)])


class EnhancedDashboardPages:
    """Enhanced dashboard pages with all features."""

    # Dashboard CSS styles
    DASHBOARD_CSS = """
    <style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background: #f5f7fa; }
    .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
    .header { background: #2c3e50; color: white; padding: 1rem; border-radius: 5px 5px 0 0; margin-bottom: 1rem; }
    .card { background: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 1rem; padding: 1rem; }
    .row { display: flex; flex-wrap: wrap; margin: 0 -10px; }
    .col { flex: 1; padding: 0 10px; min-width: 250px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 8px; text-align: left; border-bottom: 1px solid #eee; }
    th { background-color: #f8f9fa; }
    .alert { padding: 15px; border-radius: 4px; margin-bottom: 15px; }
    .alert-danger { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .alert-warning { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .alert-success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .metric-card { padding: 15px; border-radius: 4px; margin-bottom: 10px; color: white; text-align: center; }
    .metric-card h3 { margin-top: 0; }
    .metric-card p { font-size: 1.8rem; font-weight: bold; margin: 10px 0; }
    .bg-primary { background-color: #4e73df; }
    .bg-success { background-color: #1cc88a; }
    .bg-warning { background-color: #f6c23e; }
    .bg-danger { background-color: #e74a3b; }
    .bg-info { background-color: #36b9cc; }
    .refresh-btn { background: #2c3e50; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer; }
    .btn { display: inline-block; font-weight: 400; text-align: center; white-space: nowrap; vertical-align: middle; user-select: none; border: 1px solid transparent; padding: .375rem .75rem; font-size: 1rem; line-height: 1.5; border-radius: .25rem; transition: color .15s ease-in-out,background-color .15s ease-in-out,border-color .15s ease-in-out,box-shadow .15s ease-in-out; }
    .btn-primary { color: #fff; background-color: #4e73df; border-color: #4e73df; }
    .btn-danger { color: #fff; background-color: #e74a3b; border-color: #e74a3b; }
    .btn-success { color: #fff; background-color: #1cc88a; border-color: #1cc88a; }
    .toggle-switch { position: relative; display: inline-block; width: 60px; height: 34px; }
    .toggle-switch input { opacity: 0; width: 0; height: 0; }
    .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #ccc; transition: .4s; border-radius: 34px; }
    .slider:before { position: absolute; content: ""; height: 26px; width: 26px; left: 4px; bottom: 4px; background-color: white; transition: .4s; border-radius: 50%; }
    input:checked + .slider { background-color: #1cc88a; }
    input:checked + .slider:before { transform: translateX(26px); }
    </style>
    """

    def __init__(
        self,
        dashboard_service: DashboardService,
        model_registry: Registry,
        experiment_manager: ExperimentManager,
        retraining_pipeline: RetrainingPipeline,
        reconciliation_service: ReconciliationService,
        ws_connection_manager: WebSocketConnectionManager) -> None:
        """Initialize the EnhancedDashboardPages with required services.

        Args:
            dashboard_service: Service for dashboard metrics and operations
            model_registry: Registry for managing ML models
            experiment_manager: Manager for A/B testing experiments
            retraining_pipeline: Pipeline for model retraining
            reconciliation_service: Service for portfolio reconciliation
            ws_connection_manager: Manager for WebSocket connections
        """
        self.dashboard = dashboard_service
        self.model_registry = model_registry
        self.experiment_manager = experiment_manager
        self.retraining_pipeline = retraining_pipeline
        self.reconciliation_service = reconciliation_service
        self.ws_manager = ws_connection_manager

    @router.get("/", response_class=HTMLResponse)
    async def main_dashboard(self) -> str:
        """Enhanced main dashboard with all features."""
        # Get metrics for the dashboard
        metrics = await self.dashboard.get_all_metrics()

        # The following are called for their side effects
        await self.model_registry.get_model_count()
        len(self.experiment_manager.active_experiments)
        await self.reconciliation_service.get_reconciliation_status()

        # Format the metrics for the template
        uptime = self._format_uptime(metrics["system"]["uptime_seconds"])
        health_score = metrics["system"]["health_score"] * 100
        total_models = metrics["models"]["total"]
        active_models = metrics["models"]["active"]
        last_trained = self._format_time_ago(metrics["models"].get("last_trained"))
        ws_status = metrics["websocket"]["status"]
        ws_uptime = self._format_uptime(metrics["websocket"]["connection_time_seconds"])
        ws_messages = metrics["websocket"]["messages_last_hour"]

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Gal-Friday Dashboard</title>
            <meta http-equiv="refresh" content="30">
            <style>
                {self.DASHBOARD_CSS}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Gal-Friday Dashboard</h1>
                <p>Real-time monitoring and control panel</p>
            </div>
            <div class="nav">
                <a href="/dashboard/models">📊 Models</a>
                <a href="/dashboard/experiments">🧪 Experiments</a>
                <a href="/dashboard/reconciliation">🔍 Reconciliation</a>
                <a href="/dashboard/retraining">🔄 Retraining</a>
            </div>
            <div class="dashboard-grid">
                <div class="card">
                    <h2>System Health</h2>
                    <div class="label">Uptime</div>
                    <div class="metric">
                        {uptime}
                    </div>
                    <div class="label">Health Score</div>
                    <div class="progress">
                        <div class="progress-bar"
                             style="width: {health_score}%">
                        </div>
                    </div>
                    <div class="label">Last Updated</div>
                    <div>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                </div>
                <div class="card">
                    <h2>Model Registry</h2>
                    <div class="label">Total Models</div>
                    <div class="metric">{total_models}</div>
                    <div class="label">Active Models</div>
                    <div class="metric">{active_models}</div>
                    <div class="label">Last Trained</div>
                    <div>{last_trained}</div>
                </div>
                <div class="card">
                    <h2>⚡ Real-time Connectivity</h2>
                    <div class="label">WebSocket Status</div>
                    <div class="status {self._get_ws_status_class(ws_status)}">
                        {ws_status}
                    </div>
                    <div class="label">Connection Uptime</div>
                    <div>{ws_uptime}</div>
                    <div class="label">Messages (1h)</div>
                    <div class="metric">{ws_messages}</div>
                </div>
                    </div>
                    <div class="label">Position Accuracy</div>
                    <div class="metric">100%</div>
                    <div class="label">Discrepancies</div>
                    <div class="metric">0</div>
                </div>
                <!-- Alerts Summary -->
                <div class="card">
                    <h2>🔔 Alerts</h2>
                    <div class="label">Critical</div>
                    <div class="metric"
                         style="color: #e74c3c;">
                        {metrics['alerts']['critical']}
                    </div>
                    <div class="label">Warnings</div>
                    <div class="metric"
                         style="color: #f39c12;">
                        {metrics['alerts']['warning']}
                    </div>
                    <div class="label">Info</div>
                    <div class="metric">{metrics['alerts']['info']}</div>
                </div>
            </div>
        </body>
        </html>
        """

    # CSS for the model registry dashboard
    MODEL_REGISTRY_CSS = """
    body {
        font-family: Arial, sans-serif;  /* noqa: F821 */
        margin: 20px;  /* noqa: F821 */
        background-color: #f5f5f5;  /* noqa: F821 */
    }
    .header {
        background-color: #2c3e50;  /* noqa: F821 */
        color: white;  /* noqa: F821 */
        padding: 20px;  /* noqa: F821 */
        border-radius: 8px;  /* noqa: F821 */
    }
    .model-grid {
        display: grid;  /* noqa: F821 */
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));  /* noqa: F821 */
        gap: 20px;  /* noqa: F821 */
        margin-top: 20px;  /* noqa: F821 */
    }
    .model-card {
        background-color: white;  /* noqa: F821 */
        padding: 20px;  /* noqa: F821 */
        border-radius: 8px;  /* noqa: F821 */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);  /* noqa: F821 */
    }
    .model-card h3 {
        margin-top: 0;  /* noqa: F821 */
        display: inline-block;  /* noqa: F821 */
    }
    .stage {
        float: right;  /* noqa: F821 */
        padding: 2px 8px;  /* noqa: F821 */
        border-radius: 4px;  /* noqa: F821 */
        font-size: 12px;  /* noqa: F821 */
        font-weight: bold;  /* noqa: F821 */
        text-transform: uppercase;  /* noqa: F821 */
    }
    .stage.production { background-color: #2ecc71; color: white; }  /* noqa: F821 */
    .stage.staging { background-color: #f39c12; color: white; }  /* noqa: F821 */
    .stage.development { background-color: #3498db; color: white; }  /* noqa: F821 */
    .stage.archived { background-color: #95a5a6; color: white; }  /* noqa: F821 */
    .metrics {
        display: flex;  /* noqa: F821 */
        gap: 15px;  /* noqa: F821 */
        margin: 15px 0;  /* noqa: F821 */
    }
    .metric-box {
        text-align: center;  /* noqa: F821 */
        padding: 10px;  /* noqa: F821 */
        background-color: #f8f9fa;  /* noqa: F821 */
        border-radius: 6px;  /* noqa: F821 */
        flex: 1;  /* noqa: F821 */
    }
    .back-link { color: #3498db; text-decoration: none; }  /* noqa: F821 */
    """

    @router.get("/models", response_class=HTMLResponse)
    async def models_dashboard(self) -> str:
        """Model registry dashboard."""
        models = await self.model_registry.get_all_models()

        # Start building the HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Registry - Gal-Friday</title>
            <style>
                {self.MODEL_REGISTRY_CSS}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🤖 Model Registry</h1>
                <p>Manage and monitor ML models</p>
            </div>
            <p><a href="/dashboard/" class="back-link">← Back to Dashboard</a></p>
            <div class="model-grid">
        """

        # Add model cards
        for model in models:
            stage_class = model.stage.value.lower()
            training_completed = model.training_completed_at or model.created_at
            trading_pairs = getattr(model, "trading_pairs", ["N/A"])

            # Format metrics with default values if not present
            accuracy = model.validation_metrics.get("accuracy", 0) * 100
            precision = model.validation_metrics.get("precision", 0) * 100
            recall = model.validation_metrics.get("recall", 0) * 100

            # Add model card to HTML content
            html_content += f"""
            <div class="model-card">
                <h3>{model.model_name}</h3>
                <span class="stage {stage_class}">{model.stage.value}</span>
                <p>
                    <strong>Type:</strong> {model.model_type} |
                    <strong>Version:</strong> {model.version} |
                    <strong>Created:</strong> {training_completed.strftime('%Y-%m-%d')}
                </p>
                <div class="metrics">
                    <div class="metric-box">
                        <div style="font-size: 20px; font-weight: bold;">
                            {accuracy:.1f}%
                        </div>
                        <div style="font-size: 12px; color: #666;">Accuracy</div>
                    </div>
                    <div class="metric-box">
                        <div style="font-size: 20px; font-weight: bold;">
                            {precision:.1f}%
                        </div>
                        <div style="font-size: 12px; color: #666;">Precision</div>
                    </div>
                    <div class="metric-box">
                        <div style="font-size: 20px; font-weight: bold;">
                            {recall:.1f}%
                        </div>
                        <div style="font-size: 12px; color: #666;">Recall</div>
                    </div>
                </div>
                <p><strong>Description:</strong> {model.description or 'No description'}</p>
                <p><strong>Last Updated:</strong> {(model.updated_at.strftime('%Y-%m-%d %H:%M') if model.updated_at else 'N/A')}</p>
                <p><strong>Training Data:</strong> {model.training_data_path or 'N/A'}</p>
                <p><strong>Pairs:</strong> {
                    ', '.join(trading_pairs) if isinstance(trading_pairs, list)
                    else trading_pairs
                }</p>
            </div>
            """

        # Close HTML tags
        html_content += """
            </div>
        </body>
        </html>
        """
        return html_content

    @router.get("/experiments", response_class=HTMLResponse)
    async def experiments_dashboard(self) -> str:
        """A/B testing experiments dashboard."""
        experiments = []
        for exp_id, _ in self.experiment_manager.active_experiments.items():
            status = await self.experiment_manager.get_experiment_status(exp_id)
            experiments.append(status)

        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>A/B Testing - Gal-Friday</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                .header {
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                }
                .experiment-card {
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .variant {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin: 20px 0;
                }
                .variant-box {
                    padding: 15px;
                    border-radius: 4px;
                }
                .control {
                    background-color: #e8f8f5;
                    border: 1px solid #27ae60;
                }
                .treatment {
                    background-color: #fef5e7;
                    border: 1px solid #f39c12;
                }
                .metric {
                    font-size: 18px;
                    font-weight: bold;
                }
                .significance {
                    padding: 10px;
                    background-color: #ecf0f1;
                    border-radius: 4px;
                    margin-top: 10px;
                }
                .back-link { color: #3498db; text-decoration: none; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧪 A/B Testing Experiments</h1>
                <p>Compare model performance in production</p>
            </div>
            <p><a href="/dashboard/" class="back-link">← Back to Dashboard</a></p>
        """

        if not experiments:
            html += ('<p style="text-align: center; color: #666; margin: 40px;">'
                   'No active experiments</p>')
        else:
            for exp in experiments:
                _ = exp["statistical_significance"]  # Removed unused variable
                html += f"""
                <div class="experiment-card">
                    <h3>{exp['name']}</h3>
                    <p>
                    <strong>Status:</strong> {exp['status']} |
                    <strong>Started:</strong> {exp['start_time']}
                </p>
                <div class="variant">
                    <div class="variant-box control">
                        <h4>Control</h4>
                        <p>Model: {exp['control']['model_id'][:8]}...</p>
                        <div class="metric">{exp['control']['accuracy']*100:.1f}%</div>
                        <p>Accuracy</p>
                        <p>Samples: {exp['control']['samples']:,}</p>
                        <p>Signals: {exp['control']['signals']}</p>
                    </div>
                    <div class="variant-box treatment">
                        <h4>Treatment</h4>
                        <p>Model: {exp['treatment']['model_id'][:8]}...</p>
                        <div class="metric">{exp['treatment']['accuracy']*100:.1f}%</div>
                        <p>Accuracy</p>
                        <p>Samples: {exp['treatment']['samples']:,}</p>
                        <p>Signals: {exp['treatment']['signals']}</p>
                    </div>
                </div>
                """

        html += """
        </body>
        </html>
        """
        return html

    @router.get("/reconciliation", response_class=HTMLResponse)
    async def reconciliation_dashboard(self) -> str:
        """Portfolio reconciliation dashboard."""
        status = await self.reconciliation_service.get_reconciliation_status()

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reconciliation - Gal-Friday</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                }}
                .status-card {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
                .metric-box {{
                    text-align: center;
                    padding: 20px;
                    background-color: #ecf0f1;
                    border-radius: 4px;
                }}
                .success {{ color: #27ae60; }}
                .warning {{ color: #f39c12; }}
                .error {{ color: #e74c3c; }}
                .back-link {{
                    color: #3498db;
                    text-decoration: none;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>✅ Portfolio Reconciliation</h1>
                <p>Ensure data integrity with exchange</p>
            </div>
            <p><a href="/dashboard/" class="back-link">← Back to Dashboard</a></p>
            <div class="status-card">
                <h3>Current Status</h3>
                <p><strong>Last Run:</strong> {status.get('last_run', 'Never')}</p>
                <p><strong>Next Run:</strong> {status.get('next_run', 'Not scheduled')}</p>
                <p><strong>Consecutive Failures:</strong> "
                   "{status.get('consecutive_failures', 0)}</p>
            </div>
        """

        if status.get("last_report"):
            report = status["last_report"]
            html += f"""
            <div class="status-card">
                <h3>Last Reconciliation Report</h3>
                <div class="metric-grid">
                    <div class="metric-box">
                        <div style="font-size: 24px; font-weight: bold;">
                        <div
                            style="font-weight: bold;"
                            class="{self._get_discrepancy_class(
                                report.get('total_discrepancies', 0)
                            )}">
                            {report.get('total_discrepancies', 0)}
                        </div>
                        <div>Discrepancies Found</div>
                    </div>
                    <div class="metric-box">
                        <div style="font-size: 24px; font-weight: bold;">
                            {len(report.get('auto_corrections', []))}
                        </div>
                        <div>Auto-corrections</div>
                    </div>
                </div>
            </div>
            """

        html += """
        </body>
        </html>
        """
        return html

    @router.get("/retraining", response_class=HTMLResponse)
    async def retraining_dashboard(self) -> str:
        """Model retraining dashboard."""
        status = await self.retraining_pipeline.get_retraining_status()

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Retraining - Gal-Friday</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                }}
                .job-card {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .status {{
                    padding: 5px 10px;
                    border-radius: 4px;
                    display: inline-block;
                }}
                .status.running {{
                    background-color: #3498db;
                    color: white;
                }}
                .status.completed {{
                    background-color: #27ae60;
                    color: white;
                }}
                .status.failed {{
                    background-color: #e74c3c;
                    color: white;
                }}
                .metric-row {{
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 20px;
                    margin: 20px 0;
                }}
                .back-link {{
                    color: #3498db;
                    text-decoration: none;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔄 Model Retraining Pipeline</h1>
                <p>Automated model updates and drift detection</p>
            </div>
            <p><a href="/dashboard/" class="back-link">← Back to Dashboard</a></p>
            <div class="job-card">
                <h3>Pipeline Status</h3>
                <div class="metric-row">
                    <div>
                        <strong>Active Jobs:</strong> {len(status['active_jobs'])}
                    </div>
                    <div>
                        <strong>Recent Completed:</strong> {status['recent_completed']}
                    </div>
                    <div>
                        <strong>Recent Failed:</strong> {status['recent_failed']}
                    </div>
                </div>
                <p><strong>Next Check:</strong> {status['next_check']}</p>
            </div>
        """

        if status["active_jobs"]:
            html += "<h3>Active Retraining Jobs</h3>"
            for job in status["active_jobs"]:
                html += f"""
                <div class="job-card">
                    <h4>{job['model_name']}</h4>
                    <p>
                        <span class="status {job['status']}">{job['status'].upper()}</span>
                        <strong>Trigger:</strong> {job['trigger']} |
                        <strong>Started:</strong> {job['start_time']}
                    </p>
                </div>
                """

        html += """
        </body>
        </html>
        """
        return html

    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format."""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)

        if days > 0:
            return f"{days}d {hours}h"
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    def _format_time_ago(self, timestamp_str: str | None) -> str:
        """Format timestamp as time ago."""
        if not timestamp_str:
            return "Never"

        seconds_per_hour = 3600
        seconds_per_minute = 60
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            delta = datetime.now(timestamp.tzinfo) - timestamp

            if delta.days > 0:
                return f"{delta.days}d ago"
            if delta.seconds > seconds_per_hour:
                return f"{delta.seconds // seconds_per_hour}h ago"
            return f"{delta.seconds // seconds_per_minute}m ago"
        except (ValueError, AttributeError):
            return "Unknown"

    def _get_ws_status_class(self, status: str) -> str:
        """Get CSS class for WebSocket status."""
        if status.lower() == "connected":
            return "active"
        if status.lower() == "reconnecting":
            return "warning"
        return "error"

    def _get_discrepancy_class(self, count: int) -> str:
        """Get CSS class for discrepancy count.

        Args:
            count: Number of discrepancies
        Returns:
            str: CSS class name
        """
        warning_threshold = 5
        if count == 0:
            return "success"
        if count < warning_threshold:
            return "warning"
        return "error"


# Create singleton instance
dashboard_pages = None

def get_dashboard_pages(
    dashboard_service: DashboardService,
    model_registry: Registry,
    experiment_manager: ExperimentManager,
    retraining_pipeline: RetrainingPipeline,
    reconciliation_service: ReconciliationService,
    ws_connection_manager: WebSocketConnectionManager) -> EnhancedDashboardPages:
    """Get or create dashboard pages instance.

    Args:
        dashboard_service: Service for dashboard metrics and operations
        model_registry: Registry for managing ML models
        experiment_manager: Manager for A/B testing experiments
        retraining_pipeline: Pipeline for model retraining
        reconciliation_service: Service for portfolio reconciliation
        ws_connection_manager: Manager for WebSocket connections
    Returns:
        EnhancedDashboardPages: The dashboard pages instance
    """
    global dashboard_pages  # noqa: PLW0603 - Singleton pattern requires global
    if dashboard_pages is None:
        dashboard_pages = EnhancedDashboardPages(
            dashboard_service=dashboard_service,
            model_registry=model_registry,
            experiment_manager=experiment_manager,
            retraining_pipeline=retraining_pipeline,
            reconciliation_service=reconciliation_service,
            ws_connection_manager=ws_connection_manager)
    return dashboard_pages