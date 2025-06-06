# Gal Friday Real-Time Trading Dashboard

A comprehensive real-time dashboard for monitoring trading metrics, portfolio performance, and system health in the Gal Friday trading system.

## 🚀 Features

- **Real-time Data Streaming**: WebSocket-based live updates without page refresh
- **Portfolio Visualization**: Current positions, P&L, and performance metrics
- **Trading Metrics**: Win rate, Sharpe ratio, drawdown analysis
- **System Monitoring**: CPU, memory usage, and health scores
- **Alert Management**: Visual notifications and alert history
- **Responsive Design**: Works across desktop and mobile devices
- **REST API**: Programmatic access to dashboard data

## 📦 Installation

The dashboard is included with the Gal Friday system. Ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `websockets` - WebSocket support
- `aiohttp` - HTTP client (for testing)

## 🖥️ Usage

### Starting the Dashboard

1. **Simple Start** (recommended):
   ```bash
   cd gal_friday/monitoring
   python run_dashboard.py
   ```

2. **Custom Configuration**:
   ```bash
   DASHBOARD_HOST=localhost DASHBOARD_PORT=8080 python run_dashboard.py
   ```

3. **Access the Dashboard**:
   - Main Dashboard: http://localhost:8000/static/dashboard.html
   - API Root: http://localhost:8000/
   - Health Check: http://localhost:8000/health
   - WebSocket: ws://localhost:8000/ws

### Testing

Run the test suite to verify functionality:

```bash
# Unit tests
python test_dashboard.py

# Test against running server
python test_dashboard.py server 8000
```

## 🏗️ Architecture

### Components

1. **DashboardService**: Core service providing aggregated metrics
2. **RealTimeDashboard**: WebSocket-enabled real-time dashboard
3. **Dashboard HTML/JS**: Frontend client with responsive UI
4. **Widget System**: Modular dashboard components

### Data Flow

```
Portfolio Manager → Dashboard Service → Real-Time Dashboard → WebSocket → Frontend
```

### Widget Types

- `portfolio_value`: Portfolio overview and positions
- `trading_metrics`: Performance statistics  
- `system_metrics`: System health and resources
- `alert_panel`: Alerts and notifications

## 📊 API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard status |
| `/health` | GET | Health check |
| `/api/widgets` | GET | All current widget data |
| `/api/data/{widget_type}` | GET | Specific widget data |
| `/static/dashboard.html` | GET | Main dashboard UI |

### WebSocket Messages

#### Client → Server

```json
{
  "type": "ping"
}
```

```json
{
  "type": "subscribe",
  "widgets": ["portfolio_value", "trading_metrics"]
}
```

#### Server → Client

```json
{
  "type": "initial_data",
  "data": { /* all widget data */ },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

```json
{
  "type": "data_update",
  "widget_type": "portfolio_value",
  "data": { /* updated widget data */ },
  "timestamp": "2024-01-15T10:30:05Z"
}
```

## 🎨 Customization

### Update Intervals

Configure update frequencies in `dashboard_config`:

```python
dashboard_config = {
    "update_intervals": {
        "portfolio": 5,    # seconds
        "metrics": 10,
        "system": 15,
        "alerts": 30
    }
}
```

### Adding Custom Widgets

1. Define widget type in `WidgetType` enum
2. Add update method in `RealTimeDashboard`
3. Create frontend component in dashboard.html
4. Add message handler in JavaScript

### Styling

The dashboard uses modern CSS with:
- CSS Grid for responsive layout
- CSS Variables for theming
- Smooth animations and transitions
- Mobile-first responsive design

## 🔧 Configuration

### Environment Variables

- `DASHBOARD_HOST`: Server host (default: "0.0.0.0")
- `DASHBOARD_PORT`: Server port (default: 8000)

### Dashboard Config

```python
dashboard_config = {
    "update_intervals": {...},
    "max_connections": 100,
    "heartbeat_interval": 30
}
```

## 🐛 Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Check if server is running
   - Verify firewall settings
   - Check browser console for errors

2. **No Data Updates**
   - Verify portfolio manager is connected
   - Check server logs for errors
   - Ensure update tasks are running

3. **Performance Issues**
   - Reduce update frequencies
   - Limit number of concurrent connections
   - Check system resources

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check WebSocket connections:
```bash
curl http://localhost:8000/api/widgets
```

## 🧪 Testing

### Unit Tests
```bash
cd gal_friday/monitoring
python test_dashboard.py
```

### Integration Tests
```bash
# Start dashboard in terminal 1
python run_dashboard.py

# Test in terminal 2
python test_dashboard.py server 8000
```

### Load Testing

For production deployment, test WebSocket connections:

```bash
# Install wscat for WebSocket testing
npm install -g wscat

# Test WebSocket connection
wscat -c ws://localhost:8000/ws
```

## 📈 Metrics Reference

### Portfolio Metrics
- `total_equity`: Total portfolio value
- `daily_pnl`: Daily profit/loss
- `total_pnl`: Total unrealized P&L
- `cash_balance`: Available cash
- `positions_value`: Value of positions
- `positions`: List of current positions

### Trading Metrics
- `total_trades`: Number of trades executed
- `win_rate`: Percentage of profitable trades
- `sharpe_ratio`: Risk-adjusted return measure
- `max_drawdown`: Maximum portfolio decline

### System Metrics
- `health_score`: Overall system health (0-1)
- `cpu_percent`: CPU utilization
- `memory_percent`: Memory utilization
- `uptime_seconds`: System uptime

## 🚦 Production Deployment

### Security Considerations
- Use HTTPS/WSS in production
- Implement authentication/authorization
- Rate limit WebSocket connections
- Validate all client messages

### Performance Optimization
- Use Redis for caching metrics
- Implement connection pooling
- Add CDN for static assets
- Monitor resource usage

### Monitoring
- Set up alerts for high CPU/memory
- Monitor WebSocket connection counts
- Track API response times
- Log all errors for debugging

## 📝 License

This dashboard is part of the Gal Friday trading system and follows the same license terms. 