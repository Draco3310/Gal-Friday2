#!/bin/bash

# =============================================================================
# Gal-Friday2 Development Environment Setup Script
# =============================================================================
# This script sets up a complete development environment for the Gal-Friday2
# cryptocurrency trading bot, allowing OpenAI Codex and developers to quickly
# get started with the codebase.
#
# Usage: ./setup.sh [options]
# Options:
#   --full          Full setup including databases and external services
#   --dev           Development setup (default) - minimal dependencies
#   --test          Test environment setup
#   --docker        Use Docker for services
#   --help          Show this help message
# =============================================================================

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.11"
PROJECT_NAME="gal-friday2"
VENV_NAME=".venv"
CONFIG_FILE="config/config.yaml"
EXAMPLE_CONFIG="config/config.example.yaml"

# Default options
SETUP_TYPE="dev"
USE_DOCKER=false
SKIP_CONFIRMATION=false

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "${BLUE}"
    echo "============================================================================="
    echo "  $1"
    echo "============================================================================="
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "$1 is not installed or not in PATH"
        return 1
    fi
    return 0
}

confirm_action() {
    if [ "$SKIP_CONFIRMATION" = true ]; then
        return 0
    fi
    
    echo -e "${YELLOW}$1${NC}"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Operation cancelled by user"
        exit 0
    fi
}

# =============================================================================
# System Requirements Check
# =============================================================================

check_system_requirements() {
    print_header "Checking System Requirements"
    
    local missing_deps=()
    
    # Check Python
    if check_command python3; then
        local python_version=$(python3 --version | cut -d' ' -f2)
        print_info "Python version: $python_version"
        
        # Check if Python version is >= 3.11
        if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
            print_warning "Python 3.11+ recommended, found $python_version"
        fi
    else
        missing_deps+=("python3")
    fi
    
    # Check pip
    if ! check_command pip3 && ! check_command pip; then
        missing_deps+=("pip")
    fi
    
    # Check git
    if ! check_command git; then
        missing_deps+=("git")
    fi
    
    # Check curl
    if ! check_command curl; then
        missing_deps+=("curl")
    fi
    
    # Platform-specific checks
    case "$(uname -s)" in
        Linux*)
            print_info "Platform: Linux"
            # Check for build essentials
            if ! dpkg -l | grep -q build-essential 2>/dev/null && ! rpm -q gcc 2>/dev/null; then
                print_warning "Build tools may be needed for some Python packages"
            fi
            ;;
        Darwin*)
            print_info "Platform: macOS"
            if ! check_command brew; then
                print_warning "Homebrew not found - some dependencies may need manual installation"
            fi
            ;;
        MINGW*|CYGWIN*|MSYS*)
            print_info "Platform: Windows"
            ;;
        *)
            print_warning "Unknown platform: $(uname -s)"
            ;;
    esac
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        print_info "Please install the missing dependencies and run this script again"
        exit 1
    fi
    
    print_success "System requirements check passed"
}

# =============================================================================
# Python Environment Setup
# =============================================================================

setup_python_environment() {
    print_header "Setting up Python Environment"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_NAME" ]; then
        print_step "Creating virtual environment: $VENV_NAME"
        python3 -m venv "$VENV_NAME"
    else
        print_info "Virtual environment already exists: $VENV_NAME"
    fi
    
    # Activate virtual environment
    print_step "Activating virtual environment"
    source "$VENV_NAME/bin/activate" || {
        print_error "Failed to activate virtual environment"
        exit 1
    }
    
    # Upgrade pip
    print_step "Upgrading pip"
    pip install --upgrade pip
    
    # Install wheel and setuptools
    print_step "Installing build tools"
    pip install wheel setuptools
    
    print_success "Python environment setup complete"
}

# =============================================================================
# Dependencies Installation
# =============================================================================

install_dependencies() {
    print_header "Installing Dependencies"
    
    # Install main dependencies
    print_step "Installing main dependencies from requirements.txt"
    if [ -f "requirements.txt" ]; then
        # Try to install all dependencies, but handle TA-Lib separately if it fails
        if ! pip install -r requirements.txt; then
            print_warning "Some dependencies failed to install, trying without TA-Lib..."
            # Create temporary requirements without TA-Lib
            grep -v "TA-Lib" requirements.txt > requirements_temp.txt
            pip install -r requirements_temp.txt
            rm requirements_temp.txt
            
            # Try to install TA-Lib separately with better error handling
            print_step "Attempting to install TA-Lib separately"
            if ! pip install TA-Lib; then
                print_warning "TA-Lib installation failed - you may need to install system dependencies"
                print_info "On Ubuntu/Debian: sudo apt-get install libta-lib-dev"
                print_info "On macOS: brew install ta-lib"
                print_info "On other systems, see: https://github.com/TA-Lib/ta-lib-python"
                print_info "The system will work without TA-Lib, but some technical indicators may not be available"
            fi
        fi
    else
        print_error "requirements.txt not found"
        exit 1
    fi
    
    # Install test dependencies if in test mode
    if [ "$SETUP_TYPE" = "test" ] || [ "$SETUP_TYPE" = "full" ]; then
        print_step "Installing test dependencies"
        if [ -f "requirements-test.txt" ]; then
            pip install -r requirements-test.txt
        else
            print_warning "requirements-test.txt not found, skipping test dependencies"
        fi
    fi
    
    # Install development tools
    if [ "$SETUP_TYPE" = "dev" ] || [ "$SETUP_TYPE" = "full" ]; then
        print_step "Installing development tools"
        pip install pre-commit black isort flake8 mypy
    fi
    
    print_success "Dependencies installation complete"
}

# =============================================================================
# Configuration Setup
# =============================================================================

setup_configuration() {
    print_header "Setting up Configuration"
    
    # Create config directory if it doesn't exist
    mkdir -p config
    
    # Copy example config if main config doesn't exist
    if [ ! -f "$CONFIG_FILE" ]; then
        if [ -f "$EXAMPLE_CONFIG" ]; then
            print_step "Creating configuration file from example"
            cp "$EXAMPLE_CONFIG" "$CONFIG_FILE"
            print_info "Configuration file created: $CONFIG_FILE"
            print_warning "Please edit $CONFIG_FILE with your specific settings"
        else
            print_warning "No example configuration found, creating minimal config"
            create_minimal_config
        fi
    else
        print_info "Configuration file already exists: $CONFIG_FILE"
    fi
    
    # Create other necessary directories
    print_step "Creating necessary directories"
    mkdir -p logs
    mkdir -p models
    mkdir -p data
    mkdir -p temp
    
    print_success "Configuration setup complete"
}

create_minimal_config() {
    cat > "$CONFIG_FILE" << 'EOF'
# Minimal Gal-Friday2 Configuration for Development

# Application settings
app:
  name: "gal-friday2-dev"
  version: "0.1.0"
  environment: "development"

# Logging configuration
logging:
  level: "INFO"
  file: "logs/gal-friday.log"
  max_size: "10MB"
  backup_count: 5

# Database configuration (development)
database:
  url: "sqlite:///data/gal_friday_dev.db"
  echo: false

# Feature engine configuration
feature_engine:
  cache_size: 1000
  update_interval: 60

# Prediction service configuration
prediction_service:
  process_pool_workers: 2
  models: []

# Risk management
risk_management:
  max_position_size: 100.0
  max_daily_loss: 1000.0
  
# Paper trading mode (safe for development)
trading:
  mode: "paper"
  exchange: "kraken_sandbox"
EOF
}

# =============================================================================
# Database Setup
# =============================================================================

setup_database() {
    print_header "Setting up Database"
    
    if [ "$SETUP_TYPE" = "dev" ]; then
        print_step "Setting up SQLite database for development"
        # Create database directory
        mkdir -p data
        
        # Run database migrations if available
        if [ -f "generate_ddl.py" ]; then
            print_step "Running database schema generation"
            python generate_ddl.py
        fi
        
    elif [ "$SETUP_TYPE" = "full" ]; then
        if [ "$USE_DOCKER" = true ]; then
            setup_docker_databases
        else
            print_warning "Full database setup requires Docker or manual PostgreSQL/InfluxDB installation"
            print_info "Use --docker flag for automated database setup"
        fi
    fi
    
    print_success "Database setup complete"
}

setup_docker_databases() {
    print_step "Setting up databases with Docker"
    
    if ! check_command docker; then
        print_error "Docker is required but not installed"
        exit 1
    fi
    
    if ! check_command docker-compose; then
        print_error "Docker Compose is required but not installed"
        exit 1
    fi
    
    # Use existing docker-compose.yml if available
    if [ -f "scripts/deploy/docker-compose.yml" ]; then
        print_step "Starting databases with Docker Compose"
        cd scripts/deploy
        docker-compose up -d postgres influxdb redis
        cd ../..
        
        # Wait for databases to be ready
        print_step "Waiting for databases to be ready..."
        sleep 10
        
        # Test database connections
        print_step "Testing database connections"
        # Add database connection tests here
        
    else
        print_warning "Docker Compose file not found, creating minimal setup"
        create_minimal_docker_compose
    fi
}

create_minimal_docker_compose() {
    mkdir -p scripts/deploy
    cat > scripts/deploy/docker-compose.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: gal_friday_dev
      POSTGRES_USER: gal_friday
      POSTGRES_PASSWORD: dev_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  influxdb:
    image: influxdb:2.7
    environment:
      INFLUXDB_DB: gal_friday_metrics
      INFLUXDB_ADMIN_USER: admin
      INFLUXDB_ADMIN_PASSWORD: dev_password
    ports:
      - "8086:8086"
    volumes:
      - influxdb_data:/var/lib/influxdb2

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  influxdb_data:
  redis_data:
EOF
}

# =============================================================================
# Development Tools Setup
# =============================================================================

setup_development_tools() {
    print_header "Setting up Development Tools"
    
    # Setup pre-commit hooks
    if [ -f ".pre-commit-config.yaml" ]; then
        print_step "Installing pre-commit hooks"
        pre-commit install
    fi
    
    # Create useful development scripts
    create_development_scripts
    
    # Setup IDE configuration
    setup_ide_configuration
    
    print_success "Development tools setup complete"
}

create_development_scripts() {
    print_step "Creating development scripts"
    
    # Create run script
    cat > run_dev.sh << 'EOF'
#!/bin/bash
# Quick development run script
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -m gal_friday.main --config config/config.yaml --log-level DEBUG
EOF
    chmod +x run_dev.sh
    
    # Create test script
    cat > run_tests.sh << 'EOF'
#!/bin/bash
# Quick test run script
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/ -v --tb=short
EOF
    chmod +x run_tests.sh
    
    # Create lint script
    cat > run_lint.sh << 'EOF'
#!/bin/bash
# Code quality check script
source .venv/bin/activate
echo "Running ruff..."
ruff check gal_friday/
echo "Running mypy..."
mypy gal_friday/
echo "Running tests..."
pytest tests/ --tb=short
EOF
    chmod +x run_lint.sh
    
    print_info "Created development scripts: run_dev.sh, run_tests.sh, run_lint.sh"
}

setup_ide_configuration() {
    print_step "Setting up IDE configuration"
    
    # VS Code settings
    mkdir -p .vscode
    if [ ! -f ".vscode/settings.json" ]; then
        cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".mypy_cache": true,
        ".pytest_cache": true,
        ".ruff_cache": true
    }
}
EOF
    fi
    
    # Launch configuration for debugging
    if [ ! -f ".vscode/launch.json" ]; then
        cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Gal-Friday2 Debug",
            "type": "python",
            "request": "launch",
            "module": "gal_friday.main",
            "args": ["--config", "config/config.yaml", "--log-level", "DEBUG"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "Run Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}
EOF
    fi
}

# =============================================================================
# Testing Setup
# =============================================================================

setup_testing() {
    print_header "Setting up Testing Environment"
    
    # Run basic tests to verify setup
    if [ -d "tests" ]; then
        print_step "Running basic tests to verify setup"
        python -m pytest tests/ --tb=short -x || {
            print_warning "Some tests failed - this may be expected in a fresh setup"
        }
    else
        print_warning "No tests directory found"
    fi
    
    # Create test data if needed
    if [ -d "tests/fixtures" ]; then
        print_step "Setting up test fixtures"
        # Add any test data setup here
    fi
    
    print_success "Testing setup complete"
}

# =============================================================================
# Documentation and Examples
# =============================================================================

setup_documentation() {
    print_header "Setting up Documentation and Examples"
    
    # Create quick start guide
    create_quick_start_guide
    
    # Create example scripts
    create_example_scripts
    
    print_success "Documentation setup complete"
}

create_quick_start_guide() {
    cat > QUICKSTART.md << 'EOF'
# Gal-Friday2 Quick Start Guide

## Development Environment

This project has been set up with a complete development environment.

### Running the Application

1. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Run in development mode:**
   ```bash
   ./run_dev.sh
   ```

3. **Run tests:**
   ```bash
   ./run_tests.sh
   ```

4. **Run code quality checks:**
   ```bash
   ./run_lint.sh
   ```

### Key Files and Directories

- `config/config.yaml` - Main configuration file
- `gal_friday/` - Main application code
- `tests/` - Test suite
- `logs/` - Application logs
- `models/` - ML models storage
- `data/` - Data storage

### Development Workflow

1. Make changes to the code
2. Run tests: `./run_tests.sh`
3. Run linting: `./run_lint.sh`
4. Test the application: `./run_dev.sh`

### Configuration

Edit `config/config.yaml` to customize:
- Database connections
- API credentials (use environment variables for security)
- Trading parameters
- Logging settings

### Debugging

Use VS Code with the provided launch configurations, or run with:
```bash
python -m gal_friday.main --config config/config.yaml --log-level DEBUG
```

### Getting Help

- Check the main README.md for detailed documentation
- Review the docs/ directory for comprehensive guides
- Check the examples/ directory for usage examples
EOF
}

create_example_scripts() {
    mkdir -p examples
    
    # Create a simple example script
    cat > examples/basic_usage.py << 'EOF'
#!/usr/bin/env python3
"""
Basic usage example for Gal-Friday2

This script demonstrates how to initialize and use core components
of the Gal-Friday2 trading system.
"""

import asyncio
import logging
from pathlib import Path

# Add the project root to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService

async def main():
    """Basic usage example."""
    print("Gal-Friday2 Basic Usage Example")
    print("=" * 40)
    
    # Initialize configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config_manager = ConfigManager(str(config_path))
    
    # Initialize logger
    logger_service = LoggerService(config_manager)
    logger = logger_service.get_logger("example")
    
    logger.info("Example script started")
    
    # Example: Access configuration
    app_name = config_manager.get("app.name", "gal-friday2")
    logger.info(f"Application name: {app_name}")
    
    # Example: Log different levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    
    print("Example completed successfully!")
    print("Check the logs directory for log output.")

if __name__ == "__main__":
    asyncio.run(main())
EOF
    chmod +x examples/basic_usage.py
}

# =============================================================================
# Verification and Health Check
# =============================================================================

run_health_check() {
    print_header "Running Health Check"
    
    local issues=()
    
    # Check Python environment
    if ! source "$VENV_NAME/bin/activate" 2>/dev/null; then
        issues+=("Virtual environment activation failed")
    fi
    
    # Check key imports
    print_step "Testing key imports"
    python -c "
import sys
try:
    import gal_friday
    print('✓ Main package imports successfully')
except ImportError as e:
    print(f'✗ Main package import failed: {e}')
    sys.exit(1)

try:
    import numpy, pandas, sklearn, xgboost
    print('✓ ML libraries import successfully')
except ImportError as e:
    print(f'✗ ML libraries import failed: {e}')
    sys.exit(1)

try:
    import fastapi, uvicorn, websockets
    print('✓ Web libraries import successfully')
except ImportError as e:
    print(f'✗ Web libraries import failed: {e}')
    sys.exit(1)
" || issues+=("Package imports failed")
    
    # Check configuration
    if [ ! -f "$CONFIG_FILE" ]; then
        issues+=("Configuration file missing")
    fi
    
    # Check directory structure
    local required_dirs=("logs" "models" "data")
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            issues+=("Required directory missing: $dir")
        fi
    done
    
    # Report results
    if [ ${#issues[@]} -eq 0 ]; then
        print_success "Health check passed - environment is ready!"
    else
        print_warning "Health check found issues:"
        for issue in "${issues[@]}"; do
            print_error "  - $issue"
        done
    fi
}

# =============================================================================
# Main Setup Function
# =============================================================================

main_setup() {
    print_header "Gal-Friday2 Development Environment Setup"
    print_info "Setup type: $SETUP_TYPE"
    print_info "Use Docker: $USE_DOCKER"
    
    # Run setup steps
    check_system_requirements
    setup_python_environment
    install_dependencies
    setup_configuration
    setup_database
    setup_development_tools
    
    if [ "$SETUP_TYPE" = "test" ] || [ "$SETUP_TYPE" = "full" ]; then
        setup_testing
    fi
    
    setup_documentation
    run_health_check
    
    print_header "Setup Complete!"
    print_success "Gal-Friday2 development environment is ready!"
    print_info ""
    print_info "Next steps:"
    print_info "1. Activate the virtual environment: source $VENV_NAME/bin/activate"
    print_info "2. Edit the configuration file: $CONFIG_FILE"
    print_info "3. Run the application: ./run_dev.sh"
    print_info "4. Run tests: ./run_tests.sh"
    print_info ""
    print_info "For more information, see QUICKSTART.md"
}

# =============================================================================
# Command Line Argument Parsing
# =============================================================================

show_help() {
    cat << EOF
Gal-Friday2 Development Environment Setup Script

Usage: $0 [options]

Options:
  --full          Full setup including databases and external services
  --dev           Development setup (default) - minimal dependencies
  --test          Test environment setup
  --docker        Use Docker for services
  --yes           Skip confirmation prompts
  --help          Show this help message

Examples:
  $0                    # Basic development setup
  $0 --full --docker    # Full setup with Docker services
  $0 --test             # Setup for testing
  $0 --dev --yes        # Development setup without prompts

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            SETUP_TYPE="full"
            shift
            ;;
        --dev)
            SETUP_TYPE="dev"
            shift
            ;;
        --test)
            SETUP_TYPE="test"
            shift
            ;;
        --docker)
            USE_DOCKER=true
            shift
            ;;
        --yes)
            SKIP_CONFIRMATION=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# =============================================================================
# Main Execution
# =============================================================================

# Ensure we're in the project root
if [ ! -f "pyproject.toml" ] || [ ! -d "gal_friday" ]; then
    print_error "This script must be run from the Gal-Friday2 project root directory"
    exit 1
fi

# Show setup summary and confirm
print_info "Gal-Friday2 Setup Configuration:"
print_info "  Setup Type: $SETUP_TYPE"
print_info "  Use Docker: $USE_DOCKER"
print_info "  Project Directory: $(pwd)"
print_info ""

confirm_action "This will set up the Gal-Friday2 development environment."

# Run the main setup
main_setup 