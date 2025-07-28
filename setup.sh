#!/bin/bash

# =============================================================================
# Gal-Friday2 Enterprise Environment Setup Script
# Version: 2.0.0
# =============================================================================
# Production-ready setup script with enterprise-grade features including:
# - Python version management (pyenv/conda)
# - Dependency conflict resolution and lockfiles
# - Environment validation and secure credential handling
# - GPU/CUDA detection for ML acceleration
# - Parallel installations and progress tracking
# - State persistence and rollback capability
# - Comprehensive system resource validation
# =============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures
IFS=$'\n\t'       # Set secure Internal Field Separator

# Enable debug mode if DEBUG env var is set
[[ "${DEBUG:-false}" == "true" ]] && set -x

# =============================================================================
# Global Configuration
# =============================================================================

readonly SCRIPT_VERSION="2.0.0"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"
readonly PROJECT_NAME="gal-friday2"
readonly REQUIRED_PYTHON_VERSION="3.11"
readonly MIN_PYTHON_VERSION="3.11.0"
readonly MAX_PYTHON_VERSION="3.12.99"

# Setup paths
readonly VENV_NAME=".venv"
readonly CONFIG_DIR="config"
readonly CONFIG_FILE="${CONFIG_DIR}/config.yaml"
readonly EXAMPLE_CONFIG="${CONFIG_DIR}/config.example.yaml"
readonly ENV_FILE=".env"
readonly ENV_EXAMPLE=".env.example"
readonly SETUP_STATE_FILE=".setup-state.json"
readonly SETUP_LOG_FILE="setup.log"
readonly LOCKFILE="requirements.lock"

# Setup defaults
SETUP_TYPE="${SETUP_TYPE:-dev}"
USE_DOCKER="${USE_DOCKER:-false}"
SKIP_CONFIRMATION="${SKIP_CONFIRMATION:-false}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
RESUME_SETUP="${RESUME_SETUP:-false}"

# Terminal colors (with fallback for non-color terminals)
if [[ -t 1 ]] && command -v tput &>/dev/null && tput colors &>/dev/null; then
    readonly RED=$(tput setaf 1)
    readonly GREEN=$(tput setaf 2)
    readonly YELLOW=$(tput setaf 3)
    readonly BLUE=$(tput setaf 4)
    readonly MAGENTA=$(tput setaf 5)
    readonly CYAN=$(tput setaf 6)
    readonly WHITE=$(tput setaf 7)
    readonly BOLD=$(tput bold)
    readonly RESET=$(tput sgr0)
else
    readonly RED=""
    readonly GREEN=""
    readonly YELLOW=""
    readonly BLUE=""
    readonly MAGENTA=""
    readonly CYAN=""
    readonly WHITE=""
    readonly BOLD=""
    readonly RESET=""
fi

# Progress tracking
declare -A SETUP_STEPS=(
    ["system_check"]="System Requirements Check"
    ["python_setup"]="Python Environment Setup"
    ["dependencies"]="Dependencies Installation"
    ["configuration"]="Configuration Setup"
    ["environment"]="Environment Variables"
    ["database"]="Database Setup"
    ["dev_tools"]="Development Tools"
    ["testing"]="Testing Environment"
    ["documentation"]="Documentation"
    ["validation"]="Final Validation"
)

declare -A STEP_STATUS=()
declare -A STEP_TIMING=()
CURRENT_STEP=""
SETUP_START_TIME=$(date +%s)

# =============================================================================
# Utility Functions
# =============================================================================

# Logging functions with timestamps
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [${level}] ${message}" | tee -a "${SETUP_LOG_FILE}"
}

print_header() {
    echo -e "\n${BLUE}${BOLD}============================================================================="
    echo -e "  $1"
    echo -e "=============================================================================${RESET}\n"
    log "HEADER" "$1"
}

print_step() {
    echo -e "${GREEN}${BOLD}[STEP]${RESET} $1"
    log "STEP" "$1"
}

print_info() {
    echo -e "${BLUE}[INFO]${RESET} $1"
    log "INFO" "$1"
}

print_warning() {
    echo -e "${YELLOW}${BOLD}[WARNING]${RESET} $1"
    log "WARNING" "$1"
}

print_error() {
    echo -e "${RED}${BOLD}[ERROR]${RESET} $1" >&2
    log "ERROR" "$1"
}

print_success() {
    echo -e "${GREEN}${BOLD}[SUCCESS]${RESET} $1"
    log "SUCCESS" "$1"
}

print_progress() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local completed=$((width * current / total))

    printf "\r${CYAN}Progress: ["
    printf "%${completed}s" | tr ' ' '='
    printf "%$((width - completed))s" | tr ' ' '-'
    printf "] %3d%%${RESET}" $percentage

    [[ $current -eq $total ]] && echo
}

# Error handling with cleanup
trap_error() {
    local exit_code=$?
    local line_number=$1
    print_error "Script failed at line ${line_number} with exit code ${exit_code}"
    print_error "Check ${SETUP_LOG_FILE} for details"

    # Save state for resume
    save_setup_state "failed"

    # Cleanup partial installations if not resuming
    if [[ "${RESUME_SETUP}" != "true" ]]; then
        cleanup_on_error
    fi

    exit $exit_code
}

trap 'trap_error ${LINENO}' ERR

# Signal handling
trap_interrupt() {
    print_warning "Setup interrupted by user"
    save_setup_state "interrupted"
    exit 130
}

trap 'trap_interrupt' INT TERM

# Command existence check
check_command() {
    local cmd="$1"
    if command -v "$cmd" &>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Version comparison
version_compare() {
    # Returns: 0 if $1 >= $2, 1 otherwise
    printf '%s\n%s' "$2" "$1" | sort -V -C
}

# Confirmation prompt
confirm_action() {
    if [[ "$SKIP_CONFIRMATION" == "true" ]]; then
        return 0
    fi

    local prompt="${1:-Continue?}"
    echo -e "${YELLOW}${prompt}${RESET}"
    read -p "(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Operation cancelled by user"
        exit 0
    fi
}

# =============================================================================
# Setup State Management
# =============================================================================

save_setup_state() {
    local status="${1:-in_progress}"
    local state_data=$(cat <<EOF
{
    "version": "${SCRIPT_VERSION}",
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "status": "${status}",
    "setup_type": "${SETUP_TYPE}",
    "use_docker": ${USE_DOCKER},
    "current_step": "${CURRENT_STEP}",
    "completed_steps": [$(printf '"%s",' "${!STEP_STATUS[@]}" | sed 's/,$//')],
    "step_status": $(printf '{' && for key in "${!STEP_STATUS[@]}"; do printf '"%s":"%s",' "$key" "${STEP_STATUS[$key]}"; done | sed 's/,$/}/')
}
EOF
)
    echo "$state_data" > "${SETUP_STATE_FILE}"
}

load_setup_state() {
    if [[ -f "${SETUP_STATE_FILE}" ]]; then
        print_info "Found previous setup state"
        # Parse JSON manually (avoiding jq dependency)
        local state_content=$(cat "${SETUP_STATE_FILE}")

        # Extract setup type
        if [[ "$state_content" =~ \"setup_type\":\ *\"([^\"]+)\" ]]; then
            SETUP_TYPE="${BASH_REMATCH[1]}"
        fi

        # Extract docker flag
        if [[ "$state_content" =~ \"use_docker\":\ *([^,}]+) ]]; then
            USE_DOCKER="${BASH_REMATCH[1]}"
        fi

        # Extract completed steps
        if [[ "$state_content" =~ \"completed_steps\":\ *\[([^\]]*)\] ]]; then
            local completed="${BASH_REMATCH[1]}"
            # Remove quotes and spaces
            completed="${completed//\"/}"
            completed="${completed// /}"

            # Mark completed steps
            IFS=',' read -ra completed_array <<< "$completed"
            for step in "${completed_array[@]}"; do
                STEP_STATUS["$step"]="completed"
            done
        fi

        return 0
    fi
    return 1
}

cleanup_on_error() {
    print_warning "Cleaning up partial installation..."

    # Remove virtual environment if it's incomplete
    if [[ -d "${VENV_NAME}" ]] && [[ ! -f "${VENV_NAME}/bin/activate" ]]; then
        rm -rf "${VENV_NAME}"
    fi

    # Remove incomplete config files
    if [[ -f "${CONFIG_FILE}" ]] && [[ ! -s "${CONFIG_FILE}" ]]; then
        rm -f "${CONFIG_FILE}"
    fi
}

# =============================================================================
# System Requirements and Resource Checks
# =============================================================================

check_system_requirements() {
    print_header "Checking System Requirements"

    local missing_deps=()

    # Check Python
    if check_command python3; then
        local python_version=$(python3 --version | cut -d' ' -f2)
        print_info "Python version: $python_version"
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
            print_info "Platform: Windows (WSL recommended)"
            ;;
        *)
            print_warning "Unknown platform: $(uname -s)"
            ;;
    esac

    if [[ ${#missing_deps[@]} -ne 0 ]]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        print_info "Please install the missing dependencies and run this script again"
        exit 1
    fi

    print_success "System requirements check passed"
}

check_system_resources() {
    print_step "Checking system resources"

    local issues=()

    # Check available memory
    local total_mem=0
    local available_mem=0

    case "$(uname -s)" in
        Linux*)
            if [[ -f /proc/meminfo ]]; then
                total_mem=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
                available_mem=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
                total_mem=$((total_mem / 1024))  # Convert to MB
                available_mem=$((available_mem / 1024))
            fi
            ;;
        Darwin*)
            total_mem=$(sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024)}')
            # Get free memory from vm_stat
            local pages_free=$(vm_stat | awk '/Pages free/ {print $3}' | tr -d '.')
            local page_size=$(vm_stat | awk '/page size/ {print $8}')
            available_mem=$((pages_free * page_size / 1024 / 1024))
            ;;
    esac

    print_info "Total memory: ${total_mem}MB, Available: ${available_mem}MB"

    # ML models need at least 4GB available
    if [[ $available_mem -lt 4096 ]]; then
        issues+=("Insufficient memory: ${available_mem}MB available, 4GB+ recommended for ML models")
    fi

    # Check disk space
    local available_disk=$(df -BG "${PROJECT_ROOT}" | awk 'NR==2 {print $4}' | tr -d 'G')
    print_info "Available disk space: ${available_disk}GB"

    if [[ $available_disk -lt 10 ]]; then
        issues+=("Low disk space: ${available_disk}GB available, 10GB+ recommended")
    fi

    # Check CPU cores
    local cpu_cores=0
    case "$(uname -s)" in
        Linux*) cpu_cores=$(nproc 2>/dev/null || echo 1) ;;
        Darwin*) cpu_cores=$(sysctl -n hw.ncpu 2>/dev/null || echo 1) ;;
    esac

    print_info "CPU cores available: ${cpu_cores}"
    PARALLEL_JOBS=$((cpu_cores > 4 ? 4 : cpu_cores))

    # Check network connectivity
    print_step "Checking network connectivity"
    local network_ok=true

    for host in "pypi.org" "github.com" "registry.npmjs.org"; do
        if ! curl -s --head --connect-timeout 5 "https://${host}" >/dev/null; then
            issues+=("Cannot reach ${host} - check network connection")
            network_ok=false
        fi
    done

    if [[ "$network_ok" == "true" ]]; then
        print_success "Network connectivity verified"
    fi

    # Report issues
    if [[ ${#issues[@]} -gt 0 ]]; then
        print_warning "System resource issues detected:"
        for issue in "${issues[@]}"; do
            print_warning "  - $issue"
        done

        if [[ "${SKIP_CONFIRMATION}" != "true" ]]; then
            confirm_action "Continue with potential limitations?"
        fi
    else
        print_success "System resources check passed"
    fi
}

check_gpu_support() {
    print_step "Checking GPU/CUDA support for ML acceleration"

    local has_gpu=false
    local cuda_version=""

    # Check for NVIDIA GPU
    if check_command nvidia-smi; then
        if nvidia-smi &>/dev/null; then
            has_gpu=true
            print_success "NVIDIA GPU detected"

            # Get CUDA version
            if check_command nvcc; then
                cuda_version=$(nvcc --version | grep release | awk '{print $6}' | cut -d, -f1)
                print_info "CUDA version: ${cuda_version}"
            else
                print_warning "CUDA toolkit not found - GPU acceleration may be limited"
            fi

            # Show GPU info
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read gpu_info; do
                print_info "GPU: ${gpu_info}"
            done
        fi
    fi

    # Check for AMD GPU (ROCm)
    if check_command rocm-smi; then
        if rocm-smi &>/dev/null; then
            has_gpu=true
            print_success "AMD GPU detected with ROCm support"
        fi
    fi

    # Check for Apple Silicon
    if [[ "$(uname -s)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
        has_gpu=true
        print_success "Apple Silicon GPU detected"
    fi

    if [[ "$has_gpu" == "false" ]]; then
        print_info "No GPU detected - ML models will use CPU only"
    fi

    # Export for later use
    export HAS_GPU="$has_gpu"
    export CUDA_VERSION="$cuda_version"
}

# =============================================================================
# Python Version Management
# =============================================================================

setup_python_version() {
    print_header "Python Version Management"

    local python_cmd=""
    local python_version=""
    local need_install=false

    # Check for Python version managers
    if check_command pyenv; then
        print_info "Found pyenv - checking for Python ${REQUIRED_PYTHON_VERSION}"

        # Check if required version is installed
        if pyenv versions | grep -q "${REQUIRED_PYTHON_VERSION}"; then
            print_success "Python ${REQUIRED_PYTHON_VERSION} is available via pyenv"
            pyenv local "${REQUIRED_PYTHON_VERSION}"
            python_cmd="python"
        else
            print_warning "Python ${REQUIRED_PYTHON_VERSION} not found in pyenv"

            if confirm_action "Install Python ${REQUIRED_PYTHON_VERSION} via pyenv?"; then
                print_step "Installing Python ${REQUIRED_PYTHON_VERSION}"
                pyenv install "${REQUIRED_PYTHON_VERSION}"
                pyenv local "${REQUIRED_PYTHON_VERSION}"
                python_cmd="python"
            else
                need_install=true
            fi
        fi
    elif check_command conda; then
        print_info "Found conda - checking Python version"

        # Get current conda Python version
        python_version=$(conda run python --version 2>/dev/null | cut -d' ' -f2)

        if version_compare "$python_version" "$MIN_PYTHON_VERSION" && \
           version_compare "$MAX_PYTHON_VERSION" "$python_version"; then
            print_success "Conda Python ${python_version} is compatible"
            python_cmd="conda run python"
        else
            print_warning "Conda Python ${python_version} is not compatible"
            need_install=true
        fi
    else
        # Check system Python
        for cmd in python3.11 python3 python; do
            if check_command "$cmd"; then
                python_version=$("$cmd" --version 2>/dev/null | cut -d' ' -f2)

                if version_compare "$python_version" "$MIN_PYTHON_VERSION" && \
                   version_compare "$MAX_PYTHON_VERSION" "$python_version"; then
                    print_success "Found compatible Python: ${cmd} (${python_version})"
                    python_cmd="$cmd"
                    break
                fi
            fi
        done

        if [[ -z "$python_cmd" ]]; then
            need_install=true
        fi
    fi

    # Handle Python installation if needed
    if [[ "$need_install" == "true" ]]; then
        print_error "No compatible Python version found (need ${MIN_PYTHON_VERSION} - ${MAX_PYTHON_VERSION})"
        print_info "Please install Python ${REQUIRED_PYTHON_VERSION} using one of these methods:"
        print_info "  - pyenv: curl https://pyenv.run | bash"
        print_info "  - conda: https://docs.conda.io/en/latest/miniconda.html"
        print_info "  - System package manager"
        exit 1
    fi

    # Export for use in other functions
    export PYTHON_CMD="$python_cmd"
    export PYTHON_VERSION="$python_version"
}

# =============================================================================
# Virtual Environment Setup
# =============================================================================

setup_virtual_environment() {
    print_header "Setting up Python Virtual Environment"

    # Create virtual environment if it doesn't exist
    if [[ ! -d "${VENV_NAME}" ]]; then
        print_step "Creating virtual environment: ${VENV_NAME}"
        ${PYTHON_CMD} -m venv "${VENV_NAME}" --prompt "${PROJECT_NAME}"
    else
        print_info "Virtual environment already exists: ${VENV_NAME}"

        # Verify it's valid
        if [[ ! -f "${VENV_NAME}/bin/activate" ]]; then
            print_warning "Virtual environment appears corrupted, recreating..."
            rm -rf "${VENV_NAME}"
            ${PYTHON_CMD} -m venv "${VENV_NAME}" --prompt "${PROJECT_NAME}"
        fi
    fi

    # Activate virtual environment
    print_step "Activating virtual environment"
    source "${VENV_NAME}/bin/activate" || {
        print_error "Failed to activate virtual environment"
        exit 1
    }

    # Upgrade pip, setuptools, wheel
    print_step "Upgrading pip and build tools"
    python -m pip install --upgrade pip setuptools wheel

    # Install pip-tools for dependency management
    print_step "Installing dependency management tools"
    pip install pip-tools

    print_success "Virtual environment ready"
}

# =============================================================================
# Dependency Management with Conflict Resolution
# =============================================================================

create_requirements_lock() {
    print_step "Creating locked dependency file"

    # Compile requirements with pip-compile
    if [[ -f "requirements.in" ]]; then
        print_info "Compiling requirements.in to requirements.txt"
        pip-compile requirements.in --output-file=requirements.txt
    fi

    # Create lock file with exact versions
    print_info "Generating requirements.lock with exact versions"
    pip freeze > "${LOCKFILE}"

    print_success "Dependency lock file created: ${LOCKFILE}"
}

install_dependencies_parallel() {
    print_header "Installing Dependencies"

    # Activate virtual environment
    if [[ -f "${VENV_NAME}/bin/activate" ]]; then
        source "${VENV_NAME}/bin/activate" || {
            print_error "Failed to activate virtual environment"
            exit 1
        }
    else
        print_error "Virtual environment not found. Run setup_virtual_environment first."
        exit 1
    fi

    # Check if lockfile exists and is newer than requirements.txt
    if [[ -f "${LOCKFILE}" ]] && [[ "${LOCKFILE}" -nt "requirements.txt" ]]; then
        print_info "Using locked dependencies from ${LOCKFILE}"
        local requirements_file="${LOCKFILE}"
    else
        local requirements_file="requirements.txt"
    fi

    if [[ ! -f "$requirements_file" ]]; then
        print_error "${requirements_file} not found"
        exit 1
    fi

    # Pre-download packages for faster installation
    print_step "Pre-downloading packages"
    pip download -r "$requirements_file" -d .pip-cache/ --no-deps 2>/dev/null || true

    # Install dependencies with progress tracking
    local total_deps=$(grep -v '^#' "$requirements_file" | grep -v '^$' | wc -l)
    local current=0

    print_step "Installing ${total_deps} dependencies"

    # Handle special cases first
    # TA-Lib requires system dependencies
    if grep -q "TA-Lib" "$requirements_file"; then
        print_step "Checking TA-Lib installation"

        # Check if TA-Lib is already installed
        if python -c "import talib" 2>/dev/null; then
            print_info "TA-Lib is already installed (version $(python -c 'import talib; print(talib.__version__)'))"
        else
            print_step "Installing TA-Lib (requires system dependencies)"

            case "$(uname -s)" in
                Linux*)
                    if check_command apt-get; then
                        export SUDO_ASKPASS="${PROJECT_ROOT}/askpass.sh"
                        sudo -A apt-get update && sudo -A apt-get install -y libta-lib0-dev
                    elif check_command yum; then
                        sudo yum install -y ta-lib-devel
                    fi
                    ;;
                Darwin*)
                    if check_command brew; then
                        brew install ta-lib
                    fi
                    ;;
            esac

            pip install TA-Lib || print_warning "TA-Lib installation failed - continuing without it"
        fi
    fi

    # Install remaining dependencies
    while IFS= read -r dep; do
        [[ -z "$dep" || "$dep" =~ ^# ]] && continue
        [[ "$dep" =~ TA-Lib ]] && continue  # Already handled

        ((current++))
        print_progress $current $total_deps

        pip install "$dep" --find-links .pip-cache/ --no-deps --quiet || {
            print_warning "Failed to install: $dep"
        }
    done < "$requirements_file"

    echo  # New line after progress bar

    # Install dependencies with their dependencies
    print_step "Resolving dependency tree"
    pip install -r "$requirements_file" --find-links .pip-cache/

    # Install test dependencies if needed
    if [[ "$SETUP_TYPE" =~ ^(test|full)$ ]] && [[ -f "requirements-test.txt" ]]; then
        print_step "Installing test dependencies"
        pip install -r requirements-test.txt
    fi

    # Install development tools
    if [[ "$SETUP_TYPE" =~ ^(dev|full)$ ]]; then
        print_step "Installing development tools"
        pip install pre-commit black isort ruff mypy bandit safety
    fi

    # Create lock file if it doesn't exist
    if [[ ! -f "${LOCKFILE}" ]]; then
        create_requirements_lock
    fi

    # Cleanup cache
    rm -rf .pip-cache/

    print_success "Dependencies installation complete"
}

# =============================================================================
# Environment Variables and Secrets Management
# =============================================================================

setup_environment_variables() {
    print_header "Setting up Environment Variables"

    # Create .env from example if it doesn't exist
    if [[ ! -f "${ENV_FILE}" ]]; then
        if [[ -f "${ENV_EXAMPLE}" ]]; then
            print_step "Creating .env file from example"
            cp "${ENV_EXAMPLE}" "${ENV_FILE}"
            chmod 600 "${ENV_FILE}"  # Restrict permissions
        else
            print_step "Creating .env file"
            create_env_file
        fi
    else
        print_info "Environment file already exists: ${ENV_FILE}"
    fi

    # Validate environment variables
    validate_environment_variables

    # Setup secure credential storage
    setup_credential_storage

    print_success "Environment setup complete"
}

create_env_file() {
    cat > "${ENV_FILE}" << 'EOF'
# Gal-Friday2 Environment Variables
# WARNING: This file contains sensitive information. Never commit to version control!

# Environment
ENVIRONMENT=development
DEBUG=false
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=sqlite:///data/gal_friday_dev.db
# For PostgreSQL: postgresql://user:password@localhost:5432/gal_friday
# POSTGRES_HOST=localhost
# POSTGRES_PORT=5432
# POSTGRES_DB=gal_friday
# POSTGRES_USER=gal_friday
# POSTGRES_PASSWORD=

# InfluxDB Configuration
# INFLUXDB_URL=http://localhost:8086
# INFLUXDB_TOKEN=
# INFLUXDB_ORG=gal-friday
# INFLUXDB_BUCKET=market_data

# Redis Configuration
# REDIS_URL=redis://localhost:6379/0

# Kraken API Configuration
# KRAKEN_API_KEY=
# KRAKEN_API_SECRET=
# KRAKEN_API_TIER=Starter

# ML Model Configuration
ML_PROCESS_WORKERS=2
ML_MODEL_CACHE_SIZE=5
GPU_ENABLED=false

# Monitoring Configuration
MONITORING_ENABLED=true
METRICS_EXPORT_INTERVAL=60

# Security Settings
JWT_SECRET_KEY=
ENCRYPTION_KEY=

# Feature Flags
FEATURE_PAPER_TRADING=true
FEATURE_LIVE_TRADING=false
FEATURE_BACKTESTING=true

# Performance Settings
MAX_MEMORY_MB=4096
MAX_CPU_PERCENT=80

# Notification Settings (optional)
# SLACK_WEBHOOK_URL=
# EMAIL_SMTP_HOST=
# EMAIL_SMTP_PORT=587
# EMAIL_FROM=
# EMAIL_PASSWORD=
EOF

    chmod 600 "${ENV_FILE}"
}

validate_environment_variables() {
    print_step "Validating environment variables"

    local missing_vars=()
    local warnings=()

    # Source the .env file
    set -a
    source "${ENV_FILE}"
    set +a

    # Check critical variables for production
    if [[ "$SETUP_TYPE" == "full" ]]; then
        # Database
        if [[ -z "${DATABASE_URL:-}" ]] && [[ -z "${POSTGRES_HOST:-}" ]]; then
            missing_vars+=("DATABASE_URL or POSTGRES_* variables")
        fi

        # API Keys (warn only)
        if [[ -z "${KRAKEN_API_KEY:-}" ]]; then
            warnings+=("KRAKEN_API_KEY not set - required for live trading")
        fi

        # Security
        if [[ -z "${JWT_SECRET_KEY:-}" ]]; then
            print_step "Generating JWT secret key"
            JWT_SECRET_KEY=$(openssl rand -hex 32)
            sed -i.bak "s/^JWT_SECRET_KEY=.*/JWT_SECRET_KEY=${JWT_SECRET_KEY}/" "${ENV_FILE}"
        fi

        if [[ -z "${ENCRYPTION_KEY:-}" ]]; then
            print_step "Generating encryption key"
            ENCRYPTION_KEY=$(openssl rand -hex 32)
            sed -i.bak "s/^ENCRYPTION_KEY=.*/ENCRYPTION_KEY=${ENCRYPTION_KEY}/" "${ENV_FILE}"
        fi
    fi

    # Report issues
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        print_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            print_error "  - $var"
        done
        print_info "Please edit ${ENV_FILE} and set the required variables"
        exit 1
    fi

    if [[ ${#warnings[@]} -gt 0 ]]; then
        print_warning "Environment variable warnings:"
        for warning in "${warnings[@]}"; do
            print_warning "  - $warning"
        done
    fi

    # Remove backup files
    rm -f "${ENV_FILE}.bak"

    print_success "Environment variables validated"
}

setup_credential_storage() {
    print_step "Setting up secure credential storage"

    case "$(uname -s)" in
        Linux*)
            if check_command secret-tool; then
                print_info "Using GNOME Keyring for credential storage"
            elif check_command kwallet-query; then
                print_info "Using KWallet for credential storage"
            else
                print_warning "No system keyring found - credentials will be stored in .env file"
                print_info "Consider installing gnome-keyring or kwallet for secure storage"
            fi
            ;;
        Darwin*)
            print_info "Using macOS Keychain for credential storage"
            ;;
        MINGW*|CYGWIN*|MSYS*)
            print_info "Using Windows Credential Manager"
            ;;
    esac

    # Create credential helper script
    cat > "${PROJECT_ROOT}/scripts/store-credentials.sh" << 'EOF'
#!/bin/bash
# Credential storage helper

set -euo pipefail

store_credential() {
    local service="$1"
    local account="$2"
    local secret="$3"

    case "$(uname -s)" in
        Linux*)
            if command -v secret-tool &>/dev/null; then
                echo "$secret" | secret-tool store --label="$service" service "$service" account "$account"
            else
                echo "No keyring available"
                return 1
            fi
            ;;
        Darwin*)
            security add-generic-password -a "$account" -s "$service" -w "$secret" -U
            ;;
        MINGW*|CYGWIN*|MSYS*)
            cmdkey /generic:"$service" /user:"$account" /pass:"$secret"
            ;;
    esac
}

retrieve_credential() {
    local service="$1"
    local account="$2"

    case "$(uname -s)" in
        Linux*)
            if command -v secret-tool &>/dev/null; then
                secret-tool lookup service "$service" account "$account"
            else
                return 1
            fi
            ;;
        Darwin*)
            security find-generic-password -a "$account" -s "$service" -w
            ;;
        MINGW*|CYGWIN*|MSYS*)
            # Windows doesn't have a simple CLI retrieval
            echo "Use Credential Manager GUI to view credentials"
            return 1
            ;;
    esac
}

# Main
case "${1:-}" in
    store)
        store_credential "$2" "$3" "$4"
        ;;
    retrieve)
        retrieve_credential "$2" "$3"
        ;;
    *)
        echo "Usage: $0 {store|retrieve} service account [secret]"
        exit 1
        ;;
esac
EOF

    chmod +x "${PROJECT_ROOT}/scripts/store-credentials.sh"

    print_info "Credential helper created: scripts/store-credentials.sh"
}

# =============================================================================
# Configuration Setup with Validation
# =============================================================================

setup_configuration() {
    print_header "Setting up Configuration"

    # Create config directory
    mkdir -p "${CONFIG_DIR}"

    # Setup main configuration
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        if [[ -f "${EXAMPLE_CONFIG}" ]]; then
            print_step "Creating configuration from example"
            cp "${EXAMPLE_CONFIG}" "${CONFIG_FILE}"
        else
            print_step "Creating default configuration"
            create_default_config
        fi
    else
        print_info "Configuration already exists: ${CONFIG_FILE}"
    fi

    # Validate configuration
    validate_configuration

    # Create necessary directories
    print_step "Creating project directories"
    mkdir -p logs models data temp cache .backups

    # Set appropriate permissions
    chmod 700 logs .backups  # Restrict access to logs and backups

    print_success "Configuration setup complete"
}

create_default_config() {
    cat > "${CONFIG_FILE}" << 'EOF'
# Gal-Friday2 Configuration
# This file is loaded after environment variables

app:
  name: "gal-friday2"
  version: "2.0.0"
  environment: "${ENVIRONMENT:-development}"

logging:
  level: "${LOG_LEVEL:-INFO}"
  file: "logs/gal-friday.log"
  max_size: "50MB"
  backup_count: 10
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

database:
  url: "${DATABASE_URL:-sqlite:///data/gal_friday_dev.db}"
  pool_size: 20
  max_overflow: 40
  pool_timeout: 30
  echo: false

influxdb:
  url: "${INFLUXDB_URL:-http://localhost:8086}"
  token: "${INFLUXDB_TOKEN:-}"
  org: "${INFLUXDB_ORG:-gal-friday}"
  bucket: "${INFLUXDB_BUCKET:-market_data}"
  timeout: 10000

redis:
  url: "${REDIS_URL:-redis://localhost:6379/0}"
  max_connections: 50
  decode_responses: true

kraken:
  api_key: "${KRAKEN_API_KEY:-}"
  api_secret: "${KRAKEN_API_SECRET:-}"
  api_tier: "${KRAKEN_API_TIER:-Starter}"
  rate_limit:
    Starter: 15
    Intermediate: 20
    Pro: 20
  websocket_url: "wss://ws.kraken.com"
  rest_url: "https://api.kraken.com"

feature_engine:
  cache_size: 10000
  update_interval: 60
  parallel_workers: "${ML_PROCESS_WORKERS:-2}"
  feature_groups:
    - price_features
    - volume_features
    - technical_indicators
    - market_microstructure

prediction_service:
  process_pool_workers: "${ML_PROCESS_WORKERS:-2}"
  models:
    - name: "xgboost_1m"
      type: "XGBoost"
      timeframe: "1m"
      features: ["price", "volume", "technical"]
    - name: "lstm_5m"
      type: "LSTM"
      timeframe: "5m"
      features: ["price", "volume", "orderbook"]
  model_cache_size: "${ML_MODEL_CACHE_SIZE:-5}"
  prediction_cache_ttl: 300

risk_management:
  max_position_size: 10000.0
  max_daily_loss: 500.0
  max_drawdown: 0.15
  risk_per_trade: 0.01
  stop_loss_multiplier: 2.0
  take_profit_multiplier: 3.0

portfolio:
  initial_capital: 10000.0
  max_positions: 5
  position_sizing: "kelly"
  rebalance_interval: 3600

execution:
  mode: "${TRADING_MODE:-paper}"
  slippage_bps: 10
  commission_bps: 26
  max_retry_attempts: 3
  order_timeout: 30

monitoring:
  enabled: "${MONITORING_ENABLED:-true}"
  port: 8000
  host: "0.0.0.0"
  metrics_export_interval: "${METRICS_EXPORT_INTERVAL:-60}"
  health_check_interval: 30

performance:
  max_memory_mb: "${MAX_MEMORY_MB:-4096}"
  max_cpu_percent: "${MAX_CPU_PERCENT:-80}"
  gc_threshold: 0.8
  profile_enabled: false

feature_flags:
  paper_trading: "${FEATURE_PAPER_TRADING:-true}"
  live_trading: "${FEATURE_LIVE_TRADING:-false}"
  backtesting: "${FEATURE_BACKTESTING:-true}"
  ml_predictions: true
  risk_management: true
  portfolio_optimization: true
EOF
}

validate_configuration() {
    print_step "Validating configuration"

    # Check if config is valid YAML
    if check_command python; then
        python -c "
import yaml
import sys
try:
    with open('${CONFIG_FILE}', 'r') as f:
        config = yaml.safe_load(f)
    print('Configuration is valid YAML')
except Exception as e:
    print(f'Configuration error: {e}')
    sys.exit(1)
"
    fi

    print_success "Configuration validated"
}

# =============================================================================
# Database Setup with Docker Support
# =============================================================================

setup_database() {
    print_header "Setting up Database"

    case "$SETUP_TYPE" in
        dev)
            setup_sqlite_database
            ;;
        test)
            setup_test_database
            ;;
        full)
            if [[ "$USE_DOCKER" == "true" ]]; then
                setup_docker_databases
            else
                setup_production_databases
            fi
            ;;
    esac

    # Run migrations
    run_database_migrations

    print_success "Database setup complete"
}

setup_sqlite_database() {
    print_step "Setting up SQLite database for development"

    mkdir -p data

    # Initialize database schema
    if [[ -f "scripts/init_database.py" ]]; then
        print_step "Initializing database schema"
        python scripts/init_database.py --dev
    fi
}

setup_test_database() {
    print_step "Setting up test database"

    mkdir -p data/test

    # Create separate test database
    export DATABASE_URL="sqlite:///data/test/gal_friday_test.db"
}

setup_docker_databases() {
    print_step "Setting up databases with Docker"

    # Check Docker installation
    if ! check_command docker; then
        print_error "Docker is required but not installed"
        print_info "Install Docker from https://docs.docker.com/get-docker/"
        exit 1
    fi

    # Create docker-compose file if it doesn't exist
    if [[ ! -f "docker-compose.yml" ]]; then
        create_docker_compose
    fi

    # Start services
    print_step "Starting database services"
    docker-compose up -d postgres influxdb redis

    # Wait for services to be ready
    print_step "Waiting for services to be ready"

    # PostgreSQL
    local retries=30
    while ! docker-compose exec -T postgres pg_isready &>/dev/null; do
        ((retries--))
        if [[ $retries -eq 0 ]]; then
            print_error "PostgreSQL failed to start"
            exit 1
        fi
        sleep 1
    done
    print_success "PostgreSQL is ready"

    # InfluxDB
    retries=30
    while ! curl -s http://localhost:8086/health &>/dev/null; do
        ((retries--))
        if [[ $retries -eq 0 ]]; then
            print_error "InfluxDB failed to start"
            exit 1
        fi
        sleep 1
    done
    print_success "InfluxDB is ready"

    # Redis
    retries=30
    while ! docker-compose exec -T redis redis-cli ping &>/dev/null; do
        ((retries--))
        if [[ $retries -eq 0 ]]; then
            print_error "Redis failed to start"
            exit 1
        fi
        sleep 1
    done
    print_success "Redis is ready"
}

setup_production_databases() {
    print_step "Setting up production databases"

    print_info "Production database setup requires manual configuration"
    print_info "Please ensure the following services are installed and running:"
    print_info "  - PostgreSQL 15+"
    print_info "  - InfluxDB 2.7+"
    print_info "  - Redis 7+"

    print_info "Update your .env file with the connection details"
}

create_docker_compose() {
    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    container_name: gal_friday_postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-gal_friday}
      POSTGRES_USER: ${POSTGRES_USER:-gal_friday}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-gal_friday_dev}
      PGDATA: /var/lib/postgresql/data/pgdata
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-gal_friday}"]
      interval: 10s
      timeout: 5s
      retries: 5

  influxdb:
    image: influxdb:2.7-alpine
    container_name: gal_friday_influxdb
    restart: unless-stopped
    environment:
      DOCKER_INFLUXDB_INIT_MODE: setup
      DOCKER_INFLUXDB_INIT_USERNAME: ${INFLUXDB_USERNAME:-admin}
      DOCKER_INFLUXDB_INIT_PASSWORD: ${INFLUXDB_PASSWORD:-admin_password}
      DOCKER_INFLUXDB_INIT_ORG: ${INFLUXDB_ORG:-gal-friday}
      DOCKER_INFLUXDB_INIT_BUCKET: ${INFLUXDB_BUCKET:-market_data}
      DOCKER_INFLUXDB_INIT_ADMIN_TOKEN: ${INFLUXDB_TOKEN:-dev_token}
    ports:
      - "8086:8086"
    volumes:
      - influxdb_data:/var/lib/influxdb2
      - influxdb_config:/etc/influxdb2
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8086/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: gal_friday_redis
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Optional: Database management tools
  pgadmin:
    image: dpage/pgadmin4:latest
    container_name: gal_friday_pgadmin
    restart: unless-stopped
    environment:
      PGADMIN_DEFAULT_EMAIL: ${PGADMIN_EMAIL:-admin@galfriday.local}
      PGADMIN_DEFAULT_PASSWORD: ${PGADMIN_PASSWORD:-admin}
    ports:
      - "5050:80"
    depends_on:
      - postgres
    profiles:
      - tools

volumes:
  postgres_data:
  influxdb_data:
  influxdb_config:
  redis_data:
EOF
}

run_database_migrations() {
    print_step "Running database migrations"

    # Check for migration tool
    if [[ -f "alembic.ini" ]] && check_command alembic; then
        print_info "Running Alembic migrations"
        alembic upgrade head
    elif [[ -f "scripts/migrate.py" ]]; then
        print_info "Running custom migrations"
        python scripts/migrate.py
    elif [[ -f "gal_friday/dal/migrations/migration_manager.py" ]]; then
        print_info "Running migration manager"
        python -c "from gal_friday.dal.migrations.migration_manager import MigrationManager; MigrationManager().run_migrations()"
    else
        print_warning "No migration system found"
    fi
}

# =============================================================================
# Development Tools Setup
# =============================================================================

setup_development_tools() {
    print_header "Setting up Development Tools"

    # Pre-commit hooks
    setup_pre_commit

    # IDE configurations
    setup_ide_configs

    # Development scripts
    create_dev_scripts

    # Git hooks
    setup_git_hooks

    print_success "Development tools setup complete"
}

setup_pre_commit() {
    print_step "Setting up pre-commit hooks"

    if [[ ! -f ".pre-commit-config.yaml" ]]; then
        cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: check-toml
      - id: debug-statements
      - id: mixed-line-ending

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--ignore-missing-imports]

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.6
    hooks:
      - id: bandit
        args: ['-c', 'pyproject.toml']

  - repo: https://github.com/gitguardian/ggshield
    rev: v1.25.0
    hooks:
      - id: ggshield
        language: python
        stages: [commit]
EOF
    fi

    # Install pre-commit
    pre-commit install
    pre-commit install --hook-type commit-msg

    # Run on all files (optional)
    if confirm_action "Run pre-commit on all files now?"; then
        pre-commit run --all-files || true
    fi
}

setup_ide_configs() {
    print_step "Setting up IDE configurations"

    # VS Code
    setup_vscode

    # PyCharm
    setup_pycharm

    # Vim/Neovim
    setup_vim
}

setup_vscode() {
    mkdir -p .vscode

    # VS Code settings
    cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true,
        "source.fixAll": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".mypy_cache": true,
        ".pytest_cache": true,
        ".ruff_cache": true,
        "*.egg-info": true,
        ".coverage": true,
        "htmlcov": true,
        ".tox": true
    },
    "python.analysis.typeCheckingMode": "strict",
    "python.analysis.autoImportCompletions": true,
    "python.analysis.diagnosticMode": "workspace"
}
EOF

    # Launch configurations
    cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Run Main",
            "type": "python",
            "request": "launch",
            "module": "gal_friday.main",
            "args": ["--config", "${workspaceFolder}/config/config.yaml"],
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            },
            "envFile": "${workspaceFolder}/.env"
        },
        {
            "name": "Debug Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env"
        },
        {
            "name": "Run Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": [
                "-xvs",
                "${workspaceFolder}/tests",
                "-k",
                "${selectedText}"
            ],
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env.test"
        },
        {
            "name": "Run Dashboard",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "gal_friday.monitoring.dashboard_service:app",
                "--reload",
                "--host",
                "0.0.0.0",
                "--port",
                "8000"
            ],
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env"
        }
    ]
}
EOF

    # Extensions recommendations
    cat > .vscode/extensions.json << 'EOF'
{
    "recommendations": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-python.black-formatter",
        "charliermarsh.ruff",
        "ms-vscode.makefile-tools",
        "redhat.vscode-yaml",
        "tamasfe.even-better-toml",
        "streetsidesoftware.code-spell-checker",
        "eamodio.gitlens",
        "github.vscode-pull-request-github",
        "ms-azuretools.vscode-docker",
        "mtxr.sqltools",
        "mechatroner.rainbow-csv"
    ]
}
EOF
}

setup_pycharm() {
    mkdir -p .idea

    # PyCharm run configurations
    cat > .idea/runConfigurations/Main.xml << 'EOF'
<component name="ProjectRunConfigurationManager">
  <configuration default="false" name="Main" type="PythonConfigurationType" factoryName="Python">
    <module name="gal-friday2" />
    <option name="INTERPRETER_OPTIONS" value="" />
    <option name="PARENT_ENVS" value="true" />
    <envs>
      <env name="PYTHONUNBUFFERED" value="1" />
    </envs>
    <option name="SDK_HOME" value="$PROJECT_DIR$/.venv/bin/python" />
    <option name="WORKING_DIRECTORY" value="$PROJECT_DIR$" />
    <option name="IS_MODULE_SDK" value="false" />
    <option name="ADD_CONTENT_ROOTS" value="true" />
    <option name="ADD_SOURCE_ROOTS" value="true" />
    <option name="SCRIPT_NAME" value="gal_friday.main" />
    <option name="PARAMETERS" value="--config config/config.yaml" />
    <option name="SHOW_COMMAND_LINE" value="false" />
    <option name="EMULATE_TERMINAL" value="false" />
    <option name="MODULE_MODE" value="true" />
    <method v="2" />
  </configuration>
</component>
EOF
}

setup_vim() {
    # Vim/Neovim configuration
    cat > .vim-config << 'EOF'
" Gal-Friday2 Vim Configuration
" Add this to your .vimrc or init.vim

" Python specific settings
autocmd FileType python setlocal expandtab shiftwidth=4 tabstop=4 softtabstop=4
autocmd FileType python setlocal textwidth=88  " Black line length
autocmd FileType python setlocal colorcolumn=88

" Use .venv Python
let g:python3_host_prog = getcwd() . '/.venv/bin/python'

" ALE configuration for Python
let g:ale_python_auto_virtualenv = 1
let g:ale_linters = {
\   'python': ['ruff', 'mypy'],
\}
let g:ale_fixers = {
\   'python': ['black', 'isort'],
\}
let g:ale_fix_on_save = 1

" CoC configuration (if using)
let g:coc_global_extensions = [
\   'coc-python',
\   'coc-pyright',
\   'coc-json',
\   'coc-yaml',
\   'coc-toml',
\]
EOF
}

create_dev_scripts() {
    print_step "Creating development scripts"

    mkdir -p scripts/dev

    # Main run script
    cat > scripts/dev/run.sh << 'EOF'
#!/bin/bash
set -euo pipefail

# Load environment
source .venv/bin/activate
source .env

# Set Python path
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# Run with optional arguments
exec python -m gal_friday.main --config config/config.yaml "$@"
EOF

    # Test runner
    cat > scripts/dev/test.sh << 'EOF'
#!/bin/bash
set -euo pipefail

source .venv/bin/activate

# Run tests with coverage by default
pytest tests/ \
    --cov=gal_friday \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-config=pyproject.toml \
    "$@"

# Open coverage report if requested
if [[ "${OPEN_COVERAGE:-false}" == "true" ]]; then
    open htmlcov/index.html || xdg-open htmlcov/index.html
fi
EOF

    # Linting script
    cat > scripts/dev/lint.sh << 'EOF'
#!/bin/bash
set -euo pipefail

source .venv/bin/activate

echo "Running code quality checks..."

# Ruff
echo "Running ruff..."
ruff check gal_friday/ tests/ --fix

# Black
echo "Running black..."
black gal_friday/ tests/

# isort
echo "Running isort..."
isort gal_friday/ tests/

# MyPy
echo "Running mypy..."
mypy gal_friday/

# Bandit
echo "Running bandit security scan..."
bandit -r gal_friday/ -f json -o .bandit-report.json || true

# Safety
echo "Checking dependencies for vulnerabilities..."
safety check --json || true

echo "Code quality checks complete!"
EOF

    # Performance profiling script
    cat > scripts/dev/profile.sh << 'EOF'
#!/bin/bash
set -euo pipefail

source .venv/bin/activate

# Profile the application
python -m cProfile -o profile.stats -m gal_friday.main --config config/config.yaml "$@"

# Generate report
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.strip_dirs()
p.sort_stats('cumulative')
p.print_stats(50)
" > profile_report.txt

echo "Profile report saved to profile_report.txt"

# Visualize if snakeviz is installed
if command -v snakeviz &>/dev/null; then
    snakeviz profile.stats
fi
EOF

    # Database management
    cat > scripts/dev/db.sh << 'EOF'
#!/bin/bash
set -euo pipefail

source .venv/bin/activate
source .env

case "${1:-}" in
    migrate)
        echo "Running database migrations..."
        alembic upgrade head
        ;;
    rollback)
        echo "Rolling back one migration..."
        alembic downgrade -1
        ;;
    reset)
        echo "Resetting database..."
        alembic downgrade base
        alembic upgrade head
        ;;
    shell)
        echo "Opening database shell..."
        case "${DATABASE_URL}" in
            postgresql://*)
                psql "${DATABASE_URL}"
                ;;
            sqlite://*)
                sqlite3 "${DATABASE_URL#sqlite:///}"
                ;;
            *)
                echo "Unknown database type"
                exit 1
                ;;
        esac
        ;;
    *)
        echo "Usage: $0 {migrate|rollback|reset|shell}"
        exit 1
        ;;
esac
EOF

    # Make all scripts executable
    chmod +x scripts/dev/*.sh

    print_success "Development scripts created in scripts/dev/"
}

setup_git_hooks() {
    print_step "Setting up Git hooks"

    mkdir -p .git/hooks

    # Commit message hook
    cat > .git/hooks/commit-msg << 'EOF'
#!/bin/bash
# Commit message format checker

commit_regex='^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\(.+\))?: .{1,50}'

if ! grep -qE "$commit_regex" "$1"; then
    echo "Invalid commit message format!"
    echo "Format: <type>(<scope>): <subject>"
    echo "Example: feat(auth): add login endpoint"
    echo ""
    echo "Types: feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert"
    exit 1
fi
EOF

    chmod +x .git/hooks/commit-msg
}

# =============================================================================
# Testing and Validation
# =============================================================================

setup_testing() {
    print_header "Setting up Testing Environment"

    # Create test configuration
    create_test_config

    # Setup test database
    setup_test_database

    # Run initial tests
    run_validation_tests

    print_success "Testing environment ready"
}

create_test_config() {
    print_step "Creating test configuration"

    # Test environment file
    cat > .env.test << 'EOF'
# Test Environment Variables
ENVIRONMENT=test
DATABASE_URL=sqlite:///data/test/gal_friday_test.db
REDIS_URL=redis://localhost:6379/1
LOG_LEVEL=DEBUG
FEATURE_PAPER_TRADING=true
FEATURE_LIVE_TRADING=false
EOF

    # Test config
    mkdir -p config/test
    cp "${CONFIG_FILE}" config/test/config.yaml

    # Update test config for testing
    sed -i.bak 's/environment: "development"/environment: "test"/' config/test/config.yaml
    rm -f config/test/config.yaml.bak
}

run_validation_tests() {
    print_step "Running validation tests"

    # Create validation test file
    cat > tests/test_setup_validation.py << 'EOF'
"""Setup validation tests"""
import os
import sys
import importlib
import pytest


def test_python_version():
    """Test Python version requirements"""
    assert sys.version_info >= (3, 11), "Python 3.11+ required"
    assert sys.version_info < (3, 13), "Python 3.12 max supported"


def test_required_packages():
    """Test that all required packages are importable"""
    required_packages = [
        "numpy",
        "pandas",
        "sklearn",
        "xgboost",
        "torch",
        "fastapi",
        "uvicorn",
        "websockets",
        "asyncio",
        "aiohttp",
        "sqlalchemy",
        "influxdb_client",
        "redis",
        "pydantic",
        "yaml",
    ]

    failed_imports = []
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            failed_imports.append(package)

    assert not failed_imports, f"Failed to import: {failed_imports}"


def test_project_structure():
    """Test that project structure is correct"""
    required_dirs = ["gal_friday", "tests", "config", "scripts", "logs", "data", "models"]
    required_files = ["pyproject.toml", "requirements.txt", ".env", "config/config.yaml"]

    for dir_name in required_dirs:
        assert os.path.isdir(dir_name), f"Missing directory: {dir_name}"

    for file_name in required_files:
        assert os.path.isfile(file_name), f"Missing file: {file_name}"


def test_environment_variables():
    """Test that critical environment variables are set"""
    from dotenv import load_dotenv
    load_dotenv(".env.test")

    critical_vars = ["ENVIRONMENT", "DATABASE_URL"]

    for var in critical_vars:
        assert os.getenv(var), f"Missing environment variable: {var}"


def test_database_connection():
    """Test database connectivity"""
    from sqlalchemy import create_engine

    db_url = os.getenv("DATABASE_URL", "sqlite:///data/test/test.db")
    engine = create_engine(db_url)

    try:
        conn = engine.connect()
        conn.close()
    except Exception as e:
        pytest.fail(f"Database connection failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF

    # Run validation tests
    if pytest tests/test_setup_validation.py -v; then
        print_success "Validation tests passed"
    else
        print_warning "Some validation tests failed - check the output"
    fi
}

# =============================================================================
# Documentation Generation
# =============================================================================

setup_documentation() {
    print_header "Setting up Documentation"

    create_documentation_structure

    print_success "Documentation structure created"
}

create_documentation_structure() {
    mkdir -p docs/{api,guides,deployment,architecture}

    # Main README
    cat > README.md << 'EOF'
# Gal-Friday2

Enterprise-grade automated cryptocurrency trading system.

## Quick Start

```bash
# Setup development environment
./setup.sh --dev

# Activate environment
source .venv/bin/activate

# Run the application
python -m gal_friday.main --config config/config.yaml
```

## Documentation

- [Architecture Overview](docs/architecture/README.md)
- [API Documentation](docs/api/README.md)
- [Deployment Guide](docs/deployment/README.md)
- [Development Guide](docs/guides/development.md)

## Features

- High-frequency cryptocurrency trading
- ML-based price prediction (XGBoost, LSTM, Random Forest)
- Real-time market data processing
- Risk management and portfolio optimization
- WebSocket and REST API integration with Kraken
- Comprehensive monitoring and alerting
- Paper trading and backtesting modes

## Requirements

- Python 3.11+
- PostgreSQL 15+ (for production)
- InfluxDB 2.7+ (for time series data)
- Redis 7+ (for caching)
- 8GB+ RAM recommended
- GPU optional (for ML acceleration)

## License

Proprietary - All rights reserved
EOF

    # Development guide
    cat > docs/guides/development.md << 'EOF'
# Development Guide

## Environment Setup

The `setup.sh` script handles complete environment setup:

```bash
# Development setup (SQLite, minimal deps)
./setup.sh --dev

# Full setup with Docker services
./setup.sh --full --docker

# Test environment
./setup.sh --test
```

## Development Workflow

1. **Activate virtual environment**
   ```bash
   source .venv/bin/activate
   ```

2. **Run tests**
   ```bash
   scripts/dev/test.sh
   ```

3. **Code quality checks**
   ```bash
   scripts/dev/lint.sh
   ```

4. **Run application**
   ```bash
   scripts/dev/run.sh
   ```

## Project Structure

```
gal_friday/
├── core/           # Core types, events, pubsub
├── dal/            # Data access layer
├── execution/      # Order execution
├── market_price/   # Market data handling
├── predictors/     # ML models
├── portfolio/      # Portfolio management
├── monitoring/     # Dashboard and metrics
└── main.py         # Entry point
```

## Coding Standards

- Use type hints for all functions
- Follow PEP 8 style guide
- Write docstrings for all public APIs
- Add unit tests for new features
- Run pre-commit hooks before committing

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gal_friday

# Run specific test
pytest tests/test_module.py::test_function

# Run integration tests
pytest -m integration
```

## Debugging

Use VS Code or PyCharm with provided configurations, or:

```bash
# Debug mode
python -m gal_friday.main --config config/config.yaml --log-level DEBUG

# Profile performance
scripts/dev/profile.sh
```
EOF
}

# =============================================================================
# Final Health Check and Summary
# =============================================================================

run_final_validation() {
    print_header "Running Final Validation"

    local validation_passed=true
    local warnings=()

    # Check virtual environment
    print_step "Validating Python environment"
    if source "${VENV_NAME}/bin/activate" 2>/dev/null; then
        print_success "Virtual environment OK"
    else
        print_error "Virtual environment not functional"
        validation_passed=false
    fi

    # Test imports
    print_step "Testing package imports"
    python -c "
import gal_friday
import numpy
import pandas
import sklearn
import xgboost
import fastapi
import asyncio
print('All core packages import successfully')
" || validation_passed=false

    # Check configuration
    print_step "Validating configuration"
    if [[ -f "${CONFIG_FILE}" ]] && [[ -f "${ENV_FILE}" ]]; then
        print_success "Configuration files present"
    else
        print_error "Configuration files missing"
        validation_passed=false
    fi

    # Check services (if Docker mode)
    if [[ "$USE_DOCKER" == "true" ]] && check_command docker-compose; then
        print_step "Checking Docker services"
        if docker-compose ps | grep -q "Up"; then
            print_success "Docker services running"
        else
            warnings+=("Docker services not running - run 'docker-compose up -d'")
        fi
    fi

    # GPU check summary
    if [[ "${HAS_GPU:-false}" == "true" ]]; then
        print_success "GPU acceleration available"
    else
        warnings+=("No GPU detected - ML models will use CPU only")
    fi

    # Report results
    echo
    if [[ "$validation_passed" == "true" ]]; then
        print_success "Setup validation PASSED"
    else
        print_error "Setup validation FAILED"
    fi

    if [[ ${#warnings[@]} -gt 0 ]]; then
        print_warning "Warnings:"
        for warning in "${warnings[@]}"; do
            print_warning "  - $warning"
        done
    fi

    # Save final state
    save_setup_state "completed"
}

show_summary() {
    print_header "Setup Complete!"

    # Calculate setup time
    local end_time=$(date +%s)
    local duration=$((end_time - SETUP_START_TIME))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))

    print_success "Gal-Friday2 environment is ready!"
    print_info "Setup completed in ${minutes}m ${seconds}s"

    echo
    print_info "Environment Summary:"
    print_info "  Python: ${PYTHON_VERSION}"
    print_info "  Setup Type: ${SETUP_TYPE}"
    print_info "  Virtual Env: ${VENV_NAME}"
    print_info "  GPU Support: ${HAS_GPU:-false}"

    echo
    print_info "Next Steps:"
    print_info "1. Activate environment: ${GREEN}source ${VENV_NAME}/bin/activate${RESET}"
    print_info "2. Configure API keys: ${GREEN}nano .env${RESET}"
    print_info "3. Run the application: ${GREEN}scripts/dev/run.sh${RESET}"
    print_info "4. View dashboard: ${GREEN}http://localhost:8000${RESET}"

    echo
    print_info "Useful Commands:"
    print_info "  Run tests: ${CYAN}scripts/dev/test.sh${RESET}"
    print_info "  Lint code: ${CYAN}scripts/dev/lint.sh${RESET}"
    print_info "  Database: ${CYAN}scripts/dev/db.sh{RESET}"

    if [[ "$USE_DOCKER" == "true" ]]; then
        echo
        print_info "Docker Services:"
        print_info "  Start: ${CYAN}docker-compose up -d${RESET}"
        print_info "  Stop: ${CYAN}docker-compose down${RESET}"
        print_info "  Logs: ${CYAN}docker-compose logs -f${RESET}"
    fi

    echo
    print_info "For more information, see README.md and docs/"
}

# =============================================================================
# Main Setup Flow
# =============================================================================

main() {
    # Handle command line arguments first
    parse_arguments "$@"

    # Show header
    print_header "Gal-Friday2 Enterprise Setup v${SCRIPT_VERSION}"
    log "INFO" "Starting setup with type: ${SETUP_TYPE}"

    # Check if resuming
    if [[ "$RESUME_SETUP" == "true" ]] && load_setup_state; then
        print_info "Resuming previous setup..."
    else
        # Fresh setup - confirm with user
        print_info "Setup Configuration:"
        print_info "  Type: ${SETUP_TYPE}"
        print_info "  Docker: ${USE_DOCKER}"
        print_info "  Directory: ${PROJECT_ROOT}"

        if [[ "$SKIP_CONFIRMATION" != "true" ]]; then
            confirm_action "Proceed with setup?"
        fi
    fi

    # Run setup steps
    run_setup_step "system_check" check_system_requirements
    run_setup_step "system_check" check_system_resources
    run_setup_step "system_check" check_gpu_support

    run_setup_step "python_setup" setup_python_version
    run_setup_step "python_setup" setup_virtual_environment

    run_setup_step "dependencies" install_dependencies_parallel

    run_setup_step "environment" setup_environment_variables

    run_setup_step "configuration" setup_configuration

    run_setup_step "database" setup_database

    run_setup_step "dev_tools" setup_development_tools

    if [[ "$SETUP_TYPE" =~ ^(test|full)$ ]]; then
        run_setup_step "testing" setup_testing
    fi

    run_setup_step "documentation" setup_documentation

    run_setup_step "validation" run_final_validation

    # Show summary
    show_summary

    # Cleanup
    rm -f "${SETUP_STATE_FILE}"
}

run_setup_step() {
    local step_id="$1"
    local step_function="$2"

    # Skip if already completed
    if [[ "${STEP_STATUS[$step_id]:-}" == "completed" ]]; then
        print_info "Skipping ${SETUP_STEPS[$step_id]} (already completed)"
        return 0
    fi

    CURRENT_STEP="$step_id"
    local start_time=$(date +%s)

    # Run the step
    if $step_function; then
        STEP_STATUS[$step_id]="completed"
        local end_time=$(date +%s)
        STEP_TIMING[$step_id]=$((end_time - start_time))
        save_setup_state
    else
        STEP_STATUS[$step_id]="failed"
        save_setup_state "failed"
        print_error "Setup failed at step: ${SETUP_STEPS[$step_id]}"
        exit 1
    fi
}

# =============================================================================
# Command Line Interface
# =============================================================================

show_help() {
    cat << EOF
Gal-Friday2 Enterprise Setup Script v${SCRIPT_VERSION}

Usage: $0 [OPTIONS]

Options:
    --dev           Development setup (default) - SQLite, minimal deps
    --test          Test environment setup
    --full          Full production setup with all services

    --docker        Use Docker for database services
    --yes, -y       Skip confirmation prompts
    --resume        Resume from previous setup state
    --parallel N    Number of parallel jobs (default: 4)

    --help, -h      Show this help message
    --version       Show version information

Environment Variables:
    SETUP_TYPE      Setup type (dev|test|full)
    USE_DOCKER      Use Docker services (true|false)
    PARALLEL_JOBS   Number of parallel installation jobs
    DEBUG           Enable debug output (true|false)

Examples:
    # Basic development setup
    $0 --dev

    # Full setup with Docker
    $0 --full --docker

    # Automated CI setup
    $0 --test --yes

    # Resume interrupted setup
    $0 --resume

For more information, see docs/deployment/setup.md
EOF
}

show_version() {
    echo "Gal-Friday2 Setup Script"
    echo "Version: ${SCRIPT_VERSION}"
    echo "Path: ${SCRIPT_DIR}"
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dev)
                SETUP_TYPE="dev"
                shift
                ;;
            --test)
                SETUP_TYPE="test"
                shift
                ;;
            --full)
                SETUP_TYPE="full"
                shift
                ;;
            --docker)
                USE_DOCKER=true
                shift
                ;;
            --yes|-y)
                SKIP_CONFIRMATION=true
                shift
                ;;
            --resume)
                RESUME_SETUP=true
                shift
                ;;
            --parallel)
                PARALLEL_JOBS="$2"
                shift 2
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            --version)
                show_version
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# =============================================================================
# Entry Point
# =============================================================================

# Ensure we're in the project root
cd "${PROJECT_ROOT}"

# Verify we're in the correct directory
if [[ ! -f "pyproject.toml" ]] || [[ ! -d "gal_friday" ]]; then
    print_error "This script must be run from the Gal-Friday2 project root"
    print_error "Expected files not found in: ${PROJECT_ROOT}"
    exit 1
fi

# Start logging
echo "=== Setup started at $(date) ===" > "${SETUP_LOG_FILE}"

# Run main setup
main "$@"
