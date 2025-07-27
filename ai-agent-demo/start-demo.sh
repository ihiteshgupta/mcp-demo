#!/bin/bash

# AI Agent Creation Demo Startup Script
# =====================================

set -e

# Configuration
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$DEMO_DIR/venv"
HOST="${HOST:-localhost}"
PORT="${PORT:-3002}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python() {
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        log_error "Python is not installed. Please install Python 3.9 or higher."
        exit 1
    fi

    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    REQUIRED_VERSION="3.9"
    
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        log_error "Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    log_success "Using Python $PYTHON_VERSION"
}

# Function to setup virtual environment
setup_venv() {
    log_info "Setting up virtual environment..."
    
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating virtual environment..."
        $PYTHON_CMD -m venv "$VENV_DIR"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip
    
    # Install dependencies
    log_info "Installing dependencies..."
    pip install -r "$DEMO_DIR/requirements.txt"
    
    log_success "Virtual environment setup complete"
}

# Function to check MCP server connectivity
check_mcp_servers() {
    log_info "Checking MCP server connectivity..."
    
    # Define servers to check
    declare -A servers=(
        ["Prompt Builder"]="http://localhost:8006/health"
        ["LLM Provider"]="http://localhost:8002/health"
        ["Sequential Thinker"]="http://localhost:8001/health"
        ["Memory Management"]="http://localhost:8004/health"
    )
    
    local all_healthy=true
    
    for server_name in "${!servers[@]}"; do
        local url="${servers[$server_name]}"
        log_info "Checking $server_name at $url..."
        
        if command_exists curl; then
            if curl -s -f "$url" >/dev/null 2>&1; then
                log_success "$server_name is healthy"
            else
                log_warning "$server_name is not responding"
                all_healthy=false
            fi
        else
            log_warning "curl not found, skipping health check for $server_name"
        fi
    done
    
    if [ "$all_healthy" = false ]; then
        log_warning "Some MCP servers are not responding."
        log_info "The demo will work with simulated responses, but for full functionality:"
        log_info "1. Start the Prompt Builder MCP server (port 8006)"
        log_info "2. Start the underlying MCP servers (LLM Provider, Sequential Thinker, Memory)"
        echo
    fi
}

# Function to create necessary directories
setup_directories() {
    log_info "Setting up directories..."
    
    mkdir -p "$DEMO_DIR/static"
    mkdir -p "$DEMO_DIR/data"
    mkdir -p "$DEMO_DIR/logs"
    
    log_success "Directories created"
}

# Function to start the demo server
start_server() {
    log_info "Starting AI Agent Creation Demo..."
    log_info "Demo will be available at: http://$HOST:$PORT"
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Set environment variables
    export PYTHONPATH="$DEMO_DIR:$PYTHONPATH"
    
    # Start the server
    cd "$DEMO_DIR"
    
    if [ "$1" = "--dev" ] || [ "$1" = "--reload" ]; then
        log_info "Starting in development mode with auto-reload..."
        uvicorn main:app --host "$HOST" --port "$PORT" --reload --log-level info
    else
        log_info "Starting demo server..."
        uvicorn main:app --host "$HOST" --port "$PORT" --log-level info
    fi
}

# Function to display help
show_help() {
    cat << EOF
AI Agent Creation Demo Startup Script

Usage: $0 [OPTIONS]

Options:
    --dev, --reload     Start in development mode with auto-reload
    --host HOST         Set the host (default: localhost)
    --port PORT         Set the port (default: 3002)
    --skip-checks       Skip MCP server connectivity checks
    --help, -h          Show this help message

Environment Variables:
    HOST                Host to bind to (default: localhost)
    PORT                Port to bind to (default: 3002)

Examples:
    $0                          # Start demo server
    $0 --dev                    # Start in development mode
    $0 --host 0.0.0.0 --port 8080  # Custom host and port
    $0 --skip-checks            # Skip MCP server checks

Before running the demo:
1. Ensure MCP servers are running (optional for demo simulation)
2. Configure demo-config.json if needed

EOF
}

# Main function
main() {
    echo "🤖 AI Agent Creation Demo"
    echo "========================"
    echo
    
    # Parse arguments
    local skip_checks=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dev|--reload)
                DEV_MODE=true
                shift
                ;;
            --host)
                HOST="$2"
                shift 2
                ;;
            --port)
                PORT="$2"
                shift 2
                ;;
            --skip-checks)
                skip_checks=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Setup process
    check_python
    setup_directories
    setup_venv
    
    if [ "$skip_checks" = false ]; then
        check_mcp_servers
    fi
    
    # Start server
    start_server "$@"
}

# Handle interrupts gracefully
trap 'log_info "Shutting down AI Agent Creation Demo..."; exit 0' INT TERM

# Run main function
main "$@"