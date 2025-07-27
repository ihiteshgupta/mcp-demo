#!/bin/bash

# Complete MCP Demo System Startup Script
# =======================================
# This script starts the entire MCP ecosystem for the AI Agent Creation Demo

set -e

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
PID_DIR="$PROJECT_ROOT/.pids"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is available
check_port() {
    local port=$1
    if command_exists nc; then
        ! nc -z localhost $port 2>/dev/null
    else
        ! netstat -ln 2>/dev/null | grep -q ":$port "
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local name=$1
    local port=$2
    local max_attempts=30
    local attempt=1
    
    log_info "Waiting for $name to be ready on port $port..."
    
    while [ $attempt -le $max_attempts ]; do
        if command_exists curl; then
            if curl -s -f "http://localhost:$port/health" >/dev/null 2>&1; then
                log_success "$name is ready!"
                return 0
            fi
        else
            if nc -z localhost $port 2>/dev/null; then
                log_success "$name is ready!"
                return 0
            fi
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    echo
    log_warning "$name is not responding after $((max_attempts * 2)) seconds"
    return 1
}

# Function to start a service
start_service() {
    local name=$1
    local directory=$2
    local command=$3
    local port=$4
    local log_file="$LOG_DIR/${name}.log"
    local pid_file="$PID_DIR/${name}.pid"
    
    log_step "Starting $name..."
    
    # Check if port is available
    if ! check_port $port; then
        log_warning "$name port $port is already in use. Skipping..."
        return 0
    fi
    
    # Create directories
    mkdir -p "$LOG_DIR" "$PID_DIR"
    
    # Start service
    cd "$PROJECT_ROOT/$directory"
    
    # Start service in background
    eval "$command" > "$log_file" 2>&1 &
    local pid=$!
    echo $pid > "$pid_file"
    
    log_info "$name started with PID $pid (log: $log_file)"
    
    # Wait for service to be ready
    if wait_for_service "$name" $port; then
        return 0
    else
        log_error "$name failed to start properly"
        return 1
    fi
}

# Function to stop all services
stop_services() {
    log_info "Stopping all services..."
    
    if [ -d "$PID_DIR" ]; then
        for pid_file in "$PID_DIR"/*.pid; do
            if [ -f "$pid_file" ]; then
                local service_name=$(basename "$pid_file" .pid)
                local pid=$(cat "$pid_file")
                
                if kill -0 $pid 2>/dev/null; then
                    log_info "Stopping $service_name (PID: $pid)..."
                    kill $pid
                    sleep 2
                    
                    # Force kill if still running
                    if kill -0 $pid 2>/dev/null; then
                        log_warning "Force killing $service_name..."
                        kill -9 $pid
                    fi
                fi
                
                rm -f "$pid_file"
            fi
        done
        
        rmdir "$PID_DIR" 2>/dev/null || true
    fi
    
    log_success "All services stopped"
}

# Function to show service status
show_status() {
    echo
    echo "🚀 MCP Demo System Status"
    echo "========================"
    
    declare -A services=(
        ["Sequential Thinker"]="8001"
        ["LLM Provider"]="8002"
        ["Memory Management"]="8004"
        ["Prompt Builder"]="8006"
        ["AI Agent Demo"]="3002"
    )
    
    for service in "${!services[@]}"; do
        local port=${services[$service]}
        if command_exists curl && curl -s -f "http://localhost:$port/health" >/dev/null 2>&1; then
            echo -e "✅ $service: ${GREEN}Running${NC} on port $port"
        else
            echo -e "❌ $service: ${RED}Not running${NC} on port $port"
        fi
    done
    
    echo
    echo "📱 Access Points:"
    echo "• AI Agent Demo: http://localhost:3002"
    echo "• Prompt Builder API: http://localhost:8006/docs"
    echo "• LLM Provider API: http://localhost:8002/docs"
    echo
}

# Function to open browser
open_browser() {
    if [ "$OPEN_BROWSER" = "true" ] || [ "$1" = "--open" ]; then
        log_info "Opening browser..."
        
        if command_exists open; then
            open "http://localhost:3002"
        elif command_exists xdg-open; then
            xdg-open "http://localhost:3002"
        elif command_exists start; then
            start "http://localhost:3002"
        else
            log_info "Please open http://localhost:3002 in your browser"
        fi
    fi
}

# Function to show help
show_help() {
    cat << EOF
Complete MCP Demo System Startup Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    start           Start all MCP services and demo (default)
    stop            Stop all running services
    restart         Stop and restart all services
    status          Show status of all services
    logs            Show logs for all services

Options:
    --open          Open browser automatically
    --help, -h      Show this help message

Environment Variables:
    OPEN_BROWSER    Set to 'true' to open browser automatically

Services Started:
    1. Sequential Thinker MCP Server (port 8001)
    2. LLM Provider MCP Server (port 8002)
    3. Memory Management MCP Server (port 8004)
    4. Prompt Builder MCP Server (port 8006)
    5. AI Agent Demo Client (port 3002)

Examples:
    $0                  # Start all services
    $0 start --open     # Start services and open browser
    $0 stop             # Stop all services
    $0 restart          # Restart all services
    $0 status           # Check service status

EOF
}

# Function to show logs
show_logs() {
    if [ -d "$LOG_DIR" ]; then
        echo "📋 Recent logs from all services:"
        echo "================================"
        
        for log_file in "$LOG_DIR"/*.log; do
            if [ -f "$log_file" ]; then
                local service_name=$(basename "$log_file" .log)
                echo
                echo -e "${BLUE}--- $service_name ---${NC}"
                tail -10 "$log_file"
            fi
        done
    else
        log_info "No logs found. Services may not have been started yet."
    fi
}

# Function to start all services
start_all_services() {
    echo "🚀 Starting Complete MCP Demo System"
    echo "====================================="
    echo
    
    log_info "Project root: $PROJECT_ROOT"
    echo
    
    # Check Python
    if ! command_exists python3; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Start services in order
    local all_started=true
    
    # 1. Sequential Thinker MCP Server
    if start_service "Sequential Thinker" "sequential-thinker-mcp" "python main.py --port 8001" 8001; then
        log_success "Sequential Thinker MCP Server started"
    else
        all_started=false
    fi
    
    sleep 2
    
    # 2. LLM Provider MCP Server
    if start_service "LLM Provider" "mcp-servers/llm-provider" "python main.py --port 8002" 8002; then
        log_success "LLM Provider MCP Server started"
    else
        all_started=false
    fi
    
    sleep 2
    
    # 3. Memory Management MCP Server
    if start_service "Memory Management" "mcp-servers/memory" "python main.py --port 8004" 8004; then
        log_success "Memory Management MCP Server started"
    else
        all_started=false
    fi
    
    sleep 2
    
    # 4. Prompt Builder MCP Server
    if start_service "Prompt Builder" "prompt-builder-mcp" "python main.py --port 8006" 8006; then
        log_success "Prompt Builder MCP Server started"
    else
        all_started=false
    fi
    
    sleep 3
    
    # 5. AI Agent Demo Client
    if start_service "AI Agent Demo" "ai-agent-demo" "python main.py --port 3002" 3002; then
        log_success "AI Agent Demo Client started"
    else
        all_started=false
    fi
    
    echo
    if [ "$all_started" = true ]; then
        log_success "🎉 All services started successfully!"
        echo
        show_status
        open_browser "$@"
        
        echo "💡 Tips:"
        echo "• Use '$0 status' to check service health"
        echo "• Use '$0 logs' to view recent logs"
        echo "• Use '$0 stop' to stop all services"
        echo "• Press Ctrl+C to stop all services"
        echo
        
        # Keep script running to handle Ctrl+C
        log_info "Demo system is running. Press Ctrl+C to stop all services."
        
        # Wait for interrupt
        trap 'echo; log_info "Shutting down..."; stop_services; exit 0' INT TERM
        
        while true; do
            sleep 1
        done
    else
        log_error "Some services failed to start. Check logs for details."
        exit 1
    fi
}

# Main function
main() {
    case "${1:-start}" in
        start)
            start_all_services "$@"
            ;;
        stop)
            stop_services
            ;;
        restart)
            stop_services
            sleep 2
            start_all_services "$@"
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        --help|-h|help)
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"