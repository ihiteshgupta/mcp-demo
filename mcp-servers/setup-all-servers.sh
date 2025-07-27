#!/bin/bash

# Setup script for all MCP servers
# This script installs dependencies for all servers in the ecosystem

set -e

echo "🚀 Setting up MCP Servers Ecosystem"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
        print_error "Python 3.9+ is required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    print_success "Python $PYTHON_VERSION found"
}

# Setup virtual environment for a server
setup_server() {
    local server_name=$1
    local server_path=$2
    
    print_status "Setting up $server_name..."
    
    if [ ! -d "$server_path" ]; then
        print_warning "Directory $server_path does not exist, skipping $server_name"
        return
    fi
    
    cd "$server_path"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment for $server_name"
        $PYTHON_CMD -m venv venv
    fi
    
    # Activate virtual environment
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    else
        print_error "Could not find virtual environment activation script"
        return 1
    fi
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install server-specific requirements if available
    if [ -f "requirements.txt" ]; then
        print_status "Installing requirements for $server_name"
        pip install -r requirements.txt
    else
        print_status "Installing core MCP requirements for $server_name"
        pip install mcp fastapi uvicorn pydantic aiohttp
    fi
    
    # Install optional dependencies based on server type
    case $server_name in
        "llm-provider")
            print_status "Installing LLM provider dependencies"
            pip install openai anthropic boto3 requests sentence-transformers torch transformers || print_warning "Some LLM dependencies failed to install"
            ;;
        "vector-store")
            print_status "Installing vector store dependencies"
            pip install chromadb qdrant-client sentence-transformers numpy faiss-cpu || print_warning "Some vector store dependencies failed to install"
            ;;
        "memory")
            print_status "Installing memory store dependencies"
            pip install redis aiosqlite asyncpg motor psutil || print_warning "Some memory store dependencies failed to install"
            ;;
        "web-fetch")
            print_status "Installing web fetch dependencies"
            pip install beautifulsoup4 lxml trafilatura feedparser markdownify langchain || print_warning "Some web fetch dependencies failed to install"
            ;;
        "orchestrator")
            print_status "Installing orchestrator dependencies"
            pip install psutil pyyaml structlog prometheus-client || print_warning "Some orchestrator dependencies failed to install"
            ;;
    esac
    
    # Test import of main module
    if [ -f "main.py" ]; then
        print_status "Testing $server_name module import"
        if $PYTHON_CMD -c "import sys; sys.path.append('.'); import main" 2>/dev/null; then
            print_success "$server_name setup completed successfully"
        else
            print_warning "$server_name module import test failed (might be due to missing optional dependencies)"
        fi
    fi
    
    deactivate
    cd - > /dev/null
}

# Create directories if they don't exist
create_directories() {
    print_status "Creating server directories..."
    
    mkdir -p llm-provider
    mkdir -p vector-store
    mkdir -p memory
    mkdir -p web-fetch
    mkdir -p orchestrator/config
    mkdir -p data
    mkdir -p logs
    
    print_success "Directories created"
}

# Create requirements files for each server
create_requirements_files() {
    print_status "Creating individual requirements files..."
    
    # LLM Provider requirements
    cat > llm-provider/requirements.txt << 'EOF'
mcp>=0.6.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
aiohttp>=3.8.0
openai>=1.3.0
anthropic>=0.7.0
boto3>=1.28.0
requests>=2.31.0
sentence-transformers>=2.2.0
torch>=2.0.0
transformers>=4.30.0
asyncio-mqtt>=0.13.0
structlog>=23.0.0
python-json-logger>=2.0.0
orjson>=3.9.0
EOF

    # Vector Store requirements
    cat > vector-store/requirements.txt << 'EOF'
mcp>=0.6.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
aiohttp>=3.8.0
chromadb>=0.4.0
qdrant-client>=1.6.0
sentence-transformers>=2.2.0
numpy>=1.24.0
faiss-cpu>=1.7.4
langchain>=0.0.300
structlog>=23.0.0
python-json-logger>=2.0.0
EOF

    # Memory requirements
    cat > memory/requirements.txt << 'EOF'
mcp>=0.6.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
redis[hiredis]>=5.0.0
aiosqlite>=0.19.0
asyncpg>=0.28.0
motor>=3.3.0
psutil>=5.9.0
structlog>=23.0.0
python-json-logger>=2.0.0
cryptography>=41.0.0
EOF

    # Web Fetch requirements
    cat > web-fetch/requirements.txt << 'EOF'
mcp>=0.6.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
aiohttp>=3.8.0
aiofiles>=23.0.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
trafilatura>=1.6.0
feedparser>=6.0.0
markdownify>=0.11.0
langchain>=0.0.300
structlog>=23.0.0
python-json-logger>=2.0.0
EOF

    # Orchestrator requirements
    cat > orchestrator/requirements.txt << 'EOF'
mcp>=0.6.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
aiohttp>=3.8.0
psutil>=5.9.0
pyyaml>=6.0.0
structlog>=23.0.0
python-json-logger>=2.0.0
prometheus-client>=0.17.0
cryptography>=41.0.0
asyncio-mqtt>=0.13.0
EOF

    print_success "Requirements files created"
}

# Create startup scripts
create_startup_scripts() {
    print_status "Creating startup scripts..."
    
    # Start all services script
    cat > start-all.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting all MCP servers..."

# Start orchestrator in the background
cd orchestrator && python main.py --auto-start &
ORCHESTRATOR_PID=$!

echo "Orchestrator started with PID: $ORCHESTRATOR_PID"
echo "Services are being started by the orchestrator..."
echo ""
echo "Check status with:"
echo "  curl http://localhost:8000/health"
echo ""
echo "Stop with:"
echo "  kill $ORCHESTRATOR_PID"
echo ""
echo "Logs are available in the console above."

# Keep script running
wait $ORCHESTRATOR_PID
EOF

    # Stop all services script
    cat > stop-all.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping all MCP servers..."

# Kill all python processes running MCP servers
pkill -f "main.py"

echo "All MCP servers stopped"
EOF

    # Status check script
    cat > check-status.sh << 'EOF'
#!/bin/bash
echo "📊 MCP Servers Status"
echo "===================="

# Check if orchestrator is running
if pgrep -f "orchestrator.*main.py" > /dev/null; then
    echo "✅ Orchestrator: Running"
    
    # Try to get status from orchestrator
    if command -v curl &> /dev/null; then
        echo ""
        echo "Service Details:"
        curl -s http://localhost:8000/health 2>/dev/null || echo "❌ Could not connect to orchestrator API"
    fi
else
    echo "❌ Orchestrator: Not running"
fi

echo ""
echo "Process List:"
ps aux | grep "main.py" | grep -v grep || echo "No MCP server processes found"
EOF

    chmod +x start-all.sh stop-all.sh check-status.sh
    
    print_success "Startup scripts created"
}

# Create development configuration
create_dev_config() {
    print_status "Creating development configuration..."
    
    # Create .env file
    cat > .env << 'EOF'
# MCP Servers Development Configuration

# Logging
LOG_LEVEL=INFO
DEBUG_MODE=false

# Ports
ORCHESTRATOR_PORT=8000
LLM_PROVIDER_PORT=8002
VECTOR_STORE_PORT=8003
MEMORY_PORT=8004
WEB_FETCH_PORT=8005
SEQUENTIAL_THINKER_PORT=8001

# API Keys (set these for production)
# OPENAI_API_KEY=your_openai_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here
# AWS_ACCESS_KEY_ID=your_aws_key_here
# AWS_SECRET_ACCESS_KEY=your_aws_secret_here

# Database URLs
# REDIS_URL=redis://localhost:6379
# POSTGRESQL_URL=postgresql://user:pass@localhost/mcp_db
# MONGODB_URL=mongodb://localhost:27017/mcp_db

# Development settings
AUTO_START_SERVICES=true
ENABLE_HEALTH_MONITORING=true
CACHE_ENABLED=true
RATE_LIMITING=false
EOF

    print_success "Development configuration created"
}

# Main setup function
main() {
    print_status "Starting MCP Servers Ecosystem setup..."
    
    # Check prerequisites
    check_python
    
    # Create directory structure
    create_directories
    
    # Create requirements files
    create_requirements_files
    
    # Setup each server
    setup_server "orchestrator" "orchestrator"
    setup_server "llm-provider" "llm-provider"
    setup_server "vector-store" "vector-store"
    setup_server "memory" "memory"
    setup_server "web-fetch" "web-fetch"
    
    # Create startup scripts
    create_startup_scripts
    
    # Create development configuration
    create_dev_config
    
    echo ""
    print_success "🎉 MCP Servers Ecosystem setup completed!"
    echo ""
    echo "Next steps:"
    echo "1. Review and update configuration files in each server directory"
    echo "2. Set up API keys in .env file if using cloud providers"
    echo "3. Start the ecosystem: ./start-all.sh"
    echo "4. Check status: ./check-status.sh"
    echo ""
    echo "Documentation: See README.md for detailed usage instructions"
    echo "Configuration: Each server has its own config.json file"
    echo "Monitoring: Access http://localhost:8000 for orchestrator status"
    echo ""
    print_status "Happy coding! 🚀"
}

# Run main function
main "$@"