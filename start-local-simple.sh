#!/bin/bash

# MCP GenAI Demo - Simple Local Setup with LM Studio
# This script starts supporting services in Docker and apps locally

set -e

echo "🚀 Starting MCP GenAI Demo with LM Studio (Simple Setup)..."
echo "================================================================"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "✅ Docker is running"

# Check if LM Studio is running
echo "🤖 Checking LM Studio..."
if ! curl -s http://localhost:1234/v1/models >/dev/null 2>&1; then
    echo "❌ LM Studio is not running on port 1234."
    echo ""
    echo "Please:"
    echo "1. Start LM Studio desktop application"
    echo "2. Load a model (e.g., Llama 3.1 8B, Mistral 7B)"
    echo "3. Start the local server (port 1234)"
    echo ""
    echo "Download LM Studio from: https://lmstudio.ai/"
    exit 1
fi

echo "✅ LM Studio is running"

# Display available models
echo ""
echo "📋 Available LM Studio models:"
curl -s http://localhost:1234/v1/models | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    models = data.get('data', [])
    if models:
        for model in models:
            print(f\"   - {model['id']}\")
    else:
        print('   No models loaded')
except:
    print('   Could not parse model list')
" || echo "   Could not fetch model list"

# Start supporting services only
echo ""
echo "📦 Starting supporting services (Redis, ChromaDB)..."
docker-compose -f docker-compose.simple.yml up -d redis chroma

echo "⏳ Waiting for services to be ready..."

# Wait for Redis
echo "📮 Waiting for Redis..."
timeout=60
counter=0
while ! docker exec mcp-redis redis-cli ping >/dev/null 2>&1; do
    if [ $counter -ge $timeout ]; then
        echo "❌ Timeout waiting for Redis to start"
        exit 1
    fi
    sleep 2
    counter=$((counter + 2))
done
echo "✅ Redis is ready"

# Wait for ChromaDB
echo "🗃️ Waiting for ChromaDB..."
counter=0
while ! curl -s http://localhost:8002/api/v1/heartbeat >/dev/null 2>&1; do
    if [ $counter -ge $timeout ]; then
        echo "❌ Timeout waiting for ChromaDB to start"
        exit 1
    fi
    sleep 2
    counter=$((counter + 2))
done
echo "✅ ChromaDB is ready"

# Start MCP Server
echo ""
echo "🔧 Starting MCP Server..."
cd mcp-server

# Setup Python environment
if [ ! -d ".venv" ]; then
    echo "📦 Creating Python virtual environment..."
    # Try to use Python 3.11 or 3.10 for better compatibility
    if command -v python3.11 &> /dev/null; then
        python3.11 -m venv .venv
    elif command -v python3.10 &> /dev/null; then
        python3.10 -m venv .venv
    elif command -v python3.9 &> /dev/null; then
        python3.9 -m venv .venv
    else
        echo "⚠️  Using system Python 3, may have compatibility issues with Python 3.13"
        python3 -m venv .venv
    fi
fi

source .venv/bin/activate
echo "📦 Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Export environment variables for LM Studio
export REDIS_HOST=localhost
export REDIS_PORT=6379
export CHROMA_HOST=localhost
export CHROMA_PORT=8002
export HOST=0.0.0.0
export PORT=8000
export DEBUG=true
export LOG_LEVEL=info
export LMSTUDIO_BASE_URL=http://localhost:1234/v1
export LMSTUDIO_API_KEY=lm-studio
export MCP_SERVER_NAME=genai-mcp-server-lmstudio
export MCP_SERVER_VERSION=1.0.0

# Start the server
echo "🚀 Starting MCP Server on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
SERVER_PID=$!

# Give server time to start
sleep 5

# Start MCP Client
echo ""
echo "🌐 Starting MCP Client..."
cd ../mcp-client

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

# Start the client
echo "🚀 Starting Next.js client on port 3000..."
PORT=3000 npm run dev &
CLIENT_PID=$!

# Wait for client to be ready
echo "⏳ Waiting for client to start..."
counter=0
while ! curl -s http://localhost:3000 >/dev/null 2>&1; do
    if [ $counter -ge 60 ]; then
        echo "❌ Timeout waiting for client to start"
        exit 1
    fi
    sleep 2
    counter=$((counter + 2))
done

echo ""
echo "================================================================"
echo "✅ All services are running!"
echo ""
echo "🌐 Frontend:      http://localhost:3000"
echo "🔧 MCP Server:    http://localhost:8000"
echo "📚 API Docs:      http://localhost:8000/docs"
echo "🤖 LM Studio:     http://localhost:1234"
echo "📮 Redis:         http://localhost:6379"
echo "🗃️ ChromaDB:      http://localhost:8002"
echo ""
echo "ℹ️  Using LM Studio for AI inference"
echo "================================================================"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    
    # Kill local processes
    if [ ! -z "$SERVER_PID" ]; then
        echo "Stopping MCP Server..."
        kill $SERVER_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$CLIENT_PID" ]; then
        echo "Stopping MCP Client..."
        kill $CLIENT_PID 2>/dev/null || true
    fi
    
    # Stop Docker services
    echo "Stopping Docker services..."
    cd $(dirname $0)
    docker-compose -f docker-compose.simple.yml down
    
    echo "✅ All services stopped"
    exit 0
}

# Set up trap for cleanup
trap cleanup INT TERM EXIT

# Keep script running
wait