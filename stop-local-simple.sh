#!/bin/bash

# MCP GenAI Demo - Stop Simple Local Setup
# This script stops all services and processes

set -e

echo "🛑 Stopping MCP GenAI Demo (Simple Setup)..."
echo "============================================"

# Stop Node.js processes
if [ -f .mcp-client.pid ]; then
    CLIENT_PID=$(cat .mcp-client.pid)
    if ps -p $CLIENT_PID > /dev/null 2>&1; then
        echo "⏹️  Stopping MCP Client (PID: $CLIENT_PID)..."
        kill $CLIENT_PID
    fi
    rm -f .mcp-client.pid
fi

# Stop MCP Server
if [ -f .mcp-server.pid ]; then
    SERVER_PID=$(cat .mcp-server.pid)
    if ps -p $SERVER_PID > /dev/null 2>&1; then
        echo "⏹️  Stopping MCP Server (PID: $SERVER_PID)..."
        kill $SERVER_PID
    fi
    rm -f .mcp-server.pid
fi

# Check LM Studio status (but don't stop it - it's a standalone application)
echo "🔍 Checking LM Studio status..."
if curl -s http://localhost:1234/v1/models >/dev/null 2>&1; then
    echo "✅ LM Studio is still running at http://localhost:1234"
    echo "   You can keep it running for other AI applications"
else
    echo "ℹ️  LM Studio appears to be stopped or not accessible"
fi

# Clean up any remaining Python processes
echo "🧹 Cleaning up any remaining processes..."
pkill -f "uvicorn main:app" || true
pkill -f "npm run dev" || true

# Stop Docker services
echo "🐳 Stopping Docker services..."
docker-compose -f docker-compose.simple.yml down

echo "✅ All services stopped successfully!"

# Ask user if they want to remove volumes (data)
echo ""
read -p "🗑️  Do you want to remove all data (AI models, database, etc.)? This will free up disk space but require re-downloading models next time. (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧹 Removing all volumes and data..."
    docker-compose -f docker-compose.simple.yml down -v
    
    # Remove orphaned containers
    docker container prune -f
    
    # Remove unused images (optional)
    read -p "🖼️  Do you also want to remove unused Docker images to free up more space? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker image prune -f
        echo "✅ Unused images removed"
    fi
    
    echo "✅ All data removed successfully!"
    echo "💡 Next time you start, AI models will need to be re-downloaded."
else
    echo "✅ Data preserved. Models and databases will be available for next startup."
fi

# Clean up any temporary files
# No temporary files to clean up with LM Studio integration

echo ""
echo "🎯 MCP GenAI Demo cleanup complete!"
echo ""
echo "💡 Note: LM Studio remains running as a standalone application."
echo "   You can keep it running for other AI applications or stop it manually."
echo ""
echo "🚀 To start again, run: ./start-local-simple.sh"
echo "   Make sure LM Studio is still running with a model loaded."
echo ""