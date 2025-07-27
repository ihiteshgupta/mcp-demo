#!/bin/bash

# MCP GenAI Services Startup Script
# This script starts all required services for the guided demo

set -e

echo "🚀 Starting MCP GenAI Services"
echo "=============================="
echo ""
echo "Starting required services:"
echo "• LM Studio (external - must be running)"
echo "• MCP Server (Python/FastAPI)"
echo "• Frontend Client (Next.js)"
echo ""

# Colors for better visualization
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to show component status
show_component_status() {
    local component=$1
    local status=$2
    local message=$3
    
    case $status in
        "starting")
            echo -e "${YELLOW}⏳ $component:${NC} $message"
            ;;
        "active")
            echo -e "${GREEN}✅ $component:${NC} $message"
            ;;
        "processing")
            echo -e "${BLUE}🔄 $component:${NC} $message"
            ;;
        "complete")
            echo -e "${PURPLE}🎯 $component:${NC} $message"
            ;;
        "error")
            echo -e "${RED}❌ $component:${NC} $message"
            ;;
    esac
}

# Function to simulate real-time thinking
show_thinking_process() {
    local step=$1
    local detail=$2
    echo -e "${CYAN}🧠 Sequential Thinking Step $step:${NC} $detail"
    sleep 1
}

# Check if services are running
echo "🔍 Checking service status..."
echo ""

# Check LM Studio
if curl -s http://localhost:1234/v1/models >/dev/null 2>&1; then
    show_component_status "LM Studio" "active" "Running at http://localhost:1234"
else
    show_component_status "LM Studio" "error" "Not running - please start LM Studio first"
    echo ""
    echo "Please start LM Studio and load a model, then run this demo again."
    exit 1
fi

# Check MCP Server
if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    show_component_status "MCP Server" "active" "Running at http://localhost:8000"
else
    show_component_status "MCP Server" "starting" "Starting MCP Server..."
    cd mcp-server
    source .venv/bin/activate 2>/dev/null || source venv/bin/activate
    uvicorn main:app --reload --port 8000 &
    MCP_SERVER_PID=$!
    echo $MCP_SERVER_PID > ../.demo-mcp-server.pid
    cd ..
    sleep 5
    show_component_status "MCP Server" "active" "Started with PID $MCP_SERVER_PID"
fi

# Check MCP Client/Frontend
if curl -s http://localhost:3000 >/dev/null 2>&1; then
    show_component_status "Frontend" "active" "Running at http://localhost:3000"
else
    show_component_status "Frontend" "starting" "Starting Next.js frontend..."
    cd mcp-client
    npm run dev &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../.demo-frontend.pid
    cd ..
    sleep 8
    show_component_status "Frontend" "active" "Started with PID $FRONTEND_PID"
fi

echo ""
echo -e "${GREEN}🎉 ALL SERVICES STARTED!${NC}"
echo "========================"
echo ""
echo -e "${YELLOW}🌐 Access Points:${NC}"
echo "• Main Demo: http://localhost:3000"
echo "• Guided Demo: http://localhost:3000/demo/guided"
echo "• MCP Server API: http://localhost:8000/docs"
echo "• LM Studio: http://localhost:1234"
echo ""
echo -e "${CYAN}💡 Ready for Demo:${NC}"
echo "• Visit http://localhost:3000/demo/guided for step-by-step demo"
echo "• All services are running and ready for interaction"
echo "• No background automation - everything is user-controlled"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "🧹 Cleaning up demo processes..."
    
    if [ -f .demo-mcp-server.pid ]; then
        MCP_PID=$(cat .demo-mcp-server.pid)
        if ps -p $MCP_PID > /dev/null 2>&1; then
            kill $MCP_PID
            echo "Stopped MCP Server (PID: $MCP_PID)"
        fi
        rm -f .demo-mcp-server.pid
    fi
    
    if [ -f .demo-frontend.pid ]; then
        FRONTEND_PID=$(cat .demo-frontend.pid)
        if ps -p $FRONTEND_PID > /dev/null 2>&1; then
            kill $FRONTEND_PID
            echo "Stopped Frontend (PID: $FRONTEND_PID)"
        fi
        rm -f .demo-frontend.pid
    fi
    
    echo "Demo cleanup complete!"
}

# Set up cleanup on script exit
trap cleanup EXIT

echo "Press Ctrl+C to stop the demo and cleanup processes."
echo ""

echo "Press Ctrl+C to stop all services."
echo ""

# Keep the script running to maintain services
while true; do
    sleep 10
done