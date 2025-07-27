#!/usr/bin/env python3
"""
AI Agent Creation Demo Client

A comprehensive demo that showcases the complete workflow of creating AI agents
using the MCP ecosystem. This client demonstrates:

1. JSON-based configuration
2. Step-by-step agent creation process
3. Memory configuration and management
4. Sequential thinking setup
5. Prompt building and optimization
6. Final AI agent deployment and testing

The demo provides an interactive interface showing how to build AI agents
from initial concept to deployment using the MCP server ecosystem.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import webbrowser

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Agent Creation Demo",
    description="Interactive demo for creating AI agents using MCP ecosystem",
    version="1.0.0"
)

# Global state
demo_sessions = {}
mcp_clients = {}
active_websockets = set()


class MCPClient:
    """Client for communicating with MCP servers."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.connected = False
    
    async def connect(self) -> bool:
        """Connect to MCP server."""
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            health_url = f"http://{self.config['host']}:{self.config['port']}/health"
            async with self.session.get(health_url) as response:
                if response.status == 200:
                    self.connected = True
                    logger.info(f"Connected to {self.name}")
                    return True
                else:
                    logger.error(f"Health check failed for {self.name}: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Failed to connect to {self.name}: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MCP server."""
        if self.session:
            await self.session.close()
        self.connected = False
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server."""
        if not self.connected:
            return {"success": False, "error": f"Not connected to {self.name}"}
        
        try:
            # Simulate MCP tool call for demo purposes
            logger.info(f"Calling {tool_name} on {self.name} with params: {list(parameters.keys())}")
            
            # Simulate different server responses
            if self.name == "prompt-builder":
                return await self._simulate_prompt_builder_call(tool_name, parameters)
            else:
                return {"success": True, "message": f"Tool {tool_name} called successfully on {self.name}"}
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _simulate_prompt_builder_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate Prompt Builder MCP server calls."""
        await asyncio.sleep(0.5)  # Simulate processing time
        
        if tool_name == "create_prompt_building_session":
            session_id = f"demo_session_{len(demo_sessions) + 1}"
            return {
                "success": True,
                "session_id": session_id,
                "name": parameters.get("session_name", "Demo Session"),
                "current_step": "initialization",
                "next_steps": ["define_context", "configure_memory", "configure_thinking", "build_prompt", "create_agent"]
            }
        
        elif tool_name == "define_agent_context":
            return {
                "success": True,
                "session_id": parameters.get("session_id"),
                "current_step": "context_defined",
                "context_data": {
                    "agent_name": parameters.get("agent_name"),
                    "agent_description": parameters.get("agent_description"),
                    "domain": parameters.get("domain"),
                    "use_case": parameters.get("use_case"),
                    "target_audience": parameters.get("target_audience")
                },
                "next_step": "configure_memory"
            }
        
        elif tool_name == "configure_memory_settings":
            return {
                "success": True,
                "session_id": parameters.get("session_id"),
                "current_step": "memory_configured",
                "memory_config": {
                    "memory_type": parameters.get("memory_type", "conversation"),
                    "persistence": parameters.get("persistence", True),
                    "memory_session_id": f"mem_session_{hash(str(parameters)) % 1000}"
                },
                "next_step": "configure_thinking"
            }
        
        elif tool_name == "configure_sequential_thinking":
            return {
                "success": True,
                "session_id": parameters.get("session_id"),
                "current_step": "thinking_configured",
                "thinking_config": {
                    "thinking_style": parameters.get("thinking_style", "analytical"),
                    "enable_chain_of_thought": parameters.get("enable_chain_of_thought", True),
                    "thinking_chain_id": f"thinking_chain_{hash(str(parameters)) % 1000}"
                },
                "next_step": "build_prompt"
            }
        
        elif tool_name == "build_agent_prompt":
            agent_prompt = f"""# AI Agent: Customer Service Assistant

## Role and Purpose
You are a Customer Service Assistant, designed to help customers with inquiries, complaints, and support requests.

## Domain Expertise
- **Primary Domain**: Customer Service
- **Use Case**: Support and assistance
- **Target Audience**: General customers

## Behavioral Guidelines
1. Always maintain a helpful and professional tone
2. Provide accurate and relevant information
3. Ask clarifying questions when needed
4. Escalate complex issues appropriately

## Memory and Context Management
- **Memory Type**: conversation
- **Persistence**: Enabled
- **Semantic Search**: Enabled

## Reasoning Approach
- **Thinking Style**: empathetic
- **Chain of Thought**: Enabled
- **Reasoning Depth**: detailed

---

**Instructions**: Follow the above guidelines in all customer interactions. Use your memory and reasoning capabilities to provide excellent customer service."""
            
            return {
                "success": True,
                "session_id": parameters.get("session_id"),
                "current_step": "prompt_built",
                "agent_id": f"agent_{hash(str(parameters)) % 1000}",
                "agent_prompt": agent_prompt,
                "next_step": "create_agent"
            }
        
        elif tool_name == "create_ai_agent":
            test_response = """Hello! I'm your Customer Service Assistant, and I'm here to help you with any questions, concerns, or support needs you may have. 

I specialize in:
- Answering product and service questions
- Helping resolve issues and complaints
- Providing guidance on policies and procedures
- Connecting you with the right resources

I'm designed to listen carefully, understand your needs, and provide helpful, accurate information. Please feel free to share what brings you here today, and I'll do my best to assist you!"""
            
            return {
                "success": True,
                "session_id": parameters.get("session_id"),
                "current_step": "agent_created",
                "agent_id": f"agent_{hash(str(parameters)) % 1000}",
                "agent_name": "Customer Service Assistant",
                "test_message": parameters.get("test_message"),
                "test_response": test_response,
                "agent_ready": True,
                "usage_instructions": {
                    "deployment": "The agent is ready for deployment",
                    "integration": "Use the agent_id to integrate with your application",
                    "customization": "The prompt can be further customized based on feedback"
                }
            }
        
        elif tool_name == "get_session_status":
            return {
                "success": True,
                "session_id": parameters.get("session_id"),
                "current_step": "initialization",
                "message": "Session status retrieved"
            }
        
        elif tool_name == "get_mcp_servers_status":
            return {
                "success": True,
                "servers": {
                    "llm-provider": {"connected": True, "host": "localhost", "port": 8002},
                    "sequential-thinker": {"connected": True, "host": "localhost", "port": 8001},
                    "memory": {"connected": True, "host": "localhost", "port": 8004},
                    "vector-store": {"connected": False, "host": "localhost", "port": 8003}
                },
                "total_servers": 4,
                "connected_servers": 3
            }
        
        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}


class DemoSession:
    """Represents a demo session for creating an AI agent."""
    
    def __init__(self, session_id: str, config: Dict[str, Any]):
        self.session_id = session_id
        self.config = config
        self.current_step = "start"
        self.steps_completed = []
        self.agent_data = {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def update_step(self, step: str, data: Dict[str, Any] = None):
        """Update current step and data."""
        if self.current_step not in self.steps_completed:
            self.steps_completed.append(self.current_step)
        self.current_step = step
        if data:
            self.agent_data.update(data)
        self.updated_at = datetime.now()


async def initialize_mcp_clients():
    """Initialize MCP client connections."""
    config_path = Path(__file__).parent / "demo-config.json"
    
    if not config_path.exists():
        logger.error("Demo config file not found")
        return
    
    with open(config_path) as f:
        config = json.load(f)
    
    for server_name, server_config in config.get("mcp_servers", {}).items():
        if server_config.get("enabled", False):
            client = MCPClient(server_name, server_config)
            if await client.connect():
                mcp_clients[server_name] = client
                logger.info(f"Connected to {server_name}")


# API Routes
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve the main demo page."""
    templates = Jinja2Templates(directory="templates")
    return templates.TemplateResponse("demo.html", {"request": request})


@app.get("/api/demo/config")
async def get_demo_config():
    """Get demo configuration and available options."""
    config_path = Path(__file__).parent / "demo-config.json"
    
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Demo config not found")
    
    with open(config_path) as f:
        config = json.load(f)
    
    return {
        "success": True,
        "config": config,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/demo/start")
async def start_demo_session(request: dict):
    """Start a new demo session."""
    session_name = request.get("session_name", f"Demo Session {len(demo_sessions) + 1}")
    
    # Create session with Prompt Builder
    if "prompt-builder" in mcp_clients:
        result = await mcp_clients["prompt-builder"].call_tool(
            "create_prompt_building_session",
            {"session_name": session_name, "description": "Interactive demo session"}
        )
        
        if result.get("success"):
            session_id = result.get("session_id")
            demo_session = DemoSession(session_id, request)
            demo_session.update_step("session_created", {"prompt_builder_session": session_id})
            demo_sessions[session_id] = demo_session
            
            return {
                "success": True,
                "session_id": session_id,
                "session_name": session_name,
                "current_step": "session_created",
                "next_step": "define_context",
                "message": "Demo session started successfully"
            }
    
    raise HTTPException(status_code=503, detail="Prompt Builder MCP server not available")


@app.post("/api/demo/step/context")
async def define_context_step(request: dict):
    """Step 1: Define agent context and requirements."""
    session_id = request.get("session_id")
    
    if session_id not in demo_sessions:
        raise HTTPException(status_code=404, detail="Demo session not found")
    
    session = demo_sessions[session_id]
    
    if "prompt-builder" in mcp_clients:
        result = await mcp_clients["prompt-builder"].call_tool(
            "define_agent_context",
            {
                "session_id": session_id,
                "agent_name": request.get("agent_name"),
                "agent_description": request.get("agent_description"),
                "domain": request.get("domain"),
                "use_case": request.get("use_case"),
                "target_audience": request.get("target_audience", "general"),
                "additional_context": request.get("additional_context", "")
            }
        )
        
        if result.get("success"):
            session.update_step("context_defined", {"context_result": result})
            
            return {
                "success": True,
                "session_id": session_id,
                "current_step": "context_defined",
                "context_data": result.get("context_data"),
                "next_step": "configure_memory",
                "message": "Agent context defined successfully"
            }
    
    raise HTTPException(status_code=503, detail="Failed to define context")


@app.post("/api/demo/step/memory")
async def configure_memory_step(request: dict):
    """Step 2: Configure memory settings."""
    session_id = request.get("session_id")
    
    if session_id not in demo_sessions:
        raise HTTPException(status_code=404, detail="Demo session not found")
    
    session = demo_sessions[session_id]
    
    if "prompt-builder" in mcp_clients:
        result = await mcp_clients["prompt-builder"].call_tool(
            "configure_memory_settings",
            {
                "session_id": session_id,
                "memory_type": request.get("memory_type", "conversation"),
                "persistence": request.get("persistence", True),
                "memory_size_limit": request.get("memory_size_limit", 10000),
                "retention_policy": request.get("retention_policy", "fifo"),
                "enable_semantic_search": request.get("enable_semantic_search", True)
            }
        )
        
        if result.get("success"):
            session.update_step("memory_configured", {"memory_result": result})
            
            return {
                "success": True,
                "session_id": session_id,
                "current_step": "memory_configured",
                "memory_config": result.get("memory_config"),
                "next_step": "configure_thinking",
                "message": "Memory configuration completed"
            }
    
    raise HTTPException(status_code=503, detail="Failed to configure memory")


@app.post("/api/demo/step/thinking")
async def configure_thinking_step(request: dict):
    """Step 3: Configure sequential thinking."""
    session_id = request.get("session_id")
    
    if session_id not in demo_sessions:
        raise HTTPException(status_code=404, detail="Demo session not found")
    
    session = demo_sessions[session_id]
    
    if "prompt-builder" in mcp_clients:
        result = await mcp_clients["prompt-builder"].call_tool(
            "configure_sequential_thinking",
            {
                "session_id": session_id,
                "thinking_style": request.get("thinking_style", "analytical"),
                "enable_chain_of_thought": request.get("enable_chain_of_thought", True),
                "thinking_depth": request.get("thinking_depth", "detailed"),
                "reasoning_steps": request.get("reasoning_steps", 5),
                "enable_self_reflection": request.get("enable_self_reflection", True)
            }
        )
        
        if result.get("success"):
            session.update_step("thinking_configured", {"thinking_result": result})
            
            return {
                "success": True,
                "session_id": session_id,
                "current_step": "thinking_configured",
                "thinking_config": result.get("thinking_config"),
                "next_step": "build_prompt",
                "message": "Sequential thinking configuration completed"
            }
    
    raise HTTPException(status_code=503, detail="Failed to configure thinking")


@app.post("/api/demo/step/prompt")
async def build_prompt_step(request: dict):
    """Step 4: Build the agent prompt."""
    session_id = request.get("session_id")
    
    if session_id not in demo_sessions:
        raise HTTPException(status_code=404, detail="Demo session not found")
    
    session = demo_sessions[session_id]
    
    if "prompt-builder" in mcp_clients:
        result = await mcp_clients["prompt-builder"].call_tool(
            "build_agent_prompt",
            {
                "session_id": session_id,
                "prompt_type": request.get("prompt_type", "conversational"),
                "include_examples": request.get("include_examples", True),
                "optimization_goals": request.get("optimization_goals", ["clarity", "accuracy"])
            }
        )
        
        if result.get("success"):
            session.update_step("prompt_built", {"prompt_result": result})
            
            return {
                "success": True,
                "session_id": session_id,
                "current_step": "prompt_built",
                "agent_id": result.get("agent_id"),
                "agent_prompt": result.get("agent_prompt"),
                "next_step": "create_agent",
                "message": "Agent prompt built successfully"
            }
    
    raise HTTPException(status_code=503, detail="Failed to build prompt")


@app.post("/api/demo/step/agent")
async def create_agent_step(request: dict):
    """Step 5: Create and test the AI agent."""
    session_id = request.get("session_id")
    
    if session_id not in demo_sessions:
        raise HTTPException(status_code=404, detail="Demo session not found")
    
    session = demo_sessions[session_id]
    
    if "prompt-builder" in mcp_clients:
        result = await mcp_clients["prompt-builder"].call_tool(
            "create_ai_agent",
            {
                "session_id": session_id,
                "test_message": request.get("test_message", "Hello! Please introduce yourself and explain what you can help me with.")
            }
        )
        
        if result.get("success"):
            session.update_step("agent_created", {"agent_result": result})
            
            return {
                "success": True,
                "session_id": session_id,
                "current_step": "agent_created",
                "agent_id": result.get("agent_id"),
                "agent_name": result.get("agent_name"),
                "test_message": result.get("test_message"),
                "test_response": result.get("test_response"),
                "agent_ready": result.get("agent_ready"),
                "usage_instructions": result.get("usage_instructions"),
                "message": "AI Agent created and ready for use!"
            }
    
    raise HTTPException(status_code=503, detail="Failed to create agent")


@app.get("/api/demo/session/{session_id}")
async def get_session_status(session_id: str):
    """Get current session status and progress."""
    if session_id not in demo_sessions:
        raise HTTPException(status_code=404, detail="Demo session not found")
    
    session = demo_sessions[session_id]
    
    return {
        "success": True,
        "session_id": session_id,
        "current_step": session.current_step,
        "steps_completed": session.steps_completed,
        "agent_data": session.agent_data,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat()
    }


@app.get("/api/demo/health")
async def demo_health_check():
    """Check health of demo system and MCP servers."""
    mcp_status = {}
    
    for name, client in mcp_clients.items():
        mcp_status[name] = {
            "connected": client.connected,
            "host": client.config["host"],
            "port": client.config["port"]
        }
    
    return {
        "success": True,
        "demo_status": "healthy",
        "active_sessions": len(demo_sessions),
        "mcp_servers": mcp_status,
        "timestamp": datetime.now().isoformat()
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time demo updates."""
    await websocket.accept()
    active_websockets.add(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message.get("type") == "status_request":
                status = {
                    "type": "status",
                    "active_sessions": len(demo_sessions),
                    "mcp_servers": {name: client.connected for name, client in mcp_clients.items()},
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send_text(json.dumps(status))
                
    except WebSocketDisconnect:
        active_websockets.remove(websocket)


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the demo application."""
    logger.info("Starting AI Agent Creation Demo...")
    
    # Create necessary directories
    Path("templates").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    
    # Initialize MCP clients
    await initialize_mcp_clients()
    
    # Mount static files
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    logger.info("AI Agent Creation Demo started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down AI Agent Creation Demo...")
    
    for client in mcp_clients.values():
        await client.disconnect()
    
    logger.info("AI Agent Creation Demo shut down")


def main():
    """Run the demo application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Agent Creation Demo")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=3002, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--open-browser", action="store_true", help="Open browser automatically")
    
    args = parser.parse_args()
    
    if args.open_browser:
        webbrowser.open(f"http://{args.host}:{args.port}")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()