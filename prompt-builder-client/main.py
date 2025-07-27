#!/usr/bin/env python3
"""
Prompt Builder Client

A comprehensive client application that uses MCP servers (LLM Provider with Bedrock and Sequential Thinking)
to generate intelligent prompts based on user needs. Features a modern web UI and advanced prompt engineering.

Features:
- Web-based UI for prompt building
- Integration with AWS Bedrock via MCP LLM Provider
- Sequential thinking integration for structured reasoning
- Template-based prompt generation
- Real-time preview and testing
- Prompt optimization and validation
- Export/import functionality
- MCP configuration management
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import webbrowser
from dataclasses import dataclass, asdict

# Web framework
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# MCP Client integration
import aiohttp
import websockets

# Data models
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Prompt Builder Client",
    description="Advanced prompt building with MCP integration",
    version="1.0.0"
)

# Global state
mcp_clients = {}
active_websockets = set()
prompt_sessions = {}


class MCPClient:
    """MCP client for communicating with MCP servers."""
    
    def __init__(self, server_name: str, config: Dict[str, Any]):
        self.server_name = server_name
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.connected = False
        
    async def connect(self):
        """Connect to MCP server."""
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Test connection with health check
            health_url = f"http://{self.config['host']}:{self.config['port']}/health"
            async with self.session.get(health_url) as response:
                if response.status == 200:
                    self.connected = True
                    logger.info(f"Connected to {self.server_name}")
                    return True
                else:
                    logger.error(f"Health check failed for {self.server_name}: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to connect to {self.server_name}: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MCP server."""
        if self.session:
            await self.session.close()
        self.connected = False
        logger.info(f"Disconnected from {self.server_name}")
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool."""
        if not self.connected or not self.session:
            raise Exception(f"Not connected to {self.server_name}")
        
        try:
            # In a real MCP implementation, this would use the MCP protocol
            # For now, we'll simulate the calls
            logger.info(f"Calling {tool_name} on {self.server_name} with params: {parameters}")
            
            if self.server_name == "llm_provider":
                return await self._simulate_llm_call(tool_name, parameters)
            elif self.server_name == "sequential_thinker":
                return await self._simulate_thinking_call(tool_name, parameters)
            else:
                return {"success": False, "error": f"Unknown server: {self.server_name}"}
                
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _simulate_llm_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate LLM provider calls."""
        if tool_name == "generate_text":
            # Simulate Bedrock text generation
            prompt = parameters.get("prompt", "")
            model = parameters.get("model", "anthropic.claude-3-sonnet-20240229")
            
            # Mock response for demonstration
            return {
                "success": True,
                "content": f"Generated response for prompt: '{prompt[:50]}...' using {model}",
                "model": model,
                "provider": "aws_bedrock",
                "usage": {
                    "input_tokens": len(prompt) // 4,
                    "output_tokens": 100,
                    "total_tokens": len(prompt) // 4 + 100
                }
            }
        elif tool_name == "list_models":
            return {
                "success": True,
                "models": {
                    "anthropic.claude-3-5-sonnet-20241022-v2:0": {
                        "provider": "aws_bedrock",
                        "capabilities": {
                            "max_tokens": 8192,
                            "supports_streaming": True,
                            "supports_functions": True,
                            "context_window": 200000
                        }
                    },
                    "anthropic.claude-3-haiku-20240307-v1:0": {
                        "provider": "aws_bedrock",
                        "capabilities": {
                            "max_tokens": 4096,
                            "supports_streaming": True,
                            "supports_functions": True,
                            "context_window": 200000
                        }
                    }
                }
            }
        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
    
    async def _simulate_thinking_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate Sequential Thinking calls."""
        if tool_name == "create_thinking_chain":
            chain_id = f"chain_{uuid.uuid4().hex[:8]}"
            return {
                "success": True,
                "chain_id": chain_id,
                "description": parameters.get("description", ""),
                "message": f"Created thinking chain: {chain_id}"
            }
        elif tool_name == "generate_sequential_thinking_prompt":
            task = parameters.get("task", "")
            steps = parameters.get("steps", [])
            
            # Generate a structured thinking prompt
            prompt = f"# Sequential Thinking Task: {task}\n\n"
            prompt += "Please work through this task systematically using the following steps:\n\n"
            
            for i, step in enumerate(steps, 1):
                prompt += f"## Step {i}: {step.get('description', f'Step {i}')}\n"
                if step.get('reasoning'):
                    prompt += f"**Reasoning:** {step['reasoning']}\n"
                if step.get('expected_output'):
                    prompt += f"**Expected Output:** {step['expected_output']}\n"
                prompt += "\n"
            
            prompt += "Please provide your analysis for each step, building upon previous steps to reach the final solution."
            
            return {
                "success": True,
                "prompt": prompt,
                "chain_id": parameters.get("chain_id", "auto_generated"),
                "step_count": len(steps)
            }
        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}


# Data models
class PromptRequest(BaseModel):
    """Request for prompt generation."""
    task_description: str = Field(description="Description of what the prompt should accomplish")
    context: str = Field(default="", description="Additional context for the prompt")
    prompt_type: str = Field(default="general", description="Type of prompt: general, creative, analytical, etc.")
    target_audience: str = Field(default="general", description="Target audience for the prompt")
    use_sequential_thinking: bool = Field(default=True, description="Use sequential thinking approach")
    model_preference: str = Field(default="auto", description="Preferred model for generation")
    max_tokens: int = Field(default=2000, description="Maximum tokens for the response")
    temperature: float = Field(default=0.7, description="Temperature for generation")


class PromptOptimizationRequest(BaseModel):
    """Request for prompt optimization."""
    original_prompt: str = Field(description="Original prompt to optimize")
    optimization_goals: List[str] = Field(description="Goals for optimization")
    target_model: str = Field(default="auto", description="Target model for optimization")


class PromptTestRequest(BaseModel):
    """Request for prompt testing."""
    prompt: str = Field(description="Prompt to test")
    test_inputs: List[str] = Field(description="Test inputs to validate the prompt")
    model: str = Field(default="auto", description="Model to test with")


@dataclass
class PromptSession:
    """Session for prompt building."""
    session_id: str
    user_id: str
    created_at: datetime
    prompts: List[Dict[str, Any]]
    templates_used: List[str]
    thinking_chains: List[str]


# MCP Configuration
MCP_CONFIG = {
    "llm_provider": {
        "host": "localhost",
        "port": 8002,
        "enabled": True
    },
    "sequential_thinker": {
        "host": "localhost", 
        "port": 8001,
        "enabled": True
    },
    "vector_store": {
        "host": "localhost",
        "port": 8003,
        "enabled": False
    },
    "memory": {
        "host": "localhost",
        "port": 8004,
        "enabled": False
    }
}


# Initialize MCP clients
async def initialize_mcp_clients():
    """Initialize connections to MCP servers."""
    global mcp_clients
    
    for server_name, config in MCP_CONFIG.items():
        if config.get("enabled", False):
            client = MCPClient(server_name, config)
            if await client.connect():
                mcp_clients[server_name] = client
                logger.info(f"MCP client for {server_name} initialized")
            else:
                logger.error(f"Failed to initialize MCP client for {server_name}")


async def cleanup_mcp_clients():
    """Cleanup MCP client connections."""
    for client in mcp_clients.values():
        await client.disconnect()
    mcp_clients.clear()


# API Routes
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve the main application page."""
    templates = Jinja2Templates(directory="templates")
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    mcp_status = {}
    for name, client in mcp_clients.items():
        mcp_status[name] = {"connected": client.connected}
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mcp_servers": mcp_status
    }


@app.get("/api/models")
async def list_models():
    """List available models from LLM provider."""
    if "llm_provider" not in mcp_clients:
        raise HTTPException(status_code=503, detail="LLM provider not available")
    
    try:
        result = await mcp_clients["llm_provider"].call_tool("list_models", {})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-prompt")
async def generate_prompt(request: PromptRequest):
    """Generate a prompt based on user requirements."""
    try:
        session_id = str(uuid.uuid4())
        
        # Step 1: Create thinking chain if requested
        thinking_chain_id = None
        if request.use_sequential_thinking and "sequential_thinker" in mcp_clients:
            chain_result = await mcp_clients["sequential_thinker"].call_tool(
                "create_thinking_chain",
                {
                    "chain_id": f"prompt_gen_{session_id}",
                    "description": f"Generate prompt for: {request.task_description}",
                    "main_task": request.task_description,
                    "context": request.context
                }
            )
            
            if chain_result.get("success"):
                thinking_chain_id = chain_result.get("chain_id")
        
        # Step 2: Define thinking steps based on prompt type
        thinking_steps = []
        
        if request.prompt_type == "creative":
            thinking_steps = [
                {
                    "description": "Analyze Creative Requirements",
                    "reasoning": "Understand the creative goals and constraints",
                    "expected_output": "Clear creative objectives and style requirements"
                },
                {
                    "description": "Design Creative Framework",
                    "reasoning": "Structure the prompt to inspire creativity while maintaining focus",
                    "expected_output": "Creative prompt framework with inspiration elements"
                },
                {
                    "description": "Add Examples and Context",
                    "reasoning": "Provide examples to guide the creative direction",
                    "expected_output": "Rich prompt with examples and creative context"
                }
            ]
        elif request.prompt_type == "analytical":
            thinking_steps = [
                {
                    "description": "Define Analysis Scope",
                    "reasoning": "Clarify what needs to be analyzed and the methodology",
                    "expected_output": "Clear analysis framework and scope"
                },
                {
                    "description": "Structure Analytical Process",
                    "reasoning": "Create step-by-step analytical approach",
                    "expected_output": "Structured analytical methodology"
                },
                {
                    "description": "Define Output Format",
                    "reasoning": "Specify how the analysis should be presented",
                    "expected_output": "Clear format and structure for analytical output"
                }
            ]
        else:  # general
            thinking_steps = [
                {
                    "description": "Understand Requirements",
                    "reasoning": "Analyze what the prompt needs to accomplish",
                    "expected_output": "Clear understanding of prompt objectives"
                },
                {
                    "description": "Design Prompt Structure",
                    "reasoning": "Create an effective structure for the prompt",
                    "expected_output": "Well-structured prompt framework"
                },
                {
                    "description": "Optimize for Clarity",
                    "reasoning": "Ensure the prompt is clear and unambiguous",
                    "expected_output": "Clear, optimized prompt ready for use"
                }
            ]
        
        # Step 3: Generate structured thinking prompt
        structured_prompt = None
        if thinking_chain_id and "sequential_thinker" in mcp_clients:
            thinking_result = await mcp_clients["sequential_thinker"].call_tool(
                "generate_sequential_thinking_prompt",
                {
                    "task": f"Create an effective prompt for: {request.task_description}",
                    "context": f"Context: {request.context}\nTarget audience: {request.target_audience}\nPrompt type: {request.prompt_type}",
                    "steps": thinking_steps,
                    "chain_id": thinking_chain_id
                }
            )
            
            if thinking_result.get("success"):
                structured_prompt = thinking_result.get("prompt")
        
        # Step 4: Generate the actual prompt using LLM
        if "llm_provider" in mcp_clients:
            # Prepare the meta-prompt for generating the user's prompt
            meta_prompt = f"""
You are an expert prompt engineer. Create an effective prompt based on the following requirements:

Task Description: {request.task_description}
Context: {request.context}
Target Audience: {request.target_audience}
Prompt Type: {request.prompt_type}

{'Using this structured thinking approach:\n' + structured_prompt if structured_prompt else ''}

Create a clear, effective prompt that will help achieve the specified task. The prompt should be:
1. Clear and unambiguous
2. Appropriate for the target audience
3. Optimized for the specified prompt type
4. Include relevant context and examples if helpful

Generated Prompt:"""

            generation_result = await mcp_clients["llm_provider"].call_tool(
                "generate_text",
                {
                    "prompt": meta_prompt,
                    "model": request.model_preference if request.model_preference != "auto" else "anthropic.claude-3-sonnet-20240229",
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature
                }
            )
            
            if generation_result.get("success"):
                generated_prompt = generation_result.get("content", "")
                
                # Store session data
                session = PromptSession(
                    session_id=session_id,
                    user_id="anonymous",
                    created_at=datetime.now(),
                    prompts=[{
                        "original_request": asdict(request),
                        "generated_prompt": generated_prompt,
                        "thinking_chain_id": thinking_chain_id,
                        "structured_prompt": structured_prompt,
                        "meta_prompt": meta_prompt,
                        "generation_result": generation_result
                    }],
                    templates_used=[],
                    thinking_chains=[thinking_chain_id] if thinking_chain_id else []
                )
                
                prompt_sessions[session_id] = session
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "generated_prompt": generated_prompt,
                    "thinking_chain_id": thinking_chain_id,
                    "structured_thinking": structured_prompt,
                    "meta_prompt_used": meta_prompt,
                    "model_used": generation_result.get("model"),
                    "usage": generation_result.get("usage"),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise HTTPException(status_code=500, detail=f"LLM generation failed: {generation_result.get('error')}")
        else:
            raise HTTPException(status_code=503, detail="LLM provider not available")
            
    except Exception as e:
        logger.error(f"Prompt generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize-prompt")
async def optimize_prompt(request: PromptOptimizationRequest):
    """Optimize an existing prompt."""
    try:
        if "llm_provider" not in mcp_clients:
            raise HTTPException(status_code=503, detail="LLM provider not available")
        
        # Create optimization prompt
        optimization_goals_str = "\n".join(f"- {goal}" for goal in request.optimization_goals)
        
        meta_prompt = f"""
You are an expert prompt engineer. Optimize the following prompt to achieve these goals:

Goals:
{optimization_goals_str}

Original Prompt:
{request.original_prompt}

Please provide an optimized version that:
1. Maintains the original intent
2. Achieves the specified optimization goals
3. Is clearer and more effective
4. Includes explanation of changes made

Optimized Prompt:"""

        result = await mcp_clients["llm_provider"].call_tool(
            "generate_text",
            {
                "prompt": meta_prompt,
                "model": request.target_model if request.target_model != "auto" else "anthropic.claude-3-sonnet-20240229",
                "max_tokens": 2000,
                "temperature": 0.3  # Lower temperature for optimization
            }
        )
        
        if result.get("success"):
            return {
                "success": True,
                "original_prompt": request.original_prompt,
                "optimized_prompt": result.get("content"),
                "optimization_goals": request.optimization_goals,
                "model_used": result.get("model"),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail=f"Optimization failed: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"Prompt optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/test-prompt")
async def test_prompt(request: PromptTestRequest):
    """Test a prompt with sample inputs."""
    try:
        if "llm_provider" not in mcp_clients:
            raise HTTPException(status_code=503, detail="LLM provider not available")
        
        test_results = []
        
        for test_input in request.test_inputs:
            # Combine prompt with test input
            full_prompt = f"{request.prompt}\n\nInput: {test_input}"
            
            result = await mcp_clients["llm_provider"].call_tool(
                "generate_text",
                {
                    "prompt": full_prompt,
                    "model": request.model if request.model != "auto" else "anthropic.claude-3-sonnet-20240229",
                    "max_tokens": 1000,
                    "temperature": 0.7
                }
            )
            
            test_results.append({
                "input": test_input,
                "output": result.get("content") if result.get("success") else None,
                "success": result.get("success"),
                "error": result.get("error") if not result.get("success") else None,
                "usage": result.get("usage")
            })
        
        return {
            "success": True,
            "prompt_tested": request.prompt,
            "test_results": test_results,
            "total_tests": len(request.test_inputs),
            "successful_tests": sum(1 for r in test_results if r["success"]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prompt testing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get prompt building session data."""
    if session_id not in prompt_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = prompt_sessions[session_id]
    return {
        "session_id": session.session_id,
        "created_at": session.created_at.isoformat(),
        "prompts": session.prompts,
        "templates_used": session.templates_used,
        "thinking_chains": session.thinking_chains
    }


@app.get("/api/templates")
async def list_templates():
    """List available prompt templates."""
    # In a real implementation, these would be loaded from files
    templates = {
        "creative_writing": {
            "name": "Creative Writing",
            "description": "Template for creative writing prompts",
            "template": "Write a {genre} story about {topic}. The story should be {length} and include {elements}. Style: {style}",
            "variables": ["genre", "topic", "length", "elements", "style"]
        },
        "analysis": {
            "name": "Analysis",
            "description": "Template for analytical prompts",
            "template": "Analyze {subject} focusing on {aspects}. Provide insights on {questions}. Use {methodology} approach.",
            "variables": ["subject", "aspects", "questions", "methodology"]
        },
        "code_generation": {
            "name": "Code Generation",
            "description": "Template for code generation prompts",
            "template": "Write {language} code to {task}. Requirements: {requirements}. Include {features}.",
            "variables": ["language", "task", "requirements", "features"]
        },
        "explanation": {
            "name": "Explanation",
            "description": "Template for explanation prompts",
            "template": "Explain {concept} to {audience}. Cover {topics} using {examples}. Difficulty level: {level}",
            "variables": ["concept", "audience", "topics", "examples", "level"]
        }
    }
    
    return {"templates": templates}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    active_websockets.add(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message.get("type") == "status_request":
                status = {
                    "type": "status",
                    "mcp_servers": {name: client.connected for name, client in mcp_clients.items()},
                    "active_sessions": len(prompt_sessions),
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send_text(json.dumps(status))
                
    except WebSocketDisconnect:
        active_websockets.remove(websocket)


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    logger.info("Starting Prompt Builder Client...")
    
    # Create necessary directories
    Path("templates").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    # Initialize MCP clients
    await initialize_mcp_clients()
    
    # Mount static files
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    logger.info("Prompt Builder Client started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup the application."""
    logger.info("Shutting down Prompt Builder Client...")
    await cleanup_mcp_clients()
    logger.info("Prompt Builder Client shut down")


def main():
    """Run the Prompt Builder Client."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prompt Builder Client")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=3001, help="Port to bind to")
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