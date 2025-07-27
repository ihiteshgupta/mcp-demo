#!/usr/bin/env python3
"""
Prompt Builder MCP Server

An MCP server that provides intelligent prompt building capabilities by orchestrating
other MCP servers (LLM Provider, Sequential Thinking, Memory, Vector Store).

This server acts as a high-level interface for creating AI agents through structured
prompt engineering workflows.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path

import aiohttp
from fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Prompt Builder")

# Global state
mcp_clients = {}
agent_configurations = {}
prompt_sessions = {}


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    host: str
    port: int
    enabled: bool = True
    description: str = ""
    tools: List[str] = None


@dataclass
class AgentConfiguration:
    """Configuration for an AI agent."""
    agent_id: str
    name: str
    description: str
    prompt_template: str
    memory_config: Dict[str, Any]
    thinking_config: Dict[str, Any]
    llm_config: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class PromptBuildingSession:
    """Session for building prompts and creating agents."""
    session_id: str
    current_step: str
    agent_config: Optional[AgentConfiguration]
    context_data: Dict[str, Any]
    prompt_history: List[Dict[str, Any]]
    created_at: datetime


class MCPClient:
    """Client for communicating with other MCP servers."""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.connected = False
        
    async def connect(self) -> bool:
        """Connect to the MCP server."""
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Test connection
            health_url = f"http://{self.config.host}:{self.config.port}/health"
            async with self.session.get(health_url) as response:
                if response.status == 200:
                    self.connected = True
                    logger.info(f"Connected to {self.config.name}")
                    return True
                else:
                    logger.error(f"Health check failed for {self.config.name}: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to connect to {self.config.name}: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self.session:
            await self.session.close()
        self.connected = False
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server."""
        if not self.connected or not self.session:
            raise Exception(f"Not connected to {self.config.name}")
        
        try:
            # Simulate MCP tool call
            logger.info(f"Calling {tool_name} on {self.config.name}")
            
            # In a real implementation, this would use proper MCP protocol
            if self.config.name == "llm-provider":
                return await self._simulate_llm_call(tool_name, parameters)
            elif self.config.name == "sequential-thinker":
                return await self._simulate_thinking_call(tool_name, parameters)
            elif self.config.name == "memory":
                return await self._simulate_memory_call(tool_name, parameters)
            elif self.config.name == "vector-store":
                return await self._simulate_vector_call(tool_name, parameters)
            else:
                return {"success": False, "error": f"Unknown server: {self.config.name}"}
                
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _simulate_llm_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate LLM provider calls."""
        if tool_name == "generate_text":
            prompt = parameters.get("prompt", "")
            model = parameters.get("model", "anthropic.claude-3-5-sonnet-20241022-v2:0")
            
            return {
                "success": True,
                "content": f"[Generated response for: {prompt[:50]}...]",
                "model": model,
                "usage": {"input_tokens": 100, "output_tokens": 150, "total_tokens": 250}
            }
        return {"success": False, "error": f"Unknown LLM tool: {tool_name}"}
    
    async def _simulate_thinking_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate sequential thinking calls."""
        if tool_name == "create_thinking_chain":
            return {
                "success": True,
                "chain_id": f"chain_{hash(str(parameters)) % 10000}",
                "description": parameters.get("description", "")
            }
        elif tool_name == "generate_sequential_thinking_prompt":
            return {
                "success": True,
                "prompt": f"Sequential thinking prompt for: {parameters.get('task', 'unknown task')}",
                "chain_id": parameters.get("chain_id", "auto")
            }
        return {"success": False, "error": f"Unknown thinking tool: {tool_name}"}
    
    async def _simulate_memory_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate memory management calls."""
        if tool_name == "create_session":
            return {
                "success": True,
                "session_id": f"mem_{hash(str(parameters)) % 10000}",
                "type": parameters.get("session_type", "conversation")
            }
        elif tool_name == "store_memory":
            return {
                "success": True,
                "memory_id": f"memory_{hash(str(parameters)) % 10000}",
                "stored_at": datetime.now().isoformat()
            }
        return {"success": False, "error": f"Unknown memory tool: {tool_name}"}
    
    async def _simulate_vector_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate vector store calls."""
        if tool_name == "create_collection":
            return {
                "success": True,
                "collection_id": f"vec_{hash(str(parameters)) % 10000}",
                "name": parameters.get("name", "default")
            }
        return {"success": False, "error": f"Unknown vector tool: {tool_name}"}


async def initialize_mcp_clients():
    """Initialize connections to MCP servers."""
    config_path = Path(__file__).parent / "config.json"
    
    if not config_path.exists():
        logger.error("Config file not found")
        return
    
    with open(config_path) as f:
        config = json.load(f)
    
    for server_name, server_config in config.get("mcp_servers", {}).items():
        if server_config.get("enabled", False):
            mcp_config = MCPServerConfig(
                name=server_name,
                host=server_config["host"],
                port=server_config["port"],
                enabled=server_config["enabled"],
                description=server_config.get("description", ""),
                tools=server_config.get("tools", [])
            )
            
            client = MCPClient(mcp_config)
            if await client.connect():
                mcp_clients[server_name] = client
                logger.info(f"MCP client for {server_name} initialized")


# MCP Tools
@mcp.tool()
async def create_prompt_building_session(
    session_name: str,
    description: str = ""
) -> Dict[str, Any]:
    """Create a new prompt building session."""
    session_id = f"session_{len(prompt_sessions) + 1}_{hash(session_name) % 1000}"
    
    session = PromptBuildingSession(
        session_id=session_id,
        current_step="initialization",
        agent_config=None,
        context_data={},
        prompt_history=[],
        created_at=datetime.now()
    )
    
    prompt_sessions[session_id] = session
    
    return {
        "success": True,
        "session_id": session_id,
        "name": session_name,
        "description": description,
        "current_step": "initialization",
        "next_steps": ["define_context", "configure_memory", "configure_thinking", "build_prompt", "create_agent"],
        "message": f"Created prompt building session: {session_name}"
    }


@mcp.tool()
async def define_agent_context(
    session_id: str,
    agent_name: str,
    agent_description: str,
    domain: str,
    use_case: str,
    target_audience: str = "general",
    additional_context: str = ""
) -> Dict[str, Any]:
    """Define the context and requirements for the AI agent."""
    if session_id not in prompt_sessions:
        return {"success": False, "error": "Session not found"}
    
    session = prompt_sessions[session_id]
    session.current_step = "context_defined"
    session.context_data = {
        "agent_name": agent_name,
        "agent_description": agent_description,
        "domain": domain,
        "use_case": use_case,
        "target_audience": target_audience,
        "additional_context": additional_context,
        "defined_at": datetime.now().isoformat()
    }
    
    return {
        "success": True,
        "session_id": session_id,
        "current_step": "context_defined",
        "context_data": session.context_data,
        "next_step": "configure_memory",
        "message": f"Context defined for agent: {agent_name}"
    }


@mcp.tool()
async def configure_memory_settings(
    session_id: str,
    memory_type: str = "conversation",
    persistence: bool = True,
    memory_size_limit: int = 10000,
    retention_policy: str = "fifo",
    enable_semantic_search: bool = True
) -> Dict[str, Any]:
    """Configure memory settings for the AI agent."""
    if session_id not in prompt_sessions:
        return {"success": False, "error": "Session not found"}
    
    session = prompt_sessions[session_id]
    
    # Create memory session if memory client is available
    memory_session_id = None
    if "memory" in mcp_clients:
        memory_result = await mcp_clients["memory"].call_tool(
            "create_session",
            {
                "session_type": memory_type,
                "config": {
                    "persistence": persistence,
                    "size_limit": memory_size_limit,
                    "retention_policy": retention_policy
                }
            }
        )
        if memory_result.get("success"):
            memory_session_id = memory_result.get("session_id")
    
    memory_config = {
        "memory_type": memory_type,
        "persistence": persistence,
        "memory_size_limit": memory_size_limit,
        "retention_policy": retention_policy,
        "enable_semantic_search": enable_semantic_search,
        "memory_session_id": memory_session_id,
        "configured_at": datetime.now().isoformat()
    }
    
    session.current_step = "memory_configured"
    if not session.context_data:
        session.context_data = {}
    session.context_data["memory_config"] = memory_config
    
    return {
        "success": True,
        "session_id": session_id,
        "current_step": "memory_configured",
        "memory_config": memory_config,
        "memory_session_id": memory_session_id,
        "next_step": "configure_thinking",
        "message": "Memory configuration completed"
    }


@mcp.tool()
async def configure_sequential_thinking(
    session_id: str,
    thinking_style: str = "analytical",
    enable_chain_of_thought: bool = True,
    thinking_depth: str = "detailed",
    reasoning_steps: int = 5,
    enable_self_reflection: bool = True
) -> Dict[str, Any]:
    """Configure sequential thinking settings for the AI agent."""
    if session_id not in prompt_sessions:
        return {"success": False, "error": "Session not found"}
    
    session = prompt_sessions[session_id]
    
    # Create thinking chain if sequential thinker is available
    thinking_chain_id = None
    if "sequential-thinker" in mcp_clients:
        chain_result = await mcp_clients["sequential-thinker"].call_tool(
            "create_thinking_chain",
            {
                "description": f"Thinking chain for {session.context_data.get('agent_name', 'unknown agent')}",
                "style": thinking_style,
                "depth": thinking_depth
            }
        )
        if chain_result.get("success"):
            thinking_chain_id = chain_result.get("chain_id")
    
    thinking_config = {
        "thinking_style": thinking_style,
        "enable_chain_of_thought": enable_chain_of_thought,
        "thinking_depth": thinking_depth,
        "reasoning_steps": reasoning_steps,
        "enable_self_reflection": enable_self_reflection,
        "thinking_chain_id": thinking_chain_id,
        "configured_at": datetime.now().isoformat()
    }
    
    session.current_step = "thinking_configured"
    if not session.context_data:
        session.context_data = {}
    session.context_data["thinking_config"] = thinking_config
    
    return {
        "success": True,
        "session_id": session_id,
        "current_step": "thinking_configured",
        "thinking_config": thinking_config,
        "thinking_chain_id": thinking_chain_id,
        "next_step": "build_prompt",
        "message": "Sequential thinking configuration completed"
    }


@mcp.tool()
async def build_agent_prompt(
    session_id: str,
    prompt_type: str = "conversational",
    include_examples: bool = True,
    optimization_goals: List[str] = None
) -> Dict[str, Any]:
    """Build the final prompt for the AI agent using all configured settings."""
    if session_id not in prompt_sessions:
        return {"success": False, "error": "Session not found"}
    
    session = prompt_sessions[session_id]
    context = session.context_data
    
    if not context:
        return {"success": False, "error": "No context defined for session"}
    
    # Generate structured thinking prompt if available
    structured_prompt = None
    if "sequential-thinker" in mcp_clients and context.get("thinking_config"):
        thinking_result = await mcp_clients["sequential-thinker"].call_tool(
            "generate_sequential_thinking_prompt",
            {
                "task": f"Create an AI agent for {context.get('use_case', 'general purpose')}",
                "context": f"Domain: {context.get('domain', 'general')}, Audience: {context.get('target_audience', 'general')}",
                "chain_id": context.get("thinking_config", {}).get("thinking_chain_id")
            }
        )
        if thinking_result.get("success"):
            structured_prompt = thinking_result.get("prompt")
    
    # Build the comprehensive agent prompt
    agent_prompt = f"""# AI Agent: {context.get('agent_name', 'Assistant')}

## Role and Purpose
You are {context.get('agent_name', 'an AI assistant')}, {context.get('agent_description', 'designed to help users')}.

## Domain Expertise
- **Primary Domain**: {context.get('domain', 'General')}
- **Use Case**: {context.get('use_case', 'General assistance')}
- **Target Audience**: {context.get('target_audience', 'General users')}

## Behavioral Guidelines
1. Always maintain a helpful and professional tone
2. Provide accurate and relevant information within your domain
3. Ask clarifying questions when needed
4. Acknowledge limitations and suggest alternatives when appropriate

## Memory and Context Management
- **Memory Type**: {context.get('memory_config', {}).get('memory_type', 'conversation')}
- **Persistence**: {'Enabled' if context.get('memory_config', {}).get('persistence') else 'Disabled'}
- **Semantic Search**: {'Enabled' if context.get('memory_config', {}).get('enable_semantic_search') else 'Disabled'}

## Reasoning Approach
- **Thinking Style**: {context.get('thinking_config', {}).get('thinking_style', 'analytical')}
- **Chain of Thought**: {'Enabled' if context.get('thinking_config', {}).get('enable_chain_of_thought') else 'Disabled'}
- **Reasoning Depth**: {context.get('thinking_config', {}).get('thinking_depth', 'standard')}

{f'## Structured Thinking Framework\\n{structured_prompt}' if structured_prompt else ''}

## Additional Context
{context.get('additional_context', 'No additional context provided.')}

---

**Instructions**: Follow the above guidelines in all interactions. Use your memory and reasoning capabilities to provide the best possible assistance."""
    
    # Generate final prompt using LLM if available
    final_prompt = agent_prompt
    if "llm-provider" in mcp_clients:
        generation_result = await mcp_clients["llm-provider"].call_tool(
            "generate_text",
            {
                "prompt": f"Optimize and enhance this AI agent prompt for clarity and effectiveness:\\n\\n{agent_prompt}",
                "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "max_tokens": 2000,
                "temperature": 0.3
            }
        )
        if generation_result.get("success"):
            final_prompt = generation_result.get("content", agent_prompt)
    
    # Create agent configuration
    agent_config = AgentConfiguration(
        agent_id=f"agent_{hash(context.get('agent_name', 'default')) % 10000}",
        name=context.get('agent_name', 'AI Assistant'),
        description=context.get('agent_description', ''),
        prompt_template=final_prompt,
        memory_config=context.get('memory_config', {}),
        thinking_config=context.get('thinking_config', {}),
        llm_config={
            "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "temperature": 0.7,
            "max_tokens": 2000
        },
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    session.agent_config = agent_config
    session.current_step = "prompt_built"
    session.prompt_history.append({
        "step": "build_agent_prompt",
        "prompt": final_prompt,
        "timestamp": datetime.now().isoformat()
    })
    
    agent_configurations[agent_config.agent_id] = agent_config
    
    return {
        "success": True,
        "session_id": session_id,
        "current_step": "prompt_built",
        "agent_id": agent_config.agent_id,
        "agent_prompt": final_prompt,
        "agent_config": asdict(agent_config),
        "next_step": "create_agent",
        "message": f"Agent prompt built successfully for {agent_config.name}"
    }


@mcp.tool()
async def create_ai_agent(
    session_id: str,
    test_message: str = "Hello! Please introduce yourself and explain what you can help me with."
) -> Dict[str, Any]:
    """Create the final AI agent and test it with a sample message."""
    if session_id not in prompt_sessions:
        return {"success": False, "error": "Session not found"}
    
    session = prompt_sessions[session_id]
    agent_config = session.agent_config
    
    if not agent_config:
        return {"success": False, "error": "No agent configuration found. Please build the prompt first."}
    
    # Test the agent with the provided message
    test_response = None
    if "llm-provider" in mcp_clients:
        test_prompt = f"{agent_config.prompt_template}\\n\\nUser: {test_message}\\nAssistant:"
        
        test_result = await mcp_clients["llm-provider"].call_tool(
            "generate_text",
            {
                "prompt": test_prompt,
                "model": agent_config.llm_config.get("model", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
                "max_tokens": agent_config.llm_config.get("max_tokens", 1000),
                "temperature": agent_config.llm_config.get("temperature", 0.7)
            }
        )
        if test_result.get("success"):
            test_response = test_result.get("content")
    
    # Store agent in memory if available
    if "memory" in mcp_clients and agent_config.memory_config.get("memory_session_id"):
        await mcp_clients["memory"].call_tool(
            "store_memory",
            {
                "session_id": agent_config.memory_config["memory_session_id"],
                "content": {
                    "type": "agent_creation",
                    "agent_id": agent_config.agent_id,
                    "agent_name": agent_config.name,
                    "created_at": datetime.now().isoformat()
                }
            }
        )
    
    session.current_step = "agent_created"
    
    return {
        "success": True,
        "session_id": session_id,
        "current_step": "agent_created",
        "agent_id": agent_config.agent_id,
        "agent_name": agent_config.name,
        "agent_description": agent_config.description,
        "test_message": test_message,
        "test_response": test_response,
        "agent_ready": True,
        "usage_instructions": {
            "deployment": "The agent is ready for deployment",
            "integration": "Use the agent_id to integrate with your application",
            "customization": "The prompt can be further customized based on feedback"
        },
        "message": f"AI Agent '{agent_config.name}' created successfully and is ready for use!"
    }


@mcp.tool()
async def get_session_status(session_id: str) -> Dict[str, Any]:
    """Get the current status of a prompt building session."""
    if session_id not in prompt_sessions:
        return {"success": False, "error": "Session not found"}
    
    session = prompt_sessions[session_id]
    
    return {
        "success": True,
        "session_id": session_id,
        "current_step": session.current_step,
        "context_data": session.context_data,
        "agent_config": asdict(session.agent_config) if session.agent_config else None,
        "prompt_history": session.prompt_history,
        "created_at": session.created_at.isoformat(),
        "available_next_steps": get_next_steps(session.current_step)
    }


@mcp.tool()
async def list_created_agents() -> Dict[str, Any]:
    """List all created AI agents."""
    agents = []
    for agent_config in agent_configurations.values():
        agents.append({
            "agent_id": agent_config.agent_id,
            "name": agent_config.name,
            "description": agent_config.description,
            "created_at": agent_config.created_at.isoformat(),
            "updated_at": agent_config.updated_at.isoformat()
        })
    
    return {
        "success": True,
        "agents": agents,
        "total_count": len(agents),
        "message": f"Found {len(agents)} created agents"
    }


@mcp.tool()
async def get_mcp_servers_status() -> Dict[str, Any]:
    """Get the status of all connected MCP servers."""
    servers_status = {}
    
    for name, client in mcp_clients.items():
        servers_status[name] = {
            "connected": client.connected,
            "host": client.config.host,
            "port": client.config.port,
            "description": client.config.description,
            "tools": client.config.tools
        }
    
    return {
        "success": True,
        "servers": servers_status,
        "total_servers": len(mcp_clients),
        "connected_servers": sum(1 for client in mcp_clients.values() if client.connected),
        "message": f"{len(mcp_clients)} MCP servers configured"
    }


def get_next_steps(current_step: str) -> List[str]:
    """Get the available next steps based on current step."""
    step_flow = {
        "initialization": ["define_context"],
        "context_defined": ["configure_memory"],
        "memory_configured": ["configure_thinking"],
        "thinking_configured": ["build_prompt"],
        "prompt_built": ["create_agent"],
        "agent_created": ["test_agent", "deploy_agent"]
    }
    return step_flow.get(current_step, [])


# Startup event
@mcp.on_event("startup")
async def startup():
    """Initialize the MCP server."""
    logger.info("Starting Prompt Builder MCP Server...")
    await initialize_mcp_clients()
    logger.info("Prompt Builder MCP Server started successfully")


# Shutdown event
@mcp.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Shutting down Prompt Builder MCP Server...")
    for client in mcp_clients.values():
        await client.disconnect()
    logger.info("Prompt Builder MCP Server shut down")


if __name__ == "__main__":
    mcp.run()