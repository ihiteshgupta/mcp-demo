import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from config.settings import settings
from models.requests import (
    InitializeRequest, InitializeResponse,
    ListToolsRequest, ListToolsResponse, Tool,
    CallToolRequest, CallToolResponse,
    ListResourcesRequest, ListResourcesResponse, Resource,
    BusinessRuleRequest, BusinessRuleResponse, BusinessRule,
    ValidationRequest, ValidationResponse, ValidationResult,
    SearchRequest, SearchResponse
)
from utils.exceptions import MCPError, MCPMethodNotFoundError, MCPInvalidParamsError

logger = logging.getLogger(__name__)


class MCPHandlers:
    """MCP protocol message handlers."""
    
    def __init__(self, server):
        from mcp.server import MCPGenAIServer
        self.server: MCPGenAIServer = server
    
    async def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request."""
        try:
            request = InitializeRequest(**params)
            
            # Log client information
            logger.info(f"Client connected: {request.clientInfo}")
            
            response = InitializeResponse(
                capabilities={
                    "tools": {
                        "listChanged": True
                    },
                    "resources": {
                        "subscribe": False,
                        "listChanged": True
                    },
                    "logging": {}
                },
                serverInfo={
                    "name": settings.mcp_server_name,
                    "version": settings.mcp_server_version
                }
            )
            
            return response.dict()
            
        except Exception as e:
            logger.error(f"Error in initialize handler: {e}")
            raise MCPError(f"Initialize failed: {e}")
    
    async def handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP list tools request."""
        try:
            request = ListToolsRequest(**(params or {}))
            
            tools = [
                Tool(
                    name="generate_business_rule",
                    description="Generate a business rule based on context and requirements",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "context": {
                                "type": "string",
                                "description": "Business context for rule generation"
                            },
                            "requirements": {
                                "type": "string", 
                                "description": "Specific requirements for the rule"
                            },
                            "rule_id": {
                                "type": "string",
                                "description": "Optional rule ID"
                            },
                            "examples": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Example rules for reference"
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Session ID for tracking"
                            },
                            "provider": {
                                "type": "string",
                                "description": "LLM provider to use",
                                "enum": ["openai", "anthropic", "local"]
                            },
                            "model": {
                                "type": "string",
                                "description": "Specific model to use"
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Temperature for generation",
                                "minimum": 0.0,
                                "maximum": 2.0
                            }
                        },
                        "required": ["context", "requirements"]
                    }
                ),
                Tool(
                    name="validate_business_rule",
                    description="Validate a business rule and provide feedback",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "rule_content": {
                                "type": "string",
                                "description": "Rule content to validate"
                            },
                            "examples": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Validation examples"
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Session ID"
                            },
                            "provider": {
                                "type": "string",
                                "description": "LLM provider to use",
                                "enum": ["openai", "anthropic", "local"]
                            }
                        },
                        "required": ["rule_content"]
                    }
                ),
                Tool(
                    name="search_context",
                    description="Search for relevant context and information",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "minimum": 1,
                                "maximum": 20
                            },
                            "filter_metadata": {
                                "type": "object",
                                "description": "Metadata filters"
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Session ID"
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]
            
            response = ListToolsResponse(tools=tools)
            return response.dict()
            
        except Exception as e:
            logger.error(f"Error in list_tools handler: {e}")
            raise MCPError(f"List tools failed: {e}")
    
    async def handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP call tool request."""
        try:
            request = CallToolRequest(**params)
            
            if request.name == "generate_business_rule":
                return await self._handle_generate_business_rule(request.arguments)
            elif request.name == "validate_business_rule":
                return await self._handle_validate_business_rule(request.arguments)
            elif request.name == "search_context":
                return await self._handle_search_context(request.arguments)
            else:
                raise MCPMethodNotFoundError(f"Tool '{request.name}' not found")
            
        except Exception as e:
            logger.error(f"Error in call_tool handler: {e}")
            if isinstance(e, MCPError):
                raise
            raise MCPError(f"Tool call failed: {e}")
    
    async def _handle_generate_business_rule(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle business rule generation."""
        try:
            request = BusinessRuleRequest(**arguments)
            response = await self.server.handle_business_rule_generation(request)
            
            content = [
                {
                    "type": "text",
                    "text": f"Generated business rule: {response.rule.name}\n\n"
                           f"ID: {response.rule.id}\n"
                           f"Description: {response.rule.description}\n"
                           f"Condition: {response.rule.condition}\n"
                           f"Action: {response.rule.action}\n"
                           f"Priority: {response.rule.priority}\n"
                           f"Business Value: {response.rule.business_value}\n\n"
                           f"Examples:\n" + "\n".join(f"- {ex}" for ex in response.rule.examples)
                }
            ]
            
            return CallToolResponse(content=content).dict()
            
        except Exception as e:
            logger.error(f"Error generating business rule: {e}")
            content = [{"type": "text", "text": f"Error generating business rule: {e}"}]
            return CallToolResponse(content=content, isError=True).dict()
    
    async def _handle_validate_business_rule(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle business rule validation."""
        try:
            request = ValidationRequest(**arguments)
            response = await self.server.handle_rule_validation(request)
            
            content = [
                {
                    "type": "text",
                    "text": f"Validation Results\n\n"
                           f"Score: {response.validation.score}/10\n\n"
                           f"Strengths:\n" + "\n".join(f"- {s}" for s in response.validation.strengths) + "\n\n"
                           f"Issues Found:\n" + "\n".join(f"- {i}" for i in response.validation.issues) + "\n\n"
                           f"Recommendations:\n" + "\n".join(f"- {r}" for r in response.validation.recommendations) + "\n\n"
                           + (f"Revised Rule:\n{response.validation.revised_rule}\n\n" if response.validation.revised_rule else "")
                           + f"Implementation Notes:\n" + "\n".join(f"- {n}" for n in response.validation.implementation_notes)
                }
            ]
            
            return CallToolResponse(content=content).dict()
            
        except Exception as e:
            logger.error(f"Error validating business rule: {e}")
            content = [{"type": "text", "text": f"Error validating business rule: {e}"}]
            return CallToolResponse(content=content, isError=True).dict()
    
    async def _handle_search_context(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle context search."""
        try:
            request = SearchRequest(**arguments)
            response = await self.server.handle_context_search(request)
            
            results_text = f"Search Results for: '{response.query}'\n"
            results_text += f"Found {response.total_results} results\n\n"
            
            for i, result in enumerate(response.results, 1):
                results_text += f"{i}. Score: {result.get('score', 'N/A')}\n"
                results_text += f"   Content: {result.get('content', 'No content')[:200]}...\n"
                if result.get('metadata'):
                    results_text += f"   Metadata: {result['metadata']}\n"
                results_text += "\n"
            
            content = [{"type": "text", "text": results_text}]
            return CallToolResponse(content=content).dict()
            
        except Exception as e:
            logger.error(f"Error searching context: {e}")
            content = [{"type": "text", "text": f"Error searching context: {e}"}]
            return CallToolResponse(content=content, isError=True).dict()
    
    async def handle_list_resources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP list resources request."""
        try:
            request = ListResourcesRequest(**(params or {}))
            
            resources = [
                Resource(
                    uri="template://business_rule_generation",
                    name="Business Rule Generation Template",
                    description="Template for generating business rules",
                    mimeType="text/plain"
                ),
                Resource(
                    uri="template://rule_validation",
                    name="Rule Validation Template", 
                    description="Template for validating business rules",
                    mimeType="text/plain"
                ),
                Resource(
                    uri="template://context_search",
                    name="Context Search Template",
                    description="Template for context search and analysis",
                    mimeType="text/plain"
                ),
                Resource(
                    uri="info://server_status",
                    name="Server Status",
                    description="Current server status and component information",
                    mimeType="application/json"
                )
            ]
            
            response = ListResourcesResponse(resources=resources)
            return response.dict()
            
        except Exception as e:
            logger.error(f"Error in list_resources handler: {e}")
            raise MCPError(f"List resources failed: {e}")
    
    async def handle_unknown_method(self, method: str, params: Dict[str, Any]) -> None:
        """Handle unknown MCP methods."""
        logger.warning(f"Unknown MCP method called: {method}")
        raise MCPMethodNotFoundError(method)