import logging
from typing import Dict, Any, Optional

from config.settings import settings
from components.llm_provider import LLMProvider
from components.prompt_builder import PromptBuilder
from components.session_storage import SessionStorage
from components.vector_store import VectorStore
from components.business_rules import BusinessRuleGenerator
from models.requests import (
    BusinessRuleRequest, BusinessRuleResponse,
    ValidationRequest, ValidationResponse, ValidationResult,
    SearchRequest, SearchResponse
)
from mcp.handlers import MCPHandlers
from utils.exceptions import MCPError, MCPMethodNotFoundError

logger = logging.getLogger(__name__)


class MCPGenAIServer:
    """Main MCP server with GenAI components."""
    
    def __init__(self):
        # Initialize components
        self.llm_provider = LLMProvider()
        self.prompt_builder = PromptBuilder()
        # Use memory storage for demo (Redis not required)
        self.session_storage = SessionStorage(storage_type="memory")
        self.vector_store = VectorStore(
            store_type="chroma" if settings.chroma_host else "memory"
        )
        
        # Initialize business rule generator
        self.business_rule_generator = BusinessRuleGenerator(
            llm_provider=self.llm_provider,
            prompt_builder=self.prompt_builder,
            session_storage=self.session_storage,
            vector_store=self.vector_store
        )
        
        # Initialize MCP handlers
        self.handlers = MCPHandlers(self)
        
        logger.info("Initialized MCPGenAIServer with all components")
    
    async def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP message."""
        try:
            jsonrpc = message.get("jsonrpc", "2.0")
            method = message.get("method")
            params = message.get("params", {})
            message_id = message.get("id")
            
            if not method:
                return {
                    "jsonrpc": jsonrpc,
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request"
                    },
                    "id": message_id
                }
            
            # Route to appropriate handler
            if method == "initialize":
                result = await self.handlers.handle_initialize(params)
            elif method == "tools/list":
                result = await self.handlers.handle_list_tools(params)
            elif method == "tools/call":
                result = await self.handlers.handle_call_tool(params)
            elif method == "resources/list":
                result = await self.handlers.handle_list_resources(params)
            else:
                await self.handlers.handle_unknown_method(method, params)
            
            return {
                "jsonrpc": jsonrpc,
                "result": result,
                "id": message_id
            }
            
        except MCPError as e:
            logger.error(f"MCP error handling message: {e}")
            return {
                "jsonrpc": message.get("jsonrpc", "2.0"),
                "error": {
                    "code": e.code,
                    "message": e.message
                },
                "id": message.get("id")
            }
        except Exception as e:
            logger.error(f"Unexpected error handling message: {e}")
            return {
                "jsonrpc": message.get("jsonrpc", "2.0"),
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                },
                "id": message.get("id")
            }
    
    async def handle_business_rule_generation(self, request: BusinessRuleRequest) -> BusinessRuleResponse:
        """Handle business rule generation request."""
        return await self.business_rule_generator.generate_rule(request)
    
    async def handle_rule_validation(self, request: ValidationRequest) -> ValidationResponse:
        """Handle rule validation request."""
        try:
            # Create or get session
            session = None
            if request.session_id:
                session = await self.session_storage.get_session(request.session_id)
                if not session:
                    session = await self.session_storage.create_session(
                        session_id=request.session_id
                    )
            else:
                session = await self.session_storage.create_session()
            
            # Build validation prompt
            prompt = self.prompt_builder.build_validation_prompt(
                rule=request.rule_content,
                examples=request.examples or []
            )
            
            # Get validation from LLM
            completion = await self.llm_provider.generate_completion(
                prompt=prompt,
                provider=request.provider
            )
            
            # Parse validation result
            validation = await self._parse_validation_result(completion.content)
            
            # Update session
            await self.session_storage.update_session(
                session.session_id,
                {
                    "last_validation": validation.model_dump(mode='json'),
                    "validation_request": request.model_dump(mode='json')
                }
            )
            
            return ValidationResponse(
                validation=validation,
                session_id=session.session_id
            )
            
        except Exception as e:
            logger.error(f"Error validating rule: {e}")
            # Return a default validation result
            validation = ValidationResult(
                score=5,
                strengths=["Rule provided for validation"],
                issues=[f"Validation error: {e}"],
                recommendations=["Please review the rule manually"],
                implementation_notes=["Validation could not be completed automatically"]
            )
            return ValidationResponse(validation=validation)
    
    async def handle_context_search(self, request: SearchRequest) -> SearchResponse:
        """Handle context search request."""
        try:
            # Create or get session
            session = None
            if request.session_id:
                session = await self.session_storage.get_session(request.session_id)
            
            # Search vector store
            search_results = await self.vector_store.similarity_search(
                query=request.query,
                k=request.limit,
                filter_metadata=request.filter_metadata
            )
            
            # If no results from vector store, try to find similar rules
            if not search_results:
                search_results = await self.business_rule_generator.search_similar_rules(
                    query=request.query,
                    limit=request.limit
                )
                
                # Convert to expected format
                formatted_results = []
                for result in search_results:
                    formatted_results.append({
                        "content": result["content"],
                        "score": result["score"],
                        "metadata": result["metadata"]
                    })
                search_results = formatted_results
            else:
                # Convert search results to dict format
                formatted_results = []
                for result in search_results:
                    formatted_results.append({
                        "content": result.document.content,
                        "score": result.score,
                        "metadata": result.document.metadata
                    })
                search_results = formatted_results
            
            # If still no results, provide a contextual response
            if not search_results:
                # Build context search prompt
                prompt = self.prompt_builder.build_context_search_prompt(
                    query=request.query,
                    documents=[]
                )
                
                # Get contextual response from LLM
                completion = await self.llm_provider.generate_completion(prompt=prompt)
                
                search_results = [{
                    "content": completion.content,
                    "score": 0.8,
                    "metadata": {
                        "type": "llm_generated",
                        "query": request.query,
                        "model": completion.model
                    }
                }]
            
            # Update session if available
            if session:
                await self.session_storage.update_session(
                    session.session_id,
                    {
                        "last_search": {
                            "query": request.query,
                            "results_count": len(search_results)
                        }
                    }
                )
            
            return SearchResponse(
                results=search_results,
                query=request.query,
                total_results=len(search_results),
                session_id=session.session_id if session else None
            )
            
        except Exception as e:
            logger.error(f"Error searching context: {e}")
            return SearchResponse(
                results=[{
                    "content": f"Search error: {e}",
                    "score": 0.0,
                    "metadata": {"error": True}
                }],
                query=request.query,
                total_results=1
            )
    
    async def _parse_validation_result(self, content: str) -> ValidationResult:
        """Parse LLM validation response into structured result."""
        try:
            # Initialize default values
            validation_data = {
                "score": 5,
                "strengths": [],
                "issues": [],
                "recommendations": [],
                "revised_rule": None,
                "implementation_notes": []
            }
            
            lines = content.split('\n')
            current_section = None
            current_content = []
            
            for line in lines:
                line = line.strip()
                
                # Detect section headers
                if 'validation score:' in line.lower() or 'score:' in line.lower():
                    # Extract score
                    import re
                    score_match = re.search(r'(\d+)(?:/10)?', line)
                    if score_match:
                        validation_data['score'] = min(10, max(0, int(score_match.group(1))))
                elif 'strengths:' in line.lower():
                    if current_section and current_content:
                        validation_data[current_section] = current_content.copy()
                    current_section = 'strengths'
                    current_content = []
                elif 'issues found:' in line.lower() or 'issues:' in line.lower():
                    if current_section and current_content:
                        validation_data[current_section] = current_content.copy()
                    current_section = 'issues'
                    current_content = []
                elif 'recommendations:' in line.lower():
                    if current_section and current_content:
                        validation_data[current_section] = current_content.copy()
                    current_section = 'recommendations'
                    current_content = []
                elif 'revised rule:' in line.lower():
                    if current_section and current_content:
                        validation_data[current_section] = current_content.copy()
                    current_section = 'revised_rule'
                    current_content = []
                elif 'implementation notes:' in line.lower():
                    if current_section and current_content:
                        if current_section == 'revised_rule':
                            validation_data['revised_rule'] = '\n'.join(current_content).strip()
                        else:
                            validation_data[current_section] = current_content.copy()
                    current_section = 'implementation_notes'
                    current_content = []
                elif current_section and line:
                    # Clean up bullet points and numbering
                    clean_line = line
                    clean_line = re.sub(r'^\d+\.\s*', '', clean_line)
                    clean_line = re.sub(r'^[-*]\s*', '', clean_line)
                    if clean_line:
                        current_content.append(clean_line)
            
            # Handle the last section
            if current_section and current_content:
                if current_section == 'revised_rule':
                    validation_data['revised_rule'] = '\n'.join(current_content).strip()
                else:
                    validation_data[current_section] = current_content.copy()
            
            return ValidationResult(**validation_data)
            
        except Exception as e:
            logger.error(f"Error parsing validation result: {e}")
            return ValidationResult(
                score=5,
                strengths=["Rule provided"],
                issues=[f"Parsing error: {e}"],
                recommendations=["Please review manually"],
                implementation_notes=["Automatic validation parsing failed"]
            )
    
    async def get_server_status(self) -> Dict[str, Any]:
        """Get comprehensive server status."""
        try:
            # Get component status
            llm_info = self.llm_provider.get_provider_info()
            prompt_info = self.prompt_builder.get_template_info()
            session_info = self.session_storage.get_storage_info()
            vector_info = self.vector_store.get_store_info()
            
            # Get generation stats
            generation_stats = await self.business_rule_generator.get_generation_stats()
            
            return {
                "server": {
                    "name": settings.mcp_server_name,
                    "version": settings.mcp_server_version,
                    "status": "running"
                },
                "components": {
                    "llm_provider": llm_info,
                    "prompt_builder": prompt_info,
                    "session_storage": session_info,
                    "vector_store": vector_info
                },
                "statistics": generation_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting server status: {e}")
            return {
                "server": {
                    "name": settings.mcp_server_name,
                    "version": settings.mcp_server_version,
                    "status": "error",
                    "error": str(e)
                }
            }