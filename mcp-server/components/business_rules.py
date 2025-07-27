import logging
import json
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import uuid4

from components.llm_provider import LLMProvider
from components.prompt_builder import PromptBuilder
from components.session_storage import SessionStorage
from components.vector_store import VectorStore, Document
from models.requests import BusinessRule, BusinessRuleRequest, BusinessRuleResponse
from utils.exceptions import LLMProviderError, PromptBuilderError

logger = logging.getLogger(__name__)


class BusinessRuleGenerator:
    """Business rule generation and management."""
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_builder: PromptBuilder,
        session_storage: SessionStorage,
        vector_store: VectorStore
    ):
        self.llm_provider = llm_provider
        self.prompt_builder = prompt_builder
        self.session_storage = session_storage
        self.vector_store = vector_store
        logger.info("Initialized BusinessRuleGenerator")
    
    async def generate_rule(self, request: BusinessRuleRequest) -> BusinessRuleResponse:
        """Generate a business rule based on the request."""
        try:
            # Create or get session
            session = None
            if request.session_id:
                session = await self.session_storage.get_session(request.session_id)
                if not session:
                    session = await self.session_storage.create_session(
                        session_id=request.session_id,
                        user_id=request.user_id
                    )
            else:
                session = await self.session_storage.create_session(user_id=request.user_id)
            
            # Store request in session for context
            await self.session_storage.update_session(
                session.session_id,
                {
                    "last_request": request.model_dump(mode='json'),
                    "request_type": "business_rule_generation"
                }
            )
            
            # Build prompt for rule generation with sequential thinking
            use_sequential_thinking = request.metadata.get("use_sequential_thinking", True) if hasattr(request, 'metadata') and request.metadata else True
            
            if use_sequential_thinking:
                prompt = self.prompt_builder.build_business_rule_with_thinking(
                    context=request.context,
                    requirements=request.requirements,
                    rule_id=request.rule_id,
                    examples=request.examples or []
                )
                logger.info(f"Using sequential thinking for rule generation: {request.rule_id}")
            else:
                prompt = self.prompt_builder.build_business_rule_prompt(
                    context=request.context,
                    requirements=request.requirements,
                    rule_id=request.rule_id,
                    examples=request.examples or []
                )
            
            # Generate rule using LLM
            completion = await self.llm_provider.generate_completion(
                prompt=prompt,
                model=request.model,
                provider=request.provider,
                temperature=request.temperature
            )
            
            # Parse the generated rule
            rule = await self._parse_generated_rule(
                completion.content,
                request.rule_id
            )
            
            # Store the generated rule in vector store for future reference
            await self._store_rule_in_vector_store(rule, request.context)
            
            # Update session with generated rule
            await self.session_storage.update_session(
                session.session_id,
                {
                    "last_generated_rule": rule.model_dump(mode='json'),
                    "generation_info": {
                        "model": completion.model,
                        "usage": completion.usage,
                        "finish_reason": completion.finish_reason
                    }
                }
            )
            
            response = BusinessRuleResponse(
                rule=rule,
                generation_info={
                    "model": completion.model,
                    "usage": completion.usage,
                    "finish_reason": completion.finish_reason,
                    "prompt_length": len(prompt)
                },
                session_id=session.session_id
            )
            
            logger.info(f"Generated business rule: {rule.id}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating business rule: {e}")
            raise LLMProviderError(f"Failed to generate business rule: {e}")
    
    async def _parse_generated_rule(self, content: str, rule_id: str = None) -> BusinessRule:
        """Parse the LLM-generated content into a structured business rule."""
        try:
            # Generate rule ID if not provided
            if not rule_id:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                rule_id = f"BR-{timestamp}"
            
            # Initialize default values
            rule_data = {
                "id": rule_id,
                "name": "Generated Business Rule",
                "description": "",
                "condition": "",
                "action": "",
                "priority": "MEDIUM",
                "business_value": "",
                "examples": [],
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "metadata": {"generated": True}
            }
            
            # Parse the structured content
            lines = content.split('\n')
            current_section = None
            current_content = []
            
            for line in lines:
                line = line.strip()
                
                # Detect section headers
                if line.startswith('### Rule Name:'):
                    if current_section:
                        rule_data[current_section] = '\n'.join(current_content).strip()
                    current_section = 'name'
                    current_content = [line.replace('### Rule Name:', '').strip()]
                elif line.startswith('### Description:'):
                    if current_section:
                        rule_data[current_section] = '\n'.join(current_content).strip()
                    current_section = 'description'
                    current_content = []
                elif line.startswith('### Condition:'):
                    if current_section:
                        rule_data[current_section] = '\n'.join(current_content).strip()
                    current_section = 'condition'
                    current_content = []
                elif line.startswith('### Action:'):
                    if current_section:
                        rule_data[current_section] = '\n'.join(current_content).strip()
                    current_section = 'action'
                    current_content = []
                elif line.startswith('### Priority:'):
                    if current_section:
                        rule_data[current_section] = '\n'.join(current_content).strip()
                    current_section = 'priority'
                    priority_text = line.replace('### Priority:', '').strip()
                    # Extract priority level
                    if 'HIGH' in priority_text.upper():
                        rule_data['priority'] = 'HIGH'
                    elif 'LOW' in priority_text.upper():
                        rule_data['priority'] = 'LOW'
                    else:
                        rule_data['priority'] = 'MEDIUM'
                    current_section = None
                elif line.startswith('### Business Value:'):
                    if current_section:
                        rule_data[current_section] = '\n'.join(current_content).strip()
                    current_section = 'business_value'
                    current_content = []
                elif line.startswith('### Examples:'):
                    if current_section:
                        rule_data[current_section] = '\n'.join(current_content).strip()
                    current_section = 'examples'
                    current_content = []
                elif current_section and line:
                    current_content.append(line)
            
            # Handle the last section
            if current_section and current_content:
                if current_section == 'examples':
                    # Parse examples as list
                    examples_text = '\n'.join(current_content).strip()
                    # Split by numbered items or bullet points
                    examples = []
                    for example_line in examples_text.split('\n'):
                        example_line = example_line.strip()
                        # Remove numbering and bullet points
                        example_line = re.sub(r'^\d+\.\s*', '', example_line)
                        example_line = re.sub(r'^[-*]\s*', '', example_line)
                        if example_line:
                            examples.append(example_line)
                    rule_data['examples'] = examples
                else:
                    rule_data[current_section] = '\n'.join(current_content).strip()
            
            # Fallback parsing if structured format wasn't used
            if not rule_data['name'] or rule_data['name'] == 'Generated Business Rule':
                # Try to extract a title from the first line
                first_lines = content.split('\n')[:5]
                for line in first_lines:
                    line = line.strip()
                    if line and not line.startswith('#') and len(line) > 10:
                        rule_data['name'] = line[:100]  # Limit length
                        break
            
            if not rule_data['description']:
                # Use the beginning of content as description
                rule_data['description'] = content[:300] + "..." if len(content) > 300 else content
            
            return BusinessRule(**rule_data)
            
        except Exception as e:
            logger.error(f"Error parsing generated rule: {e}")
            # Return a basic rule with the raw content
            return BusinessRule(
                id=rule_id or f"BR-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                name="Generated Business Rule",
                description=content[:300] + "..." if len(content) > 300 else content,
                condition="See description for details",
                action="See description for details", 
                priority="MEDIUM",
                business_value="Generated rule requires review",
                examples=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                metadata={"generated": True, "parse_error": str(e)}
            )
    
    async def _store_rule_in_vector_store(self, rule: BusinessRule, context: str):
        """Store the generated rule in vector store for future reference."""
        try:
            # Create document content that includes rule details and context
            document_content = f"""
Business Rule: {rule.name}

ID: {rule.id}
Priority: {rule.priority}

Description: {rule.description}

Condition: {rule.condition}

Action: {rule.action}

Business Value: {rule.business_value}

Context: {context}

Examples:
{chr(10).join(f"- {example}" for example in rule.examples)}
            """.strip()
            
            document = Document(
                id=f"rule_{rule.id}",
                content=document_content,
                metadata={
                    "type": "business_rule",
                    "rule_id": rule.id,
                    "rule_name": rule.name,
                    "priority": rule.priority,
                    "created_at": rule.created_at.isoformat(),
                    "context": context[:200]  # Store truncated context
                }
            )
            
            await self.vector_store.add_documents([document])
            logger.debug(f"Stored rule {rule.id} in vector store")
            
        except Exception as e:
            logger.warning(f"Failed to store rule in vector store: {e}")
            # Don't fail the whole operation if vector storage fails
    
    async def search_similar_rules(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar rules in the vector store."""
        try:
            search_results = await self.vector_store.similarity_search(
                query=query,
                k=limit,
                filter_metadata={"type": "business_rule"}
            )
            
            results = []
            for result in search_results:
                results.append({
                    "rule_id": result.document.metadata.get("rule_id"),
                    "rule_name": result.document.metadata.get("rule_name"),
                    "content": result.document.content,
                    "score": result.score,
                    "metadata": result.document.metadata
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar rules: {e}")
            return []
    
    async def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about rule generation."""
        try:
            # Count rules in vector store
            total_rules = await self.vector_store.count_documents()
            
            # Get recent sessions
            recent_sessions = await self.session_storage.list_sessions()
            
            generation_sessions = [
                session for session in recent_sessions
                if session.data.get("request_type") == "business_rule_generation"
            ]
            
            return {
                "total_rules_generated": total_rules,
                "active_sessions": len(recent_sessions),
                "generation_sessions": len(generation_sessions),
                "llm_providers": self.llm_provider.list_providers(),
                "default_provider": self.llm_provider.default_provider
            }
            
        except Exception as e:
            logger.error(f"Error getting generation stats: {e}")
            return {
                "total_rules_generated": 0,
                "active_sessions": 0,
                "generation_sessions": 0,
                "error": str(e)
            }