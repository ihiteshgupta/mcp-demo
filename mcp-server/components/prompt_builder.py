import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound

from utils.exceptions import PromptBuilderError

logger = logging.getLogger(__name__)


class SequentialThinkingStep:
    """Represents a step in sequential thinking process."""
    
    def __init__(self, step_number: int, description: str, reasoning: str = "", expected_output: str = ""):
        self.step_number = step_number
        self.description = description
        self.reasoning = reasoning
        self.expected_output = expected_output
        self.result = None
        self.completed = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "description": self.description,
            "reasoning": self.reasoning,
            "expected_output": self.expected_output,
            "result": self.result,
            "completed": self.completed
        }


class PromptBuilder:
    """Dynamic prompt construction with templates."""
    
    def __init__(self, template_dir: str = None):
        if template_dir is None:
            template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
        
        self.template_dir = Path(template_dir)
        
        if not self.template_dir.exists():
            logger.warning(f"Template directory does not exist: {template_dir}")
            self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sequential thinking chains
        self.thinking_chains = {}
        
        try:
            self.env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                autoescape=False,
                trim_blocks=True,
                lstrip_blocks=True
            )
            logger.info(f"Initialized PromptBuilder with template directory: {template_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize Jinja2 environment: {e}")
            raise PromptBuilderError(f"Failed to initialize template environment: {e}")
    
    def _load_template(self, template_name: str) -> Template:
        """Load a Jinja2 template by name."""
        try:
            return self.env.get_template(template_name)
        except TemplateNotFound:
            logger.error(f"Template not found: {template_name}")
            raise PromptBuilderError(f"Template '{template_name}' not found")
        except Exception as e:
            logger.error(f"Error loading template {template_name}: {e}")
            raise PromptBuilderError(f"Error loading template '{template_name}': {e}")
    
    def build_business_rule_prompt(
        self, 
        context: str, 
        requirements: str,
        rule_id: str = None,
        examples: List[str] = None
    ) -> str:
        """Build prompt for business rule generation."""
        try:
            template = self._load_template("business_rule_generation.jinja2")
            
            variables = {
                "context": context,
                "requirements": requirements,
                "rule_id": rule_id,
                "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
                "examples": examples or []
            }
            
            prompt = template.render(**variables)
            logger.debug(f"Generated business rule prompt with {len(prompt)} characters")
            return prompt
            
        except Exception as e:
            logger.error(f"Error building business rule prompt: {e}")
            raise PromptBuilderError(f"Failed to build business rule prompt: {e}")
    
    def build_validation_prompt(self, rule: str, examples: List[str] = None) -> str:
        """Build prompt for rule validation."""
        try:
            template = self._load_template("rule_validation.jinja2")
            
            variables = {
                "rule_content": rule,
                "examples": examples or []
            }
            
            prompt = template.render(**variables)
            logger.debug(f"Generated validation prompt with {len(prompt)} characters")
            return prompt
            
        except Exception as e:
            logger.error(f"Error building validation prompt: {e}")
            raise PromptBuilderError(f"Failed to build validation prompt: {e}")
    
    def build_context_search_prompt(
        self, 
        query: str, 
        documents: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for context search and analysis."""
        try:
            template = self._load_template("context_search.jinja2")
            
            # Ensure documents have the expected structure
            formatted_docs = []
            for doc in documents:
                formatted_doc = {
                    "title": doc.get("title", ""),
                    "source": doc.get("source", ""),
                    "content": doc.get("content", ""),
                    "score": doc.get("score", 0.0)
                }
                formatted_docs.append(formatted_doc)
            
            variables = {
                "query": query,
                "documents": formatted_docs
            }
            
            prompt = template.render(**variables)
            logger.debug(f"Generated context search prompt with {len(prompt)} characters")
            return prompt
            
        except Exception as e:
            logger.error(f"Error building context search prompt: {e}")
            raise PromptBuilderError(f"Failed to build context search prompt: {e}")
    
    def create_thinking_chain(self, chain_id: str, description: str = "") -> str:
        """Create a new sequential thinking chain."""
        self.thinking_chains[chain_id] = {
            "id": chain_id,
            "description": description,
            "steps": [],
            "created_at": datetime.now(),
            "completed": False
        }
        logger.info(f"Created thinking chain: {chain_id}")
        return chain_id
    
    def add_thinking_step(
        self, 
        chain_id: str, 
        description: str, 
        reasoning: str = "", 
        expected_output: str = ""
    ) -> SequentialThinkingStep:
        """Add a step to sequential thinking chain."""
        if chain_id not in self.thinking_chains:
            raise PromptBuilderError(f"Thinking chain '{chain_id}' not found")
        
        chain = self.thinking_chains[chain_id]
        step_number = len(chain["steps"]) + 1
        
        step = SequentialThinkingStep(
            step_number=step_number,
            description=description,
            reasoning=reasoning,
            expected_output=expected_output
        )
        
        chain["steps"].append(step)
        logger.debug(f"Added step {step_number} to chain {chain_id}: {description}")
        return step
    
    def build_sequential_thinking_prompt(
        self, 
        chain_id: str,
        main_task: str,
        context: str = "",
        additional_instructions: str = ""
    ) -> str:
        """Build a prompt that includes sequential thinking steps."""
        if chain_id not in self.thinking_chains:
            raise PromptBuilderError(f"Thinking chain '{chain_id}' not found")
        
        chain = self.thinking_chains[chain_id]
        steps = chain["steps"]
        
        if not steps:
            raise PromptBuilderError(f"No steps defined for thinking chain '{chain_id}'")
        
        try:
            # Try to load custom template first, fallback to built-in
            try:
                template = self._load_template("sequential_thinking.jinja2")
            except PromptBuilderError:
                # Use built-in template
                template_content = self._get_builtin_sequential_template()
                template = self.env.from_string(template_content)
            
            variables = {
                "main_task": main_task,
                "context": context,
                "thinking_steps": [step.to_dict() for step in steps],
                "chain_description": chain["description"],
                "additional_instructions": additional_instructions,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            prompt = template.render(**variables)
            logger.debug(f"Generated sequential thinking prompt for chain {chain_id} with {len(steps)} steps")
            return prompt
            
        except Exception as e:
            logger.error(f"Error building sequential thinking prompt: {e}")
            raise PromptBuilderError(f"Failed to build sequential thinking prompt: {e}")
    
    def _get_builtin_sequential_template(self) -> str:
        """Get built-in sequential thinking template."""
        return '''You are an AI assistant that thinks through problems step by step using sequential reasoning.

MAIN TASK: {{ main_task }}

{% if context %}
CONTEXT:
{{ context }}
{% endif %}

SEQUENTIAL THINKING PROCESS:
You must work through the following steps in order, showing your reasoning for each step:

{% for step in thinking_steps %}
STEP {{ step.step_number }}: {{ step.description }}
{% if step.reasoning %}
Reasoning: {{ step.reasoning }}
{% endif %}
{% if step.expected_output %}
Expected Output: {{ step.expected_output }}
{% endif %}

{% endfor %}

INSTRUCTIONS:
1. Work through each step sequentially, clearly showing your thinking process
2. For each step, provide:
   - Your analysis and reasoning
   - The specific output or conclusion for that step
   - How it connects to the next step
3. Build upon previous steps to reach the final solution
4. Be thorough but concise in your reasoning

{% if additional_instructions %}
ADDITIONAL REQUIREMENTS:
{{ additional_instructions }}
{% endif %}

Begin with STEP 1 and work through each step systematically:'''
    
    def build_business_rule_with_thinking(
        self, 
        context: str, 
        requirements: str,
        rule_id: str = None,
        examples: List[str] = None
    ) -> str:
        """Build business rule prompt with sequential thinking."""
        # Create thinking chain for business rule generation
        chain_id = f"business_rule_{rule_id or datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.create_thinking_chain(chain_id, "Business Rule Generation with Sequential Reasoning")
        
        # Define thinking steps
        self.add_thinking_step(
            chain_id,
            "Analyze Business Context",
            "Understand the business domain, constraints, and objectives",
            "Clear understanding of business context and key variables"
        )
        
        self.add_thinking_step(
            chain_id,
            "Parse Requirements", 
            "Break down specific requirements into actionable components",
            "List of specific requirements and success criteria"
        )
        
        self.add_thinking_step(
            chain_id,
            "Design Rule Logic",
            "Structure conditional logic with when-then patterns",
            "Logical flow with conditions, actions, and edge cases"
        )
        
        self.add_thinking_step(
            chain_id,
            "Optimize for Business Value",
            "Ensure rule maximizes business value while minimizing risks",
            "Rule optimized for revenue, customer satisfaction, and operational efficiency"
        )
        
        self.add_thinking_step(
            chain_id,
            "Format as JSON Structure",
            "Convert logic into implementable JSON format with proper syntax",
            "Valid JSON business rule ready for system integration"
        )
        
        # Build the prompt
        main_task = f"Generate a comprehensive business rule in JSON format that addresses the given requirements within the specified business context."
        
        additional_instructions = f'''
RULE REQUIREMENTS:
{requirements}

EXAMPLES (if provided):
{chr(10).join(f"- {example}" for example in (examples or []))}

OUTPUT FORMAT:
Provide a JSON business rule with the following structure:
{{
  "rule_id": "unique_identifier",
  "name": "descriptive_name",
  "description": "detailed_description",
  "when": {{
    // conditions using JSON query syntax
  }},
  "then": {{
    // actions and consequences
  }},
  "priority": "HIGH|MEDIUM|LOW",
  "business_value": "description_of_business_impact",
  "metadata": {{
    "created_by": "ai_sequential_thinking",
    "thinking_chain": "{chain_id}",
    "version": "1.0"
  }}
}}
'''
        
        return self.build_sequential_thinking_prompt(
            chain_id=chain_id,
            main_task=main_task,
            context=context,
            additional_instructions=additional_instructions
        )
    
    def get_thinking_chain(self, chain_id: str) -> Optional[Dict[str, Any]]:
        """Get thinking chain by ID."""
        chain = self.thinking_chains.get(chain_id)
        if chain:
            return {
                "id": chain["id"],
                "description": chain["description"],
                "steps": [step.to_dict() for step in chain["steps"]],
                "created_at": chain["created_at"].isoformat(),
                "completed": chain["completed"]
            }
        return None
    
    def list_thinking_chains(self) -> List[Dict[str, Any]]:
        """List all thinking chains."""
        return [
            {
                "id": chain["id"],
                "description": chain["description"],
                "step_count": len(chain["steps"]),
                "created_at": chain["created_at"].isoformat(),
                "completed": chain["completed"]
            }
            for chain in self.thinking_chains.values()
        ]
    
    def apply_template(self, template_name: str, variables: Dict[str, Any]) -> str:
        """Apply a custom template with provided variables."""
        try:
            template = self._load_template(template_name)
            prompt = template.render(**variables)
            logger.debug(f"Applied template {template_name} with {len(prompt)} characters")
            return prompt
            
        except Exception as e:
            logger.error(f"Error applying template {template_name}: {e}")
            raise PromptBuilderError(f"Failed to apply template '{template_name}': {e}")
    
    def list_templates(self) -> List[str]:
        """List all available templates."""
        try:
            templates = []
            for file_path in self.template_dir.glob("*.jinja2"):
                templates.append(file_path.name)
            logger.debug(f"Found {len(templates)} templates")
            return sorted(templates)
        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            return []
    
    def create_custom_template(
        self, 
        template_name: str, 
        template_content: str
    ) -> bool:
        """Create a custom template."""
        try:
            if not template_name.endswith('.jinja2'):
                template_name += '.jinja2'
            
            template_path = self.template_dir / template_name
            
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(template_content)
            
            # Validate the template by trying to load it
            self._load_template(template_name)
            
            logger.info(f"Created custom template: {template_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating custom template {template_name}: {e}")
            raise PromptBuilderError(f"Failed to create template '{template_name}': {e}")
    
    def get_template_info(self) -> Dict[str, Any]:
        """Get information about the prompt builder and available templates."""
        return {
            "template_directory": str(self.template_dir),
            "available_templates": self.list_templates(),
            "builtin_methods": [
                "build_business_rule_prompt",
                "build_validation_prompt", 
                "build_context_search_prompt",
                "apply_template"
            ]
        }


# Convenience functions for common prompt patterns
def create_simple_prompt(instruction: str, context: str = "", examples: List[str] = None) -> str:
    """Create a simple prompt with instruction and context."""
    prompt_parts = [instruction]
    
    if context:
        prompt_parts.append(f"\nContext:\n{context}")
    
    if examples:
        prompt_parts.append("\nExamples:")
        for i, example in enumerate(examples, 1):
            prompt_parts.append(f"{i}. {example}")
    
    return "\n".join(prompt_parts)


def create_structured_prompt(
    role: str,
    task: str, 
    context: Dict[str, Any] = None,
    constraints: List[str] = None,
    output_format: str = None
) -> str:
    """Create a structured prompt with role, task, context, and constraints."""
    prompt_parts = [f"You are {role}."]
    
    if task:
        prompt_parts.append(f"\nTask: {task}")
    
    if context:
        prompt_parts.append("\nContext:")
        for key, value in context.items():
            prompt_parts.append(f"- {key}: {value}")
    
    if constraints:
        prompt_parts.append("\nConstraints:")
        for constraint in constraints:
            prompt_parts.append(f"- {constraint}")
    
    if output_format:
        prompt_parts.append(f"\nOutput Format:\n{output_format}")
    
    return "\n".join(prompt_parts)