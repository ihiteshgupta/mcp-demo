#!/usr/bin/env python3
"""
Sequential Thinker MCP Server

An MCP server that provides sequential thinking capabilities for structured reasoning and problem solving.
Based on the Model Context Protocol (MCP) specification and inspired by the existing MCP demo architecture.
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent directory to path for importing existing components
mcp_server_path = str(Path(__file__).parent.parent / "mcp-server")
sys.path.append(mcp_server_path)

from mcp.server.fastmcp import FastMCP
from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
from pydantic import BaseModel, Field

# Import existing sequential thinking components
try:
    from components.prompt_builder import PromptBuilder, SequentialThinkingStep
    from utils.exceptions import PromptBuilderError
except ImportError:
    # Fallback: Create minimal local versions
    class PromptBuilderError(Exception):
        pass
    
    class SequentialThinkingStep:
        def __init__(self, step_number: int, description: str, reasoning: str = "", expected_output: str = ""):
            self.step_number = step_number
            self.description = description
            self.reasoning = reasoning
            self.expected_output = expected_output
            self.result = None
            self.completed = False
            
        def to_dict(self):
            return {
                "step_number": self.step_number,
                "description": self.description,
                "reasoning": self.reasoning,
                "expected_output": self.expected_output,
                "result": self.result,
                "completed": self.completed
            }
    
    class PromptBuilder:
        def __init__(self, template_dir: str = None):
            self.thinking_chains = {}
            
        def create_thinking_chain(self, chain_id: str, description: str):
            self.thinking_chains[chain_id] = {
                "id": chain_id,
                "description": description,
                "steps": [],
                "created_at": datetime.now().isoformat(),
                "completed": False
            }
            return chain_id
            
        def add_thinking_step(self, chain_id: str, description: str, reasoning: str = "", expected_output: str = ""):
            if chain_id not in self.thinking_chains:
                raise PromptBuilderError(f"Chain {chain_id} not found")
            
            step_number = len(self.thinking_chains[chain_id]["steps"]) + 1
            step = SequentialThinkingStep(step_number, description, reasoning, expected_output)
            self.thinking_chains[chain_id]["steps"].append(step.to_dict())
            return step
            
        def get_thinking_chain(self, chain_id: str):
            return self.thinking_chains.get(chain_id)
            
        def list_thinking_chains(self):
            return list(self.thinking_chains.values())
            
        def list_templates(self):
            return ["sequential_thinking", "business_rule_generation"]
            
        def build_sequential_thinking_prompt(self, chain_id: str, main_task: str, context: str = "", additional_instructions: str = ""):
            chain = self.thinking_chains.get(chain_id)
            if not chain:
                raise PromptBuilderError(f"Chain {chain_id} not found")
                
            prompt = f"Task: {main_task}\n\n"
            if context:
                prompt += f"Context: {context}\n\n"
            
            prompt += "Please follow these sequential thinking steps:\n\n"
            for i, step in enumerate(chain["steps"], 1):
                prompt += f"Step {i}: {step['description']}\n"
                if step['reasoning']:
                    prompt += f"Reasoning: {step['reasoning']}\n"
                if step['expected_output']:
                    prompt += f"Expected Output: {step['expected_output']}\n"
                prompt += "\n"
                
            if additional_instructions:
                prompt += f"\nAdditional Instructions: {additional_instructions}"
                
            return prompt
            
        def build_business_rule_with_thinking(self, context: str, requirements: str, rule_id: str = None, examples: List[str] = None):
            prompt = f"Generate business rules based on the following:\n\n"
            prompt += f"Context: {context}\n\n"
            prompt += f"Requirements: {requirements}\n\n"
            
            if examples:
                prompt += "Examples:\n"
                for example in examples:
                    prompt += f"- {example}\n"
                prompt += "\n"
                
            if rule_id:
                prompt += f"Rule ID: {rule_id}\n\n"
                
            prompt += "Please provide a well-structured business rule with clear conditions and actions."
            return prompt
            
        def apply_template(self, template_name: str, variables: Dict[str, Any]):
            # Simple template application
            if template_name == "sequential_thinking":
                return f"Sequential thinking for: {variables.get('task', 'Unknown task')}"
            elif template_name == "business_rule_generation":
                return f"Business rule for: {variables.get('context', 'Unknown context')}"
            else:
                return f"Template {template_name} with variables: {variables}"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Sequential Thinker MCP Server")

# Global components
prompt_builder: Optional[PromptBuilder] = None


class ThinkingChainRequest(BaseModel):
    """Request model for creating thinking chains."""
    chain_id: str = Field(description="Unique identifier for the thinking chain")
    description: str = Field(description="Description of the thinking process")
    main_task: str = Field(description="The main task to solve")
    context: str = Field(default="", description="Additional context for the task")


class ThinkingStepRequest(BaseModel):
    """Request model for adding thinking steps."""
    chain_id: str = Field(description="ID of the thinking chain")
    description: str = Field(description="Description of this thinking step")
    reasoning: str = Field(default="", description="Reasoning behind this step")
    expected_output: str = Field(default="", description="Expected output from this step")


class SequentialThinkingRequest(BaseModel):
    """Request model for sequential thinking generation."""
    task: str = Field(description="The main task to solve")
    context: str = Field(default="", description="Additional context")
    steps: List[Dict[str, str]] = Field(default=[], description="Predefined thinking steps")
    additional_instructions: str = Field(default="", description="Additional instructions")


class CustomPromptRequest(BaseModel):
    """Request model for custom prompt building."""
    template_name: str = Field(description="Name of the template to use")
    variables: Dict[str, Any] = Field(description="Variables to substitute in template")


@mcp.resource("sequential-thinking://chains")
async def list_thinking_chains() -> str:
    """List all available thinking chains."""
    if not prompt_builder:
        return "Sequential thinking not initialized"
    
    chains = prompt_builder.list_thinking_chains()
    
    if not chains:
        return "No thinking chains available"
    
    result = "Available Thinking Chains:\n\n"
    for chain in chains:
        result += f"ID: {chain['id']}\n"
        result += f"Description: {chain['description']}\n"
        result += f"Steps: {chain['step_count']}\n"
        result += f"Created: {chain['created_at']}\n"
        result += f"Completed: {chain['completed']}\n\n"
    
    return result


@mcp.resource("sequential-thinking://chain/{chain_id}")
async def get_thinking_chain(chain_id: str) -> str:
    """Get details of a specific thinking chain."""
    if not prompt_builder:
        return "Sequential thinking not initialized"
    
    chain = prompt_builder.get_thinking_chain(chain_id)
    
    if not chain:
        return f"Thinking chain '{chain_id}' not found"
    
    result = f"Thinking Chain: {chain['id']}\n\n"
    result += f"Description: {chain['description']}\n"
    result += f"Created: {chain['created_at']}\n"
    result += f"Completed: {chain['completed']}\n\n"
    result += "Steps:\n"
    
    for step in chain['steps']:
        result += f"\nStep {step['step_number']}: {step['description']}\n"
        if step['reasoning']:
            result += f"Reasoning: {step['reasoning']}\n"
        if step['expected_output']:
            result += f"Expected Output: {step['expected_output']}\n"
        result += f"Completed: {step['completed']}\n"
    
    return result


@mcp.resource("sequential-thinking://templates")
async def list_templates() -> str:
    """List all available prompt templates."""
    if not prompt_builder:
        return "Sequential thinking not initialized"
    
    templates = prompt_builder.list_templates()
    
    if not templates:
        return "No templates available"
    
    result = "Available Templates:\n\n"
    for template in templates:
        result += f"- {template}\n"
    
    return result


@mcp.tool()
async def create_thinking_chain(request: ThinkingChainRequest) -> Dict[str, Any]:
    """Create a new sequential thinking chain."""
    if not prompt_builder:
        raise Exception("Sequential thinking not initialized")
    
    try:
        chain_id = prompt_builder.create_thinking_chain(
            request.chain_id, 
            request.description
        )
        
        return {
            "success": True,
            "chain_id": chain_id,
            "description": request.description,
            "message": f"Created thinking chain: {chain_id}"
        }
    
    except Exception as e:
        logger.error(f"Error creating thinking chain: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to create thinking chain"
        }


@mcp.tool()
async def add_thinking_step(request: ThinkingStepRequest) -> Dict[str, Any]:
    """Add a step to an existing thinking chain."""
    if not prompt_builder:
        raise Exception("Sequential thinking not initialized")
    
    try:
        step = prompt_builder.add_thinking_step(
            request.chain_id,
            request.description,
            request.reasoning,
            request.expected_output
        )
        
        return {
            "success": True,
            "step_number": step.step_number,
            "description": step.description,
            "chain_id": request.chain_id,
            "message": f"Added step {step.step_number} to chain {request.chain_id}"
        }
    
    except Exception as e:
        logger.error(f"Error adding thinking step: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to add thinking step"
        }


@mcp.tool()
async def generate_sequential_thinking_prompt(request: SequentialThinkingRequest) -> Dict[str, Any]:
    """Generate a sequential thinking prompt for structured reasoning."""
    if not prompt_builder:
        raise Exception("Sequential thinking not initialized")
    
    try:
        # Create a new thinking chain for this request
        chain_id = f"sequential_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        prompt_builder.create_thinking_chain(chain_id, "Auto-generated sequential thinking")
        
        # Add predefined steps or default steps
        if request.steps:
            for step_data in request.steps:
                prompt_builder.add_thinking_step(
                    chain_id,
                    step_data.get("description", ""),
                    step_data.get("reasoning", ""),
                    step_data.get("expected_output", "")
                )
        else:
            # Default thinking steps for general problem solving
            default_steps = [
                {
                    "description": "Problem Analysis",
                    "reasoning": "Understand the problem, identify key components and constraints",
                    "expected_output": "Clear problem definition with identified variables and constraints"
                },
                {
                    "description": "Solution Planning", 
                    "reasoning": "Develop a systematic approach to solve the problem",
                    "expected_output": "Step-by-step solution strategy"
                },
                {
                    "description": "Implementation Design",
                    "reasoning": "Design the specific implementation approach",
                    "expected_output": "Detailed implementation plan with specific actions"
                },
                {
                    "description": "Validation and Testing",
                    "reasoning": "Ensure the solution is correct and handles edge cases",
                    "expected_output": "Validation criteria and test cases"
                },
                {
                    "description": "Final Solution",
                    "reasoning": "Synthesize all previous steps into the final answer",
                    "expected_output": "Complete, validated solution"
                }
            ]
            
            for step_data in default_steps:
                prompt_builder.add_thinking_step(
                    chain_id,
                    step_data["description"],
                    step_data["reasoning"],
                    step_data["expected_output"]
                )
        
        # Generate the sequential thinking prompt
        prompt = prompt_builder.build_sequential_thinking_prompt(
            chain_id=chain_id,
            main_task=request.task,
            context=request.context,
            additional_instructions=request.additional_instructions
        )
        
        return {
            "success": True,
            "prompt": prompt,
            "chain_id": chain_id,
            "step_count": len(prompt_builder.thinking_chains[chain_id]["steps"]),
            "message": "Generated sequential thinking prompt"
        }
    
    except Exception as e:
        logger.error(f"Error generating sequential thinking prompt: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to generate sequential thinking prompt"
        }


@mcp.tool()
async def build_custom_prompt(request: CustomPromptRequest) -> Dict[str, Any]:
    """Build a custom prompt using available templates."""
    if not prompt_builder:
        raise Exception("Sequential thinking not initialized")
    
    try:
        prompt = prompt_builder.apply_template(
            request.template_name,
            request.variables
        )
        
        return {
            "success": True,
            "prompt": prompt,
            "template": request.template_name,
            "variables_used": list(request.variables.keys()),
            "message": f"Generated prompt using template: {request.template_name}"
        }
    
    except Exception as e:
        logger.error(f"Error building custom prompt: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to build custom prompt"
        }


@mcp.tool()
async def create_business_rule_thinking_prompt(
    context: str,
    requirements: str,
    rule_id: str = None,
    examples: List[str] = None
) -> Dict[str, Any]:
    """Create a business rule generation prompt with sequential thinking."""
    if not prompt_builder:
        raise Exception("Sequential thinking not initialized")
    
    try:
        prompt = prompt_builder.build_business_rule_with_thinking(
            context=context,
            requirements=requirements,
            rule_id=rule_id,
            examples=examples or []
        )
        
        return {
            "success": True,
            "prompt": prompt,
            "context": context,
            "requirements": requirements,
            "rule_id": rule_id,
            "message": "Generated business rule thinking prompt"
        }
    
    except Exception as e:
        logger.error(f"Error creating business rule thinking prompt: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to create business rule thinking prompt"
        }


@mcp.tool()
async def get_server_info() -> Dict[str, Any]:
    """Get information about the Sequential Thinker MCP server."""
    available_templates = []
    active_chains = 0
    
    if prompt_builder:
        available_templates = prompt_builder.list_templates()
        active_chains = len(prompt_builder.thinking_chains)
    
    return {
        "server_name": "Sequential Thinker MCP Server",
        "version": "1.0.0",
        "description": "Provides structured sequential thinking capabilities for complex reasoning tasks",
        "capabilities": [
            "Sequential thinking chain creation",
            "Step-by-step reasoning prompts",
            "Business rule generation with thinking",
            "Custom prompt building",
            "Template management"
        ],
        "available_templates": available_templates,
        "active_thinking_chains": active_chains,
        "initialized": prompt_builder is not None
    }


@mcp.tool()
async def validate_thinking_chain(chain_id: str) -> Dict[str, Any]:
    """Validate a thinking chain for completeness and logical flow."""
    if not prompt_builder:
        raise Exception("Sequential thinking not initialized")
    
    try:
        chain = prompt_builder.get_thinking_chain(chain_id)
        
        if not chain:
            return {
                "success": False,
                "error": f"Chain '{chain_id}' not found",
                "valid": False
            }
        
        steps = chain["steps"]
        validation_results = {
            "valid": True,
            "issues": [],
            "suggestions": [],
            "step_count": len(steps),
            "completeness_score": 0
        }
        
        if not steps:
            validation_results["valid"] = False
            validation_results["issues"].append("No steps defined in thinking chain")
            return {
                "success": True,
                "validation": validation_results,
                "chain_id": chain_id
            }
        
        # Check for logical flow
        step_descriptions = [step["description"] for step in steps]
        if len(set(step_descriptions)) != len(step_descriptions):
            validation_results["issues"].append("Duplicate step descriptions found")
        
        # Check step completeness
        complete_steps = sum(1 for step in steps if step.get("reasoning") and step.get("expected_output"))
        validation_results["completeness_score"] = complete_steps / len(steps) if steps else 0
        
        if validation_results["completeness_score"] < 0.8:
            validation_results["suggestions"].append("Consider adding more detailed reasoning and expected outputs")
        
        # Check for recommended minimum steps
        if len(steps) < 3:
            validation_results["suggestions"].append("Consider adding more steps for better problem decomposition")
        
        return {
            "success": True,
            "validation": validation_results,
            "chain_id": chain_id,
            "message": f"Validated thinking chain: {chain_id}"
        }
    
    except Exception as e:
        logger.error(f"Error validating thinking chain: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to validate thinking chain"
        }


async def setup_and_cleanup():
    """Initialize and cleanup server components."""
    global prompt_builder
    
    # Initialize prompt builder with templates from parent project
    try:
        template_dir = Path(__file__).parent.parent / "mcp-server" / "templates"
        prompt_builder = PromptBuilder(str(template_dir))
        logger.info("Sequential Thinker MCP Server initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize prompt builder: {e}")
        # Create with default directory if parent templates not available
        prompt_builder = PromptBuilder()
    
    yield
    
    # Cleanup
    logger.info("Sequential Thinker MCP Server shutting down")


def main():
    """Run the Sequential Thinker MCP server."""
    import argparse
    global prompt_builder
    
    parser = argparse.ArgumentParser(description="Sequential Thinker MCP Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Initialize prompt builder
    try:
        template_dir = Path(__file__).parent.parent / "mcp-server" / "templates"
        prompt_builder = PromptBuilder(str(template_dir))
        logger.info("Sequential Thinker MCP Server initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize prompt builder: {e}")
        # Create with default directory if parent templates not available
        prompt_builder = PromptBuilder()
    
    # Run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()