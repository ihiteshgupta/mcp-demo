# Sequential Thinker MCP Server

A Model Context Protocol (MCP) server that provides structured sequential thinking capabilities for complex reasoning and problem-solving tasks.

## Overview

The Sequential Thinker MCP Server enables AI systems to break down complex problems into logical, step-by-step reasoning processes. It provides tools for creating thinking chains, managing reasoning steps, and generating structured prompts that guide systematic problem-solving approaches.

## Features

### Core Capabilities
- **Sequential Thinking Chains**: Create and manage multi-step reasoning processes
- **Step-by-Step Analysis**: Break complex problems into manageable components
- **Business Rule Generation**: Apply structured thinking to business logic creation
- **Custom Prompt Building**: Use templates to generate tailored reasoning prompts
- **Validation Tools**: Verify thinking chain completeness and logical flow

### MCP Integration
- Full MCP protocol compliance (version 2024-11-05)
- Resources for viewing thinking chains and templates
- Tools for creating and managing reasoning processes
- STDIO transport support for local development
- Compatible with Claude Desktop and other MCP clients

## Installation

### Prerequisites
- Python 3.9 or higher
- MCP client (Claude Desktop, VS Code extension, etc.)

### Setup
```bash
# Clone or navigate to the sequential-thinker-mcp directory
cd sequential-thinker-mcp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Server
```bash
# Run with STDIO transport (default)
python main.py

# Run with custom log level
python main.py --log-level DEBUG
```

### MCP Client Configuration

Add to your MCP client configuration (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "sequential-thinker": {
      "command": "python",
      "args": ["/path/to/sequential-thinker-mcp/main.py"],
      "env": {}
    }
  }
}
```

## Available Tools

### Core Tools

#### `create_thinking_chain`
Create a new sequential thinking chain for structured reasoning.

**Parameters:**
- `chain_id` (string): Unique identifier for the chain
- `description` (string): Description of the thinking process
- `main_task` (string): The main task to solve
- `context` (string, optional): Additional context

**Example:**
```json
{
  "chain_id": "problem_analysis_001",
  "description": "Analyze customer retention problem",
  "main_task": "Identify why customer retention is declining",
  "context": "SaaS company with 20% churn rate increase"
}
```

#### `add_thinking_step`
Add a step to an existing thinking chain.

**Parameters:**
- `chain_id` (string): ID of the thinking chain
- `description` (string): Description of this step
- `reasoning` (string, optional): Reasoning behind the step
- `expected_output` (string, optional): Expected outcome

#### `generate_sequential_thinking_prompt`
Generate a structured reasoning prompt.

**Parameters:**
- `task` (string): The main task to solve
- `context` (string, optional): Additional context
- `steps` (array, optional): Predefined thinking steps
- `additional_instructions` (string, optional): Extra requirements

#### `create_business_rule_thinking_prompt`
Create a business rule generation prompt with sequential thinking.

**Parameters:**
- `context` (string): Business context
- `requirements` (string): Rule requirements
- `rule_id` (string, optional): Rule identifier
- `examples` (array, optional): Example scenarios

### Utility Tools

#### `validate_thinking_chain`
Validate a thinking chain for completeness and logical flow.

#### `get_server_info`
Get information about server capabilities and status.

#### `build_custom_prompt`
Build prompts using custom templates.

## Available Resources

### `sequential-thinking://chains`
List all available thinking chains with their status and metadata.

### `sequential-thinking://chain/{chain_id}`
Get detailed information about a specific thinking chain, including all steps and their completion status.

### `sequential-thinking://templates`
List all available prompt templates.

## Templates

The server includes several built-in templates:

### `sequential_thinking.jinja2`
General-purpose sequential reasoning template with customizable steps.

### `decision_making.jinja2`
Structured decision-making framework with criteria evaluation and alternative analysis.

### `research_analysis.jinja2`
Systematic research methodology template for thorough investigation and analysis.

### `business_rule_generation.jinja2`
Business rule creation template with domain analysis and logic design.

## Configuration

Edit `config.json` to customize server behavior:

```json
{
  "server": {
    "name": "Sequential Thinker MCP Server",
    "port": 8001,
    "transport": "stdio"
  },
  "thinking_chains": {
    "max_active_chains": 100,
    "default_step_count": 5,
    "auto_cleanup": true
  },
  "validation": {
    "min_steps": 3,
    "completeness_threshold": 0.8
  }
}
```

## Example Usage

### Creating a Problem-Solving Chain

```python
# 1. Create a thinking chain
create_thinking_chain({
  "chain_id": "market_analysis",
  "description": "Analyze new market opportunity",
  "main_task": "Evaluate expansion into European market",
  "context": "B2B SaaS company considering European expansion"
})

# 2. Add thinking steps
add_thinking_step({
  "chain_id": "market_analysis",
  "description": "Market Size Analysis",
  "reasoning": "Understand total addressable market in target regions",
  "expected_output": "Market size estimates with growth projections"
})

add_thinking_step({
  "chain_id": "market_analysis", 
  "description": "Competitive Landscape",
  "reasoning": "Identify key competitors and market positioning",
  "expected_output": "Competitive analysis with strengths/weaknesses"
})

# 3. Generate reasoning prompt
generate_sequential_thinking_prompt({
  "task": "Should we expand into European market?",
  "context": "Based on market analysis findings"
})
```

### Business Rule Generation

```python
create_business_rule_thinking_prompt({
  "context": "E-commerce platform with dynamic pricing needs",
  "requirements": "Create volume discount rules for B2B customers",
  "rule_id": "volume_discount_001",
  "examples": [
    "Orders over $10k get 5% discount",
    "Enterprise customers get additional 2% discount"
  ]
})
```

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Quality
```bash
# Format code
black .

# Lint code  
flake8 .

# Type checking
mypy .
```

### Adding Custom Templates

1. Create a new Jinja2 template file in the `templates/` directory
2. Use template variables for dynamic content
3. Follow the established step-by-step structure
4. Test with the `build_custom_prompt` tool

## Integration Examples

### Claude Desktop
```json
{
  "mcpServers": {
    "sequential-thinker": {
      "command": "python",
      "args": ["/path/to/sequential-thinker-mcp/main.py"],
      "env": {
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### VS Code Extension
Use with MCP-compatible VS Code extensions that support STDIO transport.

### Custom Applications
Integrate using any MCP client library:
- Python: `mcp` package
- TypeScript: `@modelcontextprotocol/sdk`
- Other languages: Community implementations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code quality checks pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- Check the configuration and logs
- Verify MCP client compatibility
- Review the examples and documentation
- Open an issue with detailed information about the problem

## Version History

- **1.0.0**: Initial release with core sequential thinking capabilities
  - Sequential thinking chain management
  - Business rule generation
  - Template-based prompt building
  - MCP protocol compliance
  - Built-in validation tools