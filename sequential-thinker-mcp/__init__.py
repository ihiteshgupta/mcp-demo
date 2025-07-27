"""
Sequential Thinker MCP Server

A Model Context Protocol (MCP) server that provides structured sequential thinking capabilities.
This server helps AI systems and users break down complex problems into logical, step-by-step
reasoning processes.

Key Features:
- Sequential thinking chain creation and management
- Step-by-step reasoning prompts
- Business rule generation with structured thinking
- Custom prompt building with templates
- Validation and quality assurance for thinking processes

Usage:
    python -m sequential_thinker_mcp.main

Or as a module:
    from sequential_thinker_mcp import SequentialThinkerMCPServer
"""

__version__ = "1.0.0"
__author__ = "Sequential Thinking Team"
__description__ = "MCP server for structured sequential thinking and reasoning"

# Export main components for external use
from .main import main, mcp

__all__ = [
    "main",
    "mcp",
    "__version__",
    "__author__", 
    "__description__"
]