# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MCP (Model Context Protocol) GenAI Demo featuring a multi-tier architecture with **guided interactive demo**, **user-controlled rule generation**, and **step-by-step learning experience**:
- **Python FastAPI MCP Server**: GenAI components hub with LLM provider, prompt builder, session storage, and vector store
- **Node.js MCP Client**: Protocol implementation with Next.js frontend for business rule generation
- **LM Studio Integration**: Local AI model serving with OpenAI-compatible API
- **Interactive Demo System**: User-controlled step-by-step guided experience

## Development Commands

### Quick Start (Recommended)
```bash
# Start all services for guided demo
./run-services.sh

# Manual services startup
./start-local-simple.sh

# Clean shutdown
./stop-local-simple.sh
```

### Manual Development Setup

#### MCP Server (Python/FastAPI)
```bash
cd mcp-server
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

#### MCP Client (Node.js/Next.js)
```bash
cd mcp-client
npm install
npm run dev          # Development server
npm run build        # Production build
npm run type-check   # TypeScript validation
npm run lint         # ESLint validation
```

#### Testing
```bash
# Python server tests
cd mcp-server
pytest tests/test_mcp_server.py -v

# Run specific test
pytest tests/test_mcp_server.py::TestMCPServer::test_business_rule_generation -v

# Test with coverage
pytest --cov=components --cov=mcp tests/
```

### LM Studio Setup
1. Download and install LM Studio from https://lmstudio.ai
2. Download a compatible model (Llama 2 7B, Mistral 7B, etc.)
3. Load model and start server on localhost:1234
4. Verify: `curl http://localhost:1234/v1/models`

## Architecture Deep Dive

### Sequential Thinking Engine
The core innovation is the **SequentialThinkingStep** system in `prompt_builder.py`:
- **5-step reasoning process**: Context Analysis → Requirements Parsing → Logic Design → Business Optimization → JSON Formatting
- **Real-time visualization**: Each thinking step is tracked and displayed live
- **Template-driven prompts**: Jinja2 templates with thinking chain integration
- **Performance metrics**: Step timing and completion tracking

### MCP Protocol Implementation
**Server Side (Python)**:
- `mcp/server.py`: Core MCP server with component orchestration
- `mcp/handlers.py`: Protocol method handlers (initialize, list_tools, call_tool)
- **Message Flow**: WebSocket → JSON-RPC 2.0 → Component dispatch → Response

**Client Side (TypeScript)**:
- `lib/mcp-client.ts`: WebSocket-based MCP client with reconnection logic
- `lib/types.ts`: Complete TypeScript interface definitions
- **Connection Management**: Auto-reconnect with exponential backoff

### Business Rule Generation Pipeline
1. **Context Analysis**: Extract domain knowledge and constraints
2. **Sequential Thinking**: 5-step reasoning process with live visualization  
3. **Template Selection**: Dynamic prompt construction with Jinja2
4. **LLM Generation**: LM Studio integration with streaming support
5. **Storage & Retrieval**: Session persistence and vector search

### Interactive Demo System Architecture
**Guided Demo Flow**:
- `run-services.sh`: Starts all required services
- `src/components/demo/GuiDedDemo.tsx`: 5-step interactive tutorial
- **User-Controlled Progression**: Each step requires user interaction
- **Pre-filled Templates**: Business scenarios with editable content

**Demo Components**:
- **Setup & Connect**: Service health checks and validation
- **Define Context**: Business domain selection with templates
- **Create Agent**: AI agent configuration with parameters
- **Generate Rules**: Rule creation with scenario templates
- **Test & Validate**: Rule testing with realistic scenarios

### Critical Data Flow Patterns

#### Pydantic v2 Serialization
**Always use** `.model_dump(mode='json')` instead of `.dict()`:
```python
# Correct
await self.session_storage.update_session(
    session.session_id,
    request.model_dump(mode='json')
)

# Incorrect (causes datetime errors)
await self.session_storage.update_session(
    session.session_id,
    request.dict()
)
```

#### LM Studio Model Selection
The `lmstudio_provider.py` filters embedding models for chat completions:
```python
# Filter out embedding models for chat
chat_models = [m for m in available_models 
               if not any(keyword in m.lower() 
                         for keyword in ['embed', 'embedding'])]
```

#### Next.js Hydration Handling
For timestamp displays, use client-side mounting check:
```typescript
const [isMounted, setIsMounted] = useState(false)
useEffect(() => setIsMounted(true), [])

// In render
{isMounted ? component.lastUpdate.toLocaleTimeString() : '--:--:--'}
```

## Component Configuration

### Storage Backend Selection
The server automatically selects storage based on environment:
- **Development**: Memory storage (no Redis required)
- **Production**: Redis for session persistence
- **Vector Store**: ChromaDB if available, else in-memory

### LLM Provider Configuration
Supports multiple providers with consistent interface:
- **LM Studio**: Local models via OpenAI-compatible API
- **OpenAI**: Cloud API with key authentication
- **Anthropic**: Claude models with key authentication  
- **Local**: Mock provider for testing

## Key File Locations

### Core Server Components
- `mcp-server/components/business_rules.py`: Main rule generation engine
- `mcp-server/components/prompt_builder.py`: Sequential thinking and templates
- `mcp-server/components/lmstudio_provider.py`: Local AI integration
- `mcp-server/components/session_storage.py`: Conversation persistence
- `mcp-server/components/vector_store.py`: Semantic search and context

### Frontend Components  
- `mcp-client/src/components/business-rules/RuleGenerator.tsx`: Main demo interface with prepopulated scenarios
- `mcp-client/src/components/demo/GuiDedDemo.tsx`: Interactive guided demo system
- `mcp-client/src/app/demo/guided/page.tsx`: Guided demo page
- `mcp-client/src/lib/mcp-client.ts`: WebSocket client with MCP protocol

### Demo and Configuration
- `TELCO_DEMO_SCRIPT.md`: Comprehensive demo scenarios for telecommunications
- `config/mcp-settings.json`: Shared configuration for both server and client
- `run-services.sh`: Service startup script for guided demo

## Service Access Points
- **Guided Demo**: http://localhost:3000/demo/guided
- **Manual Demo**: http://localhost:3000/demo  
- **Home Page**: http://localhost:3000
- **MCP Server API**: http://localhost:8000
- **FastAPI Docs**: http://localhost:8000/docs
- **MCP WebSocket**: ws://localhost:8000/mcp
- **LM Studio API**: http://localhost:1234

## Demo Scenarios

### Business Context Templates
1. **Telecommunications Company**: B2B enterprise services with volume discounts
2. **E-commerce Platform**: Multi-vendor marketplace with dynamic pricing
3. **Financial Services**: Digital banking with compliance requirements

### Rule Generation Scenarios
1. **Volume-Based Discounts**: Quantity-based pricing with customer tiers
2. **Customer Retention Rules**: At-risk customer management with interventions
3. **Fraud Detection Rules**: Transaction monitoring with risk scoring

### Testing Scenarios
- **High Volume Customer**: Enterprise customer with large orders
- **Medium Volume Regular**: Standard customer with moderate purchases
- **Edge Cases**: Minimum thresholds and exception handling

Each scenario features:
- **Pre-filled templates** with realistic business data
- **Editable parameters** for customization
- **Step-by-step progression** requiring user interaction
- **Real-time AI processing** with immediate feedback

## Error Handling Patterns

### MCP Protocol Errors
Return proper JSON-RPC 2.0 error responses:
```python
{
    "jsonrpc": "2.0",
    "error": {"code": -32603, "message": "Internal error"},
    "id": message_id
}
```

### WebSocket Connection Management
Client implements exponential backoff with maximum retry attempts and graceful degradation when server unavailable.

### Component Failure Recovery
Each component (LLM, Vector Store, Session Storage) has fallback mechanisms and graceful degradation paths to maintain demo functionality.

## Prerequisites

- **Python 3.9+** (virtual environment auto-configured by scripts)
- **Node.js 18+** for Next.js frontend
- **LM Studio** for local AI model serving (required)
- **Redis** (optional, memory storage available for development)
- **ChromaDB** (optional, in-memory fallback available)

## Important Notes

### User-Controlled Demo Experience
- **No Background Automation**: All demo activities require user interaction
- **Step-by-Step Progression**: Each step must be completed before proceeding
- **Pre-filled Templates**: All forms have realistic business data that can be edited
- **Real-time Processing**: AI generation happens when user clicks buttons
- **Complete Workflow**: From service setup to rule testing in 5 guided steps

### Removed Features
- **Automated Demo Scripts**: No background rule generation
- **Live Dashboard**: Removed real-time component visualization
- **Auto-progression**: Users control when to move between steps
- **Background APIs**: Removed automated demo endpoints