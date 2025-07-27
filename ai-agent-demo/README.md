# AI Agent Creation Demo

An interactive demonstration of creating AI agents using the MCP (Model Context Protocol) ecosystem. This demo showcases a complete workflow from initial concept to deployed AI agent through a step-by-step guided process.

## 🎯 Demo Overview

This demo demonstrates how to:

1. **Configure JSON-based settings** for MCP server integration
2. **Define agent context** and requirements
3. **Configure memory management** for conversation persistence
4. **Set up sequential thinking** for structured reasoning
5. **Build optimized prompts** using AI-powered generation
6. **Create and test AI agents** ready for deployment

### Key Features

- **Interactive Step-by-Step Flow**: Guided process with real-time feedback
- **JSON Configuration Management**: All settings managed through configuration files
- **MCP Ecosystem Integration**: Demonstrates orchestration of multiple MCP servers
- **Live Agent Testing**: Test your created agent in real-time
- **Production-Ready Output**: Get deployment-ready agent configurations

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Demo Client   │────│ Prompt Builder   │────│  LLM Provider   │
│   (FastAPI)     │    │   MCP Server     │    │   MCP Server    │
│                 │    │     (8006)       │    │     (8002)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Demo Config    │    │Sequential Thinker│    │     Memory      │
│   (JSON)        │    │   MCP Server     │    │   MCP Server    │
│                 │    │     (8001)       │    │     (8004)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Components

1. **AI Agent Demo** (Port 3002): Interactive web interface
2. **Prompt Builder MCP Server** (Port 8006): Orchestrates agent creation
3. **LLM Provider MCP Server** (Port 8002): Handles AI text generation
4. **Sequential Thinker MCP Server** (Port 8001): Structured reasoning
5. **Memory MCP Server** (Port 8004): Conversation persistence

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- MCP servers running (optional - demo works with simulation)

### Running the Demo

1. **Start the demo**:
   ```bash
   cd ai-agent-demo
   ./start-demo.sh
   ```

2. **Access the demo**:
   Open your browser to `http://localhost:3002`

3. **Follow the guided steps**:
   - Welcome & Overview
   - Define Agent Context
   - Configure Memory Settings
   - Configure Sequential Thinking
   - Build Agent Prompt
   - Create & Test Agent

### With Full MCP Ecosystem

For the complete experience with real MCP servers:

1. **Start all MCP servers**:
   ```bash
   # From the mcp-servers directory
   ./start-all.sh
   
   # Start Prompt Builder MCP server
   cd prompt-builder-mcp
   python main.py
   ```

2. **Start the demo**:
   ```bash
   cd ai-agent-demo
   ./start-demo.sh
   ```

## 📋 Demo Flow

### Step 1: Define Agent Context
Configure your AI agent's basic information:
- **Agent Name**: What to call your agent
- **Description**: What the agent does
- **Domain**: Business area (Customer Service, Education, etc.)
- **Use Case**: Specific application
- **Target Audience**: Who will use the agent

### Step 2: Configure Memory Settings
Set up how your agent remembers information:
- **Memory Type**: Conversation, Learning Progress, Analytical, etc.
- **Persistence**: Whether memory survives sessions
- **Size Limits**: How much information to retain
- **Retention Policy**: How to manage memory when full
- **Semantic Search**: Enable intelligent memory retrieval

### Step 3: Configure Sequential Thinking
Configure reasoning capabilities:
- **Thinking Style**: Analytical, Creative, Empathetic, etc.
- **Chain of Thought**: Enable step-by-step reasoning
- **Thinking Depth**: Surface to Comprehensive
- **Reasoning Steps**: Number of thinking steps (3-7)
- **Self-Reflection**: Enable self-evaluation

### Step 4: Build Agent Prompt
Automated prompt generation using:
- Sequential thinking integration
- Memory configuration
- Domain-specific optimizations
- LLM-powered prompt enhancement

### Step 5: Create & Test Agent
Final agent creation and testing:
- Agent deployment simulation
- Interactive testing interface
- Configuration export options
- Deployment instructions

## ⚙️ Configuration

### Main Configuration: `demo-config.json`

```json
{
  "demo": {
    "name": "AI Agent Creation Demo",
    "port": 3002
  },
  "mcp_servers": {
    "prompt-builder": {
      "host": "localhost",
      "port": 8006,
      "enabled": true
    }
  },
  "agent_templates": {
    "domains": {
      "customer_service": {
        "name": "Customer Service",
        "use_cases": ["support_chat", "faq_assistant"]
      }
    }
  }
}
```

### Domain Templates

Pre-configured templates for common use cases:

- **Customer Service**: Support chat, FAQ assistance, complaint handling
- **Education**: Tutoring, course assistance, skill assessment
- **Business Analysis**: Data analysis, reporting, strategic planning
- **Creative Writing**: Content creation, storytelling, copywriting
- **Technical Support**: Troubleshooting, documentation, code review

### Memory Types

- **Conversation Memory**: General conversation flow
- **Learning Progress Memory**: Educational contexts
- **Analytical Context Memory**: Business analysis
- **Creative Context Memory**: Creative workflows
- **Technical Context Memory**: Technical support

### Thinking Styles

- **Analytical**: Data-driven, methodical approach
- **Creative**: Innovative, imaginative thinking
- **Empathetic**: Human-centered, emotional intelligence
- **Systematic**: Structured, step-by-step process
- **Pedagogical**: Teaching-focused, learning optimization

## 🎮 Demo Modes

### Simulation Mode (Default)
- Works without MCP servers
- Uses realistic simulated responses
- Perfect for demonstrations and learning

### Full Integration Mode
- Requires all MCP servers running
- Real AI-powered responses
- Production-like experience

## 🛠️ Development

### Running in Development Mode
```bash
./start-demo.sh --dev
```

### Custom Configuration
```bash
# Custom host and port
./start-demo.sh --host 0.0.0.0 --port 8080

# Skip MCP server checks
./start-demo.sh --skip-checks
```

### Project Structure
```
ai-agent-demo/
├── main.py                 # FastAPI demo application
├── demo-config.json        # Demo configuration
├── requirements.txt        # Python dependencies
├── start-demo.sh           # Startup script
├── templates/
│   └── demo.html          # Interactive demo interface
└── README.md              # This file
```

## 🔧 API Endpoints

### Demo Control
- `POST /api/demo/start` - Start new demo session
- `GET /api/demo/config` - Get demo configuration
- `GET /api/demo/health` - System health check

### Demo Steps
- `POST /api/demo/step/context` - Define agent context
- `POST /api/demo/step/memory` - Configure memory
- `POST /api/demo/step/thinking` - Configure thinking
- `POST /api/demo/step/prompt` - Build prompt
- `POST /api/demo/step/agent` - Create agent

### Session Management
- `GET /api/demo/session/{session_id}` - Get session status
- `WebSocket /ws` - Real-time updates

## 🎯 Use Cases

### Educational Demonstrations
- AI/ML course demonstrations
- Prompt engineering workshops
- MCP protocol tutorials

### Business Presentations
- Customer demos for AI services
- Proof-of-concept presentations
- Stakeholder demonstrations

### Development & Testing
- MCP ecosystem testing
- Integration validation
- Performance demonstrations

## 🚨 Troubleshooting

### MCP Servers Not Available
The demo works in simulation mode without MCP servers. For full functionality:

1. Check server status: `curl http://localhost:8006/health`
2. Start MCP servers: Follow MCP server documentation
3. Verify configuration in `demo-config.json`

### Port Already in Use
```bash
./start-demo.sh --port 8080
```

### Dependencies Issues
```bash
# Recreate virtual environment
rm -rf venv
./start-demo.sh
```

## 📊 Demo Metrics

The demo tracks:
- Session completion rates
- Step-by-step progression
- Configuration choices
- Agent creation success
- User interaction patterns

## 🔮 Future Enhancements

- **Multi-language Support**: International demonstrations
- **Custom Domain Templates**: User-defined domains
- **Advanced Agent Types**: Specialized agent configurations
- **Integration Examples**: Real deployment scenarios
- **Performance Analytics**: Detailed metrics and insights

## 📄 License

This demo is part of the MCP ecosystem demonstration project.

## 🆘 Support

For support:
1. Check this README for common issues
2. Verify MCP server status and configuration
3. Review demo-config.json settings
4. Check browser console for client-side errors

The demo is designed to be self-explanatory and user-friendly, providing an excellent introduction to AI agent creation using the MCP ecosystem.