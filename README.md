# MCP Demo: AI Agent Creation Ecosystem

A comprehensive demonstration of the Model Context Protocol (MCP) ecosystem for building intelligent AI agents. This project showcases a complete workflow from configuration to deployment using multiple MCP servers working together.

## 🎯 What This Demo Shows

This demo demonstrates a **real-world AI agent creation pipeline** using:

1. **JSON-based Configuration**: Everything is configurable through JSON files
2. **Step-by-Step Workflow**: Guided process from concept to deployment
3. **MCP Server Orchestration**: Multiple servers working together seamlessly
4. **Memory Configuration**: Persistent conversation and context management
5. **Sequential Thinking**: Structured reasoning for better AI responses
6. **Prompt Engineering**: AI-powered prompt optimization
7. **Live Agent Testing**: Interactive testing of created agents

## 🚀 Quick Start

### Recommended Setup (Verified Working)

If you encounter issues with the automated scripts, use this manual approach:

```bash
# 1. Start MCP Server (Python 3.12 recommended)
cd mcp-server
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
REDIS_HOST=none CHROMA_HOST=none uvicorn main:app --host 0.0.0.0 --port 8000 --reload &

# 2. Start Frontend Client
cd ../mcp-client
npm install
PORT=3001 npm run dev &

# 3. Access the demo
open http://localhost:3001
```

### Automated Scripts (May Require Troubleshooting)

```bash
# Complete ecosystem (requires all services)
./run-complete-demo.sh --open

# Simplified setup (requires Docker)
./start-local-simple.sh

# Original business rule demo
./run-services.sh
```

**Note**: If automated scripts fail, use the "Recommended Setup" above. See [Troubleshooting](#-troubleshooting) section for common issues and solutions.

## 📚 Complete Documentation

**For detailed step-by-step instructions, demo scripts, and troubleshooting:**
👉 **[COMPLETE_DEMO_GUIDE.md](./COMPLETE_DEMO_GUIDE.md)** 👈

This comprehensive guide includes:
- 🎬 **Complete demo walkthrough** for creating a Telco business rules agent
- 🚀 **System setup and management** commands
- 🎯 **Business presentation scripts** and talking points
- 🔧 **Troubleshooting guide** and technical requirements
- 📊 **Demo variations** for different business domains

## 🎯 Key Features

### ✨ Guided Demo Experience
- **5-step interactive tutorial** from setup to testing
- **Pre-filled scenarios** for telecommunications, e-commerce, and fintech
- **User-controlled progression** through each step
- **Real-time rule generation** with AI assistance

### 🤖 Business Rule Agent Creation
- **Custom agent configuration** with business context
- **Template-based scenarios** with editable parameters
- **Sequential thinking integration** for complex reasoning
- **Interactive testing and validation** of generated rules

### 🔧 Complete Rule Development Workflow
- **Service health checks** and connection validation
- **Business context definition** with industry templates
- **AI agent creation** with custom parameters
- **Rule generation** with multiple scenario options
- **Testing and validation** with realistic test cases

## 📋 Demo Scenarios

### Business Context Templates
1. **Telecommunications Company** - B2B enterprise services with volume discounts
2. **E-commerce Platform** - Multi-vendor marketplace with dynamic pricing
3. **Financial Services** - Digital banking with compliance requirements

### Rule Generation Scenarios
1. **Volume-Based Discounts** - Quantity-based pricing with customer tiers
2. **Customer Retention Rules** - At-risk customer management with interventions
3. **Fraud Detection Rules** - Transaction monitoring with risk scoring

### Testing Scenarios
- **High Volume Customer** - Enterprise customer with large orders
- **Medium Volume Regular** - Standard customer with moderate purchases
- **Edge Cases** - Minimum thresholds and exception handling

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LM Studio     │    │   MCP Server    │    │   Next.js UI    │
│   (Local AI)    │◄──►│  (Python/Fast)  │◄──►│ (Guided Demo)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ OpenAI-Compatible│    │ Sequential      │    │ Interactive     │
│ API Interface   │    │ Thinking Engine │    │ Step-by-Step    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components
- **LLM Provider**: Unified interface with LM Studio integration
- **Prompt Builder**: Sequential thinking with structured reasoning
- **Vector Store**: Semantic context retrieval and similarity search  
- **Session Storage**: Conversation history and context preservation
- **MCP Protocol**: Real-time WebSocket communication

## 🎬 User-Controlled Demo Features

### Interactive Step-by-Step Process
- **Setup & Connect**: Service health checks and connection validation
- **Define Context**: Business domain selection with pre-filled templates
- **Create Agent**: AI agent configuration with custom parameters
- **Generate Rules**: Business rule creation with scenario templates
- **Test & Validate**: Rule testing with realistic business scenarios

### Hands-On Learning Experience
- **Click-to-proceed**: Each step requires user interaction
- **Editable templates**: Pre-filled content that can be customized
- **Real-time feedback**: Immediate results from AI processing
- **Complete workflow**: From concept to tested business rule

## 🛠️ Technical Stack

### Backend (MCP Server)
- **Python 3.9+** with FastAPI
- **Pydantic v2** for data validation
- **WebSocket** support for real-time communication
- **Jinja2** templating for dynamic prompts
- **Sequential thinking** engine with step tracking

### Frontend (MCP Client)  
- **Next.js 14** with App Router
- **TypeScript** for type safety
- **Tailwind CSS** for styling
- **Real-time WebSocket** client
- **Interactive demo** components

### AI Integration
- **LM Studio** for local model serving
- **OpenAI-compatible** API interface
- **Model flexibility** (Llama, Mistral, etc.)
- **Streaming responses** support

## 📁 Project Structure

```
mcp-demo/
├── run-services.sh             # 🚀 Start all services
├── start-local-simple.sh       # 🔧 Manual startup
├── stop-local-simple.sh        # 🛑 Clean shutdown
├── CLAUDE.md                   # 📚 Development guide
├── README.md                   # 📖 This file
│
├── mcp-server/                 # 🐍 Python FastAPI Server
│   ├── components/            # Core GenAI components
│   │   ├── business_rules.py  # Rule generation engine
│   │   ├── prompt_builder.py  # Sequential thinking
│   │   ├── llm_provider.py   # LM Studio integration
│   │   └── ...
│   ├── mcp/                  # MCP protocol implementation
│   └── templates/            # Prompt templates
│
├── mcp-client/               # ⚛️ Next.js Frontend
│   ├── src/app/
│   │   ├── demo/            # 🔧 Demo pages
│   │   │   ├── guided/      # 📚 Guided demo
│   │   │   └── page.tsx     # 🎯 Main demo page
│   │   └── page.tsx         # 🏠 Home page
│   ├── src/components/
│   │   ├── demo/            # 📊 Demo components
│   │   └── business-rules/  # 📝 Rule components
│   └── src/lib/             # 🔗 MCP client library
│
└── config/                  # ⚙️ Shared configuration
    └── mcp-settings.json
```

## 🎮 Usage

### Guided Interactive Demo
```bash
# Start all required services
./run-services.sh

# Open guided demo in browser  
open http://localhost:3000/demo/guided
```

### Manual Demo Tools
```bash
# Start services only
./start-local-simple.sh

# Access manual demo tools
open http://localhost:3000/demo
```

## 🔧 Prerequisites & Setup

### Required Software
- **LM Studio**: Download from https://lmstudio.ai
- **Python 3.9+**: For MCP server
- **Node.js 18+**: For frontend client

### LM Studio Setup
1. Download and install LM Studio
2. Download a suitable model (e.g., Llama 2 7B, Mistral 7B)
3. Load the model and start server on localhost:1234
4. Verify at: http://localhost:1234/v1/models

### Automatic Configuration
All other configuration is handled automatically by the startup scripts:
- **Python virtual environment** creation and activation
- **Dependency installation** for both server and client
- **Environment variable** setup and validation
- **Service health checks** and startup verification
- **Error handling** and recovery procedures

## 📊 Demo Analytics

### Step Completion Tracking
- **Progress Visualization**: Clear progress through 5-step process
- **Service Health**: Connection status for all required services
- **Generation Metrics**: Rule creation time and success rates
- **Test Results**: Validation outcomes for generated rules

### Learning Insights
- **Template Usage**: Which business scenarios are most popular
- **Agent Configuration**: Common parameter settings
- **Rule Complexity**: Generated rule structure analysis
- **User Journey**: How users progress through the demo

## 🚨 Troubleshooting

### Quick Diagnosis
```bash
# Check LM Studio
curl http://localhost:1234/v1/models

# Check MCP Server
curl http://localhost:8000/health

# Check Frontend
curl http://localhost:3000

# Complete reset
./stop-local-simple.sh && ./start-local-simple.sh
```

### Common Issues & Solutions

#### 1. **Services Fail to Start (ModuleNotFoundError)**
**Problem**: Missing dependencies or import errors in MCP services

**Solution**: Use the manual startup approach with proper virtual environments:
```bash
# Setup MCP Server
cd mcp-server
python3.12 -m venv .venv  # Use Python 3.12 (avoid 3.13 compatibility issues)
source .venv/bin/activate
pip install -r requirements.txt

# Start MCP Server with in-memory storage (no Docker required)
REDIS_HOST=none CHROMA_HOST=none uvicorn main:app --host 0.0.0.0 --port 8000 --reload &

# Setup Frontend Client
cd ../mcp-client
npm install
PORT=3001 npm run dev &  # Use port 3001 if 3000 is busy
```

#### 2. **Sequential Thinker Import Errors**
**Problem**: Sequential Thinker service can't import required components

**Solution**: The Sequential Thinker has been fixed with fallback implementations:
```bash
cd sequential-thinker-mcp
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py --port 8001 &
```

#### 3. **Python 3.13 Compatibility Issues**
**Problem**: Pydantic-core build failures with Python 3.13

**Solution**: Use Python 3.12 or 3.11 instead:
```bash
# Check available Python versions
python3.12 --version || python3.11 --version

# Use compatible Python version
python3.12 -m venv .venv
```

#### 4. **Port Conflicts**
**Problem**: Ports already in use

**Solution**: Kill existing processes and use alternative ports:
```bash
# Check what's using port 3000
lsof -i :3000

# Kill process if needed
kill <PID>

# Or use alternative port
PORT=3001 npm run dev
```

#### 5. **Docker Not Available**
**Problem**: start-local-simple.sh requires Docker for Redis/ChromaDB

**Solution**: Use in-memory storage instead:
```bash
# Skip Docker and use memory storage
cd mcp-server
source .venv/bin/activate
REDIS_HOST=none CHROMA_HOST=none uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### 6. **Service Health Check Failures**
**Problem**: Health checks failing because services use different transports

**Solution**: Check services directly:
```bash
# Test MCP Server
curl http://localhost:8000/health
# Should return: {"status":"healthy","server":"genai-mcp-server","version":"1.0.0"}

# Test Frontend
curl http://localhost:3001
# Should return HTML content

# Services are working if you see proper responses
```

### Verified Working Setup

After troubleshooting, this configuration is confirmed working:

```bash
# 1. MCP Server (Python 3.12 + in-memory storage)
cd mcp-server
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
REDIS_HOST=none CHROMA_HOST=none uvicorn main:app --host 0.0.0.0 --port 8000 --reload &

# 2. Frontend Client (Node.js on port 3001)
cd mcp-client
npm install
PORT=3001 npm run dev &
```

**Access Points:**
- **Main Demo**: http://localhost:3001
- **MCP Server API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### Service Status Verification

✅ **Working Services:**
- MCP Server: All components initialized (LLM Provider, Prompt Builder, Session Storage, Vector Store, Business Rules)
- Frontend: Next.js app with all demo features
- Memory Storage: Fallback storage working without Redis/ChromaDB
- LM Studio Integration: Ready (when LM Studio is running)

### Advanced Troubleshooting

#### Complete Service Restart
```bash
# Kill all related processes
pkill -f "uvicorn\|node.*dev"

# Remove old virtual environments
rm -rf mcp-server/.venv sequential-thinker-mcp/venv

# Clean restart with verified setup
# Follow the "Verified Working Setup" steps above
```

#### Debug Mode
```bash
# Start MCP server with debug logging
DEBUG=true LOG_LEVEL=debug uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Check detailed logs for issues
tail -f logs/*.log
```

## 🎯 Demo Highlights

### What You'll Experience
- **🔧 Hands-On Learning**: Interactive step-by-step tutorial
- **🎯 Business Focus**: Real-world scenarios with practical applications
- **🤖 AI Collaboration**: Work with AI to create intelligent rules
- **📋 JSON Output**: Production-ready business rule format
- **✅ Validation**: Test your rules with realistic scenarios

### Business Value Demonstration
- **⚡ Rapid Prototyping**: Create and test rules in minutes
- **🎯 Domain Expertise**: AI understands business context and constraints
- **📋 Integration Ready**: Standard JSON format for easy implementation
- **🔄 Iterative Process**: Refine rules through testing and validation
- **📈 Scalable Approach**: Methodology works across industries

## 📚 API Documentation

### MCP Server Endpoints
- **Health Check**: `GET /health`
- **Business Rules**: `POST /api/v1/business-rules/generate`
- **Rule Validation**: `POST /api/v1/business-rules/validate`
- **Context Search**: `POST /api/v1/context/search`
- **WebSocket**: `ws://localhost:8000/mcp`
- **API Docs**: `GET /docs` (FastAPI auto-generated)

### Example API Usage
```bash
# Generate business rule
curl -X POST http://localhost:8000/api/v1/business-rules/generate \
  -H "Content-Type: application/json" \
  -d '{
    "context": "Enterprise telecom provider",
    "requirements": "Volume-based pricing rules",
    "metadata": {"use_sequential_thinking": true}
  }'
```

## 🔄 Development

### Adding New Demo Scenarios
1. Update business context templates in `GuiDedDemo.tsx`
2. Add new rule generation scenarios with requirements
3. Create test cases for validation step
4. Update prompt templates in `mcp-server/templates/`

### Custom Prompt Templates
Create Jinja2 templates in `mcp-server/templates/`:
```jinja2
You are an expert {{ domain }} analyst.

MAIN TASK: {{ main_task }}

SEQUENTIAL THINKING PROCESS:
{% for step in thinking_steps %}
STEP {{ step.step_number }}: {{ step.description }}
{% endfor %}

Begin with STEP 1 and work through each step systematically.
```

## 📝 License

MIT License - Built for demonstration and educational purposes.

## 🙏 Acknowledgments

- [Model Context Protocol](https://github.com/modelcontextprotocol) for the MCP specification
- [LM Studio](https://lmstudio.ai) for local AI model serving
- [FastAPI](https://fastapi.tiangolo.com/) for the Python web framework
- [Next.js](https://nextjs.org/) for the React framework

---

🚀 **Ready to create business rules with AI?** Run `./run-services.sh` and visit http://localhost:3000/demo/guided