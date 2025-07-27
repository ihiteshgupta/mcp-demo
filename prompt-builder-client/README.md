# Prompt Builder Client

An advanced web-based client for building AI prompts using MCP (Model Context Protocol) servers. Features integration with AWS Bedrock and sequential thinking for intelligent prompt generation.

## 🚀 Features

- **Web-based Interface**: Modern, responsive UI built with Tailwind CSS and Alpine.js
- **MCP Integration**: Seamless connection to MCP servers for LLM and sequential thinking capabilities
- **AWS Bedrock Support**: Direct integration with Amazon Bedrock for enterprise-grade AI models
- **Sequential Thinking**: Structured reasoning approach for better prompt engineering
- **Template System**: Pre-built templates for common prompt types
- **Real-time Preview**: Live preview of generated prompts
- **Optimization Tools**: Built-in prompt optimization and testing features
- **Export/Import**: Save and share prompt configurations

## 📋 Prerequisites

- Python 3.9 or higher
- Access to AWS Bedrock (for LLM functionality)
- Running MCP servers:
  - LLM Provider Server (port 8002)
  - Sequential Thinker Server (port 8001)

## 🛠️ Installation

### Quick Start

1. **Clone and navigate to the project**:
   ```bash
   cd prompt-builder-client
   ```

2. **Run the startup script**:
   ```bash
   ./start-prompt-builder.sh
   ```

   The script will automatically:
   - Check Python version
   - Create a virtual environment
   - Install dependencies
   - Check MCP server connectivity
   - Start the server

3. **Access the application**:
   Open your browser and navigate to `http://localhost:3001`

### Manual Installation

1. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server**:
   ```bash
   python main.py
   ```

## 🔧 Configuration

The client uses `config.json` for configuration. Key settings include:

### MCP Servers
```json
{
  "mcp_servers": {
    "llm_provider": {
      "host": "localhost",
      "port": 8002,
      "enabled": true
    },
    "sequential_thinker": {
      "host": "localhost", 
      "port": 8001,
      "enabled": true
    }
  }
}
```

### Prompt Generation Settings
```json
{
  "prompt_generation": {
    "default_settings": {
      "use_sequential_thinking": true,
      "model_preference": "auto",
      "temperature": 0.7,
      "max_tokens": 2000
    }
  }
}
```

## 🎯 Usage

### Basic Prompt Generation

1. **Open the application** in your browser
2. **Fill in the form**:
   - Task Description: What you want the prompt to accomplish
   - Context: Additional background information
   - Prompt Type: Select from general, creative, analytical, code, or explanation
   - Target Audience: Who will use this prompt

3. **Configure advanced options** (optional):
   - Enable/disable sequential thinking
   - Select specific AI models
   - Adjust creativity/temperature settings

4. **Generate the prompt** by clicking the "Generate Prompt" button

### Using Templates

The client includes pre-built templates for common use cases:

- **Creative Writing**: Story, poem, and creative content prompts
- **Analysis**: Data analysis and research prompts  
- **Code Generation**: Programming and technical prompts
- **Explanation**: Educational and instructional prompts

Click on any template to automatically fill the form with example values.

### Optimization and Testing

After generating a prompt, you can:

- **Optimize**: Improve clarity, conciseness, creativity, or accuracy
- **Test**: Validate the prompt with sample inputs
- **Copy**: Copy the prompt to your clipboard
- **Export**: Save the prompt configuration

### Sequential Thinking

When enabled, the client uses structured reasoning:

1. **Context Analysis**: Understanding the requirements
2. **Framework Design**: Creating the prompt structure  
3. **Optimization**: Refining for clarity and effectiveness

This approach typically produces higher-quality, more targeted prompts.

## 🔗 API Endpoints

### Health Check
```
GET /api/health
```

### Generate Prompt
```
POST /api/generate-prompt
{
  "task_description": "string",
  "context": "string",
  "prompt_type": "general|creative|analytical|code|explanation",
  "target_audience": "general|technical|students|experts|beginners",
  "use_sequential_thinking": true,
  "model_preference": "auto",
  "temperature": 0.7
}
```

### Optimize Prompt
```
POST /api/optimize-prompt
{
  "original_prompt": "string",
  "optimization_goals": ["clarity", "conciseness", "creativity", "accuracy"]
}
```

### Test Prompt
```
POST /api/test-prompt
{
  "prompt": "string",
  "test_inputs": ["input1", "input2"]
}
```

## 🐳 Docker Support

### Build and Run
```bash
docker build -t prompt-builder-client .
docker run -p 3001:3001 prompt-builder-client
```

### With Docker Compose
```bash
docker-compose up prompt-builder-client
```

## 🔧 Development

### Development Mode
```bash
./start-prompt-builder.sh --dev
```

This enables:
- Auto-reload on file changes
- Detailed debug logging
- Development-specific configurations

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

### Code Structure

```
prompt-builder-client/
├── main.py                 # FastAPI application
├── config.json            # Configuration file
├── requirements.txt       # Python dependencies
├── start-prompt-builder.sh # Startup script
├── templates/             # Jinja2 templates
│   └── index.html        # Main UI template
├── static/               # Static assets
│   ├── css/
│   ├── js/
│   └── images/
├── data/                 # Data storage
└── logs/                 # Application logs
```

## 🤝 Integration with MCP Servers

### LLM Provider Server
- **Purpose**: Provides access to various AI models including AWS Bedrock
- **Port**: 8002
- **Tools**: `generate_text`, `list_models`, `get_usage_metrics`

### Sequential Thinker Server  
- **Purpose**: Implements structured reasoning for prompt engineering
- **Port**: 8001
- **Tools**: `create_thinking_chain`, `generate_sequential_thinking_prompt`

## 🚨 Troubleshooting

### MCP Servers Not Responding
1. Check if MCP servers are running:
   ```bash
   curl http://localhost:8002/health  # LLM Provider
   curl http://localhost:8001/health  # Sequential Thinker
   ```

2. Start MCP servers if needed:
   ```bash
   # From the mcp-servers directory
   ./start-all.sh
   ```

### Port Already in Use
```bash
# Use a different port
./start-prompt-builder.sh --port 8080
```

### Virtual Environment Issues
```bash
# Remove and recreate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📈 Performance

### Recommended System Requirements
- **Memory**: 4GB RAM minimum, 8GB recommended
- **CPU**: 2+ cores recommended
- **Network**: Stable internet connection for cloud AI models

### Optimization Tips
- Use appropriate model selection for your use case
- Enable caching for frequently used prompts
- Monitor MCP server health and response times

## 🔐 Security

### Environment Variables
Set sensitive configuration via environment variables:
```bash
export OPENAI_API_KEY="your-key"
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
```

### Network Security
- The client runs on localhost by default
- For production deployment, use proper authentication and HTTPS
- Configure firewall rules for MCP server ports

## 📝 License

This project is part of the MCP Demo ecosystem. See the main project license for details.

## 🆘 Support

For support and issues:
1. Check the troubleshooting section above
2. Review MCP server logs for connectivity issues
3. Ensure all prerequisites are met
4. Verify AWS credentials and permissions for Bedrock access

## 🔄 Updates

The client automatically checks MCP server connectivity on startup. Ensure your MCP servers are updated to the latest versions for optimal compatibility.