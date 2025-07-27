# Production-Grade MCP Servers Ecosystem

A comprehensive collection of production-ready Model Context Protocol (MCP) servers providing essential AI infrastructure services. This ecosystem enables seamless integration between AI applications and various data sources, services, and capabilities.

## 🏗️ Architecture Overview

The MCP servers ecosystem consists of specialized servers that can work independently or together:

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Client Applications                   │
│              (Claude Desktop, VS Code, Custom)              │
└─────────────────────┬───────────────────────────────────────┘
                      │ MCP Protocol (JSON-RPC 2.0)
┌─────────────────────┴───────────────────────────────────────┐
│                    Orchestrator                             │
│             (Service Management & Discovery)                │
└─┬─────┬─────┬─────┬─────────────────┬─────────────────────┘
  │     │     │     │                 │
┌─▼─┐ ┌─▼─┐ ┌─▼─┐ ┌─▼─┐             ┌─▼─┐
│LLM│ │Vec│ │Mem│ │Web│             │Seq│
│   │ │tor│ │ory│ │   │             │ ✓ │
│Pro│ │   │ │   │ │Fet│             │   │
│vid│ │Str│ │Mgr│ │ch │             │   │
│er │ │ore│ │   │ │   │             │   │
└───┘ └───┘ └───┘ └───┘             └───┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+ (for client applications)
- Claude Desktop or compatible MCP client

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd mcp-demo/mcp-servers
```

2. **Set up the orchestrator:**
```bash
cd orchestrator
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Install dependencies for all servers:**
```bash
# Run the setup script
./setup-all-servers.sh
```

4. **Start all services:**
```bash
cd orchestrator
python main.py --auto-start
```

### Quick Test
```bash
# Check service status
curl http://localhost:8000/health

# Test LLM provider
curl -X POST http://localhost:8002/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "model": "gpt-4o-mini"}'
```

## 📦 Server Components

### 1. LLM Provider Server (`/llm-provider`)
**Multi-source LLM integration with intelligent routing**

**Features:**
- **Multiple Providers**: AWS Bedrock, OpenAI, Anthropic, LM Studio, Ollama
- **Automatic Fallback**: Route to backup providers on failure
- **Cost Optimization**: Choose cheapest model meeting requirements
- **Usage Tracking**: Monitor tokens, costs, and performance
- **Rate Limiting**: Respect provider limits and quotas

**Key Tools:**
- `generate_text` - Generate text with model selection
- `list_models` - Show available models across providers
- `optimize_model_selection` - Recommend best model for requirements
- `get_usage_metrics` - Detailed usage and cost analytics

**Configuration:**
```json
{
  "providers": {
    "aws_bedrock": { "enabled": true, "priority": 2 },
    "lm_studio": { "enabled": true, "priority": 1 },
    "openai": { "enabled": false, "api_key": "..." }
  }
}
```

### 2. Vector Store Server (`/vector-store`)
**Semantic search and vector storage with multiple backends**

**Features:**
- **Multiple Backends**: ChromaDB, Qdrant, Pinecone, FAISS, in-memory
- **Embedding Providers**: Sentence Transformers, OpenAI, Hugging Face
- **Document Processing**: Automatic chunking and embedding
- **Metadata Filtering**: Advanced search with filters
- **Analytics**: Collection analysis and insights

**Key Tools:**
- `add_documents` - Store documents with automatic embedding
- `search_vectors` - Semantic search with similarity threshold
- `create_collection` - Set up new vector collections
- `analyze_collection` - Get collection statistics and insights

**Configuration:**
```json
{
  "vector_stores": {
    "chromadb": { "enabled": true, "persist_directory": "./data" },
    "qdrant": { "enabled": false, "url": "http://localhost:6333" }
  },
  "embedding_providers": {
    "sentence_transformers": { "model": "all-MiniLM-L6-v2" }
  }
}
```

### 3. Memory Management Server (`/memory`)
**Persistent memory with multiple storage backends**

**Features:**
- **Storage Backends**: Redis, SQLite, PostgreSQL, MongoDB, in-memory
- **Memory Types**: Short-term, working, episodic, semantic, long-term
- **Automatic Cleanup**: TTL-based expiration and importance-based cleanup
- **Session Management**: User sessions with context preservation
- **Search Capabilities**: Full-text and semantic memory search

**Key Tools:**
- `store_memory` - Save memory with type and importance
- `search_memory` - Find relevant memories
- `get_session_memories` - Retrieve session history
- `cleanup_memories` - Remove old or low-importance memories

**Memory Types:**
- `short_term` - Temporary (1 hour TTL)
- `working` - Current conversation context
- `episodic` - Specific events and conversations
- `semantic` - Factual knowledge
- `long_term` - Permanent important information

### 4. Web Fetch Server (`/web-fetch`)
**Advanced web content fetching and processing**

**Features:**
- **Content Processing**: HTML extraction, markdown conversion, RSS parsing
- **Rate Limiting**: Respect robots.txt and implement delays
- **Content Filtering**: Clean HTML, extract main content
- **Bulk Operations**: Process multiple URLs concurrently
- **Caching**: Intelligent content caching with TTL

**Key Tools:**
- `process_url` - Fetch and process web content
- `bulk_fetch` - Process multiple URLs efficiently
- `extract_links` - Get all links from a webpage
- `check_url_status` - Verify URL accessibility

**Processing Modes:**
- `extract` - Main content extraction
- `markdown` - Convert to markdown
- `raw` - Return original content
- `clean` - Sanitized HTML

### 5. Sequential Thinker Server (`/sequential-thinker`)
**Structured reasoning and problem-solving**

**Features:**
- **Thinking Chains**: Multi-step reasoning processes
- **Business Rules**: Structured rule generation with thinking
- **Template System**: Reusable reasoning templates
- **Validation**: Chain completeness and logic validation

**Key Tools:**
- `create_thinking_chain` - Set up structured reasoning
- `generate_sequential_thinking_prompt` - Create step-by-step prompts
- `create_business_rule_thinking_prompt` - Business rule generation
- `validate_thinking_chain` - Verify reasoning completeness

### 6. Orchestrator (`/orchestrator`)
**Centralized service management and monitoring**

**Features:**
- **Service Lifecycle**: Start, stop, restart services
- **Health Monitoring**: Automatic health checks and recovery
- **Configuration Management**: Centralized config with hot-reload
- **Dependency Resolution**: Start services in correct order
- **Resource Monitoring**: CPU, memory, and performance tracking

**Key Tools:**
- `start_all_services` - Launch all enabled services
- `manage_service` - Control individual services
- `get_service_logs` - Retrieve service logs
- `reload_configuration` - Update config without restart

## 🔧 Configuration

### Global Configuration
Each server has its own `config.json` with comprehensive settings:

```json
{
  "server": {
    "name": "Server Name",
    "port": 8001,
    "transport": "stdio"
  },
  "features": {
    "feature1": true,
    "feature2": false
  },
  "performance": {
    "max_concurrent_requests": 100,
    "timeout_seconds": 30
  }
}
```

### Orchestrator Configuration
Central configuration in `orchestrator/config/orchestrator.yaml`:

```yaml
services:
  llm_provider:
    enabled: true
    priority: 10
    restart_policy: on-failure
  vector_store:
    enabled: true
    priority: 20
    dependencies: []
```

## 🔌 Integration

### Claude Desktop Integration
Add to Claude Desktop's MCP configuration:

```json
{
  "mcpServers": {
    "mcp-ecosystem": {
      "command": "python",
      "args": ["orchestrator/main.py"],
      "env": {
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### VS Code Integration
Use with MCP-compatible VS Code extensions:

```json
{
  "mcp.servers": {
    "mcp-ecosystem": {
      "command": "python orchestrator/main.py",
      "transport": "stdio"
    }
  }
}
```

### Custom Applications
Use any MCP client library:

```python
from mcp import Client

client = Client(transport="stdio")
client.connect("python orchestrator/main.py")

# Use the services
result = await client.call_tool("generate_text", {
    "prompt": "Explain quantum computing",
    "model": "gpt-4o"
})
```

## 📊 Monitoring & Analytics

### Health Monitoring
- **Service Status**: Real-time status of all services
- **Resource Usage**: CPU, memory, and disk monitoring
- **Performance Metrics**: Request rates, response times, error rates
- **Automatic Recovery**: Restart failed services with backoff

### Usage Analytics
- **LLM Usage**: Token consumption, costs, model performance
- **Vector Operations**: Search patterns, collection growth
- **Memory Patterns**: Access patterns, retention analysis
- **Web Fetching**: Success rates, content processing stats

### Alerting
- **Service Failures**: Immediate notification of service issues
- **Resource Limits**: Alerts when approaching resource limits
- **Cost Monitoring**: Track and alert on usage costs
- **Performance Degradation**: Detect and alert on slow responses

## 🔒 Security

### Input Validation
- **URL Validation**: Prevent SSRF attacks in web fetching
- **Content Sanitization**: Clean HTML and prevent XSS
- **Rate Limiting**: Prevent abuse and DoS attacks
- **Input Size Limits**: Prevent memory exhaustion

### Data Protection
- **Memory Encryption**: Optional encryption for sensitive data
- **Secure Storage**: Safe handling of API keys and secrets
- **Access Control**: Service-level access restrictions
- **Audit Logging**: Track all operations for compliance

## 🚀 Performance Optimization

### Caching Strategies
- **LLM Response Caching**: Cache expensive model responses
- **Vector Search Caching**: Cache frequent similarity searches
- **Web Content Caching**: Cache fetched content with TTL
- **Memory Query Caching**: Cache frequent memory lookups

### Resource Management
- **Connection Pooling**: Reuse database and HTTP connections
- **Batch Processing**: Group operations for efficiency
- **Lazy Loading**: Load resources only when needed
- **Memory Management**: Efficient cleanup and garbage collection

### Scaling
- **Horizontal Scaling**: Run multiple instances behind load balancer
- **Vertical Scaling**: Configure resource limits per service
- **Database Scaling**: Use read replicas and sharding
- **CDN Integration**: Cache static content globally

## 🧪 Testing

### Unit Tests
```bash
# Test individual servers
cd llm-provider && pytest tests/
cd vector-store && pytest tests/
cd memory && pytest tests/
```

### Integration Tests
```bash
# Test full ecosystem
cd orchestrator && pytest integration-tests/
```

### Load Testing
```bash
# Test performance under load
cd scripts && python load-test.py
```

### End-to-End Tests
```bash
# Test complete workflows
cd e2e-tests && python test-complete-workflow.py
```

## 📚 API Documentation

### REST API (when available)
- **OpenAPI Specs**: `/docs` endpoint for each server
- **Interactive Testing**: Swagger UI for API exploration
- **Schema Validation**: Request/response validation

### MCP Protocol
- **Tool Definitions**: Structured tool schemas
- **Resource Endpoints**: Available resources and formats
- **Error Handling**: Standardized error responses

## 🛠️ Development

### Adding New Servers
1. **Create Server Directory**: `mkdir mcp-servers/my-server`
2. **Implement MCP Interface**: Use FastMCP framework
3. **Add Configuration**: Create `config.json`
4. **Register with Orchestrator**: Add to `orchestrator.yaml`
5. **Add Tests**: Create test suite

### Extending Existing Servers
1. **Add New Tools**: Implement MCP tool interface
2. **Update Configuration**: Add new config options
3. **Document Changes**: Update API documentation
4. **Add Tests**: Test new functionality

### Custom Integrations
1. **Study MCP Protocol**: Understand message format
2. **Implement Client**: Use MCP client libraries
3. **Handle Lifecycle**: Manage connections and errors
4. **Add Monitoring**: Track usage and performance

## 🔄 Deployment

### Docker Deployment
```bash
# Build all images
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps
```

### Kubernetes Deployment
```bash
# Apply manifests
kubectl apply -f k8s/

# Check pods
kubectl get pods -n mcp-ecosystem
```

### Production Considerations
- **Load Balancing**: Use nginx or cloud load balancers
- **SSL/TLS**: Encrypt all communications
- **Backup Strategy**: Regular backups of critical data
- **Monitoring Stack**: Prometheus + Grafana setup
- **Log Aggregation**: ELK stack or cloud logging

## 📋 Troubleshooting

### Common Issues

**Service Won't Start**
```bash
# Check logs
python orchestrator/main.py --log-level DEBUG

# Verify dependencies
pip list | grep mcp

# Check ports
netstat -tulpn | grep 800
```

**High Memory Usage**
```bash
# Monitor resource usage
top -p $(pgrep -f "mcp-server")

# Check cache sizes
curl http://localhost:8003/vectors/stats
```

**Connection Issues**
```bash
# Test connectivity
curl http://localhost:8000/health

# Check firewall
sudo ufw status
```

### Debug Mode
Enable debug logging in any server:
```json
{
  "logging": {
    "level": "DEBUG",
    "include_request_details": true
  }
}
```

## 🤝 Contributing

### Code Style
- **Python**: Follow PEP 8, use Black formatter
- **TypeScript**: Use Prettier, follow ESLint rules
- **Documentation**: Keep README and API docs updated

### Pull Request Process
1. **Fork Repository**: Create your own fork
2. **Feature Branch**: Create branch for your feature
3. **Add Tests**: Ensure good test coverage
4. **Update Docs**: Document new features
5. **Submit PR**: Create pull request with description

### Issue Reporting
- **Bug Reports**: Use issue template with reproduction steps
- **Feature Requests**: Describe use case and requirements
- **Security Issues**: Report privately to maintainers

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Anthropic**: For the MCP protocol specification
- **FastAPI**: For the excellent web framework
- **Pydantic**: For data validation and serialization
- **Community**: For contributions and feedback

---

**Built with ❤️ for the AI development community**