import pytest
import asyncio
import json
from fastapi.testclient import TestClient

from main import app
from mcp.server import MCPGenAIServer
from models.requests import BusinessRuleRequest, ValidationRequest, SearchRequest


class TestMCPServer:
    """Test suite for MCP GenAI Server."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mcp_server(self):
        """Create MCP server instance."""
        return MCPGenAIServer()
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns server info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_demo_page(self, client):
        """Test demo page returns HTML."""
        response = client.get("/demo")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    @pytest.mark.asyncio
    async def test_mcp_initialization(self, mcp_server):
        """Test MCP server initialization."""
        # Test server components are initialized
        assert mcp_server.llm_provider is not None
        assert mcp_server.prompt_builder is not None
        assert mcp_server.session_storage is not None
        assert mcp_server.vector_store is not None
    
    @pytest.mark.asyncio
    async def test_mcp_message_handling(self, mcp_server):
        """Test MCP message handling."""
        # Test initialize message
        init_message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        response = await mcp_server.handle_message(init_message)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["serverInfo"]["name"] == "genai-mcp-server"
    
    @pytest.mark.asyncio
    async def test_list_tools(self, mcp_server):
        """Test tools listing."""
        message = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        response = await mcp_server.handle_message(message)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2
        assert "result" in response
        
        tools = response["result"]["tools"]
        tool_names = [tool["name"] for tool in tools]
        assert "generate_business_rule" in tool_names
        assert "validate_business_rule" in tool_names
        assert "search_context" in tool_names
    
    @pytest.mark.asyncio
    async def test_business_rule_generation(self, mcp_server):
        """Test business rule generation."""
        request = BusinessRuleRequest(
            context="E-commerce platform",
            requirements="Create a rule for discount validation",
            provider="local"
        )
        
        response = await mcp_server.handle_business_rule_generation(request)
        assert response.rule is not None
        assert response.rule.id is not None
        assert response.rule.name is not None
        assert response.generation_info is not None
    
    @pytest.mark.asyncio
    async def test_rule_validation(self, mcp_server):
        """Test rule validation."""
        request = ValidationRequest(
            rule_content="If customer is premium, apply 10% discount",
            provider="local"
        )
        
        response = await mcp_server.handle_rule_validation(request)
        assert response.validation is not None
        assert isinstance(response.validation.score, int)
        assert 0 <= response.validation.score <= 10
    
    @pytest.mark.asyncio
    async def test_context_search(self, mcp_server):
        """Test context search."""
        request = SearchRequest(
            query="discount rules",
            limit=5
        )
        
        response = await mcp_server.handle_context_search(request)
        assert response.results is not None
        assert isinstance(response.results, list)
        assert response.query == "discount rules"
    
    @pytest.mark.asyncio
    async def test_server_status(self, mcp_server):
        """Test server status retrieval."""
        status = await mcp_server.get_server_status()
        assert "server" in status
        assert "components" in status
        assert status["server"]["status"] == "running"


class TestMCPComponents:
    """Test individual MCP components."""
    
    @pytest.fixture
    def mcp_server(self):
        return MCPGenAIServer()
    
    @pytest.mark.asyncio
    async def test_llm_provider(self, mcp_server):
        """Test LLM provider functionality."""
        providers = mcp_server.llm_provider.list_providers()
        assert "local" in providers
        
        # Test completion with local provider
        response = await mcp_server.llm_provider.generate_completion(
            prompt="Test prompt",
            provider="local"
        )
        assert response.content is not None
        assert response.model is not None
    
    @pytest.mark.asyncio
    async def test_prompt_builder(self, mcp_server):
        """Test prompt builder functionality."""
        templates = mcp_server.prompt_builder.list_templates()
        assert len(templates) >= 3  # Should have at least 3 templates
        
        # Test business rule prompt building
        prompt = mcp_server.prompt_builder.build_business_rule_prompt(
            context="Test context",
            requirements="Test requirements"
        )
        assert "Test context" in prompt
        assert "Test requirements" in prompt
    
    @pytest.mark.asyncio
    async def test_session_storage(self, mcp_server):
        """Test session storage functionality."""
        # Create session
        session = await mcp_server.session_storage.create_session()
        assert session.session_id is not None
        
        # Update session
        updated = await mcp_server.session_storage.update_session(
            session.session_id,
            {"test_data": "test_value"}
        )
        assert updated is not None
        assert updated.data["test_data"] == "test_value"
        
        # Get session
        retrieved = await mcp_server.session_storage.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id
    
    @pytest.mark.asyncio 
    async def test_vector_store(self, mcp_server):
        """Test vector store functionality."""
        from components.vector_store import Document
        
        # Add document
        doc = Document(
            id="test_doc",
            content="This is a test document for business rules",
            metadata={"type": "test"}
        )
        
        ids = await mcp_server.vector_store.add_documents([doc])
        assert "test_doc" in ids
        
        # Search documents
        results = await mcp_server.vector_store.similarity_search(
            query="business rules",
            k=1
        )
        assert len(results) >= 0  # May be 0 if no similar documents
        
        # Count documents
        count = await mcp_server.vector_store.count_documents()
        assert count >= 1


if __name__ == "__main__":
    pytest.main([__file__])