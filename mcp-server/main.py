from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import json
import logging
from typing import Dict, Any

from config.settings import settings
from mcp.server import MCPGenAIServer
from models.requests import BusinessRuleRequest, ValidationRequest, SearchRequest
from utils.logging import setup_logging

# Setup logging
setup_logging(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MCP GenAI Server",
    description="Model Context Protocol server with GenAI components",
    version=settings.mcp_server_version,
    debug=settings.debug
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MCP Server
mcp_server = MCPGenAIServer()

@app.get("/")
async def root():
    return {
        "name": settings.mcp_server_name,
        "version": settings.mcp_server_version,
        "status": "running",
        "endpoints": {
            "websocket": "/mcp",
            "health": "/health",
            "docs": "/docs",
            "api": {
                "business_rules": "/api/v1/business-rules/generate",
                "validation": "/api/v1/business-rules/validate", 
                "search": "/api/v1/context/search"
            }
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "server": settings.mcp_server_name,
        "version": settings.mcp_server_version
    }

@app.websocket("/mcp")
async def mcp_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("MCP WebSocket connection established")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                logger.debug(f"Received MCP message: {message}")
                
                # Process message through MCP server
                response = await mcp_server.handle_message(message)
                
                # Send response back to client
                await websocket.send_text(json.dumps(response))
                logger.debug(f"Sent MCP response: {response}")
                
            except json.JSONDecodeError:
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": "Parse error"
                    },
                    "id": None
                }
                await websocket.send_text(json.dumps(error_response))
                
            except Exception as e:
                logger.error(f"Error processing MCP message: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": "Internal error",
                        "data": str(e)
                    },
                    "id": message.get("id") if 'message' in locals() else None
                }
                await websocket.send_text(json.dumps(error_response))
                
    except WebSocketDisconnect:
        logger.info("MCP WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

# REST API endpoints for direct access (bypass MCP protocol)
@app.post("/api/v1/business-rules/generate")
async def generate_business_rule(request: BusinessRuleRequest):
    """Generate a business rule using the MCP server."""
    try:
        logger.info(f"Generating business rule via REST API")
        response = await mcp_server.handle_business_rule_generation(request)
        return response
    except Exception as e:
        logger.error(f"Error generating business rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/business-rules/validate")
async def validate_business_rule(request: ValidationRequest):
    """Validate a business rule using the MCP server."""
    try:
        logger.info(f"Validating business rule via REST API")
        response = await mcp_server.handle_rule_validation(request)
        return response
    except Exception as e:
        logger.error(f"Error validating business rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/context/search")
async def search_context(request: SearchRequest):
    """Search context using the MCP server."""
    try:
        logger.info(f"Searching context via REST API: {request.query}")
        response = await mcp_server.handle_context_search(request)
        return response
    except Exception as e:
        logger.error(f"Error searching context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/status")
async def get_server_status():
    """Get comprehensive server status."""
    try:
        status = await mcp_server.get_server_status()
        return status
    except Exception as e:
        logger.error(f"Error getting server status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/demo")
async def demo_page():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MCP GenAI Server Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .status { padding: 20px; background: #f0f0f0; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>MCP GenAI Server</h1>
            <div class="status">
                <h2>Server Status: Running</h2>
                <p><strong>Version:</strong> """ + settings.mcp_server_version + """</p>
                <p><strong>WebSocket Endpoint:</strong> ws://""" + settings.host + """:""" + str(settings.port) + """/mcp</p>
                <p><strong>API Documentation:</strong> <a href="/docs">/docs</a></p>
            </div>
            <h2>Available Components:</h2>
            <ul>
                <li>LLM Provider (OpenAI, Anthropic, Local)</li>
                <li>Prompt Builder with Templates</li>
                <li>Session Storage (Redis/Memory)</li>
                <li>Vector Store (ChromaDB)</li>
                <li>Business Rule Generation</li>
            </ul>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level
    )