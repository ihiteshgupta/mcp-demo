from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None  # For custom OpenAI-compatible APIs like Ollama
    
    # LM Studio Configuration
    lmstudio_base_url: Optional[str] = "http://localhost:1234/v1"  # Default LM Studio URL
    lmstudio_api_key: Optional[str] = "lm-studio"  # LM Studio doesn't require real auth
    
    # Anthropic Configuration
    anthropic_api_key: Optional[str] = None
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # ChromaDB Configuration
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    
    # Server Configuration
    host: str = "localhost"
    port: int = 8000
    debug: bool = True
    log_level: str = "info"
    
    # MCP Configuration
    mcp_server_name: str = "genai-mcp-server"
    mcp_server_version: str = "1.0.0"
    
    class Config:
        env_file = ".env"


settings = Settings()