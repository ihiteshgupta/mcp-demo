"""
LM Studio LLM Provider Implementation

This module provides integration with LM Studio's local server using OpenAI-compatible API.
LM Studio provides a local inference server for open-source models with OpenAI API compatibility.
"""

import asyncio
import logging
from typing import AsyncIterator, List, Optional

import openai
from components.base_providers import BaseLLMProvider, CompletionRequest, CompletionResponse
from utils.exceptions import LLMProviderError

logger = logging.getLogger(__name__)


class LMStudioProvider(BaseLLMProvider):
    """
    LM Studio LLM provider implementation.
    
    LM Studio is a local inference server that runs open-source models
    with OpenAI-compatible API endpoints.
    
    Default configuration:
    - Base URL: http://localhost:1234/v1
    - API Key: lm-studio (or any string, LM Studio doesn't enforce authentication)
    """
    
    def __init__(self, base_url: str = "http://localhost:1234/v1", api_key: str = "lm-studio"):
        """
        Initialize LM Studio provider.
        
        Args:
            base_url: LM Studio server base URL (default: http://localhost:1234/v1)
            api_key: API key (LM Studio doesn't enforce auth, any string works)
        """
        self.base_url = base_url
        self.api_key = api_key
        self.client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )
        # Common models that work well with LM Studio
        self.default_model = "local-model"  # LM Studio auto-detects loaded model
        self.embedding_model = "nomic-embed-text"  # Popular embedding model
        
        logger.info(f"Initialized LM Studio provider with base URL: {base_url}")
    
    async def health_check(self) -> bool:
        """Check if LM Studio server is available and has a model loaded."""
        try:
            # Try to list available models
            models = await self.client.models.list()
            if models.data:
                logger.info(f"LM Studio health check passed. Available models: {[m.id for m in models.data]}")
                return True
            else:
                logger.warning("LM Studio server is running but no models are loaded")
                return False
        except Exception as e:
            logger.error(f"LM Studio health check failed: {e}")
            return False
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models from LM Studio."""
        try:
            models = await self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Failed to get LM Studio models: {e}")
            return []
    
    async def generate_completion(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using LM Studio."""
        try:
            # Use the first available chat model if no specific model requested
            model = request.model or self.default_model
            if model == "local-model":
                available_models = await self.get_available_models()
                if available_models:
                    # Filter out embedding models for chat completions
                    chat_models = [m for m in available_models if not any(keyword in m.lower() for keyword in ['embed', 'embedding'])]
                    if chat_models:
                        model = chat_models[0]
                        logger.debug(f"Using LM Studio chat model: {model}")
                    else:
                        # If no chat models found, use the first available model anyway
                        model = available_models[0]
                        logger.warning(f"No dedicated chat models found, using: {model}")
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False
            )
            
            return CompletionResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage=response.usage.dict() if response.usage else {},
                finish_reason=response.choices[0].finish_reason
            )
        except Exception as e:
            logger.error(f"LM Studio completion error: {e}")
            raise LLMProviderError(f"LM Studio error: {e}")
    
    async def generate_streaming(self, request: CompletionRequest) -> AsyncIterator[str]:
        """Generate streaming completion using LM Studio."""
        try:
            # Use the first available chat model if no specific model requested
            model = request.model or self.default_model
            if model == "local-model":
                available_models = await self.get_available_models()
                if available_models:
                    # Filter out embedding models for chat completions
                    chat_models = [m for m in available_models if not any(keyword in m.lower() for keyword in ['embed', 'embedding'])]
                    if chat_models:
                        model = chat_models[0]
                        logger.debug(f"Using LM Studio chat model for streaming: {model}")
                    else:
                        # If no chat models found, use the first available model anyway
                        model = available_models[0]
                        logger.warning(f"No dedicated chat models found for streaming, using: {model}")
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True
            )
            
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"LM Studio streaming error: {e}")
            raise LLMProviderError(f"LM Studio streaming error: {e}")
    
    async def embed_text(self, text: str, model: str = None) -> List[float]:
        """
        Generate embeddings using LM Studio.
        
        Note: LM Studio supports embeddings if an embedding model is loaded.
        This requires loading a compatible embedding model in LM Studio.
        """
        try:
            embedding_model = model or self.embedding_model
            
            # Check if embedding model is available
            available_models = await self.get_available_models()
            if embedding_model not in available_models:
                # Try to find any embedding model
                embedding_models = [m for m in available_models if 'embed' in m.lower()]
                if embedding_models:
                    embedding_model = embedding_models[0]
                    logger.debug(f"Using available embedding model: {embedding_model}")
                else:
                    # Fallback to using the chat model for embeddings (not ideal but works)
                    if available_models:
                        embedding_model = available_models[0]
                        logger.warning(f"No embedding model found, using chat model: {embedding_model}")
                    else:
                        raise LLMProviderError("No models available in LM Studio")
            
            response = await self.client.embeddings.create(
                model=embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"LM Studio embedding error: {e}")
            # Fallback: generate simple embeddings from text hash
            logger.warning("Falling back to hash-based embeddings")
            return self._generate_fallback_embeddings(text)
    
    def _generate_fallback_embeddings(self, text: str) -> List[float]:
        """Generate simple hash-based embeddings as fallback."""
        import hashlib
        import struct
        
        # Create a hash of the text
        hash_bytes = hashlib.sha256(text.encode()).digest()
        
        # Convert to list of floats
        embeddings = []
        for i in range(0, len(hash_bytes), 4):
            chunk = hash_bytes[i:i+4]
            if len(chunk) == 4:
                # Convert 4 bytes to float
                value = struct.unpack('>I', chunk)[0] / (2**32)
                embeddings.append(value - 0.5)  # Center around 0
        
        # Pad or truncate to 384 dimensions (common embedding size)
        target_size = 384
        if len(embeddings) < target_size:
            embeddings.extend([0.0] * (target_size - len(embeddings)))
        else:
            embeddings = embeddings[:target_size]
        
        return embeddings
    
    async def get_server_info(self) -> dict:
        """Get information about the LM Studio server."""
        try:
            models = await self.get_available_models()
            health = await self.health_check()
            
            return {
                "server_url": self.base_url,
                "status": "healthy" if health else "unhealthy",
                "models_loaded": len(models),
                "available_models": models,
                "supports_chat": True,
                "supports_embeddings": bool(models),  # Depends on loaded models
                "supports_streaming": True
            }
        except Exception as e:
            return {
                "server_url": self.base_url,
                "status": "error",
                "error": str(e),
                "models_loaded": 0,
                "available_models": [],
                "supports_chat": False,
                "supports_embeddings": False,
                "supports_streaming": False
            }