import asyncio
import logging
from typing import AsyncIterator, Dict, List, Optional, Any

import openai
import anthropic
from config.settings import settings
from utils.exceptions import LLMProviderError
from components.base_providers import BaseLLMProvider, CompletionRequest, CompletionResponse

logger = logging.getLogger(__name__)

# Import local providers (delayed import to avoid circular dependency)
LMSTUDIO_AVAILABLE = False

try:
    from components.lmstudio_provider import LMStudioProvider
    LMSTUDIO_AVAILABLE = True
except ImportError:
    logger.warning("LM Studio provider not available")


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation."""
    
    def __init__(self, api_key: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.default_model = "gpt-4"
        self.embedding_model = "text-embedding-ada-002"
    
    async def generate_completion(self, request: CompletionRequest) -> CompletionResponse:
        try:
            response = await self.client.chat.completions.create(
                model=request.model or self.default_model,
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
            logger.error(f"OpenAI completion error: {e}")
            raise LLMProviderError(f"OpenAI error: {e}")
    
    async def generate_streaming(self, request: CompletionRequest) -> AsyncIterator[str]:
        try:
            response = await self.client.chat.completions.create(
                model=request.model or self.default_model,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True
            )
            
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise LLMProviderError(f"OpenAI streaming error: {e}")
    
    async def embed_text(self, text: str, model: str = None) -> List[float]:
        try:
            response = await self.client.embeddings.create(
                model=model or self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise LLMProviderError(f"OpenAI embedding error: {e}")


class AnthropicProvider(BaseLLMProvider):
    """Anthropic LLM provider implementation."""
    
    def __init__(self, api_key: str):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.default_model = "claude-3-sonnet-20240229"
    
    async def generate_completion(self, request: CompletionRequest) -> CompletionResponse:
        try:
            response = await self.client.messages.create(
                model=request.model or self.default_model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                messages=[{"role": "user", "content": request.prompt}]
            )
            
            return CompletionResponse(
                content=response.content[0].text,
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                finish_reason=response.stop_reason
            )
        except Exception as e:
            logger.error(f"Anthropic completion error: {e}")
            raise LLMProviderError(f"Anthropic error: {e}")
    
    async def generate_streaming(self, request: CompletionRequest) -> AsyncIterator[str]:
        try:
            async with self.client.messages.stream(
                model=request.model or self.default_model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                messages=[{"role": "user", "content": request.prompt}]
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise LLMProviderError(f"Anthropic streaming error: {e}")
    
    async def embed_text(self, text: str, model: str = None) -> List[float]:
        # Anthropic doesn't provide embedding API, so we'll use a placeholder
        # In a real implementation, you might use a different service for embeddings
        raise LLMProviderError("Anthropic does not provide embedding API")


class LocalLLMProvider(BaseLLMProvider):
    """Local LLM provider implementation (placeholder)."""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        logger.warning("LocalLLMProvider is a placeholder implementation")
    
    async def generate_completion(self, request: CompletionRequest) -> CompletionResponse:
        # Placeholder implementation
        await asyncio.sleep(0.1)  # Simulate processing time
        return CompletionResponse(
            content=f"[Local LLM Response] Generated business rule based on: {request.prompt[:100]}...",
            model="local-model",
            usage={"tokens": 50},
            finish_reason="stop"
        )
    
    async def generate_streaming(self, request: CompletionRequest) -> AsyncIterator[str]:
        # Placeholder implementation
        response_text = f"[Local LLM Streaming] Generated business rule based on: {request.prompt[:100]}..."
        for chunk in response_text.split():
            await asyncio.sleep(0.05)
            yield chunk + " "
    
    async def embed_text(self, text: str, model: str = None) -> List[float]:
        # Placeholder implementation - return random-like embeddings
        import hashlib
        hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(hash_value >> i) % 2 - 0.5 for i in range(1536)]


class LLMProvider:
    """Unified interface for multiple LLM providers."""
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.default_provider = "local"
        
        # Initialize LM Studio provider if available
        if LMSTUDIO_AVAILABLE and hasattr(settings, 'lmstudio_base_url') and settings.lmstudio_base_url:
            try:
                self.providers["lmstudio"] = LMStudioProvider(
                    base_url=settings.lmstudio_base_url,
                    api_key=settings.lmstudio_api_key or "lm-studio"
                )
                self.default_provider = "lmstudio"
                logger.info("Initialized LM Studio provider (local models)")
            except Exception as e:
                logger.warning(f"Failed to initialize LM Studio provider: {e}")
        
        # Initialize providers based on available API keys
        if settings.openai_api_key and not hasattr(settings, 'lmstudio_base_url'):
            self.providers["openai"] = OpenAIProvider(settings.openai_api_key)
            if self.default_provider == "local":
                self.default_provider = "openai"
            logger.info("Initialized OpenAI provider")
        
        if settings.anthropic_api_key:
            self.providers["anthropic"] = AnthropicProvider(settings.anthropic_api_key)
            if self.default_provider == "local":
                self.default_provider = "anthropic"
            logger.info("Initialized Anthropic provider")
        
        # Always have local provider as fallback
        if "lmstudio" not in self.providers:
            self.providers["local"] = LocalLLMProvider()
            logger.info("Initialized Local LLM provider (placeholder)")
        
        logger.info(f"Default LLM provider: {self.default_provider}")
    
    def get_provider(self, provider_name: str = None) -> BaseLLMProvider:
        """Get LLM provider by name."""
        provider_name = provider_name or self.default_provider
        
        if provider_name not in self.providers:
            raise LLMProviderError(f"Provider '{provider_name}' not available")
        
        return self.providers[provider_name]
    
    async def generate_completion(
        self, 
        prompt: str, 
        model: str = None, 
        provider: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> CompletionResponse:
        """Generate completion using specified provider."""
        provider_instance = self.get_provider(provider)
        request = CompletionRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return await provider_instance.generate_completion(request)
    
    async def generate_streaming(
        self, 
        prompt: str, 
        model: str = None, 
        provider: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> AsyncIterator[str]:
        """Generate streaming completion using specified provider."""
        provider_instance = self.get_provider(provider)
        request = CompletionRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        async for chunk in provider_instance.generate_streaming(request):
            yield chunk
    
    async def embed_text(self, text: str, provider: str = None, model: str = None) -> List[float]:
        """Generate embeddings using specified provider."""
        provider_instance = self.get_provider(provider)
        return await provider_instance.embed_text(text, model)
    
    def list_providers(self) -> List[str]:
        """List available providers."""
        return list(self.providers.keys())
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about available providers."""
        return {
            "providers": list(self.providers.keys()),
            "default": self.default_provider,
            "status": {
                provider: "available" for provider in self.providers.keys()
            }
        }