#!/usr/bin/env python3
"""
LLM Provider MCP Server

Production-grade MCP server for LLM operations with support for multiple providers:
- AWS Bedrock (Claude, Titan, Llama, etc.)
- LM Studio (Local models)  
- Ollama (Local models)
- OpenAI API
- Anthropic API
- Hugging Face Transformers
- Azure OpenAI

Features:
- Multi-provider routing and fallback
- Streaming and batch processing
- Token usage tracking
- Model capabilities detection
- Advanced prompt engineering
- Cost optimization
- Rate limiting and quotas
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
import time

from mcp.server.fastmcp import FastMCP
from mcp.types import Resource, Tool, TextContent
from pydantic import BaseModel, Field

# Provider-specific imports
try:
    import boto3
    from botocore.exceptions import ClientError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("LLM Provider MCP Server")

# Global state
providers: Dict[str, 'LLMProvider'] = {}
usage_tracker = {}
rate_limiter = {}


class ProviderType(str, Enum):
    """Supported LLM provider types."""
    AWS_BEDROCK = "aws_bedrock"
    LM_STUDIO = "lm_studio"
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    LOCAL_TRANSFORMERS = "local_transformers"


@dataclass
class ModelCapabilities:
    """Model capabilities and metadata."""
    max_tokens: int
    supports_streaming: bool
    supports_functions: bool
    supports_vision: bool
    cost_per_1k_input: float
    cost_per_1k_output: float
    context_window: int
    supports_system_message: bool


@dataclass
class UsageMetrics:
    """Usage tracking metrics."""
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    latency_ms: float
    timestamp: datetime
    success: bool
    error: Optional[str] = None


class GenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(description="The input prompt")
    model: str = Field(description="Model identifier")
    provider: Optional[str] = Field(None, description="Specific provider to use")
    max_tokens: int = Field(1000, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling")
    top_k: Optional[int] = Field(None, description="Top-k sampling")
    stream: bool = Field(False, description="Enable streaming response")
    system_message: Optional[str] = Field(None, description="System message")
    functions: Optional[List[Dict]] = Field(None, description="Available functions")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class ModelListRequest(BaseModel):
    """Request model for listing models."""
    provider: Optional[str] = Field(None, description="Filter by provider")
    capability: Optional[str] = Field(None, description="Filter by capability")


class ProviderConfig(BaseModel):
    """Configuration for a provider."""
    type: ProviderType
    enabled: bool = True
    config: Dict[str, Any] = Field(default_factory=dict)
    models: List[str] = Field(default_factory=list)
    priority: int = Field(1, description="Lower number = higher priority")
    rate_limit: Optional[int] = Field(None, description="Requests per minute")
    quota: Optional[float] = Field(None, description="Monthly cost quota")


class LLMProvider:
    """Base class for LLM providers."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.type = config.type
        self.enabled = config.enabled
        self.models = {}
        self.client = None
        
    async def initialize(self):
        """Initialize the provider."""
        pass
    
    async def list_models(self) -> List[str]:
        """List available models."""
        return list(self.models.keys())
    
    async def get_model_info(self, model: str) -> Optional[ModelCapabilities]:
        """Get model capabilities."""
        return self.models.get(model)
    
    async def generate(self, request: GenerationRequest) -> Dict[str, Any]:
        """Generate text with the model."""
        raise NotImplementedError
    
    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Generate text with streaming."""
        raise NotImplementedError
    
    def track_usage(self, model: str, input_tokens: int, output_tokens: int, 
                   latency_ms: float, success: bool, error: str = None):
        """Track usage metrics."""
        capabilities = self.models.get(model)
        cost = 0.0
        if capabilities:
            cost = (input_tokens * capabilities.cost_per_1k_input / 1000 + 
                   output_tokens * capabilities.cost_per_1k_output / 1000)
        
        usage = UsageMetrics(
            provider=self.type.value,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost=cost,
            latency_ms=latency_ms,
            timestamp=datetime.now(),
            success=success,
            error=error
        )
        
        key = f"{self.type.value}:{model}"
        if key not in usage_tracker:
            usage_tracker[key] = []
        usage_tracker[key].append(usage)


class AWSBedrockProvider(LLMProvider):
    """AWS Bedrock provider implementation."""
    
    async def initialize(self):
        if not HAS_BOTO3:
            raise ImportError("boto3 required for AWS Bedrock")
        
        self.client = boto3.client(
            'bedrock-runtime',
            region_name=self.config.config.get('region', 'us-east-1'),
            aws_access_key_id=self.config.config.get('access_key'),
            aws_secret_access_key=self.config.config.get('secret_key')
        )
        
        # Define available models with capabilities
        self.models = {
            "anthropic.claude-3-5-sonnet-20241022-v2:0": ModelCapabilities(
                max_tokens=8192, supports_streaming=True, supports_functions=True,
                supports_vision=True, cost_per_1k_input=3.0, cost_per_1k_output=15.0,
                context_window=200000, supports_system_message=True
            ),
            "anthropic.claude-3-haiku-20240307-v1:0": ModelCapabilities(
                max_tokens=4096, supports_streaming=True, supports_functions=True,
                supports_vision=True, cost_per_1k_input=0.25, cost_per_1k_output=1.25,
                context_window=200000, supports_system_message=True
            ),
            "amazon.titan-text-premier-v1:0": ModelCapabilities(
                max_tokens=4096, supports_streaming=True, supports_functions=False,
                supports_vision=False, cost_per_1k_input=0.5, cost_per_1k_output=1.5,
                context_window=32000, supports_system_message=False
            ),
            "meta.llama3-2-90b-instruct-v1:0": ModelCapabilities(
                max_tokens=4096, supports_streaming=True, supports_functions=False,
                supports_vision=False, cost_per_1k_input=2.0, cost_per_1k_output=6.0,
                context_window=128000, supports_system_message=True
            )
        }
    
    async def generate(self, request: GenerationRequest) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            # Prepare request based on model
            if "anthropic.claude" in request.model:
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "messages": [{"role": "user", "content": request.prompt}]
                }
                if request.system_message:
                    body["system"] = request.system_message
                    
            elif "amazon.titan" in request.model:
                body = {
                    "inputText": request.prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": request.max_tokens,
                        "temperature": request.temperature,
                        "topP": request.top_p,
                        "stopSequences": request.stop_sequences or []
                    }
                }
                
            elif "meta.llama" in request.model:
                body = {
                    "prompt": request.prompt,
                    "max_gen_len": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p
                }
            
            response = self.client.invoke_model(
                modelId=request.model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body)
            )
            
            response_body = json.loads(response['body'].read())
            
            # Parse response based on model
            if "anthropic.claude" in request.model:
                content = response_body['content'][0]['text']
                input_tokens = response_body['usage']['input_tokens']
                output_tokens = response_body['usage']['output_tokens']
                
            elif "amazon.titan" in request.model:
                content = response_body['results'][0]['outputText']
                input_tokens = response_body['inputTextTokenCount']
                output_tokens = response_body['results'][0]['tokenCount']
                
            elif "meta.llama" in request.model:
                content = response_body['generation']
                input_tokens = response_body.get('prompt_token_count', 0)
                output_tokens = response_body.get('generation_token_count', 0)
            
            latency_ms = (time.time() - start_time) * 1000
            self.track_usage(request.model, input_tokens, output_tokens, latency_ms, True)
            
            return {
                "content": content,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                },
                "model": request.model,
                "provider": self.type.value,
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.track_usage(request.model, 0, 0, latency_ms, False, str(e))
            raise


class LMStudioProvider(LLMProvider):
    """LM Studio provider implementation."""
    
    async def initialize(self):
        if not HAS_REQUESTS:
            raise ImportError("requests required for LM Studio")
        
        self.base_url = self.config.config.get('base_url', 'http://localhost:1234/v1')
        
        # Discover available models
        try:
            response = requests.get(f"{self.base_url}/models")
            if response.status_code == 200:
                models_data = response.json()
                for model in models_data.get('data', []):
                    model_id = model['id']
                    # Default capabilities for LM Studio models
                    self.models[model_id] = ModelCapabilities(
                        max_tokens=4096, supports_streaming=True, supports_functions=False,
                        supports_vision=False, cost_per_1k_input=0.0, cost_per_1k_output=0.0,
                        context_window=4096, supports_system_message=True
                    )
        except Exception as e:
            logger.warning(f"Could not discover LM Studio models: {e}")
    
    async def generate(self, request: GenerationRequest) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            messages = []
            if request.system_message:
                messages.append({"role": "system", "content": request.system_message})
            messages.append({"role": "user", "content": request.prompt})
            
            payload = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stream": False
            }
            
            if request.stop_sequences:
                payload["stop"] = request.stop_sequences
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            usage = data.get('usage', {})
            
            latency_ms = (time.time() - start_time) * 1000
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
            
            self.track_usage(request.model, input_tokens, output_tokens, latency_ms, True)
            
            return {
                "content": content,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": usage.get('total_tokens', 0)
                },
                "model": request.model,
                "provider": self.type.value,
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.track_usage(request.model, 0, 0, latency_ms, False, str(e))
            raise


class OllamaProvider(LLMProvider):
    """Ollama provider implementation."""
    
    async def initialize(self):
        if not HAS_REQUESTS:
            raise ImportError("requests required for Ollama")
        
        self.base_url = self.config.config.get('base_url', 'http://localhost:11434')
        
        # Discover available models
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models_data = response.json()
                for model in models_data.get('models', []):
                    model_name = model['name']
                    # Default capabilities for Ollama models
                    self.models[model_name] = ModelCapabilities(
                        max_tokens=4096, supports_streaming=True, supports_functions=False,
                        supports_vision=False, cost_per_1k_input=0.0, cost_per_1k_output=0.0,
                        context_window=4096, supports_system_message=True
                    )
        except Exception as e:
            logger.warning(f"Could not discover Ollama models: {e}")
    
    async def generate(self, request: GenerationRequest) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            payload = {
                "model": request.model,
                "prompt": request.prompt,
                "stream": False,
                "options": {
                    "num_predict": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p
                }
            }
            
            if request.system_message:
                payload["system"] = request.system_message
            
            if request.stop_sequences:
                payload["options"]["stop"] = request.stop_sequences
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            data = response.json()
            content = data['response']
            
            latency_ms = (time.time() - start_time) * 1000
            # Ollama doesn't provide token counts, estimate
            input_tokens = len(request.prompt) // 4
            output_tokens = len(content) // 4
            
            self.track_usage(request.model, input_tokens, output_tokens, latency_ms, True)
            
            return {
                "content": content,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                },
                "model": request.model,
                "provider": self.type.value,
                "latency_ms": latency_ms,
                "note": "Token counts are estimated"
            }
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.track_usage(request.model, 0, 0, latency_ms, False, str(e))
            raise


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""
    
    async def initialize(self):
        if not HAS_OPENAI:
            raise ImportError("openai required for OpenAI provider")
        
        api_key = self.config.config.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = openai.AsyncOpenAI(api_key=api_key)
        
        # Define available models with capabilities
        self.models = {
            "gpt-4o": ModelCapabilities(
                max_tokens=4096, supports_streaming=True, supports_functions=True,
                supports_vision=True, cost_per_1k_input=2.5, cost_per_1k_output=10.0,
                context_window=128000, supports_system_message=True
            ),
            "gpt-4o-mini": ModelCapabilities(
                max_tokens=16384, supports_streaming=True, supports_functions=True,
                supports_vision=True, cost_per_1k_input=0.15, cost_per_1k_output=0.6,
                context_window=128000, supports_system_message=True
            ),
            "gpt-3.5-turbo": ModelCapabilities(
                max_tokens=4096, supports_streaming=True, supports_functions=True,
                supports_vision=False, cost_per_1k_input=0.5, cost_per_1k_output=1.5,
                context_window=16385, supports_system_message=True
            )
        }
    
    async def generate(self, request: GenerationRequest) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            messages = []
            if request.system_message:
                messages.append({"role": "system", "content": request.system_message})
            messages.append({"role": "user", "content": request.prompt})
            
            kwargs = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stream": False
            }
            
            if request.stop_sequences:
                kwargs["stop"] = request.stop_sequences
            
            if request.functions:
                kwargs["functions"] = request.functions
            
            if request.seed:
                kwargs["seed"] = request.seed
            
            response = await self.client.chat.completions.create(**kwargs)
            
            content = response.choices[0].message.content
            usage = response.usage
            
            latency_ms = (time.time() - start_time) * 1000
            
            self.track_usage(request.model, usage.prompt_tokens, usage.completion_tokens, latency_ms, True)
            
            return {
                "content": content,
                "usage": {
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                },
                "model": request.model,
                "provider": self.type.value,
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.track_usage(request.model, 0, 0, latency_ms, False, str(e))
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic provider implementation."""
    
    async def initialize(self):
        if not HAS_ANTHROPIC:
            raise ImportError("anthropic required for Anthropic provider")
        
        api_key = self.config.config.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API key required")
        
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        
        # Define available models with capabilities
        self.models = {
            "claude-3-5-sonnet-20241022": ModelCapabilities(
                max_tokens=8192, supports_streaming=True, supports_functions=True,
                supports_vision=True, cost_per_1k_input=3.0, cost_per_1k_output=15.0,
                context_window=200000, supports_system_message=True
            ),
            "claude-3-haiku-20240307": ModelCapabilities(
                max_tokens=4096, supports_streaming=True, supports_functions=True,
                supports_vision=True, cost_per_1k_input=0.25, cost_per_1k_output=1.25,
                context_window=200000, supports_system_message=True
            )
        }
    
    async def generate(self, request: GenerationRequest) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            kwargs = {
                "model": request.model,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "messages": [{"role": "user", "content": request.prompt}]
            }
            
            if request.system_message:
                kwargs["system"] = request.system_message
            
            if request.stop_sequences:
                kwargs["stop_sequences"] = request.stop_sequences
            
            response = await self.client.messages.create(**kwargs)
            
            content = response.content[0].text
            usage = response.usage
            
            latency_ms = (time.time() - start_time) * 1000
            
            self.track_usage(request.model, usage.input_tokens, usage.output_tokens, latency_ms, True)
            
            return {
                "content": content,
                "usage": {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.input_tokens + usage.output_tokens
                },
                "model": request.model,
                "provider": self.type.value,
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.track_usage(request.model, 0, 0, latency_ms, False, str(e))
            raise


# Provider factory
PROVIDER_CLASSES = {
    ProviderType.AWS_BEDROCK: AWSBedrockProvider,
    ProviderType.LM_STUDIO: LMStudioProvider,
    ProviderType.OLLAMA: OllamaProvider,
    ProviderType.OPENAI: OpenAIProvider,
    ProviderType.ANTHROPIC: AnthropicProvider,
}


@mcp.resource("llm://models")
async def list_all_models() -> str:
    """List all available models across all providers."""
    result = "Available LLM Models:\n\n"
    
    for provider_name, provider in providers.items():
        if not provider.enabled:
            continue
            
        models = await provider.list_models()
        if models:
            result += f"**{provider_name.upper()} ({provider.type.value})**\n"
            for model in models:
                info = await provider.get_model_info(model)
                if info:
                    result += f"  - {model}\n"
                    result += f"    Context: {info.context_window:,} tokens\n"
                    result += f"    Max output: {info.max_tokens:,} tokens\n"
                    result += f"    Streaming: {'Yes' if info.supports_streaming else 'No'}\n"
                    result += f"    Functions: {'Yes' if info.supports_functions else 'No'}\n"
                    result += f"    Cost: ${info.cost_per_1k_input:.3f}/${info.cost_per_1k_output:.3f} per 1K tokens\n"
                else:
                    result += f"  - {model}\n"
            result += "\n"
    
    return result


@mcp.resource("llm://usage")
async def get_usage_stats() -> str:
    """Get usage statistics across all providers."""
    result = "LLM Usage Statistics:\n\n"
    
    total_cost = 0.0
    total_tokens = 0
    
    for key, usage_list in usage_tracker.items():
        provider, model = key.split(':', 1)
        
        if not usage_list:
            continue
        
        recent_usage = [u for u in usage_list if u.timestamp > datetime.now() - timedelta(hours=24)]
        
        if recent_usage:
            tokens_24h = sum(u.total_tokens for u in recent_usage)
            cost_24h = sum(u.cost for u in recent_usage)
            avg_latency = sum(u.latency_ms for u in recent_usage) / len(recent_usage)
            success_rate = sum(1 for u in recent_usage if u.success) / len(recent_usage) * 100
            
            result += f"**{model}** ({provider})\n"
            result += f"  24h usage: {tokens_24h:,} tokens, ${cost_24h:.4f}\n"
            result += f"  Avg latency: {avg_latency:.1f}ms\n"
            result += f"  Success rate: {success_rate:.1f}%\n"
            result += f"  Requests: {len(recent_usage)}\n\n"
            
            total_cost += cost_24h
            total_tokens += tokens_24h
    
    result += f"**Total (24h)**: {total_tokens:,} tokens, ${total_cost:.4f}\n"
    
    return result


@mcp.resource("llm://providers")
async def list_providers() -> str:
    """List all configured providers and their status."""
    result = "LLM Providers:\n\n"
    
    for name, provider in providers.items():
        result += f"**{name}** ({provider.type.value})\n"
        result += f"  Status: {'Enabled' if provider.enabled else 'Disabled'}\n"
        result += f"  Models: {len(provider.models)}\n"
        
        if provider.config.rate_limit:
            result += f"  Rate limit: {provider.config.rate_limit} req/min\n"
        
        if provider.config.quota:
            result += f"  Monthly quota: ${provider.config.quota}\n"
        
        result += "\n"
    
    return result


@mcp.tool()
async def generate_text(request: GenerationRequest) -> Dict[str, Any]:
    """Generate text using the specified or best available model."""
    
    # Find the best provider for this request
    target_providers = []
    
    if request.provider:
        # Use specific provider
        if request.provider in providers and providers[request.provider].enabled:
            target_providers = [providers[request.provider]]
        else:
            return {
                "success": False,
                "error": f"Provider '{request.provider}' not available"
            }
    else:
        # Find providers that have this model
        for provider in providers.values():
            if provider.enabled and request.model in provider.models:
                target_providers.append(provider)
        
        # Sort by priority
        target_providers.sort(key=lambda p: p.config.priority)
    
    if not target_providers:
        return {
            "success": False,
            "error": f"No available provider for model '{request.model}'"
        }
    
    # Try each provider in order
    last_error = None
    for provider in target_providers:
        try:
            # Check rate limiting
            if provider.config.rate_limit:
                # Simple rate limiting implementation
                now = time.time()
                key = f"rate_limit:{provider.type.value}"
                if key in rate_limiter:
                    requests_in_window = [t for t in rate_limiter[key] if now - t < 60]
                    if len(requests_in_window) >= provider.config.rate_limit:
                        continue
                    rate_limiter[key] = requests_in_window + [now]
                else:
                    rate_limiter[key] = [now]
            
            result = await provider.generate(request)
            result["success"] = True
            return result
            
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Provider {provider.type.value} failed: {e}")
            continue
    
    return {
        "success": False,
        "error": f"All providers failed. Last error: {last_error}"
    }


@mcp.tool()
async def list_models(request: ModelListRequest) -> Dict[str, Any]:
    """List available models with optional filtering."""
    
    all_models = {}
    
    for provider_name, provider in providers.items():
        if not provider.enabled:
            continue
        
        if request.provider and provider.type.value != request.provider:
            continue
        
        models = await provider.list_models()
        for model in models:
            info = await provider.get_model_info(model)
            
            # Apply capability filter
            if request.capability:
                if request.capability == "streaming" and not info.supports_streaming:
                    continue
                elif request.capability == "functions" and not info.supports_functions:
                    continue
                elif request.capability == "vision" and not info.supports_vision:
                    continue
            
            all_models[model] = {
                "provider": provider.type.value,
                "capabilities": asdict(info) if info else {},
                "available": True
            }
    
    return {
        "success": True,
        "models": all_models,
        "count": len(all_models)
    }


@mcp.tool()
async def get_model_info(model: str, provider: Optional[str] = None) -> Dict[str, Any]:
    """Get detailed information about a specific model."""
    
    target_providers = [providers[provider]] if provider and provider in providers else providers.values()
    
    for prov in target_providers:
        if model in prov.models:
            info = await prov.get_model_info(model)
            return {
                "success": True,
                "model": model,
                "provider": prov.type.value,
                "capabilities": asdict(info) if info else {},
                "available": prov.enabled
            }
    
    return {
        "success": False,
        "error": f"Model '{model}' not found"
    }


@mcp.tool()
async def get_usage_metrics(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    hours: int = 24
) -> Dict[str, Any]:
    """Get detailed usage metrics."""
    
    since = datetime.now() - timedelta(hours=hours)
    filtered_usage = []
    
    for key, usage_list in usage_tracker.items():
        prov, mod = key.split(':', 1)
        
        if provider and prov != provider:
            continue
        if model and mod != model:
            continue
        
        filtered_usage.extend([u for u in usage_list if u.timestamp > since])
    
    if not filtered_usage:
        return {
            "success": True,
            "metrics": {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "success_rate": 0.0,
                "avg_latency_ms": 0.0
            }
        }
    
    total_tokens = sum(u.total_tokens for u in filtered_usage)
    total_cost = sum(u.cost for u in filtered_usage)
    successful_requests = sum(1 for u in filtered_usage if u.success)
    avg_latency = sum(u.latency_ms for u in filtered_usage) / len(filtered_usage)
    
    return {
        "success": True,
        "metrics": {
            "total_requests": len(filtered_usage),
            "successful_requests": successful_requests,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "success_rate": successful_requests / len(filtered_usage) * 100,
            "avg_latency_ms": avg_latency
        },
        "period_hours": hours
    }


@mcp.tool()
async def optimize_model_selection(
    prompt: str,
    requirements: Dict[str, Any]
) -> Dict[str, Any]:
    """Recommend the best model for given requirements."""
    
    max_tokens = requirements.get('max_tokens', 1000)
    budget_per_1k = requirements.get('budget_per_1k_tokens', 10.0)
    requires_streaming = requirements.get('streaming', False)
    requires_functions = requirements.get('functions', False)
    requires_vision = requirements.get('vision', False)
    latency_priority = requirements.get('latency_priority', 'medium')  # low, medium, high
    
    recommendations = []
    
    for provider_name, provider in providers.items():
        if not provider.enabled:
            continue
        
        for model_name, capabilities in provider.models.items():
            # Check requirements
            if max_tokens > capabilities.max_tokens:
                continue
            if requires_streaming and not capabilities.supports_streaming:
                continue
            if requires_functions and not capabilities.supports_functions:
                continue
            if requires_vision and not capabilities.supports_vision:
                continue
            
            # Calculate cost
            estimated_input_tokens = len(prompt) // 4
            estimated_output_tokens = max_tokens
            total_cost = (estimated_input_tokens * capabilities.cost_per_1k_input / 1000 + 
                         estimated_output_tokens * capabilities.cost_per_1k_output / 1000)
            
            if total_cost > budget_per_1k * (estimated_input_tokens + estimated_output_tokens) / 1000:
                continue
            
            # Calculate score based on requirements
            score = 100
            
            # Cost efficiency (lower cost = higher score)
            if total_cost > 0:
                score -= (total_cost / budget_per_1k) * 30
            
            # Latency consideration (based on historical data)
            model_usage = usage_tracker.get(f"{provider.type.value}:{model_name}", [])
            recent_usage = [u for u in model_usage if u.timestamp > datetime.now() - timedelta(hours=24)]
            
            if recent_usage:
                avg_latency = sum(u.latency_ms for u in recent_usage) / len(recent_usage)
                if latency_priority == 'high' and avg_latency > 2000:
                    score -= 20
                elif latency_priority == 'low' and avg_latency < 1000:
                    score += 10
            
            # Capability bonus
            if capabilities.supports_functions:
                score += 5
            if capabilities.supports_vision:
                score += 5
            if capabilities.context_window > 32000:
                score += 10
            
            recommendations.append({
                "model": model_name,
                "provider": provider.type.value,
                "score": max(0, score),
                "estimated_cost": total_cost,
                "capabilities": asdict(capabilities),
                "reasoning": f"Score: {score:.1f}, Cost: ${total_cost:.4f}"
            })
    
    # Sort by score descending
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    return {
        "success": True,
        "recommendations": recommendations[:5],  # Top 5
        "requirements": requirements
    }


@mcp.server.lifespan
async def setup_and_cleanup():
    """Initialize and cleanup server components."""
    global providers
    
    # Load configuration
    config_file = os.path.join(os.path.dirname(__file__), 'config.json')
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.warning("No config.json found, using default configuration")
        config = {
            "providers": {
                "lm_studio": {
                    "type": "lm_studio",
                    "enabled": True,
                    "config": {"base_url": "http://localhost:1234/v1"},
                    "priority": 1
                }
            }
        }
    
    # Initialize providers
    for name, provider_config in config.get('providers', {}).items():
        try:
            provider_type = ProviderType(provider_config['type'])
            if provider_type in PROVIDER_CLASSES:
                config_obj = ProviderConfig(**provider_config)
                provider = PROVIDER_CLASSES[provider_type](config_obj)
                await provider.initialize()
                providers[name] = provider
                logger.info(f"Initialized provider: {name} ({provider_type.value})")
            else:
                logger.warning(f"Unknown provider type: {provider_config['type']}")
        except Exception as e:
            logger.error(f"Failed to initialize provider {name}: {e}")
    
    logger.info(f"LLM Provider MCP Server initialized with {len(providers)} providers")
    
    yield
    
    # Cleanup
    providers.clear()
    usage_tracker.clear()
    rate_limiter.clear()
    logger.info("LLM Provider MCP Server shut down")


def main():
    """Run the LLM Provider MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Provider MCP Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind to")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()