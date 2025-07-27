import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, List, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CompletionRequest:
    prompt: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 2000
    stream: bool = False


@dataclass
class CompletionResponse:
    content: str
    model: str
    usage: Dict[str, Any]
    finish_reason: str


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate_completion(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion for the given request."""
        pass
    
    @abstractmethod
    async def generate_streaming(self, request: CompletionRequest) -> AsyncIterator[str]:
        """Generate streaming completion for the given request."""
        pass
    
    @abstractmethod
    async def embed_text(self, text: str, model: str = None) -> List[float]:
        """Generate embeddings for the given text."""
        pass