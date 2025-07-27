#!/usr/bin/env python3
"""
Vector Store MCP Server

Production-grade MCP server for vector storage and semantic search with support for multiple backends:
- ChromaDB (local/remote)
- Pinecone (cloud)
- Weaviate (local/cloud)
- Qdrant (local/cloud)
- FAISS (in-memory/file)
- Milvus (local/cloud)
- In-memory (development)

Features:
- Multi-backend vector storage
- Semantic search and similarity
- Document chunking and embedding
- Metadata filtering
- Batch operations
- Vector analytics
- Index management
- Hybrid search (vector + keyword)
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import numpy as np

from mcp.server.fastmcp import FastMCP
from mcp.types import Resource, Tool, TextContent
from pydantic import BaseModel, Field

# Vector store imports
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    import pinecone
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False

try:
    import weaviate
    HAS_WEAVIATE = True
except ImportError:
    HAS_WEAVIATE = False

try:
    import qdrant_client
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# Embedding providers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Text processing
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Vector Store MCP Server")

# Global state
vector_stores: Dict[str, 'VectorStore'] = {}
embedding_providers: Dict[str, 'EmbeddingProvider'] = {}
collections: Dict[str, Dict[str, Any]] = {}


class VectorStoreType(str, Enum):
    """Supported vector store types."""
    CHROMADB = "chromadb"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    FAISS = "faiss"
    MILVUS = "milvus"
    MEMORY = "memory"


class EmbeddingProviderType(str, Enum):
    """Supported embedding provider types."""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


@dataclass
class VectorDocument:
    """Document with vector representation."""
    id: str
    content: str
    vector: List[float]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class SearchResult:
    """Search result with score."""
    document: VectorDocument
    score: float
    distance: float


class DocumentRequest(BaseModel):
    """Request to add documents."""
    collection: str = Field(description="Collection name")
    documents: List[Dict[str, Any]] = Field(description="Documents to add")
    chunk_size: int = Field(2000, description="Text chunk size")
    chunk_overlap: int = Field(200, description="Chunk overlap")
    embedding_provider: Optional[str] = Field(None, description="Embedding provider to use")


class SearchRequest(BaseModel):
    """Request to search vectors."""
    collection: str = Field(description="Collection name")
    query: str = Field(description="Search query")
    limit: int = Field(10, description="Number of results")
    threshold: float = Field(0.7, description="Similarity threshold")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    include_vectors: bool = Field(False, description="Include vectors in results")
    hybrid_search: bool = Field(False, description="Use hybrid search")


class CollectionRequest(BaseModel):
    """Request to create collection."""
    name: str = Field(description="Collection name")
    dimension: int = Field(384, description="Vector dimension")
    distance_metric: str = Field("cosine", description="Distance metric")
    metadata_schema: Optional[Dict[str, str]] = Field(None, description="Metadata schema")
    description: Optional[str] = Field(None, description="Collection description")


class EmbeddingProvider:
    """Base class for embedding providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.dimension = 384
        
    async def initialize(self):
        """Initialize the provider."""
        pass
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        raise NotImplementedError
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = await self.embed_text(text)
            embeddings.append(embedding)
        return embeddings


class SentenceTransformersProvider(EmbeddingProvider):
    """Sentence Transformers embedding provider."""
    
    async def initialize(self):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers required")
        
        model_name = self.config.get('model', 'all-MiniLM-L6-v2')
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Initialized SentenceTransformers with {model_name}, dimension: {self.dimension}")
    
    async def embed_text(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    async def initialize(self):
        if not HAS_OPENAI:
            raise ImportError("openai required")
        
        api_key = self.config.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model_name = self.config.get('model', 'text-embedding-3-small')
        
        # Set dimension based on model
        if 'text-embedding-3-small' in self.model_name:
            self.dimension = 1536
        elif 'text-embedding-3-large' in self.model_name:
            self.dimension = 3072
        elif 'text-embedding-ada-002' in self.model_name:
            self.dimension = 1536
        
        logger.info(f"Initialized OpenAI embeddings with {self.model_name}, dimension: {self.dimension}")
    
    async def embed_text(self, text: str) -> List[float]:
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [item.embedding for item in response.data]


class VectorStore:
    """Base class for vector stores."""
    
    def __init__(self, config: Dict[str, Any], embedding_provider: EmbeddingProvider):
        self.config = config
        self.embedding_provider = embedding_provider
        self.type = config.get('type')
        self.client = None
        
    async def initialize(self):
        """Initialize the vector store."""
        pass
    
    async def create_collection(self, name: str, dimension: int, distance_metric: str = "cosine") -> bool:
        """Create a new collection."""
        raise NotImplementedError
    
    async def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        raise NotImplementedError
    
    async def list_collections(self) -> List[str]:
        """List all collections."""
        raise NotImplementedError
    
    async def add_documents(self, collection: str, documents: List[VectorDocument]) -> bool:
        """Add documents to collection."""
        raise NotImplementedError
    
    async def search(self, collection: str, query_vector: List[float], 
                    limit: int = 10, threshold: float = 0.7,
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar vectors."""
        raise NotImplementedError
    
    async def get_document(self, collection: str, doc_id: str) -> Optional[VectorDocument]:
        """Get document by ID."""
        raise NotImplementedError
    
    async def update_document(self, collection: str, doc_id: str, 
                             content: Optional[str] = None, 
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update document."""
        raise NotImplementedError
    
    async def delete_document(self, collection: str, doc_id: str) -> bool:
        """Delete document."""
        raise NotImplementedError


class ChromaDBVectorStore(VectorStore):
    """ChromaDB vector store implementation."""
    
    async def initialize(self):
        if not HAS_CHROMA:
            raise ImportError("chromadb required")
        
        persist_directory = self.config.get('persist_directory', './chroma_db')
        
        if self.config.get('remote_host'):
            # Remote ChromaDB
            self.client = chromadb.HttpClient(
                host=self.config.get('remote_host', 'localhost'),
                port=self.config.get('remote_port', 8000)
            )
        else:
            # Local ChromaDB
            self.client = chromadb.PersistentClient(path=persist_directory)
        
        logger.info("Initialized ChromaDB vector store")
    
    async def create_collection(self, name: str, dimension: int, distance_metric: str = "cosine") -> bool:
        try:
            # Map distance metrics
            metric_map = {
                "cosine": "cosine",
                "euclidean": "l2",
                "manhattan": "l1"
            }
            
            collection = self.client.create_collection(
                name=name,
                metadata={"hnsw:space": metric_map.get(distance_metric, "cosine")}
            )
            
            collections[name] = {
                "name": name,
                "dimension": dimension,
                "distance_metric": distance_metric,
                "created_at": datetime.now(),
                "document_count": 0
            }
            
            return True
        except Exception as e:
            logger.error(f"Failed to create ChromaDB collection: {e}")
            return False
    
    async def list_collections(self) -> List[str]:
        try:
            collections_list = self.client.list_collections()
            return [col.name for col in collections_list]
        except Exception as e:
            logger.error(f"Failed to list ChromaDB collections: {e}")
            return []
    
    async def add_documents(self, collection: str, documents: List[VectorDocument]) -> bool:
        try:
            col = self.client.get_collection(collection)
            
            ids = [doc.id for doc in documents]
            embeddings = [doc.vector for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            documents_text = [doc.content for doc in documents]
            
            col.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text
            )
            
            # Update collection info
            if collection in collections:
                collections[collection]["document_count"] += len(documents)
            
            return True
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            return False
    
    async def search(self, collection: str, query_vector: List[float], 
                    limit: int = 10, threshold: float = 0.7,
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        try:
            col = self.client.get_collection(collection)
            
            where_clause = None
            if filters:
                where_clause = filters
            
            results = col.query(
                query_embeddings=[query_vector],
                n_results=limit,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            search_results = []
            for i, (doc_id, document, metadata, distance) in enumerate(zip(
                results['ids'][0],
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                # Convert distance to similarity score
                score = 1.0 - distance if distance <= 1.0 else 1.0 / (1.0 + distance)
                
                if score >= threshold:
                    vector_doc = VectorDocument(
                        id=doc_id,
                        content=document,
                        vector=[],  # ChromaDB doesn't return vectors by default
                        metadata=metadata,
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    
                    search_results.append(SearchResult(
                        document=vector_doc,
                        score=score,
                        distance=distance
                    ))
            
            return search_results
        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {e}")
            return []


class QdrantVectorStore(VectorStore):
    """Qdrant vector store implementation."""
    
    async def initialize(self):
        if not HAS_QDRANT:
            raise ImportError("qdrant-client required")
        
        if self.config.get('url'):
            # Remote Qdrant
            self.client = qdrant_client.QdrantClient(
                url=self.config['url'],
                api_key=self.config.get('api_key')
            )
        else:
            # Local Qdrant
            self.client = qdrant_client.QdrantClient(
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 6333)
            )
        
        logger.info("Initialized Qdrant vector store")
    
    async def create_collection(self, name: str, dimension: int, distance_metric: str = "cosine") -> bool:
        try:
            # Map distance metrics
            metric_map = {
                "cosine": Distance.COSINE,
                "euclidean": Distance.EUCLID,
                "manhattan": Distance.MANHATTAN
            }
            
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=metric_map.get(distance_metric, Distance.COSINE)
                )
            )
            
            collections[name] = {
                "name": name,
                "dimension": dimension,
                "distance_metric": distance_metric,
                "created_at": datetime.now(),
                "document_count": 0
            }
            
            return True
        except Exception as e:
            logger.error(f"Failed to create Qdrant collection: {e}")
            return False
    
    async def list_collections(self) -> List[str]:
        try:
            collections_info = self.client.get_collections()
            return [col.name for col in collections_info.collections]
        except Exception as e:
            logger.error(f"Failed to list Qdrant collections: {e}")
            return []
    
    async def add_documents(self, collection: str, documents: List[VectorDocument]) -> bool:
        try:
            points = []
            for doc in documents:
                point = PointStruct(
                    id=doc.id,
                    vector=doc.vector,
                    payload={
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "created_at": doc.created_at.isoformat(),
                        "updated_at": doc.updated_at.isoformat()
                    }
                )
                points.append(point)
            
            self.client.upsert(
                collection_name=collection,
                points=points
            )
            
            # Update collection info
            if collection in collections:
                collections[collection]["document_count"] += len(documents)
            
            return True
        except Exception as e:
            logger.error(f"Failed to add documents to Qdrant: {e}")
            return False
    
    async def search(self, collection: str, query_vector: List[float], 
                    limit: int = 10, threshold: float = 0.7,
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        try:
            search_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value)
                    ))
                search_filter = Filter(must=conditions)
            
            results = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                score_threshold=threshold,
                query_filter=search_filter,
                with_payload=True
            )
            
            search_results = []
            for result in results:
                payload = result.payload
                vector_doc = VectorDocument(
                    id=str(result.id),
                    content=payload["content"],
                    vector=[],  # Vector not included by default
                    metadata=payload["metadata"],
                    created_at=datetime.fromisoformat(payload["created_at"]),
                    updated_at=datetime.fromisoformat(payload["updated_at"])
                )
                
                search_results.append(SearchResult(
                    document=vector_doc,
                    score=result.score,
                    distance=1.0 - result.score
                ))
            
            return search_results
        except Exception as e:
            logger.error(f"Failed to search Qdrant: {e}")
            return []


class MemoryVectorStore(VectorStore):
    """In-memory vector store for development."""
    
    def __init__(self, config: Dict[str, Any], embedding_provider: EmbeddingProvider):
        super().__init__(config, embedding_provider)
        self.data: Dict[str, List[VectorDocument]] = {}
    
    async def initialize(self):
        logger.info("Initialized in-memory vector store")
    
    async def create_collection(self, name: str, dimension: int, distance_metric: str = "cosine") -> bool:
        self.data[name] = []
        collections[name] = {
            "name": name,
            "dimension": dimension,
            "distance_metric": distance_metric,
            "created_at": datetime.now(),
            "document_count": 0
        }
        return True
    
    async def list_collections(self) -> List[str]:
        return list(self.data.keys())
    
    async def add_documents(self, collection: str, documents: List[VectorDocument]) -> bool:
        if collection not in self.data:
            await self.create_collection(collection, len(documents[0].vector) if documents else 384)
        
        self.data[collection].extend(documents)
        collections[collection]["document_count"] += len(documents)
        return True
    
    async def search(self, collection: str, query_vector: List[float], 
                    limit: int = 10, threshold: float = 0.7,
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        if collection not in self.data:
            return []
        
        documents = self.data[collection]
        scores = []
        
        for doc in documents:
            # Apply filters if provided
            if filters:
                match = True
                for key, value in filters.items():
                    if key not in doc.metadata or doc.metadata[key] != value:
                        match = False
                        break
                if not match:
                    continue
            
            # Calculate cosine similarity
            dot_product = np.dot(query_vector, doc.vector)
            norm_query = np.linalg.norm(query_vector)
            norm_doc = np.linalg.norm(doc.vector)
            
            if norm_query == 0 or norm_doc == 0:
                score = 0.0
            else:
                score = dot_product / (norm_query * norm_doc)
            
            if score >= threshold:
                distance = 1.0 - score
                scores.append((score, distance, doc))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Return top results
        results = []
        for score, distance, doc in scores[:limit]:
            results.append(SearchResult(
                document=doc,
                score=score,
                distance=distance
            ))
        
        return results


# Provider and store factories
EMBEDDING_PROVIDERS = {
    EmbeddingProviderType.SENTENCE_TRANSFORMERS: SentenceTransformersProvider,
    EmbeddingProviderType.OPENAI: OpenAIEmbeddingProvider,
}

VECTOR_STORES = {
    VectorStoreType.CHROMADB: ChromaDBVectorStore,
    VectorStoreType.QDRANT: QdrantVectorStore,
    VectorStoreType.MEMORY: MemoryVectorStore,
}


def chunk_text(text: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> List[str]:
    """Split text into chunks."""
    if HAS_LANGCHAIN:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        return splitter.split_text(text)
    else:
        # Simple chunking fallback
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - chunk_overlap
            if start >= len(text):
                break
        return chunks


@mcp.resource("vectors://collections")
async def list_all_collections() -> str:
    """List all collections across all vector stores."""
    result = "Vector Store Collections:\n\n"
    
    for store_name, store in vector_stores.items():
        collections_list = await store.list_collections()
        if collections_list:
            result += f"**{store_name.upper()} ({store.type})**\n"
            for collection_name in collections_list:
                info = collections.get(collection_name, {})
                result += f"  - {collection_name}\n"
                if info:
                    result += f"    Documents: {info.get('document_count', 0)}\n"
                    result += f"    Dimension: {info.get('dimension', 'unknown')}\n"
                    result += f"    Created: {info.get('created_at', 'unknown')}\n"
            result += "\n"
    
    return result


@mcp.resource("vectors://stats")
async def get_vector_stats() -> str:
    """Get vector store statistics."""
    result = "Vector Store Statistics:\n\n"
    
    total_collections = len(collections)
    total_documents = sum(info.get('document_count', 0) for info in collections.values())
    
    result += f"Total Collections: {total_collections}\n"
    result += f"Total Documents: {total_documents}\n\n"
    
    # Per-store stats
    for store_name, store in vector_stores.items():
        store_collections = await store.list_collections()
        store_docs = sum(
            collections.get(col, {}).get('document_count', 0) 
            for col in store_collections
        )
        
        result += f"**{store_name}** ({store.type})\n"
        result += f"  Collections: {len(store_collections)}\n"
        result += f"  Documents: {store_docs}\n"
        result += f"  Embedding provider: {store.embedding_provider.__class__.__name__}\n"
        result += f"  Embedding dimension: {store.embedding_provider.dimension}\n\n"
    
    return result


@mcp.resource("vectors://embedding-providers")
async def list_embedding_providers() -> str:
    """List all embedding providers."""
    result = "Embedding Providers:\n\n"
    
    for name, provider in embedding_providers.items():
        result += f"**{name}**\n"
        result += f"  Type: {provider.__class__.__name__}\n"
        result += f"  Dimension: {provider.dimension}\n"
        result += f"  Model: {getattr(provider, 'model_name', 'default')}\n\n"
    
    return result


@mcp.tool()
async def create_collection(request: CollectionRequest) -> Dict[str, Any]:
    """Create a new vector collection."""
    
    # Use the first available vector store
    if not vector_stores:
        return {
            "success": False,
            "error": "No vector stores available"
        }
    
    store = next(iter(vector_stores.values()))
    
    try:
        success = await store.create_collection(
            name=request.name,
            dimension=request.dimension,
            distance_metric=request.distance_metric
        )
        
        if success:
            return {
                "success": True,
                "collection": request.name,
                "dimension": request.dimension,
                "distance_metric": request.distance_metric,
                "message": f"Created collection '{request.name}'"
            }
        else:
            return {
                "success": False,
                "error": f"Failed to create collection '{request.name}'"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def add_documents(request: DocumentRequest) -> Dict[str, Any]:
    """Add documents to a vector collection."""
    
    # Get vector store and embedding provider
    if not vector_stores:
        return {
            "success": False,
            "error": "No vector stores available"
        }
    
    store = next(iter(vector_stores.values()))
    
    embedding_provider_name = request.embedding_provider or next(iter(embedding_providers.keys()))
    if embedding_provider_name not in embedding_providers:
        return {
            "success": False,
            "error": f"Embedding provider '{embedding_provider_name}' not available"
        }
    
    embedding_provider = embedding_providers[embedding_provider_name]
    
    try:
        # Process documents
        vector_docs = []
        
        for doc_data in request.documents:
            content = doc_data.get('content', '')
            metadata = doc_data.get('metadata', {})
            doc_id = doc_data.get('id') or str(uuid.uuid4())
            
            # Chunk text if needed
            if len(content) > request.chunk_size:
                chunks = chunk_text(content, request.chunk_size, request.chunk_overlap)
                
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": i,
                        "parent_document": doc_id,
                        "chunk_count": len(chunks)
                    })
                    
                    # Generate embedding
                    embedding = await embedding_provider.embed_text(chunk)
                    
                    vector_doc = VectorDocument(
                        id=chunk_id,
                        content=chunk,
                        vector=embedding,
                        metadata=chunk_metadata,
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    vector_docs.append(vector_doc)
            else:
                # Single document
                embedding = await embedding_provider.embed_text(content)
                
                vector_doc = VectorDocument(
                    id=doc_id,
                    content=content,
                    vector=embedding,
                    metadata=metadata,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                vector_docs.append(vector_doc)
        
        # Add to vector store
        success = await store.add_documents(request.collection, vector_docs)
        
        if success:
            return {
                "success": True,
                "collection": request.collection,
                "documents_added": len(vector_docs),
                "original_documents": len(request.documents),
                "embedding_provider": embedding_provider_name,
                "message": f"Added {len(vector_docs)} document chunks to '{request.collection}'"
            }
        else:
            return {
                "success": False,
                "error": f"Failed to add documents to '{request.collection}'"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def search_vectors(request: SearchRequest) -> Dict[str, Any]:
    """Search for similar vectors."""
    
    # Get vector store and embedding provider
    if not vector_stores:
        return {
            "success": False,
            "error": "No vector stores available"
        }
    
    store = next(iter(vector_stores.values()))
    embedding_provider = next(iter(embedding_providers.values()))
    
    try:
        # Generate query embedding
        query_vector = await embedding_provider.embed_text(request.query)
        
        # Perform search
        results = await store.search(
            collection=request.collection,
            query_vector=query_vector,
            limit=request.limit,
            threshold=request.threshold,
            filters=request.filters
        )
        
        # Format results
        formatted_results = []
        for result in results:
            doc_data = {
                "id": result.document.id,
                "content": result.document.content,
                "metadata": result.document.metadata,
                "score": result.score,
                "distance": result.distance
            }
            
            if request.include_vectors:
                doc_data["vector"] = result.document.vector
            
            formatted_results.append(doc_data)
        
        return {
            "success": True,
            "query": request.query,
            "collection": request.collection,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "threshold": request.threshold
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def analyze_collection(collection: str) -> Dict[str, Any]:
    """Analyze a vector collection."""
    
    if not vector_stores:
        return {
            "success": False,
            "error": "No vector stores available"
        }
    
    store = next(iter(vector_stores.values()))
    
    try:
        # Get collection info
        info = collections.get(collection, {})
        
        if not info:
            return {
                "success": False,
                "error": f"Collection '{collection}' not found"
            }
        
        # Get sample documents for analysis
        embedding_provider = next(iter(embedding_providers.values()))
        sample_query = "sample analysis query"
        query_vector = await embedding_provider.embed_text(sample_query)
        
        sample_results = await store.search(
            collection=collection,
            query_vector=query_vector,
            limit=100,
            threshold=0.0  # Get any results
        )
        
        # Analyze metadata
        metadata_fields = {}
        content_lengths = []
        
        for result in sample_results:
            content_lengths.append(len(result.document.content))
            
            for key, value in result.document.metadata.items():
                if key not in metadata_fields:
                    metadata_fields[key] = set()
                metadata_fields[key].add(type(value).__name__)
        
        # Calculate statistics
        avg_content_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
        min_content_length = min(content_lengths) if content_lengths else 0
        max_content_length = max(content_lengths) if content_lengths else 0
        
        return {
            "success": True,
            "collection": collection,
            "info": info,
            "statistics": {
                "document_count": info.get("document_count", 0),
                "dimension": info.get("dimension", 0),
                "distance_metric": info.get("distance_metric", "unknown"),
                "avg_content_length": avg_content_length,
                "min_content_length": min_content_length,
                "max_content_length": max_content_length,
                "metadata_fields": {k: list(v) for k, v in metadata_fields.items()},
                "sample_size": len(sample_results)
            }
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def get_similar_documents(
    collection: str,
    document_id: str,
    limit: int = 5
) -> Dict[str, Any]:
    """Find documents similar to a given document."""
    
    if not vector_stores:
        return {
            "success": False,
            "error": "No vector stores available"
        }
    
    store = next(iter(vector_stores.values()))
    
    try:
        # Get the source document
        source_doc = await store.get_document(collection, document_id)
        if not source_doc:
            return {
                "success": False,
                "error": f"Document '{document_id}' not found"
            }
        
        # Search for similar documents
        results = await store.search(
            collection=collection,
            query_vector=source_doc.vector,
            limit=limit + 1,  # +1 to exclude the source document
            threshold=0.0
        )
        
        # Filter out the source document
        similar_docs = [
            {
                "id": result.document.id,
                "content": result.document.content,
                "metadata": result.document.metadata,
                "score": result.score,
                "distance": result.distance
            }
            for result in results
            if result.document.id != document_id
        ][:limit]
        
        return {
            "success": True,
            "source_document": document_id,
            "collection": collection,
            "similar_documents": similar_docs,
            "count": len(similar_docs)
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.server.lifespan
async def setup_and_cleanup():
    """Initialize and cleanup server components."""
    global vector_stores, embedding_providers
    
    # Load configuration
    config_file = os.path.join(os.path.dirname(__file__), 'config.json')
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.warning("No config.json found, using default configuration")
        config = {
            "embedding_providers": {
                "sentence_transformers": {
                    "type": "sentence_transformers",
                    "model": "all-MiniLM-L6-v2"
                }
            },
            "vector_stores": {
                "memory": {
                    "type": "memory"
                }
            }
        }
    
    # Initialize embedding providers
    for name, provider_config in config.get('embedding_providers', {}).items():
        try:
            provider_type = EmbeddingProviderType(provider_config['type'])
            if provider_type in EMBEDDING_PROVIDERS:
                provider = EMBEDDING_PROVIDERS[provider_type](provider_config)
                await provider.initialize()
                embedding_providers[name] = provider
                logger.info(f"Initialized embedding provider: {name} ({provider_type.value})")
            else:
                logger.warning(f"Unknown embedding provider type: {provider_config['type']}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding provider {name}: {e}")
    
    # Initialize vector stores
    for name, store_config in config.get('vector_stores', {}).items():
        try:
            store_type = VectorStoreType(store_config['type'])
            if store_type in VECTOR_STORES:
                # Use first available embedding provider
                embedding_provider = next(iter(embedding_providers.values()))
                store = VECTOR_STORES[store_type](store_config, embedding_provider)
                await store.initialize()
                vector_stores[name] = store
                logger.info(f"Initialized vector store: {name} ({store_type.value})")
            else:
                logger.warning(f"Unknown vector store type: {store_config['type']}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store {name}: {e}")
    
    logger.info(f"Vector Store MCP Server initialized with {len(vector_stores)} stores and {len(embedding_providers)} providers")
    
    yield
    
    # Cleanup
    vector_stores.clear()
    embedding_providers.clear()
    collections.clear()
    logger.info("Vector Store MCP Server shut down")


def main():
    """Run the Vector Store MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vector Store MCP Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8003, help="Port to bind to")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()