import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from uuid import uuid4

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("ChromaDB not available, using memory vector store only")

from config.settings import settings
from utils.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any] = None
    embedding: List[float] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        result = asdict(self)
        if self.created_at:
            result['created_at'] = self.created_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create document from dictionary."""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class SearchResult:
    document: Document
    score: float
    distance: float = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "score": self.score,
            "distance": self.distance
        }


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    async def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """Perform similarity search."""
        pass
    
    @abstractmethod
    async def similarity_search_by_vector(
        self, 
        embedding: List[float], 
        k: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """Perform similarity search using vector."""
        pass
    
    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID."""
        pass
    
    @abstractmethod
    async def update_document(self, document_id: str, document: Document) -> bool:
        """Update document."""
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents by IDs."""
        pass
    
    @abstractmethod
    async def list_documents(self, limit: int = 100, offset: int = 0) -> List[Document]:
        """List documents with pagination."""
        pass
    
    @abstractmethod
    async def count_documents(self) -> int:
        """Count total documents."""
        pass


class MemoryVectorStore(BaseVectorStore):
    """In-memory vector store implementation for development."""
    
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        logger.info("Initialized memory vector store")
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(a * a for a in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _simple_text_embedding(self, text: str) -> List[float]:
        """Create a simple text embedding for demonstration purposes."""
        # This is a very basic embedding based on character frequencies
        # In a real implementation, you would use proper embedding models
        embedding = [0.0] * 384  # Standard embedding size
        
        text_lower = text.lower()
        for i, char in enumerate(text_lower[:384]):
            embedding[i % 384] += ord(char) / 128.0 - 1.0
        
        # Normalize
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        added_ids = []
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = self._simple_text_embedding(doc.content)
            
            self.documents[doc.id] = doc
            added_ids.append(doc.id)
        
        logger.debug(f"Added {len(added_ids)} documents to memory vector store")
        return added_ids
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> List[SearchResult]:
        query_embedding = self._simple_text_embedding(query)
        return await self.similarity_search_by_vector(query_embedding, k, filter_metadata)
    
    async def similarity_search_by_vector(
        self, 
        embedding: List[float], 
        k: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> List[SearchResult]:
        results = []
        
        for doc in self.documents.values():
            # Apply metadata filter if provided
            if filter_metadata:
                if not all(doc.metadata.get(key) == value for key, value in filter_metadata.items()):
                    continue
            
            if doc.embedding:
                similarity = self._calculate_cosine_similarity(embedding, doc.embedding)
                distance = 1.0 - similarity
                
                results.append(SearchResult(
                    document=doc,
                    score=similarity,
                    distance=distance
                ))
        
        # Sort by similarity (highest first) and return top k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        return self.documents.get(document_id)
    
    async def update_document(self, document_id: str, document: Document) -> bool:
        if document_id in self.documents:
            if document.embedding is None:
                document.embedding = self._simple_text_embedding(document.content)
            self.documents[document_id] = document
            return True
        return False
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        deleted_count = 0
        for doc_id in document_ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
                deleted_count += 1
        
        logger.debug(f"Deleted {deleted_count} documents from memory vector store")
        return deleted_count > 0
    
    async def list_documents(self, limit: int = 100, offset: int = 0) -> List[Document]:
        all_docs = list(self.documents.values())
        return all_docs[offset:offset + limit]
    
    async def count_documents(self) -> int:
        return len(self.documents)


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation."""
    
    def __init__(self, host: str = "localhost", port: int = 8001, collection_name: str = "mcp_documents"):
        if not CHROMA_AVAILABLE:
            raise VectorStoreError("ChromaDB is not available. Install chromadb package.")
        
        try:
            self.client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=ChromaSettings(allow_reset=True, anonymized_telemetry=False)
            )
            
            self.collection_name = collection_name
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=collection_name)
                logger.info(f"Using existing ChromaDB collection: {collection_name}")
            except Exception:
                self.collection = self.client.create_collection(name=collection_name)
                logger.info(f"Created new ChromaDB collection: {collection_name}")
            
            logger.info(f"Initialized ChromaDB vector store: {host}:{port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise VectorStoreError(f"Failed to initialize ChromaDB: {e}")
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        try:
            ids = []
            contents = []
            metadatas = []
            embeddings = []
            
            for doc in documents:
                ids.append(doc.id)
                contents.append(doc.content)
                
                # Prepare metadata (ChromaDB requires serializable values)
                metadata = doc.metadata.copy()
                metadata['created_at'] = doc.created_at.isoformat()
                metadatas.append(metadata)
                
                # Use provided embedding or let ChromaDB generate one
                if doc.embedding:
                    embeddings.append(doc.embedding)
            
            # Add to ChromaDB
            if embeddings:
                self.collection.add(
                    ids=ids,
                    documents=contents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            else:
                # Let ChromaDB generate embeddings
                self.collection.add(
                    ids=ids,
                    documents=contents,
                    metadatas=metadatas
                )
            
            logger.debug(f"Added {len(ids)} documents to ChromaDB")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise VectorStoreError(f"Failed to add documents: {e}")
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> List[SearchResult]:
        try:
            # Prepare where clause for filtering
            where_clause = filter_metadata if filter_metadata else None
            
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=where_clause
            )
            
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    content = results['documents'][0][i] if results['documents'] else ""
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0.0
                    
                    # Convert distance to similarity score
                    score = 1.0 - distance if distance <= 1.0 else 0.0
                    
                    # Reconstruct created_at from metadata
                    created_at = datetime.utcnow()
                    if 'created_at' in metadata:
                        try:
                            created_at = datetime.fromisoformat(metadata.pop('created_at'))
                        except:
                            pass
                    
                    document = Document(
                        id=doc_id,
                        content=content,
                        metadata=metadata,
                        created_at=created_at
                    )
                    
                    search_results.append(SearchResult(
                        document=document,
                        score=score,
                        distance=distance
                    ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            raise VectorStoreError(f"Failed to search: {e}")
    
    async def similarity_search_by_vector(
        self, 
        embedding: List[float], 
        k: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> List[SearchResult]:
        try:
            where_clause = filter_metadata if filter_metadata else None
            
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=k,
                where=where_clause
            )
            
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    content = results['documents'][0][i] if results['documents'] else ""
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0.0
                    
                    score = 1.0 - distance if distance <= 1.0 else 0.0
                    
                    created_at = datetime.utcnow()
                    if 'created_at' in metadata:
                        try:
                            created_at = datetime.fromisoformat(metadata.pop('created_at'))
                        except:
                            pass
                    
                    document = Document(
                        id=doc_id,
                        content=content,
                        metadata=metadata,
                        created_at=created_at
                    )
                    
                    search_results.append(SearchResult(
                        document=document,
                        score=score,
                        distance=distance
                    ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB by vector: {e}")
            raise VectorStoreError(f"Failed to search by vector: {e}")
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        try:
            results = self.collection.get(ids=[document_id])
            
            if results['ids'] and document_id in results['ids']:
                idx = results['ids'].index(document_id)
                content = results['documents'][idx] if results['documents'] else ""
                metadata = results['metadatas'][idx] if results['metadatas'] else {}
                
                created_at = datetime.utcnow()
                if 'created_at' in metadata:
                    try:
                        created_at = datetime.fromisoformat(metadata.pop('created_at'))
                    except:
                        pass
                
                return Document(
                    id=document_id,
                    content=content,
                    metadata=metadata,
                    created_at=created_at
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting document from ChromaDB: {e}")
            return None
    
    async def update_document(self, document_id: str, document: Document) -> bool:
        try:
            # ChromaDB doesn't have direct update, so we need to delete and add
            await self.delete_documents([document_id])
            
            # Set the ID to match the one we're updating
            document.id = document_id
            await self.add_documents([document])
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating document in ChromaDB: {e}")
            return False
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        try:
            self.collection.delete(ids=document_ids)
            logger.debug(f"Deleted {len(document_ids)} documents from ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents from ChromaDB: {e}")
            return False
    
    async def list_documents(self, limit: int = 100, offset: int = 0) -> List[Document]:
        try:
            # ChromaDB doesn't support offset directly, so we get all and slice
            results = self.collection.get()
            
            documents = []
            if results['ids']:
                total_docs = len(results['ids'])
                start_idx = min(offset, total_docs)
                end_idx = min(offset + limit, total_docs)
                
                for i in range(start_idx, end_idx):
                    doc_id = results['ids'][i]
                    content = results['documents'][i] if results['documents'] else ""
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    
                    created_at = datetime.utcnow()
                    if 'created_at' in metadata:
                        try:
                            created_at = datetime.fromisoformat(metadata.pop('created_at'))
                        except:
                            pass
                    
                    documents.append(Document(
                        id=doc_id,
                        content=content,
                        metadata=metadata,
                        created_at=created_at
                    ))
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents from ChromaDB: {e}")
            return []
    
    async def count_documents(self) -> int:
        try:
            results = self.collection.get()
            return len(results['ids']) if results['ids'] else 0
            
        except Exception as e:
            logger.error(f"Error counting documents in ChromaDB: {e}")
            return 0


class VectorStore:
    """Main vector store class that chooses the appropriate backend."""
    
    def __init__(self, store_type: str = "memory", connection_params: Dict[str, Any] = None):
        self.store_type = store_type
        connection_params = connection_params or {}
        
        if store_type == "chroma" and CHROMA_AVAILABLE:
            try:
                self.backend = ChromaVectorStore(
                    host=connection_params.get("host", settings.chroma_host),
                    port=connection_params.get("port", settings.chroma_port),
                    collection_name=connection_params.get("collection_name", "mcp_documents")
                )
                logger.info("Using ChromaDB vector store")
            except Exception as e:
                logger.warning(f"Failed to initialize ChromaDB, falling back to memory: {e}")
                self.backend = MemoryVectorStore()
                self.store_type = "memory"
        else:
            self.backend = MemoryVectorStore()
            self.store_type = "memory"
            logger.info("Using memory vector store")
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        return await self.backend.add_documents(documents)
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> List[SearchResult]:
        return await self.backend.similarity_search(query, k, filter_metadata)
    
    async def similarity_search_by_vector(
        self, 
        embedding: List[float], 
        k: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> List[SearchResult]:
        return await self.backend.similarity_search_by_vector(embedding, k, filter_metadata)
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        return await self.backend.get_document(document_id)
    
    async def update_document(self, document_id: str, document: Document) -> bool:
        return await self.backend.update_document(document_id, document)
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        return await self.backend.delete_documents(document_ids)
    
    async def list_documents(self, limit: int = 100, offset: int = 0) -> List[Document]:
        return await self.backend.list_documents(limit, offset)
    
    async def count_documents(self) -> int:
        return await self.backend.count_documents()
    
    def get_store_info(self) -> Dict[str, Any]:
        """Get information about the vector store."""
        return {
            "type": self.store_type,
            "backend": self.backend.__class__.__name__,
            "chroma_available": CHROMA_AVAILABLE
        }


# Helper functions for common operations
async def create_document_from_text(
    text: str, 
    metadata: Dict[str, Any] = None,
    doc_id: str = None
) -> Document:
    """Create a document from text with optional metadata."""
    if doc_id is None:
        doc_id = str(uuid4())
    
    return Document(
        id=doc_id,
        content=text,
        metadata=metadata or {}
    )


async def batch_add_texts(
    vector_store: VectorStore,
    texts: List[str],
    metadatas: List[Dict[str, Any]] = None,
    ids: List[str] = None
) -> List[str]:
    """Batch add multiple texts to vector store."""
    documents = []
    
    for i, text in enumerate(texts):
        doc_id = ids[i] if ids else str(uuid4())
        metadata = metadatas[i] if metadatas else {}
        
        documents.append(Document(
            id=doc_id,
            content=text,
            metadata=metadata
        ))
    
    return await vector_store.add_documents(documents)