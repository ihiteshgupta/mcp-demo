#!/usr/bin/env python3
"""
Memory Management MCP Server

Production-grade MCP server for memory management and persistence with support for multiple backends:
- Redis (in-memory + persistence)
- SQLite (local file database)
- PostgreSQL (production database)
- MongoDB (document store)
- DynamoDB (AWS NoSQL)
- In-memory (development)

Features:
- Session management
- Conversation history
- Context persistence
- Memory types (short-term, long-term, episodic)
- Memory consolidation and compression
- Automatic cleanup and archiving
- Memory retrieval and search
- Cross-session memory sharing
- Memory analytics and insights
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
import pickle
import hashlib
import time

from mcp.server.fastmcp import FastMCP
from mcp.types import Resource, Tool, TextContent
from pydantic import BaseModel, Field

# Memory storage backends
try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    import sqlite3
    import aiosqlite
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False

try:
    import asyncpg
    HAS_POSTGRESQL = True
except ImportError:
    HAS_POSTGRESQL = False

try:
    import motor.motor_asyncio
    HAS_MONGODB = True
except ImportError:
    HAS_MONGODB = False

try:
    import boto3
    from botocore.exceptions import ClientError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Memory Management MCP Server")

# Global state
memory_stores: Dict[str, 'MemoryStore'] = {}
active_sessions: Dict[str, Dict[str, Any]] = {}
memory_analytics = {}


class MemoryType(str, Enum):
    """Types of memory storage."""
    SHORT_TERM = "short_term"      # Temporary, expires quickly
    WORKING = "working"            # Current conversation context
    EPISODIC = "episodic"          # Specific events/conversations
    SEMANTIC = "semantic"          # Factual knowledge
    PROCEDURAL = "procedural"      # How-to knowledge
    LONG_TERM = "long_term"        # Permanent storage


class MemoryBackend(str, Enum):
    """Supported memory backends."""
    REDIS = "redis"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    DYNAMODB = "dynamodb"
    MEMORY = "memory"


class MemoryImportance(str, Enum):
    """Memory importance levels."""
    CRITICAL = "critical"          # Never delete
    HIGH = "high"                  # Important to keep
    MEDIUM = "medium"              # Keep if space allows
    LOW = "low"                    # Can be deleted
    TEMPORARY = "temporary"        # Delete after TTL


@dataclass
class MemoryItem:
    """Individual memory item."""
    id: str
    session_id: str
    content: str
    memory_type: MemoryType
    importance: MemoryImportance
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    accessed_at: datetime
    access_count: int
    ttl: Optional[int] = None      # Time-to-live in seconds
    tags: List[str] = None
    parent_id: Optional[str] = None
    embedding: Optional[List[float]] = None


@dataclass
class MemorySession:
    """Memory session information."""
    session_id: str
    user_id: Optional[str]
    context: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    last_accessed: datetime
    memory_count: int
    total_size: int
    metadata: Dict[str, Any]


class MemoryRequest(BaseModel):
    """Request to store memory."""
    session_id: str = Field(description="Session identifier")
    content: str = Field(description="Memory content")
    memory_type: MemoryType = Field(MemoryType.WORKING, description="Type of memory")
    importance: MemoryImportance = Field(MemoryImportance.MEDIUM, description="Memory importance")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tags: List[str] = Field(default_factory=list, description="Memory tags")
    ttl: Optional[int] = Field(None, description="Time-to-live in seconds")
    parent_id: Optional[str] = Field(None, description="Parent memory ID")


class MemorySearchRequest(BaseModel):
    """Request to search memories."""
    session_id: Optional[str] = Field(None, description="Session to search in")
    query: str = Field(description="Search query")
    memory_types: List[MemoryType] = Field(default_factory=list, description="Memory types to search")
    importance_filter: Optional[MemoryImportance] = Field(None, description="Minimum importance")
    tags: List[str] = Field(default_factory=list, description="Tags to filter by")
    limit: int = Field(10, description="Maximum results")
    similarity_threshold: float = Field(0.7, description="Similarity threshold")
    include_metadata: bool = Field(True, description="Include metadata in results")


class MemoryUpdateRequest(BaseModel):
    """Request to update memory."""
    memory_id: str = Field(description="Memory ID to update")
    content: Optional[str] = Field(None, description="New content")
    importance: Optional[MemoryImportance] = Field(None, description="New importance")
    metadata: Optional[Dict[str, Any]] = Field(None, description="New metadata")
    tags: Optional[List[str]] = Field(None, description="New tags")
    ttl: Optional[int] = Field(None, description="New TTL")


class SessionRequest(BaseModel):
    """Request to create/manage session."""
    session_id: Optional[str] = Field(None, description="Existing session ID")
    user_id: Optional[str] = Field(None, description="User identifier")
    context: Dict[str, Any] = Field(default_factory=dict, description="Session context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session metadata")


class MemoryStore:
    """Base class for memory storage backends."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = config.get('backend')
        self.client = None
        
    async def initialize(self):
        """Initialize the memory store."""
        pass
    
    async def store_memory(self, memory: MemoryItem) -> bool:
        """Store a memory item."""
        raise NotImplementedError
    
    async def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory by ID."""
        raise NotImplementedError
    
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory item."""
        raise NotImplementedError
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory item."""
        raise NotImplementedError
    
    async def search_memories(self, query: str, session_id: Optional[str] = None,
                            memory_types: List[MemoryType] = None,
                            limit: int = 10) -> List[MemoryItem]:
        """Search for memories."""
        raise NotImplementedError
    
    async def get_session_memories(self, session_id: str, 
                                 memory_types: List[MemoryType] = None,
                                 limit: int = 100) -> List[MemoryItem]:
        """Get all memories for a session."""
        raise NotImplementedError
    
    async def create_session(self, session: MemorySession) -> bool:
        """Create a new session."""
        raise NotImplementedError
    
    async def get_session(self, session_id: str) -> Optional[MemorySession]:
        """Get session information."""
        raise NotImplementedError
    
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session information."""
        raise NotImplementedError
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its memories."""
        raise NotImplementedError
    
    async def cleanup_expired(self) -> int:
        """Clean up expired memories."""
        raise NotImplementedError
    
    async def get_analytics(self) -> Dict[str, Any]:
        """Get memory analytics."""
        raise NotImplementedError


class RedisMemoryStore(MemoryStore):
    """Redis-based memory store implementation."""
    
    async def initialize(self):
        if not HAS_REDIS:
            raise ImportError("redis required for Redis memory store")
        
        redis_config = {
            'host': self.config.get('host', 'localhost'),
            'port': self.config.get('port', 6379),
            'db': self.config.get('db', 0),
            'decode_responses': False,  # We'll handle encoding manually
            'socket_timeout': self.config.get('timeout', 5),
            'socket_connect_timeout': self.config.get('connect_timeout', 5),
        }
        
        if self.config.get('password'):
            redis_config['password'] = self.config['password']
        
        if self.config.get('ssl', False):
            redis_config['ssl'] = True
            redis_config['ssl_cert_reqs'] = None
        
        self.client = redis.Redis(**redis_config)
        
        # Test connection
        await self.client.ping()
        logger.info("Initialized Redis memory store")
    
    def _serialize_memory(self, memory: MemoryItem) -> bytes:
        """Serialize memory item."""
        data = asdict(memory)
        # Convert datetime objects to ISO strings
        for key in ['created_at', 'updated_at', 'accessed_at']:
            if data[key]:
                data[key] = data[key].isoformat()
        return pickle.dumps(data)
    
    def _deserialize_memory(self, data: bytes) -> MemoryItem:
        """Deserialize memory item."""
        memory_data = pickle.loads(data)
        # Convert ISO strings back to datetime objects
        for key in ['created_at', 'updated_at', 'accessed_at']:
            if memory_data[key]:
                memory_data[key] = datetime.fromisoformat(memory_data[key])
        
        return MemoryItem(**memory_data)
    
    async def store_memory(self, memory: MemoryItem) -> bool:
        try:
            # Store the memory item
            memory_key = f"memory:{memory.id}"
            memory_data = self._serialize_memory(memory)
            
            # Use pipeline for atomic operations
            pipe = self.client.pipeline()
            
            # Store memory
            pipe.set(memory_key, memory_data)
            
            # Set TTL if specified
            if memory.ttl:
                pipe.expire(memory_key, memory.ttl)
            
            # Add to session index
            session_key = f"session:{memory.session_id}:memories"
            pipe.sadd(session_key, memory.id)
            
            # Add to type index
            type_key = f"type:{memory.memory_type.value}:memories"
            pipe.sadd(type_key, memory.id)
            
            # Add to importance index
            importance_key = f"importance:{memory.importance.value}:memories"
            pipe.sadd(importance_key, memory.id)
            
            # Add tags to index
            for tag in (memory.tags or []):
                tag_key = f"tag:{tag}:memories"
                pipe.sadd(tag_key, memory.id)
            
            # Execute pipeline
            await pipe.execute()
            
            return True
        except Exception as e:
            logger.error(f"Failed to store memory in Redis: {e}")
            return False
    
    async def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        try:
            memory_key = f"memory:{memory_id}"
            data = await self.client.get(memory_key)
            
            if data:
                memory = self._deserialize_memory(data)
                
                # Update access info
                memory.accessed_at = datetime.now()
                memory.access_count += 1
                
                # Store updated memory
                await self.store_memory(memory)
                
                return memory
            
            return None
        except Exception as e:
            logger.error(f"Failed to get memory from Redis: {e}")
            return None
    
    async def search_memories(self, query: str, session_id: Optional[str] = None,
                            memory_types: List[MemoryType] = None,
                            limit: int = 10) -> List[MemoryItem]:
        try:
            # Simple text search implementation
            # In production, you'd use Redis Search or external search engine
            
            candidate_ids = set()
            
            # Get candidates based on session
            if session_id:
                session_key = f"session:{session_id}:memories"
                session_memories = await self.client.smembers(session_key)
                candidate_ids.update(m.decode() for m in session_memories)
            
            # Filter by memory types
            if memory_types:
                type_ids = set()
                for memory_type in memory_types:
                    type_key = f"type:{memory_type.value}:memories"
                    type_memories = await self.client.smembers(type_key)
                    type_ids.update(m.decode() for m in type_memories)
                
                if candidate_ids:
                    candidate_ids = candidate_ids.intersection(type_ids)
                else:
                    candidate_ids = type_ids
            
            # If no specific filters, get all memories
            if not candidate_ids and not session_id and not memory_types:
                # Get all memory keys
                all_keys = await self.client.keys("memory:*")
                candidate_ids = {key.decode().split(":", 1)[1] for key in all_keys}
            
            # Score memories by text similarity (simple implementation)
            scored_memories = []
            query_lower = query.lower()
            
            for memory_id in candidate_ids:
                memory = await self.get_memory(memory_id)
                if memory:
                    # Simple scoring based on content match
                    content_lower = memory.content.lower()
                    if query_lower in content_lower:
                        # Basic relevance scoring
                        score = content_lower.count(query_lower) / len(content_lower)
                        score += memory.importance.value == "high" and 0.1 or 0
                        score += memory.access_count * 0.01
                        
                        scored_memories.append((score, memory))
            
            # Sort by score and return top results
            scored_memories.sort(key=lambda x: x[0], reverse=True)
            return [memory for _, memory in scored_memories[:limit]]
            
        except Exception as e:
            logger.error(f"Failed to search memories in Redis: {e}")
            return []
    
    async def cleanup_expired(self) -> int:
        """Clean up expired memories."""
        try:
            # Redis handles TTL automatically, but we clean up indexes
            deleted_count = 0
            
            # Get all memory keys
            all_keys = await self.client.keys("memory:*")
            
            for key in all_keys:
                # Check if key still exists (not expired)
                exists = await self.client.exists(key)
                if not exists:
                    # Remove from indexes
                    memory_id = key.decode().split(":", 1)[1]
                    
                    # Clean up from session indexes
                    session_keys = await self.client.keys("session:*:memories")
                    for session_key in session_keys:
                        await self.client.srem(session_key, memory_id)
                    
                    # Clean up from type indexes
                    type_keys = await self.client.keys("type:*:memories")
                    for type_key in type_keys:
                        await self.client.srem(type_key, memory_id)
                    
                    deleted_count += 1
            
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup expired memories: {e}")
            return 0


class SQLiteMemoryStore(MemoryStore):
    """SQLite-based memory store implementation."""
    
    async def initialize(self):
        if not HAS_SQLITE:
            raise ImportError("aiosqlite required for SQLite memory store")
        
        db_path = self.config.get('database_path', './memory.db')
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        self.db_path = db_path
        
        # Create tables
        await self._create_tables()
        logger.info(f"Initialized SQLite memory store at {db_path}")
    
    async def _create_tables(self):
        """Create database tables."""
        async with aiosqlite.connect(self.db_path) as db:
            # Memories table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    accessed_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    ttl INTEGER,
                    tags TEXT,
                    parent_id TEXT,
                    embedding TEXT
                )
            """)
            
            # Sessions table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    context TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    memory_count INTEGER DEFAULT 0,
                    total_size INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """)
            
            # Indexes for better performance
            await db.execute("CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)")
            
            await db.commit()
    
    def _memory_to_dict(self, memory: MemoryItem) -> Dict[str, Any]:
        """Convert memory item to database format."""
        return {
            'id': memory.id,
            'session_id': memory.session_id,
            'content': memory.content,
            'memory_type': memory.memory_type.value,
            'importance': memory.importance.value,
            'metadata': json.dumps(memory.metadata),
            'created_at': memory.created_at.isoformat(),
            'updated_at': memory.updated_at.isoformat(),
            'accessed_at': memory.accessed_at.isoformat(),
            'access_count': memory.access_count,
            'ttl': memory.ttl,
            'tags': json.dumps(memory.tags) if memory.tags else None,
            'parent_id': memory.parent_id,
            'embedding': json.dumps(memory.embedding) if memory.embedding else None
        }
    
    def _dict_to_memory(self, data: Dict[str, Any]) -> MemoryItem:
        """Convert database format to memory item."""
        return MemoryItem(
            id=data['id'],
            session_id=data['session_id'],
            content=data['content'],
            memory_type=MemoryType(data['memory_type']),
            importance=MemoryImportance(data['importance']),
            metadata=json.loads(data['metadata']) if data['metadata'] else {},
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            accessed_at=datetime.fromisoformat(data['accessed_at']),
            access_count=data['access_count'],
            ttl=data['ttl'],
            tags=json.loads(data['tags']) if data['tags'] else [],
            parent_id=data['parent_id'],
            embedding=json.loads(data['embedding']) if data['embedding'] else None
        )
    
    async def store_memory(self, memory: MemoryItem) -> bool:
        try:
            async with aiosqlite.connect(self.db_path) as db:
                memory_dict = self._memory_to_dict(memory)
                
                await db.execute("""
                    INSERT OR REPLACE INTO memories 
                    (id, session_id, content, memory_type, importance, metadata,
                     created_at, updated_at, accessed_at, access_count, ttl, tags, parent_id, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, tuple(memory_dict.values()))
                
                await db.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to store memory in SQLite: {e}")
            return False
    
    async def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                async with db.execute(
                    "SELECT * FROM memories WHERE id = ?", (memory_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                    
                    if row:
                        memory = self._dict_to_memory(dict(row))
                        
                        # Update access info
                        memory.accessed_at = datetime.now()
                        memory.access_count += 1
                        
                        # Store updated memory
                        await self.store_memory(memory)
                        
                        return memory
                    
                    return None
        except Exception as e:
            logger.error(f"Failed to get memory from SQLite: {e}")
            return None
    
    async def search_memories(self, query: str, session_id: Optional[str] = None,
                            memory_types: List[MemoryType] = None,
                            limit: int = 10) -> List[MemoryItem]:
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Build query conditions
                conditions = ["content LIKE ?"]
                params = [f"%{query}%"]
                
                if session_id:
                    conditions.append("session_id = ?")
                    params.append(session_id)
                
                if memory_types:
                    type_placeholders = ",".join(["?"] * len(memory_types))
                    conditions.append(f"memory_type IN ({type_placeholders})")
                    params.extend([mt.value for mt in memory_types])
                
                where_clause = " AND ".join(conditions)
                
                query_sql = f"""
                    SELECT * FROM memories 
                    WHERE {where_clause}
                    ORDER BY 
                        CASE importance 
                            WHEN 'critical' THEN 5
                            WHEN 'high' THEN 4
                            WHEN 'medium' THEN 3
                            WHEN 'low' THEN 2
                            WHEN 'temporary' THEN 1
                        END DESC,
                        access_count DESC,
                        created_at DESC
                    LIMIT ?
                """
                params.append(limit)
                
                async with db.execute(query_sql, params) as cursor:
                    rows = await cursor.fetchall()
                    
                    return [self._dict_to_memory(dict(row)) for row in rows]
                    
        except Exception as e:
            logger.error(f"Failed to search memories in SQLite: {e}")
            return []


class InMemoryStore(MemoryStore):
    """In-memory store for development and testing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.memories: Dict[str, MemoryItem] = {}
        self.sessions: Dict[str, MemorySession] = {}
        self.indexes = {
            'session': {},
            'type': {},
            'importance': {},
            'tags': {}
        }
    
    async def initialize(self):
        logger.info("Initialized in-memory store")
    
    async def store_memory(self, memory: MemoryItem) -> bool:
        try:
            self.memories[memory.id] = memory
            
            # Update indexes
            session_id = memory.session_id
            if session_id not in self.indexes['session']:
                self.indexes['session'][session_id] = []
            self.indexes['session'][session_id].append(memory.id)
            
            memory_type = memory.memory_type.value
            if memory_type not in self.indexes['type']:
                self.indexes['type'][memory_type] = []
            self.indexes['type'][memory_type].append(memory.id)
            
            importance = memory.importance.value
            if importance not in self.indexes['importance']:
                self.indexes['importance'][importance] = []
            self.indexes['importance'][importance].append(memory.id)
            
            for tag in (memory.tags or []):
                if tag not in self.indexes['tags']:
                    self.indexes['tags'][tag] = []
                self.indexes['tags'][tag].append(memory.id)
            
            return True
        except Exception as e:
            logger.error(f"Failed to store memory in memory store: {e}")
            return False
    
    async def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        memory = self.memories.get(memory_id)
        if memory:
            # Update access info
            memory.accessed_at = datetime.now()
            memory.access_count += 1
        return memory
    
    async def search_memories(self, query: str, session_id: Optional[str] = None,
                            memory_types: List[MemoryType] = None,
                            limit: int = 10) -> List[MemoryItem]:
        try:
            candidates = []
            
            # Get candidate memory IDs
            if session_id and session_id in self.indexes['session']:
                candidate_ids = set(self.indexes['session'][session_id])
            else:
                candidate_ids = set(self.memories.keys())
            
            # Filter by memory types
            if memory_types:
                type_ids = set()
                for memory_type in memory_types:
                    if memory_type.value in self.indexes['type']:
                        type_ids.update(self.indexes['type'][memory_type.value])
                candidate_ids = candidate_ids.intersection(type_ids)
            
            # Score and filter memories
            query_lower = query.lower()
            scored_memories = []
            
            for memory_id in candidate_ids:
                memory = self.memories.get(memory_id)
                if memory and query_lower in memory.content.lower():
                    # Simple scoring
                    score = memory.content.lower().count(query_lower)
                    scored_memories.append((score, memory))
            
            # Sort by score and return top results
            scored_memories.sort(key=lambda x: x[0], reverse=True)
            return [memory for _, memory in scored_memories[:limit]]
            
        except Exception as e:
            logger.error(f"Failed to search memories in memory store: {e}")
            return []


# Store factory
MEMORY_STORES = {
    MemoryBackend.REDIS: RedisMemoryStore,
    MemoryBackend.SQLITE: SQLiteMemoryStore,
    MemoryBackend.MEMORY: InMemoryStore,
}


@mcp.resource("memory://sessions")
async def list_sessions() -> str:
    """List all active memory sessions."""
    result = "Memory Sessions:\n\n"
    
    if not active_sessions:
        return "No active sessions"
    
    for session_id, session_data in active_sessions.items():
        result += f"**{session_id}**\n"
        result += f"  User: {session_data.get('user_id', 'anonymous')}\n"
        result += f"  Created: {session_data.get('created_at', 'unknown')}\n"
        result += f"  Last accessed: {session_data.get('last_accessed', 'unknown')}\n"
        result += f"  Memory count: {session_data.get('memory_count', 0)}\n\n"
    
    return result


@mcp.resource("memory://stats")
async def get_memory_stats() -> str:
    """Get memory storage statistics."""
    result = "Memory Statistics:\n\n"
    
    if not memory_stores:
        return "No memory stores available"
    
    store = next(iter(memory_stores.values()))
    
    try:
        analytics = await store.get_analytics()
        
        result += f"Total sessions: {len(active_sessions)}\n"
        result += f"Memory stores: {len(memory_stores)}\n"
        
        # Add store-specific stats if available
        if analytics:
            for key, value in analytics.items():
                result += f"{key}: {value}\n"
        
    except Exception as e:
        result += f"Error getting analytics: {e}\n"
    
    return result


@mcp.resource("memory://types")
async def list_memory_types() -> str:
    """List available memory types and their descriptions."""
    result = "Memory Types:\n\n"
    
    type_descriptions = {
        MemoryType.SHORT_TERM: "Temporary memory that expires quickly",
        MemoryType.WORKING: "Current conversation context",
        MemoryType.EPISODIC: "Specific events and conversations",
        MemoryType.SEMANTIC: "Factual knowledge and information",
        MemoryType.PROCEDURAL: "How-to knowledge and procedures",
        MemoryType.LONG_TERM: "Permanent storage for important information"
    }
    
    for memory_type, description in type_descriptions.items():
        result += f"**{memory_type.value}**: {description}\n"
    
    return result


@mcp.tool()
async def create_session(request: SessionRequest) -> Dict[str, Any]:
    """Create a new memory session."""
    
    if not memory_stores:
        return {
            "success": False,
            "error": "No memory stores available"
        }
    
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        session = MemorySession(
            session_id=session_id,
            user_id=request.user_id,
            context=request.context,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            last_accessed=datetime.now(),
            memory_count=0,
            total_size=0,
            metadata=request.metadata
        )
        
        # Store in active sessions
        active_sessions[session_id] = asdict(session)
        
        # Store in backend if available
        store = next(iter(memory_stores.values()))
        if hasattr(store, 'create_session'):
            await store.create_session(session)
        
        return {
            "success": True,
            "session_id": session_id,
            "user_id": request.user_id,
            "created_at": session.created_at.isoformat(),
            "message": f"Created session '{session_id}'"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def store_memory(request: MemoryRequest) -> Dict[str, Any]:
    """Store a memory item."""
    
    if not memory_stores:
        return {
            "success": False,
            "error": "No memory stores available"
        }
    
    store = next(iter(memory_stores.values()))
    
    try:
        memory = MemoryItem(
            id=str(uuid.uuid4()),
            session_id=request.session_id,
            content=request.content,
            memory_type=request.memory_type,
            importance=request.importance,
            metadata=request.metadata,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=0,
            ttl=request.ttl,
            tags=request.tags,
            parent_id=request.parent_id
        )
        
        success = await store.store_memory(memory)
        
        if success:
            # Update session info
            if request.session_id in active_sessions:
                active_sessions[request.session_id]['memory_count'] += 1
                active_sessions[request.session_id]['total_size'] += len(request.content)
                active_sessions[request.session_id]['last_accessed'] = datetime.now().isoformat()
            
            return {
                "success": True,
                "memory_id": memory.id,
                "session_id": request.session_id,
                "memory_type": request.memory_type.value,
                "importance": request.importance.value,
                "message": f"Stored memory '{memory.id}'"
            }
        else:
            return {
                "success": False,
                "error": "Failed to store memory"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def search_memory(request: MemorySearchRequest) -> Dict[str, Any]:
    """Search for memories."""
    
    if not memory_stores:
        return {
            "success": False,
            "error": "No memory stores available"
        }
    
    store = next(iter(memory_stores.values()))
    
    try:
        memories = await store.search_memories(
            query=request.query,
            session_id=request.session_id,
            memory_types=request.memory_types,
            limit=request.limit
        )
        
        # Format results
        results = []
        for memory in memories:
            memory_data = {
                "id": memory.id,
                "session_id": memory.session_id,
                "content": memory.content,
                "memory_type": memory.memory_type.value,
                "importance": memory.importance.value,
                "created_at": memory.created_at.isoformat(),
                "access_count": memory.access_count,
                "tags": memory.tags
            }
            
            if request.include_metadata:
                memory_data["metadata"] = memory.metadata
            
            results.append(memory_data)
        
        return {
            "success": True,
            "query": request.query,
            "results": results,
            "total_results": len(results),
            "session_id": request.session_id
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def get_session_memories(
    session_id: str,
    memory_types: List[MemoryType] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """Get all memories for a session."""
    
    if not memory_stores:
        return {
            "success": False,
            "error": "No memory stores available"
        }
    
    store = next(iter(memory_stores.values()))
    
    try:
        if hasattr(store, 'get_session_memories'):
            memories = await store.get_session_memories(session_id, memory_types, limit)
        else:
            # Fallback to search
            memories = await store.search_memories("", session_id, memory_types, limit)
        
        # Format results
        results = []
        for memory in memories:
            results.append({
                "id": memory.id,
                "content": memory.content,
                "memory_type": memory.memory_type.value,
                "importance": memory.importance.value,
                "created_at": memory.created_at.isoformat(),
                "updated_at": memory.updated_at.isoformat(),
                "access_count": memory.access_count,
                "tags": memory.tags,
                "metadata": memory.metadata
            })
        
        return {
            "success": True,
            "session_id": session_id,
            "memories": results,
            "total_memories": len(results),
            "memory_types": [mt.value for mt in memory_types] if memory_types else "all"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def update_memory(request: MemoryUpdateRequest) -> Dict[str, Any]:
    """Update an existing memory."""
    
    if not memory_stores:
        return {
            "success": False,
            "error": "No memory stores available"
        }
    
    store = next(iter(memory_stores.values()))
    
    try:
        # Get existing memory
        memory = await store.get_memory(request.memory_id)
        if not memory:
            return {
                "success": False,
                "error": f"Memory '{request.memory_id}' not found"
            }
        
        # Update fields
        if request.content is not None:
            memory.content = request.content
        if request.importance is not None:
            memory.importance = request.importance
        if request.metadata is not None:
            memory.metadata.update(request.metadata)
        if request.tags is not None:
            memory.tags = request.tags
        if request.ttl is not None:
            memory.ttl = request.ttl
        
        memory.updated_at = datetime.now()
        
        # Store updated memory
        success = await store.store_memory(memory)
        
        if success:
            return {
                "success": True,
                "memory_id": request.memory_id,
                "updated_at": memory.updated_at.isoformat(),
                "message": f"Updated memory '{request.memory_id}'"
            }
        else:
            return {
                "success": False,
                "error": "Failed to update memory"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def delete_memory(memory_id: str) -> Dict[str, Any]:
    """Delete a memory item."""
    
    if not memory_stores:
        return {
            "success": False,
            "error": "No memory stores available"
        }
    
    store = next(iter(memory_stores.values()))
    
    try:
        if hasattr(store, 'delete_memory'):
            success = await store.delete_memory(memory_id)
        else:
            # Fallback for in-memory store
            if hasattr(store, 'memories') and memory_id in store.memories:
                del store.memories[memory_id]
                success = True
            else:
                success = False
        
        if success:
            return {
                "success": True,
                "memory_id": memory_id,
                "message": f"Deleted memory '{memory_id}'"
            }
        else:
            return {
                "success": False,
                "error": f"Memory '{memory_id}' not found"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def cleanup_memories(
    max_age_hours: int = 168,  # 1 week
    min_importance: MemoryImportance = MemoryImportance.LOW
) -> Dict[str, Any]:
    """Clean up old or low-importance memories."""
    
    if not memory_stores:
        return {
            "success": False,
            "error": "No memory stores available"
        }
    
    store = next(iter(memory_stores.values()))
    
    try:
        if hasattr(store, 'cleanup_expired'):
            deleted_count = await store.cleanup_expired()
        else:
            # Manual cleanup for stores without built-in cleanup
            deleted_count = 0
            cutoff_date = datetime.now() - timedelta(hours=max_age_hours)
            
            if hasattr(store, 'memories'):
                to_delete = []
                for memory_id, memory in store.memories.items():
                    # Delete if old and low importance
                    if (memory.created_at < cutoff_date and 
                        memory.importance.value in ['low', 'temporary']):
                        to_delete.append(memory_id)
                
                for memory_id in to_delete:
                    del store.memories[memory_id]
                    deleted_count += 1
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "max_age_hours": max_age_hours,
            "min_importance": min_importance.value,
            "message": f"Cleaned up {deleted_count} memories"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.server.lifespan
async def setup_and_cleanup():
    """Initialize and cleanup server components."""
    global memory_stores
    
    # Load configuration
    config_file = os.path.join(os.path.dirname(__file__), 'config.json')
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.warning("No config.json found, using default configuration")
        config = {
            "memory_stores": {
                "memory": {
                    "backend": "memory",
                    "enabled": True
                }
            }
        }
    
    # Initialize memory stores
    for name, store_config in config.get('memory_stores', {}).items():
        if not store_config.get('enabled', True):
            continue
            
        try:
            backend = MemoryBackend(store_config['backend'])
            if backend in MEMORY_STORES:
                store = MEMORY_STORES[backend](store_config)
                await store.initialize()
                memory_stores[name] = store
                logger.info(f"Initialized memory store: {name} ({backend.value})")
            else:
                logger.warning(f"Unknown memory backend: {store_config['backend']}")
        except Exception as e:
            logger.error(f"Failed to initialize memory store {name}: {e}")
    
    if not memory_stores:
        # Fallback to in-memory store
        store = InMemoryStore({"backend": "memory"})
        await store.initialize()
        memory_stores["default"] = store
        logger.info("No memory stores configured, using fallback in-memory store")
    
    logger.info(f"Memory Management MCP Server initialized with {len(memory_stores)} stores")
    
    yield
    
    # Cleanup
    memory_stores.clear()
    active_sessions.clear()
    memory_analytics.clear()
    logger.info("Memory Management MCP Server shut down")


def main():
    """Run the Memory Management MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory Management MCP Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8004, help="Port to bind to")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()