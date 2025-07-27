import json
import logging
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
from uuid import uuid4

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Redis not available, using memory storage only")

from config.settings import settings
from utils.exceptions import SessionStorageError

logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    expires_at: Optional[datetime] = None
    data: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.data is None:
            self.data = {}
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for storage."""
        result = asdict(self)
        # Convert datetime objects to ISO strings
        if self.created_at:
            result['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            result['updated_at'] = self.updated_at.isoformat()
        if self.expires_at:
            result['expires_at'] = self.expires_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionData':
        """Create session from dictionary."""
        # Convert ISO strings back to datetime objects
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if 'expires_at' in data and isinstance(data['expires_at'], str):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        
        return cls(**data)
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def update_data(self, new_data: Dict[str, Any]) -> None:
        """Update session data and timestamp."""
        self.data.update(new_data)
        self.updated_at = datetime.utcnow()


class BaseSessionStorage(ABC):
    """Abstract base class for session storage."""
    
    @abstractmethod
    async def create_session(
        self, 
        session_id: str = None, 
        user_id: str = None,
        ttl_seconds: int = 3600
    ) -> SessionData:
        """Create a new session."""
        pass
    
    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session by ID."""
        pass
    
    @abstractmethod
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> Optional[SessionData]:
        """Update session data."""
        pass
    
    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        pass
    
    @abstractmethod
    async def list_sessions(self, user_id: str = None) -> List[SessionData]:
        """List all sessions, optionally filtered by user_id."""
        pass
    
    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Clean up expired sessions and return count of cleaned sessions."""
        pass


class MemorySessionStorage(BaseSessionStorage):
    """In-memory session storage implementation."""
    
    def __init__(self):
        self.sessions: Dict[str, SessionData] = {}
        self._cleanup_task = None
        self._start_cleanup_task()
        logger.info("Initialized memory session storage")
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        """Periodically clean up expired sessions."""
        while True:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes
                await self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    async def create_session(
        self, 
        session_id: str = None, 
        user_id: str = None,
        ttl_seconds: int = 3600
    ) -> SessionData:
        if session_id is None:
            session_id = str(uuid4())
        
        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds) if ttl_seconds > 0 else None
        
        session = SessionData(
            session_id=session_id,
            user_id=user_id,
            expires_at=expires_at
        )
        
        self.sessions[session_id] = session
        logger.debug(f"Created memory session: {session_id}")
        return session
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        session = self.sessions.get(session_id)
        if session and session.is_expired():
            await self.delete_session(session_id)
            return None
        return session
    
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> Optional[SessionData]:
        session = await self.get_session(session_id)
        if session:
            session.update_data(data)
            logger.debug(f"Updated memory session: {session_id}")
        return session
    
    async def delete_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.debug(f"Deleted memory session: {session_id}")
            return True
        return False
    
    async def list_sessions(self, user_id: str = None) -> List[SessionData]:
        sessions = []
        for session in self.sessions.values():
            if not session.is_expired():
                if user_id is None or session.user_id == user_id:
                    sessions.append(session)
        return sessions
    
    async def cleanup_expired(self) -> int:
        expired_sessions = []
        for session_id, session in self.sessions.items():
            if session.is_expired():
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        if expired_sessions:
            logger.debug(f"Cleaned up {len(expired_sessions)} expired memory sessions")
        
        return len(expired_sessions)


class RedisSessionStorage(BaseSessionStorage):
    """Redis session storage implementation."""
    
    def __init__(self, host: str, port: int, db: int = 0, password: str = None):
        if not REDIS_AVAILABLE:
            raise SessionStorageError("Redis is not available. Install redis package.")
        
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True
        )
        self.key_prefix = "mcp_session:"
        logger.info(f"Initialized Redis session storage: {host}:{port}")
    
    def _make_key(self, session_id: str) -> str:
        """Create Redis key for session."""
        return f"{self.key_prefix}{session_id}"
    
    async def create_session(
        self, 
        session_id: str = None, 
        user_id: str = None,
        ttl_seconds: int = 3600
    ) -> SessionData:
        if session_id is None:
            session_id = str(uuid4())
        
        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds) if ttl_seconds > 0 else None
        
        session = SessionData(
            session_id=session_id,
            user_id=user_id,
            expires_at=expires_at
        )
        
        try:
            key = self._make_key(session_id)
            value = json.dumps(session.to_dict())
            
            if ttl_seconds > 0:
                await self.redis_client.setex(key, ttl_seconds, value)
            else:
                await self.redis_client.set(key, value)
            
            logger.debug(f"Created Redis session: {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Error creating Redis session: {e}")
            raise SessionStorageError(f"Failed to create session: {e}")
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        try:
            key = self._make_key(session_id)
            value = await self.redis_client.get(key)
            
            if value is None:
                return None
            
            session_data = json.loads(value)
            session = SessionData.from_dict(session_data)
            
            if session.is_expired():
                await self.delete_session(session_id)
                return None
            
            return session
            
        except Exception as e:
            logger.error(f"Error getting Redis session {session_id}: {e}")
            return None
    
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> Optional[SessionData]:
        session = await self.get_session(session_id)
        if session:
            session.update_data(data)
            
            try:
                key = self._make_key(session_id)
                value = json.dumps(session.to_dict())
                
                # Keep the same TTL if it exists
                ttl = await self.redis_client.ttl(key)
                if ttl > 0:
                    await self.redis_client.setex(key, ttl, value)
                else:
                    await self.redis_client.set(key, value)
                
                logger.debug(f"Updated Redis session: {session_id}")
                
            except Exception as e:
                logger.error(f"Error updating Redis session {session_id}: {e}")
                raise SessionStorageError(f"Failed to update session: {e}")
        
        return session
    
    async def delete_session(self, session_id: str) -> bool:
        try:
            key = self._make_key(session_id)
            result = await self.redis_client.delete(key)
            logger.debug(f"Deleted Redis session: {session_id}")
            return result > 0
            
        except Exception as e:
            logger.error(f"Error deleting Redis session {session_id}: {e}")
            return False
    
    async def list_sessions(self, user_id: str = None) -> List[SessionData]:
        try:
            pattern = f"{self.key_prefix}*"
            keys = await self.redis_client.keys(pattern)
            
            sessions = []
            for key in keys:
                value = await self.redis_client.get(key)
                if value:
                    try:
                        session_data = json.loads(value)
                        session = SessionData.from_dict(session_data)
                        
                        if not session.is_expired():
                            if user_id is None or session.user_id == user_id:
                                sessions.append(session)
                    except Exception as e:
                        logger.warning(f"Error parsing session data for key {key}: {e}")
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error listing Redis sessions: {e}")
            return []
    
    async def cleanup_expired(self) -> int:
        # Redis automatically handles TTL expiration, so this is mainly for manual cleanup
        try:
            pattern = f"{self.key_prefix}*"
            keys = await self.redis_client.keys(pattern)
            
            expired_count = 0
            for key in keys:
                value = await self.redis_client.get(key)
                if value:
                    try:
                        session_data = json.loads(value)
                        session = SessionData.from_dict(session_data)
                        
                        if session.is_expired():
                            await self.redis_client.delete(key)
                            expired_count += 1
                    except Exception as e:
                        logger.warning(f"Error checking expiration for key {key}: {e}")
            
            if expired_count > 0:
                logger.debug(f"Cleaned up {expired_count} expired Redis sessions")
            
            return expired_count
            
        except Exception as e:
            logger.error(f"Error cleaning up Redis sessions: {e}")
            return 0


class SessionStorage:
    """Main session storage class that chooses the appropriate backend."""
    
    def __init__(self, storage_type: str = "memory", connection_params: Dict[str, Any] = None):
        self.storage_type = storage_type
        connection_params = connection_params or {}
        
        if storage_type == "redis" and REDIS_AVAILABLE:
            try:
                self.backend = RedisSessionStorage(
                    host=connection_params.get("host", settings.redis_host),
                    port=connection_params.get("port", settings.redis_port),
                    db=connection_params.get("db", settings.redis_db),
                    password=connection_params.get("password", settings.redis_password)
                )
                logger.info("Using Redis session storage")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis storage, falling back to memory: {e}")
                self.backend = MemorySessionStorage()
                self.storage_type = "memory"
        else:
            self.backend = MemorySessionStorage()
            self.storage_type = "memory"
            logger.info("Using memory session storage")
    
    async def create_session(
        self, 
        session_id: str = None, 
        user_id: str = None,
        ttl_seconds: int = 3600
    ) -> SessionData:
        return await self.backend.create_session(session_id, user_id, ttl_seconds)
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        return await self.backend.get_session(session_id)
    
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> Optional[SessionData]:
        return await self.backend.update_session(session_id, data)
    
    async def delete_session(self, session_id: str) -> bool:
        return await self.backend.delete_session(session_id)
    
    async def list_sessions(self, user_id: str = None) -> List[SessionData]:
        return await self.backend.list_sessions(user_id)
    
    async def cleanup_expired(self) -> int:
        return await self.backend.cleanup_expired()
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the session storage."""
        return {
            "type": self.storage_type,
            "backend": self.backend.__class__.__name__,
            "redis_available": REDIS_AVAILABLE
        }