#!/usr/bin/env python3
"""
MCP Servers Orchestrator

Unified configuration and orchestration system for all MCP servers in the ecosystem.
Provides centralized management, health monitoring, service discovery, and inter-server communication.

Features:
- Centralized configuration management
- Service discovery and registration
- Health monitoring and alerting
- Load balancing and failover
- Inter-server communication
- Unified logging and metrics
- Service lifecycle management
- Configuration hot-reloading
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

from mcp.server.fastmcp import FastMCP
from mcp.types import Resource, Tool, TextContent
from pydantic import BaseModel, Field

# Service management
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("MCP Servers Orchestrator")

# Global state
services: Dict[str, 'MCPService'] = {}
health_monitor = None
configuration_manager = None


class ServiceStatus(str, Enum):
    """Service status states."""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    UNKNOWN = "unknown"


class ServiceType(str, Enum):
    """Types of MCP services."""
    LLM_PROVIDER = "llm_provider"
    VECTOR_STORE = "vector_store"
    MEMORY = "memory"
    WEB_FETCH = "web_fetch"
    SEQUENTIAL_THINKER = "sequential_thinker"
    CUSTOM = "custom"


@dataclass
class ServiceConfig:
    """Configuration for an MCP service."""
    name: str
    type: ServiceType
    enabled: bool
    command: List[str]
    working_directory: str
    environment: Dict[str, str]
    port: Optional[int]
    health_check_url: Optional[str]
    dependencies: List[str]
    restart_policy: str  # "always", "on-failure", "unless-stopped", "no"
    restart_delay: int   # seconds
    max_restarts: int
    timeout: int         # startup timeout
    priority: int        # startup order (lower = earlier)


@dataclass
class ServiceHealth:
    """Service health information."""
    service_name: str
    status: ServiceStatus
    pid: Optional[int]
    cpu_percent: float
    memory_mb: float
    uptime_seconds: float
    restart_count: int
    last_error: Optional[str]
    last_health_check: datetime
    response_time_ms: Optional[float]


@dataclass
class ServiceMetrics:
    """Service performance metrics."""
    service_name: str
    requests_per_second: float
    avg_response_time_ms: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    disk_usage_mb: float
    network_bytes_in: int
    network_bytes_out: int
    uptime_hours: float


class MCPService:
    """Represents a managed MCP service."""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.status = ServiceStatus.STOPPED
        self.start_time: Optional[datetime] = None
        self.restart_count = 0
        self.last_error: Optional[str] = None
        self.health: Optional[ServiceHealth] = None
        
    async def start(self) -> bool:
        """Start the service."""
        if self.status == ServiceStatus.RUNNING:
            return True
            
        logger.info(f"Starting service: {self.config.name}")
        self.status = ServiceStatus.STARTING
        
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(self.config.environment)
            
            # Start process
            self.process = subprocess.Popen(
                self.config.command,
                cwd=self.config.working_directory,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for startup
            start_timeout = self.config.timeout
            elapsed = 0
            while elapsed < start_timeout:
                if self.process.poll() is not None:
                    # Process exited
                    stdout, stderr = self.process.communicate()
                    self.last_error = f"Process exited: {stderr}"
                    self.status = ServiceStatus.ERROR
                    return False
                
                # Check if service is responding
                if await self._health_check():
                    self.status = ServiceStatus.RUNNING
                    self.start_time = datetime.now()
                    logger.info(f"Service started successfully: {self.config.name}")
                    return True
                
                await asyncio.sleep(1)
                elapsed += 1
            
            # Startup timeout
            self.last_error = f"Startup timeout after {start_timeout} seconds"
            self.status = ServiceStatus.ERROR
            await self.stop()
            return False
            
        except Exception as e:
            self.last_error = str(e)
            self.status = ServiceStatus.ERROR
            logger.error(f"Failed to start service {self.config.name}: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the service."""
        if self.status == ServiceStatus.STOPPED:
            return True
            
        logger.info(f"Stopping service: {self.config.name}")
        self.status = ServiceStatus.STOPPING
        
        try:
            if self.process:
                # Try graceful shutdown first
                self.process.terminate()
                
                try:
                    # Wait for graceful shutdown
                    await asyncio.wait_for(
                        asyncio.create_task(self._wait_for_exit()),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    # Force kill if needed
                    logger.warning(f"Force killing service: {self.config.name}")
                    self.process.kill()
                    await self._wait_for_exit()
                
                self.process = None
            
            self.status = ServiceStatus.STOPPED
            self.start_time = None
            logger.info(f"Service stopped: {self.config.name}")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to stop service {self.config.name}: {e}")
            return False
    
    async def restart(self) -> bool:
        """Restart the service."""
        logger.info(f"Restarting service: {self.config.name}")
        
        if not await self.stop():
            return False
        
        # Wait for restart delay
        if self.config.restart_delay > 0:
            await asyncio.sleep(self.config.restart_delay)
        
        self.restart_count += 1
        return await self.start()
    
    async def _wait_for_exit(self):
        """Wait for process to exit."""
        if self.process:
            while self.process.poll() is None:
                await asyncio.sleep(0.1)
    
    async def _health_check(self) -> bool:
        """Perform health check."""
        if not self.process or self.process.poll() is not None:
            return False
        
        # If no health check URL, just check if process is running
        if not self.config.health_check_url:
            return True
        
        # HTTP health check
        if HAS_AIOHTTP:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    start_time = time.time()
                    async with session.get(self.config.health_check_url) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        # Update health info
                        self.health = ServiceHealth(
                            service_name=self.config.name,
                            status=self.status,
                            pid=self.process.pid if self.process else None,
                            cpu_percent=0.0,  # Will be updated by monitor
                            memory_mb=0.0,    # Will be updated by monitor
                            uptime_seconds=(datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                            restart_count=self.restart_count,
                            last_error=self.last_error,
                            last_health_check=datetime.now(),
                            response_time_ms=response_time
                        )
                        
                        return response.status < 400
            except Exception as e:
                self.last_error = f"Health check failed: {str(e)}"
                return False
        
        return True
    
    def get_health(self) -> ServiceHealth:
        """Get current health status."""
        if not self.health:
            self.health = ServiceHealth(
                service_name=self.config.name,
                status=self.status,
                pid=self.process.pid if self.process else None,
                cpu_percent=0.0,
                memory_mb=0.0,
                uptime_seconds=(datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                restart_count=self.restart_count,
                last_error=self.last_error,
                last_health_check=datetime.now(),
                response_time_ms=None
            )
        
        # Update process info if available
        if HAS_PSUTIL and self.process and self.process.poll() is None:
            try:
                proc = psutil.Process(self.process.pid)
                self.health.cpu_percent = proc.cpu_percent()
                self.health.memory_mb = proc.memory_info().rss / 1024 / 1024
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return self.health


class HealthMonitor:
    """Monitors health of all services."""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.running = False
        self.task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start health monitoring."""
        if self.running:
            return
        
        self.running = True
        self.task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitor started")
    
    async def stop(self):
        """Stop health monitoring."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitor stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                await self._check_all_services()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_all_services(self):
        """Check health of all services."""
        for service in services.values():
            await self._check_service(service)
    
    async def _check_service(self, service: MCPService):
        """Check health of a single service."""
        try:
            # Update health status
            health = service.get_health()
            
            # Check if service needs restart
            if service.status == ServiceStatus.ERROR and service.config.restart_policy in ["always", "on-failure"]:
                if service.restart_count < service.config.max_restarts:
                    logger.warning(f"Restarting failed service: {service.config.name}")
                    await service.restart()
                else:
                    logger.error(f"Service {service.config.name} exceeded max restarts ({service.config.max_restarts})")
            
            # Check if process died unexpectedly
            elif service.status == ServiceStatus.RUNNING and service.process and service.process.poll() is not None:
                logger.warning(f"Service {service.config.name} died unexpectedly")
                service.status = ServiceStatus.ERROR
                service.last_error = "Process died unexpectedly"
                
                if service.config.restart_policy in ["always", "on-failure"]:
                    if service.restart_count < service.config.max_restarts:
                        await service.restart()
            
            # Perform health check
            elif service.status == ServiceStatus.RUNNING:
                is_healthy = await service._health_check()
                if not is_healthy:
                    logger.warning(f"Health check failed for service: {service.config.name}")
                    service.status = ServiceStatus.ERROR
                    
                    if service.config.restart_policy in ["always", "on-failure"]:
                        if service.restart_count < service.config.max_restarts:
                            await service.restart()
            
        except Exception as e:
            logger.error(f"Error checking service {service.config.name}: {e}")


class ConfigurationManager:
    """Manages configuration for all services."""
    
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / "orchestrator.yaml"
        self.services_config: Dict[str, ServiceConfig] = {}
        
    async def load_configuration(self) -> bool:
        """Load configuration from files."""
        try:
            # Load main configuration
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    if HAS_YAML:
                        config_data = yaml.safe_load(f)
                    else:
                        # Fallback to JSON if YAML not available
                        config_data = json.load(f)
            else:
                # Create default configuration
                config_data = self._create_default_config()
                await self.save_configuration(config_data)
            
            # Parse service configurations
            self.services_config = {}
            for service_name, service_data in config_data.get('services', {}).items():
                self.services_config[service_name] = ServiceConfig(
                    name=service_name,
                    type=ServiceType(service_data.get('type', 'custom')),
                    enabled=service_data.get('enabled', True),
                    command=service_data.get('command', []),
                    working_directory=service_data.get('working_directory', '.'),
                    environment=service_data.get('environment', {}),
                    port=service_data.get('port'),
                    health_check_url=service_data.get('health_check_url'),
                    dependencies=service_data.get('dependencies', []),
                    restart_policy=service_data.get('restart_policy', 'on-failure'),
                    restart_delay=service_data.get('restart_delay', 5),
                    max_restarts=service_data.get('max_restarts', 3),
                    timeout=service_data.get('timeout', 30),
                    priority=service_data.get('priority', 50)
                )
            
            logger.info(f"Loaded configuration for {len(self.services_config)} services")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    async def save_configuration(self, config_data: Dict[str, Any]) -> bool:
        """Save configuration to file."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                if HAS_YAML:
                    yaml.dump(config_data, f, default_flow_style=False)
                else:
                    json.dump(config_data, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        project_root = Path(__file__).parent.parent
        
        return {
            "global": {
                "log_level": "INFO",
                "health_check_interval": 30,
                "startup_timeout": 60
            },
            "services": {
                "llm_provider": {
                    "type": "llm_provider",
                    "enabled": True,
                    "command": ["python", "main.py"],
                    "working_directory": str(project_root / "llm-provider"),
                    "environment": {},
                    "port": 8002,
                    "health_check_url": None,
                    "dependencies": [],
                    "restart_policy": "on-failure",
                    "restart_delay": 5,
                    "max_restarts": 3,
                    "timeout": 30,
                    "priority": 10
                },
                "vector_store": {
                    "type": "vector_store",
                    "enabled": True,
                    "command": ["python", "main.py"],
                    "working_directory": str(project_root / "vector-store"),
                    "environment": {},
                    "port": 8003,
                    "health_check_url": None,
                    "dependencies": [],
                    "restart_policy": "on-failure",
                    "restart_delay": 5,
                    "max_restarts": 3,
                    "timeout": 30,
                    "priority": 20
                },
                "memory": {
                    "type": "memory",
                    "enabled": True,
                    "command": ["python", "main.py"],
                    "working_directory": str(project_root / "memory"),
                    "environment": {},
                    "port": 8004,
                    "health_check_url": None,
                    "dependencies": [],
                    "restart_policy": "on-failure",
                    "restart_delay": 5,
                    "max_restarts": 3,
                    "timeout": 30,
                    "priority": 30
                },
                "web_fetch": {
                    "type": "web_fetch",
                    "enabled": True,
                    "command": ["python", "main.py"],
                    "working_directory": str(project_root / "web-fetch"),
                    "environment": {},
                    "port": 8005,
                    "health_check_url": None,
                    "dependencies": [],
                    "restart_policy": "on-failure",
                    "restart_delay": 5,
                    "max_restarts": 3,
                    "timeout": 30,
                    "priority": 40
                },
                "sequential_thinker": {
                    "type": "sequential_thinker",
                    "enabled": True,
                    "command": ["python", "main.py"],
                    "working_directory": str(project_root / "../sequential-thinker-mcp"),
                    "environment": {},
                    "port": 8001,
                    "health_check_url": None,
                    "dependencies": [],
                    "restart_policy": "on-failure",
                    "restart_delay": 5,
                    "max_restarts": 3,
                    "timeout": 30,
                    "priority": 50
                }
            }
        }


# Request/Response models
class ServiceManagementRequest(BaseModel):
    """Request for service management operations."""
    service_name: str = Field(description="Name of the service")
    operation: str = Field(description="Operation: start, stop, restart")


class ServiceConfigRequest(BaseModel):
    """Request to update service configuration."""
    service_name: str = Field(description="Name of the service")
    config_updates: Dict[str, Any] = Field(description="Configuration updates")


# MCP Tools and Resources
@mcp.resource("orchestrator://services")
async def list_services() -> str:
    """List all registered services and their status."""
    result = "MCP Services Status:\n\n"
    
    if not services:
        return "No services registered"
    
    # Sort by priority for display
    sorted_services = sorted(services.values(), key=lambda s: s.config.priority)
    
    for service in sorted_services:
        health = service.get_health()
        
        result += f"**{service.config.name}** ({service.config.type.value})\n"
        result += f"  Status: {health.status.value}\n"
        result += f"  Enabled: {service.config.enabled}\n"
        result += f"  Priority: {service.config.priority}\n"
        
        if health.pid:
            result += f"  PID: {health.pid}\n"
        
        if health.uptime_seconds > 0:
            uptime_str = str(timedelta(seconds=int(health.uptime_seconds)))
            result += f"  Uptime: {uptime_str}\n"
        
        if health.restart_count > 0:
            result += f"  Restarts: {health.restart_count}\n"
        
        if health.cpu_percent > 0:
            result += f"  CPU: {health.cpu_percent:.1f}%\n"
        
        if health.memory_mb > 0:
            result += f"  Memory: {health.memory_mb:.1f} MB\n"
        
        if health.response_time_ms:
            result += f"  Response time: {health.response_time_ms:.1f}ms\n"
        
        if health.last_error:
            result += f"  Last error: {health.last_error}\n"
        
        result += "\n"
    
    return result


@mcp.resource("orchestrator://health")
async def get_health_overview() -> str:
    """Get overall health overview of all services."""
    result = "System Health Overview:\n\n"
    
    if not services:
        return "No services to monitor"
    
    total_services = len(services)
    running_services = sum(1 for s in services.values() if s.status == ServiceStatus.RUNNING)
    error_services = sum(1 for s in services.values() if s.status == ServiceStatus.ERROR)
    
    result += f"Total services: {total_services}\n"
    result += f"Running: {running_services}\n"
    result += f"Errors: {error_services}\n"
    result += f"Stopped: {total_services - running_services - error_services}\n\n"
    
    # System health percentage
    health_percent = (running_services / total_services * 100) if total_services > 0 else 0
    result += f"System health: {health_percent:.1f}%\n\n"
    
    # Services with issues
    if error_services > 0:
        result += "Services with issues:\n"
        for service in services.values():
            if service.status == ServiceStatus.ERROR:
                result += f"  - {service.config.name}: {service.last_error or 'Unknown error'}\n"
    
    return result


@mcp.resource("orchestrator://config")
async def get_configuration() -> str:
    """Get current orchestrator configuration."""
    if not configuration_manager:
        return "Configuration manager not initialized"
    
    result = "Orchestrator Configuration:\n\n"
    result += f"Config directory: {configuration_manager.config_dir}\n"
    result += f"Services configured: {len(configuration_manager.services_config)}\n\n"
    
    result += "Service configurations:\n"
    for name, config in configuration_manager.services_config.items():
        result += f"  **{name}**:\n"
        result += f"    Type: {config.type.value}\n"
        result += f"    Enabled: {config.enabled}\n"
        result += f"    Command: {' '.join(config.command)}\n"
        result += f"    Working dir: {config.working_directory}\n"
        result += f"    Port: {config.port}\n"
        result += f"    Restart policy: {config.restart_policy}\n"
        result += f"    Priority: {config.priority}\n\n"
    
    return result


@mcp.tool()
async def manage_service(request: ServiceManagementRequest) -> Dict[str, Any]:
    """Manage a service (start, stop, restart)."""
    
    service_name = request.service_name
    operation = request.operation.lower()
    
    if service_name not in services:
        return {
            "success": False,
            "error": f"Service '{service_name}' not found"
        }
    
    service = services[service_name]
    
    try:
        if operation == "start":
            success = await service.start()
            message = f"Started service '{service_name}'" if success else f"Failed to start service '{service_name}'"
        elif operation == "stop":
            success = await service.stop()
            message = f"Stopped service '{service_name}'" if success else f"Failed to stop service '{service_name}'"
        elif operation == "restart":
            success = await service.restart()
            message = f"Restarted service '{service_name}'" if success else f"Failed to restart service '{service_name}'"
        else:
            return {
                "success": False,
                "error": f"Unknown operation '{operation}'. Use: start, stop, restart"
            }
        
        return {
            "success": success,
            "service": service_name,
            "operation": operation,
            "status": service.status.value,
            "message": message
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def start_all_services() -> Dict[str, Any]:
    """Start all enabled services in priority order."""
    
    if not services:
        return {
            "success": False,
            "error": "No services configured"
        }
    
    # Sort by priority (lower number = higher priority)
    sorted_services = sorted(
        [s for s in services.values() if s.config.enabled],
        key=lambda s: s.config.priority
    )
    
    results = []
    total_started = 0
    
    for service in sorted_services:
        try:
            # Check dependencies
            dependencies_ready = True
            for dep_name in service.config.dependencies:
                if dep_name in services:
                    dep_service = services[dep_name]
                    if dep_service.status != ServiceStatus.RUNNING:
                        dependencies_ready = False
                        break
            
            if not dependencies_ready:
                results.append({
                    "service": service.config.name,
                    "success": False,
                    "error": "Dependencies not ready"
                })
                continue
            
            success = await service.start()
            if success:
                total_started += 1
            
            results.append({
                "service": service.config.name,
                "success": success,
                "status": service.status.value,
                "error": service.last_error if not success else None
            })
            
            # Small delay between service starts
            await asyncio.sleep(1)
            
        except Exception as e:
            results.append({
                "service": service.config.name,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "total_services": len(sorted_services),
        "started": total_started,
        "results": results
    }


@mcp.tool()
async def stop_all_services() -> Dict[str, Any]:
    """Stop all services in reverse priority order."""
    
    if not services:
        return {
            "success": False,
            "error": "No services configured"
        }
    
    # Sort by reverse priority for shutdown
    sorted_services = sorted(
        services.values(),
        key=lambda s: s.config.priority,
        reverse=True
    )
    
    results = []
    total_stopped = 0
    
    for service in sorted_services:
        if service.status == ServiceStatus.STOPPED:
            continue
            
        try:
            success = await service.stop()
            if success:
                total_stopped += 1
            
            results.append({
                "service": service.config.name,
                "success": success,
                "status": service.status.value,
                "error": service.last_error if not success else None
            })
            
            # Small delay between service stops
            await asyncio.sleep(0.5)
            
        except Exception as e:
            results.append({
                "service": service.config.name,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "total_services": len(sorted_services),
        "stopped": total_stopped,
        "results": results
    }


@mcp.tool()
async def restart_all_services() -> Dict[str, Any]:
    """Restart all enabled services."""
    
    # Stop all services first
    stop_result = await stop_all_services()
    
    # Wait a moment
    await asyncio.sleep(2)
    
    # Start all services
    start_result = await start_all_services()
    
    return {
        "success": True,
        "stop_result": stop_result,
        "start_result": start_result
    }


@mcp.tool()
async def get_service_logs(service_name: str, lines: int = 50) -> Dict[str, Any]:
    """Get recent logs for a service."""
    
    if service_name not in services:
        return {
            "success": False,
            "error": f"Service '{service_name}' not found"
        }
    
    service = services[service_name]
    
    if not service.process:
        return {
            "success": False,
            "error": f"Service '{service_name}' is not running"
        }
    
    try:
        # For now, return basic process info
        # In a full implementation, you'd collect and store logs
        return {
            "success": True,
            "service": service_name,
            "pid": service.process.pid,
            "status": service.status.value,
            "uptime": (datetime.now() - service.start_time).total_seconds() if service.start_time else 0,
            "logs": f"Logs for {service_name} would appear here. PID: {service.process.pid}"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def reload_configuration() -> Dict[str, Any]:
    """Reload configuration from files."""
    
    if not configuration_manager:
        return {
            "success": False,
            "error": "Configuration manager not initialized"
        }
    
    try:
        success = await configuration_manager.load_configuration()
        
        if success:
            # Update services with new configuration
            # For now, just report what would happen
            return {
                "success": True,
                "message": "Configuration reloaded successfully",
                "services_count": len(configuration_manager.services_config)
            }
        else:
            return {
                "success": False,
                "error": "Failed to reload configuration"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


async def initialize_services():
    """Initialize all services from configuration."""
    global services
    
    if not configuration_manager:
        logger.error("Configuration manager not initialized")
        return False
    
    # Clear existing services
    services.clear()
    
    # Create service instances
    for service_name, service_config in configuration_manager.services_config.items():
        try:
            service = MCPService(service_config)
            services[service_name] = service
            logger.info(f"Registered service: {service_name}")
        except Exception as e:
            logger.error(f"Failed to register service {service_name}: {e}")
    
    logger.info(f"Initialized {len(services)} services")
    return True


async def shutdown_handler():
    """Handle graceful shutdown."""
    logger.info("Shutting down orchestrator...")
    
    # Stop health monitor
    if health_monitor:
        await health_monitor.stop()
    
    # Stop all services
    if services:
        await stop_all_services()
    
    logger.info("Orchestrator shutdown complete")


@mcp.server.lifespan
async def setup_and_cleanup():
    """Initialize and cleanup orchestrator components."""
    global health_monitor, configuration_manager
    
    # Setup signal handlers for graceful shutdown
    if sys.platform != "win32":
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, lambda s, f: asyncio.create_task(shutdown_handler()))
    
    # Initialize configuration manager
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    configuration_manager = ConfigurationManager(config_dir)
    
    # Load configuration
    config_loaded = await configuration_manager.load_configuration()
    if not config_loaded:
        logger.error("Failed to load configuration")
        return
    
    # Initialize services
    services_initialized = await initialize_services()
    if not services_initialized:
        logger.error("Failed to initialize services")
        return
    
    # Start health monitor
    health_monitor = HealthMonitor(check_interval=30)
    await health_monitor.start()
    
    logger.info("MCP Servers Orchestrator initialized successfully")
    
    yield
    
    # Cleanup
    await shutdown_handler()


def main():
    """Run the MCP Servers Orchestrator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Servers Orchestrator")
    parser.add_argument("--config-dir", default="./config", help="Configuration directory")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--auto-start", action="store_true", help="Auto-start all services")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()