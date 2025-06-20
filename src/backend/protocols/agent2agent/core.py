"""
Core Agent2Agent Protocol Implementation

Provides the fundamental message types, structures, and protocol handling
for agent-to-agent communication in the ADK ecosystem.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import logging

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
import msgpack

logger = logging.getLogger(__name__)


class ProtocolVersion:
    """Protocol version management"""
    MAJOR = 2
    MINOR = 0
    PATCH = 0
    
    @classmethod
    def get_version(cls) -> str:
        return f"{cls.MAJOR}.{cls.MINOR}.{cls.PATCH}"
    
    @classmethod
    def is_compatible(cls, version: str) -> bool:
        """Check if a version is compatible with current protocol"""
        parts = version.split(".")
        if len(parts) != 3:
            return False
        
        major = int(parts[0])
        # Major version must match for compatibility
        return major == cls.MAJOR


class MessageType(Enum):
    """Agent2Agent message types"""
    
    # Discovery Messages
    AGENT_ANNOUNCE = "agent.announce"
    AGENT_DISCOVER = "agent.discover" 
    AGENT_QUERY = "agent.query"
    AGENT_GOODBYE = "agent.goodbye"
    
    # Capability Messages
    CAPABILITY_REGISTER = "capability.register"
    CAPABILITY_REQUEST = "capability.request"
    CAPABILITY_MATCH = "capability.match"
    CAPABILITY_UPDATE = "capability.update"
    
    # Task Coordination
    TASK_PROPOSE = "task.propose"
    TASK_ACCEPT = "task.accept"
    TASK_REJECT = "task.reject"
    TASK_DELEGATE = "task.delegate"
    TASK_UPDATE = "task.update"
    TASK_COMPLETE = "task.complete"
    TASK_FAILED = "task.failed"
    
    # Resource Sharing
    RESOURCE_OFFER = "resource.offer"
    RESOURCE_REQUEST = "resource.request"
    RESOURCE_GRANT = "resource.grant"
    RESOURCE_DENY = "resource.deny"
    RESOURCE_TRANSFER = "resource.transfer"
    RESOURCE_RELEASE = "resource.release"
    
    # Collaboration
    COLLAB_INVITE = "collab.invite"
    COLLAB_JOIN = "collab.join"
    COLLAB_LEAVE = "collab.leave"
    COLLAB_SYNC = "collab.sync"
    
    # Health & Monitoring
    HEALTH_CHECK = "health.check"
    HEALTH_REPORT = "health.report"
    METRICS_REPORT = "metrics.report"
    STATUS_UPDATE = "status.update"
    
    # Control Messages
    PROTOCOL_HANDSHAKE = "protocol.handshake"
    PROTOCOL_UPGRADE = "protocol.upgrade"
    ERROR = "error"
    ACK = "ack"
    NACK = "nack"


@dataclass
class Agent2AgentMessage:
    """Core message structure for agent communication"""
    
    # Message identification
    id: str
    correlation_id: Optional[str] = None  # For request-response correlation
    
    # Message routing
    type: MessageType
    sender: str
    recipient: Optional[str] = None  # None for broadcast
    
    # Message content
    payload: Dict[str, Any]
    
    # Message metadata
    timestamp: float
    version: str = ProtocolVersion.get_version()
    priority: int = 0  # 0=normal, 1=high, 2=urgent
    ttl: Optional[int] = None  # Time to live in seconds
    
    # Security
    signature: Optional[str] = None  # Cryptographic signature
    encryption_key_id: Optional[str] = None  # For encrypted payloads
    
    # Delivery guarantees
    requires_ack: bool = False
    max_retries: int = 3
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "id": self.id,
            "correlation_id": self.correlation_id,
            "type": self.type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "version": self.version,
            "priority": self.priority,
            "ttl": self.ttl,
            "signature": self.signature,
            "encryption_key_id": self.encryption_key_id,
            "requires_ack": self.requires_ack,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agent2AgentMessage":
        """Create message from dictionary"""
        return cls(
            id=data["id"],
            correlation_id=data.get("correlation_id"),
            type=MessageType(data["type"]),
            sender=data["sender"],
            recipient=data.get("recipient"),
            payload=data["payload"],
            timestamp=data["timestamp"],
            version=data.get("version", ProtocolVersion.get_version()),
            priority=data.get("priority", 0),
            ttl=data.get("ttl"),
            signature=data.get("signature"),
            encryption_key_id=data.get("encryption_key_id"),
            requires_ack=data.get("requires_ack", False),
            max_retries=data.get("max_retries", 3),
            retry_count=data.get("retry_count", 0)
        )
    
    def to_bytes(self) -> bytes:
        """Serialize message to bytes using MessagePack"""
        return msgpack.packb(self.to_dict())
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "Agent2AgentMessage":
        """Deserialize message from bytes"""
        return cls.from_dict(msgpack.unpackb(data, raw=False))
    
    def is_expired(self) -> bool:
        """Check if message has expired based on TTL"""
        if self.ttl is None:
            return False
        
        elapsed = time.time() - self.timestamp
        return elapsed > self.ttl
    
    def create_ack(self) -> "Agent2AgentMessage":
        """Create acknowledgment message"""
        return Agent2AgentMessage(
            id=str(uuid.uuid4()),
            correlation_id=self.id,
            type=MessageType.ACK,
            sender=self.recipient,
            recipient=self.sender,
            payload={"original_message_id": self.id},
            timestamp=time.time()
        )
    
    def create_error_response(self, error: str, details: Optional[Dict] = None) -> "Agent2AgentMessage":
        """Create error response message"""
        return Agent2AgentMessage(
            id=str(uuid.uuid4()),
            correlation_id=self.id,
            type=MessageType.ERROR,
            sender=self.recipient,
            recipient=self.sender,
            payload={
                "error": error,
                "details": details or {},
                "original_message_id": self.id
            },
            timestamp=time.time()
        )


class MessageHandler:
    """Base class for message handlers"""
    
    def __init__(self):
        self.handlers: Dict[MessageType, Callable] = {}
        
    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register handler for message type"""
        self.handlers[message_type] = handler
        
    async def handle_message(self, message: Agent2AgentMessage) -> Optional[Agent2AgentMessage]:
        """Handle incoming message"""
        handler = self.handlers.get(message.type)
        if not handler:
            logger.warning(f"No handler registered for message type: {message.type}")
            return None
            
        try:
            return await handler(message)
        except Exception as e:
            logger.error(f"Error handling message {message.id}: {e}")
            return message.create_error_response(str(e))


class Agent2AgentProtocol:
    """Core protocol implementation"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.message_handler = MessageHandler()
        self.pending_acks: Dict[str, asyncio.Future] = {}
        self.message_queue = asyncio.Queue()
        self.running = False
        
        # Setup default handlers
        self._setup_default_handlers()
        
    def _setup_default_handlers(self):
        """Setup default message handlers"""
        self.message_handler.register_handler(
            MessageType.HEALTH_CHECK,
            self._handle_health_check
        )
        self.message_handler.register_handler(
            MessageType.ACK,
            self._handle_ack
        )
        self.message_handler.register_handler(
            MessageType.PROTOCOL_HANDSHAKE,
            self._handle_handshake
        )
        
    async def _handle_health_check(self, message: Agent2AgentMessage) -> Agent2AgentMessage:
        """Handle health check message"""
        return Agent2AgentMessage(
            id=str(uuid.uuid4()),
            correlation_id=message.id,
            type=MessageType.HEALTH_REPORT,
            sender=self.agent_id,
            recipient=message.sender,
            payload={
                "status": "healthy",
                "timestamp": time.time(),
                "protocol_version": ProtocolVersion.get_version()
            },
            timestamp=time.time()
        )
        
    async def _handle_ack(self, message: Agent2AgentMessage) -> None:
        """Handle acknowledgment message"""
        correlation_id = message.correlation_id
        if correlation_id in self.pending_acks:
            self.pending_acks[correlation_id].set_result(message)
            
    async def _handle_handshake(self, message: Agent2AgentMessage) -> Agent2AgentMessage:
        """Handle protocol handshake"""
        peer_version = message.payload.get("version", "0.0.0")
        
        if not ProtocolVersion.is_compatible(peer_version):
            return message.create_error_response(
                "Incompatible protocol version",
                {"supported_version": ProtocolVersion.get_version()}
            )
            
        return Agent2AgentMessage(
            id=str(uuid.uuid4()),
            correlation_id=message.id,
            type=MessageType.PROTOCOL_HANDSHAKE,
            sender=self.agent_id,
            recipient=message.sender,
            payload={
                "version": ProtocolVersion.get_version(),
                "capabilities": self._get_capabilities(),
                "status": "ready"
            },
            timestamp=time.time()
        )
        
    def _get_capabilities(self) -> List[str]:
        """Get protocol capabilities"""
        return [
            "discovery",
            "negotiation",
            "task_coordination",
            "resource_sharing",
            "secure_communication",
            "health_monitoring"
        ]
        
    async def send_message(self, message: Agent2AgentMessage) -> Optional[Agent2AgentMessage]:
        """Send message and optionally wait for response"""
        # Set sender if not set
        if not message.sender:
            message.sender = self.agent_id
            
        # Generate ID if not set
        if not message.id:
            message.id = str(uuid.uuid4())
            
        # Add to queue for sending
        await self.message_queue.put(message)
        
        # Wait for acknowledgment if required
        if message.requires_ack:
            future = asyncio.Future()
            self.pending_acks[message.id] = future
            
            try:
                # Wait for ack with timeout
                ack = await asyncio.wait_for(future, timeout=30.0)
                return ack
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for ack for message {message.id}")
                del self.pending_acks[message.id]
                return None
                
        return None
        
    async def process_incoming(self, message: Agent2AgentMessage) -> Optional[Agent2AgentMessage]:
        """Process incoming message"""
        # Check if message is expired
        if message.is_expired():
            logger.warning(f"Dropping expired message {message.id}")
            return None
            
        # Check version compatibility
        if not ProtocolVersion.is_compatible(message.version):
            logger.warning(f"Incompatible message version: {message.version}")
            return message.create_error_response("Incompatible protocol version")
            
        # Handle message
        response = await self.message_handler.handle_message(message)
        
        # Send ack if required
        if message.requires_ack and message.type != MessageType.ACK:
            ack = message.create_ack()
            await self.send_message(ack)
            
        return response
        
    async def start(self):
        """Start protocol processing"""
        self.running = True
        logger.info(f"Agent2Agent protocol started for agent {self.agent_id}")
        
    async def stop(self):
        """Stop protocol processing"""
        self.running = False
        
        # Cancel pending acks
        for future in self.pending_acks.values():
            if not future.done():
                future.cancel()
                
        self.pending_acks.clear()
        logger.info(f"Agent2Agent protocol stopped for agent {self.agent_id}")


class ProtocolStats:
    """Protocol statistics tracking"""
    
    def __init__(self):
        self.messages_sent = 0
        self.messages_received = 0
        self.messages_failed = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.active_connections = 0
        self.start_time = time.time()
        
    def record_sent(self, message: Agent2AgentMessage, size: int):
        """Record sent message"""
        self.messages_sent += 1
        self.bytes_sent += size
        
    def record_received(self, message: Agent2AgentMessage, size: int):
        """Record received message"""
        self.messages_received += 1
        self.bytes_received += size
        
    def record_failed(self):
        """Record failed message"""
        self.messages_failed += 1
        
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        uptime = time.time() - self.start_time
        
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "messages_failed": self.messages_failed,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "active_connections": self.active_connections,
            "uptime_seconds": uptime,
            "avg_messages_per_second": (self.messages_sent + self.messages_received) / uptime if uptime > 0 else 0
        }