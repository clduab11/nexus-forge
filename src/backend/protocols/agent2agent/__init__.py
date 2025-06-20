"""
ADK Agent2Agent Protocol Implementation

This module provides the core implementation of the Agent2Agent protocol
for secure, bidirectional communication between AI agents in Nexus Forge.
"""

from .core import (
    Agent2AgentMessage,
    MessageType,
    Agent2AgentProtocol,
    ProtocolVersion
)

from .discovery import (
    AgentDiscoveryService,
    AgentRegistry,
    CapabilityIndex
)

from .security import (
    SecureAgentChannel,
    AgentCertificateManager,
    MessageEncryption
)

from .negotiation import (
    CapabilityNegotiationEngine,
    TaskContract,
    ContractTerms
)

from .integration import (
    Agent2AgentIntegration,
    WebSocketBridge,
    ADKProtocolAdapter
)

__all__ = [
    # Core
    "Agent2AgentMessage",
    "MessageType", 
    "Agent2AgentProtocol",
    "ProtocolVersion",
    
    # Discovery
    "AgentDiscoveryService",
    "AgentRegistry",
    "CapabilityIndex",
    
    # Security
    "SecureAgentChannel",
    "AgentCertificateManager", 
    "MessageEncryption",
    
    # Negotiation
    "CapabilityNegotiationEngine",
    "TaskContract",
    "ContractTerms",
    
    # Integration
    "Agent2AgentIntegration",
    "WebSocketBridge",
    "ADKProtocolAdapter"
]

# Protocol version
VERSION = "2.0.0"