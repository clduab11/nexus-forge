"""
Secure Communication Layer for Agent2Agent Protocol

Provides encryption, authentication, and secure channels for agent communication.
"""

import asyncio
import base64
import json
import time
import uuid
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import os

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.x509 import Certificate, load_pem_x509_certificate
import jwt

from .core import Agent2AgentMessage, MessageType

logger = logging.getLogger(__name__)


@dataclass
class AgentCertificate:
    """Agent certificate for authentication"""
    agent_id: str
    public_key: rsa.RSAPublicKey
    certificate: Certificate
    issued_at: float
    expires_at: float
    issuer: str
    
    def is_valid(self) -> bool:
        """Check if certificate is valid"""
        current_time = time.time()
        return self.issued_at <= current_time <= self.expires_at
        
    def to_pem(self) -> bytes:
        """Export certificate to PEM format"""
        return self.certificate.public_bytes(serialization.Encoding.PEM)
        
    @classmethod
    def from_pem(cls, pem_data: bytes, agent_id: str) -> "AgentCertificate":
        """Load certificate from PEM format"""
        cert = load_pem_x509_certificate(pem_data, default_backend())
        
        return cls(
            agent_id=agent_id,
            public_key=cert.public_key(),
            certificate=cert,
            issued_at=cert.not_valid_before.timestamp(),
            expires_at=cert.not_valid_after.timestamp(),
            issuer=cert.issuer.rfc4514_string()
        )


class AgentCertificateManager:
    """Manager for agent certificates"""
    
    def __init__(self, ca_cert_path: Optional[str] = None):
        self.ca_cert_path = ca_cert_path
        self.ca_cert = None
        self.certificate_cache: Dict[str, AgentCertificate] = {}
        
        if ca_cert_path and os.path.exists(ca_cert_path):
            with open(ca_cert_path, 'rb') as f:
                self.ca_cert = load_pem_x509_certificate(f.read(), default_backend())
                
    async def verify_certificate(self, certificate: AgentCertificate) -> bool:
        """Verify agent certificate"""
        # Check validity period
        if not certificate.is_valid():
            logger.warning(f"Certificate for agent {certificate.agent_id} is expired")
            return False
            
        # In production, verify against CA
        # For now, we'll do basic validation
        if self.ca_cert:
            try:
                # Verify certificate chain
                # This is simplified - real implementation would use proper chain validation
                return True
            except Exception as e:
                logger.error(f"Certificate verification failed: {e}")
                return False
                
        # If no CA cert, accept all valid certificates (development mode)
        return True
        
    async def get_certificate(self, agent_id: str) -> Optional[AgentCertificate]:
        """Get certificate for agent"""
        # Check cache
        if agent_id in self.certificate_cache:
            cert = self.certificate_cache[agent_id]
            if cert.is_valid():
                return cert
            else:
                del self.certificate_cache[agent_id]
                
        # In production, fetch from certificate store
        # For now, return None
        return None
        
    async def store_certificate(self, certificate: AgentCertificate):
        """Store agent certificate"""
        if await self.verify_certificate(certificate):
            self.certificate_cache[certificate.agent_id] = certificate
            

class MessageEncryption:
    """Message encryption/decryption service"""
    
    def __init__(self):
        self.symmetric_keys: Dict[str, bytes] = {}
        
    def generate_session_key(self) -> Tuple[str, bytes]:
        """Generate new session key"""
        key_id = str(uuid.uuid4())
        key = Fernet.generate_key()
        self.symmetric_keys[key_id] = key
        return key_id, key
        
    def encrypt_payload(self, payload: Dict[str, Any], key_id: str) -> Dict[str, Any]:
        """Encrypt message payload"""
        if key_id not in self.symmetric_keys:
            raise ValueError(f"Unknown key ID: {key_id}")
            
        # Serialize payload
        payload_json = json.dumps(payload)
        payload_bytes = payload_json.encode('utf-8')
        
        # Encrypt
        f = Fernet(self.symmetric_keys[key_id])
        encrypted = f.encrypt(payload_bytes)
        
        return {
            "encrypted": True,
            "data": base64.b64encode(encrypted).decode('utf-8'),
            "algorithm": "fernet"
        }
        
    def decrypt_payload(self, encrypted_payload: Dict[str, Any], key_id: str) -> Dict[str, Any]:
        """Decrypt message payload"""
        if key_id not in self.symmetric_keys:
            raise ValueError(f"Unknown key ID: {key_id}")
            
        if not encrypted_payload.get("encrypted"):
            return encrypted_payload
            
        # Decode
        encrypted_data = base64.b64decode(encrypted_payload["data"])
        
        # Decrypt
        f = Fernet(self.symmetric_keys[key_id])
        decrypted = f.decrypt(encrypted_data)
        
        # Deserialize
        return json.loads(decrypted.decode('utf-8'))
        
    def encrypt_with_public_key(self, data: bytes, public_key: rsa.RSAPublicKey) -> bytes:
        """Encrypt data with RSA public key"""
        return public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
    def decrypt_with_private_key(self, encrypted_data: bytes, private_key: rsa.RSAPrivateKey) -> bytes:
        """Decrypt data with RSA private key"""
        return private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )


@dataclass
class SecureChannelConfig:
    """Configuration for secure channel"""
    enable_encryption: bool = True
    enable_authentication: bool = True
    session_timeout: int = 3600  # 1 hour
    max_message_size: int = 10 * 1024 * 1024  # 10MB
    compression: bool = True


class SecureAgentChannel:
    """Secure communication channel between agents"""
    
    def __init__(
        self,
        agent_a_id: str,
        agent_b_id: str,
        certificate_manager: AgentCertificateManager,
        encryption_service: MessageEncryption,
        config: Optional[SecureChannelConfig] = None
    ):
        self.agent_a_id = agent_a_id
        self.agent_b_id = agent_b_id
        self.channel_id = str(uuid.uuid4())
        self.certificate_manager = certificate_manager
        self.encryption_service = encryption_service
        self.config = config or SecureChannelConfig()
        
        # Channel state
        self.established = False
        self.session_key_id = None
        self.session_key = None
        self.established_at = None
        self.last_activity = None
        
        # Certificates
        self.agent_a_cert = None
        self.agent_b_cert = None
        
        # Message counters for replay protection
        self.sent_counter = 0
        self.received_counters: Dict[str, int] = {}
        
    async def establish(self) -> bool:
        """Establish secure channel"""
        try:
            # Get certificates
            self.agent_a_cert = await self.certificate_manager.get_certificate(self.agent_a_id)
            self.agent_b_cert = await self.certificate_manager.get_certificate(self.agent_b_id)
            
            if not self.agent_a_cert or not self.agent_b_cert:
                logger.error("Failed to get certificates for agents")
                return False
                
            # Verify certificates
            if self.config.enable_authentication:
                if not await self.certificate_manager.verify_certificate(self.agent_a_cert):
                    logger.error(f"Invalid certificate for agent {self.agent_a_id}")
                    return False
                    
                if not await self.certificate_manager.verify_certificate(self.agent_b_cert):
                    logger.error(f"Invalid certificate for agent {self.agent_b_id}")
                    return False
                    
            # Generate session key
            if self.config.enable_encryption:
                self.session_key_id, self.session_key = self.encryption_service.generate_session_key()
                
            # Mark as established
            self.established = True
            self.established_at = time.time()
            self.last_activity = time.time()
            
            logger.info(f"Secure channel established between {self.agent_a_id} and {self.agent_b_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to establish secure channel: {e}")
            return False
            
    async def send_message(self, message: Agent2AgentMessage) -> None:
        """Send message through secure channel"""
        if not self.established:
            raise RuntimeError("Channel not established")
            
        # Check session timeout
        if self._is_session_expired():
            raise RuntimeError("Session expired")
            
        # Update activity
        self.last_activity = time.time()
        
        # Add security headers
        message.payload["_security"] = {
            "channel_id": self.channel_id,
            "counter": self.sent_counter,
            "timestamp": time.time()
        }
        
        # Encrypt payload if enabled
        if self.config.enable_encryption and self.session_key_id:
            message.payload = self.encryption_service.encrypt_payload(
                message.payload,
                self.session_key_id
            )
            message.encryption_key_id = self.session_key_id
            
        # Sign message
        if self.config.enable_authentication:
            message.signature = await self._sign_message(message)
            
        # Increment counter
        self.sent_counter += 1
        
    async def receive_message(self, message: Agent2AgentMessage) -> Agent2AgentMessage:
        """Receive and verify message through secure channel"""
        if not self.established:
            raise RuntimeError("Channel not established")
            
        # Check session timeout
        if self._is_session_expired():
            raise RuntimeError("Session expired")
            
        # Update activity
        self.last_activity = time.time()
        
        # Verify signature
        if self.config.enable_authentication and message.signature:
            if not await self._verify_signature(message):
                raise SecurityError("Invalid message signature")
                
        # Decrypt payload if encrypted
        if self.config.enable_encryption and message.encryption_key_id:
            message.payload = self.encryption_service.decrypt_payload(
                message.payload,
                message.encryption_key_id
            )
            
        # Verify security headers
        security = message.payload.get("_security", {})
        if security:
            # Check replay attack
            sender_id = message.sender
            counter = security.get("counter", 0)
            
            if sender_id in self.received_counters:
                if counter <= self.received_counters[sender_id]:
                    raise SecurityError("Possible replay attack detected")
                    
            self.received_counters[sender_id] = counter
            
            # Remove security headers from payload
            del message.payload["_security"]
            
        return message
        
    def _is_session_expired(self) -> bool:
        """Check if session has expired"""
        if not self.established_at:
            return True
            
        elapsed = time.time() - self.established_at
        return elapsed > self.config.session_timeout
        
    async def _sign_message(self, message: Agent2AgentMessage) -> str:
        """Sign message with agent's private key"""
        # In production, this would use the agent's private key
        # For now, we'll use JWT with a shared secret
        payload = {
            "message_id": message.id,
            "sender": message.sender,
            "recipient": message.recipient,
            "type": message.type.value,
            "timestamp": message.timestamp,
            "payload_hash": self._hash_payload(message.payload)
        }
        
        # Use session key as secret for JWT
        secret = self.session_key or b"development_secret"
        return jwt.encode(payload, secret, algorithm="HS256")
        
    async def _verify_signature(self, message: Agent2AgentMessage) -> bool:
        """Verify message signature"""
        try:
            # Decode JWT
            secret = self.session_key or b"development_secret"
            decoded = jwt.decode(message.signature, secret, algorithms=["HS256"])
            
            # Verify fields match
            if decoded["message_id"] != message.id:
                return False
            if decoded["sender"] != message.sender:
                return False
            if decoded["type"] != message.type.value:
                return False
                
            # Verify payload hash
            payload_hash = self._hash_payload(message.payload)
            if decoded.get("payload_hash") != payload_hash:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False
            
    def _hash_payload(self, payload: Dict[str, Any]) -> str:
        """Create hash of payload for integrity check"""
        # Remove security headers before hashing
        payload_copy = payload.copy()
        payload_copy.pop("_security", None)
        
        # Create deterministic JSON
        payload_json = json.dumps(payload_copy, sort_keys=True)
        
        # Hash using SHA256
        digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
        digest.update(payload_json.encode('utf-8'))
        return base64.b64encode(digest.finalize()).decode('utf-8')
        
    async def close(self):
        """Close secure channel"""
        self.established = False
        self.session_key = None
        self.session_key_id = None
        
        logger.info(f"Secure channel closed between {self.agent_a_id} and {self.agent_b_id}")


class SecureChannelManager:
    """Manager for secure channels between agents"""
    
    def __init__(
        self,
        certificate_manager: AgentCertificateManager,
        encryption_service: MessageEncryption
    ):
        self.certificate_manager = certificate_manager
        self.encryption_service = encryption_service
        self.channels: Dict[str, SecureAgentChannel] = {}
        self._lock = asyncio.Lock()
        
    def _get_channel_key(self, agent_a: str, agent_b: str) -> str:
        """Get consistent channel key for agent pair"""
        agents = sorted([agent_a, agent_b])
        return f"{agents[0]}:{agents[1]}"
        
    async def get_or_create_channel(
        self,
        agent_a: str,
        agent_b: str,
        config: Optional[SecureChannelConfig] = None
    ) -> SecureAgentChannel:
        """Get existing channel or create new one"""
        channel_key = self._get_channel_key(agent_a, agent_b)
        
        async with self._lock:
            # Check existing channel
            if channel_key in self.channels:
                channel = self.channels[channel_key]
                if channel.established and not channel._is_session_expired():
                    return channel
                else:
                    # Close expired channel
                    await channel.close()
                    del self.channels[channel_key]
                    
            # Create new channel
            channel = SecureAgentChannel(
                agent_a,
                agent_b,
                self.certificate_manager,
                self.encryption_service,
                config
            )
            
            # Establish channel
            if await channel.establish():
                self.channels[channel_key] = channel
                return channel
            else:
                raise RuntimeError("Failed to establish secure channel")
                
    async def close_channel(self, agent_a: str, agent_b: str):
        """Close channel between agents"""
        channel_key = self._get_channel_key(agent_a, agent_b)
        
        async with self._lock:
            if channel_key in self.channels:
                await self.channels[channel_key].close()
                del self.channels[channel_key]
                
    async def cleanup_expired_channels(self):
        """Clean up expired channels"""
        async with self._lock:
            expired_keys = []
            
            for key, channel in self.channels.items():
                if channel._is_session_expired():
                    await channel.close()
                    expired_keys.append(key)
                    
            for key in expired_keys:
                del self.channels[key]
                
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired channels")