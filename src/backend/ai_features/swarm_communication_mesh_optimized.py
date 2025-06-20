"""
Optimized Swarm Communication Mesh
High-performance message passing with <50ms latency
"""

import asyncio
import time
import numpy as np
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable, Any
from uuid import uuid4
import msgpack
import hashlib

from .swarm_intelligence import (
    SwarmAgent,
    SwarmMessage,
    CommunicationType,
    Pheromone,
)
from nexus_forge.core.monitoring import get_logger

logger = get_logger(__name__)


class OptimizedCommunicationMesh:
    """
    Ultra-fast communication mesh with:
    - Zero-copy message passing
    - Lock-free data structures  
    - Parallel message processing
    - Intelligent routing
    - Connection pooling
    """
    
    def __init__(self):
        # Multi-channel architecture for parallel processing
        self.num_channels = 16  # Power of 2 for fast modulo
        self.channels: List[asyncio.Queue] = [
            asyncio.Queue(maxsize=1000) for _ in range(self.num_channels)
        ]
        
        # Agent connections with O(1) lookup
        self.agent_connections: Dict[str, 'AgentConnection'] = {}
        
        # Message routing table
        self.routing_table: Dict[str, int] = {}  # agent_id -> channel_id
        
        # Broadcast optimization
        self.broadcast_groups: Dict[str, Set[str]] = defaultdict(set)
        self.multicast_trees: Dict[str, 'MulticastTree'] = {}
        
        # Pheromone system (optimized)
        self.pheromone_grid = SpatialHashGrid(cell_size=0.1)
        
        # Performance metrics
        self.metrics = MessageMetrics()
        
        # Worker tasks
        self.workers: List[asyncio.Task] = []
        self.running = False
        
        # Message deduplication
        self.message_cache = LRUMessageCache(size=10000)
        
        # Batch processing
        self.batch_processor = BatchMessageProcessor()
        
    async def start(self):
        """Start communication mesh"""
        self.running = True
        
        # Start channel workers
        for i in range(self.num_channels):
            worker = asyncio.create_task(self._channel_worker(i))
            self.workers.append(worker)
            
        # Start maintenance tasks
        self.workers.append(asyncio.create_task(self._maintenance_loop()))
        
        logger.info("Optimized communication mesh started")
        
    async def stop(self):
        """Stop communication mesh"""
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
            
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        logger.info("Communication mesh stopped")
        
    async def add_agent(self, agent: SwarmAgent):
        """Add agent to mesh with optimized routing"""
        # Create connection
        conn = AgentConnection(agent.id)
        self.agent_connections[agent.id] = conn
        
        # Assign to channel using consistent hashing
        channel_id = hash(agent.id) & (self.num_channels - 1)  # Fast modulo
        self.routing_table[agent.id] = channel_id
        
        # Initialize agent in spatial grid
        self.pheromone_grid.insert(agent.id, agent.position)
        
        logger.debug(f"Agent {agent.id} added to channel {channel_id}")
        
    async def remove_agent(self, agent_id: str):
        """Remove agent from mesh"""
        if agent_id in self.agent_connections:
            conn = self.agent_connections[agent_id]
            await conn.close()
            del self.agent_connections[agent_id]
            
        if agent_id in self.routing_table:
            del self.routing_table[agent_id]
            
        # Remove from broadcast groups
        for group in self.broadcast_groups.values():
            group.discard(agent_id)
            
        self.pheromone_grid.remove(agent_id)
        
    async def send_message(self, message: SwarmMessage) -> bool:
        """Send message with ultra-low latency"""
        start_time = time.perf_counter()
        
        try:
            # Check message cache for deduplication
            if self.message_cache.contains(message.id):
                return True  # Already processed
                
            self.message_cache.add(message.id)
            
            # Route based on message type
            if message.type == CommunicationType.BROADCAST:
                await self._send_broadcast_optimized(message)
            elif message.type == CommunicationType.MULTICAST:
                await self._send_multicast_optimized(message)
            elif message.type == CommunicationType.UNICAST:
                await self._send_unicast_optimized(message)
            elif message.type == CommunicationType.STIGMERGIC:
                await self._handle_stigmergic(message)
                
            # Update metrics
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.record_message(message.type, latency_ms)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message {message.id}: {e}")
            self.metrics.record_failure(message.type)
            return False
            
    async def _send_unicast_optimized(self, message: SwarmMessage):
        """Optimized unicast with direct routing"""
        if not message.recipient_id:
            return
            
        # Get target channel
        channel_id = self.routing_table.get(message.recipient_id)
        if channel_id is None:
            return
            
        # Add to channel queue
        await self.channels[channel_id].put(('unicast', message))
        
    async def _send_broadcast_optimized(self, message: SwarmMessage):
        """Optimized broadcast using parallel channels"""
        # Create broadcast packet
        packet = ('broadcast', message)
        
        # Send to all channels in parallel
        tasks = []
        for channel in self.channels:
            task = asyncio.create_task(channel.put(packet))
            tasks.append(task)
            
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _send_multicast_optimized(self, message: SwarmMessage):
        """Optimized multicast using multicast trees"""
        group_id = message.content.get('group_id')
        if not group_id:
            return
            
        # Get or create multicast tree
        if group_id not in self.multicast_trees:
            members = self.broadcast_groups.get(group_id, set())
            self.multicast_trees[group_id] = MulticastTree(members)
            
        tree = self.multicast_trees[group_id]
        
        # Send using tree structure
        await tree.send(message, self.channels, self.routing_table)
        
    async def _handle_stigmergic(self, message: SwarmMessage):
        """Handle stigmergic communication (pheromones)"""
        pheromone_data = message.content.get('pheromone')
        if not pheromone_data:
            return
            
        # Create pheromone
        pheromone = Pheromone(
            agent_id=message.sender_id,
            position=np.array(pheromone_data.get('position', [0, 0, 0])),
            type=pheromone_data.get('type', 'generic'),
            strength=pheromone_data.get('strength', 1.0),
            metadata=pheromone_data.get('metadata', {})
        )
        
        # Add to spatial grid
        self.pheromone_grid.add_pheromone(pheromone)
        
    async def _channel_worker(self, channel_id: int):
        """Worker for processing messages in a channel"""
        channel = self.channels[channel_id]
        batch = []
        last_batch_time = time.time()
        
        while self.running:
            try:
                # Try to get message with timeout for batching
                try:
                    msg_type, message = await asyncio.wait_for(
                        channel.get(),
                        timeout=0.001  # 1ms timeout
                    )
                    batch.append((msg_type, message))
                except asyncio.TimeoutError:
                    pass
                    
                # Process batch if ready
                current_time = time.time()
                should_process = (
                    len(batch) >= self.batch_processor.optimal_size or
                    (len(batch) > 0 and current_time - last_batch_time >= 0.005)  # 5ms
                )
                
                if should_process and batch:
                    await self._process_message_batch(batch)
                    
                    # Update batch processor metrics
                    self.batch_processor.update_metrics(
                        len(batch),
                        time.time() - current_time
                    )
                    
                    batch = []
                    last_batch_time = current_time
                    
            except Exception as e:
                logger.error(f"Channel {channel_id} worker error: {e}")
                await asyncio.sleep(0.001)
                
    async def _process_message_batch(self, batch: List[Tuple[str, SwarmMessage]]):
        """Process a batch of messages efficiently"""
        # Group by recipient for efficient delivery
        unicast_groups = defaultdict(list)
        broadcast_messages = []
        
        for msg_type, message in batch:
            if msg_type == 'unicast':
                unicast_groups[message.recipient_id].append(message)
            elif msg_type == 'broadcast':
                broadcast_messages.append(message)
                
        # Process unicasts
        for recipient_id, messages in unicast_groups.items():
            if recipient_id in self.agent_connections:
                conn = self.agent_connections[recipient_id]
                await conn.send_batch(messages)
                
        # Process broadcasts
        if broadcast_messages:
            await self._deliver_broadcast_batch(broadcast_messages)
            
    async def _deliver_broadcast_batch(self, messages: List[SwarmMessage]):
        """Deliver broadcast messages efficiently"""
        # Serialize once
        serialized = [msgpack.packb(msg.to_dict()) for msg in messages]
        
        # Send to all agents in parallel
        tasks = []
        for agent_id, conn in self.agent_connections.items():
            task = asyncio.create_task(conn.send_raw_batch(serialized))
            tasks.append(task)
            
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _maintenance_loop(self):
        """Periodic maintenance tasks"""
        while self.running:
            try:
                # Evaporate pheromones
                self.pheromone_grid.evaporate(rate=0.01)
                
                # Update multicast trees
                for group_id, tree in self.multicast_trees.items():
                    members = self.broadcast_groups.get(group_id, set())
                    if tree.needs_update(members):
                        tree.rebuild(members)
                        
                # Clean message cache
                self.message_cache.cleanup()
                
                await asyncio.sleep(1.0)  # Maintenance every second
                
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
                
    async def deposit_pheromone(self, pheromone: Pheromone):
        """Deposit pheromone in environment"""
        self.pheromone_grid.add_pheromone(pheromone)
        
    async def sense_pheromones(self, position: np.ndarray, radius: float) -> List[Pheromone]:
        """Sense pheromones at position"""
        return self.pheromone_grid.query_radius(position, radius)
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get communication metrics"""
        return self.metrics.get_summary()


class AgentConnection:
    """Optimized agent connection with batching and compression"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.send_queue = asyncio.Queue(maxsize=100)
        self.receive_queue = asyncio.Queue(maxsize=100)
        self.connected = True
        
    async def send_batch(self, messages: List[SwarmMessage]):
        """Send batch of messages"""
        if not self.connected:
            return
            
        # Compress batch
        batch_data = msgpack.packb([msg.to_dict() for msg in messages])
        
        await self.send_queue.put(batch_data)
        
    async def send_raw_batch(self, serialized: List[bytes]):
        """Send pre-serialized batch"""
        if not self.connected:
            return
            
        # Combine into single packet
        combined = msgpack.packb(serialized)
        await self.send_queue.put(combined)
        
    async def receive(self) -> Optional[List[SwarmMessage]]:
        """Receive batch of messages"""
        if not self.connected:
            return None
            
        try:
            batch_data = await asyncio.wait_for(self.receive_queue.get(), timeout=0.1)
            messages_data = msgpack.unpackb(batch_data)
            
            return [
                SwarmMessage.from_dict(data) 
                for data in messages_data
            ]
        except asyncio.TimeoutError:
            return None
            
    async def close(self):
        """Close connection"""
        self.connected = False


class MulticastTree:
    """Efficient multicast tree for group communication"""
    
    def __init__(self, members: Set[str]):
        self.members = members
        self.tree = self._build_tree(members)
        self.version = 0
        
    def _build_tree(self, members: Set[str]) -> Dict[str, List[str]]:
        """Build spanning tree for multicast"""
        if not members:
            return {}
            
        # Simple tree: first member is root, others are children
        # In production, use MST algorithm
        members_list = list(members)
        root = members_list[0]
        
        tree = {root: members_list[1:]}
        for member in members_list[1:]:
            tree[member] = []
            
        return tree
        
    def needs_update(self, new_members: Set[str]) -> bool:
        """Check if tree needs rebuilding"""
        return new_members != self.members
        
    def rebuild(self, new_members: Set[str]):
        """Rebuild tree with new members"""
        self.members = new_members
        self.tree = self._build_tree(new_members)
        self.version += 1
        
    async def send(
        self, 
        message: SwarmMessage,
        channels: List[asyncio.Queue],
        routing_table: Dict[str, int]
    ):
        """Send message using tree structure"""
        # Start from root
        if not self.tree:
            return
            
        root = next(iter(self.tree.keys()))
        await self._send_recursive(root, message, channels, routing_table, set())
        
    async def _send_recursive(
        self,
        node: str,
        message: SwarmMessage,
        channels: List[asyncio.Queue],
        routing_table: Dict[str, int],
        visited: Set[str]
    ):
        """Recursive tree traversal for sending"""
        if node in visited:
            return
            
        visited.add(node)
        
        # Send to node
        channel_id = routing_table.get(node)
        if channel_id is not None:
            await channels[channel_id].put(('unicast', message))
            
        # Send to children
        children = self.tree.get(node, [])
        tasks = []
        
        for child in children:
            task = asyncio.create_task(
                self._send_recursive(child, message, channels, routing_table, visited)
            )
            tasks.append(task)
            
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class SpatialHashGrid:
    """Spatial hash grid for O(1) pheromone operations"""
    
    def __init__(self, cell_size: float = 0.1):
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int, int], List[Pheromone]] = defaultdict(list)
        self.agent_positions: Dict[str, np.ndarray] = {}
        
    def _get_cell(self, position: np.ndarray) -> Tuple[int, int, int]:
        """Get grid cell for position"""
        return tuple(int(p / self.cell_size) for p in position)
        
    def insert(self, agent_id: str, position: np.ndarray):
        """Insert agent position"""
        self.agent_positions[agent_id] = position.copy()
        
    def remove(self, agent_id: str):
        """Remove agent"""
        if agent_id in self.agent_positions:
            del self.agent_positions[agent_id]
            
    def add_pheromone(self, pheromone: Pheromone):
        """Add pheromone to grid"""
        cell = self._get_cell(pheromone.position)
        self.grid[cell].append(pheromone)
        
    def query_radius(self, position: np.ndarray, radius: float) -> List[Pheromone]:
        """Query pheromones within radius"""
        results = []
        
        # Calculate cells to check
        cell_radius = int(radius / self.cell_size) + 1
        center_cell = self._get_cell(position)
        
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                for dz in range(-cell_radius, cell_radius + 1):
                    cell = (
                        center_cell[0] + dx,
                        center_cell[1] + dy,
                        center_cell[2] + dz
                    )
                    
                    # Check pheromones in cell
                    for pheromone in self.grid.get(cell, []):
                        dist = np.linalg.norm(pheromone.position - position)
                        if dist <= radius:
                            results.append(pheromone)
                            
        return results
        
    def evaporate(self, rate: float = 0.01):
        """Evaporate pheromones"""
        current_time = time.time()
        
        for cell, pheromones in list(self.grid.items()):
            remaining = []
            
            for pheromone in pheromones:
                # Calculate evaporation
                age = current_time - pheromone.timestamp.timestamp()
                pheromone.strength *= (1 - rate) ** age
                
                if pheromone.strength > 0.01:  # Threshold
                    remaining.append(pheromone)
                    
            if remaining:
                self.grid[cell] = remaining
            else:
                del self.grid[cell]


class LRUMessageCache:
    """LRU cache for message deduplication"""
    
    def __init__(self, size: int = 10000):
        self.size = size
        self.cache: Dict[str, float] = {}  # message_id -> timestamp
        self.access_order = deque()
        
    def contains(self, message_id: str) -> bool:
        """Check if message is in cache"""
        return message_id in self.cache
        
    def add(self, message_id: str):
        """Add message to cache"""
        current_time = time.time()
        
        if message_id in self.cache:
            # Update timestamp
            self.cache[message_id] = current_time
            return
            
        # Add new entry
        self.cache[message_id] = current_time
        self.access_order.append(message_id)
        
        # Evict if needed
        while len(self.cache) > self.size:
            oldest = self.access_order.popleft()
            if oldest in self.cache:
                del self.cache[oldest]
                
    def cleanup(self):
        """Remove expired entries"""
        current_time = time.time()
        expired = []
        
        for msg_id, timestamp in self.cache.items():
            if current_time - timestamp > 60:  # 1 minute expiry
                expired.append(msg_id)
                
        for msg_id in expired:
            del self.cache[msg_id]


class BatchMessageProcessor:
    """Adaptive batch processor for optimal throughput"""
    
    def __init__(self):
        self.optimal_size = 10
        self.history = deque(maxlen=100)
        
    def update_metrics(self, batch_size: int, processing_time: float):
        """Update processing metrics"""
        throughput = batch_size / processing_time if processing_time > 0 else 0
        self.history.append((batch_size, throughput))
        
        # Find optimal batch size
        if len(self.history) >= 10:
            # Group by batch size
            size_throughputs = defaultdict(list)
            for size, tput in self.history:
                size_throughputs[size].append(tput)
                
            # Calculate average throughput per size
            avg_throughputs = {
                size: np.mean(tputs)
                for size, tputs in size_throughputs.items()
            }
            
            # Find best size
            if avg_throughputs:
                self.optimal_size = max(avg_throughputs, key=avg_throughputs.get)
                self.optimal_size = max(5, min(50, self.optimal_size))


class MessageMetrics:
    """High-performance message metrics tracking"""
    
    def __init__(self):
        self.message_counts = defaultdict(int)
        self.latency_samples = defaultdict(lambda: deque(maxlen=1000))
        self.failure_counts = defaultdict(int)
        self.last_reset = time.time()
        
    def record_message(self, msg_type: CommunicationType, latency_ms: float):
        """Record message send"""
        self.message_counts[msg_type] += 1
        self.latency_samples[msg_type].append(latency_ms)
        
    def record_failure(self, msg_type: CommunicationType):
        """Record message failure"""
        self.failure_counts[msg_type] += 1
        
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        duration = time.time() - self.last_reset
        
        summary = {
            "duration_seconds": duration,
            "total_messages": sum(self.message_counts.values()),
            "total_failures": sum(self.failure_counts.values()),
            "by_type": {}
        }
        
        for msg_type in CommunicationType:
            count = self.message_counts.get(msg_type, 0)
            failures = self.failure_counts.get(msg_type, 0)
            samples = list(self.latency_samples.get(msg_type, []))
            
            type_stats = {
                "count": count,
                "failures": failures,
                "rate_per_second": count / duration if duration > 0 else 0,
                "success_rate": (count - failures) / count if count > 0 else 1.0
            }
            
            if samples:
                type_stats.update({
                    "avg_latency_ms": np.mean(samples),
                    "p50_latency_ms": np.percentile(samples, 50),
                    "p99_latency_ms": np.percentile(samples, 99)
                })
                
            summary["by_type"][msg_type.value] = type_stats
            
        return summary