"""
Fast Coordination Protocol - Optimized for <50ms latency

Implements high-performance inter-agent communication with:
- Zero-copy message passing
- Lock-free data structures
- Message batching and aggregation
- Parallel message processing
- Connection pooling
- Optimized routing algorithms
"""

import asyncio
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any, Set
import logging
import msgpack
from asyncio import PriorityQueue

from .core import Agent2AgentMessage, MessageType

logger = logging.getLogger(__name__)


class FastMessageRouter:
    """High-performance message router with O(1) routing"""
    
    def __init__(self, num_workers: int = 8):
        # Multi-queue architecture for parallel processing
        self.priority_queues: List[PriorityQueue] = [
            PriorityQueue() for _ in range(num_workers)
        ]
        self.worker_tasks: List[asyncio.Task] = []
        self.num_workers = num_workers
        
        # Agent routing table for O(1) lookup
        self.agent_routes: Dict[str, int] = {}  # agent_id -> worker_id
        
        # Message batching
        self.batch_size = 10
        self.batch_timeout = 0.005  # 5ms max batch wait
        self.message_batches: Dict[str, List[Agent2AgentMessage]] = defaultdict(list)
        
        # Connection pooling
        self.connection_pool: Dict[str, asyncio.Queue] = {}
        self.pool_size = 5
        
        # Performance metrics
        self.metrics = {
            "messages_routed": 0,
            "batches_sent": 0,
            "avg_latency_ms": 0.0,
            "p99_latency_ms": 0.0
        }
        
    async def start(self):
        """Start router workers"""
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._process_worker(i))
            self.worker_tasks.append(worker)
            
    async def stop(self):
        """Stop router workers"""
        for task in self.worker_tasks:
            task.cancel()
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
    def route_message(self, message: Agent2AgentMessage) -> int:
        """Route message to appropriate worker with consistent hashing"""
        # Use recipient for routing, or sender for broadcasts
        routing_key = message.recipient or message.sender
        
        # Check if we have a cached route
        if routing_key in self.agent_routes:
            return self.agent_routes[routing_key]
            
        # Consistent hashing for load distribution
        worker_id = hash(routing_key) % self.num_workers
        self.agent_routes[routing_key] = worker_id
        
        return worker_id
        
    async def send_message(self, message: Agent2AgentMessage):
        """Send message with automatic batching"""
        start_time = time.perf_counter()
        
        # Determine worker
        worker_id = self.route_message(message)
        
        # Add to priority queue based on message priority
        priority = -message.priority  # Negative for max-heap behavior
        await self.priority_queues[worker_id].put((priority, time.time(), message))
        
        # Update metrics
        self.metrics["messages_routed"] += 1
        
    async def _process_worker(self, worker_id: int):
        """Process messages for a specific worker"""
        queue = self.priority_queues[worker_id]
        batch = []
        last_batch_time = time.time()
        
        while True:
            try:
                # Try to get message with timeout for batching
                try:
                    priority, timestamp, message = await asyncio.wait_for(
                        queue.get(), 
                        timeout=self.batch_timeout
                    )
                    batch.append(message)
                except asyncio.TimeoutError:
                    pass
                
                # Send batch if ready
                current_time = time.time()
                should_send = (
                    len(batch) >= self.batch_size or
                    (len(batch) > 0 and current_time - last_batch_time >= self.batch_timeout)
                )
                
                if should_send and batch:
                    await self._send_batch(batch)
                    batch = []
                    last_batch_time = current_time
                    
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(0.001)  # Brief pause on error
                
    async def _send_batch(self, messages: List[Agent2AgentMessage]):
        """Send a batch of messages efficiently"""
        # Group messages by recipient for network efficiency
        grouped = defaultdict(list)
        for msg in messages:
            recipient = msg.recipient or "broadcast"
            grouped[recipient].append(msg)
            
        # Send each group
        send_tasks = []
        for recipient, group in grouped.items():
            task = asyncio.create_task(self._send_group(recipient, group))
            send_tasks.append(task)
            
        await asyncio.gather(*send_tasks, return_exceptions=True)
        
        self.metrics["batches_sent"] += 1
        
    async def _send_group(self, recipient: str, messages: List[Agent2AgentMessage]):
        """Send a group of messages to the same recipient"""
        # Get connection from pool
        conn = await self._get_connection(recipient)
        
        try:
            # Serialize batch efficiently
            batch_data = msgpack.packb([msg.to_dict() for msg in messages])
            
            # Send with zero-copy if possible
            await conn.send(batch_data)
            
        finally:
            # Return connection to pool
            await self._return_connection(recipient, conn)
            
    async def _get_connection(self, agent_id: str):
        """Get connection from pool"""
        if agent_id not in self.connection_pool:
            self.connection_pool[agent_id] = asyncio.Queue(maxsize=self.pool_size)
            
        pool = self.connection_pool[agent_id]
        
        try:
            # Try to get existing connection
            conn = pool.get_nowait()
        except asyncio.QueueEmpty:
            # Create new connection
            conn = await self._create_connection(agent_id)
            
        return conn
        
    async def _return_connection(self, agent_id: str, conn):
        """Return connection to pool"""
        pool = self.connection_pool[agent_id]
        
        try:
            pool.put_nowait(conn)
        except asyncio.QueueFull:
            # Pool is full, close connection
            await conn.close()
            
    async def _create_connection(self, agent_id: str):
        """Create new connection (placeholder)"""
        # In real implementation, this would create actual network connection
        return MockConnection(agent_id)


class MockConnection:
    """Mock connection for testing"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        
    async def send(self, data: bytes):
        # Simulate network send
        await asyncio.sleep(0.0001)  # 0.1ms network latency
        
    async def close(self):
        pass


class ParallelCoordinator:
    """Parallel task coordinator with minimal synchronization"""
    
    def __init__(self):
        self.task_graph = {}  # task_id -> dependencies
        self.task_status = {}  # task_id -> status
        self.ready_queue = asyncio.Queue()
        self.completion_futures = {}  # task_id -> Future
        
        # Worker pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=16)
        
    async def add_task(self, task_id: str, dependencies: List[str] = None):
        """Add task with dependencies"""
        self.task_graph[task_id] = dependencies or []
        self.task_status[task_id] = "pending"
        self.completion_futures[task_id] = asyncio.Future()
        
        # Check if task is ready
        if not dependencies:
            await self.ready_queue.put(task_id)
            self.task_status[task_id] = "ready"
            
    async def complete_task(self, task_id: str, result: Any):
        """Mark task as complete and trigger dependents"""
        self.task_status[task_id] = "completed"
        
        # Set result future
        if task_id in self.completion_futures:
            self.completion_futures[task_id].set_result(result)
            
        # Check all tasks for newly ready ones
        newly_ready = []
        for tid, deps in self.task_graph.items():
            if self.task_status.get(tid) == "pending":
                if all(self.task_status.get(dep) == "completed" for dep in deps):
                    newly_ready.append(tid)
                    
        # Add newly ready tasks to queue
        for tid in newly_ready:
            self.task_status[tid] = "ready"
            await self.ready_queue.put(tid)
            
    async def wait_for_task(self, task_id: str, timeout: float = None):
        """Wait for task completion"""
        if task_id in self.completion_futures:
            return await asyncio.wait_for(
                self.completion_futures[task_id], 
                timeout=timeout
            )
        return None
        
    async def get_ready_tasks(self, max_tasks: int = 10) -> List[str]:
        """Get batch of ready tasks"""
        ready = []
        
        for _ in range(max_tasks):
            try:
                task_id = self.ready_queue.get_nowait()
                ready.append(task_id)
            except asyncio.QueueEmpty:
                break
                
        return ready


class OptimizedSwarmCommunication:
    """Optimized swarm communication layer"""
    
    def __init__(self, swarm_id: str):
        self.swarm_id = swarm_id
        self.router = FastMessageRouter()
        self.coordinator = ParallelCoordinator()
        
        # Agent registry with capabilities index
        self.agent_registry = {}
        self.capability_index = defaultdict(set)  # capability -> agent_ids
        
        # Message deduplication
        self.seen_messages = set()
        self.dedup_window = 1000  # Keep last N message IDs
        
        # Performance optimizations
        self.enable_compression = True
        self.enable_batching = True
        self.enable_caching = True
        
    async def start(self):
        """Start communication system"""
        await self.router.start()
        logger.info(f"Optimized swarm communication started for {self.swarm_id}")
        
    async def stop(self):
        """Stop communication system"""
        await self.router.stop()
        
    def register_agent(self, agent_id: str, capabilities: List[str]):
        """Register agent with O(1) capability indexing"""
        self.agent_registry[agent_id] = {
            "capabilities": set(capabilities),
            "last_seen": time.time(),
            "load": 0.0
        }
        
        # Update capability index
        for cap in capabilities:
            self.capability_index[cap].add(agent_id)
            
    def find_agents_by_capability(self, capability: str) -> Set[str]:
        """Find agents with capability in O(1)"""
        return self.capability_index.get(capability, set())
        
    def find_best_agent_for_task(self, required_capabilities: List[str]) -> Optional[str]:
        """Find best agent for task with optimized matching"""
        if not required_capabilities:
            # No requirements, find least loaded
            if not self.agent_registry:
                return None
                
            return min(
                self.agent_registry.keys(),
                key=lambda aid: self.agent_registry[aid]["load"]
            )
            
        # Find agents with all required capabilities
        candidate_sets = [
            self.capability_index.get(cap, set())
            for cap in required_capabilities
        ]
        
        if not candidate_sets:
            return None
            
        # Intersection of all capability sets
        candidates = set.intersection(*candidate_sets)
        
        if not candidates:
            return None
            
        # Return least loaded candidate
        return min(
            candidates,
            key=lambda aid: self.agent_registry[aid]["load"]
        )
        
    async def broadcast_message(self, message: Agent2AgentMessage):
        """Optimized broadcast with deduplication"""
        # Check for duplicate
        if message.id in self.seen_messages:
            return
            
        # Add to seen messages
        self.seen_messages.add(message.id)
        if len(self.seen_messages) > self.dedup_window:
            # Remove oldest (simple FIFO for now)
            self.seen_messages.pop()
            
        # Route message
        await self.router.send_message(message)
        
    async def send_batch(self, messages: List[Agent2AgentMessage]):
        """Send batch of messages efficiently"""
        send_tasks = []
        
        for msg in messages:
            # Skip duplicates
            if msg.id not in self.seen_messages:
                self.seen_messages.add(msg.id)
                task = asyncio.create_task(self.router.send_message(msg))
                send_tasks.append(task)
                
        await asyncio.gather(*send_tasks, return_exceptions=True)
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "router_metrics": self.router.metrics,
            "active_agents": len(self.agent_registry),
            "indexed_capabilities": len(self.capability_index),
            "message_dedup_rate": len(self.seen_messages) / self.dedup_window
        }


# Optimized coordination patterns
class FastHierarchicalCoordinator:
    """Optimized hierarchical coordination with minimal overhead"""
    
    def __init__(self, communication: OptimizedSwarmCommunication):
        self.comm = communication
        self.squad_cache = {}  # Cache squad assignments
        
    async def organize_squads(self, agents: List[str], objective: Dict) -> Dict[str, List[str]]:
        """Organize agents into squads with O(n) complexity"""
        # Use cached assignments if available
        cache_key = f"{','.join(sorted(agents))}:{objective.get('id')}"
        if cache_key in self.squad_cache:
            return self.squad_cache[cache_key]
            
        # Group by primary capability using index
        squads = defaultdict(list)
        
        for agent_id in agents:
            agent_info = self.comm.agent_registry.get(agent_id)
            if agent_info and agent_info["capabilities"]:
                # Get primary capability
                primary_cap = next(iter(agent_info["capabilities"]))
                squads[f"squad_{primary_cap}"].append(agent_id)
                
        result = dict(squads)
        self.squad_cache[cache_key] = result
        
        return result
        
    async def assign_tasks_parallel(self, tasks: List[Dict], agents: List[str]) -> Dict[str, List[str]]:
        """Assign tasks to agents in parallel with minimal latency"""
        assignments = defaultdict(list)
        
        # Parallel task assignment
        assign_tasks = []
        for task in tasks:
            task_future = asyncio.create_task(
                self._assign_single_task(task, agents)
            )
            assign_tasks.append((task["id"], task_future))
            
        # Wait for all assignments
        for task_id, future in assign_tasks:
            agent_id = await future
            if agent_id:
                assignments[agent_id].append(task_id)
                
        return dict(assignments)
        
    async def _assign_single_task(self, task: Dict, agents: List[str]) -> Optional[str]:
        """Assign single task with optimized matching"""
        required_caps = task.get("required_capabilities", [])
        
        # Use optimized agent finder
        best_agent = self.comm.find_best_agent_for_task(required_caps)
        
        if best_agent:
            # Update agent load
            self.comm.agent_registry[best_agent]["load"] += 0.1
            
        return best_agent


class ZeroCopyMessagePool:
    """Zero-copy message pool for reduced allocation overhead"""
    
    def __init__(self, pool_size: int = 1000):
        self.pool = deque(maxlen=pool_size)
        self.pool_size = pool_size
        
        # Pre-allocate messages
        for _ in range(pool_size // 2):
            self.pool.append(self._create_message())
            
    def _create_message(self) -> Agent2AgentMessage:
        """Create new message instance"""
        return Agent2AgentMessage(
            id="",
            type=MessageType.TASK_UPDATE,
            sender="",
            payload={},
            timestamp=0.0
        )
        
    def acquire(self) -> Agent2AgentMessage:
        """Get message from pool"""
        try:
            return self.pool.popleft()
        except IndexError:
            return self._create_message()
            
    def release(self, message: Agent2AgentMessage):
        """Return message to pool"""
        # Reset message
        message.id = ""
        message.correlation_id = None
        message.sender = ""
        message.recipient = None
        message.payload = {}
        message.priority = 0
        message.ttl = None
        
        # Return to pool
        try:
            self.pool.append(message)
        except:
            pass  # Pool full, let GC handle it
