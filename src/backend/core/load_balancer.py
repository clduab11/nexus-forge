"""
Intelligent Load Balancing System
Advanced load balancing with predictive scaling and multi-region support
"""

import asyncio
import json
import logging
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategy types"""

    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    HEALTH_BASED = "health_based"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"


@dataclass
class ServerNode:
    """Server node configuration"""

    id: str
    host: str
    port: int
    weight: int = 1
    max_connections: int = 1000
    current_connections: int = 0
    health_score: float = 1.0
    avg_response_time: float = 0.0
    region: str = "default"
    capacity_utilization: float = 0.0
    last_health_check: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_healthy(self) -> bool:
        """Check if server is healthy"""
        return (
            self.health_score > 0.5
            and self.capacity_utilization < 0.9
            and (datetime.utcnow() - self.last_health_check).seconds < 30
        )

    @property
    def load_score(self) -> float:
        """Calculate overall load score (lower is better)"""
        connection_ratio = self.current_connections / self.max_connections
        return (
            connection_ratio * 0.4
            + self.capacity_utilization * 0.3
            + (1 - self.health_score) * 0.2
            + min(self.avg_response_time / 1000, 1.0) * 0.1
        )


@dataclass
class LoadBalancingMetrics:
    """Load balancing performance metrics"""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    active_connections: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class AdvancedLoadBalancer:
    """Advanced load balancer with predictive capabilities"""

    def __init__(
        self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    ):
        self.strategy = strategy
        self.server_nodes: Dict[str, ServerNode] = {}
        self.metrics_history: List[LoadBalancingMetrics] = []
        self.request_history: List[Dict[str, Any]] = []
        self.health_check_interval = 10  # seconds
        self.prediction_window = 300  # 5 minutes
        self.last_health_check = datetime.utcnow()

        # Round robin counter
        self._round_robin_index = 0

        # Predictive scaling parameters
        self.auto_scaling_enabled = True
        self.min_nodes = 2
        self.max_nodes = 10
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3

    def add_server_node(self, node: ServerNode):
        """Add a server node to the load balancer"""
        self.server_nodes[node.id] = node
        logger.info(f"Added server node {node.id} at {node.host}:{node.port}")

    def remove_server_node(self, node_id: str):
        """Remove a server node from the load balancer"""
        if node_id in self.server_nodes:
            del self.server_nodes[node_id]
            logger.info(f"Removed server node {node_id}")

    async def select_server(
        self, request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServerNode]:
        """Select the best server based on the current strategy"""
        # Filter healthy servers
        healthy_servers = [
            node for node in self.server_nodes.values() if node.is_healthy
        ]

        if not healthy_servers:
            logger.warning("No healthy servers available")
            return None

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.HEALTH_BASED:
            return self._health_based_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.PREDICTIVE:
            return await self._predictive_selection(healthy_servers, request_context)
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            return await self._adaptive_selection(healthy_servers, request_context)
        else:
            return self._round_robin_selection(healthy_servers)

    def _round_robin_selection(self, servers: List[ServerNode]) -> ServerNode:
        """Simple round robin selection"""
        if not servers:
            return None

        selected_server = servers[self._round_robin_index % len(servers)]
        self._round_robin_index += 1
        return selected_server

    def _weighted_round_robin_selection(self, servers: List[ServerNode]) -> ServerNode:
        """Weighted round robin selection based on server weights"""
        if not servers:
            return None

        # Create weighted list
        weighted_servers = []
        for server in servers:
            weighted_servers.extend([server] * server.weight)

        if not weighted_servers:
            return servers[0]

        selected_server = weighted_servers[
            self._round_robin_index % len(weighted_servers)
        ]
        self._round_robin_index += 1
        return selected_server

    def _least_connections_selection(self, servers: List[ServerNode]) -> ServerNode:
        """Select server with least active connections"""
        return min(servers, key=lambda s: s.current_connections)

    def _least_response_time_selection(self, servers: List[ServerNode]) -> ServerNode:
        """Select server with lowest average response time"""
        return min(servers, key=lambda s: s.avg_response_time)

    def _health_based_selection(self, servers: List[ServerNode]) -> ServerNode:
        """Select server based on overall health score"""
        return max(servers, key=lambda s: s.health_score)

    async def _predictive_selection(
        self, servers: List[ServerNode], request_context: Optional[Dict[str, Any]]
    ) -> ServerNode:
        """Predictive server selection based on historical data and ML"""
        # Analyze historical patterns
        predicted_loads = await self._predict_server_loads(servers)

        # Select server with lowest predicted load
        best_server = min(servers, key=lambda s: predicted_loads.get(s.id, 1.0))
        return best_server

    async def _adaptive_selection(
        self, servers: List[ServerNode], request_context: Optional[Dict[str, Any]]
    ) -> ServerNode:
        """Adaptive selection that combines multiple strategies"""
        # Calculate scores for different strategies
        scores = {}

        for server in servers:
            # Connection load score (0-1, lower is better)
            connection_score = server.current_connections / server.max_connections

            # Response time score (0-1, lower is better)
            response_time_score = min(server.avg_response_time / 1000, 1.0)

            # Health score (0-1, higher is better, so invert)
            health_score = 1 - server.health_score

            # Load score (0-1, lower is better)
            load_score = server.load_score

            # Combined adaptive score
            scores[server.id] = (
                connection_score * 0.3
                + response_time_score * 0.3
                + health_score * 0.2
                + load_score * 0.2
            )

        # Select server with lowest score
        best_server_id = min(scores.keys(), key=lambda k: scores[k])
        return self.server_nodes[best_server_id]

    async def _predict_server_loads(
        self, servers: List[ServerNode]
    ) -> Dict[str, float]:
        """Predict future server loads using historical data"""
        predictions = {}

        for server in servers:
            # Simple prediction based on current load and trend
            current_load = server.load_score

            # Analyze recent trend from metrics
            recent_metrics = (
                self.metrics_history[-10:]
                if len(self.metrics_history) >= 10
                else self.metrics_history
            )

            if len(recent_metrics) >= 2:
                # Calculate load trend
                load_trend = (
                    recent_metrics[-1].requests_per_second
                    - recent_metrics[0].requests_per_second
                ) / len(recent_metrics)
                predicted_load = current_load + (load_trend * 0.1)  # Scale trend impact
            else:
                predicted_load = current_load

            predictions[server.id] = max(0.0, min(1.0, predicted_load))

        return predictions

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming request with load balancing"""
        start_time = time.time()

        try:
            # Select best server
            selected_server = await self.select_server(request)

            if not selected_server:
                return {
                    "status": "error",
                    "error": "No healthy servers available",
                    "response_time_ms": (time.time() - start_time) * 1000,
                }

            # Update server connection count
            selected_server.current_connections += 1

            try:
                # Simulate request processing
                response = await self._process_request_on_server(
                    selected_server, request
                )

                # Update server metrics
                response_time = (time.time() - start_time) * 1000
                await self._update_server_metrics(selected_server, response_time, True)

                # Record request
                self._record_request(request, selected_server, response_time, True)

                return {
                    "status": "success",
                    "server_id": selected_server.id,
                    "response": response,
                    "response_time_ms": response_time,
                }

            finally:
                # Always decrement connection count
                selected_server.current_connections = max(
                    0, selected_server.current_connections - 1
                )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Request handling failed: {e}")

            if "selected_server" in locals():
                await self._update_server_metrics(selected_server, response_time, False)
                self._record_request(request, selected_server, response_time, False)

            return {
                "status": "error",
                "error": str(e),
                "response_time_ms": response_time,
            }

    async def _process_request_on_server(
        self, server: ServerNode, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate processing request on selected server"""
        # Simulate variable processing time based on server load
        base_delay = 0.05  # 50ms base
        load_factor = server.capacity_utilization
        processing_delay = base_delay + (load_factor * 0.1)

        await asyncio.sleep(processing_delay)

        # Simulate occasional failures based on server health
        if random.random() > server.health_score:
            raise Exception(f"Server {server.id} processing failed")

        return {
            "processed_by": server.id,
            "processing_time": processing_delay,
            "server_load": server.capacity_utilization,
        }

    async def _update_server_metrics(
        self, server: ServerNode, response_time: float, success: bool
    ):
        """Update server performance metrics"""
        # Update average response time with exponential moving average
        alpha = 0.1  # Smoothing factor
        server.avg_response_time = (
            1 - alpha
        ) * server.avg_response_time + alpha * response_time

        # Update capacity utilization
        connection_ratio = server.current_connections / server.max_connections
        server.capacity_utilization = max(
            connection_ratio, server.capacity_utilization * 0.95
        )  # Decay factor

        # Update health score based on success rate
        if success:
            server.health_score = min(1.0, server.health_score + 0.01)
        else:
            server.health_score = max(0.0, server.health_score - 0.05)

    def _record_request(
        self,
        request: Dict[str, Any],
        server: ServerNode,
        response_time: float,
        success: bool,
    ):
        """Record request for analytics and optimization"""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "server_id": server.id,
            "response_time_ms": response_time,
            "success": success,
            "request_type": request.get("type", "unknown"),
            "server_load": server.load_score,
        }

        self.request_history.append(record)

        # Keep only recent requests (last 1000)
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]

    async def run_health_checks(self):
        """Run health checks on all server nodes"""
        if (
            datetime.utcnow() - self.last_health_check
        ).seconds < self.health_check_interval:
            return

        logger.info("Running health checks on all servers")

        for server in self.server_nodes.values():
            try:
                # Simulate health check
                health_result = await self._check_server_health(server)
                server.health_score = health_result["health_score"]
                server.last_health_check = datetime.utcnow()

            except Exception as e:
                logger.warning(f"Health check failed for server {server.id}: {e}")
                server.health_score = max(0.0, server.health_score - 0.1)

        self.last_health_check = datetime.utcnow()

        # Auto-scaling check
        if self.auto_scaling_enabled:
            await self._check_auto_scaling()

    async def _check_server_health(self, server: ServerNode) -> Dict[str, Any]:
        """Perform health check on a server"""
        # Simulate health check with variable results
        await asyncio.sleep(0.01)  # Simulate network latency

        # Simulate health based on current load
        base_health = 0.9
        load_penalty = server.capacity_utilization * 0.3
        health_score = max(0.1, base_health - load_penalty + random.uniform(-0.1, 0.1))

        return {
            "health_score": health_score,
            "response_time_ms": random.uniform(5, 50),
            "status": "healthy" if health_score > 0.5 else "degraded",
        }

    async def _check_auto_scaling(self):
        """Check if auto-scaling is needed"""
        if not self.server_nodes:
            return

        # Calculate average load across all servers
        total_load = sum(
            server.capacity_utilization for server in self.server_nodes.values()
        )
        avg_load = total_load / len(self.server_nodes)

        current_server_count = len(self.server_nodes)

        # Scale up if needed
        if avg_load > self.scale_up_threshold and current_server_count < self.max_nodes:
            await self._scale_up()

        # Scale down if needed
        elif (
            avg_load < self.scale_down_threshold
            and current_server_count > self.min_nodes
        ):
            await self._scale_down()

    async def _scale_up(self):
        """Add a new server node"""
        new_node_id = f"auto_server_{len(self.server_nodes) + 1}"
        new_node = ServerNode(
            id=new_node_id,
            host=f"10.0.0.{100 + len(self.server_nodes)}",
            port=8000,
            weight=1,
            max_connections=1000,
            region="auto_scaled",
        )

        self.add_server_node(new_node)
        logger.info(f"Auto-scaled up: Added server {new_node_id}")

    async def _scale_down(self):
        """Remove the least utilized server node"""
        if len(self.server_nodes) <= self.min_nodes:
            return

        # Find server with lowest utilization
        least_utilized = min(
            self.server_nodes.values(), key=lambda s: s.capacity_utilization
        )

        # Only remove if it has no active connections
        if least_utilized.current_connections == 0:
            self.remove_server_node(least_utilized.id)
            logger.info(f"Auto-scaled down: Removed server {least_utilized.id}")

    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get current load balancing statistics"""
        total_connections = sum(
            server.current_connections for server in self.server_nodes.values()
        )
        total_capacity = sum(
            server.max_connections for server in self.server_nodes.values()
        )

        avg_health = (
            statistics.mean(
                [server.health_score for server in self.server_nodes.values()]
            )
            if self.server_nodes
            else 0
        )
        avg_response_time = (
            statistics.mean(
                [server.avg_response_time for server in self.server_nodes.values()]
            )
            if self.server_nodes
            else 0
        )

        # Calculate recent success rate
        recent_requests = (
            self.request_history[-100:]
            if len(self.request_history) >= 100
            else self.request_history
        )
        success_rate = (
            sum(1 for req in recent_requests if req["success"]) / len(recent_requests)
            if recent_requests
            else 1.0
        )

        return {
            "strategy": self.strategy.value,
            "server_count": len(self.server_nodes),
            "healthy_servers": sum(
                1 for server in self.server_nodes.values() if server.is_healthy
            ),
            "total_connections": total_connections,
            "total_capacity": total_capacity,
            "capacity_utilization": (
                total_connections / total_capacity if total_capacity > 0 else 0
            ),
            "average_health_score": avg_health,
            "average_response_time_ms": avg_response_time,
            "recent_success_rate": success_rate,
            "auto_scaling_enabled": self.auto_scaling_enabled,
            "requests_processed": len(self.request_history),
        }


async def setup_demo_load_balancer() -> AdvancedLoadBalancer:
    """Set up a demo load balancer with sample servers"""
    lb = AdvancedLoadBalancer(LoadBalancingStrategy.ADAPTIVE)

    # Add sample server nodes
    servers = [
        ServerNode("server_1", "10.0.0.10", 8001, weight=2, max_connections=500),
        ServerNode("server_2", "10.0.0.11", 8002, weight=1, max_connections=300),
        ServerNode("server_3", "10.0.0.12", 8003, weight=3, max_connections=800),
        ServerNode(
            "server_4",
            "10.0.0.13",
            8004,
            weight=1,
            max_connections=400,
            region="us-east",
        ),
        ServerNode(
            "server_5",
            "10.0.0.14",
            8005,
            weight=2,
            max_connections=600,
            region="us-west",
        ),
    ]

    for server in servers:
        lb.add_server_node(server)

    logger.info(f"Demo load balancer set up with {len(servers)} servers")
    return lb


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    async def demo_load_balancer():
        """Demo the load balancer functionality"""
        lb = await setup_demo_load_balancer()

        logger.info("🚀 Starting load balancer demo")

        # Simulate load testing
        for i in range(50):
            request = {"id": f"request_{i}", "type": "api_call", "data": {"test": True}}

            result = await lb.handle_request(request)

            if i % 10 == 0:
                stats = lb.get_load_balancing_stats()
                logger.info(
                    f"Stats: {stats['healthy_servers']}/{stats['server_count']} healthy, "
                    f"{stats['capacity_utilization']:.2f} utilization, "
                    f"{stats['recent_success_rate']:.2f} success rate"
                )

            # Run health checks periodically
            if i % 20 == 0:
                await lb.run_health_checks()

            await asyncio.sleep(0.1)

        # Final stats
        final_stats = lb.get_load_balancing_stats()
        logger.info("🎉 Load balancer demo completed")
        logger.info(f"Final stats: {json.dumps(final_stats, indent=2)}")

    asyncio.run(demo_load_balancer())
