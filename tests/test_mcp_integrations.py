"""
Test suite for MCP tool integrations (Supabase, Redis, Mem0)

Tests the real-time coordination, caching, and knowledge graph functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
import json

from nexus_forge.integrations.supabase.coordination_client import SupabaseCoordinationClient
from nexus_forge.integrations.mem0.knowledge_client import Mem0KnowledgeClient
from nexus_forge.core.cache import RedisCache, CacheStrategy
from nexus_forge.core.exceptions import CoordinationError, IntegrationError

pytestmark = pytest.mark.asyncio


class TestSupabaseCoordinationClient:
    """Test suite for Supabase coordination client"""
    
    @pytest.fixture
    def coordination_client(self):
        """Create Supabase coordination client with mocked dependencies"""
        with patch('nexus_forge.integrations.supabase.coordination_client.create_client'):
            client = SupabaseCoordinationClient(
                url="https://test.supabase.co",
                key="test-key",
                project_id="test-project"
            )
            
            # Mock the Supabase client
            client.client = MagicMock()
            client.client.table.return_value.select.return_value.execute.return_value.data = []
            
            return client
    
    async def test_agent_registration(self, coordination_client):
        """Test agent registration in coordination system"""
        # Mock successful registration
        coordination_client.client.table.return_value.insert.return_value.execute.return_value.data = [
            {"agent_id": "agent_123"}
        ]
        coordination_client.update_agent_state = AsyncMock()
        
        agent_id = await coordination_client.register_agent(
            name="TestAgent",
            agent_type="code_generator",
            capabilities={"languages": ["python", "javascript"]},
            configuration={"model": "gemini-pro"}
        )
        
        assert agent_id == "agent_123"
        
        # Verify database operations
        coordination_client.client.table.assert_called_with('agents')
        coordination_client.update_agent_state.assert_called_once_with(
            "agent_123", 'idle', {}
        )
    
    async def test_workflow_creation_and_management(self, coordination_client):
        """Test workflow creation and status updates"""
        # Mock workflow creation
        coordination_client.client.table.return_value.insert.return_value.execute.return_value.data = [
            {"workflow_id": "workflow_456"}
        ]
        
        workflow_id = await coordination_client.create_workflow(
            name="Test Workflow",
            description="A test workflow",
            definition={"steps": ["step1", "step2"]},
            priority=7
        )
        
        assert workflow_id == "workflow_456"
        
        # Test status update
        coordination_client._broadcast_event = AsyncMock()
        await coordination_client.update_workflow_status(
            workflow_id,
            "running",
            metrics={"progress": 50}
        )
        
        # Verify broadcast
        coordination_client._broadcast_event.assert_called_once()
        call_args = coordination_client._broadcast_event.call_args
        assert call_args[0][0] == "workflow_progress"
        assert call_args[0][1]["workflow_id"] == workflow_id
    
    async def test_task_assignment_with_reservation(self, coordination_client):
        """Test atomic task assignment with reservation logic"""
        task_id = "task_789"
        agent_id = "agent_123"
        
        # Mock successful task reservation
        coordination_client.client.table.return_value.update.return_value.eq.return_value.eq.return_value.execute.return_value.data = [
            {"task_id": task_id, "agent_id": agent_id}
        ]
        coordination_client.update_agent_state = AsyncMock()
        coordination_client._broadcast_event = AsyncMock()
        
        result = await coordination_client.assign_task(task_id, agent_id)
        
        assert result is True
        
        # Verify agent state update
        coordination_client.update_agent_state.assert_called_once_with(
            agent_id, 'busy', {'current_task': task_id}, task_id
        )
        
        # Verify task broadcast
        coordination_client._broadcast_event.assert_called_once()
    
    async def test_task_assignment_conflict(self, coordination_client):
        """Test handling of task assignment conflicts"""
        task_id = "task_789"
        agent_id = "agent_123"
        
        # Mock failed task reservation (already assigned)
        coordination_client.client.table.return_value.update.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
        
        result = await coordination_client.assign_task(task_id, agent_id)
        
        assert result is False
    
    async def test_real_time_subscriptions(self, coordination_client):
        """Test real-time subscription setup"""
        callback = AsyncMock()
        
        # Mock channel creation
        mock_channel = MagicMock()
        coordination_client.client.channel.return_value = mock_channel
        
        await coordination_client.subscribe_to_agent_status(callback)
        
        # Verify subscription setup
        assert callback in coordination_client.subscriptions['agent_status']
        assert 'agent_status' in coordination_client.channels
        mock_channel.on.assert_called_once()
        mock_channel.subscribe.assert_called_once()
    
    async def test_performance_metrics_recording(self, coordination_client):
        """Test performance metrics recording"""
        await coordination_client.record_metric(
            metric_type="task_execution_time",
            metric_value=45.5,
            metric_unit="seconds",
            agent_id="agent_123",
            task_id="task_456",
            additional_data={"complexity": "medium"}
        )
        
        # Verify metrics table insertion
        coordination_client.client.table.assert_called_with('performance_metrics')
    
    async def test_pending_tasks_retrieval(self, coordination_client):
        """Test retrieval of pending tasks"""
        # Mock pending tasks
        coordination_client.client.table.return_value.select.return_value.eq.return_value.order.return_value.order.return_value.limit.return_value.execute.return_value.data = [
            {
                "task_id": "task_1",
                "name": "Generate UI",
                "type": "ui_generation",
                "priority": 8
            },
            {
                "task_id": "task_2", 
                "name": "Write tests",
                "type": "testing",
                "priority": 6
            }
        ]
        
        tasks = await coordination_client.get_pending_tasks(
            agent_type="code_generator",
            limit=5
        )
        
        assert len(tasks) == 2
        assert tasks[0]["task_id"] == "task_1"
        assert tasks[0]["priority"] == 8
    
    async def test_health_check(self, coordination_client):
        """Test coordination client health check"""
        # Mock healthy response
        coordination_client.client.table.return_value.select.return_value.execute.return_value.data = [{"count": 5}]
        coordination_client.connected = True
        coordination_client.channels = {"test_channel": MagicMock()}
        
        health = await coordination_client.health_check()
        
        assert health["status"] == "healthy"
        assert health["connected"] is True
        assert health["active_channels"] == 1
        assert "timestamp" in health
    
    async def test_connection_error_handling(self, coordination_client):
        """Test error handling during connection"""
        # Mock connection failure
        coordination_client.client.table.return_value.select.return_value.execute.side_effect = Exception("Connection failed")
        
        with pytest.raises(CoordinationError) as exc_info:
            await coordination_client.connect()
        
        assert "Supabase connection failed" in str(exc_info.value)


class TestMem0KnowledgeClient:
    """Test suite for Mem0 knowledge graph client"""
    
    @pytest.fixture
    def knowledge_client(self):
        """Create Mem0 knowledge client with mocked dependencies"""
        with patch('nexus_forge.integrations.mem0.knowledge_client.RedisCache'):
            client = Mem0KnowledgeClient(
                api_key="test-key",
                orchestrator_id="orch_123"
            )
            client.cache = MagicMock()
            return client
    
    async def test_entity_creation_and_caching(self, knowledge_client):
        """Test entity creation with caching"""
        entity_data = {
            "name": "TestAgent",
            "type": "AI_AGENT",
            "properties": {"capabilities": ["code_gen"]}
        }
        
        entity_id = await knowledge_client.create_entity(entity_data)
        
        # Verify entity was cached
        assert entity_id in knowledge_client.entity_cache
        cached_entity = knowledge_client.entity_cache[entity_id]
        assert cached_entity["name"] == "TestAgent"
        assert "created_at" in cached_entity
        
        # Verify Redis caching
        knowledge_client.cache.set.assert_called_once()
        call_args = knowledge_client.cache.set.call_args
        assert call_args[0][0] == f"mem0:entity:{entity_id}"
        assert call_args[1]["strategy"] == CacheStrategy.COMPRESSED
    
    async def test_relationship_creation(self, knowledge_client):
        """Test relationship creation between entities"""
        relationship_id = await knowledge_client.create_relationship(
            from_entity="agent_1",
            to_entity="capability_1", 
            relationship_type="implements",
            properties={"confidence": 0.9}
        )
        
        # Verify relationship was cached
        assert "agent_1" in knowledge_client.relationship_cache
        relationships = knowledge_client.relationship_cache["agent_1"]
        assert len(relationships) == 1
        assert relationships[0]["to"] == "capability_1"
        assert relationships[0]["type"] == "IMPLEMENTS"
        
        # Verify Redis caching
        knowledge_client.cache.set.assert_called_once()
    
    async def test_agent_entity_creation_with_capabilities(self, knowledge_client):
        """Test agent entity creation with capability relationships"""
        knowledge_client.create_entity = AsyncMock(side_effect=["agent_123", "cap_1", "cap_2"])
        knowledge_client.create_relationship = AsyncMock(return_value="rel_123")
        
        entity_id = await knowledge_client.add_agent_entity(
            agent_id="agent_123",
            agent_type="full_stack",
            capabilities=["ui_design", "backend_dev"]
        )
        
        assert entity_id == "agent_123"
        
        # Verify entity creation calls
        assert knowledge_client.create_entity.call_count == 3  # Agent + 2 capabilities
        
        # Verify relationship creation calls
        assert knowledge_client.create_relationship.call_count == 3  # Orchestrator->Agent + Agent->2 Capabilities
    
    async def test_pattern_search_with_similarity(self, knowledge_client):
        """Test pattern search with similarity scoring"""
        # Setup pattern library
        knowledge_client.pattern_library = {
            "task_decomposition": [
                {
                    "id": "pattern_1",
                    "pattern": {
                        "properties": {"confidence": 0.9, "pattern_type": "decomposition"}
                    },
                    "embedding": [0.1, 0.2, 0.3, 0.4]
                },
                {
                    "id": "pattern_2",
                    "pattern": {
                        "properties": {"confidence": 0.7, "pattern_type": "decomposition"}
                    },
                    "embedding": [0.8, 0.7, 0.6, 0.5]
                }
            ]
        }
        
        # Mock embedding generation
        knowledge_client._generate_pattern_embedding = MagicMock(
            return_value=[0.2, 0.3, 0.4, 0.5]
        )
        
        results = await knowledge_client.search_patterns(
            query="task decomposition strategy",
            pattern_type="task_decomposition",
            min_confidence=0.5,
            limit=5
        )
        
        assert len(results) == 2
        # Should be ordered by similarity (pattern_1 should be more similar)
        assert results[0]["properties"]["confidence"] >= 0.7
    
    async def test_thinking_pattern_storage(self, knowledge_client):
        """Test storage of thinking patterns"""
        knowledge_client.create_entity = AsyncMock(return_value="pattern_456")
        knowledge_client.create_relationship = AsyncMock(return_value="rel_789")
        
        pattern_content = {
            "thinking_chain": [
                {"step": 1, "thought": "Initial analysis"},
                {"step": 2, "thought": "Deeper insight"}
            ],
            "conclusion": "Solution found",
            "confidence": 0.85
        }
        
        pattern_id = await knowledge_client.add_thinking_pattern(
            pattern_type="deep_analysis",
            pattern_content=pattern_content,
            confidence=0.85
        )
        
        assert pattern_id == "pattern_456"
        
        # Verify pattern was added to library
        assert "deep_analysis" in knowledge_client.pattern_library
        
        # Verify relationship creation (orchestrator learns from pattern)
        knowledge_client.create_relationship.assert_called_once()
        call_args = knowledge_client.create_relationship.call_args
        assert call_args[0][2] == "learns_from"
    
    async def test_agent_performance_metrics_tracking(self, knowledge_client):
        """Test tracking of agent performance metrics"""
        agent_id = "agent_123"
        
        # Mock agent entity
        knowledge_client.entity_cache[agent_id] = {
            "id": agent_id,
            "properties": {
                "performance_metrics": {
                    "tasks_completed": 10,
                    "success_rate": 0.85,
                    "average_execution_time": 45.2
                }
            }
        }
        
        metrics = await knowledge_client.get_agent_performance_metrics(agent_id)
        
        assert metrics["tasks_completed"] == 10
        assert metrics["success_rate"] == 0.85
        assert metrics["average_execution_time"] == 45.2
    
    async def test_knowledge_graph_export_import(self, knowledge_client):
        """Test knowledge graph export and import"""
        # Setup test data
        knowledge_client.entity_cache = {
            "entity_1": {"id": "entity_1", "name": "Test Entity"}
        }
        knowledge_client.relationship_cache = {
            "entity_1": [{"from": "entity_1", "to": "entity_2", "type": "CONNECTS"}]
        }
        knowledge_client.pattern_library = {
            "test_patterns": [{"id": "pattern_1", "data": "test"}]
        }
        
        # Test export
        export_data = await knowledge_client.export_knowledge_graph()
        
        assert len(export_data["entities"]) == 1
        assert len(export_data["relationships"]) == 1
        assert "test_patterns" in export_data["patterns"]
        assert "export_timestamp" in export_data
        
        # Test import
        new_client = Mem0KnowledgeClient("key", "orch")
        await new_client.import_knowledge_graph(export_data)
        
        assert len(new_client.entity_cache) == 1
        assert "entity_1" in new_client.relationship_cache
        assert "test_patterns" in new_client.pattern_library


class TestRedisCacheEnhanced:
    """Test suite for enhanced Redis cache with multi-level caching"""
    
    @pytest.fixture
    def redis_cache(self):
        """Create Redis cache with mocked client"""
        with patch('nexus_forge.core.cache.redis.StrictRedis'):
            cache = RedisCache()
            cache.client = MagicMock()
            cache.client.ping.return_value = True
            return cache
    
    def test_l1_cache_operations(self, redis_cache):
        """Test L1 (in-memory) cache operations"""
        key = "test_key"
        value = {"data": "test_value"}
        
        # Test set
        result = redis_cache.set_l1(key, value)
        assert result is True
        assert key in redis_cache.l1_cache
        
        # Test get (should hit)
        retrieved = redis_cache.get_l1(key)
        assert retrieved == value
        assert redis_cache.metrics.hits == 1
        
        # Test expiration
        import time
        redis_cache.l1_cache[key]["timestamp"] = time.time() - 400  # Expired
        retrieved = redis_cache.get_l1(key)
        assert retrieved is None
        assert key not in redis_cache.l1_cache  # Should be removed
    
    def test_l1_cache_eviction(self, redis_cache):
        """Test L1 cache LRU eviction"""
        redis_cache.l1_max_size = 2  # Small size for testing
        
        # Fill cache to capacity
        redis_cache.set_l1("key1", "value1")
        redis_cache.set_l1("key2", "value2")
        
        # Add one more (should evict oldest)
        redis_cache.set_l1("key3", "value3")
        
        assert len(redis_cache.l1_cache) == 2
        assert "key1" not in redis_cache.l1_cache  # Should be evicted
        assert "key2" in redis_cache.l1_cache
        assert "key3" in redis_cache.l1_cache
    
    def test_l2_cache_operations(self, redis_cache):
        """Test L2 (Redis session state) cache operations"""
        key = "session_key"
        value = {"session_data": "test"}
        
        # Mock Redis operations
        redis_cache.client.set.return_value = True
        redis_cache.client.get.return_value = json.dumps(value).encode('utf-8')
        
        # Test set
        result = redis_cache.set_l2(key, value, timeout=1800)
        assert result is True
        
        # Verify Redis was called with L2 prefix
        redis_cache.client.set.assert_called_once()
        call_args = redis_cache.client.set.call_args
        assert call_args[0][0] == f"l2:{key}"
        
        # Test get
        retrieved = redis_cache.get_l2(key)
        redis_cache.client.get.assert_called_with(f"l2:{key}")
    
    def test_l3_cache_operations(self, redis_cache):
        """Test L3 (Redis rate limiting) cache operations"""
        key = "rate_limit_key"
        value = {"requests": 10, "window": "hour"}
        
        # Mock Redis operations
        redis_cache.client.set.return_value = True
        redis_cache.client.get.return_value = json.dumps(value).encode('utf-8')
        
        # Test set
        result = redis_cache.set_l3(key, value, timeout=86400)
        assert result is True
        
        # Verify Redis was called with L3 prefix and long TTL
        redis_cache.client.set.assert_called_once()
        call_args = redis_cache.client.set.call_args
        assert call_args[0][0] == f"l3:{key}"
        assert call_args[1]["ex"] == 86400
        
        # Test get
        retrieved = redis_cache.get_l3(key)
        assert retrieved == value
    
    def test_cache_strategy_selection(self, redis_cache):
        """Test different caching strategies"""
        # Test simple strategy
        simple_data = {"simple": "data"}
        redis_cache.set("simple_key", simple_data, strategy=CacheStrategy.SIMPLE)
        
        # Test compressed strategy
        large_data = {"large": "x" * 2000}  # Large data for compression
        redis_cache.set("compressed_key", large_data, strategy=CacheStrategy.COMPRESSED)
        
        # Test semantic strategy
        semantic_data = {"content": "AI response", "context": "test"}
        redis_cache.set("semantic_key", semantic_data, strategy=CacheStrategy.SEMANTIC)
        
        # Verify different serialization was used
        assert redis_cache.client.set.call_count == 3
    
    def test_cache_metrics_tracking(self, redis_cache):
        """Test cache metrics tracking"""
        # Reset metrics
        redis_cache.metrics.reset()
        
        # Simulate cache operations
        redis_cache.client.get.return_value = b'{"test": "data"}'
        redis_cache.get("hit_key")  # Cache hit
        
        redis_cache.client.get.return_value = None
        redis_cache.get("miss_key")  # Cache miss
        
        redis_cache.set("set_key", {"data": "test"})  # Cache set
        
        # Check metrics
        assert redis_cache.metrics.hits == 1
        assert redis_cache.metrics.misses == 1
        assert redis_cache.metrics.sets == 1
        assert redis_cache.metrics.get_hit_rate() == 50.0  # 1 hit out of 2 gets
    
    def test_cache_invalidation_strategies(self, redis_cache):
        """Test cache invalidation by patterns and tags"""
        # Mock Redis operations
        redis_cache.client.keys.return_value = [
            b"tag:ai_response:key1",
            b"tag:ai_response:key2", 
            b"tag:other:key3"
        ]
        redis_cache.client.delete.return_value = 2
        
        # Test pattern-based invalidation
        deleted = redis_cache.delete_pattern("tag:ai_response:*")
        assert deleted >= 0  # Should attempt deletion
        
        # Test tag-based invalidation
        deleted = redis_cache.invalidate_by_tag("ai_response")
        redis_cache.client.keys.assert_called()
    
    def test_cache_health_check(self, redis_cache):
        """Test cache health check functionality"""
        # Mock successful operations
        redis_cache.client.set.return_value = True
        redis_cache.client.get.return_value = b'{"timestamp": 123456, "test": true}'
        redis_cache.client.delete.return_value = 1
        
        health = redis_cache.health_check()
        
        assert health["status"] == "healthy"
        assert health["latency_ms"] > 0
        assert health["operations_successful"]["set"] is True
        assert health["operations_successful"]["get"] is True
        assert health["operations_successful"]["delete"] is True