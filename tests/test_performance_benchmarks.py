"""
Performance benchmark tests for Nexus Forge components

Tests performance characteristics of the enhanced Starri orchestrator,
MCP tool integrations, and overall system scalability.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock
import statistics
from concurrent.futures import ThreadPoolExecutor

from nexus_forge.agents.starri.orchestrator import ThinkingMode

pytestmark = [pytest.mark.asyncio, pytest.mark.performance]


class TestThinkingPerformance:
    """Performance tests for deep thinking capabilities"""
    
    async def test_thinking_latency_benchmark(self, mock_enhanced_starri_orchestrator, performance_thresholds):
        """Benchmark thinking operation latency"""
        orchestrator = mock_enhanced_starri_orchestrator
        
        # Mock fast responses
        orchestrator.gemini_client.generate_content = AsyncMock(
            return_value={
                "content": "Quick analysis with high confidence: 0.95",
                "usage_metadata": {"total_token_count": 50}
            }
        )
        
        latencies = []
        
        # Benchmark 20 thinking operations
        for i in range(20):
            start_time = time.time()
            
            await orchestrator.think_deeply(
                prompt=f"Optimization problem {i}",
                mode=ThinkingMode.QUICK_DECISION,
                max_thinking_steps=2
            )
            
            latency = time.time() - start_time
            latencies.append(latency)
        
        # Performance assertions
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        max_latency = max(latencies)
        
        assert avg_latency < performance_thresholds["thinking_time"]["max_per_step"]
        assert p95_latency < performance_thresholds["thinking_time"]["max_per_step"] * 1.5
        assert max_latency < performance_thresholds["thinking_time"]["max_total"]
        
        print(f"\nThinking Performance Metrics:")
        print(f"  Average latency: {avg_latency:.3f}s")
        print(f"  95th percentile: {p95_latency:.3f}s")
        print(f"  Max latency: {max_latency:.3f}s")
    
    async def test_thinking_throughput_benchmark(self, mock_enhanced_starri_orchestrator):
        """Benchmark thinking operation throughput"""
        orchestrator = mock_enhanced_starri_orchestrator
        
        # Mock fast responses
        orchestrator.gemini_client.generate_content = AsyncMock(
            return_value={
                "content": "Fast response",
                "usage_metadata": {"total_token_count": 25}
            }
        )
        
        start_time = time.time()
        
        # Run 50 concurrent thinking operations
        tasks = []
        for i in range(50):
            task = orchestrator.think_deeply(
                prompt=f"Problem {i}",
                mode=ThinkingMode.QUICK_DECISION,
                max_thinking_steps=1
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        throughput = 50 / total_time  # operations per second
        
        # Should achieve at least 10 operations per second
        assert throughput >= 10.0
        
        print(f"\nThinking Throughput: {throughput:.2f} ops/sec")
    
    async def test_thinking_memory_efficiency(self, mock_enhanced_starri_orchestrator):
        """Test memory efficiency during thinking operations"""
        import psutil
        import os
        
        orchestrator = mock_enhanced_starri_orchestrator
        
        # Mock responses
        orchestrator.gemini_client.generate_content = AsyncMock(
            return_value={
                "content": "Memory efficient response",
                "usage_metadata": {"total_token_count": 100}
            }
        )
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run many thinking operations
        tasks = []
        for i in range(100):
            task = orchestrator.think_deeply(
                prompt=f"Large problem {i}" * 10,  # Larger prompts
                mode=ThinkingMode.DEEP_ANALYSIS,
                max_thinking_steps=3
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB)
        assert memory_increase < 100 * 1024 * 1024  # 100MB
        
        print(f"\nMemory increase: {memory_increase / 1024 / 1024:.2f} MB")


class TestCoordinationPerformance:
    """Performance tests for agent coordination"""
    
    async def test_agent_registration_performance(self, mock_enhanced_starri_orchestrator, sample_agent_capabilities):
        """Benchmark agent registration performance"""
        orchestrator = mock_enhanced_starri_orchestrator
        
        start_time = time.time()
        
        # Register 50 agents
        registration_tasks = []
        for i in range(50):
            task = orchestrator.register_agent(
                agent_id=f"perf_agent_{i:03d}",
                agent_type="test_agent",
                capabilities=list(sample_agent_capabilities["fullstack_developer"]),
                configuration={"model": "test", "timeout": 30}
            )
            registration_tasks.append(task)
        
        await asyncio.gather(*registration_tasks)
        
        registration_time = time.time() - start_time
        
        # Should register 50 agents in under 2 seconds
        assert registration_time < 2.0
        
        print(f"\nAgent Registration Performance:")
        print(f"  50 agents registered in: {registration_time:.3f}s")
        print(f"  Rate: {50 / registration_time:.1f} agents/sec")
    
    async def test_task_decomposition_performance(self, mock_enhanced_starri_orchestrator, performance_thresholds):
        """Benchmark task decomposition performance"""
        orchestrator = mock_enhanced_starri_orchestrator
        
        decomposition_times = []
        
        # Test decomposition for various complexity levels
        test_cases = [
            ("Simple web app", ["basic CRUD", "authentication"]),
            ("Complex dashboard", ["real-time data", "analytics", "reporting", "user management"]),
            ("Enterprise system", ["microservices", "API gateway", "database cluster", "monitoring", "CI/CD"])
        ]
        
        for description, requirements in test_cases:
            start_time = time.time()
            
            await orchestrator.decompose_task(
                task_description=description,
                requirements=requirements
            )
            
            decomposition_time = time.time() - start_time
            decomposition_times.append(decomposition_time)
        
        # All decompositions should be fast
        max_decomposition_time = max(decomposition_times)
        avg_decomposition_time = statistics.mean(decomposition_times)
        
        assert max_decomposition_time < performance_thresholds["task_decomposition"]["max_time"]
        assert avg_decomposition_time < performance_thresholds["task_decomposition"]["max_time"] / 2
        
        print(f"\nTask Decomposition Performance:")
        print(f"  Average time: {avg_decomposition_time:.3f}s")
        print(f"  Max time: {max_decomposition_time:.3f}s")
    
    async def test_workflow_coordination_performance(self, mock_enhanced_starri_orchestrator, sample_workflow_decomposition):
        """Benchmark workflow coordination performance"""
        orchestrator = mock_enhanced_starri_orchestrator
        
        # Setup test workflow
        workflow_id = sample_workflow_decomposition["workflow_id"]
        orchestrator.active_workflows[workflow_id] = sample_workflow_decomposition
        
        start_time = time.time()
        
        result = await orchestrator.coordinate_agents(
            workflow_id=workflow_id,
            execution_mode="parallel"
        )
        
        coordination_time = time.time() - start_time
        
        # Coordination should be fast
        assert coordination_time < 5.0  # Under 5 seconds for setup
        assert result["status"] == "completed"
        
        print(f"\nWorkflow Coordination Performance:")
        print(f"  Coordination time: {coordination_time:.3f}s")
    
    async def test_concurrent_workflow_performance(self, mock_enhanced_starri_orchestrator):
        """Test performance with multiple concurrent workflows"""
        orchestrator = mock_enhanced_starri_orchestrator
        
        start_time = time.time()
        
        # Create 10 concurrent workflows
        workflow_tasks = []
        for i in range(10):
            async def create_workflow(index=i):
                decomposition = await orchestrator.decompose_task(
                    task_description=f"Web app {index}",
                    requirements=[f"Feature {index}"]
                )
                
                execution = await orchestrator.coordinate_agents(
                    workflow_id=decomposition["workflow_id"]
                )
                
                return execution["status"]
            
            workflow_tasks.append(asyncio.create_task(create_workflow()))
        
        results = await asyncio.gather(*workflow_tasks)
        
        total_time = time.time() - start_time
        
        # All workflows should complete successfully
        assert all(status == "completed" for status in results)
        
        # Should handle 10 concurrent workflows efficiently
        assert total_time < 30.0  # Under 30 seconds total
        
        print(f"\nConcurrent Workflow Performance:")
        print(f"  10 workflows completed in: {total_time:.3f}s")
        print(f"  Average per workflow: {total_time / 10:.3f}s")


class TestCachePerformance:
    """Performance tests for multi-level caching"""
    
    async def test_l1_cache_performance(self, mock_redis_cache):
        """Benchmark L1 (in-memory) cache performance"""
        cache = mock_redis_cache
        
        # Test data
        test_data = {"large_response": "x" * 1000}  # 1KB
        
        # Benchmark sets
        start_time = time.time()
        for i in range(1000):
            cache.set_l1(f"l1_key_{i}", test_data)
        set_time = time.time() - start_time
        
        # Benchmark gets
        start_time = time.time()
        for i in range(1000):
            cache.get_l1(f"l1_key_{i}")
        get_time = time.time() - start_time
        
        # L1 should be very fast
        assert set_time < 0.1  # 1000 sets in <100ms
        assert get_time < 0.05  # 1000 gets in <50ms
        
        print(f"\nL1 Cache Performance:")
        print(f"  1000 sets: {set_time:.3f}s ({1000/set_time:.0f} ops/sec)")
        print(f"  1000 gets: {get_time:.3f}s ({1000/get_time:.0f} ops/sec)")
    
    async def test_cache_strategy_performance(self, mock_redis_cache):
        """Benchmark different caching strategies"""
        from nexus_forge.core.cache import CacheStrategy
        
        cache = mock_redis_cache
        
        # Test data of different sizes
        small_data = {"data": "small"}
        large_data = {"data": "x" * 10000}  # 10KB
        
        strategies = [
            CacheStrategy.SIMPLE,
            CacheStrategy.COMPRESSED,
            CacheStrategy.SEMANTIC
        ]
        
        results = {}
        
        for strategy in strategies:
            # Benchmark small data
            start_time = time.time()
            for i in range(100):
                cache.set(f"small_{strategy.value}_{i}", small_data, strategy=strategy)
            small_time = time.time() - start_time
            
            # Benchmark large data
            start_time = time.time()
            for i in range(100):
                cache.set(f"large_{strategy.value}_{i}", large_data, strategy=strategy)
            large_time = time.time() - start_time
            
            results[strategy.value] = {
                "small_data_time": small_time,
                "large_data_time": large_time
            }
        
        # All strategies should be reasonably fast
        for strategy, times in results.items():
            assert times["small_data_time"] < 1.0
            assert times["large_data_time"] < 2.0
            
            print(f"\n{strategy} Strategy Performance:")
            print(f"  Small data (100 ops): {times['small_data_time']:.3f}s")
            print(f"  Large data (100 ops): {times['large_data_time']:.3f}s")
    
    async def test_cache_hit_rate_performance(self, mock_redis_cache):
        """Test cache performance under different hit rates"""
        cache = mock_redis_cache
        
        # Configure mock to simulate different hit rates
        hit_count = 0
        total_requests = 0
        
        def mock_get(key):
            nonlocal hit_count, total_requests
            total_requests += 1
            # Simulate 80% hit rate
            if hit_count / total_requests < 0.8:
                hit_count += 1
                return b'{"cached": "data"}'
            return None
        
        cache.client.get = mock_get
        
        start_time = time.time()
        
        # Simulate 1000 cache requests
        for i in range(1000):
            cache.get(f"key_{i % 200}")  # Some key repetition for hits
        
        request_time = time.time() - start_time
        actual_hit_rate = hit_count / total_requests
        
        # Should handle 1000 requests quickly
        assert request_time < 2.0
        assert actual_hit_rate >= 0.75  # At least 75% hit rate
        
        print(f"\nCache Hit Rate Performance:")
        print(f"  1000 requests in: {request_time:.3f}s")
        print(f"  Hit rate: {actual_hit_rate:.1%}")
        print(f"  Throughput: {1000/request_time:.0f} req/sec")


class TestSystemScalability:
    """System-wide scalability tests"""
    
    async def test_memory_usage_under_load(self, mock_enhanced_starri_orchestrator):
        """Test memory usage under high load"""
        import psutil
        import os
        
        orchestrator = mock_enhanced_starri_orchestrator
        process = psutil.Process(os.getpid())
        
        initial_memory = process.memory_info().rss
        
        # Simulate high load with many concurrent operations
        tasks = []
        
        # 50 thinking operations
        for i in range(50):
            task = orchestrator.think_deeply(
                prompt=f"Complex problem {i}" * 20,  # Large prompts
                mode=ThinkingMode.DEEP_ANALYSIS,
                max_thinking_steps=3
            )
            tasks.append(task)
        
        # 20 task decompositions
        for i in range(20):
            task = orchestrator.decompose_task(
                task_description=f"Complex system {i}",
                requirements=[f"req_{j}" for j in range(10)]
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 200 * 1024 * 1024  # <200MB increase
        
        print(f"\nMemory Usage Under Load:")
        print(f"  Initial memory: {initial_memory / 1024 / 1024:.1f} MB")
        print(f"  Final memory: {final_memory / 1024 / 1024:.1f} MB")
        print(f"  Increase: {memory_increase / 1024 / 1024:.1f} MB")
    
    async def test_response_time_under_load(self, mock_enhanced_starri_orchestrator):
        """Test response time degradation under load"""
        orchestrator = mock_enhanced_starri_orchestrator
        
        # Baseline: measure response time with low load
        start_time = time.time()
        await orchestrator.think_deeply("Simple problem", ThinkingMode.QUICK_DECISION)
        baseline_time = time.time() - start_time
        
        # High load: measure response time with concurrent operations
        load_tasks = []
        for i in range(20):  # Background load
            task = orchestrator.think_deeply(
                f"Background problem {i}",
                ThinkingMode.DEEP_ANALYSIS
            )
            load_tasks.append(task)
        
        # Measure response time under load
        start_time = time.time()
        target_task = asyncio.create_task(
            orchestrator.think_deeply("Target problem", ThinkingMode.QUICK_DECISION)
        )
        
        # Let some background load start
        await asyncio.sleep(0.1)
        
        await target_task
        loaded_time = time.time() - start_time
        
        # Cleanup background tasks
        await asyncio.gather(*load_tasks)
        
        # Response time should not degrade significantly
        degradation_ratio = loaded_time / baseline_time
        assert degradation_ratio < 3.0  # Less than 3x slower under load
        
        print(f"\nResponse Time Under Load:")
        print(f"  Baseline: {baseline_time:.3f}s")
        print(f"  Under load: {loaded_time:.3f}s")
        print(f"  Degradation: {degradation_ratio:.1f}x")
    
    async def test_error_rate_under_stress(self, mock_enhanced_starri_orchestrator):
        """Test error rate under stress conditions"""
        orchestrator = mock_enhanced_starri_orchestrator
        
        # Configure some operations to fail randomly
        original_think = orchestrator.think_deeply
        failure_count = 0
        
        async def sometimes_failing_think(*args, **kwargs):
            nonlocal failure_count
            # Simulate 5% failure rate
            if failure_count % 20 == 19:  # Every 20th call fails
                failure_count += 1
                raise Exception("Simulated failure")
            failure_count += 1
            return await original_think(*args, **kwargs)
        
        orchestrator.think_deeply = sometimes_failing_think
        
        # Run many operations and count failures
        tasks = []
        for i in range(100):
            task = asyncio.create_task(
                orchestrator.think_deeply(f"Problem {i}", ThinkingMode.QUICK_DECISION)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes and failures
        successes = sum(1 for r in results if not isinstance(r, Exception))
        failures = sum(1 for r in results if isinstance(r, Exception))
        error_rate = failures / len(results)
        
        # Error rate should match expected rate (5%)
        assert 0.03 <= error_rate <= 0.07  # 3-7% acceptable range
        assert successes >= 93  # At least 93% success rate
        
        print(f"\nError Rate Under Stress:")
        print(f"  Total operations: {len(results)}")
        print(f"  Successes: {successes}")
        print(f"  Failures: {failures}")
        print(f"  Error rate: {error_rate:.1%}")


class TestResourceUtilization:
    """Resource utilization benchmarks"""
    
    async def test_cpu_utilization(self, mock_enhanced_starri_orchestrator):
        """Monitor CPU utilization during operations"""
        import psutil
        import threading
        
        orchestrator = mock_enhanced_starri_orchestrator
        
        # Monitor CPU usage in background
        cpu_readings = []
        monitoring = True
        
        def monitor_cpu():
            while monitoring:
                cpu_readings.append(psutil.cpu_percent(interval=0.1))
        
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Perform CPU-intensive operations
        tasks = []
        for i in range(30):
            task = orchestrator.think_deeply(
                f"CPU intensive problem {i}",
                ThinkingMode.DEEP_ANALYSIS,
                max_thinking_steps=5
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Stop monitoring
        monitoring = False
        monitor_thread.join()
        
        # Analyze CPU usage
        avg_cpu = statistics.mean(cpu_readings) if cpu_readings else 0
        max_cpu = max(cpu_readings) if cpu_readings else 0
        
        # CPU usage should be reasonable
        assert max_cpu < 90.0  # Should not max out CPU
        
        print(f"\nCPU Utilization:")
        print(f"  Average CPU: {avg_cpu:.1f}%")
        print(f"  Peak CPU: {max_cpu:.1f}%")
    
    async def test_network_efficiency(self, mock_enhanced_starri_orchestrator):
        """Test network request efficiency"""
        orchestrator = mock_enhanced_starri_orchestrator
        
        # Count API calls
        api_call_count = 0
        
        async def counting_generate_content(*args, **kwargs):
            nonlocal api_call_count
            api_call_count += 1
            return {
                "content": "Generated content",
                "usage_metadata": {"total_token_count": 100}
            }
        
        orchestrator.gemini_client.generate_content = counting_generate_content
        
        # Perform operations that should reuse results
        start_time = time.time()
        
        # Same prompt multiple times (should benefit from caching)
        for i in range(10):
            await orchestrator.think_deeply(
                "Repeated optimization problem",
                ThinkingMode.QUICK_DECISION
            )
        
        operation_time = time.time() - start_time
        
        # Should be efficient with API calls
        api_calls_per_operation = api_call_count / 10
        
        # With good caching, should reduce API calls
        assert api_calls_per_operation <= 3.0  # Max 3 calls per operation
        
        print(f"\nNetwork Efficiency:")
        print(f"  Total API calls: {api_call_count}")
        print(f"  API calls per operation: {api_calls_per_operation:.1f}")
        print(f"  Total time: {operation_time:.3f}s")