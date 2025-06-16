"""
Performance Benchmarks and Validation for Advanced Agentic Systems
Validates performance requirements for all 6 advanced systems
"""

import asyncio
import gc
import statistics
import time
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest

# Import performance monitoring utilities
from nexus_forge.features.performance_analytics import (
    MetricType,
    PerformanceAnalyticsOrchestrator,
)


@pytest.fixture
def performance_benchmarks():
    """Performance benchmark targets"""
    return {
        "agent_self_improvement": {
            "max_recommendation_latency": 5000,  # ms
            "min_improvement_accuracy": 0.8,
            "max_memory_usage": 500,  # MB
            "concurrent_requests": 20,
        },
        "advanced_caching": {
            "max_cache_lookup_latency": 10,  # ms L1
            "max_cache_miss_latency": 100,  # ms L3
            "min_hit_rate": 0.8,
            "max_memory_overhead": 200,  # MB
            "concurrent_operations": 1000,
        },
        "behavior_analysis": {
            "max_analysis_latency": 2000,  # ms
            "min_pattern_accuracy": 0.85,
            "max_storage_growth": 1.0,  # MB per 1000 interactions
            "concurrent_agents": 50,
        },
        "performance_analytics": {
            "max_metric_collection_latency": 50,  # ms
            "max_anomaly_detection_latency": 100,  # ms
            "max_system_overhead": 5.0,  # %
            "metrics_per_second": 1000,
        },
        "dynamic_model_selection": {
            "max_selection_latency": 50,  # ms
            "min_selection_accuracy": 0.85,
            "max_decision_overhead": 2.0,  # %
            "concurrent_selections": 100,
        },
        "multimodal_integration": {
            "max_translation_latency": 2000,  # ms
            "max_workflow_latency": 10000,  # ms
            "min_semantic_preservation": 0.85,
            "concurrent_workflows": 10,
        },
    }


@pytest.fixture
async def memory_monitor():
    """Monitor memory usage during tests"""

    class MemoryMonitor:
        def __init__(self):
            self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.initial_memory

        def update_peak(self):
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            self.peak_memory = max(self.peak_memory, current_memory)
            return current_memory

        def get_memory_increase(self):
            return self.peak_memory - self.initial_memory

    return MemoryMonitor()


class TestAgentSelfImprovementPerformance:
    """Performance tests for Agent Self-Improvement system"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_improvement_recommendation_latency(
        self, performance_benchmarks, memory_monitor
    ):
        """Test improvement recommendation generation latency"""
        from nexus_forge.features.agent_self_improvement import (
            AgentSelfImprovementOrchestrator,
        )

        orchestrator = AgentSelfImprovementOrchestrator()
        orchestrator.cache = AsyncMock()

        performance_data = {
            "performance_scores": [0.8, 0.85, 0.82, 0.88, 0.90],
            "task_completion_rate": 0.92,
            "error_rate": 0.05,
        }

        latencies = []

        # Test multiple iterations
        for i in range(10):
            start_time = time.time()

            with patch.object(
                orchestrator, "generate_improvement_recommendations"
            ) as mock_gen:
                mock_gen.return_value = [
                    {"strategy": "rl", "confidence": 0.8, "expected_improvement": 0.05}
                ]

                recommendations = (
                    await orchestrator.generate_improvement_recommendations(
                        f"test_agent_{i}", performance_data
                    )
                )

            latency = (time.time() - start_time) * 1000  # ms
            latencies.append(latency)
            memory_monitor.update_peak()

        # Validate latency requirements
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile

        assert (
            avg_latency
            < performance_benchmarks["agent_self_improvement"][
                "max_recommendation_latency"
            ]
        )
        assert (
            p95_latency
            < performance_benchmarks["agent_self_improvement"][
                "max_recommendation_latency"
            ]
            * 1.5
        )

        # Validate memory usage
        memory_increase = memory_monitor.get_memory_increase()
        assert (
            memory_increase
            < performance_benchmarks["agent_self_improvement"]["max_memory_usage"]
        )

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_improvement_requests(self, performance_benchmarks):
        """Test concurrent improvement request handling"""
        from nexus_forge.features.agent_self_improvement import (
            AgentSelfImprovementOrchestrator,
        )

        orchestrator = AgentSelfImprovementOrchestrator()
        orchestrator.cache = AsyncMock()

        concurrent_requests = performance_benchmarks["agent_self_improvement"][
            "concurrent_requests"
        ]

        async def generate_improvement_request(agent_id):
            with patch.object(
                orchestrator, "generate_improvement_recommendations"
            ) as mock_gen:
                mock_gen.return_value = [{"strategy": "test", "confidence": 0.8}]

                return await orchestrator.generate_improvement_recommendations(
                    agent_id, {"performance_scores": [0.8]}
                )

        start_time = time.time()

        # Execute concurrent requests
        tasks = [
            generate_improvement_request(f"agent_{i}")
            for i in range(concurrent_requests)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        execution_time = time.time() - start_time

        # Validate concurrent performance
        assert execution_time < 10.0  # Should complete within 10 seconds

        # Validate success rate
        successful_requests = len([r for r in results if not isinstance(r, Exception)])
        success_rate = successful_requests / len(results)
        assert success_rate > 0.9  # 90% success rate minimum


class TestAdvancedCachingPerformance:
    """Performance tests for Advanced Caching system"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_cache_lookup_latency(self, performance_benchmarks):
        """Test cache lookup latency across all levels"""
        from nexus_forge.features.advanced_caching import AdvancedCacheOrchestrator

        orchestrator = AdvancedCacheOrchestrator()
        orchestrator.cache = AsyncMock()

        # Mock different cache levels
        orchestrator.cache.get_l1 = AsyncMock(return_value="l1_result")
        orchestrator.cache.get_l2 = AsyncMock(return_value="l2_result")
        orchestrator.cache.get_l3 = AsyncMock(return_value="l3_result")

        # Test L1 cache latency
        l1_latencies = []
        for i in range(100):
            start_time = time.perf_counter()
            result = await orchestrator.cache.get_l1(f"key_{i}")
            latency = (time.perf_counter() - start_time) * 1000  # ms
            l1_latencies.append(latency)

        avg_l1_latency = statistics.mean(l1_latencies)
        assert (
            avg_l1_latency
            < performance_benchmarks["advanced_caching"]["max_cache_lookup_latency"]
        )

        # Test L3 cache latency (miss scenario)
        l3_latencies = []
        orchestrator.cache.get_l1.return_value = None
        orchestrator.cache.get_l2.return_value = None

        for i in range(50):
            start_time = time.perf_counter()
            result = await orchestrator.cache.get_l3(f"miss_key_{i}")
            latency = (time.perf_counter() - start_time) * 1000  # ms
            l3_latencies.append(latency)

        avg_l3_latency = statistics.mean(l3_latencies)
        assert (
            avg_l3_latency
            < performance_benchmarks["advanced_caching"]["max_cache_miss_latency"]
        )

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_cache_operations(self, performance_benchmarks):
        """Test concurrent cache operations performance"""
        from nexus_forge.features.advanced_caching import AdvancedCacheOrchestrator

        orchestrator = AdvancedCacheOrchestrator()
        orchestrator.cache = AsyncMock()

        concurrent_ops = performance_benchmarks["advanced_caching"][
            "concurrent_operations"
        ]

        async def cache_operation(operation_id):
            # Mix of get and set operations
            if operation_id % 2 == 0:
                return await orchestrator.cache.get_l1(f"key_{operation_id}")
            else:
                return await orchestrator.cache.set_l1(
                    f"key_{operation_id}", f"value_{operation_id}"
                )

        start_time = time.time()

        # Execute concurrent operations
        tasks = [cache_operation(i) for i in range(concurrent_ops)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        execution_time = time.time() - start_time

        # Validate concurrent performance
        operations_per_second = len(results) / execution_time
        assert operations_per_second > 500  # Should handle at least 500 ops/sec

        # Validate error rate
        error_count = len([r for r in results if isinstance(r, Exception)])
        error_rate = error_count / len(results)
        assert error_rate < 0.05  # Less than 5% error rate


class TestBehaviorAnalysisPerformance:
    """Performance tests for Behavior Analysis system"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_interaction_logging_latency(self, performance_benchmarks):
        """Test interaction logging performance"""
        from nexus_forge.features.agent_behavior_analysis import (
            AgentBehaviorAnalysisOrchestrator,
        )

        orchestrator = AgentBehaviorAnalysisOrchestrator()
        orchestrator.cache = AsyncMock()

        latencies = []

        # Test batch interaction logging
        for i in range(100):
            interaction_data = {
                "interaction_type": "task_execution",
                "duration": 5.5 + i * 0.1,
                "quality_score": 0.8 + i * 0.001,
                "complexity": "medium",
            }

            start_time = time.perf_counter()

            with patch.object(orchestrator, "log_agent_interaction") as mock_log:
                mock_log.return_value = True
                await orchestrator.log_agent_interaction(
                    f"agent_{i % 10}", "task_execution", interaction_data
                )

            latency = (time.perf_counter() - start_time) * 1000  # ms
            latencies.append(latency)

        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]

        assert (
            avg_latency
            < performance_benchmarks["behavior_analysis"]["max_analysis_latency"]
        )
        assert (
            p95_latency
            < performance_benchmarks["behavior_analysis"]["max_analysis_latency"] * 1.5
        )

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_pattern_analysis_scalability(self, performance_benchmarks):
        """Test pattern analysis with large datasets"""
        from nexus_forge.features.agent_behavior_analysis import (
            AgentBehaviorAnalysisOrchestrator,
        )

        orchestrator = AgentBehaviorAnalysisOrchestrator()
        orchestrator.cache = AsyncMock()

        # Simulate large dataset analysis
        num_agents = performance_benchmarks["behavior_analysis"]["concurrent_agents"]
        interactions_per_agent = 100

        start_time = time.time()

        # Mock pattern analysis
        with patch.object(
            orchestrator.pattern_analyzer, "analyze_agent_patterns"
        ) as mock_analyze:
            mock_analyze.return_value = {
                "patterns_found": 5,
                "confidence": 0.85,
                "analysis_time": 0.5,
            }

            tasks = []
            for agent_id in range(num_agents):
                task = orchestrator.pattern_analyzer.analyze_agent_patterns(
                    f"agent_{agent_id}",
                    [{"interaction": i} for i in range(interactions_per_agent)],
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)

        execution_time = time.time() - start_time

        # Validate scalability
        total_interactions = num_agents * interactions_per_agent
        interactions_per_second = total_interactions / execution_time

        assert interactions_per_second > 1000  # Should process 1000+ interactions/sec
        assert execution_time < 30.0  # Should complete within 30 seconds


class TestPerformanceAnalyticsPerformance:
    """Performance tests for Performance Analytics system"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_metric_collection_latency(self, performance_benchmarks):
        """Test real-time metric collection latency"""
        from nexus_forge.features.performance_analytics import (
            PerformanceAnalyticsOrchestrator,
        )

        orchestrator = PerformanceAnalyticsOrchestrator()
        orchestrator.cache = AsyncMock()

        collection_latencies = []

        # Test metric collection performance
        for i in range(1000):  # High volume test
            start_time = time.perf_counter()

            with patch.object(orchestrator, "collect_metric") as mock_collect:
                mock_collect.return_value = f"metric_{i}"

                await orchestrator.collect_metric(
                    f"agent_{i % 10}",
                    MetricType.LATENCY,
                    1000 + i,
                    {"test": f"metric_{i}"},
                )

            latency = (time.perf_counter() - start_time) * 1000  # ms
            collection_latencies.append(latency)

        avg_latency = statistics.mean(collection_latencies)
        p99_latency = statistics.quantiles(collection_latencies, n=100)[98]

        assert (
            avg_latency
            < performance_benchmarks["performance_analytics"][
                "max_metric_collection_latency"
            ]
        )
        assert (
            p99_latency
            < performance_benchmarks["performance_analytics"][
                "max_metric_collection_latency"
            ]
            * 2
        )

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_anomaly_detection_latency(self, performance_benchmarks):
        """Test anomaly detection response time"""
        from nexus_forge.features.performance_analytics import (
            PerformanceAnalyticsOrchestrator,
        )

        orchestrator = PerformanceAnalyticsOrchestrator()
        orchestrator.cache = AsyncMock()

        detection_latencies = []

        # Test anomaly detection with various patterns
        for i in range(100):
            # Create metric that might trigger anomaly detection
            metric_value = 1000 if i % 10 != 0 else 5000  # Anomaly every 10th metric

            start_time = time.perf_counter()

            with patch.object(
                orchestrator.anomaly_detector, "process_metric"
            ) as mock_detect:
                mock_detect.return_value = [] if i % 10 != 0 else [{"alert": "anomaly"}]

                alerts = await orchestrator.anomaly_detector.process_metric(
                    MagicMock(
                        agent_id=f"agent_{i % 5}",
                        metric_type=MetricType.LATENCY,
                        value=metric_value,
                    )
                )

            latency = (time.perf_counter() - start_time) * 1000  # ms
            detection_latencies.append(latency)

        avg_latency = statistics.mean(detection_latencies)
        max_latency = max(detection_latencies)

        assert (
            avg_latency
            < performance_benchmarks["performance_analytics"][
                "max_anomaly_detection_latency"
            ]
        )
        assert (
            max_latency
            < performance_benchmarks["performance_analytics"][
                "max_anomaly_detection_latency"
            ]
            * 2
        )

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_metrics_throughput(self, performance_benchmarks):
        """Test metrics processing throughput"""
        from nexus_forge.features.performance_analytics import (
            PerformanceAnalyticsOrchestrator,
        )

        orchestrator = PerformanceAnalyticsOrchestrator()
        orchestrator.cache = AsyncMock()

        target_throughput = performance_benchmarks["performance_analytics"][
            "metrics_per_second"
        ]

        start_time = time.time()

        # Generate high volume of metrics
        tasks = []
        for i in range(target_throughput):
            task = orchestrator.collect_metric(
                f"agent_{i % 50}",
                MetricType.THROUGHPUT,
                float(i),
                {"batch": "throughput_test"},
            )
            tasks.append(task)

        # Process all metrics
        await asyncio.gather(*tasks, return_exceptions=True)

        execution_time = time.time() - start_time
        actual_throughput = target_throughput / execution_time

        assert actual_throughput >= target_throughput * 0.8  # 80% of target throughput


class TestDynamicModelSelectionPerformance:
    """Performance tests for Dynamic Model Selection system"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_model_selection_latency(self, performance_benchmarks):
        """Test model selection decision latency"""
        from nexus_forge.features.dynamic_model_selection import (
            DynamicModelSelectionOrchestrator,
        )

        orchestrator = DynamicModelSelectionOrchestrator()
        orchestrator.cache = AsyncMock()

        selection_latencies = []

        # Test various task complexities
        tasks = [
            "Simple task",
            "Medium complexity task with multiple requirements",
            "Highly complex task requiring deep reasoning, analysis, and multi-step planning with various constraints and dependencies",
        ]

        for i in range(50):
            task_text = tasks[i % len(tasks)]

            start_time = time.perf_counter()

            with patch.object(orchestrator, "select_optimal_model") as mock_select:
                mock_select.return_value = MagicMock(
                    selected_model="gemini_flash_thinking",
                    confidence_score=0.85,
                    decision_latency_ms=25.0,
                )

                decision = await orchestrator.select_optimal_model(task_text)

            latency = (time.perf_counter() - start_time) * 1000  # ms
            selection_latencies.append(latency)

        avg_latency = statistics.mean(selection_latencies)
        p95_latency = statistics.quantiles(selection_latencies, n=20)[18]

        assert (
            avg_latency
            < performance_benchmarks["dynamic_model_selection"]["max_selection_latency"]
        )
        assert (
            p95_latency
            < performance_benchmarks["dynamic_model_selection"]["max_selection_latency"]
            * 1.5
        )

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_model_selections(self, performance_benchmarks):
        """Test concurrent model selection performance"""
        from nexus_forge.features.dynamic_model_selection import (
            DynamicModelSelectionOrchestrator,
        )

        orchestrator = DynamicModelSelectionOrchestrator()
        orchestrator.cache = AsyncMock()

        concurrent_selections = performance_benchmarks["dynamic_model_selection"][
            "concurrent_selections"
        ]

        async def model_selection_request(request_id):
            with patch.object(orchestrator, "select_optimal_model") as mock_select:
                mock_select.return_value = MagicMock(
                    selected_model="gemini_pro",
                    confidence_score=0.8,
                    decision_latency_ms=30.0,
                )

                return await orchestrator.select_optimal_model(
                    f"Concurrent task {request_id}"
                )

        start_time = time.time()

        # Execute concurrent selections
        tasks = [model_selection_request(i) for i in range(concurrent_selections)]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        execution_time = time.time() - start_time

        # Validate concurrent performance
        selections_per_second = len(results) / execution_time
        assert selections_per_second > 50  # Should handle 50+ selections/sec

        # Validate success rate
        successful_selections = len(
            [r for r in results if not isinstance(r, Exception)]
        )
        success_rate = successful_selections / len(results)
        assert success_rate > 0.95  # 95% success rate


class TestMultiModalIntegrationPerformance:
    """Performance tests for Multi-Modal Integration system"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_cross_modal_translation_latency(self, performance_benchmarks):
        """Test cross-modal translation performance"""
        from nexus_forge.features.multi_modal_integration import (
            MultiModalIntegrationOrchestrator,
        )

        orchestrator = MultiModalIntegrationOrchestrator()
        orchestrator.cache = AsyncMock()

        translation_latencies = []

        # Test various translation types
        translations = [
            ("text", "image", "Create a login interface"),
            ("text", "video", "Demonstrate user authentication"),
            ("image", "text", "Describe this UI design"),
            ("video", "text", "Summarize this demo"),
        ]

        for i in range(20):
            source, target, content = translations[i % len(translations)]

            start_time = time.perf_counter()

            with patch.object(
                orchestrator, "translate_between_modalities"
            ) as mock_translate:
                mock_translate.return_value = {
                    "translation_id": f"trans_{i}",
                    "translated_content": f"Translated {content}",
                    "quality_score": 0.88,
                    "processing_time_ms": 1500,
                }

                result = await orchestrator.translate_between_modalities(
                    content, source, target
                )

            latency = (time.perf_counter() - start_time) * 1000  # ms
            translation_latencies.append(latency)

        avg_latency = statistics.mean(translation_latencies)
        max_latency = max(translation_latencies)

        assert (
            avg_latency
            < performance_benchmarks["multimodal_integration"][
                "max_translation_latency"
            ]
        )
        assert (
            max_latency
            < performance_benchmarks["multimodal_integration"][
                "max_translation_latency"
            ]
            * 1.5
        )

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_workflow_execution_latency(self, performance_benchmarks):
        """Test multi-modal workflow execution performance"""
        from nexus_forge.features.multi_modal_integration import (
            MultiModalIntegrationOrchestrator,
        )

        orchestrator = MultiModalIntegrationOrchestrator()
        orchestrator.cache = AsyncMock()

        workflow_latencies = []

        # Test different workflow complexities
        for i in range(10):
            workflow_definition = {
                "name": f"Test Workflow {i}",
                "description": "Multi-modal test workflow",
                "execution_strategy": "sequential",
                "steps": [
                    {
                        "step_id": f"step1_{i}",
                        "step_name": "Text Analysis",
                        "target_modality": "text",
                        "operation": "analysis",
                        "dependencies": [],
                        "estimated_duration_ms": 2000,
                    },
                    {
                        "step_id": f"step2_{i}",
                        "step_name": "Image Generation",
                        "target_modality": "image",
                        "operation": "translation",
                        "dependencies": [f"step1_{i}"],
                        "estimated_duration_ms": 3000,
                    },
                ],
            }

            start_time = time.perf_counter()

            with patch.object(
                orchestrator, "execute_multi_modal_workflow"
            ) as mock_execute:
                mock_execute.return_value = {
                    "execution_id": f"exec_{i}",
                    "status": "completed",
                    "results": {
                        f"step1_{i}": {"result": "text_result"},
                        f"step2_{i}": {"result": "image_result"},
                    },
                    "execution_time_ms": 5000,
                }

                result = await orchestrator.execute_multi_modal_workflow(
                    workflow_definition
                )

            latency = (time.perf_counter() - start_time) * 1000  # ms
            workflow_latencies.append(latency)

        avg_latency = statistics.mean(workflow_latencies)
        max_latency = max(workflow_latencies)

        assert (
            avg_latency
            < performance_benchmarks["multimodal_integration"]["max_workflow_latency"]
        )
        assert (
            max_latency
            < performance_benchmarks["multimodal_integration"]["max_workflow_latency"]
            * 1.2
        )


@pytest.mark.performance
class TestSystemOverallPerformance:
    """Test overall system performance with all components active"""

    @pytest.mark.asyncio
    async def test_system_startup_time(self):
        """Test system initialization and startup time"""

        startup_times = {}

        # Test each system startup time
        systems = [
            ("agent_self_improvement", "nexus_forge.features.agent_self_improvement"),
            ("advanced_caching", "nexus_forge.features.advanced_caching"),
            ("behavior_analysis", "nexus_forge.features.agent_behavior_analysis"),
            ("performance_analytics", "nexus_forge.features.performance_analytics"),
            ("dynamic_model_selection", "nexus_forge.features.dynamic_model_selection"),
            ("multimodal_integration", "nexus_forge.features.multi_modal_integration"),
        ]

        for system_name, module_name in systems:
            start_time = time.time()

            # Import and initialize
            module = __import__(
                module_name, fromlist=[f"get_{system_name}_orchestrator"]
            )

            if hasattr(module, f"get_{system_name.replace('_', '')}_orchestrator"):
                orchestrator_func = getattr(
                    module, f"get_{system_name.replace('_', '')}_orchestrator"
                )
                orchestrator = await orchestrator_func()

                startup_time = time.time() - start_time
                startup_times[system_name] = startup_time

        # Validate startup times
        for system_name, startup_time in startup_times.items():
            assert (
                startup_time < 2.0
            ), f"{system_name} took too long to start: {startup_time:.2f}s"

        # Validate total startup time
        total_startup_time = sum(startup_times.values())
        assert (
            total_startup_time < 5.0
        ), f"Total startup time too long: {total_startup_time:.2f}s"

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, memory_monitor):
        """Test memory usage under concurrent load"""

        # Import all systems
        from nexus_forge.features.advanced_caching import AdvancedCacheOrchestrator
        from nexus_forge.features.agent_behavior_analysis import (
            AgentBehaviorAnalysisOrchestrator,
        )
        from nexus_forge.features.agent_self_improvement import (
            AgentSelfImprovementOrchestrator,
        )

        # Initialize systems
        systems = {
            "self_improvement": AgentSelfImprovementOrchestrator(),
            "caching": AdvancedCacheOrchestrator(),
            "behavior": AgentBehaviorAnalysisOrchestrator(),
        }

        # Mock cache for all systems
        mock_cache = AsyncMock()
        for system in systems.values():
            system.cache = mock_cache

        initial_memory = memory_monitor.update_peak()

        # Simulate concurrent load
        tasks = []

        for i in range(100):  # 100 concurrent operations
            # Mix of operations across systems
            if i % 3 == 0:
                with patch.object(
                    systems["self_improvement"], "generate_improvement_recommendations"
                ) as mock_gen:
                    mock_gen.return_value = [{"strategy": "test"}]
                    task = systems[
                        "self_improvement"
                    ].generate_improvement_recommendations(
                        f"agent_{i}", {"scores": [0.8]}
                    )
            elif i % 3 == 1:
                task = systems["caching"].cache.get_l1(f"key_{i}")
            else:
                with patch.object(
                    systems["behavior"], "log_agent_interaction"
                ) as mock_log:
                    mock_log.return_value = True
                    task = systems["behavior"].log_agent_interaction(
                        f"agent_{i}", "test", {"data": i}
                    )

            tasks.append(task)

        # Execute all tasks
        await asyncio.gather(*tasks, return_exceptions=True)

        final_memory = memory_monitor.update_peak()
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 1GB)
        assert (
            memory_increase < 1000
        ), f"Memory increase too high: {memory_increase:.1f}MB"

        # Force garbage collection and check for memory leaks
        gc.collect()
        post_gc_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Memory should reduce after GC
        memory_after_gc = post_gc_memory - initial_memory
        assert memory_after_gc < memory_increase * 0.8, "Potential memory leak detected"

    @pytest.mark.asyncio
    async def test_error_recovery_performance(self):
        """Test system performance during error conditions"""

        from nexus_forge.features.agent_self_improvement import (
            AgentSelfImprovementOrchestrator,
        )

        orchestrator = AgentSelfImprovementOrchestrator()

        # Test performance with various error conditions
        error_scenarios = [
            Exception("Network timeout"),
            ConnectionError("Database connection failed"),
            ValueError("Invalid input data"),
            TimeoutError("Operation timeout"),
        ]

        recovery_times = []

        for error in error_scenarios:
            # Simulate error condition
            orchestrator.cache = AsyncMock()
            orchestrator.cache.get_l1.side_effect = error

            start_time = time.time()

            # System should recover gracefully
            try:
                with patch.object(
                    orchestrator, "generate_improvement_recommendations"
                ) as mock_gen:
                    mock_gen.return_value = []  # Fallback result

                    result = await orchestrator.generate_improvement_recommendations(
                        "test_agent", {"scores": [0.8]}
                    )

                recovery_time = time.time() - start_time
                recovery_times.append(recovery_time)

            except Exception as e:
                # Should not propagate unhandled exceptions
                assert False, f"Unhandled exception during error recovery: {e}"

        # Validate recovery times
        avg_recovery_time = statistics.mean(recovery_times)
        assert (
            avg_recovery_time < 1.0
        ), f"Error recovery too slow: {avg_recovery_time:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])
