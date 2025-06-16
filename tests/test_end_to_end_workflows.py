"""
End-to-end workflow tests for Nexus Forge with enhanced Starri orchestration

Tests complete workflows from user prompt to deployed application,
including all MCP tool integrations and multi-agent coordination.
"""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus_forge.agents.starri.orchestrator import (
    AgentCapability,
    StarriOrchestrator,
    ThinkingMode,
)

pytestmark = pytest.mark.asyncio


class TestCompleteWorkflows:
    """Test complete end-to-end workflows"""

    @pytest.fixture
    async def full_orchestrator(self):
        """Create fully configured orchestrator for E2E tests"""
        with patch("nexus_forge.agents.starri.orchestrator.GeminiClient"):
            with patch(
                "nexus_forge.agents.starri.orchestrator.SupabaseCoordinationClient"
            ):
                with patch(
                    "nexus_forge.agents.starri.orchestrator.Mem0KnowledgeClient"
                ):
                    orchestrator = StarriOrchestrator(
                        project_id="e2e-test-project",
                        supabase_url="https://test.supabase.co",
                        supabase_key="test-key",
                    )

                    # Mock all integrations
                    orchestrator.gemini_client = AsyncMock()
                    orchestrator.coordination_client = AsyncMock()
                    orchestrator.knowledge_client = AsyncMock()
                    orchestrator.cache = AsyncMock()

                    # Mock connection success
                    orchestrator.coordination_client.connect.return_value = True
                    orchestrator.coordination_client.register_agent.return_value = (
                        "orch_agent_123"
                    )
                    orchestrator.knowledge_client.initialize_orchestrator_knowledge.return_value = (
                        None
                    )

                    # Initialize orchestrator
                    await orchestrator.initialize()

                    # Register sample agents
                    await self._register_sample_agents(orchestrator)

                    yield orchestrator

    async def _register_sample_agents(self, orchestrator):
        """Register sample agents for testing"""
        agents = [
            {
                "id": "ui_designer_001",
                "type": "ui_designer",
                "capabilities": [
                    AgentCapability.UI_DESIGN,
                    AgentCapability.IMAGE_GENERATION,
                ],
            },
            {
                "id": "backend_dev_001",
                "type": "backend_developer",
                "capabilities": [
                    AgentCapability.CODE_GENERATION,
                    AgentCapability.API_INTEGRATION,
                ],
            },
            {
                "id": "fullstack_dev_001",
                "type": "fullstack_developer",
                "capabilities": [
                    AgentCapability.CODE_GENERATION,
                    AgentCapability.UI_DESIGN,
                    AgentCapability.TESTING,
                    AgentCapability.DEPLOYMENT,
                ],
            },
            {
                "id": "qa_engineer_001",
                "type": "qa_engineer",
                "capabilities": [
                    AgentCapability.TESTING,
                    AgentCapability.DATA_ANALYSIS,
                ],
            },
        ]

        for agent in agents:
            await orchestrator.register_agent(
                agent_id=agent["id"],
                agent_type=agent["type"],
                capabilities=agent["capabilities"],
                configuration={"model": "gemini-pro", "timeout": 300},
            )

    async def test_simple_web_app_workflow(self, full_orchestrator):
        """Test building a simple web application end-to-end"""

        # Step 1: Deep thinking analysis of the request
        full_orchestrator.think_deeply = AsyncMock(
            return_value={
                "thinking_chain": [
                    {
                        "step": 1,
                        "thought": "User wants a task management app. Need to analyze requirements: CRUD operations, user auth, real-time updates.",
                        "confidence": 0.8,
                    },
                    {
                        "step": 2,
                        "thought": "This is a medium complexity project. Can be built with React frontend, FastAPI backend, PostgreSQL database.",
                        "confidence": 0.9,
                    },
                ],
                "conclusion": {
                    "conclusion": "Build a task management application with modern web stack",
                    "confidence": 0.9,
                },
                "confidence": 0.9,
            }
        )

        # Step 2: Task decomposition
        full_orchestrator.decompose_task = AsyncMock(
            return_value={
                "workflow_id": "workflow_task_app_001",
                "decomposition": {
                    "subtasks": [
                        {
                            "id": "design_ui",
                            "description": "Design user interface mockups and components",
                            "required_capabilities": ["ui_design"],
                            "estimated_duration": "45m",
                            "dependencies": [],
                        },
                        {
                            "id": "implement_frontend",
                            "description": "Implement React frontend with components",
                            "required_capabilities": ["code_generation", "ui_design"],
                            "estimated_duration": "90m",
                            "dependencies": ["design_ui"],
                        },
                        {
                            "id": "implement_backend",
                            "description": "Create FastAPI backend with authentication",
                            "required_capabilities": [
                                "code_generation",
                                "api_integration",
                            ],
                            "estimated_duration": "120m",
                            "dependencies": [],
                        },
                        {
                            "id": "write_tests",
                            "description": "Write comprehensive test suite",
                            "required_capabilities": ["testing"],
                            "estimated_duration": "60m",
                            "dependencies": ["implement_frontend", "implement_backend"],
                        },
                        {
                            "id": "deploy_app",
                            "description": "Deploy to cloud platform",
                            "required_capabilities": ["deployment"],
                            "estimated_duration": "30m",
                            "dependencies": ["write_tests"],
                        },
                    ],
                    "estimated_duration": "5.5 hours",
                    "execution_strategy": "mixed",  # Some parallel, some sequential
                },
                "confidence": 0.85,
            }
        )

        # Step 3: Agent coordination
        full_orchestrator.coordinate_agents = AsyncMock(
            return_value={
                "workflow_id": "workflow_task_app_001",
                "status": "completed",
                "results": {
                    "design_ui": {
                        "mockups": ["dashboard.png", "task_list.png"],
                        "components": ["TaskCard", "AddTaskForm", "FilterBar"],
                        "design_system": {"colors": "#007bff", "fonts": "Inter"},
                    },
                    "implement_frontend": {
                        "files": [
                            "src/App.js",
                            "src/components/TaskCard.js",
                            "src/pages/Dashboard.js",
                        ],
                        "package_json": {"react": "18.2.0", "typescript": "4.9.0"},
                    },
                    "implement_backend": {
                        "files": ["main.py", "models.py", "auth.py", "database.py"],
                        "requirements_txt": ["fastapi", "sqlalchemy", "pydantic"],
                    },
                    "write_tests": {
                        "test_files": ["test_api.py", "test_frontend.js"],
                        "coverage": "92%",
                    },
                    "deploy_app": {
                        "deployment_url": "https://task-app-abc123.cloudrun.app",
                        "status": "healthy",
                        "performance": {"response_time": "150ms", "uptime": "100%"},
                    },
                },
                "reflection": {
                    "reflection": "Workflow executed successfully with high efficiency",
                    "quality_score": 0.95,
                    "improvements": [
                        "Consider implementing caching for better performance"
                    ],
                },
                "metrics": {
                    "total_time": 4800,  # 80 minutes (faster than estimated)
                    "tasks_completed": 5,
                    "tasks_failed": 0,
                    "efficiency_score": 0.94,
                },
            }
        )

        # Execute the complete workflow
        user_prompt = "Build a task management web application with user authentication, CRUD operations for tasks, and a clean modern UI"

        # Step 1: Initial analysis
        analysis = await full_orchestrator.think_deeply(
            prompt=user_prompt, mode=ThinkingMode.DEEP_ANALYSIS
        )

        # Step 2: Task decomposition
        decomposition_result = await full_orchestrator.decompose_task(
            task_description=user_prompt,
            requirements=[
                "User authentication and authorization",
                "Create, read, update, delete tasks",
                "Modern responsive UI",
                "Real-time updates",
                "Search and filtering",
                "Data persistence",
            ],
            constraints={"budget": "medium", "timeline": "1 week", "priority": 8},
        )

        # Step 3: Coordinate execution
        execution_result = await full_orchestrator.coordinate_agents(
            workflow_id=decomposition_result["workflow_id"], execution_mode="mixed"
        )

        # Verify workflow completion
        assert execution_result["status"] == "completed"
        assert execution_result["metrics"]["tasks_completed"] == 5
        assert execution_result["metrics"]["tasks_failed"] == 0

        # Verify all major components were delivered
        results = execution_result["results"]
        assert "design_ui" in results
        assert "implement_frontend" in results
        assert "implement_backend" in results
        assert "write_tests" in results
        assert "deploy_app" in results

        # Verify deployment success
        assert results["deploy_app"]["deployment_url"].startswith("https://")
        assert results["deploy_app"]["status"] == "healthy"

        # Verify quality metrics
        assert execution_result["reflection"]["quality_score"] > 0.9
        assert float(results["write_tests"]["coverage"].rstrip("%")) > 90

    async def test_complex_data_dashboard_workflow(self, full_orchestrator):
        """Test building a complex analytics dashboard"""

        # Mock complex thinking process
        full_orchestrator.think_deeply = AsyncMock(
            side_effect=[
                # Initial analysis
                {
                    "thinking_chain": [
                        {
                            "step": 1,
                            "thought": "Analytics dashboard requires real-time data processing, visualization, and performance optimization.",
                            "confidence": 0.8,
                        },
                        {
                            "step": 2,
                            "thought": "High complexity project. Need data pipeline, API optimization, interactive charts, and scalable architecture.",
                            "confidence": 0.85,
                        },
                        {
                            "step": 3,
                            "thought": "Recommend microservices architecture with event streaming for real-time updates.",
                            "confidence": 0.9,
                        },
                    ],
                    "conclusion": {
                        "conclusion": "Complex analytics dashboard with microservices",
                        "confidence": 0.9,
                    },
                    "confidence": 0.9,
                    "mode": "deep_analysis",
                },
                # Coordination planning
                {
                    "conclusion": {
                        "coordination_plan": "parallel_with_dependencies",
                        "agent_assignments": {
                            "data_pipeline": "backend_dev_001",
                            "dashboard_ui": "ui_designer_001",
                            "api_optimization": "backend_dev_001",
                            "testing_suite": "qa_engineer_001",
                        },
                        "critical_path": [
                            "data_pipeline",
                            "api_optimization",
                            "dashboard_ui",
                        ],
                    },
                    "confidence": 0.88,
                    "mode": "coordination",
                },
                # Execution reflection
                {
                    "reflection": "Complex workflow handled well with microservices approach",
                    "quality_score": 0.92,
                    "improvements": [
                        "Add monitoring and alerting",
                        "Implement caching strategy",
                    ],
                    "mode": "reflection",
                },
            ]
        )

        # Mock task decomposition for complex project
        full_orchestrator._parse_task_decomposition = MagicMock(
            return_value={
                "subtasks": [
                    {
                        "id": "design_data_architecture",
                        "description": "Design scalable data architecture and pipeline",
                        "required_capabilities": ["data_analysis", "api_integration"],
                        "estimated_duration": "120m",
                        "complexity": "high",
                    },
                    {
                        "id": "build_data_pipeline",
                        "description": "Implement real-time data ingestion and processing",
                        "required_capabilities": ["code_generation", "data_analysis"],
                        "estimated_duration": "180m",
                        "complexity": "high",
                    },
                    {
                        "id": "create_dashboard_ui",
                        "description": "Build interactive dashboard with charts and filters",
                        "required_capabilities": ["ui_design", "code_generation"],
                        "estimated_duration": "150m",
                        "complexity": "medium",
                    },
                    {
                        "id": "optimize_performance",
                        "description": "Implement caching, indexing, and optimization",
                        "required_capabilities": ["optimization", "code_generation"],
                        "estimated_duration": "90m",
                        "complexity": "medium",
                    },
                    {
                        "id": "comprehensive_testing",
                        "description": "Performance testing, load testing, integration testing",
                        "required_capabilities": ["testing", "data_analysis"],
                        "estimated_duration": "120m",
                        "complexity": "medium",
                    },
                ],
                "total_duration": "11 hours",
                "execution_strategy": "parallel_with_dependencies",
                "critical_path": [
                    "design_data_architecture",
                    "build_data_pipeline",
                    "optimize_performance",
                ],
            }
        )

        # Mock workflow creation and validation
        full_orchestrator.coordination_client.create_workflow.return_value = (
            "complex_dashboard_workflow_001"
        )
        full_orchestrator._validate_decomposition = AsyncMock(
            return_value={
                "valid": True,
                "errors": [],
                "warnings": ["High complexity project"],
            }
        )

        # Mock successful execution with metrics
        full_orchestrator._execute_coordination_plan = AsyncMock(
            return_value={
                "status": "completed",
                "results": {
                    "design_data_architecture": {
                        "architecture_docs": ["data_flow.md", "api_spec.yaml"],
                        "database_schema": "optimized_schema.sql",
                        "performance_requirements": {
                            "throughput": "10k rps",
                            "latency": "<100ms",
                        },
                    },
                    "build_data_pipeline": {
                        "pipeline_code": [
                            "ingestion.py",
                            "processing.py",
                            "streaming.py",
                        ],
                        "docker_configs": ["Dockerfile", "docker-compose.yml"],
                        "throughput_achieved": "12k rps",
                    },
                    "create_dashboard_ui": {
                        "ui_components": ["DataGrid", "ChartContainer", "FilterPanel"],
                        "responsive_design": True,
                        "accessibility_score": "AA compliant",
                    },
                    "optimize_performance": {
                        "caching_strategy": "Redis + CDN",
                        "database_indexes": "optimized",
                        "response_time": "avg 45ms",
                    },
                    "comprehensive_testing": {
                        "test_coverage": "96%",
                        "load_test_results": "passed at 15k rps",
                        "security_scan": "no critical issues",
                    },
                },
                "metrics": {
                    "total_time": 7200,  # 2 hours (much faster than estimated)
                    "tasks_completed": 5,
                    "tasks_failed": 0,
                    "parallel_efficiency": 0.91,
                },
            }
        )

        full_orchestrator._monitor_workflow_execution = AsyncMock()

        # Execute complex workflow
        user_prompt = """
        Build a real-time analytics dashboard that can handle high-volume data streams,
        with interactive visualizations, filtering capabilities, user management,
        and performance optimization for handling 10k+ concurrent users
        """

        # Deep analysis phase
        analysis = await full_orchestrator.think_deeply(
            prompt=user_prompt, mode=ThinkingMode.DEEP_ANALYSIS, max_thinking_steps=5
        )

        # Task decomposition phase
        decomposition = await full_orchestrator.decompose_task(
            task_description=user_prompt,
            requirements=[
                "Real-time data processing",
                "Interactive visualizations",
                "High-performance backend",
                "Scalable architecture",
                "User authentication and roles",
                "Advanced filtering and search",
                "Mobile responsive design",
                "Performance monitoring",
            ],
            constraints={
                "performance": {
                    "max_response_time": "100ms",
                    "min_throughput": "10k rps",
                },
                "scalability": {"concurrent_users": 10000},
                "budget": "high",
                "timeline": "2 weeks",
            },
        )

        # Coordination and execution phase
        execution = await full_orchestrator.coordinate_agents(
            workflow_id=decomposition["workflow_id"], execution_mode="parallel"
        )

        # Verify complex workflow success
        assert execution["status"] == "completed"
        assert execution["metrics"]["tasks_completed"] == 5
        assert execution["metrics"]["parallel_efficiency"] > 0.85

        # Verify performance requirements met
        results = execution["results"]
        throughput = (
            int(results["build_data_pipeline"]["throughput_achieved"].split("k")[0])
            * 1000
        )
        assert throughput >= 10000  # Met throughput requirement

        response_time = int(
            results["optimize_performance"]["response_time"].split()[1].rstrip("ms")
        )
        assert response_time <= 100  # Met latency requirement

        # Verify quality standards
        test_coverage = float(
            results["comprehensive_testing"]["test_coverage"].rstrip("%")
        )
        assert test_coverage >= 95

        assert results["create_dashboard_ui"]["accessibility_score"] == "AA compliant"
        assert "no critical issues" in results["comprehensive_testing"]["security_scan"]

    async def test_workflow_error_recovery(self, full_orchestrator):
        """Test workflow error recovery and adaptation"""

        # Mock a workflow with partial failure
        full_orchestrator.think_deeply = AsyncMock(
            return_value={
                "conclusion": {
                    "conclusion": "Standard web app workflow",
                    "confidence": 0.8,
                },
                "confidence": 0.8,
            }
        )

        full_orchestrator.decompose_task = AsyncMock(
            return_value={
                "workflow_id": "error_recovery_test",
                "decomposition": {
                    "subtasks": [
                        {"id": "task_1", "description": "Successful task"},
                        {"id": "task_2", "description": "Failing task"},
                        {"id": "task_3", "description": "Recovery task"},
                    ]
                },
            }
        )

        # Mock execution with partial failure and recovery
        execution_calls = []

        async def mock_execute_single_task(workflow_id, subtask):
            execution_calls.append(subtask["id"])

            if subtask["id"] == "task_2":
                # Simulate failure
                raise Exception("Simulated task failure")
            else:
                # Simulate success
                return {"status": "completed", "result": f"Success for {subtask['id']}"}

        full_orchestrator._execute_single_task = mock_execute_single_task
        full_orchestrator.coordination_client.update_workflow_status = AsyncMock()
        full_orchestrator.coordination_client.create_task = AsyncMock(
            side_effect=["t1", "t2", "t3"]
        )
        full_orchestrator.coordination_client.assign_task = AsyncMock(return_value=True)
        full_orchestrator._find_suitable_agents = MagicMock(return_value=["agent_1"])

        # Override the execution plan to test error handling
        async def mock_execute_plan(workflow_id, plan, mode):
            results = {
                "status": "partial_success",
                "results": {},
                "metrics": {"tasks_completed": 2, "tasks_failed": 1},
            }

            # Try to execute all tasks
            for task_id in ["task_1", "task_2", "task_3"]:
                try:
                    if task_id == "task_2":
                        results["results"][task_id] = {"error": "Simulated failure"}
                    else:
                        results["results"][task_id] = {"status": "completed"}
                        results["metrics"]["tasks_completed"] += 1
                except Exception as e:
                    results["results"][task_id] = {"error": str(e)}
                    results["metrics"]["tasks_failed"] += 1

            return results

        full_orchestrator._execute_coordination_plan = mock_execute_plan
        full_orchestrator._monitor_workflow_execution = AsyncMock()
        full_orchestrator._reflect_on_execution = AsyncMock(
            return_value={
                "reflection": "Workflow had partial failure but recovery was successful",
                "lessons_learned": [
                    "Implement better error handling",
                    "Add retry mechanisms",
                ],
            }
        )

        # Execute workflow with error
        user_prompt = "Build a simple web app (error recovery test)"

        decomposition = await full_orchestrator.decompose_task(
            task_description=user_prompt, requirements=["Basic functionality"]
        )

        execution = await full_orchestrator.coordinate_agents(
            workflow_id=decomposition["workflow_id"]
        )

        # Verify error handling
        assert execution["status"] == "partial_success"
        assert execution["metrics"]["tasks_failed"] > 0
        assert "error" in execution["results"]["task_2"]

        # Verify reflection captured the issues
        assert "partial failure" in execution["reflection"]["reflection"]
        assert len(execution["reflection"]["lessons_learned"]) > 0

    async def test_concurrent_workflows(self, full_orchestrator):
        """Test handling multiple concurrent workflows"""

        # Mock multiple thinking processes
        full_orchestrator.think_deeply = AsyncMock(
            return_value={
                "conclusion": {
                    "conclusion": "Concurrent workflow execution",
                    "confidence": 0.85,
                },
                "confidence": 0.85,
            }
        )

        # Mock decomposition for multiple workflows
        workflow_counter = 0

        async def mock_decompose_task(task_description, requirements, constraints=None):
            nonlocal workflow_counter
            workflow_counter += 1
            return {
                "workflow_id": f"concurrent_workflow_{workflow_counter}",
                "decomposition": {
                    "subtasks": [
                        {
                            "id": f"task_{workflow_counter}_1",
                            "description": f"Task 1 for workflow {workflow_counter}",
                        },
                        {
                            "id": f"task_{workflow_counter}_2",
                            "description": f"Task 2 for workflow {workflow_counter}",
                        },
                    ]
                },
            }

        full_orchestrator.decompose_task = mock_decompose_task

        # Mock coordination for multiple workflows
        async def mock_coordinate_agents(workflow_id, execution_mode="parallel"):
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "results": {f"task_1": "success", f"task_2": "success"},
                "metrics": {"tasks_completed": 2, "tasks_failed": 0},
            }

        full_orchestrator.coordinate_agents = mock_coordinate_agents

        # Create multiple concurrent workflows
        workflow_tasks = []

        for i in range(3):

            async def create_workflow(index=i):
                # Decompose task
                decomposition = await full_orchestrator.decompose_task(
                    task_description=f"Build web app {index + 1}",
                    requirements=[f"Feature {index + 1}"],
                )

                # Execute workflow
                execution = await full_orchestrator.coordinate_agents(
                    workflow_id=decomposition["workflow_id"]
                )

                return {"decomposition": decomposition, "execution": execution}

            workflow_tasks.append(asyncio.create_task(create_workflow()))

        # Wait for all workflows to complete
        results = await asyncio.gather(*workflow_tasks)

        # Verify all workflows completed successfully
        assert len(results) == 3

        for i, result in enumerate(results):
            assert (
                result["decomposition"]["workflow_id"] == f"concurrent_workflow_{i + 1}"
            )
            assert result["execution"]["status"] == "completed"
            assert result["execution"]["metrics"]["tasks_completed"] == 2

        # Verify orchestrator handled concurrent load
        status = await full_orchestrator.get_orchestrator_status()
        assert status["status"] == "operational"


class TestPerformanceBenchmarks:
    """Performance and scalability benchmarks"""

    @pytest.fixture
    def benchmark_orchestrator(self):
        """Lightweight orchestrator for performance testing"""
        with patch("nexus_forge.agents.starri.orchestrator.GeminiClient"):
            with patch(
                "nexus_forge.agents.starri.orchestrator.SupabaseCoordinationClient"
            ):
                with patch(
                    "nexus_forge.agents.starri.orchestrator.Mem0KnowledgeClient"
                ):
                    orchestrator = StarriOrchestrator("bench", "url", "key")

                    # Mock with fast responses
                    orchestrator.gemini_client = AsyncMock()
                    orchestrator.coordination_client = AsyncMock()
                    orchestrator.knowledge_client = AsyncMock()
                    orchestrator.cache = AsyncMock()

                    return orchestrator

    async def test_thinking_performance_benchmark(self, benchmark_orchestrator):
        """Benchmark thinking performance"""
        import time

        # Mock fast Gemini responses
        benchmark_orchestrator.gemini_client.generate_content = AsyncMock(
            return_value={
                "content": "Quick analysis with confidence: 0.9. Solution found.",
                "usage_metadata": {"total_token_count": 50},
            }
        )

        # Benchmark thinking speed
        start_time = time.time()

        thinking_tasks = []
        for i in range(10):
            task = benchmark_orchestrator.think_deeply(
                prompt=f"Optimization problem {i}",
                mode=ThinkingMode.QUICK_DECISION,
                max_thinking_steps=2,
            )
            thinking_tasks.append(task)

        results = await asyncio.gather(*thinking_tasks)

        end_time = time.time()
        total_time = end_time - start_time

        # Performance assertions
        assert len(results) == 10
        assert (
            total_time < 5.0
        )  # Should complete 10 thinking processes in under 5 seconds
        assert all(result["confidence"] > 0.8 for result in results)

        # Calculate average thinking time
        avg_thinking_time = sum(result["thinking_time"] for result in results) / len(
            results
        )
        assert avg_thinking_time < 1.0  # Each thinking process should be under 1 second

    async def test_agent_registration_scalability(self, benchmark_orchestrator):
        """Test agent registration scalability"""
        import time

        # Mock fast registrations
        benchmark_orchestrator.knowledge_client.add_agent_entity = AsyncMock(
            return_value="entity_id"
        )

        start_time = time.time()

        # Register 100 agents concurrently
        registration_tasks = []
        for i in range(100):
            task = benchmark_orchestrator.register_agent(
                agent_id=f"agent_{i:03d}",
                agent_type="test_agent",
                capabilities=[AgentCapability.CODE_GENERATION],
                configuration={"model": "test"},
            )
            registration_tasks.append(task)

        await asyncio.gather(*registration_tasks)

        end_time = time.time()
        registration_time = end_time - start_time

        # Performance assertions
        assert len(benchmark_orchestrator.registered_agents) == 100
        assert registration_time < 2.0  # Should register 100 agents in under 2 seconds

        # Verify capability mapping efficiency
        assert (
            len(
                benchmark_orchestrator.agent_capabilities[
                    AgentCapability.CODE_GENERATION
                ]
            )
            == 100
        )

    async def test_cache_performance_benchmark(self, benchmark_orchestrator):
        """Benchmark cache performance across all levels"""
        import time

        cache = benchmark_orchestrator.cache

        # Mock cache operations with realistic timing
        cache.set_l1.return_value = True
        cache.set_l2.return_value = True
        cache.set_l3.return_value = True
        cache.get_l1.return_value = None  # Force cache miss for timing
        cache.get_l2.return_value = None
        cache.get_l3.return_value = None

        test_data = {"large_response": "x" * 10000}  # 10KB test data

        # Benchmark L1 cache (in-memory)
        start_time = time.time()
        for i in range(1000):
            cache.set_l1(f"l1_key_{i}", test_data)
        l1_time = time.time() - start_time

        # Benchmark L2 cache (Redis session)
        start_time = time.time()
        for i in range(100):
            cache.set_l2(f"l2_key_{i}", test_data)
        l2_time = time.time() - start_time

        # Benchmark L3 cache (Redis long-term)
        start_time = time.time()
        for i in range(100):
            cache.set_l3(f"l3_key_{i}", test_data)
        l3_time = time.time() - start_time

        # Performance assertions
        assert l1_time < 0.1  # L1 should be very fast (1000 ops < 100ms)
        assert l2_time < 1.0  # L2 should be fast (100 ops < 1s)
        assert l3_time < 1.0  # L3 should be fast (100 ops < 1s)

        # Verify call counts
        assert cache.set_l1.call_count == 1000
        assert cache.set_l2.call_count == 100
        assert cache.set_l3.call_count == 100
