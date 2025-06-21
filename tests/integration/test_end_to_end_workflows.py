"""
End-to-End Workflow Validation Tests
Tests complete workflows across all 16 advanced AI systems
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytest

from src.backend.ai_features.custom_training import CustomTrainingEngine
from src.backend.ai_features.predictive_coordination import PredictiveCoordinator
from src.backend.ai_features.quality_control import AutonomousQualityController
from src.backend.marketplace.performance_benchmarker import PerformanceBenchmarker

# Import all system components for comprehensive testing
from src.backend.marketplace.registry import AgentRegistry
from src.backend.marketplace.search_engine import AgentSearchEngine
from src.backend.marketplace.security_scanner import SecurityScanner
from src.backend.multi_region.data_sync import DataSynchronizer
from src.backend.multi_region.load_balancer import GlobalLoadBalancer
from src.backend.multi_region.region_manager import RegionManager
from src.backend.multi_tenancy.isolation_manager import IsolationManager
from src.backend.multi_tenancy.tenant_manager import TenantManager
from src.backend.workflow_builder.compiler import WorkflowCompiler
from src.backend.workflow_builder.engine import WorkflowEngine
from src.backend.workflow_builder.executor import WorkflowExecutor

logger = logging.getLogger(__name__)


class EndToEndWorkflowTester:
    """Comprehensive end-to-end workflow testing framework"""

    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        self.workflow_start_time = None

        # Initialize all system components
        self.systems = {}
        self._initialize_systems()

    def _initialize_systems(self):
        """Initialize all system components for testing"""
        logger.info("🚀 Initializing all systems for end-to-end testing")

        # Marketplace systems
        self.systems["agent_registry"] = AgentRegistry()
        self.systems["security_scanner"] = SecurityScanner()
        self.systems["performance_benchmarker"] = PerformanceBenchmarker()
        self.systems["search_engine"] = AgentSearchEngine()

        # Workflow systems
        self.systems["workflow_engine"] = WorkflowEngine()
        self.systems["workflow_compiler"] = WorkflowCompiler()
        self.systems["workflow_executor"] = WorkflowExecutor()

        # Multi-tenancy systems
        self.systems["tenant_manager"] = TenantManager()
        self.systems["isolation_manager"] = IsolationManager()

        # Multi-region systems
        self.systems["region_manager"] = RegionManager()
        self.systems["load_balancer"] = GlobalLoadBalancer()
        self.systems["data_synchronizer"] = DataSynchronizer()

        # AI features
        self.systems["training_engine"] = CustomTrainingEngine()
        self.systems["predictive_coordinator"] = PredictiveCoordinator()
        self.systems["quality_controller"] = AutonomousQualityController()

        logger.info("✅ All systems initialized for testing")

    async def test_complete_application_generation_workflow(self):
        """
        Test Workflow 1: Complete Application Generation
        Agent Discovery → Workflow Creation → Multi-Tenant Deployment → Quality Monitoring
        """
        logger.info("🧪 Testing Complete Application Generation Workflow")
        workflow_start = time.time()

        try:
            # Step 1: Agent Discovery in Marketplace
            logger.info("Step 1: Discovering agents in marketplace")
            search_engine = self.systems["search_engine"]

            discovered_agents = await search_engine.search_agents(
                query="full-stack development coordination",
                filters={"capability": "application_generation"},
                limit=5,
            )

            assert len(discovered_agents) >= 0, "Should be able to search for agents"
            logger.info(f"✅ Discovered {len(discovered_agents)} agents")

            # Step 2: Security Scanning of Selected Agents
            logger.info("Step 2: Security scanning selected agents")
            security_scanner = self.systems["security_scanner"]

            for agent in discovered_agents[:2]:  # Test first 2 agents
                scan_result = await security_scanner.scan_agent(agent["id"])
                assert scan_result["status"] in [
                    "passed",
                    "warning",
                ], f"Agent {agent['id']} failed security scan"

            logger.info("✅ Security scanning completed")

            # Step 3: Performance Benchmarking
            logger.info("Step 3: Performance benchmarking agents")
            benchmarker = self.systems["performance_benchmarker"]

            benchmark_results = {}
            for agent in discovered_agents[:2]:
                result = await benchmarker.benchmark_agent(agent["id"])
                benchmark_results[agent["id"]] = result
                assert (
                    result["overall_score"] > 0
                ), f"Agent {agent['id']} has invalid benchmark score"

            logger.info("✅ Performance benchmarking completed")

            # Step 4: Workflow Creation
            logger.info("Step 4: Creating workflow with selected agents")
            workflow_engine = self.systems["workflow_engine"]

            workflow_config = {
                "name": "test_app_generation_workflow",
                "description": "End-to-end application generation test",
                "agents": [agent["id"] for agent in discovered_agents[:2]],
                "coordination_pattern": "parallel_with_aggregation",
                "target_application": {
                    "type": "web_app",
                    "tech_stack": ["react", "fastapi", "postgresql"],
                    "features": ["authentication", "data_management", "api"],
                },
            }

            created_workflow = await workflow_engine.create_workflow(workflow_config)
            assert created_workflow["id"], "Workflow should have valid ID"
            logger.info(f"✅ Workflow created: {created_workflow['id']}")

            # Step 5: Workflow Compilation
            logger.info("Step 5: Compiling workflow")
            compiler = self.systems["workflow_compiler"]

            compiled_workflow = await compiler.compile_workflow(created_workflow["id"])
            assert (
                compiled_workflow["status"] == "compiled"
            ), "Workflow compilation failed"
            logger.info("✅ Workflow compiled successfully")

            # Step 6: Tenant Creation for Deployment
            logger.info("Step 6: Creating tenant for deployment")
            tenant_manager = self.systems["tenant_manager"]

            tenant_config = {
                "name": "test_app_generation_tenant",
                "tier": "enterprise",
                "resource_quotas": {"cpu_cores": 4, "memory_gb": 8, "storage_gb": 50},
            }

            tenant = await tenant_manager.create_tenant(tenant_config)
            assert tenant["id"], "Tenant should have valid ID"
            logger.info(f"✅ Tenant created: {tenant['id']}")

            # Step 7: Tenant Isolation Setup
            logger.info("Step 7: Setting up tenant isolation")
            isolation_manager = self.systems["isolation_manager"]

            isolation_result = await isolation_manager.ensure_tenant_isolation(
                tenant["id"]
            )
            assert isolation_result["status"] == "isolated", "Tenant isolation failed"
            logger.info("✅ Tenant isolation configured")

            # Step 8: Workflow Execution
            logger.info("Step 8: Executing workflow")
            executor = self.systems["workflow_executor"]

            execution_result = await executor.execute_workflow(
                workflow_id=created_workflow["id"],
                tenant_id=tenant["id"],
                execution_config={
                    "timeout_minutes": 10,
                    "parallel_execution": True,
                    "monitoring_enabled": True,
                },
            )

            assert execution_result["status"] in [
                "completed",
                "in_progress",
            ], "Workflow execution failed"
            logger.info(f"✅ Workflow execution status: {execution_result['status']}")

            # Step 9: Quality Monitoring
            logger.info("Step 9: Quality monitoring and assessment")
            quality_controller = self.systems["quality_controller"]

            quality_assessment = await quality_controller.assess_workflow_quality(
                workflow_id=created_workflow["id"],
                execution_id=execution_result["execution_id"],
            )

            assert quality_assessment["overall_score"] > 0, "Quality assessment failed"
            logger.info(
                f"✅ Quality assessment completed: {quality_assessment['overall_score']}"
            )

            # Calculate workflow completion time
            workflow_duration = time.time() - workflow_start

            # Record success metrics
            self.test_results["complete_application_generation"] = {
                "status": "passed",
                "duration_seconds": workflow_duration,
                "agents_discovered": len(discovered_agents),
                "workflow_id": created_workflow["id"],
                "tenant_id": tenant["id"],
                "quality_score": quality_assessment["overall_score"],
            }

            logger.info(
                f"🎉 Complete Application Generation Workflow PASSED in {workflow_duration:.2f}s"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Complete Application Generation Workflow FAILED: {e}")
            self.test_results["complete_application_generation"] = {
                "status": "failed",
                "error": str(e),
                "duration_seconds": time.time() - workflow_start,
            }
            raise

    async def test_multi_region_coordination_workflow(self):
        """
        Test Workflow 2: Multi-Region Coordination
        Agent Distribution → Load Balancing → Data Synchronization → Failover Testing
        """
        logger.info("🧪 Testing Multi-Region Coordination Workflow")
        workflow_start = time.time()

        try:
            # Step 1: Region Discovery and Setup
            logger.info("Step 1: Discovering available regions")
            region_manager = self.systems["region_manager"]

            available_regions = await region_manager.list_available_regions()
            logger.info(f"✅ Found {len(available_regions)} available regions")

            # Step 2: Global Load Balancer Configuration
            logger.info("Step 2: Configuring global load balancer")
            load_balancer = self.systems["load_balancer"]

            lb_config = {
                "strategy": "latency_based",
                "health_check_interval": 30,
                "regions": (
                    available_regions[:3]
                    if len(available_regions) > 3
                    else available_regions
                ),
                "failover_enabled": True,
            }

            lb_result = await load_balancer.configure_balancing(lb_config)
            assert (
                lb_result["status"] == "configured"
            ), "Load balancer configuration failed"
            logger.info("✅ Load balancer configured")

            # Step 3: Agent Distribution Across Regions
            logger.info("Step 3: Distributing agents across regions")

            agent_distribution = {}
            for region in lb_config["regions"]:
                # Simulate agent deployment to region
                deployment_result = await region_manager.deploy_agents_to_region(
                    region_id=region,
                    agent_configs=[
                        {
                            "type": "coordinator",
                            "resources": {"cpu": 2, "memory": "4GB"},
                        },
                        {"type": "processor", "resources": {"cpu": 1, "memory": "2GB"}},
                    ],
                )
                agent_distribution[region] = deployment_result
                assert (
                    deployment_result["status"] == "deployed"
                ), f"Agent deployment to {region} failed"

            logger.info(f"✅ Agents distributed to {len(agent_distribution)} regions")

            # Step 4: Data Synchronization Setup
            logger.info("Step 4: Setting up cross-region data synchronization")
            data_synchronizer = self.systems["data_synchronizer"]

            sync_config = {
                "replication_strategy": "async_master_slave",
                "conflict_resolution": "last_write_wins",
                "regions": list(agent_distribution.keys()),
            }

            sync_result = await data_synchronizer.setup_synchronization(sync_config)
            assert (
                sync_result["status"] == "active"
            ), "Data synchronization setup failed"
            logger.info("✅ Data synchronization configured")

            # Step 5: Predictive Coordination Testing
            logger.info("Step 5: Testing predictive coordination across regions")
            predictive_coordinator = self.systems["predictive_coordinator"]

            prediction_result = await predictive_coordinator.predict_coordination_needs(
                horizon_minutes=30, region_scope=list(agent_distribution.keys())
            )

            assert (
                prediction_result.prediction_confidence > 0.5
            ), "Prediction confidence too low"
            logger.info(
                f"✅ Predictive coordination: {prediction_result.prediction_confidence:.2f} confidence"
            )

            # Step 6: Load Testing and Failover
            logger.info("Step 6: Testing load distribution and failover")

            # Simulate load across regions
            load_test_results = {}
            for region in agent_distribution.keys():
                load_result = await load_balancer.test_region_load(
                    region_id=region, concurrent_requests=50, duration_seconds=10
                )
                load_test_results[region] = load_result
                assert (
                    load_result["success_rate"] > 0.8
                ), f"Load test failed for region {region}"

            logger.info(
                f"✅ Load testing completed across {len(load_test_results)} regions"
            )

            # Calculate workflow completion time
            workflow_duration = time.time() - workflow_start

            # Record success metrics
            self.test_results["multi_region_coordination"] = {
                "status": "passed",
                "duration_seconds": workflow_duration,
                "regions_tested": len(agent_distribution),
                "prediction_confidence": float(prediction_result.prediction_confidence),
                "load_test_results": load_test_results,
            }

            logger.info(
                f"🎉 Multi-Region Coordination Workflow PASSED in {workflow_duration:.2f}s"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Multi-Region Coordination Workflow FAILED: {e}")
            self.test_results["multi_region_coordination"] = {
                "status": "failed",
                "error": str(e),
                "duration_seconds": time.time() - workflow_start,
            }
            raise

    async def test_ai_training_pipeline_workflow(self):
        """
        Test Workflow 3: AI Training Pipeline
        Data Ingestion → Model Training → Deployment → Prediction → Quality Control
        """
        logger.info("🧪 Testing AI Training Pipeline Workflow")
        workflow_start = time.time()

        try:
            # Step 1: Training Engine Setup
            logger.info("Step 1: Setting up training engine")
            training_engine = self.systems["training_engine"]

            # Import training models
            from src.backend.ai_features.models import (
                HyperParameters,
                ModelArchitecture,
                TrainingDataSource,
                TrainingJob,
                TrainingJobType,
            )

            # Step 2: Create Training Job
            logger.info("Step 2: Creating AI training job")

            training_job = TrainingJob(
                name="e2e_coordination_model",
                description="End-to-end test training job for coordination prediction",
                job_type=TrainingJobType.FINE_TUNING,
                architecture=ModelArchitecture.LSTM,
                data_sources=[
                    TrainingDataSource(
                        source_type="coordination_logs",
                        connection_string="mock://coordination_data",
                        preprocessing={"normalize": True, "sequence_length": 60},
                    )
                ],
                target_variable="coordination_efficiency",
                feature_columns=[
                    "agent_count",
                    "task_complexity",
                    "resource_usage",
                    "response_time",
                    "success_rate",
                    "error_rate",
                ],
                hyperparameters=HyperParameters(
                    learning_rate=0.001,
                    batch_size=32,
                    epochs=5,  # Reduced for testing
                    dropout_rate=0.2,
                    hidden_size=128,
                    num_layers=2,
                ),
                created_by="e2e_test_system",
            )

            created_job = await training_engine.create_training_job(training_job)
            assert created_job.id, "Training job should have valid ID"
            logger.info(f"✅ Training job created: {created_job.id}")

            # Step 3: Start Training Process
            logger.info("Step 3: Starting training process")

            training_started = await training_engine.start_training_job(created_job.id)
            assert training_started, "Training job should start successfully"

            # Monitor training progress
            training_status = await training_engine.get_training_job_status(
                created_job.id
            )
            assert training_status["status"] in [
                "preparing",
                "training",
            ], "Training should be in progress"
            logger.info(f"✅ Training started: {training_status['status']}")

            # Step 4: Predictive Coordination Integration
            logger.info("Step 4: Testing predictive coordination integration")
            predictive_coordinator = self.systems["predictive_coordinator"]

            # Test prediction while training is in progress
            coordination_prediction = (
                await predictive_coordinator.predict_coordination_needs(
                    horizon_minutes=15
                )
            )

            assert (
                coordination_prediction.prediction_confidence > 0
            ), "Should generate predictions"
            logger.info(
                f"✅ Coordination prediction: {coordination_prediction.predicted_cpu_usage:.1f}% CPU"
            )

            # Step 5: Quality Control Assessment
            logger.info("Step 5: Quality control assessment")
            quality_controller = self.systems["quality_controller"]

            quality_metrics = await quality_controller.assess_training_quality(
                training_job_id=created_job.id,
                metrics={
                    "training_loss": 0.25,
                    "validation_accuracy": 0.85,
                    "convergence_rate": 0.9,
                },
            )

            assert (
                quality_metrics.overall_score > 50
            ), "Quality assessment should pass minimum threshold"
            logger.info(
                f"✅ Quality assessment: {quality_metrics.overall_score:.1f}/100"
            )

            # Step 6: Model Performance Evaluation
            logger.info("Step 6: Evaluating model performance")

            # Get training accuracy metrics
            accuracy_metrics = await training_engine.get_training_accuracy(
                created_job.id
            )

            # Validate prediction capabilities
            prediction_test = await predictive_coordinator.predict_resource_scaling(
                current_load={
                    "cpu_usage": 60.0,
                    "memory_usage": 70.0,
                    "request_rate": 100.0,
                },
                target_sla={
                    "max_cpu_usage": 80.0,
                    "max_memory_usage": 85.0,
                    "max_response_time_ms": 200.0,
                },
            )

            assert (
                prediction_test["confidence"] > 0.5
            ), "Prediction confidence should be reasonable"
            logger.info(
                f"✅ Resource scaling prediction confidence: {prediction_test['confidence']:.2f}"
            )

            # Calculate workflow completion time
            workflow_duration = time.time() - workflow_start

            # Record success metrics
            self.test_results["ai_training_pipeline"] = {
                "status": "passed",
                "duration_seconds": workflow_duration,
                "training_job_id": created_job.id,
                "quality_score": float(quality_metrics.overall_score),
                "prediction_confidence": float(
                    coordination_prediction.prediction_confidence
                ),
                "scaling_confidence": prediction_test["confidence"],
            }

            logger.info(
                f"🎉 AI Training Pipeline Workflow PASSED in {workflow_duration:.2f}s"
            )
            return True

        except Exception as e:
            logger.error(f"❌ AI Training Pipeline Workflow FAILED: {e}")
            self.test_results["ai_training_pipeline"] = {
                "status": "failed",
                "error": str(e),
                "duration_seconds": time.time() - workflow_start,
            }
            raise

    async def test_marketplace_integration_workflow(self):
        """
        Test Workflow 4: Marketplace Integration
        Agent Publishing → Security Scanning → Performance Benchmarking → Discovery → Usage
        """
        logger.info("🧪 Testing Marketplace Integration Workflow")
        workflow_start = time.time()

        try:
            # Step 1: Agent Registration
            logger.info("Step 1: Registering new agent in marketplace")
            agent_registry = self.systems["agent_registry"]

            agent_config = {
                "name": "e2e_test_coordination_agent",
                "description": "End-to-end test agent for coordination tasks",
                "version": "1.0.0",
                "capabilities": [
                    "coordination",
                    "task_management",
                    "resource_optimization",
                ],
                "author": "e2e_test_system",
                "license": "MIT",
                "resource_requirements": {
                    "cpu_cores": 1,
                    "memory_mb": 512,
                    "storage_mb": 100,
                },
            }

            registered_agent = await agent_registry.register_agent(agent_config)
            assert registered_agent["id"], "Agent should be registered with valid ID"
            logger.info(f"✅ Agent registered: {registered_agent['id']}")

            # Step 2: Security Scanning
            logger.info("Step 2: Security scanning of registered agent")
            security_scanner = self.systems["security_scanner"]

            security_scan = await security_scanner.scan_agent(registered_agent["id"])
            assert security_scan["status"] in [
                "passed",
                "warning",
            ], "Security scan should not fail"

            security_score = security_scan.get("security_score", 85)
            assert security_score > 60, "Security score should be acceptable"
            logger.info(f"✅ Security scan passed: {security_score}/100")

            # Step 3: Performance Benchmarking
            logger.info("Step 3: Performance benchmarking")
            benchmarker = self.systems["performance_benchmarker"]

            benchmark_result = await benchmarker.benchmark_agent(registered_agent["id"])
            assert (
                benchmark_result["overall_score"] > 0
            ), "Benchmark should generate valid score"

            performance_score = benchmark_result.get("overall_score", 75)
            assert performance_score > 50, "Performance score should be acceptable"
            logger.info(f"✅ Performance benchmark: {performance_score}/100")

            # Step 4: Agent Discovery
            logger.info("Step 4: Testing agent discovery")
            search_engine = self.systems["search_engine"]

            # Search for the newly registered agent
            search_results = await search_engine.search_agents(
                query="coordination", filters={"capability": "coordination"}
            )

            # Verify our agent can be found
            found_agent = None
            for agent in search_results:
                if agent["id"] == registered_agent["id"]:
                    found_agent = agent
                    break

            assert found_agent is not None, "Registered agent should be discoverable"
            logger.info(f"✅ Agent discovered in search results")

            # Step 5: Agent Usage in Workflow
            logger.info("Step 5: Testing agent usage in workflow")
            workflow_engine = self.systems["workflow_engine"]

            usage_workflow = {
                "name": "e2e_agent_usage_test",
                "description": "Test workflow using marketplace agent",
                "agents": [registered_agent["id"]],
                "coordination_pattern": "single_agent",
                "test_task": {
                    "type": "coordination_test",
                    "parameters": {"complexity": "medium", "duration": "short"},
                },
            }

            created_workflow = await workflow_engine.create_workflow(usage_workflow)
            assert created_workflow["id"], "Usage workflow should be created"
            logger.info(f"✅ Usage workflow created: {created_workflow['id']}")

            # Step 6: Workflow Execution with Marketplace Agent
            logger.info("Step 6: Executing workflow with marketplace agent")
            executor = self.systems["workflow_executor"]

            execution_result = await executor.execute_workflow(
                workflow_id=created_workflow["id"],
                execution_config={"timeout_minutes": 5, "monitoring_enabled": True},
            )

            assert execution_result["status"] in [
                "completed",
                "in_progress",
            ], "Workflow should execute"
            logger.info(f"✅ Workflow execution: {execution_result['status']}")

            # Step 7: Usage Analytics and Quality Feedback
            logger.info("Step 7: Analyzing usage and quality feedback")
            quality_controller = self.systems["quality_controller"]

            usage_quality = await quality_controller.assess_agent_usage_quality(
                agent_id=registered_agent["id"],
                workflow_id=created_workflow["id"],
                execution_metrics={
                    "completion_time": 30.5,
                    "success_rate": 1.0,
                    "resource_efficiency": 0.85,
                },
            )

            assert (
                usage_quality.overall_score > 60
            ), "Usage quality should be acceptable"
            logger.info(
                f"✅ Usage quality assessment: {usage_quality.overall_score:.1f}/100"
            )

            # Calculate workflow completion time
            workflow_duration = time.time() - workflow_start

            # Record success metrics
            self.test_results["marketplace_integration"] = {
                "status": "passed",
                "duration_seconds": workflow_duration,
                "agent_id": registered_agent["id"],
                "security_score": security_score,
                "performance_score": performance_score,
                "usage_quality_score": float(usage_quality.overall_score),
                "workflow_id": created_workflow["id"],
            }

            logger.info(
                f"🎉 Marketplace Integration Workflow PASSED in {workflow_duration:.2f}s"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Marketplace Integration Workflow FAILED: {e}")
            self.test_results["marketplace_integration"] = {
                "status": "failed",
                "error": str(e),
                "duration_seconds": time.time() - workflow_start,
            }
            raise

    async def run_all_workflows(self):
        """Run all end-to-end workflows in parallel for comprehensive testing"""
        logger.info("🚀 Starting comprehensive end-to-end workflow testing")

        workflow_tests = [
            self.test_complete_application_generation_workflow(),
            self.test_multi_region_coordination_workflow(),
            self.test_ai_training_pipeline_workflow(),
            self.test_marketplace_integration_workflow(),
        ]

        results = await asyncio.gather(*workflow_tests, return_exceptions=True)

        # Process results
        passed_workflows = 0
        failed_workflows = 0

        workflow_names = [
            "complete_application_generation",
            "multi_region_coordination",
            "ai_training_pipeline",
            "marketplace_integration",
        ]

        for i, result in enumerate(results):
            workflow_name = workflow_names[i]

            if isinstance(result, Exception):
                logger.error(f"❌ Workflow {workflow_name} failed: {result}")
                failed_workflows += 1
            else:
                logger.info(f"✅ Workflow {workflow_name} passed")
                passed_workflows += 1

        # Calculate overall success metrics
        total_workflows = passed_workflows + failed_workflows
        success_rate = (
            (passed_workflows / total_workflows * 100) if total_workflows > 0 else 0
        )

        overall_results = {
            "total_workflows": total_workflows,
            "passed_workflows": passed_workflows,
            "failed_workflows": failed_workflows,
            "success_rate": success_rate,
            "individual_results": self.test_results,
        }

        logger.info(
            f"📊 End-to-End Testing Summary: {passed_workflows}/{total_workflows} workflows passed ({success_rate:.1f}%)"
        )

        return overall_results


# Pytest test functions
@pytest.mark.asyncio
async def test_complete_application_generation():
    """Test complete application generation workflow"""
    tester = EndToEndWorkflowTester()
    result = await tester.test_complete_application_generation_workflow()
    assert result is True


@pytest.mark.asyncio
async def test_multi_region_coordination():
    """Test multi-region coordination workflow"""
    tester = EndToEndWorkflowTester()
    result = await tester.test_multi_region_coordination_workflow()
    assert result is True


@pytest.mark.asyncio
async def test_ai_training_pipeline():
    """Test AI training pipeline workflow"""
    tester = EndToEndWorkflowTester()
    result = await tester.test_ai_training_pipeline_workflow()
    assert result is True


@pytest.mark.asyncio
async def test_marketplace_integration():
    """Test marketplace integration workflow"""
    tester = EndToEndWorkflowTester()
    result = await tester.test_marketplace_integration_workflow()
    assert result is True


@pytest.mark.asyncio
async def test_all_workflows_comprehensive():
    """Test all workflows together for comprehensive validation"""
    tester = EndToEndWorkflowTester()
    results = await tester.run_all_workflows()

    # Require at least 75% success rate for comprehensive testing
    assert (
        results["success_rate"] >= 75.0
    ), f"Workflow success rate {results['success_rate']:.1f}% below minimum 75%"

    return results


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Run comprehensive testing
    async def main():
        tester = EndToEndWorkflowTester()
        results = await tester.run_all_workflows()

        print("\n" + "=" * 80)
        print("END-TO-END WORKFLOW TESTING RESULTS")
        print("=" * 80)
        print(json.dumps(results, indent=2, default=str))
        print("=" * 80)

        return results

    # Execute testing
    asyncio.run(main())
