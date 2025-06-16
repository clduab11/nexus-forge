"""
Comprehensive tests for Agent Self-Improvement System
Tests reinforcement learning, collaborative debate, and automated code review
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus_forge.core.cache import RedisCache
from nexus_forge.features.agent_self_improvement import (
    AgentSelfImprovementOrchestrator,
    AutomatedCodeReviewer,
    CollaborativeDebateSystem,
    ImprovementStrategy,
    ReinforcementLearningEngine,
    get_self_improvement_orchestrator,
    start_agent_self_improvement_system,
)


@pytest.fixture
async def mock_redis_cache():
    """Mock Redis cache for testing"""
    cache = AsyncMock(spec=RedisCache)
    cache.set_l1 = AsyncMock(return_value=True)
    cache.set_l2 = AsyncMock(return_value=True)
    cache.get_l1 = AsyncMock(return_value=None)
    cache.get_l2 = AsyncMock(return_value=None)
    cache.get = AsyncMock(return_value=None)
    return cache


@pytest.fixture
async def improvement_orchestrator(mock_redis_cache):
    """Create test orchestrator instance"""
    orchestrator = AgentSelfImprovementOrchestrator()
    orchestrator.cache = mock_redis_cache
    return orchestrator


@pytest.fixture
def sample_agent_performance():
    """Sample agent performance data"""
    return {
        "agent_id": "test_agent",
        "performance_scores": [0.8, 0.85, 0.82, 0.88, 0.90],
        "task_completion_rate": 0.92,
        "error_rate": 0.05,
        "efficiency_score": 0.87,
        "learning_progress": 0.75,
    }


@pytest.fixture
def sample_code_snippet():
    """Sample code for review testing"""
    return """
async def process_user_data(user_data):
    # Process user data
    if user_data:
        return user_data.get('name', '')
    return ''
"""


class TestReinforcementLearningEngine:
    """Test reinforcement learning components"""

    @pytest.mark.asyncio
    async def test_q_learning_initialization(self, mock_redis_cache):
        """Test Q-learning engine initialization"""
        engine = ReinforcementLearningEngine(mock_redis_cache)

        assert engine.learning_rate == 0.1
        assert engine.discount_factor == 0.9
        assert engine.exploration_rate == 0.1
        assert len(engine.q_table) == 0

    @pytest.mark.asyncio
    async def test_q_learning_update(self, mock_redis_cache):
        """Test Q-learning value updates"""
        engine = ReinforcementLearningEngine(mock_redis_cache)

        # Update Q-value
        await engine.update_q_value("test_state", "test_action", 0.8, "next_state")

        # Verify Q-table updated
        assert ("test_state", "test_action") in engine.q_table
        assert engine.q_table[("test_state", "test_action")] > 0

    @pytest.mark.asyncio
    async def test_action_selection(self, mock_redis_cache):
        """Test action selection strategy"""
        engine = ReinforcementLearningEngine(mock_redis_cache)

        # Add some Q-values
        engine.q_table[("test_state", "action1")] = 0.8
        engine.q_table[("test_state", "action2")] = 0.6

        # Test action selection
        action = await engine.select_action("test_state", ["action1", "action2"])
        assert action in ["action1", "action2"]

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_learning_performance(self, mock_redis_cache):
        """Test learning performance and convergence"""
        engine = ReinforcementLearningEngine(mock_redis_cache)

        start_time = time.time()

        # Simulate learning episodes
        for i in range(100):
            state = f"state_{i % 10}"
            action = f"action_{i % 3}"
            reward = 0.8 if i % 2 == 0 else 0.6
            next_state = f"state_{(i + 1) % 10}"

            await engine.update_q_value(state, action, reward, next_state)

        learning_time = time.time() - start_time

        # Performance assertions
        assert learning_time < 1.0  # Should complete in under 1 second
        assert len(engine.q_table) > 0
        assert all(isinstance(v, float) for v in engine.q_table.values())


class TestCollaborativeDebateSystem:
    """Test collaborative debate components"""

    @pytest.mark.asyncio
    async def test_debate_initialization(self, mock_redis_cache):
        """Test debate system initialization"""
        debate_system = CollaborativeDebateSystem(mock_redis_cache)

        assert len(debate_system.debate_participants) == 3
        assert "generator" in debate_system.debate_participants
        assert "critic" in debate_system.debate_participants
        assert "moderator" in debate_system.debate_participants

    @pytest.mark.asyncio
    async def test_debate_round_execution(self, mock_redis_cache):
        """Test single debate round"""
        debate_system = CollaborativeDebateSystem(mock_redis_cache)

        proposal = {
            "type": "optimization",
            "description": "Optimize caching strategy",
            "impact_estimate": 0.15,
        }

        # Mock participant responses
        with patch.object(debate_system, "_get_participant_response") as mock_response:
            mock_response.return_value = {
                "participant": "generator",
                "response": "Implement L2 cache with TTL",
                "confidence": 0.8,
                "reasoning": "Will reduce latency",
            }

            result = await debate_system.conduct_debate_round(proposal, "test_agent")

            assert "consensus_reached" in result
            assert "final_recommendation" in result
            assert result["debate_rounds"] > 0

    @pytest.mark.asyncio
    async def test_consensus_evaluation(self, mock_redis_cache):
        """Test consensus evaluation logic"""
        debate_system = CollaborativeDebateSystem(mock_redis_cache)

        responses = [
            {"confidence": 0.9, "support": True},
            {"confidence": 0.8, "support": True},
            {"confidence": 0.7, "support": False},
        ]

        consensus = await debate_system._evaluate_consensus(responses)

        assert isinstance(consensus, dict)
        assert "consensus_reached" in consensus
        assert "confidence_score" in consensus


class TestAutomatedCodeReviewer:
    """Test automated code review components"""

    @pytest.mark.asyncio
    async def test_code_reviewer_initialization(self, mock_redis_cache):
        """Test code reviewer initialization"""
        reviewer = AutomatedCodeReviewer(mock_redis_cache)

        assert len(reviewer.review_criteria) > 0
        assert "security" in reviewer.review_criteria
        assert "performance" in reviewer.review_criteria
        assert "maintainability" in reviewer.review_criteria

    @pytest.mark.asyncio
    async def test_code_analysis(self, mock_redis_cache, sample_code_snippet):
        """Test code analysis functionality"""
        reviewer = AutomatedCodeReviewer(mock_redis_cache)

        analysis = await reviewer.analyze_code(sample_code_snippet, "test_agent")

        assert "review_id" in analysis
        assert "overall_score" in analysis
        assert "security_issues" in analysis
        assert "performance_issues" in analysis
        assert "maintainability_score" in analysis
        assert analysis["overall_score"] >= 0.0
        assert analysis["overall_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_security_analysis(self, mock_redis_cache):
        """Test security vulnerability detection"""
        reviewer = AutomatedCodeReviewer(mock_redis_cache)

        # Code with potential security issue
        insecure_code = """
def execute_query(query):
    cursor.execute(query)  # SQL injection risk
    return cursor.fetchall()
"""

        analysis = await reviewer.analyze_code(insecure_code, "test_agent")

        # Should detect security issues
        assert len(analysis["security_issues"]) > 0
        assert any(
            "injection" in issue["description"].lower()
            for issue in analysis["security_issues"]
        )

    @pytest.mark.asyncio
    async def test_performance_analysis(self, mock_redis_cache):
        """Test performance issue detection"""
        reviewer = AutomatedCodeReviewer(mock_redis_cache)

        # Code with performance issues
        slow_code = """
def process_data(data):
    result = []
    for item in data:
        for other_item in data:  # O(n²) complexity
            if item == other_item:
                result.append(item)
    return result
"""

        analysis = await reviewer.analyze_code(slow_code, "test_agent")

        # Should detect performance issues
        assert len(analysis["performance_issues"]) > 0


class TestAgentSelfImprovementOrchestrator:
    """Test main orchestrator functionality"""

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, improvement_orchestrator):
        """Test orchestrator initialization"""
        assert improvement_orchestrator.rl_engine is not None
        assert improvement_orchestrator.debate_system is not None
        assert improvement_orchestrator.code_reviewer is not None
        assert not improvement_orchestrator.active

    @pytest.mark.asyncio
    async def test_improvement_recommendation(
        self, improvement_orchestrator, sample_agent_performance
    ):
        """Test improvement recommendation generation"""
        recommendations = (
            await improvement_orchestrator.generate_improvement_recommendations(
                "test_agent", sample_agent_performance
            )
        )

        assert len(recommendations) > 0
        for rec in recommendations:
            assert "strategy" in rec
            assert "confidence" in rec
            assert "expected_improvement" in rec
            assert rec["confidence"] >= 0.0
            assert rec["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_strategy_selection(self, improvement_orchestrator):
        """Test improvement strategy selection"""
        strategies = [
            ImprovementStrategy.REINFORCEMENT_LEARNING,
            ImprovementStrategy.COLLABORATIVE_DEBATE,
            ImprovementStrategy.AUTOMATED_CODE_REVIEW,
        ]

        selected = await improvement_orchestrator._select_improvement_strategies(
            "test_agent", 0.75, strategies
        )

        assert len(selected) > 0
        assert all(isinstance(s, ImprovementStrategy) for s in selected)

    @pytest.mark.asyncio
    async def test_improvement_execution(self, improvement_orchestrator):
        """Test improvement execution workflow"""
        with patch.object(
            improvement_orchestrator.rl_engine, "optimize_agent_policy"
        ) as mock_rl:
            mock_rl.return_value = {"optimization_applied": True, "improvement": 0.05}

            result = await improvement_orchestrator.execute_improvement(
                "test_agent",
                ImprovementStrategy.REINFORCEMENT_LEARNING,
                {"learning_rate": 0.1},
            )

            assert "success" in result
            assert "improvement_metrics" in result
            mock_rl.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_improvement_cycle(
        self, improvement_orchestrator, sample_agent_performance
    ):
        """Test complete improvement cycle"""
        # Mock external dependencies
        with patch.object(
            improvement_orchestrator, "_analyze_agent_performance"
        ) as mock_analyze:
            mock_analyze.return_value = {
                "performance_score": 0.8,
                "areas_for_improvement": ["efficiency", "error_handling"],
                "learning_potential": 0.7,
            }

            result = await improvement_orchestrator.analyze_and_improve_agent(
                "test_agent", sample_agent_performance
            )

            assert "analysis_results" in result
            assert "improvement_recommendations" in result
            assert "execution_results" in result

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_improvement_system_performance(self, improvement_orchestrator):
        """Test improvement system performance under load"""
        start_time = time.time()

        # Simulate multiple concurrent improvement requests
        tasks = []
        for i in range(10):
            performance_data = {
                "agent_id": f"agent_{i}",
                "performance_scores": [0.7 + i * 0.02],
                "task_completion_rate": 0.8 + i * 0.01,
            }
            task = improvement_orchestrator.generate_improvement_recommendations(
                f"agent_{i}", performance_data
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        execution_time = time.time() - start_time

        # Performance assertions
        assert execution_time < 5.0  # Should complete in under 5 seconds
        assert len(results) == 10
        assert all(len(result) > 0 for result in results)


class TestGlobalFunctions:
    """Test global functions and integration points"""

    @pytest.mark.asyncio
    async def test_get_self_improvement_orchestrator(self):
        """Test global orchestrator singleton"""
        orchestrator1 = await get_self_improvement_orchestrator()
        orchestrator2 = await get_self_improvement_orchestrator()

        assert orchestrator1 is orchestrator2  # Should be singleton
        assert isinstance(orchestrator1, AgentSelfImprovementOrchestrator)

    @pytest.mark.asyncio
    async def test_start_improvement_system(self):
        """Test system startup function"""
        with patch(
            "nexus_forge.features.agent_self_improvement.get_self_improvement_orchestrator"
        ) as mock_get:
            mock_orchestrator = AsyncMock()
            mock_get.return_value = mock_orchestrator

            await start_agent_self_improvement_system()

            mock_orchestrator.start_self_improvement.assert_called_once()


class TestErrorHandling:
    """Test error handling and resilience"""

    @pytest.mark.asyncio
    async def test_invalid_agent_performance_data(self, improvement_orchestrator):
        """Test handling of invalid performance data"""
        invalid_data = {"invalid": "data"}

        recommendations = (
            await improvement_orchestrator.generate_improvement_recommendations(
                "test_agent", invalid_data
            )
        )

        # Should handle gracefully and return at least fallback recommendations
        assert isinstance(recommendations, list)

    @pytest.mark.asyncio
    async def test_cache_failure_resilience(self, improvement_orchestrator):
        """Test resilience to cache failures"""
        # Mock cache failure
        improvement_orchestrator.cache.set_l1.side_effect = Exception("Cache error")

        # Should still work despite cache failure
        result = await improvement_orchestrator.generate_improvement_recommendations(
            "test_agent", {"performance_scores": [0.8]}
        )

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_network_timeout_handling(self, improvement_orchestrator):
        """Test handling of network timeouts"""
        with patch.object(
            improvement_orchestrator.debate_system, "conduct_debate_round"
        ) as mock_debate:
            mock_debate.side_effect = asyncio.TimeoutError("Network timeout")

            # Should handle timeout gracefully
            result = await improvement_orchestrator.execute_improvement(
                "test_agent", ImprovementStrategy.COLLABORATIVE_DEBATE, {}
            )

            assert "success" in result
            assert not result["success"]  # Should fail gracefully


@pytest.mark.integration
class TestSystemIntegration:
    """Test integration with other systems"""

    @pytest.mark.asyncio
    async def test_performance_analytics_integration(self, improvement_orchestrator):
        """Test integration with performance analytics"""
        with patch.object(improvement_orchestrator.cache, "get_l1") as mock_get:
            mock_get.return_value = {
                "system_performance": {"avg_latency": 1200, "success_rate": 0.95}
            }

            # Should incorporate performance analytics data
            recommendations = (
                await improvement_orchestrator.generate_improvement_recommendations(
                    "test_agent", {"performance_scores": [0.8]}
                )
            )

            assert len(recommendations) > 0

    @pytest.mark.asyncio
    async def test_mem0_knowledge_integration(self, improvement_orchestrator):
        """Test integration with Mem0 knowledge graph"""
        # Mock Mem0 integration
        with patch.object(
            improvement_orchestrator, "_store_learning_patterns"
        ) as mock_store:
            mock_store.return_value = True

            await improvement_orchestrator.execute_improvement(
                "test_agent", ImprovementStrategy.REINFORCEMENT_LEARNING, {}
            )

            # Should store learning patterns in knowledge graph
            mock_store.assert_called()

    @pytest.mark.asyncio
    async def test_behavior_analysis_integration(self, improvement_orchestrator):
        """Test integration with behavior analysis"""
        with patch.object(improvement_orchestrator.cache, "get_l2") as mock_get:
            mock_get.return_value = {
                "agent_behavior_patterns": {
                    "test_agent": {
                        "collaboration_score": 0.9,
                        "efficiency_trends": [0.8, 0.85, 0.9],
                    }
                }
            }

            # Should incorporate behavior analysis insights
            result = await improvement_orchestrator.analyze_and_improve_agent(
                "test_agent", {"performance_scores": [0.8]}
            )

            assert "analysis_results" in result


if __name__ == "__main__":
    pytest.main([__file__])
