"""
Swarm Intelligence Integration Guide
Example code and integration patterns for using the swarm intelligence system
"""

import asyncio
from typing import List, Dict, Any

from nexus_forge.backend.ai_features.swarm_intelligence import (
    SwarmIntelligence,
    SwarmObjective,
    SwarmPattern,
    SwarmStrategy,
)
from nexus_forge.backend.ai_features.swarm_optimization_algorithms import (
    SwarmOptimizationManager,
    OptimizationProblem,
)
from nexus_forge.backend.ai_features.swarm_execution_engine import (
    SwarmExecutionEngine,
    ExecutionContext,
)
from nexus_forge.backend.ai_features.swarm_monitoring_dashboard import (
    SwarmMonitor,
    SwarmAnalytics,
    SwarmDashboard,
)


class SwarmIntegrationExamples:
    """Examples of how to integrate swarm intelligence into applications"""
    
    @staticmethod
    async def example_research_swarm():
        """Example: Create a research swarm to analyze a topic"""
        # Initialize swarm
        swarm = SwarmIntelligence(
            project_id="nexus-forge-demo",
            gemini_api_key="your-api-key"
        )
        
        # Define research objective
        objective = SwarmObjective(
            description="Research quantum computing applications in healthcare",
            strategy=SwarmStrategy.RESEARCH,
            success_criteria={
                "sources": 20,          # Minimum 20 sources
                "quality_score": 0.8,   # 80% quality threshold
                "coverage": 0.9,        # 90% topic coverage
                "peer_reviewed": 10     # At least 10 peer-reviewed sources
            },
            constraints={
                "time_limit": 3600,     # 1 hour
                "cost_limit": 10.0,     # $10 max cost
                "language": "en"        # English sources only
            },
            priority=8
        )
        
        # Execute with mesh pattern for collaborative research
        result = await swarm.coordinate_swarm(
            objective=objective,
            pattern=SwarmPattern.MESH,
            max_agents=15
        )
        
        # Process results
        print(f"Research completed with {result.confidence:.2%} confidence")
        print(f"Found {len(result.completed_tasks)} sources")
        print(f"Emergence patterns detected: {result.emergence_patterns}")
        
        return result
    
    @staticmethod
    async def example_development_swarm():
        """Example: Create a development swarm for building features"""
        swarm = SwarmIntelligence(project_id="nexus-forge-demo")
        
        # Define development objective
        objective = SwarmObjective(
            description="Implement user authentication system with JWT tokens",
            strategy=SwarmStrategy.DEVELOPMENT,
            success_criteria={
                "features": ["login", "logout", "refresh", "password_reset"],
                "test_coverage": 0.9,
                "security_score": 0.95,
                "performance": {"login_time_ms": 200}
            },
            constraints={
                "framework": "FastAPI",
                "database": "PostgreSQL",
                "time_limit": 7200  # 2 hours
            },
            priority=9
        )
        
        # Use hierarchical pattern for structured development
        result = await swarm.coordinate_swarm(
            objective=objective,
            pattern=SwarmPattern.HIERARCHICAL,
            max_agents=20
        )
        
        # Extract code artifacts
        code_artifacts = [
            task_result for task_id, task_result in result.results.items()
            if "code" in str(task_result)
        ]
        
        print(f"Development completed: {result.status}")
        print(f"Generated {len(code_artifacts)} code artifacts")
        
        return result
    
    @staticmethod
    async def example_optimization_swarm():
        """Example: Use swarm optimization for resource allocation"""
        # Define optimization problem
        def resource_cost_function(allocation):
            """Minimize cost while meeting constraints"""
            # allocation = [cpu_units, memory_gb, storage_gb, bandwidth_mbps]
            costs = [0.1, 0.05, 0.01, 0.02]  # Cost per unit
            total_cost = sum(allocation[i] * costs[i] for i in range(4))
            
            # Add penalties for constraint violations
            penalty = 0
            if sum(allocation) > 100:  # Total resource limit
                penalty += 100 * (sum(allocation) - 100)
            
            return total_cost + penalty
        
        problem = OptimizationProblem(
            name="Cloud Resource Allocation",
            objective_function=resource_cost_function,
            bounds=[(0, 50), (0, 32), (0, 1000), (0, 1000)],  # Min/max for each resource
            dimension=4,
            minimize=True
        )
        
        # Run multiple algorithms and compare
        optimizer = SwarmOptimizationManager()
        results = await optimizer.compare_algorithms(
            problem,
            algorithms=["PSO", "ACO", "BCO", "FA"],
            max_iterations=100,
            verbose=True
        )
        
        # Find best result
        best_algo, best_result = optimizer.get_best_algorithm(results)
        
        print(f"Best algorithm: {best_algo}")
        print(f"Optimal allocation: {best_result.best_solution}")
        print(f"Minimum cost: ${best_result.best_fitness:.2f}")
        
        return best_result
    
    @staticmethod
    async def example_adaptive_swarm():
        """Example: Adaptive swarm that changes patterns based on performance"""
        swarm = SwarmIntelligence(project_id="nexus-forge-demo")
        
        # Complex objective that may require adaptation
        objective = SwarmObjective(
            description="Analyze and optimize e-commerce recommendation engine",
            strategy=SwarmStrategy.ANALYSIS,
            success_criteria={
                "metrics_improved": ["click_through_rate", "conversion_rate"],
                "improvement_threshold": 0.15,  # 15% improvement
                "a_b_test_confidence": 0.95
            },
            constraints={
                "existing_data": True,
                "real_time_processing": True,
                "budget": 1000
            }
        )
        
        # Start with adaptive pattern
        result = await swarm.coordinate_swarm(
            objective=objective,
            pattern=SwarmPattern.ADAPTIVE,
            max_agents=25
        )
        
        # Check adaptations
        print(f"Swarm adapted {result.metadata.get('adaptations', 0)} times")
        print(f"Final pattern: {result.metadata.get('final_pattern', 'unknown')}")
        
        return result
    
    @staticmethod
    async def example_monitoring_dashboard():
        """Example: Set up real-time monitoring dashboard"""
        # Initialize monitoring components
        monitor = SwarmMonitor(buffer_size=1000)
        analytics = SwarmAnalytics()
        dashboard = SwarmDashboard(monitor, analytics)
        
        # Start a swarm with monitoring
        swarm = SwarmIntelligence(project_id="nexus-forge-demo")
        
        # Simple objective for demonstration
        objective = SwarmObjective(
            description="Monitor system health and performance",
            strategy=SwarmStrategy.MAINTENANCE,
            priority=7
        )
        
        # Monitoring callback
        async def monitoring_callback(data):
            """Process monitoring updates"""
            metrics = await monitor.collect_metrics(
                swarm_id=data.get("execution_id", "unknown"),
                agents=data.get("agents", []),
                tasks=data.get("tasks", []),
                messages=data.get("messages", []),
                pattern=SwarmPattern.DISTRIBUTED
            )
            
            # Check for alerts
            if monitor.active_alerts:
                print(f"ALERT: {monitor.active_alerts[-1]['message']}")
            
            # Update analytics
            if len(monitor.metrics_buffer) > 10:
                analysis = await analytics.analyze_swarm_performance(
                    list(monitor.metrics_buffer)
                )
                
                if analysis.get("optimization_opportunities"):
                    print("Optimization opportunities detected:")
                    for opt in analysis["optimization_opportunities"][:3]:
                        print(f"  - {opt['recommendation']}")
        
        # Execute with monitoring
        execution_engine = SwarmExecutionEngine(project_id="nexus-forge-demo")
        
        context = ExecutionContext(
            objective=objective,
            agents=await swarm._form_swarm(
                objective, 
                {"complexity": 0.5}, 
                SwarmPattern.DISTRIBUTED,
                10,
                []
            ),
            tasks=[]  # Would be populated by decomposition
        )
        
        result = await execution_engine.execute_swarm(
            context=context,
            monitoring_callback=monitoring_callback
        )
        
        # Generate final report
        report = await analytics.generate_report(
            swarm_id="demo-swarm",
            metrics_history=list(monitor.metrics_buffer)
        )
        
        print(f"Swarm health: {report['executive_summary']['overall_health']}")
        print(f"Key recommendations: {len(report['recommendations'])}")
        
        return report
    
    @staticmethod
    async def example_distributed_processing():
        """Example: Distributed data processing with stigmergic coordination"""
        swarm = SwarmIntelligence(project_id="nexus-forge-demo")
        
        # Data processing objective
        objective = SwarmObjective(
            description="Process and analyze 1TB of log data for anomalies",
            strategy=SwarmStrategy.ANALYSIS,
            success_criteria={
                "data_processed": "100%",
                "anomalies_detected": True,
                "false_positive_rate": 0.05
            },
            constraints={
                "parallel_processing": True,
                "memory_per_agent": "4GB",
                "completion_time": 3600  # 1 hour
            }
        )
        
        # Use distributed pattern with stigmergic coordination
        result = await swarm.coordinate_swarm(
            objective=objective,
            pattern=SwarmPattern.DISTRIBUTED,
            max_agents=50  # Large swarm for parallel processing
        )
        
        # Check pheromone trails (indicating processing patterns)
        pheromone_data = result.metadata.get("pheromone_trails", {})
        
        print(f"Processing completed: {result.status}")
        print(f"Anomalies found: {len(result.results.get('anomalies', []))}")
        print(f"Pheromone trails formed: {pheromone_data.get('trails_detected', False)}")
        
        return result


class SwarmIntegrationPatterns:
    """Common integration patterns for swarm intelligence"""
    
    @staticmethod
    async def pattern_map_reduce_swarm(data_chunks: List[Any], map_func, reduce_func):
        """Map-Reduce pattern using swarm intelligence"""
        swarm = SwarmIntelligence(project_id="nexus-forge-demo")
        
        # Create objective for map-reduce
        objective = SwarmObjective(
            description=f"Process {len(data_chunks)} data chunks using map-reduce",
            strategy=SwarmStrategy.ANALYSIS,
            metadata={
                "map_function": map_func.__name__,
                "reduce_function": reduce_func.__name__,
                "data_chunks": data_chunks
            }
        )
        
        # Use mesh pattern for distributed processing
        result = await swarm.coordinate_swarm(
            objective=objective,
            pattern=SwarmPattern.MESH,
            max_agents=min(len(data_chunks), 20)
        )
        
        # Extract and reduce results
        mapped_results = [
            result.results.get(f"chunk_{i}", None)
            for i in range(len(data_chunks))
        ]
        
        final_result = reduce_func(mapped_results)
        
        return final_result
    
    @staticmethod
    async def pattern_pipeline_swarm(stages: List[Dict[str, Any]]):
        """Pipeline pattern with different swarm patterns per stage"""
        results = []
        
        for i, stage in enumerate(stages):
            swarm = SwarmIntelligence(project_id="nexus-forge-demo")
            
            # Create stage objective
            objective = SwarmObjective(
                description=stage["description"],
                strategy=stage.get("strategy", SwarmStrategy.ANALYSIS),
                metadata={
                    "stage": i + 1,
                    "total_stages": len(stages),
                    "previous_results": results[-1] if results else None
                }
            )
            
            # Execute stage with specified pattern
            result = await swarm.coordinate_swarm(
                objective=objective,
                pattern=stage.get("pattern", SwarmPattern.HIERARCHICAL),
                max_agents=stage.get("max_agents", 10)
            )
            
            results.append(result)
            
            # Check if stage failed
            if result.status == "failed":
                print(f"Pipeline failed at stage {i + 1}: {stage['description']}")
                break
        
        return results
    
    @staticmethod
    async def pattern_consensus_swarm(proposals: List[Dict[str, Any]]):
        """Consensus pattern for decision making"""
        swarm = SwarmIntelligence(project_id="nexus-forge-demo")
        
        # Create consensus objective
        objective = SwarmObjective(
            description="Reach consensus on proposals through swarm intelligence",
            strategy=SwarmStrategy.ANALYSIS,
            metadata={
                "proposals": proposals,
                "consensus_threshold": 0.75
            }
        )
        
        # Use mesh pattern for peer-to-peer consensus
        result = await swarm.coordinate_swarm(
            objective=objective,
            pattern=SwarmPattern.MESH,
            max_agents=len(proposals) * 2  # Multiple agents per proposal
        )
        
        # Extract consensus decision
        consensus = result.results.get("consensus_decision", None)
        confidence = result.results.get("consensus_confidence", 0.0)
        
        return {
            "decision": consensus,
            "confidence": confidence,
            "dissenting_opinions": result.results.get("dissenting", [])
        }


# Example usage
async def main():
    """Run integration examples"""
    print("=== Swarm Intelligence Integration Examples ===\n")
    
    # Example 1: Research Swarm
    print("1. Research Swarm Example")
    try:
        research_result = await SwarmIntegrationExamples.example_research_swarm()
        print("   Research completed successfully\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # Example 2: Optimization
    print("2. Optimization Swarm Example")
    try:
        optimization_result = await SwarmIntegrationExamples.example_optimization_swarm()
        print("   Optimization completed successfully\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # Example 3: Map-Reduce Pattern
    print("3. Map-Reduce Pattern Example")
    try:
        data = list(range(100))
        result = await SwarmIntegrationPatterns.pattern_map_reduce_swarm(
            data_chunks=[data[i:i+10] for i in range(0, 100, 10)],
            map_func=lambda x: sum(x),
            reduce_func=lambda x: sum(x)
        )
        print(f"   Map-Reduce result: {result}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    print("Integration examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
