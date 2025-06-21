"""
Swarm Coordination Latency Benchmark
Measures and compares coordination latency between original and optimized implementations
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from datetime import datetime, timezone

from .swarm_intelligence import (
    SwarmAgent,
    SwarmObjective,
    SwarmStrategy,
    SwarmIntelligence,
)
from .swarm_coordination_patterns import (
    HierarchicalSwarmCoordinator,
    MeshSwarmCoordinator,
)
from .swarm_coordination_optimized import (
    UltraFastSwarmCoordinator,
)
from nexus_forge.core.monitoring import get_logger

logger = get_logger(__name__)


class SwarmLatencyBenchmark:
    """Benchmark suite for swarm coordination latency"""
    
    def __init__(self):
        self.results = []
        self.test_scenarios = [
            ("small_swarm", 5, 10),    # 5 agents, 10 tasks
            ("medium_swarm", 20, 50),  # 20 agents, 50 tasks
            ("large_swarm", 50, 200),  # 50 agents, 200 tasks
            ("xlarge_swarm", 100, 500), # 100 agents, 500 tasks
        ]
        
    async def run_all_benchmarks(self) -> pd.DataFrame:
        """Run all benchmark scenarios"""
        logger.info("Starting swarm coordination latency benchmarks...")
        
        for scenario_name, num_agents, num_tasks in self.test_scenarios:
            logger.info(f"\nRunning scenario: {scenario_name} ({num_agents} agents, {num_tasks} tasks)")
            
            # Test original coordinators
            await self._benchmark_coordinator(
                "hierarchical_original",
                scenario_name,
                num_agents,
                num_tasks,
                self._create_hierarchical_original
            )
            
            await self._benchmark_coordinator(
                "mesh_original",
                scenario_name,
                num_agents,
                num_tasks,
                self._create_mesh_original
            )
            
            # Test optimized coordinator
            await self._benchmark_coordinator(
                "ultra_fast_optimized",
                scenario_name,
                num_agents,
                num_tasks,
                self._create_ultra_fast
            )
            
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Calculate improvements
        self._calculate_improvements(df)
        
        return df
        
    async def _benchmark_coordinator(
        self,
        coordinator_type: str,
        scenario_name: str,
        num_agents: int,
        num_tasks: int,
        coordinator_factory
    ):
        """Benchmark a specific coordinator"""
        # Create swarm and agents
        swarm = SwarmIntelligence()
        agents = self._create_agents(num_agents)
        
        for agent in agents:
            swarm.agents[agent.id] = agent
            
        # Create objective
        objective = SwarmObjective(
            description=f"Benchmark objective for {scenario_name}",
            strategy=SwarmStrategy.DEVELOPMENT,
            priority=8,
            metadata={"expected_tasks": num_tasks}
        )
        
        # Create coordinator
        coordinator = coordinator_factory(swarm)
        
        # Warm up
        logger.info(f"Warming up {coordinator_type}...")
        await self._warmup_coordinator(coordinator, objective, agents[:2])
        
        # Run multiple iterations
        iterations = 5
        latencies = []
        message_counts = []
        throughputs = []
        
        for i in range(iterations):
            logger.info(f"  Iteration {i+1}/{iterations}")
            
            # Clear previous state
            swarm.tasks.clear()
            for agent in agents:
                agent.completed_tasks.clear()
                agent.status = "idle"
                agent.load = 0.0
                
            # Measure execution
            start_time = time.perf_counter()
            
            result = await coordinator.execute(objective, agents, {})
            
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            latencies.append(execution_time_ms)
            
            # Extract metrics from result
            if hasattr(coordinator, 'coordination_metrics'):
                metrics = coordinator.coordination_metrics
                if 'throughput_msg_s' in metrics:
                    throughputs.append(metrics['throughput_msg_s'])
                    
            if hasattr(coordinator, 'comm') and hasattr(coordinator.comm, 'router'):
                message_counts.append(coordinator.comm.router.metrics.get('messages_routed', 0))
            else:
                message_counts.append(0)
                
        # Record results
        result_entry = {
            'coordinator_type': coordinator_type,
            'scenario': scenario_name,
            'num_agents': num_agents,
            'num_tasks': num_tasks,
            'avg_latency_ms': np.mean(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p99_latency_ms': np.percentile(latencies, 99),
            'std_latency_ms': np.std(latencies),
            'avg_messages': np.mean(message_counts) if message_counts else 0,
            'avg_throughput_msg_s': np.mean(throughputs) if throughputs else 0,
        }
        
        self.results.append(result_entry)
        
        logger.info(f"  Average latency: {result_entry['avg_latency_ms']:.2f}ms")
        logger.info(f"  P99 latency: {result_entry['p99_latency_ms']:.2f}ms")
        
    async def _warmup_coordinator(self, coordinator, objective, agents):
        """Warm up coordinator to ensure fair comparison"""
        try:
            await asyncio.wait_for(
                coordinator.execute(objective, agents, {}),
                timeout=5.0
            )
        except:
            pass  # Warmup errors are ok
            
    def _create_agents(self, count: int) -> List[SwarmAgent]:
        """Create test agents"""
        agents = []
        
        capabilities_pool = [
            ["code_generation", "testing"],
            ["web_search", "analysis"],
            ["data_processing", "visualization"],
            ["optimization", "debugging"],
            ["documentation", "review"],
        ]
        
        for i in range(count):
            agent = SwarmAgent(
                name=f"agent_{i}",
                type="benchmark_agent",
                capabilities=capabilities_pool[i % len(capabilities_pool)],
                performance_score=0.8 + (i % 5) * 0.04  # 0.8 to 0.96
            )
            agents.append(agent)
            
        return agents
        
    def _create_hierarchical_original(self, swarm):
        """Create original hierarchical coordinator"""
        return HierarchicalSwarmCoordinator(swarm)
        
    def _create_mesh_original(self, swarm):
        """Create original mesh coordinator"""
        return MeshSwarmCoordinator(swarm)
        
    def _create_ultra_fast(self, swarm):
        """Create optimized ultra-fast coordinator"""
        return UltraFastSwarmCoordinator(swarm)
        
    def _calculate_improvements(self, df: pd.DataFrame):
        """Calculate performance improvements"""
        logger.info("\n=== Performance Improvements ===")
        
        scenarios = df['scenario'].unique()
        
        for scenario in scenarios:
            scenario_data = df[df['scenario'] == scenario]
            
            # Get baseline (hierarchical original)
            baseline = scenario_data[scenario_data['coordinator_type'] == 'hierarchical_original'].iloc[0]
            optimized = scenario_data[scenario_data['coordinator_type'] == 'ultra_fast_optimized'].iloc[0]
            
            # Calculate improvements
            latency_improvement = (baseline['avg_latency_ms'] - optimized['avg_latency_ms']) / baseline['avg_latency_ms'] * 100
            p99_improvement = (baseline['p99_latency_ms'] - optimized['p99_latency_ms']) / baseline['p99_latency_ms'] * 100
            
            logger.info(f"\n{scenario}:")
            logger.info(f"  Baseline avg latency: {baseline['avg_latency_ms']:.2f}ms")
            logger.info(f"  Optimized avg latency: {optimized['avg_latency_ms']:.2f}ms")
            logger.info(f"  Improvement: {latency_improvement:.1f}%")
            logger.info(f"  P99 improvement: {p99_improvement:.1f}%")
            
            # Check if we meet <50ms target
            if optimized['avg_latency_ms'] < 50:
                logger.info("  ✓ MEETS <50ms TARGET")
            else:
                logger.info("  ✗ Does not meet <50ms target")
                
    def generate_report(self, df: pd.DataFrame) -> str:
        """Generate detailed benchmark report"""
        report = []
        report.append("# Swarm Coordination Latency Benchmark Report")
        report.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        report.append("\n## Executive Summary")
        
        # Overall improvements
        baseline_avg = df[df['coordinator_type'] == 'hierarchical_original']['avg_latency_ms'].mean()
        optimized_avg = df[df['coordinator_type'] == 'ultra_fast_optimized']['avg_latency_ms'].mean()
        overall_improvement = (baseline_avg - optimized_avg) / baseline_avg * 100
        
        report.append(f"- Overall latency improvement: {overall_improvement:.1f}%")
        report.append(f"- Baseline average: {baseline_avg:.2f}ms")
        report.append(f"- Optimized average: {optimized_avg:.2f}ms")
        
        # Target achievement
        scenarios_meeting_target = df[
            (df['coordinator_type'] == 'ultra_fast_optimized') & 
            (df['avg_latency_ms'] < 50)
        ]['scenario'].tolist()
        
        report.append(f"- Scenarios meeting <50ms target: {len(scenarios_meeting_target)}/{len(df['scenario'].unique())}")
        
        # Detailed results
        report.append("\n## Detailed Results")
        
        for scenario in df['scenario'].unique():
            report.append(f"\n### {scenario}")
            scenario_data = df[df['scenario'] == scenario]
            
            for _, row in scenario_data.iterrows():
                report.append(f"\n**{row['coordinator_type']}**")
                report.append(f"- Average latency: {row['avg_latency_ms']:.2f}ms")
                report.append(f"- Min/Max: {row['min_latency_ms']:.2f}ms / {row['max_latency_ms']:.2f}ms")
                report.append(f"- P50/P99: {row['p50_latency_ms']:.2f}ms / {row['p99_latency_ms']:.2f}ms")
                report.append(f"- Messages: {row['avg_messages']:.0f}")
                
                if row['avg_throughput_msg_s'] > 0:
                    report.append(f"- Throughput: {row['avg_throughput_msg_s']:.0f} msg/s")
                    
        # Optimization techniques
        report.append("\n## Optimization Techniques Applied")
        report.append("1. **Zero-copy message passing**: Eliminated memory allocation overhead")
        report.append("2. **Lock-free data structures**: Reduced synchronization costs")
        report.append("3. **Parallel message processing**: 16 concurrent channels")
        report.append("4. **Message batching**: 5ms adaptive batching window")
        report.append("5. **Connection pooling**: Reused connections reduce setup time")
        report.append("6. **O(1) routing**: Hash-based routing tables")
        report.append("7. **Predictive scheduling**: ML-based task assignment")
        report.append("8. **Spatial indexing**: O(1) pheromone operations")
        
        return "\n".join(report)
        
    def plot_results(self, df: pd.DataFrame):
        """Plot benchmark results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Swarm Coordination Latency Benchmark Results', fontsize=16)
        
        # Plot 1: Average latency comparison
        ax1 = axes[0, 0]
        scenarios = df['scenario'].unique()
        coordinator_types = df['coordinator_type'].unique()
        
        x = np.arange(len(scenarios))
        width = 0.25
        
        for i, coord_type in enumerate(coordinator_types):
            data = df[df['coordinator_type'] == coord_type]
            values = [data[data['scenario'] == s]['avg_latency_ms'].values[0] for s in scenarios]
            ax1.bar(x + i*width, values, width, label=coord_type)
            
        ax1.axhline(y=50, color='r', linestyle='--', label='50ms target')
        ax1.set_xlabel('Scenario')
        ax1.set_ylabel('Average Latency (ms)')
        ax1.set_title('Average Latency by Scenario')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(scenarios, rotation=45)
        ax1.legend()
        ax1.set_yscale('log')
        
        # Plot 2: P99 latency
        ax2 = axes[0, 1]
        for i, coord_type in enumerate(coordinator_types):
            data = df[df['coordinator_type'] == coord_type]
            values = [data[data['scenario'] == s]['p99_latency_ms'].values[0] for s in scenarios]
            ax2.bar(x + i*width, values, width, label=coord_type)
            
        ax2.axhline(y=50, color='r', linestyle='--', label='50ms target')
        ax2.set_xlabel('Scenario')
        ax2.set_ylabel('P99 Latency (ms)')
        ax2.set_title('P99 Latency by Scenario')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(scenarios, rotation=45)
        ax2.legend()
        ax2.set_yscale('log')
        
        # Plot 3: Scaling behavior
        ax3 = axes[1, 0]
        for coord_type in coordinator_types:
            data = df[df['coordinator_type'] == coord_type]
            ax3.plot(data['num_agents'], data['avg_latency_ms'], 'o-', label=coord_type)
            
        ax3.axhline(y=50, color='r', linestyle='--', label='50ms target')
        ax3.set_xlabel('Number of Agents')
        ax3.set_ylabel('Average Latency (ms)')
        ax3.set_title('Scaling Behavior')
        ax3.legend()
        ax3.set_yscale('log')
        
        # Plot 4: Message efficiency
        ax4 = axes[1, 1]
        optimized_data = df[df['coordinator_type'] == 'ultra_fast_optimized']
        if not optimized_data.empty and optimized_data['avg_messages'].sum() > 0:
            ax4.scatter(optimized_data['avg_messages'], optimized_data['avg_latency_ms'])
            for _, row in optimized_data.iterrows():
                ax4.annotate(row['scenario'], (row['avg_messages'], row['avg_latency_ms']))
                
        ax4.set_xlabel('Average Messages')
        ax4.set_ylabel('Average Latency (ms)')
        ax4.set_title('Message Efficiency (Optimized Coordinator)')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'swarm_latency_benchmark_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Benchmark plot saved to {filename}")
        
        return fig


async def run_benchmark():
    """Run the complete benchmark suite"""
    benchmark = SwarmLatencyBenchmark()
    
    # Run benchmarks
    df = await benchmark.run_all_benchmarks()
    
    # Generate report
    report = benchmark.generate_report(df)
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f'swarm_latency_report_{timestamp}.md'
    
    with open(report_filename, 'w') as f:
        f.write(report)
        
    logger.info(f"\nBenchmark report saved to {report_filename}")
    
    # Generate plots
    try:
        benchmark.plot_results(df)
    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")
        
    # Save raw data
    csv_filename = f'swarm_latency_data_{timestamp}.csv'
    df.to_csv(csv_filename, index=False)
    logger.info(f"Raw data saved to {csv_filename}")
    
    return df, report


if __name__ == "__main__":
    # Run benchmark
    asyncio.run(run_benchmark())
