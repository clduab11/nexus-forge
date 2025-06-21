"""
Performance Benchmarking Suite
Measures throughput and performance improvements in nexus-forge
"""

import asyncio
import json
import time
import statistics
from datetime import datetime, timezone
from typing import Dict, Any
import numpy as np
import psutil

# Import both original and optimized engines
from src.backend.ai_features.swarm_execution_engine import (
    SwarmExecutionEngine as OriginalEngine,
    ExecutionContext,
    SwarmTask,
    SwarmAgent,
    SwarmObjective
)
from src.backend.ai_features.swarm_execution_optimized import (
    OptimizedSwarmExecutionEngine
)


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_info": self._get_system_info(),
            "benchmarks": {}
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context"""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else "N/A",
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "platform": {
                "system": psutil.os.name,
                "python_version": psutil.sys.version.split()[0]
            }
        }
    
    async def run_all_benchmarks(self):
        """Run all performance benchmarks"""
        print("Starting Performance Benchmark Suite...")
        print(f"System: {self.results['system_info']['cpu_count']} CPUs, "
              f"{self.results['system_info']['memory_gb']}GB RAM")
        print("-" * 60)
        
        # Benchmark 1: Throughput Test
        await self.benchmark_throughput()
        
        # Benchmark 2: Latency Test
        await self.benchmark_latency()
        
        # Benchmark 3: Concurrency Test
        await self.benchmark_concurrency()
        
        # Benchmark 4: Resource Efficiency Test
        await self.benchmark_resource_efficiency()
        
        # Benchmark 5: Cache Performance Test
        await self.benchmark_cache_performance()
        
        # Save results
        self._save_results()
        self._print_summary()
    
    async def benchmark_throughput(self):
        """Benchmark task throughput"""
        print("\n1. THROUGHPUT BENCHMARK")
        print("-" * 40)
        
        task_counts = [100, 500, 1000, 5000]
        results = {"original": {}, "optimized": {}}
        
        for count in task_counts:
            print(f"\nTesting with {count} tasks...")
            
            # Create test context
            context = self._create_test_context(count)
            
            # Test original engine
            original_engine = OriginalEngine("benchmark_project")
            start_time = time.time()
            
            try:
                result = await asyncio.wait_for(
                    original_engine.execute_swarm(context),
                    timeout=300  # 5 minute timeout
                )
                original_time = time.time() - start_time
                original_throughput = count / original_time
                
                results["original"][count] = {
                    "execution_time": original_time,
                    "throughput": original_throughput,
                    "success_rate": len(result.completed_tasks) / count
                }
                
                print(f"  Original: {original_throughput:.2f} tasks/sec")
            except asyncio.TimeoutError:
                print("  Original: TIMEOUT")
                results["original"][count] = {"error": "timeout"}
            finally:
                await original_engine.shutdown()
            
            # Test optimized engine
            optimized_engine = OptimizedSwarmExecutionEngine("benchmark_project")
            start_time = time.time()
            
            try:
                result = await asyncio.wait_for(
                    optimized_engine.execute_swarm(context),
                    timeout=300
                )
                optimized_time = time.time() - start_time
                optimized_throughput = count / optimized_time
                
                results["optimized"][count] = {
                    "execution_time": optimized_time,
                    "throughput": optimized_throughput,
                    "success_rate": len(result.completed_tasks) / count,
                    "cache_hit_rate": result.performance_metrics.get("cache_hit_rate", 0)
                }
                
                print(f"  Optimized: {optimized_throughput:.2f} tasks/sec")
                
                # Calculate improvement
                if count in results["original"] and "throughput" in results["original"][count]:
                    improvement = (optimized_throughput / original_throughput - 1) * 100
                    print(f"  Improvement: {improvement:.1f}%")
                    
            except asyncio.TimeoutError:
                print("  Optimized: TIMEOUT")
                results["optimized"][count] = {"error": "timeout"}
            finally:
                await optimized_engine.shutdown()
        
        self.results["benchmarks"]["throughput"] = results
    
    async def benchmark_latency(self):
        """Benchmark task latency"""
        print("\n2. LATENCY BENCHMARK")
        print("-" * 40)
        
        iterations = 100
        results = {"original": [], "optimized": []}
        
        print(f"Running {iterations} individual task executions...")
        
        for i in range(iterations):
            # Single task context
            context = self._create_test_context(1)
            
            # Test original
            original_engine = OriginalEngine("benchmark_project")
            start_time = time.time()
            
            try:
                await original_engine.execute_swarm(context)
                latency = (time.time() - start_time) * 1000  # Convert to ms
                results["original"].append(latency)
            except Exception:
                results["original"].append(None)
            finally:
                await original_engine.shutdown()
            
            # Test optimized
            optimized_engine = OptimizedSwarmExecutionEngine("benchmark_project")
            start_time = time.time()
            
            try:
                await optimized_engine.execute_swarm(context)
                latency = (time.time() - start_time) * 1000  # Convert to ms
                results["optimized"].append(latency)
            except Exception:
                results["optimized"].append(None)
            finally:
                await optimized_engine.shutdown()
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{iterations}")
        
        # Calculate statistics
        original_latencies = [l for l in results["original"] if l is not None]
        optimized_latencies = [l for l in results["optimized"] if l is not None]
        
        self.results["benchmarks"]["latency"] = {
            "original": {
                "mean": statistics.mean(original_latencies) if original_latencies else 0,
                "median": statistics.median(original_latencies) if original_latencies else 0,
                "p95": np.percentile(original_latencies, 95) if original_latencies else 0,
                "p99": np.percentile(original_latencies, 99) if original_latencies else 0,
            },
            "optimized": {
                "mean": statistics.mean(optimized_latencies) if optimized_latencies else 0,
                "median": statistics.median(optimized_latencies) if optimized_latencies else 0,
                "p95": np.percentile(optimized_latencies, 95) if optimized_latencies else 0,
                "p99": np.percentile(optimized_latencies, 99) if optimized_latencies else 0,
            }
        }
        
        print("\nLatency Results:")
        print(f"  Original - Mean: {self.results['benchmarks']['latency']['original']['mean']:.2f}ms, "
              f"P95: {self.results['benchmarks']['latency']['original']['p95']:.2f}ms")
        print(f"  Optimized - Mean: {self.results['benchmarks']['latency']['optimized']['mean']:.2f}ms, "
              f"P95: {self.results['benchmarks']['latency']['optimized']['p95']:.2f}ms")
        
        if original_latencies and optimized_latencies:
            improvement = (1 - self.results['benchmarks']['latency']['optimized']['mean'] / 
                          self.results['benchmarks']['latency']['original']['mean']) * 100
            print(f"  Improvement: {improvement:.1f}% reduction")
    
    async def benchmark_concurrency(self):
        """Benchmark concurrent execution scaling"""
        print("\n3. CONCURRENCY BENCHMARK")
        print("-" * 40)
        
        agent_counts = [1, 5, 10, 20, 50]
        results = {"original": {}, "optimized": {}}
        
        for agent_count in agent_counts:
            print(f"\nTesting with {agent_count} agents...")
            
            # Create context with multiple agents
            context = self._create_test_context(1000, agent_count)
            
            # Test original
            original_engine = OriginalEngine("benchmark_project")
            start_time = time.time()
            
            try:
                await original_engine.execute_swarm(context)
                original_time = time.time() - start_time
                results["original"][agent_count] = {
                    "execution_time": original_time,
                    "throughput": 1000 / original_time
                }
                print(f"  Original: {original_time:.2f}s")
            except Exception as e:
                print(f"  Original: ERROR - {str(e)}")
                results["original"][agent_count] = {"error": str(e)}
            finally:
                await original_engine.shutdown()
            
            # Test optimized
            optimized_engine = OptimizedSwarmExecutionEngine("benchmark_project")
            start_time = time.time()
            
            try:
                await optimized_engine.execute_swarm(context)
                optimized_time = time.time() - start_time
                results["optimized"][agent_count] = {
                    "execution_time": optimized_time,
                    "throughput": 1000 / optimized_time
                }
                print(f"  Optimized: {optimized_time:.2f}s")
                
                if agent_count in results["original"] and "execution_time" in results["original"][agent_count]:
                    speedup = results["original"][agent_count]["execution_time"] / optimized_time
                    print(f"  Speedup: {speedup:.2f}x")
                    
            except Exception as e:
                print(f"  Optimized: ERROR - {str(e)}")
                results["optimized"][agent_count] = {"error": str(e)}
            finally:
                await optimized_engine.shutdown()
        
        self.results["benchmarks"]["concurrency"] = results
    
    async def benchmark_resource_efficiency(self):
        """Benchmark resource usage efficiency"""
        print("\n4. RESOURCE EFFICIENCY BENCHMARK")
        print("-" * 40)
        
        # Monitor resource usage during execution
        context = self._create_test_context(5000)
        results = {"original": {}, "optimized": {}}
        
        # Test original
        print("\nMonitoring original engine resources...")
        original_engine = OriginalEngine("benchmark_project")
        
        initial_cpu = psutil.cpu_percent(interval=0.1)
        initial_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        
        start_time = time.time()
        
        try:
            # Monitor during execution
            cpu_samples = []
            memory_samples = []
            
            async def monitor():
                while True:
                    cpu_samples.append(psutil.cpu_percent(interval=0.1))
                    memory_samples.append(psutil.Process().memory_info().rss / (1024**2))
                    await asyncio.sleep(0.5)
            
            monitor_task = asyncio.create_task(monitor())
            
            await original_engine.execute_swarm(context)
            execution_time = time.time() - start_time
            
            monitor_task.cancel()
            
            results["original"] = {
                "execution_time": execution_time,
                "avg_cpu": statistics.mean(cpu_samples) if cpu_samples else 0,
                "peak_cpu": max(cpu_samples) if cpu_samples else 0,
                "avg_memory": statistics.mean(memory_samples) if memory_samples else 0,
                "peak_memory": max(memory_samples) if memory_samples else 0,
            }
            
            print(f"  CPU - Avg: {results['original']['avg_cpu']:.1f}%, "
                  f"Peak: {results['original']['peak_cpu']:.1f}%")
            print(f"  Memory - Avg: {results['original']['avg_memory']:.1f}MB, "
                  f"Peak: {results['original']['peak_memory']:.1f}MB")
                  
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results["original"] = {"error": str(e)}
        finally:
            await original_engine.shutdown()
        
        # Test optimized
        print("\nMonitoring optimized engine resources...")
        optimized_engine = OptimizedSwarmExecutionEngine("benchmark_project")
        
        initial_cpu = psutil.cpu_percent(interval=0.1)
        initial_memory = psutil.Process().memory_info().rss / (1024**2)
        
        start_time = time.time()
        
        try:
            cpu_samples = []
            memory_samples = []
            
            monitor_task = asyncio.create_task(monitor())
            
            await optimized_engine.execute_swarm(context)
            execution_time = time.time() - start_time
            
            monitor_task.cancel()
            
            results["optimized"] = {
                "execution_time": execution_time,
                "avg_cpu": statistics.mean(cpu_samples) if cpu_samples else 0,
                "peak_cpu": max(cpu_samples) if cpu_samples else 0,
                "avg_memory": statistics.mean(memory_samples) if memory_samples else 0,
                "peak_memory": max(memory_samples) if memory_samples else 0,
            }
            
            print(f"  CPU - Avg: {results['optimized']['avg_cpu']:.1f}%, "
                  f"Peak: {results['optimized']['peak_cpu']:.1f}%")
            print(f"  Memory - Avg: {results['optimized']['avg_memory']:.1f}MB, "
                  f"Peak: {results['optimized']['peak_memory']:.1f}MB")
            
            # Calculate efficiency improvement
            if "avg_cpu" in results["original"] and results["original"]["avg_cpu"] > 0:
                cpu_efficiency = (1 - results["optimized"]["avg_cpu"] / results["original"]["avg_cpu"]) * 100
                print(f"  CPU Efficiency Improvement: {cpu_efficiency:.1f}%")
                
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results["optimized"] = {"error": str(e)}
        finally:
            await optimized_engine.shutdown()
        
        self.results["benchmarks"]["resource_efficiency"] = results
    
    async def benchmark_cache_performance(self):
        """Benchmark cache performance impact"""
        print("\n5. CACHE PERFORMANCE BENCHMARK")
        print("-" * 40)
        
        # Create context with repeated tasks
        tasks = []
        for i in range(100):
            for j in range(10):  # 10 copies of each task
                task = SwarmTask(
                    id=f"task_{i}_{j}",
                    type="compute",
                    data={"value": i},
                    priority=5
                )
                tasks.append(task)
        
        context = ExecutionContext(
            objective=SwarmObjective(id="cache_test", description="Cache performance test"),
            agents=[self._create_test_agent(f"agent_{i}") for i in range(5)],
            tasks=tasks
        )
        
        # Test with optimized engine (has caching)
        print("\nTesting cache effectiveness...")
        optimized_engine = OptimizedSwarmExecutionEngine("benchmark_project")
        
        start_time = time.time()
        
        try:
            result = await optimized_engine.execute_swarm(context)
            execution_time = time.time() - start_time
            
            cache_hit_rate = result.performance_metrics.get("cache_hit_rate", 0)
            
            self.results["benchmarks"]["cache_performance"] = {
                "total_tasks": len(tasks),
                "unique_tasks": 100,
                "execution_time": execution_time,
                "cache_hit_rate": cache_hit_rate,
                "throughput": len(tasks) / execution_time,
                "estimated_speedup": 1 / (1 - cache_hit_rate * 0.9) if cache_hit_rate > 0 else 1
            }
            
            print(f"  Cache Hit Rate: {cache_hit_rate * 100:.1f}%")
            print(f"  Throughput: {len(tasks) / execution_time:.2f} tasks/sec")
            print(f"  Estimated Speedup from Cache: {self.results['benchmarks']['cache_performance']['estimated_speedup']:.2f}x")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            self.results["benchmarks"]["cache_performance"] = {"error": str(e)}
        finally:
            await optimized_engine.shutdown()
    
    def _create_test_context(self, task_count: int, agent_count: int = 10) -> ExecutionContext:
        """Create test execution context"""
        # Create agents
        agents = []
        for i in range(agent_count):
            agent = self._create_test_agent(f"agent_{i}")
            agents.append(agent)
        
        # Create tasks
        tasks = []
        for i in range(task_count):
            task_type = ["compute", "io", "transform"][i % 3]
            task = SwarmTask(
                id=f"task_{i}",
                type=task_type,
                data={
                    "index": i,
                    "payload": f"Test data for task {i}",
                    "size": i % 100
                },
                priority=i % 10 + 1,
                required_capabilities=[task_type]
            )
            tasks.append(task)
        
        # Create objective
        objective = SwarmObjective(
            id="benchmark_obj",
            description=f"Benchmark with {task_count} tasks and {agent_count} agents",
            success_criteria=["complete_all_tasks"]
        )
        
        return ExecutionContext(
            objective=objective,
            agents=agents,
            tasks=tasks,
            config={
                "max_parallel": agent_count * 10,
                "enable_caching": True,
                "batch_size": 25
            }
        )
    
    def _create_test_agent(self, agent_id: str) -> SwarmAgent:
        """Create test agent"""
        return SwarmAgent(
            id=agent_id,
            name=f"Test Agent {agent_id}",
            type="benchmark",
            capabilities=["compute", "io", "transform"],
            status="ready",
            load=0.0,
            performance_score=0.9
        )
    
    def _save_results(self):
        """Save benchmark results to file"""
        filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {filename}")
    
    def _print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 60)
        
        # Throughput improvement
        if "throughput" in self.results["benchmarks"]:
            throughput_improvements = []
            for count in self.results["benchmarks"]["throughput"]["optimized"]:
                if (count in self.results["benchmarks"]["throughput"]["original"] and
                    "throughput" in self.results["benchmarks"]["throughput"]["original"][count] and
                    "throughput" in self.results["benchmarks"]["throughput"]["optimized"][count]):
                    
                    original = self.results["benchmarks"]["throughput"]["original"][count]["throughput"]
                    optimized = self.results["benchmarks"]["throughput"]["optimized"][count]["throughput"]
                    improvement = (optimized / original - 1) * 100
                    throughput_improvements.append(improvement)
            
            if throughput_improvements:
                avg_throughput_improvement = statistics.mean(throughput_improvements)
                print(f"\nAverage Throughput Improvement: {avg_throughput_improvement:.1f}%")
        
        # Latency improvement
        if "latency" in self.results["benchmarks"]:
            if ("mean" in self.results["benchmarks"]["latency"]["original"] and
                "mean" in self.results["benchmarks"]["latency"]["optimized"]):
                
                original_latency = self.results["benchmarks"]["latency"]["original"]["mean"]
                optimized_latency = self.results["benchmarks"]["latency"]["optimized"]["mean"]
                latency_reduction = (1 - optimized_latency / original_latency) * 100
                print(f"Average Latency Reduction: {latency_reduction:.1f}%")
        
        # Resource efficiency
        if "resource_efficiency" in self.results["benchmarks"]:
            if ("avg_cpu" in self.results["benchmarks"]["resource_efficiency"]["original"] and
                "avg_cpu" in self.results["benchmarks"]["resource_efficiency"]["optimized"]):
                
                original_cpu = self.results["benchmarks"]["resource_efficiency"]["original"]["avg_cpu"]
                optimized_cpu = self.results["benchmarks"]["resource_efficiency"]["optimized"]["avg_cpu"]
                cpu_reduction = (1 - optimized_cpu / original_cpu) * 100
                print(f"CPU Usage Reduction: {cpu_reduction:.1f}%")
        
        # Cache effectiveness
        if "cache_performance" in self.results["benchmarks"]:
            if "cache_hit_rate" in self.results["benchmarks"]["cache_performance"]:
                cache_hit_rate = self.results["benchmarks"]["cache_performance"]["cache_hit_rate"]
                print(f"Cache Hit Rate: {cache_hit_rate * 100:.1f}%")
        
        print("\n" + "=" * 60)


async def main():
    """Run performance benchmarks"""
    benchmark = PerformanceBenchmark()
    await benchmark.run_all_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())
