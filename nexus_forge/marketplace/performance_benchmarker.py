"""
Performance benchmarking for agent packages
Tests latency, throughput, and resource usage
"""

import time
import asyncio
import psutil
import statistics
from typing import Dict, List, Any, Optional
import tempfile
import importlib.util
import sys
from pathlib import Path
import json
import zipfile

from .models import PerformanceMetrics, AgentManifest


class PerformanceBenchmarker:
    """Benchmark agent performance across various metrics"""
    
    # Benchmark configuration
    WARMUP_REQUESTS = 10
    BENCHMARK_REQUESTS = 100
    CONCURRENT_REQUESTS = 10
    TIMEOUT_SECONDS = 30
    
    # Test inputs for different agent types
    TEST_INPUTS = {
        "natural_language": [
            {"text": "Hello, how are you?"},
            {"text": "What is the capital of France?"},
            {"text": "Explain quantum computing in simple terms."}
        ],
        "code_generation": [
            {"prompt": "Write a Python function to calculate fibonacci"},
            {"prompt": "Create a REST API endpoint"},
            {"prompt": "Implement a binary search algorithm"}
        ],
        "data_processing": [
            {"data": [1, 2, 3, 4, 5], "operation": "sum"},
            {"data": {"key": "value"}, "operation": "transform"},
            {"data": "sample text", "operation": "analyze"}
        ],
        "default": [
            {"input": "test"},
            {"query": "benchmark"},
            {"data": "sample"}
        ]
    }
    
    async def benchmark_agent(
        self, package_path: str, manifest: AgentManifest
    ) -> PerformanceMetrics:
        """
        Run comprehensive performance benchmarks on an agent
        
        Args:
            package_path: Path to agent package
            manifest: Agent manifest
            
        Returns:
            Performance metrics
        """
        # Extract and load agent
        with tempfile.TemporaryDirectory() as temp_dir:
            agent_instance = await self._load_agent(package_path, manifest, temp_dir)
            
            # Select appropriate test inputs
            test_inputs = self._get_test_inputs(manifest.category)
            
            # Warmup phase
            await self._warmup(agent_instance, test_inputs)
            
            # Run benchmarks
            latency_results = await self._benchmark_latency(agent_instance, test_inputs)
            throughput_results = await self._benchmark_throughput(agent_instance, test_inputs)
            resource_results = await self._benchmark_resources(agent_instance, test_inputs)
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                latency_results, throughput_results, resource_results
            )
            
            return metrics
    
    async def _load_agent(
        self, package_path: str, manifest: AgentManifest, extract_dir: str
    ) -> Any:
        """Load agent from package"""
        # Extract package
        with zipfile.ZipFile(package_path, 'r') as zip_file:
            zip_file.extractall(extract_dir)
        
        # Add to Python path
        sys.path.insert(0, extract_dir)
        
        try:
            # Parse main class
            module_name, class_name = manifest.main_class.rsplit('.', 1)
            
            # Import module
            spec = importlib.util.spec_from_file_location(
                module_name,
                Path(extract_dir) / f"{module_name.replace('.', '/')}.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get agent class
            agent_class = getattr(module, class_name)
            
            # Initialize agent
            config = manifest.config_schema or {}
            agent_instance = agent_class(config)
            
            return agent_instance
            
        finally:
            # Clean up sys.path
            sys.path.remove(extract_dir)
    
    def _get_test_inputs(self, category: str) -> List[Dict[str, Any]]:
        """Get appropriate test inputs for agent category"""
        return self.TEST_INPUTS.get(category.value, self.TEST_INPUTS["default"])
    
    async def _warmup(self, agent: Any, test_inputs: List[Dict[str, Any]]) -> None:
        """Warmup agent before benchmarking"""
        for _ in range(self.WARMUP_REQUESTS):
            for input_data in test_inputs:
                try:
                    await asyncio.wait_for(
                        agent.process(input_data),
                        timeout=self.TIMEOUT_SECONDS
                    )
                except Exception:
                    pass  # Ignore errors during warmup
    
    async def _benchmark_latency(
        self, agent: Any, test_inputs: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Benchmark request latency"""
        latencies = []
        errors = 0
        
        for _ in range(self.BENCHMARK_REQUESTS):
            for input_data in test_inputs:
                start_time = time.perf_counter()
                
                try:
                    await asyncio.wait_for(
                        agent.process(input_data),
                        timeout=self.TIMEOUT_SECONDS
                    )
                    latency = (time.perf_counter() - start_time) * 1000  # Convert to ms
                    latencies.append(latency)
                except Exception:
                    errors += 1
        
        return {
            "latencies": latencies,
            "errors": errors
        }
    
    async def _benchmark_throughput(
        self, agent: Any, test_inputs: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Benchmark throughput (requests per second)"""
        start_time = time.perf_counter()
        completed_requests = 0
        
        # Run concurrent requests
        tasks = []
        for _ in range(self.BENCHMARK_REQUESTS // self.CONCURRENT_REQUESTS):
            batch_tasks = []
            for _ in range(self.CONCURRENT_REQUESTS):
                for input_data in test_inputs:
                    task = asyncio.create_task(
                        self._process_with_timeout(agent, input_data)
                    )
                    batch_tasks.append(task)
            
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            completed_requests += sum(1 for r in results if not isinstance(r, Exception))
            tasks.extend(batch_tasks)
        
        duration = time.perf_counter() - start_time
        throughput = completed_requests / duration
        
        return {
            "throughput": throughput,
            "duration": duration,
            "completed": completed_requests
        }
    
    async def _benchmark_resources(
        self, agent: Any, test_inputs: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Benchmark resource usage"""
        memory_usage = []
        cpu_usage = []
        
        process = psutil.Process()
        
        # Monitor resources during execution
        monitor_task = asyncio.create_task(
            self._monitor_resources(process, memory_usage, cpu_usage)
        )
        
        # Run benchmark
        for _ in range(20):  # Reduced iterations for resource monitoring
            for input_data in test_inputs:
                try:
                    await asyncio.wait_for(
                        agent.process(input_data),
                        timeout=self.TIMEOUT_SECONDS
                    )
                except Exception:
                    pass
        
        # Stop monitoring
        monitor_task.cancel()
        
        return {
            "memory_usage": memory_usage,
            "cpu_usage": cpu_usage
        }
    
    async def _process_with_timeout(
        self, agent: Any, input_data: Dict[str, Any]
    ) -> Any:
        """Process request with timeout"""
        try:
            return await asyncio.wait_for(
                agent.process(input_data),
                timeout=self.TIMEOUT_SECONDS
            )
        except Exception:
            raise
    
    async def _monitor_resources(
        self, process: psutil.Process,
        memory_list: List[float],
        cpu_list: List[float]
    ) -> None:
        """Monitor resource usage"""
        while True:
            try:
                # Memory in MB
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_list.append(memory_mb)
                
                # CPU percentage
                cpu_percent = process.cpu_percent(interval=0.1)
                cpu_list.append(cpu_percent)
                
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception:
                pass
    
    def _calculate_metrics(
        self,
        latency_results: Dict[str, Any],
        throughput_results: Dict[str, float],
        resource_results: Dict[str, List[float]]
    ) -> PerformanceMetrics:
        """Calculate final performance metrics"""
        latencies = latency_results["latencies"]
        errors = latency_results["errors"]
        
        # Calculate latency percentiles
        if latencies:
            avg_latency = statistics.mean(latencies)
            p50_latency = statistics.median(latencies)
            p95_latency = self._percentile(latencies, 95)
            p99_latency = self._percentile(latencies, 99)
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = 0
        
        # Calculate resource usage
        memory_usage = resource_results["memory_usage"]
        cpu_usage = resource_results["cpu_usage"]
        
        if memory_usage:
            avg_memory = statistics.mean(memory_usage)
            peak_memory = max(memory_usage)
        else:
            avg_memory = peak_memory = 0
        
        if cpu_usage:
            avg_cpu = statistics.mean(cpu_usage)
        else:
            avg_cpu = 0
        
        # Calculate rates
        total_requests = len(latencies) + errors
        success_rate = len(latencies) / total_requests if total_requests > 0 else 0
        error_rate = errors / total_requests if total_requests > 0 else 0
        
        # Calculate overall performance score
        performance_score = self._calculate_performance_score(
            avg_latency, p95_latency, throughput_results["throughput"],
            success_rate, avg_memory, avg_cpu
        )
        
        return PerformanceMetrics(
            avg_response_time=avg_latency,
            p50_response_time=p50_latency,
            p95_response_time=p95_latency,
            p99_response_time=p99_latency,
            avg_memory_mb=avg_memory,
            peak_memory_mb=peak_memory,
            avg_cpu_percent=avg_cpu,
            requests_per_second=throughput_results["throughput"],
            success_rate=success_rate,
            error_rate=error_rate,
            performance_score=performance_score
        )
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not data:
            return 0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        
        if index >= len(sorted_data):
            return sorted_data[-1]
        
        return sorted_data[index]
    
    def _calculate_performance_score(
        self,
        avg_latency: float,
        p95_latency: float,
        throughput: float,
        success_rate: float,
        avg_memory: float,
        avg_cpu: float
    ) -> float:
        """Calculate overall performance score (0-100)"""
        # Scoring weights
        weights = {
            "latency": 0.3,
            "throughput": 0.25,
            "success_rate": 0.25,
            "resource_usage": 0.2
        }
        
        # Latency score (lower is better)
        # Good: <100ms, Acceptable: <500ms, Poor: >1000ms
        if avg_latency < 100:
            latency_score = 100
        elif avg_latency < 500:
            latency_score = 100 - ((avg_latency - 100) / 400) * 50
        elif avg_latency < 1000:
            latency_score = 50 - ((avg_latency - 500) / 500) * 30
        else:
            latency_score = max(0, 20 - (avg_latency - 1000) / 100)
        
        # Throughput score (higher is better)
        # Good: >100 RPS, Acceptable: >10 RPS, Poor: <10 RPS
        if throughput > 100:
            throughput_score = 100
        elif throughput > 10:
            throughput_score = 50 + ((throughput - 10) / 90) * 50
        else:
            throughput_score = (throughput / 10) * 50
        
        # Success rate score
        success_score = success_rate * 100
        
        # Resource usage score (lower is better)
        memory_score = max(0, 100 - (avg_memory / 1000) * 50)  # Penalize >2GB
        cpu_score = max(0, 100 - avg_cpu)  # Penalize high CPU
        resource_score = (memory_score + cpu_score) / 2
        
        # Calculate weighted score
        overall_score = (
            latency_score * weights["latency"] +
            throughput_score * weights["throughput"] +
            success_score * weights["success_rate"] +
            resource_score * weights["resource_usage"]
        )
        
        return min(100, max(0, overall_score))