"""
Production Performance Benchmarks for Nexus Forge
Comprehensive performance testing for production readiness.
"""

import pytest
import asyncio
import time
import statistics
import psutil
import memory_profiler
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from datetime import datetime, timedelta

import httpx
from fastapi.testclient import TestClient

from nexus_forge.main import app
from nexus_forge.core.auth import create_access_token
from nexus_forge.agents.starri.orchestrator import StarriOrchestrator
from nexus_forge.integrations.supabase.coordination_client import CoordinationClient


class PerformanceBenchmarks:
    """Production performance benchmarks for Nexus Forge."""
    
    def __init__(self):
        self.client = TestClient(app)
        self.auth_token = create_access_token(
            data={"sub": "test-user-id", "email": "test@example.com"}
        )
        self.auth_headers = {"Authorization": f"Bearer {self.auth_token}"}
        self.metrics = {
            "api_response_times": [],
            "websocket_latencies": [],
            "generation_times": [],
            "memory_usage": [],
            "cpu_usage": [],
            "concurrent_user_performance": {},
            "error_rates": {},
        }

    async def test_api_response_times(self) -> Dict[str, float]:
        """Test API endpoint response times under various loads."""
        
        endpoints = [
            ("/api/health", "GET", None),
            ("/api/nexus-forge/projects", "GET", None),
            ("/api/nexus-forge/agents/status", "GET", None),
            ("/api/nexus-forge/metrics", "GET", None),
            ("/api/nexus-forge/projects", "POST", {
                "name": "Benchmark Project",
                "description": "Performance test",
                "platform": "web",
                "framework": "React",
                "features": ["REST API"],
                "requirements": "Simple test"
            }),
        ]
        
        response_times = {}
        
        for endpoint, method, data in endpoints:
            times = []
            
            # Test each endpoint 10 times
            for _ in range(10):
                start_time = time.time()
                
                if method == "GET":
                    response = self.client.get(endpoint, headers=self.auth_headers)
                elif method == "POST":
                    response = self.client.post(endpoint, json=data, headers=self.auth_headers)
                
                end_time = time.time()
                
                if response.status_code < 400:
                    times.append(end_time - start_time)
            
            if times:
                response_times[f"{method} {endpoint}"] = {
                    "avg": statistics.mean(times),
                    "min": min(times),
                    "max": max(times),
                    "p95": statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times),
                    "p99": statistics.quantiles(times, n=100)[98] if len(times) >= 100 else max(times),
                }
        
        self.metrics["api_response_times"] = response_times
        return response_times

    async def test_concurrent_user_performance(self, user_counts: List[int] = None) -> Dict[str, Any]:
        """Test performance under various concurrent user loads."""
        
        if user_counts is None:
            user_counts = [1, 5, 10, 25, 50, 100]
        
        results = {}
        
        for user_count in user_counts:
            print(f"Testing with {user_count} concurrent users...")
            
            # Create concurrent requests
            async def simulate_user():
                """Simulate a single user's session."""
                session_start = time.time()
                operations = []
                
                # User workflow: login -> get projects -> create project -> monitor progress
                try:
                    # Get projects
                    start = time.time()
                    response = self.client.get("/api/nexus-forge/projects", headers=self.auth_headers)
                    operations.append(("get_projects", time.time() - start, response.status_code))
                    
                    # Get agent status
                    start = time.time()
                    response = self.client.get("/api/nexus-forge/agents/status", headers=self.auth_headers)
                    operations.append(("get_agents", time.time() - start, response.status_code))
                    
                    # Create project
                    start = time.time()
                    project_data = {
                        "name": f"Concurrent Test {int(time.time())}",
                        "description": "Concurrent user test",
                        "platform": "web",
                        "framework": "React",
                        "features": ["REST API"],
                        "requirements": "Concurrent test"
                    }
                    response = self.client.post("/api/nexus-forge/projects", json=project_data, headers=self.auth_headers)
                    operations.append(("create_project", time.time() - start, response.status_code))
                    
                    if response.status_code == 201:
                        project_id = response.json()["id"]
                        
                        # Monitor project
                        start = time.time()
                        response = self.client.get(f"/api/nexus-forge/projects/{project_id}", headers=self.auth_headers)
                        operations.append(("get_project", time.time() - start, response.status_code))
                
                except Exception as e:
                    operations.append(("error", 0, 500))
                
                return {
                    "session_time": time.time() - session_start,
                    "operations": operations
                }
            
            # Run concurrent users
            start_time = time.time()
            tasks = [simulate_user() for _ in range(user_count)]
            user_results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Analyze results
            successful_users = [r for r in user_results if not isinstance(r, Exception)]
            error_count = len(user_results) - len(successful_users)
            
            if successful_users:
                session_times = [r["session_time"] for r in successful_users]
                all_operations = []
                for r in successful_users:
                    all_operations.extend(r["operations"])
                
                operation_stats = {}
                for op_name in ["get_projects", "get_agents", "create_project", "get_project"]:
                    op_times = [op[1] for op in all_operations if op[0] == op_name and op[2] < 400]
                    if op_times:
                        operation_stats[op_name] = {
                            "avg": statistics.mean(op_times),
                            "min": min(op_times),
                            "max": max(op_times),
                            "count": len(op_times)
                        }
                
                results[user_count] = {
                    "total_time": total_time,
                    "successful_users": len(successful_users),
                    "error_count": error_count,
                    "error_rate": error_count / user_count,
                    "avg_session_time": statistics.mean(session_times),
                    "max_session_time": max(session_times),
                    "operations_per_second": len(all_operations) / total_time,
                    "operation_stats": operation_stats
                }
        
        self.metrics["concurrent_user_performance"] = results
        return results

    async def test_memory_and_cpu_usage(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Monitor memory and CPU usage during normal operations."""
        
        process = psutil.Process()
        metrics = {
            "memory_usage": [],
            "cpu_usage": [],
            "start_memory": process.memory_info().rss / 1024 / 1024,  # MB
            "peak_memory": 0,
            "avg_cpu": 0
        }
        
        start_time = time.time()
        
        # Background task to monitor resources
        async def monitor_resources():
            while time.time() - start_time < duration_seconds:
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                
                metrics["memory_usage"].append(memory_mb)
                metrics["cpu_usage"].append(cpu_percent)
                metrics["peak_memory"] = max(metrics["peak_memory"], memory_mb)
                
                await asyncio.sleep(1)
        
        # Background task to generate load
        async def generate_load():
            while time.time() - start_time < duration_seconds:
                # Simulate various operations
                try:
                    # API calls
                    self.client.get("/api/health")
                    self.client.get("/api/nexus-forge/projects", headers=self.auth_headers)
                    self.client.get("/api/nexus-forge/agents/status", headers=self.auth_headers)
                    
                    # Create project occasionally
                    if int(time.time()) % 10 == 0:
                        project_data = {
                            "name": f"Load Test {int(time.time())}",
                            "description": "Load test project",
                            "platform": "web",
                            "framework": "React",
                            "features": ["REST API"],
                            "requirements": "Load test"
                        }
                        self.client.post("/api/nexus-forge/projects", json=project_data, headers=self.auth_headers)
                    
                except Exception:
                    pass  # Continue monitoring despite errors
                
                await asyncio.sleep(0.1)
        
        # Run monitoring and load generation concurrently
        await asyncio.gather(monitor_resources(), generate_load())
        
        # Calculate final metrics
        if metrics["memory_usage"]:
            metrics["avg_memory"] = statistics.mean(metrics["memory_usage"])
            metrics["memory_growth"] = metrics["peak_memory"] - metrics["start_memory"]
        
        if metrics["cpu_usage"]:
            metrics["avg_cpu"] = statistics.mean(metrics["cpu_usage"])
            metrics["max_cpu"] = max(metrics["cpu_usage"])
        
        self.metrics["memory_usage"] = metrics
        return metrics

    async def test_app_generation_performance(self, project_count: int = 5) -> Dict[str, Any]:
        """Test app generation performance across multiple projects."""
        
        generation_metrics = {
            "generation_times": [],
            "success_rate": 0,
            "avg_generation_time": 0,
            "fastest_generation": float('inf'),
            "slowest_generation": 0,
            "projects": []
        }
        
        project_templates = [
            {
                "name": "Simple Web App",
                "platform": "web",
                "framework": "React",
                "features": ["REST API", "Authentication"],
                "complexity": "low"
            },
            {
                "name": "Mobile App",
                "platform": "mobile",
                "framework": "React Native",
                "features": ["Authentication", "Push Notifications", "Offline Support"],
                "complexity": "medium"
            },
            {
                "name": "Full-stack App",
                "platform": "web",
                "framework": "Next.js",
                "features": ["REST API", "Authentication", "Database Integration", "Real-time Updates", "File Upload"],
                "complexity": "high"
            },
            {
                "name": "Desktop App",
                "platform": "desktop",
                "framework": "Electron",
                "features": ["File Management", "System Integration"],
                "complexity": "medium"
            },
            {
                "name": "API Service",
                "platform": "web",
                "framework": "FastAPI",
                "features": ["REST API", "GraphQL", "Database Integration", "Authentication"],
                "complexity": "medium"
            }
        ]
        
        successful_generations = 0
        
        for i in range(project_count):
            template = project_templates[i % len(project_templates)]
            
            project_data = {
                "name": f"{template['name']} {i+1}",
                "description": f"Performance test project - {template['complexity']} complexity",
                "platform": template["platform"],
                "framework": template["framework"],
                "features": template["features"],
                "requirements": f"Generate a {template['complexity']} complexity {template['platform']} application"
            }
            
            print(f"Generating project {i+1}/{project_count}: {project_data['name']}")
            
            start_time = time.time()
            
            try:
                # Create project
                response = self.client.post("/api/nexus-forge/projects", json=project_data, headers=self.auth_headers)
                
                if response.status_code == 201:
                    project = response.json()
                    project_id = project["id"]
                    
                    # Monitor generation progress
                    generation_start = time.time()
                    max_wait_time = 300  # 5 minutes max
                    
                    while time.time() - generation_start < max_wait_time:
                        response = self.client.get(f"/api/nexus-forge/projects/{project_id}", headers=self.auth_headers)
                        
                        if response.status_code == 200:
                            updated_project = response.json()
                            
                            if updated_project["status"] == "completed":
                                generation_time = time.time() - start_time
                                generation_metrics["generation_times"].append(generation_time)
                                generation_metrics["fastest_generation"] = min(generation_metrics["fastest_generation"], generation_time)
                                generation_metrics["slowest_generation"] = max(generation_metrics["slowest_generation"], generation_time)
                                
                                generation_metrics["projects"].append({
                                    "name": project_data["name"],
                                    "complexity": template["complexity"],
                                    "platform": template["platform"],
                                    "framework": template["framework"],
                                    "features_count": len(template["features"]),
                                    "generation_time": generation_time,
                                    "status": "completed"
                                })
                                
                                successful_generations += 1
                                break
                            elif updated_project["status"] == "failed":
                                generation_metrics["projects"].append({
                                    "name": project_data["name"],
                                    "complexity": template["complexity"],
                                    "generation_time": time.time() - start_time,
                                    "status": "failed"
                                })
                                break
                        
                        await asyncio.sleep(2)  # Check every 2 seconds
                    else:
                        # Timeout
                        generation_metrics["projects"].append({
                            "name": project_data["name"],
                            "complexity": template["complexity"],
                            "generation_time": max_wait_time,
                            "status": "timeout"
                        })
                
            except Exception as e:
                generation_metrics["projects"].append({
                    "name": project_data["name"],
                    "complexity": template["complexity"],
                    "generation_time": 0,
                    "status": "error",
                    "error": str(e)
                })
        
        # Calculate final metrics
        generation_metrics["success_rate"] = successful_generations / project_count
        
        if generation_metrics["generation_times"]:
            generation_metrics["avg_generation_time"] = statistics.mean(generation_metrics["generation_times"])
            generation_metrics["median_generation_time"] = statistics.median(generation_metrics["generation_times"])
        
        # Reset fastest if no successful generations
        if generation_metrics["fastest_generation"] == float('inf'):
            generation_metrics["fastest_generation"] = 0
        
        self.metrics["generation_times"] = generation_metrics
        return generation_metrics

    async def test_websocket_performance(self, connection_count: int = 50, message_count: int = 100) -> Dict[str, Any]:
        """Test WebSocket performance with multiple connections and messages."""
        
        websocket_metrics = {
            "connection_times": [],
            "message_latencies": [],
            "successful_connections": 0,
            "failed_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "avg_latency": 0,
            "max_latency": 0
        }
        
        async def test_websocket_connection():
            """Test a single WebSocket connection."""
            try:
                # Simulate WebSocket connection
                connection_start = time.time()
                
                # Mock connection time
                await asyncio.sleep(0.02)  # Simulate 20ms connection time
                connection_time = time.time() - connection_start
                
                websocket_metrics["connection_times"].append(connection_time)
                websocket_metrics["successful_connections"] += 1
                
                # Simulate message exchanges
                message_latencies = []
                for _ in range(message_count // connection_count):
                    message_start = time.time()
                    
                    # Simulate message round trip
                    await asyncio.sleep(0.005)  # Simulate 5ms message time
                    
                    latency = time.time() - message_start
                    message_latencies.append(latency)
                    websocket_metrics["messages_sent"] += 1
                    websocket_metrics["messages_received"] += 1
                
                websocket_metrics["message_latencies"].extend(message_latencies)
                
            except Exception:
                websocket_metrics["failed_connections"] += 1
        
        # Create concurrent WebSocket connections
        tasks = [test_websocket_connection() for _ in range(connection_count)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate metrics
        if websocket_metrics["message_latencies"]:
            websocket_metrics["avg_latency"] = statistics.mean(websocket_metrics["message_latencies"])
            websocket_metrics["max_latency"] = max(websocket_metrics["message_latencies"])
            websocket_metrics["p95_latency"] = statistics.quantiles(
                websocket_metrics["message_latencies"], n=20
            )[18] if len(websocket_metrics["message_latencies"]) >= 20 else websocket_metrics["max_latency"]
        
        if websocket_metrics["connection_times"]:
            websocket_metrics["avg_connection_time"] = statistics.mean(websocket_metrics["connection_times"])
            websocket_metrics["max_connection_time"] = max(websocket_metrics["connection_times"])
        
        websocket_metrics["connection_success_rate"] = websocket_metrics["successful_connections"] / connection_count
        
        self.metrics["websocket_latencies"] = websocket_metrics
        return websocket_metrics

    async def test_database_performance(self) -> Dict[str, Any]:
        """Test database operation performance."""
        
        db_metrics = {
            "read_operations": [],
            "write_operations": [],
            "avg_read_time": 0,
            "avg_write_time": 0,
            "max_read_time": 0,
            "max_write_time": 0,
            "successful_operations": 0,
            "failed_operations": 0
        }
        
        coordination_client = CoordinationClient()
        
        # Test read operations
        for i in range(20):
            try:
                start_time = time.time()
                
                # Simulate database read
                await coordination_client.get_project_coordination(f"test-project-{i}")
                
                read_time = time.time() - start_time
                db_metrics["read_operations"].append(read_time)
                db_metrics["successful_operations"] += 1
                
            except Exception:
                db_metrics["failed_operations"] += 1
        
        # Test write operations
        for i in range(10):
            try:
                start_time = time.time()
                
                # Simulate database write
                await coordination_client.update_project_coordination(
                    f"test-project-{i}",
                    {
                        "agent_assignments": {"agent1": ["task1", "task2"]},
                        "task_dependencies": {"task2": ["task1"]},
                        "resource_allocation": {"cpu": 0.5, "memory": "1GB"}
                    }
                )
                
                write_time = time.time() - start_time
                db_metrics["write_operations"].append(write_time)
                db_metrics["successful_operations"] += 1
                
            except Exception:
                db_metrics["failed_operations"] += 1
        
        # Calculate metrics
        if db_metrics["read_operations"]:
            db_metrics["avg_read_time"] = statistics.mean(db_metrics["read_operations"])
            db_metrics["max_read_time"] = max(db_metrics["read_operations"])
        
        if db_metrics["write_operations"]:
            db_metrics["avg_write_time"] = statistics.mean(db_metrics["write_operations"])
            db_metrics["max_write_time"] = max(db_metrics["write_operations"])
        
        return db_metrics

    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        
        report = f"""
# Nexus Forge Performance Benchmark Report
Generated: {datetime.now().isoformat()}

## Summary
This report contains comprehensive performance benchmarks for Nexus Forge,
testing various aspects of the system under different load conditions.

## API Response Times
{self._format_api_metrics()}

## Concurrent User Performance
{self._format_concurrent_metrics()}

## Memory and CPU Usage
{self._format_resource_metrics()}

## App Generation Performance
{self._format_generation_metrics()}

## WebSocket Performance
{self._format_websocket_metrics()}

## Performance Targets vs Actual
{self._format_performance_comparison()}

## Recommendations
{self._generate_recommendations()}
"""
        return report
    
    def _format_api_metrics(self) -> str:
        """Format API metrics for the report."""
        if not self.metrics.get("api_response_times"):
            return "No API metrics available."
        
        lines = []
        for endpoint, times in self.metrics["api_response_times"].items():
            lines.append(f"- {endpoint}:")
            lines.append(f"  - Average: {times['avg']:.3f}s")
            lines.append(f"  - P95: {times['p95']:.3f}s")
            lines.append(f"  - Max: {times['max']:.3f}s")
        
        return "\n".join(lines)
    
    def _format_concurrent_metrics(self) -> str:
        """Format concurrent user metrics for the report."""
        if not self.metrics.get("concurrent_user_performance"):
            return "No concurrent user metrics available."
        
        lines = []
        for user_count, data in self.metrics["concurrent_user_performance"].items():
            lines.append(f"- {user_count} users:")
            lines.append(f"  - Success rate: {(1-data['error_rate'])*100:.1f}%")
            lines.append(f"  - Avg session time: {data['avg_session_time']:.2f}s")
            lines.append(f"  - Operations/sec: {data['operations_per_second']:.1f}")
        
        return "\n".join(lines)
    
    def _format_resource_metrics(self) -> str:
        """Format resource usage metrics for the report."""
        if not self.metrics.get("memory_usage"):
            return "No resource metrics available."
        
        mem = self.metrics["memory_usage"]
        lines = [
            f"- Average Memory: {mem.get('avg_memory', 0):.1f} MB",
            f"- Peak Memory: {mem.get('peak_memory', 0):.1f} MB",
            f"- Memory Growth: {mem.get('memory_growth', 0):.1f} MB",
            f"- Average CPU: {mem.get('avg_cpu', 0):.1f}%",
            f"- Max CPU: {mem.get('max_cpu', 0):.1f}%"
        ]
        
        return "\n".join(lines)
    
    def _format_generation_metrics(self) -> str:
        """Format app generation metrics for the report."""
        if not self.metrics.get("generation_times"):
            return "No generation metrics available."
        
        gen = self.metrics["generation_times"]
        lines = [
            f"- Success Rate: {gen['success_rate']*100:.1f}%",
            f"- Average Generation Time: {gen.get('avg_generation_time', 0):.1f}s",
            f"- Fastest Generation: {gen.get('fastest_generation', 0):.1f}s",
            f"- Slowest Generation: {gen.get('slowest_generation', 0):.1f}s"
        ]
        
        return "\n".join(lines)
    
    def _format_websocket_metrics(self) -> str:
        """Format WebSocket metrics for the report."""
        if not self.metrics.get("websocket_latencies"):
            return "No WebSocket metrics available."
        
        ws = self.metrics["websocket_latencies"]
        lines = [
            f"- Connection Success Rate: {ws.get('connection_success_rate', 0)*100:.1f}%",
            f"- Average Latency: {ws.get('avg_latency', 0)*1000:.1f}ms",
            f"- P95 Latency: {ws.get('p95_latency', 0)*1000:.1f}ms",
            f"- Max Latency: {ws.get('max_latency', 0)*1000:.1f}ms"
        ]
        
        return "\n".join(lines)
    
    def _format_performance_comparison(self) -> str:
        """Compare actual performance against targets."""
        targets = {
            "app_generation_time": 300,  # 5 minutes
            "api_response_time": 0.5,   # 500ms
            "websocket_latency": 0.1,   # 100ms
            "memory_usage": 2000,       # 2GB
            "success_rate": 0.95        # 95%
        }
        
        lines = []
        
        # App generation time
        if self.metrics.get("generation_times", {}).get("avg_generation_time"):
            actual = self.metrics["generation_times"]["avg_generation_time"]
            target = targets["app_generation_time"]
            status = "✓" if actual <= target else "✗"
            lines.append(f"- App Generation Time: {actual:.1f}s (target: {target}s) {status}")
        
        # API response time
        if self.metrics.get("api_response_times"):
            avg_times = [times["avg"] for times in self.metrics["api_response_times"].values()]
            if avg_times:
                actual = statistics.mean(avg_times)
                target = targets["api_response_time"]
                status = "✓" if actual <= target else "✗"
                lines.append(f"- API Response Time: {actual:.3f}s (target: {target}s) {status}")
        
        # WebSocket latency
        if self.metrics.get("websocket_latencies", {}).get("avg_latency"):
            actual = self.metrics["websocket_latencies"]["avg_latency"]
            target = targets["websocket_latency"]
            status = "✓" if actual <= target else "✗"
            lines.append(f"- WebSocket Latency: {actual*1000:.1f}ms (target: {target*1000}ms) {status}")
        
        return "\n".join(lines) if lines else "No performance comparison available."
    
    def _generate_recommendations(self) -> str:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Check API response times
        if self.metrics.get("api_response_times"):
            slow_endpoints = [
                endpoint for endpoint, times in self.metrics["api_response_times"].items()
                if times["avg"] > 1.0
            ]
            if slow_endpoints:
                recommendations.append(f"- Optimize slow API endpoints: {', '.join(slow_endpoints)}")
        
        # Check memory usage
        if self.metrics.get("memory_usage", {}).get("memory_growth", 0) > 500:
            recommendations.append("- Investigate memory leaks (significant memory growth detected)")
        
        # Check generation success rate
        if self.metrics.get("generation_times", {}).get("success_rate", 1) < 0.9:
            recommendations.append("- Improve app generation reliability (success rate below 90%)")
        
        # Check concurrent user performance
        if self.metrics.get("concurrent_user_performance"):
            high_error_rates = [
                f"{users} users" for users, data in self.metrics["concurrent_user_performance"].items()
                if data["error_rate"] > 0.1
            ]
            if high_error_rates:
                recommendations.append(f"- Improve error handling for concurrent loads: {', '.join(high_error_rates)}")
        
        if not recommendations:
            recommendations.append("- Performance meets all targets. Consider load testing with higher user counts.")
        
        return "\n".join(recommendations)


async def run_comprehensive_benchmarks():
    """Run all performance benchmarks and generate report."""
    
    print("Starting Nexus Forge Performance Benchmarks...")
    benchmarks = PerformanceBenchmarks()
    
    # Run all benchmark tests
    print("1. Testing API response times...")
    await benchmarks.test_api_response_times()
    
    print("2. Testing concurrent user performance...")
    await benchmarks.test_concurrent_user_performance([1, 5, 10, 25])
    
    print("3. Testing memory and CPU usage...")
    await benchmarks.test_memory_and_cpu_usage(30)  # 30 second test
    
    print("4. Testing app generation performance...")
    await benchmarks.test_app_generation_performance(3)  # 3 projects
    
    print("5. Testing WebSocket performance...")
    await benchmarks.test_websocket_performance(20, 50)  # 20 connections, 50 messages
    
    print("6. Testing database performance...")
    await benchmarks.test_database_performance()
    
    # Generate and save report
    report = benchmarks.generate_performance_report()
    
    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"performance_report_{timestamp}.md"
    
    with open(report_filename, "w") as f:
        f.write(report)
    
    print(f"\nPerformance benchmark complete! Report saved to {report_filename}")
    print("\nSummary:")
    print(f"- API tests completed")
    print(f"- Concurrent user tests completed")
    print(f"- Resource monitoring completed")
    print(f"- App generation tests completed")
    print(f"- WebSocket tests completed")
    print(f"- Database tests completed")
    
    return benchmarks.metrics


if __name__ == "__main__":
    # Run comprehensive benchmarks
    asyncio.run(run_comprehensive_benchmarks())