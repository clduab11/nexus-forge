"""
Swarm Coordination Patterns
Implementation of various swarm coordination patterns for different scenarios
"""

import asyncio
import json
import logging
import random
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import numpy as np
import networkx as nx

from .swarm_intelligence import (
    SwarmAgent,
    SwarmCoordinator,
    SwarmMessage,
    SwarmObjective,
    SwarmTask,
    CommunicationType,
    Pheromone,
)
from nexus_forge.core.exceptions import CoordinationError
from nexus_forge.core.monitoring import get_logger

logger = get_logger(__name__)


class HierarchicalSwarmCoordinator(SwarmCoordinator):
    """
    Tree-structured swarm with clear command hierarchy
    Best for: Complex projects with clear subtask dependencies
    """
    
    async def execute(
        self,
        objective: SwarmObjective,
        agents: List[SwarmAgent],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute hierarchical coordination with commander and squad structure"""
        logger.info(f"Executing hierarchical swarm for objective: {objective.id}")
        
        # Initialize execution tracking
        completed_tasks = []
        failed_tasks = []
        results = {}
        
        try:
            # Select commander (highest performance score)
            commander = max(agents, key=lambda a: a.performance_score)
            commander.metadata["role"] = "commander"
            
            # Organize remaining agents into squads
            squads = await self._organize_squads(
                agents=[a for a in agents if a.id != commander.id],
                strategy=objective.strategy
            )
            
            # Commander decomposes objective
            subtasks = await self._decompose_objective(objective, commander, len(squads))
            
            # Assign tasks to squads
            squad_assignments = await self._assign_tasks_to_squads(subtasks, squads)
            
            # Execute with hierarchical coordination
            execution_results = await self._execute_hierarchical(
                commander,
                squad_assignments,
                objective
            )
            
            # Aggregate results
            for squad_id, squad_result in execution_results.items():
                results[squad_id] = squad_result["results"]
                completed_tasks.extend(squad_result["completed"])
                failed_tasks.extend(squad_result["failed"])
            
            # Calculate overall status
            total_tasks = len(completed_tasks) + len(failed_tasks)
            success_rate = len(completed_tasks) / total_tasks if total_tasks > 0 else 0
            
            status = "completed" if success_rate > 0.8 else "partial" if success_rate > 0.5 else "failed"
            
            return {
                "status": status,
                "results": results,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "confidence": success_rate,
                "hierarchy": {
                    "commander": commander.id,
                    "squads": {k: [a.id for a in v["agents"]] for k, v in squads.items()}
                }
            }
            
        except Exception as e:
            logger.error(f"Hierarchical coordination failed: {e}")
            return {
                "status": "failed",
                "results": {"error": str(e)},
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "confidence": 0.0
            }
    
    async def _organize_squads(
        self,
        agents: List[SwarmAgent],
        strategy: Any
    ) -> Dict[str, Dict[str, Any]]:
        """Organize agents into functional squads"""
        squads = defaultdict(lambda: {"agents": [], "leader": None})
        
        # Group by primary capability
        capability_groups = defaultdict(list)
        for agent in agents:
            if agent.capabilities:
                primary_capability = agent.capabilities[0]
                capability_groups[primary_capability].append(agent)
        
        # Form squads from capability groups
        squad_id = 0
        for capability, group_agents in capability_groups.items():
            if group_agents:
                squad_name = f"squad_{squad_id}_{capability}"
                
                # Select squad leader (best performer in group)
                leader = max(group_agents, key=lambda a: a.performance_score)
                leader.metadata["role"] = "squad_leader"
                
                squads[squad_name] = {
                    "agents": group_agents,
                    "leader": leader,
                    "capability": capability
                }
                
                squad_id += 1
        
        return dict(squads)
    
    async def _decompose_objective(
        self,
        objective: SwarmObjective,
        commander: SwarmAgent,
        num_squads: int
    ) -> List[SwarmTask]:
        """Commander decomposes objective into subtasks"""
        # Simulate intelligent decomposition
        subtasks = []
        
        # Create tasks based on objective complexity
        base_task_count = max(num_squads, 5)
        
        for i in range(base_task_count):
            task = SwarmTask(
                description=f"Subtask {i+1} for {objective.description}",
                priority=objective.priority,
                required_capabilities=self._determine_required_capabilities(i, objective),
                estimated_duration=None  # Will be estimated by squad
            )
            
            # Add dependencies for sequential tasks
            if i > 0 and random.random() > 0.5:
                task.dependencies.append(subtasks[i-1].id)
            
            subtasks.append(task)
        
        # Store in swarm task registry
        for task in subtasks:
            self.swarm.tasks[task.id] = task
        
        return subtasks
    
    def _determine_required_capabilities(
        self,
        task_index: int,
        objective: SwarmObjective
    ) -> List[str]:
        """Determine required capabilities for a task"""
        base_capabilities = ["analysis", "processing"]
        
        # Add strategy-specific capabilities
        if objective.strategy.value == "research":
            base_capabilities.extend(["web_search", "data_collection"])
        elif objective.strategy.value == "development":
            base_capabilities.extend(["code_generation", "testing"])
        elif objective.strategy.value == "analysis":
            base_capabilities.extend(["statistical_modeling", "visualization"])
        
        return base_capabilities[:2]  # Limit to 2 capabilities per task
    
    async def _assign_tasks_to_squads(
        self,
        tasks: List[SwarmTask],
        squads: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[SwarmTask]]:
        """Assign tasks to squads based on capabilities"""
        assignments = defaultdict(list)
        
        for task in tasks:
            best_squad = None
            best_score = 0
            
            # Find best matching squad
            for squad_name, squad_info in squads.items():
                # Calculate capability match score
                squad_capabilities = set()
                for agent in squad_info["agents"]:
                    squad_capabilities.update(agent.capabilities)
                
                match_score = len(
                    set(task.required_capabilities) & squad_capabilities
                ) / len(task.required_capabilities) if task.required_capabilities else 0.5
                
                if match_score > best_score:
                    best_score = match_score
                    best_squad = squad_name
            
            if best_squad:
                assignments[best_squad].append(task)
        
        return dict(assignments)
    
    async def _execute_hierarchical(
        self,
        commander: SwarmAgent,
        squad_assignments: Dict[str, List[SwarmTask]],
        objective: SwarmObjective
    ) -> Dict[str, Dict[str, Any]]:
        """Execute tasks with hierarchical coordination"""
        execution_results = {}
        
        # Commander monitors overall progress
        commander_task = asyncio.create_task(
            self._commander_monitor(commander, squad_assignments, objective)
        )
        
        # Execute squad tasks in parallel
        squad_tasks = []
        for squad_name, tasks in squad_assignments.items():
            squad_task = asyncio.create_task(
                self._execute_squad_tasks(squad_name, tasks)
            )
            squad_tasks.append((squad_name, squad_task))
        
        # Wait for all squads to complete
        for squad_name, task in squad_tasks:
            try:
                result = await task
                execution_results[squad_name] = result
            except Exception as e:
                logger.error(f"Squad {squad_name} execution failed: {e}")
                execution_results[squad_name] = {
                    "results": {"error": str(e)},
                    "completed": [],
                    "failed": [t.id for t in squad_assignments[squad_name]]
                }
        
        # Cancel commander monitoring
        commander_task.cancel()
        
        return execution_results
    
    async def _commander_monitor(
        self,
        commander: SwarmAgent,
        squad_assignments: Dict[str, List[SwarmTask]],
        objective: SwarmObjective
    ):
        """Commander monitors and coordinates squad progress"""
        while True:
            # Check squad progress
            overall_progress = await self._calculate_overall_progress(squad_assignments)
            
            # Broadcast status update
            status_message = SwarmMessage(
                sender_id=commander.id,
                type=CommunicationType.BROADCAST,
                content={
                    "type": "status_update",
                    "progress": overall_progress,
                    "objective_id": objective.id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            
            await self.swarm.communication_mesh.send_message(status_message)
            
            # Check for issues and intervene if needed
            if overall_progress.get("issues", []):
                await self._commander_intervention(commander, overall_progress["issues"])
            
            await asyncio.sleep(5)  # Monitor every 5 seconds
    
    async def _calculate_overall_progress(
        self,
        squad_assignments: Dict[str, List[SwarmTask]]
    ) -> Dict[str, Any]:
        """Calculate overall execution progress"""
        total_tasks = sum(len(tasks) for tasks in squad_assignments.values())
        completed_tasks = 0
        failed_tasks = 0
        issues = []
        
        for tasks in squad_assignments.values():
            for task in tasks:
                if task.id in self.swarm.tasks:
                    task_obj = self.swarm.tasks[task.id]
                    if task_obj.status == "completed":
                        completed_tasks += 1
                    elif task_obj.status == "failed":
                        failed_tasks += 1
                        issues.append(f"Task {task.id} failed")
        
        progress_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        return {
            "percentage": progress_percentage,
            "completed": completed_tasks,
            "failed": failed_tasks,
            "total": total_tasks,
            "issues": issues
        }
    
    async def _commander_intervention(
        self,
        commander: SwarmAgent,
        issues: List[str]
    ):
        """Commander intervenes to resolve issues"""
        for issue in issues:
            # Simple intervention: reassign failed tasks
            if "failed" in issue:
                task_id = issue.split()[1]
                
                # Find available agent
                available_agents = [
                    agent for agent in self.swarm.agents.values()
                    if agent.status == "idle" and agent.id != commander.id
                ]
                
                if available_agents:
                    # Reassign to best available agent
                    best_agent = max(available_agents, key=lambda a: a.performance_score)
                    
                    reassign_message = SwarmMessage(
                        sender_id=commander.id,
                        recipient_id=best_agent.id,
                        type=CommunicationType.UNICAST,
                        content={
                            "type": "task_reassignment",
                            "task_id": task_id,
                            "priority": "high"
                        }
                    )
                    
                    await self.swarm.communication_mesh.send_message(reassign_message)
    
    async def _execute_squad_tasks(
        self,
        squad_name: str,
        tasks: List[SwarmTask]
    ) -> Dict[str, Any]:
        """Execute tasks assigned to a squad"""
        completed = []
        failed = []
        results = {}
        
        # Execute tasks respecting dependencies
        task_futures = {}
        
        for task in tasks:
            # Wait for dependencies
            if task.dependencies:
                for dep_id in task.dependencies:
                    if dep_id in task_futures:
                        await task_futures[dep_id]
            
            # Execute task
            future = asyncio.create_task(self._execute_single_task(task, squad_name))
            task_futures[task.id] = future
        
        # Wait for all tasks to complete
        for task_id, future in task_futures.items():
            try:
                result = await future
                if result["status"] == "completed":
                    completed.append(task_id)
                    results[task_id] = result["result"]
                else:
                    failed.append(task_id)
            except Exception as e:
                logger.error(f"Task {task_id} execution failed: {e}")
                failed.append(task_id)
        
        return {
            "results": results,
            "completed": completed,
            "failed": failed
        }
    
    async def _execute_single_task(
        self,
        task: SwarmTask,
        squad_name: str
    ) -> Dict[str, Any]:
        """Execute a single task"""
        # Update task status
        task.status = "running"
        task.started_at = datetime.now(timezone.utc)
        
        # Simulate task execution
        await asyncio.sleep(random.uniform(1, 5))
        
        # Determine success (90% success rate for demo)
        success = random.random() < 0.9
        
        task.status = "completed" if success else "failed"
        task.completed_at = datetime.now(timezone.utc)
        
        if success:
            task.result = {
                "output": f"Result for {task.description}",
                "squad": squad_name,
                "duration": (task.completed_at - task.started_at).total_seconds()
            }
        
        return {
            "status": task.status,
            "result": task.result
        }


class MeshSwarmCoordinator(SwarmCoordinator):
    """
    Fully connected peer-to-peer swarm
    Best for: Distributed tasks without central control
    """
    
    async def execute(
        self,
        objective: SwarmObjective,
        agents: List[SwarmAgent],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute mesh coordination with peer-to-peer communication"""
        logger.info(f"Executing mesh swarm for objective: {objective.id}")
        
        completed_tasks = []
        failed_tasks = []
        results = {}
        
        try:
            # Initialize consensus protocol
            consensus = ByzantineConsensus(agents)
            
            # Create task marketplace
            marketplace = TaskMarketplace(self.swarm)
            
            # Agents collectively decompose objective
            tasks = await self._collective_decomposition(agents, objective, consensus)
            
            # Add tasks to marketplace
            for task in tasks:
                await marketplace.add_task(task)
            
            # Agents bid on tasks
            task_assignments = await self._run_task_auction(agents, marketplace)
            
            # Execute with peer coordination
            execution_results = await self._execute_mesh_coordination(
                agents,
                task_assignments,
                consensus
            )
            
            # Aggregate peer results
            for agent_id, agent_results in execution_results.items():
                results[agent_id] = agent_results["results"]
                completed_tasks.extend(agent_results["completed"])
                failed_tasks.extend(agent_results["failed"])
            
            # Calculate consensus on final result
            final_consensus = await consensus.reach_consensus(
                "final_result",
                results,
                timeout=30
            )
            
            success_rate = len(completed_tasks) / (len(completed_tasks) + len(failed_tasks))
            
            return {
                "status": "completed" if success_rate > 0.8 else "partial",
                "results": final_consensus,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "confidence": consensus.confidence,
                "mesh_topology": self._get_mesh_topology(agents)
            }
            
        except Exception as e:
            logger.error(f"Mesh coordination failed: {e}")
            return {
                "status": "failed",
                "results": {"error": str(e)},
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "confidence": 0.0
            }
    
    async def _collective_decomposition(
        self,
        agents: List[SwarmAgent],
        objective: SwarmObjective,
        consensus: 'ByzantineConsensus'
    ) -> List[SwarmTask]:
        """Agents collectively decompose objective through consensus"""
        proposals = []
        
        # Each agent proposes task decomposition
        for agent in agents[:5]:  # Limit to first 5 agents for efficiency
            agent_proposal = await self._agent_propose_decomposition(agent, objective)
            proposals.append((agent.id, agent_proposal))
        
        # Reach consensus on decomposition
        consensus_decomposition = await consensus.reach_consensus(
            "task_decomposition",
            proposals,
            timeout=10
        )
        
        # Convert consensus to tasks
        tasks = []
        for i, task_desc in enumerate(consensus_decomposition):
            task = SwarmTask(
                description=task_desc,
                priority=objective.priority,
                required_capabilities=["generic_processing"]  # Will be refined during bidding
            )
            tasks.append(task)
            self.swarm.tasks[task.id] = task
        
        return tasks
    
    async def _agent_propose_decomposition(
        self,
        agent: SwarmAgent,
        objective: SwarmObjective
    ) -> List[str]:
        """Individual agent proposes task decomposition"""
        # Simulate agent reasoning about decomposition
        base_tasks = [
            f"Analyze requirements for {objective.description}",
            f"Design solution approach for {objective.description}",
            f"Implement core functionality for {objective.description}",
            f"Test and validate solution for {objective.description}",
            f"Optimize and refine for {objective.description}"
        ]
        
        # Add agent-specific insights based on capabilities
        if "research" in agent.capabilities:
            base_tasks.insert(0, f"Research background for {objective.description}")
        elif "optimization" in agent.capabilities:
            base_tasks.append(f"Performance optimization for {objective.description}")
        
        return base_tasks
    
    async def _run_task_auction(
        self,
        agents: List[SwarmAgent],
        marketplace: 'TaskMarketplace'
    ) -> Dict[str, List[SwarmTask]]:
        """Run decentralized task auction"""
        assignments = defaultdict(list)
        
        # Get all available tasks
        available_tasks = await marketplace.get_available_tasks()
        
        # Each agent bids on tasks
        bids = {}
        for task in available_tasks:
            task_bids = []
            
            for agent in agents:
                if agent.status == "idle" or agent.load < 0.8:
                    bid = await self._calculate_agent_bid(agent, task)
                    task_bids.append((agent.id, bid))
            
            bids[task.id] = task_bids
        
        # Award tasks to highest bidders
        for task_id, task_bids in bids.items():
            if task_bids:
                # Sort by bid value
                task_bids.sort(key=lambda x: x[1], reverse=True)
                winner_id = task_bids[0][0]
                
                task = self.swarm.tasks[task_id]
                task.agent_id = winner_id
                assignments[winner_id].append(task)
                
                # Update agent load
                agent = self.swarm.agents[winner_id]
                agent.load = min(1.0, agent.load + 0.2)
                agent.status = "busy"
        
        return dict(assignments)
    
    async def _calculate_agent_bid(
        self,
        agent: SwarmAgent,
        task: SwarmTask
    ) -> float:
        """Calculate agent's bid for a task"""
        # Base bid on capability match
        capability_match = len(
            set(agent.capabilities) & set(task.required_capabilities)
        ) / len(task.required_capabilities) if task.required_capabilities else 0.5
        
        # Factor in current load (less loaded agents bid higher)
        load_factor = 1.0 - agent.load
        
        # Factor in performance score
        performance_factor = agent.performance_score
        
        # Calculate final bid
        bid = (capability_match * 0.4 + load_factor * 0.3 + performance_factor * 0.3)
        
        return bid
    
    async def _execute_mesh_coordination(
        self,
        agents: List[SwarmAgent],
        task_assignments: Dict[str, List[SwarmTask]],
        consensus: 'ByzantineConsensus'
    ) -> Dict[str, Dict[str, Any]]:
        """Execute tasks with mesh coordination"""
        execution_results = {}
        
        # Create peer monitoring tasks
        monitoring_tasks = []
        for agent in agents:
            monitor_task = asyncio.create_task(
                self._peer_monitor(agent, agents, consensus)
            )
            monitoring_tasks.append(monitor_task)
        
        # Execute assigned tasks
        agent_tasks = []
        for agent_id, tasks in task_assignments.items():
            if agent_id in self.swarm.agents:
                agent = self.swarm.agents[agent_id]
                agent_task = asyncio.create_task(
                    self._execute_agent_tasks(agent, tasks, consensus)
                )
                agent_tasks.append((agent_id, agent_task))
        
        # Wait for all agents to complete
        for agent_id, task in agent_tasks:
            try:
                result = await task
                execution_results[agent_id] = result
            except Exception as e:
                logger.error(f"Agent {agent_id} execution failed: {e}")
                execution_results[agent_id] = {
                    "results": {"error": str(e)},
                    "completed": [],
                    "failed": [t.id for t in task_assignments.get(agent_id, [])]
                }
        
        # Cancel monitoring
        for task in monitoring_tasks:
            task.cancel()
        
        return execution_results
    
    async def _peer_monitor(
        self,
        agent: SwarmAgent,
        peers: List[SwarmAgent],
        consensus: 'ByzantineConsensus'
    ):
        """Agent monitors peer health and progress"""
        while True:
            # Check random peers
            peers_to_check = random.sample(
                [p for p in peers if p.id != agent.id],
                min(3, len(peers) - 1)
            )
            
            for peer in peers_to_check:
                # Send health check
                health_msg = SwarmMessage(
                    sender_id=agent.id,
                    recipient_id=peer.id,
                    type=CommunicationType.UNICAST,
                    content={
                        "type": "health_check",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                
                await self.swarm.communication_mesh.send_message(health_msg)
            
            # Share status with peers
            status_msg = SwarmMessage(
                sender_id=agent.id,
                type=CommunicationType.MULTICAST,
                content={
                    "type": "peer_status",
                    "agent_id": agent.id,
                    "load": agent.load,
                    "completed_tasks": len(agent.completed_tasks),
                    "status": agent.status
                }
            )
            
            await self.swarm.communication_mesh.send_message(status_msg)
            
            await asyncio.sleep(10)  # Monitor every 10 seconds
    
    async def _execute_agent_tasks(
        self,
        agent: SwarmAgent,
        tasks: List[SwarmTask],
        consensus: 'ByzantineConsensus'
    ) -> Dict[str, Any]:
        """Execute tasks assigned to an agent with peer coordination"""
        completed = []
        failed = []
        results = {}
        
        for task in tasks:
            try:
                # Announce task start to peers
                start_msg = SwarmMessage(
                    sender_id=agent.id,
                    type=CommunicationType.MULTICAST,
                    content={
                        "type": "task_started",
                        "task_id": task.id,
                        "agent_id": agent.id
                    }
                )
                await self.swarm.communication_mesh.send_message(start_msg)
                
                # Execute task
                result = await self._execute_task_with_peer_support(agent, task, consensus)
                
                if result["status"] == "completed":
                    completed.append(task.id)
                    results[task.id] = result["result"]
                    
                    # Share success with peers
                    success_msg = SwarmMessage(
                        sender_id=agent.id,
                        type=CommunicationType.MULTICAST,
                        content={
                            "type": "task_completed",
                            "task_id": task.id,
                            "result_summary": str(result["result"])[:100]
                        }
                    )
                    await self.swarm.communication_mesh.send_message(success_msg)
                else:
                    failed.append(task.id)
                    
            except Exception as e:
                logger.error(f"Task {task.id} execution failed: {e}")
                failed.append(task.id)
        
        # Update agent state
        agent.completed_tasks.extend(completed)
        agent.load = max(0, agent.load - 0.2 * len(tasks))
        agent.status = "idle" if agent.load < 0.2 else "busy"
        
        return {
            "results": results,
            "completed": completed,
            "failed": failed
        }
    
    async def _execute_task_with_peer_support(
        self,
        agent: SwarmAgent,
        task: SwarmTask,
        consensus: 'ByzantineConsensus'
    ) -> Dict[str, Any]:
        """Execute task with potential peer assistance"""
        # Update task status
        task.status = "running"
        task.started_at = datetime.now(timezone.utc)
        
        # Check if task needs peer assistance
        if task.priority > 7 or len(task.required_capabilities) > 2:
            # Request peer assistance
            assistance_msg = SwarmMessage(
                sender_id=agent.id,
                type=CommunicationType.BROADCAST,
                content={
                    "type": "assistance_request",
                    "task_id": task.id,
                    "required_capabilities": task.required_capabilities
                }
            )
            await self.swarm.communication_mesh.send_message(assistance_msg)
            
            # Wait briefly for responses
            await asyncio.sleep(2)
        
        # Simulate task execution
        await asyncio.sleep(random.uniform(2, 6))
        
        # Determine success
        success = random.random() < 0.85
        
        task.status = "completed" if success else "failed"
        task.completed_at = datetime.now(timezone.utc)
        
        if success:
            task.result = {
                "output": f"Mesh result for {task.description}",
                "agent": agent.id,
                "peer_assisted": task.priority > 7,
                "duration": (task.completed_at - task.started_at).total_seconds()
            }
        
        return {
            "status": task.status,
            "result": task.result
        }
    
    def _get_mesh_topology(self, agents: List[SwarmAgent]) -> Dict[str, Any]:
        """Get current mesh network topology"""
        topology = {
            "nodes": len(agents),
            "edges": 0,
            "average_degree": 0,
            "connected": True
        }
        
        # Calculate mesh statistics
        total_connections = sum(len(agent.neighbors) for agent in agents)
        topology["edges"] = total_connections // 2  # Each edge counted twice
        topology["average_degree"] = total_connections / len(agents) if agents else 0
        
        return topology


class AdaptiveSwarmCoordinator(SwarmCoordinator):
    """
    Dynamic swarm that changes structure based on performance
    Best for: Complex, evolving objectives
    """
    
    async def execute(
        self,
        objective: SwarmObjective,
        agents: List[SwarmAgent],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute adaptive coordination with dynamic restructuring"""
        logger.info(f"Executing adaptive swarm for objective: {objective.id}")
        
        completed_tasks = []
        failed_tasks = []
        results = {}
        
        try:
            # Initialize with default pattern
            current_pattern = SwarmPattern.HIERARCHICAL
            coordinator = None
            
            # Performance monitoring
            performance_monitor = PerformanceMonitor()
            structure_optimizer = StructureOptimizer(self.swarm)
            
            # Adaptive execution loop
            adaptation_count = 0
            max_adaptations = 3
            
            while adaptation_count < max_adaptations:
                # Execute with current pattern
                if current_pattern == SwarmPattern.HIERARCHICAL:
                    coordinator = HierarchicalSwarmCoordinator(self.swarm)
                elif current_pattern == SwarmPattern.MESH:
                    coordinator = MeshSwarmCoordinator(self.swarm)
                else:
                    coordinator = DistributedSwarmCoordinator(self.swarm)
                
                # Execute for a time window
                execution_task = asyncio.create_task(
                    coordinator.execute(objective, agents, config)
                )
                
                # Monitor performance
                monitoring_task = asyncio.create_task(
                    self._monitor_and_adapt(
                        performance_monitor,
                        structure_optimizer,
                        agents,
                        objective
                    )
                )
                
                # Wait for execution or adaptation trigger
                done, pending = await asyncio.wait(
                    [execution_task, monitoring_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                if execution_task in done:
                    # Execution completed
                    result = await execution_task
                    monitoring_task.cancel()
                    
                    results.update(result.get("results", {}))
                    completed_tasks.extend(result.get("completed_tasks", []))
                    failed_tasks.extend(result.get("failed_tasks", []))
                    
                    break
                    
                else:
                    # Adaptation triggered
                    adaptation = await monitoring_task
                    execution_task.cancel()
                    
                    if adaptation["should_adapt"]:
                        # Apply adaptation
                        current_pattern = adaptation["new_pattern"]
                        agents = await self._apply_adaptation(
                            agents,
                            adaptation,
                            objective
                        )
                        
                        adaptation_count += 1
                        logger.info(
                            f"Adapted to {current_pattern.value} pattern "
                            f"(adaptation {adaptation_count}/{max_adaptations})"
                        )
                    else:
                        break
            
            # Calculate final metrics
            success_rate = len(completed_tasks) / (len(completed_tasks) + len(failed_tasks))
            
            return {
                "status": "completed" if success_rate > 0.8 else "partial",
                "results": results,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "confidence": success_rate,
                "adaptations": adaptation_count,
                "final_pattern": current_pattern.value
            }
            
        except Exception as e:
            logger.error(f"Adaptive coordination failed: {e}")
            return {
                "status": "failed",
                "results": {"error": str(e)},
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "confidence": 0.0
            }
    
    async def _monitor_and_adapt(
        self,
        performance_monitor: 'PerformanceMonitor',
        structure_optimizer: 'StructureOptimizer',
        agents: List[SwarmAgent],
        objective: SwarmObjective
    ) -> Dict[str, Any]:
        """Monitor performance and determine if adaptation is needed"""
        monitoring_window = 30  # seconds
        samples = []
        
        start_time = time.time()
        while time.time() - start_time < monitoring_window:
            # Collect performance metrics
            metrics = await performance_monitor.collect_metrics(agents, self.swarm.tasks)
            samples.append(metrics)
            
            await asyncio.sleep(5)  # Sample every 5 seconds
        
        # Analyze performance trends
        analysis = await performance_monitor.analyze_trends(samples)
        
        # Determine if adaptation is needed
        should_adapt = (
            analysis["efficiency"] < 0.6 or
            analysis["latency"] > 200 or
            analysis["failure_rate"] > 0.3 or
            analysis["bottleneck_detected"]
        )
        
        adaptation_recommendation = {
            "should_adapt": should_adapt,
            "new_pattern": SwarmPattern.HIERARCHICAL,
            "reason": "performance baseline"
        }
        
        if should_adapt:
            # Determine new pattern
            if analysis["bottleneck_detected"] and analysis["communication_overhead"] > 0.3:
                adaptation_recommendation["new_pattern"] = SwarmPattern.DISTRIBUTED
                adaptation_recommendation["reason"] = "communication bottleneck"
            elif analysis["coordination_overhead"] > 0.4:
                adaptation_recommendation["new_pattern"] = SwarmPattern.MESH
                adaptation_recommendation["reason"] = "coordination overhead"
            elif analysis["efficiency"] < 0.5:
                adaptation_recommendation["new_pattern"] = SwarmPattern.HIERARCHICAL
                adaptation_recommendation["reason"] = "low efficiency"
            
            # Get structure optimization recommendations
            optimization = await structure_optimizer.optimize(
                agents,
                objective,
                analysis
            )
            adaptation_recommendation.update(optimization)
        
        return adaptation_recommendation
    
    async def _apply_adaptation(
        self,
        agents: List[SwarmAgent],
        adaptation: Dict[str, Any],
        objective: SwarmObjective
    ) -> List[SwarmAgent]:
        """Apply adaptation to swarm structure"""
        # Reset agent states
        for agent in agents:
            agent.status = "idle"
            agent.current_task = None
            agent.neighbors.clear()
        
        # Apply agent additions/removals
        if "add_agents" in adaptation:
            for agent_spec in adaptation["add_agents"]:
                new_agent = SwarmAgent(
                    name=f"{agent_spec['type']}_adaptive_{len(agents)}",
                    type=agent_spec["type"],
                    capabilities=self.swarm._get_agent_capabilities(
                        agent_spec["type"],
                        objective.strategy
                    )
                )
                agents.append(new_agent)
                self.swarm.agents[new_agent.id] = new_agent
        
        if "remove_agents" in adaptation:
            agents_to_remove = []
            for agent_id in adaptation["remove_agents"]:
                agent = next((a for a in agents if a.id == agent_id), None)
                if agent and len(agent.completed_tasks) == 0:
                    agents_to_remove.append(agent)
            
            for agent in agents_to_remove:
                agents.remove(agent)
                del self.swarm.agents[agent.id]
        
        # Rebuild communication mesh for new pattern
        await self.swarm.communication_mesh.establish_mesh_network(agents)
        
        # Apply parameter adjustments
        if "parameter_adjustments" in adaptation:
            for param, value in adaptation["parameter_adjustments"].items():
                # Apply adjustments (implementation specific)
                logger.info(f"Adjusted {param} to {value}")
        
        return agents


class DistributedSwarmCoordinator(SwarmCoordinator):
    """
    Fully distributed coordination without central control
    Best for: Highly parallel, independent tasks
    """
    
    async def execute(
        self,
        objective: SwarmObjective,
        agents: List[SwarmAgent],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute distributed coordination with stigmergic communication"""
        logger.info(f"Executing distributed swarm for objective: {objective.id}")
        
        completed_tasks = []
        failed_tasks = []
        results = {}
        
        try:
            # Initialize stigmergic environment
            environment = StigmergicEnvironment(self.swarm.communication_mesh)
            
            # Agents independently explore and create tasks
            exploration_tasks = []
            for agent in agents:
                task = asyncio.create_task(
                    self._agent_explore_and_execute(
                        agent,
                        objective,
                        environment
                    )
                )
                exploration_tasks.append((agent.id, task))
            
            # Monitor emergence
            emergence_task = asyncio.create_task(
                self._monitor_stigmergic_emergence(agents, environment)
            )
            
            # Wait for all agents to complete exploration
            agent_results = {}
            for agent_id, task in exploration_tasks:
                try:
                    result = await task
                    agent_results[agent_id] = result
                    results[agent_id] = result["results"]
                    completed_tasks.extend(result["completed"])
                    failed_tasks.extend(result["failed"])
                except Exception as e:
                    logger.error(f"Agent {agent_id} exploration failed: {e}")
                    agent_results[agent_id] = {
                        "results": {"error": str(e)},
                        "completed": [],
                        "failed": []
                    }
            
            # Get emergence patterns
            emergence_patterns = await emergence_task
            
            # Aggregate distributed results
            aggregated_results = await self._aggregate_distributed_results(
                agent_results,
                emergence_patterns
            )
            
            success_rate = len(completed_tasks) / (len(completed_tasks) + len(failed_tasks))
            
            return {
                "status": "completed" if success_rate > 0.7 else "partial",
                "results": aggregated_results,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "confidence": success_rate,
                "emergence_patterns": emergence_patterns,
                "pheromone_trails": environment.get_pheromone_summary()
            }
            
        except Exception as e:
            logger.error(f"Distributed coordination failed: {e}")
            return {
                "status": "failed",
                "results": {"error": str(e)},
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "confidence": 0.0
            }
    
    async def _agent_explore_and_execute(
        self,
        agent: SwarmAgent,
        objective: SwarmObjective,
        environment: 'StigmergicEnvironment'
    ) -> Dict[str, Any]:
        """Agent autonomously explores and executes tasks"""
        completed = []
        failed = []
        results = {}
        discovered_tasks = []
        
        exploration_duration = 60  # seconds
        start_time = time.time()
        
        while time.time() - start_time < exploration_duration:
            # Sense environment
            pheromones = await environment.sense(agent.position)
            
            # Decide action based on pheromones
            action = await self._decide_action(agent, pheromones, objective)
            
            if action["type"] == "explore":
                # Move to new position
                agent.position += action["direction"] * 0.1
                agent.position = np.clip(agent.position, 0, 1)  # Keep in bounds
                
                # Possibly discover new task
                if random.random() < 0.3:  # 30% chance
                    task = await self._discover_task(agent, objective)
                    discovered_tasks.append(task)
                    
                    # Deposit task pheromone
                    await environment.deposit_pheromone(
                        Pheromone(
                            agent_id=agent.id,
                            position=agent.position.copy(),
                            type="task_discovered",
                            strength=1.0,
                            metadata={"task_id": task.id}
                        )
                    )
                    
            elif action["type"] == "exploit":
                # Execute discovered task
                if discovered_tasks:
                    task = discovered_tasks.pop(0)
                    result = await self._execute_distributed_task(agent, task)
                    
                    if result["status"] == "completed":
                        completed.append(task.id)
                        results[task.id] = result["result"]
                        
                        # Deposit success pheromone
                        await environment.deposit_pheromone(
                            Pheromone(
                                agent_id=agent.id,
                                position=agent.position.copy(),
                                type="task_completed",
                                strength=2.0,
                                metadata={"task_id": task.id}
                            )
                        )
                    else:
                        failed.append(task.id)
                        
            elif action["type"] == "follow":
                # Follow pheromone trail
                if action["target_position"] is not None:
                    direction = action["target_position"] - agent.position
                    agent.position += direction * 0.2
                    agent.position = np.clip(agent.position, 0, 1)
            
            # Small delay to prevent tight loop
            await asyncio.sleep(0.5)
        
        # Update agent final state
        agent.completed_tasks.extend(completed)
        
        return {
            "results": results,
            "completed": completed,
            "failed": failed,
            "discovered_tasks": len(discovered_tasks) + len(completed) + len(failed)
        }
    
    async def _decide_action(
        self,
        agent: SwarmAgent,
        pheromones: List[Pheromone],
        objective: SwarmObjective
    ) -> Dict[str, Any]:
        """Decide agent action based on pheromones and state"""
        # Analyze pheromone types
        task_pheromones = [p for p in pheromones if p.type == "task_discovered"]
        success_pheromones = [p for p in pheromones if p.type == "task_completed"]
        
        # Exploration vs exploitation decision
        exploration_tendency = 0.7 - len(agent.completed_tasks) * 0.1
        
        if random.random() < exploration_tendency:
            # Explore new areas
            return {
                "type": "explore",
                "direction": np.random.randn(3) * 0.5
            }
        elif task_pheromones:
            # Exploit discovered tasks
            return {"type": "exploit"}
        elif success_pheromones:
            # Follow success trails
            strongest = max(success_pheromones, key=lambda p: p.strength)
            return {
                "type": "follow",
                "target_position": strongest.position
            }
        else:
            # Random walk
            return {
                "type": "explore",
                "direction": np.random.randn(3) * 0.3
            }
    
    async def _discover_task(
        self,
        agent: SwarmAgent,
        objective: SwarmObjective
    ) -> SwarmTask:
        """Agent discovers a new task"""
        task = SwarmTask(
            description=f"Distributed task for {objective.description} at position {agent.position}",
            priority=random.randint(3, 8),
            required_capabilities=random.sample(agent.capabilities, k=min(2, len(agent.capabilities)))
        )
        
        self.swarm.tasks[task.id] = task
        return task
    
    async def _execute_distributed_task(
        self,
        agent: SwarmAgent,
        task: SwarmTask
    ) -> Dict[str, Any]:
        """Execute task in distributed manner"""
        task.status = "running"
        task.started_at = datetime.now(timezone.utc)
        task.agent_id = agent.id
        
        # Simulate distributed execution
        await asyncio.sleep(random.uniform(1, 4))
        
        # Success rate based on capability match
        capability_match = len(
            set(agent.capabilities) & set(task.required_capabilities)
        ) / len(task.required_capabilities) if task.required_capabilities else 0.8
        
        success = random.random() < (0.7 + capability_match * 0.3)
        
        task.status = "completed" if success else "failed"
        task.completed_at = datetime.now(timezone.utc)
        
        if success:
            task.result = {
                "output": f"Distributed result for {task.description}",
                "agent": agent.id,
                "position": agent.position.tolist(),
                "duration": (task.completed_at - task.started_at).total_seconds()
            }
        
        return {
            "status": task.status,
            "result": task.result
        }
    
    async def _monitor_stigmergic_emergence(
        self,
        agents: List[SwarmAgent],
        environment: 'StigmergicEnvironment'
    ) -> List[str]:
        """Monitor for emergent patterns in stigmergic system"""
        patterns = []
        monitoring_duration = 60
        start_time = time.time()
        
        while time.time() - start_time < monitoring_duration:
            # Check for trail formation
            trails = environment.detect_pheromone_trails()
            if trails:
                patterns.append("pheromone_trails")
            
            # Check for agent clustering around resources
            clusters = self._detect_agent_clusters(agents)
            if clusters > 1:
                patterns.append("resource_clustering")
            
            # Check for collective problem solving
            task_completion_rate = self._calculate_task_completion_rate()
            if task_completion_rate > 0.8:
                patterns.append("collective_efficiency")
            
            await asyncio.sleep(10)
        
        return list(set(patterns))
    
    def _detect_agent_clusters(self, agents: List[SwarmAgent]) -> int:
        """Detect number of agent clusters"""
        if len(agents) < 3:
            return 0
        
        # Simple clustering based on position proximity
        clusters = 0
        visited = set()
        
        for i, agent in enumerate(agents):
            if i in visited:
                continue
                
            cluster_members = [i]
            visited.add(i)
            
            # Find nearby agents
            for j, other in enumerate(agents[i+1:], i+1):
                if j not in visited:
                    distance = np.linalg.norm(agent.position - other.position)
                    if distance < 0.3:  # Proximity threshold
                        cluster_members.append(j)
                        visited.add(j)
            
            if len(cluster_members) >= 2:
                clusters += 1
        
        return clusters
    
    def _calculate_task_completion_rate(self) -> float:
        """Calculate recent task completion rate"""
        total_tasks = len(self.swarm.tasks)
        if total_tasks == 0:
            return 0.0
        
        completed = sum(
            1 for task in self.swarm.tasks.values()
            if task.status == "completed"
        )
        
        return completed / total_tasks
    
    async def _aggregate_distributed_results(
        self,
        agent_results: Dict[str, Dict[str, Any]],
        emergence_patterns: List[str]
    ) -> Dict[str, Any]:
        """Aggregate results from distributed execution"""
        aggregated = {
            "total_tasks_discovered": sum(
                r.get("discovered_tasks", 0) for r in agent_results.values()
            ),
            "emergence_patterns": emergence_patterns,
            "agent_contributions": {}
        }
        
        # Merge all results
        all_results = {}
        for agent_id, result in agent_results.items():
            all_results.update(result.get("results", {}))
            
            # Track agent contributions
            aggregated["agent_contributions"][agent_id] = {
                "tasks_completed": len(result.get("completed", [])),
                "tasks_failed": len(result.get("failed", [])),
                "efficiency": len(result.get("completed", [])) / 
                            (len(result.get("completed", [])) + len(result.get("failed", [])) + 1)
            }
        
        aggregated["merged_results"] = all_results
        
        return aggregated


# Supporting Classes

class ByzantineConsensus:
    """Byzantine fault-tolerant consensus for mesh swarms"""
    
    def __init__(self, agents: List[SwarmAgent]):
        self.agents = agents
        self.votes = defaultdict(lambda: defaultdict(list))
        self.confidence = 0.0
    
    async def reach_consensus(
        self,
        topic: str,
        proposals: Any,
        timeout: float = 10.0
    ) -> Any:
        """Reach consensus on a topic through voting"""
        # Simplified Byzantine consensus
        if isinstance(proposals, list) and len(proposals) > 0:
            # Each agent votes
            votes = defaultdict(int)
            
            for proposal in proposals:
                # Hash proposal for voting
                proposal_key = str(proposal)[:100]
                votes[proposal_key] += 1
            
            # Find proposal with most votes
            if votes:
                best_proposal = max(votes, key=votes.get)
                self.confidence = votes[best_proposal] / len(self.agents)
                
                # Find original proposal
                for proposal in proposals:
                    if str(proposal)[:100] == best_proposal:
                        return proposal
        
        return proposals[0] if proposals else None


class TaskMarketplace:
    """Decentralized task marketplace for mesh swarms"""
    
    def __init__(self, swarm):
        self.swarm = swarm
        self.available_tasks = asyncio.Queue()
        self.task_bids = defaultdict(list)
    
    async def add_task(self, task: SwarmTask):
        """Add task to marketplace"""
        await self.available_tasks.put(task)
    
    async def get_available_tasks(self) -> List[SwarmTask]:
        """Get all available tasks"""
        tasks = []
        
        # Drain queue temporarily
        while not self.available_tasks.empty():
            try:
                task = self.available_tasks.get_nowait()
                tasks.append(task)
            except asyncio.QueueEmpty:
                break
        
        # Put tasks back
        for task in tasks:
            await self.available_tasks.put(task)
        
        return tasks


class PerformanceMonitor:
    """Monitor swarm performance metrics"""
    
    async def collect_metrics(
        self,
        agents: List[SwarmAgent],
        tasks: Dict[str, SwarmTask]
    ) -> Dict[str, float]:
        """Collect current performance metrics"""
        total_tasks = len(tasks)
        completed_tasks = sum(1 for t in tasks.values() if t.status == "completed")
        failed_tasks = sum(1 for t in tasks.values() if t.status == "failed")
        running_tasks = sum(1 for t in tasks.values() if t.status == "running")
        
        # Agent metrics
        avg_load = np.mean([a.load for a in agents]) if agents else 0
        idle_agents = sum(1 for a in agents if a.status == "idle")
        
        # Task metrics
        task_durations = []
        for task in tasks.values():
            if task.completed_at and task.started_at:
                duration = (task.completed_at - task.started_at).total_seconds()
                task_durations.append(duration)
        
        avg_task_duration = np.mean(task_durations) if task_durations else 0
        
        return {
            "efficiency": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "failure_rate": failed_tasks / total_tasks if total_tasks > 0 else 0,
            "latency": avg_task_duration,
            "throughput": completed_tasks / max(1, avg_task_duration),
            "agent_utilization": 1 - (idle_agents / len(agents)) if agents else 0,
            "average_load": avg_load,
            "running_tasks": running_tasks
        }
    
    async def analyze_trends(
        self,
        samples: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Analyze performance trends"""
        if not samples:
            return {
                "efficiency": 0.5,
                "latency": 100,
                "failure_rate": 0.1,
                "bottleneck_detected": False,
                "communication_overhead": 0.1,
                "coordination_overhead": 0.1
            }
        
        # Calculate trends
        efficiencies = [s["efficiency"] for s in samples]
        latencies = [s["latency"] for s in samples]
        failure_rates = [s["failure_rate"] for s in samples]
        
        # Detect bottlenecks
        utilizations = [s["agent_utilization"] for s in samples]
        bottleneck_detected = (
            np.std(utilizations) > 0.3 or  # Uneven load distribution
            any(s["running_tasks"] > len(samples[0]) * 0.5 for s in samples)  # Task queue buildup
        )
        
        return {
            "efficiency": np.mean(efficiencies),
            "latency": np.mean(latencies),
            "failure_rate": np.mean(failure_rates),
            "bottleneck_detected": bottleneck_detected,
            "communication_overhead": 0.2,  # Simplified
            "coordination_overhead": 0.2,  # Simplified
            "efficiency_trend": "declining" if efficiencies[-1] < efficiencies[0] else "stable"
        }


class StructureOptimizer:
    """Optimize swarm structure dynamically"""
    
    def __init__(self, swarm):
        self.swarm = swarm
    
    async def optimize(
        self,
        agents: List[SwarmAgent],
        objective: SwarmObjective,
        performance_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate structure optimization recommendations"""
        recommendations = {
            "add_agents": [],
            "remove_agents": [],
            "parameter_adjustments": {}
        }
        
        # Agent count optimization
        if performance_analysis["efficiency"] < 0.5:
            # Add specialized agents
            recommendations["add_agents"].append({
                "type": "optimizer",
                "count": 2
            })
        
        if performance_analysis["bottleneck_detected"]:
            # Add agents to relieve bottleneck
            recommendations["add_agents"].append({
                "type": "processor",
                "count": 3
            })
        
        # Remove underperforming agents
        underperformers = [
            a for a in agents
            if a.performance_score < 0.5 and len(a.completed_tasks) < 2
        ]
        recommendations["remove_agents"] = [a.id for a in underperformers[:2]]
        
        # Parameter adjustments
        if performance_analysis["communication_overhead"] > 0.3:
            recommendations["parameter_adjustments"]["message_frequency"] = 0.5
        
        if performance_analysis["coordination_overhead"] > 0.3:
            recommendations["parameter_adjustments"]["consensus_threshold"] = 0.6
        
        return recommendations


class StigmergicEnvironment:
    """Environment for stigmergic coordination"""
    
    def __init__(self, communication_mesh):
        self.mesh = communication_mesh
        self.pheromone_grid = defaultdict(list)
    
    async def deposit_pheromone(self, pheromone: Pheromone):
        """Deposit pheromone in environment"""
        await self.mesh.deposit_pheromone(pheromone)
    
    async def sense(self, position: np.ndarray, radius: float = 0.5) -> List[Pheromone]:
        """Sense pheromones at position"""
        return await self.mesh.sense_pheromones(position, radius)
    
    def detect_pheromone_trails(self) -> bool:
        """Detect if pheromone trails have formed"""
        # Check for connected pheromone deposits
        strong_pheromones = []
        
        for pheromones in self.mesh.pheromone_map.values():
            strong = [p for p in pheromones if p.strength > 0.5]
            strong_pheromones.extend(strong)
        
        # Simple trail detection: multiple strong pheromones in proximity
        if len(strong_pheromones) < 3:
            return False
        
        # Check for connected path
        for i, p1 in enumerate(strong_pheromones):
            nearby_count = sum(
                1 for p2 in strong_pheromones[i+1:]
                if np.linalg.norm(p1.position - p2.position) < 0.3
            )
            if nearby_count >= 2:
                return True
        
        return False
    
    def get_pheromone_summary(self) -> Dict[str, Any]:
        """Get summary of pheromone state"""
        total_pheromones = sum(len(p) for p in self.mesh.pheromone_map.values())
        
        pheromone_types = defaultdict(int)
        for pheromones in self.mesh.pheromone_map.values():
            for p in pheromones:
                pheromone_types[p.type] += 1
        
        return {
            "total_pheromones": total_pheromones,
            "pheromone_types": dict(pheromone_types),
            "active_locations": len(self.mesh.pheromone_map),
            "trails_detected": self.detect_pheromone_trails()
        }