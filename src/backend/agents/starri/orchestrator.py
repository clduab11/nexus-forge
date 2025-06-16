"""
Starri Orchestrator - Advanced AI Coordinator with Gemini-2.5-Flash-Thinking

This orchestrator uses Google's Gemini-2.5-Flash-Thinking model for deep reasoning,
reflection, and complex task coordination across multiple AI agents.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from nexus_forge.core.cache import CacheStrategy, RedisCache
from nexus_forge.core.exceptions import (
    AgentError,
    CoordinationError,
    OrchestrationError,
    TaskDecompositionError,
)
from nexus_forge.core.monitoring import get_logger
from nexus_forge.integrations.google.gemini_client import GeminiClient
from nexus_forge.integrations.mem0.knowledge_client import Mem0KnowledgeClient
from nexus_forge.integrations.supabase.coordination_client import (
    SupabaseCoordinationClient,
)

logger = get_logger(__name__)


class ThinkingMode(Enum):
    """Different thinking modes for the orchestrator"""

    DEEP_ANALYSIS = "deep_analysis"  # For complex problem solving
    QUICK_DECISION = "quick_decision"  # For rapid responses
    REFLECTION = "reflection"  # For self-evaluation
    PLANNING = "planning"  # For task decomposition
    COORDINATION = "coordination"  # For agent management


class AgentCapability(Enum):
    """Capabilities that agents can have"""

    CODE_GENERATION = "code_generation"
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    TEXT_GENERATION = "text_generation"
    DATA_ANALYSIS = "data_analysis"
    API_INTEGRATION = "api_integration"
    UI_DESIGN = "ui_design"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    OPTIMIZATION = "optimization"


class StarriOrchestrator:
    """
    Master AI Orchestrator using Gemini-2.5-Flash-Thinking for intelligent coordination

    Features:
    - Deep thinking and reflection capabilities
    - Dynamic task decomposition
    - Real-time agent coordination
    - Knowledge graph integration
    - Performance monitoring
    """

    def __init__(
        self,
        project_id: str,
        supabase_url: str,
        supabase_key: str,
        mem0_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        redis_url: Optional[str] = None,
    ):
        """Initialize the Starri orchestrator with all integrations"""
        self.project_id = project_id
        self.orchestrator_id = f"starri_{uuid4().hex[:8]}"

        # Initialize Gemini client with Flash-Thinking model
        self.gemini_client = GeminiClient(project_id=project_id, api_key=gemini_api_key)

        # Initialize coordination client
        self.coordination_client = SupabaseCoordinationClient(
            url=supabase_url, key=supabase_key, project_id=project_id
        )

        # Initialize knowledge client
        self.knowledge_client = Mem0KnowledgeClient(
            api_key=mem0_api_key or "", orchestrator_id=self.orchestrator_id
        )

        # Initialize cache
        self.cache = RedisCache()

        # Agent registry
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_capabilities: Dict[AgentCapability, List[str]] = {}

        # Task management
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()

        # Performance tracking
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_thinking_time": 0.0,
            "total_thinking_time": 0.0,
            "reflection_count": 0,
        }

        # Thinking chain storage
        self.thinking_chains: Dict[str, List[Dict[str, Any]]] = {}

        logger.info(f"Starri Orchestrator initialized with ID: {self.orchestrator_id}")

    async def initialize(self):
        """Initialize all connections and warm up the orchestrator"""
        try:
            # Connect to Supabase
            await self.coordination_client.connect()

            # Register orchestrator as an agent
            await self.coordination_client.register_agent(
                name=f"Starri-{self.orchestrator_id}",
                agent_type="orchestrator",
                capabilities={
                    "thinking_modes": [mode.value for mode in ThinkingMode],
                    "coordination": True,
                    "task_decomposition": True,
                    "reflection": True,
                },
                configuration={
                    "model": "gemini-2.5-flash-thinking",
                    "max_thinking_depth": 10,
                    "reflection_threshold": 0.7,
                },
            )

            # Initialize knowledge graph entities
            await self.knowledge_client.initialize_orchestrator_knowledge()

            # Start background tasks
            asyncio.create_task(self._monitor_agents())
            asyncio.create_task(self._process_task_queue())

            logger.info("Starri Orchestrator fully initialized")

        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise OrchestrationError(f"Initialization failed: {e}")

    async def think_deeply(
        self,
        prompt: str,
        mode: ThinkingMode = ThinkingMode.DEEP_ANALYSIS,
        context: Optional[Dict[str, Any]] = None,
        max_thinking_steps: int = 10,
    ) -> Dict[str, Any]:
        """
        Use Gemini-2.5-Flash-Thinking for deep reasoning and analysis

        Args:
            prompt: The problem or question to think about
            mode: The thinking mode to use
            context: Additional context for thinking
            max_thinking_steps: Maximum thinking iterations

        Returns:
            Dict containing thinking chain, conclusion, and confidence
        """
        start_time = time.time()
        thinking_chain = []

        try:
            # Prepare thinking prompt based on mode
            thinking_prompt = self._prepare_thinking_prompt(prompt, mode, context)

            # Check cache for similar thinking patterns
            cache_key = f"thinking:{mode.value}:{hash(prompt)}"
            cached_result = self.cache.get(cache_key, CacheStrategy.SEMANTIC)
            if cached_result and cached_result.get("confidence", 0) > 0.9:
                logger.info("Using cached high-confidence thinking result")
                return cached_result

            # Initialize thinking chain
            current_thought = {
                "step": 0,
                "mode": mode.value,
                "prompt": prompt,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Iterative thinking process
            for step in range(max_thinking_steps):
                # Generate next thought using Flash-Thinking
                thought_result = await self.gemini_client.generate_content(
                    prompt=thinking_prompt,
                    model_type="flash",  # This will use gemini-2.5-flash-thinking
                    generation_config={
                        "temperature": (
                            0.7 if mode == ThinkingMode.DEEP_ANALYSIS else 0.4
                        ),
                        "max_output_tokens": 2048,
                        "top_p": 0.9,
                    },
                )

                # Parse thinking result
                thought_content = thought_result["content"]
                current_thought = {
                    "step": step + 1,
                    "thought": thought_content,
                    "confidence": self._extract_confidence(thought_content),
                    "requires_more_thinking": self._check_needs_more_thinking(
                        thought_content
                    ),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                thinking_chain.append(current_thought)

                # Check if we've reached a conclusion
                if (
                    not current_thought["requires_more_thinking"]
                    or current_thought["confidence"] > 0.95
                ):
                    break

                # Prepare next iteration prompt
                thinking_prompt = self._prepare_continuation_prompt(
                    thinking_chain, mode
                )

            # Reflect on the thinking process
            reflection = await self._reflect_on_thinking(thinking_chain, mode)

            # Extract final conclusion
            conclusion = self._synthesize_conclusion(thinking_chain, reflection)

            # Calculate metrics
            thinking_time = time.time() - start_time
            self.metrics["total_thinking_time"] += thinking_time
            self.metrics["average_thinking_time"] = self.metrics[
                "total_thinking_time"
            ] / (self.metrics["reflection_count"] + 1)

            result = {
                "thinking_chain": thinking_chain,
                "reflection": reflection,
                "conclusion": conclusion,
                "confidence": conclusion.get("confidence", 0.0),
                "thinking_time": thinking_time,
                "mode": mode.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Cache high-confidence results
            if result["confidence"] > 0.8:
                self.cache.set(
                    cache_key, result, timeout=3600, strategy=CacheStrategy.SEMANTIC
                )

            # Store in knowledge graph
            await self.knowledge_client.add_thinking_pattern(
                pattern_type=mode.value,
                pattern_content=result,
                confidence=result["confidence"],
            )

            return result

        except Exception as e:
            logger.error(f"Deep thinking failed: {e}")
            raise OrchestrationError(f"Thinking process failed: {e}")

    async def decompose_task(
        self,
        task_description: str,
        requirements: List[str],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Decompose a complex task into subtasks using deep thinking

        Args:
            task_description: High-level task description
            requirements: List of requirements
            constraints: Optional constraints (budget, time, resources)

        Returns:
            Dict containing task breakdown and execution plan
        """
        try:
            # Use planning mode for task decomposition
            thinking_result = await self.think_deeply(
                prompt=f"""
                Task: {task_description}
                
                Requirements:
                {json.dumps(requirements, indent=2)}
                
                Constraints:
                {json.dumps(constraints or {}, indent=2)}
                
                Please decompose this task into:
                1. Atomic subtasks that can be assigned to specific agents
                2. Dependencies between subtasks
                3. Required agent capabilities for each subtask
                4. Estimated complexity and duration
                5. Success criteria for each subtask
                """,
                mode=ThinkingMode.PLANNING,
                context={
                    "available_agents": list(self.registered_agents.keys()),
                    "agent_capabilities": {
                        cap.value: agents
                        for cap, agents in self.agent_capabilities.items()
                    },
                },
            )

            # Parse decomposition from thinking result
            decomposition = self._parse_task_decomposition(
                thinking_result["conclusion"]
            )

            # Validate decomposition
            validation_result = await self._validate_decomposition(decomposition)
            if not validation_result["valid"]:
                raise TaskDecompositionError(
                    f"Invalid decomposition: {validation_result['errors']}"
                )

            # Create workflow from decomposition
            workflow_id = await self.coordination_client.create_workflow(
                name=f"Task: {task_description[:50]}...",
                description=task_description,
                definition=decomposition,
                priority=constraints.get("priority", 5) if constraints else 5,
            )

            # Store workflow metadata
            self.active_workflows[workflow_id] = {
                "decomposition": decomposition,
                "thinking_result": thinking_result,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "initialized",
            }

            return {
                "workflow_id": workflow_id,
                "decomposition": decomposition,
                "estimated_duration": decomposition.get("total_duration", "unknown"),
                "required_agents": decomposition.get("required_agents", []),
                "confidence": thinking_result["confidence"],
            }

        except Exception as e:
            logger.error(f"Task decomposition failed: {e}")
            raise TaskDecompositionError(f"Failed to decompose task: {e}")

    async def coordinate_agents(
        self, workflow_id: str, execution_mode: str = "parallel"
    ) -> Dict[str, Any]:
        """
        Coordinate multiple agents to execute a workflow

        Args:
            workflow_id: The workflow to execute
            execution_mode: "parallel" or "sequential"

        Returns:
            Dict containing execution results
        """
        try:
            workflow = self.active_workflows.get(workflow_id)
            if not workflow:
                raise OrchestrationError(f"Workflow {workflow_id} not found")

            decomposition = workflow["decomposition"]
            subtasks = decomposition["subtasks"]

            # Use coordination thinking mode
            coordination_plan = await self.think_deeply(
                prompt=f"""
                Coordinate execution of {len(subtasks)} subtasks:
                {json.dumps(subtasks, indent=2)}
                
                Execution mode: {execution_mode}
                Available agents: {json.dumps(list(self.registered_agents.keys()))}
                
                Create an optimal coordination plan considering:
                1. Agent capabilities and current workload
                2. Task dependencies
                3. Resource constraints
                4. Parallel execution opportunities
                5. Error handling strategies
                """,
                mode=ThinkingMode.COORDINATION,
            )

            # Execute coordination plan
            execution_results = await self._execute_coordination_plan(
                workflow_id, coordination_plan["conclusion"], execution_mode
            )

            # Monitor and adapt during execution
            monitoring_task = asyncio.create_task(
                self._monitor_workflow_execution(workflow_id)
            )

            # Wait for completion or timeout
            try:
                await asyncio.wait_for(
                    monitoring_task, timeout=decomposition.get("max_duration", 3600)
                )
            except asyncio.TimeoutError:
                logger.warning(f"Workflow {workflow_id} execution timeout")
                await self._handle_workflow_timeout(workflow_id)

            # Reflect on execution
            execution_reflection = await self._reflect_on_execution(
                workflow_id, execution_results
            )

            return {
                "workflow_id": workflow_id,
                "status": execution_results["status"],
                "results": execution_results["results"],
                "reflection": execution_reflection,
                "metrics": execution_results["metrics"],
            }

        except Exception as e:
            logger.error(f"Agent coordination failed: {e}")
            raise CoordinationError(f"Failed to coordinate agents: {e}")

    async def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[AgentCapability],
        configuration: Dict[str, Any],
    ):
        """Register an agent with the orchestrator"""
        try:
            # Store agent information
            self.registered_agents[agent_id] = {
                "type": agent_type,
                "capabilities": [cap.value for cap in capabilities],
                "configuration": configuration,
                "status": "online",
                "registered_at": datetime.now(timezone.utc).isoformat(),
            }

            # Update capability mapping
            for capability in capabilities:
                if capability not in self.agent_capabilities:
                    self.agent_capabilities[capability] = []
                self.agent_capabilities[capability].append(agent_id)

            # Add to knowledge graph
            await self.knowledge_client.add_agent_entity(
                agent_id=agent_id,
                agent_type=agent_type,
                capabilities=[cap.value for cap in capabilities],
            )

            logger.info(
                f"Registered agent {agent_id} with capabilities: {capabilities}"
            )

        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            raise AgentError(f"Agent registration failed: {e}")

    async def _prepare_thinking_prompt(
        self, prompt: str, mode: ThinkingMode, context: Optional[Dict[str, Any]]
    ) -> str:
        """Prepare a thinking prompt based on mode"""
        base_prompt = f"""
        You are Starri, an advanced AI orchestrator using deep thinking capabilities.
        Current thinking mode: {mode.value}
        
        Context:
        {json.dumps(context or {}, indent=2)}
        
        Task: {prompt}
        
        """

        mode_specific_instructions = {
            ThinkingMode.DEEP_ANALYSIS: """
            Perform deep analysis:
            1. Break down the problem into fundamental components
            2. Identify hidden assumptions and implications
            3. Consider multiple perspectives and edge cases
            4. Synthesize insights into actionable conclusions
            5. Provide confidence level (0-1) for your analysis
            """,
            ThinkingMode.QUICK_DECISION: """
            Make a quick but informed decision:
            1. Identify the most critical factors
            2. Apply heuristics and patterns from past experience
            3. Provide a clear recommendation with rationale
            4. Note any risks or uncertainties
            """,
            ThinkingMode.REFLECTION: """
            Reflect on the process or outcome:
            1. Evaluate what worked well and what didn't
            2. Identify lessons learned and patterns
            3. Suggest improvements for future iterations
            4. Assess overall effectiveness
            """,
            ThinkingMode.PLANNING: """
            Create a comprehensive plan:
            1. Define clear objectives and success criteria
            2. Break down into atomic, assignable tasks
            3. Identify dependencies and critical path
            4. Estimate resources and timelines
            5. Include risk mitigation strategies
            """,
            ThinkingMode.COORDINATION: """
            Design optimal coordination strategy:
            1. Match tasks to agent capabilities
            2. Balance workload across agents
            3. Minimize dependencies and bottlenecks
            4. Plan for parallel execution where possible
            5. Include fallback strategies
            """,
        }

        return base_prompt + mode_specific_instructions.get(
            mode, "Think step by step about this problem."
        )

    async def _reflect_on_thinking(
        self, thinking_chain: List[Dict[str, Any]], mode: ThinkingMode
    ) -> Dict[str, Any]:
        """Reflect on the thinking process"""
        self.metrics["reflection_count"] += 1

        reflection_prompt = f"""
        Reflect on this thinking process:
        
        Mode: {mode.value}
        Steps taken: {len(thinking_chain)}
        
        Thinking chain summary:
        {json.dumps([{
            "step": t["step"],
            "confidence": t.get("confidence", 0),
            "key_insight": t["thought"][:200] + "..."
        } for t in thinking_chain], indent=2)}
        
        Evaluate:
        1. Was the thinking process efficient?
        2. Were all important aspects considered?
        3. What could be improved?
        4. What patterns emerged?
        5. Overall quality score (0-1)?
        """

        reflection_result = await self.gemini_client.generate_content(
            prompt=reflection_prompt,
            model_type="flash",
            generation_config={"temperature": 0.3, "max_output_tokens": 1024},
        )

        return {
            "reflection": reflection_result["content"],
            "quality_score": self._extract_quality_score(reflection_result["content"]),
            "improvements": self._extract_improvements(reflection_result["content"]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _extract_confidence(self, thought_content: str) -> float:
        """Extract confidence level from thinking content"""
        # Look for confidence indicators in the text
        import re

        # Try to find explicit confidence mentions
        confidence_match = re.search(
            r"confidence[:\s]+(\d+\.?\d*)%?", thought_content.lower()
        )
        if confidence_match:
            conf_value = float(confidence_match.group(1))
            return conf_value / 100 if conf_value > 1 else conf_value

        # Use heuristics based on certainty words
        high_confidence_words = [
            "certain",
            "definitely",
            "clearly",
            "obviously",
            "undoubtedly",
        ]
        medium_confidence_words = ["likely", "probably", "seems", "appears", "suggests"]
        low_confidence_words = ["uncertain", "unclear", "possibly", "might", "perhaps"]

        thought_lower = thought_content.lower()

        high_count = sum(1 for word in high_confidence_words if word in thought_lower)
        medium_count = sum(
            1 for word in medium_confidence_words if word in thought_lower
        )
        low_count = sum(1 for word in low_confidence_words if word in thought_lower)

        if high_count > low_count + medium_count:
            return 0.9
        elif low_count > high_count + medium_count:
            return 0.3
        elif medium_count > 0:
            return 0.6
        else:
            return 0.5  # Default confidence

    def _check_needs_more_thinking(self, thought_content: str) -> bool:
        """Check if more thinking is needed"""
        # Indicators that more thinking is needed
        continuation_phrases = [
            "need to consider",
            "requires further",
            "additionally",
            "however",
            "on the other hand",
            "but",
            "questions remain",
            "unclear",
            "investigate further",
        ]

        # Indicators that thinking is complete
        completion_phrases = [
            "in conclusion",
            "therefore",
            "final recommendation",
            "the solution is",
            "to summarize",
            "in summary",
            "the answer is",
        ]

        thought_lower = thought_content.lower()

        has_continuation = any(
            phrase in thought_lower for phrase in continuation_phrases
        )
        has_completion = any(phrase in thought_lower for phrase in completion_phrases)

        return has_continuation and not has_completion

    def _synthesize_conclusion(
        self, thinking_chain: List[Dict[str, Any]], reflection: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize final conclusion from thinking chain"""
        # Get the last high-confidence thought
        high_confidence_thoughts = [
            t for t in thinking_chain if t.get("confidence", 0) > 0.7
        ]

        if high_confidence_thoughts:
            final_thought = high_confidence_thoughts[-1]
        else:
            final_thought = thinking_chain[-1] if thinking_chain else {}

        return {
            "conclusion": final_thought.get("thought", "No clear conclusion reached"),
            "confidence": final_thought.get("confidence", 0.5),
            "based_on_steps": len(thinking_chain),
            "quality_score": reflection.get("quality_score", 0.5),
            "key_insights": self._extract_key_insights(thinking_chain),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _extract_key_insights(self, thinking_chain: List[Dict[str, Any]]) -> List[str]:
        """Extract key insights from thinking chain"""
        insights = []

        for thought in thinking_chain:
            content = thought.get("thought", "")
            # Simple heuristic: look for numbered points or bullet points
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if (
                    line.startswith(("1.", "2.", "3.", "•", "-", "*"))
                    and len(line) > 10
                    and thought.get("confidence", 0) > 0.6
                ):
                    insights.append(line)

        return insights[:10]  # Limit to top 10 insights

    def _extract_quality_score(self, reflection_content: str) -> float:
        """Extract quality score from reflection"""
        import re

        score_match = re.search(
            r"quality score[:\s]+(\d+\.?\d*)", reflection_content.lower()
        )
        if score_match:
            return float(score_match.group(1))

        # Fallback: analyze sentiment of reflection
        positive_words = [
            "excellent",
            "effective",
            "comprehensive",
            "thorough",
            "insightful",
        ]
        negative_words = ["poor", "incomplete", "shallow", "inadequate", "insufficient"]

        reflection_lower = reflection_content.lower()
        positive_count = sum(1 for word in positive_words if word in reflection_lower)
        negative_count = sum(1 for word in negative_words if word in reflection_lower)

        if positive_count > negative_count:
            return 0.8
        elif negative_count > positive_count:
            return 0.4
        else:
            return 0.6

    def _extract_improvements(self, reflection_content: str) -> List[str]:
        """Extract improvement suggestions from reflection"""
        improvements = []

        # Look for improvement indicators
        lines = reflection_content.split("\n")
        capture_next = False

        for line in lines:
            line = line.strip()
            if any(
                indicator in line.lower()
                for indicator in [
                    "could be improved",
                    "improvement",
                    "should",
                    "recommendation",
                    "suggest",
                ]
            ):
                capture_next = True
            elif capture_next and line and not line.endswith(":"):
                improvements.append(line)
                if len(improvements) >= 5:  # Limit to 5 improvements
                    break

        return improvements

    def _parse_task_decomposition(self, conclusion: Dict[str, Any]) -> Dict[str, Any]:
        """Parse task decomposition from conclusion"""
        # This would typically use more sophisticated parsing
        # For now, we'll create a structured decomposition
        content = conclusion.get("conclusion", "")

        # Extract subtasks (simplified parsing)
        subtasks = []
        lines = content.split("\n")
        current_subtask = None

        for line in lines:
            line = line.strip()
            # Look for subtask indicators
            if line.startswith(("Subtask", "Task", "1.", "2.", "3.")):
                if current_subtask:
                    subtasks.append(current_subtask)
                current_subtask = {
                    "id": f"task_{len(subtasks) + 1}",
                    "description": line,
                    "dependencies": [],
                    "required_capabilities": [],
                    "estimated_duration": "30m",
                    "complexity": "medium",
                }
            elif current_subtask and line:
                # Add details to current subtask
                if "depends on" in line.lower():
                    current_subtask["dependencies"].append(line)
                elif "requires" in line.lower():
                    current_subtask["required_capabilities"].append(line)

        if current_subtask:
            subtasks.append(current_subtask)

        return {
            "subtasks": subtasks,
            "total_duration": f"{len(subtasks) * 30}m",
            "required_agents": list(
                set(
                    cap
                    for task in subtasks
                    for cap in task.get("required_capabilities", [])
                )
            ),
            "execution_strategy": "parallel" if len(subtasks) > 3 else "sequential",
        }

    async def _validate_decomposition(
        self, decomposition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate task decomposition"""
        errors = []
        warnings = []

        # Check for required fields
        if not decomposition.get("subtasks"):
            errors.append("No subtasks defined")

        # Validate each subtask
        for i, subtask in enumerate(decomposition.get("subtasks", [])):
            if not subtask.get("description"):
                errors.append(f"Subtask {i+1} missing description")

            # Check if required capabilities are available
            for capability in subtask.get("required_capabilities", []):
                if not any(
                    capability in str(agents)
                    for agents in self.agent_capabilities.values()
                ):
                    warnings.append(f"No agent available for capability: {capability}")

        # Check for circular dependencies
        # (Simplified check - in production, use topological sort)

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    async def _execute_coordination_plan(
        self, workflow_id: str, plan: Dict[str, Any], execution_mode: str
    ) -> Dict[str, Any]:
        """Execute the coordination plan"""
        results = {
            "status": "started",
            "results": {},
            "metrics": {
                "start_time": time.time(),
                "tasks_completed": 0,
                "tasks_failed": 0,
            },
        }

        try:
            # Update workflow status
            await self.coordination_client.update_workflow_status(
                workflow_id, "running"
            )

            # Get subtasks from plan
            subtasks = self.active_workflows[workflow_id]["decomposition"]["subtasks"]

            if execution_mode == "parallel":
                # Execute tasks in parallel
                task_futures = []
                for subtask in subtasks:
                    task_future = asyncio.create_task(
                        self._execute_single_task(workflow_id, subtask)
                    )
                    task_futures.append((subtask["id"], task_future))

                # Wait for all tasks
                for task_id, future in task_futures:
                    try:
                        result = await future
                        results["results"][task_id] = result
                        results["metrics"]["tasks_completed"] += 1
                    except Exception as e:
                        results["results"][task_id] = {"error": str(e)}
                        results["metrics"]["tasks_failed"] += 1

            else:  # sequential
                # Execute tasks sequentially
                for subtask in subtasks:
                    try:
                        result = await self._execute_single_task(workflow_id, subtask)
                        results["results"][subtask["id"]] = result
                        results["metrics"]["tasks_completed"] += 1
                    except Exception as e:
                        results["results"][subtask["id"]] = {"error": str(e)}
                        results["metrics"]["tasks_failed"] += 1
                        # Stop on failure in sequential mode
                        break

            # Calculate final status
            if results["metrics"]["tasks_failed"] == 0:
                results["status"] = "completed"
            elif results["metrics"]["tasks_completed"] > 0:
                results["status"] = "partial_success"
            else:
                results["status"] = "failed"

            # Update workflow status
            await self.coordination_client.update_workflow_status(
                workflow_id, results["status"], metrics=results["metrics"]
            )

            results["metrics"]["total_time"] = (
                time.time() - results["metrics"]["start_time"]
            )

        except Exception as e:
            logger.error(f"Coordination plan execution failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)

        return results

    async def _execute_single_task(
        self, workflow_id: str, subtask: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single subtask"""
        # Find suitable agent
        required_capabilities = subtask.get("required_capabilities", [])
        suitable_agents = self._find_suitable_agents(required_capabilities)

        if not suitable_agents:
            raise AgentError(f"No suitable agent found for task: {subtask['id']}")

        # Select best agent (simplified - could use load balancing)
        selected_agent = suitable_agents[0]

        # Create task in coordination system
        task_id = await self.coordination_client.create_task(
            workflow_id=workflow_id,
            name=subtask["description"],
            task_type="ai_task",
            input_data=subtask,
            agent_id=selected_agent,
        )

        # Assign task to agent
        assigned = await self.coordination_client.assign_task(task_id, selected_agent)
        if not assigned:
            raise AgentError(
                f"Failed to assign task {task_id} to agent {selected_agent}"
            )

        # Wait for task completion (simplified - in production, use proper event system)
        max_wait = 300  # 5 minutes
        start_time = time.time()

        while time.time() - start_time < max_wait:
            task_status = (
                await self.coordination_client.client.table("tasks")
                .select("*")
                .eq("task_id", task_id)
                .execute()
            )

            if task_status.data and task_status.data[0]["status"] in [
                "completed",
                "failed",
            ]:
                return task_status.data[0]

            await asyncio.sleep(5)  # Check every 5 seconds

        raise TimeoutError(f"Task {task_id} execution timeout")

    def _find_suitable_agents(self, required_capabilities: List[str]) -> List[str]:
        """Find agents that match required capabilities"""
        suitable_agents = []

        for agent_id, agent_info in self.registered_agents.items():
            agent_capabilities = agent_info.get("capabilities", [])
            # Check if agent has any of the required capabilities
            if any(
                req_cap in str(agent_capabilities) for req_cap in required_capabilities
            ):
                if agent_info.get("status") == "online":
                    suitable_agents.append(agent_id)

        return suitable_agents

    async def _monitor_workflow_execution(self, workflow_id: str):
        """Monitor workflow execution and adapt as needed"""
        while workflow_id in self.active_workflows:
            try:
                # Check workflow status
                workflow_status = (
                    await self.coordination_client.client.table("workflows")
                    .select("*")
                    .eq("workflow_id", workflow_id)
                    .execute()
                )

                if workflow_status.data:
                    status = workflow_status.data[0]["status"]
                    if status in ["completed", "failed", "cancelled"]:
                        break

                # Check for stuck tasks
                tasks = (
                    await self.coordination_client.client.table("tasks")
                    .select("*")
                    .eq("workflow_id", workflow_id)
                    .execute()
                )

                for task in tasks.data:
                    if task["status"] == "running":
                        # Check if task is stuck (simplified)
                        started_at = datetime.fromisoformat(task.get("started_at", ""))
                        if (
                            datetime.now(timezone.utc) - started_at
                        ).seconds > 600:  # 10 minutes
                            logger.warning(f"Task {task['task_id']} appears stuck")
                            # Could implement recovery strategies here

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error monitoring workflow {workflow_id}: {e}")
                await asyncio.sleep(30)

    async def _handle_workflow_timeout(self, workflow_id: str):
        """Handle workflow timeout"""
        logger.warning(f"Handling timeout for workflow {workflow_id}")

        # Update workflow status
        await self.coordination_client.update_workflow_status(
            workflow_id, "timeout", error_message="Workflow execution timeout"
        )

        # Cancel any running tasks
        tasks = (
            await self.coordination_client.client.table("tasks")
            .select("*")
            .eq("workflow_id", workflow_id)
            .eq("status", "running")
            .execute()
        )

        for task in tasks.data:
            await self.coordination_client.update_task_status(
                task["task_id"],
                "cancelled",
                error_details={"reason": "workflow_timeout"},
            )

    async def _reflect_on_execution(
        self, workflow_id: str, execution_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Reflect on workflow execution"""
        reflection_prompt = f"""
        Reflect on this workflow execution:
        
        Workflow ID: {workflow_id}
        Status: {execution_results['status']}
        Tasks completed: {execution_results['metrics']['tasks_completed']}
        Tasks failed: {execution_results['metrics']['tasks_failed']}
        Total time: {execution_results['metrics'].get('total_time', 0):.2f} seconds
        
        Results summary:
        {json.dumps(execution_results['results'], indent=2)}
        
        Analyze:
        1. What went well?
        2. What could be improved?
        3. Were there any unexpected issues?
        4. How efficient was the execution?
        5. Recommendations for similar workflows?
        """

        reflection_result = await self.think_deeply(
            prompt=reflection_prompt,
            mode=ThinkingMode.REFLECTION,
            context={
                "workflow_id": workflow_id,
                "execution_results": execution_results,
            },
        )

        # Store reflection in knowledge graph
        await self.knowledge_client.add_execution_pattern(
            workflow_type="task_execution",
            pattern_content=reflection_result,
            success_rate=(
                execution_results["metrics"]["tasks_completed"]
                / (
                    execution_results["metrics"]["tasks_completed"]
                    + execution_results["metrics"]["tasks_failed"]
                )
                if execution_results["metrics"]["tasks_completed"]
                + execution_results["metrics"]["tasks_failed"]
                > 0
                else 0
            ),
        )

        return reflection_result

    async def _monitor_agents(self):
        """Background task to monitor agent health"""
        while True:
            try:
                for agent_id in list(self.registered_agents.keys()):
                    # Check agent heartbeat
                    agent_state = await self.coordination_client.get_agent_state(
                        agent_id
                    )
                    if agent_state:
                        last_heartbeat = datetime.fromisoformat(
                            agent_state.get("last_heartbeat", "")
                        )
                        if (
                            datetime.now(timezone.utc) - last_heartbeat
                        ).seconds > 300:  # 5 minutes
                            logger.warning(f"Agent {agent_id} appears offline")
                            self.registered_agents[agent_id]["status"] = "offline"

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error monitoring agents: {e}")
                await asyncio.sleep(60)

    async def _process_task_queue(self):
        """Background task to process queued tasks"""
        while True:
            try:
                if not self.task_queue.empty():
                    task = await self.task_queue.get()
                    # Process task
                    logger.info(f"Processing queued task: {task}")
                    # Implementation depends on task type

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error processing task queue: {e}")
                await asyncio.sleep(5)

    async def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get current orchestrator status"""
        return {
            "orchestrator_id": self.orchestrator_id,
            "status": "operational",
            "registered_agents": len(self.registered_agents),
            "active_agents": sum(
                1
                for agent in self.registered_agents.values()
                if agent["status"] == "online"
            ),
            "active_workflows": len(self.active_workflows),
            "metrics": self.metrics,
            "thinking_chains_stored": len(self.thinking_chains),
            "cache_stats": self.cache.get_cache_stats(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        logger.info("Shutting down Starri Orchestrator")

        # Cancel active workflows
        for workflow_id in list(self.active_workflows.keys()):
            await self.coordination_client.update_workflow_status(
                workflow_id, "cancelled", error_message="Orchestrator shutdown"
            )

        # Disconnect from services
        await self.coordination_client.disconnect()

        # Save metrics and patterns
        await self.knowledge_client.save_orchestrator_state(
            self.orchestrator_id, self.metrics, self.thinking_chains
        )

        logger.info("Starri Orchestrator shutdown complete")
