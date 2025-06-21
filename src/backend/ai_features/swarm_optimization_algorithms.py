"""
Bio-Inspired Swarm Optimization Algorithms
Implementation of PSO, ACO, BCO, Firefly and other nature-inspired algorithms
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Set, Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import networkx as nx
from scipy.spatial.distance import euclidean

from nexus_forge.core.monitoring import get_logger

logger = get_logger(__name__)


# Base Classes
@dataclass
class OptimizationProblem:
    """Definition of an optimization problem"""
    id: str = field(default_factory=lambda: f"opt_{uuid4().hex[:8]}")
    name: str = ""
    objective_function: Optional[Callable] = None
    constraints: List[Callable] = field(default_factory=list)
    bounds: List[Tuple[float, float]] = field(default_factory=list)
    dimension: int = 0
    minimize: bool = True  # True for minimization, False for maximization
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result of optimization"""
    problem_id: str = ""
    algorithm: str = ""
    best_solution: np.ndarray = field(default_factory=lambda: np.array([]))
    best_fitness: float = float('inf')
    convergence_history: List[float] = field(default_factory=list)
    iterations: int = 0
    execution_time: float = 0.0
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


# Particle Swarm Optimization (PSO)
@dataclass
class Particle:
    """Individual particle in PSO"""
    id: str = field(default_factory=lambda: f"particle_{uuid4().hex[:8]}")
    position: np.ndarray = field(default_factory=lambda: np.random.rand(10))
    velocity: np.ndarray = field(default_factory=lambda: np.random.randn(10) * 0.1)
    best_position: np.ndarray = field(default_factory=lambda: np.random.rand(10))
    best_fitness: float = float('inf')
    fitness: float = float('inf')


class ParticleSwarmOptimization:
    """
    Particle Swarm Optimization Algorithm
    Simulates social behavior of bird flocking or fish schooling
    """
    
    def __init__(self, swarm_size: int = 50):
        self.swarm_size = swarm_size
        self.w = 0.7298  # Inertia weight
        self.c1 = 1.49618  # Cognitive parameter
        self.c2 = 1.49618  # Social parameter
        self.particles: List[Particle] = []
        self.global_best_position: Optional[np.ndarray] = None
        self.global_best_fitness: float = float('inf')
        
    async def optimize(
        self,
        problem: OptimizationProblem,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = False
    ) -> OptimizationResult:
        """Run PSO optimization"""
        start_time = time.time()
        result = OptimizationResult(
            problem_id=problem.id,
            algorithm="PSO"
        )
        
        try:
            # Initialize swarm
            await self._initialize_swarm(problem)
            
            # Optimization loop
            for iteration in range(max_iterations):
                # Update particles
                improved = await self._update_particles(problem)
                
                # Record convergence
                result.convergence_history.append(self.global_best_fitness)
                
                if verbose and iteration % 10 == 0:
                    logger.info(
                        f"PSO Iteration {iteration}: "
                        f"Best fitness = {self.global_best_fitness:.6f}"
                    )
                
                # Check convergence
                if len(result.convergence_history) > 10:
                    recent_improvement = abs(
                        result.convergence_history[-1] - 
                        result.convergence_history[-10]
                    )
                    if recent_improvement < tolerance:
                        logger.info(f"PSO converged after {iteration} iterations")
                        break
                
                # Adaptive parameters
                self._adapt_parameters(iteration, max_iterations)
            
            # Set final result
            result.best_solution = self.global_best_position.copy()
            result.best_fitness = self.global_best_fitness
            result.iterations = len(result.convergence_history)
            result.execution_time = time.time() - start_time
            result.success = True
            
            # Add metadata
            result.metadata = {
                "final_inertia": self.w,
                "swarm_diversity": self._calculate_swarm_diversity(),
                "convergence_rate": self._calculate_convergence_rate(result.convergence_history)
            }
            
        except Exception as e:
            logger.error(f"PSO optimization failed: {e}")
            result.success = False
            result.metadata["error"] = str(e)
            
        return result
    
    async def _initialize_swarm(self, problem: OptimizationProblem):
        """Initialize particle swarm"""
        self.particles = []
        self.global_best_fitness = float('inf') if problem.minimize else float('-inf')
        
        for _ in range(self.swarm_size):
            # Random position within bounds
            position = np.array([
                np.random.uniform(low, high)
                for low, high in problem.bounds
            ])
            
            # Random velocity
            velocity = np.array([
                np.random.uniform(-(high-low)*0.1, (high-low)*0.1)
                for low, high in problem.bounds
            ])
            
            particle = Particle(
                position=position,
                velocity=velocity,
                best_position=position.copy()
            )
            
            # Evaluate fitness
            particle.fitness = await self._evaluate_fitness(particle.position, problem)
            particle.best_fitness = particle.fitness
            
            # Update global best
            if self._is_better(particle.fitness, self.global_best_fitness, problem.minimize):
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()
            
            self.particles.append(particle)
    
    async def _update_particles(self, problem: OptimizationProblem) -> int:
        """Update particle positions and velocities"""
        improved = 0
        
        for particle in self.particles:
            # Update velocity
            r1 = np.random.rand(len(particle.position))
            r2 = np.random.rand(len(particle.position))
            
            cognitive = self.c1 * r1 * (particle.best_position - particle.position)
            social = self.c2 * r2 * (self.global_best_position - particle.position)
            
            particle.velocity = self.w * particle.velocity + cognitive + social
            
            # Apply velocity limits
            max_velocity = [(high-low)*0.2 for low, high in problem.bounds]
            particle.velocity = np.clip(particle.velocity, -max_velocity, max_velocity)
            
            # Update position
            particle.position = particle.position + particle.velocity
            
            # Apply position bounds
            for i, (low, high) in enumerate(problem.bounds):
                if particle.position[i] < low:
                    particle.position[i] = low
                    particle.velocity[i] = -particle.velocity[i] * 0.5
                elif particle.position[i] > high:
                    particle.position[i] = high
                    particle.velocity[i] = -particle.velocity[i] * 0.5
            
            # Evaluate new position
            particle.fitness = await self._evaluate_fitness(particle.position, problem)
            
            # Update personal best
            if self._is_better(particle.fitness, particle.best_fitness, problem.minimize):
                particle.best_fitness = particle.fitness
                particle.best_position = particle.position.copy()
                improved += 1
                
                # Update global best
                if self._is_better(particle.fitness, self.global_best_fitness, problem.minimize):
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()
        
        return improved
    
    async def _evaluate_fitness(
        self, 
        position: np.ndarray, 
        problem: OptimizationProblem
    ) -> float:
        """Evaluate fitness of a position"""
        if problem.objective_function:
            fitness = problem.objective_function(position)
            
            # Apply constraints as penalties
            for constraint in problem.constraints:
                violation = constraint(position)
                if violation > 0:
                    fitness += violation * 1000  # Penalty factor
            
            return fitness
        return 0.0
    
    def _is_better(self, fitness1: float, fitness2: float, minimize: bool) -> bool:
        """Check if fitness1 is better than fitness2"""
        return fitness1 < fitness2 if minimize else fitness1 > fitness2
    
    def _adapt_parameters(self, iteration: int, max_iterations: int):
        """Adapt PSO parameters during optimization"""
        # Linear decrease of inertia weight
        self.w = 0.9 - (0.9 - 0.4) * iteration / max_iterations
        
        # Time-varying acceleration coefficients
        self.c1 = 2.5 - 2.0 * iteration / max_iterations
        self.c2 = 0.5 + 2.0 * iteration / max_iterations
    
    def _calculate_swarm_diversity(self) -> float:
        """Calculate diversity of swarm positions"""
        if not self.particles:
            return 0.0
            
        positions = np.array([p.position for p in self.particles])
        centroid = np.mean(positions, axis=0)
        
        diversity = np.mean([
            euclidean(pos, centroid) for pos in positions
        ])
        
        return diversity
    
    def _calculate_convergence_rate(self, history: List[float]) -> float:
        """Calculate convergence rate"""
        if len(history) < 2:
            return 0.0
            
        improvements = []
        for i in range(1, len(history)):
            if history[i-1] != 0:
                improvement = abs(history[i] - history[i-1]) / abs(history[i-1])
                improvements.append(improvement)
        
        return np.mean(improvements) if improvements else 0.0


# Ant Colony Optimization (ACO)
@dataclass
class Ant:
    """Individual ant in ACO"""
    id: str = field(default_factory=lambda: f"ant_{uuid4().hex[:8]}")
    current_node: int = 0
    path: List[int] = field(default_factory=list)
    path_cost: float = 0.0
    tabu_list: Set[int] = field(default_factory=set)


class AntColonyOptimization:
    """
    Ant Colony Optimization Algorithm
    Simulates foraging behavior of ants to find optimal paths
    """
    
    def __init__(self, ant_count: int = 50):
        self.ant_count = ant_count
        self.alpha = 1.0  # Pheromone importance
        self.beta = 2.0   # Heuristic importance
        self.rho = 0.1    # Evaporation rate
        self.Q = 100      # Pheromone deposit factor
        self.pheromone_matrix: Optional[np.ndarray] = None
        self.heuristic_matrix: Optional[np.ndarray] = None
        self.best_path: List[int] = []
        self.best_cost: float = float('inf')
        
    async def optimize_tsp(
        self,
        distance_matrix: np.ndarray,
        max_iterations: int = 100,
        verbose: bool = False
    ) -> OptimizationResult:
        """Solve Traveling Salesman Problem using ACO"""
        start_time = time.time()
        n_cities = len(distance_matrix)
        
        result = OptimizationResult(
            algorithm="ACO-TSP"
        )
        
        try:
            # Initialize pheromone and heuristic matrices
            self.pheromone_matrix = np.ones((n_cities, n_cities)) * 0.1
            self.heuristic_matrix = 1.0 / (distance_matrix + 1e-10)
            np.fill_diagonal(self.heuristic_matrix, 0)
            
            # Main optimization loop
            for iteration in range(max_iterations):
                # Create ants
                ants = [Ant(current_node=random.randint(0, n_cities-1)) 
                       for _ in range(self.ant_count)]
                
                # Construct solutions
                for ant in ants:
                    await self._construct_solution(ant, n_cities)
                    ant.path_cost = self._calculate_path_cost(ant.path, distance_matrix)
                    
                    # Update best solution
                    if ant.path_cost < self.best_cost:
                        self.best_cost = ant.path_cost
                        self.best_path = ant.path.copy()
                
                # Update pheromones
                self._update_pheromones(ants)
                
                # Record convergence
                result.convergence_history.append(self.best_cost)
                
                if verbose and iteration % 10 == 0:
                    logger.info(
                        f"ACO Iteration {iteration}: "
                        f"Best cost = {self.best_cost:.2f}"
                    )
                
                # Adaptive parameters
                self._adapt_aco_parameters(iteration, max_iterations)
            
            # Set final result
            result.best_solution = np.array(self.best_path)
            result.best_fitness = self.best_cost
            result.iterations = max_iterations
            result.execution_time = time.time() - start_time
            result.success = True
            
            result.metadata = {
                "path_length": len(self.best_path),
                "pheromone_stats": self._get_pheromone_statistics()
            }
            
        except Exception as e:
            logger.error(f"ACO optimization failed: {e}")
            result.success = False
            result.metadata["error"] = str(e)
            
        return result
    
    async def optimize_path(
        self,
        graph: nx.Graph,
        start_node: Any,
        end_node: Any,
        max_iterations: int = 100
    ) -> OptimizationResult:
        """Find optimal path in a graph"""
        start_time = time.time()
        
        result = OptimizationResult(
            algorithm="ACO-PathFinding"
        )
        
        try:
            # Convert graph to adjacency matrix
            nodes = list(graph.nodes())
            n_nodes = len(nodes)
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            
            # Build distance matrix
            distance_matrix = np.full((n_nodes, n_nodes), np.inf)
            for u, v, data in graph.edges(data=True):
                i, j = node_to_idx[u], node_to_idx[v]
                weight = data.get('weight', 1.0)
                distance_matrix[i][j] = weight
                distance_matrix[j][i] = weight  # Undirected
            
            # Initialize pheromones
            self.pheromone_matrix = np.ones((n_nodes, n_nodes)) * 0.1
            self.heuristic_matrix = 1.0 / (distance_matrix + 1e-10)
            
            start_idx = node_to_idx[start_node]
            end_idx = node_to_idx[end_node]
            
            # Optimization loop
            for iteration in range(max_iterations):
                paths = []
                
                # Generate ant paths
                for _ in range(self.ant_count):
                    path = await self._find_path(start_idx, end_idx, distance_matrix)
                    if path:
                        cost = self._calculate_path_cost(path, distance_matrix)
                        paths.append((path, cost))
                
                # Update best path
                if paths:
                    best_ant_path, best_ant_cost = min(paths, key=lambda x: x[1])
                    if best_ant_cost < self.best_cost:
                        self.best_cost = best_ant_cost
                        self.best_path = best_ant_path
                
                # Update pheromones
                self._update_path_pheromones(paths)
                
                result.convergence_history.append(self.best_cost)
            
            # Convert indices back to nodes
            if self.best_path:
                result.best_solution = np.array([nodes[i] for i in self.best_path])
                result.best_fitness = self.best_cost
                result.success = True
            
            result.iterations = max_iterations
            result.execution_time = time.time() - start_time
            
        except Exception as e:
            logger.error(f"ACO path optimization failed: {e}")
            result.success = False
            result.metadata["error"] = str(e)
            
        return result
    
    async def _construct_solution(self, ant: Ant, n_cities: int):
        """Construct a complete tour for an ant"""
        ant.path = [ant.current_node]
        ant.tabu_list = {ant.current_node}
        
        while len(ant.path) < n_cities:
            next_city = self._select_next_city(ant, n_cities)
            ant.path.append(next_city)
            ant.tabu_list.add(next_city)
            ant.current_node = next_city
        
        # Return to start
        ant.path.append(ant.path[0])
    
    def _select_next_city(self, ant: Ant, n_cities: int) -> int:
        """Select next city using probabilistic rule"""
        current = ant.current_node
        
        # Calculate probabilities
        probabilities = np.zeros(n_cities)
        
        for city in range(n_cities):
            if city not in ant.tabu_list:
                pheromone = self.pheromone_matrix[current][city] ** self.alpha
                heuristic = self.heuristic_matrix[current][city] ** self.beta
                probabilities[city] = pheromone * heuristic
        
        # Normalize probabilities
        total = np.sum(probabilities)
        if total > 0:
            probabilities /= total
            
            # Roulette wheel selection
            cumsum = np.cumsum(probabilities)
            r = random.random()
            
            for i, p in enumerate(cumsum):
                if r <= p:
                    return i
        
        # If no valid city found, return random unvisited
        unvisited = list(set(range(n_cities)) - ant.tabu_list)
        return random.choice(unvisited) if unvisited else 0
    
    def _calculate_path_cost(self, path: List[int], distance_matrix: np.ndarray) -> float:
        """Calculate total cost of a path"""
        cost = 0.0
        for i in range(len(path) - 1):
            cost += distance_matrix[path[i]][path[i+1]]
        return cost
    
    def _update_pheromones(self, ants: List[Ant]):
        """Update pheromone matrix"""
        # Evaporation
        self.pheromone_matrix *= (1 - self.rho)
        
        # Add new pheromones
        for ant in ants:
            if ant.path_cost > 0:
                pheromone_amount = self.Q / ant.path_cost
                
                for i in range(len(ant.path) - 1):
                    u, v = ant.path[i], ant.path[i+1]
                    self.pheromone_matrix[u][v] += pheromone_amount
                    self.pheromone_matrix[v][u] += pheromone_amount
    
    async def _find_path(
        self, 
        start: int, 
        end: int, 
        distance_matrix: np.ndarray
    ) -> Optional[List[int]]:
        """Find path from start to end node"""
        current = start
        path = [current]
        visited = {current}
        
        while current != end and len(visited) < len(distance_matrix):
            # Get next node
            neighbors = []
            for next_node in range(len(distance_matrix)):
                if next_node not in visited and distance_matrix[current][next_node] < np.inf:
                    neighbors.append(next_node)
            
            if not neighbors:
                return None  # Dead end
            
            # Select next node probabilistically
            probabilities = []
            for neighbor in neighbors:
                pheromone = self.pheromone_matrix[current][neighbor] ** self.alpha
                heuristic = self.heuristic_matrix[current][neighbor] ** self.beta
                probabilities.append(pheromone * heuristic)
            
            # Normalize and select
            total = sum(probabilities)
            if total > 0:
                probabilities = [p/total for p in probabilities]
                current = np.random.choice(neighbors, p=probabilities)
            else:
                current = random.choice(neighbors)
            
            path.append(current)
            visited.add(current)
        
        return path if current == end else None
    
    def _update_path_pheromones(self, paths: List[Tuple[List[int], float]]):
        """Update pheromones for path finding"""
        # Evaporation
        self.pheromone_matrix *= (1 - self.rho)
        
        # Add new pheromones
        for path, cost in paths:
            if cost > 0:
                pheromone_amount = self.Q / cost
                
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    self.pheromone_matrix[u][v] += pheromone_amount
    
    def _adapt_aco_parameters(self, iteration: int, max_iterations: int):
        """Adapt ACO parameters during optimization"""
        # Increase pheromone importance over time
        self.alpha = 1.0 + 1.0 * iteration / max_iterations
        
        # Decrease evaporation rate
        self.rho = 0.1 * (1 - iteration / max_iterations)
    
    def _get_pheromone_statistics(self) -> Dict[str, float]:
        """Get pheromone matrix statistics"""
        if self.pheromone_matrix is None:
            return {}
            
        return {
            "mean": float(np.mean(self.pheromone_matrix)),
            "std": float(np.std(self.pheromone_matrix)),
            "max": float(np.max(self.pheromone_matrix)),
            "min": float(np.min(self.pheromone_matrix))
        }


# Bee Colony Optimization (BCO)
@dataclass
class Bee:
    """Individual bee in BCO"""
    id: str = field(default_factory=lambda: f"bee_{uuid4().hex[:8]}")
    type: str = "scout"  # scout, worker, onlooker
    position: np.ndarray = field(default_factory=lambda: np.random.rand(10))
    fitness: float = float('inf')
    trial_count: int = 0
    nectar_amount: float = 0.0


class BeeColonyOptimization:
    """
    Artificial Bee Colony Optimization
    Simulates intelligent foraging behavior of honey bee swarms
    """
    
    def __init__(self, colony_size: int = 50):
        self.colony_size = colony_size
        self.employed_bees = colony_size // 2
        self.onlooker_bees = colony_size // 2
        self.scout_bees = 1
        self.limit = 100  # Abandonment limit
        self.food_sources: List[Bee] = []
        self.best_position: Optional[np.ndarray] = None
        self.best_fitness: float = float('inf')
        
    async def optimize(
        self,
        problem: OptimizationProblem,
        max_iterations: int = 100,
        verbose: bool = False
    ) -> OptimizationResult:
        """Run BCO optimization"""
        start_time = time.time()
        result = OptimizationResult(
            problem_id=problem.id,
            algorithm="BCO"
        )
        
        try:
            # Initialize food sources
            await self._initialize_food_sources(problem)
            
            for iteration in range(max_iterations):
                # Employed bee phase
                await self._employed_bee_phase(problem)
                
                # Calculate probabilities for onlooker bees
                probabilities = self._calculate_probabilities()
                
                # Onlooker bee phase
                await self._onlooker_bee_phase(problem, probabilities)
                
                # Scout bee phase
                await self._scout_bee_phase(problem)
                
                # Record convergence
                result.convergence_history.append(self.best_fitness)
                
                if verbose and iteration % 10 == 0:
                    logger.info(
                        f"BCO Iteration {iteration}: "
                        f"Best fitness = {self.best_fitness:.6f}"
                    )
            
            # Set final result
            result.best_solution = self.best_position.copy()
            result.best_fitness = self.best_fitness
            result.iterations = max_iterations
            result.execution_time = time.time() - start_time
            result.success = True
            
            result.metadata = {
                "abandoned_sources": sum(1 for bee in self.food_sources if bee.trial_count >= self.limit),
                "nectar_distribution": self._get_nectar_distribution()
            }
            
        except Exception as e:
            logger.error(f"BCO optimization failed: {e}")
            result.success = False
            result.metadata["error"] = str(e)
            
        return result
    
    async def _initialize_food_sources(self, problem: OptimizationProblem):
        """Initialize food source positions"""
        self.food_sources = []
        self.best_fitness = float('inf') if problem.minimize else float('-inf')
        
        for i in range(self.employed_bees):
            # Random position within bounds
            position = np.array([
                np.random.uniform(low, high)
                for low, high in problem.bounds
            ])
            
            bee = Bee(
                type="employed",
                position=position
            )
            
            # Evaluate fitness
            bee.fitness = await self._evaluate_fitness(position, problem)
            bee.nectar_amount = 1.0 / (1.0 + bee.fitness) if bee.fitness >= 0 else 1.0 + abs(bee.fitness)
            
            # Update global best
            if self._is_better(bee.fitness, self.best_fitness, problem.minimize):
                self.best_fitness = bee.fitness
                self.best_position = bee.position.copy()
            
            self.food_sources.append(bee)
    
    async def _employed_bee_phase(self, problem: OptimizationProblem):
        """Employed bees search for better food sources"""
        for bee in self.food_sources:
            # Generate new solution
            new_position = await self._generate_neighbor_solution(bee, problem)
            
            # Evaluate new solution
            new_fitness = await self._evaluate_fitness(new_position, problem)
            
            # Greedy selection
            if self._is_better(new_fitness, bee.fitness, problem.minimize):
                bee.position = new_position
                bee.fitness = new_fitness
                bee.nectar_amount = 1.0 / (1.0 + new_fitness) if new_fitness >= 0 else 1.0 + abs(new_fitness)
                bee.trial_count = 0
                
                # Update global best
                if self._is_better(new_fitness, self.best_fitness, problem.minimize):
                    self.best_fitness = new_fitness
                    self.best_position = new_position.copy()
            else:
                bee.trial_count += 1
    
    def _calculate_probabilities(self) -> np.ndarray:
        """Calculate selection probabilities for onlooker bees"""
        nectar_amounts = np.array([bee.nectar_amount for bee in self.food_sources])
        total_nectar = np.sum(nectar_amounts)
        
        if total_nectar > 0:
            return nectar_amounts / total_nectar
        else:
            return np.ones(len(self.food_sources)) / len(self.food_sources)
    
    async def _onlooker_bee_phase(
        self, 
        problem: OptimizationProblem, 
        probabilities: np.ndarray
    ):
        """Onlooker bees select food sources based on nectar amount"""
        for _ in range(self.onlooker_bees):
            # Select food source using roulette wheel
            selected_idx = np.random.choice(len(self.food_sources), p=probabilities)
            selected_bee = self.food_sources[selected_idx]
            
            # Generate new solution
            new_position = await self._generate_neighbor_solution(selected_bee, problem)
            
            # Evaluate new solution
            new_fitness = await self._evaluate_fitness(new_position, problem)
            
            # Greedy selection
            if self._is_better(new_fitness, selected_bee.fitness, problem.minimize):
                selected_bee.position = new_position
                selected_bee.fitness = new_fitness
                selected_bee.nectar_amount = 1.0 / (1.0 + new_fitness) if new_fitness >= 0 else 1.0 + abs(new_fitness)
                selected_bee.trial_count = 0
                
                # Update global best
                if self._is_better(new_fitness, self.best_fitness, problem.minimize):
                    self.best_fitness = new_fitness
                    self.best_position = new_position.copy()
            else:
                selected_bee.trial_count += 1
    
    async def _scout_bee_phase(self, problem: OptimizationProblem):
        """Scout bees search for new food sources"""
        for bee in self.food_sources:
            if bee.trial_count >= self.limit:
                # Abandon food source and search for new one
                bee.position = np.array([
                    np.random.uniform(low, high)
                    for low, high in problem.bounds
                ])
                
                bee.fitness = await self._evaluate_fitness(bee.position, problem)
                bee.nectar_amount = 1.0 / (1.0 + bee.fitness) if bee.fitness >= 0 else 1.0 + abs(bee.fitness)
                bee.trial_count = 0
                bee.type = "scout"
                
                # Update global best
                if self._is_better(bee.fitness, self.best_fitness, problem.minimize):
                    self.best_fitness = bee.fitness
                    self.best_position = bee.position.copy()
    
    async def _generate_neighbor_solution(
        self, 
        bee: Bee, 
        problem: OptimizationProblem
    ) -> np.ndarray:
        """Generate neighbor solution"""
        # Select random partner
        partners = [b for b in self.food_sources if b.id != bee.id]
        partner = random.choice(partners)
        
        # Generate new position
        phi = np.random.uniform(-1, 1, len(bee.position))
        new_position = bee.position + phi * (bee.position - partner.position)
        
        # Apply bounds
        for i, (low, high) in enumerate(problem.bounds):
            new_position[i] = np.clip(new_position[i], low, high)
        
        return new_position
    
    async def _evaluate_fitness(
        self, 
        position: np.ndarray, 
        problem: OptimizationProblem
    ) -> float:
        """Evaluate fitness of a position"""
        if problem.objective_function:
            fitness = problem.objective_function(position)
            
            # Apply constraints as penalties
            for constraint in problem.constraints:
                violation = constraint(position)
                if violation > 0:
                    fitness += violation * 1000
            
            return fitness
        return 0.0
    
    def _is_better(self, fitness1: float, fitness2: float, minimize: bool) -> bool:
        """Check if fitness1 is better than fitness2"""
        return fitness1 < fitness2 if minimize else fitness1 > fitness2
    
    def _get_nectar_distribution(self) -> Dict[str, float]:
        """Get nectar amount distribution statistics"""
        nectar_amounts = [bee.nectar_amount for bee in self.food_sources]
        
        return {
            "mean": float(np.mean(nectar_amounts)),
            "std": float(np.std(nectar_amounts)),
            "max": float(np.max(nectar_amounts)),
            "min": float(np.min(nectar_amounts))
        }


# Firefly Algorithm
@dataclass
class Firefly:
    """Individual firefly in FA"""
    id: str = field(default_factory=lambda: f"firefly_{uuid4().hex[:8]}")
    position: np.ndarray = field(default_factory=lambda: np.random.rand(10))
    brightness: float = 0.0
    attractiveness: float = 1.0


class FireflyAlgorithm:
    """
    Firefly Algorithm
    Based on flashing behavior of fireflies for mating
    """
    
    def __init__(self, swarm_size: int = 25):
        self.swarm_size = swarm_size
        self.alpha = 0.2  # Randomization parameter
        self.beta_base = 1.0  # Attractiveness at r=0
        self.gamma = 1.0  # Light absorption coefficient
        self.fireflies: List[Firefly] = []
        self.best_position: Optional[np.ndarray] = None
        self.best_brightness: float = float('-inf')
        
    async def optimize(
        self,
        problem: OptimizationProblem,
        max_iterations: int = 100,
        verbose: bool = False
    ) -> OptimizationResult:
        """Run Firefly Algorithm optimization"""
        start_time = time.time()
        result = OptimizationResult(
            problem_id=problem.id,
            algorithm="FireflyAlgorithm"
        )
        
        try:
            # Initialize fireflies
            await self._initialize_fireflies(problem)
            
            for iteration in range(max_iterations):
                # Update firefly positions
                await self._update_fireflies(problem)
                
                # Record convergence
                result.convergence_history.append(-self.best_brightness if problem.minimize else self.best_brightness)
                
                if verbose and iteration % 10 == 0:
                    logger.info(
                        f"FA Iteration {iteration}: "
                        f"Best brightness = {self.best_brightness:.6f}"
                    )
                
                # Adapt parameters
                self._adapt_fa_parameters(iteration, max_iterations)
            
            # Set final result
            result.best_solution = self.best_position.copy()
            result.best_fitness = -self.best_brightness if problem.minimize else self.best_brightness
            result.iterations = max_iterations
            result.execution_time = time.time() - start_time
            result.success = True
            
            result.metadata = {
                "firefly_distribution": self._get_firefly_distribution(),
                "final_alpha": self.alpha
            }
            
        except Exception as e:
            logger.error(f"Firefly optimization failed: {e}")
            result.success = False
            result.metadata["error"] = str(e)
            
        return result
    
    async def _initialize_fireflies(self, problem: OptimizationProblem):
        """Initialize firefly swarm"""
        self.fireflies = []
        self.best_brightness = float('-inf')
        
        for _ in range(self.swarm_size):
            # Random position within bounds
            position = np.array([
                np.random.uniform(low, high)
                for low, high in problem.bounds
            ])
            
            firefly = Firefly(position=position)
            
            # Calculate brightness (inverse of cost for minimization)
            fitness = await self._evaluate_fitness(position, problem)
            firefly.brightness = -fitness if problem.minimize else fitness
            
            # Update global best
            if firefly.brightness > self.best_brightness:
                self.best_brightness = firefly.brightness
                self.best_position = firefly.position.copy()
            
            self.fireflies.append(firefly)
    
    async def _update_fireflies(self, problem: OptimizationProblem):
        """Update firefly positions based on attraction"""
        for i, firefly_i in enumerate(self.fireflies):
            for j, firefly_j in enumerate(self.fireflies):
                if firefly_j.brightness > firefly_i.brightness:
                    # Calculate distance
                    distance = euclidean(firefly_i.position, firefly_j.position)
                    
                    # Calculate attraction
                    attraction = self.beta_base * np.exp(-self.gamma * distance ** 2)
                    
                    # Move firefly_i toward firefly_j
                    random_vector = self.alpha * (np.random.rand(len(firefly_i.position)) - 0.5)
                    
                    firefly_i.position = (
                        firefly_i.position +
                        attraction * (firefly_j.position - firefly_i.position) +
                        random_vector
                    )
                    
                    # Apply bounds
                    for k, (low, high) in enumerate(problem.bounds):
                        firefly_i.position[k] = np.clip(firefly_i.position[k], low, high)
                    
                    # Update brightness
                    fitness = await self._evaluate_fitness(firefly_i.position, problem)
                    firefly_i.brightness = -fitness if problem.minimize else fitness
                    
                    # Update global best
                    if firefly_i.brightness > self.best_brightness:
                        self.best_brightness = firefly_i.brightness
                        self.best_position = firefly_i.position.copy()
    
    async def _evaluate_fitness(
        self, 
        position: np.ndarray, 
        problem: OptimizationProblem
    ) -> float:
        """Evaluate fitness of a position"""
        if problem.objective_function:
            fitness = problem.objective_function(position)
            
            # Apply constraints as penalties
            for constraint in problem.constraints:
                violation = constraint(position)
                if violation > 0:
                    fitness += violation * 1000
            
            return fitness
        return 0.0
    
    def _adapt_fa_parameters(self, iteration: int, max_iterations: int):
        """Adapt Firefly Algorithm parameters"""
        # Decrease randomization parameter
        self.alpha = 0.2 * (1 - iteration / max_iterations)
        
        # Optionally adapt gamma for better convergence
        self.gamma = 1.0 + 0.5 * iteration / max_iterations
    
    def _get_firefly_distribution(self) -> Dict[str, Any]:
        """Get firefly position distribution"""
        positions = np.array([f.position for f in self.fireflies])
        centroid = np.mean(positions, axis=0)
        
        return {
            "centroid": centroid.tolist(),
            "spread": float(np.mean([euclidean(pos, centroid) for pos in positions])),
            "brightness_variance": float(np.var([f.brightness for f in self.fireflies]))
        }


# Grey Wolf Optimizer
@dataclass
class Wolf:
    """Individual wolf in GWO"""
    id: str = field(default_factory=lambda: f"wolf_{uuid4().hex[:8]}")
    position: np.ndarray = field(default_factory=lambda: np.random.rand(10))
    fitness: float = float('inf')
    rank: str = "omega"  # alpha, beta, delta, omega


class GreyWolfOptimizer:
    """
    Grey Wolf Optimizer
    Simulates leadership hierarchy and hunting behavior of grey wolves
    """
    
    def __init__(self, pack_size: int = 30):
        self.pack_size = pack_size
        self.wolves: List[Wolf] = []
        self.alpha: Optional[Wolf] = None
        self.beta: Optional[Wolf] = None
        self.delta: Optional[Wolf] = None
        
    async def optimize(
        self,
        problem: OptimizationProblem,
        max_iterations: int = 100,
        verbose: bool = False
    ) -> OptimizationResult:
        """Run Grey Wolf Optimizer"""
        start_time = time.time()
        result = OptimizationResult(
            problem_id=problem.id,
            algorithm="GWO"
        )
        
        try:
            # Initialize wolf pack
            await self._initialize_pack(problem)
            
            for iteration in range(max_iterations):
                # Update wolf positions
                await self._update_positions(problem, iteration, max_iterations)
                
                # Update hierarchy
                await self._update_hierarchy(problem)
                
                # Record convergence
                if self.alpha:
                    result.convergence_history.append(self.alpha.fitness)
                
                if verbose and iteration % 10 == 0 and self.alpha:
                    logger.info(
                        f"GWO Iteration {iteration}: "
                        f"Alpha fitness = {self.alpha.fitness:.6f}"
                    )
            
            # Set final result
            if self.alpha:
                result.best_solution = self.alpha.position.copy()
                result.best_fitness = self.alpha.fitness
                result.success = True
            
            result.iterations = max_iterations
            result.execution_time = time.time() - start_time
            
            result.metadata = {
                "pack_hierarchy": self._get_hierarchy_info()
            }
            
        except Exception as e:
            logger.error(f"GWO optimization failed: {e}")
            result.success = False
            result.metadata["error"] = str(e)
            
        return result
    
    async def _initialize_pack(self, problem: OptimizationProblem):
        """Initialize wolf pack"""
        self.wolves = []
        
        for _ in range(self.pack_size):
            # Random position within bounds
            position = np.array([
                np.random.uniform(low, high)
                for low, high in problem.bounds
            ])
            
            wolf = Wolf(position=position)
            wolf.fitness = await self._evaluate_fitness(position, problem)
            
            self.wolves.append(wolf)
        
        # Initialize hierarchy
        await self._update_hierarchy(problem)
    
    async def _update_hierarchy(self, problem: OptimizationProblem):
        """Update wolf hierarchy based on fitness"""
        # Sort wolves by fitness
        sorted_wolves = sorted(
            self.wolves, 
            key=lambda w: w.fitness,
            reverse=not problem.minimize
        )
        
        # Assign ranks
        if len(sorted_wolves) >= 1:
            self.alpha = sorted_wolves[0]
            self.alpha.rank = "alpha"
        
        if len(sorted_wolves) >= 2:
            self.beta = sorted_wolves[1]
            self.beta.rank = "beta"
        
        if len(sorted_wolves) >= 3:
            self.delta = sorted_wolves[2]
            self.delta.rank = "delta"
        
        # Rest are omegas
        for wolf in sorted_wolves[3:]:
            wolf.rank = "omega"
    
    async def _update_positions(
        self,
        problem: OptimizationProblem,
        iteration: int,
        max_iterations: int
    ):
        """Update wolf positions based on hunting behavior"""
        if not self.alpha or not self.beta or not self.delta:
            return
        
        # Calculate a (decreases linearly from 2 to 0)
        a = 2 - 2 * iteration / max_iterations
        
        for wolf in self.wolves:
            if wolf.rank != "omega":
                continue
            
            # Calculate distances to alpha, beta, delta
            r1, r2 = np.random.rand(2)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * self.alpha.position - wolf.position)
            X1 = self.alpha.position - A1 * D_alpha
            
            r1, r2 = np.random.rand(2)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * self.beta.position - wolf.position)
            X2 = self.beta.position - A2 * D_beta
            
            r1, r2 = np.random.rand(2)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * self.delta.position - wolf.position)
            X3 = self.delta.position - A3 * D_delta
            
            # Update position
            wolf.position = (X1 + X2 + X3) / 3
            
            # Apply bounds
            for i, (low, high) in enumerate(problem.bounds):
                wolf.position[i] = np.clip(wolf.position[i], low, high)
            
            # Evaluate new position
            wolf.fitness = await self._evaluate_fitness(wolf.position, problem)
    
    async def _evaluate_fitness(
        self, 
        position: np.ndarray, 
        problem: OptimizationProblem
    ) -> float:
        """Evaluate fitness of a position"""
        if problem.objective_function:
            fitness = problem.objective_function(position)
            
            # Apply constraints as penalties
            for constraint in problem.constraints:
                violation = constraint(position)
                if violation > 0:
                    fitness += violation * 1000
            
            return fitness
        return 0.0
    
    def _get_hierarchy_info(self) -> Dict[str, Any]:
        """Get wolf pack hierarchy information"""
        info = {
            "alpha_fitness": self.alpha.fitness if self.alpha else None,
            "beta_fitness": self.beta.fitness if self.beta else None,
            "delta_fitness": self.delta.fitness if self.delta else None,
            "omega_count": sum(1 for w in self.wolves if w.rank == "omega")
        }
        
        return info


# Cuckoo Search Algorithm
class CuckooSearch:
    """
    Cuckoo Search Algorithm
    Based on brood parasitism of some cuckoo species
    """
    
    def __init__(self, nest_count: int = 25):
        self.nest_count = nest_count
        self.pa = 0.25  # Probability of discovering alien eggs
        self.alpha = 1.0  # Step size
        self.nests: List[np.ndarray] = []
        self.fitness: List[float] = []
        self.best_nest: Optional[np.ndarray] = None
        self.best_fitness: float = float('inf')
        
    async def optimize(
        self,
        problem: OptimizationProblem,
        max_iterations: int = 100,
        verbose: bool = False
    ) -> OptimizationResult:
        """Run Cuckoo Search optimization"""
        start_time = time.time()
        result = OptimizationResult(
            problem_id=problem.id,
            algorithm="CuckooSearch"
        )
        
        try:
            # Initialize nests
            await self._initialize_nests(problem)
            
            for iteration in range(max_iterations):
                # Generate new solutions via Levy flights
                new_nest = await self._get_cuckoo_via_levy_flight(problem)
                
                # Evaluate and potentially replace random nest
                j = np.random.randint(0, self.nest_count)
                if self._is_better(new_nest[1], self.fitness[j], problem.minimize):
                    self.nests[j] = new_nest[0]
                    self.fitness[j] = new_nest[1]
                
                # Abandon worst nests
                await self._abandon_worst_nests(problem)
                
                # Update best
                best_idx = np.argmin(self.fitness) if problem.minimize else np.argmax(self.fitness)
                if self._is_better(self.fitness[best_idx], self.best_fitness, problem.minimize):
                    self.best_fitness = self.fitness[best_idx]
                    self.best_nest = self.nests[best_idx].copy()
                
                # Record convergence
                result.convergence_history.append(self.best_fitness)
                
                if verbose and iteration % 10 == 0:
                    logger.info(
                        f"CS Iteration {iteration}: "
                        f"Best fitness = {self.best_fitness:.6f}"
                    )
            
            # Set final result
            result.best_solution = self.best_nest.copy()
            result.best_fitness = self.best_fitness
            result.iterations = max_iterations
            result.execution_time = time.time() - start_time
            result.success = True
            
            result.metadata = {
                "final_diversity": self._calculate_diversity()
            }
            
        except Exception as e:
            logger.error(f"Cuckoo Search failed: {e}")
            result.success = False
            result.metadata["error"] = str(e)
            
        return result
    
    async def _initialize_nests(self, problem: OptimizationProblem):
        """Initialize nest positions"""
        self.nests = []
        self.fitness = []
        self.best_fitness = float('inf') if problem.minimize else float('-inf')
        
        for _ in range(self.nest_count):
            # Random position within bounds
            position = np.array([
                np.random.uniform(low, high)
                for low, high in problem.bounds
            ])
            
            fitness = await self._evaluate_fitness(position, problem)
            
            self.nests.append(position)
            self.fitness.append(fitness)
            
            # Update best
            if self._is_better(fitness, self.best_fitness, problem.minimize):
                self.best_fitness = fitness
                self.best_nest = position.copy()
    
    async def _get_cuckoo_via_levy_flight(
        self, 
        problem: OptimizationProblem
    ) -> Tuple[np.ndarray, float]:
        """Generate new solution via Levy flight"""
        # Select random nest
        i = np.random.randint(0, self.nest_count)
        current_nest = self.nests[i]
        
        # Levy flight
        step = self._levy_flight(len(current_nest))
        new_nest = current_nest + self.alpha * step
        
        # Apply bounds
        for j, (low, high) in enumerate(problem.bounds):
            new_nest[j] = np.clip(new_nest[j], low, high)
        
        # Evaluate
        fitness = await self._evaluate_fitness(new_nest, problem)
        
        return new_nest, fitness
    
    def _levy_flight(self, dimension: int) -> np.ndarray:
        """Generate Levy flight step"""
        # Levy exponent
        beta = 1.5
        
        # Calculate sigma
        sigma = (
            math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
            (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2))
        ) ** (1 / beta)
        
        # Generate step
        u = np.random.randn(dimension) * sigma
        v = np.random.randn(dimension)
        step = u / (np.abs(v) ** (1 / beta))
        
        return step
    
    async def _abandon_worst_nests(self, problem: OptimizationProblem):
        """Abandon worst nests and build new ones"""
        # Sort nests by fitness
        sorted_indices = np.argsort(self.fitness)
        if not problem.minimize:
            sorted_indices = sorted_indices[::-1]
        
        # Number of nests to abandon
        n_abandon = int(self.pa * self.nest_count)
        
        # Replace worst nests
        for idx in sorted_indices[-n_abandon:]:
            # Generate new random position
            new_position = np.array([
                np.random.uniform(low, high)
                for low, high in problem.bounds
            ])
            
            new_fitness = await self._evaluate_fitness(new_position, problem)
            
            self.nests[idx] = new_position
            self.fitness[idx] = new_fitness
    
    async def _evaluate_fitness(
        self, 
        position: np.ndarray, 
        problem: OptimizationProblem
    ) -> float:
        """Evaluate fitness of a position"""
        if problem.objective_function:
            fitness = problem.objective_function(position)
            
            # Apply constraints as penalties
            for constraint in problem.constraints:
                violation = constraint(position)
                if violation > 0:
                    fitness += violation * 1000
            
            return fitness
        return 0.0
    
    def _is_better(self, fitness1: float, fitness2: float, minimize: bool) -> bool:
        """Check if fitness1 is better than fitness2"""
        return fitness1 < fitness2 if minimize else fitness1 > fitness2
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity"""
        if not self.nests:
            return 0.0
        
        positions = np.array(self.nests)
        centroid = np.mean(positions, axis=0)
        
        diversity = np.mean([euclidean(pos, centroid) for pos in positions])
        
        return float(diversity)


# Optimization Manager
class SwarmOptimizationManager:
    """
    Manager for running various swarm optimization algorithms
    """
    
    def __init__(self):
        self.algorithms = {
            "PSO": ParticleSwarmOptimization,
            "ACO": AntColonyOptimization,
            "BCO": BeeColonyOptimization,
            "FA": FireflyAlgorithm,
            "GWO": GreyWolfOptimizer,
            "CS": CuckooSearch
        }
        self.results_cache: Dict[str, OptimizationResult] = {}
        
    async def optimize(
        self,
        problem: OptimizationProblem,
        algorithm: str = "PSO",
        **kwargs
    ) -> OptimizationResult:
        """Run optimization using specified algorithm"""
        if algorithm not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Create algorithm instance
        algo_class = self.algorithms[algorithm]
        algo_instance = algo_class()
        
        # Run optimization
        result = await algo_instance.optimize(problem, **kwargs)
        
        # Cache result
        self.results_cache[result.problem_id] = result
        
        return result
    
    async def compare_algorithms(
        self,
        problem: OptimizationProblem,
        algorithms: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, OptimizationResult]:
        """Compare multiple algorithms on the same problem"""
        if algorithms is None:
            algorithms = list(self.algorithms.keys())
        
        results = {}
        
        for algo_name in algorithms:
            logger.info(f"Running {algo_name} optimization...")
            result = await self.optimize(problem, algo_name, **kwargs)
            results[algo_name] = result
        
        return results
    
    def get_best_algorithm(
        self, 
        results: Dict[str, OptimizationResult]
    ) -> Tuple[str, OptimizationResult]:
        """Determine best performing algorithm"""
        best_algo = None
        best_result = None
        best_fitness = float('inf')
        
        for algo_name, result in results.items():
            if result.success and result.best_fitness < best_fitness:
                best_fitness = result.best_fitness
                best_algo = algo_name
                best_result = result
        
        return best_algo, best_result
