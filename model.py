"""
Model definition for Collaborative Cryptographic Parameter Optimization.

This module defines the main model that coordinates multiple explorer agents
searching for optimal lattice-based cryptography parameters.
"""

import mesa
import random
import math
from agents import ExplorerAgent


class CryptoOptimizationModel(mesa.Model):
    """
    A model where multiple agents collaboratively search for optimal
    lattice-based cryptographic parameters.
    
    The model implements a particle swarm optimization approach where
    agents explore the parameter space and share findings with neighbors.
    """
    
    def __init__(
        self,
        n_explorers=20,
        alpha=0.7,
        beta=0.3,
        communication_topology="ring",
        seed=None
    ):
        """
        Initialize the cryptographic parameter optimization model.
        
        Args:
            n_explorers: Number of explorer agents
            alpha: Weight for security in fitness function (0-1)
            beta: Weight for performance cost in fitness function (0-1)
            communication_topology: How agents communicate ("ring", "all", "random")
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        if seed is not None:
            random.seed(seed)
        
        # Model parameters
        self.n_explorers = n_explorers
        self.alpha = alpha
        self.beta = beta
        self.communication_topology = communication_topology
        
        # Parameter space bounds
        self.bounds = {
            'n': (256, 2048),      # Lattice dimension
            'q': (2048, 8192),     # Modulus
            'sigma': (2.0, 5.0)    # Gaussian noise std deviation
        }
        
        # Global best tracking
        self.global_best_params = None
        self.global_best_fitness = -float('inf')
        
        # Scheduler for agent activation
        # self.schedule = mesa.time.SimultaneousActivation(self)
        self.schedule = mesa.time.RandomActivation(self)
        
        # Create explorer agents
        self._create_agents()
        
        # Setup communication network
        self.setup_communication_topology()
        
        # Initialize global best from initial positions
        self.update_global_best()
        
        # Data collection
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Global_Best_Fitness": lambda m: m.global_best_fitness,
                "Global_Best_N": lambda m: m.global_best_params['n'] if m.global_best_params else None,
                "Global_Best_Q": lambda m: m.global_best_params['q'] if m.global_best_params else None,
                "Global_Best_Sigma": lambda m: m.global_best_params['sigma'] if m.global_best_params else None,
                "Average_Fitness": self.compute_average_fitness,
                "Diversity": self.compute_diversity,
                "Convergence_Rate": self.compute_convergence_rate
            },
            agent_reporters={
                "Fitness": "fitness_personal",
                "N": lambda a: a.current_params['n'],
                "Q": lambda a: a.current_params['q'],
                "Sigma": lambda a: a.current_params['sigma']
            }
        )
        
        # Initialize data collection
        self.datacollector.collect(self)
    
    def _create_agents(self):
        """
        Create explorer agents with random initial positions.
        
        Agents start with low-to-medium parameters to allow visible progression
        during optimization (rather than starting near optimal values).
        """
        for i in range(self.n_explorers):
            # Start with smaller values to see progression
            # n: favor smaller values initially (256 or 512)
            # q: start in lower-middle range (3000-5000)
            # sigma: start in middle-upper range (2.5-4.0)
            initial_params = {
                'n': random.choice([256, 256, 512, 512, 1024]),  # 40% chance of 256, 40% of 512, 20% of 1024
                'q': random.randint(3000, 5000),  # Lower-middle range
                'sigma': random.uniform(2.5, 4.0)  # Middle-upper range
            }
            
            # Variable inertia for diversity
            # Reduced range (0.4-0.7) to slow down convergence and prevent boundary rushing
            inertia = random.uniform(0.4, 0.7)
            
            agent = ExplorerAgent(
                model=self,
                initial_params=initial_params,
                inertia=inertia
            )
            
            self.schedule.add(agent)
    
    def setup_communication_topology(self):
        """
        Establish communication links between agents based on topology.
        
        Topologies:
        - "ring": Each agent communicates with 2 neighbors (circular)
        - "all": Full connectivity (all-to-all)
        - "random": Each agent has 4 random neighbors
        """
        agents = list(self.schedule.agents)
        
        if self.communication_topology == "ring":
            # Ring topology: bi-directional circular neighbors
            for i, agent in enumerate(agents):
                left_neighbor = agents[(i - 1) % len(agents)]
                right_neighbor = agents[(i + 1) % len(agents)]
                agent.neighbors_list = [left_neighbor, right_neighbor]
        
        elif self.communication_topology == "all":
            # Complete graph: everyone communicates with everyone
            for agent in agents:
                agent.neighbors_list = [a for a in agents if a != agent]
        
        elif self.communication_topology == "random":
            # Random connections: 4 neighbors per agent
            for agent in agents:
                others = [a for a in agents if a != agent]
                k = min(4, len(others))
                agent.neighbors_list = random.sample(others, k)
        
        else:
            raise ValueError(f"Unknown topology: {self.communication_topology}")
    
    def step(self):
        """
        Execute one step of the model.
        
        1. All agents perform their step (simultaneously)
        2. Update global best
        3. Collect data
        """
        # Step all agents
        self.schedule.step()
        
        # Update global best from all agents
        self.update_global_best()
        
        # Collect data
        self.datacollector.collect(self)
    
    def update_global_best(self):
        """
        Update the global best parameters found by any agent.
        """
        for agent in self.schedule.agents:
            if agent.fitness_personal > self.global_best_fitness:
                self.global_best_fitness = agent.fitness_personal
                self.global_best_params = agent.best_personal.copy()
    
    def compute_average_fitness(self):
        """
        Calculate average fitness across all agents.
        
        Returns:
            float: Mean fitness value
        """
        if not self.schedule.agents:
            return 0
        
        total_fitness = sum(agent.fitness_personal for agent in self.schedule.agents)
        return total_fitness / len(self.schedule.agents)
    
    def compute_diversity(self):
        """
        Measure diversity of agent positions in parameter space.
        
        Uses standard deviation of 'n' values as a proxy for diversity.
        Higher diversity means agents are more spread out.
        
        Returns:
            float: Diversity metric
        """
        if not self.schedule.agents:
            return 0
        
        n_values = [agent.current_params['n'] for agent in self.schedule.agents]
        
        mean_n = sum(n_values) / len(n_values)
        variance = sum((n - mean_n) ** 2 for n in n_values) / len(n_values)
        std_dev = math.sqrt(variance)
        
        return std_dev
    
    def compute_convergence_rate(self):
        """
        Measure how quickly the swarm is converging.
        
        Returns the percentage of agents within 10% of global best fitness.
        
        Returns:
            float: Convergence rate (0-1)
        """
        if not self.schedule.agents:
            return 0
        
        # Handle case where global_best_fitness is negative or zero
        if self.global_best_fitness <= 0:
            # Use distance from worst instead
            fitness_values = [agent.fitness_personal for agent in self.schedule.agents]
            if not fitness_values:
                return 0
            worst_fitness = min(fitness_values)
            fitness_range = self.global_best_fitness - worst_fitness
            
            if fitness_range <= 0:
                return 1.0  # All agents have same fitness
            
            # Count agents within 10% of the range from the best
            threshold = self.global_best_fitness - 0.1 * fitness_range
            converged_agents = sum(
                1 for agent in self.schedule.agents 
                if agent.fitness_personal >= threshold
            )
        else:
            # Original logic for positive fitness
            threshold = 0.9 * self.global_best_fitness
            converged_agents = sum(
                1 for agent in self.schedule.agents 
                if agent.fitness_personal >= threshold
            )
        
        return converged_agents / len(self.schedule.agents)
    
    def get_security_level(self, params):
        """
        Estimate security level using Lindner-Peikert approximation.
        
        Args:
            params: Dict with 'n', 'q', 'sigma'
        
        Returns:
            float: Estimated security in bits
        """
        n = params['n']
        q = params['q']
        sigma = params['sigma']
        
        # Re-use the logic from Agent class (duplication for access from model)
        if sigma <= 0 or q <= 0 or n <= 0:
            return 0
            
        try:
            log2_q = math.log2(q)
            if q <= sigma: return 0 # Degenerate case
            
            log2_ratio = math.log2(q / sigma)
            
            # Root Hermite Factor log
            log2_delta = (log2_ratio ** 2) / (4 * n * log2_q)
            
            if log2_delta <= 0: return 0
            security_bits = (1.8 / log2_delta) - 110
            
            return max(0, security_bits)
        except:
            return 0
    
    def get_performance_cost(self, params):
        """
        Estimate computational cost for given parameters.
        
        Args:
            params: Dict with 'n', 'q', 'sigma'
        
        Returns:
            float: Normalized cost metric
        """
        n = params['n']
        q = params['q']
        
        cost = (n ** 2) * math.log2(q) / 1e6
        return cost
    
    def run_model(self, n_steps=100):
        """
        Run the model for a specified number of steps.
        
        Args:
            n_steps: Number of simulation steps
        """
        for _ in range(n_steps):
            self.step()
    
    def get_results_summary(self):
        """
        Get a summary of optimization results.
        
        Returns:
            dict: Summary statistics and best parameters
        """
        return {
            'best_params': self.global_best_params,
            'best_fitness': self.global_best_fitness,
            'security_bits': self.get_security_level(self.global_best_params) if self.global_best_params else None,
            'performance_cost': self.get_performance_cost(self.global_best_params) if self.global_best_params else None,
            'avg_fitness': self.compute_average_fitness(),
            'diversity': self.compute_diversity(),
            'convergence': self.compute_convergence_rate(),
            'total_steps': self.schedule.steps
        }