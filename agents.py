"""
Agent definitions for the Collaborative Cryptographic Parameter Optimization Model.

This module defines the ExplorerAgent class that searches for optimal 
lattice-based cryptography parameters using swarm intelligence.
"""

import mesa
import random
import math


class ExplorerAgent(mesa.Agent):
    """
    An agent that explores the parameter space of lattice-based cryptography
    to find optimal configurations balancing security and performance.
    
    The agent uses Particle Swarm Optimization (PSO) principles:
    - Maintains current position (parameters)
    - Tracks personal best
    - Communicates with neighbors
    - Updates velocity based on personal and social learning
    """
    
    def __init__(self, model, initial_params, inertia=0.7):
        """
        Initialize an explorer agent.
        
        Args:
            model: Reference to the model instance
            initial_params: Dict with 'n', 'q', 'sigma' initial values
            inertia: Inertia weight for velocity updates (0.5-0.9)
        """
        super().__init__(model)
        
        # Current position in parameter space
        self.current_params = initial_params.copy()

        # Compute velocity limits
        self.v_limits = self._compute_velocity_limits()
        
        # Velocity in parameter space
        self.velocity = {
            'n': random.uniform(-self.v_limits['n']/4, self.v_limits['n']/4),
            'q': random.uniform(-self.v_limits['q']/4, self.v_limits['q']/4),
            'sigma': random.uniform(-self.v_limits['sigma']/4, self.v_limits['sigma']/4)
        }
        
        # Personal best
        self.best_personal = initial_params.copy()
        self.fitness_personal = self.evaluate_fitness()
        
        # PSO parameters
        self.inertia = inertia
        self.cognitive_weight = 1.8  # Reduced from 2.0 for smoother convergence
        self.social_weight = 1.8     # Reduced from 2.0 for smoother convergence
        
        # Communication
        self.neighbors_list = []
        self.inbox = []
        
        # History tracking
        self.fitness_history = [self.fitness_personal]

    def _compute_velocity_limits(self):
        """
        Compute velocity limits for each parameter.
        
        Reduced velocity limits to prevent agents from rushing to boundaries
        and disappearing from visualization. These limits balance:
        - Exploration: Agents can still search the parameter space
        - Stability: Prevents wild oscillations and boundary collisions
        - Visualization: Keeps agents visible in parameter space plots
        
        Standard PSO practices recommend 10-20% of parameter range for V_max
        to ensure stable convergence without premature boundary attraction.
        
        Returns:
            dict: Velocity limits (V_max) for each parameter.
        """
        bounds = self.model.bounds
        
        return {
            'n': 0.15 * (bounds['n'][1] - bounds['n'][0]),      # ~270 per step
            'q': 0.05 * (bounds['q'][1] - bounds['q'][0]),      # ~307 per step (reduced from 0.08)
            'sigma': 0.15 * (bounds['sigma'][1] - bounds['sigma'][0])  # ~0.45 per step (reduced from 0.33)
        }
        
    def evaluate_fitness(self):
        """
        Evaluate the fitness of current parameters.
        
        Fitness = α * Security - β * Cost (normalized to 0-100 scale)
        
        Returns:
            float: Fitness value (higher is better)
        """
        n = self.current_params['n']
        q = self.current_params['q']
        sigma = self.current_params['sigma']
        
        # Security estimation (simplified Lattice estimator)
        # Based on BKZ block size needed for attack
        # Security roughly scales with n * log(q) / log(sigma)
        # NOTE: This formula gives values in range ~500-9000 bits (not realistic but used for optimization)
        security_bits = (n * math.log2(q)) / (2 * math.log2(sigma) + 1)
        
        # Performance cost (operations scale with n^2 * log(q))
        cost = (n ** 2) * math.log2(q)

        # Calculate normalization bounds based on actual parameter space
        # Using the model bounds to get realistic min/max values
        bounds = self.model.bounds

        # S_MIN = self.model.bounds['n'][0] * math.log2(self.model.bounds['q'][0]) / (2 * math.log2(self.model.bounds['sigma'][1]) + 1)
        # S_MAX = self.model.bounds['n'][1] * math.log2(self.model.bounds['q'][1]) / (2 * math.log2(self.model.bounds['sigma'][0]) + 1)
        # C_MIN = self.model.bounds['n'][0]**2 * math.log2(self.model.bounds['q'][0])
        # C_MAX = self.model.bounds['n'][1]**2 * math.log2(self.model.bounds['q'][1])
        
        # Minimum security: smallest n, smallest q, largest sigma
        S_MIN = bounds['n'][0] * math.log2(bounds['q'][0]) / (2 * math.log2(bounds['sigma'][1]) + 1)
        # Maximum security: largest n, largest q, smallest sigma  
        S_MAX = bounds['n'][1] * math.log2(bounds['q'][1]) / (2 * math.log2(bounds['sigma'][0]) + 1)
        
        # Minimum cost: smallest n, smallest q
        C_MIN = bounds['n'][0]**2 * math.log2(bounds['q'][0])
        # Maximum cost: largest n, largest q
        C_MAX = bounds['n'][1]**2 * math.log2(bounds['q'][1])

        # Normalize security and cost to 0-1 range
        security_normalized = max(0, min(1, (security_bits - S_MIN) / (S_MAX - S_MIN)))
        cost_normalized = max(0, min(1, (cost - C_MIN) / (C_MAX - C_MIN)))

        # Combined fitness (scaled to 0-100 for better visibility)
        alpha = self.model.alpha  # Security weight
        beta = self.model.beta    # Cost weight
        
        # Scale to 0-100 range for better visualization
        fitness = 100 * (alpha * security_normalized - beta * cost_normalized)
        
        return fitness

    
    def step(self):
        """
        Execute one step of the agent's behavior.
        
        Follows PSO algorithm:
        1. Communicate findings with neighbors
        2. Learn from neighbors
        3. Update velocity
        4. Move to new position
        5. Evaluate new position
        """
        # 1. Share current best with neighbors
        self.communicate_findings()
        
        # 2. Process messages from neighbors
        self.learn_from_neighbors()
        
        # 3. Update velocity based on PSO rules
        self.update_velocity()
        
        # 4. Move to new position
        self.move()
        
        # 5. Evaluate new position
        current_fitness = self.evaluate_fitness()
        self.fitness_history.append(current_fitness)
        
        # 6. Update personal best if improved
        if current_fitness > self.fitness_personal:
            self.fitness_personal = current_fitness
            self.best_personal = self.current_params.copy()
    
    def communicate_findings(self):
        """
        Send personal best findings to neighboring agents.
        """
        message = {
            'sender_id': self.unique_id,
            'params': self.best_personal.copy(),
            'fitness': self.fitness_personal,
            'timestamp': self.model.schedule.steps
        }
        
        for neighbor in self.neighbors_list:
            neighbor.receive_message(message)
    
    def receive_message(self, message):
        """
        Receive a message from another agent.
        
        Args:
            message: Dict containing sender info, params, and fitness
        """
        self.inbox.append(message)
    
    def learn_from_neighbors(self):
        """
        Update knowledge based on neighbors' findings.
        
        Implements the social component of PSO by finding the best
        solution among neighbors.
        """
        if not self.inbox:
            return
        
        # Find best neighbor solution
        best_neighbor_msg = max(self.inbox, key=lambda m: m['fitness'])
        best_neighbor_params = best_neighbor_msg['params']
        
        # Store for velocity update
        self.best_neighbor = best_neighbor_params
        
        # Clear inbox
        self.inbox = []
    
    def update_velocity(self):
        """
        Update velocity using PSO formula.
        
        v(t+1) = w*v(t) + c1*r1*(pbest - x(t)) + c2*r2*(nbest - x(t))
        
        where:
        - w: inertia weight
        - c1: cognitive weight
        - c2: social weight
        - r1, r2: random values [0,1]
        - pbest: personal best
        - nbest: neighborhood best
        """
        for param in ['n', 'q', 'sigma']:
            # Inertia component
            inertia_component = self.inertia * self.velocity[param]
            
            # Cognitive component (personal best)
            r1 = random.random()
            cognitive_component = self.cognitive_weight * r1 * (
                self.best_personal[param] - self.current_params[param]
            )
            
            # Social component (neighbor best)
            social_component = 0
            if hasattr(self, 'best_neighbor'):
                r2 = random.random()
                social_component = self.social_weight * r2 * (
                    self.best_neighbor[param] - self.current_params[param]
                )
            
            # Update velocity
            self.velocity[param] = (inertia_component + cognitive_component + social_component)

            self.velocity[param] = max(
                -self.v_limits[param],
                min(self.velocity[param], self.v_limits[param])
            )
    
    def move(self):
        """
        Update position based on velocity, respecting bounds.
        """
        bounds = self.model.bounds
        
        # Update each parameter
        for param in ['n', 'q', 'sigma']:
            new_value = self.current_params[param] + self.velocity[param]
            
            # Apply bounds
            min_val, max_val = bounds[param]
            new_value = max(min_val, min(new_value, max_val))
            
            # Special handling for 'n' - must be power of 2
            if param == 'n':
                # Find nearest power of 2
                powers = [256, 512, 1024, 2048]
                new_value = min(powers, key=lambda x: abs(x - new_value))
            
            # Special handling for 'q' - should be prime-like
            elif param == 'q':
                new_value = int(new_value)
            
            self.current_params[param] = new_value