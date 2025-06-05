import numpy as np
import logging; logging.disable(logging.CRITICAL)

from base import Baseline

#############################
# Differential Evolution (DE)
#############################

class DE(Baseline):
    def __init__(self, init_obs, bounds, F=0.8, CR=0.9):
        """
        Parameters:
            init_obs : list/array of initial observations. Each observation is expected to have
                       self.dim elements for the candidate and one additional element for the objective.
            bounds : a tuple (lower_bounds, upper_bounds)
            F : scaling factor for mutation (default 0.8)
            CR: crossover probability (default 0.9)
        """
        super().__init__(init_obs, bounds)
        self.F = F
        self.CR = CR
        # Initialize population and objective values using NumPy for vectorized handling.
        self.population = np.array([obs[:self.dim] for obs in init_obs])
        self.scores = np.array([obs[self.dim] for obs in init_obs])
        self.pop_size = self.population.shape[0]
        self.current_idx = 0  # Pointer for round-robin updates

    def propose(self):
        """Generate a candidate solution for the current individual using DE mutation and crossover."""
        i = self.current_idx
        # Pick three distinct indices different from i.
        indices = np.delete(np.arange(self.pop_size), i)
        r1, r2, r3 = np.random.choice(indices, 3, replace=False)
        
        # Mutation: vectorized computation using NumPy.
        mutant = self.population[r1] + self.F * (self.population[r2] - self.population[r3])
        
        # Crossover: combine the current individual and the mutant.
        # Using vectorized operations with a random mask.
        mask = np.random.rand(self.dim) < self.CR
        trial = np.where(mask, mutant, self.population[i])
        
        # Enforce bounds using NumPy's clip.
        lower_bounds, upper_bounds = self.bounds
        trial = np.clip(trial, lower_bounds, upper_bounds)
        
        # Store candidate for update.
        self.proposed_idx = i
        self.proposed_candidate = trial
        
        return trial, False

    def update(self, new_obs):
        """
        Accept the new observation (candidate solution and its objective value) and update the population.
        If the candidate is better (lower objective) than the current individual, replace it.
        """
        super().update(new_obs)
        candidate_obj = new_obs[self.dim]
        if candidate_obj < self.scores[self.proposed_idx]:
            self.population[self.proposed_idx] = self.proposed_candidate
            self.scores[self.proposed_idx] = candidate_obj
        # Advance to the next individual.
        self.current_idx = (self.current_idx + 1) % self.pop_size

#############################
# Particle Swarm Optimization (PSO)
#############################

class PSO(Baseline):
    def __init__(self, init_obs, bounds, w=0.5, c1=1.5, c2=1.5):
        """
        Parameters:
            init_obs : list/array of initial observations. Each observation has self.dim elements for the candidate
                       and one additional element for the objective.
            bounds : a tuple (lower_bounds, upper_bounds)
            w   : inertia weight
            c1  : cognitive (personal best) acceleration coefficient
            c2  : social (global best) acceleration coefficient
        """
        super().__init__(init_obs, bounds)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        # Initialize positions and scores.
        self.population = np.array([obs[:self.dim] for obs in init_obs])
        self.scores = np.array([obs[self.dim] for obs in init_obs])
        self.pop_size = self.population.shape[0]
        # Set personal bests as the initial positions.
        self.pbest = self.population.copy()
        self.pbest_scores = self.scores.copy()
        # Determine global best.
        best_idx = np.argmin(self.scores)
        self.global_best = self.population[best_idx]
        self.global_best_score = self.scores[best_idx]
        # Initialize velocities with small random values.
        self.velocities = np.random.uniform(-1, 1, size=self.population.shape)
        self.current_idx = 0  # Pointer for sequential updates

    def propose(self):
        """Generate a new candidate for the current particle using the PSO velocity update rule."""
        i = self.current_idx
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        
        # Update velocity vector using vectorized arithmetic.
        self.velocities[i] = (self.w * self.velocities[i] +
                              self.c1 * r1 * (self.pbest[i] - self.population[i]) +
                              self.c2 * r2 * (self.global_best - self.population[i]))
        candidate = self.population[i] + self.velocities[i]
        
        # Enforce the variable bounds.
        lower_bounds, upper_bounds = self.bounds
        candidate = np.clip(candidate, lower_bounds, upper_bounds)
        
        # Store the candidate for the update step.
        self.proposed_idx = i
        self.proposed_candidate = candidate
        
        return candidate, False

    def update(self, new_obs):
        """
        Update the current particle with the new observation.
        Adjust personal and global bests if the candidate's objective improves.
        """
        super().update(new_obs)
        candidate_obj = new_obs[self.dim]
        i = self.proposed_idx
        
        # Update the particle's current position and score.
        self.population[i] = self.proposed_candidate
        self.scores[i] = candidate_obj
        
        # Update personal best if there is an improvement.
        if candidate_obj < self.pbest_scores[i]:
            self.pbest[i] = self.proposed_candidate
            self.pbest_scores[i] = candidate_obj
        
        # Update global best if this candidate is the best seen so far.
        if candidate_obj < self.global_best_score:
            self.global_best = self.proposed_candidate
            self.global_best_score = candidate_obj
        
        # Move on to the next particle.
        self.current_idx = (self.current_idx + 1) % self.pop_size
