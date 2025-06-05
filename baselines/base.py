import numpy as np
import torch

class Baseline:

    def __init__(self, init_obs, bounds, *args):
        self.bounds = bounds # bounds for the input space (shape: (2, n_dims) where n_dims is the number of input dimensions)
        self.dim = bounds.shape[1] # number of input dimensions
        self.X = init_obs[:, :self.dim]
        self.y = init_obs[:, self.dim]

    def propose(self):
        '''
        Run the algorithm to propose a new observation (X)
        '''
        raise NotImplementedError

    def update(self, new_obs):
        '''
        Update the algorithm with the new observation (X, y)
        '''
        self.X = np.vstack([self.X, new_obs[:self.dim]])
        self.y = np.append(self.y, new_obs[self.dim]) 