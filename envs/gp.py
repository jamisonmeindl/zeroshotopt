import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import GPy
import contextlib
from tqdm import tqdm
import sys
from scipy.stats.qmc import LatinHypercube
from scipy.optimize import minimize
from itertools import product
from enum import Enum, auto
from scipy.optimize import differential_evolution
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import time

sys.path.append('..')
sys.path.append('../baselines')

device = 'cpu'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

class KernelType(Enum):
    RBF = auto()
    MATERN32 = auto()
    MATERN52 = auto()
    RATIONAL_QUADRATIC = auto()
    EXPONENTIAL = auto()
    COSINE = auto()

class GP:
    def __init__(self, N=2, seed=None, use_ard=True, max_kernels=2):
        self.N = N
        self.seed = seed
        self.use_ard = use_ard
        self.max_kernels = max_kernels
        if seed is not None:
            set_seed(seed)
        self.variances = [np.exp(np.random.uniform(np.log(0.1), np.log(2.0))) for i in range(max_kernels)]
        self.lengthscales = [self._sample_lengthscale() for i in range(max_kernels)]
        self.num_samples = self._sample_num_samples()
        self.kernel = self._create_random_kernel()
        self.X_initial = self._sample_initial_points()
        self.K_initial = self.kernel.K(self.X_initial)
        self.K_initial += 1e-5 * np.eye(self.K_initial.shape[0])
        self.y_sampled_initial = np.random.multivariate_normal(np.zeros(self.num_samples), self.K_initial)
        self.v=0
        self.opt_x = None

    def _sample_lengthscale(self):
        if self.use_ard:
            return np.exp(np.random.uniform(np.log(0.1), np.log(10), size=self.N))
        else:
            return np.exp(np.random.uniform(np.log(0.1), np.log(10)))

    def _sample_num_samples(self):
        return np.random.randint(10*self.N, 30*self.N)

    def _sample_initial_points(self):
        sampler = LatinHypercube(d=self.N, seed =self.seed)
        return sampler.random(n=self.num_samples) * 2 - 1  

    def _create_base_kernel(self, kernel_type, variance, lengthscale):
        if kernel_type == KernelType.RBF:
            return GPy.kern.RBF(input_dim=self.N, variance=variance, lengthscale=lengthscale, ARD=self.use_ard)
        elif kernel_type == KernelType.MATERN32:
            return GPy.kern.Matern32(input_dim=self.N, variance=variance, lengthscale=lengthscale, ARD=self.use_ard)
        elif kernel_type == KernelType.MATERN52:
            return GPy.kern.Matern52(input_dim=self.N, variance=variance, lengthscale=lengthscale, ARD=self.use_ard)
        elif kernel_type == KernelType.RATIONAL_QUADRATIC:
            alpha = np.exp(np.random.uniform(np.log(0.1), np.log(2.0)))
            return GPy.kern.RatQuad(input_dim=self.N, variance=variance, lengthscale=lengthscale, power=alpha, ARD=self.use_ard)
        elif kernel_type == KernelType.EXPONENTIAL:
            return GPy.kern.Exponential(input_dim=self.N, variance=variance, lengthscale=lengthscale, ARD=self.use_ard)
        elif kernel_type == KernelType.COSINE:
            return GPy.kern.Cosine(input_dim=self.N, variance=variance, lengthscale=lengthscale, ARD=self.use_ard)
        else:
            raise ValueError("Unsupported kernel type")

    def _create_random_kernel(self):
        num_kernels = np.random.randint(1, self.max_kernels + 1)
        kernels = [self._create_base_kernel(random.choice(list(KernelType)), variance, lengthscale) for variance, lengthscale in zip(self.variances[:num_kernels], self.lengthscales[:num_kernels])]
        
        if len(kernels) == 1:
            return kernels[0]
        else:
            combination_type = np.random.choice(["add", "multiply"])
            if combination_type == "add":
                return np.sum(kernels)
            else:
                return np.prod(kernels)  

    def evaluate(self, x):
        with temp_seed(0):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            K_new_initial = self.kernel.K(x, self.X_initial)
            K_new_new = self.kernel.K(x)
            y_mean_new = K_new_initial @ np.linalg.solve(self.K_initial, self.y_sampled_initial)
            K_new_conditional = K_new_new - K_new_initial @ np.linalg.solve(self.K_initial, K_new_initial.T)
            return np.random.multivariate_normal(y_mean_new, K_new_conditional).flatten()

    def get_kernel_description(self):
        def describe_kernel(kernel):
            if isinstance(kernel, GPy.kern.Add):
                return " + ".join([describe_kernel(k) for k in kernel.parts])
            elif isinstance(kernel, GPy.kern.Prod):
                return " * ".join([describe_kernel(k) for k in kernel.parts])
            else:
                return kernel.__class__.__name__
        
        return describe_kernel(self.kernel)

    def grid(self, dim):
        assert self.N == 2, 'must be 2 dimensional'
        x = np.linspace(-1, 1, dim)
        y = np.linspace(-1, 1, dim)
        X, Y = np.meshgrid(x, y)
        with torch.no_grad():
            Z = np.vectorize(lambda x, y: self.evaluate((np.array([x, y]))))(X, Y)
        return X,Y,Z

    def visualize(self, ax=None, X=None, Y=None, Z=None, init_points=None, points=None, dynamic=False, savefig=False, save_path='', score = 0):
        assert self.N == 2, 'must be 2 dimensional'
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        if X is None:
            X, Y, Z = self.grid(200)
        contour = ax.contourf(X, Y, Z, 20)
        all_points = []
        
        if init_points is not None:
            x_vals, y_vals = zip(*init_points)
            ax.scatter(x_vals, y_vals, color='grey', s=40, label="Initial Sample Points")
            all_points.extend(init_points)
        
        if points is not None:

            all_points.extend(points)
            points = np.array(points)
            indices = np.arange(len(all_points))
            norm = Normalize(vmin=0, vmax=len(points) - 1)
            cmap = plt.cm.Reds  
            sm = ScalarMappable(norm=norm, cmap=cmap)

            for i in range(len(points)):
                point = points[i]
                color = sm.to_rgba(indices[i])  
                ax.scatter(point[0], point[1], color=color, s=40) 
            all_points.extend(points)
        
        if all_points:
            all_points = np.array(all_points)
            function_values = np.empty((all_points.shape[0]))
            for i in range(function_values.shape[0]):
                function_values[i] = self.evaluate(all_points[i])
            min_value = np.min(function_values)
            min_index = np.argmin(function_values)
            min_point = all_points[min_index]
            max_index = np.argmax(function_values)
            ax.scatter(min_point[0], min_point[1], color='white', marker='*', s=100, label=f'Trajectory Min Value: {min_value:.4f}')
            
            
        if self.opt_x is not None:
            ax.scatter(self.opt_x[0], self.opt_x[1], color='red', marker='*', s=600, label=f'Global Min Value: {self.v:.4f}')
        ax.scatter([], [], color='none', label=f'Score: {score:.4f}')

        # ax.set_xticks([])
        # ax.set_yticks([])
        plt.legend()
        plt.tight_layout()
        if savefig:
            plt.savefig(f'{save_path}/GP_{self.seed}_{self.get_kernel_description()}.png')
    
    


class GPEnv(gym.Env):
    def __init__(self, N = 2):
        self.N = N
        self.f = None
        self.action_space = Box(-1, 1, (N, ), np.float32) #
        self.observation_space = Box(-np.inf, np.inf, (N+1,), np.float32)
        self.f_min = float('inf')

    def step(self, x):
        f_x = self.f.evaluate(x)

        if self.f.v is None:
            r_n = max(0, self.f_min - f_x.item())
            self.f_min = min(self.f_min, f_x.item())
        else:
            self.f_min = min(self.f_min, f_x.item())
            r_n = self.f.v - self.f_min 
        observation = np.concatenate([x, f_x], dtype=np.float32)
        return observation, r_n, False, False, {}
    
    def reset(self, seed=None, options=None):
        super().reset()

        if self.f is None or (options is not None and options['reset_f'] == True):
            self.f = GP(N = self.N, seed=seed)
            if (options is not None and options['find_min'] == True):
                self.f.opt_x, self.f.v = self.find_minimum()
            else:
                self.f.opt_x, self.f.v = None, None

        self.action_space.seed(seed)
        action = self.action_space.sample()
        f_action = self.f.evaluate(action)
        self.f_min = f_action.item()
        
        observation = np.concatenate((action, f_action),dtype=np.float32)
        
        return observation, {}

    def find_minimum(self):
        def objective_function(x):
            return self.f.evaluate(np.array(x))
        local_refinements = self.N
        bounds = [(-1, 1)] * self.N
        
        result_global = differential_evolution(objective_function, bounds=bounds, maxiter=2500)
        return result_global.x, result_global.fun


if __name__ == "__main__":
    env = GPEnv(N=2)
    env.reset(seed=0, options={'reset_f':True})
    print('step: ', env.step(np.array([0.5, 0.5])))



    