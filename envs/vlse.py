import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import botorch
from botorch.test_functions import *
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from scipy.optimize import differential_evolution


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

restrictions = {
    'ackley': None,
    'beale': 2,
    'branin': 2,
    'bukin': 2,
    'cos8': 8,
    'dropwave': 2,
    'dixonprice': None,
    'eggholder': 2,
    'griewank': None,
    'hartmann': 6, # only dim 3 or 6, only using 6
    'holdertable': 2,
    'levy': None,
    'michalewicz': None,
    'powell': None,
    'rastrigin': None, # any
    'rosenbrock':None, # any
    'shekel': 4, # 4
    'sixhumpcamel': 2, # 2
    'styblinskitang':None, # any
    'threehumpcamel': 2, # 2
}

PROBLEMS = {
    'ackley': Ackley,
    'beale': Beale,
    'branin': Branin,
    'bukin': Bukin,
    'cosine8': Cosine8,
    'dropwave':DropWave,
    'dixonprice': DixonPrice,
    'eggholder': EggHolder,
    'griewank': Griewank,
    'hartmann': Hartmann, # only dim 3 or 6, only using 6
    'holdertable': HolderTable,
    'levy': Levy,
    'michalewicz':Michalewicz,
    'powell':Powell,
    'rastrigin': Rastrigin, # any
    'rosenbrock':Rosenbrock, # any
    'shekel': Shekel, # 4
    'sixhumpcamel': SixHumpCamel, # 2
    'styblinskitang':StyblinskiTang, # any
    'threehumpcamel': ThreeHumpCamel, # 2
}

class BOProblem:
    def __init__(self, N=2, seed=None, problem_type='threehumpcamel'):
        self.opt_x = None
        self.problem_type = problem_type
        self.problems = PROBLEMS
        self.dim = None
        self.bounds = None
        self.f = None
        self.v = None
        self.x_min = None
        self.x_mins = None
        self.seed = seed
        # self.opt_x = None

        problem_class = self.problems[problem_type]
        # manual settings
        # if problem_type == 'hartmann': problem_class.dim = 6

        # Match dimensions
        if hasattr(problem_class, 'dim'): 
            self.dim = problem_class.dim
            if self.dim != N: f'WARNING: type {problem_class} requires dim {problem_class.dim}, but attempted {N}. defaulting to dim {self.dim}'
            self.f = problem_class()
        else:
            self.f = problem_class(dim=N)
            self.dim = N

        # Find bounds
        if hasattr(problem_class, '_bounds'): self.bounds = problem_class._bounds
        else: self.bounds = self.f._bounds

        # locate global minimum
        if problem_class._optimizers: 
            self.x_mins = [self._normalize(x) for x in problem_class._optimizers]
            self.x_min = self.x_mins[0]
        elif self.f._optimizers:
            self.x_mins = [self._normalize(x) for x in self.f._optimizers]
            self.x_min = self.x_mins[0]

        if hasattr(problem_class, '_optimal_value'): self.v = problem_class._optimal_value
        else: self.v = self.f._optimal_value


    def get_dim(self):
        return self.dim

    def get_min_x(self):
        return self.x_min

    def get_min_y(self):
        return self.v

    def get_kernel_description(self):
        return self.problem_type

    def _normalize(self, x):
        '''
        Map x from bounds interval to [-1, 1] for every dimension
        '''
        bounds = np.array(self.bounds)
        min_bounds = bounds[:, 0]
        max_bounds = bounds[:, 1]
        x_normalized = 2 * (x - min_bounds) / (max_bounds - min_bounds) - 1
        return x_normalized

    def _denormalize(self, x):
        '''
        map x in [-1,1] to bounds interval for every dim
        '''
        bounds = np.array(self.bounds)
        min_bounds = bounds[:, 0]
        max_bounds = bounds[:, 1]
        x_denormalized = (x + 1) / 2 * (max_bounds - min_bounds) + min_bounds

        return x_denormalized

    def evaluate(self, x):
        x = self._denormalize(x)
        x = torch.tensor(x).float()
        return np.array([self.f.evaluate_true(x).item()])

    def grid(self, dim):
        assert self.dim == 2, 'must be 2 dimensional'
        x = np.linspace(-1, 1, dim)
        y = np.linspace(-1, 1, dim)
        X, Y = np.meshgrid(x, y)
        with torch.no_grad():
            Z = np.vectorize(lambda x, y: self.evaluate((np.array([x, y]))))(X, Y)
        return X,Y,Z
    def visualize(self, ax=None, X=None, Y=None, Z=None, init_points=None, points=None, dynamic=False, savefig=False):
        # assert self.N == 2, 'must be 2 dimensional'
        
        
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
            cmap = plt.cm.Reds  # Single color gradient (e.g., 'viridis', 'plasma', etc.)
            sm = ScalarMappable(norm=norm, cmap=cmap)

            # Plot trajectory points as dots with gradient
            for i in range(len(points)):
                point = points[i]
                color = sm.to_rgba(indices[i])  # Use the index to determine the color
                ax.scatter(point[0], point[1], color=color, s=40)  # Plot each point as a dot
            
        
        if self.x_min is not None:
            ax.scatter(self.x_min[0], self.x_min[1], color='red', marker='*', s=100, label=f'Global Min Value: {self.v:.4f}')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        if savefig:
            plt.savefig(f'VLSE_{self.problem_type}_{self.seed}.png')

class BOProblemEnv(gym.Env):
    def __init__(self, N = 2, problem_type="threehumpcamel"):
        self.N = N if not restrictions[problem_type] else restrictions[problem_type]
        self.f = None
        self.f_min = float('inf')
        self.action_space = Box(-1, 1, (self.N,), np.float32)
        self.observation_space = Box(-np.inf, np.inf, (self.N+1,), np.float32)

        self.problem_type = problem_type
    
    def step(self, x):
        f_x = self.f.evaluate(x)
        self.f_min = min(self.f_min, f_x.item())
        r_n = self.f.v - self.f_min
        observation = np.concatenate([x, f_x], dtype=np.float32)
        return observation, r_n, False, False, {}
    
    def reset(self, seed=None, options=None):
        super().reset()
        set_seed(seed)

        if self.f is None or (options is not None and options['reset_f'] == True):
            self.f = BOProblem(N = self.N, seed=seed, problem_type=self.problem_type)

            if self.f.v is None:
                _, self.f.v = self.find_minimum()

        # Sample action, reset minimum so far
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
        # print(result_global.fun)
        return result_global.x, result_global.fun

if __name__ == "__main__":
    print('vlse')

   