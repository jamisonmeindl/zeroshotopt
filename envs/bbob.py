# blackbox optimization benchmarking https://numbbo.github.io/coco/testsuites/bbob

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import torch
import random
import bbobtorch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
'''
Separable Functions
f1	Sphere Function
f2	Separable Ellipsoidal Function
f3	Rastrigin Function
f4	Büche-Rastrigin Function
f5	Linear Slope
Functions with low or moderate conditioning
f6	Attractive Sector Function
f7	Step Ellipsoidal Function
f8	Rosenbrock Function, original
f9	Rosenbrock Function, rotated
Functions with high conditioning and unimodal
f10	Ellipsoidal Function
f11	Discus Function
f12	Bent Cigar Function
f13	Sharp Ridge Function
f14	Different Powers Function
Multi-modal functions with adequate global structure
f15	Rastrigin Function
f16	Weierstrass Function
f17	Schaffer's F7 Function
f18	Schaffer's F7 Function, moderately ill-conditioned
f19	Composite Griewank-Rosenbrock Function F8F2
Multi-modal functions with weak global structure
f20	Schwefel Function
f21	Gallagher's Gaussian 101-me Peaks Function
f22	Gallagher's Gaussian 21-hi Peaks Function
f23	Katsuura Function
f24	Lunacek bi-Rastrigin Function
'''


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BBOB:
    def __init__(self, N: int = 2, seed = None , fn_type:int = 1):
        # N: num dimensions
        # seed: seed
        # fn_type: int [1,24]
        self.fn_type = fn_type
        mapping = {
            1: bbobtorch.create_f01,
            2: bbobtorch.create_f02,
            3: bbobtorch.create_f03,
            4: bbobtorch.create_f04,
            5: bbobtorch.create_f05,
            6: bbobtorch.create_f06,
            7: bbobtorch.create_f07, 
            8: bbobtorch.create_f08,
            9: bbobtorch.create_f09,
            10: bbobtorch.create_f10,
            11: bbobtorch.create_f11,
            12: bbobtorch.create_f12,
            13: bbobtorch.create_f13,
            14: bbobtorch.create_f14,
            15: bbobtorch.create_f15,
            16: bbobtorch.create_f16,
            17: bbobtorch.create_f17,
            18: bbobtorch.create_f18,
            19: bbobtorch.create_f19,
            20: bbobtorch.create_f20,
            21: bbobtorch.create_f21,
            22: bbobtorch.create_f22,
            23: bbobtorch.create_f23,
            24: bbobtorch.create_f24,
        }

        self.names = {
            1: 'Sphere Function',
            2: 'Separable Ellipsoidal Function',
            3: 'Rastrigin Function',
            4:'Büche-Rastrigin Function',
            5: 'Linear Slope',
            6: 'Attractive Sector Function',
            7: 'Step Ellipsoidal Function',
            8: 'Rosenbrock Function, original',
            9: 'Rosenbrock Function, rotated',
            10: 'Ellipsoidal Function',
            11: 'Discus Function',
            12: 'Bent Cigar Function',
            13: 'Sharp Ridge Function',
            14: 'Different Powers Function',
            15: 'Rastrigin Function',
            16: 'Weierstrass Function',
            17: 'Schaffers F7 Function',
            18: 'Schaffers F7 Function, moderately ill-conditioned',
            19: 'Composite Griewank-Rosenbrock Function F8F2',
            20: 'Schwefel Function',
            21: 'Gallaghers Gaussian 101-me Peaks Function',
            22: 'Gallaghers Gaussian 21-hi Peaks Function f23	Katsuura Function',
            23: 'Katsuura Function',
            24: 'Lunacek bi-Rastrigin Function',
        }
        self.device = 'cpu'
        self.seed = seed if seed else np.random.randint(0, 1000000)
        self.fn = mapping[fn_type](N, dev = self.device, seed=self.seed)
        self.x_min = -5
        self.x_max = 5
        self.M = self.fn.x_opt
        self.v = self.fn.f_opt.item() 
        self.N = N
    
    def normalize(self,x):
        '''
        convert range [x_min, x_max] to [-1,1]
        '''
        normalized_x = (x - self.x_min)/(self.x_max-self.x_min)*2 - 1
        return normalized_x

    def denormalize(self,x):
        '''
        convert range to [-1,1] to [x_min, x_max]
        '''
        denormalize_x = (x + 1)/2 *(self.x_max-self.x_min) + self.x_min
        return denormalize_x 

    def evaluate(self, x):
        '''
        assume x values are in range [-1,1], scales them into range [x_min,x_max] and evaluates it
        '''
        # needs to be a float for bbobtorch to operate with
        x = torch.tensor(x).to(self.device).float().unsqueeze(0)  #add batch dimension for bbob function eval
        # scale x to -5,5 range
        y = self.fn(self.denormalize(x)).cpu() # is a float
        return y
    def get_kernel_description(self):
        return self.names[self.fn_type]


    def grid(self, dim):
        assert self.N == 2, 'must be 2 dimensional'
        x = np.linspace(-1, 1, dim)
        y = np.linspace(-1, 1, dim)
        X, Y = np.meshgrid(x, y)
        with torch.no_grad():
            Z = np.vectorize(lambda x, y: self.evaluate(np.array([x, y])))(X, Y)
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
            
            
        if self.M is not None:
            global_min = self.normalize(self.M)
            ax.scatter(global_min[0], global_min[1], color='red', marker='*', s=600, label=f'Global Min Value: {self.v:.4f}')
        ax.scatter([], [], color='none', label=f'Score: {score:.4f}')

        # ax.set_xticks([])
        # ax.set_yticks([])
        plt.legend()
        plt.tight_layout()

        if savefig:
            plt.savefig(f'{save_path}/BBOB{self.fn_type}_{self.seed}.png')
    


    
    
class BBOBEnv(gym.Env):
    def __init__(self, fn_type = 1, N = 2):
        self.fn_type = fn_type
        self.N = N
        self.f = BBOB(fn_type = fn_type, N = N) 
        self.action_space = Box(-1, 1, (N, ), np.float32) #
        self.observation_space = Box(-np.inf, np.inf, (N+1,), np.float32)
        self.f_min = float('inf')
    
    def step(self, x):
        f_x = self.f.evaluate(x)
        self.f_min = min(self.f_min, f_x.item())
        r_n = self.f.v - self.f_min 
        observation = np.concatenate([x, f_x], dtype=np.float32)
        return observation, r_n, False, False, {}
    
    def reset(self, seed=None, options=None):
        super().reset()
        set_seed(seed)
        if options is not None and options['reset_f'] == True:
            self.f = BBOB(N = self.N, seed=seed, fn_type=self.fn_type)
        self.action_space.seed(seed)
        action = self.action_space.sample()
        f_action = self.f.evaluate(action)
        self.f_min = f_action.item()
        observation = np.concatenate([action, f_action],dtype=np.float32)
        return observation, {}

    def seed(self, seed):
        set_seed(seed)


if __name__ == "__main__":
    N = 6
    env = BBOBEnv(N=N, fn_type=2)
    env.reset(seed=0)

    # # testing env
    epsilon = 1e-3
    print('------------')
    print('fn type: ', env.f.fn_type)
    print('step: ', env.step(np.array([1]*N)))
    print('optimal step: ', env.step(env.f.normalize(env.f.fn.x_opt.cpu().numpy())))
    print('near optimal step: ', env.step(env.f.normalize(env.f.fn.x_opt.cpu().numpy() - epsilon)))
    print('optimal val: ', env.f.v)

    print('-----seed 0-------')
    env.reset(seed=0, options={'reset_f':True})
    print('fn type: ', env.f.fn_type)
    print('step: ', env.step(np.array([0.5]*N)))
    print('optimal step: ', env.step(env.f.normalize(env.f.fn.x_opt.cpu().numpy())))
    print('near optimal step: ', env.step(env.f.normalize(env.f.fn.x_opt.cpu().numpy() - epsilon)))
    print('optimal val: ', env.f.v)
