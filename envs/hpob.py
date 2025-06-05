'''
HPO-B
https://arxiv.org/pdf/2106.06257
https://github.com/releaunifreiburg/HPO-B

To use, download data files from above links
'''
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import xgboost as xgb
import json
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


SEARCH_SPACES = ['4796', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970', '5971', '6766', '6767', '6794', '7607', '7609', '5889']
SAVED_SURROGATES_DIR = "../envs/saved-surrogates/"
SURROGATES_FILE = SAVED_SURROGATES_DIR+"summary-stats.json"
META_TRAIN_PATH = '../envs/hpob-data/meta-train-dataset.json'
META_TEST_PATH = '../envs/hpob-data/meta-test-dataset.json'
META_VALIDATION_PATH = '../envs/hpob-data/meta-validation-dataset.json'


dims = {
    '4796': 3,
    '5527': 8,
    '5636': 6,
    '5859': 6,
    '5860': 2,
    '5891': 8,
    '5906': 16,
    '5965': 10,
    '5970': 2,
    '5971': 16,
    '6766': 2,
    '6767': 18,
    '6794': 10,
    '7607': 9,
    '7609': 9,
    '5889': 6
}



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

class HPOB:
    def __init__(self, mode, search_space_id, seed =None):
        '''
        search spaces: model with respective hyperparameters to optimize
        '''
        self.search_space_id = search_space_id
        self.seed = seed
        self.v = -1 


        if mode == 'train':
            with open(META_TRAIN_PATH, "rb") as f:
                self.dataset_ids = list(json.load(f)[self.search_space_id].keys())
        elif mode == 'test':
            with open(META_TEST_PATH, "rb") as f:
                self.dataset_ids = list(json.load(f)[self.search_space_id].keys())

        with open(SURROGATES_FILE) as f:
            self.surrogates_stats = json.load(f)

        self.dataset_id = random.choice(self.dataset_ids)

        self.surrogate_name='surrogate-'+self.search_space_id+'-'+self.dataset_id
        self.bst_surrogate = xgb.Booster()
        self.bst_surrogate.load_model(SAVED_SURROGATES_DIR+self.surrogate_name+'.json')

        self.y_min = self.surrogates_stats[self.surrogate_name]["y_min"]
        self.y_max = self.surrogates_stats[self.surrogate_name]["y_max"]
        
        self.search_space_dim = dims[self.search_space_id]
        

    def get_kernel_description(self):
        return f'{self.search_space_id}_{self.dataset_id}_{self.seed}'

    def normalize_y(self,y):
        '''
        convert range [y_min, y_max] to [0,1]
        '''
        return np.clip((y-self.y_min)/(self.y_max-self.y_min + 1e-12),0,1)
    
    def normalize_x(self,x):
        '''
        converts [-1,1] to [0,1]
        '''
        return (x+1)/2
        
    def evaluate(self, x):

        # print(f'step 1: x: {x} ')
        x = self.normalize_x(x)
        x_q = xgb.DMatrix(np.expand_dims(x,axis=0))
        # print(f'step 2: x_q: {x_q} ')
        new_y = self.bst_surrogate.predict(x_q)
        # print(f'step 3: y: {new_y} ')
        norm_y = self.normalize_y(new_y)
        # print(norm_y)
        return -norm_y

    def visualize(self, ax=None, X=None, Y=None, Z=None, init_points=None, points=None, dynamic=False, savefig=False):
        
       
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
 
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        if savefig:
            plt.savefig(f'HPOB_{self.get_kernel_description()}.png')
    
    
    def visualize_space(self):
        assert self.search_space_dim == 2, 'must be 2d'
        x = np.linspace(-1, 1, 200)
        y = np.linspace(-1, 1, 200)
        X, Y = np.meshgrid(x, y)
        if self.Z is None:
            with tqdm(total=np.prod(X.shape).item()) as pbar:
                self.Z = np.vectorize(lambda x, y: self.evaluate([x, y],pbar))(X, Y)

        fig, ax = plt.subplots()
        contour = ax.contourf(X, Y, self.Z, 200)

        ax.set_title(f"HPO-B {self.search_space_id}-{self.dataset_id}")
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        plt.savefig(f'hpob_{self.search_space_id}_{self.dataset_id}.png')

    def get_spaces(self):
        return list(self.datasets.keys())

    def grid(self, dim):
        x = np.linspace(-1, 1, dim)
        y = np.linspace(-1, 1, dim)
        X, Y = np.meshgrid(x, y)
        with torch.no_grad():
            Z = np.vectorize(lambda x, y: self.evaluate((torch.tensor([x, y], requires_grad=False, dtype=torch.float64))))(X, Y)
        return X,Y,Z
    
    def get_datasets(self, search_space):
        if search_space not in self.datasets:
            valid_spaces = ' '.join(self.get_spaces())
            raise KeyError(f'invalid search space, use one of the following {valid_spaces}' )
        else:
            return self.datasets[search_space]
        

class HPOBEnv(gym.Env):
    def __init__(self, mode, search_space_id='5970', dataset_id=None):
        self.search_space_id = search_space_id
        self.mode = mode
        self.f = HPOB(mode =self.mode,seed=None, search_space_id=self.search_space_id)
        self.action_space = Box(-1, 1, (self.f.search_space_dim, ), np.float32)
        self.observation_space = Box(-np.inf, np.inf, (self.f.search_space_dim+1,), np.float32)
        self.f_min = float('inf')
        
    def get_datasets(self):
        return list(self.meta_test_data[self.search_space_id].keys())

    def step(self, x):
        f_x = self.f.evaluate(x)
        self.f_min = min(self.f_min, f_x.item())

        r_n = -np.log(max(f_x - self.f.v,0) + 1e-8).item() 
        observation = np.concatenate((x, f_x), dtype=np.float32)

        return observation, r_n, False, False, {}
    
    def reset(self, seed=None, options=None):
        super().reset()
        set_seed(seed)
        if options is not None and options['reset_f'] == True:
            self.f = HPOB(mode=self.mode,seed=seed, search_space_id=self.search_space_id)
        self.action_space.seed(seed)
        action = self.action_space.sample()
        f_action = self.f.evaluate(action)
        self.f_min = f_action.item()
        observation = np.concatenate((action, f_action), dtype=np.float32)
        return observation, {}


