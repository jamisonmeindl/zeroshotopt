"""
This script evaluates baseline methods of the specified functions.
This includes BO and classical gradient free optimization methods.
Results are saved to a specified output directory.

Usage:
python test.py \
  --env-id bbob_2d \
  --num-envs 100 \
  --num-proc 48 \
  --output-dir test_100/bbob_2d_40 \
  --num-steps 40
"""
import argparse
import os, sys
os.environ['OMP_NUM_THREADS'] = '1'
from distutils.util import strtobool
import random
import numpy as np
import torch
import gymnasium as gym
import warnings
import csv  
warnings.filterwarnings("ignore")

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import envs
from envs.utils import get_initial_observations, render_env_saved
from envs.valid_envs import vlse_envs
from baselines import algos
from utils import parallel_execute
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="GKLSEnv-v0",
        help="the id of the environment")
    parser.add_argument("--num-init-steps", type=int, default=10,
        help="the number of initial steps before rollout")
    parser.add_argument("--num-steps", type=int, default=20,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--seed", type=int, default=200000000,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--num-envs", type=int, default=100,
        help="number of environments to test on")
    parser.add_argument("--num-proc", type=int, default=4,
        help="number of parallel processes to use")
    parser.add_argument("--output-dir", type=str, default="outputs",
        help="directory to save the outputs")
    
    args = parser.parse_args()

    return args


def test(idx, seed, env_id, num_init_steps, num_steps, keys, visualize =False):
    '''
    Tests individual seed for specified function, steps, and methods
    '''
    random.seed(seed + idx)
    np.random.seed(seed + idx)
    torch.manual_seed(seed + idx)

    if 'vlse' in env_id:
        dim = env_id.split('_')[-1][:-1]
        env_ids = vlse_envs[dim]
        env = gym.make(f'BOProblemEnv-{env_ids[idx%len(env_ids)]}-{dim}D-v0')
    elif 'bbob' in env_id:
        num_bbob = 24
        dim = env_id.split('_')[-1][:-1]
        env = gym.make(f'BBOBEnv-{dim}D-f{(idx%num_bbob)+1}-v0')
    else:
        env = gym.make(env_id)
    
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    obs, _ = env.reset(seed=seed + idx, options={'reset_f': True, 'find_min': True})
    initial_obs = get_initial_observations(gym.vector.SyncVectorEnv([lambda: env]), n_obs=num_init_steps-1, seed=seed + idx).squeeze(1)
    bounds = np.vstack([env.action_space.low, env.action_space.high])
    action_dim = env.action_space.shape[0]
    initial = np.vstack([initial_obs, obs])

    algos_instances = {}
    action_arrs = {}
    output_arrs = {}
    distance_arrs = {}
    
    for key in keys:
        try:
            algo_name = key.split('_')
            if len(algo_name) < 2:
                algos_instances[key] = algos[key](initial, bounds)
            else:
                algos_instances[key] = algos[algo_name[0]](initial, bounds,use_rbf_kernel = (algo_name[1] == 'rbf'))
        except:
            algos_instances[key] = None
        action_arrs[key] = np.empty((num_init_steps + num_steps, action_dim))
        output_arrs[key] = np.empty((num_init_steps + num_steps,))
        distance_arrs[key] = np.empty((num_steps+1,))
 
    j = 0
    for i in range(num_init_steps):
        for key in keys:
            action_arrs[key][j] = initial[j, :action_dim]
            output_arrs[key][j] = initial[j, action_dim]
        j += 1
    for key in keys:
        distance_arrs[key][0] = initial[:,action_dim].min() - env.unwrapped.f.v
        
    while j < num_steps + num_init_steps:
        for key in keys:
            try:
                action, _ = algos_instances[key].propose()

                obs, _, _, _, _ = env.step(action)

              
                algos_instances[key].update(obs)
                action_arrs[key][j] = obs[:action_dim]
                output_arrs[key][j] = obs[action_dim]
                distance_arrs[key][j-num_init_steps+1] = float(output_arrs[key][:j+1].min()) - env.unwrapped.f.v

            except Exception as e:
                action_arrs[key][j] = np.nan  
                output_arrs[key][j] = np.nan
                distance_arrs[key][j-num_init_steps+1] = np.nan

        j += 1

    
    best_algo = min(
        (algo for algo in keys if not np.isnan(output_arrs[algo]).all()), 
        key=lambda algo: np.nanmin(output_arrs[algo])
    )
    env.unwrapped.f.v = min(env.unwrapped.f.v, np.nanmin(output_arrs[best_algo]))

    algo_paths = {}
    for key in keys:
        algo_paths[key] = {
            'seed': seed + idx,
            'actions': action_arrs[key],
            'outputs': output_arrs[key],
            'minimum': env.unwrapped.f.v,
            'algo': key,
            'trajectory_minimum': np.nanmin(output_arrs[best_algo]),
            'env_type': env.unwrapped.f.get_kernel_description(),
        }
        
    if visualize:
        env.unwrapped.f.visualize(init_points = action_arrs[best_algo][:num_init_steps,:2], points = action_arrs[best_algo][num_init_steps:,:2], savefig=True)

    return keys, idx, distance_arrs, algo_paths


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    
    args = parse_args()
    device = 'cpu'
    torch.set_default_device(device)

    torch.backends.cudnn.deterministic = args.torch_deterministic

    
    algorithms = ['gp-ei', 'gp-logei', 'gp-ucb', 'gp-ts', 'gp-jes', 'gp-mes']  
    kernels = ['matern', 'rbf']

    keys = []
    for algo in algorithms:
        for kernel in kernels:
            keys.append(algo+'_'+kernel)
    keys.extend(['random', 'cmaes', 'de', 'pso'])

    distance_to_minimum = np.zeros((len(keys), args.num_envs, args.num_steps + 1))
    all_paths = np.empty((len(keys), args.num_envs), dtype=object) 

    visualize = False

    worker_args = []
    assert args.num_proc >= 1

    for i in range(args.num_envs):
        worker_args.append((i, args.seed, args.env_id, args.num_init_steps, args.num_steps, keys, visualize))

    for algo, idx, arrs, outputs in parallel_execute(test, worker_args, args.num_proc):
        for key in keys:
            distance_to_minimum[keys.index(key)][idx] = arrs[key]
            all_paths[keys.index(key)][idx] = outputs[key]

    os.makedirs(args.output_dir, exist_ok=True)

    for key in keys:

        result_path = os.path.join(args.output_dir, f'{key}_outputs.pkl')
        with open(result_path, 'wb') as f:
            pickle.dump(all_paths[keys.index(key)], f)
        print(f"Output results saved to {result_path}")
