"""
This script generates data using BO methods on the specified GP functions.
Results are saved to a specified output directory.

Usage:
python generate.py \
  --cuda False \
  --result-dir sample/train_2d_40 \
  --num-envs 100 \
  --num-proc 48 \
  --seed 0 \
  --env-id GPEnv-2D-v0 \
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
warnings.filterwarnings("ignore")

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import envs
from envs.utils import get_initial_observations, render_env_saved
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
    parser.add_argument("--seed", type=int, default=0,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--num-envs", type=int, default=100,
        help="number of environments to test on")
    parser.add_argument("--num-proc", type=int, default=4,
        help="number of parallel processes to use")
    parser.add_argument("--result-dir", type=str, default="results",
        help="directory to save the results")
    parser.add_argument("--task-id", type=int, default=0,
        help="task id")
    parser.add_argument("--num-tasks", type=int, default=1,
        help="number of tasks")
    
    args = parser.parse_args()

    return args


def test(idx, seed, env_id, num_init_steps, num_steps, keys, verbose=False):
    random.seed(seed + idx)
    np.random.seed(seed + idx)
    torch.manual_seed(seed + idx)

    env = gym.make(env_id)
    env = gym.wrappers.ClipAction(env)
    action_dim = env.action_space.shape[0]
    
    obs, _ = env.reset(seed=seed + idx, options={'reset_f': True, 'find_min': False})
    initial_obs = get_initial_observations(gym.vector.SyncVectorEnv([lambda: env]), n_obs=num_init_steps-1, seed=seed + idx).squeeze(1)
    bounds = np.vstack([env.action_space.low, env.action_space.high])
    done = False
    j = 0
    initial = np.vstack([initial_obs, obs])

    algos_instances = {}
    action_arrs = {}
    output_arrs = {}
    min_arrs = {}
    distance_arrs = {}
    

    for key in keys:
        try:
            algo_name = key.split('_')
            if len(algo_name) < 2:
                algos_instances[key] = algos[key](initial, bounds)
            else:
                algos_instances[key] = algos[algo_name[0]](initial, bounds, use_rbf_kernel = (algo_name[1] == 'rbf'))
        except:
            algos_instances[key] = None
        action_arrs[key] = np.empty((num_init_steps + num_steps, action_dim))
        output_arrs[key] = np.empty((num_init_steps + num_steps,))
        min_arrs[key] = np.empty((num_init_steps + num_steps,))
        distance_arrs[key] = np.empty((num_init_steps + num_steps,))

    j = 0
    for i in range(num_init_steps):
        for key in keys:
            action_arrs[key][j] = initial[j, :action_dim]
            output_arrs[key][j] = initial[j, action_dim]
        j += 1

    while j < num_steps + num_init_steps:
        for key in keys:
            try:
                action, _ = algos_instances[key].propose()
                obs, _, _, _, _ = env.step(action)
                algos_instances[key].update(obs)
                action_arrs[key][j] = obs[:action_dim]
                output_arrs[key][j] = obs[action_dim]
            except:
                action_arrs[key][j] = np.nan  
                output_arrs[key][j] = np.nan

        j += 1
    
    best_algo = min(
        (algo for algo in keys if not np.isnan(output_arrs[algo]).all()), 
        key=lambda algo: np.nanmin(output_arrs[algo])
    )

    cumulative_min = None
    cumulative_max = None
    traj_totals = None
    count = 0

    for key in keys:
        traj_outputs = output_arrs[key]
        if not np.isnan(traj_outputs).any():
            if cumulative_min is None:
                traj_totals = np.minimum.accumulate(np.where(np.isnan(traj_outputs), np.inf, traj_outputs))
                cumulative_min = np.minimum.accumulate(np.where(np.isnan(traj_outputs), np.inf, traj_outputs))
                cumulative_max = np.maximum.accumulate(np.where(np.isnan(traj_outputs), np.inf, traj_outputs))
            else:
                cumulative_max = np.maximum(cumulative_max, np.maximum.accumulate(np.where(np.isnan(traj_outputs), np.inf, traj_outputs)))
                cumulative_min = np.minimum(cumulative_min, np.minimum.accumulate(np.where(np.isnan(traj_outputs), np.inf, traj_outputs)))
                traj_totals += np.minimum.accumulate(np.where(np.isnan(traj_outputs), np.inf, traj_outputs))
            count += 1
    traj_mean = traj_totals / count

    algo_paths = []
    for key in keys:
        algo_paths.append({
            'seed': seed + idx,
            'actions': action_arrs[key],
            'outputs': output_arrs[key],
            'trajectory_minimum': cumulative_min,
            'trajectory_maximum': cumulative_max,
            'trajectory_mean': traj_mean,
            'algo': key,
            'env_type': env.unwrapped.f.get_kernel_description()
        })

    return idx, algo_paths


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    torch.set_default_device(device)

    torch.backends.cudnn.deterministic = args.torch_deterministic

    num_envs_task = args.num_envs // args.num_tasks

    algorithms = ['gp-ei', 'gp-logei', 'gp-ucb', 'gp-ts', 'gp-jes', 'gp-mes'] 
    kernels = ['matern', 'rbf']

    keys = []
    for algo in algorithms:
        for kernel in kernels:
            keys.append(algo+'_'+kernel)

    all_paths = np.empty(num_envs_task*len(keys), dtype=object) 


    worker_args = []
    assert args.num_proc >= 1
    for i in range(num_envs_task):
        worker_args.append((i, args.seed + num_envs_task*args.task_id, args.env_id, args.num_init_steps, args.num_steps, keys, True if args.num_proc == 1 else False))

    for idx, paths in parallel_execute(test, worker_args, args.num_proc):
        for i in range(len(paths)):
            all_paths[idx*len(paths)+i] = paths[i]


    os.makedirs(args.result_dir, exist_ok=True)

    result_path_all = os.path.join(args.result_dir, f'{args.env_id}_multi_alg_{args.seed + num_envs_task*args.task_id}_{args.seed + num_envs_task*args.task_id + num_envs_task}.pkl')

    with open(result_path_all, 'wb') as f:
        pickle.dump(all_paths, f)
    print(f"All Results saved to {result_path_all}")

