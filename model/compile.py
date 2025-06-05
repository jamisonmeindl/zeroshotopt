"""
This script is used to preprocess trajectory data to prepare for training. 
The paths and number of functions used for each path are specified in 
the variant dictionary.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import json
import os
import math
import scipy.special


# Configuration for dataset generation
# Sample configuration
variant = {
    # Number of functions for each dataset
    'num_functions': {
        'train_2d_40': 10000,
        'train_3d_40': 10000,
        'valid_2d_40': 5000,
        'valid_3d_40': 5000,
    },
    # Dimension of each dataset
    'dims': {
        'train_2d_40': 2,
        'train_3d_40': 3,           
        'valid_2d_40': 2,
        'valid_3d_40': 3,
    },
    # Lengths to include for each dataset
    'lengths': {
        'train_2d_40': [10,20,30,40],
        'train_3d_40': [10,20,30,40],      
        'valid_2d_40': [10,20,30,40],
        'valid_3d_40': [10,20,30,40],
    },
    # Number of initial random steps
    'num_init_steps': 10,
    # State normalization strategy
    'norm': 'traj_minmax',
}

datasets = variant['num_functions'].keys()

files = []

folder = 'traj_minmax_data'
os.makedirs(folder, exist_ok=True)

total_functions = sum(variant["num_functions"][dataset]*len(variant["lengths"][dataset]) for dataset in datasets)

num_algs = 12
traj_completed = 0

# Iterate through each dataset specified above
for dataset in datasets:
    dataset_path = f'../baselines/data/{dataset}' # Update path for your dataset
    traj_completed_dataset = 0
    token_files = {}
    metadata_files = {}
    weight_files = {}

    # Create output files
    for length in variant["lengths"][dataset]:
        name = f'{dataset.split("/")[0]}_length{length}_dim{variant["dims"][dataset]}_norm{variant["norm"]}_count{variant["num_functions"][dataset]}'
        files.append(f'{folder}/{name}')
        token_shape = (variant["num_functions"][dataset], num_algs, variant['num_init_steps']+length, (variant["dims"][dataset]+1))
        token_memmap_file = f'{folder}/{name}_token_memmap.npy'
        token_memmap = np.memmap(token_memmap_file, dtype='float32', mode='w+', shape=token_shape)
        token_memmap[:] = -100
        token_files[length] = token_memmap

        weight_shape = (variant["num_functions"][dataset], num_algs)
        weight_memmap_file = f'{folder}/{name}_weight_memmap.npy'
        weight_memmap = np.memmap(weight_memmap_file, dtype='float32', mode='w+', shape=weight_shape)
        weight_files[length] = weight_memmap

        metadata_shape = (variant["num_functions"][dataset], num_algs, 3)
        metadata_memmap_file = f'{folder}/{name}_metadata_memmap.npy'
        metadata_memmap = np.memmap(metadata_memmap_file, dtype='float32', mode='w+', shape=metadata_shape)
        metadata_files[length] = metadata_memmap
        metadata_memmap[:] = -100
        
        info = {
            "token_shape": token_shape,
            "weight_shape": weight_shape,
            "metadata_shape": metadata_shape,
        } 

        info.update(variant)

        info_file = f'{folder}/{name}_info.json'
        print(info_file)
        with open(info_file, 'w') as f:
            json.dump(info, f)

    for filename in sorted(os.listdir(dataset_path)):
        if filename.endswith(".pkl"):
            filepath = os.path.join(dataset_path, filename)
            with open(filepath, 'rb') as file:
                trajectories_dataset = pickle.load(file)
                print(f"Loaded {file}")
                
                max_traj = variant['num_functions'][dataset] - traj_completed_dataset
                i = 0
                j = 0
                
                while j < max_traj and i < len(trajectories_dataset):
                    advantages = {}
                    for length in variant["lengths"][dataset]:
                        advantages[length] = np.full((num_algs,), -100, dtype = np.float32)
                    for l in range(num_algs):
                        path = trajectories_dataset[i+l]
                        
                        if 'outputs' in path and isinstance(path['outputs'], np.ndarray):
                            if not np.isnan(path['outputs']).any(): 
                                for length in variant["lengths"][dataset]:
                                    traj_min = path['trajectory_minimum'][variant['num_init_steps']+length-1]
                                    traj_max = path['trajectory_maximum'][variant['num_init_steps']+length-1]

                                    median = np.median(path['outputs'][:variant['num_init_steps']])
                                    if variant['norm'] == 'traj_minmax':                                        
                                        norm_states = (path['outputs'][:variant['num_init_steps']+length] - traj_min) / (traj_max - traj_min + 1e-6)
                                        norm_states = np.clip(norm_states, 0, 1)

                                    norm_reward = (path['outputs'][:variant['num_init_steps']+length].min() - traj_min) / (median - traj_min + 1e-5)
                                    norm_reward = np.clip(norm_reward, 0, 1)
                                    norm_reward = np.sqrt(norm_reward)

                                    mean_norm_reward = (path['trajectory_mean'][variant['num_init_steps']+length-1] - traj_min) / (median - traj_min + 1e-5)
                                    mean_norm_reward = np.clip(mean_norm_reward, 0, 1)
                                    mean_norm_reward = np.sqrt(mean_norm_reward)

                                    norm_reward_difference = mean_norm_reward - norm_reward

                                    actions = path['actions'][:variant['num_init_steps']+length]
                                    
                                    metadata_files[length][j+traj_completed_dataset, l, 0] = norm_reward
                                    metadata_files[length][j+traj_completed_dataset, l, 1] = actions.shape[0]
                                    metadata_files[length][j+traj_completed_dataset, l, 2] = actions.shape[1]
                                    advantages[length][l] = norm_reward_difference

                                    token_files[length][j+traj_completed_dataset, l, :actions.shape[0], :actions.shape[1]] = actions

                                    token_files[length][j+traj_completed_dataset, l, :actions.shape[0], actions.shape[1]] = norm_states
                    for length in variant["lengths"][dataset]:
                        valid_advantages = advantages[length] > -1
                        
                        alpha = 0.5
                        advantages[length][valid_advantages] = (advantages[length][valid_advantages] - advantages[length][valid_advantages].min()) / (advantages[length][valid_advantages].max() - advantages[length][valid_advantages].min()+1e-5)
                        advantages[length][valid_advantages] = scipy.special.softmax(advantages[length][valid_advantages] / alpha)
                        advantages[length][valid_advantages] /= np.sum(advantages[length][valid_advantages])
                        advantages[length][~valid_advantages] = 0
                        weight_files[length][j+traj_completed_dataset, :] = advantages[length]

                    i+=num_algs
                    j+=1
                    
                traj_completed += j*len(variant["lengths"][dataset])
                traj_completed_dataset += j
                print(f"{traj_completed_dataset} out of {variant['num_functions'][dataset]} in dataset")
                print(f"{traj_completed} out of {total_functions} in dataset")
        if traj_completed_dataset == variant['num_functions'][dataset]:
            for length in variant["lengths"][dataset]:
                # Flush changes to disk
                metadata_files[length].flush()
                print(metadata_files[length].shape)
                token_files[length].flush()
                print(token_files[length].shape)    
                weight_files[length].flush()
                print(weight_files[length].shape)    
            break   

print(files)
