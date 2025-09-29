"""
Example usage:

CUDA_VISIBLE_DEVICES=0 python test_state_kv.py  \
    --model-path ZeroShotOptState/ckpt.pt \
    --num-envs 100 \
    --env-id bbob_2d \
    --num-steps 40 \
    --length-type adaptive \
    --norm-type traj_minmax_scaled_high \
    --sampling top_p \
    --input-dir '../baselines/test_100/bbob_2d_40' \
    --output-dir '../baselines/test_100/bbob_2d_40' \
    --ev-style linear \
    --batch-size 4 \
    --num-action-bins 2000

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

from envs.utils import get_initial_observations
from envs.valid_envs import vlse_envs
from model_kv import GPTConfig, GPT
import pickle
from tqdm import tqdm


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
    parser.add_argument("--result-dir", type=str, default="results",
        help="directory to save the results")
    parser.add_argument("--input-dir", type=str, default="inputs",
        help="directory to use as the inputs")
    parser.add_argument("--output-dir", type=str, default="outputs",
        help="directory to save the outputs")
    parser.add_argument("--time-dir", type=str, default="times",
        help="directory to save the times")
    parser.add_argument("--model-path", type=str, required=True,
        help="saved model path")
    parser.add_argument("--visualize", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="visualize") 
    parser.add_argument("--norm-type", type=str, default="pred_min",
        help="normalization method")   
    parser.add_argument("--length-type", type=str, default="adaptive",
        help="normalization method")   
    parser.add_argument("--sampling", type=str, default="top_p",
        help="sampling method")  
    parser.add_argument("--temperature", type=float, default=1.0,
        help="sampling temperature")
    parser.add_argument("--ev-style", type=str, default="linear",
        help="ev method")  
    parser.add_argument("--batch-size", type=int, default=64,
        help="batch size")
    parser.add_argument("--num-action-bins", type=int, default=2000,
        help="action bins")

    args = parser.parse_args()

    return args

def test(idx, seed, env_id, num_init_steps, num_steps, model,device, norm_type, length_type, sampling, overall_min_achieved, min_achieved, max_achieved, temperature, ev_style, batch_size, num_action_bins, visualize=False, name=''):
    # if verbose:
    #     print(f"Testing environment {idx+1}")
    random.seed(seed + idx)
    np.random.seed(seed + idx)
    torch.manual_seed(seed + idx)
    # print(seed+idx)
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
    obs, _ = env.reset(seed=seed + idx, options = {'reset_f': True, 'find_min': False})
    env.unwrapped.f.v = overall_min_achieved[seed+idx]
    initial_obs = get_initial_observations(gym.vector.SyncVectorEnv([lambda: env]), n_obs=num_init_steps-1, seed=seed + idx).squeeze(1)
    action_dim = env.action_space.shape[0]
    initial = np.vstack([initial_obs, obs])
  
    action_arr = np.empty((num_init_steps + num_steps, action_dim))
    output_arr = np.empty((num_init_steps + num_steps,))
    distance_arr = np.empty((num_steps+1,))

    variant = {
        'num_action_bins': num_action_bins,
        'max_steps': 100,
        'max_dim': 50
    }

    # Fill initial actions and outputs
    j = 0
    for i in range(num_init_steps):
        action_arr[j] = initial[j, :action_dim]
        output_arr[j] = initial[j, action_dim]
        j += 1
    
    distance_arr[0] = initial[:,action_dim].min() - env.f.v
    obs_history = torch.tensor(initial).unsqueeze(0)

    action_bins = torch.tensor(np.linspace(0, 1, variant['num_action_bins']-1))
    midpoints = ((action_bins[:-1] + action_bins[1:]) / 2.0).to(device)

    with torch.no_grad():
        target = env.unwrapped.f.v
        while j < num_steps + num_init_steps:
            history = obs_history.clone()
            actions = history[:,:,:action_dim]
            states = history[:,:,action_dim:action_dim+1]
            maximum = np.max(states[-1,:,-1].numpy())
            minimum = np.min(states[-1,:,-1].numpy())

            
            if norm_type == 'traj_minmax':
                norm_states = (states - minimum) / (maximum - minimum + 1e-5)
                upper_scaled_change = 0.1
                lower_scaled_change = 0.1
                norm_states = (norm_states * (1-(upper_scaled_change + lower_scaled_change))) + lower_scaled_change
            elif norm_type == 'traj_minmax_scaled':
                norm_states = (states - minimum) / (maximum - minimum + 1e-5)
                upper_scaled_change = 0.05 + 0.05*((num_init_steps+num_steps-j)/num_steps)
                lower_scaled_change = 0.1 + 0.15*((num_init_steps+num_steps-j)/num_steps)
                norm_states = (norm_states * (1-(upper_scaled_change + lower_scaled_change))) + lower_scaled_change
            elif norm_type == 'traj_minmax_scaled_high':
                norm_states = (states - minimum) / (maximum - minimum + 1e-5)
                upper_scaled_change = 0.05 + 0.05*((num_init_steps+num_steps-j)/num_steps)
                lower_scaled_change = 0.1 + 0.4*((num_init_steps+num_steps-j)/num_steps)
                norm_states = (norm_states * (1-(upper_scaled_change + lower_scaled_change))) + lower_scaled_change

                
            if torch.isnan(norm_states).any():
                print("Found NaN in norm_states!")
            min_state = norm_states.min()    
            min_bucket = torch.bucketize(min_state, action_bins, right=False)
            norm_states = np.clip(norm_states, 0, 1)
            actions = (actions + 1)/2

            
            combined_data = np.ravel(np.column_stack((*np.split(actions.squeeze(0).numpy(), actions.shape[-1], axis=-1), norm_states.squeeze(0).numpy())))
            steps = np.arange(actions.shape[1])
            step_tokens = np.repeat(steps, actions.shape[2]+1)

            dims = np.arange(actions.shape[2]+1)[::-1]
            dim_tokens = np.tile(dims, actions.shape[1])
            if length_type == 'adaptive':
                extra_values = np.array([0, (num_steps+num_init_steps)/100])
            elif length_type == 'short':
                extra_values = np.array([0, (20)/100])
            
            extra_step_values = np.array([variant['max_steps'], variant['max_steps']+1])
            extra_dim_values = np.array([variant['max_dim']+1, variant['max_dim']+2])

            B = batch_size
            minibatch_size = min(batch_size,8)
            
            full_samples = torch.empty((B,action_dim)).to(device)
            combined_data = torch.tensor(np.concatenate([extra_values, combined_data], axis=0), dtype=torch.float32).unsqueeze(0).to(device)
            step_tokens = torch.tensor(np.concatenate([extra_step_values, step_tokens], axis=0)).unsqueeze(0).to(device)
            dim_tokens = torch.tensor(np.concatenate([extra_dim_values, dim_tokens], axis=0)).unsqueeze(0).to(device)
            logits, _, original_kv = model(
                combined_data, step_tokens, dim_tokens)
            
            logits = logits[:, -1].cpu().expand(minibatch_size, -1)
            
            expanded_kv = list((
                k.repeat_interleave(minibatch_size, dim=0),
                v.repeat_interleave(minibatch_size, dim=0)
            ) for (k,v) in original_kv)
                        
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)

            expected_values = []
            for l in range(0,B,minibatch_size):
                batch_kv = expanded_kv.copy()
                samples = None
                steps = None
                dims = None
                for k in range(action_dim):
                
                    if k > 0:
                        logits, _ , batch_kv = model(samples, steps, dims, past_kv=batch_kv)
                        logits = logits[:, -1].cpu()
                        logits = logits / temperature
                        probs = torch.softmax(logits, dim=-1)
                    
                    p = 0.9
                    num_bins = variant['num_action_bins']
                    samples = []
                    for i in range(minibatch_size):
                        sorted_probs, sorted_indices = torch.sort(probs[i], descending=True)
                        
                        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                        cutoff = torch.where(cumulative_probs > p)[0]
                        cutoff_index = cutoff[0].item() if len(cutoff) > 0 else sorted_probs.size(0) - 1

                        top_p_probs = sorted_probs[:cutoff_index + 1]
                        top_p_indices = sorted_indices[:cutoff_index + 1]
                        top_p_probs = top_p_probs / top_p_probs.sum()

                        sample_idx = top_p_indices[torch.multinomial(top_p_probs, num_samples=1)].item()

                        if sample_idx == 0:
                            sample = torch.tensor([0.0]).unsqueeze(0)
                        elif sample_idx == num_bins - 1:
                            sample = torch.tensor([1.0]).unsqueeze(0)
                        else:
                            lower_edge = action_bins[sample_idx - 1]
                            upper_edge = action_bins[sample_idx]
                            sample = torch.tensor([(lower_edge + upper_edge) / 2.0]).unsqueeze(0)

                        samples.append(sample)
                    
                    # final tensor: [B, 1]
                    samples = torch.cat(samples, dim=0).to(device=device, dtype=torch.float32)
                    
                    full_samples[l:l+minibatch_size,k] = samples.squeeze(-1)

                    steps = torch.tensor([j]).expand(minibatch_size,-1).to(device)

                    dims = torch.tensor([action_dim-k]).expand(minibatch_size,-1).to(device)
                logits, _ , _ = model(samples, steps, dims, past_kv=batch_kv)
                
                logits_last = logits[:, -1]
                probs = torch.softmax(logits_last, dim=-1)
                
                weights = min_state - midpoints

                if ev_style == 'square':
                    weights = weights ** 2
              
                if min_bucket == 0:
                    ev = probs[:, 0]
                else:
                    ev = (probs[:, 1:min_bucket] * weights[:min_bucket-1]).sum(dim=1) + probs[:, 0]
                expected_values.extend(ev.detach().cpu().tolist())
            
            expected_values = torch.tensor(expected_values)
            max_index = torch.argmax(expected_values).item()
            
            action = (2*full_samples[max_index,:].cpu()-1).numpy()
            action_arr[j] = action  

            obs, _, _, _, _ = env.step(action)
            obs = torch.Tensor(obs)
            obs_history = torch.cat([obs_history, obs.unsqueeze(0).unsqueeze(0)], dim=1)
            distance_arr[j+1 - num_init_steps] = float(obs_history[:, :, action_dim].min().cpu().numpy()) - env.unwrapped.f.v
            j += 1
        
        outputs = obs_history[0,:, action_dim].numpy()
        env.unwrapped.f.v = min(env.unwrapped.f.v, np.nanmin(outputs))
        med = np.median(outputs[:num_init_steps])
        best = np.min(outputs)
        denom = med - env.unwrapped.f.v
        numer = med - best

        score = 1.0 if denom == 0 else numer / denom
        output_path = {
            'outputs': outputs,
            'actions': obs_history[0,:,:action_dim].numpy(),
            'minimum': env.unwrapped.f.v,
            'model_minimum':best,
            'score': score, 
            'env_type': env.f.get_kernel_description(),
            'algo': 'model',
            'seed': seed + idx,
        }
        if visualize:
            if action_dim == 2:
                env.unwrapped.f.visualize(init_points = obs_history.numpy()[0,:num_init_steps,:2], points = obs_history.numpy()[0,num_init_steps:,:2], savefig=True, save_path=name, score = score)
            print(output_path)

    return idx, distance_arr, output_path


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    torch.backends.cudnn.deterministic = args.torch_deterministic

    ckpt_path = args.model_path
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint['model_args']
    
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model Parameters: {pytorch_total_params}')
    model.eval()
    model = model.to(device)
    model = torch.compile(model) 

    distance_to_minimum = np.zeros((args.num_envs, args.num_steps + 1))
    output_paths = np.empty((args.num_envs,), dtype=object)

    files = [
        'gp-ei_matern_outputs.pkl', 'gp-ei_rbf_outputs.pkl',
        'gp-jes_matern_outputs.pkl', 'gp-jes_rbf_outputs.pkl',
        'gp-logei_matern_outputs.pkl', 'gp-logei_rbf_outputs.pkl',
        'gp-mes_matern_outputs.pkl', 'gp-mes_rbf_outputs.pkl',
        'gp-ts_matern_outputs.pkl', 'gp-ts_rbf_outputs.pkl',
        'gp-ucb_matern_outputs.pkl', 'gp-ucb_rbf_outputs.pkl']

    min_achieved = {}
    overall_min_achieved = {}

    max_achieved = {}


    for filename in os.listdir(args.input_dir):
        full_path = os.path.join(args.input_dir, filename)
        if full_path.split('/')[-1] in files:
            with open(full_path, "rb") as f:
                data = pickle.load(f)
                for traj in data:
                    if traj['seed'] in min_achieved.keys():
                        min_achieved[traj['seed']] = min(min_achieved[traj['seed']], np.nanmin(traj['outputs'][:args.num_init_steps+args.num_steps]))
                        overall_min_achieved[traj['seed']] = min(overall_min_achieved[traj['seed']], np.nanmin(traj['outputs']))
                        max_achieved[traj['seed']] = max(max_achieved[traj['seed']], np.nanmax(traj['outputs'][:args.num_init_steps+args.num_steps]))
                    else:
                        min_achieved[traj['seed']] = np.nanmin(traj['outputs'][:args.num_init_steps+args.num_steps])
                        overall_min_achieved[traj['seed']] = min(np.nanmin(traj['outputs']), traj['minimum'])
                        max_achieved[traj['seed']] = np.nanmax(traj['outputs'][:args.num_init_steps+args.num_steps])
                
    name = f'{args.model_path.split("/")[0]}_{args.norm_type}_{args.length_type}_{args.sampling}_{args.temperature}_{args.ev_style}_{args.batch_size}_state'
    os.makedirs(name, exist_ok=True)
    worker_args = []
    assert args.num_proc >= 1
    for i in range(args.num_envs):
        worker_args.append((i, args.seed, args.env_id, args.num_init_steps, args.num_steps, model, device, args.norm_type, args.length_type, args.sampling, overall_min_achieved, min_achieved, max_achieved, args.temperature, args.ev_style, args.batch_size, args.num_action_bins, args.visualize, name))

    total_score = 0

    for args_tuple in tqdm(worker_args, desc="Evaluating"):
        idx, distance_arr, path = test(*args_tuple)
        distance_to_minimum[idx] = distance_arr
        output_paths[idx] = path
        total_score += path['score']

    os.makedirs(args.output_dir, exist_ok=True)

    print(total_score/args.num_envs)

    print(distance_to_minimum[:,-1].mean())

    output_path = os.path.join(args.output_dir, f'{name}_outputs.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(output_paths, f)
    print(f"Output results saved to {output_path}")
