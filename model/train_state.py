"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ train_state.py --config simple_model.yaml

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --nproc-per-node=4 train_state.py --config simple_model.yaml
"""

import os
import time
import math
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

from datetime import timedelta

from model import GPTConfig, GPT
from torch.utils.data import Dataset, DataLoader
import json
import sys
project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import yaml
import argparse

parser = argparse.ArgumentParser(description='Training script with configurable YAML file')
parser.add_argument('--config', type=str, default='full_model.yaml', help='Path to the YAML config file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

class TrajectoryDataset(Dataset):
    def __init__(self, paths, augment, uniform):
        self.paths = paths
        self.action_bins = torch.tensor(np.linspace(0, 1, config['num_action_bins'] - 1))
        self.augment = augment
        self.uniform = uniform
        self.infos = {}
        self.token_shapes = {}
        self.metadata_shapes = {}
        self.weight_shapes = {}

        self.data_paths = {}
        self.metadata_paths = {}
        self.weight_paths = {}

        self.num_functions = 0
        self.sizes = []
        for path in paths:
            with open(f'{path}_info.json', 'r') as f:
                info = json.load(f)
            self.infos[path] = info
            self.token_shapes[path] = tuple(info['token_shape'])
            self.metadata_shapes[path] = tuple(info['metadata_shape'])
            self.weight_shapes[path] = tuple(info['weight_shape'])
            self.data_paths[path] = path + '_token_memmap.npy'
            self.metadata_paths[path] = path + '_metadata_memmap.npy'
            self.weight_paths[path] = path + '_weight_memmap.npy'

            
            self.num_functions += self.token_shapes[path][0]
            self.sizes.append(self.token_shapes[path][0])

        self.cumulative_sizes = np.cumsum(self.sizes)
        self.offsets = np.roll(self.cumulative_sizes, shift=1)
        self.offsets[0] = 0
        self.offset_dict = {name: (offset, size) for name, offset, size in zip(self.paths, self.offsets, self.sizes)}

        print(f'Uniform: {self.uniform}')
        print(f'Augment: {self.augment}')
        print(f'{self.num_functions} Trajectories Loaded')

        print(f'{paths} Initialized')

    def __len__(self):
        return self.num_functions
    
    def global_to_local(self,global_index):
        dataset_idx = np.searchsorted(self.cumulative_sizes, global_index, side="right")
        dataset_name = self.paths[dataset_idx]
        local_idx = global_index - self.offsets[dataset_idx]
        return dataset_name, local_idx

    def __getitem__(self, idx):
        """Retrieve an item with alias sampling."""
        dataset_name, local_idx = self.global_to_local(idx)

        # Load data
        token_data = np.memmap(self.data_paths[dataset_name], dtype='float32', mode='r', shape=self.token_shapes[dataset_name])
        full_metadata = np.memmap(self.metadata_paths[dataset_name], dtype='float32', mode='r', shape=self.metadata_shapes[dataset_name])
        full_weights = np.memmap(self.weight_paths[dataset_name], dtype='float32', mode='r', shape=self.weight_shapes[dataset_name])


        weights = torch.tensor(full_weights[local_idx], dtype=torch.float32)

        if not self.uniform:
            sampled_index = torch.multinomial(weights, num_samples=1, replacement=False).item()
        else:
            nonzero_indices = torch.where(weights > 0)[0]
            sampled_index = nonzero_indices[
                torch.randint(0, nonzero_indices.size(0), (1,)).item()
            ]

        tokens = torch.tensor(token_data[local_idx, sampled_index].copy(), dtype=torch.float32)
        metadata = torch.tensor(full_metadata[local_idx, sampled_index].copy(), dtype=torch.float32)

        dim = int(metadata[2])
        step_count = int(metadata[1])

        if self.augment:
            chosen_orientation = torch.randperm(dim)
            tokens[:, :dim] = tokens[:, chosen_orientation]

            flip_vector = 2 * torch.randint(0, 2, (dim,)) - 1
            tokens[:, :dim] = tokens[:, :dim] * flip_vector
            

        target_actions = torch.full((step_count, dim + 1), -100, dtype=torch.long)
        
        tokens[:, :dim] = (tokens[:, :dim] + 1) / 2
        vals = tokens[self.infos[dataset_name]['num_init_steps']:step_count, :dim+1].contiguous()

        bucketized = torch.bucketize(vals, self.action_bins, right=False)
        bucketized = bucketized + 1

        epsilon = 1e-6
        bucketized[vals <= 0.0 + epsilon] = 0
        bucketized[vals >= 1.0 - epsilon] = config['num_action_bins'] - 1
        target_actions[self.infos[dataset_name]['num_init_steps']:step_count, :dim+1] = bucketized


        flattened_targets = target_actions[:, :dim + 1].flatten()
        target_tokens = torch.full((config['train_length'] * (config['train_dim'] + 1),), -100, dtype=torch.long)
        target_tokens[:flattened_targets.shape[0]] = flattened_targets

        
        flattened_tokens = tokens[:, :dim + 1].flatten()
        flattened_tokens = torch.cat([metadata[0:1], metadata[1:2] / config['max_step'], flattened_tokens])

        full_tokens = torch.full((2 + (config['train_dim'] + 1) * config['train_length'],), 0, dtype=torch.float32)
        full_tokens[:flattened_tokens.shape[0]] = flattened_tokens

        dims = torch.full((2 + (config['train_dim'] + 1) * config['train_length'],), 0, dtype=torch.long)
        start_dims = torch.arange(dim, -1, -1)
        dim_tokens = start_dims.repeat(step_count)
        extra_dim_values = torch.tensor([config['max_dim'] + 1, config['max_dim'] + 2], dtype=torch.long)
        dim_tokens = torch.cat([extra_dim_values, dim_tokens])
        dims[:dim_tokens.shape[0]] = dim_tokens

        steps = torch.full((2 + (config['train_dim'] + 1) * config['train_length'],), 0, dtype=torch.long)
        start_steps = torch.arange(step_count)
        step_tokens = np.repeat(start_steps, dim + 1)
        extra_step_values = torch.tensor([config['max_step'], config['max_step'] + 1], dtype=torch.long)
        step_tokens = torch.cat([extra_step_values, step_tokens])
        steps[:step_tokens.shape[0]] = step_tokens

        return full_tokens, steps, dims, target_tokens

out_dir = f'agent_{config["train_dim"]}D_{config["train_length"]}length_{config["sample_type"]}_{config["norm_type"]}_{config["model_size"]}_{config["num_action_bins"]}bins'
print(out_dir)

wandb_run_name = out_dir 

with open(f'{config["paths"][0]}_info.json', 'r') as f:
    info = json.load(f)


block_size = config['max_dim']*config['max_step']+2
print(block_size)
input_embd_dim = config['num_action_bins'] + 1
print(input_embd_dim)
# model

if config['model_size'] == 'extrasmall':
    n_layer = 6
    n_head = 8
    n_embd = 256
elif config['model_size'] == 'small':
    n_layer = 12
    n_head = 8
    n_embd = 512
elif config['model_size'] == 'medium':
    n_layer = 16
    n_head = 16
    n_embd = 768
elif config['model_size'] == 'large':
    n_layer = 16
    n_head = 16
    n_embd = 1024


print(f'Learning Rate: {config["learning_rate"]}')
print(f'Weight Decay: {config["weight_decay"]}')

lr_decay_iters = config['max_iters'] # should be ~= max_iters per Chinchilla
min_lr = config['learning_rate'] / 10 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

device = 'cuda' 
dtype = config['dtype'] 
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=config['backend'],timeout=timedelta(minutes=120))
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert config['gradient_accumulation_steps'] % ddp_world_size == 0
    config['gradient_accumulation_steps'] //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = config['gradient_accumulation_steps'] * ddp_world_size * config['batch_size'] * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")


if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

valid_datasets = []
if config['sample_type'] == 'weighted':
    dataset = TrajectoryDataset(config['paths'], uniform = False, augment = True)
    for path in config['valid_paths']:
        valid_datasets.append(TrajectoryDataset([path], uniform = False, augment = False))
else:
    dataset = TrajectoryDataset(config['paths'], uniform = True, augment = True)
    for path in config['valid_paths']:
        valid_datasets.append(TrajectoryDataset([path], uniform = True, augment = False))


epoch = 0 


if ddp:
    sampler = DistributedSampler(dataset, num_replicas=ddp_world_size, rank=ddp_rank,shuffle=True,drop_last=True)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=sampler, num_workers=8, pin_memory=True, persistent_workers=True)
    data_loader.sampler.set_epoch(epoch)
else:
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)

valid_data_loaders = []
for valid_dataset in valid_datasets:
    valid_data_loaders.append(DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True))

data_iter = iter(data_loader)
valid_data_iters = []

for valid_data_loader in valid_data_loaders:
    valid_data_iters.append(iter(valid_data_loader))

def get_valid_batch(valid_data_iter, valid_data_loader):
    try:
        trajectories, steps, dims, targets = next(valid_data_iter)
    except StopIteration:
        valid_data_iter = iter(valid_data_loader)
        trajectories, steps, dims, targets = next(valid_data_iter)
    trajectories, steps, dims, targets  = trajectories.to(device, non_blocking=True), steps.to(device, non_blocking=True), dims.to(device, non_blocking=True), targets.to(device, non_blocking=True)
    return trajectories, steps, dims, targets, valid_data_iter

def get_batch(data_iter, data_loader, split, epoch):
    try:
        trajectories, steps, dims, targets = next(data_iter)
    except StopIteration:
        epoch += 1
        if ddp:
            print(f"Rank {torch.distributed.get_rank()} starting epoch {epoch}")
        else:
            print(f"Starting epoch {epoch}")
        if hasattr(data_loader.sampler, 'set_epoch'):
            data_loader.sampler.set_epoch(epoch)
        data_iter = iter(data_loader)
        if ddp:
            torch.distributed.barrier()
        trajectories, steps, dims, targets = next(data_iter)
    trajectories, steps, dims, targets  = trajectories.to(device, non_blocking=True), steps.to(device, non_blocking=True), dims.to(device, non_blocking=True), targets.to(device, non_blocking=True)
    return trajectories, steps, dims, targets, epoch, data_iter

def top_p_sample(logits, p=0.9):
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff_index = (cumulative_probs > p).nonzero(as_tuple=True)[0][0]
    top_p_probs = sorted_probs[:cutoff_index + 1]
    top_p_indices = sorted_indices[:cutoff_index + 1]
    
    top_p_probs = top_p_probs / top_p_probs.sum()
    
    sampled_index = top_p_indices[torch.multinomial(top_p_probs, num_samples=1)]
    return sampled_index

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 0

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=config['bias'], vocab_size=None, dropout=config['dropout'], max_dim = config['max_dim'], max_step = config['max_step'],
                  input_embd_dim=input_embd_dim, output_embd_dim=config['num_action_bins']) # start with model_args from command line
if config['init_from'] == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif config['init_from'] == 'resume':
    print(f"Resuming training from {config['input_dir']}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(config['input_dir'], 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'max_dim','max_step', 'input_embd_dim', 'output_embd_dim']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = 0#checkpoint['iter_num']
    best_val_loss = -1#checkpoint['best_val_loss']
elif config['init_from'].startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {config['init_from']}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=config['dropout'])
    model = GPT.from_pretrained(config['init_from'], override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(config['weight_decay'], config['learning_rate'], (config['beta1'], config['beta2']), device_type)
if config['init_from'] == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])



@torch.no_grad()
def estimate_loss(valid_data_iters):
    out = {}
    model.eval()

    # Calculate validation loss
    for k in range(len(valid_data_iters)):
        total_loss = 0
        for i in range(len(valid_data_loaders[k])):
            trajectories, steps, dims, targets, valid_data_iter = get_valid_batch(valid_data_iters[k], valid_data_loaders[k])
            valid_data_iters[k] = valid_data_iter

            _, loss = model(trajectories, steps, dims, targets)
            total_loss += loss.item()
        out[config['valid_paths'][k].split('/')[-1]] = total_loss / len(valid_data_loaders[k])

    
    model.train()
    return out, valid_data_iters

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < config['warmup_iters']:
        return config['learning_rate'] * it / config['warmup_iters']
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config['warmup_iters']) / (lr_decay_iters - config['warmup_iters'])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (config['learning_rate'] - min_lr)

# logging
if config['wandb_log'] and master_process:
    import wandb
    wandb.init(project=config['wandb_project'], name=wandb_run_name, config=config, mode = 'online')

# training loop
trajectories, steps, dims, targets, epoch, data_iter = get_batch(data_iter, data_loader, 'train', epoch)

t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if config['decay_lr'] else config['learning_rate']
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % config['eval_interval'] == 0 and master_process:
        losses, valid_data_iters = estimate_loss(valid_data_iters)
        total_val_loss = 0 
        for k in range(len(config['valid_paths'])):
            print(f"step {iter_num}: {config['valid_paths'][k].split('/')[-1]} test score {losses[config['valid_paths'][k].split('/')[-1]]:.4f}")
            total_val_loss += losses[config['valid_paths'][k].split('/')[-1]]
        total_val_loss = total_val_loss / len(config['valid_paths'])

        if config['wandb_log']:
            wandb.log({
                **{f"val/{config['valid_paths'][k].split('/')[-1]}": losses[config['valid_paths'][k].split('/')[-1]] for k in range(len(config['valid_paths']))},
                "val/total_val_loss": total_val_loss
            })
    if iter_num % config['save_interval'] == 0 and master_process:
        if iter_num > 0:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, f'ckpt.pt'))
    if iter_num == 0 and config['eval_only']:
        break
    total_loss = 0
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(config['gradient_accumulation_steps']):
        if ddp:
            model.require_backward_grad_sync = (micro_step == config['gradient_accumulation_steps'] - 1)
        with ctx:
            logits, loss = model(trajectories, steps, dims, targets)
            loss = loss / config['gradient_accumulation_steps'] # scale the loss to account for gradient accumulation
            total_loss += loss.detach().item()
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        trajectories, steps, dims, targets, epoch, data_iter = get_batch(data_iter, data_loader, 'train', epoch)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        
    # clip the gradient
    if config['grad_clip'] != 0.0:
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        if master_process:

            
            if iter_num % config['log_interval'] == 0:
                if total_norm > config['grad_clip']:
                    print(f"Gradients were clipped. Total norm before clipping: {total_norm:.2f}")
                else:
                    print(f"Gradients were not clipped. Total norm: {total_norm:.2f}")
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % config['log_interval'] == 0 and master_process:
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(config['batch_size'] * config['gradient_accumulation_steps'], dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            if config['wandb_log']:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": total_loss,
                    "lr": lr,
                    "mfu": running_mfu*100, 
                    "time": dt*1000,
                })
        print(f"iter {iter_num}: loss {total_loss:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        
    iter_num += 1
    local_iter_num += 1
    
    # termination conditions
    if iter_num > config['max_iters']:
        break

if ddp:
    destroy_process_group()