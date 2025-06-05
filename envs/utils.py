import numpy as np
from scipy.stats.qmc import LatinHypercube
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch


def get_initial_observations(envs, n_obs, seed=None):

    num_envs = len(envs.envs)
    obs_dim, action_dim = envs.single_observation_space.shape[0], envs.single_action_space.shape[0]
    if obs_dim == action_dim + 1: # zeroth order
        order = 0
    elif obs_dim == action_dim + 1 + action_dim: # first order
        order = 1
    elif obs_dim == action_dim + 1 + action_dim + action_dim ** 2: # second order
        order = 2
    else:
        raise ValueError(f"Environment dimension error")
    action_lb, action_ub = envs.single_action_space.low, envs.single_action_space.high
    initial_obs = np.empty((n_obs, num_envs, obs_dim))

    # Generate LHC sampler
    sampler = LatinHypercube(d=action_dim, seed=seed)

    # For each environment, generate unique random actions and evaluate them
    for j, env in enumerate(envs.envs):
        random_actions = sampler.random(n=n_obs) * (action_ub - action_lb) + action_lb
        for i, action in enumerate(random_actions):
            obs,_,_,_,_ = env.step(action)
            initial_obs[i, j] = obs
    return initial_obs


def render_env(env, obs_history, num_init_steps, static=False):
    points = [np.array([obs[0], obs[1]]) for obs in obs_history]
    fig, ax = plt.subplots()
    if static:
        env.unwrapped.f.visualize(ax, init_points=points[:num_init_steps], points=points[num_init_steps:])
        plt.show()
    else:
        # Animation over the 2D contour plot
        def update(num, points, ax):
            ax.clear()
            env.unwrapped.f.visualize(ax, init_points=points[:num_init_steps], points=points[num_init_steps:num_init_steps+num+1])
            return ax
        ani = FuncAnimation(fig, update, frames=len(obs_history) - num_init_steps + 5, fargs=(points, ax), repeat=True)
        print("saving")
        ani.save('animation3.gif', writer='pillow')  # To save the animation
        print("saved")
        plt.show()


def render_env_saved(env, obs_history, num_init_steps, static=False, name ='animation3.gif'):
    points = [np.array([obs[0], obs[1]]) for obs in obs_history]
    fig, ax = plt.subplots()
    X,Y,Z = env.unwrapped.f.grid(200)
    if static:
        env.unwrapped.f.visualize(ax, init_points=points[:num_init_steps], points=points[num_init_steps:],X=X,Y=Y,Z=Z)
        plt.show()
    else:
        # Animation over the 2D contour plot
        def update(num, points, ax):
            ax.clear()
            env.unwrapped.f.visualize(ax, init_points=points[:num_init_steps], points=points[num_init_steps:num_init_steps+num+1],X=X,Y=Y,Z=Z)
            return ax
        ani = FuncAnimation(fig, update, frames=obs_history.shape[0], fargs=(points, ax), repeat=True)
        print("saving")
        ani.save(name, writer='pillow')  # To save the animation
        print("saved")
        plt.show()

def render_env_comparison(env, initial_points, points1, points2, static=False):
    fig, ax = plt.subplots()
    if static:
        env.unwrapped.f.visualize_comparison(ax, initial_points, points1, points2)
        plt.savefig('comp.png')
    else:
        # Animation over the 2D contour plot
        def update(num, points1, points2, ax):
            ax.clear()
            env.unwrapped.f.visualize_comparison(ax, initial_points, points1[:,:num,:], points2[:,:num,:])
            return ax
        ani = FuncAnimation(fig, update, frames=points1.shape[1], fargs=(points1, points2, ax), repeat=True)
        print("saving")
        ani.save('animation3.gif', writer='pillow')  # To save the animation
        print("saved")
        plt.show()

def render_env_comparison_saved(env, initial_points, points1, points2, static=False):
    X,Y,Z = env.unwrapped.f.grid(200)
    fig, ax = plt.subplots()
    if static:
        env.unwrapped.f.visualize_comparison(ax, initial_points, points1, points2,X=X,Y=Y,Z=Z)
        plt.savefig('comp.png')
    else:
        # Animation over the 2D contour plot
        def update(num, points1, points2, ax):
            ax.clear()
            env.unwrapped.f.visualize_comparison(ax, initial_points, points1[:,:num,:], points2[:,:num,:],X=X,Y=Y,Z=Z)
            return ax
        ani = FuncAnimation(fig, update, frames=max(points1.shape[1],points2.shape[1]), fargs=(points1, points2, ax), repeat=True)
        print("saving")
        ani.save('animation3.gif', writer='pillow')  # To save the animation
        print("saved")
        plt.show()


if __name__ == '__main__':

    import os
    import sys

    project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    sys.path.append(project_base_dir)

    import gymnasium as gym
    import envs
    env_id = 'QuadraticEnv-v0'
    num_envs = 20
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make(env_id) for i in range(num_envs)]
    )
    initial_obs = get_initial_observations(envs, n_obs=10)

    print(initial_obs.shape)