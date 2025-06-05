import gymnasium as gym

STEPS = 100000

for i in range(1,25):
    for d in range(1,41):
        env_name = f'BBOBEnv-{d}D-f{i}-v0'
        gym.register(
            id=env_name,
            entry_point='envs.bbob:BBOBEnv',
            kwargs={'fn_type': i, 'N':d},
            max_episode_steps=STEPS
        )
        gym.register(
            id=env_name + '-gradient',
            entry_point='envs.bbob_gradient:GradientBBOBEnv',
            kwargs={'fn_type': i},
            max_episode_steps=STEPS
        )
        gym.register(
            id=env_name + '-hessian',
            entry_point='envs.bbob_gradient:HessianBBOBEnv',
            kwargs={'fn_type': i},
            max_episode_steps=STEPS
        )

IDs = ['4796','5527','5636','5859','5860','5891','5906','5965','5970','5971','6766','6767','6794','7607','7609','5889']
for search_space_id in IDs:
    for mode in ['train', 'test', 'validate']:
        gym.register(
            id=f'HPOBEnv-{mode}-{search_space_id}-v0',
            entry_point='envs.hpob:HPOBEnv',
            kwargs={'search_space_id': search_space_id, 'mode': mode},
            max_episode_steps=STEPS
        )



for i in range(1,50):
    gym.register(
    id=f'GPEnv-{i}D-v0',
    entry_point='envs.gp:GPEnv',
    kwargs={'N': i},
    max_episode_steps=STEPS
)


for i in range(1,50):
    gym.register(
    id=f'GPMeanEnv-{i}D-v0',
    entry_point='envs.gp_mean:GPEnv',
    kwargs={'N': i},
    max_episode_steps=STEPS
)

from .valid_envs import vlse_envs

for key in vlse_envs.keys():
    for problem in vlse_envs[key]:
        gym.register(
            id=f'BOProblemEnv-{problem}-{key}D-v0',
            entry_point='envs.vlse:BOProblemEnv',
            kwargs={'N': int(key), 'problem_type': problem},
            max_episode_steps=STEPS
        )


    

