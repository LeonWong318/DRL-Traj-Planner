"""
Code used to train the continous DRL agents, DDPG and TD3.

Eight different example agent variants are present, the first four of which 
corresponds to the DDPG algorithm, while the second four are TD3. You can 
select which example agent to train and evaluate by setting the ``index`` 
varaible as the first argument from the command line. 
This is generally done by the slurm array function as seen in ``SLURM_jobscript.sh``.
"""

import sys
import gymnasium as gym
import numpy as np
import random

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from utils.plotresults import plot_training_results
from stable_baselines3.common.env_checker import check_env
from torch import no_grad
from pkg_ddpg_td3.utils.map import generate_map_dynamic, generate_map_corridor, generate_map_mpc, generate_map_eval, generate_map_scene_1, generate_map_scene_2
# from pkg_ddpg_td3.utils.map_simple import  generate_simple_map_easy, generate_simple_map_static, generate_simple_map_nonconvex, generate_simple_map_dynamic,generate_simple_map_nonconvex_static, generate_simple_map_dynamic3
from pkg_ddpg_td3.utils.map_simple import *
from pkg_ddpg_td3.utils.map_multi_robot import generate_map_multi_robot1, generate_map_multi_robot2, generate_map_multi_robot3
from pkg_ddpg_td3.environment import MapDescription
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from typing import Callable
from pkg_ddpg_td3.utils.per_ddpg import PerDDPG
from pkg_ddpg_td3.utils.per_td3 import PerTD3

# from main_pre_continous import generate_map

import tqdm
def generate_map() -> MapDescription:
    return random.choice([generate_map_dynamic, generate_simple_map_dynamic,generate_simple_map_nonconvex,generate_simple_map_static])()
    return random.choice([generate_map_dynamic])()

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def step_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        lr = initial_value

        if progress_remaining < 0.5:
            lr = initial_value/2
        else:
            lr = initial_value
            
        return lr

    return func

def run():
    
    # Selects which predefined agent model to use
    # index = int(sys.argv[1])      #training on cluster
    index = 0                       #training local
    run_vers = 13
    # Load a pre-trained model
    load_checkpoint = True

    # Select the path where the model should be stored
    path = f'./Model/training/variant-{index}' + f'/run{run_vers}'
    # path = './Model/td3/image'
    # path = './Model/td3/ray'
    # path = './Model/ddpg/image'
    # path = './Model/ddpg/ray'
    
    # Parameters for different example agent models 
    variant = [
        {
            'algorithm' : "DDPG",
            'env_name': 'TrajectoryPlannerEnvironmentImgsReward1-v0',
            'net_arch': [64, 64],
            'per': True,
            'device': 'auto',
        },
        {
            'algorithm' : "DDPG",
            'env_name': 'TrajectoryPlannerEnvironmentImgsReward2-v0',
            'net_arch': [64, 64],
            'per': True,
            'device': 'auto',
        },
        {
            'algorithm' : "DDPG",
            'env_name': 'TrajectoryPlannerEnvironmentRaysReward1-v0',
            'net_arch': [16, 16],
            'per': True,
            'device': 'cpu',
        },
        {
            'algorithm' : "DDPG",
            'env_name': 'TrajectoryPlannerEnvironmentRaysReward2-v0',
            'net_arch': [16, 16],
            'per': True,
            'device': 'cpu',
        },
        # TD3
        {
            'algorithm' : "TD3",
            'env_name': 'TrajectoryPlannerEnvironmentImgsReward1-v0',
            'net_arch': [64, 64],
            'per': True,
            'device': 'auto',
        },
        {
            'algorithm' : "TD3",
            'env_name': 'TrajectoryPlannerEnvironmentImgsReward2-v0',
            'net_arch': [64, 64],
            'per': True,
            'device': 'auto',
        },
        {
            'algorithm' : "TD3",
            'env_name': 'TrajectoryPlannerEnvironmentRaysReward1-v0',
            'net_arch': [16, 16],
            'per': True,
            'device': 'cpu',
        },
        {
            'algorithm' : "TD3",
            'env_name': 'TrajectoryPlannerEnvironmentRaysReward2-v0',
            'net_arch': [16, 16],
            'per': True,
            'device': 'cpu',
        },
    ][index]

    tot_timesteps = 7e6
    n_cpu = 20
    
    # """
    # test_scene_1_dict = {1: [1, 2, 3], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4], 4: [1, 2]}
    # test_scene_2_dict = {1: [1, 2, 3]}

    # rl_index: 0 = image, 1 = ray
    # decision_mode: 0 = MPC, 1 = DDPG, 2 = TD3, 3 = Hybrid DDPG, 4 = Hybrid TD3  
    # """

    scene_option = (1, 1, 2)
    # generate_map(*scene_option)

    env_eval = gym.make(variant['env_name'], generate_map=generate_map_mpc(11))
    vec_env = make_vec_env(variant['env_name'], n_envs=n_cpu, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'generate_map': generate_simple_map_static})
    vec_env_eval = make_vec_env(variant['env_name'], n_envs=n_cpu, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'generate_map': generate_simple_map_static})
    # check_env(vec_env)

    n_actions  = vec_env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions),sigma=0.1 * np.ones(n_actions))

    if variant["algorithm"] == "DDPG" and not variant["per"]:
        Algorithm = DDPG
    elif variant["algorithm"] == "DDPG" and variant["per"]:
        Algorithm = PerDDPG
    elif variant["algorithm"] == "TD3" and not variant["per"]:
        Algorithm = TD3
    elif variant["algorithm"] == "TD3" and variant["per"]:
        Algorithm = PerTD3

    
    eval_callback = EvalCallback(vec_env_eval,
                                 best_model_save_path=path,
                                 log_path=path,
                                 eval_freq=max((tot_timesteps / 1000) // n_cpu, 1),
                                 n_eval_episodes=n_cpu)

    if load_checkpoint:
        model = Algorithm.load(f"{path}/best_model", env=env_eval)
        # plot_training_results(path)

        with no_grad():
            rew_list = []
            for j in range(1):
                obs = env_eval.reset()
                
                cum_ret = 0
                for i in range(0, 1000):
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env_eval.step(action)
                    cum_ret += reward
                    if i % 3 == 0: # Only render every third frame for performance (matplotlib is slow)
                        # vec_env.render("human")
                        env_eval.render()
                    if done:
                        # print(cum_ret)
                        print(cum_ret)
                        rew_list.append(cum_ret)
                        env_eval.reset()
                        break
            print(f'mean={sum(rew_list)/len(rew_list)}')
    
    
    else:
        model = Algorithm("MultiInputPolicy",
                    vec_env, 
                    # learning_rate=linear_schedule(0.0001),
                    learning_rate=0.0001, 
                    buffer_size=int(1e6), 
                    learning_starts=100_000, gamma=0.98,
                    # tau=0.1, # detta ska jag titta på imorgon
                    # train_freq=12, # detta ska jag titta på imorgon
                    gradient_steps=-1,
                    action_noise = action_noise,
                    policy_kwargs={'net_arch': variant['net_arch']},
                    verbose=1,
                    device=variant['device'],
                )

        # Train the model    
        model.learn(total_timesteps=tot_timesteps, log_interval=4, progress_bar=True, callback=eval_callback)


        # Save the model
        model.save(f"{path}/final_model")

                    
    
if __name__ == "__main__":
    run()
