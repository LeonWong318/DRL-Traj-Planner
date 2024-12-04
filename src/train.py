"""
Code used to train the continous DRL agents, DDPG and TD3.

Eight different example agent variants are present, the first four of which 
corresponds to the DDPG algorithm, while the second four are TD3. You can 
select which example agent to train and evaluate by setting the ``index`` 
varaible as the first argument from the command line. 
This is generally done by the slurm array function as seen in ``SLURM_jobscript.sh``.
"""

import random
import logging
from datetime import datetime
from pathlib import Path

import wandb
import torch
import numpy as np
from pkg_ddpg_td3.utils.map import generate_map_eval
from pkg_torchrl.env import make_env
from pkg_torchrl.sac import SAC
from pkg_torchrl.ppo import PPO
from pkg_torchrl.td3 import TD3
from pkg_torchrl.ddpg import DDPG
from pkg_map.utils import get_map

from configs import BaseConfig

logging.basicConfig(level=logging.INFO)


def run():
    config = BaseConfig()
    if config.use_wandb:
        _ = wandb.init(
            project="DRL-Traj-Planner",
            tags=["continuous_training"],
        )
        config = BaseConfig(**wandb.config)
        wandb.config.update(config.model_dump())

    logging.info(f"seed: {config.seed}")

    # set random seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # get function to generate a map
    generate_map = get_map(config.map_key)

    # create train and eval env
    train_env = make_env(config, generate_map=generate_map)#, use_wandb=True)
    if config.map_key == 'random':
        eval_env = make_env(config, max_eps_steps=1000, generate_map=generate_map_eval)
    else:
        eval_env = make_env(config, generate_map=generate_map)

    # create model
    algo_config = getattr(config, config.algo.lower())
    model = eval(config.algo.upper())(algo_config, train_env, eval_env)
    file_dir = Path(__file__).resolve().parents[1]
    models_path = file_dir / 'Model' / 'testing'

    # create directory to save final model
    timestamp = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    path = models_path / f"{timestamp}_{config.algo.upper()}"
    path.mkdir(exist_ok=True, parents=True)

    if config.use_wandb:
        wandb.config["path"] = path
        wandb.config["map"] = generate_map.__name__

    # train model
    # model.to(torch.device('cuda'))
    model.train(use_wandb=config.use_wandb)
    model.save(f"{path}/final_model.pth")
    logging.info(f"Final model saved to {path}")


if __name__ == "__main__":
    run()
