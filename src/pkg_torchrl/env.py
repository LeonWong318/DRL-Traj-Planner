import torch
import os
from torchrl.envs import (
    default_info_dict_reader,
    step_mdp,
    GymEnv,
    CatTensors,
    Compose,
    DoubleToFloat,
    TransformedEnv,
    ParallelEnv,
    ToTensorImage,
    ObservationNorm,
)
from torchrl.envs.transforms import (
    InitTracker,
    RewardSum,
    StepCounter,
    VecNorm
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
def find_latest_dir(base_path, prefix="Episode_"):
    """
    Find the latest directory matching a specific prefix under the base path.

    Args:
        base_path (str): The parent directory to search in.
        prefix (str): The prefix of the directories to match.

    Returns:
        str: The latest directory path or None if no directories match.
    """

    # Filter directories matching the prefix
    dirs = [d for d in os.listdir(base_path) if d.startswith(prefix) and os.path.isdir(os.path.join(base_path, d))]
    if not dirs:
        return None

    # Sort directories by numerical suffix
    dirs.sort(key=lambda d: int(d[len(prefix):]) if d[len(prefix):].isdigit() else -1)
    return os.path.join(base_path, dirs[-1])

def auto_save(obj, base_path, step, file_prefix="img", file_extension=".pt"):
    """
    Save the object to a unique directory (if step=0) or an existing directory.

    Args:
        obj: The object to save.
        base_path (str): The parent directory where the new directory will be created.
        step (int): The step index. Creates a new directory if step=0.
        file_prefix (str): The base name of the file.
        file_extension (str): The file extension (e.g., '.pt').
    """
    if step == 0:
        # Generate a unique subdirectory under base_path
        index = 1
        while True:
            new_dir = os.path.join(base_path, f"Episode_{index}")
            if not os.path.exists(new_dir):  # Ensure directory is unique
                os.makedirs(new_dir)  # Create the directory
                break
            index += 1
        print(f"Created new directory: {new_dir}")
    else:
        new_dir = find_latest_dir(base_path)  # Use the existing base path
    
    # Save the file with step index in the determined directory
    filename = f"{file_prefix}_{step}{file_extension}"
    file_path = os.path.join(new_dir, filename)
    
    # Save the object
    torch.save(obj, file_path)

def make_env(config, max_eps_steps=None, **kwargs):
    if max_eps_steps is None:
        max_eps_steps = config.sac.max_eps_steps

    raw_env = GymEnv(config.env_name,
                     w1=config.w1,
                     w2=config.w2,
                     w3=config.w3,
                     w4=config.w4,
                     w5=config.w5,
                     config=config,
                     device=config.device,
                     **kwargs)

    def make_t_env():
        if "Img" in config.env_name:
            transform_list = [
                ToTensorImage(in_keys=["external"], out_keys=["pixels"], shape_tolerant=True),
                InitTracker(),
                StepCounter(max_eps_steps),
                DoubleToFloat(),
                RewardSum(),
                ObservationNorm(standard_normal=True, in_keys=["pixels"]),
            ]
        else:
            transform_list = [
                InitTracker(),
                StepCounter(max_eps_steps),
                DoubleToFloat(),
                RewardSum(),
                CatTensors(in_keys=['internal', 'external'], out_key="observation"),
            ]

        if config.use_vec_norm:
            transform_list += [VecNorm(decay=0.9),]
        t_env = TransformedEnv(raw_env, Compose(*transform_list))
        if "Img" in config.env_name:
            t_env.transform[-1].init_stats(1000, cat_dim=0, reduce_dim=[-1, -2, -4], keep_dims=(-1, -2))
        reader = default_info_dict_reader(["success", "collided", "full_reward", "base_reward", "true_reward", "reward_tensor"])
        t_env.set_info_dict_reader(info_dict_reader=reader)
        return t_env

    if config.n_envs == 1:
        env = make_t_env()
    else:
        env = ParallelEnv(
            create_env_fn=lambda: make_t_env(),
            num_workers=config.n_envs,
            pin_memory=False,
        )

    return env


def render_rollout(eval_env, model, config, n_steps=2_000, is_torchrl=True):
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        state = eval_env.reset()
        
        steps = 0
        ep_rwd = torch.zeros(1).to(config.device)
        for i in range(n_steps):
            action = model.model["policy"](state)
            next_state = eval_env.step(action)
            obs = next_state["pixels"]
            ## uncomment below if you want to save the data
            # auto_save(obs,"../test_data",steps)
            steps += 1
            ep_rwd += next_state['next']['reward']

            # Only render every third frame for performance (matplotlib is slow)
            if i % 3 == 0 and i > 0:
                eval_env.render()

            if next_state['next']['done'] or steps > config.sac.max_eps_steps:
                print(f'Episode reward {ep_rwd.item():.2f}')
                state = eval_env.reset()
                steps = 0
                ep_rwd = torch.zeros(1).to(config.device)
            else:
                state = step_mdp(next_state)