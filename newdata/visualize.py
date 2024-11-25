import os
import numpy as np
import matplotlib.pyplot as plt
import torch

root_path = "."  # 当前工作目录

def tensor2img(tensor):
    """
    Convert a tensor to an image and display it.
    Args:
        tensor (torch.Tensor): A tensor with shape (C, H, W).
    """
    image_array = tensor.permute(1, 2, 0).cpu().numpy()
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    image_array = (image_array * 255).astype(np.uint8)
    
    plt.figure()
    plt.imshow(image_array)
    plt.axis('off')  # 隐藏坐标轴
    plt.show()

def display_all_images_in_episode(episode_path):
    """
    Display all .pt files as images from a given directory.
    Args:
        episode_path (str): Path to the directory containing .pt files.
    """
    pt_files = [file for file in os.listdir(episode_path) if file.endswith('.pt')]
    pt_files.sort()
    for pt_file in pt_files:
        file_path = os.path.join(episode_path, pt_file)
        tensor = torch.load(file_path)
        tensor2img(tensor)

def display_nth_image_across_episodes(root_path, n):
    """
    Display the nth .pt file from each episode directory.
    Args:
        root_path (str): Root directory containing all episode directories.
        n (int): Index of the file to display from each episode.
    """
    for episode in range(1, 22):
        episode_path = os.path.join(root_path, f'Episode_{episode}')
        try:
            pt_files = [file for file in os.listdir(episode_path) if file.endswith('.pt')]
            pt_files.sort()
            if len(pt_files) > n:
                file_path = os.path.join(episode_path, pt_files[n])
                tensor = torch.load(file_path)
                tensor2img(tensor)
            else:
                print(f"No {n+1}th file in {episode_path}.")
        except Exception as e:
            print(f"Error processing files in {episode_path}: {e}")


def display_each_channel_in_episode(episode_path):
    """
    Channel 1: Current occupancy grid.
    Channel 2: Previous(5 step before) occupancy grid.
    Channel 3: Distance field.
    Func:
        Plot each channel independently as heatmap.
    """
    pt_files = [file for file in os.listdir(episode_path) if file.endswith('.pt')]
    pt_files.sort()
    for pt_file in pt_files:
        file_path = os.path.join(episode_path, pt_file)
        tensor = torch.load(file_path)
        for i in range(3):
            plt.figure()
            plt.imshow(tensor[i], cmap='hot')  
            plt.title(f'Channel {i+1}')
            plt.colorbar()
            plt.show()
        




# Example Usage
episode_number = 5  # 选择展示第5个episode的所有图片
# display_all_images_in_episode(os.path.join(root_path, f'Episode_{episode_number}'))

display_each_channel_in_episode(os.path.join(root_path, f'Episode_{episode_number}'))

# nth_image = 0  # 选择展示每个episode中的第1个图片
# display_nth_image_across_episodes(root_path, nth_image)
