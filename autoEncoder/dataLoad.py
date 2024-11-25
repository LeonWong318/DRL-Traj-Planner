import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class EpisodeDataset(Dataset):
    """
    Dataset that loads tensors from episodes stored in a structured directory.
    """

    def __init__(self, base_path="../data", episode_prefix="Episode_", file_prefix="img_", file_extension=".pt"):
        """
        Initialize the dataset.

        Args:
            base_path (str): The parent directory containing episode folders.
            episode_prefix (str): The prefix for episode directories (e.g., "Episode_").
            file_prefix (str): The prefix for files within each episode (e.g., "img_").
            file_extension (str): The file extension of the saved files (e.g., ".pt").
        """
        self.base_path = base_path
        self.episode_prefix = episode_prefix
        self.file_prefix = file_prefix
        self.file_extension = file_extension
        self.tensor_paths = []

        # Collect all tensor file paths with their episode and step indices
        episodes = [
            d for d in os.listdir(base_path)
            if d.startswith(episode_prefix) and os.path.isdir(os.path.join(base_path, d))
        ]
        episodes.sort(key=lambda e: int(e[len(episode_prefix):]) if e[len(episode_prefix):].isdigit() else -1)

        for episode in episodes:
            episode_path = os.path.join(base_path, episode)
            files = [
                f for f in os.listdir(episode_path)
                if f.startswith(file_prefix) and f.endswith(file_extension)
            ]
            files.sort(key=lambda f: int(f[len(file_prefix):-len(file_extension)]) if f[len(file_prefix):-len(file_extension)].isdigit() else -1)

            for file in files:
                self.tensor_paths.append(os.path.join(episode_path, file))

    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        """
        Load the tensor at the given index. 
        """
        tensor_path = self.tensor_paths[idx]
        return torch.load(tensor_path, weights_only=True)  # Only return the tensor


def create_dataloaders(base_path, batch_size=32, train_split=0.8, shuffle=True):
    """
    Create DataLoaders for training and testing from the EpisodeDataset.

    Args:
        base_path (str): The base directory containing episode folders.
        batch_size (int): Batch size for the DataLoaders.
        train_split (float): Proportion of data to use for training.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader: DataLoader for training data.
        DataLoader: DataLoader for testing data.
    """
    dataset = EpisodeDataset(base_path)
    train_size = int(len(dataset) * train_split)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def create_test_loader(base_path, batch_size=32):
    """
    Create a DataLoader that uses all data for testing.

    Args:
        base_path (str): The base directory containing episode folders.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: DataLoader for the test dataset.
    """
    # Load the full dataset without splitting
    dataset = EpisodeDataset(base_path)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# # Example: Iterate through train_loader
# for batch_idx, batch in enumerate(train_loader):
#     print(f"Batch {batch_idx + 1} shape: {batch.shape}")
