import torch
import matplotlib.pyplot as plt
from autoEncoder import Autoencoder
from ResNet_based_autoEncoder import ResNetBasedAutoencoder
from dataLoad import create_dataloaders, create_test_loader
import os
import re
import glob
import argparse

# Function to extract latent_dim from file name
def extract_latent_dim_from_filename(filename):
    # Match a number before 'e' in the file name (e.g., "8e100" -> 8)
    match = re.search(r'(\d+)e', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract latent_dim from filename: {filename}")

# Function to get the latest model path (based on creation/modification time)
def get_latest_model_path(base_path='model/'):
    # Get a list of model files in the base directory
    model_files = glob.glob(os.path.join(base_path, '*.pth'))
    
    # Sort the model files by modification time (latest first)
    model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Return the most recent model file, or None if there are no files
    return os.path.basename(model_files[0]) if model_files else None

def load_model(model_path, base_channel_size, latent_dim, num_input_channels, width, height, arch_name):
    """
    Load the trained autoencoder model from a saved state.
    """
    if arch_name == 'autoencoder':
        model = Autoencoder(
            base_channel_size=base_channel_size,
            latent_dim=latent_dim,
            num_input_channels=num_input_channels,
            width=width,
            height=height
        )
    elif arch_name == 'ResNet':
        model = ResNetBasedAutoencoder(
            base_channel_size=base_channel_size,
            latent_dim=latent_dim,
            num_input_channels=num_input_channels,
            width=width,
            height=height
        )
    checkpoint = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu""cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(checkpoint, strict=False)
    model.eval()  # Set model to evaluation mode
    return model

def evaluate_model(model, test_loader, num_samples=5):
    """
    Evaluate the model on the test dataset.

    Args:
        model (Autoencoder): The trained autoencoder.
        test_loader (DataLoader): DataLoader for the test dataset.
        num_samples (int): Number of samples to visualize.

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    model.to(device)

    # Compute reconstruction loss
    total_loss = 0
    num_batches = 0
    for batch in test_loader:
        batch = batch.to(device)
        with torch.no_grad():
            reconstructed = model(batch)
            loss = torch.nn.functional.mse_loss(batch, reconstructed, reduction="sum")
            total_loss += loss.item()
        num_batches += 1
    average_loss = total_loss / len(test_loader.dataset)

    print(f"Reconstruction Loss: {average_loss:.4f}")

    # Visualize a few samples
    test_iter = iter(test_loader)
    batch = next(test_iter).to(device)

    with torch.no_grad():
        reconstructed = model(batch)

    # Select a few samples to visualize
    for i in range(min(num_samples, batch.size(0))):
        plt.figure(figsize=(6, 3))
        # Original image
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(batch[i].permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        # Reconstructed image
        plt.subplot(1, 2, 2)
        plt.title("Reconstructed")
        plt.imshow(reconstructed[i].permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.show()
    
    
    # plot each channel independently
    for i in range(min(num_samples, len(batch))):
        original_tensor = batch[i].cpu().numpy()
        reconstructed_tensor = reconstructed[i].cpu().numpy()

        fig, axes = plt.subplots(2, 3, figsize=(10, 7)) 
        
        for j in range(3):
            im = axes[0, j].imshow(original_tensor[j], cmap='hot') 
            axes[0, j].set_title(f"Original: Channel {j+1}")
            axes[0, j].axis('off')  
            if j == 2:  
                plt.colorbar(im, ax=axes[0, j])

        for j in range(3):
            im = axes[1, j].imshow(reconstructed_tensor[j], cmap='hot') 
            axes[1, j].set_title(f"Reconstructed: Channel {j+1}")
            axes[1, j].axis('off') 
            if j == 2:  
                plt.colorbar(im, ax=axes[1, j])

        fig.suptitle(f"Sample {i+1}: Original vs Reconstructed", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--model', type=str, default=get_latest_model_path(), help='Dimensionality of the latent space')
    parser.add_argument('--ld', type=int, default=extract_latent_dim_from_filename(get_latest_model_path()), help='Number of epochs for training')
    parser.add_argument('--arc', type=str, default='ResNet', help='The architecture you want to evaluate')
    return parser.parse_args()

def main():
    # Parameters
    args = parse_args()
    base_path = "../test_data"  # Path to the dataset
    batch_size = 64        # Batch size for evaluation
    model_path = os.path.join("model/",args.model)   # Auto using latest model
    
    print(f"Evaluating model: {args.model}")
    latent_dim = args.ld      # Dimensionality of the latent space
    base_channel_size = 32 # Base number of channels in the encoder/decoder
    width, height, num_input_channels = 54, 54, 3  # Input dimensions
    arch_name = args.arc
    # Create DataLoader for the test set
    test_loader = create_test_loader(base_path, batch_size=batch_size)

    # Load the model
    model = load_model(model_path, base_channel_size, latent_dim, num_input_channels, width, height, arch_name)

    # Evaluate the model
    evaluate_model(model, test_loader, num_samples=5)

if __name__ == "__main__":
    main()
