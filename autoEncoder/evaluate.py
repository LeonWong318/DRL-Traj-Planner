import torch
import matplotlib.pyplot as plt
from autoEncoder import Autoencoder
from dataLoad import create_dataloaders

def load_model(model_path, base_channel_size, latent_dim, num_input_channels, width, height):
    """
    Load the trained autoencoder model from a saved state.
    """
    model = Autoencoder(
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

def main():
    # Parameters
    base_path = "../test_data"  # Path to the dataset
    batch_size = 64        # Batch size for evaluation
    model_path = "model/autoencoder_alldata_8e100.pth"  # Path to the trained model
    latent_dim = 8      # Dimensionality of the latent space
    base_channel_size = 32 # Base number of channels in the encoder/decoder
    width, height, num_input_channels = 54, 54, 3  # Input dimensions

    # Create DataLoader for the test set
    _, test_loader = create_dataloaders(base_path, batch_size=batch_size, train_split=0.1)

    # Load the model
    model = load_model(model_path, base_channel_size, latent_dim, num_input_channels, width, height)

    # Evaluate the model
    evaluate_model(model, test_loader, num_samples=5)

if __name__ == "__main__":
    main()
