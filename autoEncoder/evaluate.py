from autoEncoder import Autoencoder  # Assuming the model class is in this module
import torch
from dataLoad import create_dataloaders


base_path = "../trial_data"  # Path to the dataset
batch_size = 64        # Batch size for training
latent_dim = 32       # Dimensionality of the latent space
base_channel_size = 32 # Base number of channels in the encoder/decoder
num_epochs = 50        # Number of epochs to train
width, height, num_input_channels = 54, 54, 3  # Updated dimensions

# Create DataLoaders
train_loader, val_loader = create_dataloaders(base_path, batch_size=batch_size, train_split=0.8)

# Initialize the model
model = Autoencoder(
    base_channel_size=base_channel_size,
    latent_dim=latent_dim,
    num_input_channels=num_input_channels,
    width=width,
    height=height
)


# Load the saved state dictionary
model.load_state_dict(torch.load("autoencoder_final.pth"))
model.eval()  # Set the model to evaluation mode




def evaluate_model(model, dataloader):
    model.eval()
    total_mse = 0.0
    total_count = 0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(model.device)
            reconstructions = model(images)
            mse = F.mse_loss(reconstructions, images, reduction='sum')
            total_mse += mse.item()
            total_count += images.size(0)
    
    average_mse = total_mse / total_count
    print(f"Average MSE: {average_mse}")

# Call this function after training
evaluate_model(model, val_loader)