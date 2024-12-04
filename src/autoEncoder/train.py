import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import AutoLocator


from autoEncoder import Autoencoder
from ResNet_based_autoEncoder import ResNetBasedAutoencoder
from VAE import VAE
from dataLoad import create_dataloaders

# To store the loss
train_losses = []
val_losses = []
class LossLoggerCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        global train_losses, val_losses

        # Access logged metrics
        train_loss = trainer.callback_metrics.get("train_loss")
        val_loss = trainer.callback_metrics.get("val_loss")
        
        # if isinstance(train_loss, torch.Tensor):
        train_losses.append(float(train_loss))  # Convert tensor to float
        val_losses.append(float(val_loss))  # Convert tensor to float
        # else:
        #     train_losses.append(train_loss)  
        #     val_losses.append(val_loss)  
            
def parse_args():
    parser = argparse.ArgumentParser(description='Train a simple model')
    parser.add_argument('--ld', type=int, default=128, help='Dimensionality of the latent space')
    parser.add_argument('--e', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--arc', type=str, default='VAE', help='The architecture you want to train')
    return parser.parse_args()

def main():
    # Parameters
    args = parse_args()
    base_path = "../newdata"  # Path to the dataset
    batch_size = 64        # Batch size for training
    latent_dim = args.ld       # Dimensionality of the latent space
    base_channel_size = 32 # Base number of channels in the encoder/decoder
    num_epochs = args.e        # Number of epochs to train
    width, height, num_input_channels = 54, 54, 3  # Updated dimensions for input
    arch_name = args.arc
    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(base_path, batch_size=batch_size, train_split=0.8)

    # Initialize the model
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
    elif arch_name == 'VAE':
        model = VAE(
            base_channel_size=base_channel_size,
            latent_dim=latent_dim,
            num_input_channels=num_input_channels,
            width=width,
            height=height
        )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename=f"{arch_name}""-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    # early_stopping_callback = EarlyStopping(
    #     monitor="val_loss",
    #     patience=10,
    #     mode="min"
    # )
    
    loss_logger = LossLoggerCallback()
    # Trainer
    trainer = Trainer(
        max_epochs=num_epochs,
        #callbacks=[checkpoint_callback, early_stopping_callback],
        callbacks = [checkpoint_callback, loss_logger],
        accelerator="auto",  # Use GPU if available
        logger=False,  # Disable Lightning loggers
    )
    
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    #print("Train Loss:", trainer.logged_metrics.get("train_loss").item())
    #print("Validation Loss:", trainer.logged_metrics.get("val_loss").item())
    # Save the final model
    torch.save(model.state_dict(), f"./model/{arch_name}_allnewdata_{latent_dim}e{num_epochs}.pth")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot and save the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {latent_dim} Latent over {num_epochs} Epochs')
    plt.legend()
    # Ensure x-axis shows only integers
    epochs = range(0, len(train_losses))
    plt.xticks(epochs)  # Set x-ticks to integers (0-based index)
    plt.gca().xaxis.set_major_locator(AutoLocator())
    plt.savefig(f'./figure/{arch_name}_allnewdata_{latent_dim}e{num_epochs}_loss_curve_{timestamp}.png')
    plt.show()

if __name__ == "__main__":
    main()
