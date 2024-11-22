import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from autoEncoder import Autoencoder
from dataLoad import create_dataloaders

def main():
    # Parameters
    base_path = "../trial_data"  # Path to the dataset
    batch_size = 64        # Batch size for training
    latent_dim = 8       # Dimensionality of the latent space
    base_channel_size = 32 # Base number of channels in the encoder/decoder
    num_epochs = 50        # Number of epochs to train
    width, height, num_input_channels = 54, 54, 3  # Updated dimensions for input

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

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="autoencoder-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min"
    )

    # Trainer
    trainer = Trainer(
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator="auto",  # Use GPU if available
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Save the final model
    torch.save(model.state_dict(), "autoencoder_final_{val_loss:.2f}.pth")

if __name__ == "__main__":
    main()
