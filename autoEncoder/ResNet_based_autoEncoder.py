import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torchvision.models as models


class ResNetEncoder(nn.Module):
    def __init__(self, latent_dim: int, pretrained: bool = False, input_size: int = 54):
        super().__init__()
        # Load a ResNet model
        base_model = models.resnet18(pretrained=pretrained)
        
        # Remove the fully connected and pooling layers
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])  # Up to the last conv layer

        # Define a dummy forward pass to calculate the output size
        dummy_input = torch.zeros(1, 3, input_size, input_size)
        feature_map_size = self.feature_extractor(dummy_input).shape  # Get output size of feature extractor
        self.flatten_size = feature_map_size[1] * feature_map_size[2] * feature_map_size[3]  # Flatten the feature maps

        # Add a flattening layer and a final linear layer for latent space mapping
        self.flatten = nn.Flatten()
        self.latent_mapping = nn.Linear(self.flatten_size, latent_dim)  # Map the feature map to latent_dim

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.latent_mapping(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_input_channels, base_channel_size, latent_dim, act_fn=nn.GELU):
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * 7 * 7 * c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 -> 27x27
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 27x27 -> 54x54
            act_fn(),
            nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, padding=1),  # Optionally adjust to change dimensions
            nn.Tanh()
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 7, 7)
        x = self.net(x)
        x = F.interpolate(x, size=(54, 54))  # Explicit resizing to 56x56
        return x


class ResNetBasedAutoencoder(pl.LightningModule):
    def __init__(self,
                 base_channel_size: int,
                 latent_dim: int,
                 encoder_class: object = ResNetEncoder,
                 decoder_class: object = Decoder,
                 num_input_channels: int = 3,
                 width: int = 54,
                 height: int = 54,
                 pretrained_encoder: bool = False):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(latent_dim=latent_dim, pretrained=pretrained_encoder, input_size=width)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case).
        """
        # Assume batch is a tensor directly
        x = batch  # No labels provided in your dataset
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=0)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=20,
                                                         min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self
