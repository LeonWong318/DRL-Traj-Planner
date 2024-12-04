from dataLoad import create_dataloaders

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pytorch_lightning as pl

class VAE(pl.LightningModule):
    def __init__(self, 
                 base_channel_size: int, 
                 latent_dim: int, 
                 num_input_channels: int = 3, 
                 width: int = 32, 
                 height: int = 32):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.base_channel_size = base_channel_size
        self.num_input_channels = num_input_channels
        self.width = width
        self.height = height

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(num_input_channels, base_channel_size, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.BatchNorm2d(base_channel_size),
            nn.ReLU(),
            nn.Conv2d(base_channel_size, base_channel_size * 2, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.BatchNorm2d(base_channel_size * 2),
            nn.ReLU(),
            nn.Conv2d(base_channel_size * 2, base_channel_size * 4, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.BatchNorm2d(base_channel_size * 4),
            nn.ReLU()
        )

        # Flattened dimension after the encoder
        flattened_dim = (width // 8) * (height // 8) * (base_channel_size * 4)

        # Latent space
        self.fc_mu = nn.Linear(flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(flattened_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, flattened_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channel_size * 4, base_channel_size * 2, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.BatchNorm2d(base_channel_size * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channel_size * 2, base_channel_size, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.BatchNorm2d(base_channel_size),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channel_size, num_input_channels, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(h.size(0), self.base_channel_size * 4, self.width // 8, self.height // 8)
        h = self.decoder(h)
        # Explicitly resize to match input size
        h = F.interpolate(h, size=(self.height, self.width), mode="bilinear", align_corners=False)
        return h


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    
    def _get_reconstruction_loss(self, batch):
        """
        Calculate the reconstruction loss (MSE).
        """
        x = batch  # Assuming batch contains only images
        recon_x, _, _ = self(x)
        loss = F.mse_loss(recon_x, x, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=0)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, min_lr=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)




# base_path = "../newdata"  # Path to your dataset
# dataloader, val_loader = create_dataloaders(base_path, batch_size=32, train_split=0.8)

# model = VAE(latent_dim=64)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# num_epochs = 10
# train_losses = []
# train_recon_losses = []
# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0
#     train_recon_loss = 0
#     progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}, Training")
#     for x in progress_bar:
#         x = x.to(device)
#         optimizer.zero_grad()
#         recon_x, mu, logvar = model(x)
#         loss, recon_loss = loss_function(recon_x, x, mu, logvar)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#         train_recon_loss += recon_loss.item()
#         progress_bar.set_postfix(loss=f"{recon_loss.item():.4f}")   #only print reconstruction loss no kl loss

#     train_loss /= len(dataloader.dataset)
#     train_losses.append(train_loss)

#     train_recon_loss /= len(dataloader.dataset)
#     train_recon_losses.append(train_recon_loss)

#     # 验证阶段
#     model.eval()
#     val_loss = 0
#     val_recon_loss = 0
#     val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Validation")
#     with torch.no_grad():
#         for x in val_progress_bar:
#             x = x.to(device)
#             recon_x, mu, logvar = model(x)
#             loss, recon_loss = loss_function(recon_x, x, mu, logvar)
#             val_loss += loss.item()
#             val_recon_loss += recon_loss.item()
#             val_progress_bar.set_postfix(val_loss=f"{recon_loss.item():.4f}")

#     val_loss /= len(val_loader.dataset)
#     val_recon_loss /= len(val_loader.dataset)
#     print(train_loss - train_recon_loss)
#     print(f"Epoch {epoch+1}, Training Loss: {train_recon_loss:.4f}, Validation Loss: {val_recon_loss:.4f}")


# torch.save(model.state_dict(), 'vae_model_64e10mean.pth')
# print("Model saved as vae_model.pth")


# plt.figure(figsize=(10,5))
# plt.plot(range(1, num_epochs+1), train_losses, marker='o')
# plt.title('Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.savefig("vae_train_loss_64e10mean.png")
# plt.show()

# base_path = "../newdata"  # Path to the dataset
# batch_size = 64        # Batch size for training
# latent_dim = 8       # Dimensionality of the latent space
# base_channel_size = 32 # Base number of channels in the encoder/decoder
# num_epochs = 100        # Number of epochs to train
# width, height, num_input_channels = 54, 54, 3  # Updated dimensions for input



# # Initialize the model
    
# model = VAE(latent_dim)
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total parameters: {total_params}")