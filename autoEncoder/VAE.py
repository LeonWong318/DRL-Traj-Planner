from dataLoad import create_dataloaders

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # Output size: (32, 27, 27)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # Output size: (64, 13, 13)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # Output size: (128, 6, 6)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128 * 6 * 6, latent_dim)
        self.fc_logvar = nn.Linear(128 * 6 * 6, latent_dim)

        # 解码器
        self.decoder_input = nn.Linear(latent_dim, 128 * 6 * 6)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output size: (64, 12, 12)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # Output size: (32, 24, 24)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=10, stride=2, padding=1),   # Output size: (3, 54, 54)
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        # print(f"Encoder output shape: {h.shape}")
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 128, 6, 6)
        h = self.decoder(h)
        # print(f"Decoder output shape: {h.shape}")  # 应该输出 (batch_size, 3, 54, 54)
        return h

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def loss_function(recon_x, x, mu, logvar, beta=1):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld_loss



# base_path = "../newdata"  # Path to your dataset
# dataloader, val_loader = create_dataloaders(base_path, batch_size=32, train_split=0.8)

# model = VAE(latent_dim=64)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# num_epochs = 50
# train_losses = []

# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0
#     progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}, Training")
#     for x in progress_bar:
#         x = x.to(device)
#         optimizer.zero_grad()
#         recon_x, mu, logvar = model(x)
#         loss = loss_function(recon_x, x, mu, logvar)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#         progress_bar.set_postfix(loss=f"{loss.item():.4f}")

#     train_loss /= len(dataloader.dataset)
#     train_losses.append(train_loss)

#     # 验证阶段
#     model.eval()
#     val_loss = 0
#     val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Validation")
#     with torch.no_grad():
#         for x in val_progress_bar:
#             x = x.to(device)
#             recon_x, mu, logvar = model(x)
#             loss = loss_function(recon_x, x, mu, logvar)
#             val_loss += loss.item()
#             val_progress_bar.set_postfix(val_loss=f"{loss.item():.4f}")

#     val_loss /= len(val_loader.dataset)

#     print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")


# torch.save(model.state_dict(), 'vae_model.pth')
# print("Model saved as vae_model.pth")


# plt.figure(figsize=(10,5))
# plt.plot(range(1, num_epochs+1), train_losses, marker='o')
# plt.title('Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.savefig("vae_train_loss.png")
# plt.show()

