# -*- coding: utf-8 -*-

"""
Created on Nov 27, 2024
"""

import os
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.utils import save_image

from dataLoad import create_dataloaders

class ConvVAE(nn.Module):
    def __init__(self, latent_size):
        super(ConvVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 3x54x54 -> 32x27x27
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x27x27 -> 64x13x13
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(64 * 13 * 13, latent_size * 2)  # Adjust to match flattened size
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64 * 13 * 13),  # Latent to flattened feature size
            nn.ReLU(),
            nn.Unflatten(1, (64, 13, 13)),  # Reshape back to conv output
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64x13x13 -> 32x27x27
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 32x27x27 -> 3x54x54
            nn.Sigmoid()  # Scale to [0, 1]
        )
    
    def encoder_forward(self, X):
        out = self.encoder(X)
        mu, log_var = torch.chunk(out, 2, dim=1)  # Split into mean and log variance
        return mu, log_var

    def decoder_forward(self, z):
        return self.decoder(z)

    def reparameterization(self, mu, log_var):
        epsilon = torch.randn_like(log_var)
        z = mu + epsilon * torch.sqrt(log_var.exp())
        return z

    def loss(self, X, mu_prime, mu, log_var):
        reconstruction_loss = torch.mean((X - mu_prime) ** 2)
        latent_loss = torch.mean(0.5 * (log_var.exp() + mu**2 - 1 - log_var).sum(dim=1))
        return reconstruction_loss + latent_loss

    def forward(self, X):
        mu, log_var = self.encoder_forward(X)
        z = self.reparameterization(mu, log_var)
        mu_prime = self.decoder_forward(z)
        return mu_prime, mu, log_var



# # Training loop
# def train(model, optimizer, data_loader, device):
#     model.train()
#     total_loss = 0
#     pbar = tqdm(data_loader)
#     for X, _ in pbar:
#         X = X.to(device)
#         model.zero_grad()

#         mu_prime, mu, log_var = model(X)
#         loss = model.loss(X, mu_prime, mu, log_var)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         pbar.set_description(f"Loss: {loss.item():.4f}")
#     return total_loss / len(data_loader)

def train(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    pbar = tqdm(data_loader)

    for batch in pbar:
        X = batch[0]  # Ignore labels, only take images
        X = X.to(device)
        model.zero_grad()

        # Forward pass
        mu_prime, mu, log_var = model(X)

        # Compute loss
        loss = model.loss(X, mu_prime, mu, log_var)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_description(f"Loss: {loss.item():.4f}")
    
    return total_loss / len(data_loader)




@torch.no_grad()
def save_results(model, latent_size, device, save_path='./img/vae_samples.png'):
    z = torch.randn(100, latent_size).to(device)
    out = model.decoder_forward(z)
    save_image(out.view(-1, 3, 54, 54), save_path, nrow=10, normalize=True)


# Main function
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Hyperparameters
    base_path = "../newdata"  # Path to your dataset
    batch_size = 64
    epochs = 50
    latent_size = 64
    lr = 0.001

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(base_path, batch_size=batch_size, train_split=0.8)
    for batch in train_loader:
        print(f"Batch shape: {batch.shape}")  # Should print [batch_size, 3, 54, 54]
        break
    
    
    # Initialize model and optimizer
    model = ConvVAE(latent_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    print("Start Training...")
    for epoch in range(1, epochs + 1):
        avg_loss = train(model, optimizer, train_loader, device)
        print(f"Epoch: {epoch}, AvgLoss: {avg_loss:.4f}")
    print("Training complete.")

    # Save results
    os.makedirs('./img', exist_ok=True)
    save_results(model, latent_size, device)
    print("Sample images saved to './img/vae_samples.png'")


if __name__ == '__main__':
    main()
