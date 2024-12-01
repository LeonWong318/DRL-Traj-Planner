import torch
import matplotlib.pyplot as plt
import numpy as np
from VAE1 import VAE
import torch.nn.functional as F
from dataLoad import create_test_loader

model = VAE(latent_dim = 20)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('vae_model.pth', map_location=device))

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.eval()


# load testset
base_path = "../test_data"  # Path to the dataset
batch_size = 32
# Create DataLoader for the test set

test_dataloader = create_test_loader(base_path, batch_size=batch_size)

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_kld = 0
    total_recon_loss = 0
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            recon_x, mu, logvar = model(x)
            recon_loss = F.mse_loss(recon_x, x, reduction='sum')
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kld_loss
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kld += kld_loss.item()
    avg_loss = total_loss / len(dataloader.dataset)
    avg_recon_loss = total_recon_loss / len(dataloader.dataset)
    avg_kld = total_kld / len(dataloader.dataset)
    print(f"Test Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, KLD Loss: {avg_kld:.4f}")


evaluate_model(model, test_dataloader, device)

data_iter = iter(test_dataloader)
x = next(data_iter)
x = x.to(device)
with torch.no_grad():
    recon_x, _, _ = model(x)
x = x.cpu().numpy()
recon_x = recon_x.cpu().numpy()
plt.figure(figsize=(16, 4))
for i in range(8):
    ax = plt.subplot(2, 8, i + 1)
    plt.imshow(np.transpose(x[i], (1, 2, 0)))
    plt.title("Original")
    plt.axis('off')
    ax = plt.subplot(2, 8, i + 9)
    plt.imshow(np.transpose(recon_x[i], (1, 2, 0)))
    plt.title("Reconstructed")
    plt.axis('off')
plt.show()
