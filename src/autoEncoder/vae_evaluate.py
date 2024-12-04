import torch
import matplotlib.pyplot as plt
import numpy as np
from VAE import VAE
import torch.nn.functional as F
from dataLoad import create_test_loader

# 加载模型
model = VAE(latent_dim=128)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('vae_model_128.pth', map_location=device))
model.to(device)
model.eval()

# 加载测试集
base_path = "../newdata" 
# base_path = "../test_data"  # 数据集路径
batch_size = 32
test_loader = create_test_loader(base_path, batch_size=batch_size)


test_iter = iter(test_loader)
batch = next(test_iter).to(device)

with torch.no_grad():
    reconstructed,_,_ = model(batch)  
    
# plot each channel independently
num_samples = 10


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

# for i in range(min(num_samples, len(batch))):
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