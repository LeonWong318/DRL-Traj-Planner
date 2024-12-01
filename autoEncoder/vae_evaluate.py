import torch
import matplotlib.pyplot as plt
import numpy as np
from VAE import VAE
import torch.nn.functional as F
from dataLoad import create_test_loader

# 加载模型
model = VAE(latent_dim=64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('vae_model.pth', map_location=device))
model.to(device)
model.eval()

# 加载测试集
base_path = "../newdata" 
# base_path = "../test_data"  # 数据集路径
batch_size = 32
test_loader = create_test_loader(base_path, batch_size=batch_size)

# 测试一批数据
# data = next(iter(test_loader))
# data = data.to(device)  # 确保数据也在正确的设备上
# with torch.no_grad():
#     reconstructed, _, _ = model(data)

# 绘制原始图像和重建图像
# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
# for i in range(5):
#     # 显示原始图像
#     ax = axes[0, i]
#     original_img = data[i].cpu().permute(1, 2, 0)  # 移动到CPU并调整通道位置
#     ax.imshow(original_img)
#     ax.axis('off')
#     ax.set_title('Original')

#     # 显示重建图像
#     ax = axes[1, i]
#     reconstructed_img = reconstructed[i].cpu().permute(1, 2, 0)  # 移动到CPU并调整通道位置
#     ax.imshow(reconstructed_img)
#     ax.axis('off')
#     ax.set_title('Reconstructed')

# plt.show()

test_iter = iter(test_loader)
batch = next(test_iter).to(device)

with torch.no_grad():
    reconstructed,_,_ = model(batch)  
    
# plot each channel independently
num_samples = 10
for i in range(min(num_samples, len(batch))):
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