
from dataLoad import create_dataloaders
from autoEncoder import Autoencoder

base_path = "../data"
train_loader, test_loader = create_dataloaders(base_path, batch_size=64)


