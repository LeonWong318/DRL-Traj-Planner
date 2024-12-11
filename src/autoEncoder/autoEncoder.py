import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class SequentialEncoder(nn.Module):
    def __init__(self, seq_len, num_input_channels, base_channel_size, latent_dim, act_fn=nn.GELU):
        super().__init__()
        self.seq_len = seq_len
        c_hid = base_channel_size
        self.conv = nn.Sequential(
            nn.Conv3d(num_input_channels, c_hid, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(c_hid),
            act_fn(),
            nn.Conv3d(c_hid, 2 * c_hid, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(2 * c_hid),
            act_fn(),
            nn.Flatten(),
            nn.Linear(2 * c_hid * (seq_len) * 7 * 7, latent_dim)
        )

    def forward(self, x):
        return self.conv(x)

class SequentialDecoder(nn.Module):
    def __init__(self, future_len, num_input_channels, base_channel_size, latent_dim, act_fn=nn.GELU):
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * c_hid * future_len * 7 * 7),
            act_fn()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(2 * c_hid, c_hid, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)),
            nn.BatchNorm3d(c_hid),
            act_fn(),
            nn.ConvTranspose3d(c_hid, num_input_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0], -1, future_len, 7, 7)  # Reshape to 3D
        return self.deconv(x)

class Autoencoder(pl.LightningModule):
    def __init__(self, seq_len, future_len, base_channel_size, latent_dim, num_input_channels):
        super().__init__()
        self.encoder = SequentialEncoder(seq_len, num_input_channels, base_channel_size, latent_dim)
        self.decoder = SequentialDecoder(future_len, num_input_channels, base_channel_size, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        input_seq, target_seq = batch
        prediction = self.forward(input_seq)
        loss = F.mse_loss(prediction, target_seq, reduction="mean")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=10, min_lr=1e-5
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)