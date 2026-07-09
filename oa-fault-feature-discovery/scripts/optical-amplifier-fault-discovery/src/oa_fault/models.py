import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class FaultTCN(nn.Module):
    def __init__(self, n_features, hidden_channels=96, num_blocks=5, kernel_size=3, dropout=0.15):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv1d(n_features, hidden_channels, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[
            TemporalBlock(hidden_channels, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        return self.head(x)


class WindowVAE(nn.Module):
    def __init__(self, n_features, window_len, latent_dim=32, hidden_channels=64):
        super().__init__()
        self.n_features = n_features
        self.window_len = window_len
        flat_dim = n_features * window_len
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_channels * 4),
            nn.ReLU(),
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_channels * 2, latent_dim)
        self.logvar = nn.Linear(hidden_channels * 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels * 2),
            nn.ReLU(),
            nn.Linear(hidden_channels * 2, hidden_channels * 4),
            nn.ReLU(),
            nn.Linear(hidden_channels * 4, flat_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        logvar = torch.clamp(self.logvar(h), min=-10.0, max=10.0)
        return self.mu(h), logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder(z)
        return x.view(-1, self.n_features, self.window_len)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta=0.01):
    rec = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta * kl, rec.detach(), kl.detach()
