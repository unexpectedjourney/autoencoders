import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims=None):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        encoder_blocks = []
        input_dim = in_channels
        for hidden_dim in hidden_dims:
            encoder_block = nn.Sequential(
                nn.Conv2d(
                    input_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.LeakyReLU(),
            )
            encoder_blocks.append(encoder_block)
            input_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_blocks)
        self.lin_mu = nn.Linear(hidden_dims[-1]*16, latent_dim)
        self.lin_var = nn.Linear(hidden_dims[-1]*16, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*16*16)

        decoder_blocks = []
        for i in range(len(hidden_dims) - 1, 0, -1):
            if i == 0:
                out_channels = hidden_dims[i]
            else:
                out_channels = hidden_dims[i-1]
            decoder_block = nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[i],
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
            )
            decoder_blocks.append(decoder_block)

        self.decoder = nn.Sequential(*decoder_blocks)

        self.final_layer = nn.Sequential(
            nn.Conv2d(
                hidden_dims[0],
                out_channels=3,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.Tanh()
        )

    def encode(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        mu = self.lin_mu(result)
        log_var = self.lin_var(result)

        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 16, 16)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), mu, log_var]

    @staticmethod
    def loss_function(reconstructed, x, mu, log_var, size):
        reconstruction_loss = F.mse_loss(reconstructed, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = reconstruction_loss + kld_loss
        return {
            'loss': loss,
            'reconstruction_loss': reconstruction_loss.detach(),
            'kld': -kld_loss.detach()
        }

    def generate(self, x):
        return self.forward(x)[0]

    def sample(self, z):
        return self.decode(z)
