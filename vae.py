import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from torchvision import transforms
import numpy as np
import torch.optim as optim

INPUT_DIM = 28 * 28
batch_size = 32

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(INPUT_DIM, 100)
        self.linear2 = nn.Linear(100, 25)

    def forward(self, X):
        hidden1 = F.relu(self.linear1(X))
        latent = F.relu(self.linear2(hidden1))

        return latent

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(25, 100)
        self.linear2 = nn.Linear(100, INPUT_DIM)

    def forward(self, X):
        hidden1 = F.relu(self.linear1(X))
        reconstructed_output = F.relu(self.linear2(hidden1))

        return reconstructed_output

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.mean = nn.Linear(25, 25)
        self.log_variance = nn.Linear(25, 25)

    # X is the input images
    def forward(self, X):
        hidden = self.encoder(X)

        mu = self.mean(hidden)
        log_var = self.log_variance(hidden)

        var = torch.exp(log_var)

        noise = torch.from_numpy(np.random.normal(0, 1, size=var.size())).float()

        latent_space = mu + var * noise

        recon_output = self.decoder(latent_space)

        self.mu = mu
        self.var = var

        return recon_output


transform = transforms.Compose(
    [transforms.ToTensor()])

mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)

dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

vae = VAE()

def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


recon_loss = nn.MSELoss()
optimizer = optim.Adam(vae.parameters(), lr=0.0001)

for epoch_i in range(30):
    for data_batch in dataloader:
        X, y = data_batch

        X = X.reshape(batch_size, INPUT_DIM)

        optimizer.zero_grad()

        X_recon = vae(X)

        loss = recon_loss(X_recon, X) + latent_loss(vae.mu, vae.var)
        loss.backward()

        optimizer.step()

    print(epoch_i, loss.item())

