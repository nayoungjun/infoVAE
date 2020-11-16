import torch
from torch import nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Encoder with one convolutional layer and two fully connected feedforward layers
    """

    def __init__(self, mc, num_latent, input_size=32):
        """
        initialize layer

        Args:
            mc: model complexity
            num_latent: dimension of the latent space
            num_ff: number of units in fully-connected layer
        """
        super().__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=mc * 2, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=mc * 2, out_channels=mc * 4, kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels=mc * 4, out_channels=mc * 8, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=mc * 8, out_channels=mc * 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=mc * 16, out_channels=mc * 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.fc_mean = nn.Linear(mc * 32, num_latent)
        self.fc_sd = nn.Linear(mc * 32, num_latent)

        self.convt1 = nn.ConvTranspose2d(in_channels=num_latent, out_channels=mc * 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(in_channels=mc * 16, out_channels=mc * 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(in_channels=mc * 8, out_channels=mc * 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convt4 = nn.ConvTranspose2d(in_channels=mc * 4, out_channels=mc * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convt5 = nn.ConvTranspose2d(in_channels=mc * 2, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)


    def encoder(self, input):
        """
        run input through 3 convolutional layers and a fully-connected feedforward layer

        Args:
            input (torch.Tensor): 3 x self.input_size x self.input_size tensor (3 x 32 x 32)
        Returns:
            mean layer (1 x mean_size) and sd layer (1 x sd_size)
        """

        out = self.pool(self.conv1(input).relu())
        out = self.pool(self.conv2(out).relu())
        out = self.pool(self.conv3(out).relu())
        out = self.pool(self.conv4(out).relu())
        out = torch.squeeze(self.pool(self.conv5(out).relu()))

        out_mean = self.fc_mean(out)
        out_logvar = self.fc_sd(out)
        out_logvar = torch.exp(out_logvar)

        return out_mean, out_logvar

    def reparameterize(self, mean_vec, logvar_vec):
        """
        reparameterization trick

        Args:
            mean_vec (torch.Tensor): (1 x mean_size)
            sd_vec (torch.Tensor): (1 x sd_size)
        """

        epsilon = torch.randn_like(logvar_vec)
        z = mean_vec + logvar_vec * epsilon

        return z

    def decoder(self, z):

        out = z.unsqueeze(-1).unsqueeze(-1) # shape = [Batch, num_latent, 1, 1]
        out = self.convt1(out).relu()
        out = self.convt2(out).relu()
        out = self.convt3(out).relu()
        out = self.convt4(out).relu()
        out = self.convt5(out).sigmoid()

        return out

    def forward(self, input):
        mean_vec, logvar_vec = self.encoder(input)
        z = self.reparameterize(mean_vec, logvar_vec)
        recon = self.decoder(z)

        return recon, mean_vec, logvar_vec


def ELBO(reconstruct_input, input, mean_vec, logvar_vec):

    recon_loss = F.binary_cross_entropy(reconstruct_input, input, size_average=False)
    KL = -0.5 * torch.sum(1 + logvar_vec - mean_vec ** 2 - logvar_vec.exp())

    return recon_loss, KL

def mmd(z, z_sample):
    def compute_kernel(z, z_sample):
        kernel_input = (z.unsqueeze(1) - z_sample.unsqueeze(0)).pow(2).mean(2)
        return torch.exp(-kernel_input) # (x_size, y_size)

    zz = compute_kernel(z, z)
    ss = compute_kernel(z_sample, z_sample)
    zs = compute_kernel(z, z_sample)
    return zz.mean() + ss.mean() - 2 * zs.mean()
