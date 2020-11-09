import torch
from torch import nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Encoder with one convolutional layer and two fully connected feedforward layers
    """

    def __init__(self, conv1_channel, conv1_filter, pool1_filter, conv2_channel, conv2_filter,
                 pool2_filter, conv3_channel, conv3_filter, num_fc1, num_mean, num_sd, num_fc2, out_size, input_size=32):
        """
        initialize layer

        Args:
            num_channel: number of convolutional channels of the encoder
            filter_size: size of each convolutional filters
            num_ff: number of units in fully-connected layer
        """
        super().__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=conv1_channel, kernel_size=conv1_filter)
        self.pool1 = nn.MaxPool2d(kernel_size=pool1_filter)
        self.conv2 = nn.Conv2d(in_channels=conv1_channel, out_channels=conv2_channel, kernel_size=conv2_filter)
        self.pool2 = nn.MaxPool2d(kernel_size=pool2_filter)
        self.conv3 = nn.Conv2d(in_channels=conv2_channel, out_channels=conv3_channel, kernel_size=conv3_filter)

        self.fc = nn.Linear(conv3_channel, num_fc1)
        self.fc_mean = nn.Linear(num_fc1, num_mean)
        self.fc_sd = nn.Linear(num_fc1, num_sd)
        self.decoder_fc1 = nn.Linear(num_mean, num_fc1)
        self.decoder_fc2 = nn.Linear(num_fc1, num_fc2)
        self.decoder_fc3 = nn.Linear(num_fc2, out_size)

    def encoder(self, input):
        """
        run input through 3 convolutional layers and a fully-connected feedforward layer

        Args:
            input (torch.Tensor): 3 x self.input_size x self.input_size tensor (3 x 32 x 32)
        Returns:
            mean layer (1 x mean_size) and sd layer (1 x sd_size)
        """

        out = self.conv1(input)
        out = out.relu()
        out = self.pool1(out)
        out = self.conv2(out)
        out = out.relu()
        out = self.pool2(out)
        out = self.conv3(out)
        out = out.relu()
        out = torch.squeeze(out)

        out = self.fc(out)
        out = out.relu()  # out should be a shape of 1 x num_fc
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
        out = self.decoder_fc1(z)
        out = out.relu()
        out = self.decoder_fc2(out)
        out = out.relu()
        out = self.decoder_fc3(out)
        out = torch.reshape(out, [-1, 3, self.input_size, self.input_size])
        out = torch.sigmoid(out)

        return out

    def forward(self, input, mean_size=4, sd_size=4):
        mean_vec, logvar_vec = self.encoder(input)
        z = self.reparameterize(mean_vec, logvar_vec)
        recon = self.decoder(z)

        return recon, mean_vec, logvar_vec


def ELBO(reconstruct_input, input, mean_vec, logvar_vec):

    recon_loss = F.mse_loss(reconstruct_input, input, reduction='mean')
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