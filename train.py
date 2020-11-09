import argparse
import os
from datetime import datetime

import torch
from torch import optim
import torchvision.transforms as transforms
from tqdm import trange
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.tensorboard import SummaryWriter


from model import VAE, ELBO, mmd
from utils import cycle, tile_images

## get CIFAR-10 dataset
# trainset = CIFAR10('.', train=True, download=True, transform=transforms.ToTensor())
# testset = CIFAR10('.', train=False, download=True, transform=transforms.ToTensor())

transform = transforms.Compose([
    lambda img: img.convert("RGB"),
    transforms.Resize(32),
    transforms.ToTensor()
])
trainset = MNIST('.', train=True, download=True, transform=transform)
testset = MNIST('.', train=False, download=True, transform=transform)

## Make the dataset iterable
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)


def train(logdir, num_epoch, conv1_channel, conv1_filter, pool1_filter, conv2_channel, conv2_filter,
          pool2_filter, conv3_channel, conv3_filter, num_fc1, num_mean, num_sd, num_fc2, out_size, beta, gamma):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VAE(conv1_channel, conv1_filter, pool1_filter, conv2_channel, conv2_filter,
                pool2_filter, conv3_channel, conv3_filter, num_fc1, num_mean, num_sd, num_fc2, out_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter(logdir)
    iteration = 0

    z_samples = torch.randn(16, num_mean).to(device)

    with trange(num_epoch) as loop:
        for epoch in loop:
            for num_batch, (images, _) in enumerate(train_loader):

                optimizer.zero_grad()
                images = images.to(device)
                recon, mean_vec, logvar_vec = model.forward(images)
                recon_loss, KL = ELBO(recon, images, mean_vec, logvar_vec)
                z = model.reparameterize(mean_vec, logvar_vec)
                z_samples = torch.randn_like(z)
                MMD = mmd(z, z_samples)

                loss = recon_loss + beta * KL + gamma * MMD

                loss.backward()
                optimizer.step()

                writer.add_scalar('loss', loss.item(), global_step=iteration)
                writer.add_scalar('loss/recon', recon_loss.item(), global_step=iteration)
                writer.add_scalar('loss/KL', KL.item(), global_step=iteration)
                writer.add_scalar('loss/MMD', MMD.item(), global_step=iteration)

                if iteration % 100 == 0:
                    print("epoch ", epoch, "recon_loss: ", recon_loss.item(), "KL: ", KL.item(), "MMD:", MMD.item(), "loss", loss.item())

                    if num_batch == 0:
                        recon = torch.stack([trainset[i][0] for i in range(16)])
                        writer.add_image('images/original', tile_images(recon))

                    recon = model.forward(torch.stack([trainset[i][0] for i in range(16)]).to(device))[0].cpu()
                    writer.add_image('images/reconstructions', tile_images(recon.detach()), global_step=iteration)
                    generated = model.decoder(z_samples).cpu()
                    writer.add_image('image/generated', tile_images(generated.detach()), global_step=iteration)

                    torch.save(model.state_dict(), os.path.join(logdir, "model.pt"))

                iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CIFAR10 VAE implementation for Fall 2020 Info Theory Final Project",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--logdir', type=str, default=os.path.join('runs', datetime.now().strftime('%y%m%d-%H%M%S')),
                        help='the directory to save the logs')
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--beta', type=int, default=1)
    parser.add_argument('--gamma', type=int, default=0)
    parser.add_argument('--conv1_channel', type=int, default=16)
    parser.add_argument('--conv1_filter', type=int, default=4)
    parser.add_argument('--pool1_filter', type=int, default=2)
    parser.add_argument('--conv2_channel', type=int, default=32)
    parser.add_argument('--conv2_filter', type=int, default=6)
    parser.add_argument('--pool2_filter', type=int, default=4)
    parser.add_argument('--conv3_channel', type=int, default=64)
    parser.add_argument('--conv3_filter', type=int, default=2)
    parser.add_argument('--num_mean', type=int, default=8)
    parser.add_argument('--num_sd', type=int, default=8)
    parser.add_argument('--num_fc1', type=int, default=16)
    parser.add_argument('--num_fc2', type=int, default=64)
    parser.add_argument('--out_size', type=int, default=3072)

    train(**vars(parser.parse_args()))
