import os
from time import sleep
import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm

from architecture import Generator, Discriminator
from data_loading import ImageFeatureFolder


parser = argparse.ArgumentParser("The best N Group - N1")

parser.add_argument('--dataset_dir', type=str, default='../celeba')
parser.add_argument('--result_dir', type=str, default='./celeba_result')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
parser.add_argument('--checkpoint_prefix', type=str, default='')
parser.add_argument('--save_checkpoints', type=bool, default=False)
parser.add_argument('--condition_file', type=str, default='./list_attr_celeba.txt')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--nepoch', type=int, default=20)
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--nz', type=int, default=100)  # number of noise dimension
parser.add_argument('--nc', type=int, default=3)  # number of result channel
parser.add_argument('--nfeature', type=int, default=40)
parser.add_argument('--lr', type=float, default=0.0002)
betas = (0.0, 0.99)  # adam optimizer beta1, beta2




class Trainer:
    def __init__(self):
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)
        self.loss = nn.MSELoss()
        self.optimizer_generator = optim.Adam(self.generator.parameters(), lr=config.lr, betas=betas)
        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=config.lr, betas=betas)

        self.generator.to(device)
        self.discriminator.to(device)
        self.loss.to(device)

    def train(self, dataloader):
        noise = Variable(FloatTensor(config.batch_size, config.nz, 1, 1).to(device))
        label_real = Variable(FloatTensor(config.batch_size, 1).fill_(1).to(device))
        label_fake = Variable(FloatTensor(config.batch_size, 1).fill_(0).to(device))
        for epoch in range(config.nepoch):
            for i, (data, attr) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}"):
                # train discriminator
                self.discriminator.zero_grad()
                batch_size = data.size(0)
                label_real.data.resize_(batch_size, 1).fill_(1)
                label_fake.data.resize_(batch_size, 1).fill_(0)
                noise.data.resize_(batch_size, config.nz, 1, 1).normal_(0, 1)

                attr = Variable(attr.to(device))
                real = Variable(data.to(device))
                d_real = self.discriminator(real, attr)

                fake = self.generator(noise, attr, config)
                d_fake = self.discriminator(fake.detach(), attr)  # not update generator

                d_loss = self.loss(d_real, label_real) + self.loss(d_fake, label_fake)  # real label
                d_loss.backward()
                self.optimizer_discriminator.step()

                # train generator
                self.generator.zero_grad()
                d_fake = self.discriminator(fake, attr)
                # trick the fake into being real
                g_loss = self.loss(d_fake, label_real)
                g_loss.backward()
                self.optimizer_generator.step()
                # if i % 50 == 0:
                #   tqdm.write(f"[{i}/{len(dataloader)}] batches | epoch {epoch +1} batch{i} pictures saved")
                #   vutils.save_image(
                #     fake.data, f'{config.result_dir}/result_epoch_{epoch + 1}_batch_{i+1}.png', normalize=True)

            tqdm.write(f"epoch{epoch +1} d_real: {d_real}, d_fake: {d_fake}")
            vutils.save_image(fake.data, f'{config.result_dir}/result_epoch_{epoch + 1}.png', normalize=True)
            # do checkpointing
            if config.save_checkpoints:
                torch.save(self.generator.state_dict(),
                           f'{config.checkpoint_dir}/{config.checkpoint_prefix}generator_epoch_{epoch+1}.pt')
                torch.save(self.discriminator.state_dict(),
                           f'{config.checkpoint_dir}/{config.checkpoint_prefix}discriminator_epoch_{epoch+1}.pt')


if __name__ == '__main__':
    config, _ = parser.parse_known_args()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        FloatTensor = torch.cuda.FloatTensor
        print("Running on the GPU")
    else:
        FloatTensor = torch.FloatTensor
        device = torch.device("cpu")
        print("Running on the CPU")

    # Create dirs if not already there
    os.makedirs(config.result_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    print("Loading Data")
    dataset = ImageFeatureFolder(config.dataset_dir, config.condition_file, transform=transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, num_workers=config.workers, drop_last=True, shuffle=True, pin_memory=True)
    trainer = Trainer()
    print("Starting Training")
    trainer.train(dataloader)
