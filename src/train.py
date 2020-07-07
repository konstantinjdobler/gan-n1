import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
import torchvision.utils as vutils
import torchvision.transforms as transforms

from architecture import Generator, Discriminator
from data_loading import ImageFeatureFolder

parser = argparse.ArgumentParser("cDCGAN")

parser.add_argument('--dataset_dir', type=str, default='../celeba')
parser.add_argument('--result_dir', type=str, default='./celeba_result')
parser.add_argument('--condition_file', type=str,
                    default='./list_attr_celeba.txt')

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--nepoch', type=int, default=20)
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--nz', type=int, default=100)  # number of noise dimension
parser.add_argument('--nc', type=int, default=3)  # number of result channel
parser.add_argument('--nfeature', type=int, default=40)
parser.add_argument('--lr', type=float, default=0.0002)
betas = (0.0, 0.99)  # adam optimizer beta1, beta2

config, _ = parser.parse_known_args()
device = "cuda"

class Trainer:
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.loss = nn.MSELoss()
        self.optimizer_g = optim.Adam(
            self.generator.parameters(), lr=config.lr, betas=betas)
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(), lr=config.lr, betas=betas)

        self.generator.to(device)
        self.discriminator.to(device)
        self.loss.to(device)

    def train(self, dataloader):
        noise = Variable(torch.FloatTensor(config.batch_size, config.nz, 1, 1).to(device))
        label_real = Variable(torch.FloatTensor(config.batch_size, 1).fill_(1).to(device))
        label_fake = Variable(torch.FloatTensor(config.batch_size, 1).fill_(0).to(device))
        for epoch in range(config.nepoch):
            for i, (data, attr) in enumerate(dataloader, 0):
                # train discriminator
                self.discriminator.zero_grad()

                batch_size = data.size(0)
                label_real.data.resize(batch_size, 1).fill_(1)
                label_fake.data.resize(batch_size, 1).fill_(0)
                noise.data.resize_(batch_size, config.nz, 1, 1).normal_(0, 1)

                attr = Variable(attr.to(device))
                real = Variable(data.to(device))
                d_real = self.discriminator(real, attr)

                fake = self.generator(noise, attr)
                d_fake = self.discriminator(
                    fake.detach(), attr)  # not update generator

                d_loss = self.loss(d_real, label_real) + \
                    self.loss(d_fake, label_fake)  # real label
                d_loss.backward()
                self.optimizer_d.step()

                # train generator
                self.generator.zero_grad()
                d_fake = self.discriminator(fake, attr)
                # trick the fake into being real
                g_loss = self.loss(d_fake, label_real)
                g_loss.backward()
                self.optimizer_g.step()
                if i % 50 == 0:
                    print(f"[{i}/{len(dataloader)}] batches | epoch{epoch} batch{i} pictures saved")
                    vutils.save_image(fake.data, '{}/result_epoch_{:03d}_batch_{:03d}.png'.format(config.result_dir, epoch, i), normalize=True)
            print("epoch{:03d} d_real: {}, d_fake: {}".format(
                epoch, d_real.mean(), d_fake.mean()))
            vutils.save_image(
                fake.data, '{}/result_epoch_{:03d}.png'.format(config.result_dir, epoch), normalize=True)


if __name__ == '__main__':
    dataset = ImageFeatureFolder(config.dataset_dir, config.condition_file, transform=transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, num_workers=config.workers, drop_last=True, shuffle=True, pin_memory=True)
    trainer = Trainer()
    trainer.train(dataloader)