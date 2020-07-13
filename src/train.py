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
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

from architecture import Generator, Discriminator, weights_init
from data_loading import ImageFeatureFolder

from datetime import datetime


parser = argparse.ArgumentParser("The best N Group - N1")

parser.add_argument('--dataset-dir',  type=str, default='../celeba')
parser.add_argument('--result-dir', type=str, default='./fake_samples')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
parser.add_argument('--checkpoint-prefix', type=str, default=datetime.now().strftime("%d-%m-%Y_%H_%M_%S"))
parser.add_argument('-s', '--save-checkpoints', dest='save_checkpoints', action='store_true')
parser.add_argument('--nrs', '--no-random-sample', dest='random_sample', action='store_false',
                    help='save random samples of fake faces during training')
parser.add_argument('--ii', '--training-info-interval', dest='training_info_interval', type=int, default=800,
                    help='controls how often during an epoch info is printed')
parser.add_argument('--si', '--sample-interval', dest='sample_interval', type=int, default=1500,
                    help='controls how often during an epoch sample images are saved ')
parser.add_argument('--condition-file', type=str, default='./list_attr_celeba.txt')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--nz', type=int, default=100)  # number of noise dimension
parser.add_argument('--nc', type=int, default=3)  # number of result channel
parser.add_argument('--nfeature', type=int, default=40)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--seed', dest='manual_seed', type=int, required=False)
parser.add_argument('-g', '--generator-path', dest='generator_path', help='use pretrained generator')
parser.add_argument('-d', '--discriminator-path', dest='discriminator_path', help='use pretrained discriminator')
parser.add_argument('--no-label-smoothing', dest='label_smoothing', action='store_false')
parser.add_argument('--no-label-flipping', dest='label_flipping', action='store_false')

parser.add_argument('--print-loss', dest='print_loss', action='store_true')
parser.add_argument('--show-loss-plot', dest='show_loss_plot', action='store_true')

parser.add_argument('--fixed-noise-sample', dest='fixed_noise_sample', action='store_true',
                    help='show model progression by generating samples with the same fixed noise vector during training')
parser.add_argument('--target-image-size', type=int, default=64)
parser.add_argument('--gf', '--generator-filters', dest='generator_filters', type=int, default=64)
parser.add_argument('--df', '--discriminator-filters', dest='discriminator_filters', type=int, default=64)
parser.add_argument('--no-hd-crop', dest='hd_crop', action='store_false')


parser.set_defaults(save_checkpoints=False, random_sample=True, label_flipping=True,
                    label_smoothing=True, print_loss=False, show_loss_plot=False, fixed_noise_sample=False, hd_crop=True)

# beta1 is a hyperparameter, suggestion from hack repo is 0.5
betas = (0.5, 0.99)  # adam optimizer beta1, beta2


class Trainer:
    def __init__(self):
        self.generator = Generator(config).to(device)
        self.discriminator = Discriminator(config).to(device)
        # experiment with different loss functions, TODO: test out Wasserstein loss
        self.loss = nn.BCELoss().to(device)
        self.optimizer_generator = optim.Adam(self.generator.parameters(), lr=config.lr, betas=betas)
        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=config.lr, betas=betas)

        self.generator.apply(weights_init)
        if config.generator_path is not None:
            self.generator.load_state_dict(torch.load(config.generator_path))
        # print("Generator: ", self.generator)

        self.discriminator.apply(weights_init)
        if config.discriminator_path is not None:
            self.discriminator.load_state_dict(torch.load(config.discriminator_path))
        # print("Discriminator: ", self.discriminator)

        self.LOG = {
            "loss_descriminator": [],
            "loss_generator": []
        }

    def randomly_flip_labels(self, labels, p: float = 0.05):
        number_of_labels_to_flip = int(p * labels.shape[0])
        indices_to_flip = random.choices([i for i in range(labels.shape[0])], k=number_of_labels_to_flip)
        # flip chosen labels
        labels[indices_to_flip] = 1 - labels[indices_to_flip]
        return labels

    def train(self, dataloader):
        # for progress visualization
        fixed_noise = torch.randn(config.batch_size, config.nz, 1, 1, device=device)
        fixed_attr = torch.FloatTensor(config.nfeature, config.batch_size).uniform_(0, 2).gt(1).int().float().to(device)

        z_noise = Variable(FloatTensor(config.batch_size, config.nz, 1, 1)).to(device)
        generator_target = Variable(FloatTensor(config.batch_size, 1).fill_(1)).to(device)
        discriminator_target_real = Variable(FloatTensor(config.batch_size, 1).fill_(1)).to(device)
        discriminator_target_fake = Variable(FloatTensor(config.batch_size, 1).fill_(0)).to(device)
        for epoch in range(config.epochs):
            for i, (data, attr) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}"):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                self.discriminator.zero_grad()

                if config.label_smoothing:
                    discriminator_target_real.data.uniform_(0.7, 1.0)  # one-sided label smoothing trick
                if config.label_flipping:
                    discriminator_target_real.data = self.randomly_flip_labels(
                        discriminator_target_real.data, p=0.05)  # label flipping trick
                    discriminator_target_fake.data = self.randomly_flip_labels(
                        discriminator_target_fake.data, p=0.05)  # label flipping trick

                z_noise.data.normal_(0, 1)

                attr = Variable(attr).to(device)
                real_faces = Variable(data).to(device)
                d_real_faces = self.discriminator(real_faces, attr, config)

                fake_faces = self.generator(z_noise, attr, config)
                d_fake_faces = self.discriminator(fake_faces.detach(), attr, config)  # not update generator

                d_loss = self.loss(d_real_faces, discriminator_target_real) + \
                    self.loss(d_fake_faces, discriminator_target_fake)
                d_loss.backward()
                self.optimizer_discriminator.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################

                self.generator.zero_grad()

                # TODO: test if we want to train the generator with new fake faces or instead use the ones we already used wiht the discriminator
                # noise.data.normal_(0, 1)
                # fake_faces = self.generator(z_noise, attr, config)

                d_fake = self.discriminator(fake_faces, attr, config)
                g_loss = self.loss(d_fake, generator_target)
                g_loss.backward()
                self.optimizer_generator.step()
                self.batch_training_info_and_samples(epoch, i, g_loss, d_loss, config,
                                                     fake_faces, fixed_noise, fixed_attr)
            self.epoch_training_info_and_samples(epoch, g_loss, d_loss, config, fake_faces, fixed_noise, fixed_attr)
            ######### epoch finished ##########

    def batch_training_info_and_samples(self, epoch, batch, g_loss, d_loss, config, fake_faces, fixed_noise, fixed_attr):
        self.LOG["loss_generator"].append(g_loss)
        self.LOG["loss_descriminator"].append(d_loss)

        if config.print_loss and batch > 0 and batch % config.training_info_interval == 0:
            if config.print_loss:
                tqdm.write(
                    f"epoch {epoch+1} batch {batch} | generator loss: {g_loss} | discriminator loss: {d_loss}")
        if batch % config.sample_interval == 0:
            if config.random_sample:
                vutils.save_image(
                    fake_faces.data[:min(config.batch_size, 32)], f'{config.result_dir}/{config.checkpoint_prefix}/result_epoch_{epoch + 1}_batch_{batch}.png', normalize=True)
            if config.fixed_noise_sample:
                with torch.no_grad():
                    fixed_fake = self.generator(fixed_noise, fixed_attr, config)
                    vutils.save_image(fixed_fake.detach()[:min(config.batch_size, 32)],
                                      f'{config.result_dir}/{config.checkpoint_prefix}/fixed_noise_result_epoch_{epoch + 1}_batch_{batch}.png', normalize=True)

    def epoch_training_info_and_samples(self, epoch, g_loss, d_loss, config, fake_faces, fixed_noise, fixed_attr):
        for key, values in self.LOG.values():
            plt.plot(values, label=key)
        plt.legend(loc="upper left")
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.savefig(f"{config.result_dir}/{config.checkpoint_prefix}/loss_visualization_{epoch}.png")

        if config.show_loss_plot:
            plt.show()

        if config.print_loss:
            tqdm.write(f"epoch {epoch+1} | generator loss: {g_loss} | discriminator loss: {d_loss}")
        if config.random_sample:
            vutils.save_image(
                fake_faces.data[:min(config.batch_size, 32)], f'{config.result_dir}/{config.checkpoint_prefix}/result_epoch_{epoch + 1}.png', normalize=True)
        if config.fixed_noise_sample:
            with torch.no_grad():
                fixed_fake = self.generator(fixed_noise, fixed_attr, config)
                vutils.save_image(fixed_fake.detach()[:min(config.batch_size, 32)],
                                  f'{config.result_dir}/{config.checkpoint_prefix}/fixed_noise_result_epoch_{epoch + 1}.png', normalize=True)
        if config.save_checkpoints:
            torch.save(self.generator.state_dict(),
                       f'{config.checkpoint_dir}/{config.checkpoint_prefix}/generator_epoch_{epoch+1}.pt')
            torch.save(self.discriminator.state_dict(),
                       f'{config.checkpoint_dir}/{config.checkpoint_prefix}/discriminator_epoch_{epoch+1}.pt')


if __name__ == '__main__':
    config, _ = parser.parse_known_args()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        FloatTensor = torch.cuda.FloatTensor
        print("Running on the GPU")
    else:
        FloatTensor = torch.FloatTensor
        device = torch.device("cpu")
        print("Running on the CPU")

    if config.manual_seed is None:
        config.manual_seed = random.randint(1, 10000)
        print("Random Seed: ", config.manual_seed)

    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)

    # Boost performace by selecting optimal torch-internal algorithms for hardware config
    # This adds a bit of overhaed at the beginning, but is faster at every other iteration
    # Training data needs to be of constant shape for full effect
    cudnn.benchmark = True

    # Create dirs if not already there
    if(config.random_sample or config.fixed_noise_sample):
        print(f"Sample fake images will be saved to {config.result_dir}/{config.checkpoint_prefix}")
        os.makedirs(f"{config.result_dir}/{config.checkpoint_prefix}", exist_ok=True)
    if(config.save_checkpoints):
        print(f"Checkpoints will be saved to {config.checkpoint_dir}/{config.checkpoint_prefix}")
        os.makedirs(f"{config.checkpoint_dir}/{config.checkpoint_prefix}", exist_ok=True)

    print("Loading Data")
    transformers = [] if config.hd_crop else [transforms.CenterCrop(178)]
    transformers.extend([transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = ImageFeatureFolder(config.dataset_dir, config.condition_file, transform=transforms.Compose(transformers))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, num_workers=config.workers, drop_last=True, shuffle=True, pin_memory=True)

    print("Starting Training")
    trainer = Trainer()
    trainer.train(dataloader)
