import os
from time import sleep

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

from PIL import Image
import numpy as np
import math

import torch.nn.functional as F

from architecture_pgan import Generator, Discriminator, weights_init
from data_loading import AttribDataset

from arg_parser import ArgParser

# beta1 is a hyperparameter, suggestion from hack repo is 0.5
#betas = (0.5, 0.99)  # adam optimizer beta1, beta2

class BaseConfig():
    r"""
    An empty class used for configuration members
    """

    def __init__(self, orig=None):
        if orig is not None:
            print("cawet")

class NumpyResize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return np.array(img.resize(self.size, resample=Image.BILINEAR))

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class NumpyFlip(object):

    def __init__(self, p=0.5):
        self.p = p
        random.seed(None)

    def __call__(self, img):
        if random.random() < self.p:
            return np.flip(img, 1).copy()
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

def isinf(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("The argument is not a tensor", str(tensor))
    return tensor.abs() == math.inf


def isnan(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("The argument is not a tensor", str(tensor))
    return tensor != tensor
    
class NumpyToTensor(object):

    def __init__(self):
        return

    def __call__(self, img):
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)

        return transforms.functional.to_tensor(img)

def finiteCheck(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    for p in parameters:
        infGrads = isinf(p.grad.data)
        p.grad.data[infGrads] = 0

        nanGrads = isnan(p.grad.data)
        p.grad.data[nanGrads] = 0

def WGANGPGradientPenalty(input, fake, discriminator, weight, backward=True):
    r"""
    Gradient penalty as described in
    "Improved Training of Wasserstein GANs"
    https://arxiv.org/pdf/1704.00028.pdf

    Args:

        - input (Tensor): batch of real data
        - fake (Tensor): batch of generated data. Must have the same size
          as the input
        - discrimator (nn.Module): discriminator network
        - weight (float): weight to apply to the penalty term
        - backward (bool): loss backpropagation
    """

    batchSize = input.size(0)
    alpha = torch.rand(batchSize, 1)
    alpha = alpha.expand(batchSize, int(input.nelement() /
                                        batchSize)).contiguous().view(
                                            input.size())
    alpha = alpha.to(input.device)
    interpolates = alpha * input + ((1 - alpha) * fake)

    interpolates = torch.autograd.Variable(
        interpolates, requires_grad=True)

    decisionInterpolate = discriminator(interpolates)
    decisionInterpolate = decisionInterpolate[:, 0].sum()

    gradients = torch.autograd.grad(outputs=decisionInterpolate,
                                    inputs=interpolates,
                                    create_graph=True, retain_graph=True)

    gradients = gradients[0].view(batchSize, -1)
    gradients = (gradients * gradients).sum(dim=1).sqrt()
    gradient_penalty = (((gradients - 1.0)**2)).sum() * weight

    if backward:
        gradient_penalty.backward(retain_graph=True)

    return gradient_penalty.item()

class ProgressiveGAN:
    def __init__(self, config, load_state_path=None):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.latent_vector_dimension = 512
        self.image_channels = 3 #RGB
        self.learning_rate = config['learning_rate']

        self.config = config
        self.config['alpha'] = 0

        self.generator = Generator(output_image_channels = self.image_channels, latent_vector_dimension= self.latent_vector_dimension, initial_layer_channels=config['depth_scales'][0]).to(self.device)
        self.discriminator = Discriminator(input_image_channels = self.image_channels, decision_layer_size= 1, initial_layer_channels=config['depth_scales'][0]).to(self.device)
        
        self.optimizer_generator = self.get_optimizer(self.generator)
        self.optimizer_discriminator = self.get_optimizer(self.discriminator)
        
        # If path is give, load states and overwrite components with the loaded state
        if load_state_path:
            states = Store.load(load_state_path)

            # Bring models to the same architectural structure as the loaded state requires
            for _ in range(in_state['scaleIteration']):
                discriminator.add_new_layer()
                generator.add_new_layer()
            
            # Overwrite Model States
            discriminator.load_state_dict(config['discriminatorState'])
            generator.load_state_dict(config['generatorState'])
            # Overwrite Optimizer States
            optimizer_discriminator.load_state_dict(config['dOptimizer'])
            optimizer_generator.load_state_dict(config['gOptimizer'])

        

    def train_on_batch(self, input_batch):
        input_batch = input_batch.to(self.device)
        batch_size = input_batch.size()[0]
        self.optimizer_discriminator.zero_grad()

        prediction_real_data = self.discriminator(input_batch)
        latent_vector = torch.randn(batch_size, self.latent_vector_dimension).to(self.device)
        generated_fake_data = self.generator(latent_vector).detach()
        prediction_fake_data = self.discriminator(generated_fake_data)

        discriminator_loss = self.loss(prediction_fake_data, prediction_real_data)
        
        WGANGPGradientPenalty(input_batch, generated_fake_data, self.discriminator, self.config['lambda_gp'])
        discriminator_loss.backward(retain_graph=True)
        finiteCheck(self.discriminator.parameters())
        self.optimizer_discriminator.step()

        self.optimizer_discriminator.zero_grad()
        self.optimizer_generator.zero_grad()

        latent_vector = torch.randn(batch_size, self.latent_vector_dimension).to(self.device)
        generated_fake_data = self.generator(latent_vector)
        prediction_fake_data = self.discriminator(generated_fake_data)
        generator_loss = self.loss(prediction_fake_data)
        generator_loss.backward(retain_graph=True)
        finiteCheck(self.generator.parameters())
        self.optimizer_generator.step()
        


    def loss(self, prediction_fake_data, prediction_real_data=None):
        real_data_sum = prediction_real_data[:, 0].sum() if prediction_real_data is not None else 0
        fake_data_sum = prediction_fake_data[:, 0].sum()
        return fake_data_sum - real_data_sum



    def add_new_layer(self, new_layer_channels):
        self.generator.add_new_layer(new_layer_channels)
        self.discriminator.add_new_layer(new_layer_channels)

        self.move_to_device()
    
    def get_optimizer(self, model):
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                        betas=[0, 0.99], lr=self.learning_rate)

    def move_to_device(self):
        r"""
        Move the current networks and solvers to the GPU.
        This function must be called each time generator or discriminator is modified
        """

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.optimizer_discriminator = self.get_optimizer(self.discriminator)
        self.optimizer_generator = self.get_optimizer(self.generator)

        self.optimizer_discriminator.zero_grad()
        self.optimizer_generator.zero_grad()

    def get_size(self):
        return self.generator.get_output_size()

    def update_alpha(self, new_alpha):
        self.generator.set_new_alpha(new_alpha)
        self.discriminator.set_new_alpha(new_alpha)
        self.config['alpha'] = new_alpha

    def generate_image(self, number_of_images):
        input = torch.randn(number_of_images, self.latent_vector_dimension).to(self.device)
        return self.generator(input).detach().cpu()


class Trainer:
    def __init__(self, config):
        self.path_db = config.dataset_dir

        self.start_scale = 0
        self.start_iter = 0
        self.config = config

        self.model_config = {}
        self.model_config['max_iter_at_scale'] = [48000, 96000, 96000, 96000, 96000, 96000, 96000, 96000, 200000]
        #self.model_config['max_iter_at_scale'] = [10, 10, 10, 10, 10, 10, 10, 10, 10]

        # alpha config
        self.model_config['alpha_jump_mode'] = "linear"
        self.model_config['iter_alpha_jump'] = []
        self.model_config['alpha_jump_vals'] = []
        #self.model_config.iter_alpha_jump = [[],[0, 1000, 2000], [0, 1000, 4000, 8000, 16000],[0, 2000, 4000, 8000]]
        #self.model_config.alpha_jump_vals = [[],[1., 0.5, 0], [1, 0.75, 0.5, 0.25, 0.], [1., 0.75, 0.5, 0.]]
        self.model_config['alpha_n_jumps'] = [0, 600, 600, 600, 600, 600, 600, 600, 600]
        self.model_config['alpha_size_jumps'] = [0, 32, 32, 32, 32, 32, 32, 32, 32, 32]

        # base config
        self.model_config['depth_scales'] = [512, 512, 512, 512, 256, 128, 64, 32, 16]
        self.model_config['mini_batch_size'] = 16
        self.model_config['dim_latent_vector'] = 512
        self.model_config['lambda_gp'] = 10
        self.model_config["epsilon_d"] = 0.001
        self.model_config["learning_rate"] = config.lr

        # conditional config
        self.model_config["weight_condition_g"] = 0.0
        self.model_config["weight_condition_d"] = 0.0

        self.update_alpha_jumps(self.model_config['alpha_n_jumps'], self.model_config['alpha_size_jumps'])
        
        self.scale_sanity_check()

        self.init_model()

    def init_model(self):
        self.model = ProgressiveGAN(self.model_config)


    def scale_sanity_check(self):
        n_scales = min(len(self.model_config['depth_scales']),
                    len(self.model_config['max_iter_at_scale']),
                    len(self.model_config['iter_alpha_jump']),
                    len(self.model_config['alpha_jump_vals']))

        self.model_config['depth_scales'] = self.model_config['depth_scales'][:n_scales]
        self.model_config['max_iter_at_scale'] = self.model_config['max_iter_at_scale'][:n_scales]
        self.model_config['iter_alpha_jump'] = self.model_config['iter_alpha_jump'][:n_scales]
        self.model_config['alpha_jump_vals'] = self.model_config['alpha_jump_vals'][:n_scales]

        self.model_config['size_scales'] = [4]
        for scale in range(1, n_scales):
            self.model_config['size_scales'].append(
                self.model_config['size_scales'][-1] * 2)

        self.model_config['n_scales'] = n_scales


    def update_alpha_jumps(self, n_jump_scale, size_jump_scale):
        n_scales = min(len(n_jump_scale), len(size_jump_scale))

        for scale in range(n_scales):
            self.model_config['iter_alpha_jump'].append([])
            self.model_config['alpha_jump_vals'].append([])

            if n_jump_scale[scale] == 0:
                self.model_config['iter_alpha_jump'][-1].append(0)
                self.model_config['alpha_jump_vals'][-1].append(0.0)
                continue

            diff_jump = 1.0 / float(n_jump_scale[scale])
            curr_val = 1.0
            curr_iter = 0

            while curr_val > 0:

                self.model_config['iter_alpha_jump'][-1].append(curr_iter)
                self.model_config['alpha_jump_vals'][-1].append(curr_val)

                curr_iter += size_jump_scale[scale]
                curr_val -= diff_jump

            self.model_config['iter_alpha_jump'][-1].append(curr_iter)
            self.model_config['alpha_jump_vals'][-1].append(0.0)

    def in_scale_update(self, iter, scale, input_real):
        if self.index_jump_alpha < len(self.model_config['iter_alpha_jump'][scale]):
            if iter == self.model_config['iter_alpha_jump'][scale][self.index_jump_alpha]:
                alpha = self.model_config['alpha_jump_vals'][scale][self.index_jump_alpha]
                self.model.update_alpha(alpha)
                self.index_jump_alpha += 1

        if self.model.config['alpha'] > 0:
            low_res_real = F.avg_pool2d(input_real, (2, 2))
            low_res_real = F.upsample(
                low_res_real, scale_factor=2, mode='nearest')

            alpha = self.model.config['alpha']
            input_real = alpha * low_res_real + (1-alpha) * input_real

        return input_real


    def update_dataset_for_scale(self, scale):
        pass
        # self.model_config['mini_batch_size'] = get_min_occurence(
        #     self.miniBatchScheduler, scale, self.model_config['mini_batch_size'])
        # self.path_db = get_min_occurence(
        #     self.datasetProfile, scale, self.path_db)
        # # Scale scheduler
        # if self.configScheduler is not None:
        #     if scale in self.configScheduler:
        #         print("Scale %d, updating the training configuration" % scale)
        #         print(self.configScheduler[scale])
        #         self.model.updateConfig(self.configScheduler[scale])


    def get_db_loader(self, scale):
        dataset = self.get_dataset(scale)
        return torch.utils.data.DataLoader(dataset, batch_size = self.model_config['mini_batch_size'], shuffle=True, num_workers=self.config.workers)

    def get_dataset(self, scale, size=None):
        if size is None:
            size = self.model.get_size()

        print("size", size)
        transform_list = [NumpyResize(size),
                        NumpyToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        transform = transforms.Compose(transform_list)

        return AttribDataset(self.path_db,
                            transform=transform,
                            attribDictPath=None,
                            specificAttrib=None,
                            mimicImageFolder=False)

    def train(self):
        n_scales = len(self.model_config['depth_scales'])

        for scale in range(self.start_scale, n_scales):

            self.update_dataset_for_scale(scale)

            db_loader = self.get_db_loader(scale)
            size_db = len(db_loader)

            shift_iter = 0
            if self.start_iter > 0:
                shift_iter = self.start_iter
                self.start_iter = 0

            shift_alpha = 0
            while shift_alpha < len(self.model_config['iter_alpha_jump'][scale]) and \
                    self.model_config['iter_alpha_jump'][scale][shift_alpha] < shift_iter:
                shift_alpha += 1

            while shift_iter < self.model_config['max_iter_at_scale'][scale]:

                self.index_jump_alpha = shift_alpha
                status = self.train_on_epoch(db_loader, scale,
                                        shift_iter=shift_iter,
                                        max_iter=self.model_config['max_iter_at_scale'][scale])

                if not status:
                    return False

                shift_iter += size_db
                while shift_alpha < len(self.model_config['iter_alpha_jump'][scale]) and \
                        self.model_config['iter_alpha_jump'][scale][shift_alpha] < shift_iter:
                    shift_alpha += 1

            if scale == n_scales - 1:
                break

            self.model.add_new_layer(self.model_config['depth_scales'][scale + 1])

        self.start_scale = n_scales
        self.start_iter = self.model_config['max_iter_at_scale'][-1]
        return True

    def train_on_epoch(self,
                 dbLoader,
                 scale,
                 shift_iter=0,
                 max_iter=-1):
        i = shift_iter

        for item, data in enumerate(dbLoader, 0):

            inputs_real = data[0]
            labels = data[1]

            if inputs_real.size()[0] < self.model_config['mini_batch_size']:
                continue

            # Additionnal updates inside a scale
            inputs_real = self.in_scale_update(i, scale, inputs_real)

            if len(data) > 2:
                mask = data[2]
                allLosses = self.model.train_on_batch(inputs_real)
            else:
                allLosses = self.model.train_on_batch(inputs_real)

            i += 1

            if i % 100 == 0:
                print("Iteration: ", i)

            if i % config.sample_interval == 0:
                #print losses
                self.generate_image(scale, i)

            if i == max_iter:
                return True

        return True

    def generate_image(self, scale, iteration):
        image = self.model.generate_image(self.model_config['mini_batch_size'])
        vutils.save_image(image.data[:self.model_config['mini_batch_size']], f'{config.result_dir}/{config.checkpoint_prefix}/scale' + str(scale) + '_iter' + str(iteration) + '.png', normalize=True)


    def add_new_scales(self, config_new_scales):

        self.update_alpha_jumps(config_new_scales["alpha_n_jumps"],
                                config_new_scales["alpha_size_jumps"])

        self.model_config['depth_scales'] = self.model_config['depth_scales'] + \
            config_new_scales["depth_scales"]
        self.model_config['max_iter_at_scale'] = self.model_config['max_iter_at_scale'] + \
            config_new_scales["max_iter_at_scale"]

        self.scale_sanity_check()

class Store:
    @staticmethod
    def save(generator, discriminator, gOptimizer, dOptimizer, config, scaleIteration):
        out_state = {'scaleIteration': scaleIteration,
                     'config': config,
                     'generatorState': generator.state_dict(),
                     'discriminatorState': discriminator.state_dict(),
                     'generatorOptimizer': gOptimizer.state_dict(),
                     'discriminatorOptimizer': dOptimizer.state_dict()}

        torch.save(out_state, f"../data/cpgan_{scale}.pt")

    @staticmethod
    def load(path):
        return torch.load(path)


# class Trainer:
#     def __init__(self):
#         self.generator = Generator(config).to(device)
#         self.discriminator = Discriminator(config).to(device)
#         # experiment with different loss functions, TODO: test out Wasserstein loss
#         self.loss = nn.BCELoss().to(device)
#         self.optimizer_generator = optim.Adam(self.generator.parameters(), lr=config.lr, betas=betas)
#         self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=config.lr, betas=betas)

#         self.generator.apply(weights_init)
#         if config.generator_path is not None:
#             self.generator.load_state_dict(torch.load(config.generator_path))
#         # print("Generator: ", self.generator)

#         self.discriminator.apply(weights_init)
#         if config.discriminator_path is not None:
#             self.discriminator.load_state_dict(torch.load(config.discriminator_path))
#         # print("Discriminator: ", self.discriminator)

#         self.LOG = {
#             "loss_descriminator": [],
#             "loss_generator": []
#         }

#     def randomly_flip_labels(self, labels, p: float = 0.05):
#         number_of_labels_to_flip = int(p * labels.shape[0])
#         indices_to_flip = random.choices([i for i in range(labels.shape[0])], k=number_of_labels_to_flip)
#         # flip chosen labels
#         labels[indices_to_flip] = 1 - labels[indices_to_flip]
#         return labels

#     def train(self, dataloader):
#         # for progress visualization
#         fixed_noise = torch.randn(config.batch_size, config.nz, 1, 1, device=device)
#         fixed_attr = torch.FloatTensor(config.nfeature, config.batch_size).uniform_(0, 2).gt(1).int().float().to(device)

#         z_noise = Variable(FloatTensor(config.batch_size, config.nz, 1, 1)).to(device)
#         generator_target = Variable(FloatTensor(config.batch_size, 1).fill_(1)).to(device)
#         discriminator_target_real = Variable(FloatTensor(config.batch_size, 1).fill_(1)).to(device)
#         discriminator_target_fake = Variable(FloatTensor(config.batch_size, 1).fill_(0)).to(device)
#         for epoch in range(config.epochs):
#             for i, (data, attr) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}"):

#                 ############################
#                 # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
#                 ###########################

#                 self.discriminator.zero_grad()

#                 if config.label_smoothing:
#                     discriminator_target_real.data.uniform_(0.7, 1.0)  # one-sided label smoothing trick
#                 if config.label_flipping:
#                     discriminator_target_real.data = self.randomly_flip_labels(
#                         discriminator_target_real.data, p=0.05)  # label flipping trick
#                     discriminator_target_fake.data = self.randomly_flip_labels(
#                         discriminator_target_fake.data, p=0.05)  # label flipping trick

#                 z_noise.data.normal_(0, 1)

#                 attr = Variable(attr).to(device)
#                 real_faces = Variable(data).to(device)
#                 d_real_faces = self.discriminator(real_faces, attr, config)

#                 fake_faces = self.generator(z_noise, attr, config)
#                 d_fake_faces = self.discriminator(fake_faces.detach(), attr, config)  # not update generator

#                 d_loss = self.loss(d_real_faces, discriminator_target_real) + \
#                     self.loss(d_fake_faces, discriminator_target_fake)
#                 d_loss.backward()
#                 self.optimizer_discriminator.step()

#                 ############################
#                 # (2) Update G network: maximize log(D(G(z)))
#                 ###########################

#                 self.generator.zero_grad()

#                 # TODO: test if we want to train the generator with new fake faces or instead use the ones we already used wiht the discriminator
#                 # noise.data.normal_(0, 1)
#                 # fake_faces = self.generator(z_noise, attr, config)

#                 d_fake = self.discriminator(fake_faces, attr, config)
#                 g_loss = self.loss(d_fake, generator_target)
#                 g_loss.backward()
#                 self.optimizer_generator.step()
#                 self.batch_training_info_and_samples(epoch, i, g_loss, d_loss, config,
#                                                      fake_faces, fixed_noise, fixed_attr)
#             self.epoch_training_info_and_samples(epoch, g_loss, d_loss, config, fake_faces, fixed_noise, fixed_attr)
#             ######### epoch finished ##########

#     def batch_training_info_and_samples(self, epoch, batch, g_loss, d_loss, config, fake_faces, fixed_noise, fixed_attr):
#         self.LOG["loss_generator"].append(g_loss)
#         self.LOG["loss_descriminator"].append(d_loss)

#         if config.print_loss and batch > 0 and batch % config.training_info_interval == 0:
#             if config.print_loss:
#                 tqdm.write(
#                     f"epoch {epoch+1} batch {batch} | generator loss: {g_loss} | discriminator loss: {d_loss}")
#         if batch % config.sample_interval == 0:
#             if config.random_sample:
#                 vutils.save_image(
#                     fake_faces.data[:min(config.batch_size, 32)], f'{config.result_dir}/{config.checkpoint_prefix}/result_epoch_{epoch + 1}_batch_{batch}.png', normalize=True)
#             if config.fixed_noise_sample:
#                 with torch.no_grad():
#                     fixed_fake = self.generator(fixed_noise, fixed_attr, config)
#                     vutils.save_image(fixed_fake.detach()[:min(config.batch_size, 32)],
#                                       f'{config.result_dir}/{config.checkpoint_prefix}/fixed_noise_result_epoch_{epoch + 1}_batch_{batch}.png', normalize=True)

#     def epoch_training_info_and_samples(self, epoch, g_loss, d_loss, config, fake_faces, fixed_noise, fixed_attr):
#         for key, values in self.LOG.items():
#             plt.plot(values, label=key)
#         plt.legend(loc="upper left")
#         plt.ylabel('Loss')
#         plt.xlabel('Iterations')
#         plt.savefig(f"{config.result_dir}/{config.checkpoint_prefix}/loss_visualization_{epoch}.png")

#         if config.show_loss_plot:
#             plt.show()

#         if config.print_loss:
#             tqdm.write(f"epoch {epoch+1} | generator loss: {g_loss} | discriminator loss: {d_loss}")
#         if config.random_sample:
#             vutils.save_image(
#                 fake_faces.data[:min(config.batch_size, 32)], f'{config.result_dir}/{config.checkpoint_prefix}/result_epoch_{epoch + 1}.png', normalize=True)
#         if config.fixed_noise_sample:
#             with torch.no_grad():
#                 fixed_fake = self.generator(fixed_noise, fixed_attr, config)
#                 vutils.save_image(fixed_fake.detach()[:min(config.batch_size, 32)],
#                                   f'{config.result_dir}/{config.checkpoint_prefix}/fixed_noise_result_epoch_{epoch + 1}.png', normalize=True)
#         if config.save_checkpoints:
#             torch.save(self.generator.state_dict(),
#                        f'{config.checkpoint_dir}/{config.checkpoint_prefix}/generator_epoch_{epoch+1}.pt')
#             torch.save(self.discriminator.state_dict(),
#                        f'{config.checkpoint_dir}/{config.checkpoint_prefix}/discriminator_epoch_{epoch+1}.pt')


if __name__ == '__main__':
    config = ArgParser().get_config()
    # if torch.cuda.is_available():
    #     device = torch.device("cuda:0")
    #     FloatTensor = torch.cuda.FloatTensor
    #     print("Running on the GPU")
    # else:
    #     FloatTensor = torch.FloatTensor
    #     device = torch.device("cpu")
    #     print("Running on the CPU")

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

    # print("Loading Data")
    # transformers = [] if config.hd_crop else [transforms.CenterCrop(178)]
    # transformers.extend([transforms.ToTensor(),
    #                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # dataset = ImageFeatureFolder(config.dataset_dir, config.condition_file, transform=transforms.Compose(transformers))

    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=config.batch_size, num_workers=config.workers, drop_last=True, shuffle=True, pin_memory=True)

    # print("Starting Training")
    # trainer = Trainer(config)
    # trainer.train(dataloader)
    
    trainer = Trainer(config)
    trainer.train()
