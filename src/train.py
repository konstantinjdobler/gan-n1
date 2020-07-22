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

from architecture_pgan import Generator, Discriminator
from data_loading import AttribDataset, ImageFeatureFolder

from helper.arg_parser import ArgParser
from helper.store import Store
from helper.criterions import WGANGP_loss, WGANGP_gradient_penalty, Epsilon_loss, ACGANCriterion
from helper.checks import isinf, isnan, finite_check
from helper.numpy import NumpyFlip, NumpyToTensor

# beta1 is a hyperparameter, suggestion from hack repo is 0.5
# betas = (0.5, 0.99)  # adam optimizer beta1, beta2


class ProgressiveGAN:
    def __init__(self, config, checkpoint_path=None):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.classification_criterion = ACGANCriterion
        self.classification_loss = nn.BCELoss()
        self.noise_vector_dim = 512
        self.category_vector_dim = 40  # self.classification_criterion.get_input_dim()
        self.latent_vector_dimension = self.noise_vector_dim  # + self.category_vector_dim
        self.image_channels = 3  # RGB
        self.learning_rate = config['learning_rate']

        self.config = config
        self.config['alpha'] = 0

        self.lossD = 0
        self.lossG = 0

        self.loss_history = []

        self.num_of_scale_iterations = 0

        self.generator = Generator(output_image_channels=self.image_channels, latent_vector_dimension=self.latent_vector_dimension,
                                   initial_layer_channels=config['scaling_layer_channels'][0]).to(self.device)
        self.discriminator = Discriminator(input_image_channels=self.image_channels, decision_layer_size=1,
                                           initial_layer_channels=config['scaling_layer_channels'][0]).to(self.device)

        self.optimizer_generator = self.get_optimizer(self.generator)
        self.optimizer_discriminator = self.get_optimizer(self.discriminator)

        # If path is give, load states and overwrite components with the loaded state
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(checkpoint_path):
        states = Store.load(checkpoint_path)

        # Overwrite internal attributes
        self.config = states['config']
        self.self.num_of_scale_iterations = in_state['scaleIteration']
        self.learning_rate = states['config']['learning_rate']

        # Bring models to the same architectural structure as the loaded state requires
        for scale_iteration in range(1, self.self.num_of_scale_iterations + 1):
            scaling_layer_channel = self.config['scaling_layer_channels'][scale_iteration]
            self.discriminator.add_new_layer(scaling_layer_channel)
            self.generator.add_new_layer(scaling_layer_channel)

        # Overwrite Model States
        self.discriminator.load_state_dict(config['discriminatorState'])
        self.generator.load_state_dict(config['generatorState'])
        # Overwrite Optimizer States
        self.optimizer_discriminator.load_state_dict(config['dOptimizer'])
        self.optimizer_generator.load_state_dict(config['gOptimizer'])

    def save_checkpoint(self):
        checkpoint = {'scaleIteration': self.num_of_scale_iterations,
                      'config': self.config,
                      'generatorState': self.generator.state_dict(),
                      'discriminatorState': self.discriminator.state_dict(),
                      'generatorOptimizer': self.optimizer_generator.state_dict(),
                      'discriminatorOptimizer': self.optimizer_discriminator.state_dict()}

        Store.save(
            checkpoint, f"{self.config['result_dir']}/{self.config['checkpoint_prefix']}/", self.num_of_scale_iterations)

    def train_on_batch(self, input_batch, labels):
        input_batch = input_batch.to(self.device)
        batch_size = input_batch.size()[0]
        labels_zero_one = labels.clone().detach()
        labels_zero_one[labels_zero_one == -1] = 0

        ####### TRAIN DISCRIMINATOR ##########
        self.optimizer_discriminator.zero_grad()
        prediction_real_data, prediction_real_labels = self.discriminator(input_batch)
        latent_vector = torch.randn(batch_size, self.latent_vector_dimension).to(self.device)
        #latent_vector, target_cat_noise = self.build_noise_data(batch_size)
        generated_fake_data = self.generator(latent_vector, labels).detach()
        prediction_fake_data, prediction_fake_classes = self.discriminator(generated_fake_data)

        #classification_loss_d = self.classification_criterion.loss(prediction_real_data, labels) * self.config['weight_condition_d']
        # classification_loss_d.backward(retain_graph=True)

        classification_loss = self.classification_loss(prediction_real_labels, labels_zero_one)
        classification_loss.backward(retain_graph=True)

        discriminator_loss = WGANGP_loss(prediction_fake_data, should_be_real=False) + \
            WGANGP_loss(prediction_real_data, should_be_real=True)

        WGANGP_gradient_penalty(input_batch, generated_fake_data, self.discriminator, self.config['lambda_gp'])
        discriminator_loss += Epsilon_loss(prediction_real_data, self.config['epsilon_d'])

        discriminator_loss.backward()
        finite_check(self.discriminator.parameters())
        self.optimizer_discriminator.step()

        ########## TRAIN GENERATOR #########
        self.optimizer_discriminator.zero_grad()
        self.optimizer_generator.zero_grad()

        latent_vector = torch.randn(batch_size, self.latent_vector_dimension).to(self.device)
        #latent_vector, target_cat_noise = self.build_noise_data(batch_size)
        generated_fake_data = self.generator(latent_vector, labels)
        prediction_fake_data, prediction_fake_classes = self.discriminator(generated_fake_data)

        #classification_loss_g = self.classification_criterion.loss(prediction_fake_data, target_cat_noise) * self.config['weight_condition_g']
        # classification_loss_g.backward(retain_graph=True)
        classification_loss = self.classification_loss(prediction_fake_classes, labels_zero_one)
        classification_loss.backward(retain_graph=True)
        generator_loss = WGANGP_loss(prediction_fake_data, should_be_real=True)
        generator_loss.backward()

        finite_check(self.generator.parameters())
        self.optimizer_generator.step()

        self.lossD = discriminator_loss
        self.lossG = generator_loss

        self.loss_history.append((discriminator_loss.item(), generator_loss.item()))

    def build_noise_data(self, number_of_samples):
        input_latent = torch.randn(number_of_samples, self.noise_vector_dim).to(self.device)

        target_rand_cat, latent_rand_cat = ACGANCriterion.build_random_criterion_tensor(number_of_samples)
        target_rand_cat = target_rand_cat.to(ACGANCriterion.device)
        latent_rand_cat = latent_rand_cat.to(ACGANCriterion.device)

        input_latent = torch.cat((input_latent, latent_rand_cat), dim=1)
        return input_latent, target_rand_cat

    def add_new_layer(self, new_layer_channels):
        self.generator.add_new_layer(new_layer_channels)
        self.discriminator.add_new_layer(new_layer_channels)
        self.num_of_scale_iterations += 1
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
        print('Updated alpha: ', new_alpha)
        self.generator.set_new_alpha(new_alpha)
        self.discriminator.set_new_alpha(new_alpha)
        self.config['alpha'] = new_alpha

    def generate_image(self, number_of_images, labels):
        latent_vector = torch.randn(number_of_images, self.latent_vector_dimension).to(self.device)
        return self.generator(latent_vector, labels).detach().cpu()


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
        self.model_config['scaling_layer_channels'] = [512, 512, 512, 512, 256, 128, 64, 32, 16]
        self.model_config['mini_batch_size'] = 16
        self.model_config['dim_latent_vector'] = 512
        self.model_config['lambda_gp'] = 10
        self.model_config["epsilon_d"] = 0.001
        self.model_config["learning_rate"] = config.lr

        # conditional config
        self.model_config["weight_condition_g"] = 1.0
        self.model_config["weight_condition_d"] = 1.0

        self.model_config["result_dir"] = config.result_dir
        self.model_config["checkpoint_prefix"] = config.checkpoint_prefix

        self.update_alpha_jumps(self.model_config['alpha_n_jumps'], self.model_config['alpha_size_jumps'])

        self.scale_sanity_check()

        self.init_model()

    def init_model(self):
        self.model = ProgressiveGAN(self.model_config, self.config.checkpoint)

    def scale_sanity_check(self):
        number_scaling_layers = min(len(self.model_config['scaling_layer_channels']),
                                    len(self.model_config['max_iter_at_scale']),
                                    len(self.model_config['iter_alpha_jump']),
                                    len(self.model_config['alpha_jump_vals']))

        self.model_config['scaling_layer_channels'] = self.model_config['scaling_layer_channels'][:number_scaling_layers]
        self.model_config['max_iter_at_scale'] = self.model_config['max_iter_at_scale'][:number_scaling_layers]
        self.model_config['iter_alpha_jump'] = self.model_config['iter_alpha_jump'][:number_scaling_layers]
        self.model_config['alpha_jump_vals'] = self.model_config['alpha_jump_vals'][:number_scaling_layers]

        self.model_config['size_scales'] = [4]
        for scale in range(1, number_scaling_layers):
            self.model_config['size_scales'].append(
                self.model_config['size_scales'][-1] * 2)

        self.model_config['number_scaling_layers'] = number_scaling_layers

    def update_alpha_jumps(self, n_jump_scale, size_jump_scale):
        number_scaling_layers = min(len(n_jump_scale), len(size_jump_scale))

        for scale in range(number_scaling_layers):
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
        return torch.utils.data.DataLoader(dataset, batch_size=self.model_config['mini_batch_size'], shuffle=True, num_workers=self.config.workers)

    def get_dataset(self, scale_iteration, size=None):
        if size is None:
            size = self.model.get_size()


        scaled_image_resolution = 4 * 2**scale_iteration
        print("size", size)

        path = self.path_db + f'{scaled_image_resolution}'
        resizeSize = None
        if not os.path.isdir(path):
            path = self.path_db
            resizeSize = size

        transform_list = [NumpyToTensor(resizeSize),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        transform = transforms.Compose(transform_list)

        # TODO: actiavte multi resolution image dataset when you have it on your computer
        data_set = ImageFeatureFolder(image_root=path,
                                      attribute_file=self.config.condition_file, transform=transform)

        # data_set = AttribDataset(self.path_db, #+ f'/{scaled_image_resolution}',
        #                     transform=transform,
        #                     attribDictPath=self.config.condition_file,
        #                     specificAttrib=None,
        #                     mimicImageFolder=False)

        # ACGANCriterion.set_key_order(data_set.getKeyOrders(), self.model.device)
        return data_set

    def train(self):
        number_scaling_layers = len(self.model_config['scaling_layer_channels'])

        for scale in range(self.start_scale, number_scaling_layers):

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

            if scale == number_scaling_layers - 1:
                break

            self.model.add_new_layer(self.model_config['scaling_layer_channels'][scale + 1])

        self.start_scale = number_scaling_layers
        self.start_iter = self.model_config['max_iter_at_scale'][-1]
        return True

    def train_on_epoch(self,
                       dbLoader,
                       scale,
                       shift_iter=0,
                       max_iter=-1):
        i = shift_iter

        for item, (data, labels) in enumerate(dbLoader, 0):

            inputs_real, labels = data.to(self.model.device), labels.to(self.model.device)

            if inputs_real.size()[0] < self.model_config['mini_batch_size']:
                continue

            # Additionnal updates inside a scale
            inputs_real = self.in_scale_update(i, scale, inputs_real)

            allLosses = self.model.train_on_batch(inputs_real, labels)
            # if len(data) > 2:
            #     mask = data[2]
            #     allLosses = self.model.train_on_batch(inputs_real)
            # else:
            #     allLosses = self.model.train_on_batch(inputs_real)

            i += 1

            if i % config.training_info_interval == 0:
                print("Iteration: ", i, " Alpha: ", self.model.config['alpha'])
                print("Generator Loss: ", str(self.model.lossG.item()),
                      " Discrimnator Loss: ", str(self.model.lossD.item()))

            # Save Image and Model Checkpoint
            if i % config.sample_interval == 0:
                self.generate_image(scale, i, labels)
                self.model.save_checkpoint()
                self.write_loss_to_file(scale)

            if i == max_iter:
                return True

        return True

    def generate_image(self, scale, iteration, labels):
        image = self.model.generate_image(self.model_config['mini_batch_size'], labels)
        vutils.save_image(image.data[:self.model_config['mini_batch_size']], f'{self.config.result_dir}/{self.config.checkpoint_prefix}/scale' + str(
            scale) + '_iter' + str(iteration) + '.png', normalize=True)

    def write_loss_to_file(self, scale):
        with open(f'{self.config.result_dir}/{self.config.checkpoint_prefix}/losses' + str(scale) + '.txt', "a") as loss_file:
            loss_file.writelines(((",".join(str(x) for x in loss_entry) + '\n')
                                  for loss_entry in self.model.loss_history))
            self.model.loss_history = []

    def add_new_scales(self, config_new_scales):

        self.update_alpha_jumps(config_new_scales["alpha_n_jumps"],
                                config_new_scales["alpha_size_jumps"])

        self.model_config['scaling_layer_channels'] = self.model_config['scaling_layer_channels'] + \
            config_new_scales["scaling_layer_channels"]
        self.model_config['max_iter_at_scale'] = self.model_config['max_iter_at_scale'] + \
            config_new_scales["max_iter_at_scale"]

        self.scale_sanity_check()


if __name__ == '__main__':
    config = ArgParser().get_config()

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
