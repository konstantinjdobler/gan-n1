import os

import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random

from PIL import Image
import numpy as np

from architecture_pgan import Generator, Discriminator
from data_loading import ImageFeatureFolder
from helper.arg_parser import ArgParser
from helper.store import Store
from helper.criterions import WGANGP_loss, WGANGP_gradient_penalty, Epsilon_loss
from helper.checks import isinf, isnan, finite_check
from helper.numpy import NumpyFlip, NumpyToTensor
from helper.visualization import save_tensor_as_image
import model_config

class ProgressiveGAN:
    def __init__(self, config, checkpoint_path=None):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.classification_loss = nn.BCELoss()
        self.latent_vector_dimension = config['dim_latent_vector']
        self.learning_rate = config['learning_rate']
        self.image_channels = 3
        self.config = config
        self.config['alpha'] = 0
        self.current_scale_level = 0
        self.num_of_iterations = 0

        self.loss_history = []

        self.generator = Generator(output_image_channels=self.image_channels, latent_vector_dimension=self.latent_vector_dimension,
                                   initial_layer_channels=config['scaling_layer_channels'][0]).to(self.device)
        self.discriminator = Discriminator(input_image_channels=self.image_channels, decision_layer_size=1,
                                           initial_layer_channels=config['scaling_layer_channels'][0]).to(self.device)

        self.optimizer_generator = self.get_optimizer(self.generator)
        self.optimizer_discriminator = self.get_optimizer(self.discriminator)

        self.fixed_noise = torch.randn(config['mini_batch_size'][0], self.latent_vector_dimension).to(self.device)

        # If path is given, load states and overwrite components with the loaded state
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        states = Store.load(checkpoint_path)

        # Overwrite internal attributes
        self.config = states['config']
        self.current_scale_level = states['current_scale_level']
        self.num_of_iterations = states['num_iterations']
        self.learning_rate = states['config']['learning_rate']

        # Bring models to the same architectural structure as the loaded state requires
        for scale_iteration in range(1, self.current_scale_level + 1):
            scaling_layer_channel = self.config['scaling_layer_channels'][scale_iteration]
            self.discriminator.add_new_layer(scaling_layer_channel)
            self.generator.add_new_layer(scaling_layer_channel)

        # Overwrite Model States
        self.discriminator.load_state_dict(states['discriminator_state'])
        self.generator.load_state_dict(states['generator_state'])

        self.move_to_device()

        # Overwrite Optimizer States
        self.optimizer_discriminator.load_state_dict(states['discriminator_optimizer'])
        self.optimizer_generator.load_state_dict(states['generator_optimizer'])

        if self.num_of_iterations >= self.config['max_iter_at_scale'][self.current_scale_level]:
            self.add_new_layer(self.config['scaling_layer_channels'][self.current_scale_level + 1])
            self.num_of_iterations = 0

    def save_checkpoint(self, num_of_iterations_in_scale):
        checkpoint = {'current_scale_level': self.current_scale_level,
                      'num_iterations' : num_of_iterations_in_scale,   # TODO: I added this, but it is not yet read anywhere. It should be used for initializing the start_iter of the Trainer
                      'config': self.config,
                      'generator_state': self.generator.state_dict(),
                      'discriminator_state': self.discriminator.state_dict(),
                      'generator_optimizer': self.optimizer_generator.state_dict(),
                      'discriminator_optimizer': self.optimizer_discriminator.state_dict()}

        Store.save(
            checkpoint, f"{self.config['result_dir']}/{self.config['checkpoint_prefix']}/", self.current_scale_level)

    def train_on_batch(self, input_batch, labels):
        input_batch = input_batch.to(self.device)
        batch_size = input_batch.size()[0]
        labels_zero_one = labels.clone().detach()
        labels_zero_one[labels_zero_one == -1] = 0

        ####### TRAIN DISCRIMINATOR ##########
        self.optimizer_discriminator.zero_grad()
        prediction_real_data, prediction_real_labels = self.discriminator(input_batch)
        latent_vector = torch.randn(batch_size, self.latent_vector_dimension).to(self.device)
        generated_fake_data = self.generator(latent_vector, labels).detach()
        prediction_fake_data, prediction_fake_classes = self.discriminator(generated_fake_data)

        # classification loss
        classification_loss = self.classification_loss(prediction_real_labels, labels_zero_one)
        classification_loss.backward(retain_graph=True)

        # wasserstein loss
        discriminator_loss = WGANGP_loss(prediction_fake_data, should_be_real=False) + \
            WGANGP_loss(prediction_real_data, should_be_real=True)

        # gradient penalty + epsilon loss
        WGANGP_gradient_penalty(input_batch, generated_fake_data, self.discriminator, self.config['lambda_gp'])
        discriminator_loss += Epsilon_loss(prediction_real_data, self.config['epsilon_d'])

        discriminator_loss.backward()
        finite_check(self.discriminator.parameters())
        self.optimizer_discriminator.step()

        ########## TRAIN GENERATOR #########
        self.optimizer_discriminator.zero_grad()
        self.optimizer_generator.zero_grad()

        latent_vector = torch.randn(batch_size, self.latent_vector_dimension).to(self.device)
        generated_fake_data = self.generator(latent_vector, labels)
        prediction_fake_data, prediction_fake_classes = self.discriminator(generated_fake_data)

        # classification loss
        classification_loss = self.classification_loss(prediction_fake_classes, labels_zero_one)
        classification_loss.backward(retain_graph=True)

        # wasserstein loss
        generator_loss = WGANGP_loss(prediction_fake_data, should_be_real=True)

        generator_loss.backward()
        finite_check(self.generator.parameters())
        self.optimizer_generator.step()

        # store loss
        self.loss_history.append((discriminator_loss.item(), generator_loss.item()))

    def add_new_layer(self, new_layer_channels):
        self.generator.add_new_layer(new_layer_channels)
        self.discriminator.add_new_layer(new_layer_channels)
        self.current_scale_level += 1
        self.move_to_device()

    def get_optimizer(self, model):
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                          betas=[0, 0.99], lr=self.learning_rate)

    def move_to_device(self):
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

    def generate_image(self, number_of_images, labels, fixed_noise):
        latent_vector = torch.randn(number_of_images, self.latent_vector_dimension).to(self.device)
        if fixed_noise:
            latent_vector = self.fixed_noise[:number_of_images]

        return self.generator(latent_vector, labels).detach().cpu()


class Trainer:
    def __init__(self, config):
        self.path_db = config.dataset_dir

        self.start_scale = 0
        self.start_iter = 0
        self.config = config

        self.model_config = model_config.model_config
        self.model_config["result_dir"] = config.result_dir
        self.model_config["checkpoint_prefix"] = config.checkpoint_prefix

        self.update_alpha_jumps(self.model_config['alpha_n_jumps'], self.model_config['alpha_size_jumps'])

        self.scale_sanity_check()

        self.init_model()
        self.start_scale = self.model.current_scale_level
        self.start_iter = self.model.num_of_iterations

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

    def get_db_loader(self, scale):
        dataset = self.get_dataset(scale)
        return torch.utils.data.DataLoader(dataset, batch_size=self.model_config['mini_batch_size'][scale], shuffle=True, num_workers=self.config.workers)

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

        data_set = ImageFeatureFolder(image_root=path,
                                      attribute_file=self.config.condition_file, transform=transform)

        return data_set

    def train(self):
        number_scaling_layers = len(self.model_config['scaling_layer_channels'])

        for scale in range(self.start_scale, number_scaling_layers):

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
                self.train_on_epoch(db_loader, scale,
                                             shift_iter=shift_iter,
                                             max_iter=self.model_config['max_iter_at_scale'][scale])

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

            if inputs_real.size()[0] < self.model_config['mini_batch_size'][scale]:
                continue

            # Additional updates inside a scale
            inputs_real = self.in_scale_update(i, scale, inputs_real)
            self.model.train_on_batch(inputs_real, labels)

            i += 1

            # Printing, image output and checkpoint saving
            if i % config.training_info_interval == 0:
                print("Iteration: ", i, " Alpha: ", self.model.config['alpha'])
                print("Generator Loss: ", str(self.model.loss_history[-1][0]),
                      " Discrimnator Loss: ", str(self.model.loss_history[-1][1]))
            
            if config.save_checkpoints and i % config.checkpoint_interval == 0:
                self.model.save_checkpoint(i)
                self.write_loss_to_file(scale)

            if config.random_image_samples and i % config.sample_interval == 0:
                self.generate_image(scale, i, labels)

            if i == max_iter:
                return

        return

    def generate_image(self, scale, iteration, labels):
        image = self.model.generate_image(self.model_config['mini_batch_size'][scale], labels, self.config.fixed_noise_sample)
        save_tensor_as_image(image.data[:self.model_config['mini_batch_size'][scale]], 128, f'{self.config.result_dir}/{self.config.checkpoint_prefix}/scale' + str(scale) + '_iter' + str(iteration) + '.jpg')

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

    cudnn.benchmark = True

    # Create dirs if not already there
    if(config.random_image_samples):
        print(f"Sample fake images and checkpoints will be saved to {config.result_dir}/{config.checkpoint_prefix}")
        os.makedirs(f"{config.result_dir}/{config.checkpoint_prefix}", exist_ok=True)

    trainer = Trainer(config)
    trainer.train()
