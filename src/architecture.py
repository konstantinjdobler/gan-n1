import torch.nn as nn
import torch

from math import log


class ConvTranspose2dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 4,
                 stride: int = 2, padding: int = 1, bias=False,
                 upsampling_factor: int = None,  # must be divisible by 2
                 activation_function=nn.ReLU(True),
                 batch_norm: bool = True):

        super(ConvTranspose2dBlock, self).__init__()
        if upsampling_factor:
            # This ensures output dimension are scaled up by upsampling_factor
            stride = upsampling_factor
            kernel_size = 2 * upsampling_factor
            padding = upsampling_factor // 2

        self.conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = activation_function

    def forward(self, x):
        out = self.conv_layer(x)
        if self.batch_norm:
            out = self.batch_norm(out)
        return self.activation(out)


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 4,
                 stride: int = 2, padding: int = 1, bias=False,
                 downsampling_factor: int = None,  # must be divisible by 2
                 activation_function=nn.LeakyReLU(0.2, inplace=True),  # from GAN Hacks
                 batch_norm: bool = True):

        super(Conv2dBlock, self).__init__()
        if downsampling_factor:
            # This ensures output dimension are scaled down by downsampling_factor
            stride = downsampling_factor
            kernel_size = 2 * downsampling_factor
            padding = downsampling_factor // 2

        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = activation_function

    def forward(self, x):
        out = self.conv_layer(x)
        if self.batch_norm:
            out = self.batch_norm(out)
        return self.activation(out)


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        # middle_scaling_layers = log(config.target_image_size, 2) - 3 # end layer has umsampling=2, first layer outputs 4x4
        adjustment_to_image_size = config.target_image_size // 64  # 64 is standard output
        self.linear_layer = nn.Linear(config.nz + config.nfeature, config.generator_filters * 16)
        self.convolution_layers = nn.Sequential(
            ConvTranspose2dBlock(in_channels=config.nz + config.nfeature,
                                 out_channels=config.generator_filters * 8,
                                 kernel_size=4, stride=1, padding=0),

            ConvTranspose2dBlock(in_channels=config.generator_filters * 8,
                                 out_channels=config.generator_filters * 4, upsampling_factor=2),

            ConvTranspose2dBlock(in_channels=config.generator_filters * 4,
                                 out_channels=config.generator_filters * 2, upsampling_factor=2),

            ConvTranspose2dBlock(in_channels=config.generator_filters * 2,
                                 out_channels=config.generator_filters,
                                 upsampling_factor=2 * adjustment_to_image_size),

            ConvTranspose2dBlock(in_channels=config.generator_filters,
                                 out_channels=config.nc, upsampling_factor=2,
                                 activation_function=nn.Tanh(), batch_norm=False),
        )

    def forward(self, x, attr, config):
        attr = attr.view(-1, config.nfeature, 1, 1)
        x = torch.cat([x, attr], 1)
        #x = self.linear_layer(x)
        #x = x.view(-1, config.generator_filters * 16, 1, 1)
        return self.convolution_layers(x)


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        # self.feature_input = nn.Linear(config.nfeature,
        #                                config.target_image_size * config.target_image_size)
        self.decision_layer = nn.Sequential(nn.Linear(config.discriminator_filters * 16, 1), nn.Sigmoid())
        self.classification_layer = nn.Linear(config.discriminator_filters * 16, config.nfeature)

        adjustment_to_image_size = config.target_image_size // 64  # 64 is standard input
        self.convolution_layers = nn.Sequential(
            Conv2dBlock(in_channels=config.nc, out_channels=config.discriminator_filters,
                        downsampling_factor=2, batch_norm=False),

            Conv2dBlock(in_channels=config.discriminator_filters,
                        out_channels=config.discriminator_filters * 2,
                        downsampling_factor=2*adjustment_to_image_size),

            Conv2dBlock(in_channels=config.discriminator_filters * 2,
                        out_channels=config.discriminator_filters * 4,
                        downsampling_factor=2),

            Conv2dBlock(in_channels=config.discriminator_filters * 4,
                        out_channels=config.discriminator_filters * 8,
                        downsampling_factor=2),

            Conv2dBlock(in_channels=config.discriminator_filters * 8,
                        out_channels=config.discriminator_filters * 16, kernel_size=4, stride=1, padding=0,
                        batch_norm=False),
        )

    def forward(self, x, config):
        # attr = self.feature_input(attr).view(-1, 1, config.target_image_size, config.target_image_size)
        # x = torch.cat([x, attr], 1)
        x = self.convolution_layers(x)
        x = x.view(-1, config.discriminator_filters*16 )

        decision = self.decision_layer(x).view(-1, 1)
        classification = self.classification_layer(x).view(-1, config.nfeature)
        return decision, classification


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname == 'Conv2d' or classname == 'ConvTranspose2d':
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
