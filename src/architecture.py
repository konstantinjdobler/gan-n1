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
        if self.activation:
            return self.activation(out)
        else:
            return out


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
        if self.activation:
            return self.activation(out)
        else:
            return out


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        # middle_scaling_layers = log(config.target_image_size, 2) - 3 # end layer has umsampling=2, first layer outputs 4x4
        adjustment_to_image_size = config.target_image_size // 64  # 64 is standard output
        self.main = nn.Sequential(
            # input: (config.nz + config.nfeature)
            ConvTranspose2dBlock(in_channels=config.nz + config.nfeature,
                                 out_channels=config.generator_filters * 8,
                                 kernel_size=4, stride=1, padding=0),
            # state: (generator_filters*8) x 4 x 4

            ConvTranspose2dBlock(in_channels=config.generator_filters * 8,
                                 out_channels=config.generator_filters * 4, upsampling_factor=2),
            # state: (generator_filters*4) x 8 x 8

            ConvTranspose2dBlock(in_channels=config.generator_filters * 4,
                                 out_channels=config.generator_filters * 2, upsampling_factor=2),
            # state: (generator_filters*2) x 16 x 16 (using image size 64)

            ConvTranspose2dBlock(in_channels=config.generator_filters * 2,
                                 out_channels=config.generator_filters,
                                 upsampling_factor=2 * adjustment_to_image_size),
            # state: (generator_filters) x 32 x 32

            ConvTranspose2dBlock(in_channels=config.generator_filters,
                                 out_channels=config.nc, upsampling_factor=2,
                                 activation_function=nn.Tanh(), batch_norm=False),
            # output: (nc) x 64 x 64
        )

    def forward(self, x, attr, config):
        attr = attr.view(-1, config.nfeature, 1, 1)
        x = torch.cat([x, attr], 1)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.feature_input = nn.Linear(config.nfeature,
                                       config.target_image_size * config.target_image_size)

        self.filters = config.discriminator_filters

        adjustment_to_image_size = config.target_image_size // 64  # 64 is standard input
        self.main = nn.Sequential(
            # input: (nc) x 64 x 64 (using image size 64)
            Conv2dBlock(in_channels=config.nc + 1, out_channels=config.discriminator_filters,
                        downsampling_factor=2, batch_norm=False),
            # state: (discriminator_filters) x 32 x 32

            Conv2dBlock(in_channels=config.discriminator_filters,
                        out_channels=config.discriminator_filters * 2,
                        downsampling_factor=2*adjustment_to_image_size),
            # state: (discriminator_filters*2) x 16 x 16

            Conv2dBlock(in_channels=config.discriminator_filters * 2,
                        out_channels=config.discriminator_filters * 4,
                        downsampling_factor=2),
            # state: (discriminator_filters*4) x 8 x 8

            Conv2dBlock(in_channels=config.discriminator_filters * 4,
                        out_channels=config.discriminator_filters * 8,
                        downsampling_factor=2)
            # state: (discriminator_filters*8) x 4 x 4

            #Conv2dBlock(in_channels=config.discriminator_filters * 8,
            #            out_channels=1, kernel_size=4, stride=1, padding=0,
            #            batch_norm=False, activation_function=None),
            # output: 1 x 1 x 1
        )

        self.linear = nn.Linear(config.discriminator_filters * 8 * 4 * 4, 1)

    def forward(self, x, attr, config):
        attr = self.feature_input(attr).view(-1, 1, config.target_image_size, config.target_image_size)
        x = torch.cat([x, attr], 1)
        out = self.main(x)
        out = out.view(-1, 4*4*8*self.filters)
        return self.linear(out)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname == 'Conv2d' or classname == 'ConvTranspose2d':
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
