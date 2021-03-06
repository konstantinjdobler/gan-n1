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
        # end layer has upsampling=2, first layer outputs 4x4
        num_middle_scaling_layers = int(log(config.target_image_size, 2) - 3)

        # as many scaling layers as necessary to scale to the target image size
        middle_scaling_layers = [ConvTranspose2dBlock(in_channels=config.generator_filters * 2**(i+1),
                                                      out_channels=config.generator_filters * 2**i,
                                                      upsampling_factor=2) for i in reversed(range(num_middle_scaling_layers))]
        self.main = nn.Sequential(
            ConvTranspose2dBlock(in_channels=config.nz + config.nfeature,
                                 out_channels=config.generator_filters * (2**num_middle_scaling_layers),
                                 kernel_size=4, stride=1, padding=0),
            *middle_scaling_layers,
            ConvTranspose2dBlock(in_channels=config.generator_filters,
                                 out_channels=config.nc, upsampling_factor=2,
                                 activation_function=nn.Tanh(), batch_norm=False),
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

        # end layer has upsampling=2, first layer outputs 4x4
        num_middle_scaling_layers = int(log(config.target_image_size, 2) - 3)
        # as many scaling layers as necessary to scale to the target image size
        middle_scaling_layers = [Conv2dBlock(in_channels=config.generator_filters * 2**i,
                                             out_channels=config.generator_filters * 2**(i + 1),
                                             downsampling_factor=2) for i in range(num_middle_scaling_layers)]
        self.main = nn.Sequential(
            Conv2dBlock(in_channels=config.nc + 1, out_channels=config.discriminator_filters,
                        downsampling_factor=2, batch_norm=False),
            *middle_scaling_layers,
            Conv2dBlock(in_channels=config.discriminator_filters * 2**num_middle_scaling_layers,
                        out_channels=1, kernel_size=4, stride=1, padding=0,
                        batch_norm=False, activation_function=nn.Sigmoid()),
        )

    def forward(self, x, attr, config):
        attr = self.feature_input(attr).view(-1, 1, config.target_image_size, config.target_image_size)
        x = torch.cat([x, attr], 1)
        return self.main(x).view(-1, 1)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname == 'Conv2d' or classname == 'ConvTranspose2d':
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
