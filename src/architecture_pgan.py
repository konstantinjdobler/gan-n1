import torch.nn as nn
import torch

import torch.nn.functional as F

import math
from numpy import prod


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def miniBatchStdDev(x, sub_group_size=4):
    size = x.size()
    sub_group_size = min(size[0], sub_group_size)
    if size[0] % sub_group_size != 0:
        sub_group_size = size[0]
    G = int(size[0] / sub_group_size)
    if sub_group_size > 1:
        y = x.view(-1, sub_group_size, size[1], size[2], size[3])
        y = torch.var(y, 1)
        y = torch.sqrt(y + 1e-8)
        y = y.view(G, -1)
        y = torch.mean(y, 1).view(G, 1)
        y = y.expand(G, size[2]*size[3]).view((G, 1, 1, size[2], size[3]))
        y = y.expand(G, sub_group_size, -1, -1, -1)
        y = y.contiguous().view((-1, 1, size[2], size[3]))
    else:
        y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)

    return torch.cat([x, y], dim=1)

class NormalizationLayer(nn.Module):

    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def forward(self, x, epsilon=1e-8):
        return x * (((x**2).mean(dim=1, keepdim=True) + epsilon).rsqrt())


def Upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1:
        return x
    s = x.size()
    x = x.view(-1, s[1], s[2], 1, s[3], 1)
    x = x.expand(-1, s[1], s[2], factor, s[3], factor)
    x = x.contiguous().view(-1, s[1], s[2] * factor, s[3] * factor)
    return x


def getLayerNormalizationFactor(x):
    size = x.weight.size()
    fan_in = prod(size[1:])

    return math.sqrt(2.0 / fan_in)


class ConstrainedLayer(nn.Module):
    def __init__(self,
                 module,
                 lrMul=1.0):

        super(ConstrainedLayer, self).__init__()

        self.module = module
        self.equalized =True

        self.module.bias.data.fill_(0)
        self.module.weight.data.normal_(0, 1)
        self.module.weight.data /= lrMul
        self.weight = getLayerNormalizationFactor(self.module) * lrMul

    def forward(self, x):
        x = self.module(x)
        x *= self.weight
        return x


class EqualizedConv2d(ConstrainedLayer):

    def __init__(self,
                 nChannelsPrevious,
                 nChannels,
                 kernelSize,
                 padding=0,
                 **kwargs):
        ConstrainedLayer.__init__(self,
                                  nn.Conv2d(nChannelsPrevious, nChannels,
                                            kernelSize, padding=padding,
                                            bias=True),
                                  **kwargs)


class EqualizedLinear(ConstrainedLayer):

    def __init__(self,
                 nChannelsPrevious,
                 nChannels,
                 bias=True,
                 **kwargs):
        ConstrainedLayer.__init__(self,
                                  nn.Linear(nChannelsPrevious, nChannels,
                                  bias=bias), **kwargs)


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
    def __init__(self, latent_vector_dimension, output_image_channels, initial_layer_channels, generation_activation=None):
        super(Generator, self).__init__()
        # middle_scaling_layers = log(config.target_image_size, 2) - 3 # end layer has umsampling=2, first layer outputs 4x4
        self.initial_layer_channels = initial_layer_channels
        self.output_image_channels = output_image_channels
        self.alpha = 0
        self.activation_function = torch.nn.LeakyReLU(0.2)


        # init format layer
        self.layer_channels = [initial_layer_channels]
        self.init_format_layer(latent_vector_dimension)

        self.scale_layers = nn.ModuleList([])
        self.to_rgb_layers = nn.ModuleList([EqualizedConv2d(self.initial_layer_channels, self.output_image_channels, 1)])
        self.initial_layer = nn.ModuleList([EqualizedConv2d(self.initial_layer_channels, self.initial_layer_channels, 3, padding=1)])

        self.normalization_layer = NormalizationLayer()

        self.generation_activation = generation_activation

    def init_format_layer(self, latent_vector_dimension):
        self.latent_vector_dimension = latent_vector_dimension
        self.format_layer = EqualizedLinear(self.latent_vector_dimension, 16 * self.layer_channels[0])

    def get_output_size(self):
        side = 4 * (2**(len(self.to_rgb_layers) - 1))
        return (side, side)  
        
    def add_new_layer(self, new_layer_channels):
        previous_layer_channels = self.layer_channels[-1]
        self.layer_channels.append(new_layer_channels)

        # create new scale layer group and append to all scale layers
        new_scale_layers_group = nn.ModuleList([
            EqualizedConv2d(previous_layer_channels,
                            new_layer_channels,
                            3,
                            padding=1),
            EqualizedConv2d(new_layer_channels, 
                            new_layer_channels,
                            3, 
                            padding=1)
        ])
        self.scale_layers.append(new_scale_layers_group)

        # create and append new rgb layer
        self.to_rgb_layers.append(EqualizedConv2d(  new_layer_channels,
                                                    self.output_image_channels,
                                                    1))
                                                    
                                                    
    def set_new_alpha(self, alpha):
        self.alpha = alpha
    

    def forward(self, x):
         ## Normalize the input ?
        x = self.normalization_layer(x)
        x = x.view(-1, num_flat_features(x))
        # format layer
        x = self.activation_function(self.format_layer(x))
        x = x.view(x.size()[0], -1, 4, 4)

        x = self.normalization_layer(x)

        # Scale 0 (no upsampling)
        for conv_layer in self.initial_layer:
            x = self.activation_function(conv_layer(x))
            x = self.normalization_layer(x)

        # Dirty, find a better way
        if self.alpha > 0 and len(self.scale_layers) == 1:
            y = self.to_rbg_layers[-2](x)
            y = Upscale2d(y)

        # Upper scales
        for scale, layer_group in enumerate(self.scale_layers, 0):
            x = Upscale2d(x)
            for conv_layer in layer_group:
                x = self.activation_function(conv_layer(x))
                x = self.normalization_layer(x)

            if self.alpha > 0 and scale == (len(self.scale_layers) - 2):
                y = self.to_rgb_layers[-2](x)
                y = Upscale2d(y)

        # To RGB (no alpha parameter for now)
        x = self.to_rgb_layers[-1](x)

        # Blending with the lower resolution output when alpha > 0
        if self.alpha > 0:
            x = self.alpha * y + (1.0-self.alpha) * x

        if self.generation_activation is not None:
            x = self.generation_activation(x)

        return x



class Discriminator(nn.Module):
    def __init__(self, input_image_channels, decision_layer_size, initial_layer_channels):
        super(Discriminator, self).__init__()
        # Initialization paramneters
        self.input_image_channels = input_image_channels
        self.initial_layer_channels = initial_layer_channels

        # Initalize the scales
        self.layer_channels = [self.initial_layer_channels]
        self.scale_layers = nn.ModuleList()
        self.from_rgb_layers = nn.ModuleList()

        self.merge_layers = nn.ModuleList()

        # Initialize the last layer
        self.decision_layer = EqualizedLinear(self.layer_channels[0], decision_layer_size)

        # Layer 0
        self.group_scale_zero = nn.ModuleList([
            EqualizedConv2d(self.initial_layer_channels, self.initial_layer_channels, 3, padding=1),
            EqualizedLinear(self.initial_layer_channels * 16, self.initial_layer_channels)
        ])
        self.from_rgb_layers.append(EqualizedConv2d(input_image_channels, self.initial_layer_channels, 1))

        # # Minibatch standard deviation
        # dim_entry_scale_0 = depthScale0
        # if miniBatchNormalization:
        #     dim_entry_scale_0 += 1

        # self.miniBatchNormalization = miniBatchNormalization


        # Initalize the upscaling parameters
        self.alpha = 0

        # Leaky relu activation
        self.leaky_relu = torch.nn.LeakyReLU(0.2)

    
    
    def add_new_layer(self, new_scale = None):
        depth_last_scale = self.layer_channels[-1]
        depth_new_scale = new_scale if new_scale else depth_last_scale * 2
        self.layer_channels.append(depth_new_scale)

        # create new scale layer group and append to all scale layers
        new_scale_layers_group = nn.ModuleList([
            EqualizedConv2d(depth_new_scale,
                            depth_new_scale,
                            3,
                            padding=1),
            EqualizedConv2d(depth_new_scale, 
                            depth_last_scale,
                            3, 
                            padding=1)
        ])
        self.scale_layers.append(new_scale_layers_group)

        # create and append new rgb layer
        self.to_rgb_layers.append(EqualizedConv2d(  self.input_image_channels,
                                                    depth_new_scale,
                                                    1))
                                                    

    def set_new_alpha(self, alpha):
        r"""
        Update the value of the merging factor alpha

        Args:

            - alpha (float): merging factor, must be in [0, 1]
        """

        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in [0,1]")

        if not self.from_rgb_layers:
            raise AttributeError("Can't set an alpha layer if only the scale 0"
                                 "is defined")

        self.alpha = alpha


    def forward(self, x):
        # Alpha blending
        if self.alpha > 0 and len(self.from_rgb_layers) > 1:
            y = F.avg_pool2d(x, (2, 2))
            y = self.leaky_relu(self.from_rgb_layers[- 2](y))

        # From RGB layer
        x = self.leaky_relu(self.from_rgb_layers[-1](x))

        # Caution: we must explore the layers group in reverse order !
        # Explore all scales before 0
        merge_layer = self.alpha > 0 and len(self.scale_layers) > 1

        for group_layer in reversed(self.scale_layers):

            for layer in group_layer:
                x = self.leaky_relu(layer(x))

            x = nn.AvgPool2d((2, 2))(x)

            if merge_layer:
                merge_layer = False
                x = self.alpha * y + (1-self.alpha) * x


        # Now the scale 0

        # # Minibatch standard deviation
        # if self.miniBatchNormalization:
        #     x = miniBatchStdDev(x)

        x = self.leaky_relu(self.group_scale_zero[0](x))

        x = x.view(-1, num_flat_features(x))
        x = self.leaky_relu(self.group_scale_zero[1](x))

        out = self.decision_layer(x)

        return out


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname == 'Conv2d' or classname == 'ConvTranspose2d':
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
