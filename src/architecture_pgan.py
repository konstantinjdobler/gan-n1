import torch.nn as nn
import torch

import torch.nn.functional as F

import math
from numpy import prod


def mini_batch_std_dev(x, sub_group_size=4):
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

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class NormalizationLayer(nn.Module):

    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def forward(self, x, epsilon=1e-8):
        return x * (((x**2).mean(dim=1, keepdim=True) + epsilon).rsqrt())


def upscale_2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1:
        return x
    s = x.size()
    x = x.view(-1, s[1], s[2], 1, s[3], 1)
    x = x.expand(-1, s[1], s[2], factor, s[3], factor)
    x = x.contiguous().view(-1, s[1], s[2] * factor, s[3] * factor)
    return x


def get_layer_normalization_factor(x):
    size = x.weight.size()
    fan_in = prod(size[1:])

    return math.sqrt(2.0 / fan_in)


class ConstrainedLayer(nn.Module):
    def __init__(self,
                 module,
                 lrMul=1.0):

        super(ConstrainedLayer, self).__init__()

        self.module = module
        self.equalized = True

        self.module.bias.data.fill_(0)
        self.module.weight.data.normal_(0, 1)
        self.module.weight.data /= lrMul
        self.weight = get_layer_normalization_factor(self.module) * lrMul

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


class Generator(nn.Module):
    def __init__(self, latent_vector_dimension, output_image_channels, initial_layer_channels, num_classes=40, generation_activation=nn.Tanh()):
        super(Generator, self).__init__()
        self.output_image_channels = output_image_channels
        self.alpha = 0
        self.activation_function = torch.nn.LeakyReLU(0.2)

        # init different layers
        self.layer_channels = [initial_layer_channels]
        self.format_layer = EqualizedLinear(latent_vector_dimension + num_classes, 4 * 4 * self.layer_channels[0])
        self.scale_layers = nn.ModuleList([])
        self.to_rgb_layers = nn.ModuleList(
            [EqualizedConv2d(initial_layer_channels, self.output_image_channels, 1)])
        self.initial_layer = nn.ModuleList(
            [EqualizedConv2d(initial_layer_channels, initial_layer_channels, 3, padding=1)])

        self.normalization_layer = NormalizationLayer()

        self.generation_activation = generation_activation

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
        self.to_rgb_layers.append(EqualizedConv2d(new_layer_channels,
                                                  self.output_image_channels,
                                                  1))

    def set_new_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, x, labels):
        # # Normalize the input ?
        x = self.normalization_layer(x)
        x = x.view(-1, num_flat_features(x)) # maybe not use this?
        
        # format layer
        x = torch.cat((x,labels), dim=1)
        x = self.activation_function(self.format_layer(x))
        x = x.view(x.size()[0], -1, 4, 4)

        x = self.normalization_layer(x)

        # Scale 0 (no upsampling)
        for conv_layer in self.initial_layer:
            x = self.activation_function(conv_layer(x))
            x = self.normalization_layer(x)

        if self.alpha > 0 and len(self.scale_layers) == 1:
            y = self.to_rgb_layers[-2](x)
            y = upscale_2d(y)

        # Upper scales
        for scale, layer_group in enumerate(self.scale_layers, 0):
            x = upscale_2d(x)
            for conv_layer in layer_group:
                x = self.activation_function(conv_layer(x))
                x = self.normalization_layer(x)

            if self.alpha > 0 and scale == (len(self.scale_layers) - 2):
                y = self.to_rgb_layers[-2](x)
                y = upscale_2d(y)

        # To RGB (no alpha parameter for now)
        x = self.to_rgb_layers[-1](x)

        # Blending with the lower resolution output when alpha > 0
        if self.alpha > 0:
            x = self.alpha * y + (1.0-self.alpha) * x

        if self.generation_activation is not None:
            x = self.generation_activation(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, input_image_channels, decision_layer_size, initial_layer_channels, num_classes=40):
        super(Discriminator, self).__init__()
        self.input_image_channels = input_image_channels
        self.alpha = 0
        self.leaky_relu = torch.nn.LeakyReLU(0.2)

        # Initalize the scales
        self.layer_channels = [initial_layer_channels]

        self.scale_layers = nn.ModuleList()
        self.from_rgb_layers = nn.ModuleList()
        self.merge_layers = nn.ModuleList()

        # Initialize the last layer
        self.decision_layer = EqualizedLinear(self.layer_channels[0], decision_layer_size)

        self.classification_layer = nn.Sequential(
            EqualizedLinear(self.layer_channels[0], num_classes),
            nn.Sigmoid()
        )

        # Layer 0
        self.initial_layer = nn.ModuleList([
            EqualizedConv2d(initial_layer_channels + 1, initial_layer_channels, 3, padding=1),
            EqualizedLinear(initial_layer_channels * 4 * 4, initial_layer_channels)
        ])
        self.from_rgb_layers.append(EqualizedConv2d(input_image_channels, initial_layer_channels, 1))

    def add_new_layer(self, new_scale=None):
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
        self.from_rgb_layers.append(EqualizedConv2d(self.input_image_channels,
                                                    depth_new_scale,
                                                    1))

    def set_new_alpha(self, alpha):
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

        x = mini_batch_std_dev(x)

        x = self.leaky_relu(self.initial_layer[0](x))

        x = x.view(-1, num_flat_features(x))
        x = self.leaky_relu(self.initial_layer[1](x))

        decision_if_real = self.decision_layer(x)
        classification_output = self.classification_layer(x)

        return decision_if_real, classification_output
