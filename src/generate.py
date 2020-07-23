import argparse
import torch
import torch.nn as nn
from architecture import Generator, Discriminator, weights_init
import torchvision.utils as vutils
from torch.autograd import Variable


parser = argparse.ArgumentParser("Generate image of faces")

parser.add_argument('--result-path', dest='result_path', type=str, default='./fake_samples/generatedImage.png')
parser.add_argument('-g', '--generator-path', dest='generator_path', type=str, default='')
parser.add_argument('-a', '--attributes', type=str, default='')
parser.add_argument('-n', '--number-of-images', dest='number_of_images', type=int, default=1)
parser.add_argument('-r', '--image-resolution', dest='image_resolution', type=int, default=256)


def loadAttributes(attributesPath):
    with open(attributesPath) as file:
        lines = [line.rstrip() for line in file]
    attributes = torch.FloatTensor([list(map(float, line.split(',')[-1])) for line in lines])
    return attributes


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

    attributes = loadAttributes(config.attributes).to(device)

    config.target_image_size = config.image_resolution
    config.nz = 100
    config.nc = 3
    config.nfeature = 40
    config.generator_filters = 64

    generator = Generator(config).to(device)
    generator.load_state_dict(torch.load(config.generator_path))
    generator.eval()

    noise = Variable(FloatTensor(config.number_of_images, 100, 1, 1)).to(device)
    noise.data.normal_(0, 1)
    faces = generator(noise, attributes.repeat(config.number_of_images,1), config)

    vutils.save_image(faces.data[:config.number_of_images], f'{config.result_path}', normalize=True)


