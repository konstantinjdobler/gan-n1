import argparse
import torch
import torch.nn as nn
from architecture_pgan import Generator
from helper.visualization import save_tensor_as_image
from helper.store import Store


parser = argparse.ArgumentParser("Generate image of faces")

parser.add_argument('--result-path', dest='result_path', type=str, default='./fake_samples/generatedImage.jpg')
parser.add_argument('-c', '--checkpoint-path', dest='checkpoint_path', type=str, default='')
parser.add_argument('-a', '--attributes', type=str, default='')
parser.add_argument('-n', '--number-of-images', dest='number_of_images', type=int, default=1)
parser.add_argument('-r', '--image-resolution', dest='image_resolution', type=int, default=128)


def loadAttributes(attributesPath):
    with open(attributesPath) as file:
        lines = [line.rstrip() for line in file]
    attributes = torch.FloatTensor([list(map(float, line.split(',')[1:])) for line in lines])
    return attributes


if __name__ == '__main__':
    args, _ = parser.parse_known_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        FloatTensor = torch.cuda.FloatTensor
        print("Running on the GPU")
    else:
        FloatTensor = torch.FloatTensor
        device = torch.device("cpu")
        print("Running on the CPU")

    attributes = loadAttributes(args.attributes).to(device)

    states = Store.load(args.checkpoint_path)

    generator = Generator(output_image_channels=3, latent_vector_dimension=states['config']['dim_latent_vector'], initial_layer_channels=states['config']['scaling_layer_channels'][0]).to(device)
    
    for scale_iteration in range(1, states['current_scale_level'] + 1):
        scaling_layer_channel = states['config']['scaling_layer_channels'][scale_iteration]
        generator.add_new_layer(scaling_layer_channel)
    generator.load_state_dict(states['generator_state'])
    generator.to(device)

    labels = attributes.view(40).repeat(args.number_of_images, 1)
    noise = torch.randn(args.number_of_images, states['config']['dim_latent_vector']).to(device)
    faces = generator(noise, labels).detach().cpu()

    save_tensor_as_image(faces.data[:args.number_of_images], args.image_resolution, args.result_path)


