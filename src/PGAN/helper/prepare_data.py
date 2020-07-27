import argparse
from PIL import Image
import os

parser = argparse.ArgumentParser("Prescale images for different resolutions")
parser.add_argument('path', type=str, help='Path to the image dataset. The scaled version will be placed next to it.')

def scale_images(dir_path, classes=None):
    classes = [4, 8, 16, 32, 64, 128, 256, 512, 1024] if classes is None else classes
    for i in os.listdir(dir_path):
        img_path = os.path.join(dir_path, i)
        if not os.path.isfile(img_path):
            continue
        im = Image.open(os.path.join(dir_path, i))
        im_info = im.info
        if im is None:
            continue
        o_width, o_height = im.size
        min_dim = o_width if o_width < o_height else o_height
        im = image_centre_crop(im, (min_dim, min_dim))

        for class_ in classes:
            im_resized = resize_image(im, (class_, class_))
            folder = os.path.join(os.path.dirname(dir_path),
                                "{0}_{1}".format(os.path.basename(dir_path), "scaled"),
                                str(class_), str(class_))
            os.makedirs(folder, exist_ok=True)
            im_resized.save(os.path.join(folder, i), **im_info)
    return image_dataset


def resize_image(im, size):
    return im.resize(size, Image.ANTIALIAS)


def image_centre_crop(im, outsize):
    im_width, im_height = im.size
    out_width, out_height = outsize
    left = int(round((im_width - out_width) / 2.))
    upper = int(round((im_height - out_height) / 2.))
    right = left + out_width
    lower = upper + out_height
    return im.crop((left, upper, right, lower))


if __name__ == "__main__":
    args, _ = parser.parse_known_args()
    scale_images(args.path)