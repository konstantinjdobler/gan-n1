import torch
import torchvision.transforms as transforms
import random
from PIL import Image
import numpy as np

# class NumpyResize(object):

#     def __init__(self, size):
#         self.size = size

#     def __call__(self, img):
#         if not isinstance(img, Image.Image):
#             img = Image.fromarray(img)
#         return np.array(img.resize(self.size, resample=Image.BILINEAR))

#     def __repr__(self):
#         return self.__class__.__name__ + '(p={})'.format(self.p)


class NumpyFlip(object):

    def __init__(self, p=0.5):
        self.p = p
        random.seed(None)

    def __call__(self, img):
        if random.random() < self.p:
            return np.flip(img, 1).copy()
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
class NumpyToTensor(object):

    def __init__(self, size = None):
        self.size = size
        return

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        if self.size is not None:
            img = np.array(img.resize(self.size, resample=Image.BILINEAR))
        else:
            img = np.array(img)

        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)

        return transforms.functional.to_tensor(img)