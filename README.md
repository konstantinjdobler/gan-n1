# GANs for generating novel face images
Deep Learning Course @ HPI SoSe 2020
Group N1

This repository contains two GAN architectures for generating novel images. We tested them on the CelebA dataset to generate faces.
We got inspiration and also used code fragments from:
* https://github.com/shanexn/pytorch-pggan
* https://github.com/facebookresearch/pytorch_GAN_zoo
* https://github.com/caffeinism/cDC-GAN-pytorch
* https://github.com/eriklindernoren/PyTorch-GAN


# Installation
This project requires Python 3.8. Install all necessary libraries from the requirements.txt.

# Content of this repository
As the architectures of our two models are very different, we decided to separate their source code into different subfolders in order to keep the code from growing too complex.
## DCGAN
In src/DCGAN, you can find all necessary files for using the DCGAN architecture.

### Training
Call **train.py** to initiate a training
```
python train.py --dataset-dir path/to/dataset --condition-file /path/to/condition/file.txt
```
Additional parameters:
* *result-dir*: Path to the parent directory of the folder, where images, checkpoints and loss files are saved
* *checkpoint-prefix*: Name of the folder, where images, checkpoints and loss files are saved (the current datetime by default)
* *generator-path*: Path to a pretrained generator to resume training
* *discriminator-path*: Path to a pretrained discriminator to resume training
* *no-checkpoints-save*: If activated, np checkpoints are saved during training
* *no-random-sample*: If activated, no image samples are generated during training
* *sample-interval*: How often during training, images samples should be generated and checkpoints should be saved
* *fixed-noise-sample*: If activated, a fixed noise sample is used for generating images during training
* *show-loss-plot*: If activated, a loss plot is shown during training
* *seed*: The seed for the PRNG
* *workers*: The number of workers used for data loading
* *batch-size*: The batch size to use
* *epochs*: The number of epochs to train
* *no-label-smoothing*: Switches off label smoothing
* *no-label-flipping*: Switches off label flipping
* *target-image-size*: The size of the training images


### Image generation
Call **generate.py** to generate novel images, using a trained net
```
python generate.py --generator-path path/to/generator --atributes path/to/attributes/file --result-path /result/image/path.png
```
Additional parameters:
* *number-of-images*: The number of images that should be generated. All the images are saved as one image collage
* *image-resolution*: The resolution of the generated images (must be equal to the resolution, the generator was trained on)

You can either use one of the prefilled attribute files in src/attribute_files or customize your own attribute_file. 

## PGAN

In src/PGAN, you can find all necessary files for using the PGAN architecture.

### Training
It can speed up the training if the images are resized prior to training. You can use the helper/prepare_data.py for this.
```
python prepare_data.py /path/to/folder/in/which/the/dataset/folder/is/located
```

Call **train.py** to initiate a training
```
python train.py --dataset-dir path/to/dataset --condition-file /path/to/condition/file.txt
```
The path to the dataset is the folder in which either the folder with all images or the folders containing different prescaled datasets lie.

Additional parameters:
* *checkpoint*: Path to a checkpoint for resuming a previous training
* *result-dir*: Path to the parent directory of the folder, where images, checkpoints and loss files are saved
* *checkpoint-prefix*: Name of the folder, where images, checkpoints and loss files are saved (the current datetime by default)
* *no-checkpoints-save*: If activated, no checkpoints are saved during training
* *checkpoint-interval*: How often during training, checkpoints should be saved
* *no-random-sample*: If activated, no image samples are generated during training
* *sample-interval*: How often during training, images samples should be generated
* *fixed-noise-sample*: If activated, a fixed noise sample is used for generating images during training
* *training-info-interval*: How often during training, information about the current iteration, loss and alpha value should be printed
* *workers*: The number of workers used for data loading
* *seed*: The seed for the PRNG


### Image generation
Call **generate.py** to generate novel images, using a trained net
```
python generate.py --checkpoint-path path/to/generator/checkpoint --atributes path/to/attributes/file --result-path /result/image/path.png
```
Additional parameters:
* *number-of-images*: The number of images that should be generated. All the images are saved as one image collage
* *image-resolution*: Only influences the resolution with which the images are saved. The resolution of the generated images is defined by the used generator

### Loss evaluation
Call **helper/make_chart.py** to plot the losses that are written per resolution as txt file.
```
python make_chart.py /path/to/loss/file.txt --chart-title Losses4x4
```
