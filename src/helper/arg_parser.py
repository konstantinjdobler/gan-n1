from datetime import datetime
import argparse

class ArgParser:

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--dataset-dir',  dest='dataset_dir', type=str, default='../celeba')
        self.parser.add_argument('--result-dir', type=str, default='./fake_samples')
        self.parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
        self.parser.add_argument('--checkpoint-prefix', type=str, default=datetime.now().strftime("%d-%m-%Y_%H_%M_%S"))
        self.parser.add_argument('-s', '--save-checkpoints', dest='save_checkpoints', action='store_true')
        self.parser.add_argument('--nrs', '--no-random-sample', dest='random_sample', action='store_false',
                            help='save random samples of fake faces during training')
        self.parser.add_argument('--ii', '--training-info-interval', dest='training_info_interval', type=int, default=100,
                            help='controls how often during an epoch info is printed')
        self.parser.add_argument('--si', '--sample-interval', dest='sample_interval', type=int, default=16000,
                            help='controls how often during an epoch sample images are saved ')
        self.parser.add_argument('--condition-file', type=str, default='./list_attr_celeba.txt')
        self.parser.add_argument('--batch-size', type=int, default=32)
        self.parser.add_argument('--epochs', type=int, default=20)
        self.parser.add_argument('--workers', type=int, default=4)
        self.parser.add_argument('--nz', type=int, default=100)  # number of noise dimension
        self.parser.add_argument('--nc', type=int, default=3)  # number of result channel
        self.parser.add_argument('--nfeature', type=int, default=40)
        self.parser.add_argument('--lr', type=float, default=0.0001)
        self.parser.add_argument('--seed', dest='manual_seed', type=int, required=False)
        self.parser.add_argument('-g', '--generator-path', dest='generator_path', help='use pretrained generator')
        self.parser.add_argument('-d', '--discriminator-path', dest='discriminator_path', help='use pretrained discriminator')
        self.parser.add_argument('--no-label-smoothing', dest='label_smoothing', action='store_false')
        self.parser.add_argument('--no-label-flipping', dest='label_flipping', action='store_false')

        self.parser.add_argument('--print-loss', dest='print_loss', action='store_true')
        self.parser.add_argument('--show-loss-plot', dest='show_loss_plot', action='store_true')

        self.parser.add_argument('--fixed-noise-sample', dest='fixed_noise_sample', action='store_true',
                            help='show model progression by generating samples with the same fixed noise vector during training')
        self.parser.add_argument('--target-image-size', type=int, default=64)
        self.parser.add_argument('--gf', '--generator-filters', dest='generator_filters', type=int, default=64)
        self.parser.add_argument('--df', '--discriminator-filters', dest='discriminator_filters', type=int, default=64)
        self.parser.add_argument('--no-hd-crop', dest='hd_crop', action='store_false')
        self.parser.add_argument('--init-depth-scale', type=int, default=512)

        self.parser.add_argument('--checkpoint', type=str, default=None)


        self.parser.set_defaults(save_checkpoints=False, random_sample=True, label_flipping=True,
                            label_smoothing=True, print_loss=False, show_loss_plot=False, fixed_noise_sample=False, hd_crop=True)

    def get_config(self):
        config, _ = self.parser.parse_known_args()
        return config
    
        