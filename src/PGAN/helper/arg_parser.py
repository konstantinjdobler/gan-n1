from datetime import datetime
import argparse

class ArgParser:

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--dataset-dir',  dest='dataset_dir', type=str, default='../celeba')
        self.parser.add_argument('--condition-file', type=str, default='./list_attr_celeba.txt')
        self.parser.add_argument('--checkpoint', type=str, default=None)

        self.parser.add_argument('--result-dir', type=str, default='./fake_samples')
        self.parser.add_argument('--checkpoint-prefix', type=str, default=datetime.now().strftime("%d-%m-%Y_%H_%M_%S"))

        self.parser.add_argument('-ncs', '--no-checkpoints-save', dest='save_checkpoints', action='store_false',
                            help='do not save checkpoints in regular intervals during training')
        self.parser.add_argument('--nrs', '--no-random-sample', dest='random_image_samples', action='store_false',
                            help='do not save random samples of fake faces during training')
        self.parser.add_argument('--ii', '--training-info-interval', dest='training_info_interval', type=int, default=100,
                            help='controls how often during training info is printed')
        self.parser.add_argument('--si', '--sample-interval', dest='sample_interval', type=int, default=16000,
                            help='controls how often during training sample images are saved ')
        self.parser.add_argument('--ci', '--checkpoint-interval', dest='checkpoint_interval', type=int, default=16000,
                            help='controls how often during training a checkpoint is saved')

        self.parser.add_argument('--workers', type=int, default=8)

        self.parser.add_argument('--seed', dest='manual_seed', type=int, required=False)
        self.parser.add_argument('--fixed-noise-sample', dest='fixed_noise_sample', action='store_true',
                            help='show model progression by generating samples with the same fixed noise vector during training')

        self.parser.set_defaults(save_checkpoints=True, random_image_samples=True,
                            fixed_noise_sample=False)

    def get_config(self):
        config, _ = self.parser.parse_known_args()
        return config
    
        