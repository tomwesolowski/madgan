import numpy as np
import os

import datasets
import utils

from parameters import HParamSelect, ParamsDict

'''
Params definition. You can use use auxiliary classes like HParamSelect and HParamRange
to add some randomness into exact values selection process.
'''
params_defs = ParamsDict(
    description='mnist',
    batch_size=HParamSelect([128]),
    noise_size=64,
    noise_scale=0.1,
    pretrain_steps=10**2,
    steps=20000,
    epochs=1,
    dis_lr=0.0001,
    dis_filters=HParamSelect([96]),
    dis_filters_size=HParamSelect([5]),
    num_generators=10,
    gen_lr=0.0001,
    gen_filters=HParamSelect([96]),
    gen_filters_size=HParamSelect([5]),
    gen_keep_dropout=0.9,
    use_batch_norm=HParamSelect([True]),
    loss_diff_threshold=10.0,
    loss_diff_threshold_back=1.0,
    gen_scope='gen',
    dis_scope='dis',
    save_steps=np.array([i for i in range(5)]) * 4000,
    save_old_steps=np.arange(20) * 10000,
    switch_model_loss_decay=0.95,
    summaries_steps=25,
    prints_steps=20,
    draw_steps=4000,
    debug=False,
    images_dir='~/storage/madgan/images',
    checkpoints_dir='~/storage/madgan/checkpoints',
    summaries_dir='~/storage/madgan/summaries',
    # To run on CIFAR-10, please use Cifar10Dataset
    dataset=lambda: datasets.MnistDataset('~/data/mnist'),
    gaussian_mis=[10, 20, 60, 80, 110],
    gaussian_sigmas=[3, 3, 2, 2, 1],
    gaussian_size=2000,
    model_path='',
    mode='train',
    train_test='test',
    nb_generated=10000,
    take_dis_layer='prob',
    show_h_images=5,
    show_w_images=5,
    show_figsize=5)


class GANParams(ParamsDict):
    """A GANParams contains all the settings needed for GAN training."""

    def __init__(self, *args, **kwargs):
        super(GANParams, self).__init__(*args, **kwargs)
        # Parse parameters.
        self.name = 'gan_%s_%s_gen_%d' % (
            self.description,
            utils.get_date(),
            self.num_generators)
        self.checkpoint_dir = os.path.expanduser('%s/%s' % (self.checkpoints_dir, self.name))
        self.images_dir = os.path.expanduser('%s/%s/' % (self.images_dir, self.name))
        self.summaries_dir = os.path.expanduser('%s/%s' % (self.summaries_dir, self.name))
        for directory in [self.checkpoint_dir, self.images_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)