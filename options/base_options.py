import argparse
import os

import torch

from util import util


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='pix2pixHD', help='which model to use')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32],
                                 help="Supported data type i.e. 8, 16, 32 bit")
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        self.parser.add_argument('--fp16', action='store_true', default=False, help='train with AMP')
        self.parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

        # input/output sizes
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--label_nc', type=int, default=0, help='# of input label channels')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        self.parser.add_argument('--primitive', type=str, default='seg_edges',
                                 help="define the primitive image to be used, can be: [manual/seg/edges/seg_edges]"
                                      "for [manual/seg/edges] only one primitive should be provided under <dataroot>/<phase>_<primitive>,"
                                      "for [seg_edges], you must provide the segmentation map and edges in two separate "
                                      "files under <dataroot>/seg and <dataroot>/edges, note that the edge map must be binary"
                                      "if only the segmentation map is provided, the edge map will be created automatically using canny")
        self.parser.add_argument('--dataroot', type=str, default='./datasets/face/')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # augmentations
        self.parser.add_argument('--resize_or_crop', type=str, default='none',
                                 help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data argumentation')

        self.parser.add_argument('--affine_aug', type=str, default="none",
                                 help='which affine transformations to apply, combination of: '
                                      '"shearx_sheary_translationx_translationy_rotation", each of the chosen '
                                      'transformation will be applied with probability 0.5')

        self.parser.add_argument('--tps_aug', type=int, default=0, help='apply tps augmentations during training')
        self.parser.add_argument('--tps_points_per_dim', type=int, default=3)
        self.parser.add_argument('--tps_percent', type=float, default=0.99, help='apply tps augmentations during training')

        self.parser.add_argument('--cutmix_aug', type=int, default=0, help='apply cutmix augmentations during training')
        self.parser.add_argument('--cutmix_min_size', type=int, default=32, help='cutmix min patch size')
        self.parser.add_argument('--cutmix_max_size', type=int, default=96, help='cutmix max patch size')

        self.parser.add_argument('--canny_aug', type=int, default=0, help='apply canny augmentations during training')
        self.parser.add_argument('--canny_color', type=int, default=0, help='should the canny edges be black or white (0=black, 1=white)')
        self.parser.add_argument('--test_canny_sigma', type=float, default=2, help='the canny sigma value for the test images')
        self.parser.add_argument('--canny_sigma_l_bound', type=float, default=1.2, help='lower bound for cannys sigma value')
        self.parser.add_argument('--canny_sigma_u_bound', type=float, default=3, help='upper bound for cannys sigma value')
        self.parser.add_argument('--canny_sigma_step', type=float, default=0.3, help='step size for cannys sigma value')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512, help='display window size')
        self.parser.add_argument('--tf_log', action='store_true',
                                 help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_downsample_global', type=int, default=4,
                                 help='number of downsampling layers in netG')
        self.parser.add_argument('--n_blocks_global', type=int, default=9,
                                 help='number of residual blocks in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3,
                                 help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')
        self.parser.add_argument('--niter_fix_global', type=int, default=0,
                                 help='number of epochs that we only train the outmost local enhancer')

        # for instance-wise features
        self.parser.add_argument('--no_instance', action='store_true',
                                 help='if specified, do *not* add instance map as input')
        self.parser.add_argument('--instance_feat', action='store_true',
                                 help='if specified, add encoded instance features as input')
        self.parser.add_argument('--label_feat', action='store_true',
                                 help='if specified, add encoded label features as input')
        self.parser.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')
        self.parser.add_argument('--load_features', action='store_true',
                                 help='if specified, load precomputed feature maps')
        self.parser.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder')
        self.parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        self.parser.add_argument('--n_clusters', type=int, default=10, help='number of clusters for features')

        self.parser.add_argument('--name', type=str)
        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test
        print("name", self.opt.name)

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        print(self.opt.gpu_ids)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')

        self.opt.affine_transforms = {"shearx": [-0.3, 0.3],
                                      "sheary": [-0.3, 0.3],
                                      "translationx": [-0.2, 0.2],
                                      "translationy": [-0.2, 0.2],
                                      "rotation": [-0.1, 0.1]}
        return self.opt
