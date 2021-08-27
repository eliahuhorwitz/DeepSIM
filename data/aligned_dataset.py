import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import torch
import imageio
from skimage import util, feature
from skimage.color import rgb2gray
import numpy as np

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        if opt.primitive != "seg_edges":
            dir_A = "_" + opt.primitive
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            self.A_paths = sorted(make_dataset(self.dir_A))
            self.A = Image.open(self.A_paths[0])

        else:
            self.dir_A = os.path.join(opt.dataroot, opt.phase + "_seg")
            self.A_paths = sorted(make_dataset(self.dir_A))
            # the seg input will be saved as "A"
            self.A = Image.open(self.A_paths[0])
            self.dir_A_edges = os.path.join(opt.dataroot, opt.phase + "_edges")
            self.A_paths_edges = sorted(make_dataset(self.dir_A_edges))
            if not os.path.exists(self.dir_A_edges):
                os.makedirs(self.dir_A_edges)
            self.A_edges = Image.open(self.A_paths_edges[0]) if self.A_paths_edges else None

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))
            self.B = Image.open(self.B_paths[0]).convert('RGB')
            if opt.primitive == "seg_edges" and not self.A_edges:
                self.A_edges = Image.fromarray(util.invert(feature.canny(rgb2gray(np.array(self.B)), sigma=0.5)))

        self.adjust_input_size(opt)

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))
        self.dataset_size = len(self.A_paths)


    def adjust_input_size(self, opt):
        """
        change image size once when loading the image.
        :return:
        """
        # TODO: if we use instance map, support resize method = NEAREST
        ow, oh = self.A.size
        # for cuda memory capacity
        if max(ow, oh) > 1000:
            ow, oh = ow // 2, oh // 2

        if opt.resize_or_crop == 'none' or opt.resize_or_crop == "crop":
            # input size should fit architecture params
            base = float(2 ** opt.n_downsample_global)
            if opt.netG == 'local':
                base *= (2 ** opt.n_local_enhancers)
            new_h = int(round(oh / base) * base)
            new_w = int(round(ow / base) * base)

        elif 'resize' in opt.resize_or_crop:
            new_h, new_w = opt.loadSize, opt.loadSize

        elif 'scale_width' in opt.resize_or_crop:
            if ow != opt.loadSize:
                new_w = opt.loadSize
                new_h = int(opt.loadSize * oh / ow)

        self.A = self.A.resize((new_w, new_h), Image.NEAREST)
        if opt.primitive == "seg_edges":
            self.A_edges = self.A_edges.resize((new_w, new_h), Image.NEAREST)
        if self.opt.isTrain:
            self.B = self.B.resize((new_w, new_h), Image.BICUBIC)

    def __getitem__(self, index):
        A = self.A
        B = self.B
        params = get_params(self.opt, B.size, B)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params, is_primitive=True, is_edges=self.opt.primitive == "edges")
            A_img = A.convert('RGB')
            A_tensor = transform_A(A_img)
            if self.opt.primitive == "seg_edges":
                # apply transforms only on the edges and then fuse it to the seg
                transform_A_edges = get_transform(self.opt, params, is_primitive=True, is_edges=True)
                A_edges = self.A_edges.convert('RGB')
                A_edges_tensor = transform_A_edges(A_edges)
                if self.opt.canny_color == 0:
                    A_tensor[A_edges_tensor == A_edges_tensor.min()] = A_edges_tensor.min()
                else:
                    A_tensor[A_edges_tensor == A_edges_tensor.max()] = A_edges_tensor.max()
            
        else:
            transform_A = get_transform(self.opt, params, normalize=False, is_primitive=True)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B = self.B
            transform_B = get_transform(self.opt, params, is_primitive=False)
            B_tensor = transform_B(B)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': self.A_paths[0]}

        return input_dict

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'


# TODO : fix test loader as well with scale adjustment
class AlignedDataset_test(BaseDataset):
    def initialize(self, opt):
        print("in initialize")
        self.opt = opt
        self.root = opt.dataroot

        if opt.vid_input:
            print(os.path.join(opt.dataroot + opt.phase))
            reader = imageio.get_reader(os.path.join(opt.dataroot + opt.phase), 'ffmpeg')
            opt.phase = 'vid_frames'
            dir_A = "_" + opt.primitive if self.opt.primitive != "seg_edges" else '_seg'
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            if not os.path.exists(self.dir_A):
                os.mkdir(self.dir_A)
            i = 0
            for im in reader:
                print(i)
                if i == 240:
                    break
                imageio.imwrite("%s/%d.png" % (self.dir_A, i), im)
                i += 1

        ### input A (label maps)
        dir_A = "_" + opt.primitive if self.opt.primitive != "seg_edges" else '_seg'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        if opt.primitive == "seg_edges":
            self.dir_A_edges = os.path.join(opt.dataroot, opt.phase + "_edges")
            if not os.path.exists(self.dir_A_edges):
                os.makedirs(self.dir_A_edges)
            self.A_paths_edges = sorted(make_dataset(self.dir_A_edges))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths)
        print("dataset_size", self.dataset_size)

    def adjust_input_size(self, opt):
        """
        change image size once when loading the image.
        :return:
        """
        ow, oh = self.A.size
        # for cuda memory capacity
        if max(ow, oh) > 1000:
            ow, oh = ow // 2, oh // 2

        if opt.resize_or_crop == 'none':
            # input size should fit architecture params
            base = float(2 ** opt.n_downsample_global)
            if opt.netG == 'local':
                base *= (2 ** opt.n_local_enhancers)
            new_h = int(round(oh / base) * base)
            new_w = int(round(ow / base) * base)

        elif 'resize' in opt.resize_or_crop:
            new_h, new_w = opt.loadSize, opt.loadSize

        elif 'scale_width' in opt.resize_or_crop:
            if ow != opt.loadSize:
                new_w = opt.loadSize
                new_h = int(opt.loadSize * oh / ow)

        self.A = self.A.resize((new_w, new_h), Image.NEAREST)
        if opt.primitive == "seg_edges":
            self.A_edges = self.A_edges.resize((new_w, new_h), Image.NEAREST)

    def __getitem__(self, index):
        ### input A (label maps)
        A_path = self.A_paths[index]
        self.A = Image.open(A_path)
        params = get_params(self.opt, self.A.size, self.A)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params, is_primitive=True)
            self.A = self.A.convert('RGB')
            if self.opt.primitive == "seg_edges":
                transform_A_edges = get_transform(self.opt, params, is_primitive=True, is_edges=True)
                if self.A_paths_edges:
                    self.A_edges = Image.open(self.A_paths_edges[index])
                else:
                    # at inference time in this mode you must provide the ground truth to extract the edges from
                    self.dir_B = os.path.join(self.opt.dataroot, self.opt.phase + "_B")
                    self.B_paths = sorted(make_dataset(self.dir_B))
                    self.B = Image.open(self.B_paths[index]).convert('RGB')
                    A_edges = feature.canny(rgb2gray(np.array(self.B)), sigma=self.opt.test_canny_sigma)
                    if self.opt.canny_color ==0:
                        self.A_edges = Image.fromarray(util.invert(A_edges))
                    else:
                        self.A_edges = Image.fromarray(A_edges)
                self.A_edges = self.A_edges.convert('RGB')
            self.adjust_input_size(self.opt)
            A_tensor = transform_A(self.A)
            if self.opt.primitive == "seg_edges":
                A_edges_tensor = transform_A_edges(self.A_edges)
                if self.opt.canny_color == 0:
                    A_tensor[A_edges_tensor == A_edges_tensor.min()] = A_edges_tensor.min()
                else:
                    A_tensor[A_edges_tensor == A_edges_tensor.max()] = A_edges_tensor.max()


        else:
            transform_A = get_transform(self.opt, params, normalize=False, is_primitive=True)
            A_tensor = transform_A(self.A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
                      'feat': feat_tensor, 'path': A_path}
        return input_dict

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_test'
