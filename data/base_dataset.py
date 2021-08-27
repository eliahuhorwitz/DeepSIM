import random
from random import choices

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from skimage import util, feature
from skimage.color import rgb2gray
from skimage import morphology

from util import tps_warp


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def get_params(opt, size, input_im):
    new_w, new_h = size
    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    params = {'crop_pos': (x, y), 'crop': random.random() > 0.5, "flip": random.random() > 0.5}

    for affine_trans in opt.affine_transforms.keys():
        apply_affine_trans = (random.random() > 0.5) and affine_trans in opt.affine_aug
        if apply_affine_trans:
            params[affine_trans] = random.uniform(opt.affine_transforms[affine_trans][0],
                                                  opt.affine_transforms[affine_trans][1])

    # choose: affine / tps / identity
    apply_tps = random.random() < opt.tps_percent
    apply_affine = not apply_tps
    params["apply_affine"] = apply_affine

    np_im = np.array(input_im)
    if opt.tps_aug:
        src = tps_warp._get_regular_grid(np_im, points_per_dim=opt.tps_points_per_dim)
        dst = tps_warp._generate_random_vectors(np_im, src, scale=0.1 * new_w)
        params['tps'] = {'src': src, 'dst': dst, 'apply_tps': apply_tps}

    if opt.cutmix_aug:
        patch_size = random.randint(opt.cutmix_min_size, opt.cutmix_max_size)
        first_cutmix_x = random.randint(0, np.maximum(0, new_w - patch_size))
        first_cutmix_y = random.randint(0, np.maximum(0, new_h - patch_size))
        second_cutmix_x = random.randint(0, np.maximum(0, new_w - patch_size))
        second_cutmix_y = random.randint(0, np.maximum(0, new_h - patch_size))
        params['cutmix'] = {'first_cutmix_x': first_cutmix_x, 'first_cutmix_y': first_cutmix_y,
                            'second_cutmix_x': second_cutmix_x, 'second_cutmix_y': second_cutmix_y,
                            'patch_size': patch_size, 'apply': random.random() > 0.5}

    if opt.canny_aug:
        canny_img = _create_canny_aug(np_im, opt.canny_color, opt.canny_sigma_l_bound, opt.canny_sigma_u_bound, opt.canny_sigma_step)
        params['canny_img'] = canny_img
    return params


def get_transform(opt, params, normalize=True, is_primitive=False, is_edges=False):
    transform_list = []
    if opt.isTrain:
        # transforms applied only to the primitive
        if is_primitive:
            if opt.canny_aug and is_edges:
                transform_list.append(
                    transforms.Lambda(lambda img: __add_canny_img(img, params['canny_img'])))

        # transforms applied to both the primitive and the real image
        if not opt.no_flip:
            transform_list.append(
                transforms.Lambda(lambda img: __flip(img, params['flip'])))

        if opt.affine_aug != "none":
            transform_list.append(
                transforms.Lambda(lambda img: __affine(img, params)))

        if opt.tps_aug:
            transform_list.append(
                transforms.Lambda(lambda img: __apply_tps(img, params['tps'])))

        if opt.cutmix_aug:
            if 'cutmix' in params:
                transform_list.append(
                    transforms.Lambda(lambda img: __apply_cutmix(img, params['cutmix'])))

        if 'crop' in opt.resize_or_crop:
            transform_list.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.fineSize, params['crop'])))

    if is_edges:
        transform_list.append(transforms.Lambda(lambda img: __binary_thresh(img,opt.canny_color)))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


# ====== primitive and real image augmentations ======
def __crop(img, pos, size, crop):
    if crop:
        ow, oh = img.size
        x1, y1 = pos
        tw = th = size
        if (ow > tw or oh > th):
            im = img.crop((x1, y1, x1 + tw, y1 + th))
            im = im.resize((ow, oh), Image.BICUBIC)
            return im
    return img


def __flip(img, flip):
    if flip:
        im = img.transpose(Image.FLIP_LEFT_RIGHT)
        return im
    return img


def __affine(img, params):
    if params["apply_affine"]:
        affine_map = {"shearx": __apply_shear_x,
                      "sheary": __apply_shear_y,
                      "translationx": __apply_translation_x,
                      "translationy": __apply_translation_y,
                      "rotation": __apply_rotation}
        for affine_trans in affine_map.keys():
            if affine_trans in params.keys():
                img = affine_map[affine_trans](img, params[affine_trans])
    return img


def __apply_tps(img, tps_params):
    new_im = img
    if tps_params['apply_tps']:
        np_im = np.array(img)
        np_im = tps_warp.tps_warp_2(np_im, tps_params['dst'], tps_params['src'])
        new_im = Image.fromarray(np_im)
    return new_im


def __apply_cutmix(img, cutmix_params):
    if cutmix_params["apply"]:
        np_im = np.array(img)
        patch_size = cutmix_params["patch_size"]
        first_patch = np_im[cutmix_params["first_cutmix_y"]:cutmix_params["first_cutmix_y"] + patch_size,
                      cutmix_params["first_cutmix_x"]:cutmix_params["first_cutmix_x"] + patch_size, :].copy()
        second_patch = np_im[cutmix_params["second_cutmix_y"]:cutmix_params["second_cutmix_y"] + patch_size,
                       cutmix_params["second_cutmix_x"]:cutmix_params["second_cutmix_x"] + patch_size, :].copy()
        np_im[cutmix_params["first_cutmix_y"]:cutmix_params["first_cutmix_y"] + patch_size,
        cutmix_params["first_cutmix_x"]:cutmix_params["first_cutmix_x"] + patch_size, :] = second_patch
        np_im[cutmix_params["second_cutmix_y"]:cutmix_params["second_cutmix_y"] + patch_size,
        cutmix_params["second_cutmix_x"]:cutmix_params["second_cutmix_x"] + patch_size, :] = first_patch
        new_im = Image.fromarray(np_im)
        return new_im
    return img


def __apply_shear_x(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))


def __apply_shear_y(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))


def __apply_translation_x(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.2 <= v <= 0.2
    v = v * img.size[0]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def __apply_translation_y(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.2 <= v <= 0.2
    v = v * img.size[1]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def __apply_rotation(img, v):  # [-10, 10]
    v = v * 10
    assert -10 <= v <= 10
    return img.rotate(v)


# ====== primitive augmentations ======
def _create_canny_aug(np_im,canny_color, l_bound, u_bound, step):
    population = np.arange(l_bound, u_bound, step)
    canny_sigma = choices(population)
    img_gray = rgb2gray(np_im)
    img_canny = feature.canny(img_gray, sigma=canny_sigma[0])
    if canny_color ==0:
        img_canny = util.invert(img_canny)
    return img_canny


def __add_canny_img(input_im, canny_img):
    canny_lst = [canny_img.astype(np.int) for i in range(np.array(input_im).ndim)]
    canny_stack = (np.stack(canny_lst, axis=2) * 255).astype(np.uint8)
    return Image.fromarray(canny_stack)


def __binary_thresh(edges,canny_color):
    np_edges = np.array(edges)

    if canny_color == 0:
        np_edges[np_edges != np_edges.max()] = np_edges.min()
    else:
        np_edges[np_edges != np_edges.min()] = np_edges.max()
    return Image.fromarray(np_edges)



