from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import numpy as np
import os, fnmatch
import imageio
import matplotlib.pyplot as plt
import skimage.transform

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()    
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def crop_im(input_path, output_path):
    im = imageio.imread(input_path)
    print(im.shape)
    im = im[:, 420: 1500, :]
    im = skimage.transform.resize(im, (640, 640),
                                          mode='reflect', preserve_range=False,
                                          anti_aliasing=True, order=1).astype("float32")
    print(im.shape)
    plt.imshow(im)
    imageio.imwrite(output_path, im)
    plt.show()


def frames_to_vid(webpage):
    input_path = webpage.get_image_dir()
    output_vid_path = os.path.join(os.path.abspath(os.path.join(input_path, os.pardir)), "vid_res")
    if not os.path.exists(output_vid_path):
        os.mkdir(output_vid_path)
    files_input_label = fnmatch.filter(os.listdir(input_path), '*_input_label.jpg')
    files_synthesize = fnmatch.filter(os.listdir(input_path), '*_synthesized_image.jpg')
    len_files = min(len(files_input_label), len(files_synthesize))
    file_list_input = []
    file_list_synthesize = []
    input_name = "%d_input_label.jpg"
    synthesize_name = "%d_synthesized_image.jpg"
    for i in range(len_files):
        cur_input_path = os.path.join(input_path, (input_name % i))
        cur_synthesize_path = os.path.join(input_path, (synthesize_name % i))
        if os.path.exists(cur_input_path) and os.path.exists(cur_synthesize_path):
            file_list_input.append(cur_input_path)
            file_list_synthesize.append(cur_synthesize_path)
    #
    writer = imageio.get_writer(os.path.join(output_vid_path, 'input_vid.mov'), fps=24)
    for im in file_list_input:
        writer.append_data(imageio.imread(im))
    writer.close()

    writer = imageio.get_writer(os.path.join(output_vid_path, 'synthesize_vid.mov'), fps=24)
    for im in file_list_synthesize:
        writer.append_data(imageio.imread(im))
    writer.close()
    print("vid output saved to [%s]" % output_vid_path)
