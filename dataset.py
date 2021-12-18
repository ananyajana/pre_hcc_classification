import torch.utils.data as data
from PIL import Image
import h5py
import numpy as np
import torch
from skimage.util import random_noise
from skimage.transform import rotate
from skimage import exposure
from numpy import random

def is_hdf5_file(filename):
    return filename.lower().endswith('.h5')


def get_keys(hdf5_path):
    with h5py.File(hdf5_path, 'r') as file:
        return list(file.keys())

def h5_loader6(data, opt=None):
    ct_data = data['CT']
    label = data['label'][()]
    scale = []

    ct_imgs = []
    for key in ct_data.keys():
        scale.append(1)         # scale 1 means resolution 224x224 resized from 512x512
        img = ct_data[key][()]
        if opt is not None and opt.model['use_resnet'] == 1:
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.repeat(img, 3, axis=2)
            ct_imgs.append(Image.fromarray(img.astype('uint8'), 'RGB'))
        else:
            ct_imgs.append(Image.fromarray(img.astype('uint8'), 'L'))

    # get the crops now
    ct_data = data['CT2']
    for key in ct_data.keys():
        scale.append(2)         # scale 2 means resolution 224x224 croppped from 512x512 would be used for an auxiliary task
        img = ct_data[key][()]
        if opt is not None and opt.model['use_resnet'] == 1:
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.repeat(img, 3, axis=2)
            ct_imgs.append(Image.fromarray(img.astype('uint8'), 'RGB'))
        else:
            ct_imgs.append(Image.fromarray(img.astype('uint8'), 'L'))

    # get the crops now
    ct_data = data['CT3']
    for key in ct_data.keys():
        scale.append(0)         # scale 0 means resolution 224 downsized to 112x122 and then upsampled, this would be used for an auxiliary task
        img = ct_data[key][()]
        # this is because resnet uses 3 channels and we append the same data in every channel
        if opt is not None and opt.model['use_resnet'] == 1:
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.repeat(img, 3, axis=2)
            ct_imgs.append(Image.fromarray(img.astype('uint8'), 'RGB'))
        else:
            ct_imgs.append(Image.fromarray(img.astype('uint8'), 'L'))

    idx = np.arange(len(scale))
    # we fix the seed for reproducibility of result
    random.seed(3)
    random.shuffle(idx)

    # shuffle the corresponding images and labels else the labels would
    # be consecutive groups
    ct_imgs_shfld = []
    scales_shfld = []
    for i in range(len(idx)):
        ct_imgs_shfld.append(ct_imgs[idx[i]])
        scales_shfld.append(scale[idx[i]])

    return ct_imgs_shfld, label, scales_shfld

class HCCDataset6(data.Dataset):
    def __init__(self, hdf5_path, transform, opt=None):
        super(HCCDataset6, self).__init__()
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.keys = get_keys(self.hdf5_path)
        self.opt = opt

    def __getitem__(self, index):
        hdf5_file = h5py.File(self.hdf5_path)
        slide_data = hdf5_file[self.keys[index]]
        ct_scan, label, scale = h5_loader6(slide_data, self.opt)
        ct_tensor = []
        for i in range(len(ct_scan)):
            ct_tensor.append(self.transform(ct_scan[i]).unsqueeze(0))
        
        scale = torch.from_numpy(np.array(scale))
        return torch.cat(ct_tensor, dim=0), \
            torch.tensor(label).unsqueeze(0).long(),\
            scale.unsqueeze(0).long()

    def __len__(self):
        return len(self.keys)    
    
