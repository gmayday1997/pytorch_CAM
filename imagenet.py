import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import scipy.io
import scipy.misc as m
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImagenetDataset(Dataset):

    def __init__(self,img_path,label_path,file_name_txt_path,split_flag, transform=None, target_transform=None,loader= default_loader):

        self.label_path = label_path
        self.img_path = img_path
        self.img_txt_path = file_name_txt_path
        self.flag = split_flag
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.read_label()
        self.imgs = self.read_images()

    def read_label(self):

        if self.flag == 'train':

            self.clidx_subfold_dict = {}
            self.clidx_name_dict = {}
            map_cl_name_subfold = np.loadtxt(self.label_path,dtype=str)
            for m in map_cl_name_subfold:

                assert len(m) == 3
                subfold_idx, idx, cl_name = m
                self.clidx_subfold_dict.setdefault(subfold_idx,int(idx)-1)
                self.clidx_name_dict.setdefault(int(idx)-1,cl_name)

        if self.flag == 'valid':

            self.cl_list = np.loadtxt(self.label_path, dtype=str)

    def make_fold_dataset(self):

        dir = self.img_path
        class_to_idx = self.clidx_subfold_dict
        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)

        return images

    def make_single_file_dataset(self):

        dir = self.img_path
        cl_list = self.cl_list
        img_list = np.loadtxt(self.img_txt_path,dtype= str)
        images = []
        assert len(img_list) == len(cl_list)
        for img_path,cl in zip(img_list,cl_list):

            full_img_path = os.path.join(dir,img_path)
            item = (full_img_path,int(cl)-1)
            images.append(item)

        return images

    def read_images(self):

        if self.flag == 'train':
            imgs = self.make_fold_dataset()

        if self.flag == 'valid':
            imgs = self.make_single_file_dataset()

        return imgs

    def __getitem__(self, index):

        path, target = self.imgs[index]
        img = self.loader(path)
        #print img.numpy()
        if self.transform is not None:
            img = self.transform(img)
            #im = img.numpy()
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):

        return len(self.imgs)
