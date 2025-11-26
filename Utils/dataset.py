import torch.utils.data as data
import torch
import numpy as np
import os
import sys
sys.path.append('./')
import random
from PIL import Image


def is_image_file(img_ext):
    return img_ext == 'tif' or img_ext == 'tiff' or img_ext == 'png' or img_ext == 'jpg' or img_ext == 'jpeg' or img_ext == 'JPEG'

def PIL_loader(img_path):
    if is_image_file(img_path.split('.')[-1]):
        img = Image.open(img_path)
        return img

def one_hot_encode(label, label_idx_list):
    num_classes = len(label_idx_list)
    new_label = torch.full_like(label, 99, dtype=torch.long)
    for i, label_idx in enumerate(label_idx_list):
        isIdx = (label == label_idx)
        new_label = torch.where(isIdx, torch.tensor(i, dtype=torch.long), new_label)
    if (new_label == 99).any():
        raise ValueError("Label index not found in label_idx_list, please check the inputs!")
    _, H, W = label.shape
    output = torch.zeros((num_classes, H, W), dtype=torch.uint8).scatter(0, new_label, 1)
    return output


class PretrainDataset(data.Dataset):
    def __init__(self, cfg, trans, data_path, img_loader=None):
        self.data_path = data_path
        self.img_path_list = []
        self.get_img_path_list_from_dir(self.data_path)

        if img_loader is None:
            self.loader = PIL_loader
        else:
            self.loader = img_loader
        
        self.trans = trans

    def get_img_path_list_from_dir(self, dir_path):
        img_fullname_list = os.listdir(dir_path)
        for img_fullname in img_fullname_list:
            if len(img_fullname.split('.')) == 1:
                self.get_img_path_list_from_dir(dir_path + '/' + img_fullname)
            elif is_image_file(img_fullname.split('.')[-1]):
                self.img_path_list.append(dir_path + '/' + img_fullname)
            else:
                continue

    def __getitem__(self, idx):
        img = None
        while img is None:
            img_path = self.img_path_list[idx]
            img = self.loader(img_path)
            idx = random.randint(0, self.__len__() - 1)
        auged_crops_list = self.trans(img)    # chw
        return auged_crops_list

    def __len__(self):
        return len(self.img_path_list)
