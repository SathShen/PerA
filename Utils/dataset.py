import torch.utils.data as data
import torch
import numpy as np
import os
import sys
sys.path.append('./')
import random
from Utils.plot import show_examples, show_augs
from yacs.config import CfgNode as CN
import torchvision.transforms.functional as F
import xml.etree.ElementTree as ET
import torch.distributed as dist
import time
from PIL import Image


def is_image_file(img_ext):
    return img_ext == 'tif' or img_ext == 'tiff' or img_ext == 'png' or img_ext == 'jpg' or img_ext == 'jpeg' or img_ext == 'JPEG'

def PIL_loader(img_path):
    if is_image_file(img_path.split('.')[-1]):
        img = Image.open(img_path)
        return img

def one_hot_encode(label, label_idx_list):
    """创建独热编码以实现多类loss计算"""
    # 索引为labels(1,H,W),分配1到num_classes维度(0)  zeros[index[0][i][j]] [i] [j] = 1
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
    

"""
image classification dataset builder for finetune
"""
class ImageClassificationDataset(data.Dataset):
    """
    读取图像分类数据集，图像分类数据集文件夹结构如下：
        num_depth: 为数据集文件夹深度，如：/data/class1/class2/image.png，则num_depth=2
        label_depth: label在文件夹名中的位置,如：/data/class1/class2/image.png，data文件夹的深度为0，
                     若要分类class1，则label_class=1
    Args:
        cfg: 配置文件
        data_path: 数据集路径
        is_aug: 是否进行数据增强
        img_loader: 图像读取函数，默认为default_loader
    """
    def __init__(self, cfg, trans, data_path, img_loader=None):
        self.data_path = data_path
        self.num_depth = cfg.FINETUNE.IC.NUM_DEPTH
        self.label_depth = cfg.FINETUNE.IC.LABEL_DEPTH
        self.class_list = self.get_class_list()
        self.num_classes = len(self.class_list)
        cfg.defrost()
        cfg.FINETUNE.IC.NUM_CLASSES = self.num_classes
        cfg.freeze()

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
            if len(img_fullname.split('.')) == 1 and img_fullname:
                self.get_img_path_list_from_dir(dir_path + '/' + img_fullname)
            elif is_image_file(img_fullname.split('.')[-1]):
                image_dict = {}
                image_dict['image_path'] = dir_path + '/' + img_fullname
                image_dict['label'] = dir_path.split('/')[-(self.num_depth + 1 - self.label_depth)]
                self.img_path_list.append(image_dict)
            else:
                continue

    def get_class_list(self):
        class_name_list = self.get_all_class_name(self.data_path, self.label_depth)
        class_name_list.sort()
        class_list = []
        for i, class_str in enumerate(class_name_list):
            class_list.append([i, class_str])
        return class_list
    
    def get_all_class_name(self, root_folder, target_depth, current_depth=0):
        folders = []
        if current_depth == target_depth:
            folders.append(os.path.basename(root_folder))
        elif current_depth < target_depth:
            for folder_name in os.listdir(root_folder):
                folder_path = os.path.join(root_folder, folder_name)
                if os.path.isdir(folder_path):
                    folders.extend(self.get_all_class_name(folder_path, target_depth, current_depth + 1))
        return folders

    def label2id(self, label):
        for class_id, class_name in self.class_list:
            if label == class_name:
                return int(class_id)

    def id2label(self, input_id):
        for class_id, class_name in self.class_list:
            if input_id == class_id:
                return class_name

    def __getitem__(self, idx):
        img = None
        while img is None:
            img_path = self.img_path_list[idx]['image_path']
            label_id = self.label2id(self.img_path_list[idx]['label'])
            img = self.loader(img_path)
            idx = random.randint(0, self.__len__() - 1)
        auged_image = self.trans(img)    # chw
        label_id = torch.tensor(label_id, dtype=torch.long)
        images_dict = {}
        images_dict['images'] = auged_image
        images_dict['label_ids'] = label_id
        images_dict['images_name'] = img_path.split('/')[-1].split('.')[0]

        return images_dict

    def __len__(self):
        return len(self.img_path_list)
    
"""
semantic segmentation dataset builder for finetune
"""
class SemanticSegmentationDataset(data.Dataset):
    def __init__(self, cfg, trans, data_path, img_loader=None):
        self.data_path = data_path
        self.img_folder = f"{data_path}/image"
        self.lab_folder = f"{data_path}/label"
        self.img_fullname_list = os.listdir(self.img_folder)
        self.lab_fullname_list = os.listdir(self.lab_folder)
        self.img_ext = self.img_fullname_list[0].split('.')[-1]  # 图片后缀
        self.lab_ext = self.lab_fullname_list[0].split('.')[-1]
        self.img_name_list = list(map(lambda x: x[:-(len(self.img_ext) + 1)], self.img_fullname_list))
        self.class_list = cfg.FINETUNE.SEG.CLASS_LIST

        if img_loader is None:
            self.loader = PIL_loader
        else:
            self.loader = img_loader
        
        self.trans = trans

    def __getitem__(self, idx):
        img = None
        lab = None
        while (img is None) or (lab is None):
            img_name = self.img_name_list[idx]
            img_path = f"{self.img_folder}/{img_name}.{self.img_ext}"
            lab_path = f"{self.lab_folder}/{img_name}.{self.lab_ext}"
            img = self.loader(img_path)
            lab = self.loader(lab_path)
            idx = random.randint(0, self.__len__() - 1)
        img_auged, lab_auged = self.trans(img, lab)    # chw
        onehot_lab_auged = one_hot_encode(lab_auged.unsqueeze(0), self.class_list)
        images_dict = {}
        images_dict['images'] = img_auged
        images_dict['labels'] = onehot_lab_auged
        images_dict['images_name'] = img_name
        return images_dict

    def __len__(self):
        return len(self.img_name_list)
    
    def pred2label(self, pred):
        pred = torch.argmax(pred, dim=0)
        new_label = torch.zeros_like(pred, dtype=torch.uint8)
        for i, label_idx in enumerate(self.class_list):
            label_idx = torch.tensor(label_idx, dtype=torch.uint8)
            new_label = torch.where(pred == i, label_idx, new_label)
        return new_label
    

""" 
object detection dataset builder for finetune
"""
def get_boxes_from_xml(xml_path):
    if not os.path.exists(xml_path):
        return None, None, None, None
    tree = ET.parse(xml_path) # 解析xml文件
    root = tree.getroot() # 获取根节点
    filename = root.find("filename").text # 获取图片名称

    size = root.find("size") # 获取图片尺寸
    width = int(size.find("width").text) # 获取图片宽度
    height = int(size.find("height").text) # 获取图片高度
    depth = int(size.find("depth").text) # 获取图片深度

    objects = root.findall("object") # 获取所有物体节点
    labels = [] # 存储物体类别
    boxes = [] # 存储物体边界框
    for obj in objects: # 遍历每个物体节点
        name = obj.find("name").text # 获取物体类别
        labels.append(name) # 添加到类别列表
        bndbox = obj.find("bndbox") # 获取物体边界框
        xmin = int(bndbox.find("xmin").text) # 获取左上角x坐标
        ymin = int(bndbox.find("ymin").text) # 获取左上角y坐标
        xmax = int(bndbox.find("xmax").text) # 获取右下角x坐标
        ymax = int(bndbox.find("ymax").text) # 获取右下角y坐标
        boxes.append([xmin, ymin, xmax, ymax])  # 添加到边界框列表
    return filename, (height, width, depth), labels, boxes # 返回图片名称，图片尺寸，物体类别和边界框


class ObjectDetectionDataset(data.Dataset):
    def __init__(self, cfg, trans, data_path, img_loader=None):
        self.data_path = data_path
        self.img_folder = f"{data_path}/image"
        self.anno_folder = f"{data_path}/annotation"
        self.img_fullname_list = os.listdir(self.img_folder)
        self.img_ext = self.img_fullname_list[0].split('.')[-1]  # 图片后缀
        self.img_name_list = list(map(lambda x: x[:-(len(self.img_ext) + 1)], self.img_fullname_list))
        self.class_list = self.get_class_list()
        self.num_classes = len(self.class_list)

        cfg.defrost()
        cfg.FINETUNE.DET.NUM_CLASSES = self.num_classes
        cfg.freeze()

        if img_loader is None:
            self.loader = PIL_loader
        else:
            self.loader = img_loader
        
        self.trans = trans

    def get_class_list(self):
        class_list = []
        for xml in os.listdir(self.anno_folder):
            _, _, labels, _ = get_boxes_from_xml(f"{self.anno_folder}/{xml}")
            for label in labels:
                if label not in class_list:
                    class_list.append(label)
        class_list.sort()
        ret_list = []
        for i, class_str in enumerate(class_list):
            ret_list.append([i + 1, class_str])
        return ret_list
            
    def __len__(self):
        return len(self.img_fullname_list)
    
    def labels2id(self, label):
        for class_id, class_name in self.class_list:
            if label == class_name:
                return class_id
        raise ValueError("Label not found in class_list, please check the inputs!")
    
    def id2label(self, input_id):
        for class_id, class_name in self.class_list:
            if input_id == class_id:
                return class_name
        raise ValueError("ID not found in class_list, please check the inputs!")

    def __getitem__(self, idx):
        img = None
        labels = None
        bboxes = None
        while (img is None) or (labels is None) or (bboxes is None):
            img_name = self.img_name_list[idx]
            img_path = f"{self.img_folder}/{img_name}.{self.img_ext}"
            anno_path = f"{self.anno_folder}/{img_name}.xml"
            img = self.loader(img_path)
            _, _, labels, bboxes = get_boxes_from_xml(anno_path)
            idx = random.randint(0, self.__len__() - 1)
        labels_ids = list(map(lambda x: self.labels2id(x), labels))
        for bbox in bboxes:
            if bbox[0] == bbox[2]:
                bbox[2] += 1
            if bbox[1] == bbox[3]:
                bbox[3] += 1
        bboxes, labels_ids = torch.from_numpy(np.array(bboxes)), torch.from_numpy(np.array(labels_ids))
        imgs_auged, bboxes_auged, labels_auged = self.trans(img, bboxes, labels_ids)    # chw

        return imgs_auged, labels_auged, bboxes_auged, img_name
    

"""
change detection dataset builder for finetune
"""
class ChangeDetectionDatasetSingle(data.Dataset):
    def __init__(self, cfg, trans, data_path, img_loader=None):
        self.data_path = data_path
        self.imgA_folder = f"{data_path}/A"
        self.imgB_folder = f"{data_path}/B"
        self.lab_folder = f"{data_path}/label"
        self.imgA_fullname_list = os.listdir(self.imgA_folder)
        self.imgB_fullname_list = os.listdir(self.imgB_folder)
        self.lab_fullname_list = os.listdir(self.lab_folder)
        self.img_ext = self.imgA_fullname_list[0].split('.')[-1]  # 图片后缀
        self.lab_ext = self.lab_fullname_list[0].split('.')[-1]
        self.img_name_list = list(map(lambda x: x[:-(len(self.img_ext) + 1)], self.imgA_fullname_list))
        self.class_list = cfg.FINETUNE.CD.CLASS_LIST

        if img_loader is None:
            self.loader = PIL_loader
        else:
            self.loader = img_loader
        
        self.trans = trans

    def __getitem__(self, idx):
        imgA = None
        imgB = None
        lab = None
        while (imgA is None) or (imgB is None) or (lab is None):
            img_name = self.img_name_list[idx]
            imgA_path = f"{self.imgA_folder}/{img_name}.{self.img_ext}"
            imgB_path = f"{self.imgB_folder}/{img_name}.{self.img_ext}"
            lab_path = f"{self.lab_folder}/{img_name}.{self.lab_ext}"
            imgA = self.loader(imgA_path)
            imgB = self.loader(imgB_path)
            lab = self.loader(lab_path)
            idx = random.randint(0, self.__len__() - 1)
        imgA_auged, imgB_auged, lab_auged = self.trans(imgA, imgB, lab)    # chw
        onehot_lab_auged = one_hot_encode(lab_auged.unsqueeze(0), self.class_list)
        images_dict = {}
        images_dict['imagesA'] = imgA_auged
        images_dict['imagesB'] = imgB_auged
        images_dict['labels'] = onehot_lab_auged
        images_dict['images_name'] = img_name
        return images_dict

    def __len__(self):
        return len(self.img_name_list)
    
    def pred2label(self, pred):
        pred = torch.argmax(pred, dim=0)
        new_label = torch.zeros_like(pred, dtype=torch.uint8)
        for i, label_idx in enumerate(self.class_list):
            label_idx = torch.tensor(label_idx, dtype=torch.uint8)
            new_label = torch.where(pred == i, label_idx, new_label)
        return new_label
    


class ChangeDetectionDatasetMulti(data.Dataset):
    def __init__(self, cfg, trans, data_path, img_loader=None):
        self.data_path = data_path
        self.imgA_folder = f"{data_path}/A"
        self.imgB_folder = f"{data_path}/B"
        self.labA_folder = f"{data_path}/labelA"
        self.labB_folder = f"{data_path}/labelB"
        self.imgA_fullname_list = os.listdir(self.imgA_folder)
        self.imgB_fullname_list = os.listdir(self.imgB_folder)
        self.labA_fullname_list = os.listdir(self.labA_folder)
        self.labB_fullname_list = os.listdir(self.labB_folder)
        self.img_ext = self.imgA_fullname_list[0].split('.')[-1]  # 图片后缀
        self.lab_ext = self.labA_fullname_list[0].split('.')[-1]
        self.img_name_list = list(map(lambda x: x[:-(len(self.img_ext) + 1)], self.imgA_fullname_list))
        self.class_list = cfg.FINETUNE.CD.CLASS_LIST
        self.mask_id = cfg.FINETUNE.CD.MASK_ID

        if img_loader is None:
            self.loader = PIL_loader
        else:
            self.loader = img_loader
        
        self.trans = trans

    def __getitem__(self, idx):
        imgA = None
        imgB = None
        labA = None
        labB = None
        while (imgA is None) or (imgB is None) or (labA is None) or (labB is None):
            img_name = self.img_name_list[idx]
            imgA_path = f"{self.imgA_folder}/{img_name}.{self.img_ext}"
            imgB_path = f"{self.imgB_folder}/{img_name}.{self.img_ext}"
            labA_path = f"{self.labA_folder}/{img_name}.{self.lab_ext}"
            labB_path = f"{self.labB_folder}/{img_name}.{self.lab_ext}"
            imgA = self.loader(imgA_path)
            imgB = self.loader(imgB_path)
            labA = self.loader(labA_path)
            labB = self.loader(labB_path)
            idx = random.randint(0, self.__len__() - 1)
        # build mask
        mask = np.zeros_like(labA, dtype=np.uint8)
        mask[labA != self.mask_id] = 1
        imgA_auged, imgB_auged, labA_auged, labB_auged, mask_auged = self.trans(imgA, imgB, labA, labB, mask)    # chw
        onehot_labA_auged = one_hot_encode(labA_auged.unsqueeze(0), self.class_list)
        onehot_labB_auged = one_hot_encode(labB_auged.unsqueeze(0), self.class_list)
        onehot_mask_auged = one_hot_encode(mask_auged.unsqueeze(0), [0, 1])
        images_dict = {}
        images_dict['imagesA'] = imgA_auged
        images_dict['imagesB'] = imgB_auged
        images_dict['labelsA'] = onehot_labA_auged
        images_dict['labelsB'] = onehot_labB_auged
        images_dict['masks'] = onehot_mask_auged
        return images_dict

    def __len__(self):
        return len(self.img_name_list)
    
class InferenceDatasetSEG(data.Dataset):
    def __init__(self, cfg, trans, data_path, img_loader=None):
        self.data_path = data_path
        self.finetune_type = cfg.FINETUNE.TYPE
        self.img_path_list = []
        self.get_img_path_list_from_dir(self.data_path)

        if img_loader is None:
            self.loader = PIL_loader
        else:
            self.loader = img_loader
        self.class_list = cfg.FINETUNE.SEG.CLASS_LIST
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
        auged_image = self.trans(img)    # chw
        images_dict = {}
        images_dict['images'] = auged_image
        images_dict['images_name'] = img_path.split('/')[-1].split('.')[0]
        return images_dict

    def __len__(self):
        return len(self.img_path_list)
    
    def pred2label(self, pred):
        pred = torch.argmax(pred, dim=0)
        new_label = torch.zeros_like(pred, dtype=torch.uint8)
        for i, label_idx in enumerate(self.class_list):
            label_idx = torch.tensor(label_idx, dtype=torch.uint8)
            new_label = torch.where(pred == i, label_idx, new_label)
        return new_label
    
class InferenceDatasetCD(data.Dataset):
    def __init__(self, cfg, trans, data_path, img_loader=None):
        self.data_path = data_path
        self.imgA_folder = f"{data_path}/A"
        self.imgB_folder = f"{data_path}/B"
        self.imgA_fullname_list = os.listdir(self.imgA_folder)
        self.imgB_fullname_list = os.listdir(self.imgB_folder)
        self.img_ext = self.imgA_fullname_list[0].split('.')[-1]  # 图片后缀
        self.img_name_list = list(map(lambda x: x[:-(len(self.img_ext) + 1)], self.imgA_fullname_list))
        self.class_list = cfg.FINETUNE.CD.CLASS_LIST

        if img_loader is None:
            self.loader = PIL_loader
        else:
            self.loader = img_loader
        
        self.trans = trans

    def __getitem__(self, idx):
        imgA = None
        imgB = None
        lab = None
        while (imgA is None) or (imgB is None):
            img_name = self.img_name_list[idx]
            imgA_path = f"{self.imgA_folder}/{img_name}.{self.img_ext}"
            imgB_path = f"{self.imgB_folder}/{img_name}.{self.img_ext}"
            imgA = self.loader(imgA_path)
            imgB = self.loader(imgB_path)
            idx = random.randint(0, self.__len__() - 1)
        imgA_auged, imgB_auged = self.trans(imgA, imgB, lab)    # chw
        images_dict = {}
        images_dict['imagesA'] = imgA_auged
        images_dict['imagesB'] = imgB_auged
        images_dict['images_name'] = img_name
        return images_dict

    def __len__(self):
        return len(self.img_name_list)
    
    def pred2label(self, pred):
        pred = torch.argmax(pred, dim=0)
        new_label = torch.zeros_like(pred, dtype=torch.uint8)
        for i, label_idx in enumerate(self.class_list):
            label_idx = torch.tensor(label_idx, dtype=torch.uint8)
            new_label = torch.where(pred == i, label_idx, new_label)
        return new_label


    

# def data_test(cfgs):
#     dataset1 = PretrainDataset(cfgs)
#     num_rows = 2
#     num_cols = 4
#     rint = random.randint(0, dataset1.__len__() - num_cols* num_rows)
#     imgs = []
#     for i in range(num_cols * num_rows):
#         img = dataset1[rint + i]
#         imgs.append(img)
#     show_examples(imgs, num_rows, num_cols)


# def aug_test(cfgs):
#     dataset1 = PretrainDataset(cfgs)
#     num_rows = 5
#     num_cols = 10
#     rint = random.randint(0, dataset1.__len__() - 1)
#     img = dataset1[rint].numpy().transpose((1, 2, 0))           # 因为要做增强，所以要先转numpy
#     trans = transforms.Compose([                # 只要加了jitter和resizecrop中一个就会正常，否则cutout就会重复应用，不知道为什么
#         transforms.ToTensor(),
#         transforms.RandomHorizontalFlip(),  # 随机水平反转
#         transforms.RandomVerticalFlip(),    # 随机垂直反转
#         transforms.RandomResizedCrop(size=(cfgs.AUG.CROP_SIZE, cfgs.AUG.CROP_SIZE), scale=(cfgs.AUG.CROP_PER, 1), 
#                                      ratio=(1 - cfgs.AUG.RESIZE_RATIO, 1 + cfgs.AUG.RESIZE_RATIO)),
#         transforms.ColorJitter(brightness=cfgs.AUG.INTENSITY, contrast=cfgs.AUG.CONTRAST,
#                                saturation=cfgs.AUG.SATURATION, hue=cfgs.AUG.HUE),
#         HazeSimulation(),
#         Cutout(0.5)
#                                ])
#     show_augs(img, trans, num_rows, num_cols)


if __name__ == "__main__":
    test_cfg = CN()
    test_cfg.DATA = CN()
    test_cfg.DATA.TRAIN_DATA_PATH = r'F:\Test_data\GID_water\train\image'
    test_cfg.AUG = CN()
    test_cfg.AUG.IS_AUG = False
    test_cfg.AUG.INTENSITY = 0.4
    test_cfg.AUG.HUE = 0.2
    test_cfg.AUG.SATURATION = 0.3
    test_cfg.AUG.CONTRAST = 0.3
    test_cfg.AUG.CROP_PER = 0.4
    test_cfg.AUG.RESIZE_RATIO = 0.3
    test_cfg.AUG.CROP_SIZE = 512

    # data_test(test_cfg)
    # aug_test(test_cfg)

