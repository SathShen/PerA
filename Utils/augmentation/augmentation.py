import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torch
import random
import PIL
from PIL import Image, ImageFilter, ImageOps
import math
import collections
import sys
from torchvision import transforms
import torchvision.transforms.functional as F
import numbers
import numpy as np


def interp_mode(method):
    if method == 'bicubic':
        return InterpolationMode.BICUBIC
    elif method == 'lanczos':
        return InterpolationMode.LANCZOS
    elif method == 'hamming':
        return InterpolationMode.HAMMING
    elif method == 'nearest':
        return InterpolationMode.NEAREST
    else:
        return InterpolationMode.BILINEAR


class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class PerAToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, normalize=True):
        self.normalize = normalize

    def __call__(self, views):
        if isinstance(views, tuple):
            view0 = views[0]
            view1 = views[1]
        else:
            AssertionError("views should be tuple")

        if self.normalize:
            return F.to_tensor(view0), F.to_tensor(view1)
        else:
            return torch.from_numpy(np.array(view0).transpose((2, 0, 1))),\
                   torch.from_numpy(np.array(view1).transpose((2, 0, 1)))


class PerANormalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, views):
        if isinstance(views, tuple):
            view0 = views[0]
            view1 = views[1]
        else:
            AssertionError("views should be tuple")

        view0 = F.normalize(view0, self.mean, self.std, self.inplace)
        view1 = F.normalize(view1, self.mean, self.std, self.inplace)
        return view0, view1




class PerAColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=(0.8, 0.1), pgray=0.1):
        self.p = p
        self.pgray = pgray
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, image):
        if random.random() < self.p[0]:
            if random.random() < self.pgray:
                view0 = transforms.functional.to_grayscale(image, num_output_channels=3)
            else:
                transform0 = self.get_params(self.brightness, self.contrast,
                                            self.saturation, self.hue)
                view0 = transform0(image)
        else:
            view0 = image.copy()
        
        if random.random() < self.p[1]:
            transform1 = self.get_params(self.brightness, self.contrast,
                                        self.saturation, self.hue)
            view1 = transform1(image)
        else:
            view1 = image.copy()

        return view0, view1


class DinoV1Augmentation(object):
    def __init__(self, cfg):
        color_jitter1 = transforms.RandomApply(torch.nn.ModuleList([
            transforms.ColorJitter(brightness=cfg.AUG.INTENSITY, contrast=cfg.AUG.CONTRAST,
                                   saturation=cfg.AUG.SATURATION, hue=cfg.AUG.HUE)]), p=0.8)
        color_jitter2 = transforms.RandomApply(torch.nn.ModuleList([
            transforms.ColorJitter(brightness=cfg.AUG.INTENSITY, contrast=cfg.AUG.CONTRAST,
                                   saturation=cfg.AUG.SATURATION, hue=cfg.AUG.HUE)]), p=0.1)
        flips = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5)])
        # cutout = Cutout(p=0.2, scale=(0.1, 0.4), ratio=(3./5, 5./3))

        # first global crop
        self.global_transfo1 = transforms.Compose([
            
            transforms.RandomResizedCrop(cfg.AUG.GLOBAL_CROP_SIZE, scale=cfg.AUG.GLOBAL_SCALE, interpolation=InterpolationMode.BILINEAR),
            flips,
            color_jitter1,
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD),
            # HazeSimulation(p=1, t=(0.5, 0.7))
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(cfg.AUG.GLOBAL_CROP_SIZE, scale=cfg.AUG.GLOBAL_SCALE, interpolation=InterpolationMode.BILINEAR),
            flips,
            color_jitter2,
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD),
            # HazeSimulation(p=0.1, t=(0.3, 0.7)),
            # cutout
        ])
        # transformation for the local small crops
        self.num_local_crops = cfg.AUG.NUM_LOCAL
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(cfg.AUG.LOCAL_CROP_SIZE, scale=cfg.AUG.LOCAL_SCALE, interpolation=InterpolationMode.BILINEAR),
            flips,
            color_jitter1,
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD),
            # HazeSimulation(p=0.5, t=(0.5, 0.7))
        ])

    def __call__(self, image):
        # [global 1, global 2, local 1, local 2, local 3...]
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.num_local_crops):
            crops.append(self.local_transfo(image))
        return crops


class DinoV2Augmentation(object):
    def __init__(self, cfg):
        color_jitter1 = transforms.RandomApply(torch.nn.ModuleList([
            transforms.ColorJitter(brightness=cfg.AUG.INTENSITY, contrast=cfg.AUG.CONTRAST,
                                   saturation=cfg.AUG.SATURATION, hue=cfg.AUG.HUE)]), p=0.8)
        color_jitter2 = transforms.RandomApply(torch.nn.ModuleList([
            transforms.ColorJitter(brightness=cfg.AUG.INTENSITY, contrast=cfg.AUG.CONTRAST,
                                   saturation=cfg.AUG.SATURATION, hue=cfg.AUG.HUE)]), p=0.1)
        flips = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5)])
        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(cfg.AUG.GLOBAL_CROP_SIZE, scale=cfg.AUG.GLOBAL_SCALE, interpolation=InterpolationMode.BILINEAR, antialias=True),
            flips,
            color_jitter1,
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD, inplace=True)
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(cfg.AUG.GLOBAL_CROP_SIZE, scale=cfg.AUG.GLOBAL_SCALE, interpolation=InterpolationMode.BILINEAR, antialias=True),
            flips,
            color_jitter2,
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD, inplace=True)
        ])
        # transformation for the local small crops
        self.num_local_crops = cfg.AUG.NUM_LOCAL
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(cfg.AUG.LOCAL_CROP_SIZE, scale=cfg.AUG.LOCAL_SCALE, interpolation=InterpolationMode.BILINEAR, antialias=True),
            flips,
            color_jitter1,
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD, inplace=True)
        ])


    def __call__(self, image):
        output = {}

        # global crops:
        global_crop_1 = self.global_transfo1(image)
        global_crop_2 = self.global_transfo2(image)
        output["global_crops"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [self.local_transfo(image) for _ in range(self.num_local_crops)]
        output["local_crops"] = local_crops
        # output["offsets"] = ()
        return output


class PerAAugmentation(object):
    def __init__(self, cfg):
        self.num_global_crop_pairs = cfg.AUG.NUM_GLOBAL
        self.num_local_crop_pairs = cfg.AUG.NUM_LOCAL

        flips = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5)])
        color_jitter = PerAColorJitter(brightness=cfg.AUG.INTENSITY,
                            contrast=cfg.AUG.CONTRAST,
                            saturation=cfg.AUG.SATURATION,
                            hue=cfg.AUG.HUE,
                            p=(0.8, 0.1))
        
        self.global_pair_trans = transforms.Compose([
            transforms.RandomResizedCrop(cfg.AUG.GLOBAL_CROP_SIZE, 
                                         scale=cfg.AUG.GLOBAL_SCALE, 
                                         ratio=cfg.AUG.GLOBAL_RATIO, 
                                         interpolation=InterpolationMode.BILINEAR, 
                                         antialias=True),
            flips,
            color_jitter,
            PerAToTensor(),
            PerANormalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD, inplace=True)
        ])
        
        self.local_pair_trans = transforms.Compose([
            transforms.RandomResizedCrop(cfg.AUG.LOCAL_CROP_SIZE, 
                                         scale=cfg.AUG.LOCAL_SCALE, 
                                         ratio=cfg.AUG.LOCAL_RATIO, 
                                         interpolation=InterpolationMode.BILINEAR, 
                                         antialias=True),
            flips,
            color_jitter,
            PerAToTensor(),
            PerANormalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD, inplace=True)
        ])

    def __call__(self, image):
        # [global 1, global 2, local 1, local 2, local 3...]
        S_crop_pairs = {}
        T_crop_pairs = {}
        S_crop_pairs['global_crops'] = []
        T_crop_pairs['global_crops'] = []
        S_crop_pairs['local_crops'] = []
        T_crop_pairs['local_crops'] = []
        for _ in range(self.num_global_crop_pairs):
            global_pair = self.global_pair_trans(image)
            S_crop_pairs['global_crops'].append(global_pair[0])
            T_crop_pairs['global_crops'].append(global_pair[1])
        for _ in range(self.num_local_crop_pairs):
            local_pair = self.local_pair_trans(image)
            S_crop_pairs['local_crops'].append(local_pair[0])
            T_crop_pairs['local_crops'].append(local_pair[1])
        return S_crop_pairs, T_crop_pairs




#  -----------------------moco v3 agumentation--------------------------------


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)


class MocoV3Augmentation(object):
    def __init__(self, cfg):
        normalize = transforms.Normalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD, inplace=True)

        # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
        self.base_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(cfg.AUG.GLOBAL_CROP_SIZE, 
                                         scale=cfg.AUG.GLOBAL_SCALE, 
                                         interpolation=InterpolationMode.BILINEAR, 
                                         antialias=True),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=cfg.AUG.INTENSITY, contrast=cfg.AUG.CONTRAST,
                                   saturation=cfg.AUG.SATURATION, hue=cfg.AUG.HUE)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        self.base_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(cfg.AUG.GLOBAL_CROP_SIZE, 
                                         scale=cfg.AUG.GLOBAL_SCALE, 
                                         interpolation=InterpolationMode.BILINEAR, 
                                         antialias=True),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=cfg.AUG.INTENSITY, contrast=cfg.AUG.CONTRAST,
                                   saturation=cfg.AUG.SATURATION, hue=cfg.AUG.HUE)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([Solarize()], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return im1, im2
