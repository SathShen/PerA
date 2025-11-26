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

# -------------------------------------------------------------
#  Extended Transforms for Semantic Segmentation
# -------------------------------------------------------------
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

class SEGCompose(object):
    """Composes several transforms together.
    Args: transforms (list of ``Transform`` objects): list of transforms to compose."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lbl):
        for t in self.transforms:
            img, lbl = t(img, lbl)
        return img, lbl
    
    def append(self, transform):
        self.transforms.append(transform)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class SEGCenterCrop(object):
    """Crops the given PIL Image at the center.
    Args: size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is made."""
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, lbl):
        """  Args: img (PIL Image): Image to be cropped.
        Returns: PIL Image: Cropped image."""
        return F.center_crop(img, self.size), F.center_crop(lbl, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class SEGRandomScale(object):
    def __init__(self, scale_range, interpolation=InterpolationMode.BILINEAR):
        self.scale_range = scale_range
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be scaled.
            lbl (PIL Image): Label to be scaled.
        Returns:
            PIL Image: Rescaled image.
            PIL Image: Rescaled label.
        """
        assert img.size == lbl.size
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        target_size = (int(img.size[1]*scale), int(img.size[0]*scale))
        return F.resize(img, target_size, self.interpolation, antialias=True), F.resize(lbl, target_size, InterpolationMode.NEAREST)


class SEGScale(object):
    """Resize the input PIL Image to the given scale.
    Args:
        Scale (sequence or int): scale factors
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, scale, interpolation=InterpolationMode.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be scaled.
            lbl (PIL Image): Label to be scaled.
        Returns:
            PIL Image: Rescaled image.
            PIL Image: Rescaled label.
        """
        assert img.size == lbl.size
        target_size = (int(img.size[1]*self.scale), int(img.size[0]*self.scale)) # (H, W)
        return F.resize(img, target_size, self.interpolation, antialias=True), F.resize(lbl, target_size, InterpolationMode.NEAREST)


class SEGRandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, lbl):
        """
            img (PIL Image): Image to be rotated.
            lbl (PIL Image): Label to be rotated.
        Returns:
            PIL Image: Rotated image.
            PIL Image: Rotated label.
        """

        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.resample, self.expand, self.center), F.rotate(lbl, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class SEGRandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(lbl)
        return img, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class SEGRandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be flipped.
            lbl (PIL Image): Label to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
            PIL Image: Randomly flipped label.
        """
        if random.random() < self.p:
            return F.vflip(img), F.vflip(lbl)
        return img, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class SEGPad(object):
    def __init__(self, diviser=32):
        self.diviser = diviser
    
    def __call__(self, img, lbl):
        h, w = img.size
        ph = (h//32+1)*32 - h if h%32!=0 else 0
        pw = (w//32+1)*32 - w if w%32!=0 else 0
        im = F.pad(img, ( pw//2, pw-pw//2, ph//2, ph-ph//2) )
        lbl = F.pad(lbl, ( pw//2, pw-pw//2, ph//2, ph-ph//2))
        return im, lbl


class SEGToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, normalize=True, target_type='uint8'):
        self.normalize = normalize
        self.target_type = target_type

    def __call__(self, pic, lbl):
        """
        Note that labels will not be normalized to [0, 1].
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
            lbl (PIL Image or numpy.ndarray): Label to be converted to tensor. 
        Returns:
            Tensor: Converted image and label
        """
        if self.normalize:
            return F.to_tensor(pic), torch.from_numpy(np.array(lbl, dtype=self.target_type))
        else:
            return torch.from_numpy(np.array(pic, dtype=np.float32).transpose((2, 0, 1))),\
                   torch.from_numpy(np.array(lbl, dtype=self.target_type))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class SEGNormalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, lbl):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            tensor (Tensor): Tensor of label. A dummy input for ExtCompose
        Returns:
            Tensor: Normalized Tensor image.
            Tensor: Unchanged Tensor label
        """
        return F.normalize(tensor, self.mean, self.std, inplace=True), lbl

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class SEGRandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be cropped.
            lbl (PIL Image): Label to be cropped.
        Returns:
            PIL Image: Cropped image.
            PIL Image: Cropped label.
        """
        assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s'%(img.size, lbl.size)
        if self.padding > 0:
            img = F.pad(img, self.padding)
            lbl = F.pad(lbl, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[1] - lbl.size[0]) / 2))

        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[0] - lbl.size[1]) / 2))

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(lbl, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class SEGResize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        return F.resize(img, self.size, self.interpolation, antialias=True), F.resize(lbl, self.size, InterpolationMode.NEAREST)


class SEGRandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                 interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            print("Scale and ratio should be of kind (min, max)")
            sys.exit()

        self.interpolation = interpolation

        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop."""
        if isinstance(img, torch.Tensor):
            _, height, width = img.shape
        elif isinstance(img, PIL.Image.Image):
            width, height = img.size
        else:
            raise TypeError("Unexpected type {}".format(type(img)))
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img, lbl):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=True),\
               F.resized_crop(lbl, i, j, h, w, self.size, InterpolationMode.NEAREST)


    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str})"
        return format_string


class SEGColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
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

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img), lbl

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


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
# -------------------------------------------------------------
#  Augmentations for Change Detection
# -------------------------------------------------------------
class CDSToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, normalize=True, target_type='uint8'):
        self.normalize = normalize
        self.target_type = target_type

    def __call__(self, imga, imgb, lbl):
        """
        Note that labels will not be normalized to [0, 1].
        Args:
            imga (PIL Image or numpy.ndarray): Image to be converted to tensor.
            imgb (PIL Image or numpy.ndarray): Image to be converted to tensor.
            lbl (PIL Image or numpy.ndarray): Label to be converted to tensor. 
        Returns:
            Tensor: Converted image and label
        """
        if lbl is None:
            if self.normalize:
                return F.to_tensor(imga), F.to_tensor(imgb)
            else:
                return torch.from_numpy(np.array(imga, dtype=np.float32).transpose((2, 0, 1))),\
                        torch.from_numpy(np.array(imgb, dtype=np.float32).transpose((2, 0, 1)))
        if self.normalize:
            return F.to_tensor(imga), F.to_tensor(imgb), torch.from_numpy(np.array(lbl, dtype=self.target_type))
        else:
            return torch.from_numpy(np.array(imga, dtype=np.float32).transpose((2, 0, 1))),\
                   torch.from_numpy(np.array(imgb, dtype=np.float32).transpose((2, 0, 1))),\
                   torch.from_numpy(np.array(lbl, dtype=self.target_type))

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class CDSNormalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, imga, imgb, lbl):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(imga, self.mean, self.std, inplace=True), F.normalize(imgb, self.mean, self.std, inplace=True), lbl
    
class CDSCompose(object):
    """Composes several transforms together.
    Args: transforms (list of ``Transform`` objects): list of transforms to compose."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imga, imgb, lbl):
        for t in self.transforms:
            imga, imgb, lbl = t(imga, imgb, lbl)
        return imga, imgb, lbl
    
    def append(self, transform):
        self.transforms.append(transform)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
class CDSResize(object):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imga, imgb, lbl):
        return F.resize(imga, self.size, self.interpolation, antialias=True),\
            F.resize(imgb, self.size, self.interpolation, antialias=True),\
            F.resize(lbl, self.size, InterpolationMode.NEAREST)
    

class CDSRandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imga, imgb, lbl):
        """
        Args:
            imga (PIL Image): Image to be flipped.
            imgb (PIL Image): Image to be flipped.
            lbl (PIL Image): Label to be flipped.
        """
        if random.random() < self.p:
            return F.hflip(imga), F.hflip(imgb), F.hflip(lbl)
        return imga, imgb, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class CDSRandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imga, imgb, lbl):
        """
        Args:
            imga (PIL Image): Image to be flipped.
            imgb (PIL Image): Image to be flipped.
            lbl (PIL Image): Label to be flipped.
        """
        if random.random() < self.p:
            return F.vflip(imga), F.vflip(imgb), F.vflip(lbl)
        return imga, imgb, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    

class CDSRandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                 interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            print("Scale and ratio should be of kind (min, max)")
            sys.exit()

        self.interpolation = interpolation

        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop."""
        if isinstance(img, torch.Tensor):
            _, height, width = img.shape
        elif isinstance(img, PIL.Image.Image):
            width, height = img.size
        else:
            raise TypeError("Unexpected type {}".format(type(img)))
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, imga, imgb, lbl):
        i, j, h, w = self.get_params(imga, self.scale, self.ratio)
        return F.resized_crop(imga, i, j, h, w, self.size, self.interpolation, antialias=True),\
               F.resized_crop(imgb, i, j, h, w, self.size, self.interpolation, antialias=True),\
               F.resized_crop(lbl, i, j, h, w, self.size, InterpolationMode.NEAREST)


    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str})"
        return format_string


class CDSColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
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
    def get_trans(brightness, contrast, saturation, hue):
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

    def __call__(self, imga, imgb, lbl):
        transform = self.get_trans(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        transed_imga = transform(imga)
        transed_imgb = transform(imgb)
        return transed_imga, transed_imgb, lbl

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class CDSChangeOrder(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, imga, imgb, lbl):
        """
        Args:
            imga (PIL Image): Image to be flipped.
            imgb (PIL Image): Image to be flipped.
            lbl (PIL Image): Label to be flipped.
        """
        if random.random() < self.p:
            return imgb, imga, lbl
        return imga, imgb, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    

class CDMToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, normalize=True, target_type='uint8'):
        self.normalize = normalize
        self.target_type = target_type

    def __call__(self, imga, imgb, lbla, lblb, mask):
        if self.normalize:
            return F.to_tensor(imga),\
                    F.to_tensor(imgb),\
                    torch.from_numpy(np.array(lbla, dtype=self.target_type)),\
                    torch.from_numpy(np.array(lblb, dtype=self.target_type)),\
                    torch.from_numpy(np.array(mask, dtype=self.target_type))
        else:
            return torch.from_numpy(np.array(imga, dtype=np.float32).transpose((2, 0, 1))),\
                   torch.from_numpy(np.array(imgb, dtype=np.float32).transpose((2, 0, 1))),\
                   torch.from_numpy(np.array(lbla, dtype=self.target_type)),\
                   torch.from_numpy(np.array(lblb, dtype=self.target_type)),\
                   torch.from_numpy(np.array(mask, dtype=self.target_type))

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class CDMNormalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, imga, imgb, lbla, lblb, mask):
        return F.normalize(imga, self.mean, self.std, inplace=True),\
               F.normalize(imgb, self.mean, self.std, inplace=True),\
               lbla, lblb, mask

    
class CDMCompose(object):
    """Composes several transforms together.
    Args: transforms (list of ``Transform`` objects): list of transforms to compose."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imga, imgb, lbla, lblb, mask):
        for t in self.transforms:
            imga, imgb, lbla, lblb, mask = t(imga, imgb, lbla, lblb, mask)
        return imga, imgb, lbla, lblb, mask
    
    def append(self, transform):
        self.transforms.append(transform)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    

class CDMResize(object):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imga, imgb, lbla, lblb, mask):
        return F.resize(imga, self.size, self.interpolation, antialias=True),\
            F.resize(imgb, self.size, self.interpolation, antialias=True),\
            F.resize(lbla, self.size, InterpolationMode.NEAREST),\
            F.resize(lblb, self.size, InterpolationMode.NEAREST),\
            F.resize(mask, self.size, InterpolationMode.NEAREST)
    
class CDMRandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imga, imgb, lbla, lblb, mask):
        """
        Args:
            imga (PIL Image): Image to be flipped.
            imgb (PIL Image): Image to be flipped.
            lbl (PIL Image): Label to be flipped.
        """
        if random.random() < self.p:
            return F.hflip(imga), F.hflip(imgb), F.hflip(lbla), F.hflip(lblb), F.hflip(mask)
        return imga, imgb, lbla, lblb, mask

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class CDMRandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imga, imgb, lbla, lblb, mask):
        """
        Args:
            imga (PIL Image): Image to be flipped.
            imgb (PIL Image): Image to be flipped.
            lbl (PIL Image): Label to be flipped.
        """
        if random.random() < self.p:
            return F.vflip(imga), F.vflip(imgb), F.vflip(lbla), F.vflip(lblb), F.vflip(mask)
        return imga, imgb, lbla, lblb, mask

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    

class CDMRandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                 interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            print("Scale and ratio should be of kind (min, max)")
            sys.exit()

        self.interpolation = interpolation

        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop."""
        if isinstance(img, torch.Tensor):
            _, height, width = img.shape
        elif isinstance(img, PIL.Image.Image):
            width, height = img.size
        else:
            raise TypeError("Unexpected type {}".format(type(img)))
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, imga, imgb, lbla, lblb, mask):
        i, j, h, w = self.get_params(imga, self.scale, self.ratio)
        return F.resized_crop(imga, i, j, h, w, self.size, self.interpolation, antialias=True),\
               F.resized_crop(imgb, i, j, h, w, self.size, self.interpolation, antialias=True),\
               F.resized_crop(lbla, i, j, h, w, self.size, InterpolationMode.NEAREST),\
               F.resized_crop(lblb, i, j, h, w, self.size, InterpolationMode.NEAREST),\
               F.resized_crop(mask, i, j, h, w, self.size, InterpolationMode.NEAREST)


    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str})"
        return format_string


class CDMColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
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
    def get_trans(brightness, contrast, saturation, hue):
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

    def __call__(self, imga, imgb, lbla, lblb, mask):
        transform = self.get_trans(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        transed_imga = transform(imga)
        transed_imgb = transform(imgb)
        return transed_imga, transed_imgb, lbla, lblb, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


# -------------------------------------------------------------
#  Augmentations for Object Detection
# -------------------------------------------------------------
class DETCompose(object):
    """Composes several transforms together.
    Args: transforms (list of ``Transform`` objects): list of transforms to compose."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes, lab):
        for t in self.transforms:
            img, bboxes, lab = t(img, bboxes, lab)
        return img, bboxes, lab
    
    def append(self, transform):
        self.transforms.append(transform)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    

class DETToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, normalize=True):
        self.normalize = normalize

    def __call__(self, img, bboxes, lab):
        if self.normalize:
            return F.to_tensor(img), bboxes, lab
        else:
            return torch.from_numpy(np.array(img, dtype=np.float32).transpose((2, 0, 1))), bboxes, lab

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class DETNormalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, bboxes, lab):
        img = F.normalize(img, mean=self.mean, std=self.std, inplace=True)
        return img, bboxes, lab


    


class DETResize(object):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def get_resize_bbox(self, bboxes, lab, h, w, target_size):
        if isinstance(target_size, tuple):
            target_size = target_size[0]

        ratios = target_size / torch.tensor([w, h])
        new_bboxes = bboxes.clone().float()
        new_lab = lab.clone()
        new_bboxes[:, :2] = new_bboxes[:, :2] * ratios
        new_bboxes[:, 2:] = new_bboxes[:, 2:] * ratios
                    
        return new_bboxes, new_lab

    def __call__(self, img, new_bboxes, new_labs):
        new_bboxes, new_labs = self.get_resize_bbox(new_bboxes, new_labs, img.size[0], img.size[1], self.size)
        return F.resize(img, self.size, self.interpolation, antialias=True), new_bboxes, new_labs


    
class DETRandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, bboxes, lab):
        if isinstance(img, torch.Tensor):
            _, H, W = img.shape
        elif isinstance(img, PIL.Image.Image):
            W, H = img.size
        else:
            raise TypeError("Unexpected type {}".format(type(img)))
        
        if random.random() < self.prob:
            img = F.hflip(img)
            bboxes = self.H_bbox_flip(bboxes, (H, W))
            return img, bboxes, lab
        return img, bboxes, lab
    
    def H_bbox_flip(self, bboxes, size):
        H, W = size
        bboxes = bboxes.clone()

        x_max = W - bboxes[:, 0]
        x_min = W - bboxes[:, 2]
        bboxes[:, 0] = x_min
        bboxes[:, 2] = x_max
        return bboxes
    

class DETRandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, bboxes, lab):
        if isinstance(img, torch.Tensor):
            _, H, W = img.shape
        elif isinstance(img, PIL.Image.Image):
            W, H = img.size
        else:
            raise TypeError("Unexpected type {}".format(type(img)))
        
        if random.random() < self.prob:
            img = F.vflip(img)
            bboxes = self.V_bbox_flip(bboxes, (H, W))
            return img, bboxes, lab
        return img, bboxes, lab
    
    def V_bbox_flip(self, bboxes, size):
        H, W = size
        bboxes = bboxes.clone()
        
        y_max = H - bboxes[:, 1]
        y_min = H - bboxes[:, 3]
        bboxes[:, 1] = y_min
        bboxes[:, 3] = y_max
        return bboxes    
    
class DETRandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            print("Scale and ratio should be of kind (min, max)")
            sys.exit()
        self.interpolation = interpolation

        self.scale = scale
        self.ratio = ratio

    def get_crop_bbox(self, bboxes, lab, i, j, h, w):
        # 调整bbox的坐标以匹配裁剪后的图像
        new_bboxes = bboxes.clone()
        new_lab = lab.clone()
        new_bboxes[:, :2] = new_bboxes[:, :2] - torch.tensor([j, i])  # 减去左上角的坐标
        new_bboxes[:, 2:] = new_bboxes[:, 2:] - torch.tensor([j, i])  # 减去左上角的坐标
        # 确保bbox在裁剪后的图像内
        new_bboxes[:, :2] = torch.clamp(new_bboxes[:, :2], min=0)
        new_bboxes[:, 2:] = torch.clamp(new_bboxes[:, 2:], max=torch.tensor([w, h]))
        # 去掉框面积为0的bbox
        mask = (new_bboxes[:, 2] > new_bboxes[:, 0]) & (new_bboxes[:, 3] > new_bboxes[:, 1])
        new_bboxes = new_bboxes[mask]
        new_lab = new_lab[mask]
        return new_bboxes, new_lab
    
    def get_resize_bbox(self, bboxes, lab, h, w, target_size):
        # 调整bbox的坐标以匹配缩放后的图像
        if isinstance(target_size, tuple):
            target_size = target_size[0]

        ratios = target_size / torch.tensor([w, h])
        new_bboxes = bboxes.clone()
        new_lab = lab.clone()
        new_bboxes[:, :2] = new_bboxes[:, :2] * ratios
        new_bboxes[:, 2:] = new_bboxes[:, 2:] * ratios
        # 去掉框面积为0的bbox
        mask = (new_bboxes[:, 2] > new_bboxes[:, 0]) & (new_bboxes[:, 3] > new_bboxes[:, 1])
        new_bboxes = new_bboxes[mask]
        new_lab = new_lab[mask]
        return new_bboxes, new_lab

    def get_params_with_bbox(self, img, scale, ratio, bboxes, labs):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped
            bboxes (list): Bounding boxes of the original image, a tensor of dimensions (n, 4)
            lab (list): Labels of the original image, a tensor of dimensions (n)
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        
        if isinstance(img, torch.Tensor):
            _, height, width = img.shape
        elif isinstance(img, PIL.Image.Image):
            width, height = img.size
        else:
            raise TypeError("Unexpected type {}".format(type(img)))
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        ret = False
        count = 0
        while not ret:
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()

                new_bboxes, new_lab = self.get_crop_bbox(bboxes, labs, i, j, h, w)
                new_bboxes, new_lab = self.get_resize_bbox(new_bboxes, new_lab, h, w, self.size)

                if len(new_bboxes) != 0 and len(new_bboxes) == len(new_lab):
                    ret = True
                else:
                    count += 1
                    if count > 10:
                        return 0, 0, height, width, bboxes, labs

        return i, j, h, w, new_bboxes, new_lab


    def __call__(self, img, bboxes, lab):
        i, j, h, w, new_bboxes, new_lab = self.get_params_with_bbox(img, self.scale, self.ratio, bboxes, lab)
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=True)
        return img, new_bboxes, new_lab
    

class DETColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
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

    def __call__(self, img, bboxes, lab):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img), bboxes, lab

    

# -------------------------------------------------------------
#  Augmentations for Shifted Multi-Crop
# -------------------------------------------------------------

class SmucCompose(object):
    """Composes several transforms together.
    Args: transforms (list of ``Transform`` objects): list of transforms to compose."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_or_crops):
        for t in self.transforms:
            img_or_crops = t(img_or_crops)
        return img_or_crops
    
    def append(self, transform):
        self.transforms.append(transform)



class SmucRandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, views):
        if isinstance(views, tuple):
            view0 = views[0]
            view1 = views[1]
        else:
            AssertionError("views should be tuple")

        if random.random() < self.p:
            view0 = F.hflip(view0)
        if random.random() < self.p:
            view1 = F.hflip(view1)
        return view0, view1




class SmucRandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, views):
        if isinstance(views, tuple):
            view0 = views[0]
            view1 = views[1]
        else:
            AssertionError("views should be tuple")

        if random.random() < self.p:
            view0 = F.vflip(view0)
        if random.random() < self.p:
            view1 = F.vflip(view1)
        return view0, view1


class SmucRandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                 interpolation=InterpolationMode.BILINEAR, shift_per=(-0.25, 0.25)):
        super().__init__()
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            print("Scale and ratio should be of kind (min, max)")
            sys.exit()

        self.interpolation = interpolation

        self.scale = scale
        self.ratio = ratio

        self.shift_per = shift_per
        

    @staticmethod
    def get_params(img, scale, ratio, shift_per):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop."""
        if isinstance(img, torch.Tensor):
            _, height, width = img.shape
        elif isinstance(img, PIL.Image.Image):
            width, height = img.size
        else:
            raise TypeError("Unexpected type {}".format(type(img)))
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            aspect_scale = torch.empty(1).uniform_(scale[0], scale[1]).item()
            target_area = area * aspect_scale
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
            aspect_h_shift_per = torch.empty(1).uniform_(shift_per[0], shift_per[1]).item()
            aspect_w_shift_per = torch.empty(1).uniform_(shift_per[0], shift_per[1]).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            shift_w = int(round(abs(w * aspect_w_shift_per)))
            shift_h = int(round(abs(h * aspect_h_shift_per)))

            if 0 < w + shift_w <= width and 0 < h + shift_h <= height:
                if aspect_w_shift_per < 0:
                    si = torch.randint(0, height - h - shift_h + 1, size=(1,)).item()
                    i = si + shift_h
                else:
                    i = torch.randint(0, height - h - shift_h + 1, size=(1,)).item()
                    si = i + shift_h

                if aspect_h_shift_per < 0:
                    sj = torch.randint(0, width - w - shift_w + 1, size=(1,)).item()
                    j = sj + shift_w
                else:
                    j = torch.randint(0, width - w - shift_w + 1, size=(1,)).item()
                    sj = j + shift_w

                return i, j, si, sj, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        si = i
        sj = j
        return i, j, si, sj, h, w

    def __call__(self, img):
        i, j, si, sj, h, w = self.get_params(img, self.scale, self.ratio, self.shift_per)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=True),\
               F.resized_crop(img, si, sj, h, w, self.size, self.interpolation, antialias=True)

class SmucColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=(0.8, 0.1)):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        
        self.p = p

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

    def __call__(self, views):
        if isinstance(views, tuple):
            view0 = views[0]
            view1 = views[1]
        else:
            AssertionError("views should be tuple")

        if random.random() < self.p[0]:
            transform0 = self.get_params(self.brightness, self.contrast,
                                        self.saturation, self.hue)
            view0 = transform0(view0)

        if random.random() < self.p[1]:
            transform1 = self.get_params(self.brightness, self.contrast,
                                        self.saturation, self.hue)
            view1 = transform1(view1)

        return view0, view1

# -------------------------------------------------------------
#  Augmentations for PerA
# -------------------------------------------------------------
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


class PerARandomPatchMask(object):
    def __init__(self, crop_size, patch_size, sample_ratio_S=0.5, sample_ratio_T=0.5, pixel_mask_ratio=0.8):
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.patch_solution = crop_size // patch_size
        self.sample_ratio_T = sample_ratio_T
        self.sample_ratio_S = sample_ratio_S
        self.pixel_mask_ratio = pixel_mask_ratio
            
    def __call__(self, views):
        if isinstance(views, tuple):
            viewS = views[0]
            viewT = views[1]
        else:
            AssertionError("views should be tuple")

        src_patch_mask = torch.rand(self.patch_solution, self.patch_solution)
        mask_S = src_patch_mask < self.sample_ratio_S
        mask_S = mask_S.repeat_interleave(self.patch_size,dim=0).repeat_interleave(self.patch_size,dim=1)
        mask_T = src_patch_mask > self.sample_ratio_T
        mask_T = mask_T.repeat_interleave(self.patch_size,dim=0).repeat_interleave(self.patch_size,dim=1)
        src_pixel_mask = torch.rand(self.crop_size, self.crop_size)
        pixel_mask = src_pixel_mask > self.pixel_mask_ratio
        mask_T = mask_T | pixel_mask

        viewS = viewS * mask_S
        viewT = viewT * mask_T
        return viewS, viewT

# -------------------------------------------------------------
#  Other testAugmentations
# -------------------------------------------------------------
# class GaussianBlur(transforms.RandomApply):
#     """
#     Apply Gaussian Blur to the PIL image.
#     """

#     def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
#         # NOTE: torchvision is applying 1 - probability to return the original image
#         keep_p = 1 - p
#         transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
#         super().__init__(transforms=[transform], p=keep_p)


class HazeSimulation(object):
    """
    Haze simulation augmentation

    Args:
        p (float): probability of applying HazeSimulation
        t (float, float): minimum / maximum transmission ratio
    """
    def __init__(self, p=0.2, t=(0.3, 0.7)):
        self.p = p
        if isinstance(t, tuple) and len(t) == 2:
            self.tm = t
        elif isinstance(t, float):
            self.tm = (t, 1.)
        else:
            raise TypeError('t should be float or tuple with length 2')

    def __call__(self, img):
        if random.random() > self.p:
            return img
        trans_ratio = random.uniform(*self.tm)
        self.transmition_map = torch.full(img.shape, trans_ratio)
        num_pixels = img.shape[-2] * img.shape[-1]
        num_A = num_pixels // 100
        A = (torch.sort(img.sum(dim=0).reshape(-1), descending=True)[0][:num_A] / 3).mean() 
        img = img * self.transmition_map + A * (1 - self.transmition_map)
        return img
    

class Cutout(object):
    """
    Cutout augmentation
    Args:
        p (float): probability of applying Cutout
        s (float, float): minimum / maximum proportion of cutout area against input image size
        r (float, float): minimum / maximum ratio of cutout area against input image area
    """
    def __init__(self, p=0.2, scale=(0.1, 0.4), ratio=(3./5, 5./3)):
        self.p = p

        if isinstance(scale, tuple) and len(scale) == 2:
            self.scale = scale
        elif isinstance(scale, float):
            self.scale = (0, scale)
        else:
            raise TypeError('scale should be float or tuple with length 2')
        
        if isinstance(ratio, tuple) and len(ratio) == 2:
            self.ratio = ratio
        elif isinstance(ratio, float):
            self.ratio = (1 - ratio, 1 + ratio)
        else:
            raise TypeError('ratio should be float or tuple with length 2')

    def __call__(self, img):
        if random.random() > self.p:
            return img
        scale = random.uniform(*self.scale)
        ratio = random.uniform(*self.ratio)
        imgh, imgw = img.shape[-2:]
        centery = random.randint(0, imgh - 1)
        centerx = random.randint(0, imgw - 1)
        h = int(imgh * scale)
        w = int(imgw * scale * ratio)
        y1 = max(0, centery - h // 2)
        y2 = min(imgh, centery + h // 2)
        x1 = max(0, centerx - w // 2)
        x2 = min(imgw, centerx + w // 2)
        img[:, y1:y2, x1:x2] = 1.
        return img



# -------------------------------------------------------------
#  Augmentations for Pretraining
# -------------------------------------------------------------
    


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

class DinoV2AugmentationSMuC(object):
    def __init__(self, cfg):
        self.num_global_crop_pairs = cfg.AUG.NUM_GLOBAL
        self.num_local_crop_pairs = cfg.AUG.NUM_LOCAL
        
        self.global_pair_trans = SmucCompose([
            transforms.ToTensor(),
            SmucRandomResizedCrop(size=cfg.AUG.GLOBAL_CROP_SIZE, 
                                scale=cfg.AUG.GLOBAL_SCALE, 
                                ratio=cfg.AUG.GLOBAL_RATIO,
                                interpolation=InterpolationMode.BILINEAR,
                                shift_per=cfg.AUG.SHIFT_PER),
            SmucRandomHorizontalFlip(p=0.5),
            SmucRandomVerticalFlip(p=0.5),
            SmucColorJitter(brightness=cfg.AUG.INTENSITY, 
                contrast=cfg.AUG.CONTRAST, 
                saturation=cfg.AUG.SATURATION, 
                hue=cfg.AUG.HUE,
                p=(0.8, 0.1))
        ])
        
        self.local_pair_trans = transforms.Compose([
            transforms.ToTensor(),
            SmucRandomResizedCrop(size=cfg.AUG.LOCAL_CROP_SIZE, 
                                scale=cfg.AUG.LOCAL_SCALE, 
                                ratio=cfg.AUG.LOCAL_RATIO,
                                interpolation=InterpolationMode.BILINEAR,
                                shift_per=cfg.AUG.SHIFT_PER),
            SmucRandomHorizontalFlip(p=0.5),
            SmucRandomVerticalFlip(p=0.5),
            SmucColorJitter(brightness=cfg.AUG.INTENSITY, 
                contrast=cfg.AUG.CONTRAST, 
                saturation=cfg.AUG.SATURATION, 
                hue=cfg.AUG.HUE,
                p=(0.8, 0.1))
        ])


    def __call__(self, image):
        print("DinoV2AugmentationSMuC is deprecated.")
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
