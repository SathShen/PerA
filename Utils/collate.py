# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import math
import numpy as np


class MaskingGenerator:
    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask


def dinov2smuc_collate(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # dtype = torch.half  # TODO: Remove
    n_global_crops = len(samples_list[0][0]["global_crops"])  # sample->sample0->S_crop_pairs->global_crops
    n_local_crops = len(samples_list[0][0]["local_crops"])

    S_collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])
    T_collated_global_crops = torch.stack([s[1]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])
    S_collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])
    T_collated_local_crops = torch.stack([s[1]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

    B = len(S_collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        "S_collated_global_crops": S_collated_global_crops.to(dtype),
        "T_collated_global_crops": T_collated_global_crops.to(dtype),
        "S_collated_local_crops": S_collated_local_crops.to(dtype),
        "T_collated_local_crops": T_collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }



def dinov2_collate(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # dtype = torch.half  # TODO: Remove
    n_global_crops = len(samples_list[0]["global_crops"])
    n_local_crops = len(samples_list[0]["local_crops"])

    collated_global_crops = torch.stack([s["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

    collated_local_crops = torch.stack([s["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }


def pera_collate(samples_list):
    n_global_crops = len(samples_list[0][0]["global_crops"])  # sample->sample0->S_crop_pairs->global_crops
    n_local_crops = len(samples_list[0][0]["local_crops"])

    # list to tensor can cause memory leak
    S_collated_global_crops = np.array([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])
    T_collated_global_crops = np.array([s[1]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])
    S_collated_local_crops = np.array([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])
    T_collated_local_crops = np.array([s[1]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

    S_collated_global_crops = torch.from_numpy(S_collated_global_crops)
    T_collated_global_crops = torch.from_numpy(T_collated_global_crops)
    S_collated_local_crops = torch.from_numpy(S_collated_local_crops)
    T_collated_local_crops = torch.from_numpy(T_collated_local_crops)

    return {
        "S_collated_global_crops": S_collated_global_crops,
        "T_collated_global_crops": T_collated_global_crops,
        "S_collated_local_crops": S_collated_local_crops,
        "T_collated_local_crops": T_collated_local_crops,
    }





# 处理batch中num_objects不一致的情况
def det_collate_fn(batch):
    imgs = []
    targets = []
    for img, label, bbox in batch:
        imgs.append(img)
        target = {}
        target['labels'] = label.to(dtype=torch.int64)
        target['boxes'] = bbox.to(dtype=torch.float32)
        targets.append(target)

    image_dict = {}
    image_dict['images'] = torch.stack(imgs, dim=0)
    image_dict['targets'] = targets
    return image_dict


def off_det_collate_fn(batch):
    imgs = []
    targets = []
    for img, label, bbox, img_name in batch:
        imgs.append(img)
        target = {}
        target['labels'] = label.to(dtype=torch.int64)
        target['boxes'] = bbox.to(dtype=torch.float32)
        targets.append(target)

    image_dict = {}
    image_dict['images'] = imgs
    image_dict['targets'] = targets
    return image_dict