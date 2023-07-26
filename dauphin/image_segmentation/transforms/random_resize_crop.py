# Copyright (c) 2020 Sen Wu. All Rights Reserved.


import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import List, Tuple

import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode, functional as F

from dauphin.image_segmentation.transforms.transform import DauphinTransform


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class RandomResizedCrop(DauphinTransform):
    def __init__(
        self,
        size,
        name=None,
        prob=1.0,
        level=0,
        scale=(0.4, 2.0),
        ratio=(0.75, 1.333_333_333_333_333_3),
        interpolation=InterpolationMode.BILINEAR,
    ):
        self.size = _setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        )

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

        super().__init__(name, prob, level)

    @staticmethod
    def get_params(
        img: Tensor, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = F._get_image_size(img)
        area = height * width

        for _ in range(10):

            if (scale[0] < 1 and scale[1] < 1) or scale[0] > 1 and scale[1] > 1:
                random_scale_size = torch.empty(1).uniform_(scale[0], scale[1]).item()
            else:
                random_scale_size = random.choice(
                    [
                        torch.empty(1).uniform_(scale[0], 1).item(),
                        torch.empty(1).uniform_(1, scale[1]).item(),
                    ]
                )

            target_area = area * random_scale_size
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

            elif w > width and h > height:
                i = torch.randint(height - h, 0, size=(1,)).item()
                j = torch.randint(width - w, 0, size=(1,)).item()
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

    def transform(self, img, label=None, **kwargs):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        if label:
            return (
                self._resized_crop(img, i, j, h, w),
                {
                    key: self._resized_crop(label[key], i, j, h, w)
                    for key in label.keys()
                },
            )
        else:
            return self._resized_crop(img, i, j, h, w)

    def _resized_crop(self, img, i, j, h, w):
        if isinstance(img, list):
            return [
                F.resized_crop(im, i, j, h, w, self.size, self.interpolation)
                for im in img
            ]
        else:
            return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        return (
            f"<Transform ({self.name}), prob={self.prob}, level={self.level}, "
            f"size={self.size}, scale={self.scale}, ratio={self.ratio}, "
            f"interpolation={self.interpolation}>"
        )
