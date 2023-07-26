# Copyright (c) 2020 Sen Wu. All Rights Reserved.


import random

from PIL import Image

from dauphin.image_segmentation.transforms.transform import DauphinTransform
from dauphin.image_segmentation.transforms.utils import categorize_value


class ShearY(DauphinTransform):

    value_range = (0.0, 0.3)

    def __init__(self, name=None, prob=1.0, level=0):
        super().__init__(name, prob, level)

    def transform(self, img, label=None, **kwargs):
        degree = categorize_value(self.level, self.value_range, "float")
        if random.random() > 0.5:
            degree = -degree

        if label:
            return (
                self._sheary(img, degree),
                {key: self._sheary(label[key], degree) for key in label.keys()},
            )
        else:
            return self._sheary(img, degree)

    def _sheary(self, img, degree):
        if isinstance(img, list):
            return [
                i.transform(i.size, Image.AFFINE, (1, 0, 0, degree, 1, 0)) for i in img
            ]
        else:
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, degree, 1, 0))
