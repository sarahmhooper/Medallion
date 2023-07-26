# Copyright (c) 2020 Sen Wu. All Rights Reserved.


import random

from dauphin.image_segmentation.transforms.transform import DauphinTransform
from dauphin.image_segmentation.transforms.utils import categorize_value


class Rotate(DauphinTransform):

    value_range = (0, 180)

    def __init__(self, name=None, prob=1.0, level=0):
        super().__init__(name, prob, level)

    def transform(self, img, label=None, **kwargs):
        degree = categorize_value(self.level, self.value_range, "float")
        if random.random() > 0.5:
            degree = -degree

        if label:
            return (
                self._rotate(img, degree),
                {key: self._rotate(label[key], degree) for key in label.keys()},
            )
        else:
            return self._rotate(img, degree)

    def _rotate(self, img, degree):
        if isinstance(img, list):
            return [i.rotate(degree) for i in img]
        else:
            return img.rotate(degree)
