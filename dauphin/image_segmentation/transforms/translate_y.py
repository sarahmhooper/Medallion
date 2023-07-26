# Copyright (c) 2020 Sen Wu. All Rights Reserved.


import random

from PIL import Image

from dauphin.image_segmentation.transforms.transform import DauphinTransform
from dauphin.image_segmentation.transforms.utils import categorize_value


class TranslateY(DauphinTransform):
    def __init__(self, name=None, prob=1.0, level=0, max_degree=50):
        self.max_degree = max_degree
        self.value_range = (0, self.max_degree)
        super().__init__(name, prob, level)

    def transform(self, img, label=None, **kwargs):
        degree = categorize_value(self.level, self.value_range, "float")
        if random.random() > 0.5:
            degree = -degree

        if label:
            return (
                self._translatey(img, degree),
                {key: self._translatey(label[key], degree) for key in label.keys()},
            )
        else:
            return self._translatey(img, degree)

    def _translatey(self, img, degree):
        if isinstance(img, list):
            return [
                i.transform(i.size, Image.AFFINE, (1, 0, 0, 0, 1, degree)) for i in img
            ]
        else:
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, degree))

    def __repr__(self):
        return (
            f"<Transform ({self.name}), prob={self.prob}, level={self.level}, "
            f"max_degree={self.max_degree}>"
        )
