# Copyright (c) 2020 Sen Wu. All Rights Reserved.


from PIL import ImageOps

from dauphin.image_segmentation.transforms.transform import DauphinTransform
from dauphin.image_segmentation.transforms.utils import categorize_value


class Posterize(DauphinTransform):

    value_range = (0, 4)

    def __init__(self, name=None, prob=1.0, level=0):
        super().__init__(name, prob, level)

    def transform(self, img, label=None, **kwargs):
        degree = categorize_value(self.level, self.value_range, "int")

        if label:
            return self._posterize(img, degree), label
        else:
            return self._posterize(img, degree)

    def _posterize(self, img, degree):
        if isinstance(img, list):
            return [ImageOps.posterize(i, degree) for i in img]
        else:
            return ImageOps.posterize(img, degree)
