# Copyright (c) 2020 Sen Wu. All Rights Reserved.


from PIL import ImageOps

from dauphin.image_segmentation.transforms.transform import DauphinTransform
from dauphin.image_segmentation.transforms.utils import categorize_value


class Solarize(DauphinTransform):

    value_range = (0, 256)

    def __init__(self, name=None, prob=1.0, level=0):
        super().__init__(name, prob, level)

    def transform(self, img, label=None, **kwargs):
        degree = categorize_value(self.level, self.value_range, "float")
        if label:
            return self._solarize(img, degree), label
        else:
            return self._solarize(img, degree)

    def _solarize(self, img, degree):
        if isinstance(img, list):
            return [ImageOps.solarize(i, degree) for i in img]
        else:
            return ImageOps.solarize(img, degree)
