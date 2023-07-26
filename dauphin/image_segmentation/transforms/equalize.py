# Copyright (c) 2020 Sen Wu. All Rights Reserved.


from PIL import ImageOps

from dauphin.image_segmentation.transforms.transform import DauphinTransform


class Equalize(DauphinTransform):
    def __init__(self, name=None, prob=1.0, level=0):
        super().__init__(name, prob, level)

    def transform(self, img, label=None, **kwargs):
        if label:
            return self._equalize(img), label
        else:
            return self._equalize(img)

    def _equalize(self, img):
        if isinstance(img, list):
            return [ImageOps.equalize(i) for i in img]
        else:
            return ImageOps.equalize(img)
