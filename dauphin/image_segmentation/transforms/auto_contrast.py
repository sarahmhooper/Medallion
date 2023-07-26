# Copyright (c) 2020 Sen Wu. All Rights Reserved.


from PIL import ImageOps

from dauphin.image_segmentation.transforms.transform import DauphinTransform


class AutoContrast(DauphinTransform):
    def __init__(self, name=None, prob=1.0, level=0):
        super().__init__(name, prob, level)

    def transform(self, img, label=None, **kwargs):
        if label:
            return self._autocontrast(img), label
        else:
            return self._autocontrast(img)

    def _autocontrast(self, img):
        if isinstance(img, list):
            return [ImageOps.autocontrast(i) for i in img]
        else:
            return ImageOps.autocontrast(img)
