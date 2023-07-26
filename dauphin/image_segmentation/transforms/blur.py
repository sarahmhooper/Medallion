# Copyright (c) 2020 Sen Wu. All Rights Reserved.


from PIL import ImageFilter

from dauphin.image_segmentation.transforms.transform import DauphinTransform


class Blur(DauphinTransform):
    def __init__(self, name=None, prob=1.0, level=0):
        super().__init__(name, prob, level)

    def transform(self, img, label=None, **kwargs):
        if label:
            return self._filter(img), label
        else:
            return self._filter(img)

    def _filter(self, img):
        if isinstance(img, list):
            return [i.filter(ImageFilter.BLUR) for i in img]
        else:
            return img.filter(ImageFilter.BLUR)
