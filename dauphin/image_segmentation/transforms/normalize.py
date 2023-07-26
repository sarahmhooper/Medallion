# Copyright (c) 2020 Sen Wu. All Rights Reserved.


from torchvision import transforms as transforms

from dauphin.image_segmentation.transforms.transform import DauphinTransform


class Normalize(DauphinTransform):
    def __init__(self, mean, std, name=None, prob=1.0, level=0):
        self.mean = mean
        self.std = std
        self.transform_func = transforms.Normalize(mean, std)

        super().__init__(name, prob, level)

    def transform(self, img, label=None, **kwargs):
        if label:
            return self._normalize(img), label
        else:
            return self._normalize(img)

    def _normalize(self, img):
        if isinstance(img, list):
            return [self.transform_func(i) for i in img]
        else:
            return self.transform_func(img)
