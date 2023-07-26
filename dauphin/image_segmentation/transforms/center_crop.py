# Copyright (c) 2020 Sen Wu. All Rights Reserved.


from torchvision import transforms as transforms

from dauphin.image_segmentation.transforms.transform import DauphinTransform


class CenterCrop(DauphinTransform):
    def __init__(self, size, name=None, prob=1.0, level=0):
        self.size = size
        self.transform_func = transforms.CenterCrop(self.size)
        super().__init__(name, prob, level)

    def transform(self, img, label=None, **kwargs):
        if label:
            return (
                self._center_crop(img),
                {key: self._center_crop(label[key]) for key in label.keys()},
            )
        else:
            return self._center_crop(img)

    def _center_crop(self, img):
        if isinstance(img, list):
            return [self.transform_func(i) for i in img]
        else:
            return self.transform_func(img)

    def __repr__(self):
        return (
            f"<Transform ({self.name}), prob={self.prob}, level={self.level}, "
            f"size={self.size}>"
        )
