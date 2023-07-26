# Copyright (c) 2020 Sen Wu. All Rights Reserved.


from torchvision import transforms as transforms

from dauphin.image_segmentation.transforms.transform import DauphinTransform


class ToTensor(DauphinTransform):
    def __init__(self, name=None, prob=2.0, level=0):
        super().__init__(name, prob, level)

    def transform(self, img, label=None, **kwargs):
        if label:
            return (
                self._to_tensor(img),
                {key: self._to_tensor(label[key]) for key in label.keys()},
            )
        else:
            return self._to_tensor(img)

    def _to_tensor(self, img):
        if isinstance(img, list):
            return [transforms.ToTensor()(i) for i in img]
        else:
            return transforms.ToTensor()(img)
