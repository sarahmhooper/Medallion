# Copyright (c) 2020 Sen Wu. All Rights Reserved.


from PIL import Image

from dauphin.image_segmentation.transforms.transform import DauphinTransform


class HorizontalFlip(DauphinTransform):
    def __init__(self, name=None, prob=1.0, level=0):
        super().__init__(name, prob, level)

    def transform(self, img, label=None, **kwargs):
        if label:
            return (
                self._transpose(img),
                {key: self._transpose(label[key]) for key in label.keys()},
            )
        else:
            return self._transpose(img)

    def _transpose(self, img):
        if isinstance(img, list):
            return [i.transpose(Image.FLIP_LEFT_RIGHT) for i in img]
        else:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
