# Copyright (c) 2020 Sen Wu. All Rights Reserved.


from dauphin.image_segmentation.transforms.transform import DauphinTransform


class Identity(DauphinTransform):
    def __init__(self, name=None, prob=1.0, level=0):
        super().__init__(name, prob, level)

    def transform(self, img, label=None, **kwargs):
        if label:
            return img, label
        else:
            return img
