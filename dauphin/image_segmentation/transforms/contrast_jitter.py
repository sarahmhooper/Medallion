# Copyright (c) 2020 Sen Wu. All Rights Reserved.


import numpy as np
import torch

from dauphin.image_segmentation.transforms.to_pil_image import ToPILImage
from dauphin.image_segmentation.transforms.to_tensor import ToTensor
from dauphin.image_segmentation.transforms.transform import DauphinTransform


class ContrastJitter(DauphinTransform):
    def __init__(self, name=None, prob=1.0, level=0):
        super().__init__(name, prob, level)

    def transform(self, img, label=None, **kwargs):
        tensor_image = ToTensor()(img)
        transformed_image = ToPILImage()(self.contrast_jitter(tensor_image))

        if label:
            return transformed_image, label
        else:
            return transformed_image

    def contrast_jitter(self, image):
        con = np.random.uniform(0.5, 1)
        b = np.random.uniform(-1, 1)
        g = np.random.uniform(0.2, 1.7)
        if isinstance(image, list):
            res = []
            for img in image:
                img = img.squeeze(0).numpy()
                img = img * con + b
                img -= np.min(img)
                img = img ** g
                res.append(torch.from_numpy(img).unsqueeze(0))
            return res
        else:
            image = image.squeeze(0).numpy()
            image = image * con + b
            image -= np.min(image)
            image = image ** g
            return torch.from_numpy(image).unsqueeze(0)
