# Copyright (c) 2020 Sen Wu. All Rights Reserved.


import random

import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from dauphin.image_segmentation.transforms.to_pil_image import ToPILImage
from dauphin.image_segmentation.transforms.to_tensor import ToTensor
from dauphin.image_segmentation.transforms.transform import DauphinTransform
from dauphin.image_segmentation.transforms.utils import categorize_value


class ElasticTransform(DauphinTransform):

    alpha_range = (1000, 1400)
    sigma_range = (30, 45)

    def __init__(self, alpha=None, sigma=None, name=None, prob=1.0, level=0):
        self.alpha = alpha
        self.sigma = sigma
        super().__init__(name, prob, level)

    def transform(self, img, label=None, **kwargs):
        alpha = (
            self.alpha
            if self.alpha is not None
            else categorize_value(random.random(), self.alpha_range, "float")
        )
        sigma = (
            self.sigma
            if self.sigma is not None
            else categorize_value(random.random(), self.sigma_range, "float")
        )
        if label:
            tensor_image, tensor_label = ToTensor()(img, label)
            transformed_image, transformed_label = self.elastic_transform(
                tensor_image, label=tensor_label, alpha=alpha, sigma=sigma
            )
            new_image, new_label = ToPILImage()(transformed_image, transformed_label)
            return new_image, new_label
        else:
            tensor_image = ToTensor()(img)
            transformed_image = self.elastic_transform(
                tensor_image, alpha=alpha, sigma=sigma
            )
            new_image = ToPILImage()(transformed_image)
            return new_image

    def __repr__(self):
        return (
            f"<Transform ({self.name}), prob={self.prob}, "
            f"alpha={self.alpha}, sigma={self.sigma}>"
        )

    def elastic_transform(
        self, image, label=None, alpha=2000, sigma=30, random_state=None
    ):
        """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        """
        if isinstance(image, list):
            image = [np.squeeze(i.numpy()) for i in image]
            shape = image[0].shape
        else:
            image = np.squeeze(image.numpy())
            shape = image.shape
        # assert len(image.shape) == 2, image.shape

        if random_state is None:
            random_state = np.random.RandomState(None)

        dx = (
            gaussian_filter(
                (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
            )
            * alpha
        )
        dy = (
            gaussian_filter(
                (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
            )
            * alpha
        )

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

        if isinstance(image, list):
            new_image = [
                torch.from_numpy(
                    map_coordinates(i, indices, order=1).reshape(shape)
                ).unsqueeze(0)
                for i in image
            ]
        else:
            new_image = torch.from_numpy(
                map_coordinates(image, indices, order=1).reshape(shape)
            ).unsqueeze(0)

        if label:
            if isinstance(image, list):
                for key in label.keys():
                    label[key] = [np.squeeze(i.numpy()) for i in label[key]]
                    label[key] = [
                        torch.from_numpy(
                            map_coordinates(i, indices, order=0).reshape(shape)
                        ).unsqueeze(0)
                        for i in label[key]
                    ]
            else:
                for key in label.keys():
                    label[key] = np.squeeze(label[key].numpy())
                    label[key] = torch.from_numpy(
                        map_coordinates(label[key], indices, order=0).reshape(shape)
                    ).unsqueeze(0)
            return new_image, label
        else:
            return new_image
