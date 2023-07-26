# Copyright (c) 2020 Sen Wu. All Rights Reserved.


import numbers
from collections.abc import Sequence
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, functional as F

from dauphin.image_segmentation.transforms.transform import DauphinTransform

_pil_interpolation_to_str = {
    Image.NEAREST: "PIL.Image.NEAREST",
    Image.BILINEAR: "PIL.Image.BILINEAR",
    Image.BICUBIC: "PIL.Image.BICUBIC",
    Image.LANCZOS: "PIL.Image.LANCZOS",
    Image.HAMMING: "PIL.Image.HAMMING",
    Image.BOX: "PIL.Image.BOX",
}


def _setup_angle(x, name, req_sizes=(2,)):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError(
                "If {} is a single number, it must be positive.".format(name)
            )
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]


def _check_sequence_input(x, name, req_sizes):
    msg = (
        req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    )
    if not isinstance(x, Sequence):
        raise TypeError("{} should be a sequence of length {}.".format(name, msg))
    if len(x) not in req_sizes:
        raise ValueError("{} should be sequence of length {}.".format(name, msg))


class RandomAffine(DauphinTransform):
    def __init__(
        self,
        degrees=80,
        translate=(0.1, 0.1),
        scale=(0.9, 1.3),
        shear=(0, 25),
        interpolation=InterpolationMode.NEAREST,
        fill=0,
        name=None,
        prob=1.0,
        level=0,
    ):
        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))

        if translate is not None:
            _check_sequence_input(translate, "translate", req_sizes=(2,))
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            _check_sequence_input(scale, "scale", req_sizes=(2,))
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            self.shear = _setup_angle(shear, name="shear", req_sizes=(2, 4))
        else:
            self.shear = shear

        self.interpolation = interpolation
        self.fill = fill

        super().__init__(name, prob, level)

    @staticmethod
    def get_params(
        degrees: List[float],
        translate: Optional[List[float]],
        scale_ranges: Optional[List[float]],
        shears: Optional[List[float]],
        img_size: List[int],
    ) -> Tuple[float, Tuple[int, int], float, Tuple[float, float]]:
        """Get parameters for affine transformation

        Returns:
            params to be passed to the affine transformation
        """
        angle = float(
            torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item()
        )
        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = float(
                torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item()
            )
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear

    def transform(self, img, label=None, **kwargs):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """
        if isinstance(img, list):
            img_size = F._get_image_size(img[0])
        else:
            img_size = F._get_image_size(img)

        ret = self.get_params(
            self.degrees, self.translate, self.scale, self.shear, img_size
        )

        if label:
            return (
                self._affine(img, ret),
                {key: self._affine(label[key], ret) for key in label.keys()},
            )
        else:
            return self._affine(img, ret)

    def _affine(self, img, ret):
        if isinstance(img, list):
            return [
                F.affine(i, *ret, interpolation=self.interpolation, fill=self.fill)
                for i in img
            ]
        else:
            return F.affine(img, *ret, interpolation=self.interpolation, fill=self.fill)

    def __repr__(self):
        s = "{name}(degrees={degrees}"
        if self.translate is not None:
            s += ", translate={translate}"
        if self.scale is not None:
            s += ", scale={scale}"
        if self.shear is not None:
            s += ", shear={shear}"
        if self.interpolation > 0:
            s += ", interpolation={interpolation}"
        if self.fill != 0:
            s += ", fill={fill}"
        s += ")"
        d = dict(self.__dict__)
        d["interpolation"] = _pil_interpolation_to_str[d["interpolation"]]
        return s.format(**d)
