# Copyright (c) 2020 Sen Wu. All Rights Reserved.


import numpy as np
from PIL import Image

from dauphin.image_segmentation.transforms.compose import Compose
from dauphin.image_segmentation.transforms.transform import DauphinTransform


# TODO: Doesn't support 3d now
class Mixup(DauphinTransform):
    def __init__(self, name=None, prob=1.0, level=0, alpha=1.0):
        self.alpha = alpha

        super().__init__(name, prob, level)

    def transform(self, img, label=None, **kwargs):
        # import pdb; pdb.set_trace()
        X_dict = kwargs["X_dict"]
        Y_dict = kwargs["Y_dict"]
        transforms = kwargs["transforms"]

        if self.alpha > 0.0:
            mix_ratio = np.random.beta(self.alpha, self.alpha)
        else:
            mix_ratio = 1.0

        idx = np.random.randint(len(X_dict["image"]))

        # Calc all transforms before mixup
        prev_transforms = transforms[: kwargs["idx"]]

        other_img = X_dict["image"][idx]

        # Apply all prev mixup transforms
        if label:
            other_label = {name: Y_dict[name][idx].copy() for name in Y_dict.keys()}
            cand_img, cand_label = Compose(prev_transforms)(
                other_img, other_label, **kwargs
            )
        else:
            cand_img = Compose(prev_transforms)(other_img, **kwargs)

        mixup_img = Image.blend(img, cand_img, 1.0 - mix_ratio)
        if label:
            mixup_label = {
                key: Image.blend(label[key], cand_label[key], 1.0 - mix_ratio)
                for key in label.keys()
            }
            return mixup_img, mixup_label
        else:
            return mixup_img

    def __repr__(self):
        return (
            f"<Transform ({self.name}), prob={self.prob}, level={self.level}, "
            f"alpha={self.alpha}"
        )
