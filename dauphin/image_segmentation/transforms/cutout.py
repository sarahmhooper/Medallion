# Copyright (c) 2020 Sen Wu. All Rights Reserved.


import numpy as np
from PIL import ImageDraw

from dauphin.image_segmentation.transforms.transform import DauphinTransform
from dauphin.image_segmentation.transforms.utils import categorize_value


class Cutout(DauphinTransform):
    def __init__(self, name=None, prob=1.0, level=0, max_pixel=20, color=None):
        self.max_pixel = max_pixel
        self.value_range = (0, self.max_pixel)
        self.color = color
        super().__init__(name, prob, level)

    def transform(self, img, label=None, **kwargs):
        while True:
            # same_label = True
            if isinstance(img, list):
                new_img = [i.copy() for i in img]
            else:
                new_img = img.copy()
            if label:
                if isinstance(img, list):
                    new_label = {
                        key: [i.copy() for i in label[key]] for key in label.keys()
                    }
                else:
                    new_label = {key: label[key].copy() for key in label.keys()}
            degree = categorize_value(self.level, self.value_range, "int")

            if isinstance(img, list):
                width, height = img[0].size
                img_mode = img[0].mode
            else:
                width, height = img.size
                img_mode = img.mode

            x0 = np.random.uniform(width)
            y0 = np.random.uniform(height)

            x0 = int(max(0, x0 - degree / 2.0))
            y0 = int(max(0, y0 - degree / 2.0))
            x1 = min(width, x0 + degree)
            y1 = min(height, y0 + degree)

            xy = (x0, y0, x1, y1)

            if self.color is not None:
                color = self.color
            elif img_mode == "RGB":
                color = (125, 123, 114)
            elif img_mode == "L":
                color = 0
            else:
                raise ValueError(f"Unspported image mode {img_mode}")

            if isinstance(img, list):
                for i in new_img:
                    ImageDraw.Draw(i).rectangle(xy, color)
            else:
                ImageDraw.Draw(new_img).rectangle(xy, color)
            if label:
                for key in new_label.keys():
                    # label[key] = label[key].copy()
                    if isinstance(img, list):
                        for i in new_label[key]:
                            ImageDraw.Draw(i).rectangle(xy, color)
                    else:
                        ImageDraw.Draw(new_label[key]).rectangle(xy, color)
                #     diff = ImageChops.difference(label[key], new_label[key])
                #     # import pdb; pdb.set_trace()
                #     max_diff = np.array(diff).max()
                #     # print("max diff:", max_diff)
                #     if key != "labels_0" and max_diff > 1e-3:
                #         same_label = True
                # if same_label:
                return new_img, new_label
            else:
                return new_img

    def __repr__(self):
        return (
            f"<Transform ({self.name}), prob={self.prob}, level={self.level}, "
            f"max_pixel={self.max_pixel}>"
        )
