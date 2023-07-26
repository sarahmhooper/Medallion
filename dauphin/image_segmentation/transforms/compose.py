# Copyright (c) 2020 Sen Wu. All Rights Reserved.


class Compose(object):
    """Composes several transforms together.

    Originally from:
      https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#Compose

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label=None, **kwargs):
        for idx, t in enumerate(self.transforms):
            kwargs["idx"] = idx
            if label:
                img, label = t(img, label, **kwargs)
            else:
                img = t(img, None, **kwargs)
        if label:
            return img, label
        else:
            return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"

        return format_string
