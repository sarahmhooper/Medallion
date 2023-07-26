from dauphin.image_segmentation.transforms.auto_contrast import AutoContrast
from dauphin.image_segmentation.transforms.blur import Blur
from dauphin.image_segmentation.transforms.brightness import Brightness
from dauphin.image_segmentation.transforms.center_crop import CenterCrop
from dauphin.image_segmentation.transforms.color import Color
from dauphin.image_segmentation.transforms.contrast import Contrast
from dauphin.image_segmentation.transforms.contrast_jitter import ContrastJitter
from dauphin.image_segmentation.transforms.cutout import Cutout
from dauphin.image_segmentation.transforms.do_nothing import DoNothing
from dauphin.image_segmentation.transforms.elastic_transform import ElasticTransform
from dauphin.image_segmentation.transforms.equalize import Equalize
from dauphin.image_segmentation.transforms.horizontal_filp import HorizontalFlip
from dauphin.image_segmentation.transforms.identity import Identity
from dauphin.image_segmentation.transforms.invert import Invert
from dauphin.image_segmentation.transforms.mixup import Mixup
from dauphin.image_segmentation.transforms.posterize import Posterize
from dauphin.image_segmentation.transforms.random_affine import RandomAffine
from dauphin.image_segmentation.transforms.random_crop import RandomCrop
from dauphin.image_segmentation.transforms.random_resize_crop import RandomResizedCrop
from dauphin.image_segmentation.transforms.resize import Resize
from dauphin.image_segmentation.transforms.rotate import Rotate
from dauphin.image_segmentation.transforms.sharpness import Sharpness
from dauphin.image_segmentation.transforms.shear_x import ShearX
from dauphin.image_segmentation.transforms.shear_y import ShearY
from dauphin.image_segmentation.transforms.smooth import Smooth
from dauphin.image_segmentation.transforms.solarize import Solarize
from dauphin.image_segmentation.transforms.translate_x import TranslateX
from dauphin.image_segmentation.transforms.translate_y import TranslateY
from dauphin.image_segmentation.transforms.vertical_flip import VerticalFlip

ALL_TRANSFORMS = {
    "AutoContrast": AutoContrast,
    "Blur": Blur,
    "Brightness": Brightness,
    "CenterCrop": CenterCrop,
    "Color": Color,
    "Contrast": Contrast,
    "ContrastJitter": ContrastJitter,
    "Cutout": Cutout,
    "DoNothing": DoNothing,
    "ElasticTransform": ElasticTransform,
    "Equalize": Equalize,
    "HorizontalFlip": HorizontalFlip,
    "Identity": Identity,
    "Invert": Invert,
    "Mixup": Mixup,
    "Posterize": Posterize,
    "RandomCrop": RandomCrop,
    "RandomResizedCrop": RandomResizedCrop,
    "Resize": Resize,
    "Rotate": Rotate,
    "RandomAffine": RandomAffine,
    "Sharpness": Sharpness,
    "ShearX": ShearX,
    "ShearY": ShearY,
    "Smooth": Smooth,
    "Solarize": Solarize,
    "TranslateX": TranslateX,
    "TranslateY": TranslateY,
    "VerticalFlip": VerticalFlip,
}
