import logging
import random

from dauphin.image_segmentation.transforms import ALL_TRANSFORMS

logger = logging.getLogger(__name__)

PRECISION = 3


def parse_sequence(x, type="int"):
    x = x.split(",")
    if len(x) == 1:
        return int(x[0]) if type == "int" else float(x[0])
    return tuple([int(_) if type == "int" else float(_) for _ in x])


def parse_transform(policy):
    parsed_transforms = []

    if policy is None:
        return parsed_transforms

    transforms = policy.split("@")
    for transform in transforms:
        name = transform.split("_")[0]
        settings = transform.split("_")[1:] if len(transform.split("_")) > 1 else []
        if name in ALL_TRANSFORMS:
            prob = random.random()
            level = random.randint(0, 10 ** PRECISION) / float(10 ** PRECISION)
            config = {"prob": prob, "level": level}
            for setting in settings:
                if setting.startswith("PD"):
                    config["padding"] = parse_sequence(setting[2:], type="int")
                elif setting.startswith("PIN"):
                    config["pad_if_needed"] = bool(setting[3:])
                elif setting.startswith("PM"):
                    config["padding_mode"] = str(setting[2:])
                elif setting.startswith("MP"):
                    config["max_pixel"] = int(setting[2:])
                elif setting.startswith("MD"):
                    config["max_degree"] = int(setting[2:])
                elif setting.startswith("P"):
                    config["prob"] = float(setting[1:])
                elif setting.startswith("L"):
                    config["level"] = float(setting[1:])
                elif setting.startswith("S"):
                    config["size"] = parse_sequence(setting[1:], type="int")
                elif setting.startswith("A"):
                    config["alpha"] = float(setting[1:])
                elif setting.startswith("R"):
                    config["same_class_ratio"] = float(setting[1:])
                elif setting.startswith("I"):
                    config["interpolation"] = int(setting[1:])
                elif setting.startswith("B"):
                    config["brightness"] = float(setting[1:])
                elif setting.startswith("C"):
                    config["contrast"] = float(setting[1:])
                elif setting.startswith("T"):
                    config["saturation"] = float(setting[1:])
            parsed_transforms.append(ALL_TRANSFORMS[name](**config))
        else:
            raise ValueError(f"Unrecognized transformation {transforms}")

    return parsed_transforms


class Augmentation(object):
    """Given the augment policy to generate the list of augmentation functions."""

    def __init__(self, augment_policy, num_comp=1):
        self.augment_policy = augment_policy # any, seg-invar
        self.num_comp = num_comp

        # Default transformations applied to all images
        self.default_transforms = []
        logger.info(
            f"Default transformations: {self.default_transforms}"
        )

        self.consistency_default_transforms = []
        logger.info(
            f"Default consistency transformations: "
            f"{self.consistency_default_transforms}"
        )

        # This list of transforms will be used for the labeled data and before the consistency loss
        self.any_transforms = [
            # "AutoContrast_P1.0",
            "Brightness_P1.0",
            # "Contrast_P1.0",
            "ContrastJitter_P1.0",
            "ElasticTransform_P1.0",
            # "Cutout_P1.0",
            "Equalize_P1.0",
            # "Rotate_P1.0",
            "Sharpness_P1.0",
            # "ShearX_P1.0",
            # "ShearY_P1.0",
            # "TranslateX_P1.0",
            # "TranslateY_P1.0",
            "RandomAffine_P1.0",
            # "DoNothing_P1.0",
        ]
        logger.info(
            f"All transformations: {self.any_transforms}"
        )

        # This list of transforms will be used for the consistency loss
        self.consistency_transforms = [
            "AutoContrast_P1.0",
            "Brightness_P1.0",
            "Contrast_P1.0",
            "ContrastJitter_P1.0",
            "Equalize_P1.0",
            "Cutout_P1.0",
            "Sharpness_P1.0",
       ]
        logger.info(
            f"All consistency loss transformations: "
            f"{self.consistency_transforms}"
        )

    def __call__(self):
        if self.augment_policy == "any":
            transform_keys = "@".join(
                random.choices(
                    self.any_transforms, k=self.num_comp
                )
                + self.default_transforms
            )
            transforms = parse_transform(transform_keys)
        
        elif self.augment_policy == "seg-invar":
            transform_keys = "@".join(
                random.choices(
                    self.consistency_transforms,
                    k=self.num_comp,
                )
                + self.consistency_default_transforms
            )
            transforms = parse_transform(transform_keys)
        
        else:
            raise ValueError('Unknown augmentation policy,',self.augment_policy)

        return transforms
