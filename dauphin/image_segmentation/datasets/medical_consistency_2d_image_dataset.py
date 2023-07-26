# Copyright (c) 2020 Sen Wu. All Rights Reserved.


import logging
import os
import glob

import numpy as np
import torch
import csv
from emmental.data import EmmentalDataset

from dauphin.image_segmentation.config import KEY_DELIMITER
from dauphin.image_segmentation.datasets.utils import custom_preproc_op
from dauphin.image_segmentation.transforms.center_crop import CenterCrop
from dauphin.image_segmentation.transforms.compose import Compose
from dauphin.image_segmentation.transforms.to_pil_image import ToPILImage
from dauphin.image_segmentation.transforms.to_tensor import ToTensor

logger = logging.getLogger(__name__)


class MedicalConsistency2DImageDataset(EmmentalDataset):
    """Dataset to load medical consistency loss 2d image dataset."""

    def __init__(
        self,
        name,
        datapath,
        split="train",
        transform_cls=None,
        consistency_transform_cls=None,
        k=1,
        image_size_r=224,
        image_size_c=224,
        hist_eq=False,
        csv_fn=None,
    ):
        
        # Init 
        X_dict = {"image_name": [], "image_path": [], "image_slice": []}
        Y_dict = {"seg_path": []} # Dummy var
        self.task_name = name
        self.image_size_r = image_size_r
        self.image_size_c = image_size_c
        self.hist_eq = hist_eq

        # Identify which images we are using 
        selected_keys = []
        if split == "train" and csv_fn: # Use key/indexes specified in CSV
            with open(csv_fn, newline="") as f:
                data = list(csv.reader(f))
            for key, idx in data:
                selected_keys.append(split + KEY_DELIMITER + key + KEY_DELIMITER + idx)
                X_dict["image_name"].append(key)
                X_dict["image_path"].append(os.path.join(datapath, split, f"{key}_image.npy"))
                X_dict["image_slice"].append(idx)
                Y_dict["seg_path"].append(None)

        else: # Use all available key/index pairs in dir
            for img_path in glob.glob(
                os.path.join(datapath, split, "*_image.npy")
             ):
                key = img_path.split('/')[-1].split('_image')[0]
                img = np.load(img_path)
                for idx in range(img.shape[-1]):
                    selected_keys.append(str(split) + KEY_DELIMITER + key + KEY_DELIMITER + str(idx))
                    X_dict["image_name"].append(key)
                    X_dict["image_path"].append(os.path.join(datapath, split, f"{key}_image.npy"))
                    X_dict["image_slice"].append(idx)
                    Y_dict["seg_path"].append(None)
                        
        logger.info(f"Random samples: {selected_keys}")
        self.selected_keys = selected_keys
        
        # Setup augmentation
        self.transform_cls = transform_cls
        self.consistency_transform_cls = consistency_transform_cls
        self.transforms = None
        self.consistency_transforms = None
        self.default_transforms = [ToTensor()]

        # How many augmented samples to augment for each sample
        self.k = k if k is not None else 1

        super().__init__(name, X_dict=X_dict, Y_dict=Y_dict, uid="image_name")

    def gen_transforms(self):
        if self.transform_cls is not None:
            return self.transform_cls()
        else:
            return []

    def get_consistency_transforms(self):
        if self.consistency_transform_cls is not None:
            return self.consistency_transform_cls()
        else:
            return []

    def __getitem__(self, index):
        """Get item by index.

        Args:
          index(index): The index of the item.

        Returns:
          Dict[str, Any]: x_dict
        """
        
        # Get info for this index 
        split, key, idx = self.selected_keys[index].split(KEY_DELIMITER)
        idx = int(idx)
        
        # Read selected image from stored data 
        img_path = self.X_dict["image_path"][index]
        img = np.load(img_path)[:, :, idx:idx+1].astype("float32")
        img = img - np.min(img)
        img = img / np.max(img)
        
        # Put image in correct format
        img = np.squeeze(
                    custom_preproc_op(
                        img,
                        self.image_size_r,
                        self.image_size_c,
                        hist=self.hist_eq,
                    )
                )
        
        
        img = ToPILImage()(
            torch.from_numpy(img).unsqueeze(0)
        )
        
        x_dict = {"image_name": self.selected_keys[index], "image": img}
        y_dict = {"labels": torch.from_numpy(np.array(range(1)))} # Create a dummy variable containing empty labels

        # Apply k transforms
        new_x_dict = {}
        new_y_dict = {"labels": []}

        for name, feature in x_dict.items():
            if name not in new_x_dict:
                new_x_dict[name] = []
            if name == self.uid:
                if self.k > 1:
                    for i in range(self.k):
                        new_x_dict[name].append(f"{feature}{KEY_DELIMITER}{i}")
                else:
                    new_x_dict[name].append(feature)
            elif name == "image":
                if "consistency_image" not in new_x_dict:
                    new_x_dict["consistency_image"] = []

                self.transforms = (
                    [] if self.transform_cls is None else self.gen_transforms()
                )
                transformed_img = Compose(self.transforms)(
                    feature,
                    X_dict=self.X_dict,
                    Y_dict=self.Y_dict,
                    transforms=self.transforms,
                )
                for i in range(self.k):
                    self.consistency_transforms = (
                        self.get_consistency_transforms()
                    )
                    consistency_img = Compose(
                        self.consistency_transforms + self.default_transforms
                    )(transformed_img)
                    new_x_dict[f"consistency_{name}"].append(consistency_img)
                    new_y_dict["labels"].append(y_dict["labels"])

                new_x_dict[name].append(
                    Compose(self.default_transforms)(transformed_img)
                )
            else:
                for i in range(self.k):
                    new_x_dict[name].append(feature)

        return new_x_dict, new_y_dict
