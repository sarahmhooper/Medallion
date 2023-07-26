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


class Medical2DImageDataset(EmmentalDataset):
    """Dataset to load medical 2d image dataset."""

    def __init__(
        self,
        name,
        datapath,
        segpath,
        split="train",
        num_classes=2,
        transform_cls=None,
        k=1,
        image_size_r=224,
        image_size_c=224,
        hist_eq=False,
        csv_fn=None,
    ):
        
        # Init 
        X_dict = {"image_name": [], "image_path": [], "image_slice": []}
        Y_dict = {"seg_path": []}
        self.task_name = name
        self.label_classes = num_classes
        self.image_size_r = image_size_r
        self.image_size_c = image_size_c
        self.hist_eq = hist_eq

        # Identify which labeled images we are using 
        selected_keys=[]

        if split == "train" and csv_fn and segpath: # If train CSV is supplied, use key/indexes specified in CSV for training
            with open(csv_fn, newline="") as f:
                data = list(csv.reader(f))
            for key, idx in data:
                selected_keys.append(split + KEY_DELIMITER + key + KEY_DELIMITER + idx)
                X_dict["image_name"].append(key)
                X_dict["image_path"].append(os.path.join(datapath, split, f"{key}_image.npy"))
                X_dict["image_slice"].append(idx)
                Y_dict["seg_path"].append(os.path.join(segpath, split, f"{key}_seg.npy"))
            
        else: # If evaluating on val/test data, or if no train CSV is supplied, just get all images in dir
             for img_path in glob.glob(
                os.path.join(datapath, split, "*_image.npy")
             ):
                if segpath: # Select all images that also have a seg mask
                    key = img_path.split('/')[-1].split('_image')[0]
                    seg_path = os.path.join(segpath, split, f"{key}_seg.npy")
                    if os.path.exists(seg_path):
                        img = np.load(img_path)
                        for idx in range(img.shape[-1]):
                            selected_keys.append(str(split) + KEY_DELIMITER + key + KEY_DELIMITER + str(idx))
                            X_dict["image_name"].append(key)
                            X_dict["image_path"].append(os.path.join(datapath, split, f"{key}_image.npy"))
                            X_dict["image_slice"].append(idx)
                            Y_dict["seg_path"].append(seg_path)
                        
                else: # Select all images, regardless of if they have a seg mask (used to predict pseudolabel on unlabeled train data)
                    key = img_path.split('/')[-1].split('_image')[0]
                    img = np.load(img_path)
                    for idx in range(img.shape[-1]):
                        selected_keys.append(str(split) + KEY_DELIMITER + key + KEY_DELIMITER + str(idx))
                        X_dict["image_name"].append(key)
                        X_dict["image_path"].append(os.path.join(datapath, split, f"{key}_image.npy"))
                        X_dict["image_slice"].append(idx)
                        Y_dict["seg_path"].append(segpath)

        logger.info(f"Samples: {selected_keys}")
        self.selected_keys = selected_keys

        # Setup augmentation
        self.transform_cls = transform_cls
        self.transforms = None
        self.defaults = [ToTensor()]

        # How many augmented samples to augment for each sample
        self.k = k if k is not None else 1
        
        super().__init__(name, X_dict=X_dict, Y_dict=Y_dict, uid="image_name")

    def gen_transforms(self):
        if self.transform_cls is not None:
            return self.transform_cls()
        else:
            return []

    def __getitem__(self, index):
        """Get item by index.

        Args:
          index(index): The index of the item.

        Returns:
          Tuple[Dict[str, Any], Dict[str, Tensor]]: Tuple of x_dict and y_dict
        """
        
        # Get info for this index 
        split, key, idx = self.selected_keys[index].split(KEY_DELIMITER)
        idx = int(idx)
        
        # Read selected image/seg pairs 
        img_path = self.X_dict['image_path'][index]
        seg_path = self.Y_dict['seg_path'][index]
        
        img = np.load(img_path)[:, :, idx:idx+1].astype("float32")
        img = img - np.min(img)
        img = img / np.max(img)
        if not seg_path: 
            label = np.zeros_like(img)
        elif os.path.exists(seg_path): 
            label = np.load(seg_path)[:, :, idx:idx+1].astype("float32")
        else:
            raise ValueError('Path to specified segmentation mask, '+seg_path+', does not exist.')
            
        img = np.squeeze(custom_preproc_op(img, self.image_size_r, self.image_size_c, hist=self.hist_eq))
        label = np.squeeze(custom_preproc_op(label, self.image_size_r, self.image_size_c, hist=False, order=0))     
        
        # Put img and labels in correct format
        img = torch.from_numpy(img).unsqueeze(0)
        if len(label.shape) == 2:
            one_hot = torch.zeros(label.shape + (self.label_classes,))
            label = one_hot.scatter_(
                2, torch.from_numpy(label.astype("int")).unsqueeze(2), 1.0
            )
        else:
            label = torch.from_numpy(label.astype("float32"))
            
        
        y_dict = {
            f"labels_{cidx}": label[:, :, cidx].unsqueeze(0)
            for cidx in range(self.label_classes)
        }

        img, y_dict = ToPILImage()(img, y_dict)           
        x_dict = {"image_name": self.selected_keys[index], "image": img}
        
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
                for i in range(self.k):
                    self.transforms = self.gen_transforms() + self.defaults

                    transformed_img, transformed_labels = Compose(self.transforms)(
                        feature,
                        y_dict,
                        transforms=self.transforms,
                    )

                    new_x_dict[name].append(transformed_img)

                    new_y_dict["labels"].append(
                        torch.stack(
                            [
                                transformed_labels[f"labels_{cidx}"]
                                for cidx in range(self.label_classes)
                            ],
                            dim=3,
                        )
                    )
            else:
                for i in range(self.k):
                    new_x_dict[name].append(feature)

        return new_x_dict, new_y_dict
