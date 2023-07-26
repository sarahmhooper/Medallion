# Copyright (c) 2021 Sen Wu. All Rights Reserved.

import logging
import os

import numpy as np
import yaml

from dauphin.image_segmentation.config import KEY_DELIMITER
from dauphin.image_segmentation.models import MODEL_DIM

logger = logging.getLogger(__name__)


def save_res_to_npy(filepath, model, results):
    logger.info(f"Dumping results to {filepath}")
    if not os.path.exists(filepath): os.makedirs(filepath)
    res_dict = {}

    # 2d model case
    slice_res_dict = {}
    for idx in range(len(results["ids"])):
        split, id, slice_id = results["ids"][idx].split(KEY_DELIMITER)
        if id not in slice_res_dict:
            slice_res_dict[id] = {"slice_ids": [], "preds": []}
        slice_res_dict[id]["slice_ids"].append(int(slice_id))
        slice_res_dict[id]["preds"].append(results["preds"][idx])
    for patient_id in slice_res_dict.keys():
        patient_predicted_seg = np.stack(
            [
                x
                for _, x in sorted(
                    zip(
                        slice_res_dict[patient_id]["slice_ids"],
                        slice_res_dict[patient_id]["preds"],
                    ),
                    key=lambda pair: pair[0],
                )
            ],
            axis=2,
        )
        res_dict[patient_id] = patient_predicted_seg

    for patient_id, patient_predicted_seg in res_dict.items():
        # Save relevant patient info into h5
        save_path = os.path.join(filepath, patient_id+'_seg.npy')
        np.save(save_path, patient_predicted_seg.astype('float32'))
        
        save_path = os.path.join(filepath, patient_id+'_binarized.npy')
        np.save(save_path, np.argmax(patient_predicted_seg,-1).astype('float32'))
            


def load_data_config(filename):
    with open(filename) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config
