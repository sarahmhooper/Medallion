# Copyright (c) 2021 Sen Wu. All Rights Reserved.

import logging
import math
from functools import partial

import numpy as np
import torch
from emmental.modules.identity_module import IdentityModule
from emmental.scorer import Scorer
from emmental.task import EmmentalTask
from emmental.utils.utils import prob_to_pred
from torch import nn
from torch.nn import functional as F

from dauphin.image_segmentation.config import KEY_DELIMITER
from dauphin.image_segmentation.models import ALL_MODELS
from dauphin.image_segmentation.modules.soft_cross_entropy_loss import (
    SoftCrossEntropyLoss,
)

from emmental.task import Action as Act

logger = logging.getLogger(__name__)

# Define loss function
SCE = SoftCrossEntropyLoss(reduction="none")


def sce_loss(
    num_classes, module_name, uncertainty, intermediate_output_dict, Y
):
    batch_size = intermediate_output_dict[module_name].size()[0]
    loss = SCE(
        intermediate_output_dict[module_name]
        .permute(0, 2, 3, 1)
        .reshape(-1, num_classes),
        Y.reshape(-1, num_classes),
    )
    if uncertainty:
        mask = (
            (
                torch.max(Y.reshape(-1, num_classes),1)[0]
                - 0.5
            )
            > uncertainty
        )
        loss = torch.sum((loss * mask.float()).view(batch_size, -1), 1) / torch.clamp(
            torch.sum(mask.view(batch_size, -1), 1), min=1
        )
        return loss
    else:
        loss = torch.mean(loss.view(batch_size, -1), 1)
        return loss


# Define output function
def output_classification(module_name, intermediate_output_dict):
    return F.softmax(
        intermediate_output_dict[module_name].permute(0, 2, 3, 1), dim=-1
    )

# Define score function
def dice(label_mapping, golds, probs, preds, uids, sample_scores, return_sample_scores):
    
    # Calculate average score over sample scores
    if sample_scores is not None:
        
        return_dict = {}
        for class_name, class_idx in label_mapping.items():
            
            all_slice_metrics = {key.split('_%^&^%_')[1].split('_@#$#@_')[1]:[] for key in sample_scores.keys()}
            for key, sl_pix_metrics in sample_scores.items():
                dice_class_name, uid_key, _ = key.split('_%^&^%_')
                if class_name == dice_class_name:
                    all_slice_metrics[uid_key.split('_@#$#@_')[1]] += [(sl_pix_metrics[0][0],sl_pix_metrics[0][1])]
            
            volume_avg_dice = []
            for pid, pix_metrics in all_slice_metrics.items():
                volume_pix_ov = np.sum([ov for ov, pos in pix_metrics])
                volume_pix_pos = np.sum([pos for ov, pos in pix_metrics])
                if volume_pix_pos==0: volume_avg_dice += [1.0]
                else: volume_avg_dice += [2*volume_pix_ov/volume_pix_pos]
            return_dict[f"{class_name}_DICE"] = np.mean(volume_avg_dice)
            
        return return_dict
    
    
    # Convert probabilistic label to hard label        
    preds = prob_to_pred(probs)
    golds = prob_to_pred(golds)
    batch_size = golds.shape[0]

    preds = preds.reshape(batch_size, -1)
    golds = golds.reshape(batch_size, -1)

    gold_dict = {uid:[] for uid in uids}
    pred_dict = {uid:[] for uid in uids}

    for batch_idx in range(batch_size):
        uid_key = uids[batch_idx]
        pred_dict[uid_key].append(preds[batch_idx])
        gold_dict[uid_key].append(golds[batch_idx])

    for uid_key in pred_dict.keys():
        pred_dict[uid_key] = np.array(pred_dict[uid_key])
        gold_dict[uid_key] = np.array(gold_dict[uid_key])

        
    # Calculate image-level dice scores using probs, preds, and golds.
    return_dict = {}
    for class_name, class_idx in label_mapping.items():
        uid_list = []
        pix_overlap = []
        pix_pos = []
        for uid_key in pred_dict.keys():
            pix_overlap += [np.sum(
                np.logical_and(pred_dict[uid_key] == class_idx, gold_dict[uid_key] == class_idx)
            )]
            pix_pos += [
                np.sum(pred_dict[uid_key] == class_idx)
                + np.sum(gold_dict[uid_key] == class_idx
            )]
            uid_list += [uid_key]

        if return_sample_scores:
            for uid_key, sl_pix_ov, sl_pix_pos in zip(uid_list, pix_overlap, pix_pos):
                return_dict[f"{class_name}_%^&^%_{uid_key}_%^&^%_DICE"] = [(sl_pix_ov, sl_pix_pos)]
        
        else: 
            all_slice_metrics = {uid_key.split(KEY_DELIMITER)[1]:[] for uid_key in uid_list}
            for uid_key, sl_pix_ov, sl_pix_pos in zip(uid_list, pix_overlap, pix_pos):
                all_slice_metrics[uid_key.split(KEY_DELIMITER)[1]] += [(sl_pix_ov,sl_pix_pos)]
            
            volume_avg_dice = []
            for pid, pix_metrics in all_slice_metrics.items():
                volume_pix_ov = np.sum([ov for ov, pos in pix_metrics])
                volume_pix_pos = np.sum([pos for ov, pos in pix_metrics])
                if volume_pix_pos==0: volume_avg_dice += [1.0]
                else: volume_avg_dice += [2*volume_pix_ov/volume_pix_pos]
            return_dict[f"{class_name}_DICE"] = np.mean(volume_avg_dice)
            
    return return_dict


def create_task(args):
    if args.consistency_datapath is None: no_classifier = False
    else: no_classifier = True
    if args.model in ["GadgetronResUnet18"]:
        feature_extractor = ALL_MODELS[args.model](
            H=args.image_size_r,
            W=args.image_size_c,
            C=args.data_config["num_classes"],
            no_classifier=no_classifier,
        )
        pred_head = (
            nn.Conv2d(feature_extractor.output_dim, args.data_config["num_classes"], 1)
            if no_classifier
            else IdentityModule()
        )

        if isinstance(pred_head, nn.Conv2d):
            n = (
                pred_head.kernel_size[0]
                * pred_head.kernel_size[1]
                * pred_head.out_channels
            )
            pred_head.weight.data.normal_(0, math.sqrt(2.0 / n))
    elif args.model in ["UNet"]:
        feature_extractor = ALL_MODELS[args.model](
            n_classes=args.data_config["num_classes"],
            no_classifier=no_classifier,
        )
        pred_head = (
            nn.Conv2d(feature_extractor.output_dim, args.data_config["num_classes"], 1)
            if no_classifier
            else IdentityModule()
        )

        if isinstance(pred_head, nn.Conv2d):
            n = (
                pred_head.kernel_size[0]
                * pred_head.kernel_size[1]
                * pred_head.out_channels
            )
            pred_head.weight.data.normal_(0, math.sqrt(2.0 / n))
    elif args.model in ["Res50"]:
        feature_extractor = ALL_MODELS[args.model](
            num_classes=args.data_config["num_classes"],
            no_classifier=no_classifier,
        )
        pred_head = (
            ALL_MODELS["FCNHead"](2048, args.data_config["num_classes"])
            if no_classifier
            else IdentityModule()
        )

    else:
        raise ValueError(f"Invalid model {args.model}, need to define model in tasks/segmentation_2d_task.py")
    
    task = EmmentalTask(
        name=args.task,
        module_pool=nn.ModuleDict(
            {"feature": feature_extractor, f"{args.task}_pred_head": pred_head}
        ),
        task_flow=[
            Act(name="feature", module="feature", inputs=[("_input_", "image")]),
            Act(name=f"{args.task}_pred_head", module=f"{args.task}_pred_head", inputs=[("feature", 0)])
        ],
        loss_func=partial(
            sce_loss,
            args.data_config["num_classes"],
            f"{args.task}_pred_head",
            args.uncertainty,
        ),
        output_func=partial(output_classification, f"{args.task}_pred_head"),
        scorer=Scorer(
            customize_metric_funcs={
                "DICE": partial(dice, args.data_config["label_mapping"])
            }
        ),
        sample_scorer=Scorer(
            customize_metric_funcs={
                "DICE": partial(dice, args.data_config["label_mapping"])
            }
        ),
        require_prob_for_eval=True,
        require_pred_for_eval=False,
        weight=1.0,
    )
    logger.info(f"Built model: {task.module_pool}")

    return task
