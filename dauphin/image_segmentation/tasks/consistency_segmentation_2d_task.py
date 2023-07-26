# Copyright (c) 2021 Sen Wu. All Rights Reserved.


import logging
from functools import partial

from emmental.task import EmmentalTask
from torch import nn

from dauphin.image_segmentation.models import ALL_MODELS
from dauphin.image_segmentation.modules.consistency_loss import ConsistencyLoss
from dauphin.image_segmentation.modules.prep_consistency_feature import (
    PrepConsistencyFeature,
)

from emmental.task import Action as Act

logger = logging.getLogger(__name__)

# Define loss function
CL = ConsistencyLoss()


def consistency_loss(module_name, intermediate_output_dict, Y):
    aug_img_size = intermediate_output_dict["_input_"]["image"].size()[0]
    loss = CL(
        intermediate_output_dict[module_name][:aug_img_size],
        intermediate_output_dict[module_name][aug_img_size:],
    )
    return loss


def create_task(args):
    if args.model in ["GadgetronResUnet18"]:
        feature_extractor = ALL_MODELS[args.model](
            H=args.image_size_r,
            W=args.image_size_c,
            C=args.data_config["num_classes"],
            no_classifier=True,
        )
    elif args.model in ["UNet"]:
        feature_extractor = ALL_MODELS[args.model](
            n_classes=args.data_config["num_classes"],
            no_classifier=True,
        )
    elif args.model in ["Res50"]:
        feature_extractor = ALL_MODELS[args.model](
            num_classes=args.data_config["num_classes"],
            no_classifier=True,
        )
    else:
        raise ValueError(f"Invalid model {args.model}, need to define model in tasks/consistency_segmentation_2d_task.py")

    task = EmmentalTask(
        name=f"consistency_{args.task}",
        module_pool=nn.ModuleDict(
            {"prep_feature": PrepConsistencyFeature(), "feature": feature_extractor}
        ),
        task_flow=[
            Act(name="prep_feature", module="prep_feature", inputs=[("_input_", "image"), ("_input_", "consistency_image")]),
            Act(name="feature", module="feature", inputs=[("prep_feature", 0)])
        ],
        loss_func=partial(consistency_loss, "feature"),
        output_func=None,
        scorer=None,
        sample_scorer=None,
        require_prob_for_eval=False,
        require_pred_for_eval=False,
        weight=1.0,
    )
    logger.info(f"Built model: {task.module_pool}")

    return task
