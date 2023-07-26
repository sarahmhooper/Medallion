import logging
import os
import pickle
import sys

import emmental
import numpy as np
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.utils.parse_args import parse_args_to_config
from torch.backends import cudnn

from dauphin.image_segmentation.config import KEY_DELIMITER
from dauphin.image_segmentation.augment_policy import Augmentation
from dauphin.image_segmentation.consistency_scheduler import ConAugScheduler
from dauphin.image_segmentation.data import get_dataloaders
from dauphin.image_segmentation.scheduler import AugScheduler
from dauphin.image_segmentation.task import create_task
from dauphin.image_segmentation.utils import load_data_config, save_res_to_npy
from dauphin.utils import write_to_file, write_to_json_file  # get_sha,

logger = logging.getLogger(__name__)


def main(args):
    
    # Initialize 
    args.data_config = load_data_config(os.path.join(args.datapath, "config.yaml"))

    config = parse_args_to_config(args)
    emmental.init(
        log_dir=config["meta_config"]["log_path"],
        use_exact_log_path=config["meta_config"]["use_exact_log_path"],
        config=config,
    )

    # Set cudnn benchmark
    # torch.set_deterministic(True)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # Log configuration into files
    cmd_msg = " ".join(sys.argv)
    logger.info(f"COMMAND: {cmd_msg}")
    write_to_file(f"{emmental.Meta.log_path}/cmd.txt", cmd_msg)

    logger.info(f"Config: {emmental.Meta.config}")
    write_to_file(f"{emmental.Meta.log_path}/config.txt", emmental.Meta.config)

    # Create dataloaders
    dataloaders = get_dataloaders(args)

    # Assign transforms to dataloaders
    for idx in range(len(dataloaders)):
        if dataloaders[idx].split in args.train_split:
            if "consistency" not in dataloaders[idx].dataset.name:
                dataloaders[idx].dataset.transform_cls = Augmentation(
                    augment_policy="any",
                    num_comp=args.num_comp,
                )
            else:
                dataloaders[idx].dataset.transform_cls = Augmentation(
                    augment_policy="any",
                    num_comp=args.num_comp,
                )
                dataloaders[idx].dataset.consistency_transform_cls = Augmentation(
                    augment_policy="seg-invar",
                    num_comp=args.num_comp,
                )

    if args.scheduler == "augment":
        config["learner_config"]["task_scheduler_config"][
            "task_scheduler"
        ] = AugScheduler(
            augment_k=args.augment_k,
            enlarge=args.augment_enlarge,
            fillup=args.fillup,
            trim=args.trim,
        )
    else:
        config["learner_config"]["task_scheduler_config"][
            "task_scheduler"
        ] = ConAugScheduler(
            augment_k=args.augment_k,
            enlarge=args.augment_enlarge,
            n_sup_batches_per_step=args.n_sup_batches_per_step,
            n_unsup_batches_per_step=args.n_unsup_batches_per_step,
        )

    emmental.Meta.config["learner_config"]["task_scheduler_config"][
        "task_scheduler"
    ] = config["learner_config"]["task_scheduler_config"]["task_scheduler"]

    # Create tasks
    model = EmmentalModel(name=f"{args.task}")
    model.add_tasks(create_task(args))

    # Load saved model
    if config["model_config"]["model_path"] is not None:
        model.load(config["model_config"]["model_path"])

    if args.train:
        emmental_learner = EmmentalLearner()
        emmental_learner.learn(model, dataloaders)

    # Remove all extra augmentation policy
    for idx in range(len(dataloaders)):
        dataloaders[idx].dataset.transform_cls = None

    if args.record_scores:
        scores = model.score(
            [dl for dl in dataloaders if "consistency" not in dl.dataset.name]
        )

        # Save metrics and models
        logger.info(f"Metrics: {scores}")
        scores["log_path"] = emmental.Meta.log_path
        write_to_json_file(f"{emmental.Meta.log_path}/metrics.txt", scores)
        if args.train: model.save(f"{emmental.Meta.log_path}/last_model.pth")

    # predict and save preds on val data
    if args.val_data_available:
        res_filename = os.path.join(emmental.Meta.log_path, "final_preds", "val")
        dl = [_ for _ in dataloaders if _.split == "val"][0]
        
        model.save_preds_to_numpy(dl,res_filename,KEY_DELIMITER)
        
    # predict and save preds on test data
    if args.test_data_available:
        res_filename = os.path.join(emmental.Meta.log_path, "final_preds", "test")
        dl = [_ for _ in dataloaders if _.split == "test"][0]
        
        model.save_preds_to_numpy(dl,res_filename,KEY_DELIMITER)
        
    # predict and save preds on train data
    if args.predict_on_train and args.train_data_available:

        # predict on ALL train data, not just data specified in the csv
        args.csv_fn = None
        args.augment_k = 1
        args.train_segpath = None
        dataloaders = get_dataloaders(args)

        res_filename = os.path.join(emmental.Meta.log_path, "final_preds", "train")
        dl = [
            _
            for _ in dataloaders
            if _.split == "train" and "consistency" not in _.dataset.name
        ][0]
        
        model.save_preds_to_numpy(dl,res_filename,KEY_DELIMITER)
        