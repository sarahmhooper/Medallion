# Copyright (c) 2021 Sen Wu. All Rights Reserved.


import logging

from emmental.data import EmmentalDataLoader

from dauphin.image_segmentation.models import MODEL_DIM
from dauphin.image_segmentation.datasets import ALL_CONSISTENCY_DATASETS, ALL_DATASETS

logger = logging.getLogger(__name__)


def get_dataloaders(args):
    dataloaders = []
    datasets = {}

    for split in ["train", "val", "test"]:
        if split == "train" and args.train_data_available:
            datasets[split] = ALL_DATASETS[MODEL_DIM[args.model]](
                args.task,
                args.datapath,
                args.train_segpath,
                split,
                num_classes=args.data_config["num_classes"],
                k=args.augment_k,
                image_size_r=args.image_size_r,
                image_size_c=args.image_size_c,
                hist_eq=args.hist_eq,
                csv_fn=args.csv_fn,
            )
        elif split == "val" and args.val_data_available:
            datasets[split] = ALL_DATASETS[MODEL_DIM[args.model]](
                args.task,
                args.datapath,
                args.eval_segpath,
                split,
                num_classes=args.data_config["num_classes"],
                image_size_r=args.image_size_r,
                image_size_c=args.image_size_c,
                hist_eq=args.hist_eq,
            )
        elif split == "test" and args.test_data_available:
            datasets[split] = ALL_DATASETS[MODEL_DIM[args.model]](
                args.task,
                args.datapath,
                args.eval_segpath,
                split,
                num_classes=args.data_config["num_classes"],
                image_size_r=args.image_size_r,
                image_size_c=args.image_size_c,
                hist_eq=args.hist_eq,
            )
            
    for split, dataset in datasets.items():
        dataloaders.append(
            EmmentalDataLoader(
                task_to_label_dict={args.task: "labels"},
                dataset=dataset,
                split=split,
                shuffle=True if split in ["train"] else False,
                batch_size=args.batch_size
                if split in args.train_split or args.valid_batch_size is None
                else args.valid_batch_size,
                num_workers=4,
            )
        )
        logger.info(
            f"Built dataloader for {args.task} {split} set with {len(dataset)} "
            f"samples (Shuffle={split in args.train_split}, "
            f"Batch size={dataloaders[-1].batch_size})."
        )

    if args.consistency_datapath:
        datasets = {}

        for split in ["train"]:
            datasets[split] = ALL_CONSISTENCY_DATASETS[MODEL_DIM[args.model]](
                f"consistency_{args.task}",
                args.consistency_datapath,
                split,
                k=args.augment_k,
                image_size_r=args.image_size_r,
                image_size_c=args.image_size_c,
                hist_eq=args.hist_eq,
                csv_fn=args.consistency_csv_fn,
            )

        for split, dataset in datasets.items():
            dataloaders.append(
                EmmentalDataLoader(
                    task_to_label_dict={f"consistency_{args.task}": "labels"},
                    dataset=dataset,
                    split=split,
                    shuffle=True if split in ["train"] else False,
                    batch_size=args.consistency_batch_size,
                    num_workers=4,
                )
            )

            logger.info(
                f"Built dataloader for consistency_{args.task} {split} set "
                f"with {len(dataset)} "
                f"samples (Shuffle={split in args.train_split}, "
                f"Batch size={dataloaders[-1].batch_size})."
            )

    return dataloaders
