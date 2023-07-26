import itertools
import logging
import math
from random import choices
from typing import Dict, Iterator, List, Tuple, Union

import emmental
import numpy as np
import torch
from emmental.data import EmmentalDataLoader
from emmental.schedulers.scheduler import Scheduler, Batch
from sklearn.preprocessing import normalize
from torch import Tensor

logger = logging.getLogger(__name__)


class ConAugScheduler(Scheduler):
    """Generate batch generator from all dataloaders in round robin order for training.
    Args:
      fillup(bool): Whether fillup to make all dataloader the same size.
    """

    def __init__(
        self,
        fillup: bool = False,
        trim: bool = False,
        augment_k: int = None,
        enlarge: int = 1,
        n_sup_batches_per_step=1,
        n_unsup_batches_per_step=1,
    ) -> None:
        super().__init__()
        self.fillup = fillup
        self.trim = trim
        self.augment_k = augment_k
        self.enlarge = enlarge
        self.n_sup_batches_per_step = (
            n_sup_batches_per_step if n_sup_batches_per_step else 1
        )
        self.n_unsup_batches_per_step = (
            n_unsup_batches_per_step if n_unsup_batches_per_step else 1
        )
        self.first_time = True

        assert (
            self.augment_k is None or self.enlarge <= self.augment_k
        ), f"{self.enlarge} <= {self.augment_k}"

    def get_num_batches(self, dataloaders: List[EmmentalDataLoader]) -> int:
        """Get total number of batches per epoch.
        Args:
          dataloaders(list): List of dataloaders.
        Returns:
          int: Total number of batches per epoch.
        """
        batch_counts = [len(dataloader) for dataloader in dataloaders]
        # if self.fillup:
        #     batch_counts = [max(batch_counts)] * len(dataloaders)
        # elif self.trim:
        #     batch_counts = [min(batch_counts)] * len(dataloaders)

        for idx in range(len(dataloaders)):
            if dataloaders[idx].n_batches:
                batch_counts[idx] = dataloaders[idx].n_batches

        base_cnt = 0
        total_steps = 0
        for idx in range(len(dataloaders)):
            if not dataloaders[idx].dataset.task_name.startswith("consistency_"):
                base_cnt += self.n_sup_batches_per_step
                total_steps = max(
                    total_steps,
                    math.ceil(batch_counts[idx] / self.n_sup_batches_per_step),
                )
            else:
                base_cnt += self.n_unsup_batches_per_step

        return base_cnt * total_steps

    def get_batches(
        self, dataloaders: List[EmmentalDataLoader], model
    ) -> Iterator[
        Tuple[
            List[str],
            Dict[str, Union[Tensor, List[str]]],
            Dict[str, Tensor],
            Dict[str, str],
            str,
            str,
        ]
    ]:
        """Generate batch generator from all dataloaders in round robin order.
        Args:
          dataloaders(list): List of dataloaders.
        Returns:
          genertor: A generator of all batches.
        """
        task_to_label_dicts = [
            dataloader.task_to_label_dict for dataloader in dataloaders
        ]
        uid_names = [dataloader.uid for dataloader in dataloaders]
        data_names = [dataloader.data_name for dataloader in dataloaders]
        splits = [dataloader.split for dataloader in dataloaders]
        data_loaders = [iter(dataloader) for dataloader in dataloaders]

        # Calc the batch size for each dataloader
        # batch_counts = [len(dataloader) for dataloader in dataloaders]
        # if self.fillup:
        #     batch_counts = [max(batch_counts)] * len(dataloaders)
        # elif self.trim:
        #     batch_counts = [min(batch_counts)] * len(dataloaders)

        # for idx in range(len(dataloaders)):
        #     if dataloaders[idx].n_batches:
        #         batch_counts[idx] = dataloaders[idx].n_batches
        batch_counts = [len(dataloader) for dataloader in dataloaders]
        # if self.fillup:
        #     batch_counts = [max(batch_counts)] * len(dataloaders)
        # elif self.trim:
        #     batch_counts = [min(batch_counts)] * len(dataloaders)

        for idx in range(len(dataloaders)):
            if dataloaders[idx].n_batches:
                batch_counts[idx] = dataloaders[idx].n_batches

        total_steps = 0
        for idx in range(len(dataloaders)):
            if not dataloaders[idx].dataset.task_name.startswith("consistency_"):
                total_steps = max(
                    total_steps,
                    math.ceil(batch_counts[idx] / self.n_sup_batches_per_step),
                )

        indexer = []
        for idx in range(len(dataloaders)):
            if not dataloaders[idx].dataset.task_name.startswith("consistency_"):
                indexer.extend([idx] * self.n_sup_batches_per_step)
            else:
                indexer.extend([idx] * self.n_unsup_batches_per_step)

        if (
            len(indexer)
            != emmental.Meta.config["learner_config"]["optimizer_config"][
                "gradient_accumulation_steps"
            ]
        ) and self.first_time:
            gradient_accumulation_steps = emmental.Meta.config["learner_config"][
                "optimizer_config"
            ]["gradient_accumulation_steps"]
            logger.warn(
                f"Number of gradient accumulation steps "
                f"({gradient_accumulation_steps}) "
                f"doesn't match batches in each step ({len(indexer)})!!!"
            )
            self.first_time = False

        dataloader_indexer = list(
            itertools.chain.from_iterable([indexer] * total_steps)
        )

        for data_loader_idx in dataloader_indexer:
            uid_name = uid_names[data_loader_idx]
            try:
                out = next(data_loaders[data_loader_idx])
            except StopIteration:
                data_loaders[data_loader_idx] = iter(dataloaders[data_loader_idx])
                out = next(data_loaders[data_loader_idx])

            if isinstance(out, dict):
                X_dict = out
                Y_dict = None
            else:
                X_dict, Y_dict = out

            if self.augment_k and self.augment_k > 1 and self.augment_k > self.enlarge:
                model.eval()
                with torch.no_grad():
                    uid_dict, loss_dict, prob_dict, gold_dict = model(
                        X_dict[uid_name],
                        X_dict,
                        Y_dict,
                        task_to_label_dicts[data_loader_idx],
                        return_probs=False,
                    )
                model.train()

                # Collect losses
                loss_dist = list(loss_dict.values())[0].detach().cpu().numpy()

                # row-based weighted sampling
                dist = normalize(
                    np.array(loss_dist).reshape(-1, self.augment_k), axis=1, norm="l1"
                )
                select_idx = np.vstack(
                    [
                        i * self.augment_k
                        + np.array(
                            choices(range(self.augment_k), dist[i], k=self.enlarge)
                        )
                        if max(dist[i]) > 0
                        else i * self.augment_k
                        + np.array(choices(range(self.augment_k), k=self.enlarge))
                        for i in range(dist.shape[0])
                    ]
                ).reshape(-1)

                if "mask" in X_dict:
                    X_new_dict = {"image": [], "mask": [], uid_name: []}
                else:
                    X_new_dict = {"image": [], uid_name: []}
                Y_new_dict = {"labels": []}

                for idx in select_idx:
                    X_new_dict[uid_name].append(X_dict[uid_name][idx])
                if "consistency_image" not in X_dict:
                    X_new_dict["image"] = X_dict["image"][select_idx]
                    if "mask" in X_dict:
                        X_new_dict["mask"] = X_dict["mask"][select_idx]
                else:
                    X_new_dict["image"] = X_dict["image"]
                    if "mask" in X_dict:
                        X_new_dict["mask"] = X_dict["mask"]
                if "consistency_image" in X_dict:
                    X_new_dict["consistency_image"] = X_dict["consistency_image"][select_idx]
                Y_new_dict["labels"] = Y_dict["labels"][select_idx]

                X_dict = X_new_dict
                Y_dict = Y_new_dict
                
            yield Batch(
                X_dict[uid_name],
                X_dict,
                Y_dict,
                task_to_label_dicts[data_loader_idx],
                data_names[data_loader_idx],
                splits[data_loader_idx],
            )

            