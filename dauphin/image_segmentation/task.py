# Copyright (c) 2021 Sen Wu. All Rights Reserved.


from dauphin.image_segmentation.models import MODEL_DIM
from dauphin.image_segmentation.tasks import ALL_CONSISTENCY_TASKS, ALL_TASKS


def create_task(args):
    tasks = []

    # Create segmentation task
    tasks.append(ALL_TASKS[MODEL_DIM[args.model]](args))

    # Create contrastive task
    tasks.append(ALL_CONSISTENCY_TASKS[MODEL_DIM[args.model]](args))

    return tasks
