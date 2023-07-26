# Copyright (c) 2021 Sen Wu. All Rights Reserved.


from dauphin.image_segmentation.tasks.consistency_segmentation_2d_task import (
    create_task as create_consistency_segmentation_2d_task,
)
from dauphin.image_segmentation.tasks.segmentation_2d_task import (
    create_task as create_segmentation_2d_task,
)

ALL_TASKS = {
    "2d": create_segmentation_2d_task,
}

ALL_CONSISTENCY_TASKS = {
    "2d": create_consistency_segmentation_2d_task,
}
