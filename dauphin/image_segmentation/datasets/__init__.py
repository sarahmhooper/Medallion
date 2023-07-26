from dauphin.image_segmentation.datasets.medical_2d_image_dataset import (
    Medical2DImageDataset,
)
from dauphin.image_segmentation.datasets.medical_consistency_2d_image_dataset import (
    MedicalConsistency2DImageDataset,
)

ALL_DATASETS = {
    "2d": Medical2DImageDataset,
}

ALL_CONSISTENCY_DATASETS = {
    "2d": MedicalConsistency2DImageDataset,
}
