# Copyright (c) 2021 Sen Wu. All Rights Reserved.


import torch
import torch.nn.functional as F
from torch import nn

from dauphin.image_segmentation.modules.one_hot import one_hot


class DiceLoss(nn.Module):
    """
    Contrastive dice loss function **NOT TESTED THOROUGHLY**.
    """

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.eps: float = 1e-6

    def forward(self, pred, target):
        # compute softmax over the classes axis
        pred = F.softmax(pred, dim=-1)
        pred = torch.argmax(pred,dim=-1)

        pred = one_hot(
            pred,
            num_classes=target.shape[-1],
            device=target.device,
            eps=0
            dtype=target.dtype,
        )
        pred = pred.permute(0,2,3,1)

        pred = pred[:,:,:,1:] # Remove background class
        target = target[:,:,:,1:] # Remove background class

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(pred * target, dims)
        cardinality = torch.sum(pred + target, dims)

        dice_score = 2.0 * intersection / (cardinality + self.eps)
        
        dice_score[cardinality==0] = 1.0

        return 1.0 - dice_score
