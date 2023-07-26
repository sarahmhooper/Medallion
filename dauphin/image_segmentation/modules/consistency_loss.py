# Copyright (c) 2021 Sen Wu. All Rights Reserved.


import numpy as np
import torch
from torch import nn


class ConsistencyLoss(nn.Module):
    """
    Consistency loss function.
    """

    def __init__(self):
        super(ConsistencyLoss, self).__init__()
        self.CEL = torch.nn.CosineEmbeddingLoss(reduction="none")

    def forward(self, input1, input2):
        bsz1 = input1.size()[0]
        bsz2 = input2.size()[0]
        fdim = input1.size()[1]
        dim = input1.size()[1] * input1.size()[2] * input1.size()[3]

        input1 = input1.permute(0, 2, 3, 1).reshape(-1, fdim)
        input2 = input2.permute(0, 2, 3, 1).reshape(-1, fdim)
        k = bsz2 // bsz1
        input1 = (
            input1.reshape(bsz1, -1).repeat(1, k).reshape(-1, dim).reshape(-1, fdim)
        )

        loss = self.CEL(
            input1,
            input2,
            torch.from_numpy(np.array([1] * input1.size()[0])).to(input1.device),
        )
        loss = loss.view(bsz2, -1)
        return loss.mean(1)
