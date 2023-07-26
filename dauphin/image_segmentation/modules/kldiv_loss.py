# Copyright (c) 2021 Sen Wu. All Rights Reserved.


import torch
from torch import nn


class KLDivLoss(nn.Module):
    """
    Contrastive KL div loss function **NOT TESTED THOROUGHLY**.
    """

    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.KLL = torch.nn.KLDivLoss(reduction="none")
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input1, input2):
        bsz = input1.size()[0]
        fdim = input1.size()[1]

        input1 = input1.permute(0, 2, 3, 1).reshape(-1, fdim)
        input2 = input2.permute(0, 2, 3, 1).reshape(-1, fdim)

        loss = self.KLL(self.log_softmax(input1), self.softmax(input2))
        loss = torch.sum(loss, dim=1)
        loss = loss.view(bsz, -1)

        return loss.mean(1)
