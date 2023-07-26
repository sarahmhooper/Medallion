# Copyright (c) 2021 Sen Wu. All Rights Reserved.


import torch
from torch import Tensor, nn


class PrepConsistencyFeature(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f1: Tensor, f2: Tensor) -> Tensor:  # type:ignore
        return torch.cat((f1, f2), 0)
