"""
    Siamese network part with DeepID2
"""

import torch
import torch.nn as nn
from models.deepid2 import DeepID2


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.branch_1 = nn.Sequential(
            DeepID2()
        )

        self.branch_2 = nn.Sequential(
            DeepID2()
        )

    def forward(self, x, y):
        out_1 = self.branch_1(x)
        out_2 = self.branch_2(y)

        return out_1, out_2
