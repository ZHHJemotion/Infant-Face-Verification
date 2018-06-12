"""
    THe Net Model consists of DeepID2 and Residual Units

"""

import torch.nn as nn
from models.Siamese_net import SiameseNetwork
from models.resnet import ResNet17


class DeepID2_ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepID2_ResNet, self).__init__()
        self.SiameseNet = SiameseNetwork()
        self.ResNet = ResNet17()

        self.num_classes = num_classes

        self.fc1 = nn.Sequential(
            nn.Linear(64*14*14, 1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 2)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2)
        )

    def forward(self, x, y):
        feat_1, feat_2 = self.SiameseNet(x, y)

        # ------------------------------------------------------------#
        # fc feature will be used for contrastive loss function
        # ------------------------------------------------------------#
        fc_feat_1 = feat_1.view(feat_1.size(0), -1)
        fc_feat_2 = feat_2.view(feat_2.size(0), -1)
        fc_feat_1 = self.fc1(fc_feat_1)
        fc_feat_2 = self.fc1(fc_feat_2)

        # here: get the difference between both features from DeepID2
        diff_feat = feat_1 - feat_2

        diff_feat = self.ResNet(diff_feat)
        diff_feat = self.fc2(diff_feat)

        return fc_feat_1, fc_feat_2, diff_feat
