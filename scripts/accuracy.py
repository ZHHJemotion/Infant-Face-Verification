import torch
import torch.nn as nn
import torch.nn.functional as F


class Accuracy(object):
    def __init__(self, batch_size):
        super(Accuracy, self).__init__()
        self.batch_size = batch_size
