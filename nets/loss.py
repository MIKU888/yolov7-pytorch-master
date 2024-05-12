import math
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaeLoss(nn.Module):
    def __init__(self, origin, out, mask):
        super(MaeLoss, self).__init__()
        self.origin = origin
        self.out = out
        self.mask = mask
        self.loss = nn.MSELoss()

    def __call__(self):
        return self.loss(self.out*(1-self.mask), self.origin*(1-self.mask))
