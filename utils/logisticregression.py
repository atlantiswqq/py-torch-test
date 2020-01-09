# _*_ coding:utf-8 _*_
# Author:Atlantis
# Date:2020-01-09

import torch.nn as nn


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_idm, out_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_idm, out_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

