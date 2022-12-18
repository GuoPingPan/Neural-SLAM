# The following code is largely borrowed from:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/distributions.py

import torch
import torch.nn as nn

from utils.model import AddBias

"""
Modify standard PyTorch distributions so they are compatible with this code.
这里的目的应该是为了修改维度
"""

# https://blog.csdn.net/HaoZiHuang/article/details/126356668

"""Dicrete"""
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: \
    log_prob_cat(self, actions.squeeze(-1))
FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)


"""Box"""
FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: \
    log_prob_normal(self, actions).sum(-1, keepdim=False)

entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


class Categorical(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        # 输出空间所有动作的概率
        x = self.linear(x)

        # 用空间所有动作的概率对离散分布进行初始化
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        self.fc_mean = nn.Linear(num_inputs, num_outputs)
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        # n,2
        action_mean = self.fc_mean(x)

        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        # 添加可学习噪声
        action_logstd = self.logstd(zeros)

        # 按照当前的输出为均值，传入可学习的噪声方差
        return FixedNormal(action_mean, action_logstd.exp())
