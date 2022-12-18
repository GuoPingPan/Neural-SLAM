from utils.model import ChannelPool, get_grid
import torch

# a = torch.rand(10,3,244,244)
# pool = ChannelPool(1)
#
# out = pool(a)
# print(out.shape)

# a = get_grid(torch.randn(10,3),(10,2,100,100),'cpu')

import cv2

# m = torch.Tensor(
#     [[[0.7071,-0.7071,0],
#      [0.7071,0.7071,0]]]
# )
# print(m)
# import torch.nn.functional as F
#
# grid = F.affine_grid(m,size=(1,3,10,10))
# print(grid.shape)
# print(grid[0,2,4,:])


# a = {}
# a['a'] = (a.get('a1',0.1),a.get('a2',0.1))
# print(a)
# c = a.pop('a1',None)
# print(c)

# full_map = torch.zeros(10,10).float()
# local_map = torch.zeros(5,5).float()
#
# local_map = full_map[int(full_map.shape[0]/2 - local_map.shape[0]/2):int(full_map.shape[0]/2 - local_map.shape[0]/2)+local_map.shape[0],\
#             int(full_map.shape[1]/2 - local_map.shape[1]/2):int(full_map.shape[1]/2 - local_map.shape[1]/2)+local_map.shape[1]]
#
# full_map[4-1:4+1,4-1:4+1] = 10
# print(local_map)

import torch.nn as nn
a = torch.randn(10,8,240,240)

# main = nn.Sequential(
#     nn.MaxPool2d(2),
#     nn.Conv2d(8, 32, 3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(2),
#     nn.Conv2d(32, 64, 3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(2),
#     nn.Conv2d(64, 128, 3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(2),
#     nn.Conv2d(128, 64, 3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.Conv2d(64, 32, 3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.Flatten()
# )
# out = main(a)
# print(out.shape)
# print(240*240*2)

# The following code is largely borrowed from:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/distributions.py



#
# import torch
# import torch.nn as nn
#
# from utils.model import AddBias
#
# """
# Modify standard PyTorch distributions so they are compatible with this code.
# """
#
# """Dicrete"""
# FixedCategorical = torch.distributions.Categorical
#
#
#
# old_sample = FixedCategorical.sample
# FixedCategorical.sample = lambda self: old_sample(self)
#
# log_prob_cat = FixedCategorical.log_prob
# FixedCategorical.log_probs = lambda self, actions: \
#     log_prob_cat(self, actions.squeeze(-1))
# FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)
#
#
# """Box"""
# FixedNormal = torch.distributions.Normal
# log_prob_normal = FixedNormal.log_prob
# FixedNormal.log_probs = lambda self, actions: \
#     log_prob_normal(self, actions).sum(-1, keepdim=False)
#
# entropy = FixedNormal.entropy
# FixedNormal.entropy = lambda self: entropy(self).sum(-1)
#
# FixedNormal.mode = lambda self: self.mean
#
#
# class DiagGaussian(nn.Module):
#
#     def __init__(self, num_inputs, num_outputs):
#         super(DiagGaussian, self).__init__()
#
#         self.fc_mean = nn.Linear(num_inputs, num_outputs)
#         self.logstd = AddBias(torch.zeros(num_outputs))
#
#     def forward(self, x):
#         # n,2
#         action_mean = self.fc_mean(x)
#
#         print(action_mean)
#
#         zeros = torch.zeros(action_mean.size())
#         if x.is_cuda:
#             zeros = zeros.cuda()
#
#         # 添加可学习噪声
#         action_logstd = self.logstd(zeros)
#
#         # 按照当前的输出为均值，传入可学习的噪声方差
#         return FixedNormal(action_mean, action_logstd.exp())
#
#
# dist = DiagGaussian(256,2)
#
# a = torch.randn(3,256)
#
# d = dist(a)
#
# print(type(d))
# print(d.mode())
# # print(a)
# print(d.sample())
# # print(dist.logstd._bias)
# print(d.log_prob(d.sample()))
# print(d.log_probs(d.sample()))


# import cv2
#
# img = cv2.imread(r'C:\Users\73106\Pictures\t01a1402407a2d7747b.jpg')
# cv2.imshow('a',img)
# cv2.waitKey(0)
