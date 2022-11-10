'''Loss Prediction Module in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''
import torch
import torch.nn as nn 
import torch.nn.functional as F 


class LossNet(nn.Module):
    # def __init__(self, feature_sizes=[32, 16, 8, 4], num_channels=[256,512,1024,2048], interm_dim=512):
    def __init__(self, feature_sizes=[256, 128, 64, 32], num_channels=[256, 512, 1024, 2048], interm_dim=512): # 224, 224 size
    # def __init__(self, feature_sizes=[32, 16, 8, 4], num_channels=[30976, 61952, 123904, 247808], interm_dim=128): # 360,360 size
        super(LossNet, self).__init__()
        
        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        # print("self.FC1", self.FC1)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        # print("self.FC2", self.FC2)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        # print("self.FC3", self.FC3)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)
        # print("self.FC4", self.FC4)

        self.linear = nn.Linear(4 * interm_dim, 1)
        # print("interm_idm", 4 * interm_dim )
        
    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        # print("------------------------")
        # print("out1 shape", out1.shape)
        
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        # print("------------------------")
        # print("out2 shape", out2.shape)
        
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        
        # print("------------------------")
        # print("out3 shape", out3.shape)
        
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        
        out4 = F.relu(self.FC4(out4))
        
        # print("------------------")
        # print(torch.cat((out1, out2, out3, out4)))
        
        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out