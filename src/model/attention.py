import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common
from model.ENLA import ENLA
import math


class ENLCA(nn.Module):
    def __init__(self, channel=128, reduction=2, ksize=3, scale=3, stride=1, softmax_scale=10, average=True,
                 conv=common.default_conv, res_scale=0.1):
        super(ENLCA, self).__init__()
        self.conv_match1 = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=None)
        self.conv_match2 = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=None)
        self.conv_assembly = common.BasicBlock(conv, channel, channel, 1, bn=False, act=None)
        self.res_scale = res_scale
        self.attn_fn = ENLA(
            dim_heads=channel // reduction,
            nb_features=128,
        )
        self.k = math.sqrt(6)

    def forward(self, input):
        x_embed_1 = self.conv_match1(input)
        x_embed_2 = self.conv_match2(input)
        x_assembly = self.conv_assembly(input) #[B,C,H,W]
        x_embed_2 = F.normalize(x_embed_2, p=2, dim=1,eps=5e-5)*self.k
        x_embed_1 = F.normalize(x_embed_1, p=2, dim=1, eps=5e-5)*self.k
        N, C, H, W = x_embed_1.shape
        loss = 0
        if self.training:
            score = torch.matmul(x_embed_1.permute(0, 2, 3, 1).view((N, H * W, C)),
                                 x_embed_2.view(N, C, H * W))  # [N,H*W,H*W]
            score = torch.exp(score)
            score = torch.sort(score, dim=2, descending=True)[0]
            positive = torch.mean(score[:, :, :5], dim=2)
            negative = torch.mean(score[:, :, 50:], dim=2)  # [N,H*W]
            loss = F.relu(-1 * torch.log(positive / (negative + 1e-6))+1)
            loss = torch.mean(loss)

        x_embed_1 = x_embed_1.permute(0, 2, 3, 1).view(N,1, H * W, C)
        x_embed_2 = x_embed_2.permute(0, 2, 3, 1).view(N,1, H * W, C)
        x_assembly = x_assembly.permute(0, 2, 3, 1).view(N,1, H*W, -1 )
        x_final = self.attn_fn(x_embed_1, x_embed_2, x_assembly).squeeze(1)# (1, H*W, C)
        return x_final.permute(0, 2, 1).view(N, -1, H, W)*self.res_scale+input,loss
