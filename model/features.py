import torch
import torch.nn as nn

from utils.deconvolution import torch_richardson_lucy


class Conv(nn.Module):
    def __init__(self , input_channels , n_feats , kernel_size , stride = 1 ,padding=0 , bias=True , bn = False , act=False ):
        super(Conv , self).__init__()
        m = []
        m.append(nn.Conv2d(input_channels , n_feats , kernel_size , stride , padding , bias=bias))
        if bn: m.append(nn.BatchNorm2d(n_feats))
        if act:m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)
    def forward(self, input):
        return self.body(input)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, padding = 0 ,bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, padding = padding , bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        n_feats1 = 16
        kernel_size = 5
        self.n_colors = 3

        FeatureBlock = [
            Conv(self.n_colors, n_feats1, kernel_size, padding=2, act=True),
            ResBlock(Conv, n_feats1, kernel_size, padding=2),
            ResBlock(Conv, n_feats1, kernel_size, padding=2),
            ResBlock(Conv, n_feats1, kernel_size, padding=2),
        ]

        self.FeatureBlock = nn.Sequential(*FeatureBlock)


    def load_weights(self, path):
        self.load_state_dict(torch.load(path))


    def forward(self, input, kernel=None):
        feature_out = self.FeatureBlock(input)

        feature_rl = torch.zeros(feature_out.size())
        for i in range(feature_out.shape[1]):
            feature_rl[:,i] = torch_richardson_lucy(feature_out[:,i].T, kernel)

        return feature_out, feature_rl
