'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import numpy as np

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def reg_loss(self):
        reg_loss = 0.0
        cnt = 0
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                reg_loss += torch.sum(torch.abs(m.weight))
        return reg_loss
    
    def l1reg_loss(self):
        reg_loss = 0.0
        cnt = 0
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                reg_loss += torch.sum(torch.abs(m.weight))
        return reg_loss

    def l12reg_loss(self):
        reg_loss = 0.0
        cnt = 0
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                reg_loss += torch.sum(torch.abs(m.weight).sqrt())
        return reg_loss

    def l23reg_loss(self):
        reg_loss = 0.0
        cnt = 0 
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                reg_loss += torch.sum(torch.abs(m.weight).pow(2/3))
        return reg_loss

    def exact_sparsity(self):
        nnz = 0.0
        total_param = 0.0
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                total_param += np.prod(m.weight.data.shape)
                nnz += torch.sum(m.weight.data != 0).detach().item()
        ratio = nnz / total_param
        return ratio

    def quantized_ratio(self):
        total_param = 0.0
        quantized_param = 0.0
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                total_param += np.prod(m.weight.data.shape)
                total_param += np.prod(m.bias.data.shape)
                quantized_param += torch.sum(m.weight.data != 1.0).detach().item()
                quantized_param += torch.sum(m.weight.data != -1.0).detach().item()
        ratio = quantized_param / total_param
        return ratio

    def sparsity_level(self):
        nnz_2 = 0.0
        nnz_3 = 0.0
        nnz_4 = 0.0
        total_param = 0.0
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                total_param += np.prod(m.weight.data.shape)
                nnz_2 += torch.sum(m.weight.data.abs() >= 0.01).detach().item()
                nnz_3 += torch.sum(m.weight.data.abs() >= 0.001).detach().item()
                nnz_4 += torch.sum(m.weight.data.abs() >= 0.0001).detach().item()
        ratio_2 = nnz_2 / total_param
        ratio_3 = nnz_3 / total_param
        ratio_4 = nnz_4 / total_param
        return ratio_2, ratio_3

def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
