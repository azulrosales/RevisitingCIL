'''
Reference:
https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/cifar100-class-incremental/modified_linear.py
'''
import math
import torch
from torch import nn
from torch.nn import functional as F

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        # Cosine similarity
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}