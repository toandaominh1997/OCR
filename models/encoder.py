import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from modules.densenet import Densenet
from modules.efficientnet import EfficientNet

class Encoder(nn.Module):
    def __init__(self, dropout_rate = 0.2, net='densenet'):
        super(Encoder, self).__init__()
        if net == 'densenet':
            self.model = Densenet(dropout_rate=dropout_rate)
        elif net == 'efficientnet':
            self.model = EfficientNet.from_name('efficientnet-b0')

    def forward(self, input):
        out = self.model(input)
        print('output: ', out.size())
        return out
