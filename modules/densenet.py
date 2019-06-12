import torch 
import torch.nn as nn
import torch.nn.functional as F

class _ConvBlock(nn.Sequential):
    def __init__(self, input_channel, growth_rate, dropout_rate=0.2):
        super(_ConvBlock, self).__init__()

        self.add_module('norm1_1', nn.BatchNorm2d(input_channel)),
        self.add_module('relu2_1', nn.ReLU(inplace=True)),
        self.add_module('conv2_1', nn.Conv2d(input_channel, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))

        self.dropout_rate = dropout_rate

    def forward(self, x):
        new_features = super(_ConvBlock, self).forward(x)
        if self.dropout_rate > 0:
            new_features = F.dropout(new_features, p=self.dropout_rate, training=self.training)

        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, nb_layers, nb_filter, growth_rate, dropout_rate=0.2, weight_decay=1e-4):
        super(_DenseBlock, self).__init__()

        for i in range(nb_layers):
            layer = _ConvBlock(nb_filter + i * growth_rate, growth_rate, dropout_rate)
            self.add_module('conv_block%d' % (i + 1), layer)

class _TransitionBlock(nn.Sequential):
    def __init__(self, nb_in_filter, nb_out_filter, dropout_rate=None):
        super(_TransitionBlock, self).__init__()

        self.add_module('norm', nn.BatchNorm2d(nb_in_filter))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(nb_in_filter, nb_out_filter, kernel_size=1, stride=1, bias=False))

        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class Densenet(nn.Module):
    def __init__(self, dropout_rate):
        super(Densenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, bias=False)

        self.dense_block1 = _DenseBlock(nb_layers=4, nb_filter=64, growth_rate=16, dropout_rate=dropout_rate)
        self.trans_block1 = _TransitionBlock(nb_in_filter=64 + 16 * 4, nb_out_filter=128)

        self.dense_block2 = _DenseBlock(nb_layers=6, nb_filter=128, growth_rate=16, dropout_rate=dropout_rate)
        self.trans_block2 = _TransitionBlock(nb_in_filter=128 + 16*6, nb_out_filter=224)

        self.dense_block3 = _DenseBlock(nb_layers=4, nb_filter=224, growth_rate=16, dropout_rate=dropout_rate)

        self.batch_norm4 = nn.BatchNorm2d(288)
    def forward(self, input):
        out = self.conv1(input[:, :, :, :] - 0.5)

        out = self.dense_block1(out)
        out = self.trans_block1(out)

        out = self.dense_block2(out)
        out = self.trans_block2(out)

        out = self.dense_block3(out)

        src = F.relu(self.batch_norm4(out), inplace=True)
        return src
