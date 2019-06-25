import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import math

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

class Encoder(nn.Module):
    def __init__(self, dropout_rate = 0.2):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, bias=False)

        self.dense_block1 = _DenseBlock(nb_layers=4, nb_filter=64, growth_rate=16, dropout_rate=dropout_rate)
        self.trans_block1 = _TransitionBlock(nb_in_filter=64 + 16 * 4, nb_out_filter=128)

        self.dense_block2 = _DenseBlock(nb_layers=6, nb_filter=128, growth_rate=16, dropout_rate=dropout_rate)
        self.trans_block2 = _TransitionBlock(nb_in_filter=128 + 16*6, nb_out_filter=224)

        self.dense_block3 = _DenseBlock(nb_layers=4, nb_filter=224, growth_rate=16, dropout_rate=dropout_rate)

        self.batch_norm4 = nn.BatchNorm2d(288)

    def forward(self, src): # (b, c, h, w)
        batch_size = src.size(0)

        out = self.conv1(src[:, :, :, :] - 0.5)

        out = self.dense_block1(out)
        out = self.trans_block1(out)

        out = self.dense_block2(out)
        out = self.trans_block2(out)

        out = self.dense_block3(out)

        src = F.relu(self.batch_norm4(out), inplace=True)

        return src


class BidirectionalLSTM(nn.Module):
    def __init__(self, num_features, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(num_features, hidden_size, bidirectional=True)
        self.output = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_features):
        recurrent, _ = self.rnn(input_features)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.output(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, use_cuda, seqlen=26, relation_aware=False):
        super(AttentionLayer, self).__init__()

        self.output_dim = output_dim
        self.use_cuda = use_cuda
        self.relation_aware = relation_aware

        self.linear_v = nn.Linear(input_dim, output_dim, bias=False)
        self.linear_q = nn.Linear(input_dim, output_dim, bias=False)
        self.linear_k = nn.Linear(input_dim, output_dim, bias=False)

        if self.relation_aware:
            self.alpha_V = nn.Parameter(torch.zeros((seqlen, seqlen, output_dim)))
            self.alpha_K = nn.Parameter(torch.zeros((seqlen, seqlen, output_dim)))

    def forward(self, x):

        batch_size, seq_len, num_features = x.size()
        x_k = self.linear_k(x)
        x_q = self.linear_q(x)
        x_v = self.linear_v(x)
        if not self.relation_aware:
            atten_energies = torch.matmul(x_q, x_k.transpose(2, 1))/math.sqrt(self.output_dim)

            atten_energies = torch.stack([F.softmax(atten_energies[i]) for i in range(batch_size)])
            z = torch.matmul(atten_energies, x_v)

        else:
            alpha_V = nn.Parameter(torch.zeros((seq_len, seq_len, self.output_dim)))
            alpha_K = nn.Parameter(torch.zeros((seq_len, seq_len, self.output_dim)))
            atten_energies = Variable(torch.zeros((batch_size, seq_len, seq_len)))
            z = Variable(torch.zeros((batch_size, seq_len, self.output_dim)))
            if self.use_cuda:
                z = z.cuda()
                atten_energies = atten_energies.cuda()
                alpha_K = alpha_K.cuda()
                alpha_V = alpha_V.cuda()
            for i in range(seq_len):
                x_k_ = x_k + alpha_K[i]
                atten_energy = torch.matmul(x_q[:, i].unsqueeze(1), x_k_.transpose(2, 1))/math.sqrt(self.output_dim)
                atten_energy = F.softmax(atten_energy.squeeze(1)).unsqueeze(1)
                x_v_ = x_v + alpha_V[i]
                z[:, i] = torch.matmul(atten_energy, x_v_).squeeze(1)
                atten_energies[:, i] = atten_energy.squeeze(1)
        return z, atten_energies


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim = int(256 * 2), n_classes = 200, relation_aware=False):
        super(Decoder, self).__init__()

       #self.attend_layer = AttentionLayer(
       #     input_dim=hidden_dim,
       #     output_dim=hidden_dim,
       #     use_cuda=True,
       #     relation_aware=relation_aware
       # )
        self.rnn1 = BidirectionalLSTM(input_dim, hidden_dim, hidden_dim)
        self.rnn2 = BidirectionalLSTM(hidden_dim, hidden_dim, n_classes)


    def forward(self, X):
        X = X.view(X.size(0), X.size(1)*X.size(2), X.size(3))
        X = X.permute(2, 0, 1)
        X = self.rnn1(X)
        #X = X.permute(1, 0, 2)
        #attend, attend_energies = self.attend_layer(X)
        #attend = attend.permute(1, 0, 2)
        output = self.rnn2(X)
        return output

class Model(nn.Module):
    def __init__(self, n_classes, fixed_height = 48):
        super(Model, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder(input_dim=int(fixed_height * 288 / 8), n_classes=n_classes)

        self.crnn = nn.Sequential(
            self.encoder,
            self.decoder
        )

        for p in self.crnn.parameters():
            if isinstance(p, nn.Conv2d):
                nn.init.kaiming_normal_(p.weight)
            elif isinstance(p, nn.BatchNorm2d):
                nn.init.constant_(p.weight, 1)
                nn.init.constant_(p.bias, 0)
            else:
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, input):
        output = self.crnn(input)

        return output


image = torch.randn(32, 1, 48, 500)

model = Model(n_classes=78)

output = model(image)

print(output.size())
