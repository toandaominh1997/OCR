import torch
import torch.nn as nn
from torch.nn import functional as F

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
