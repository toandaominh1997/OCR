import torch
import torch.nn as nn
from torch.nn import functional as F
from modules.bilstm import BidirectionalLSTM

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim = int(256 * 2), num_class = 200, relation_aware=False, net='densenet'):
        super(Decoder, self).__init__()

       #self.attend_layer = AttentionLayer(
       #     input_dim=hidden_dim,
       #     output_dim=hidden_dim,
       #     use_cuda=True,
       #     relation_aware=relation_aware
       # )
        self.rnn1 = BidirectionalLSTM(input_dim, hidden_dim, hidden_dim)
        self.rnn2 = BidirectionalLSTM(hidden_dim, hidden_dim, num_class)


    def forward(self, X):
        X = X.view(X.size(0), X.size(1)*X.size(2), X.size(3))
        X = X.permute(2, 0, 1)
        X = self.rnn1(X)
        #X = X.permute(1, 0, 2)
        #attend, attend_energies = self.attend_layer(X)
        #attend = attend.permute(1, 0, 2)
        output = self.rnn2(X)
        return output