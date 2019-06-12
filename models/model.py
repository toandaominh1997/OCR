import torch 
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder
from models.decoder import Decoder


class Model(nn.Module):
    def __init__(self, num_classes, fixed_height = 48, net='efficientnet'):
        super(Model, self).__init__()
        self.encoder = Encoder(net = net)
        self.decoder = Decoder(input_dim=int(640), num_class=num_classes)
        self.crnn = nn.Sequential(
            self.encoder,
            self.decoder
        )
        self.log_softmax = nn.Softmax(dim=2)

    def forward(self, input):
        output = self.crnn(input)
        output = self.log_softmax(output)
        return output


