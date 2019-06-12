import torch 
import torch.nn as nn
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

    def forward(self, input):
        output = self.crnn(input)

        return output


