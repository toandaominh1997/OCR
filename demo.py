import torch 
from models.model import Model
input = torch.randn(32, 1, 48, 600)

model = Model(num_classes = 1000)
output = model(input)

print('output: ', output.size())
