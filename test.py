import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss
import time

parser = argparse.ArgumentParser()

parser.add_argument('--root', required=True, help='path to dataset')
parser.add_argument('--train_label', required=True, help='path to dataset')
parser.add_argument('--valid_label', required=True, help='path to dataset')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=256, help='input hidden size')
parser.add_argument('--height', type=int, default=48, help='the height of the input image to network')
parser.add_argument('--num_class', type=int, default=48, help='the number class of the input image to network')
parser.add_argument('--num_epoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
