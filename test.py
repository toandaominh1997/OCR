import os
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss
import time
import datetime

from dataset import dataset
from dataset import aug

from model import dcrnn

from util import util
from util import convert
from util import metric

parser = argparse.ArgumentParser()

parser.add_argument('--root', required=True, help='path to dataset')
parser.add_argument('--train_label', required=True, help='path to dataset')
parser.add_argument('--valid_label', required=True, help='path to dataset')
parser.add_argument('--test_label', default=None, help='path to dataset')
parser.add_argument('--num_worker', type=int, help='number of data loading workers', default=10)
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=256, help='input hidden size')
parser.add_argument('--height', type=int, default=48, help='the height of the input image to network')
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--num_class', type=int, default=48, help='the number class of the input image to network')
parser.add_argument('--num_epoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for neural network')
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--num_gpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--resume', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--save_dir', default='saved', help='Where to store samples and models')
parser.add_argument('--manual_seed', type=int, default=1234, help='reproduce experiemnt')

args = parser.parse_args()
start_time = datetime.datetime.now().strftime('%m-%d_%H%M%S')
if(not os.path.exists(os.path.join(args.save_dir, start_time))):
    os.makedirs(os.path.join(args.save_dir, start_time))
random.seed(args.manual_seed)
np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
cudnn.benchmark = True
args.alphabet = util.get_vocab(root=args.root, label=args.train_label)

if(torch.cuda.is_available() and args.cuda):
    torch.cuda.set_device(util.get_gpu())
if(torch.cuda.is_available() and not args.cuda):
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

test_transform = aug.test_transforms(height=args.height)

if(args.test_label is not None):
    test_dataset = dataset.ocrDataset(args=args, root=args.root, label=args.test_label, train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=int(args.num_worker), collate_fn=dataset.alignCollate())


args.num_class = len(args.alphabet) + 1
converter = convert.strLabelConverter(args.alphabet)

model = dcrnn.Model(n_classes=args.num_class, fixed_height=args.height)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

if(args.resume!=''):
    print('loading pretrained model from {}'.format(args.resume))
    model, _ = util.resume_checkpoint(model, optimizer, args.resume)
criterion = CTCLoss()
if(torch.cuda.is_available() and args.cuda):
    model = model.cuda()
    criterion = criterion.cuda()

def evaluate(data_loader):
    print(['start evaluate'])
    model.eval()
    total_loss=0
    accBF = 0.0
    accBC = 0.0
    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader):
            batch_size = image.size(0)
            image = image.cuda()
            label, target_size = converter.encode(target)
            output = model(image)
            output_size = Variable(torch.IntTensor([output.size(0)] * batch_size))
            loss = criterion(output, label, output_size, target_size)/batch_size
            _, output = output.max(2)
            output = output.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(output.data, output_size.data, raw=False)
            accBF += metric.by_field(sim_preds, target)
            accBC += metric.by_char(sim_preds, target)
        total_loss /=len(data_loader)
        return total_loss, accBF/len(data_loader), accBC/len(data_loader)
    


if __name__ == '__main__':
    main()
