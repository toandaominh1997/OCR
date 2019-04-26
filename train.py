import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss
import time
# import from file config
from util import util
from dataset import dataset
from dataset import aug
from util import convert
from model import dcrnn
from model import crnn
from model import metric
from model import googlenet
parser = argparse.ArgumentParser()

parser.add_argument('--root', required=True, help='path to dataset')
parser.add_argument('--train_label', required=True, help='path to dataset')
parser.add_argument('--valid_label', required=True, help='path to dataset')
parser.add_argument('--num_worker', type=int, help='number of data loading workers', default=10)
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--num_epoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--height', type=int, default=48, help='the height of the input image to network')
parser.add_argument('--num_class', type=int, default=48, help='the number class of the input image to network')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for neural network')
parser.add_argument('--gpu_id', type=int, default=-1, help='id of GPUs to use')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
opt = parser.parse_args()
if(opt.gpu_id==-1):
    opt.gpu_id = util.get_gpu()
print(opt)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True
torch.cuda.set_device(opt.gpu_id)

train_transform = aug.train_transforms(height=opt.height)
test_transform = aug.test_transforms(height=opt.height)

train_dataset = dataset.ocrDataset(root=opt.root, label=opt.train_label, transform=train_transform)
test_dataset = dataset.ocrDataset(root=opt.root, label=opt.valid_label, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_worker), collate_fn=dataset.alignCollate())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=int(opt.num_worker), collate_fn=dataset.alignCollate())

opt.alphabet = util.get_vocab(root='ocr_kaggle/data', label=opt.train_label)

opt.num_class = len(opt.alphabet)+1
converter = convert.strLabelConverter(opt.alphabet)
criterion = CTCLoss()


# model = dcrnn.Model(n_classes=opt.num_class, fixed_height=opt.height)
model = googlenet.Model(num_class=opt.num_class)
data = torch.FloatTensor(opt.batch_size, 1, 64, 600)
target = torch.IntTensor(opt.batch_size * 5)
length = torch.IntTensor(opt.batch_size)


if(opt.gpu_id>=0):
    model = model.cuda()
    data = data.cuda()
    criterion = criterion.cuda()
data = Variable(data)
target = Variable(target)
length = Variable(length)

optimizer = optim.Adam(model.parameters(), lr=opt.lr)



def train_epoch(model, data_loader):
    total_loss = 0
    model.train()
    start = time.time()
    for idx, (_data, _target) in enumerate(data_loader):
        batch_size = _data.size(0)
        util.loadData(data, _data)
        t, l = converter.encode(_target)
        util.loadData(target, t)
        util.loadData(length, l)
        optimizer.zero_grad()
        output = model(data)
        output_size = Variable(torch.IntTensor([output.size(0)] * batch_size))
        loss = criterion(output, target, output_size, length) / batch_size
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if((idx+1)%1000==0):
            print('Loss: {}'.format(total_loss/idx))
    return total_loss/len(data_loader)
def valid(model, data_loader):
    model.eval()
    total_val_loss = 0
    accBF = 0.0
    accBC = 0.0
    with torch.no_grad():
        for idx, (_data, _target) in enumerate(data_loader):
            batch_size = _data.size(0)
            util.loadData(data, _data)
            t, l = converter.encode(_target)
            util.loadData(target, t)
            util.loadData(length, l)
            output = model(data)
            output_size = Variable(torch.IntTensor([output.size(0)] * batch_size))
            loss = criterion(output, target, output_size, length) / batch_size
            total_val_loss += loss.item()
            _, output = output.max(2)
            output = output.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(output.data, output_size.data, raw=False)
            accBF += metric.by_field(sim_preds, _target)
            accBC += metric.by_char(sim_preds, _target)
        print('Test-Loss: {}, accBF: {}, accBC: {}'.format(total_val_loss/len(data_loader), accBF/len(data_loader), accBC/len(data_loader)))


for epoch in range(1, opt.num_epoch):
    start = time.time()
    loss= train_epoch(model, train_loader)  
    print('Time: {}, Epoch: {}/{}, Loss: {}'.format(time.time()-start, epoch, opt.num_epoch, loss))
    valid(model, test_loader)