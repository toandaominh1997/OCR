import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import zipfile
import pandas as pd
from torch.utils.data import Dataset
import os
from io import BytesIO
from PIL import Image
from imgaug import augmenters as iaa
from warpctc_pytorch import CTCLoss
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import cv2
import random
import time
from tqdm import tqdm


def get_vocab(root, label):
#     archive = zipfile.ZipFile('/content/drive/My Drive/IAM.zip', 'r')
#     filename=archive.read(label)
    filename = os.path.join(root, label)
    df = pd.read_json(filename, typ='series', encoding="utf-8")
    df = pd.DataFrame(df)
    df = df.reset_index()
    df.columns = ['index', 'label']
    alphabets = ''.join(sorted(set(''.join(df['label'].get_values()))))
    return alphabets

alphabets = get_vocab('../input/iam/IAM', label='train-labels.json')

print(alphabets)

# Transform
def random_dilate(img):
    img = np.array(img)
    img = cv2.dilate(img, np.ones(shape=(random.randint(1,3), random.randint(1,3)), dtype=np.uint8))
    return Image.fromarray(img)

def random_erode(img):
    img = np.array(img)
    img = cv2.erode(img, np.ones(shape=(random.randint(1,3), random.randint(1,3)), dtype=np.uint8))
    return Image.fromarray(img)

class ResizeImage:
    def __init__(self, height):
        self.height = height 
    def __call__(self, img):
        img = np.array(img)
        h,w = img.shape[:2]

        new_w = int(self.height / h * w)
        img = cv2.resize(img, (new_w, self.height), interpolation=cv2.INTER_CUBIC)

        return Image.fromarray(img)

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.5,
                          iaa.OneOf(
                              [
                                  iaa.GaussianBlur(sigma=(0, 3.0)),
                                  iaa.AverageBlur(k=(3, 11)),
                                  iaa.MedianBlur(k=(3, 11))
                              ])
                          ),

        ])

    def __call__(self, img):
        img = np.array(img)
        transformed_img =  self.aug.augment_image(img)

        return Image.fromarray(transformed_img)
def train_transforms(height):
    transform = transforms.Compose([

            transforms.RandomApply(
                [
                    random_dilate,
                ],
                p=0.15),

            transforms.RandomApply(
                [
                    random_erode,
                ],
                p=0.15),

            transforms.RandomApply(
                [
                    ImgAugTransform(),
                ],
                p=0.15),
                
            transforms.RandomApply(
                [
                    transforms.Pad(3, fill=255, padding_mode='constant'),
                ],
                p=0.15),

            transforms.RandomApply(
                [
                    transforms.Pad(3, fill=255, padding_mode='reflect'),
                ],
                p=0.15),

            transforms.RandomAffine(degrees=5, scale=(0.9, 1.1), shear=5, resample=Image.NEAREST),
            ResizeImage(height),
            transforms.ToTensor()
        ])
    return transform
def test_transforms(height):
    transform = transforms.Compose([
        ResizeImage(height),
        transforms.ToTensor() 
    ])
    return transform
def target_transforms():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform

class lmdbDataset(Dataset):
    def __init__(self, root, label, transform=None, target_transform=None):
#         self.archive = zipfile.ZipFile('/content/drive/My Drive/IAM.zip', 'r')
        (self.path, self.label) = self.parse_path(root, label)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform 
        
    def supportdata(self, root, label):
#         filename = self.archive.read(label)
        filename = os.path.join(root, label)
        df = pd.read_json(filename, typ='series')
        df = pd.DataFrame(df)
        df = df.reset_index()
        df.columns = ['index', 'label']
        return df

    def parse_path(self, root, label):
        df = pd.DataFrame()
        df = self.supportdata(root, label)
        return (df['index'].tolist(), df['label'].tolist())
    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):
        try:
#             archive = zipfile.ZipFile('/content/drive/My Drive/IAM.zip', 'r')
#             filename = BytesIO(archive.read(os.path.join('words', self.path[index])))
            filename = os.path.join(self.root, 'words')
            filename = os.path.join(filename, self.path[index])
            img = Image.open(filename).convert('L')
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]
        if self.transform is not None:
            img = self.transform(img)
        label = self.label[index].encode()
        if self.target_transform is not None:
            label = self.target_transform(label)
        return (img, label)

class alignCollate(object):
    def __init__(self):
        pass
    def __call__(self, batch):
        images, labels = zip(*batch)
        c = images[0].size(0)
        h = max([p.size(1) for p in images])
        w = max([p.size(2) for p in images])
        batch_images = torch.zeros(len(images), c, h, w).fill_(1)
        for i, image in enumerate(images):
            started_h = max(0, random.randint(0, h-image.size(1)))
            started_w = max(0, random.randint(0, w-image.size(2)))
            batch_images[i,:, started_h:started_h+image.size(1), started_w:started_w+image.size(2)] = image
        return batch_images, labels
train_transform = train_transforms(height=48)
test_transform = test_transforms(height=48)

train_dataset = lmdbDataset(root='../input/iam/IAM', label='valid-labels.json', transform=train_transform)
test_dataset = lmdbDataset(root='../input/iam/IAM', label='valid-labels.json', transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=int(4), collate_fn=alignCollate())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=int(4), collate_fn=alignCollate())

class strLabelConverter(object):
    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for -1 index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        length = []
        result = []
        for item in text:
            item = item.decode('utf-8', 'strict')
            item = item.replace('\n', '').replace('\r\n', '')
            length.append(len(item))
            for char in item:
                index = self.dict[char]
                result.append(index)

        text = result
        # print(text,length)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
num_class = len(alphabets)+1
converter = strLabelConverter(alphabets)
criterion = CTCLoss()

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
model = Model(n_classes=num_class)
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def by_field(predict, target):
    with torch.no_grad():
        correct=0
        targets = []
        for i in target:
            targets.append(i.decode('utf-8', 'strict'))
        for pred, tar in zip(predict, targets):
            if(pred==tar):
                correct+=1
    return correct / float(len(targets))

def by_char(predict, target):
    with torch.no_grad():
        targets = []
        correct = 0
        total_target = 0
        for i in target:
            targets.append(i.decode('utf-8', 'strict'))
        for pred, tar in zip(predict, targets):
            total_target+=len(tar)
            for p, t in zip(pred, tar):
                if(p==t):
                    correct+=1
    return correct/float(total_target)

def valid():
    model.eval()
    total_val_loss = 0
    accBF = 0.9
    accBC = 0.0
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target
            batch_size = data.size(0)
            optimizer.zero_grad()
            t, length = converter.encode(target)
            output = model(data)
            output_size = Variable(torch.IntTensor([output.size(0)] * batch_size))
            loss = criterion(output, t, output_size, length) / batch_size
            total_val_loss += loss.item()
            _, output = output.max(2)
            output = output.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(output.data, output_size.data, raw=False)
            accBF += by_field(sim_preds, target)
            accBC += by_char(sim_preds, target)
        print('Test-Loss: {}, accBF: {}, accBC: {}'.format(total_loss/len(test_loader), accBF/len(test_loader), accBC/len(test_loader)))

epochs = 50

for epoch in range(1, epochs):
    accBF = 0.0
    accBC = 0.0
    total_loss = 0
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target
        batch_size = data.size(0)
        optimizer.zero_grad()
        t, length = converter.encode(target)
        output = model(data)
        output_size = Variable(torch.IntTensor([output.size(0)] * batch_size))
        loss = criterion(output, t, output_size, length) / batch_size
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, output = output.max(2)
        output = output.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(output.data, output_size.data, raw=False)
        accBF += by_field(sim_preds, target)
        accBC += by_char(sim_preds, target)
        if((idx+1)%1000==0):
            print(('Index: {}/{}, Loss: {}'.format(idx, len(train_loader), total_loss/idx)))    
    print('Epoch: {}/{}, Loss: {}, accBF: {}, accBC: {}'.format(epoch, epochs, 
                        total_loss/len(train_loader), accBF/len(train_loader), accBC/len(train_loader)))
    valid()